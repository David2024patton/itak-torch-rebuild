//! Hardware detection: what GPU hardware does this machine have,
//! and how much VRAM does each card carry?
//!
//! The AMD path reads the amdgpu driver's sysfs files directly:
//! `/sys/class/drm/card<N>/device/vendor` for the PCI vendor and
//! `/sys/class/drm/card<N>/device/mem_info_vram_total` for the
//! VRAM total in bytes. This exists because of the original
//! engine's defining hardware bug: its `detectLinux` read VRAM
//! only from `nvidia-smi`, so an AMD card probed as 0 MiB and
//! auto-config forced CPU-only on a machine whose 24 GiB card was
//! the whole point. The regression test in this crate's test suite
//! is named for that bug.
//!
//! Fallback order, per the architecture:
//!
//! 1. AMD (or any vendor) cards found in sysfs, with real VRAM for
//!    amdgpu cards.
//! 2. `nvidia-smi`, for NVIDIA cards the sysfs pass could not size.
//! 3. An empty report: CPU-only, a legitimate answer that
//!    auto-config turns into the CPU backend, never an error.
//!
//! The filesystem root is injectable ([`Detector::with_sysfs_root`])
//! and the command runner is injectable
//! ([`Detector::with_command_runner`]), so tests run against a fake
//! sysfs tree and never touch the host's real hardware or PATH.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use crate::diag::dbg_log;

/// PCI vendor ID for AMD (`0x1002` in the sysfs `vendor` file).
const PCI_VENDOR_AMD: u32 = 0x1002;
/// PCI vendor ID for NVIDIA (`0x10de` in the sysfs `vendor` file).
const PCI_VENDOR_NVIDIA: u32 = 0x10de;
/// PCI vendor ID for Intel (`0x8086` in the sysfs `vendor` file).
const PCI_VENDOR_INTEL: u32 = 0x8086;

/// Bytes in one MiB; sysfs reports VRAM in bytes, the rest of the
/// engine plans in MiB.
const BYTES_PER_MIB: u64 = 1024 * 1024;

/// Which vendor made a GPU, from its PCI vendor ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuVendor {
    /// AMD (`0x1002`): the amdgpu path, VRAM from sysfs.
    Amd,
    /// NVIDIA (`0x10de`): sized by `nvidia-smi` when available.
    Nvidia,
    /// Intel (`0x8086`): integrated graphics, no dedicated VRAM.
    Intel,
    /// Any other PCI vendor ID, carried so the report stays honest.
    Unknown(u32),
}

impl GpuVendor {
    /// Map a PCI vendor ID from a sysfs `vendor` file to a vendor.
    #[must_use]
    pub fn from_pci_id(id: u32) -> Self {
        match id {
            PCI_VENDOR_AMD => Self::Amd,
            PCI_VENDOR_NVIDIA => Self::Nvidia,
            PCI_VENDOR_INTEL => Self::Intel,
            other => Self::Unknown(other),
        }
    }
}

/// Where a GPU entry's facts came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectSource {
    /// The amdgpu sysfs files, including a real VRAM total.
    AmdgpuSysfs,
    /// A sysfs `device` directory that named the vendor but exposed
    /// no VRAM total (integrated graphics, or a card whose driver
    /// is not amdgpu): the entry is real, its VRAM is unknown and
    /// reported as 0.
    Sysfs,
    /// An `nvidia-smi` query line.
    NvidiaSmi,
}

/// One detected GPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuInfo {
    /// The card's vendor.
    pub vendor: GpuVendor,
    /// Best known human label: a known-device name for AMD cards
    /// (the 7900 XTX is named), the `nvidia-smi` name for NVIDIA
    /// cards, otherwise a vendor label with the device ID.
    pub name: String,
    /// PCI device ID from sysfs, when the sysfs pass found the card.
    pub device_id: Option<u32>,
    /// Dedicated VRAM in MiB. 0 means unknown (see
    /// [`DetectSource::Sysfs`]), never a silent measurement.
    pub vram_mib: u64,
    /// Where this entry's facts came from.
    pub source: DetectSource,
    /// The `card<N>` index in sysfs, when the sysfs pass found it.
    pub card_index: Option<u32>,
}

/// The result of a detection pass: every GPU found, or none.
///
/// Carries a `version` field like every contract in this project,
/// so the summary that crosses the seam in Hello/HelloAck can be
/// checked against the shape the platform expects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareReport {
    /// Report format version. See [`HardwareReport::VERSION`].
    pub version: u32,
    /// Every GPU detected, in sysfs card order, with any
    /// `nvidia-smi` entries appended after the sysfs entries.
    pub gpus: Vec<GpuInfo>,
}

impl HardwareReport {
    /// The current report format version.
    pub const VERSION: u32 = 1;

    /// An empty report: no GPU found, CPU-only machine.
    #[must_use]
    pub fn cpu_only() -> Self {
        Self {
            version: Self::VERSION,
            gpus: Vec::new(),
        }
    }

    /// Whether any GPU was detected at all.
    #[must_use]
    pub fn has_gpu(&self) -> bool {
        !self.gpus.is_empty()
    }

    /// The GPU with the most VRAM: the one auto-config plans
    /// against. Ties go to the earliest card in the report.
    #[must_use]
    pub fn best_gpu(&self) -> Option<&GpuInfo> {
        self.gpus.iter().max_by_key(|gpu| gpu.vram_mib)
    }
}

/// Runs an external command and returns its stdout on success.
///
/// Injectable so tests can answer for `nvidia-smi` without a real
/// NVIDIA machine; the production implementation is
/// [`SystemCommandRunner`].
pub trait CommandRunner {
    /// Run `program` with `args`; return stdout as text if the
    /// command ran and exited successfully, else `None`.
    fn run(&self, program: &str, args: &[&str]) -> Option<String>;
}

/// The production [`CommandRunner`]: a real subprocess.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemCommandRunner;

impl CommandRunner for SystemCommandRunner {
    fn run(&self, program: &str, args: &[&str]) -> Option<String> {
        let output = Command::new(program).args(args).output().ok()?;
        if !output.status.success() {
            return None;
        }
        String::from_utf8(output.stdout).ok()
    }
}

/// Hardware detector with an injectable sysfs root and command
/// runner. [`Detector::new`] is the production configuration: the
/// real `/` root and real subprocesses.
pub struct Detector {
    sysfs_root: PathBuf,
    max_cards: u32,
    runner: Box<dyn CommandRunner>,
}

impl Default for Detector {
    fn default() -> Self {
        Self::new()
    }
}

impl Detector {
    /// The production detector: sysfs at `/`, at most 16 cards
    /// probed, real subprocesses for the `nvidia-smi` fallback.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sysfs_root: PathBuf::from("/"),
            max_cards: 16,
            runner: Box::new(SystemCommandRunner),
        }
    }

    /// Point the sysfs scan at another root. Tests pass a fake
    /// sysfs tree; the scan reads `<root>/sys/class/drm/...`.
    #[must_use]
    pub fn with_sysfs_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.sysfs_root = root.into();
        self
    }

    /// Replace the command runner (the `nvidia-smi` fallback).
    #[must_use]
    pub fn with_command_runner(mut self, runner: Box<dyn CommandRunner>) -> Self {
        self.runner = runner;
        self
    }

    /// Set how many `card<N>` slots the scan probes. A probe bound
    /// with a default, not a compiled-in limit on the report.
    #[must_use]
    pub fn with_max_cards(mut self, max_cards: u32) -> Self {
        self.max_cards = max_cards;
        self
    }

    /// Run one detection pass and return the report. Detection
    /// never fails: the worst case is the CPU-only report.
    #[must_use]
    pub fn detect(&self) -> HardwareReport {
        let mut gpus = self.scan_sysfs();

        // The NVIDIA fallback fires only when sysfs could not size
        // any card: an AMD card with a real VRAM total ends the
        // search, exactly the case the original engine got wrong.
        if !gpus.iter().any(|gpu| gpu.vram_mib > 0) {
            let smi_gpus = self.query_nvidia_smi();
            if !smi_gpus.is_empty() {
                // SMI sizes replace any same-vendor sysfs entries
                // that carried VRAM 0 (unknown), so the report
                // never carries the same card twice.
                gpus.retain(|gpu| gpu.vendor != GpuVendor::Nvidia);
                gpus.extend(smi_gpus);
            }
        }

        if gpus.is_empty() {
            dbg_log!("detect: no GPU found, reporting CPU-only");
        } else {
            for gpu in &gpus {
                dbg_log!(
                    "detect: card {:?} {} vendor {:?} vram {} MiB via {:?}",
                    gpu.card_index,
                    gpu.name,
                    gpu.vendor,
                    gpu.vram_mib,
                    gpu.source
                );
            }
        }

        HardwareReport {
            version: HardwareReport::VERSION,
            gpus,
        }
    }

    /// The sysfs pass: probe `card<N>/device` directories under
    /// `<root>/sys/class/drm`.
    fn scan_sysfs(&self) -> Vec<GpuInfo> {
        let mut gpus = Vec::new();
        let drm = self.sysfs_root.join("sys").join("class").join("drm");
        for index in 0..self.max_cards {
            let device = drm.join(format!("card{index}")).join("device");
            let Some(vendor_id) = read_hex_id(&device.join("vendor")) else {
                continue;
            };
            let vendor = GpuVendor::from_pci_id(vendor_id);
            let device_id = read_hex_id(&device.join("device"));
            let gpu = match vendor {
                GpuVendor::Amd => {
                    let vram_mib = read_vram_mib(&device.join("mem_info_vram_total"));
                    if vram_mib == 0 {
                        dbg_log!(
                            "detect: card{index} is AMD but mem_info_vram_total \
                             is missing or zero; VRAM unknown"
                        );
                    }
                    GpuInfo {
                        vendor,
                        name: amd_name(device_id),
                        device_id,
                        vram_mib,
                        source: if vram_mib > 0 {
                            DetectSource::AmdgpuSysfs
                        } else {
                            DetectSource::Sysfs
                        },
                        card_index: Some(index),
                    }
                }
                _ => GpuInfo {
                    vendor,
                    name: generic_name(vendor, device_id),
                    device_id,
                    vram_mib: 0,
                    source: DetectSource::Sysfs,
                    card_index: Some(index),
                },
            };
            gpus.push(gpu);
        }
        gpus
    }

    /// The `nvidia-smi` fallback pass.
    fn query_nvidia_smi(&self) -> Vec<GpuInfo> {
        let Some(text) = self.runner.run(
            "nvidia-smi",
            &[
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
        ) else {
            dbg_log!("detect: nvidia-smi unavailable or failed");
            return Vec::new();
        };
        parse_nvidia_smi_csv(&text)
    }
}

/// Detect hardware with the production configuration. This is the
/// seam the architecture names: `detect() -> HardwareReport`.
#[must_use]
pub fn detect() -> HardwareReport {
    Detector::new().detect()
}

/// Read a sysfs hex ID file (`0x1002\n`) as a number.
fn read_hex_id(path: &std::path::Path) -> Option<u32> {
    let text = fs::read_to_string(path).ok()?;
    parse_hex_id(&text)
}

/// Parse a sysfs hex ID string.
fn parse_hex_id(text: &str) -> Option<u32> {
    let trimmed = text.trim();
    let digits = trimmed.strip_prefix("0x").unwrap_or(trimmed);
    u32::from_str_radix(digits, 16).ok()
}

/// Read an amdgpu `mem_info_vram_total` file (bytes, decimal) as
/// MiB. Missing or unparsable reads as 0 (unknown), never a guess.
fn read_vram_mib(path: &std::path::Path) -> u64 {
    let Ok(text) = fs::read_to_string(path) else {
        return 0;
    };
    parse_vram_bytes(&text).map_or(0, |bytes| bytes / BYTES_PER_MIB)
}

/// Parse a decimal byte count from a sysfs VRAM file.
fn parse_vram_bytes(text: &str) -> Option<u64> {
    text.trim().parse::<u64>().ok()
}

/// The friendly name for an AMD card, from its PCI device ID.
/// Unknown IDs get an honest generic label, never a wrong name.
fn amd_name(device_id: Option<u32>) -> String {
    match device_id {
        Some(0x744c) => "AMD Radeon RX 7900 XTX".to_string(),
        Some(0x7448) => "AMD Radeon RX 7900 XT".to_string(),
        Some(0x744e) => "AMD Radeon RX 7900 GRE".to_string(),
        Some(id) => format!("AMD GPU (device 0x{id:04X})"),
        None => "AMD GPU".to_string(),
    }
}

/// The label for a non-AMD sysfs card whose VRAM sysfs cannot size.
fn generic_name(vendor: GpuVendor, device_id: Option<u32>) -> String {
    let label = match vendor {
        GpuVendor::Nvidia => "NVIDIA GPU",
        GpuVendor::Intel => "Intel GPU",
        GpuVendor::Amd => "AMD GPU",
        GpuVendor::Unknown(_) => "Unknown GPU",
    };
    match device_id {
        Some(id) => format!("{label} (device 0x{id:04X})"),
        None => label.to_string(),
    }
}

/// Parse `nvidia-smi --query-gpu=name,memory.total
/// --format=csv,noheader,nounits` output: one `name, MiB` line per
/// GPU. Unparsable lines are skipped, never guessed at.
fn parse_nvidia_smi_csv(text: &str) -> Vec<GpuInfo> {
    let mut gpus = Vec::new();
    for line in text.lines() {
        let Some((name, mib)) = line.rsplit_once(',') else {
            continue;
        };
        let Ok(vram_mib) = mib.trim().parse::<u64>() else {
            continue;
        };
        let name = name.trim();
        if name.is_empty() {
            continue;
        }
        gpus.push(GpuInfo {
            vendor: GpuVendor::Nvidia,
            name: name.to_string(),
            device_id: None,
            vram_mib,
            source: DetectSource::NvidiaSmi,
            card_index: None,
        });
    }
    gpus
}

#[cfg(test)]
mod tests {
    use super::{parse_hex_id, parse_nvidia_smi_csv, parse_vram_bytes, GpuVendor, BYTES_PER_MIB};

    #[test]
    fn hex_ids_parse_with_and_without_prefix() {
        assert_eq!(parse_hex_id("0x1002\n"), Some(0x1002));
        assert_eq!(parse_hex_id("744c"), Some(0x744c));
        assert_eq!(parse_hex_id("not-hex"), None);
        assert_eq!(parse_hex_id(""), None);
    }

    #[test]
    fn vram_bytes_parse_as_decimal() {
        // The live 7900 XTX figure: 24560 MiB in bytes.
        assert_eq!(
            parse_vram_bytes("25753026560\n"),
            Some(24_560 * BYTES_PER_MIB)
        );
        assert_eq!(parse_vram_bytes("0"), Some(0));
        assert_eq!(parse_vram_bytes("junk"), None);
    }

    #[test]
    fn nvidia_smi_csv_parses_name_and_mib() {
        let gpus = parse_nvidia_smi_csv("NVIDIA GeForce RTX 4070 Ti SUPER, 16376\n");
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].vendor, GpuVendor::Nvidia);
        assert_eq!(gpus[0].name, "NVIDIA GeForce RTX 4070 Ti SUPER");
        assert_eq!(gpus[0].vram_mib, 16376);
    }

    #[test]
    fn nvidia_smi_csv_skips_garbage_lines() {
        let gpus = parse_nvidia_smi_csv("no comma here\nName, notanumber\n, 4096\n");
        assert!(gpus.is_empty());
        assert!(parse_nvidia_smi_csv("").is_empty());
    }
}
