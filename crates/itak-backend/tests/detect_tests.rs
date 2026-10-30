//! Detection and auto-config tests, run entirely against a fake
//! sysfs tree and a fake command runner: no test here touches the
//! host's real hardware, PATH, or `nvidia-smi`.
//!
//! The first test is the named regression for the original
//! engine's hardware bug (PLAN S7, TODO T-4.3): the original
//! `detectLinux` read VRAM only from `nvidia-smi`, so the RX 7900
//! XTX probed as 0 MiB and AutoConfig forced CPU-only. The fixture
//! is that machine: an AMD card in sysfs with the live card's real
//! `mem_info_vram_total`, and no `nvidia-smi` at all.

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use itak_backend::{
    auto_select, AutoConfig, BackendKind, CommandRunner, DetectSource, Detector, GpuVendor,
    SelectionReason,
};

/// The live RX 7900 XTX's `mem_info_vram_total`: 24560 MiB in
/// bytes, the figure the recovered original verified on the real
/// card after its sysfs fix.
const VRAM_7900_XTX_BYTES: u64 = 25_753_026_560;
/// 24560 MiB, the same figure in the units the report uses.
const VRAM_7900_XTX_MIB: u64 = 24_560;

/// A fake sysfs tree in a unique temp directory, removed on drop.
struct FakeSysfs {
    root: PathBuf,
}

impl FakeSysfs {
    fn new(test: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let root = std::env::temp_dir().join(format!(
            "itak-backend-{test}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(root.join("sys/class/drm")).expect("create fake drm dir");
        Self { root }
    }

    /// Add one `card<index>` with the given sysfs files. `vram_bytes`
    /// writes `mem_info_vram_total`; `None` leaves the file absent,
    /// as on drivers that do not expose it.
    fn card(
        &self,
        index: u32,
        vendor: &str,
        device: Option<&str>,
        vram_bytes: Option<u64>,
    ) -> &Self {
        let dir = self.root.join(format!("sys/class/drm/card{index}/device"));
        fs::create_dir_all(&dir).expect("create fake device dir");
        fs::write(dir.join("vendor"), vendor).expect("write vendor");
        if let Some(device) = device {
            fs::write(dir.join("device"), device).expect("write device");
        }
        if let Some(bytes) = vram_bytes {
            fs::write(dir.join("mem_info_vram_total"), bytes.to_string())
                .expect("write vram total");
        }
        self
    }

    /// A detector pointed at this tree with the given SMI runner.
    fn detector(&self, runner: FakeRunner) -> Detector {
        Detector::new()
            .with_sysfs_root(self.root.clone())
            .with_command_runner(Box::new(runner))
    }
}

impl Drop for FakeSysfs {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

/// A command runner that answers with canned stdout (or fails,
/// like a machine with no `nvidia-smi` installed).
struct FakeRunner {
    smi_output: Option<String>,
}

impl FakeRunner {
    /// No `nvidia-smi`: every command fails.
    fn none() -> Self {
        Self { smi_output: None }
    }

    /// `nvidia-smi` answers with the given CSV text.
    fn smi(text: &str) -> Self {
        Self {
            smi_output: Some(text.to_string()),
        }
    }
}

impl CommandRunner for FakeRunner {
    fn run(&self, program: &str, _args: &[&str]) -> Option<String> {
        if program == "nvidia-smi" {
            self.smi_output.clone()
        } else {
            None
        }
    }
}

/// THE regression test, named for the original bug (TODO T-4.3):
/// the 7900 XTX reported as 0 MiB forcing CPU-only. On this
/// fixture (AMD in sysfs, no `nvidia-smi` anywhere) the original
/// engine saw 0 MiB and chose CPU. This rebuild must see the full
/// 24560 MiB from amdgpu sysfs and choose the GPU.
#[test]
fn regression_7900_xtx_reported_zero_mib_forced_cpu_only() {
    let sysfs = FakeSysfs::new("regression-7900xtx");
    sysfs.card(0, "0x1002\n", Some("0x744c\n"), Some(VRAM_7900_XTX_BYTES));

    // No nvidia-smi: the exact condition that sank the original.
    let report = sysfs.detector(FakeRunner::none()).detect();

    assert_eq!(report.gpus.len(), 1, "one GPU in the report: {report:?}");
    let gpu = &report.gpus[0];
    assert_eq!(gpu.vendor, GpuVendor::Amd);
    assert_eq!(gpu.name, "AMD Radeon RX 7900 XTX");
    assert_eq!(gpu.source, DetectSource::AmdgpuSysfs);
    assert_eq!(
        gpu.vram_mib, VRAM_7900_XTX_MIB,
        "the bug read this as 0 MiB; sysfs must report the full total"
    );

    let selection = auto_select(&report, &AutoConfig::default());
    assert_eq!(
        selection.kind,
        BackendKind::WgpuVulkan,
        "the bug forced CPU-only here; the GPU must be chosen: {selection:?}"
    );
    assert_eq!(selection.reason, SelectionReason::GpuSelected);
    assert_eq!(
        selection.gpu_name.as_deref(),
        Some("AMD Radeon RX 7900 XTX")
    );
    assert!(
        selection.usable_vram_mib > 20_000 && selection.usable_vram_mib <= VRAM_7900_XTX_MIB,
        "budget is the configured fraction of the card: {selection:?}"
    );
}

/// The NVIDIA fallback still works: sysfs empty, `nvidia-smi`
/// answers, the card is sized and chosen.
#[test]
fn nvidia_smi_fallback_detects_and_selects_gpu() {
    let sysfs = FakeSysfs::new("nvidia-fallback");
    let report = sysfs
        .detector(FakeRunner::smi("NVIDIA GeForce RTX 4070 Ti SUPER, 16376\n"))
        .detect();

    assert_eq!(report.gpus.len(), 1);
    let gpu = &report.gpus[0];
    assert_eq!(gpu.vendor, GpuVendor::Nvidia);
    assert_eq!(gpu.source, DetectSource::NvidiaSmi);
    assert_eq!(gpu.vram_mib, 16376);

    let selection = auto_select(&report, &AutoConfig::default());
    assert_eq!(selection.kind, BackendKind::WgpuVulkan);
    assert_eq!(selection.reason, SelectionReason::GpuSelected);
}

/// The CPU-only fallback still works: no cards, no `nvidia-smi`,
/// an honest empty report and a CPU verdict that says why.
#[test]
fn cpu_only_when_no_gpu_found() {
    let sysfs = FakeSysfs::new("cpu-only");
    let report = sysfs.detector(FakeRunner::none()).detect();

    assert!(!report.has_gpu());
    assert!(report.best_gpu().is_none());

    let selection = auto_select(&report, &AutoConfig::default());
    assert_eq!(selection.kind, BackendKind::CpuSimd);
    assert_eq!(selection.reason, SelectionReason::NoGpuDetected);
    assert_eq!(selection.usable_vram_mib, 0);
}

/// An AMD card whose VRAM is below the configured minimum falls
/// back to CPU, with the shortfall named in the reason. This is
/// also the honest shape of the original failure mode: a card
/// seen with VRAM unknown (0) is a fallback with a reason, never
/// a silent CPU-only machine.
#[test]
fn amd_below_min_vram_falls_back_to_cpu_with_reason() {
    let sysfs = FakeSysfs::new("below-min");
    // 512 MiB card: real sysfs VRAM, far below the 4096 default.
    sysfs.card(0, "0x1002\n", Some("0x1638\n"), Some(512 * 1024 * 1024));

    let report = sysfs.detector(FakeRunner::none()).detect();
    assert_eq!(report.gpus[0].vram_mib, 512);

    let selection = auto_select(&report, &AutoConfig::default());
    assert_eq!(selection.kind, BackendKind::CpuSimd);
    assert_eq!(
        selection.reason,
        SelectionReason::GpuBelowMinVram {
            found_mib: 512,
            required_mib: 4096
        }
    );
}

/// An AMD card with no `mem_info_vram_total` (the driver did not
/// expose it) is reported with VRAM 0 and source Sysfs, so the
/// gap is visible in the report itself, and auto-config falls
/// back with the reason named instead of guessing.
#[test]
fn amd_without_vram_file_reports_unknown_and_falls_back() {
    let sysfs = FakeSysfs::new("vram-unknown");
    sysfs.card(0, "0x1002\n", Some("0x744c\n"), None);

    let report = sysfs.detector(FakeRunner::none()).detect();
    assert_eq!(report.gpus.len(), 1);
    assert_eq!(report.gpus[0].vram_mib, 0);
    assert_eq!(report.gpus[0].source, DetectSource::Sysfs);

    let selection = auto_select(&report, &AutoConfig::default());
    assert_eq!(selection.kind, BackendKind::CpuSimd);
    assert_eq!(
        selection.reason,
        SelectionReason::GpuBelowMinVram {
            found_mib: 0,
            required_mib: 4096
        }
    );
}

/// The VRAM minimum is config, not a compiled constant: the same
/// 7900 XTX fixture flips verdicts as the operator moves the
/// threshold across the card's size.
#[test]
fn min_vram_threshold_is_config_not_a_constant() {
    let sysfs = FakeSysfs::new("threshold-config");
    sysfs.card(0, "0x1002\n", Some("0x744c\n"), Some(VRAM_7900_XTX_BYTES));
    let report = sysfs.detector(FakeRunner::none()).detect();

    let strict = AutoConfig {
        min_gpu_vram_mib: 32_768,
        ..AutoConfig::default()
    };
    let selection = auto_select(&report, &strict);
    assert_eq!(selection.kind, BackendKind::CpuSimd);
    assert_eq!(
        selection.reason,
        SelectionReason::GpuBelowMinVram {
            found_mib: VRAM_7900_XTX_MIB,
            required_mib: 32_768
        }
    );

    let lenient = AutoConfig {
        min_gpu_vram_mib: 1024,
        ..AutoConfig::default()
    };
    let selection = auto_select(&report, &lenient);
    assert_eq!(selection.kind, BackendKind::WgpuVulkan);
}

/// The operator switch beats hardware: `prefer_gpu` off chooses
/// CPU on a machine with a fully capable GPU, and says so.
#[test]
fn operator_can_force_cpu() {
    let sysfs = FakeSysfs::new("force-cpu");
    sysfs.card(0, "0x1002\n", Some("0x744c\n"), Some(VRAM_7900_XTX_BYTES));
    let report = sysfs.detector(FakeRunner::none()).detect();

    let config = AutoConfig {
        prefer_gpu: false,
        ..AutoConfig::default()
    };
    let selection = auto_select(&report, &config);
    assert_eq!(selection.kind, BackendKind::CpuSimd);
    assert_eq!(selection.reason, SelectionReason::OperatorForcedCpu);
}

/// With an integrated Intel GPU in card0 and the discrete AMD
/// card in card1, detection reports both and auto-config plans
/// against the discrete card.
#[test]
fn discrete_amd_preferred_over_integrated_intel() {
    let sysfs = FakeSysfs::new("multi-card");
    sysfs.card(0, "0x8086\n", Some("0x4680\n"), None);
    sysfs.card(1, "0x1002\n", Some("0x744c\n"), Some(VRAM_7900_XTX_BYTES));

    let report = sysfs.detector(FakeRunner::none()).detect();
    assert_eq!(report.gpus.len(), 2);
    assert_eq!(report.gpus[0].vendor, GpuVendor::Intel);
    assert_eq!(report.gpus[0].vram_mib, 0);

    let best = report.best_gpu().expect("a best GPU");
    assert_eq!(best.vendor, GpuVendor::Amd);
    assert_eq!(best.vram_mib, VRAM_7900_XTX_MIB);

    let selection = auto_select(&report, &AutoConfig::default());
    assert_eq!(selection.kind, BackendKind::WgpuVulkan);
    assert_eq!(
        selection.gpu_name.as_deref(),
        Some("AMD Radeon RX 7900 XTX")
    );
}
