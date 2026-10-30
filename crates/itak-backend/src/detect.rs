//! Hardware detection and backend selection.
//!
//! [`detect`] is the function ARCHITECTURE 3.1 names for this
//! module. It ranks adapters by the BACKEND.md Section 2 policy,
//! applies the Section 6 fallback ladder, cross-checks VRAM against
//! sysfs, and returns a [`HardwareReport`] carrying the decision and
//! every diag event it emitted.
//!
//! Selection is a policy, never a heuristic: the config's
//! [`GpuPolicy`] and the adapters' capabilities decide, in that
//! order. Nothing here looks at model files, file sizes, or tensor
//! shapes (ARCHITECTURE 5.3: the serve path never routes by file
//! size; engine selection is by capability).
//!
//! The probe seam: adapter enumeration belongs to the wgpu stack,
//! which lands with the GPU backend in TODO T-4.2. Until then,
//! [`SystemProbe`] enumerates no adapters (it reads the CPU and the
//! sysfs VRAM truth), so an `Auto` selection on this build lands on
//! the CPU backend at L4 with that reason recorded, which is the
//! honest state of the build. The ranking policy itself is complete
//! and is unit-tested against synthetic adapter lists; T-4.2 wires
//! the wgpu enumerator into [`AdapterProbe`] without touching the
//! policy.

use crate::config::{BackendConfig, GpuPolicy};
use crate::diag;
use crate::report::{
    BackendKind, CpuInfo, DiagEvent, FallbackLevel, HardwareReport, SelectionOutcome, SimdFeatures,
};

/// The graphics API an adapter was enumerated through. Only Vulkan
/// is selectable: DECISIONS 2026-10-26 fixes Vulkan as the GPU path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterBackend {
    /// Vulkan (the selectable path).
    Vulkan,
    /// Anything else (seen, reported, never selected).
    Other,
}

/// The device class an adapter reported. Mirrors the wgpu
/// `DeviceType` vocabulary of the BACKEND.md Section 2 probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    /// A discrete GPU (the 7900 XTX class). First rank.
    DiscreteGpu,
    /// An integrated GPU. Selectable only at ladder level L1, only
    /// when configuration allows it.
    IntegratedGpu,
    /// A software rasterizer (llvmpipe). Counts as no adapter:
    /// BACKEND.md Section 2 rejects it outright, because it would
    /// report success while running slower than this crate's own
    /// CPU backend.
    Cpu,
    /// A device class the probe did not classify. Treated as
    /// non-discrete for ranking.
    Other,
}

/// One enumerated GPU adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterInfo {
    /// Adapter name as the driver reports it.
    pub name: String,
    /// Graphics API it was enumerated through.
    pub backend: AdapterBackend,
    /// Device class it reported.
    pub device_type: DeviceType,
    /// VRAM the adapter reports, when it reports any. Cross-checked
    /// against sysfs at selection (BACKEND.md Section 2).
    pub vram_bytes: Option<u64>,
}

/// The hardware probe seam. Production code uses [`SystemProbe`];
/// tests inject synthetic hardware. Implementations must be honest:
/// an adapter that cannot be used must not be reported as usable.
pub trait AdapterProbe {
    /// Enumerate GPU adapters.
    fn adapters(&self) -> Vec<AdapterInfo>;
    /// Total VRAM in bytes from AMD sysfs
    /// (`/sys/class/drm/card*/device/mem_info_vram_total`), when a
    /// readable AMD device exists.
    fn sysfs_vram_total_bytes(&self) -> Option<u64>;
    /// The host CPU facts.
    fn cpu_info(&self) -> CpuInfo;
}

/// The production probe: CPU facts and the sysfs VRAM truth from
/// the live host. Adapter enumeration arrives with the wgpu backend
/// (TODO T-4.2) and is honestly empty until then.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemProbe;

impl AdapterProbe for SystemProbe {
    fn adapters(&self) -> Vec<AdapterInfo> {
        // No wgpu stack is linked into this build yet (T-4.2).
        // Returning an empty list is the truth; selection records
        // it as the L4 reason rather than pretending a GPU exists.
        Vec::new()
    }

    fn sysfs_vram_total_bytes(&self) -> Option<u64> {
        // The largest readable AMD VRAM total wins: hosts carry
        // several drm cards (the 7900 XTX is card2 on the bench
        // host), and only the discrete card publishes
        // mem_info_vram_total.
        let mut best: Option<u64> = None;
        if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
            for entry in entries.flatten() {
                let path = entry.path().join("device/mem_info_vram_total");
                if let Ok(text) = std::fs::read_to_string(&path) {
                    if let Ok(value) = text.trim().parse::<u64>() {
                        best = Some(best.map_or(value, |b: u64| b.max(value)));
                    }
                }
            }
        }
        best
    }

    fn cpu_info(&self) -> CpuInfo {
        let model = std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|text| {
                text.lines().find_map(|line| {
                    line.strip_prefix("model name")
                        .and_then(|rest| rest.split(':').nth(1))
                        .map(|name| name.trim().to_string())
                })
            })
            .unwrap_or_default();
        let logical_cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        CpuInfo {
            model,
            logical_cores,
            simd: SimdFeatures {
                #[cfg(target_arch = "x86_64")]
                avx2: std::is_x86_feature_detected!("avx2"),
                #[cfg(not(target_arch = "x86_64"))]
                avx2: false,
                #[cfg(target_arch = "x86_64")]
                avx512f: std::is_x86_feature_detected!("avx512f"),
                #[cfg(not(target_arch = "x86_64"))]
                avx512f: false,
                #[cfg(target_arch = "x86_64")]
                fma: std::is_x86_feature_detected!("fma"),
                #[cfg(not(target_arch = "x86_64"))]
                fma: false,
                #[cfg(target_arch = "aarch64")]
                neon: std::arch::is_aarch64_feature_detected!("neon"),
                #[cfg(not(target_arch = "aarch64"))]
                neon: false,
            },
        }
    }
}

/// Rank adapters by the BACKEND.md Section 2 policy and return the
/// winner: Vulkan first, discrete before non-discrete, software
/// adapters rejected outright, non-discrete adapters considered
/// only when `allow_non_discrete` is set. Among equals, enumeration
/// order decides (the probe's order is the driver's order).
#[must_use]
pub fn rank_adapters(adapters: &[AdapterInfo], allow_non_discrete: bool) -> Option<&AdapterInfo> {
    let usable =
        |a: &AdapterInfo| a.backend == AdapterBackend::Vulkan && a.device_type != DeviceType::Cpu;
    adapters
        .iter()
        .find(|a| usable(a) && a.device_type == DeviceType::DiscreteGpu)
        .or_else(|| {
            if allow_non_discrete {
                adapters.iter().find(|a| usable(a))
            } else {
                None
            }
        })
}

/// Why no adapter was selected, in operator language, naming the
/// decisive fact (no enumeration, software-only, non-discrete-only).
fn no_adapter_reason(adapters: &[AdapterInfo], allow_non_discrete: bool) -> String {
    if adapters.is_empty() {
        return "no GPU adapters were enumerated by this build (wgpu enumeration lands in T-4.2)"
            .to_string();
    }
    let software: Vec<&str> = adapters
        .iter()
        .filter(|a| a.device_type == DeviceType::Cpu)
        .map(|a| a.name.as_str())
        .collect();
    if software.len() == adapters.len() {
        return format!(
            "only software adapters present ({}), which count as no adapter",
            software.join(", ")
        );
    }
    if !allow_non_discrete {
        let names: Vec<&str> = adapters
            .iter()
            .filter(|a| a.device_type != DeviceType::Cpu)
            .map(|a| a.name.as_str())
            .collect();
        return format!(
            "no discrete Vulkan adapter; non-discrete adapters ({}) are not allowed by configuration",
            names.join(", ")
        );
    }
    "no Vulkan adapter among those enumerated".to_string()
}

/// Detect hardware and select a backend under `config`, probing
/// through `probe`. See the module docs for the policy. The returned
/// report is the complete, queryable record of the decision.
#[must_use]
pub fn detect(config: &BackendConfig, probe: &dyn AdapterProbe) -> HardwareReport {
    let cpu = probe.cpu_info();
    let adapters = probe.adapters();
    let sysfs_vram_total_bytes = probe.sysfs_vram_total_bytes();
    let vram_budget_bytes =
        sysfs_vram_total_bytes.map(|total| total.saturating_sub(config.vram_reserve_bytes));
    let mut events: Vec<DiagEvent> = Vec::new();
    let emit = |code: &'static str, detail: String, events: &mut Vec<DiagEvent>| {
        diag::debug(code, &detail);
        events.push(DiagEvent { code, detail });
    };

    let (outcome, chosen_adapter, reason) = match config.gpu_policy {
        GpuPolicy::Disabled => {
            let reason = "GPU backend disabled by configuration; CPU backend selected".to_string();
            emit(
                "backend.fallback_cpu",
                "reason=config_disabled".to_string(),
                &mut events,
            );
            (
                SelectionOutcome::Selected {
                    kind: BackendKind::Cpu,
                    level: FallbackLevel::L4,
                },
                None,
                reason,
            )
        }
        GpuPolicy::Auto | GpuPolicy::Required => {
            match rank_adapters(&adapters, config.allow_non_discrete_gpu) {
                Some(adapter) => {
                    let adapter = adapter.clone();
                    let level = if adapter.device_type == DeviceType::DiscreteGpu {
                        FallbackLevel::L0
                    } else {
                        FallbackLevel::L1
                    };
                    emit(
                        "backend.selected",
                        format!(
                            "backend={} adapter={} level={level:?}",
                            BackendKind::WgpuVulkan.name(),
                            adapter.name
                        ),
                        &mut events,
                    );
                    // The Section 2 cross-check: adapter-reported
                    // VRAM and the sysfs truth must agree; a
                    // disagreement is itself a diag event.
                    if let (Some(reported), Some(sysfs)) =
                        (adapter.vram_bytes, sysfs_vram_total_bytes)
                    {
                        let diff = reported.abs_diff(sysfs);
                        if diff * 100 > sysfs {
                            emit(
                                "backend.vram_disagreement",
                                format!(
                                    "adapter={} adapter_vram={reported} sysfs_vram={sysfs}",
                                    adapter.name
                                ),
                                &mut events,
                            );
                        }
                    }
                    let reason = match level {
                        FallbackLevel::L0 => {
                            format!("discrete Vulkan adapter selected: {}", adapter.name)
                        }
                        FallbackLevel::L1 => format!(
                            "non-discrete Vulkan adapter selected, allowed by configuration: {}",
                            adapter.name
                        ),
                        FallbackLevel::L4 => {
                            unreachable!("a chosen adapter never lands on the CPU level")
                        }
                    };
                    (
                        SelectionOutcome::Selected {
                            kind: BackendKind::WgpuVulkan,
                            level,
                        },
                        Some(adapter),
                        reason,
                    )
                }
                None => {
                    let why = no_adapter_reason(&adapters, config.allow_non_discrete_gpu);
                    if config.gpu_policy == GpuPolicy::Required {
                        emit("backend.unavailable", format!("reason={why}"), &mut events);
                        let reason = format!("GPU policy is Required but {why}");
                        (
                            SelectionOutcome::Refused {
                                reason: reason.clone(),
                            },
                            None,
                            reason,
                        )
                    } else {
                        emit("backend.fallback_cpu", format!("reason={why}"), &mut events);
                        (
                            SelectionOutcome::Selected {
                                kind: BackendKind::Cpu,
                                level: FallbackLevel::L4,
                            },
                            None,
                            format!("CPU backend selected: {why}"),
                        )
                    }
                }
            }
        }
    };

    diag::debug(
        "detect",
        &format!(
            "adapters={} sysfs_vram={sysfs_vram_total_bytes:?} outcome={outcome:?}",
            adapters.len()
        ),
    );
    HardwareReport {
        cpu,
        adapters,
        chosen_adapter,
        sysfs_vram_total_bytes,
        vram_budget_bytes,
        outcome,
        reason,
        events,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter(name: &str, backend: AdapterBackend, device_type: DeviceType) -> AdapterInfo {
        AdapterInfo {
            name: name.to_string(),
            backend,
            device_type,
            vram_bytes: None,
        }
    }

    #[test]
    fn discrete_vulkan_wins_over_integrated_and_software() {
        // The bench host's exact trio (BACKEND.md Section 2):
        // software first in the list on purpose, to prove ranking
        // is by class, not by order.
        let adapters = vec![
            adapter("llvmpipe", AdapterBackend::Vulkan, DeviceType::Cpu),
            adapter(
                "Intel UHD 770",
                AdapterBackend::Vulkan,
                DeviceType::IntegratedGpu,
            ),
            adapter(
                "RX 7900 XTX",
                AdapterBackend::Vulkan,
                DeviceType::DiscreteGpu,
            ),
        ];
        let winner = rank_adapters(&adapters, false).expect("a winner exists");
        assert_eq!(winner.name, "RX 7900 XTX");
    }

    #[test]
    fn software_adapter_is_never_selected() {
        let adapters = vec![adapter("llvmpipe", AdapterBackend::Vulkan, DeviceType::Cpu)];
        assert!(rank_adapters(&adapters, true).is_none());
    }

    #[test]
    fn non_discrete_needs_the_config_gate() {
        let adapters = vec![adapter(
            "Intel UHD 770",
            AdapterBackend::Vulkan,
            DeviceType::IntegratedGpu,
        )];
        assert!(rank_adapters(&adapters, false).is_none());
        assert_eq!(
            rank_adapters(&adapters, true).map(|a| a.name.as_str()),
            Some("Intel UHD 770")
        );
    }

    #[test]
    fn non_vulkan_is_never_selected() {
        let adapters = vec![adapter(
            "Mystery GPU",
            AdapterBackend::Other,
            DeviceType::DiscreteGpu,
        )];
        assert!(rank_adapters(&adapters, true).is_none());
    }
}
