//! Auto-config: turn a [`HardwareReport`] into a backend choice.
//!
//! Every limit here is a field on [`AutoConfig`] with a default,
//! never a compiled-in constant (David's rule): the minimum VRAM a
//! GPU needs before the engine plans against it, and the fraction
//! of VRAM the engine may budget, are operator-tunable config.
//! Tests prove the point by moving the threshold across a fixed
//! hardware fixture and watching the verdict flip.
//!
//! Every fallback (GPU present but not chosen, GPU disabled by the
//! operator) is a named reason in the selection and a line on the
//! diag channel, per the architecture: a fallback is a fact the
//! operator can pull out of the module, never a silent default.

use crate::diag::dbg_log;
use crate::hwdetect::HardwareReport;

/// Auto-config limits and switches, each with a default.
///
/// In the unified crate these are the same limits
/// [`crate::BackendConfig`] carries (with the same defaults, from
/// the same constants); this struct remains the configuration
/// [`auto_select`] was proven against, so its tests and call sites
/// keep the shape they were written to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoConfig {
    /// Minimum dedicated VRAM, in MiB, for a GPU to be chosen.
    /// Below this the card cannot hold a model this engine serves,
    /// so selection falls back to CPU with the reason named.
    /// Default: 4096 MiB.
    pub min_gpu_vram_mib: u64,
    /// Fraction of the chosen GPU's VRAM the engine may budget
    /// for weights and KV state; the rest stays free for the
    /// driver and other processes. Clamped to `0.0..=1.0`.
    /// Default: 0.90.
    pub vram_budget_fraction: f32,
    /// Operator switch: when false, selection is CPU even on a
    /// machine with a capable GPU. Default: true.
    pub prefer_gpu: bool,
}

impl Default for AutoConfig {
    fn default() -> Self {
        Self {
            min_gpu_vram_mib: crate::config::DEFAULT_MIN_GPU_VRAM_MIB,
            vram_budget_fraction: crate::config::DEFAULT_VRAM_BUDGET_FRACTION,
            prefer_gpu: true,
        }
    }
}

/// The backend auto-config can choose. The wgpu Vulkan variant is
/// the GPU path from the architecture (the 7900 XTX path); the CPU
/// SIMD variant is always available. This is auto-config's own
/// two-way verdict type, distinct from the selection policy's
/// [`crate::report::BackendKind`] vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// wgpu on Vulkan against the chosen GPU.
    WgpuVulkan,
    /// CPU SIMD, the always-available fallback.
    CpuSimd,
}

/// Why auto-config chose what it chose. Fallbacks are variants,
/// not log-only events, so callers and tests can assert on them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionReason {
    /// A GPU met the configured minimum VRAM and was chosen.
    GpuSelected,
    /// Detection found no GPU at all.
    NoGpuDetected,
    /// GPUs were found but the best one is below the configured
    /// minimum VRAM (including VRAM unknown, reported as 0).
    GpuBelowMinVram {
        /// VRAM of the best GPU found, in MiB.
        found_mib: u64,
        /// The configured minimum it failed to meet, in MiB.
        required_mib: u64,
    },
    /// The operator set `prefer_gpu` off in config.
    OperatorForcedCpu,
}

/// The auto-config verdict: which backend, on which GPU, with how
/// much VRAM budget, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendSelection {
    /// The chosen backend.
    pub kind: BackendKind,
    /// The chosen GPU's name, when a GPU was chosen.
    pub gpu_name: Option<String>,
    /// VRAM the engine may budget, in MiB: the chosen GPU's VRAM
    /// times the configured budget fraction. 0 for CPU selections.
    pub usable_vram_mib: u64,
    /// Why this selection was made.
    pub reason: SelectionReason,
}

/// Choose a backend for a hardware report under a config.
///
/// Selection order: the operator switch wins first; then the best
/// GPU (most VRAM) is chosen if it meets the configured minimum;
/// otherwise CPU, with the reason distinguishing "no GPU exists"
/// from "a GPU exists but is too small".
#[must_use]
pub fn auto_select(report: &HardwareReport, config: &AutoConfig) -> BackendSelection {
    if !config.prefer_gpu {
        dbg_log!("auto-config: operator forced CPU (prefer_gpu off)");
        return cpu_selection(SelectionReason::OperatorForcedCpu);
    }

    let Some(best) = report.best_gpu() else {
        dbg_log!("auto-config: no GPU detected, CPU backend");
        return cpu_selection(SelectionReason::NoGpuDetected);
    };

    if best.vram_mib < config.min_gpu_vram_mib {
        dbg_log!(
            "auto-config: best GPU {} has {} MiB, below the configured \
             minimum {} MiB; CPU backend",
            best.name,
            best.vram_mib,
            config.min_gpu_vram_mib
        );
        return cpu_selection(SelectionReason::GpuBelowMinVram {
            found_mib: best.vram_mib,
            required_mib: config.min_gpu_vram_mib,
        });
    }

    let fraction = f64::from(config.vram_budget_fraction.clamp(0.0, 1.0));
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let usable_vram_mib = (fraction * best.vram_mib as f64) as u64;
    dbg_log!(
        "auto-config: GPU {} selected, {} MiB VRAM, {} MiB budgeted",
        best.name,
        best.vram_mib,
        usable_vram_mib
    );
    BackendSelection {
        kind: BackendKind::WgpuVulkan,
        gpu_name: Some(best.name.clone()),
        usable_vram_mib,
        reason: SelectionReason::GpuSelected,
    }
}

/// A CPU selection with a named reason.
fn cpu_selection(reason: SelectionReason) -> BackendSelection {
    BackendSelection {
        kind: BackendKind::CpuSimd,
        gpu_name: None,
        usable_vram_mib: 0,
        reason,
    }
}
