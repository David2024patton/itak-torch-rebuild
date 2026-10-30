//! Backend selection configuration.
//!
//! Per David's build law, every operational limit here is config
//! with a default, never a compiled-in constant: the GPU policy, the
//! non-discrete adapter gate (BACKEND.md Section 6, level L1), and
//! the VRAM reserve (BACKEND.md Section 5) are all fields.

/// How selection treats the GPU backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuPolicy {
    /// Use the GPU when a suitable adapter exists; otherwise land
    /// on the CPU backend at ladder level L4, with the reason
    /// recorded (BACKEND.md Section 6).
    #[default]
    Auto,
    /// The GPU backend is disabled by configuration. Selection
    /// lands on the CPU backend at L4 without consulting the
    /// adapter list: this is the T-4.1 proof configuration, the
    /// engine serving on CPU only with the GPU off.
    Disabled,
    /// The GPU backend must be selected; if no suitable adapter
    /// exists, selection refuses (see
    /// [`crate::SelectionOutcome::Refused`]) instead of silently
    /// serving on CPU.
    Required,
}

/// Default VRAM reserve: bytes held back from the sysfs VRAM total
/// when computing the planning budget (BACKEND.md Section 5: budget
/// is sysfs total minus a configured reserve). Two GiB covers the
/// display stack and other tenants of the card; it is a default,
/// and operators override it in config.
pub const DEFAULT_VRAM_RESERVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Default minimum dedicated VRAM, in MiB, for a GPU to be chosen
/// by auto-config (lane 16's limit, shared by both configs in the
/// unified crate).
pub const DEFAULT_MIN_GPU_VRAM_MIB: u64 = 4096;

/// Default fraction of the chosen GPU's VRAM the engine may budget
/// for weights and KV state (lane 16's limit, shared by both
/// configs in the unified crate).
pub const DEFAULT_VRAM_BUDGET_FRACTION: f32 = 0.90;

/// Configuration for [`crate::detect`] and backend construction.
///
/// Integration 2026-10-29: this is the crate's one selection
/// config. It carries lane 17's policy fields (`gpu_policy`,
/// `allow_non_discrete_gpu`, `vram_reserve_bytes`) AND lane 16's
/// auto-config limits (`min_gpu_vram_mib`, `vram_budget_fraction`,
/// `prefer_gpu`), each with its default, so an operator tunes one
/// struct. [`crate::autoconfig::AutoConfig`] remains the limits
/// struct [`crate::autoconfig::auto_select`] was proven against;
/// its defaults are the same constants defined here.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendConfig {
    /// How the GPU backend participates in selection.
    pub gpu_policy: GpuPolicy,
    /// Whether a non-discrete adapter (an integrated GPU) may be
    /// selected, at ladder level L1. Off by default: BACKEND.md
    /// Section 6 enters L1 only when configuration explicitly
    /// allows it, never by preference drift. Software adapters
    /// (llvmpipe) are never selectable under any setting.
    pub allow_non_discrete_gpu: bool,
    /// Bytes reserved against the sysfs VRAM total when the
    /// planning budget is computed.
    pub vram_reserve_bytes: u64,
    /// Minimum dedicated VRAM, in MiB, for a GPU to be chosen by
    /// the auto-config path. Below this the card cannot hold a
    /// model this engine serves.
    pub min_gpu_vram_mib: u64,
    /// Fraction of the chosen GPU's VRAM the auto-config path may
    /// budget for weights and KV state; the rest stays free for
    /// the driver and other processes.
    pub vram_budget_fraction: f32,
    /// Operator switch for the auto-config path: when false,
    /// selection is CPU even on a machine with a capable GPU.
    pub prefer_gpu: bool,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            gpu_policy: GpuPolicy::Auto,
            allow_non_discrete_gpu: false,
            vram_reserve_bytes: DEFAULT_VRAM_RESERVE_BYTES,
            min_gpu_vram_mib: DEFAULT_MIN_GPU_VRAM_MIB,
            vram_budget_fraction: DEFAULT_VRAM_BUDGET_FRACTION,
            prefer_gpu: true,
        }
    }
}

impl BackendConfig {
    /// Configuration with the GPU backend disabled: the CPU-only
    /// serving proof of TODO T-4.1.
    #[must_use]
    pub fn cpu_only() -> Self {
        Self {
            gpu_policy: GpuPolicy::Disabled,
            ..Self::default()
        }
    }
}
