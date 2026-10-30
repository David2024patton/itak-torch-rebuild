//! K8: embedding gather, logit softcap and scale. Not yet
//! implemented.
//!
//! The embedding gather reads one 6,656-wide row per token from
//! the Q4_K table (the gather itself is the kernel; the table's
//! rows are K1-shaped data). The softcap (20.0) and the logit
//! scale (0.19611613) come from the arch module and fuse into
//! K3's epilogue when K3 lands; the forward applies them on host
//! today (`muse_glimmer::softcap_scale`), so this module owns
//! the gather refusal only.

use muse_glimmer::{GlimmerError, Result};

/// Refuse `step` because it needs the embedding gather kernel.
pub(crate) fn refuse<T>(step: &'static str) -> Result<T> {
    Err(GlimmerError::StepFailed {
        step,
        detail: "kernel K8 (embedding gather, BACKEND.md Section 3) is not \
                 implemented yet; embedding lookup cannot run on the wgpu backend \
                 until it lands"
            .to_string(),
    })
}
