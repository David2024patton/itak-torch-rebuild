//! K7: SwiGLU elementwise activation. Not yet implemented.
//!
//! `silu(gate) * up` over the 19,968-wide FFN intermediate; an
//! epilogue of the up projection when fused, writing the activated
//! intermediate in f16 when K2's down projection will stage it in
//! LDS (BACKEND.md Section 4). The gate and up projections
//! themselves are Q4_K and already run through K1; the FFN step
//! refuses here, at the activation, until K7 lands.

use muse_glimmer::{GlimmerError, Result};

/// Refuse `step` because it needs the SwiGLU kernel.
pub(crate) fn refuse<T>(step: &'static str) -> Result<T> {
    Err(GlimmerError::StepFailed {
        step,
        detail: "kernel K7 (SwiGLU elementwise, BACKEND.md Section 3) is not \
                 implemented yet; the FFN block cannot finish on the wgpu backend \
                 until it lands"
            .to_string(),
    })
}
