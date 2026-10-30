//! K2: Q6_K dequant GEMV. Not yet implemented.
//!
//! Second-largest stream in Glimmer (17.1 percent of bytes): the
//! FFN down projection (19,968 to 6,656) and the attention V
//! projection. Its input is the FFN intermediate, which at 19,968
//! wide exceeds the 64 KiB workgroup storage in f32; BACKEND.md
//! Section 4 leaves the staging choice (f16 in LDS versus global
//! memory through the Infinity Cache) to this kernel's bench.
//! Until it lands, every Q6_K projection refuses here by name.

use muse_glimmer::{GlimmerError, Result};

/// Refuse `step` because it needs the Q6_K kernel.
pub(crate) fn refuse<T>(step: &'static str) -> Result<T> {
    Err(GlimmerError::StepFailed {
        step,
        detail: "kernel K2 (Q6_K dequant GEMV, BACKEND.md Section 3) is not implemented yet; \
                 Q6_K projections (FFN down, attention V) cannot run on the wgpu backend \
                 until it lands"
            .to_string(),
    })
}
