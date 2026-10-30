//! K4: RMSNorm fused with the residual add. Not yet implemented.
//!
//! The bytes are trivial (the 6,656-wide hidden vector, twice per
//! layer); the point is fusion, so the hidden vector is not
//! re-read from VRAM between the residual add and the norm. This
//! module will also carry the weightless variant the embedding
//! norm uses (`rms_norm_plain`). Until it lands, both norm steps
//! refuse here by name.

use muse_glimmer::{GlimmerError, Result};

/// Refuse `step` because it needs the fused norm kernel.
pub(crate) fn refuse<T>(step: &'static str) -> Result<T> {
    Err(GlimmerError::StepFailed {
        step,
        detail: "kernel K4 (RMSNorm fused with the residual add, BACKEND.md Section 3) \
                 is not implemented yet; norm steps cannot run on the wgpu backend \
                 until it lands"
            .to_string(),
    })
}
