//! K6: GQA decode attention. Not yet implemented.
//!
//! 32 Q heads over 2 KV heads, head dim 128, in two variants: the
//! ring variant for the 39 sliding-window layers (window 2,048)
//! and a paged split-K variant with online softmax for the 13
//! global layers (up to 1.66 GiB of KV reads per token at the full
//! 131K context). It consumes K5's rotated, appended KV. Until K5
//! and K6 land, the `attention` step refuses here by name.

use muse_glimmer::{GlimmerError, Result};

/// Refuse `step` because it needs the attention kernels.
pub(crate) fn refuse<T>(step: &'static str) -> Result<T> {
    Err(GlimmerError::StepFailed {
        step,
        detail: "kernels K5 (RoPE plus KV append) and K6 (GQA decode attention), \
                 BACKEND.md Section 3, are not implemented yet; attention cannot run \
                 on the wgpu backend until they land"
            .to_string(),
    })
}
