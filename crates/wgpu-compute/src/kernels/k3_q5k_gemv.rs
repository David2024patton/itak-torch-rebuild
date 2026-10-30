//! K3: Q5_K dequant GEMV. Not yet implemented.
//!
//! The untied output head: 6,656 to 202,048, about 0.86 GiB, 5.5
//! percent of the stream, running every token. Its 202,048 rows
//! exceed the 65,535-workgroup dispatch cap, which is why the K1
//! grid scheme is already two-dimensional. The logit softcap and
//! scale fuse into this kernel's epilogue when it lands (K8).
//! Until then, `lm_head` refuses here by name.

use muse_glimmer::{GlimmerError, Result};

/// Refuse `step` because it needs the Q5_K kernel.
pub(crate) fn refuse<T>(step: &'static str) -> Result<T> {
    Err(GlimmerError::StepFailed {
        step,
        detail: "kernel K3 (Q5_K dequant GEMV for the output head, BACKEND.md Section 3) \
                 is not implemented yet; the Q5_K head cannot run on the wgpu backend \
                 until it lands"
            .to_string(),
    })
}
