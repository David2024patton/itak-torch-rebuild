//! Error types for the wgpu backend.
//!
//! Same rule as the rest of the rebuild: every failure is loud and
//! names the thing that failed. Device selection failures list every
//! adapter that was seen and why none qualified; tensor refusals
//! carry the tensor name and both byte counts; a kernel that has not
//! landed yet is a named refusal, never a substituted result.

use std::fmt;

/// Everything that can go wrong while selecting a device, uploading
/// a tensor, or running a kernel in this crate.
#[derive(Debug)]
pub enum WgpuError {
    /// No adapter satisfied the selection policy. `seen` describes
    /// every adapter wgpu enumerated, so the refusal shows whether
    /// the card was absent, present but the wrong kind, or present
    /// only as a software rasterizer (which BACKEND.md Section 6
    /// counts as no adapter at all).
    NoSuitableAdapter {
        /// One line per enumerated adapter: name, backend, type.
        seen: Vec<String>,
        /// Whether the policy in force allowed non-discrete GPUs.
        allow_non_discrete: bool,
    },
    /// The selected adapter could not produce a device.
    DeviceRequest {
        /// The wgpu failure, in its own words.
        detail: String,
    },
    /// The granted device limits fall below what a kernel or a
    /// tensor requires. This is BACKEND.md Section 6's fallback
    /// event, surfaced as an error instead of a surprise mid-load.
    LimitShortfall {
        /// Which limit fell short.
        what: &'static str,
        /// Bytes (or units) the work requires.
        required: u64,
        /// What the device actually granted.
        granted: u64,
    },
    /// A tensor's declared geometry and its bytes disagree, or its
    /// shape is not one the addressed kernel can run.
    InvalidTensor {
        /// The full GGUF tensor name.
        name: String,
        /// What is wrong, with both numbers where numbers differ.
        detail: String,
    },
    /// Shader compilation or pipeline creation failed.
    Pipeline {
        /// The wgpu failure, in its own words.
        detail: String,
    },
    /// A buffer map (readback) failed or returned an error.
    Readback {
        /// The wgpu failure, in its own words.
        detail: String,
    },
}

impl fmt::Display for WgpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WgpuError::NoSuitableAdapter {
                seen,
                allow_non_discrete,
            } => {
                write!(
                    f,
                    "no suitable GPU adapter (allow_non_discrete={allow_non_discrete}); enumerated: [{}]",
                    seen.join("; ")
                )
            }
            WgpuError::DeviceRequest { detail } => {
                write!(f, "request_device failed: {detail}")
            }
            WgpuError::LimitShortfall {
                what,
                required,
                granted,
            } => {
                write!(
                    f,
                    "device limit {what} too small: need {required}, granted {granted}"
                )
            }
            WgpuError::InvalidTensor { name, detail } => {
                write!(f, "tensor {name} is not runnable as given: {detail}")
            }
            WgpuError::Pipeline { detail } => {
                write!(f, "shader or pipeline creation failed: {detail}")
            }
            WgpuError::Readback { detail } => {
                write!(f, "buffer readback failed: {detail}")
            }
        }
    }
}

impl std::error::Error for WgpuError {}

/// Result alias for wgpu backend operations.
pub type Result<T> = std::result::Result<T, WgpuError>;
