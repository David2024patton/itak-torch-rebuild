//! Error types for the kernels.
//!
//! Design rule (the same one the `gguf` loader was rebuilt under,
//! from the 2026-10-29 audit): every failure is loud and names the
//! thing that failed. A quantized payload whose length is not a
//! whole number of blocks is an error here, never a silently
//! truncated dequantization: a kernel never indexes a payload field
//! it has not checked (ARCHITECTURE.md, live test F3).

use std::fmt;

/// Result alias for kernel operations.
pub type Result<T> = std::result::Result<T, KernelError>;

/// Everything a kernel call can refuse on.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum KernelError {
    /// The byte length of a quantized payload is not a whole number
    /// of blocks of its format. This is the malformed-payload error:
    /// the kernel stops here rather than decoding a partial block
    /// into wrong numbers.
    MalformedBlockLength {
        /// Name of the operation that rejected the payload.
        op: &'static str,
        /// The byte length it was given.
        actual: usize,
        /// Bytes per block of the format; `actual` must be a
        /// multiple of this.
        block_bytes: usize,
    },
    /// A caller-provided output buffer did not have exactly the
    /// length the operation writes.
    OutputLengthMismatch {
        /// Name of the operation that rejected the buffer.
        op: &'static str,
        /// The length the operation requires.
        expected: usize,
        /// The length it was given.
        actual: usize,
    },
    /// Two slices that must have equal length did not.
    LengthMismatch {
        /// Name of the operation that rejected the input.
        op: &'static str,
        /// The length the operation required.
        expected: usize,
        /// The length it was given.
        actual: usize,
    },
    /// A flat buffer did not match its declared shape.
    ShapeMismatch {
        /// Name of the operation that rejected the input.
        op: &'static str,
        /// Element or byte count implied by the declared shape.
        expected: usize,
        /// Count of the buffer actually passed.
        actual: usize,
    },
    /// A configuration parameter or a shape parameter was out of
    /// range (including a shape product that overflows `usize`).
    InvalidConfig {
        /// Name of the operation that rejected the value.
        op: &'static str,
        /// Name of the offending parameter.
        param: &'static str,
        /// Why the value is invalid.
        reason: &'static str,
    },
    /// A call's shapes imply more elements than the configured
    /// ceiling allows. See [`crate::KernelConfig`].
    LimitExceeded {
        /// Name of the operation that refused.
        op: &'static str,
        /// The configured ceiling.
        limit: usize,
        /// The element count the call would have covered.
        actual: usize,
    },
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MalformedBlockLength {
                op,
                actual,
                block_bytes,
            } => write!(
                f,
                "{op}: malformed payload, {actual} bytes is not a whole number of {block_bytes}-byte blocks"
            ),
            Self::OutputLengthMismatch {
                op,
                expected,
                actual,
            } => write!(
                f,
                "{op}: output buffer length mismatch, expected {expected}, got {actual}"
            ),
            Self::LengthMismatch {
                op,
                expected,
                actual,
            } => write!(
                f,
                "{op}: length mismatch, expected {expected}, got {actual}"
            ),
            Self::ShapeMismatch {
                op,
                expected,
                actual,
            } => write!(
                f,
                "{op}: buffer holds {actual}, shape requires {expected}"
            ),
            Self::InvalidConfig { op, param, reason } => {
                write!(f, "{op}: invalid {param}: {reason}")
            }
            Self::LimitExceeded { op, limit, actual } => write!(
                f,
                "{op}: element count {actual} exceeds configured limit {limit}"
            ),
        }
    }
}

impl std::error::Error for KernelError {}
