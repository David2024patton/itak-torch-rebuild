//! Named errors for the backend crate. Every fallible operation
//! returns one of these; nothing in this crate panics on a shape
//! mismatch, a bad payload, or an out-of-range index supplied by a
//! caller, and nothing silently substitutes a different backend or a
//! placeholder result (the old engine's failure mode, closed by
//! ARCHITECTURE 5.2 and 5.3).

use std::fmt;

use gguf::GgmlType;
use torch_math::MathError;

use crate::report::Op;

/// Everything that can go wrong in a backend operation or in
/// backend selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    /// A math-layer refusal (shape, length, or configuration),
    /// carried through unchanged so the op and parameter that failed
    /// stay named. The scalar ops implemented inside this crate
    /// (softmax, attention, elementwise) report through this variant
    /// too: one error vocabulary for all math.
    Math(MathError),
    /// A quantized payload's byte length disagrees with the exact
    /// length its type and element count require. The payload is
    /// refused before any byte is indexed (ARCHITECTURE 5.2: a
    /// kernel never indexes a payload field it has not checked).
    ByteLengthMismatch {
        /// Name of the operation that rejected the payload.
        op: &'static str,
        /// The payload's declared element type.
        ty: GgmlType,
        /// Byte length the type and element count require.
        expected: u64,
        /// Byte length actually supplied.
        actual: u64,
    },
    /// The element count is not a whole number of quantization
    /// blocks for the type, so no valid byte length exists.
    BadElementCount {
        /// Name of the operation that rejected the count.
        op: &'static str,
        /// The element type whose block size does not divide it.
        ty: GgmlType,
        /// The element count supplied.
        elements: usize,
    },
    /// An index (an embedding token id) outside the table it indexes.
    IndexOutOfRange {
        /// Name of the operation that rejected the index.
        op: &'static str,
        /// The index supplied.
        index: u64,
        /// The number of rows the table actually has.
        len: u64,
    },
    /// A backend was asked for an operation its capability report
    /// does not list. Distinct from a per-type gap: this is a whole
    /// op the backend does not implement.
    UnsupportedOp {
        /// The operation that was requested.
        op: Op,
    },
    /// Selection under a `Required` GPU policy found no suitable
    /// adapter. The engine refuses to start rather than silently
    /// serving on a backend the operator did not accept
    /// (ARCHITECTURE Section 4: a fallback is an explicit state,
    /// never a silent one; a required GPU that is absent is a
    /// refusal, not a fallback).
    NoBackendAvailable {
        /// Why no backend satisfied the policy.
        reason: String,
    },
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Math(e) => write!(f, "{e}"),
            Self::ByteLengthMismatch {
                op,
                ty,
                expected,
                actual,
            } => write!(
                f,
                "{op}: {ty:?} payload holds {actual} bytes, type and element count require {expected}"
            ),
            Self::BadElementCount { op, ty, elements } => write!(
                f,
                "{op}: {elements} elements is not a whole number of {ty:?} blocks"
            ),
            Self::IndexOutOfRange { op, index, len } => {
                write!(f, "{op}: index {index} out of range, table has {len} rows")
            }
            Self::UnsupportedOp { op } => {
                write!(f, "backend does not implement op {}", op.name())
            }
            Self::NoBackendAvailable { reason } => {
                write!(f, "no backend available: {reason}")
            }
        }
    }
}

impl std::error::Error for BackendError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Math(e) => Some(e),
            _ => None,
        }
    }
}

impl From<MathError> for BackendError {
    fn from(e: MathError) -> Self {
        Self::Math(e)
    }
}
