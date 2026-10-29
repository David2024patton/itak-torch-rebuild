//! Named errors for the attention crate. Every fallible operation
//! returns one of these; nothing in this crate panics on a shape
//! mismatch or a bad index supplied by a caller.

use std::fmt;

/// Everything that can go wrong in the KV cache or the attention math.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttentionError {
    /// A dimension that must be positive was zero.
    ZeroDimension {
        /// Which dimension was zero.
        what: &'static str,
    },
    /// A sliding-window layer was given a window of zero.
    InvalidWindow {
        /// The offending window size.
        window: usize,
    },
    /// A tensor slice did not have the expected element count.
    ShapeMismatch {
        /// Which tensor or index mismatched.
        what: &'static str,
        /// Element count (or index bound) that was required.
        expected: usize,
        /// Element count (or index) that was supplied.
        actual: usize,
    },
    /// Layer index outside the cache's layer pattern.
    InvalidLayer {
        /// The requested layer.
        layer: usize,
        /// How many layers the cache actually has.
        layer_count: usize,
    },
    /// Appends must arrive in position order per layer.
    OutOfOrderPosition {
        /// The layer being appended to.
        layer: usize,
        /// The position the layer expected next.
        expected: usize,
        /// The position that was supplied.
        actual: usize,
    },
    /// Grouped-query attention needs the query heads to divide evenly
    /// across the KV heads.
    NonDivisibleHeads {
        /// Query head count supplied.
        num_q_heads: usize,
        /// KV head count in the cache.
        num_kv_heads: usize,
    },
    /// A query position has no keys it may attend to (the causal and
    /// window masks excluded everything, e.g. an empty cache).
    NoVisibleKeys {
        /// The query position that found nothing.
        position: usize,
    },
}

impl fmt::Display for AttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension { what } => {
                write!(f, "{what} must be positive, got 0")
            }
            Self::InvalidWindow { window } => {
                write!(f, "sliding window must be positive, got {window}")
            }
            Self::ShapeMismatch {
                what,
                expected,
                actual,
            } => write!(f, "{what}: got {actual}, expected {expected}"),
            Self::InvalidLayer { layer, layer_count } => {
                write!(
                    f,
                    "layer {layer} out of range, cache has {layer_count} layers"
                )
            }
            Self::OutOfOrderPosition {
                layer,
                expected,
                actual,
            } => write!(
                f,
                "layer {layer} appends must be in order: expected position {expected}, got {actual}"
            ),
            Self::NonDivisibleHeads {
                num_q_heads,
                num_kv_heads,
            } => write!(
                f,
                "{num_q_heads} query heads do not divide evenly across {num_kv_heads} KV heads"
            ),
            Self::NoVisibleKeys { position } => {
                write!(f, "query at position {position} has no visible keys")
            }
        }
    }
}

impl std::error::Error for AttentionError {}
