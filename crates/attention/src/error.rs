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
    /// A checkpoint was requested while another checkpoint is still
    /// outstanding on the same cache. One checkpoint per cache at a
    /// time; release it before taking the next (SPEC-itak-spec
    /// Section 5.2).
    CheckpointBusy,
    /// A checkpoint handle does not name the cache's live checkpoint:
    /// it was already released, it belongs to a different cache, or
    /// no checkpoint is outstanding. Rewinding or releasing through a
    /// stale handle is refused, never applied to the wrong snapshot.
    StaleCheckpoint {
        /// The stale checkpoint's serial number.
        checkpoint: u64,
    },
    /// The checkpoint round bound exceeds the smallest sliding
    /// window in the cache's pattern. A round that long would write
    /// some ring slot twice, and no restore could be exact
    /// (SPEC-itak-spec Section 5.2 bounds a round by the window).
    CheckpointTooDeep {
        /// The round bound that was requested.
        max_round_tokens: usize,
        /// The smallest windowed window in the pattern.
        window: usize,
    },
    /// While a checkpoint is live, an append tried to land at or
    /// beyond the first position the checkpoint did not save. The
    /// append is refused so a later rewind can never meet a slot it
    /// cannot restore.
    CheckpointRangeExceeded {
        /// The layer being appended to.
        layer: usize,
        /// The position that was refused.
        position: usize,
        /// The first position outside the checkpoint's saved range.
        limit: usize,
    },
    /// A rewind target outside `[base, appended]` for a layer: below
    /// the checkpoint's recorded base, or ahead of what the layer has
    /// actually appended. The rewind is refused whole; no layer is
    /// partially restored.
    RewindOutOfRange {
        /// The first layer the target is invalid for.
        layer: usize,
        /// The layer's position when the checkpoint was taken.
        base: usize,
        /// The layer's current appended count.
        appended: usize,
        /// The target that was requested.
        target: usize,
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
            Self::CheckpointBusy => {
                write!(
                    f,
                    "a checkpoint is already outstanding on this cache; release it before taking another"
                )
            }
            Self::StaleCheckpoint { checkpoint } => {
                write!(
                    f,
                    "checkpoint {checkpoint} is not this cache's live checkpoint (already released or foreign)"
                )
            }
            Self::CheckpointTooDeep {
                max_round_tokens,
                window,
            } => write!(
                f,
                "checkpoint round bound {max_round_tokens} exceeds the smallest sliding window {window}: a round that long would overwrite a ring slot twice and no rewind could be exact"
            ),
            Self::CheckpointRangeExceeded {
                layer,
                position,
                limit,
            } => write!(
                f,
                "layer {layer} append at position {position} is outside the live checkpoint's saved range (first unsaved position {limit}); refusing so a later rewind cannot meet a slot it cannot restore"
            ),
            Self::RewindOutOfRange {
                layer,
                base,
                appended,
                target,
            } => write!(
                f,
                "rewind target {target} is outside layer {layer}'s checkpoint range [{base}, {appended}]"
            ),
        }
    }
}

impl std::error::Error for AttentionError {}
