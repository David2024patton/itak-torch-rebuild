//! Error types for the dflash module.
//!
//! Design rule (carried from the 2026-10-29 audit of the original Go
//! engine): every failure is loud and names the thing that failed. A
//! missing metadata key, a missing tensor, a conditioning input of the
//! wrong width, or a file carrying a DFlash variant this module does
//! not implement is a named error, never a fallback value and never a
//! silently skipped feature.

use std::fmt;

/// Result alias for dflash operations.
pub type Result<T> = std::result::Result<T, DflashError>;

/// Everything that can go wrong while configuring, loading, or running
/// a DFlash draft model.
#[derive(Debug)]
pub enum DflashError {
    /// A metadata key the config requires is absent from the file.
    MissingMetadata {
        /// The full GGUF metadata key that was required.
        key: String,
    },
    /// A metadata key is present but its value cannot describe a model
    /// this module can run (impossible head counts, a sliding pattern
    /// whose length disagrees with the layer count, and so on).
    InvalidMetadata {
        /// The full GGUF metadata key at fault.
        key: String,
        /// What is wrong with the value.
        detail: String,
    },
    /// `general.architecture` names a different architecture. This
    /// module serves exactly one arch string and refuses all others.
    WrongArchitecture {
        /// The architecture string found in the file.
        found: String,
    },
    /// The file carries a DFlash-family feature this module does not
    /// implement (a DFlash2 selector, a DSpark Markov head, a `d2t`
    /// vocabulary map, a non-silu hidden activation). Refused by name
    /// rather than loaded-and-ignored: running the plain graph over
    /// such a file would produce confident garbage.
    UnsupportedFeature {
        /// The tensor or metadata key that carries the feature.
        name: String,
        /// Which feature it is and why it is refused.
        detail: String,
    },
    /// A tensor the forward needs is not in the weight source.
    MissingTensor {
        /// The full GGUF tensor name that was required.
        name: String,
    },
    /// A tensor is present in the weight source but cannot be
    /// delivered as stored (for example a ggml type the source
    /// cannot dequantize). Kept distinct from
    /// [`DflashError::MissingTensor`]: the name exists, the bytes
    /// are unusable.
    TensorUnreadable {
        /// The full GGUF tensor name that was requested.
        name: String,
        /// Why the source refused it, in the source's own words.
        detail: String,
    },
    /// A tensor is present but its dimensions disagree with the shape
    /// the config requires for it.
    ShapeMismatch {
        /// The full GGUF tensor name.
        name: String,
        /// Dimensions the config requires, in GGUF stored order.
        expected: Vec<u64>,
        /// Dimensions the source actually carries, in GGUF stored order.
        actual: Vec<u64>,
    },
    /// A tensor's dimensions imply one element count but its data
    /// holds another, so the tensor cannot be what its name claims.
    TensorDataLength {
        /// The full GGUF tensor name.
        name: String,
        /// Element count implied by the dimensions.
        expected: u64,
        /// Element count actually present in the data.
        actual: usize,
    },
    /// The target tensors the draft borrows (embedding table, output
    /// head) are absent: this draft file carries neither of its own,
    /// so a forward without caller-supplied target tensors has
    /// nothing lawful to read. Named separately from
    /// [`DflashError::MissingTensor`] because the tensor was never
    /// expected in the draft file itself.
    MissingTargetTensors {
        /// Which borrowed tensor is missing (`token_embd.weight` or
        /// `output.weight`, in the target file's naming).
        which: &'static str,
    },
    /// Conditioning features handed to the injection pass have the
    /// wrong row width: the fused input must be exactly the `fc`
    /// projection's input width (target layer count times the
    /// target's hidden size).
    ConditioningWidth {
        /// Row width the loaded `fc.weight` requires.
        expected: usize,
        /// Row width the caller supplied.
        actual: usize,
    },
    /// An injection or block position is not usable: positions must
    /// be strictly increasing within one call. (Re-injecting a
    /// position a cache already holds is not an error: it overwrites
    /// that position's K/V, which is what llama.cpp's cache does
    /// when the target reprocesses a position after a rewind.)
    PositionConflict {
        /// The offending absolute position.
        position: usize,
        /// What is wrong with it.
        detail: String,
    },
    /// A forward or injection was called with zero rows.
    EmptyInput,
    /// A token id is outside the vocabulary.
    TokenOutOfRange {
        /// The offending token id.
        token: u32,
        /// The vocabulary size.
        vocab_size: u32,
    },
    /// A block handed to the drafting forward is longer than the
    /// file's `dflash.block_size`: the draft was built for blocks of
    /// at most that many positions.
    BlockTooLong {
        /// Block length the caller supplied.
        len: usize,
        /// The file's block size.
        block_size: u32,
    },
    /// A pass would run past the model's context length.
    SequenceTooLong {
        /// Total sequence length the pass would reach.
        total: usize,
        /// The model's context length, from its metadata.
        context_length: u32,
    },
    /// A torch-math primitive refused its inputs.
    Math(torch_math::MathError),
}

impl fmt::Display for DflashError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingMetadata { key } => {
                write!(f, "required GGUF metadata key is absent: {key}")
            }
            Self::InvalidMetadata { key, detail } => {
                write!(f, "GGUF metadata key {key} is invalid: {detail}")
            }
            Self::WrongArchitecture { found } => write!(
                f,
                "this module serves architecture \"dflash\" only, file declares \"{found}\""
            ),
            Self::UnsupportedFeature { name, detail } => {
                write!(f, "unsupported DFlash feature at {name}: {detail}")
            }
            Self::MissingTensor { name } => {
                write!(
                    f,
                    "required tensor is absent from the weight source: {name}"
                )
            }
            Self::TensorUnreadable { name, detail } => {
                write!(
                    f,
                    "tensor {name} is present in the weight source but unreadable: {detail}"
                )
            }
            Self::ShapeMismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "tensor {name}: shape {actual:?} does not match the shape the config requires {expected:?}"
            ),
            Self::TensorDataLength {
                name,
                expected,
                actual,
            } => write!(
                f,
                "tensor {name}: dimensions imply {expected} elements but the data holds {actual}"
            ),
            Self::MissingTargetTensors { which } => write!(
                f,
                "the draft file carries no {which} of its own, and no target tensor was supplied to borrow"
            ),
            Self::ConditioningWidth { expected, actual } => write!(
                f,
                "conditioning rows are {actual} wide but the fused projection requires {expected} (target layers x target hidden)"
            ),
            Self::PositionConflict { position, detail } => {
                write!(f, "position {position}: {detail}")
            }
            Self::EmptyInput => write!(f, "called with zero rows"),
            Self::TokenOutOfRange { token, vocab_size } => write!(
                f,
                "token id {token} is outside the vocabulary of {vocab_size}"
            ),
            Self::BlockTooLong { len, block_size } => write!(
                f,
                "block of {len} positions exceeds the file's block_size of {block_size}"
            ),
            Self::SequenceTooLong {
                total,
                context_length,
            } => write!(
                f,
                "sequence would reach {total} tokens, past the model's context length of {context_length}"
            ),
            Self::Math(e) => write!(f, "math: {e}"),
        }
    }
}

impl std::error::Error for DflashError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Math(e) => Some(e),
            _ => None,
        }
    }
}

impl From<torch_math::MathError> for DflashError {
    fn from(e: torch_math::MathError) -> Self {
        Self::Math(e)
    }
}
