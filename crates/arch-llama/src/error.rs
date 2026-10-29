//! Error types for the arch-llama module.
//!
//! Design rule (carried from the 2026-10-29 audit of the original Go
//! engine): every failure is loud and names the thing that failed. The
//! original loader silently substituted pseudo-random weights for any
//! tensor it could not read, dropped Qwen2's attention biases without a
//! word, and invented a 32000-token vocabulary when the real one was
//! missing. Those defect classes are banned here: a missing metadata
//! key, a missing tensor, or a shape that disagrees with the config is
//! a named error, never a fallback value.

use std::fmt;

/// Result alias for arch-llama operations.
pub type Result<T> = std::result::Result<T, LlamaError>;

/// Everything that can go wrong while configuring, loading, or running
/// a llama-family or qwen2-family model.
#[derive(Debug)]
pub enum LlamaError {
    /// A metadata key the config requires is absent from the file.
    MissingMetadata {
        /// The full GGUF metadata key that was required.
        key: String,
    },
    /// A metadata key is present but its value cannot describe a model
    /// this module can run (impossible head counts, per-layer head
    /// arrays meant for hybrid modules, and so on).
    InvalidMetadata {
        /// The full GGUF metadata key at fault.
        key: String,
        /// What is wrong with the value.
        detail: String,
    },
    /// `general.architecture` names an architecture this module does
    /// not serve. This module serves exactly `llama` and `qwen2`.
    WrongArchitecture {
        /// The architecture string found in the file.
        found: String,
    },
    /// A tensor the forward needs is not in the weight source. The old
    /// engine kept random placeholder weights in this situation; here
    /// it is an error, always.
    MissingTensor {
        /// The full GGUF tensor name that was required.
        name: String,
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
    /// `forward` was called with zero tokens.
    EmptyInput,
    /// A token id is outside the model's vocabulary.
    TokenOutOfRange {
        /// The offending token id.
        token: u32,
        /// The model's vocabulary size.
        vocab_size: u32,
    },
    /// A forward pass would run past the model's context length.
    SequenceTooLong {
        /// Total sequence length the pass would reach.
        total: usize,
        /// The model's context length, from its metadata.
        context_length: u32,
    },
    /// The layers of a KV cache disagree about the next position,
    /// which means the cache was advanced unevenly and cannot be
    /// trusted for this sequence.
    CacheDesync {
        /// The first layer whose position disagreed.
        layer: usize,
        /// The position every layer was expected to be at (layer 0's).
        expected: usize,
        /// The position the disagreeing layer is actually at.
        actual: usize,
    },
    /// The GGUF loader refused something (a tensor it cannot
    /// dequantize, a malformed file, a mistyped metadata value).
    Gguf(gguf::GgufError),
    /// A torch-math primitive refused its inputs.
    Math(torch_math::MathError),
    /// The attention crate refused its inputs or the cache operation.
    Attention(attention::AttentionError),
}

impl fmt::Display for LlamaError {
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
                "this module serves architectures \"llama\" and \"qwen2\" only, file declares \"{found}\""
            ),
            Self::MissingTensor { name } => {
                write!(
                    f,
                    "required tensor is absent from the weight source: {name}"
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
            Self::EmptyInput => write!(f, "forward called with zero tokens"),
            Self::TokenOutOfRange { token, vocab_size } => write!(
                f,
                "token id {token} is outside the vocabulary of {vocab_size}"
            ),
            Self::SequenceTooLong {
                total,
                context_length,
            } => write!(
                f,
                "sequence would reach {total} tokens, past the model's context length of {context_length}"
            ),
            Self::CacheDesync {
                layer,
                expected,
                actual,
            } => write!(
                f,
                "KV cache desync: layer {layer} is at position {actual}, expected {expected}"
            ),
            Self::Gguf(e) => write!(f, "gguf loader: {e}"),
            Self::Math(e) => write!(f, "math: {e}"),
            Self::Attention(e) => write!(f, "attention: {e}"),
        }
    }
}

impl std::error::Error for LlamaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gguf(e) => Some(e),
            Self::Math(e) => Some(e),
            Self::Attention(e) => Some(e),
            _ => None,
        }
    }
}

impl From<gguf::GgufError> for LlamaError {
    fn from(e: gguf::GgufError) -> Self {
        // A tensor the loader cannot find is this module's own named
        // error, so callers see one vocabulary for the common case.
        match e {
            gguf::GgufError::MissingTensor(name) => Self::MissingTensor { name },
            other => Self::Gguf(other),
        }
    }
}

impl From<torch_math::MathError> for LlamaError {
    fn from(e: torch_math::MathError) -> Self {
        Self::Math(e)
    }
}

impl From<attention::AttentionError> for LlamaError {
    fn from(e: attention::AttentionError) -> Self {
        Self::Attention(e)
    }
}
