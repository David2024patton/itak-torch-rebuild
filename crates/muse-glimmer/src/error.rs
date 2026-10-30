//! Error types for the muse-glimmer module.
//!
//! Design rule (carried from the 2026-10-29 audit of the original Go
//! engine): every failure is loud and names the thing that failed. The
//! original loader silently substituted pseudo-random weights for any
//! tensor it could not read; that defect class is banned here. A missing
//! metadata key, a missing tensor, or a forward step that has no
//! implementation yet is a named error, never a fallback value.

use std::fmt;

/// Result alias for muse-glimmer operations.
pub type Result<T> = std::result::Result<T, GlimmerError>;

/// Everything that can go wrong while configuring, loading, or running
/// the Muse Glimmer architecture module.
#[derive(Debug)]
pub enum GlimmerError {
    /// A metadata key the config requires is absent from the file.
    MissingMetadata {
        /// The full GGUF metadata key that was required.
        key: &'static str,
    },
    /// A metadata key is present but its value cannot describe a model
    /// (wrong length pattern, impossible head counts, and so on).
    InvalidMetadata {
        /// The full GGUF metadata key at fault.
        key: &'static str,
        /// What is wrong with the value.
        detail: String,
    },
    /// `general.architecture` names a different architecture. This module
    /// serves exactly one arch string and refuses all others.
    WrongArchitecture {
        /// The architecture string found in the file.
        found: String,
    },
    /// A tensor the forward needs is not in the weight source. The old
    /// engine kept random placeholder weights in this situation; here it
    /// is an error, always.
    MissingTensor {
        /// The full GGUF tensor name that was required.
        name: String,
    },
    /// A tensor is present in the weight source but cannot be delivered
    /// as stored (for example a ggml type the source cannot size or
    /// read). Kept distinct from [`GlimmerError::MissingTensor`]: the
    /// name exists, the bytes are unusable, and conflating the two
    /// sends debugging in the wrong direction.
    TensorUnreadable {
        /// The full GGUF tensor name that was requested.
        name: String,
        /// Why the source refused it, in the source's own words.
        detail: String,
    },
    /// A forward step has no implementation yet. Skeleton stage: the
    /// math lives in the kernel crate, and until a real `Compute`
    /// implementation is wired in, the step refuses by name instead of
    /// returning placeholder numbers.
    Unimplemented {
        /// Stable name of the step that refused (for example `embed`).
        step: &'static str,
    },
    /// A forward step ran and failed: a shape mismatch, an
    /// unsupported tensor dtype, or an error handed up from the
    /// kernel or KV crates. Carries the step name plus the underlying
    /// failure in its own words, so a real backend's refusals stay as
    /// loud and specific as the skeleton's.
    StepFailed {
        /// Stable name of the step that failed (for example
        /// `attention`).
        step: &'static str,
        /// The underlying failure, in the failing component's own
        /// words.
        detail: String,
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
    /// Sampling was asked to draw from an empty logit slice.
    EmptyLogits,
    /// Every logit is negative infinity or NaN, so no probability
    /// distribution exists to draw from. Named instead of letting a
    /// softmax produce NaNs and a draw pick an arbitrary token.
    NoViableToken,
    /// A sampling parameter is outside the range where it means
    /// anything (a negative temperature, a top-p above 1, a zero
    /// repetition penalty). The config is refused whole, never clamped
    /// silently into a different behavior than the caller asked for.
    InvalidSampling {
        /// The config field at fault (for example `temperature`).
        field: &'static str,
        /// What is wrong with the value.
        detail: String,
    },
    /// A dimension measurement was requested with a shape or config
    /// that cannot describe one: a tensor length that does not match
    /// its declared rows and cols, a zero column count, a config
    /// field outside its meaningful range. Named instead of panicking
    /// or returning a number computed from a different shape than the
    /// caller declared.
    InvalidDimension {
        /// What is wrong with the request.
        detail: String,
    },
}

impl fmt::Display for GlimmerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GlimmerError::MissingMetadata { key } => {
                write!(f, "required GGUF metadata key is absent: {key}")
            }
            GlimmerError::InvalidMetadata { key, detail } => {
                write!(f, "GGUF metadata key {key} is invalid: {detail}")
            }
            GlimmerError::WrongArchitecture { found } => {
                write!(
                    f,
                    "this module serves architecture \"muse-glimmer\" only, file declares \"{found}\""
                )
            }
            GlimmerError::MissingTensor { name } => {
                write!(
                    f,
                    "required tensor is absent from the weight source: {name}"
                )
            }
            GlimmerError::TensorUnreadable { name, detail } => {
                write!(
                    f,
                    "tensor {name} is present in the weight source but unreadable: {detail}"
                )
            }
            GlimmerError::Unimplemented { step } => {
                write!(f, "forward step is not implemented yet: {step}")
            }
            GlimmerError::StepFailed { step, detail } => {
                write!(f, "forward step {step} failed: {detail}")
            }
            GlimmerError::EmptyInput => {
                write!(f, "forward called with zero tokens")
            }
            GlimmerError::TokenOutOfRange { token, vocab_size } => {
                write!(
                    f,
                    "token id {token} is outside the vocabulary of {vocab_size}"
                )
            }
            GlimmerError::EmptyLogits => {
                write!(f, "sampling called with zero logits")
            }
            GlimmerError::NoViableToken => {
                write!(
                    f,
                    "every logit is negative infinity or NaN; no token can be drawn"
                )
            }
            GlimmerError::InvalidSampling { field, detail } => {
                write!(f, "sampling config field {field} is invalid: {detail}")
            }
            GlimmerError::InvalidDimension { detail } => {
                write!(f, "dimension measurement request is invalid: {detail}")
            }
        }
    }
}

impl std::error::Error for GlimmerError {}
