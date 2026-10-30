//! Error types for the loader bridge.
//!
//! Design rule (carried from the 2026-10-29 audit of the original Go
//! engine): every failure is loud and names the thing that failed. The
//! coverage failure names every missing tensor, not just the first,
//! because an inventory check that dribbles out one name per run is a
//! debugging tax.

use std::fmt;

use gguf::GgufError;
use muse_glimmer::GlimmerError;

/// Result alias for loader operations.
pub type Result<T> = std::result::Result<T, LoaderError>;

/// Everything that can go wrong while bridging a GGUF file into an
/// architecture module.
#[derive(Debug)]
pub enum LoaderError {
    /// The tensor inventory check found expected tensors absent from
    /// the file. Carries the complete missing list, in the arch
    /// module's expected order.
    MissingTensors(Vec<String>),
    /// The architecture module refused something (bad config, a tensor
    /// it could not take). Passed through unwrapped in kind so callers
    /// can match on the arch module's own error variants.
    Glimmer(GlimmerError),
    /// The dflash module refused something. Same pass-through rule
    /// as [`LoaderError::Glimmer`].
    Dflash(dflash::DflashError),
    /// The GGUF loader refused something (parse or read failure).
    Gguf(GgufError),
}

impl fmt::Display for LoaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoaderError::MissingTensors(names) => write!(
                f,
                "tensor inventory incomplete: {} expected tensor(s) absent: {}",
                names.len(),
                names.join(", ")
            ),
            LoaderError::Glimmer(e) => write!(f, "{e}"),
            LoaderError::Dflash(e) => write!(f, "{e}"),
            LoaderError::Gguf(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for LoaderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoaderError::MissingTensors(_) => None,
            LoaderError::Glimmer(e) => Some(e),
            LoaderError::Dflash(e) => Some(e),
            LoaderError::Gguf(e) => Some(e),
        }
    }
}

impl From<GlimmerError> for LoaderError {
    fn from(e: GlimmerError) -> Self {
        Self::Glimmer(e)
    }
}

impl From<dflash::DflashError> for LoaderError {
    fn from(e: dflash::DflashError) -> Self {
        Self::Dflash(e)
    }
}

impl From<GgufError> for LoaderError {
    fn from(e: GgufError) -> Self {
        Self::Gguf(e)
    }
}
