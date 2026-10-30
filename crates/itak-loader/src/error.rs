//! Error types for the loader bridge.
//!
//! Design rule (carried from the 2026-10-29 audit of the original Go
//! engine): every failure is loud and names the thing that failed. The
//! coverage failure names every missing tensor, not just the first,
//! because an inventory check that dribbles out one name per run is a
//! debugging tax.

use std::fmt;

use arch_llama::LlamaError;
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
    /// The llama-family architecture module refused something (bad
    /// config, a tensor it could not take). Passed through unwrapped
    /// in kind, same as the Glimmer variant.
    Llama(LlamaError),
    /// The GGUF loader refused something (parse or read failure).
    Gguf(GgufError),
    /// The file's own `general.architecture` string names no module
    /// in the arch registry. Carries the string the file declared and
    /// the complete supported list, so the refusal itself teaches the
    /// caller what this engine loads (ARCHITECTURE.md 5.3: an
    /// unrecognized string is a named refusal listing what is
    /// supported).
    UnknownArchitecture {
        /// The architecture string the file declared.
        found: String,
        /// Every architecture string the registry can route, sorted.
        supported: Vec<String>,
    },
    /// A second module tried to register an architecture string that
    /// already has one. The registry is the only load path, so a
    /// string must route to exactly one module.
    DuplicateArchitecture {
        /// The architecture string that was registered twice.
        architecture: String,
    },
    /// Engine selection found no candidate able to serve an
    /// architecture by capability (the module provides no forward, no
    /// available backend lists the architecture). Selection is by
    /// capability, never by file size; the detail states which
    /// capability check failed.
    NoCapableEngine {
        /// The architecture that could not be served.
        architecture: String,
        /// Why every candidate was refused, in plain words.
        detail: String,
    },
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
            LoaderError::Llama(e) => write!(f, "{e}"),
            LoaderError::Gguf(e) => write!(f, "{e}"),
            LoaderError::UnknownArchitecture { found, supported } => write!(
                f,
                "unsupported architecture \"{found}\": this engine loads [{}]",
                supported.join(", ")
            ),
            LoaderError::DuplicateArchitecture { architecture } => write!(
                f,
                "architecture \"{architecture}\" is already registered: \
                 one architecture string routes to exactly one module"
            ),
            LoaderError::NoCapableEngine {
                architecture,
                detail,
            } => write!(
                f,
                "no capable engine for architecture \"{architecture}\": {detail}"
            ),
        }
    }
}

impl std::error::Error for LoaderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoaderError::MissingTensors(_) => None,
            LoaderError::Glimmer(e) => Some(e),
            LoaderError::Llama(e) => Some(e),
            LoaderError::Gguf(e) => Some(e),
            LoaderError::UnknownArchitecture { .. } => None,
            LoaderError::DuplicateArchitecture { .. } => None,
            LoaderError::NoCapableEngine { .. } => None,
        }
    }
}

impl From<GlimmerError> for LoaderError {
    fn from(e: GlimmerError) -> Self {
        Self::Glimmer(e)
    }
}

impl From<LlamaError> for LoaderError {
    fn from(e: LlamaError) -> Self {
        Self::Llama(e)
    }
}

impl From<GgufError> for LoaderError {
    fn from(e: GgufError) -> Self {
        Self::Gguf(e)
    }
}
