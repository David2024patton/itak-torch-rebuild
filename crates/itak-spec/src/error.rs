//! Named refusals for the speculative decoding loop (SPEC-itak-spec Section
//! 6.1). Every variant is a state the loop refuses by name; nothing in this
//! crate fails silently, panics on bad input, or falls back without an event.

use dflash::DflashError;
use muse_glimmer::GlimmerError;

/// Every way session creation or a round can refuse, named.
#[derive(Debug)]
pub enum SpecError {
    /// The chosen drafting mode's machinery does not exist in this build.
    /// N-gram needs its own state machinery and Medusa needs its heads
    /// loader; neither is in this crate yet, and asking for either refuses
    /// here instead of silently running something else.
    ModeUnavailable {
        /// The mode name as it appears in configuration (`ngram`, `medusa`).
        mode: &'static str,
    },
    /// DFlash mode was selected but no draft model was loaded.
    DraftNotLoaded,
    /// The attached draft declares an architecture other than `dflash`.
    DraftArchMismatch {
        /// The architecture string the draft file declares.
        found: String,
    },
    /// The draft model failed to build from its file.
    DraftLoadFailed {
        /// Why the load failed.
        detail: String,
    },
    /// The caller's speculation envelope (rounds, cache budget) does not
    /// match what this session was configured with.
    EnvelopeMismatch {
        /// The envelope field that differs.
        field: &'static str,
        /// What the session was configured with.
        expected: usize,
        /// What the caller asked for.
        actual: usize,
    },
    /// A Medusa heads artifact carries an unsupported format version.
    ArtifactVersionUnsupported {
        /// The version the artifact declares.
        found: u32,
        /// The newest version this build reads.
        supported: u32,
    },
    /// Medusa heads loaded but are all zeros: drafting with them would be
    /// drafting with noise, so the session refuses instead.
    ZeroInitializedHeads,
    /// The configured draft depth is zero or beyond what one draft block
    /// (or the target's largest legal block) can propose.
    DepthOutOfRange {
        /// The configured depth.
        n_max: usize,
        /// The draft block size, which bounds the depth.
        block_size: usize,
    },
    /// A KV checkpoint is already live; rounds never nest checkpoints.
    CheckpointBusy,
    /// The cache position after a round disagrees with what the loop
    /// committed: the loop and the cache have drifted apart, and the
    /// session refuses to continue on a lying cache.
    CacheOutOfSync {
        /// The position the loop committed to.
        expected: usize,
        /// The position the cache reports.
        actual: usize,
    },
    /// Grammar-constrained sampling and speculation are both active. The
    /// correction walk assumes the full vocabulary; enabling both refuses
    /// until the grammar-aware walk exists.
    GrammarConflict,
    /// The target's verify forward refused mid-round. Under the default
    /// error policy the cache has already been rewound to the round's base
    /// and the checkpoint released when this surfaces.
    VerifyRefused {
        /// The target-side refusal.
        source: Box<GlimmerError>,
    },
    /// The draft side refused: a block forward, a conditioning injection,
    /// or a commit-time sync failure in the draft's own cache.
    DraftRefused {
        /// The draft-side refusal.
        source: Box<DflashError>,
    },
    /// A target forward outside the verify step refused (prefill, or a
    /// plain non-speculative step).
    TargetRefused {
        /// The target-side refusal.
        source: Box<GlimmerError>,
    },
    /// A target KV checkpoint, rewind, or release refused.
    CacheRefused {
        /// Which cache operation refused (`checkpoint`, `rewind`, `release`).
        op: &'static str,
        /// The cache-side refusal.
        source: Box<GlimmerError>,
    },
    /// The draft and the target disagree on a shared geometric fact, so
    /// captures and distributions could never line up.
    DraftTargetMismatch {
        /// The field that disagrees (`vocab_size`, `target_layer`).
        field: &'static str,
        /// The target's value.
        expected: usize,
        /// The draft's value.
        actual: usize,
    },
    /// One round needs `depth + 1` checkpoint slots (the pending token plus
    /// the drafts) but the round budget allows fewer.
    RoundBoundTooSmall {
        /// The configured depth.
        depth: usize,
        /// The configured per-round token bound.
        max_round_tokens: usize,
    },
    /// The session was driven out of order: a round before `begin`, a
    /// second `begin`, a round after the session stopped, or a draft commit
    /// whose staged captures do not cover the committed tokens.
    SessionState {
        /// What was wrong with the call order or staged state.
        detail: String,
    },
    /// A sampler entry point was handed inputs that violate its stated
    /// contract: an empty logit row, a correction walk whose target rows
    /// do not cover the proposal plus the bonus row, or a proposal token
    /// outside the rows it indexes. The sampler refuses by name instead
    /// of panicking on the indexing (panel B4, executed probes).
    SamplerContract {
        /// The entry point that refused (`distribution`,
        /// `correct_greedy`, `correct_sampled`).
        entry: &'static str,
        /// What was wrong with the inputs.
        detail: String,
    },
}

impl std::fmt::Display for SpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpecError::ModeUnavailable { mode } => {
                write!(f, "drafting mode {mode} is not available in this build")
            }
            SpecError::DraftNotLoaded => {
                write!(f, "speculation mode requires a loaded draft model")
            }
            SpecError::DraftArchMismatch { found } => {
                write!(f, "draft architecture is {found}, expected dflash")
            }
            SpecError::DraftLoadFailed { detail } => {
                write!(f, "draft model failed to load: {detail}")
            }
            SpecError::EnvelopeMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "speculation envelope mismatch on {field}: session has {expected}, caller asked {actual}"
            ),
            SpecError::ArtifactVersionUnsupported { found, supported } => write!(
                f,
                "draft artifact version {found} is not supported (newest supported: {supported})"
            ),
            SpecError::ZeroInitializedHeads => {
                write!(f, "draft heads are zero-initialized; refusing to draft with noise")
            }
            SpecError::DepthOutOfRange { n_max, block_size } => write!(
                f,
                "draft depth {n_max} is out of range for draft block size {block_size}"
            ),
            SpecError::CheckpointBusy => {
                write!(f, "a KV checkpoint is already live; rounds never nest checkpoints")
            }
            SpecError::CacheOutOfSync { expected, actual } => write!(
                f,
                "cache position {actual} does not match the committed position {expected}"
            ),
            SpecError::GrammarConflict => write!(
                f,
                "grammar-constrained sampling and speculation cannot run together yet"
            ),
            SpecError::VerifyRefused { source } => {
                write!(f, "verify forward refused: {source}")
            }
            SpecError::DraftRefused { source } => {
                write!(f, "draft refused: {source}")
            }
            SpecError::TargetRefused { source } => {
                write!(f, "target forward refused: {source}")
            }
            SpecError::CacheRefused { op, source } => {
                write!(f, "cache {op} refused: {source}")
            }
            SpecError::DraftTargetMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "draft/target mismatch on {field}: target has {expected}, draft has {actual}"
            ),
            SpecError::RoundBoundTooSmall {
                depth,
                max_round_tokens,
            } => write!(
                f,
                "round bound {max_round_tokens} cannot hold depth {depth} plus the pending token"
            ),
            SpecError::SessionState { detail } => write!(f, "session state error: {detail}"),
            SpecError::SamplerContract { entry, detail } => {
                write!(f, "sampler contract refused at {entry}: {detail}")
            }
        }
    }
}

impl std::error::Error for SpecError {}

/// The loop's result type.
pub type Result<T> = std::result::Result<T, SpecError>;
