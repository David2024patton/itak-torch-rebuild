//! itak-spec: the speculative decoding loop of the iTaK Torch engine
//! (SPEC-itak-spec). A [`SpecSession`] runs DFlash block drafting against
//! the Glimmer target with exact-output verification.
//!
//! One round, per the spec's Section 4: the draft proposes up to `depth`
//! tokens from the last committed position; the target verifies the whole
//! block in one `forward_block` under a KV checkpoint; the correction
//! walk in [`sampler`] accepts the prefix the target agrees with and
//! emits one correction or bonus token; the cache rewinds to the
//! committed count; the draft is told exactly the newly committed tokens
//! and receives their captured layer-input rows as conditioning.
//!
//! The guarantee that matters: at temperature 0 the emitted sequence is
//! token-identical to non-speculative greedy generation from the same
//! target. The sampled arm implements the spec's rejection-sampling
//! correction exactly, so sampled emission matches the target's
//! distribution. Depth defaults to 6, the operating point the llama.cpp
//! sweep measured on 2026-10-30 (`bench/gpu-proof-2026-10-30.md`), and is
//! configuration, never a constant. Diagnostics and cumulative stats are
//! built in from birth ([`diag`], [`SpecStats`]).
//!
//! Known gaps, named rather than papered over: N-gram and Medusa drafting
//! refuse with `ModeUnavailable` (their machinery is not in this crate);
//! grammar-constrained speculation refuses with `GrammarConflict`; the
//! loop drives the CPU target path, so it proves the mechanism, not the
//! serving speed (TODO T-9.6's speed half belongs to the GPU path).

#![forbid(unsafe_code)]

pub mod config;
pub mod diag;
pub mod error;
pub mod rng;
pub mod sampler;
pub mod session;
pub mod speculator;

#[cfg(test)]
mod testkit;

pub use config::{OnVerifyError, SpecConfig, SpecMode};
pub use diag::{Diag, NullDiag, StderrDiag};
pub use error::{Result, SpecError};
pub use rng::Rng;
pub use session::{RoundOutcome, SpecSession, SpecStats};
pub use speculator::{
    BlockProposal, DflashSpeculator, DraftArm, DraftDist, Proposal, SharedCompute, Speculator,
};
