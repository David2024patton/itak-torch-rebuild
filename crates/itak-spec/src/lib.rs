//! itak-spec: the speculative decoding loop of the iTaK Torch engine
//! (SPEC-itak-spec). A [`SpecSession`] runs block drafting against
//! the Glimmer target with exact-output verification: DFlash block
//! drafting from a separate draft model, or n-gram prompt-lookup
//! drafting ([`NgramProposer`]) from the sequence itself, with no
//! draft model at all.
//!
//! One round, per the spec's Section 4: the draft proposes up to `depth`
//! tokens from the last committed position; the target verifies the whole
//! block in one `forward_block` under a KV checkpoint; the correction
//! walk in [`sampler`] accepts the prefix the target agrees with and
//! emits one correction or bonus token; the cache rewinds to the
//! committed count; the draft is told exactly the newly committed tokens
//! and receives their captured layer-input rows as conditioning.
//!
//! The T-9.7 depth sweep ([`sweep`]) re-measures the depth default on
//! this stack: it drives sessions through a caller-supplied factory,
//! so it runs against whatever backend the sessions were built on.
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
//! Known gaps, named rather than papered over: Medusa drafting
//! refuses with `ModeUnavailable` (its machinery is not in this
//! crate); grammar-constrained speculation refuses with `GrammarConflict`. The
//! loop drives its target through the [`driver::TargetDriver`] seam:
//! the CPU driver ([`CpuTarget`]) proves the mechanism, and the
//! GPU-resident driver ([`ResidentTarget`], TODO T-9.6) runs the
//! identical loop against the resident GPU forward. The draft model
//! has no GPU path in this workspace, so drafting runs on the CPU
//! stack behind the same `Speculator` seam on both drivers (the
//! bench note `bench/spec-gpu-port-2026-11-03.md` records what that
//! costs the speed bar).

#![forbid(unsafe_code)]

pub mod config;
pub mod diag;
pub mod driver;
pub mod error;
pub mod ngram;
pub mod resident;
pub mod rng;
pub mod sampler;
pub mod session;
pub mod speculator;
pub mod sweep;

#[cfg(test)]
mod testkit;

pub use config::{DraftBackend, OnVerifyError, SpecConfig, SpecMode};
pub use diag::{Diag, NullDiag, StderrDiag};
pub use driver::{CpuTarget, TargetDriver};
pub use error::{Result, SpecError};
pub use ngram::{NgramConfig, NgramProposer};
pub use resident::ResidentTarget;
pub use rng::Rng;
pub use session::{RoundOutcome, SpecSession, SpecStats};
pub use speculator::{
    BlockProposal, DflashSpeculator, DraftArm, DraftDist, Proposal, SharedCompute, Speculator,
};
pub use sweep::{run_sweep, DepthRow, SweepConfig, SweepError, SweepReport};
