//! `cpu-compute`: the CPU reference implementation of
//! [`muse_glimmer::Compute`], the arithmetic seam of the Muse
//! Glimmer architecture module.
//!
//! This is the crate the skeleton was waiting for. Where
//! [`muse_glimmer::UnimplementedCompute`] refuses every step by name,
//! [`CpuCompute`] does the real arithmetic: dequantization through
//! the `gguf` crate, dense math through `torch-math`, and grouped-
//! query attention over the `attention` crate's KV cache. It is a
//! correctness reference, not a speed contender: weights are
//! dequantized on every call and attention runs one query row at a
//! time, so the specialized kernels that come later can be verified
//! against simple, obviously-correct code.
//!
//! Two design points the trait forced, stated plainly:
//!
//! - **KV storage lives here, not in the caller's cache.**
//!   [`muse_glimmer::Cache`] is a position contract only (`position`
//!   and `advance`); it has no KV surface, and the forward hands the
//!   compute layer nothing else. So [`CpuCompute`] owns an
//!   `attention::KvCache`, built from the first config it serves,
//!   and cross-checks it against the position contract on every
//!   attention call: if the cache's stored positions and the
//!   caller's position disagree, the step refuses by name instead of
//!   attending to the wrong history. One [`CpuCompute`] serves one
//!   sequence; [`CpuCompute::reset`] abandons it.
//! - **Prefill and decode are the same code path.** Attention
//!   processes one position at a time: rotate, append to the cache,
//!   attend. A multi-token prefill is that loop repeated, and a
//!   decode step is the loop run once, so the two cannot drift
//!   apart. The integration tests prove the property end to end:
//!   incremental decode logits match one-shot prefill logits for the
//!   same sequence.
//!
//! The forward's step order, including the two placements the
//! architecture crate flagged for verification (post norms after
//! each residual add; the gate projection reading the normed
//! attention input), is `muse-glimmer`'s, not this crate's: this
//! crate supplies arithmetic for the steps in the order it is
//! called. Its tests pin the implemented order numerically, with the
//! assumptions named where they bite.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod cpu;
pub mod diag;

pub use cpu::CpuCompute;
