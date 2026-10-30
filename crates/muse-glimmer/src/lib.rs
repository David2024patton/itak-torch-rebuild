//! Muse Glimmer architecture module for the iTaK Torch Rust engine.
//!
//! One crate, one architecture: `muse-glimmer`, the 52-layer hybrid
//! measured on 2026-10-29 from the real Muse-Glimmer-30B GGUF (39
//! sliding-window layers with a 2048 window, 13 global layers, 32 Q
//! heads over 2 KV heads, hidden 6656, FFN 19968, logit softcap 20.0
//! with scale 0.19611613). Full dissection:
//! `~/workspace/glimmer-blueprint/GLIMMER-DISSECTION.md`.
//!
//! Status: skeleton stage (TODO T-P.3, opened ahead of its Phase 2 gate
//! on David's 2026-10-29 order). What is real:
//!
//! - [`GlimmerConfig`] parsed from the file's own GGUF metadata, every
//!   required key named when absent, the layer pattern derived from
//!   metadata when present and otherwise from the measured pattern,
//!   with its origin recorded in the config.
//! - The full measured tensor inventory ([`Weights`], 14 tensors per
//!   layer including the Q/K norms, attention gate, and post norms the
//!   Llama-shaped template in the old engine never looked up), loaded
//!   through the [`TensorSource`] seam.
//! - The structured forward in [`GlimmerModel::forward`]: validated
//!   input, the complete per-layer loop, final norm, untied head, and
//!   the real softcap-and-scale finishing math. The capturing
//!   variants ([`GlimmerModel::forward_capturing`],
//!   [`GlimmerModel::forward_block_capturing`]) additionally return
//!   the residual rows entering named layers, the conditioning
//!   input the DFlash draft consumes (SPEC-itak-spec Q1); capture
//!   is additive and changes no step of the verified pass.
//! - Fail-loud errors everywhere: a missing key, a missing tensor, or
//!   an unimplemented step is a named [`GlimmerError`], never a
//!   substituted value. That is the audit's central lesson, applied.
//!
//! What is trait-bound, waiting on sibling crates: metadata and tensor
//! access (the `itak-loader` crate, T-2.1, implements [`Metadata`] and
//! [`TensorSource`] over the `gguf` crate), the arithmetic (the kernels crate, T-2.2,
//! implements [`Compute`]), and KV storage (the KV crate, T-2.4,
//! implements [`Cache`] with real ring and paged storage behind the
//! position contract defined here). Until a real [`Compute`] is wired
//! in, [`UnimplementedCompute`] makes the forward refuse at its first
//! step, by name. A real one now exists: the `cpu-compute` crate's
//! `CpuCompute` does the arithmetic through `torch-math` and keeps KV
//! state in the `attention` crate's cache, cross-checked against the
//! position contract defined here.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod cache;
pub mod compute;
pub mod config;
pub mod diag;
pub mod error;
pub mod layer;
pub mod metadata;
pub mod model;
pub mod weights;

#[cfg(test)]
mod testkit;

pub use cache::{Cache, PositionCache};
pub use compute::{AttentionRequest, Compute, UnimplementedCompute};
pub use config::{GlimmerConfig, ARCHITECTURE};
pub use diag::{Diag, NullDiag, StderrDiag};
pub use error::{GlimmerError, Result};
pub use layer::{LayerKind, PatternSource};
pub use metadata::{MapMetadata, Metadata};
pub use model::{softcap_scale, GlimmerModel, LayerInputCapture};
pub use weights::{expected_tensor_names, Dtype, LayerWeights, TensorSource, TensorView, Weights};
