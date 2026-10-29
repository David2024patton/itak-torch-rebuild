//! Generic dense architecture module for the iTaK Torch Rust
//! engine: the `llama` and `qwen2` families.
//!
//! One crate, one block shape: pre-norm transformer layers with
//! rotary position embedding, grouped-query attention over a real
//! KV cache, and a SwiGLU feed-forward network. That is the shape
//! both families share; what differs is in the files, and this crate
//! reads it from the files: Qwen2's attention biases (which the
//! original Go engine silently dropped) are part of the tensor map,
//! the RMSNorm epsilon and RoPE base come from each file's own
//! metadata, and a model whose `output.weight` is absent runs on its
//! embedding table as a tied head, explicitly, never by substitution
//! for a tensor that failed to read.
//!
//! Status: blueprint stage 2 (ARCHITECTURE.md), the first verified
//! text stage. The conformance models are TinyLlama-1.1B (`llama`)
//! and Qwen2.5-1.5B (`qwen2`), dissected 2026-10-29
//! (`~/workspace/model-teardown/models/`). What is real:
//!
//! - [`LlamaConfig`] parsed from the file's own GGUF metadata,
//!   every required key named when absent, no fabricated defaults.
//! - The full tensor inventory ([`Weights`]) for both families,
//!   including the optional q/k/v biases, loaded through the
//!   [`TensorSource`] seam (the `gguf` crate's file implements it)
//!   with every shape checked against the config.
//! - The forward in [`LlamaModel::forward`]: embedding, per-layer
//!   pre-norm, RoPE, GQA attention against the `attention` crate's
//!   KV cache, SwiGLU FFN, final norm, head. Prefill and incremental
//!   decode are the same code path over the shared cache; the tests
//!   prove their logits agree.
//! - Fail-loud errors everywhere: a missing key, a missing tensor,
//!   or a wrong shape is a named [`LlamaError`], never a substituted
//!   value. That is the audit's central lesson, applied.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod config;
pub mod diag;
pub mod error;
pub mod model;
pub mod weights;

#[cfg(test)]
mod testkit;

pub use config::{LlamaConfig, ARCHITECTURES};
pub use diag::{Diag, NullDiag, StderrDiag};
pub use error::{LlamaError, Result};
pub use model::LlamaModel;
pub use weights::{expected_tensor_names, LayerWeights, TensorData, TensorSource, Weights};
