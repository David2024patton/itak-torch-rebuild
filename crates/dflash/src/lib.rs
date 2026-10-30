//! DFlash draft architecture module for the iTaK Torch rebuild's
//! speculation stack (SPEC-itak-spec Section 3, dependency E3).
//!
//! One crate, one architecture: `dflash`, the block draft model
//! that conditions on a target model's hidden states. The shipped
//! pair is Muse-Glimmer-30B and its draft
//! `dflash-Muse-Glimmer-30B-Q4_K_M.gguf` (5 layers, hidden 6656,
//! 32 Q heads over 8 KV heads, head dim 128, block size 16, target
//! layers [2, 14, 26, 38, 50], census 2026-10-30).
//!
//! The conditioning wiring was settled from evidence before this
//! module was written (SPEC-itak-spec Q1, settled 2026-10-30): the
//! draft consumes the target's LAYER-INPUT residual rows at the
//! layers its own `dflash.target_layers` metadata names,
//! concatenated per position, fused through its `fc` projection
//! into K/V in its own cache at the target positions; drafting then
//! runs a noise block (anchor token, then the file's mask token)
//! through the draft layers with non-causal block attention,
//! borrowing the target's embedding table and output head, both of
//! which this file lacks. Sources: llama.cpp `src/models/dflash.cpp`,
//! `common/speculative.cpp`, `src/models/muse-glimmer.cpp`,
//! `src/llama-model.cpp` (RoPE pairing: NEOX for this arch).
//!
//! What is real:
//!
//! - [`DflashConfig`] parsed from the file's own GGUF metadata,
//!   every required key named when absent, no silent defaults.
//! - The full measured tensor inventory ([`Weights`], 3 globals
//!   plus 11 per layer, 58 total), loaded through the
//!   [`TensorSource`] seam with every shape checked, and the
//!   DFlash2/DSpark/d2t variant gates refused by name at load.
//! - [`DflashModel::inject`]: the conditioning fusion and KV
//!   injection pass. [`DflashModel::forward_block`]: the drafting
//!   pass producing per-position logits for a block. Both proven
//!   arithmetically on CPU (closed-form and naive-reference tests
//!   in this crate) and against the real file on the server.
//! - Fail-loud errors everywhere: a missing key, a missing tensor,
//!   a wrong conditioning width, or an unsupported variant is a
//!   named [`DflashError`], never a substituted value.
//!
//! Loading from a real GGUF file goes through the `itak-loader`
//! crate, which implements this module's [`Metadata`] and
//! [`TensorSource`] traits over the `gguf` crate and gates the load
//! on the full coverage check (`load_dflash_config`,
//! `coverage_dflash`, `load_dflash_weights`).
//!
//! Not here, by design: the speculative verify loop that consumes
//! these logits (the next spec task), GPU kernels for the draft,
//! and the DFlash2/DSpark variants (refused by name at load).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod cache;
pub mod config;
pub mod diag;
pub mod error;
pub mod metadata;
pub mod model;
pub mod weights;

#[cfg(test)]
mod testkit;

pub use cache::{DflashCache, KvRow};
pub use config::{DflashConfig, ARCHITECTURE};
pub use diag::{Diag, NullDiag, StderrDiag};
pub use error::{DflashError, Result};
pub use metadata::{MapMetadata, Metadata};
pub use model::{DflashModel, TargetTensors};
pub use weights::{
    expected_tensor_names, optional_tensor_names, LayerWeights, TensorData, TensorSource, Weights,
};
