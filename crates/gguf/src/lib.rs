//! Strict GGUF v3 loading for the iTaK Torch rebuild.
//!
//! This crate replaces the original Go GGUF loader, which the 2026-10-29
//! audit found had three defect classes this crate is built to make
//! impossible:
//!
//! 1. **Missing dequantizers.** The Go loader had no Q3_K, Q5_K, or Q8_1
//!    readers. Here every type in the teardown corpus (F32, F16, BF16,
//!    Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K)
//!    dequantizes with ggml-exact math, and every other type is a hard
//!    error naming the tensor and the type. The full mainline type table
//!    lives in [`ggml`]; fork-only types are namespaced there by fork,
//!    never renumbered into mainline ids.
//! 2. **A type-id collision.** The Go loader treated mainline ggml type
//!    id 30 as I2_S; mainline id 30 is BF16. The id table in [`ggml`] is
//!    mainline and pinned by tests.
//! 3. **Silent placeholder weights.** The Go loader substituted
//!    pseudo-random weights for anything it could not read. This crate has
//!    no placeholder path: a tensor that is absent, unreadable, or of an
//!    unsupported type is an error, full stop. [`coverage`] makes that
//!    checkable per file: a coverage report names every required tensor
//!    that is missing, unreadable, or misshapen, and `verify_coverage`
//!    refuses the load when any exist.
//!
//! Diagnostics are built in from birth (David's rule): structured
//! `key=value` debug lines flow through [`diag`] whenever `GGUF_DEBUG` is
//! set or [`diag::set_debug_enabled`] is called, and every hard error is
//! logged at the point of refusal.
//!
//! Quick start:
//!
//! ```no_run
//! use std::path::Path;
//! let file = gguf::GgufFile::open(Path::new("model.gguf")).unwrap();
//! println!("arch: {}", file.architecture().unwrap());
//! let config = file.model_config().unwrap();
//! let weights = file.load_f32("token_embd.weight").unwrap();
//! ```

#![forbid(unsafe_code)]

pub mod config;
pub mod coverage;
pub mod dequant;
pub mod diag;
pub mod error;
pub mod file;
pub mod ggml;
pub mod metadata;

pub use config::{ModelConfig, TokenizerInfo};
pub use coverage::{CoverageReport, RequiredTensor, TensorCoverage, TensorStatus};
pub use error::{GgufError, Result};
pub use file::{GgufFile, TensorInfo};
pub use ggml::GgmlType;
pub use metadata::{Metadata, MetadataValue};
