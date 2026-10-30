//! `itak-kernels`: quantized tensor math for the iTaK Torch Rust
//! engine (PLAN S5, TODO T-2.2).
//!
//! PLAN.md assigns this crate the quantized side of the math: Q4_K,
//! Q6_K, and Q8_0 dequantization and matmul first, and it names the
//! bar every kernel here is held to: pure functions over buffers, no
//! silent wrong numbers, and a payload field is never indexed before
//! its length has been checked (ARCHITECTURE.md, live test F3, the
//! safeMatMul lesson from the old Go engine).
//!
//! What exists today is the first half of T-2.2: the Q4_K kernels in
//! [`q4k`] (dequantize, and matmul as the GEMV the token loop runs).
//! Q6_K and Q8_0 kernels are the second half of T-2.2 and are not
//! here yet; the dense f32 primitives (RMSNorm, RoPE, softmax,
//! SiLU) live in the `torch-math` reference crate and are not
//! duplicated here.
//!
//! House rules, as everywhere in this workspace:
//!
//! - **Limits are config.** The one operational limit these kernels
//!   have (how many elements a single call may cover) is
//!   [`KernelConfig`], with a default, never a constant in a kernel
//!   body. Format geometry (256 elements, 144 bytes per Q4_K block)
//!   is not a limit; it is the file format, taken from the `gguf`
//!   crate's type definitions and pinned by tests.
//! - **Diagnostics from birth.** [`diag`] is the structured logging
//!   seam (off by default); refusals are logged through it, and
//!   every error is a named [`KernelError`], never a panic and never
//!   a substituted value.
//! - **No allocation surprises.** Each function's documentation
//!   states exactly what it allocates; the dequantizer allocates
//!   nothing at all.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod config;
pub mod diag;
pub mod error;
pub mod q4k;

pub use config::KernelConfig;
pub use diag::set_debug;
pub use error::{KernelError, Result};
pub use q4k::{dequantize_q4k, dequantize_q4k_vec, matvec_q4k, BLOCK_BYTES, BLOCK_ELEMENTS};
