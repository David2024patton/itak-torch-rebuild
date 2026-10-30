//! `wgpu-compute`: the wgpu (Vulkan) implementation of
//! [`muse_glimmer::Compute`], sibling to `cpu-compute`.
//!
//! Naming: PLAN.md and BACKEND.md call the backend family
//! `itak-backend` (the backend trait, `detect()`, and its
//! implementations under one module contract). The implementations
//! land as their own crates beside each other, the way
//! `cpu-compute` landed for the CPU SIMD slot without taking the
//! family name; this crate is the family's wgpu member, and its
//! [`detect`] is the family's hardware detection: adapter facts,
//! a written selection policy, granted-limits verification, and
//! the chosen adapter recorded for the health surface.
//!
//! What is real in this landing (TODO T-4.2, first tier):
//!
//! - **Device selection** ([`detect`]): Vulkan discrete GPU or a
//!   named refusal listing every adapter seen. llvmpipe counts as
//!   no adapter, per BACKEND.md Section 6. The device opens with
//!   the adapter's own limits and the granted limits are verified
//!   against the kernels' floor before anything runs.
//! - **K1, the Q4_K dequant GEMV** ([`kernels::k1_q4k_gemv`]):
//!   77.3 percent of Glimmer's streamed bytes, reading the ggml
//!   block layout directly in WGSL, checked against the `gguf`
//!   crate's CPU dequantizer plus `torch-math` on synthetic rows
//!   here and on the real file's tensors through the `k1_check`
//!   example, and benched against llama.cpp's measured 708 GB/s
//!   through the `k1_bench` example (results in BACKEND.md).
//!
//! What is not here yet, by design: kernels K2 to K8 live in their
//! own modules under [`kernels`] with their shapes documented,
//! and every `Compute` step that needs one refuses through that
//! module with a named [`muse_glimmer::GlimmerError::StepFailed`].
//! A forward over this backend therefore runs its Q4_K
//! projections on the GPU and stops, by name, at the first step
//! whose kernel has not landed. Nothing falls back silently;
//! per-op CPU composition is the caller's load-time decision
//! (BACKEND.md Section 6, level L2).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod compute;
pub mod detect;
pub mod device;
pub mod diag;
pub mod error;
pub mod kernels;

pub use compute::WgpuCompute;
pub use detect::{detect, AdapterFacts, DetectPolicy};
pub use device::GpuContext;
pub use error::{Result, WgpuError};
pub use kernels::k1_q4k_gemv::{K1Call, K1Pipeline, K1Shape, K1Weights};
