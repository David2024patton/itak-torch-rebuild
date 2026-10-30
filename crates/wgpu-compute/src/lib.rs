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
//! What is real in this landing (TODO T-4.2, first tiers):
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
//! - **K2, the Q6_K dequant GEMV** ([`kernels::k2_q6k_gemv`]):
//!   17.1 percent of the stream (FFN down, attention V), same
//!   tuned parallel shape as K1, with a host-side repack into a
//!   word-aligned device layout (the 210-byte block is not a
//!   whole number of words; the module docs own the reasoning).
//!   Checked and benched through the `k2_check` and `k2_bench`
//!   examples.
//! - **K3, the Q5_K dequant GEMV** ([`kernels::k3_q5k_gemv`]):
//!   the 202,048-row output head, two-dimensional dispatch over
//!   the 65,535-workgroup cap. Checked and benched through the
//!   `k3_check` and `k3_bench` examples.
//!
//! Also wired: K4 (RMSNorm, plain and weighted, plus the
//! residual add) and K7 (SwiGLU and the attention gate), so the
//! FFN runs end to end; K5 (RoPE plus KV append) and K6
//! (grouped-query attention over device-resident KV buffers,
//! with the `attention` crate's cache mirrored on the host for
//! positions and cross-checks), so attention runs end to end;
//! and K8 (the embedding gather in every dtype `cpu-compute`
//! accepts, plus the scale-then-softcap head finishing kernel,
//! proven against `muse_glimmer::softcap_scale`; the trait
//! path's model applies the finishing itself, the resident
//! path runs this kernel). A forward over this backend
//! therefore runs every step on the GPU. Nothing falls back
//! silently; per-op CPU composition is the caller's load-time
//! decision (BACKEND.md Section 6, level L2).
//!
//! The device-resident forward ([`session`]) composes the same
//! kernels without the trait's host round trips: weights
//! upload once, activations and KV stay on the device, and
//! each forward chunk is one command-buffer chain with only
//! the finished logits read back. The trait path remains the
//! executable reference; the resident path is pinned to it by
//! the cross-path session test.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod compute;
pub mod detect;
pub mod device;
pub mod diag;
pub mod error;
pub mod kernels;
pub mod session;

pub use compute::WgpuCompute;
pub use detect::{detect, AdapterFacts, DetectPolicy};
pub use device::GpuContext;
pub use error::{Result, WgpuError};
pub use kernels::k10_q4k_gemm::{GemmShape, K10Call, K10Pipeline, GEMM_MAX_BATCH};
pub use kernels::k11_q6k_gemm::{K11Call, K11Pipeline};
pub use kernels::k1_q4k_gemv::{
    K1ActGeom, K1Call, K1Pipeline, K1Shape, K1Weights, ACT_GATED, ACT_NONE, ACT_SWIGLU,
};
pub use kernels::k2_q6k_gemv::{K2Call, K2Pipeline, K2Shape, K2Weights};
pub use kernels::k3_q5k_gemv::{
    K3Call, K3Pipeline, K3Shape, K3Weights, FINISH_NONE, FINISH_SCALE, FINISH_SOFTCAP,
};
pub use kernels::k4_rmsnorm::{K4Call, K4Pipeline};
pub use kernels::k5_rope_kv::{K5AppendCall, K5Pipeline, K5RotateCall};
pub use kernels::k6_attention::{
    AttendGeom, DeviceKv, GpuKvState, K6FlashCall, K6Pipeline, K6ReduceCall, K6ScoresCall, LayerKv,
};
pub use kernels::k7_swiglu::{K7Call, K7Pipeline};
pub use kernels::k8_head::{
    K8ArgmaxCall, K8FinishCall, K8GatherRowsCall, K8Pipeline, ARGMAX_GROUPS,
};
pub use kernels::k9_attn_prep::{K9Pipeline, K9PrepCall, PrepArgs};
pub use session::{ResidentSession, SessionStats};
