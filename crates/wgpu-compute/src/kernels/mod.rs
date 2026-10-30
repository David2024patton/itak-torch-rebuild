//! The kernel set, one module per kernel, in BACKEND.md
//! Section 3's priority order (bytes streamed per token on
//! Glimmer, then position on the critical path).
//!
//! The full set is implemented: K1, K2, and K3 (the three GEMV
//! kernels cover Glimmer's whole weight stream: Q4_K, Q6_K,
//! Q5_K), K4 (norms and the residual add), K5 (RoPE plus KV
//! append), K6 (grouped-query attention over the resident KV
//! buffers K5 fills), K7 (the elementwise activations), and K8
//! (the embedding gather and the head finishing math). Every
//! `Compute` step runs on the device; nothing in this crate
//! silently substitutes another path for a kernel (BACKEND.md
//! Section 6's ladder puts per-op CPU fallback at load time, as
//! a disclosed composition decision by the caller, not as a
//! quiet detour inside a step).

pub mod k1_q4k_gemv;
pub mod k2_q6k_gemv;
pub mod k3_q5k_gemv;
pub mod k4_rmsnorm;
pub mod k5_rope_kv;
pub mod k6_attention;
pub mod k7_swiglu;
pub mod k8_head;
