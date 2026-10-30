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
//!
//! Past the numbered set, the K9 module is a fusion product
//! for the device-resident forward: the attention preparation
//! (the Q/K norms, the sliding layers' rotations, and the KV
//! append) as one dispatch, proven against the K4/K5
//! composition it replaces in that path only. K10 and K11 are
//! the prefill GEMMs (Q4_K and Q6_K): tiled matrix-matrix
//! forms of K1 and K2 that stage dequantized weight tiles in
//! workgroup memory so a chunk's weights stream once instead
//! of once per row, proven against the GEMV composition they
//! replace in the resident session's chunk plans only. K12,
//! K13, and K14 are the prefill MMQ set: K12 quantizes the
//! chunk's activations to Q8_1 on the device (carrying the
//! activation transforms the GEMM pair fuses into its
//! product), and K13 (Q4_K) / K14 (Q6_K) run the product as
//! DP4A integer dots against the weights' packed payloads,
//! the design llama.cpp's MMQ reaches its prefill rates
//! with. The session prefers the MMQ form over the GEMM form
//! per projection when the device built it; the GEMM stays
//! as the named fallback for devices and shapes the MMQ
//! refuses.

pub mod k10_q4k_gemm;
pub mod k11_q6k_gemm;
pub mod k12_q8_quant;
pub mod k13_q4k_mmq;
pub mod k14_q6k_mmq;
pub mod k1_q4k_gemv;
pub mod k2_q6k_gemv;
pub mod k3_q5k_gemv;
pub mod k4_rmsnorm;
pub mod k5_rope_kv;
pub mod k6_attention;
pub mod k7_swiglu;
pub mod k8_head;
pub mod k9_attn_prep;
