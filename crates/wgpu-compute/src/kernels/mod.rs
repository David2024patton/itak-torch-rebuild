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
//! refuses. K15 (Q4_K) and K16 (Q6_K) are the wide-tile
//! double-buffered MMQ forms: a 128-class output tile, a
//! 4x8 per-thread sub-tile, and half-super-block staging
//! steps pipelined through two workgroup buffer sets, so
//! staging and compute overlap instead of alternating
//! across two barriers per super-block. The session tries
//! the wide form first at chunk row counts past its
//! threshold, then the standard MMQ, then the GEMM: one
//! ladder, every refusal named.
//!
//! K17 is the DFlash draft set: the NEOX (half-split)
//! rotate and the draft's non-causal, two-source
//! (committed plus block) sliding-window attention, the
//! two primitives the draft forward needs that the
//! Glimmer kernels do not provide; the rest of the draft
//! forward composes K1/K2/K3/K4/K7/K8 unchanged.
//!
//! K18 is the small-batch product set: K13/K14's MMQ
//! math at a 16-row batch cap, one weight stream per
//! dispatch for the 2..=16 row band the speculative
//! loop runs at, in all three quant families (Q4_K,
//! Q5_K, Q6_K). The session routes to it only under
//! its env-set threshold, which defaults off; see the
//! module's own docs for the reasoning.
//!
//! K19 is the counter-proposal to K18 in the same
//! band: the Q4_K small-batch product in llama.cpp's
//! mul_mat_vec register-streaming shape, batch
//! specialized into the shader as literals, weights
//! dequantized in registers with no weight staging in
//! workgroup memory at all (the streaming-rate
//! research's structural verdict on both staged
//! attempts). The session routes to it only under its
//! own env-set threshold, which also defaults off.
//!
//! K21 is the Q5_K arm of the register-streaming
//! small-batch family (K19 carries Q4_K): the same
//! 2..=16 band as K18, but llama.cpp mul_mat_vec's
//! shape instead of the MMQ's: weights dequantize in
//! registers and are never staged in workgroup
//! memory, the batch lives inside the workgroup as
//! generation-specialized register accumulators, and
//! the only workgroup array is the reduction
//! partials. The session routes to it only under the
//! shared MMV env threshold, which ships open at the
//! full band (16) since the 2026-11-06 baseline
//! verdict and is narrowed per shape by the band
//! routing table; see the module's own docs for the
//! reasoning.
//!
//! K21F is K21's finish-carrying form: the same
//! register-streaming Q5_K product with the head's
//! fused finishing map (logit scale, tanh softcap,
//! K3's modes and expressions) applied at the
//! reduction's final store, built because the
//! 2026-11-06 verdict showed plain K21 never fires
//! in the real forward: on the current model file
//! the Q5_K tensors are the head, and the head sites
//! prepare a finish this variant now carries. It is
//! registered at backend construction and routed at
//! the resident block head only under two gates: its
//! own switch (`WGPU_COMPUTE_K21F_HEAD`, default on
//! since the wave18 retest met the reopen criterion
//! the guard wrote after the 2026-11-07
//! defaults-confirmation identity failure; `0`
//! closes it) and the shared MMV band; see
//! bench/k21-finish-2026-11-06.md,
//! bench/k21f-default-guard-2026-11-07.md, and
//! bench/k21f-default-flip-2026-11-07.md.
//!
//! K20 is the Q6_K register-streaming small-batch
//! product: K18's Q6_K band (2..=16 rows, one weight
//! stream per dispatch) in llama.cpp `mul_mat_vec`'s
//! shape instead of the MMQ shape, dequantizing in
//! registers and staging nothing but reduction
//! partials, as the counter-proposal to K18's and the
//! wide MMQ's workgroup-memory staging (the streaming
//! measurements behind it are in the module's own
//! docs). It shares its routing threshold with the
//! Q4_K register-streaming product (env
//! `WGPU_COMPUTE_MMV_MAX_ROWS`, shipped default 16,
//! the full band; set it to 1 to close the gate).
//!
//! K22 is the Q4_K MMQ product reorganized for
//! occupancy: the prefill profile measured K13 at
//! VGPR 256 (the architectural cap) streaming at under
//! 5% of the card's ceiling, so K22 runs the same MMQ
//! math in llama.cpp mul_mmq's register organization
//! (a per-thread register cache consumed column by
//! column, scalar integer dots, immediate folds) at the
//! same tile coverage. The session routes to it only
//! under its own env switch, which defaults off; see
//! the module's own docs for the reasoning.
//!
//! K23 is the same Q4_K MMQ product with the one
//! structural piece K22 left behind: llama.cpp's staging
//! repack. The staging rounds shrink to llama's 128-element
//! quantum, the staged qs arrays take llama's QPITCH skew,
//! and the scale pairs split into their own arrays, all
//! transcribed from mul_mmq_cm1; K13's thread organization
//! and math are unchanged, so a compiled or measured
//! difference is attributable to the staging alone. The
//! session routes to it only under its own env switch,
//! which defaults off; see the module's own docs and
//! `bench/k23-mmq-repack-2026-11-06.md`.

pub mod k10_q4k_gemm;
pub mod k11_q6k_gemm;
pub mod k12_q8_quant;
pub mod k13_q4k_mmq;
pub mod k14_q6k_mmq;
pub mod k15_q4k_mmq_wide;
pub mod k16_q6k_mmq_wide;
pub mod k17_dflash;
pub mod k18_small_batch;
pub mod k19_q4k_mmv;
pub mod k1_q4k_gemv;
pub mod k20_q6k_mmv;
pub mod k21_q5k_mmv;
pub mod k21_q5k_mmv_finish;
pub mod k22_q4k_mmq;
pub mod k23_q4k_coopmat;
pub mod k23_q4k_mmq;
pub mod k24_q4k_coopmat;
pub mod k2_q6k_gemv;
pub mod k3_q5k_gemv;
pub mod k4_rmsnorm;
pub mod k5_rope_kv;
pub mod k6_attention;
pub mod k7_swiglu;
pub mod k8_head;
pub mod k9_attn_prep;
pub mod splitk_reduce;
