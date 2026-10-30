//! The kernel set, one module per kernel, in BACKEND.md
//! Section 3's priority order (bytes streamed per token on
//! Glimmer, then position on the critical path).
//!
//! K1 is implemented. K2 to K8 are modules with their shapes and
//! duties documented and a single named refusal where the
//! `Compute` implementation calls them: nothing in this crate
//! silently substitutes another path for a missing kernel
//! (BACKEND.md Section 6's ladder puts per-op CPU fallback at load
//! time, as a disclosed composition decision by the caller, not as
//! a quiet detour inside a step).

pub mod k1_q4k_gemv;
pub mod k2_q6k_gemv;
pub mod k3_q5k_gemv;
pub mod k4_rmsnorm;
pub mod k5_rope_kv;
pub mod k6_attention;
pub mod k7_swiglu;
pub mod k8_head;
