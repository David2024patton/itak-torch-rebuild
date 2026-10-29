//! The math the forward delegates, behind a trait.
//!
//! This module owns the architecture: which steps run, in what order,
//! with which weights. It does not own arithmetic. Dequantization,
//! matmul, RoPE, norms, and softmax belong to the kernel crate (TODO
//! T-2.2), executing on the backend crate; they enter here through
//! [`Compute`]. That split is the seam that lets kernels be specialized
//! for Glimmer's exact shapes (6656 by 19968 FFN, the Q4_K/Q6_K mix) and
//! benchmarked without touching the architecture code.
//!
//! Layout convention for every activation tensor below: row-major
//! `[rows, width]`, flattened. `rows` is the number of positions (times
//! heads, for the per-head norm steps), stated per method.
//!
//! [`UnimplementedCompute`] is the skeleton-stage implementation: every
//! step refuses with [`GlimmerError::Unimplemented`] naming itself. The
//! forward loop is fully structured and testable against it, and no
//! caller can mistake its refusals for output.

use crate::cache::Cache;
use crate::config::GlimmerConfig;
use crate::error::{GlimmerError, Result};
use crate::layer::LayerKind;
use crate::weights::{LayerWeights, TensorView};

/// Everything one attention step needs, bundled: the bundled shape
/// keeps the [`Compute::attention`] signature stable as the kernel
/// crate's needs grow, instead of growing a parameter list.
pub struct AttentionRequest<'a> {
    /// Query projection, already Q-normed: `[positions, q_width]`.
    pub q: &'a [f32],
    /// Key projection, already K-normed: `[positions, k_width]`.
    pub k: &'a [f32],
    /// Value projection: `[positions, v_width]`.
    pub v: &'a [f32],
    /// Layer index, for addressing this layer's KV storage.
    pub layer: usize,
    /// The layer's attention kind.
    pub kind: LayerKind,
    /// The model configuration.
    pub cfg: &'a GlimmerConfig,
}

/// The operations one forward pass needs, in the order the forward
/// calls them. Implementations live in the kernel crate; this trait is
/// the contract they must satisfy.
pub trait Compute {
    /// Look up the embedding rows for `tokens`, returning
    /// `[tokens.len(), hidden]`.
    fn embed(&self, table: &TensorView, tokens: &[u32], cfg: &GlimmerConfig) -> Result<Vec<f32>>;

    /// RMSNorm over the last dimension of `x`, which holds `rows` rows.
    /// Used for the pre norms (width hidden or FFN input), the per-head
    /// Q and K norms (width head dim, rows = positions times heads),
    /// the post norms, and the final norm.
    fn rms_norm(&self, x: &[f32], weight: &TensorView, rows: usize, eps: f32) -> Result<Vec<f32>>;

    /// Linear projection `x` by `weight`, `[rows, in]` to `[rows, out]`,
    /// dequantizing the weight as needed.
    fn proj(&self, x: &[f32], weight: &TensorView, rows: usize) -> Result<Vec<f32>>;

    /// Attention for one layer: applies RoPE to the request's `q` and
    /// `k` at the cache's current position, attends under the layer's
    /// kind (sliding-window layers see only the last
    /// `cfg.sliding_window` positions), reads and writes KV through
    /// `cache`, and returns the attention output at query width,
    /// `[positions, q_width]`.
    fn attention(&self, req: &AttentionRequest<'_>, cache: &mut dyn Cache) -> Result<Vec<f32>>;

    /// Apply the attention gate: elementwise product of the attention
    /// output with the sigmoid of the gate projection (gated attention,
    /// per the `attn_gate.weight` tensor in the real file).
    fn attn_gate(&self, attn_out: &[f32], gate: &[f32]) -> Result<Vec<f32>>;

    /// Residual add of two equal-length activations.
    fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>>;

    /// The SwiGLU feed-forward block: `down(silu(gate(x)) * up(x))`,
    /// returning `[rows, hidden]`.
    fn swiglu_ffn(
        &self,
        x: &[f32],
        w: &LayerWeights,
        rows: usize,
        cfg: &GlimmerConfig,
    ) -> Result<Vec<f32>>;

    /// Project the final position's normed hidden state to vocabulary
    /// logits, `[vocab]`. Softcapping and scaling are the model's job,
    /// not this step's (see `softcap_scale` in the model module).
    fn lm_head(&self, x: &[f32], head: &TensorView, cfg: &GlimmerConfig) -> Result<Vec<f32>>;
}

/// Skeleton-stage [`Compute`]: every step refuses by name. Present so
/// the model can be assembled, validated, and structurally tested before
/// the kernel crate lands, and so any accidental production use fails
/// loud at the first step instead of emitting placeholder numbers.
#[derive(Debug, Default)]
pub struct UnimplementedCompute;

impl Compute for UnimplementedCompute {
    fn embed(
        &self,
        _table: &TensorView,
        _tokens: &[u32],
        _cfg: &GlimmerConfig,
    ) -> Result<Vec<f32>> {
        Err(GlimmerError::Unimplemented { step: "embed" })
    }

    fn rms_norm(
        &self,
        _x: &[f32],
        _weight: &TensorView,
        _rows: usize,
        _eps: f32,
    ) -> Result<Vec<f32>> {
        Err(GlimmerError::Unimplemented { step: "rms_norm" })
    }

    fn proj(&self, _x: &[f32], _weight: &TensorView, _rows: usize) -> Result<Vec<f32>> {
        Err(GlimmerError::Unimplemented { step: "proj" })
    }

    fn attention(&self, _req: &AttentionRequest<'_>, _cache: &mut dyn Cache) -> Result<Vec<f32>> {
        Err(GlimmerError::Unimplemented { step: "attention" })
    }

    fn attn_gate(&self, _attn_out: &[f32], _gate: &[f32]) -> Result<Vec<f32>> {
        Err(GlimmerError::Unimplemented { step: "attn_gate" })
    }

    fn add(&self, _a: &[f32], _b: &[f32]) -> Result<Vec<f32>> {
        Err(GlimmerError::Unimplemented { step: "add" })
    }

    fn swiglu_ffn(
        &self,
        _x: &[f32],
        _w: &LayerWeights,
        _rows: usize,
        _cfg: &GlimmerConfig,
    ) -> Result<Vec<f32>> {
        Err(GlimmerError::Unimplemented { step: "swiglu_ffn" })
    }

    fn lm_head(&self, _x: &[f32], _head: &TensorView, _cfg: &GlimmerConfig) -> Result<Vec<f32>> {
        Err(GlimmerError::Unimplemented { step: "lm_head" })
    }
}
