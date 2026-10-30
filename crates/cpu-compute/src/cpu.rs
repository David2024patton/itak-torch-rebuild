//! [`CpuCompute`]: real f32 arithmetic behind the
//! [`muse_glimmer::Compute`] trait. See the crate docs for the two
//! design points the trait forced (backend-owned KV storage, and
//! prefill and decode sharing one row-at-a-time attention path).
//!
//! Every refusal is a [`GlimmerError::StepFailed`] naming the step
//! and carrying the underlying failure in its own words: a shape
//! mismatch, a tensor dtype this backend cannot dequantize, a
//! `torch-math` or `attention` error, or KV state that disagrees
//! with the caller's sequence position. Nothing is substituted,
//! zero-filled, or silently skipped, which is the audit lesson this
//! whole rebuild exists to apply.

use std::cell::RefCell;

use attention::{AttentionParams, Checkpoint, KvCache, LayerKind as AttnLayerKind};
use gguf::dequant::dequantize;
use gguf::ggml::type_name;
use gguf::GgmlType;
use muse_glimmer::{
    AttentionRequest, Cache, Compute, Dtype, GlimmerConfig, GlimmerError, LayerKind, LayerWeights,
    Result, TensorView,
};

use crate::diag::dbg_log;

/// Build a [`GlimmerError::StepFailed`] for `step` with `detail`.
fn step_failed(step: &'static str, detail: impl Into<String>) -> GlimmerError {
    GlimmerError::StepFailed {
        step,
        detail: detail.into(),
    }
}

/// The mainline ggml type id of a [`Dtype`], for naming types in
/// refusals. Ids are the ones pinned in `muse-glimmer`'s weights
/// module (`Q4_0` is 2, `Bf16` is 30, and so on).
fn dtype_id(dtype: Dtype) -> u32 {
    match dtype {
        Dtype::F32 => 0,
        Dtype::F16 => 1,
        Dtype::Q4_0 => 2,
        Dtype::Q8_0 => 8,
        Dtype::Q4K => 12,
        Dtype::Q5K => 13,
        Dtype::Q6K => 14,
        Dtype::Bf16 => 30,
        Dtype::Unknown(id) => id,
    }
}

/// The `gguf` crate's dequantizable type for a tensor, or a refusal
/// naming the tensor and the type it actually carries. The `gguf`
/// crate supports exactly F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K;
/// anything else (Q4_0, or an id this workspace does not name) is a
/// hard error here, never a fallback.
fn ggml_type(view: &TensorView, step: &'static str) -> Result<GgmlType> {
    match view.dtype {
        Dtype::F32 => Ok(GgmlType::F32),
        Dtype::F16 => Ok(GgmlType::F16),
        Dtype::Q8_0 => Ok(GgmlType::Q8_0),
        Dtype::Q4K => Ok(GgmlType::Q4K),
        Dtype::Q5K => Ok(GgmlType::Q5K),
        Dtype::Q6K => Ok(GgmlType::Q6K),
        Dtype::Bf16 => Ok(GgmlType::Bf16),
        other => Err(step_failed(
            step,
            format!(
                "tensor {} has dtype {} (ggml type id {}), which the CPU reference backend cannot dequantize",
                view.name,
                type_name(dtype_id(other)),
                dtype_id(other)
            ),
        )),
    }
}

/// Dequantize a whole tensor to f32, validating the byte length
/// against the type's block geometry first so a truncated or
/// mislabeled tensor fails with its name and both lengths.
fn dequantize_view(view: &TensorView, step: &'static str) -> Result<Vec<f32>> {
    let ty = ggml_type(view, step)?;
    let elements = usize::try_from(view.elements()).map_err(|_| {
        step_failed(
            step,
            format!("tensor {} element count does not fit in usize", view.name),
        )
    })?;
    let expected = ty.byte_len(elements as u64).ok_or_else(|| {
        step_failed(
            step,
            format!(
                "tensor {} has {elements} elements, not a whole number of {} blocks",
                view.name,
                type_name(ty.id())
            ),
        )
    })?;
    if expected != view.data.len() as u64 {
        return Err(step_failed(
            step,
            format!(
                "tensor {} holds {} bytes but {elements} elements of {} need {expected}",
                view.name,
                view.data.len(),
                type_name(ty.id())
            ),
        ));
    }
    Ok(dequantize(ty, &view.data, elements))
}

/// Dequantize `rows` consecutive rows of `row_elems` elements from a
/// tensor, used by the embedding lookup so a single token does not
/// dequantize the whole table. When a row is a whole number of
/// quantization blocks (the usual case: hidden 6656 is 26 Q4_K
/// blocks), only the rows' bytes are touched; otherwise the whole
/// tensor is dequantized and the rows sliced out of the result.
fn dequantize_rows(
    view: &TensorView,
    step: &'static str,
    row_elems: usize,
    rows: std::ops::Range<usize>,
) -> Result<Vec<f32>> {
    let ty = ggml_type(view, step)?;
    let block = ty.block_size() as usize;
    if row_elems % block == 0 {
        let row_bytes = (row_elems / block) * ty.block_bytes() as usize;
        let start = rows.start * row_bytes;
        let end = rows.end * row_bytes;
        let slice = view.data.get(start..end).ok_or_else(|| {
            step_failed(
                step,
                format!(
                    "tensor {} holds {} bytes, too few for rows {rows:?} of {row_elems} elements",
                    view.name,
                    view.data.len()
                ),
            )
        })?;
        Ok(dequantize(ty, slice, (rows.end - rows.start) * row_elems))
    } else {
        let full = dequantize_view(view, step)?;
        let start = rows.start * row_elems;
        let end = rows.end * row_elems;
        let slice = full.get(start..end).ok_or_else(|| {
            step_failed(
                step,
                format!(
                    "tensor {} has {} elements, too few for rows {rows:?} of {row_elems}",
                    view.name,
                    full.len()
                ),
            )
        })?;
        Ok(slice.to_vec())
    }
}

/// Logistic sigmoid, computed in the branch that keeps the
/// exponential finite at both tails.
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// The shape a [`KvState`] was built from. A `CpuCompute` is bound
/// to the first config it serves; a different shape arriving later
/// is a refusal, not a silent rebuild (a rebuild would drop the
/// stored sequence without anyone deciding that).
#[derive(Clone, Debug, PartialEq, Eq)]
struct CacheShape {
    layers: Vec<LayerKind>,
    kv_heads: u32,
    key_length: u32,
    value_length: u32,
    sliding_window: u32,
}

impl CacheShape {
    fn of(cfg: &GlimmerConfig) -> Self {
        CacheShape {
            layers: cfg.layers.clone(),
            kv_heads: cfg.kv_heads,
            key_length: cfg.key_length,
            value_length: cfg.value_length,
            sliding_window: cfg.sliding_window,
        }
    }
}

/// The backend-owned KV storage: the `attention` crate's cache plus
/// the shape it was built from. See the crate docs for why the
/// storage lives here (the `Cache` trait the forward hands over is
/// a position contract only).
#[derive(Debug)]
struct KvState {
    cache: KvCache,
    shape: CacheShape,
}

impl KvState {
    fn build(cfg: &GlimmerConfig, step: &'static str) -> Result<Self> {
        let pattern: Vec<AttnLayerKind> = cfg
            .layers
            .iter()
            .map(|kind| match kind {
                LayerKind::Global => AttnLayerKind::Global,
                LayerKind::SlidingWindow => AttnLayerKind::Windowed {
                    window: cfg.sliding_window as usize,
                },
            })
            .collect();
        let cache = KvCache::new(&pattern, cfg.kv_heads as usize, cfg.key_length as usize)
            .map_err(|e| step_failed(step, e.to_string()))?;
        dbg_log!(
            "kv cache built layers={} kv_heads={} head_dim={} window={}",
            pattern.len(),
            cfg.kv_heads,
            cfg.key_length,
            cfg.sliding_window
        );
        Ok(KvState {
            cache,
            shape: CacheShape::of(cfg),
        })
    }

    fn check_shape(&self, cfg: &GlimmerConfig, step: &'static str) -> Result<()> {
        if self.shape != CacheShape::of(cfg) {
            return Err(step_failed(
                step,
                "this CpuCompute already serves a model of a different shape; \
                 construct a fresh CpuCompute per model (or reset this one) \
                 instead of retargeting it mid-sequence",
            ));
        }
        Ok(())
    }
}

/// The CPU reference [`Compute`] backend. Stateful in exactly one
/// way: it owns the KV cache of the one sequence it is serving.
/// Construct one per sequence (or per model run), and call
/// [`CpuCompute::reset`] to abandon a sequence and start over.
#[derive(Debug, Default)]
pub struct CpuCompute {
    kv: RefCell<Option<KvState>>,
}

impl CpuCompute {
    /// A fresh backend with no KV state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop all stored KV state. The next attention call rebuilds
    /// the cache from its config and expects the caller's sequence
    /// position to be 0 again.
    pub fn reset(&self) {
        dbg_log!("kv state reset");
        *self.kv.borrow_mut() = None;
    }

    /// Checkpoint the backend's KV cache before a speculative round
    /// (SPEC-itak-spec Section 5, dependency E2): the round may
    /// append up to `max_round_tokens` positions per layer, and
    /// [`CpuCompute::cache_rewind`] can then restore the exact
    /// pre-round state, ring evictions included.
    ///
    /// The KV state must already exist, which it does after the
    /// first forward (a speculative session checkpoints after
    /// prefill, per the spec's round structure); checkpointing a
    /// backend that has served nothing yet is refused by name
    /// rather than silently binding the checkpoint to an unbuilt
    /// cache.
    pub fn cache_checkpoint(&self, max_round_tokens: usize) -> Result<Checkpoint> {
        let step = "cache_checkpoint";
        let mut guard = self.kv.borrow_mut();
        let state = guard.as_mut().ok_or_else(|| {
            step_failed(
                step,
                "no KV state exists yet: a speculative round checkpoints after a first \
                 forward has built the cache (the spec's prefill), not before",
            )
        })?;
        state
            .cache
            .checkpoint(max_round_tokens)
            .map_err(|e| step_failed(step, e.to_string()))
    }

    /// Rewind the backend's KV cache to `target` positions against a
    /// live checkpoint from [`CpuCompute::cache_checkpoint`]:
    /// committing a round is a rewind to `base + 1 + accepted`,
    /// aborting it is a rewind to `base`.
    ///
    /// Pair this with [`muse_glimmer::Cache::rewind_to`] on the
    /// model's position cache, with the same target, in the same
    /// step (SPEC-itak-spec Section 5.4: position and storage move
    /// together). Rewinding only one of the two is caught by the
    /// attention step's position cross-check on the next forward,
    /// which refuses by name rather than attending to a history the
    /// two sides disagree about.
    pub fn cache_rewind(&self, checkpoint: &Checkpoint, target: usize) -> Result<()> {
        let step = "cache_rewind";
        let mut guard = self.kv.borrow_mut();
        let state = guard.as_mut().ok_or_else(|| {
            step_failed(
                step,
                "no KV state exists: the sequence was reset after the checkpoint was taken",
            )
        })?;
        state
            .cache
            .rewind_to(checkpoint, target)
            .map_err(|e| step_failed(step, e.to_string()))
    }

    /// Release a checkpoint taken with
    /// [`CpuCompute::cache_checkpoint`] once its round is resolved.
    pub fn cache_release(&self, checkpoint: &Checkpoint) -> Result<()> {
        let step = "cache_release";
        let mut guard = self.kv.borrow_mut();
        let state = guard.as_mut().ok_or_else(|| {
            step_failed(
                step,
                "no KV state exists: the sequence was reset after the checkpoint was taken",
            )
        })?;
        state
            .cache
            .release(checkpoint)
            .map_err(|e| step_failed(step, e.to_string()))
    }

    /// Shared projection body: `proj` calls it under its own step
    /// name, the FFN under `swiglu_ffn`, so refusals name the step
    /// the forward thinks it is in.
    fn proj_impl(
        &self,
        x: &[f32],
        weight: &TensorView,
        rows: usize,
        step: &'static str,
    ) -> Result<Vec<f32>> {
        if weight.shape.len() != 2 {
            return Err(step_failed(
                step,
                format!(
                    "projection tensor {} has shape {:?}, expected [in, out] in GGUF stored order",
                    weight.name, weight.shape
                ),
            ));
        }
        let (inn, out) = (weight.shape[0] as usize, weight.shape[1] as usize);
        if x.len() != rows * inn {
            return Err(step_failed(
                step,
                format!(
                    "input holds {} elements but {rows} rows of width {inn} need {}",
                    x.len(),
                    rows * inn
                ),
            ));
        }
        let w = dequantize_view(weight, step)?;
        if rows == 1 {
            // Decode shape: one token against the whole matrix.
            return torch_math::matvec(&w, out, inn, x)
                .map_err(|e| step_failed(step, e.to_string()));
        }
        // Prefill shape: transpose the weight once, then one matmul.
        // The stored layout is [out][in] row-major; the transpose is
        // [in][out], which is what matmul's right operand wants.
        let mut wt = vec![0.0f32; inn * out];
        for (o, w_row) in w.chunks_exact(inn).enumerate() {
            for (i, &val) in w_row.iter().enumerate() {
                wt[i * out + o] = val;
            }
        }
        torch_math::matmul(x, rows, inn, &wt, inn, out)
            .map_err(|e| step_failed(step, e.to_string()))
    }

    /// Apply RoPE to every head of one position's projection row.
    /// The pairing is llama.cpp's NORM mode (adjacent channels):
    /// `llama_model_rope_type` selects LLAMA_ROPE_TYPE_NORM for
    /// muse-glimmer, and the converter unpermuted Q/K into that
    /// layout, so the half-split form torch-math's `rope` implements
    /// for the llama/qwen2 families would pair the wrong channels
    /// here (a Run B follow-up find; the two pairings disagree
    /// numerically from the first rotated position).
    fn rope_row(
        row: &[f32],
        heads: usize,
        head_dim: usize,
        position: usize,
        base: f32,
        step: &'static str,
    ) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(row.len());
        for h in 0..heads {
            let rotated = torch_math::rope_interleaved(
                &row[h * head_dim..(h + 1) * head_dim],
                position,
                base,
                head_dim,
            )
            .map_err(|e| step_failed(step, e.to_string()))?;
            out.extend_from_slice(&rotated);
        }
        Ok(out)
    }
}

impl Compute for CpuCompute {
    fn embed(&self, table: &TensorView, tokens: &[u32], cfg: &GlimmerConfig) -> Result<Vec<f32>> {
        let step = "embed";
        let hidden = cfg.hidden as usize;
        if table.shape.len() != 2
            || table.shape[0] as usize != hidden
            || table.shape[1] as usize != cfg.vocab_size as usize
        {
            return Err(step_failed(
                step,
                format!(
                    "embedding table {} has shape {:?}, expected [{}, {}] in GGUF stored order",
                    table.name, table.shape, hidden, cfg.vocab_size
                ),
            ));
        }
        let mut out = Vec::with_capacity(tokens.len() * hidden);
        for &token in tokens {
            let row = dequantize_rows(table, step, hidden, token as usize..token as usize + 1)?;
            out.extend_from_slice(&row);
        }
        Ok(out)
    }

    fn rms_norm(&self, x: &[f32], weight: &TensorView, rows: usize, eps: f32) -> Result<Vec<f32>> {
        let step = "rms_norm";
        if rows == 0 || x.len() % rows != 0 {
            return Err(step_failed(
                step,
                format!(
                    "input holds {} elements, not a whole number of {rows} rows",
                    x.len()
                ),
            ));
        }
        let width = x.len() / rows;
        let w = dequantize_view(weight, step)?;
        if w.len() != width {
            return Err(step_failed(
                step,
                format!(
                    "norm weight {} has {} elements but the rows it normalizes are {width} wide",
                    weight.name,
                    w.len()
                ),
            ));
        }
        let mut out = Vec::with_capacity(x.len());
        for r in 0..rows {
            let row = torch_math::rms_norm(&x[r * width..(r + 1) * width], &w, eps)
                .map_err(|e| step_failed(step, e.to_string()))?;
            out.extend_from_slice(&row);
        }
        Ok(out)
    }

    fn rms_norm_plain(&self, x: &[f32], rows: usize, eps: f32) -> Result<Vec<f32>> {
        let step = "rms_norm_plain";
        if rows == 0 || x.len() % rows != 0 {
            return Err(step_failed(
                step,
                format!(
                    "input holds {} elements, not a whole number of {rows} rows",
                    x.len()
                ),
            ));
        }
        let width = x.len() / rows;
        // Weightless: normalize by RMS only. torch_math::rms_norm
        // wants a weight slice, so hand it ones rather than growing
        // a second kernel for the same arithmetic.
        let ones = vec![1.0f32; width];
        let mut out = Vec::with_capacity(x.len());
        for r in 0..rows {
            let row = torch_math::rms_norm(&x[r * width..(r + 1) * width], &ones, eps)
                .map_err(|e| step_failed(step, e.to_string()))?;
            out.extend_from_slice(&row);
        }
        Ok(out)
    }

    fn proj(&self, x: &[f32], weight: &TensorView, rows: usize) -> Result<Vec<f32>> {
        self.proj_impl(x, weight, rows, "proj")
    }

    fn attention(&self, req: &AttentionRequest<'_>, cache: &mut dyn Cache) -> Result<Vec<f32>> {
        let step = "attention";
        let cfg = req.cfg;
        // The attention crate stores keys and values at one head
        // dimension. Glimmer's real file has key_length ==
        // value_length (128), as does every config this backend can
        // serve; a file where they differ is refused by name rather
        // than silently reshaped.
        if cfg.key_length != cfg.value_length {
            return Err(step_failed(
                step,
                format!(
                    "key_length {} != value_length {}: the attention crate's KvCache stores keys and values at one head dimension",
                    cfg.key_length, cfg.value_length
                ),
            ));
        }
        if cfg.layer_kind(req.layer) != Some(req.kind) {
            return Err(step_failed(
                step,
                format!(
                    "request kind {:?} does not match the config's kind for layer {}",
                    req.kind, req.layer
                ),
            ));
        }
        let q_width = cfg.q_width() as usize;
        let k_width = cfg.k_width() as usize;
        let v_width = cfg.v_width() as usize;
        if q_width == 0 || req.q.len() % q_width != 0 {
            return Err(step_failed(
                step,
                format!(
                    "query projection holds {} elements, not a whole number of {q_width}-wide rows",
                    req.q.len()
                ),
            ));
        }
        let rows = req.q.len() / q_width;
        if req.k.len() != rows * k_width || req.v.len() != rows * v_width {
            return Err(step_failed(
                step,
                format!(
                    "key/value projections hold {}/{} elements for {rows} rows, expected {}/{}",
                    req.k.len(),
                    req.v.len(),
                    rows * k_width,
                    rows * v_width
                ),
            ));
        }

        let start = cache.position();
        let mut guard = self.kv.borrow_mut();
        if guard.is_none() {
            *guard = Some(KvState::build(cfg, step)?);
        }
        let state = guard.as_mut().expect("kv state was just built");
        state.check_shape(cfg, step)?;
        let appended = state
            .cache
            .next_position(req.layer)
            .map_err(|e| step_failed(step, e.to_string()))?;
        if appended != start {
            return Err(step_failed(
                step,
                format!(
                    "KV state for layer {} is at position {appended} but the sequence cache is at position {start}: \
                     the two disagree, so attending would read the wrong history. \
                     Pair one Cache with one CpuCompute per sequence, or call CpuCompute::reset \
                     when a sequence is abandoned (a forward that failed midway leaves the KV state \
                     ahead of the cache and also needs a reset)",
                    req.layer
                ),
            ));
        }

        let head_dim = cfg.key_length as usize;
        let params = AttentionParams::new();
        let mut out = Vec::with_capacity(rows * q_width);
        // One position at a time: rotate (sliding layers only),
        // append, attend. Prefill and decode share this exact path
        // (decode is the rows == 1 case), and the causal and window
        // masks inside gqa_attention are stated over absolute
        // positions, so a query never sees a later position even
        // though the whole call's keys end up stored.
        for i in 0..rows {
            let position = start + i;
            // RoPE runs on the sliding-window layers only, at the
            // SWA base; global layers are NoPE and their Q/K pass
            // through unrotated (src/models/muse-glimmer.cpp:
            // "RoPE runs on the SWA layers, NoPE on full ones").
            let (q_rot, k_rot) = if req.kind == LayerKind::SlidingWindow {
                (
                    Self::rope_row(
                        &req.q[i * q_width..(i + 1) * q_width],
                        cfg.q_heads as usize,
                        head_dim,
                        position,
                        cfg.rope_freq_base_swa,
                        step,
                    )?,
                    Self::rope_row(
                        &req.k[i * k_width..(i + 1) * k_width],
                        cfg.kv_heads as usize,
                        head_dim,
                        position,
                        cfg.rope_freq_base_swa,
                        step,
                    )?,
                )
            } else {
                (
                    req.q[i * q_width..(i + 1) * q_width].to_vec(),
                    req.k[i * k_width..(i + 1) * k_width].to_vec(),
                )
            };
            state
                .cache
                .append(
                    req.layer,
                    &k_rot,
                    &req.v[i * v_width..(i + 1) * v_width],
                    position,
                )
                .map_err(|e| step_failed(step, e.to_string()))?;
            let view = state
                .cache
                .view(req.layer)
                .map_err(|e| step_failed(step, e.to_string()))?;
            let row = attention::gqa_attention(
                &q_rot,
                1,
                cfg.q_heads as usize,
                &[position],
                &view,
                &params,
            )
            .map_err(|e| step_failed(step, e.to_string()))?;
            out.extend_from_slice(&row);
        }
        dbg_log!(
            "attention layer={} kind={:?} start={} rows={}",
            req.layer,
            req.kind,
            start,
            rows
        );
        Ok(out)
    }

    fn attn_gate(&self, attn_out: &[f32], gate: &[f32]) -> Result<Vec<f32>> {
        if attn_out.len() != gate.len() {
            return Err(step_failed(
                "attn_gate",
                format!(
                    "attention output holds {} elements but the gate projection holds {}",
                    attn_out.len(),
                    gate.len()
                ),
            ));
        }
        Ok(attn_out
            .iter()
            .zip(gate.iter())
            .map(|(a, g)| a * sigmoid(*g))
            .collect())
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if a.len() != b.len() {
            return Err(step_failed(
                "add",
                format!(
                    "residual operands hold {} and {} elements",
                    a.len(),
                    b.len()
                ),
            ));
        }
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
    }

    fn swiglu_ffn(
        &self,
        x: &[f32],
        w: &LayerWeights,
        rows: usize,
        _cfg: &GlimmerConfig,
    ) -> Result<Vec<f32>> {
        let step = "swiglu_ffn";
        let gate = self.proj_impl(x, &w.ffn_gate, rows, step)?;
        let up = self.proj_impl(x, &w.ffn_up, rows, step)?;
        let act = torch_math::swiglu(&gate, &up).map_err(|e| step_failed(step, e.to_string()))?;
        self.proj_impl(&act, &w.ffn_down, rows, step)
    }

    fn lm_head(&self, x: &[f32], head: &TensorView, cfg: &GlimmerConfig) -> Result<Vec<f32>> {
        let step = "lm_head";
        if head.shape.len() != 2 {
            return Err(step_failed(
                step,
                format!(
                    "output head {} has shape {:?}, expected [hidden, vocab] in GGUF stored order",
                    head.name, head.shape
                ),
            ));
        }
        let (inn, vocab) = (head.shape[0] as usize, head.shape[1] as usize);
        if inn != x.len() {
            return Err(step_failed(
                step,
                format!(
                    "output head {} reads {inn} inputs but the normed hidden state holds {}",
                    head.name,
                    x.len()
                ),
            ));
        }
        if vocab != cfg.vocab_size as usize {
            return Err(step_failed(
                step,
                format!(
                    "output head {} produces {vocab} logits but the config's vocabulary is {}",
                    head.name, cfg.vocab_size
                ),
            ));
        }
        let w = dequantize_view(head, step)?;
        torch_math::matvec(&w, vocab, inn, x).map_err(|e| step_failed(step, e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// A minimal config for direct trait calls: only the fields the
    /// called step reads need to be plausible.
    fn call_cfg() -> GlimmerConfig {
        GlimmerConfig {
            name: None,
            block_count: 1,
            context_length: 16,
            hidden: 4,
            ffn_hidden: 8,
            q_heads: 2,
            kv_heads: 1,
            key_length: 2,
            value_length: 2,
            rope_freq_base: 10_000.0,
            rope_freq_base_swa: 10_000.0,
            rms_eps: 1e-6,
            sliding_window: 8,
            logit_softcap: 20.0,
            logit_scale: 1.0,
            vocab_size: 4,
            layers: vec![LayerKind::Global],
            pattern_source: muse_glimmer::PatternSource::MeasuredFallback,
        }
    }

    fn f32_view(name: &str, shape: &[u64], values: &[f32]) -> TensorView {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        TensorView {
            name: name.to_string(),
            dtype: Dtype::F32,
            shape: shape.to_vec(),
            data: Arc::from(data.into_boxed_slice()),
        }
    }

    #[test]
    fn cache_checkpoint_before_any_forward_refuses_by_name() {
        // No forward has run, so no KV state exists to snapshot; the
        // refusal names that instead of building an empty cache the
        // caller never asked for.
        let c = CpuCompute::new();
        let err = c.cache_checkpoint(4).expect_err("no KV state yet");
        match err {
            GlimmerError::StepFailed { step, detail } => {
                assert_eq!(step, "cache_checkpoint");
                assert!(detail.contains("no KV state"), "{detail}");
            }
            other => panic!("expected StepFailed, got {other:?}"),
        }
        // Rewind and release against the same empty backend refuse
        // the same way.
        let other = CpuCompute::new();
        let mut cache = KvCache::new(&[AttnLayerKind::Global], 1, 2).expect("cache builds");
        let handle = cache.checkpoint(1).expect("checkpoint");
        let err = other.cache_rewind(&handle, 0).expect_err("no KV state");
        assert!(matches!(
            err,
            GlimmerError::StepFailed {
                step: "cache_rewind",
                ..
            }
        ));
        let err = other.cache_release(&handle).expect_err("no KV state");
        assert!(matches!(
            err,
            GlimmerError::StepFailed {
                step: "cache_release",
                ..
            }
        ));
        cache.release(&handle).expect("release");
    }

    #[test]
    fn attn_gate_applies_sigmoid() {
        let c = CpuCompute::new();
        let out = c.attn_gate(&[2.0, -4.0], &[0.0, 2.0]).expect("gate");
        assert_eq!(out[0], 1.0); // 2 * sigmoid(0) = 2 * 0.5
                                 // -4 * sigmoid(2) = -4 * 0.88079707...
        assert!((out[1] - -3.523_188_3).abs() < 1e-5);
    }

    #[test]
    fn attn_gate_length_mismatch_refuses() {
        let c = CpuCompute::new();
        let err = c.attn_gate(&[1.0], &[1.0, 2.0]).expect_err("mismatch");
        assert!(matches!(
            err,
            GlimmerError::StepFailed {
                step: "attn_gate",
                ..
            }
        ));
    }

    #[test]
    fn add_length_mismatch_refuses() {
        let c = CpuCompute::new();
        let err = c.add(&[1.0, 2.0], &[1.0]).expect_err("mismatch");
        assert!(matches!(err, GlimmerError::StepFailed { step: "add", .. }));
    }

    #[test]
    fn proj_matvec_and_matmul_paths_agree() {
        let c = CpuCompute::new();
        // Weight stored [in=2, out=2], data rows are output rows:
        // [[1, 2], [3, 4]].
        let w = f32_view("w", &[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let one = c.proj(&[5.0, 6.0], &w, 1).expect("matvec path");
        assert_eq!(one, vec![17.0, 39.0]);
        let two = c.proj(&[5.0, 6.0, 7.0, 8.0], &w, 2).expect("matmul path");
        assert_eq!(two, vec![17.0, 39.0, 23.0, 53.0]);
    }

    #[test]
    fn proj_shape_mismatch_refuses() {
        let c = CpuCompute::new();
        let w = f32_view("w", &[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let err = c.proj(&[1.0, 2.0, 3.0], &w, 1).expect_err("bad input");
        assert!(matches!(err, GlimmerError::StepFailed { step: "proj", .. }));
    }

    #[test]
    fn rms_norm_normalizes_each_row_at_its_own_width() {
        let c = CpuCompute::new();
        // Per-head norm shape: 4 rows of width 2 under one weight.
        let w = f32_view("n", &[2], &[1.0, 1.0]);
        let out = c
            .rms_norm(&[3.0, 4.0, 3.0, 4.0, 0.0, 2.0, 1.0, 0.0], &w, 4, 1e-6)
            .expect("norm");
        // Row [3, 4]: rms = sqrt((9 + 16) / 2) = sqrt(12.5).
        let inv = 1.0 / (12.5f32 + 1e-6).sqrt();
        assert!((out[0] - 3.0 * inv).abs() < 1e-6);
        assert!((out[1] - 4.0 * inv).abs() < 1e-6);
        // Row [0, 2]: rms = sqrt(2).
        let inv2 = 1.0 / (2.0f32 + 1e-6).sqrt();
        assert!((out[5] - 2.0 * inv2).abs() < 1e-6);
    }

    #[test]
    fn embed_reads_q8_0_rows() {
        let c = CpuCompute::new();
        // Table of 4 tokens x hidden 32, Q8_0: one 34-byte block per
        // row, f16 scale 1.0 (0x3c00) then 32 signed bytes. Row t
        // holds the values t, t, ..., t pattern shifted by element.
        let mut data = Vec::new();
        for t in 0..4i8 {
            data.extend_from_slice(&0x3c00u16.to_le_bytes());
            for i in 0..32i8 {
                data.push((t * 10 + i) as u8);
            }
        }
        let mut cfg = call_cfg();
        cfg.hidden = 32;
        let table = TensorView {
            name: "token_embd.weight".to_string(),
            dtype: Dtype::Q8_0,
            shape: vec![32, 4],
            data: Arc::from(data.into_boxed_slice()),
        };
        let out = c.embed(&table, &[2], &cfg).expect("embed");
        assert_eq!(out.len(), 32);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, (20 + i) as f32);
        }
    }

    #[test]
    fn embed_unknown_dtype_refuses_by_name() {
        let c = CpuCompute::new();
        let cfg = call_cfg();
        let table = TensorView {
            name: "token_embd.weight".to_string(),
            dtype: Dtype::Unknown(77),
            shape: vec![4, 4],
            data: Arc::from(Vec::new().into_boxed_slice()),
        };
        let err = c.embed(&table, &[0], &cfg).expect_err("unknown dtype");
        match err {
            GlimmerError::StepFailed { step, detail } => {
                assert_eq!(step, "embed");
                assert!(detail.contains("token_embd.weight"), "{detail}");
                assert!(detail.contains("77"), "{detail}");
            }
            other => panic!("expected StepFailed, got {other:?}"),
        }
    }
}
