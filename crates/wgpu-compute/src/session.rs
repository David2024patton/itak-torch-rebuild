//! The device-resident forward: [`ResidentSession`].
//!
//! The `Compute` trait path (see [`crate::compute`]) is the
//! backend's correctness reference, but its currency is host
//! slices: every step uploads its input and reads its output
//! back, and the attention step reads the rotated keys back to
//! the host mirror once per layer per token. A Glimmer decode
//! token through that path is on the order of a thousand
//! blocking submissions and measures at 1.98 tok/s against
//! llama.cpp's 44.31 (bench/gpu-proof-2026-10-30.md), while the
//! kernels themselves bench at llama.cpp's bandwidth in
//! isolation. The composition is the cost, not the math.
//!
//! This module is the composition fixed. A session uploads the
//! whole weight set once, owns the activation scratch, the
//! per-layer KV buffers ([`DeviceKv`], the storage half of the
//! trait path's `GpuKvState`), and the logits buffer, and runs
//! each forward chunk as ONE command-buffer chain: the
//! embedding gather (K8, token ids read from a device buffer),
//! the weightless embedding norm, then per layer the exact
//! step order `muse_glimmer::GlimmerModel::forward_rows` walks
//! (norms K4 with the fused post-norm-plus-residual entry,
//! projections K1/K2/K3, the fused attention preparation K9
//! (the Q/K norms, the sliding layers' RoPE, and the KV append
//! as one dispatch), attention K6, gate and SwiGLU K7), the
//! final
//! norm, the head, and the K8 finishing, with the logits copy
//! into the readback staging appended to the same encoder. The
//! only host traffic per chunk is the token ids in (one queue
//! write of a few bytes per token) and the finished logits out
//! (one map). Nothing else crosses the bus.
//!
//! One source of truth for the step order: this driver mirrors
//! `GlimmerModel::forward_rows` line for line, and the crate's
//! resident-session test runs the same synthetic model through
//! the CPU reference, the trait path, and this path and
//! compares every token, so a step-order drift in either
//! driver fails a gate. The settled conventions are the
//! model's own (see `muse-glimmer`'s model docs): post norms
//! normalize the branch at the hardcoded epsilon below and
//! only then add the untouched residual; RoPE applies to the
//! sliding-window layers only; the head finishing scales
//! first, then softcaps. Argmax stays on the host: the
//! session returns the finished logits, and the caller picks
//! the token exactly as it does from the trait path.
//!
//! Buffer discipline: activations ping-pong between two
//! stream buffers, one chunk of rows at a time. The fused
//! post-norm entry (K4) is what closes each half-layer: it
//! reads the residual stream and the branch and writes the
//! other stream, so no separate add pass exists anywhere in
//! the chain.
//!
//! Debug: session construction and forward log buffer sizes,
//! positions, and submission counts through `crate::diag`;
//! the counts themselves are also returned in
//! [`SessionStats`].

use std::collections::HashMap;

use gguf::ggml::GgmlType;
use muse_glimmer::{
    Dtype, GlimmerConfig, GlimmerError, LayerKind, LayerWeights, Result, TensorView, Weights,
};

use crate::compute::WgpuCompute;
use crate::device::GpuContext;
use crate::diag::dbg_log;
use crate::error::WgpuError;
use crate::kernels::k1_q4k_gemv::{K1ActGeom, K1Call, K1Weights};
use crate::kernels::k2_q6k_gemv::{K2Call, K2Weights};
use crate::kernels::k3_q5k_gemv::{K3Call, K3Weights};
use crate::kernels::k4_rmsnorm::{K4Call, CHAIN_MAX_WIDTH};
use crate::kernels::k6_attention::{AttendGeom, DeviceKv, K6FlashCall, K6ReduceCall, K6ScoresCall};
use crate::kernels::k7_swiglu::K7Call;
use crate::kernels::k8_head::{K8FinishCall, K8GatherRowsCall, GATHER_NORM_MAX_WIDTH};
use crate::kernels::k9_attn_prep::{K9PrepCall, PrepArgs};

/// The post-norm epsilon, hardcoded to match the reference
/// converter and `muse_glimmer::model`'s own `POST_NORM_EPS`
/// (not file metadata, and not exported by that crate). The
/// resident-session cross-path test pins this value's effect:
/// if either side's epsilon drifts, the token streams diverge
/// and the gate fails.
const POST_NORM_EPS: f32 = 1e-8;

/// Most rows one forward chunk processes. Decode forwards one
/// row; prefill chunks the prompt under this cap and the
/// scores budget below.
const MAX_CHUNK_ROWS: usize = 64;

/// Cap on the attention scores scratch, in elements
/// (rows x query heads x stored positions): 2^24 f32 values,
/// 64 MiB. The chunk row count shrinks under long contexts so
/// the scores buffer never outgrows this.
const SCORES_BUDGET_ELEMS: usize = 1 << 24;

/// The windowed layers' ring capacity in a session:
/// `sliding_window + MAX_CHUNK_ROWS - 1`. The headroom is a
/// correctness requirement, not tuning. A chunk appends all
/// its rows before its attention runs, so the oldest query in
/// the chunk (at `start`) still needs its whole window
/// `[start - window + 1, start]` resident after the newest
/// row (at `start + rows - 1`) has been appended: the ring
/// must span `window + rows - 1` positions or the appends
/// evict keys the chunk's own early queries attend to (and a
/// fully-masked query row reduces to 0/0). The mask window
/// stays the config's window; only the storage ring grows.
fn ring_capacity(cfg: &GlimmerConfig) -> usize {
    cfg.sliding_window as usize + MAX_CHUNK_ROWS - 1
}

/// Wrap a backend failure in the model's step-failure shape,
/// under the same step names the trait path reports, so a
/// refusal reads identically whichever path raised it.
fn we(error: WgpuError, step: &'static str) -> GlimmerError {
    GlimmerError::StepFailed {
        step,
        detail: error.to_string(),
    }
}

/// Counters a session accumulates over its life: the receipt
/// the resident path exists to produce. `submissions` counts
/// forward submissions only; KV growth copies (geometric,
/// amortized away over a run) count separately in
/// `aux_submissions` so the steady-state number is never
/// dressed up.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SessionStats {
    /// Forward calls served.
    pub forwards: u64,
    /// Chunks those forwards split into (one submission each).
    pub chunks: u64,
    /// Queue submissions carrying a forward chain.
    pub submissions: u64,
    /// KV growth copy submissions outside the forward chains.
    pub aux_submissions: u64,
    /// Logits readbacks (one per chunk).
    pub readbacks: u64,
    /// Compute dispatches recorded (across all chunks).
    pub dispatches: u64,
    /// Tokens forwarded.
    pub tokens: u64,
}

/// A projection matrix after the session's one-time upload:
/// the kernel weight handle of whichever family the tensor's
/// dtype belongs to. Selection mirrors the trait path's
/// `ensure_uploaded` exactly (Q4_K, Q6_K, Q5_K; everything
/// else refused by name), so the two paths accept and refuse
/// the same files.
#[derive(Debug)]
enum ProjWeights {
    /// Q4_K weights for K1.
    Q4K(K1Weights),
    /// Q6_K weights for K2.
    Q6K(K2Weights),
    /// Q5_K weights for K3.
    Q5K(K3Weights),
}

impl ProjWeights {
    /// Element count of the resident weight bytes, for logging.
    fn byte_len(&self) -> usize {
        match self {
            ProjWeights::Q4K(w) => w.byte_len as usize,
            ProjWeights::Q6K(w) => w.byte_len as usize,
            ProjWeights::Q5K(w) => w.byte_len as usize,
        }
    }

    /// Total output rows this matrix produces per token.
    fn out_rows(&self) -> usize {
        match self {
            ProjWeights::Q4K(w) => w.out_rows,
            ProjWeights::Q6K(w) => w.out_rows,
            ProjWeights::Q5K(w) => w.out_rows,
        }
    }
}

/// One prepared projection dispatch in a forward plan.
#[derive(Debug)]
enum ProjCall {
    /// K1 dispatch.
    Q4K(K1Call),
    /// K2 dispatch.
    Q6K(K2Call),
    /// K3 dispatch.
    Q5K(K3Call),
}

impl ProjCall {
    /// Record this dispatch into the open compute `pass`.
    fn encode_into(&self, pass: &mut wgpu::ComputePass<'_>) {
        match self {
            ProjCall::Q4K(c) => c.encode_into(pass),
            ProjCall::Q6K(c) => c.encode_into(pass),
            ProjCall::Q5K(c) => c.encode_into(pass),
        }
    }
}

/// Upload one projection tensor for the session, once. The
/// refusal text matches the trait path's projection entry, so
/// a dtype with no GEMV family fails the same way on both
/// paths.
fn upload_proj(
    backend: &WgpuCompute,
    view: &TensorView,
    step: &'static str,
) -> Result<ProjWeights> {
    if view.shape.len() != 2 {
        return Err(GlimmerError::StepFailed {
            step,
            detail: format!(
                "tensor {} has shape {:?}; expected [in, out]",
                view.name, view.shape
            ),
        });
    }
    let (inn, out) = (view.shape[0] as usize, view.shape[1] as usize);
    let ctx = backend.context();
    match view.dtype {
        Dtype::Q4K => backend
            .k1()
            .upload(ctx, &view.name, &view.data, inn, out)
            .map(ProjWeights::Q4K)
            .map_err(|e| we(e, step)),
        Dtype::Q6K => backend
            .k2()
            .upload(ctx, &view.name, &view.data, inn, out)
            .map(ProjWeights::Q6K)
            .map_err(|e| we(e, step)),
        Dtype::Q5K => backend
            .k3()
            .upload(ctx, &view.name, &view.data, inn, out)
            .map(ProjWeights::Q5K)
            .map_err(|e| we(e, step)),
        other => Err(GlimmerError::StepFailed {
            step,
            detail: format!(
                "tensor {} has dtype {other:?}; the gemv kernels cover Q4_K, Q6_K, and Q5_K",
                view.name
            ),
        }),
    }
}

/// Prepare one projection dispatch over resident buffers.
fn prepare_proj(
    backend: &WgpuCompute,
    weights: &ProjWeights,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    batch: usize,
    step: &'static str,
) -> Result<ProjCall> {
    let ctx = backend.context();
    let batch = batch as u32;
    match weights {
        ProjWeights::Q4K(w) => backend
            .k1()
            .prepare(ctx, w, input, output, batch)
            .map(ProjCall::Q4K)
            .map_err(|e| we(e, step)),
        ProjWeights::Q6K(w) => backend
            .k2()
            .prepare(ctx, w, input, output, batch)
            .map(ProjCall::Q6K)
            .map_err(|e| we(e, step)),
        ProjWeights::Q5K(w) => backend
            .k3()
            .prepare(ctx, w, input, output, batch)
            .map(ProjCall::Q5K)
            .map_err(|e| we(e, step)),
    }
}

/// Dequantize one norm weight to f32 on the host and upload it
/// once. The dtype acceptance mirrors the trait path's
/// `rms_norm` conversion exactly: F32, F16, BF16, Q8_0, Q4_K,
/// Q5_K, Q6_K dequantize; anything else refuses by name.
fn upload_norm(
    backend: &WgpuCompute,
    view: &TensorView,
    step: &'static str,
) -> Result<wgpu::Buffer> {
    let ty = match view.dtype {
        Dtype::F32 => GgmlType::F32,
        Dtype::F16 => GgmlType::F16,
        Dtype::Bf16 => GgmlType::Bf16,
        Dtype::Q8_0 => GgmlType::Q8_0,
        Dtype::Q4K => GgmlType::Q4K,
        Dtype::Q5K => GgmlType::Q5K,
        Dtype::Q6K => GgmlType::Q6K,
        other => {
            return Err(GlimmerError::StepFailed {
                step,
                detail: format!(
                    "norm weight {} has dtype {other:?}; rms_norm dequantizes f32, f16, bf16, \
                     Q8_0, Q4_K, Q5_K, or Q6_K",
                    view.name
                ),
            })
        }
    };
    if view.shape.len() != 1 || view.shape[0] == 0 {
        return Err(GlimmerError::StepFailed {
            step,
            detail: format!(
                "norm weight {} has shape {:?}; expected a non-empty row vector",
                view.name, view.shape
            ),
        });
    }
    let n = view.shape[0] as usize;
    let values = gguf::dequant::dequantize(ty, &view.data, n);
    Ok(backend.context().upload_f32(&view.name, &values))
}

/// Upload several same-input-width Q4_K projection tensors
/// as ONE concatenated weight set (see
/// [`crate::kernels::k1_q4k_gemv::K1Pipeline::upload_multi`]):
/// the batched form the plan builder dispatches once for what
/// were several projections. The shape check matches
/// [`upload_proj`]'s, so a malformed part refuses the same way
/// on both paths.
fn upload_batch(
    backend: &WgpuCompute,
    views: &[&TensorView],
    step: &'static str,
) -> Result<ProjWeights> {
    let mut parts = Vec::with_capacity(views.len());
    for view in views {
        if view.shape.len() != 2 {
            return Err(GlimmerError::StepFailed {
                step,
                detail: format!(
                    "tensor {} has shape {:?}; expected [in, out]",
                    view.name, view.shape
                ),
            });
        }
        parts.push((
            view.data.as_ref(),
            view.shape[0] as usize,
            view.shape[1] as usize,
        ));
    }
    backend
        .k1()
        .upload_multi(backend.context(), &views[0].name, &parts)
        .map(ProjWeights::Q4K)
        .map_err(|e| we(e, step))
}

/// Read an individual projection handle a batching decision
/// was supposed to leave behind: unreachable by construction
/// (the upload keeps the handle exactly when no batch or pair
/// covers it), refused in the session's shape if it happens.
fn need(w: &Option<ProjWeights>) -> Result<&ProjWeights> {
    w.as_ref().ok_or_else(|| GlimmerError::StepFailed {
        step: "session",
        detail: "a layer weight is missing beside its batching decision".to_string(),
    })
}

/// One layer's resident weights: projection handles plus the
/// six norm vectors as f32 buffers. Projections the plan can
/// batch or pair (the Q/K/V set, the FFN gate/up pair) upload
/// concatenated when their dtypes allow it, and the individual
/// handles beside a batch stay `None`: a layer's weights live
/// on the device exactly once, in whichever form its plan
/// dispatches.
#[derive(Debug)]
struct ResidentLayer {
    /// Sliding or global, from the config pattern.
    kind: LayerKind,
    /// Attention norm vector, `[hidden]`.
    attn_norm: wgpu::Buffer,
    /// Query norm vector, `[head_dim]`.
    attn_q_norm: wgpu::Buffer,
    /// Key norm vector, `[head_dim]`.
    attn_k_norm: wgpu::Buffer,
    /// Post-attention norm vector, `[hidden]`.
    post_attention_norm: wgpu::Buffer,
    /// Feed-forward norm vector, `[hidden]`.
    ffn_norm: wgpu::Buffer,
    /// Post-feed-forward norm vector, `[hidden]`.
    post_ffw_norm: wgpu::Buffer,
    /// The concatenated attention weights: Q, K, V (plus the
    /// gate segment when [`ResidentLayer::batch_gate`]
    /// holds), or, when V is another family, the partial
    /// batch Q, K, gate (see [`ResidentLayer::batch_has_v`]).
    attn_batch: Option<ProjWeights>,
    /// Whether the batch carries the value projection. When
    /// it does not, V keeps its own projection and buffer,
    /// and the gate rides as the batch's third segment.
    batch_has_v: bool,
    /// Whether the batch carries the attention gate as its
    /// last segment (only when the output projection is Q4_K
    /// and so can consume the gate fused; see the plan
    /// builder).
    batch_gate: bool,
    /// Query projection (absent when batched).
    attn_q: Option<ProjWeights>,
    /// Key projection (absent when batched).
    attn_k: Option<ProjWeights>,
    /// Value projection (absent when batched).
    attn_v: Option<ProjWeights>,
    /// Attention gate projection (absent when it rides the
    /// batch as the fourth segment).
    attn_gate: Option<ProjWeights>,
    /// Attention output projection.
    attn_output: ProjWeights,
    /// The concatenated FFN gate+up weights, present when
    /// both are Q4_K and the down projection can consume the
    /// pair with the SwiGLU activation fused (a Q4_K or Q6_K
    /// down; see the plan builder).
    ffn_pair: Option<ProjWeights>,
    /// FFN gate projection (absent when paired).
    ffn_gate: Option<ProjWeights>,
    /// FFN up projection (absent when paired).
    ffn_up: Option<ProjWeights>,
    /// FFN down projection.
    ffn_down: ProjWeights,
}

impl ResidentLayer {
    /// Upload one layer's weights for the session.
    fn upload(backend: &WgpuCompute, w: &LayerWeights, kind: LayerKind) -> Result<Self> {
        let mat = |v: &TensorView| v.shape.len() == 2;
        // The attention batch: Q, K, V concatenated when all
        // three are Q4_K, with the gate riding as a fourth
        // segment when it and the output projection are also
        // Q4_K (the output projection then consumes the gate
        // fused and the separate gate projection disappears).
        // When V is another family (a Q6_K V on half the real
        // model's layers) but Q, K, the gate, and the output
        // projection are Q4_K, the partial batch Q, K, gate
        // still forms and V keeps its own projection: the
        // three fused projections share one activation and sit
        // adjacent in the step order either way.
        let qk_q4k = [&w.attn_q, &w.attn_k]
            .iter()
            .all(|v| v.dtype == Dtype::Q4K && mat(v));
        let v_q4k = w.attn_v.dtype == Dtype::Q4K && mat(&w.attn_v);
        let gate_fusable = w.attn_gate.dtype == Dtype::Q4K
            && mat(&w.attn_gate)
            && w.attn_output.dtype == Dtype::Q4K
            && mat(&w.attn_output);
        let (attn_batch, batch_has_v, batch_gate) = if qk_q4k && v_q4k {
            let mut views = vec![&w.attn_q, &w.attn_k, &w.attn_v];
            if gate_fusable {
                views.push(&w.attn_gate);
            }
            (
                Some(upload_batch(backend, &views, "proj")?),
                true,
                gate_fusable,
            )
        } else if qk_q4k && gate_fusable {
            (
                Some(upload_batch(backend, &[&w.attn_q, &w.attn_k, &w.attn_gate], "proj")?),
                false,
                true,
            )
        } else {
            (None, false, false)
        };
        let batched = attn_batch.is_some();
        // The FFN pair: gate and up concatenated when both
        // are Q4_K and the down projection's family carries
        // the fused SwiGLU form (K1 or K2); any other down
        // keeps the separate projections and the K7 stage.
        let down_fuses = matches!(w.ffn_down.dtype, Dtype::Q4K | Dtype::Q6K) && mat(&w.ffn_down);
        let ffn_pair = if down_fuses
            && w.ffn_gate.dtype == Dtype::Q4K
            && mat(&w.ffn_gate)
            && w.ffn_up.dtype == Dtype::Q4K
            && mat(&w.ffn_up)
        {
            Some(upload_batch(backend, &[&w.ffn_gate, &w.ffn_up], "swiglu_ffn")?)
        } else {
            None
        };
        let paired = ffn_pair.is_some();
        Ok(ResidentLayer {
            kind,
            attn_norm: upload_norm(backend, &w.attn_norm, "rms_norm")?,
            attn_q_norm: upload_norm(backend, &w.attn_q_norm, "rms_norm")?,
            attn_k_norm: upload_norm(backend, &w.attn_k_norm, "rms_norm")?,
            post_attention_norm: upload_norm(backend, &w.post_attention_norm, "rms_norm")?,
            ffn_norm: upload_norm(backend, &w.ffn_norm, "rms_norm")?,
            post_ffw_norm: upload_norm(backend, &w.post_ffw_norm, "rms_norm")?,
            attn_batch,
            batch_has_v,
            batch_gate,
            attn_q: if batched {
                None
            } else {
                Some(upload_proj(backend, &w.attn_q, "proj")?)
            },
            attn_k: if batched {
                None
            } else {
                Some(upload_proj(backend, &w.attn_k, "proj")?)
            },
            attn_v: if batch_has_v {
                None
            } else {
                Some(upload_proj(backend, &w.attn_v, "proj")?)
            },
            attn_gate: if batch_gate {
                None
            } else {
                Some(upload_proj(backend, &w.attn_gate, "proj")?)
            },
            attn_output: upload_proj(backend, &w.attn_output, "proj")?,
            ffn_pair,
            ffn_gate: if paired {
                None
            } else {
                Some(upload_proj(backend, &w.ffn_gate, "swiglu_ffn")?)
            },
            ffn_up: if paired {
                None
            } else {
                Some(upload_proj(backend, &w.ffn_up, "swiglu_ffn")?)
            },
            ffn_down: upload_proj(backend, &w.ffn_down, "swiglu_ffn")?,
        })
    }

    /// Resident weight bytes across the layer, for the build
    /// log line.
    fn weight_bytes(&self) -> usize {
        let opt = |w: &Option<ProjWeights>| w.as_ref().map_or(0, |p| p.byte_len());
        opt(&self.attn_batch)
            + opt(&self.attn_q)
            + opt(&self.attn_k)
            + opt(&self.attn_v)
            + opt(&self.attn_gate)
            + self.attn_output.byte_len()
            + opt(&self.ffn_pair)
            + opt(&self.ffn_gate)
            + opt(&self.ffn_up)
            + self.ffn_down.byte_len()
    }
}

/// Activation scratch for one chunk geometry. Stream buffers
/// A and B ping-pong: the embedding gather lands in B, the
/// weightless norm writes A, each layer's post-attention fused
/// norm reads A and writes B, its post-FFN fused norm reads B
/// and writes A, so every layer enters with the residual in A
/// and leaves it in A.
#[derive(Debug)]
struct Scratch {
    /// Token id buffer, `[rows]` u32, rewritten per chunk.
    tokens: wgpu::Buffer,
    /// Residual stream A, `[rows][hidden]`.
    stream_a: wgpu::Buffer,
    /// Residual stream B, `[rows][hidden]`.
    stream_b: wgpu::Buffer,
    /// Normed hidden rows, `[rows][hidden]`.
    h: wgpu::Buffer,
    /// Post-norm branch rows, `[rows][hidden]`.
    branch: wgpu::Buffer,
    /// Batched QKV projection rows, `[rows][batch width]`:
    /// the concatenated projection's output, laid out per
    /// layer as Q, K, V (and the gate) segments. Sized for the
    /// widest batch among the layers; a layer whose batch is
    /// narrower simply packs its rows tighter at the front.
    qkv: wgpu::Buffer,
    /// Paired FFN gate+up rows, `[rows][2 * ffn]`: the paired
    /// projection's output, gate segment first.
    ffn_pair: wgpu::Buffer,
    /// Query rows as projected, `[rows][q_width]`.
    q: wgpu::Buffer,
    /// Query rows after the Q norm (and, on sliding layers,
    /// the rotation), `[rows][q_width]`: the fused preparation
    /// (K9) writes here from the raw projection rows (a buffer
    /// may not be bound read-only and read-write in one
    /// dispatch, so no step in this design aliases its input),
    /// and the scores pass reads it. Keys need no such buffer:
    /// K9 writes the prepared keys straight into the KV store.
    q_normed: wgpu::Buffer,
    /// Key rows as projected, `[rows][kv_width]`.
    k: wgpu::Buffer,
    /// Value rows, `[rows][kv_width]`.
    v: wgpu::Buffer,
    /// Attention output rows from the reduce, `[rows][q_width]`.
    attn: wgpu::Buffer,
    /// Attention rows after the gate multiply, `[rows][q_width]`.
    attn_gated: wgpu::Buffer,
    /// Attention gate rows, `[rows][q_width]`.
    gate: wgpu::Buffer,
    /// FFN gate rows, `[rows][ffn]`.
    ffn_gate: wgpu::Buffer,
    /// FFN up rows, `[rows][ffn]`.
    ffn_up: wgpu::Buffer,
    /// FFN activated rows, `[rows][ffn]`.
    ffn_act: wgpu::Buffer,
    /// Attention scores, `[rows][q_heads][scores_cap]`.
    scores: wgpu::Buffer,
    /// The last row of stream A, copied out for the head,
    /// `[hidden]`.
    last_row: wgpu::Buffer,
    /// The normed last row, `[hidden]`.
    head_in: wgpu::Buffer,
    /// Finished logits, `[vocab]`.
    logits: wgpu::Buffer,
    /// Logits readback staging, `[vocab]`, `COPY_DST | MAP_READ`.
    staging: wgpu::Buffer,
}

impl Scratch {
    /// Allocate the scratch for `rows` chunk rows with a
    /// scores capacity of `scores_cap` stored positions.
    /// `qkv_width` is the widest batched-QKV row among the
    /// layers (0 when no layer batches; the buffer then takes
    /// a token element and is never bound).
    fn new(
        ctx: &GpuContext,
        cfg: &GlimmerConfig,
        rows: usize,
        scores_cap: usize,
        qkv_width: usize,
    ) -> Self {
        let act = |label: &str, elems: usize| ctx.scratch_buffer(label, (elems * 4) as u64);
        let hidden = cfg.hidden as usize;
        let q_width = cfg.q_width() as usize;
        let kv_width = cfg.k_width() as usize;
        let ffn = cfg.ffn_hidden as usize;
        let vocab = cfg.vocab_size as usize;
        Scratch {
            tokens: ctx.scratch_buffer("session tokens", (rows * 4) as u64),
            stream_a: act("session stream a", rows * hidden),
            stream_b: act("session stream b", rows * hidden),
            h: act("session h", rows * hidden),
            branch: act("session branch", rows * hidden),
            qkv: act("session qkv", rows * qkv_width.max(1)),
            ffn_pair: act("session ffn pair", rows * 2 * ffn),
            q: act("session q", rows * q_width),
            q_normed: act("session q normed", rows * q_width),
            k: act("session k", rows * kv_width),
            v: act("session v", rows * kv_width),
            attn: act("session attn", rows * q_width),
            attn_gated: act("session attn gated", rows * q_width),
            gate: act("session gate", rows * q_width),
            ffn_gate: act("session ffn gate", rows * ffn),
            ffn_up: act("session ffn up", rows * ffn),
            ffn_act: act("session ffn act", rows * ffn),
            scores: act("session scores", rows * cfg.q_heads as usize * scores_cap),
            last_row: act("session last row", hidden),
            head_in: act("session head in", hidden),
            logits: act("session logits", vocab),
            staging: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("session logits staging"),
                size: (vocab * 4) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
        }
    }
}

/// One layer's prepared dispatches within a chunk plan, in
/// forward order. The position-dependent calls (prep, scores,
/// reduce) are refreshed per chunk through their setters
/// before encoding; everything else is fixed at plan build.
#[derive(Debug)]
struct LayerPlan {
    /// `rms_norm(h = norm(stream_a, attn_norm))` (absent for
    /// layers past the first on chained plans: the previous
    /// layer's boundary chain already produced this layer's
    /// `h`).
    attn_norm: Option<K4Call>,
    /// Q projection (or the whole batched QKV projection,
    /// when the layer batches), h to q (or to the qkv rows).
    proj_q: ProjCall,
    /// K projection (absent when batched).
    proj_k: Option<ProjCall>,
    /// V projection (absent when batched).
    proj_v: Option<ProjCall>,
    /// The fused attention preparation (K9): the Q and K
    /// norms, the sliding layers' RoPE, and the KV append in
    /// one dispatch.
    prep: K9PrepCall,
    /// Attention scores over the resident KV (two-pass form,
    /// absent on decode plans).
    scores: Option<K6ScoresCall>,
    /// Attention reduce over the resident KV (two-pass form,
    /// absent on decode plans).
    reduce: Option<K6ReduceCall>,
    /// The fused flash attention (decode plans: one row,
    /// where the two-pass form's scores buffer and second
    /// dispatch are pure overhead; chunk plans keep two-pass).
    flash: Option<K6FlashCall>,
    /// Gate projection, h to gate (absent when the gate rides
    /// the QKV batch; present beside a fused output projection
    /// when the gate uploads separately).
    proj_gate: Option<ProjCall>,
    /// Gated multiply, q-attention rows by gate rows (absent
    /// when the output projection consumes the gate fused).
    gate_mul: Option<K7Call>,
    /// Output projection, attention rows to branch (with the
    /// gate activation fused in when `gate_mul` is absent).
    proj_o: ProjCall,
    /// Fused post-attention norm and residual add:
    /// stream_b = norm(branch) + stream_a (unchained plans
    /// only; chained plans fold it into `post_chain`).
    post_attn: Option<K4Call>,
    /// The two-stage chain: post-attention norm + residual
    /// and the FFN norm in one dispatch (chained plans only).
    post_chain: Option<K4Call>,
    /// `rms_norm(h = norm(stream_b, ffn_norm))` (unchained
    /// plans only).
    ffn_norm: Option<K4Call>,
    /// FFN gate projection (or the paired gate+up projection,
    /// when the layer pairs), h to the gate rows (or the pair
    /// rows).
    proj_fg: ProjCall,
    /// FFN up projection (absent when paired).
    proj_fu: Option<ProjCall>,
    /// SwiGLU activation, gate and up to act (absent when the
    /// down projection consumes the pair with the activation
    /// fused).
    swiglu: Option<K7Call>,
    /// FFN down projection, activated rows to branch.
    proj_fd: ProjCall,
    /// The boundary chain: post-FFN norm + residual and the
    /// NEXT layer's attention norm in one dispatch (chained
    /// plans, every layer but the last).
    boundary: Option<K4Call>,
    /// The close: post-FFN norm + residual with the head's
    /// final norm folded in, emitting the head input from
    /// the last row (chained plans, the last layer only).
    close: Option<K4Call>,
    /// Fused post-FFN norm and residual add:
    /// stream_a = norm(branch) + stream_b (unchained plans,
    /// and the last layer of chained plans).
    post_ffw: Option<K4Call>,
    /// Compute dispatches this layer records, counted at
    /// build from the fusions its weights qualified for.
    dispatch_count: u64,
}

impl LayerPlan {
    /// Refresh the position-dependent uniforms for a chunk
    /// whose KV view holds `stored` entries starting at
    /// absolute position `first_pos`, with queries at absolute
    /// position `start` onward.
    fn set_geometry(&mut self, ctx: &GpuContext, stored: usize, first_pos: usize, start: usize) {
        self.prep.set_start_position(ctx, start);
        if let Some(flash) = &mut self.flash {
            flash.set_geometry(ctx, stored, first_pos, start);
        }
        if let Some(scores) = &mut self.scores {
            scores.set_geometry(ctx, stored, first_pos, start);
        }
        if let Some(reduce) = &mut self.reduce {
            reduce.set_geometry(ctx, stored, first_pos);
        }
    }

    /// Record the pre-attention dispatches: the attention
    /// norm and the Q/K/V projections (one batched dispatch
    /// when the layer batches). The fused preparation opens
    /// [`LayerPlan::encode_attention`].
    fn encode_pre_attention(&self, pass: &mut wgpu::ComputePass<'_>) {
        if let Some(n) = &self.attn_norm {
            n.encode_into(pass);
        }
        self.proj_q.encode_into(pass);
        if let Some(k) = &self.proj_k {
            k.encode_into(pass);
        }
        if let Some(v) = &self.proj_v {
            v.encode_into(pass);
        }
    }

    /// Record the attention dispatches proper: the fused
    /// preparation (norms, rotation, append), then the fused
    /// flash on decode plans or scores + reduce on chunk
    /// plans.
    fn encode_attention(&self, pass: &mut wgpu::ComputePass<'_>) {
        self.prep.encode_into(pass);
        if let Some(flash) = &self.flash {
            flash.encode_into(pass);
        }
        if let Some(scores) = &self.scores {
            scores.encode_into(pass);
        }
        if let Some(reduce) = &self.reduce {
            reduce.encode_into(pass);
        }
    }

    /// Record the post-attention dispatches: the gate
    /// projection and multiply (each absent when the output
    /// projection fuses the gate), the output projection, the
    /// fused post-attention norm, and the whole FFN half
    /// closing with the fused post-FFN norm.
    fn encode_post_attention(&self, pass: &mut wgpu::ComputePass<'_>) {
        if let Some(g) = &self.proj_gate {
            g.encode_into(pass);
        }
        if let Some(m) = &self.gate_mul {
            m.encode_into(pass);
        }
        self.proj_o.encode_into(pass);
        if let Some(p) = &self.post_attn {
            p.encode_into(pass);
        }
        if let Some(c) = &self.post_chain {
            c.encode_into(pass);
        }
        if let Some(n) = &self.ffn_norm {
            n.encode_into(pass);
        }
        self.proj_fg.encode_into(pass);
        if let Some(u) = &self.proj_fu {
            u.encode_into(pass);
        }
        if let Some(s) = &self.swiglu {
            s.encode_into(pass);
        }
        self.proj_fd.encode_into(pass);
        if let Some(b) = &self.boundary {
            b.encode_into(pass);
        }
        if let Some(c) = &self.close {
            c.encode_into(pass);
        }
        if let Some(p) = &self.post_ffw {
            p.encode_into(pass);
        }
    }

    /// Compute dispatches this layer contributes, counted at
    /// build time from the fusions its weights qualified for
    /// (see the plan builder's accounting).
    fn dispatch_count(&self) -> u64 {
        self.dispatch_count
    }
}

/// A cached forward plan for one chunk geometry: the scratch
/// plus every prepared dispatch of the chain. Plans bind the
/// session's weight buffers, its scratch, and the KV buffers
/// as they stood at build time; a KV growth replaces a layer's
/// buffers, which invalidates the plan (the session compares
/// capacities and rebuilds).
#[derive(Debug)]
struct ChunkPlan {
    /// Chunk row count this plan serves.
    rows: usize,
    /// Scores capacity this plan's scratch was built with.
    scores_cap: usize,
    /// KV capacities per layer at build time.
    kv_caps: Vec<usize>,
    /// The chunk's activation scratch.
    scratch: Scratch,
    /// Embedding gather over the device token buffer (the
    /// fused gather+norm form when the hidden width fits
    /// its register residency, writing stream A directly).
    gather: K8GatherRowsCall,
    /// Weightless embedding norm, stream B to stream A
    /// (absent when the gather is the fused gather+norm).
    embed_norm: Option<K4Call>,
    /// Per-layer dispatch plans, in layer order.
    layers: Vec<LayerPlan>,
    /// Final norm over the copied-out last row (absent on
    /// chained plans: the last layer's close folds it in
    /// and emits the head input directly).
    final_norm: Option<K4Call>,
    /// Head projection, head input to logits.
    head: ProjCall,
    /// Head finishing over the logits, in place (absent when
    /// the head is Q5_K: the finishing map is fused into the
    /// head kernel's final store instead).
    finish: Option<K8FinishCall>,
    /// Compute dispatches one chunk of this plan records.
    dispatch_count: u64,
}

impl ChunkPlan {
    /// Build the plan: allocate scratch and prepare every
    /// dispatch against the session's resident buffers.
    ///
    /// # Errors
    ///
    /// The first kernel refusal, wrapped under its step name.
    #[allow(clippy::too_many_arguments)]
    fn build(
        backend: &WgpuCompute,
        cfg: &GlimmerConfig,
        layers: &[ResidentLayer],
        final_norm_w: &wgpu::Buffer,
        output: &ProjWeights,
        embedding: &wgpu::Buffer,
        embedding_dtype: Dtype,
        kv: &DeviceKv,
        rows: usize,
        scores_cap: usize,
    ) -> Result<Self> {
        let ctx = backend.context();
        let qkv_width = layers
            .iter()
            .filter_map(|l| l.attn_batch.as_ref().map(ProjWeights::out_rows))
            .max()
            .unwrap_or(0);
        let scratch = Scratch::new(ctx, cfg, rows, scores_cap, qkv_width);
        let hidden = cfg.hidden as usize;
        let head_dim = cfg.key_length;
        let q_width = cfg.q_width() as usize;
        let kv_width = cfg.k_width() as usize;
        let ffn = cfg.ffn_hidden as usize;
        let eps = cfg.rms_eps;
        // (The plan builder reads individual handles through
        // `need`, which refuses in the session's shape when a
        // batching decision left one behind; see below.)

        // The embedding gather and the weightless norm over
        // it: one fused dispatch when the hidden width fits
        // the fused kernel's register residency (the normed
        // rows land in stream A directly), the two dispatches
        // otherwise.
        let (gather, embed_norm) = if hidden <= GATHER_NORM_MAX_WIDTH {
            let gather = backend.k8().prepare_gather_rows_norm(
                ctx,
                embedding,
                &scratch.tokens,
                &scratch.stream_a,
                rows,
                hidden,
                eps,
                embedding_dtype,
            )?;
            (gather, None)
        } else {
            let gather = backend.k8().prepare_gather_rows(
                ctx,
                embedding,
                &scratch.tokens,
                &scratch.stream_b,
                rows,
                hidden,
                embedding_dtype,
            )?;
            let embed_norm = backend.k4().prepare_rms_norm_plain(
                ctx,
                &scratch.stream_b,
                &scratch.stream_a,
                rows,
                hidden,
                eps,
            );
            (gather, Some(embed_norm))
        };

        let mut layer_plans = Vec::with_capacity(layers.len());
        let mut kv_caps = Vec::with_capacity(layers.len());
        for (i, layer) in layers.iter().enumerate() {
            let kv_layer = kv.layer(i, "attention")?;
            kv_caps.push(kv_layer.capacity);
            // The append slots positions modulo the ring
            // capacity (window plus chunk headroom), and K6
            // reads them back modulo the same ring; the mask
            // window stays the config's window.
            let window = match layer.kind {
                LayerKind::SlidingWindow => Some(cfg.sliding_window as usize),
                LayerKind::Global => None,
            };
            let ring = window.map(|_| ring_capacity(cfg));
            // The norm chains (K4's two-stage form) run when
            // the hidden width fits their register residency;
            // wider models keep the separate norm dispatches.
            let chained = hidden <= CHAIN_MAX_WIDTH;
            let last = i + 1 == layers.len();
            let attn_norm = if chained && i > 0 {
                // The previous layer's boundary chain writes
                // this layer's h.
                None
            } else {
                Some(backend.k4().prepare_rms_norm(
                    ctx,
                    &scratch.stream_a,
                    &layer.attn_norm,
                    &scratch.h,
                    rows,
                    hidden,
                    eps,
                ))
            };
            // The attention projections and the fused
            // preparation (K9: the Q/K norms, the sliding
            // layers' rotation, and the KV append in one
            // dispatch). Batched layers project Q, K, V (and
            // the gate, when it rides along) in ONE dispatch
            // into the qkv rows, and the preparation reads its
            // segments out of them; a partial batch (Q, K,
            // gate) leaves V on its own projection and buffer,
            // and the preparation reads V from there (its own
            // stride). Unbatched layers keep the three
            // projections and the preparation reads the three
            // buffers. Either way the preparation's math is
            // the K4/K5 composition the trait path still runs
            // as the reference.
            let (proj_q, proj_k, proj_v, prep) = if let Some(batch) = &layer.attn_batch {
                let call = prepare_proj(backend, batch, &scratch.h, &scratch.qkv, rows, "proj")?;
                let stride = batch.out_rows();
                let (proj_v, v_in, v_stride, v_off) = if layer.batch_has_v {
                    (None, &scratch.qkv, stride, q_width + kv_width)
                } else {
                    let proj_v = prepare_proj(
                        backend,
                        need(&layer.attn_v)?,
                        &scratch.h,
                        &scratch.v,
                        rows,
                        "proj",
                    )?;
                    (Some(proj_v), &scratch.v, kv_width, 0)
                };
                let prep = backend
                    .k9()
                    .prepare(
                        ctx,
                        &PrepArgs {
                            q_in: &scratch.qkv,
                            q_stride: stride,
                            q_off: 0,
                            k_in: &scratch.qkv,
                            kv_stride: stride,
                            k_off: q_width,
                            v_in,
                            v_stride,
                            v_off,
                            q_w: &layer.attn_q_norm,
                            k_w: &layer.attn_k_norm,
                            q_out: &scratch.q_normed,
                            kv_k: &kv_layer.k,
                            kv_v: &kv_layer.v,
                            rows,
                            q_heads: cfg.q_heads as usize,
                            kv_heads: cfg.kv_heads as usize,
                            head_dim: head_dim as usize,
                            start_pos: 0,
                            ring,
                            rotate: layer.kind == LayerKind::SlidingWindow,
                            base: cfg.rope_freq_base_swa,
                            eps,
                        },
                    )
                    .map_err(|e| we(e, "attention"))?;
                (call, None, proj_v, prep)
            } else {
                let proj_q = prepare_proj(
                    backend,
                    need(&layer.attn_q)?,
                    &scratch.h,
                    &scratch.q,
                    rows,
                    "proj",
                )?;
                let proj_k = prepare_proj(
                    backend,
                    need(&layer.attn_k)?,
                    &scratch.h,
                    &scratch.k,
                    rows,
                    "proj",
                )?;
                let proj_v = prepare_proj(
                    backend,
                    need(&layer.attn_v)?,
                    &scratch.h,
                    &scratch.v,
                    rows,
                    "proj",
                )?;
                let prep = backend
                    .k9()
                    .prepare(
                        ctx,
                        &PrepArgs {
                            q_in: &scratch.q,
                            q_stride: q_width,
                            q_off: 0,
                            k_in: &scratch.k,
                            kv_stride: kv_width,
                            k_off: 0,
                            v_in: &scratch.v,
                            v_stride: kv_width,
                            v_off: 0,
                            q_w: &layer.attn_q_norm,
                            k_w: &layer.attn_k_norm,
                            q_out: &scratch.q_normed,
                            kv_k: &kv_layer.k,
                            kv_v: &kv_layer.v,
                            rows,
                            q_heads: cfg.q_heads as usize,
                            kv_heads: cfg.kv_heads as usize,
                            head_dim: head_dim as usize,
                            start_pos: 0,
                            ring,
                            rotate: layer.kind == LayerKind::SlidingWindow,
                            base: cfg.rope_freq_base_swa,
                            eps,
                        },
                    )
                    .map_err(|e| we(e, "attention"))?;
                (proj_q, Some(proj_k), Some(proj_v), prep)
            };
            let geom = AttendGeom {
                rows,
                q_heads: cfg.q_heads as usize,
                kv_heads: cfg.kv_heads as usize,
                head_dim: head_dim as usize,
                stored: 1,
                first_pos: 0,
                start_pos: 0,
                window,
                ring,
            };
            // Decode plans (one row) run the fused flash
            // attention; chunk plans keep the two-pass form,
            // whose scores grid amortizes the extra dispatch
            // over the chunk's rows.
            let (scores, reduce, flash) = if rows == 1 {
                let flash = backend
                    .k6()
                    .prepare_flash(ctx, &scratch.q_normed, kv_layer, &scratch.attn, &geom)
                    .map_err(|e| we(e, "attention"))?;
                (None, None, Some(flash))
            } else {
                let scores = backend
                    .k6()
                    .prepare_scores(ctx, &scratch.q_normed, kv_layer, &scratch.scores, &geom)
                    .map_err(|e| we(e, "attention"))?;
                let reduce = backend
                    .k6()
                    .prepare_reduce(ctx, &scratch.scores, kv_layer, &scratch.attn, &geom)
                    .map_err(|e| we(e, "attention"))?;
                (Some(scores), Some(reduce), None)
            };
            // The gate and the output projection. When the
            // output weights are Q4_K, the gate multiply fuses
            // into the output projection's activation stage
            // (K1's gated form, K7's exact sigmoid over the
            // same gate values), reading the gate rows from
            // the batch's fourth segment when they ride it, or
            // from their own projection otherwise; the K7
            // multiply dispatch disappears either way. Any
            // other output family keeps the trait composition:
            // separate gate projection, K7 multiply, plain
            // output projection.
            let (proj_gate, gate_mul, proj_o, gate_fused) =
                if let ProjWeights::Q4K(ow) = &layer.attn_output {
                    if layer.batch_gate {
                        let batch = need(&layer.attn_batch)?;
                        // The gate is the batch's last
                        // segment: fourth when V rides the
                        // batch, third on a partial batch.
                        let gate_off = if layer.batch_has_v {
                            q_width + 2 * kv_width
                        } else {
                            q_width + kv_width
                        };
                        let call = backend
                            .k1_gated()
                            .prepare_act(
                                ctx,
                                ow,
                                &scratch.attn,
                                &scratch.qkv,
                                &scratch.branch,
                                rows as u32,
                                &K1ActGeom {
                                    a_stride_elems: q_width,
                                    b_stride_elems: batch.out_rows(),
                                    b_off_elems: gate_off,
                                },
                            )
                            .map(ProjCall::Q4K)
                            .map_err(|e| we(e, "proj"))?;
                        (None, None, call, true)
                    } else if matches!(layer.attn_gate, Some(ProjWeights::Q4K(_))) {
                        let proj_gate = prepare_proj(
                            backend,
                            need(&layer.attn_gate)?,
                            &scratch.h,
                            &scratch.gate,
                            rows,
                            "proj",
                        )?;
                        let call = backend
                            .k1_gated()
                            .prepare_act(
                                ctx,
                                ow,
                                &scratch.attn,
                                &scratch.gate,
                                &scratch.branch,
                                rows as u32,
                                &K1ActGeom {
                                    a_stride_elems: q_width,
                                    b_stride_elems: q_width,
                                    b_off_elems: 0,
                                },
                            )
                            .map(ProjCall::Q4K)
                            .map_err(|e| we(e, "proj"))?;
                        (Some(proj_gate), None, call, true)
                    } else {
                        let proj_gate = prepare_proj(
                            backend,
                            need(&layer.attn_gate)?,
                            &scratch.h,
                            &scratch.gate,
                            rows,
                            "proj",
                        )?;
                        let gate_mul = backend.k7().prepare_gate(
                            ctx,
                            &scratch.attn,
                            &scratch.gate,
                            &scratch.attn_gated,
                            rows * q_width,
                        );
                        let proj_o = prepare_proj(
                            backend,
                            &layer.attn_output,
                            &scratch.attn_gated,
                            &scratch.branch,
                            rows,
                            "proj",
                        )?;
                        (Some(proj_gate), Some(gate_mul), proj_o, false)
                    }
                } else {
                    let proj_gate = prepare_proj(
                        backend,
                        need(&layer.attn_gate)?,
                        &scratch.h,
                        &scratch.gate,
                        rows,
                        "proj",
                    )?;
                    let gate_mul = backend.k7().prepare_gate(
                        ctx,
                        &scratch.attn,
                        &scratch.gate,
                        &scratch.attn_gated,
                        rows * q_width,
                    );
                    let proj_o = prepare_proj(
                        backend,
                        &layer.attn_output,
                        &scratch.attn_gated,
                        &scratch.branch,
                        rows,
                        "proj",
                    )?;
                    (Some(proj_gate), Some(gate_mul), proj_o, false)
                };
            // The post-attention close and the FFN norm: one
            // chained dispatch when chained (the intermediate
            // stream row stays in the kernel's registers
            // between the stages), the two dispatches
            // otherwise.
            let (post_attn, post_chain, ffn_norm) = if chained {
                let chain = backend
                    .k4()
                    .prepare_norm_chain(
                        ctx,
                        &scratch.stream_a,
                        &scratch.branch,
                        &layer.post_attention_norm,
                        &layer.ffn_norm,
                        &scratch.stream_b,
                        &scratch.h,
                        rows,
                        hidden,
                        POST_NORM_EPS,
                        eps,
                    )
                    .map_err(|e| we(e, "rms_norm"))?;
                (None, Some(chain), None)
            } else {
                let post_attn = backend.k4().prepare_norm_residual(
                    ctx,
                    &scratch.stream_a,
                    &scratch.branch,
                    &layer.post_attention_norm,
                    &scratch.stream_b,
                    rows,
                    hidden,
                    POST_NORM_EPS,
                );
                let ffn_norm = backend.k4().prepare_rms_norm(
                    ctx,
                    &scratch.stream_b,
                    &layer.ffn_norm,
                    &scratch.h,
                    rows,
                    hidden,
                    eps,
                );
                (Some(post_attn), None, Some(ffn_norm))
            };
            // The FFN half. Paired layers project gate and up
            // in ONE dispatch into the pair rows, and the down
            // projection consumes the pair with the SwiGLU
            // activation fused into its activation stage
            // (K1's or K2's SwiGLU form, K7's exact silu over
            // the same gate/up values): the up projection and
            // the K7 dispatch both disappear. Unpaired layers
            // keep the trait composition: two projections, the
            // K7 activation, the plain down projection.
            let (proj_fg, proj_fu, swiglu, proj_fd) = if let Some(pair) = &layer.ffn_pair {
                let proj_fg = prepare_proj(
                    backend,
                    pair,
                    &scratch.h,
                    &scratch.ffn_pair,
                    rows,
                    "swiglu_ffn",
                )?;
                let geom = K1ActGeom {
                    a_stride_elems: 2 * ffn,
                    b_stride_elems: 2 * ffn,
                    b_off_elems: ffn,
                };
                let proj_fd = match &layer.ffn_down {
                    ProjWeights::Q4K(dw) => backend
                        .k1_swiglu()
                        .prepare_act(
                            ctx,
                            dw,
                            &scratch.ffn_pair,
                            &scratch.ffn_pair,
                            &scratch.branch,
                            rows as u32,
                            &geom,
                        )
                        .map(ProjCall::Q4K)
                        .map_err(|e| we(e, "swiglu_ffn"))?,
                    ProjWeights::Q6K(dw) => backend
                        .k2_swiglu()
                        .prepare_act(
                            ctx,
                            dw,
                            &scratch.ffn_pair,
                            &scratch.ffn_pair,
                            &scratch.branch,
                            rows as u32,
                            &geom,
                        )
                        .map(ProjCall::Q6K)
                        .map_err(|e| we(e, "swiglu_ffn"))?,
                    _ => {
                        return Err(GlimmerError::StepFailed {
                            step: "session",
                            detail: "an ffn pair was uploaded beside a down projection \
                                     with no fused swiglu form"
                                .to_string(),
                        })
                    }
                };
                (proj_fg, None, None, proj_fd)
            } else {
                let proj_fg = prepare_proj(
                    backend,
                    need(&layer.ffn_gate)?,
                    &scratch.h,
                    &scratch.ffn_gate,
                    rows,
                    "swiglu_ffn",
                )?;
                let proj_fu = prepare_proj(
                    backend,
                    need(&layer.ffn_up)?,
                    &scratch.h,
                    &scratch.ffn_up,
                    rows,
                    "swiglu_ffn",
                )?;
                let swiglu = backend.k7().prepare_swiglu(
                    ctx,
                    &scratch.ffn_gate,
                    &scratch.ffn_up,
                    &scratch.ffn_act,
                    rows * ffn,
                );
                let proj_fd = prepare_proj(
                    backend,
                    &layer.ffn_down,
                    &scratch.ffn_act,
                    &scratch.branch,
                    rows,
                    "swiglu_ffn",
                )?;
                (proj_fg, Some(proj_fu), Some(swiglu), proj_fd)
            };
            // The layer's close: chained plans fold the
            // post-FFN norm and the NEXT layer's attention
            // norm into one boundary chain; the last layer's
            // close folds the head's final norm in instead
            // (its stream's last row becomes the head input
            // in the same dispatch). Unchained plans close
            // with the plain post-FFN norm.
            let (boundary, close, post_ffw) = if chained && !last {
                let next = &layers[i + 1];
                let chain = backend
                    .k4()
                    .prepare_norm_chain(
                        ctx,
                        &scratch.stream_b,
                        &scratch.branch,
                        &layer.post_ffw_norm,
                        &next.attn_norm,
                        &scratch.stream_a,
                        &scratch.h,
                        rows,
                        hidden,
                        POST_NORM_EPS,
                        eps,
                    )
                    .map_err(|e| we(e, "rms_norm"))?;
                (Some(chain), None, None)
            } else if chained {
                let close = backend
                    .k4()
                    .prepare_norm_close(
                        ctx,
                        &scratch.stream_b,
                        &scratch.branch,
                        &layer.post_ffw_norm,
                        final_norm_w,
                        &scratch.stream_a,
                        &scratch.head_in,
                        rows,
                        hidden,
                        POST_NORM_EPS,
                        eps,
                    )
                    .map_err(|e| we(e, "rms_norm"))?;
                (None, Some(close), None)
            } else {
                let post_ffw = backend.k4().prepare_norm_residual(
                    ctx,
                    &scratch.stream_b,
                    &scratch.branch,
                    &layer.post_ffw_norm,
                    &scratch.stream_a,
                    rows,
                    hidden,
                    POST_NORM_EPS,
                );
                (None, None, Some(post_ffw))
            };
            // Dispatch accounting, from the unfused layer's
            // 17: the QKV batch removes the K and V
            // projections (and the gate projection when the
            // gate rides the batch), the fused gate removes
            // the K7 multiply, the FFN pair removes the up
            // projection and the K7 SwiGLU, and the flash
            // attention removes one of the two K6 dispatches
            // on decode plans.
            let mut dispatch_count: u64 = 17;
            if rows == 1 {
                dispatch_count -= 1;
            }
            if layer.attn_batch.is_some() {
                // A full batch removes the K and V
                // projections; a partial batch removes K and
                // the gate projection instead (V stays).
                dispatch_count -= 2;
                if layer.batch_gate && layer.batch_has_v {
                    dispatch_count -= 1;
                }
            }
            if gate_fused {
                dispatch_count -= 1;
            }
            if layer.ffn_pair.is_some() {
                dispatch_count -= 2;
            }
            if chained {
                // The post-attention chain removes one norm
                // dispatch per layer, and the boundary chain
                // removes the attention norm of every layer
                // past the first (folded into the previous
                // layer's close).
                dispatch_count -= 1;
                if i > 0 {
                    dispatch_count -= 1;
                }
            }
            layer_plans.push(LayerPlan {
                attn_norm,
                proj_q,
                proj_k,
                proj_v,
                prep,
                scores,
                reduce,
                flash,
                proj_gate,
                gate_mul,
                proj_o,
                post_attn,
                post_chain,
                ffn_norm,
                proj_fg,
                proj_fu,
                swiglu,
                proj_fd,
                boundary,
                close,
                post_ffw,
                dispatch_count,
            });
        }

        // The final norm: folded into the last layer's close
        // on chained plans (the close emits the head input);
        // a standalone dispatch over the copied-out last row
        // otherwise.
        let final_norm = if hidden <= CHAIN_MAX_WIDTH {
            None
        } else {
            Some(backend.k4().prepare_rms_norm(
                ctx,
                &scratch.last_row,
                final_norm_w,
                &scratch.head_in,
                1,
                hidden,
                eps,
            ))
        };
        // The head and its finishing: a Q5_K head takes the
        // finishing map fused into its kernel's final store
        // (K3's finish modes, K8's exact expressions applied
        // once per logit at its only store); any other head
        // family keeps the separate K8 finishing dispatch.
        let (head, finish) = if let ProjWeights::Q5K(w) = output {
            let mode = if cfg.logit_softcap > 0.0 {
                crate::kernels::k3_q5k_gemv::FINISH_SOFTCAP
            } else {
                crate::kernels::k3_q5k_gemv::FINISH_SCALE
            };
            let head = backend
                .k3()
                .prepare_with_finish(
                    ctx,
                    w,
                    &scratch.head_in,
                    &scratch.logits,
                    1,
                    mode,
                    cfg.logit_scale,
                    cfg.logit_softcap,
                )
                .map(ProjCall::Q5K)
                .map_err(|e| we(e, "lm_head"))?;
            (head, None)
        } else {
            let head = prepare_proj(
                backend,
                output,
                &scratch.head_in,
                &scratch.logits,
                1,
                "lm_head",
            )?;
            let finish = if cfg.logit_softcap > 0.0 {
                backend
                    .k8()
                    .prepare_finish(
                        ctx,
                        &scratch.logits,
                        cfg.vocab_size as usize,
                        cfg.logit_scale,
                        cfg.logit_softcap,
                    )
                    .map_err(|e| we(e, "lm_head"))?
            } else {
                backend.k8().prepare_scale(
                    ctx,
                    &scratch.logits,
                    cfg.vocab_size as usize,
                    cfg.logit_scale,
                )
            };
            (head, Some(finish))
        };
        let dispatch_count = 1
            + u64::from(embed_norm.is_some())
            + layer_plans
                .iter()
                .map(LayerPlan::dispatch_count)
                .sum::<u64>()
            + 1
            + u64::from(final_norm.is_some())
            + u64::from(finish.is_some());
        Ok(ChunkPlan {
            rows,
            scores_cap,
            kv_caps,
            scratch,
            gather,
            embed_norm,
            layers: layer_plans,
            final_norm,
            head,
            finish,
            dispatch_count,
        })
    }

    /// Whether the plan still binds the session's current KV
    /// buffers (capacities unchanged since build) and its
    /// scores scratch covers `needed_cap` stored positions.
    fn is_current(&self, kv: &DeviceKv, needed_cap: usize) -> bool {
        self.scores_cap >= needed_cap
            && self.kv_caps.len() == kv.layer_count()
            && self
                .kv_caps
                .iter()
                .enumerate()
                .all(|(i, cap)| kv.layer(i, "attention").is_ok_and(|l| l.capacity == *cap))
    }

    /// Refresh every position-dependent uniform for a chunk
    /// starting at absolute position `start`, given each
    /// layer's post-append view (`stored`, `first_pos`).
    fn set_positions(&mut self, ctx: &GpuContext, start: usize, views: &[(usize, usize)]) {
        for (plan, (stored, first_pos)) in self.layers.iter_mut().zip(views.iter()) {
            plan.set_geometry(ctx, *stored, *first_pos, start);
        }
    }

    /// Record the chunk's main chain into the open compute
    /// `pass`: gather, embedding norm, every layer in forward
    /// order (the preparation carries the sliding layers'
    /// RoPE). The whole chain is ONE pass: a pass per dispatch
    /// measured at roughly 26 microseconds of combined
    /// host-record and GPU transition time per dispatch on the
    /// proof hardware, which at 1,071 dispatches per Glimmer
    /// token cost more than the weight streaming itself;
    /// inside one pass wgpu inserts the hazard barriers
    /// between dispatches and nothing else is paid per
    /// dispatch but the launch.
    fn encode_chain(&self, pass: &mut wgpu::ComputePass<'_>) {
        self.gather.encode_into(pass);
        if let Some(n) = &self.embed_norm {
            n.encode_into(pass);
        }
        for plan in &self.layers {
            plan.encode_pre_attention(pass);
            plan.encode_attention(pass);
            plan.encode_post_attention(pass);
        }
    }

    /// Copy the stream's last row into the head's input
    /// buffer (an encoder-level operation, recorded between
    /// the chain pass and the head pass).
    fn copy_last_row(&self, encoder: &mut wgpu::CommandEncoder, cfg: &GlimmerConfig) {
        let hidden_bytes = u64::from(cfg.hidden) * 4;
        encoder.copy_buffer_to_buffer(
            &self.scratch.stream_a,
            (self.rows - 1) as u64 * hidden_bytes,
            &self.scratch.last_row,
            0,
            hidden_bytes,
        );
    }

    /// Record the head segment into the open compute `pass`:
    /// final norm, head projection, finishing, in place over
    /// the logits buffer.
    fn encode_head(&self, pass: &mut wgpu::ComputePass<'_>) {
        if let Some(n) = &self.final_norm {
            n.encode_into(pass);
        }
        self.head.encode_into(pass);
        if let Some(f) = &self.finish {
            f.encode_into(pass);
        }
    }

    /// Copy the finished logits into the readback staging
    /// (an encoder-level operation, recorded last).
    fn copy_logits(&self, encoder: &mut wgpu::CommandEncoder, cfg: &GlimmerConfig) {
        encoder.copy_buffer_to_buffer(
            &self.scratch.logits,
            0,
            &self.scratch.staging,
            0,
            u64::from(cfg.vocab_size) * 4,
        );
    }
}

/// A device-resident Glimmer inference session over one
/// [`WgpuCompute`] backend: weights uploaded once at
/// construction, KV resident across forwards, one submission
/// per forward chunk.
///
/// The session borrows the backend; build the backend unboxed
/// (not through the `Compute` trait object) so the session can
/// reach the kernel pipelines.
#[derive(Debug)]
pub struct ResidentSession<'a> {
    /// The backend whose device and pipelines the session uses.
    backend: &'a WgpuCompute,
    /// The model configuration.
    cfg: GlimmerConfig,
    /// The embedding table, resident in its file dtype.
    embedding: wgpu::Buffer,
    /// The embedding table's dtype (K8 gather family selector).
    embedding_dtype: Dtype,
    /// The final norm vector as an f32 buffer.
    final_norm: wgpu::Buffer,
    /// The head (lm_head) projection.
    output: ProjWeights,
    /// Per-layer resident weights.
    layers: Vec<ResidentLayer>,
    /// Per-layer resident KV storage.
    kv: DeviceKv,
    /// Tokens forwarded so far (the KV's next position).
    position: usize,
    /// Cached chunk plans by row count.
    plans: HashMap<usize, ChunkPlan>,
    /// Accumulated counters.
    stats: SessionStats,
}

impl<'a> ResidentSession<'a> {
    /// Build a session: upload every weight tensor of `weights`
    /// to the device once and lay out the KV storage. No
    /// forward runs here; chunk plans build lazily on the
    /// first forward of each row count.
    ///
    /// # Errors
    ///
    /// The model's step-failure shape, under the step that owns
    /// the tensor: a config/weights layer count mismatch, a
    /// key/value width mismatch (the K6 layout stores one row
    /// width for both), a head dimension past the K6 reduce
    /// bound, or a tensor whose dtype its kernel family does
    /// not cover, each refused by name before any upload of
    /// that tensor.
    pub fn new(backend: &'a WgpuCompute, cfg: &GlimmerConfig, weights: &Weights) -> Result<Self> {
        if weights.layers.len() != cfg.layers.len() {
            return Err(GlimmerError::StepFailed {
                step: "session",
                detail: format!(
                    "weights carry {} layers but the config pattern has {}",
                    weights.layers.len(),
                    cfg.layers.len()
                ),
            });
        }
        if cfg.key_length != cfg.value_length {
            return Err(GlimmerError::StepFailed {
                step: "attention",
                detail: format!(
                    "key length {} differs from value length {}; the gpu kv layout stores \
                     keys and values at one row width",
                    cfg.key_length, cfg.value_length
                ),
            });
        }
        if cfg.key_length > 128 {
            return Err(GlimmerError::StepFailed {
                step: "attention",
                detail: format!(
                    "head dim {} exceeds the k6 reduce thread bound of 128",
                    cfg.key_length
                ),
            });
        }
        let ctx = backend.context();
        ctx.check_tensor_fits(&weights.embedding.name, weights.embedding.data.len() as u64)
            .map_err(|e| we(e, "embed"))?;
        let embedding = ctx.upload_bytes(&weights.embedding.name, &weights.embedding.data);
        let final_norm = upload_norm(backend, &weights.final_norm, "rms_norm")?;
        let output = upload_proj(backend, &weights.output, "lm_head")?;
        if output.out_rows() != cfg.vocab_size as usize {
            return Err(GlimmerError::StepFailed {
                step: "lm_head",
                detail: format!(
                    "head matrix produces {} rows but the vocab is {}",
                    output.out_rows(),
                    cfg.vocab_size
                ),
            });
        }
        let mut layers = Vec::with_capacity(weights.layers.len());
        let mut weight_bytes = weights.embedding.data.len();
        for (w, kind) in weights.layers.iter().zip(cfg.layers.iter()) {
            let layer = ResidentLayer::upload(backend, w, *kind)?;
            weight_bytes += layer.weight_bytes();
            layers.push(layer);
        }
        let pattern: Vec<attention::LayerKind> = cfg
            .layers
            .iter()
            .map(|kind| match kind {
                LayerKind::Global => attention::LayerKind::Global,
                LayerKind::SlidingWindow => attention::LayerKind::Windowed {
                    window: cfg.sliding_window as usize,
                },
            })
            .collect();
        dbg_log!(
            "session built: layers={} resident weight bytes={} vocab={} hidden={}",
            layers.len(),
            weight_bytes,
            cfg.vocab_size,
            cfg.hidden
        );
        Ok(ResidentSession {
            backend,
            cfg: cfg.clone(),
            embedding,
            embedding_dtype: weights.embedding.dtype,
            final_norm,
            output,
            layers,
            kv: DeviceKv::new(&pattern, ring_capacity(cfg)),
            position: 0,
            plans: HashMap::new(),
            stats: SessionStats::default(),
        })
    }

    /// The model configuration the session was built for.
    #[must_use]
    pub fn config(&self) -> &GlimmerConfig {
        &self.cfg
    }

    /// Tokens forwarded so far; the next forward continues at
    /// this position.
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    /// The session's accumulated counters.
    #[must_use]
    pub fn stats(&self) -> SessionStats {
        self.stats
    }

    /// Clear the KV and rewind to position zero. Chunk plans
    /// bind the old KV buffers, so they are dropped with it.
    pub fn reset(&mut self) {
        let pattern: Vec<attention::LayerKind> = self
            .cfg
            .layers
            .iter()
            .map(|kind| match kind {
                LayerKind::Global => attention::LayerKind::Global,
                LayerKind::SlidingWindow => attention::LayerKind::Windowed {
                    window: self.cfg.sliding_window as usize,
                },
            })
            .collect();
        self.kv = DeviceKv::new(&pattern, ring_capacity(&self.cfg));
        self.plans.clear();
        self.position = 0;
        dbg_log!("session reset: kv cleared, position 0");
    }

    /// Run one forward over `tokens` and return the finished
    /// logits of the last token, exactly the values the trait
    /// path's `GlimmerModel::forward` returns. The prompt is
    /// processed in chunks (one submission each); a
    /// single-token decode is one chunk, one submission, one
    /// readback.
    ///
    /// # Errors
    ///
    /// [`GlimmerError::EmptyInput`] on an empty slice and
    /// [`GlimmerError::TokenOutOfRange`] naming the first token
    /// past the vocabulary, matching the model's own
    /// validation; kernel refusals surface as
    /// [`GlimmerError::StepFailed`] under the owning step.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(GlimmerError::EmptyInput);
        }
        for &token in tokens {
            if token >= self.cfg.vocab_size {
                return Err(GlimmerError::TokenOutOfRange {
                    token,
                    vocab_size: self.cfg.vocab_size,
                });
            }
        }
        self.stats.forwards += 1;
        self.stats.tokens += tokens.len() as u64;
        let mut rest = tokens;
        let mut logits = Vec::new();
        while !rest.is_empty() {
            let rows = self.chunk_rows(rest.len());
            logits = self.forward_chunk(&rest[..rows])?;
            rest = &rest[rows..];
        }
        Ok(logits)
    }

    /// Rows for the next chunk of an `n`-token remainder: the
    /// chunk cap, shrunk so the scores scratch
    /// (rows x query heads x stored-after-chunk) stays under
    /// the budget at the deepest position this chunk reaches.
    fn chunk_rows(&self, n: usize) -> usize {
        let stored_max = (self.position + n).max(1);
        let by_budget = (SCORES_BUDGET_ELEMS / (self.cfg.q_heads as usize * stored_max)).max(1);
        n.min(MAX_CHUNK_ROWS).min(by_budget)
    }

    /// Stored entries and first stored position for layer
    /// `kind` after a chunk ending at absolute position `end`.
    /// Windowed layers expose the whole ring (window plus
    /// chunk headroom); the K6 mask applies the true window
    /// per query, so the extra entries are simply never
    /// visible to the queries they are too old for.
    fn view_of(&self, kind: LayerKind, end: usize) -> (usize, usize) {
        match kind {
            LayerKind::Global => (end, 0),
            LayerKind::SlidingWindow => {
                let stored = end.min(ring_capacity(&self.cfg));
                (stored, end - stored)
            }
        }
    }

    /// Run one chunk: ensure KV, refresh the plan's positions,
    /// upload the token ids, encode the whole chain into one
    /// command buffer, submit once, read back only the logits.
    fn forward_chunk(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let ctx = self.backend.context();
        let rows = tokens.len();
        let start = self.position;
        let end = start + rows;

        // Grow the KV first: growth submits its own copy
        // command buffers (counted apart) and replaces layer
        // buffers, which the plan currency check below sees.
        let kv_width = self.cfg.k_width() as usize;
        let caps_before: Vec<usize> = (0..self.layers.len())
            .map(|i| self.kv.layer(i, "attention").map_or(0, |l| l.capacity))
            .collect();
        for i in 0..self.layers.len() {
            self.kv.ensure_layer(ctx, i, kv_width, end, "attention")?;
        }
        let grew = (0..self.layers.len())
            .any(|i| self.kv.layer(i, "attention").map_or(0, |l| l.capacity) != caps_before[i]);
        if grew {
            self.stats.aux_submissions += 1;
        }

        // The scores scratch must cover the deepest layer view
        // after this chunk: the full end position when any
        // layer is global, else the window.
        let any_global = self.cfg.layers.contains(&LayerKind::Global);
        let needed_cap = if any_global {
            end
        } else {
            end.min(ring_capacity(&self.cfg))
        };
        let current = self
            .plans
            .get(&rows)
            .is_some_and(|p| p.is_current(&self.kv, needed_cap));
        if !current {
            let cap = if needed_cap <= 1024 {
                1024
            } else {
                needed_cap
                    .max(
                        self.plans
                            .get(&rows)
                            .map_or(0, |p| p.scores_cap.saturating_mul(2)),
                    )
                    .next_power_of_two()
            }
            .min(self.cfg.context_length as usize)
            .max(needed_cap);
            let plan = ChunkPlan::build(
                self.backend,
                &self.cfg,
                &self.layers,
                &self.final_norm,
                &self.output,
                &self.embedding,
                self.embedding_dtype,
                &self.kv,
                rows,
                cap,
            )?;
            self.plans.insert(rows, plan);
        }

        let views: Vec<(usize, usize)> = self
            .cfg
            .layers
            .iter()
            .map(|kind| self.view_of(*kind, end))
            .collect();
        let plan = self
            .plans
            .get_mut(&rows)
            .ok_or_else(|| GlimmerError::StepFailed {
                step: "session",
                detail: format!("chunk plan for {rows} rows missing after build"),
            })?;
        plan.set_positions(ctx, start, &views);
        let token_bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        ctx.queue
            .write_buffer(&plan.scratch.tokens, 0, &token_bytes);
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resident forward"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("resident forward chain"),
                timestamp_writes: None,
            });
            plan.encode_chain(&mut pass);
        }
        // The last-row copy feeds the standalone final norm;
        // a folded close already emitted the head input.
        if plan.final_norm.is_some() {
            plan.copy_last_row(&mut encoder, &self.cfg);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("resident forward head"),
                timestamp_writes: None,
            });
            plan.encode_head(&mut pass);
        }
        plan.copy_logits(&mut encoder, &self.cfg);
        let dispatches = plan.dispatch_count;
        let staging = plan.scratch.staging.clone();
        ctx.queue.submit([encoder.finish()]);
        self.stats.submissions += 1;
        self.stats.chunks += 1;
        self.stats.dispatches += dispatches;
        let logits = ctx
            .map_staging(&staging, self.cfg.vocab_size as usize)
            .map_err(|e| we(e, "lm_head"))?;
        self.stats.readbacks += 1;
        self.position = end;
        dbg_log!(
            "session forward: rows={rows} start={start} end={end} dispatches={dispatches} \
             submissions={}",
            self.stats.submissions
        );
        Ok(logits)
    }
}
