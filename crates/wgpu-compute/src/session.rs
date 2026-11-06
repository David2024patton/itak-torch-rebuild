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
//! whole weight set once (and flushes the upload queue before
//! construction returns, so the copies are paid at
//! construction, not inside the first forward), owns the
//! activation scratch, the
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
//! (one map). Nothing else crosses the bus. A multi-chunk
//! `forward` maps only its final chunk's logits (see
//! [`ResidentSession::forward`]): the earlier chunks' copies
//! land in their staging unread, because only the last
//! chunk's logits are ever returned, and skipping their maps
//! is what lets a prefill's chunk submissions pipeline
//! instead of draining the device between chunks.
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
//! first, then softcaps. Two readback shapes share the one
//! chain: `forward` returns the finished logits and the
//! caller picks the token exactly as it does from the trait
//! path, while `forward_argmax` appends K8's on-device
//! reduction to the head segment and reads back eight bytes
//! (the winning index and logit, the host sampler's greedy
//! pick) for callers on the greedy path.
//!
//! Two more shapes serve the speculative loop (TODO T-9.6).
//! The block tail runs the same final norm and head over
//! EVERY row of the chunk and reads back `[rows][vocab]`
//! logits: a verify block's per-position distributions, from
//! the one chain, never a second forward. The capturing
//! forwards additionally copy each named layer's input rows
//! (the residual stream A at that layer's entry, exactly what
//! `GlimmerModel::forward_rows` captures) into a readback
//! staging at the layer boundary: the chain's single pass
//! splits into one pass per layer range so the encoder-level
//! copies can sit between them. And [`ResidentSession::
//! rewind_to`] restores an earlier position: KV slots are
//! addressed by absolute position, views derive from the
//! position, and the ring headroom bounds how far back a
//! rewind may reach (see the method's own contract).
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
//! [`SessionStats`]. Per-dispatch GPU timing
//! (`crate::timestamps`) needs no caller cooperation either:
//! in timing mode the session drains the recorder after every
//! chunk into its [`DiagAccumulator`] and finishes the report
//! (the summary table to `WGPU_COMPUTE_DIAG_OUT`, when set)
//! at end of run.

use std::collections::HashMap;

use gguf::ggml::GgmlType;
use muse_glimmer::{
    Dtype, GlimmerConfig, GlimmerError, LayerInputCapture, LayerKind, LayerWeights, Result,
    TensorView, Weights,
};

use crate::compute::{mmv_shape_of, MmvFamily, WgpuCompute};
use crate::device::GpuContext;
use crate::diag::dbg_log;
use crate::error::WgpuError;
use crate::kernels::k10_q4k_gemm::K10Call;
use crate::kernels::k11_q6k_gemm::K11Call;
use crate::kernels::k12_q8_quant::K12QuantCall;
use crate::kernels::k13_q4k_mmq::K13Call;
use crate::kernels::k14_q6k_mmq::K14Call;
use crate::kernels::k15_q4k_mmq_wide::K15Call;
use crate::kernels::k16_q6k_mmq_wide::K16Call;
use crate::kernels::k18_small_batch::K18Call;
use crate::kernels::k19_q4k_mmv::K19Call;
use crate::kernels::k1_q4k_gemv::{K1ActGeom, K1Call, K1Weights};
use crate::kernels::k20_q6k_mmv::K20Call;
use crate::kernels::k21_q5k_mmv::K21Call;
use crate::kernels::k22_q4k_mmq::K22Call;
use crate::kernels::k23_q4k_mmq::K23Call;
use crate::kernels::k2_q6k_gemv::{K2Call, K2Weights};
use crate::kernels::k3_q5k_gemv::{K3Call, K3Weights};
use crate::kernels::k4_rmsnorm::{K4Call, CHAIN_MAX_WIDTH};
use crate::kernels::k6_attention::{AttendGeom, DeviceKv, K6FlashCall};
use crate::kernels::k7_swiglu::K7Call;
use crate::kernels::k8_head::{
    K8ArgmaxCall, K8FinishCall, K8GatherRowsCall, ARGMAX_GROUPS, GATHER_NORM_MAX_WIDTH,
};
use crate::kernels::k9_attn_prep::{K9PrepCall, PrepArgs};
use crate::timestamps::DiagAccumulator;

/// The post-norm epsilon, hardcoded to match the reference
/// converter and `muse_glimmer::model`'s own `POST_NORM_EPS`
/// (not file metadata, and not exported by that crate). The
/// resident-session cross-path test pins this value's effect:
/// if either side's epsilon drifts, the token streams diverge
/// and the gate fails.
const POST_NORM_EPS: f32 = 1e-8;

/// The structural ceiling on rows one forward chunk
/// processes: the ring headroom algebra (window plus cap
/// minus one), the rewind bound, and the scratch design are
/// all built around chunks no larger than this. The
/// operational cap a session actually runs resolves from
/// config at construction ([`CHUNK_ROWS_ENV`]) and never
/// exceeds this ceiling.
const MAX_CHUNK_ROWS: usize = 64;

/// Environment variable naming the chunk row cap override:
/// the most rows one prefill chunk processes. Operational
/// config, not a compiled constant (house law): lower it to
/// shrink per-chunk scratch and the KV ring headroom with
/// it. An unparsable or out-of-range value falls back to
/// [`DEFAULT_CHUNK_ROWS`], with the fallback logged at
/// session construction.
pub const CHUNK_ROWS_ENV: &str = "WGPU_COMPUTE_CHUNK_ROWS";

/// The chunk row cap when the environment does not override
/// it: the structural ceiling, and the value every recorded
/// bench ran at.
pub const DEFAULT_CHUNK_ROWS: usize = 64;

/// Resolve the chunk row cap from the raw environment value:
/// a plain count inside `1..=MAX_CHUNK_ROWS`, or the default
/// when unset, unparsable, or out of range. Pure, so the
/// sizing law is unit-testable without a device.
fn resolve_chunk_rows(raw: Option<&str>) -> usize {
    match raw.and_then(|s| s.trim().parse::<usize>().ok()) {
        Some(n) if (1..=MAX_CHUNK_ROWS).contains(&n) => n,
        _ => DEFAULT_CHUNK_ROWS,
    }
}

/// Rows for a chunk of an `n`-token remainder under cap
/// `cap`: the remainder, capped. Pure sizing math, unit
/// tested; the session's `chunk_rows` is this over its
/// resolved cap. Before the flash switch this also shrank
/// under a scores-buffer budget; flash materializes no
/// scores, so the budget is gone and chunks hold the cap at
/// any context length.
fn chunk_rows_for(cap: usize, n: usize) -> usize {
    n.min(cap)
}

/// Whether chunk `index` of a `count`-chunk forward
/// defers its tail readback (the logits vector, or the
/// argmax tail's eight bytes): every chunk but the
/// last. A forward returns only its final chunk's tail,
/// so an earlier chunk's readback would drain the
/// device (the full poll inside `map_staging`) just to
/// discard its contents, and until it returned the host
/// could not prepare the next chunk: the submissions
/// would serialize against the device. A deferred chunk
/// encodes and submits identically (same chain, same
/// head, same logits copy into staging, same diag
/// stamps); only its map is skipped, so the host runs
/// ahead preparing chunk N+1 while the device still
/// runs chunk N. The final chunk always reads back (it
/// carries the return value), so a single-chunk forward
/// (every decode step) never defers. Pure, so the
/// pipelining contract is unit-testable without a
/// device.
fn defers_readback(index: usize, count: usize) -> bool {
    index + 1 < count
}

/// Chunk row count at which a plan's projections switch from
/// the GEMV kernels to the matrix-matrix kernels: the MMQ
/// pair (K12 quantize plus K13/K14, when the backend built
/// them) first, the tiled GEMM kernels (K10/K11, when the
/// backend built them) behind them as the named fallback.
/// The matrix forms stage each weight tile once per chunk,
/// so past a handful of rows they beat re-streaming the
/// weights per row, and below it the GEMV's decode-tuned
/// shape is the right tool. Decode plans (one row) never
/// reach this threshold. The bench record behind the value
/// is in bench/gpu-resident-2026-10-30.md.
const GEMM_MIN_ROWS: usize = 16;

/// The windowed layers' ring capacity in a session:
/// `sliding_window + chunk_cap - 1`, where `chunk_cap` is the
/// session's resolved chunk row cap. The headroom is a
/// correctness requirement, not tuning. A chunk appends all
/// its rows before its attention runs, so the oldest query in
/// the chunk (at `start`) still needs its whole window
/// `[start - window + 1, start]` resident after the newest
/// row (at `start + rows - 1`) has been appended: the ring
/// must span `window + rows - 1` positions or the appends
/// evict keys the chunk's own early queries attend to (and a
/// fully-masked query row reduces to 0/0). The mask window
/// stays the config's window; only the storage ring grows.
fn ring_capacity(cfg: &GlimmerConfig, chunk_cap: usize) -> usize {
    cfg.sliding_window as usize + chunk_cap - 1
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

/// Fold one chunk's captures into a forward's accumulated
/// captures: each layer's rows concatenate in chunk order
/// (chunks run in position order), and layers stay in
/// ascending order because every chunk reports the same
/// ascending list.
fn merge_captures(acc: &mut Vec<LayerInputCapture>, chunk: Vec<LayerInputCapture>) {
    for capture in chunk {
        match acc.iter_mut().find(|c| c.layer == capture.layer) {
            Some(existing) => existing.rows.extend_from_slice(&capture.rows),
            None => acc.push(capture),
        }
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
    /// Logits readbacks (one per `forward` call: a
    /// multi-chunk forward defers every chunk's readback
    /// but the last chunk's, whose single blocking map
    /// collects the only logits the call returns; the
    /// per-chunk logits copies still land in staging,
    /// unmapped).
    pub readbacks: u64,
    /// Argmax readbacks (one per `forward_argmax` call,
    /// under the same deferral as [`Self::readbacks`]):
    /// eight bytes each (the winning index and logit)
    /// instead of the logits vector's 808 KB at the real
    /// vocabulary.
    pub argmax_readbacks: u64,
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
pub(crate) enum ProjWeights {
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

/// One prepared projection dispatch in a forward plan: a
/// GEMV call on decode-shaped plans, or its matrix-matrix
/// counterpart on chunk plans past [`GEMM_MIN_ROWS`] when
/// the backend built the matrix pipelines: the MMQ pair
/// (a K12 quantize dispatch feeding a K13/K14 product
/// dispatch) when the device built the MMQ set, the GEMM
/// dispatch (K10/K11) behind it as the fallback. The forms
/// bind the same resident weight buffers and the same
/// `[rows][width]` activation layout, so the choice is
/// per-plan and invisible to the chain's step order.
///
/// An MMQ form whose activation is one stage of a layer's
/// `h` may carry NO quantize of its own (`quant: None`):
/// its quantize is the layer's shared one (see [`HShare`]),
/// prepared once over `h` into the dedicated scratch pair
/// and encoded by the layer plan ahead of every member
/// product. The product binds the same buffers either way,
/// so the bytes it reads are the per-call quantize's bytes,
/// bit for bit.
#[derive(Debug)]
enum ProjCall {
    /// K1 dispatch.
    Q4K(K1Call),
    /// K2 dispatch.
    Q6K(K2Call),
    /// K3 dispatch.
    Q5K(K3Call),
    /// K10 (Q4_K GEMM) dispatch.
    GemmQ4K(K10Call),
    /// K11 (Q6_K GEMM) dispatch.
    GemmQ6K(K11Call),
    /// K12 quantize + K13 (Q4_K MMQ) dispatches.
    MmqQ4K {
        /// The activation quantize dispatch, or `None` when
        /// this projection reads its layer's shared `h`
        /// quantize instead (see the enum's docs).
        quant: Option<K12QuantCall>,
        /// The integer product dispatch.
        mmq: K13Call,
    },
    /// K12 quantize + K14 (Q6_K MMQ) dispatches.
    MmqQ6K {
        /// The activation quantize dispatch, or `None` when
        /// this projection reads its layer's shared `h`
        /// quantize instead (see the enum's docs).
        quant: Option<K12QuantCall>,
        /// The integer product dispatch.
        mmq: K14Call,
    },
    /// K12 quantize + K15 (Q4_K wide-tile double-buffered
    /// MMQ) dispatches.
    MmqWideQ4K {
        /// The activation quantize dispatch, or `None` when
        /// this projection reads its layer's shared `h`
        /// quantize instead (see the enum's docs).
        quant: Option<K12QuantCall>,
        /// The integer product dispatch.
        mmq: K15Call,
    },
    /// K12 quantize + K16 (Q6_K wide-tile double-buffered
    /// MMQ) dispatches.
    MmqWideQ6K {
        /// The activation quantize dispatch, or `None` when
        /// this projection reads its layer's shared `h`
        /// quantize instead (see the enum's docs).
        quant: Option<K12QuantCall>,
        /// The integer product dispatch.
        mmq: K16Call,
    },
    /// K12 quantize + K22 (Q4_K MMQ in the
    /// register-organization prototype form) dispatches.
    /// Constructed only when the backend's K22 switch is
    /// on; at shipped defaults this variant never appears
    /// in a plan.
    MmqK22Q4K {
        /// The activation quantize dispatch, or `None` when
        /// this projection reads its layer's shared `h`
        /// quantize instead (see the enum's docs).
        quant: Option<K12QuantCall>,
        /// The integer product dispatch.
        mmq: K22Call,
    },
    /// K12 quantize + K23 (Q4_K MMQ in the
    /// staging-repack prototype form) dispatches.
    /// Constructed only when the backend's K23 switch is
    /// on and its K22 switch is off; at shipped defaults
    /// this variant never appears in a plan.
    MmqK23Q4K {
        /// The activation quantize dispatch, or `None` when
        /// this projection reads its layer's shared `h`
        /// quantize instead (see the enum's docs).
        quant: Option<K12QuantCall>,
        /// The integer product dispatch.
        mmq: K23Call,
    },
    /// K12 quantize + K18 (Q4_K small-batch) dispatches.
    SmallQ4K {
        /// The activation quantize dispatch.
        quant: K12QuantCall,
        /// The integer product dispatch.
        mmq: K18Call,
    },
    /// K12 quantize + K18 (Q6_K small-batch) dispatches.
    SmallQ6K {
        /// The activation quantize dispatch.
        quant: K12QuantCall,
        /// The integer product dispatch.
        mmq: K18Call,
    },
    /// K12 quantize + K18 (Q5_K small-batch) dispatches.
    SmallQ5K {
        /// The activation quantize dispatch.
        quant: K12QuantCall,
        /// The integer product dispatch.
        mmq: K18Call,
    },
    /// K19 (Q4_K register-streaming small-batch) dispatch:
    /// a single dispatch, no quantize in front of it (the
    /// kernel consumes the f32 rows directly, so its
    /// dispatch count is the GEMV's).
    MmvQ4K(K19Call),
    /// K21 (Q5_K register-streaming small-batch) dispatch:
    /// one dispatch, no quantize (the MMV form reads the
    /// f32 input directly, like the GEMV).
    MmvQ5K(K21Call),
    /// K20 (Q6_K register-streaming small-batch) dispatch:
    /// a single dispatch, no quantize in front of it (the
    /// kernel consumes the f32 rows directly, so its
    /// dispatch count is the GEMV's).
    MmvQ6K(K20Call),
    /// K1 dispatches chunked at power-of-two row counts
    /// (the chunked GEMV form): one windowed GEMV call
    /// per chunk, in row order, covering the batch.
    ChunkedQ4K(Vec<K1Call>),
    /// K2 dispatches chunked at power-of-two row counts
    /// (the chunked GEMV form; see [`ProjCall::ChunkedQ4K`]).
    ChunkedQ6K(Vec<K2Call>),
    /// K3 dispatches chunked at power-of-two row counts
    /// (the chunked GEMV form; see [`ProjCall::ChunkedQ4K`]).
    ChunkedQ5K(Vec<K3Call>),
}

impl ProjCall {
    /// Record this dispatch into the open compute `pass`.
    fn encode_into(&self, pass: &mut wgpu::ComputePass<'_>) {
        match self {
            ProjCall::Q4K(c) => c.encode_into(pass),
            ProjCall::Q6K(c) => c.encode_into(pass),
            ProjCall::Q5K(c) => c.encode_into(pass),
            ProjCall::GemmQ4K(c) => c.encode_into(pass),
            ProjCall::GemmQ6K(c) => c.encode_into(pass),
            ProjCall::MmqQ4K { quant, mmq } => {
                if let Some(q) = quant {
                    q.encode_into(pass);
                }
                mmq.encode_into(pass);
            }
            ProjCall::MmqQ6K { quant, mmq } => {
                if let Some(q) = quant {
                    q.encode_into(pass);
                }
                mmq.encode_into(pass);
            }
            ProjCall::MmqWideQ4K { quant, mmq } => {
                if let Some(q) = quant {
                    q.encode_into(pass);
                }
                mmq.encode_into(pass);
            }
            ProjCall::MmqWideQ6K { quant, mmq } => {
                if let Some(q) = quant {
                    q.encode_into(pass);
                }
                mmq.encode_into(pass);
            }
            ProjCall::MmqK22Q4K { quant, mmq } => {
                if let Some(q) = quant {
                    q.encode_into(pass);
                }
                mmq.encode_into(pass);
            }
            ProjCall::MmqK23Q4K { quant, mmq } => {
                if let Some(q) = quant {
                    q.encode_into(pass);
                }
                mmq.encode_into(pass);
            }
            ProjCall::SmallQ4K { quant, mmq } => {
                quant.encode_into(pass);
                mmq.encode_into(pass);
            }
            ProjCall::SmallQ6K { quant, mmq } => {
                quant.encode_into(pass);
                mmq.encode_into(pass);
            }
            ProjCall::SmallQ5K { quant, mmq } => {
                quant.encode_into(pass);
                mmq.encode_into(pass);
            }
            ProjCall::MmvQ4K(c) => c.encode_into(pass),
            ProjCall::MmvQ5K(c) => c.encode_into(pass),
            ProjCall::MmvQ6K(c) => c.encode_into(pass),
            ProjCall::ChunkedQ4K(calls) => {
                for c in calls {
                    c.encode_into(pass);
                }
            }
            ProjCall::ChunkedQ6K(calls) => {
                for c in calls {
                    c.encode_into(pass);
                }
            }
            ProjCall::ChunkedQ5K(calls) => {
                for c in calls {
                    c.encode_into(pass);
                }
            }
        }
    }

    /// Compute dispatches this projection records into the
    /// open pass: one for every form but the quantized
    /// forms, whose quantize is a dispatch of its own. An
    /// MMQ form riding its layer's shared quantize records
    /// one: the shared dispatch is counted once on the
    /// layer plan, not once per member. The K18 small
    /// forms always prepare their own quantize (they
    /// never ride the share), so they record two. A
    /// chunked GEMV form records one per chunk dispatch.
    fn dispatches(&self) -> u64 {
        match self {
            ProjCall::MmqQ4K { quant, .. }
            | ProjCall::MmqQ6K { quant, .. }
            | ProjCall::MmqWideQ4K { quant, .. }
            | ProjCall::MmqWideQ6K { quant, .. }
            | ProjCall::MmqK22Q4K { quant, .. }
            | ProjCall::MmqK23Q4K { quant, .. } => 1 + u64::from(quant.is_some()),
            ProjCall::SmallQ4K { .. } | ProjCall::SmallQ6K { .. } | ProjCall::SmallQ5K { .. } => 2,
            ProjCall::ChunkedQ4K(calls) => calls.len() as u64,
            ProjCall::ChunkedQ6K(calls) => calls.len() as u64,
            ProjCall::ChunkedQ5K(calls) => calls.len() as u64,
            _ => 1,
        }
    }
}

/// Upload one projection tensor for the session, once. The
/// refusal text matches the trait path's projection entry, so
/// a dtype with no GEMV family fails the same way on both
/// paths.
pub(crate) fn upload_proj(
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

/// A projection matrix uploaded for the small-batch
/// sweep harness (`examples/mmv_sweep.rs`): the same
/// resident weight handle the session builds through
/// [`upload_proj`], from raw parts instead of a
/// [`TensorView`], so the harness can sweep synthetic
/// tensors at real model shapes. Prepares go through
/// [`prepare_proj`] itself, the session's own routing
/// ladder, so the variant a sweep measures is the
/// variant the session would dispatch under the same
/// environment: the two can never drift onto different
/// kernels.
pub struct ProjBench {
    weights: ProjWeights,
    in_features: usize,
    out_features: usize,
}

impl ProjBench {
    /// Upload one synthetic projection tensor. The
    /// dtype acceptance and refusal text mirror
    /// [`upload_proj`] exactly: Q4_K, Q6_K, and Q5_K
    /// upload; everything else is refused by name.
    pub fn upload(
        backend: &WgpuCompute,
        name: &str,
        dtype: Dtype,
        raw: &[u8],
        in_features: usize,
        out_features: usize,
    ) -> Result<ProjBench> {
        let step = "proj_bench";
        let ctx = backend.context();
        let weights = match dtype {
            Dtype::Q4K => backend
                .k1()
                .upload(ctx, name, raw, in_features, out_features)
                .map(ProjWeights::Q4K)
                .map_err(|e| we(e, step))?,
            Dtype::Q6K => backend
                .k2()
                .upload(ctx, name, raw, in_features, out_features)
                .map(ProjWeights::Q6K)
                .map_err(|e| we(e, step))?,
            Dtype::Q5K => backend
                .k3()
                .upload(ctx, name, raw, in_features, out_features)
                .map(ProjWeights::Q5K)
                .map_err(|e| we(e, step))?,
            other => {
                return Err(GlimmerError::StepFailed {
                    step,
                    detail: format!(
                        "tensor {name} has dtype {other:?}; the gemv kernels cover Q4_K, Q6_K, and Q5_K"
                    ),
                });
            }
        };
        Ok(ProjBench {
            weights,
            in_features,
            out_features,
        })
    }

    /// Resident weight bytes of the uploaded tensor:
    /// the stream the timed dispatch reads.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.weights.byte_len()
    }

    /// Input width of the projection.
    #[must_use]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Output width of the projection.
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Prepare one projection dispatch at `batch` rows
    /// through the session's routing ladder
    /// ([`prepare_proj`], no shared `h` quantize: the
    /// bench measures one projection at a time, so each
    /// quantized form prepares its own quantize exactly
    /// as it does for a non-`h` projection in a forward).
    pub fn prepare(
        &self,
        backend: &WgpuCompute,
        input: &wgpu::Buffer,
        q8_qs: &wgpu::Buffer,
        q8_ds: &wgpu::Buffer,
        output: &wgpu::Buffer,
        batch: usize,
    ) -> Result<ProjBenchCall> {
        prepare_proj(
            backend,
            &self.weights,
            input,
            q8_qs,
            q8_ds,
            output,
            batch,
            "proj_bench",
            None,
        )
        .map(|call| ProjBenchCall { call })
    }
}

/// One prepared projection dispatch for the sweep
/// harness: the session's own [`ProjCall`], wrapped so
/// the harness can encode it, name the kernel form the
/// routing chose, and classify its numerics (a form
/// that quantizes its activation answers to the wider
/// quantization band, not to reassociation noise).
pub struct ProjBenchCall {
    call: ProjCall,
}

impl ProjBenchCall {
    /// Record this dispatch into the open compute pass.
    pub fn encode_into(&self, pass: &mut wgpu::ComputePass<'_>) {
        self.call.encode_into(pass);
    }

    /// Compute dispatches this projection records: the
    /// session's own accounting (see
    /// [`ProjCall::dispatches`]).
    #[must_use]
    pub fn dispatches(&self) -> u64 {
        self.call.dispatches()
    }

    /// The product kernel the routing chose, under the
    /// diag timestamps system's name for it. A composed
    /// form (a K12 quantize feeding its product) is
    /// named by its product kernel; the quantize is
    /// part of the measured composition.
    #[must_use]
    pub fn kernel_name(&self) -> &'static str {
        match &self.call {
            ProjCall::Q4K(_) => "k1_q4k_gemv",
            ProjCall::Q6K(_) => "k2_q6k_gemv",
            ProjCall::Q5K(_) => "k3_q5k_gemv",
            ProjCall::GemmQ4K(_) => "k10_q4k_gemm",
            ProjCall::GemmQ6K(_) => "k11_q6k_gemm",
            ProjCall::MmqQ4K { .. } => "k13_q4k_mmq",
            ProjCall::MmqQ6K { .. } => "k14_q6k_mmq",
            ProjCall::MmqK22Q4K { .. } => "k22_q4k_mmq",
            ProjCall::MmqK23Q4K { .. } => "k23_q4k_mmq",
            ProjCall::MmqWideQ4K { .. } => "k15_q4k_mmq_wide",
            ProjCall::MmqWideQ6K { .. } => "k16_q6k_mmq_wide",
            ProjCall::SmallQ4K { .. } => "k18_q4k_small",
            ProjCall::SmallQ6K { .. } => "k18_q6k_small",
            ProjCall::SmallQ5K { .. } => "k18_q5k_small",
            ProjCall::MmvQ4K(_) => "k19_q4k_mmv",
            ProjCall::MmvQ6K(_) => "k20_q6k_mmv",
            ProjCall::MmvQ5K(_) => "k21_q5k_mmv",
            ProjCall::ChunkedQ4K(_) => "k1_q4k_gemv",
            ProjCall::ChunkedQ6K(_) => "k2_q6k_gemv",
            ProjCall::ChunkedQ5K(_) => "k3_q5k_gemv",
        }
    }

    /// Whether this form quantizes its activation to
    /// Q8 before the product (the MMQ and small-batch
    /// forms): its output answers to the crate's 5%
    /// quantization band against the f32 GEMV form, not
    /// to the tight reassociation bound.
    #[must_use]
    pub fn quantized(&self) -> bool {
        !matches!(
            self.call,
            ProjCall::Q4K(_)
                | ProjCall::Q6K(_)
                | ProjCall::Q5K(_)
                | ProjCall::GemmQ4K(_)
                | ProjCall::GemmQ6K(_)
                | ProjCall::MmvQ4K(_)
                | ProjCall::MmvQ6K(_)
                | ProjCall::MmvQ5K(_)
                | ProjCall::ChunkedQ4K(_)
                | ProjCall::ChunkedQ6K(_)
                | ProjCall::ChunkedQ5K(_)
        )
    }
}

/// Choose the Q4_K MMQ product for an already-prepared
/// K12 quantize call: the wide product (K15) first when
/// the batch is at or past the backend's wide row
/// threshold, then the K22 register-organization
/// prototype when the backend's K22 switch is on, then
/// the K23 staging-repack prototype when the backend's
/// K23 switch is on, the standard product (K13) behind
/// all three. With both prototype switches off (the
/// shipped default) the ladder is the shipped ladder
/// exactly: the K22 and K23 rungs are skipped without
/// being consulted. `None` when no pipeline
/// exists or every prepare refuses: the caller then
/// takes the GEMM form. Every refusal is logged by name;
/// the choice is made here, at plan-build time, and
/// never substitutes inside a dispatch. All the MMQ
/// projection forms (plain, gated, SwiGLU) share this
/// product choice: the activation transform lives in the
/// quantize call they hand over, upstream of the
/// product, so the ladder is one code path per family.
fn mmq_product_q4k(
    backend: &WgpuCompute,
    weights: &K1Weights,
    q8_qs: &wgpu::Buffer,
    q8_ds: &wgpu::Buffer,
    output: &wgpu::Buffer,
    batch: u32,
    quant: Option<K12QuantCall>,
) -> Option<ProjCall> {
    let ctx = backend.context();
    if batch >= backend.wide_min_rows() {
        if let Some(wide) = backend.k15() {
            match wide.prepare(ctx, weights, q8_qs, q8_ds, output, batch) {
                Ok(mmq) => return Some(ProjCall::MmqWideQ4K { quant, mmq }),
                Err(e) => {
                    dbg_log!(
                        "mmq wide q4k refused at the product prepare: {e}; \
                         the standard mmq form serves it"
                    );
                }
            }
        }
    }
    if backend.k22_mmq() {
        if let Some(k22) = backend.k22() {
            match k22.prepare(ctx, weights, q8_qs, q8_ds, output, batch) {
                Ok(mmq) => return Some(ProjCall::MmqK22Q4K { quant, mmq }),
                Err(e) => {
                    dbg_log!(
                        "mmq k22 q4k refused at the product prepare: {e}; \
                         the standard mmq form serves it"
                    );
                }
            }
        }
    }
    if backend.k23_mmq() {
        if let Some(k23) = backend.k23() {
            match k23.prepare(ctx, weights, q8_qs, q8_ds, output, batch) {
                Ok(mmq) => return Some(ProjCall::MmqK23Q4K { quant, mmq }),
                Err(e) => {
                    dbg_log!(
                        "mmq k23 q4k refused at the product prepare: {e}; \
                         the standard mmq form serves it"
                    );
                }
            }
        }
    }
    let mmq = backend.k13()?;
    match mmq.prepare(ctx, weights, q8_qs, q8_ds, output, batch) {
        Ok(mmq) => Some(ProjCall::MmqQ4K { quant, mmq }),
        Err(e) => {
            dbg_log!("mmq q4k refused at the product prepare: {e}; the gemm form serves it");
            None
        }
    }
}

/// Choose the Q6_K MMQ product for an already-prepared
/// K12 quantize call: the wide product (K16) first past
/// the wide row threshold, the standard product (K14)
/// behind it. See [`mmq_product_q4k`] for the contract.
fn mmq_product_q6k(
    backend: &WgpuCompute,
    weights: &K2Weights,
    q8_qs: &wgpu::Buffer,
    q8_ds: &wgpu::Buffer,
    output: &wgpu::Buffer,
    batch: u32,
    quant: Option<K12QuantCall>,
) -> Option<ProjCall> {
    let ctx = backend.context();
    if batch >= backend.wide_min_rows() {
        if let Some(wide) = backend.k16() {
            match wide.prepare(ctx, weights, q8_qs, q8_ds, output, batch) {
                Ok(mmq) => return Some(ProjCall::MmqWideQ6K { quant, mmq }),
                Err(e) => {
                    dbg_log!(
                        "mmq wide q6k refused at the product prepare: {e}; \
                         the standard mmq form serves it"
                    );
                }
            }
        }
    }
    let mmq = backend.k14()?;
    match mmq.prepare(ctx, weights, q8_qs, q8_ds, output, batch) {
        Ok(mmq) => Some(ProjCall::MmqQ6K { quant, mmq }),
        Err(e) => {
            dbg_log!("mmq q6k refused at the product prepare: {e}; the gemm form serves it");
            None
        }
    }
}

/// One layer stage's shared K12 quantize of the layer's
/// `h` activation, and the scratch pair it lands in.
///
/// A layer reads the same `h` content through several
/// projections in one stage: the attention stage (the QKV
/// batch, or Q, K, V separately, plus the separate V of a
/// partial batch and a separate gate projection) all read
/// the `h` the attention norm produced, and the FFN stage
/// (the gate+up pair, or gate and up separately) reads the
/// `h` the post-attention chain produced. Every one of
/// those projections, in MMQ form, used to prepare its own
/// K12 plain quantize of that identical buffer at the
/// identical batch and width: the same kernel over the
/// same bytes, so the outputs were the same bytes computed
/// again. The share prepares the quantize ONCE per stage,
/// into the plan's dedicated pair (`Scratch::q8h_qs` /
/// `Scratch::q8h_ds`, kept apart from the general pair so
/// no single-consumer quantize can clobber it mid-stage),
/// and each member product binds that pair and carries no
/// quantize of its own: byte-identical by construction,
/// because the member products read exactly the bytes
/// their own quantize dispatches would have written.
///
/// The two stages are two shares over two different `h`
/// contents (the post-attention chain rewrites `h` between
/// them); they reuse the one dedicated pair in turn, which
/// is safe because every attention-stage product is encoded
/// before the FFN stage's quantize (the gate projection,
/// the attention stage's last consumer, opens the
/// post-attention encode, ahead of the chain that produces
/// the FFN stage's `h`).
///
/// A projection joins only when its input width matches
/// the width the share was prepared at; a mismatched width
/// keeps its own quantize over the general pair, exactly
/// as before the share existed. The share's call is
/// prepared lazily by its first member and installed only
/// when that member's product also prepares: a member
/// whose product refuses falls back to the GEMM form with
/// the share untouched.
#[derive(Debug)]
struct HShare<'a> {
    /// The shared quantized activations (the dedicated
    /// pair's packed int8 words).
    qs: &'a wgpu::Buffer,
    /// The shared quantized scales, beside [`HShare::qs`].
    ds: &'a wgpu::Buffer,
    /// The shared quantize call, once the first member has
    /// installed it.
    call: Option<K12QuantCall>,
    /// The input width the share quantizes at, set with
    /// the call.
    in_features: Option<usize>,
    /// Projections riding the share (their products bind
    /// the pair and record no quantize of their own).
    members: usize,
}

impl<'a> HShare<'a> {
    /// A fresh share over the plan's dedicated pair.
    fn new(qs: &'a wgpu::Buffer, ds: &'a wgpu::Buffer) -> Self {
        HShare {
            qs,
            ds,
            call: None,
            in_features: None,
            members: 0,
        }
    }

    /// Whether `in_features` may ride this share: the
    /// width is unset (no member yet) or matches the width
    /// the share quantizes at.
    fn admits(&self, in_features: usize) -> bool {
        self.in_features.is_none() || self.in_features == Some(in_features)
    }

    /// Dispatches the share contributes to the layer: one
    /// when at least one member rides it, none otherwise.
    fn dispatches(&self) -> u64 {
        u64::from(self.call.is_some())
    }
}

/// Try the MMQ form of a plain Q4_K projection: the K12
/// plain quantize of `input` feeding the product
/// [`mmq_product_q4k`] chooses. `None` (with the refusal
/// logged by name) when the device built no MMQ pipeline
/// or a prepare refuses the shape: the caller then takes
/// the GEMM form. This is the fallback ladder's
/// composition step, at plan-build time; it never
/// substitutes inside a dispatch.
#[allow(clippy::too_many_arguments)]
fn try_mmq_q4k(
    backend: &WgpuCompute,
    weights: &K1Weights,
    input: &wgpu::Buffer,
    q8_qs: &wgpu::Buffer,
    q8_ds: &wgpu::Buffer,
    output: &wgpu::Buffer,
    batch: u32,
    share: Option<&mut HShare<'_>>,
) -> Option<ProjCall> {
    if backend.k13().is_none() && backend.k15().is_none() {
        return None;
    }
    let ctx = backend.context();
    if let Some(share) = share {
        if share.admits(weights.in_features) {
            // The shared form: the product binds the
            // share's pair and carries no quantize; the
            // share's quantize is prepared now if this is
            // its first member, but installed only when
            // the product prepares too, so a refusal here
            // leaves the share exactly as it was and the
            // caller takes the GEMM form.
            let mut fresh = None;
            if share.call.is_none() {
                match backend.k12().prepare(
                    ctx,
                    input,
                    share.qs,
                    share.ds,
                    batch,
                    weights.in_features,
                ) {
                    Ok(q) => fresh = Some(q),
                    Err(e) => {
                        dbg_log!(
                            "mmq q4k refused at the shared quantize prepare: {e}; \
                             the gemm form serves it"
                        );
                        return None;
                    }
                }
            }
            let call = mmq_product_q4k(backend, weights, share.qs, share.ds, output, batch, None)?;
            if share.call.is_none() {
                share.call = fresh.take();
                share.in_features = Some(weights.in_features);
            }
            share.members += 1;
            return Some(call);
        }
        // A width the share does not quantize at never
        // joins it: the projection keeps its own quantize
        // over the general pair, below.
    }
    let quant = match backend
        .k12()
        .prepare(ctx, input, q8_qs, q8_ds, batch, weights.in_features)
    {
        Ok(q) => q,
        Err(e) => {
            dbg_log!("mmq q4k refused at the quantize prepare: {e}; the gemm form serves it");
            return None;
        }
    };
    mmq_product_q4k(backend, weights, q8_qs, q8_ds, output, batch, Some(quant))
}

/// Try the MMQ form of a plain Q6_K projection (K12 plain
/// quantize feeding the product [`mmq_product_q6k`]
/// chooses). See [`try_mmq_q4k`] for the fallback
/// contract.
#[allow(clippy::too_many_arguments)]
fn try_mmq_q6k(
    backend: &WgpuCompute,
    weights: &K2Weights,
    input: &wgpu::Buffer,
    q8_qs: &wgpu::Buffer,
    q8_ds: &wgpu::Buffer,
    output: &wgpu::Buffer,
    batch: u32,
    share: Option<&mut HShare<'_>>,
) -> Option<ProjCall> {
    if backend.k14().is_none() && backend.k16().is_none() {
        return None;
    }
    let ctx = backend.context();
    if let Some(share) = share {
        if share.admits(weights.in_features) {
            // The shared form, as in [`try_mmq_q4k`].
            let mut fresh = None;
            if share.call.is_none() {
                match backend.k12().prepare(
                    ctx,
                    input,
                    share.qs,
                    share.ds,
                    batch,
                    weights.in_features,
                ) {
                    Ok(q) => fresh = Some(q),
                    Err(e) => {
                        dbg_log!(
                            "mmq q6k refused at the shared quantize prepare: {e}; \
                             the gemm form serves it"
                        );
                        return None;
                    }
                }
            }
            let call = mmq_product_q6k(backend, weights, share.qs, share.ds, output, batch, None)?;
            if share.call.is_none() {
                share.call = fresh.take();
                share.in_features = Some(weights.in_features);
            }
            share.members += 1;
            return Some(call);
        }
    }
    let quant = match backend
        .k12()
        .prepare(ctx, input, q8_qs, q8_ds, batch, weights.in_features)
    {
        Ok(q) => q,
        Err(e) => {
            dbg_log!("mmq q6k refused at the quantize prepare: {e}; the gemm form serves it");
            return None;
        }
    };
    mmq_product_q6k(backend, weights, q8_qs, q8_ds, output, batch, Some(quant))
}

/// Whether `batch` sits in the small-batch band the
/// backend's config enables: 2 rows up to the resolved
/// `small_batch_max_rows` threshold (default 1, which
/// enables nothing). The kernel's own cap
/// ([`crate::kernels::k18_small_batch::SMALL_MAX_BATCH`])
/// bounds the band from above at prepare.
fn small_batch_applies(backend: &WgpuCompute, batch: u32) -> bool {
    batch >= 2 && batch <= backend.small_batch_max_rows()
}

/// Whether the register-streaming product for
/// `family` serves a projection of shape
/// (`in_features` x `out_rows`) at `batch` rows: the
/// backend's routing table
/// ([`WgpuCompute::mmv_routes`]) admits the batch
/// only under the global ceiling (2 rows up to the
/// resolved `mmv_max_rows`, default 1, which enables
/// nothing) AND inside the (family, shape) cell of
/// measured wins from the 2026-11-06 sweep
/// (bench/mmv-band-routing-2026-11-06.md). A batch
/// the ceiling admits but the cell excludes is the
/// mid-band cliff the table exists to remove; it is
/// logged by name so an A/B can see the fall-throughs
/// it is made of, and the ladder continues exactly
/// as it did before the table existed. The kernels'
/// own band ([`crate::kernels::k19_q4k_mmv::MMV_MIN_BATCH`]
/// ..=[`crate::kernels::k19_q4k_mmv::MMV_MAX_BATCH`],
/// the same constants in every family module) still
/// bounds every cell at prepare.
fn mmv_applies(
    backend: &WgpuCompute,
    family: MmvFamily,
    in_features: usize,
    out_rows: usize,
    batch: u32,
) -> bool {
    let shape = mmv_shape_of(in_features, out_rows);
    if backend.mmv_routes(family, shape, batch) {
        return true;
    }
    if batch >= 2 && batch <= backend.mmv_max_rows() {
        dbg_log!(
            "mmv {} {} excludes batch {batch} (in {in_features}, out {out_rows}); \
             the ladder continues",
            family.name(),
            shape.name()
        );
    }
    false
}

/// Try the K19 register-streaming form of a plain Q4_K
/// projection: one dispatch over the f32 input rows
/// themselves, no quantize in front of it. `None` when
/// the batch is outside the configured band or the
/// prepare refuses the shape (logged by name): the
/// caller then continues down the ladder. This is the
/// fallback ladder's composition step, at plan-build
/// time; it never substitutes inside a dispatch.
fn try_mmv_q4k(
    backend: &WgpuCompute,
    weights: &K1Weights,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    batch: u32,
) -> Option<ProjCall> {
    if !mmv_applies(
        backend,
        MmvFamily::Q4K,
        weights.in_features,
        weights.out_rows,
        batch,
    ) {
        return None;
    }
    match backend
        .k19()
        .prepare(backend.context(), weights, input, output, batch)
    {
        Ok(c) => Some(ProjCall::MmvQ4K(c)),
        Err(e) => {
            dbg_log!("mmv q4k refused at the product prepare: {e}; the ladder continues");
            None
        }
    }
}

/// Try the K20 register-streaming form of a plain Q6_K
/// projection: one dispatch over the f32 input rows
/// themselves, no quantize in front of it. `None` when
/// the batch is outside the configured band or the
/// prepare refuses the shape (logged by name): the
/// caller then continues down the ladder. This is the
/// fallback ladder's composition step, at plan-build
/// time; it never substitutes inside a dispatch.
fn try_mmv_q6k(
    backend: &WgpuCompute,
    weights: &K2Weights,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    batch: u32,
) -> Option<ProjCall> {
    if !mmv_applies(
        backend,
        MmvFamily::Q6K,
        weights.in_features,
        weights.out_rows,
        batch,
    ) {
        return None;
    }
    match backend
        .k20()
        .prepare(backend.context(), weights, input, output, batch)
    {
        Ok(c) => Some(ProjCall::MmvQ6K(c)),
        Err(e) => {
            dbg_log!("mmv q6k refused at the product prepare: {e}; the ladder continues");
            None
        }
    }
}

/// Try the K18 small-batch form of a plain Q4_K
/// projection: the K12 plain quantize of `input`
/// feeding the K18 product. `None` when the batch is
/// outside the configured band, the device built no
/// K18 Q4_K pipeline, or a prepare refuses the shape
/// (logged by name): the caller then takes the GEMV
/// form. This is the fallback ladder's composition
/// step, at plan-build time; it never substitutes
/// inside a dispatch.
fn try_small_q4k(
    backend: &WgpuCompute,
    weights: &K1Weights,
    input: &wgpu::Buffer,
    q8_qs: &wgpu::Buffer,
    q8_ds: &wgpu::Buffer,
    output: &wgpu::Buffer,
    batch: u32,
) -> Option<ProjCall> {
    if !small_batch_applies(backend, batch) {
        return None;
    }
    let small = backend.k18_q4k()?;
    let ctx = backend.context();
    let quant = match backend
        .k12()
        .prepare(ctx, input, q8_qs, q8_ds, batch, weights.in_features)
    {
        Ok(q) => q,
        Err(e) => {
            dbg_log!("small q4k refused at the quantize prepare: {e}; the gemv form serves it");
            return None;
        }
    };
    match small.prepare(ctx, weights, q8_qs, q8_ds, output, batch) {
        Ok(mmq) => Some(ProjCall::SmallQ4K { quant, mmq }),
        Err(e) => {
            dbg_log!("small q4k refused at the product prepare: {e}; the gemv form serves it");
            None
        }
    }
}

/// Try the K18 small-batch form of a plain Q6_K
/// projection. See [`try_small_q4k`] for the contract.
fn try_small_q6k(
    backend: &WgpuCompute,
    weights: &K2Weights,
    input: &wgpu::Buffer,
    q8_qs: &wgpu::Buffer,
    q8_ds: &wgpu::Buffer,
    output: &wgpu::Buffer,
    batch: u32,
) -> Option<ProjCall> {
    if !small_batch_applies(backend, batch) {
        return None;
    }
    let small = backend.k18_q6k()?;
    let ctx = backend.context();
    let quant = match backend
        .k12()
        .prepare(ctx, input, q8_qs, q8_ds, batch, weights.in_features)
    {
        Ok(q) => q,
        Err(e) => {
            dbg_log!("small q6k refused at the quantize prepare: {e}; the gemv form serves it");
            return None;
        }
    };
    match small.prepare(ctx, weights, q8_qs, q8_ds, output, batch) {
        Ok(mmq) => Some(ProjCall::SmallQ6K { quant, mmq }),
        Err(e) => {
            dbg_log!("small q6k refused at the product prepare: {e}; the gemv form serves it");
            None
        }
    }
}

/// Try the K18 small-batch form of a plain Q5_K
/// projection. See [`try_small_q4k`] for the contract.
fn try_small_q5k(
    backend: &WgpuCompute,
    weights: &K3Weights,
    input: &wgpu::Buffer,
    q8_qs: &wgpu::Buffer,
    q8_ds: &wgpu::Buffer,
    output: &wgpu::Buffer,
    batch: u32,
) -> Option<ProjCall> {
    if !small_batch_applies(backend, batch) {
        return None;
    }
    let small = backend.k18_q5k()?;
    let ctx = backend.context();
    let quant = match backend
        .k12()
        .prepare(ctx, input, q8_qs, q8_ds, batch, weights.in_features)
    {
        Ok(q) => q,
        Err(e) => {
            dbg_log!("small q5k refused at the quantize prepare: {e}; the gemv form serves it");
            return None;
        }
    };
    match small.prepare(ctx, weights, q8_qs, q8_ds, output, batch) {
        Ok(mmq) => Some(ProjCall::SmallQ5K { quant, mmq }),
        Err(e) => {
            dbg_log!("small q5k refused at the product prepare: {e}; the gemv form serves it");
            None
        }
    }
}

/// Whether `batch` sits in the chunked-GEMV band the
/// backend's config enables: 2 rows up to the resolved
/// `chunked_gemv_max_rows` threshold (default 1, which
/// enables nothing), and below [`GEMM_MIN_ROWS`], where
/// the matrix forms own the ladder.
fn chunked_gemv_applies(backend: &WgpuCompute, batch: u32) -> bool {
    batch >= 2 && batch <= backend.chunked_gemv_max_rows() && (batch as usize) < GEMM_MIN_ROWS
}

/// Decompose `batch` into power-of-two row chunks,
/// largest first, as `(start, len)` pairs in row order:
/// 7 decomposes to `[(0, 4), (4, 2), (6, 1)]`. Every
/// chunk's row count is a power of two: the counts the
/// published strix-llama.cpp RADV report found scaling
/// through the small-batch path, where arbitrary counts
/// (3, 5, 6) ran several times slower than 1, 2, 4 (see
/// bench/chunked-gemv-2026-11-05.md). A power-of-two
/// batch decomposes to itself, one chunk.
fn pow2_chunks(batch: u32) -> Vec<(u32, u32)> {
    let mut chunks = Vec::new();
    let mut start = 0u32;
    let mut rest = batch;
    while rest > 0 {
        let len = 1u32 << (31 - rest.leading_zeros());
        chunks.push((start, len));
        start += len;
        rest -= len;
    }
    chunks
}

/// Try the chunked GEMV form of a plain Q4_K projection:
/// the batch served as one windowed K1 dispatch per
/// power-of-two chunk (see [`pow2_chunks`]), instead of
/// one dispatch at the batch's own count. `None` when
/// the batch is outside the configured band or a window
/// prepare refuses (logged by name): the caller then
/// takes the whole-batch GEMV form. This is the fallback
/// ladder's composition step, at plan-build time; it
/// never substitutes inside a dispatch. The form
/// prepares no quantize: like every GEMV form it reads
/// the resident activation rows directly, so the
/// quantize-once sharing the quantized forms ride is
/// neither duplicated nor disturbed by chunking.
fn try_chunked_q4k(
    backend: &WgpuCompute,
    weights: &K1Weights,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    batch: u32,
) -> Option<ProjCall> {
    if !chunked_gemv_applies(backend, batch) {
        return None;
    }
    let ctx = backend.context();
    let mut calls = Vec::new();
    for (start, len) in pow2_chunks(batch) {
        match backend
            .k1()
            .prepare_rows(ctx, weights, input, output, start, len)
        {
            Ok(c) => calls.push(c),
            Err(e) => {
                dbg_log!("chunked q4k refused at the window prepare: {e}; the gemv form serves it");
                return None;
            }
        }
    }
    Some(ProjCall::ChunkedQ4K(calls))
}

/// Try the chunked GEMV form of a plain Q6_K
/// projection. See [`try_chunked_q4k`] for the contract.
fn try_chunked_q6k(
    backend: &WgpuCompute,
    weights: &K2Weights,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    batch: u32,
) -> Option<ProjCall> {
    if !chunked_gemv_applies(backend, batch) {
        return None;
    }
    let ctx = backend.context();
    let mut calls = Vec::new();
    for (start, len) in pow2_chunks(batch) {
        match backend
            .k2()
            .prepare_rows(ctx, weights, input, output, start, len)
        {
            Ok(c) => calls.push(c),
            Err(e) => {
                dbg_log!("chunked q6k refused at the window prepare: {e}; the gemv form serves it");
                return None;
            }
        }
    }
    Some(ProjCall::ChunkedQ6K(calls))
}

/// Try the chunked GEMV form of a plain Q5_K
/// projection. See [`try_chunked_q4k`] for the contract.
fn try_chunked_q5k(
    backend: &WgpuCompute,
    weights: &K3Weights,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    batch: u32,
) -> Option<ProjCall> {
    if !chunked_gemv_applies(backend, batch) {
        return None;
    }
    let ctx = backend.context();
    let mut calls = Vec::new();
    for (start, len) in pow2_chunks(batch) {
        match backend
            .k3()
            .prepare_rows(ctx, weights, input, output, start, len)
        {
            Ok(c) => calls.push(c),
            Err(e) => {
                dbg_log!("chunked q5k refused at the window prepare: {e}; the gemv form serves it");
                return None;
            }
        }
    }
    Some(ProjCall::ChunkedQ5K(calls))
}

/// Try the K21 register-streaming form of a plain
/// Q5_K projection: the f32 input feeds the product
/// directly, no quantize dispatch. `None` when the
/// batch is outside the configured band or the
/// prepare refuses the shape (logged by name): the
/// caller then continues down the ladder. This is the
/// fallback ladder's composition step, at plan-build
/// time; it never substitutes inside a dispatch.
fn try_mmv_q5k(
    backend: &WgpuCompute,
    weights: &K3Weights,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    batch: u32,
) -> Option<ProjCall> {
    if !mmv_applies(
        backend,
        MmvFamily::Q5K,
        weights.in_features,
        weights.out_rows,
        batch,
    ) {
        return None;
    }
    let ctx = backend.context();
    match backend.k21().prepare(ctx, weights, input, output, batch) {
        Ok(c) => Some(ProjCall::MmvQ5K(c)),
        Err(e) => {
            dbg_log!("mmv q5k refused at the product prepare: {e}; the ladder continues");
            None
        }
    }
}

/// Prepare one projection dispatch over resident buffers.
/// At chunk sizes past [`GEMM_MIN_ROWS`] the Q4_K and Q6_K
/// families dispatch through the matrix-matrix forms: the
/// MMQ pair when the backend built it, the GEMM kernels
/// (K10/K11) behind them when it built those; below the
/// threshold, and for Q5_K (the one-row head) at any size,
/// the GEMV kernels run. Three rungs sit between those:
/// in the register-streaming band (2 rows up to the
/// backend's configured `mmv_max_rows`, shipped off),
/// the MMV register-streaming products serve their
/// families first in that family's ladder when its band
/// is open and the (family, shape) cell of the
/// measured-win routing table admits the batch (K19 for
/// Q4_K, K20 for Q6_K, K21 for Q5_K;
/// bench/mmv-band-routing-2026-11-06.md);
/// the register-streaming form is the band's structural
/// challenger, and the A/B should not be confounded; see
/// bench/mmv-prototype-2026-11-05.md,
/// bench/mmv-q6k-2026-11-05.md and
/// bench/mmv-q5k-2026-11-05.md); behind them, in the
/// small-batch band (2 rows up to the backend's
/// configured `small_batch_max_rows`, shipped off), the
/// K18 small-batch products serve all three families
/// when the backend built them; and in the chunked band
/// (2 rows up to the backend's configured
/// `chunked_gemv_max_rows`, shipped off), the GEMV
/// kernels themselves serve the batch as several
/// windowed dispatches at power-of-two row counts (the
/// chunked GEMV form), tried after K18 and before the
/// single whole-batch GEMV dispatch. So each family's
/// ladder reads: MMV, then K18, then chunked, then the
/// single whole-batch GEMV dispatch. Every form
/// consumes the same resident buffers, so the choice
/// never changes the chain's data layout. `share` is the layer stage's
/// shared `h` quantize (see [`HShare`]) when this
/// projection reads the layer's `h`, and `None` for every
/// other input: the GEMV and GEMM forms never consult
/// it, since they quantize nothing, and neither do the
/// K18 forms, which prepare their own quantize exactly
/// as the MMQ forms did before the share existed.
#[allow(clippy::too_many_arguments)]
fn prepare_proj(
    backend: &WgpuCompute,
    weights: &ProjWeights,
    input: &wgpu::Buffer,
    q8_qs: &wgpu::Buffer,
    q8_ds: &wgpu::Buffer,
    output: &wgpu::Buffer,
    batch: usize,
    step: &'static str,
    share: Option<&mut HShare<'_>>,
) -> Result<ProjCall> {
    let ctx = backend.context();
    let matrix = batch >= GEMM_MIN_ROWS;
    let batch = batch as u32;
    match weights {
        ProjWeights::Q4K(w) => {
            if matrix {
                if let Some(c) = try_mmq_q4k(backend, w, input, q8_qs, q8_ds, output, batch, share)
                {
                    return Ok(c);
                }
                if let Some(g) = backend.k10() {
                    return g
                        .prepare(ctx, w, input, output, batch)
                        .map(ProjCall::GemmQ4K)
                        .map_err(|e| we(e, step));
                }
            }
            if let Some(c) = try_mmv_q4k(backend, w, input, output, batch) {
                return Ok(c);
            }
            if let Some(c) = try_small_q4k(backend, w, input, q8_qs, q8_ds, output, batch) {
                return Ok(c);
            }
            if let Some(c) = try_chunked_q4k(backend, w, input, output, batch) {
                return Ok(c);
            }
            backend
                .k1()
                .prepare(ctx, w, input, output, batch)
                .map(ProjCall::Q4K)
                .map_err(|e| we(e, step))
        }
        ProjWeights::Q6K(w) => {
            if matrix {
                if let Some(c) = try_mmq_q6k(backend, w, input, q8_qs, q8_ds, output, batch, share)
                {
                    return Ok(c);
                }
                if let Some(g) = backend.k11() {
                    return g
                        .prepare(ctx, w, input, output, batch)
                        .map(ProjCall::GemmQ6K)
                        .map_err(|e| we(e, step));
                }
            }
            if let Some(c) = try_mmv_q6k(backend, w, input, output, batch) {
                return Ok(c);
            }
            if let Some(c) = try_small_q6k(backend, w, input, q8_qs, q8_ds, output, batch) {
                return Ok(c);
            }
            if let Some(c) = try_chunked_q6k(backend, w, input, output, batch) {
                return Ok(c);
            }
            backend
                .k2()
                .prepare(ctx, w, input, output, batch)
                .map(ProjCall::Q6K)
                .map_err(|e| we(e, step))
        }
        ProjWeights::Q5K(w) => {
            if let Some(c) = try_mmv_q5k(backend, w, input, output, batch) {
                return Ok(c);
            }
            if let Some(c) = try_small_q5k(backend, w, input, q8_qs, q8_ds, output, batch) {
                return Ok(c);
            }
            if let Some(c) = try_chunked_q5k(backend, w, input, output, batch) {
                return Ok(c);
            }
            backend
                .k3()
                .prepare(ctx, w, input, output, batch)
                .map(ProjCall::Q5K)
                .map_err(|e| we(e, step))
        }
    }
}

/// Prepare the gated output projection (a Q4_K projection
/// whose activation is the attention rows gated by a second
/// stream): on chunk plans past [`GEMM_MIN_ROWS`], the MMQ
/// form (the gated K12 quantize feeding the product
/// [`mmq_product_q4k`] chooses: the wide K15 form past the
/// wide row threshold, the standard K13 form behind it)
/// when the backend built it, the GEMM's gated form behind
/// it, K1's gated form otherwise. The geometry contract is
/// the same [`K1ActGeom`] in every form.
#[allow(clippy::too_many_arguments)]
fn prepare_gated_proj(
    backend: &WgpuCompute,
    weights: &K1Weights,
    input_a: &wgpu::Buffer,
    input_b: &wgpu::Buffer,
    q8_qs: &wgpu::Buffer,
    q8_ds: &wgpu::Buffer,
    output: &wgpu::Buffer,
    batch: usize,
    geom: &K1ActGeom,
    step: &'static str,
) -> Result<ProjCall> {
    let ctx = backend.context();
    if batch >= GEMM_MIN_ROWS {
        if backend.k13().is_some() || backend.k15().is_some() {
            let quant = backend.k12_gated().prepare_act(
                ctx,
                input_a,
                input_b,
                q8_qs,
                q8_ds,
                batch as u32,
                weights.in_features,
                geom,
            );
            match quant {
                Ok(quant) => {
                    if let Some(c) = mmq_product_q4k(
                        backend,
                        weights,
                        q8_qs,
                        q8_ds,
                        output,
                        batch as u32,
                        Some(quant),
                    ) {
                        return Ok(c);
                    }
                }
                Err(e) => {
                    dbg_log!(
                        "mmq gated q4k refused at the quantize prepare: {e}; \
                         the gemm form serves it"
                    );
                }
            }
        }
        if let Some(g) = backend.k10_gated() {
            return g
                .prepare_act(ctx, weights, input_a, input_b, output, batch as u32, geom)
                .map(ProjCall::GemmQ4K)
                .map_err(|e| we(e, step));
        }
    }
    backend
        .k1_gated()
        .prepare_act(ctx, weights, input_a, input_b, output, batch as u32, geom)
        .map(ProjCall::Q4K)
        .map_err(|e| we(e, step))
}

/// Prepare a SwiGLU-fused down projection over whichever
/// family the down weights belong to: on chunk plans past
/// [`GEMM_MIN_ROWS`], the MMQ form (the SwiGLU K12 quantize
/// feeding the family's product kernel) when the backend
/// built it, the GEMM's SwiGLU form behind it, the family's
/// GEMV SwiGLU form otherwise.
#[allow(clippy::too_many_arguments)]
fn prepare_swiglu_proj(
    backend: &WgpuCompute,
    weights: &ProjWeights,
    input_a: &wgpu::Buffer,
    input_b: &wgpu::Buffer,
    q8_qs: &wgpu::Buffer,
    q8_ds: &wgpu::Buffer,
    output: &wgpu::Buffer,
    batch: usize,
    geom: &K1ActGeom,
    step: &'static str,
) -> Result<ProjCall> {
    let ctx = backend.context();
    let matrix = batch >= GEMM_MIN_ROWS;
    match weights {
        ProjWeights::Q4K(w) => {
            if matrix {
                if backend.k13().is_some() || backend.k15().is_some() {
                    let quant = backend.k12_swiglu().prepare_act(
                        ctx,
                        input_a,
                        input_b,
                        q8_qs,
                        q8_ds,
                        batch as u32,
                        w.in_features,
                        geom,
                    );
                    match quant {
                        Ok(quant) => {
                            if let Some(c) = mmq_product_q4k(
                                backend,
                                w,
                                q8_qs,
                                q8_ds,
                                output,
                                batch as u32,
                                Some(quant),
                            ) {
                                return Ok(c);
                            }
                        }
                        Err(e) => {
                            dbg_log!(
                                "mmq swiglu q4k refused at the quantize prepare: {e}; \
                                 the gemm form serves it"
                            );
                        }
                    }
                }
                if let Some(g) = backend.k10_swiglu() {
                    return g
                        .prepare_act(ctx, w, input_a, input_b, output, batch as u32, geom)
                        .map(ProjCall::GemmQ4K)
                        .map_err(|e| we(e, step));
                }
            }
            backend
                .k1_swiglu()
                .prepare_act(ctx, w, input_a, input_b, output, batch as u32, geom)
                .map(ProjCall::Q4K)
                .map_err(|e| we(e, step))
        }
        ProjWeights::Q6K(w) => {
            if matrix {
                if backend.k14().is_some() || backend.k16().is_some() {
                    let quant = backend.k12_swiglu().prepare_act(
                        ctx,
                        input_a,
                        input_b,
                        q8_qs,
                        q8_ds,
                        batch as u32,
                        w.in_features,
                        geom,
                    );
                    match quant {
                        Ok(quant) => {
                            if let Some(c) = mmq_product_q6k(
                                backend,
                                w,
                                q8_qs,
                                q8_ds,
                                output,
                                batch as u32,
                                Some(quant),
                            ) {
                                return Ok(c);
                            }
                        }
                        Err(e) => {
                            dbg_log!(
                                "mmq swiglu q6k refused at the quantize prepare: {e}; \
                                 the gemm form serves it"
                            );
                        }
                    }
                }
                if let Some(g) = backend.k11_swiglu() {
                    return g
                        .prepare_act(ctx, w, input_a, input_b, output, batch as u32, geom)
                        .map(ProjCall::GemmQ6K)
                        .map_err(|e| we(e, step));
                }
            }
            backend
                .k2_swiglu()
                .prepare_act(ctx, w, input_a, input_b, output, batch as u32, geom)
                .map(ProjCall::Q6K)
                .map_err(|e| we(e, step))
        }
        ProjWeights::Q5K(_) => Err(GlimmerError::StepFailed {
            step,
            detail: "a swiglu-fused down projection has no Q5_K form".to_string(),
        }),
    }
}

/// Dequantize one norm weight to f32 on the host and upload it
/// once. The dtype acceptance mirrors the trait path's
/// `rms_norm` conversion exactly: F32, F16, BF16, Q8_0, Q4_K,
/// Q5_K, Q6_K dequantize; anything else refuses by name.
pub(crate) fn upload_norm(
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
                Some(upload_batch(
                    backend,
                    &[&w.attn_q, &w.attn_k, &w.attn_gate],
                    "proj",
                )?),
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
            Some(upload_batch(
                backend,
                &[&w.ffn_gate, &w.ffn_up],
                "swiglu_ffn",
            )?)
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
    /// and the flash attention reads it. Keys need no such buffer:
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
    /// K12's quantized activations for the chain's
    /// single-consumer quantizes: the packed int8 words,
    /// `[rows][blocks][8]` u32, blocks counted at the widest
    /// projection input a plan quantizes (hidden, attention,
    /// or FFN width). One buffer serves them all: dispatches
    /// inside a pass are totally ordered, each quantize
    /// immediately precedes its product, and no two of these
    /// quantized forms are ever live at once. The layers'
    /// shared `h` quantizes do NOT land here; they have the
    /// dedicated pair below, because their bytes must stay
    /// live across a whole stage while this pair's forms
    /// come and go.
    q8_qs: wgpu::Buffer,
    /// K12's quantized activation scales: one `vec4<f32>`
    /// per block, beside [`Scratch::q8_qs`].
    q8_ds: wgpu::Buffer,
    /// The layers' shared `h` quantizes land here: the
    /// packed int8 words of one stage's quantize of `h`,
    /// bound by every member product of that stage (see
    /// [`HShare`]). Sized exactly like the general pair: a
    /// stage quantizes `h` at the hidden width, which the
    /// general pair's block count already covers. The two
    /// stages of a layer reuse this pair in turn; see
    /// [`HShare`] for why that is safe.
    q8h_qs: wgpu::Buffer,
    /// The shared quantize's scales, beside
    /// [`Scratch::q8h_qs`].
    q8h_ds: wgpu::Buffer,
    /// The last row of stream A, copied out for the head,
    /// `[hidden]`.
    last_row: wgpu::Buffer,
    /// The normed last row, `[hidden]`.
    head_in: wgpu::Buffer,
    /// Finished logits, `[vocab]`.
    logits: wgpu::Buffer,
    /// Logits readback staging, `[vocab]`, `COPY_DST | MAP_READ`.
    staging: wgpu::Buffer,
    /// Argmax reduction partials, `[ARGMAX_GROUPS][2]` u32
    /// (value bits, index) pairs.
    argmax_partials: wgpu::Buffer,
    /// Argmax reduction result, `[2]` u32 (winning index,
    /// winning logit's bits).
    argmax_result: wgpu::Buffer,
    /// Argmax readback staging, `[2]` u32, `COPY_DST |
    /// MAP_READ`: the eight bytes an argmax-tail chunk returns
    /// instead of the logits vector.
    argmax_staging: wgpu::Buffer,
    /// The final-normed hidden rows for the block tail,
    /// `[rows][hidden]`: the standalone final norm over every
    /// row of stream A, where the standard head consumes only
    /// the last row (through `head_in`).
    block_head_in: wgpu::Buffer,
    /// Finished logits for every row, `[rows][vocab]`: the
    /// block tail's output, read back by verify rounds.
    block_logits: wgpu::Buffer,
    /// Block logits readback staging, `[rows][vocab]`,
    /// `COPY_DST | MAP_READ`.
    block_staging: wgpu::Buffer,
}

impl Scratch {
    /// Allocate the scratch for `rows` chunk rows. There is
    /// deliberately no attention scores buffer: flash
    /// attention never materializes scores, so no scratch
    /// here grows with the stored position count.
    /// `qkv_width` is the widest batched-QKV row among the
    /// layers (0 when no layer batches; the buffer then takes
    /// a token element and is never bound).
    fn new(ctx: &GpuContext, cfg: &GlimmerConfig, rows: usize, qkv_width: usize) -> Self {
        let act = |label: &str, elems: usize| ctx.scratch_buffer(label, (elems * 4) as u64);
        let hidden = cfg.hidden as usize;
        let q_width = cfg.q_width() as usize;
        let kv_width = cfg.k_width() as usize;
        let ffn = cfg.ffn_hidden as usize;
        let vocab = cfg.vocab_size as usize;
        // Q8_1 blocks at the widest width any projection in
        // the chain quantizes (see the field docs).
        let q8_blocks = hidden.max(q_width).max(ffn).div_ceil(32);
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
            q8_qs: act("session q8 qs", rows * q8_blocks * 8),
            q8_ds: act("session q8 ds", rows * q8_blocks * 4),
            q8h_qs: act("session q8h qs", rows * q8_blocks * 8),
            q8h_ds: act("session q8h ds", rows * q8_blocks * 4),
            last_row: act("session last row", hidden),
            head_in: act("session head in", hidden),
            logits: act("session logits", vocab),
            staging: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("session logits staging"),
                size: (vocab * 4) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            argmax_partials: act("session argmax partials", ARGMAX_GROUPS * 2),
            argmax_result: act("session argmax result", 2),
            argmax_staging: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("session argmax staging"),
                size: 8,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            block_head_in: act("session block head in", rows * hidden),
            block_logits: act("session block logits", rows * vocab),
            block_staging: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("session block logits staging"),
                size: (rows * vocab * 4) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
        }
    }
}

/// One layer's prepared dispatches within a chunk plan, in
/// forward order. The position-dependent calls (prep, flash)
/// are refreshed per chunk through their setters
/// before encoding; everything else is fixed at plan build.
#[derive(Debug)]
struct LayerPlan {
    /// `rms_norm(h = norm(stream_a, attn_norm))` (absent for
    /// layers past the first on chained plans: the previous
    /// layer's boundary chain already produced this layer's
    /// `h`).
    attn_norm: Option<K4Call>,
    /// The attention stage's shared K12 quantize of `h`
    /// (absent when no attention-stage projection took an
    /// MMQ form: decode plans and GEMM-served plans
    /// quantize nothing). Encoded right after the attention
    /// norm, ahead of every member product; the gate
    /// projection, encoded in the post-attention half, is
    /// its last reader.
    quant_attn: Option<K12QuantCall>,
    /// The FFN stage's shared K12 quantize of `h` (absent
    /// on the same terms as [`LayerPlan::quant_attn`]).
    /// Encoded after the post-attention chain produces the
    /// FFN stage's `h`, ahead of the gate and up products.
    quant_ffn: Option<K12QuantCall>,
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
    /// The fused flash attention over the resident KV: the
    /// only attention form a session plan runs, at every row
    /// count (decode rows and prefill chunks alike). The
    /// two-pass scores + reduce form stays with the trait
    /// path, which steps one row at a time through the K6
    /// one-shot entry points.
    flash: K6FlashCall,
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
        self.flash.set_geometry(ctx, stored, first_pos, start);
    }

    /// Record the pre-attention dispatches: the attention
    /// norm and the Q/K/V projections (one batched dispatch
    /// when the layer batches). The fused preparation opens
    /// [`LayerPlan::encode_attention`].
    fn encode_pre_attention(&self, pass: &mut wgpu::ComputePass<'_>) {
        if let Some(n) = &self.attn_norm {
            n.encode_into(pass);
        }
        if let Some(q) = &self.quant_attn {
            q.encode_into(pass);
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
    /// flash attention, the plan's one attention dispatch at
    /// every row count.
    fn encode_attention(&self, pass: &mut wgpu::ComputePass<'_>) {
        self.prep.encode_into(pass);
        self.flash.encode_into(pass);
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
        if let Some(q) = &self.quant_ffn {
            q.encode_into(pass);
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
    /// The on-device argmax reduction over the finished
    /// logits (an argmax-tail chunk appends its two
    /// dispatches to the head segment and reads back eight
    /// bytes instead of the logits vector).
    argmax: K8ArgmaxCall,
    /// Block tail, first dispatch: the final norm over every
    /// row of stream A into the block head input. The chain
    /// leaves every row's final residual in stream A (the last
    /// layer's close writes it for all rows even though it
    /// emits the standard head input for the last row only),
    /// so this reproduces the head input per row.
    block_final_norm: K4Call,
    /// Block tail head projection, block head input to block
    /// logits, at the chunk's full batch.
    block_head: ProjCall,
    /// Block tail finishing over the block logits, in place
    /// (absent when the head is Q5_K, exactly as the standard
    /// head's `finish`).
    block_finish: Option<K8FinishCall>,
    /// Compute dispatches one block-tail chunk records in its
    /// head segment (the block final norm, the head, and the
    /// finishing when separate).
    block_dispatch_count: u64,
    /// Compute dispatches one chunk of this plan records.
    dispatch_count: u64,
}

/// Which fusions a layer's weights qualified for: the
/// shape half of the layer's dispatch accounting, kept
/// apart from the prepared calls so the accounting is a
/// pure function the unit tests can pin without a device.
#[derive(Clone, Copy, Debug)]
struct LayerFusionShape {
    /// The QKV batch formed (the full Q, K, V batch or
    /// the partial Q, K, gate batch).
    batched: bool,
    /// The batch carries both V and the gate (a full
    /// batch with the gate riding): the separate gate
    /// projection disappears as well.
    full_batch_gate: bool,
    /// The output projection consumes the gate fused, so
    /// the K7 gate multiply disappears.
    gate_fused: bool,
    /// The FFN gate+up pair formed: the up projection and
    /// the K7 SwiGLU disappear.
    ffn_paired: bool,
    /// The norm chains run (the hidden width fits their
    /// register residency).
    chained: bool,
    /// The first layer: on chained plans its attention
    /// norm is its own dispatch, where every later
    /// layer's rides the previous layer's boundary chain.
    first: bool,
}

/// Count one layer's dispatches in a chunk plan.
///
/// From the unfused layer's 17, less one because the
/// flash attention is a single K6 dispatch where the
/// two-pass form was two. The QKV batch removes two
/// projections (K and V on a full batch; K and the gate
/// projection on a partial batch, where V stays), and one
/// more when a full batch also carries the gate. The
/// fused gate removes the K7 multiply, the FFN pair
/// removes the up projection and the K7 SwiGLU, and the
/// chains remove one norm dispatch per layer plus the
/// attention norm of every layer past the first (folded
/// into the previous layer's boundary chain).
///
/// The base counts every projection as one dispatch.
/// `proj_dispatches` carries each projection's real count
/// in chain order (Q, K, V, gate, output, FFN gate, FFN
/// up, FFN down), ONE for an absent projection, and the
/// extras over one are added back: an MMQ projection with
/// its own K12 quantize records two. A projection riding
/// its stage's shared quantize records one (the product
/// alone), and `shared_quantizes` adds the shared
/// dispatches back, once per stage that gained a member:
/// that is the whole arithmetic of quantize-once, one
/// quantize per stage in place of one per member.
fn layer_dispatch_count(
    shape: LayerFusionShape,
    proj_dispatches: [u64; 8],
    shared_quantizes: u64,
) -> u64 {
    let mut count: u64 = 16;
    if shape.batched {
        count -= 2;
        if shape.full_batch_gate {
            count -= 1;
        }
    }
    if shape.gate_fused {
        count -= 1;
    }
    if shape.ffn_paired {
        count -= 2;
    }
    if shape.chained {
        count -= 1;
        if !shape.first {
            count -= 1;
        }
    }
    for d in proj_dispatches {
        count += d - 1;
    }
    count + shared_quantizes
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
        ring_cap: usize,
    ) -> Result<Self> {
        let ctx = backend.context();
        let qkv_width = layers
            .iter()
            .filter_map(|l| l.attn_batch.as_ref().map(ProjWeights::out_rows))
            .max()
            .unwrap_or(0);
        let scratch = Scratch::new(ctx, cfg, rows, qkv_width);
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
            let ring = window.map(|_| ring_cap);
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
            // The layer's two shared `h` quantizes (see
            // HShare): one for the attention stage's
            // projections, one for the FFN stage's, both
            // landing in the dedicated scratch pair.
            // Members join as their projections are
            // prepared below; the calls move onto the
            // layer plan at its construction.
            let mut share_attn = HShare::new(&scratch.q8h_qs, &scratch.q8h_ds);
            let mut share_ffn = HShare::new(&scratch.q8h_qs, &scratch.q8h_ds);
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
                let call = prepare_proj(
                    backend,
                    batch,
                    &scratch.h,
                    &scratch.q8_qs,
                    &scratch.q8_ds,
                    &scratch.qkv,
                    rows,
                    "proj",
                    Some(&mut share_attn),
                )?;
                let stride = batch.out_rows();
                let (proj_v, v_in, v_stride, v_off) = if layer.batch_has_v {
                    (None, &scratch.qkv, stride, q_width + kv_width)
                } else {
                    let proj_v = prepare_proj(
                        backend,
                        need(&layer.attn_v)?,
                        &scratch.h,
                        &scratch.q8_qs,
                        &scratch.q8_ds,
                        &scratch.v,
                        rows,
                        "proj",
                        Some(&mut share_attn),
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
                    &scratch.q8_qs,
                    &scratch.q8_ds,
                    &scratch.q,
                    rows,
                    "proj",
                    Some(&mut share_attn),
                )?;
                let proj_k = prepare_proj(
                    backend,
                    need(&layer.attn_k)?,
                    &scratch.h,
                    &scratch.q8_qs,
                    &scratch.q8_ds,
                    &scratch.k,
                    rows,
                    "proj",
                    Some(&mut share_attn),
                )?;
                let proj_v = prepare_proj(
                    backend,
                    need(&layer.attn_v)?,
                    &scratch.h,
                    &scratch.q8_qs,
                    &scratch.q8_ds,
                    &scratch.v,
                    rows,
                    "proj",
                    Some(&mut share_attn),
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
            // Every plan runs the fused flash attention:
            // decode rows and prefill chunks alike. The
            // kernel is shape-generic (one workgroup per
            // query row and head, the causal position taken
            // per row, keys slotted modulo the ring), so the
            // sliding mask and the chunked ring carry over
            // unchanged, and no scores buffer is materialized
            // at any context length.
            //
            // With the backend's staged-flash switch on
            // (WGPU_COMPUTE_FLASH_STAGED), multi-row plans
            // take the staged form instead: same bindings,
            // params, and output, with K/V walked in blocks
            // of 32 keys staged in workgroup memory (see
            // FLASH_STAGED_SHADER). Single-row plans keep
            // the shipped form even with the switch on: a
            // one-row workgroup would stage a block for one
            // active row of eight, the same reasoning that
            // routes llama's N == 1 calls to its scalar path
            // (ggml-vulkan.cpp, get_fa_tuning_params). With
            // the switch off, or on a device whose staged
            // pipeline was never built, this is exactly the
            // shipped call.
            let flash = if backend.flash_staged()
                && rows > 1
                && backend.k6().flash_staged_available()
            {
                backend.k6().prepare_flash_staged(
                    ctx,
                    &scratch.q_normed,
                    kv_layer,
                    &scratch.attn,
                    &geom,
                )
            } else {
                backend
                    .k6()
                    .prepare_flash(ctx, &scratch.q_normed, kv_layer, &scratch.attn, &geom)
            }
            .map_err(|e| we(e, "attention"))?;
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
                        let call = prepare_gated_proj(
                            backend,
                            ow,
                            &scratch.attn,
                            &scratch.qkv,
                            &scratch.q8_qs,
                            &scratch.q8_ds,
                            &scratch.branch,
                            rows,
                            &K1ActGeom {
                                a_stride_elems: q_width,
                                b_stride_elems: batch.out_rows(),
                                b_off_elems: gate_off,
                            },
                            "proj",
                        )?;
                        (None, None, call, true)
                    } else if matches!(layer.attn_gate, Some(ProjWeights::Q4K(_))) {
                        let proj_gate = prepare_proj(
                            backend,
                            need(&layer.attn_gate)?,
                            &scratch.h,
                            &scratch.q8_qs,
                            &scratch.q8_ds,
                            &scratch.gate,
                            rows,
                            "proj",
                            Some(&mut share_attn),
                        )?;
                        let call = prepare_gated_proj(
                            backend,
                            ow,
                            &scratch.attn,
                            &scratch.gate,
                            &scratch.q8_qs,
                            &scratch.q8_ds,
                            &scratch.branch,
                            rows,
                            &K1ActGeom {
                                a_stride_elems: q_width,
                                b_stride_elems: q_width,
                                b_off_elems: 0,
                            },
                            "proj",
                        )?;
                        (Some(proj_gate), None, call, true)
                    } else {
                        let proj_gate = prepare_proj(
                            backend,
                            need(&layer.attn_gate)?,
                            &scratch.h,
                            &scratch.q8_qs,
                            &scratch.q8_ds,
                            &scratch.gate,
                            rows,
                            "proj",
                            Some(&mut share_attn),
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
                            &scratch.q8_qs,
                            &scratch.q8_ds,
                            &scratch.branch,
                            rows,
                            "proj",
                            None,
                        )?;
                        (Some(proj_gate), Some(gate_mul), proj_o, false)
                    }
                } else {
                    let proj_gate = prepare_proj(
                        backend,
                        need(&layer.attn_gate)?,
                        &scratch.h,
                        &scratch.q8_qs,
                        &scratch.q8_ds,
                        &scratch.gate,
                        rows,
                        "proj",
                        Some(&mut share_attn),
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
                        &scratch.q8_qs,
                        &scratch.q8_ds,
                        &scratch.branch,
                        rows,
                        "proj",
                        None,
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
                    &scratch.q8_qs,
                    &scratch.q8_ds,
                    &scratch.ffn_pair,
                    rows,
                    "swiglu_ffn",
                    Some(&mut share_ffn),
                )?;
                let geom = K1ActGeom {
                    a_stride_elems: 2 * ffn,
                    b_stride_elems: 2 * ffn,
                    b_off_elems: ffn,
                };
                let proj_fd = prepare_swiglu_proj(
                    backend,
                    &layer.ffn_down,
                    &scratch.ffn_pair,
                    &scratch.ffn_pair,
                    &scratch.q8_qs,
                    &scratch.q8_ds,
                    &scratch.branch,
                    rows,
                    &geom,
                    "swiglu_ffn",
                )?;
                (proj_fg, None, None, proj_fd)
            } else {
                let proj_fg = prepare_proj(
                    backend,
                    need(&layer.ffn_gate)?,
                    &scratch.h,
                    &scratch.q8_qs,
                    &scratch.q8_ds,
                    &scratch.ffn_gate,
                    rows,
                    "swiglu_ffn",
                    Some(&mut share_ffn),
                )?;
                let proj_fu = prepare_proj(
                    backend,
                    need(&layer.ffn_up)?,
                    &scratch.h,
                    &scratch.q8_qs,
                    &scratch.q8_ds,
                    &scratch.ffn_up,
                    rows,
                    "swiglu_ffn",
                    Some(&mut share_ffn),
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
                    &scratch.q8_qs,
                    &scratch.q8_ds,
                    &scratch.branch,
                    rows,
                    "swiglu_ffn",
                    None,
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
            // Dispatch accounting (layer_dispatch_count
            // carries the arithmetic): the fusion shape
            // from the layer's weights, the prepared
            // projections' own counts (an absent projection
            // counts one, contributing no extra), and one
            // dispatch per shared quantize that gained a
            // member.
            let dispatch_count = layer_dispatch_count(
                LayerFusionShape {
                    batched: layer.attn_batch.is_some(),
                    full_batch_gate: layer.batch_gate && layer.batch_has_v,
                    gate_fused,
                    ffn_paired: layer.ffn_pair.is_some(),
                    chained,
                    first: i == 0,
                },
                [
                    proj_q.dispatches(),
                    proj_k.as_ref().map_or(1, ProjCall::dispatches),
                    proj_v.as_ref().map_or(1, ProjCall::dispatches),
                    proj_gate.as_ref().map_or(1, ProjCall::dispatches),
                    proj_o.dispatches(),
                    proj_fg.dispatches(),
                    proj_fu.as_ref().map_or(1, ProjCall::dispatches),
                    proj_fd.dispatches(),
                ],
                share_attn.dispatches() + share_ffn.dispatches(),
            );
            layer_plans.push(LayerPlan {
                attn_norm,
                quant_attn: share_attn.call,
                quant_ffn: share_ffn.call,
                proj_q,
                proj_k,
                proj_v,
                prep,
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
                &scratch.q8_qs,
                &scratch.q8_ds,
                &scratch.logits,
                1,
                "lm_head",
                None,
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
        let argmax = backend
            .k8()
            .prepare_argmax(
                ctx,
                &scratch.logits,
                cfg.vocab_size as usize,
                &scratch.argmax_partials,
                &scratch.argmax_result,
            )
            .map_err(|e| we(e, "lm_head"))?;
        // The block head segment: the standard head's family
        // split at the chunk's full batch. A Q5_K head takes
        // its fused finishing at any batch (K3's contract is
        // [batch, out]); the other families take their GEMV
        // form directly rather than prepare_proj's matrix
        // forms, because the head is a GEMV-shaped projection
        // at every size the standard head runs (one row), and
        // the block tail must not change the head's arithmetic
        // family as the batch grows.
        let block_final_norm = backend.k4().prepare_rms_norm(
            ctx,
            &scratch.stream_a,
            final_norm_w,
            &scratch.block_head_in,
            rows,
            hidden,
            eps,
        );
        let (block_head, block_finish) = if let ProjWeights::Q5K(w) = output {
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
                    &scratch.block_head_in,
                    &scratch.block_logits,
                    rows as u32,
                    mode,
                    cfg.logit_scale,
                    cfg.logit_softcap,
                )
                .map(ProjCall::Q5K)
                .map_err(|e| we(e, "lm_head"))?;
            (head, None)
        } else {
            let head = match output {
                ProjWeights::Q4K(w) => backend
                    .k1()
                    .prepare(
                        ctx,
                        w,
                        &scratch.block_head_in,
                        &scratch.block_logits,
                        rows as u32,
                    )
                    .map(ProjCall::Q4K)
                    .map_err(|e| we(e, "lm_head"))?,
                ProjWeights::Q6K(w) => backend
                    .k2()
                    .prepare(
                        ctx,
                        w,
                        &scratch.block_head_in,
                        &scratch.block_logits,
                        rows as u32,
                    )
                    .map(ProjCall::Q6K)
                    .map_err(|e| we(e, "lm_head"))?,
                ProjWeights::Q5K(_) => unreachable!("the Q5_K head is handled above"),
            };
            let finish = if cfg.logit_softcap > 0.0 {
                backend
                    .k8()
                    .prepare_finish(
                        ctx,
                        &scratch.block_logits,
                        rows * cfg.vocab_size as usize,
                        cfg.logit_scale,
                        cfg.logit_softcap,
                    )
                    .map_err(|e| we(e, "lm_head"))?
            } else {
                backend.k8().prepare_scale(
                    ctx,
                    &scratch.block_logits,
                    rows * cfg.vocab_size as usize,
                    cfg.logit_scale,
                )
            };
            (head, Some(finish))
        };
        let block_dispatch_count = 1 + block_head.dispatches() + u64::from(block_finish.is_some());
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
            kv_caps,
            scratch,
            gather,
            embed_norm,
            layers: layer_plans,
            final_norm,
            head,
            finish,
            argmax,
            block_final_norm,
            block_head,
            block_finish,
            block_dispatch_count,
            dispatch_count,
        })
    }

    /// Whether the plan still binds the session's current KV
    /// buffers (capacities unchanged since build). Flash
    /// attention keeps no position-sized scratch, so KV
    /// currency is the only condition.
    fn is_current(&self, kv: &DeviceKv) -> bool {
        self.kv_caps.len() == kv.layer_count()
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
        self.encode_open(pass);
        self.encode_layers(pass, 0..self.layers.len());
    }

    /// Record the chain's opening into the open compute
    /// `pass`: the embedding gather and the weightless
    /// embedding norm. Split from [`Self::encode_layers`] so
    /// a capturing chunk can stop the chain at a layer
    /// boundary (an encoder copy of stream A) and resume it
    /// in a fresh pass.
    fn encode_open(&self, pass: &mut wgpu::ComputePass<'_>) {
        self.gather.encode_into(pass);
        if let Some(n) = &self.embed_norm {
            n.encode_into(pass);
        }
    }

    /// Record one contiguous range of layers into the open
    /// compute `pass`, in forward order.
    fn encode_layers(&self, pass: &mut wgpu::ComputePass<'_>, range: std::ops::Range<usize>) {
        for plan in &self.layers[range] {
            plan.encode_pre_attention(pass);
            plan.encode_attention(pass);
            plan.encode_post_attention(pass);
        }
    }

    /// Record the block head segment into the open compute
    /// `pass`: the block final norm, the block head
    /// projection, and its finishing, in place over the
    /// block logits buffer.
    fn encode_block_head(&self, pass: &mut wgpu::ComputePass<'_>) {
        self.block_final_norm.encode_into(pass);
        self.block_head.encode_into(pass);
        if let Some(f) = &self.block_finish {
            f.encode_into(pass);
        }
    }

    /// Copy the finished block logits into their readback
    /// staging (an encoder-level operation, recorded last on
    /// a block-tail chunk).
    fn copy_block_logits(&self, encoder: &mut wgpu::CommandEncoder, cfg: &GlimmerConfig) {
        encoder.copy_buffer_to_buffer(
            &self.scratch.block_logits,
            0,
            &self.scratch.block_staging,
            0,
            (self.rows * cfg.vocab_size as usize * 4) as u64,
        );
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

    /// Record the argmax reduction into the open head `pass`,
    /// after the head segment: the two reduction dispatches
    /// over the finished logits.
    fn encode_argmax(&self, pass: &mut wgpu::ComputePass<'_>) {
        self.argmax.encode_into(pass);
    }

    /// Copy the argmax result (winning index and logit, two
    /// words) into its readback staging (an encoder-level
    /// operation, recorded last on an argmax-tail chunk).
    fn copy_argmax(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(
            &self.scratch.argmax_result,
            0,
            &self.scratch.argmax_staging,
            0,
            8,
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
    /// The chunk row cap resolved from config at construction
    /// ([`CHUNK_ROWS_ENV`]): prefill chunks never exceed it,
    /// and the windowed layers' ring headroom is sized from
    /// it (window plus cap minus one).
    chunk_cap: usize,
    /// Tokens forwarded so far (the KV's next position).
    position: usize,
    /// Cached chunk plans by row count.
    plans: HashMap<usize, ChunkPlan>,
    /// Accumulated counters.
    stats: SessionStats,
    /// The diag timing accumulator, present only when the
    /// backend's context carries a timestamp recorder
    /// (timing mode on a capable device). `None` is the off
    /// path: no drains, no report, no behavior change.
    diag_acc: Option<DiagAccumulator>,
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
        let chunk_cap = resolve_chunk_rows(std::env::var(CHUNK_ROWS_ENV).ok().as_deref());
        dbg_log!(
            "session chunk config: cap={chunk_cap} rows (env {CHUNK_ROWS_ENV}, \
             default {DEFAULT_CHUNK_ROWS}, ceiling {MAX_CHUNK_ROWS})"
        );
        dbg_log!(
            "session built: layers={} resident weight bytes={} vocab={} hidden={}",
            layers.len(),
            weight_bytes,
            cfg.vocab_size,
            cfg.hidden
        );
        // The diag accumulator exists exactly when the
        // context carries a recorder; it reads its report
        // destination from the environment once, here.
        let diag_acc = ctx.diag_recorder().map(|_| DiagAccumulator::from_env());
        if let Some(acc) = &diag_acc {
            dbg_log!(
                "session diag timing on: drains after every chunk, report to {}",
                acc.out_path().map_or_else(
                    || "the diag channel only".to_string(),
                    |p| p.display().to_string()
                )
            );
        }
        // Flush the weight uploads before the session is
        // handed out. `upload_bytes` stages its copies on
        // the queue: they execute in queue order ahead of
        // the next submission, so an unflushed session
        // leaves the whole weight set (16.76 GB on the real
        // Glimmer file) to be copied in front of the first
        // forward's first chunk, and that chunk's blocking
        // readback is where the copy time lands: the
        // prefill profile's ~1.7 s first-slice latency
        // (bench/prefill-profile-2026-11-05.md), in the
        // prefill wall but in no kernel's timing, since
        // copies are not dispatches. Submitting an empty
        // command buffer flushes the queue's pending
        // writes and the poll pays the copies here, at
        // construction, where "session built" should mean
        // the weights are resident. The total copy work is
        // unchanged; only its attribution moves.
        ctx.queue.submit([ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("session upload flush"),
            })
            .finish()]);
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GlimmerError::StepFailed {
                step: "session",
                detail: format!("flushing the session's weight uploads failed: {e}"),
            })?;
        Ok(ResidentSession {
            backend,
            cfg: cfg.clone(),
            embedding,
            embedding_dtype: weights.embedding.dtype,
            final_norm,
            output,
            layers,
            kv: DeviceKv::new(&pattern, ring_capacity(cfg, chunk_cap)),
            chunk_cap,
            position: 0,
            plans: HashMap::new(),
            stats: SessionStats::default(),
            diag_acc,
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
        self.kv = DeviceKv::new(&pattern, ring_capacity(&self.cfg, self.chunk_cap));
        self.plans.clear();
        self.position = 0;
        dbg_log!("session reset: kv cleared, position 0");
    }

    /// Run one forward over `tokens` and return the finished
    /// logits of the last token, exactly the values the trait
    /// path's `GlimmerModel::forward` returns. The prompt is
    /// processed in chunks (one submission each); a
    /// single-token decode is one chunk, one submission, one
    /// readback. A multi-chunk forward reads back only its
    /// final chunk (see [`defers_readback`]): the chunk
    /// submissions pipeline, the host preparing chunk N+1
    /// while the device still runs chunk N, and the final
    /// chunk's blocking map is the call's only device wait.
    ///
    /// # Errors
    ///
    /// [`GlimmerError::EmptyInput`] on an empty slice and
    /// [`GlimmerError::TokenOutOfRange`] naming the first token
    /// past the vocabulary, matching the model's own
    /// validation; kernel refusals surface as
    /// [`GlimmerError::StepFailed`] under the owning step.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.validate_tokens(tokens)?;
        self.stats.forwards += 1;
        self.stats.tokens += tokens.len() as u64;
        // The chunk count is known before any chunk runs
        // (the cap is fixed for the session's life), so the
        // loop can mark every chunk but the last deferred:
        // their logits are computed and copied exactly as
        // before but never mapped, and the submissions
        // pipeline instead of each one draining the device
        // on a blocking readback whose contents the next
        // iteration would overwrite unread.
        let mut count = 0usize;
        let mut remaining = tokens.len();
        while remaining > 0 {
            let rows = self.chunk_rows(remaining);
            count += 1;
            remaining -= rows;
        }
        let mut rest = tokens;
        let mut logits = Vec::new();
        let mut index = 0usize;
        while !rest.is_empty() {
            let rows = self.chunk_rows(rest.len());
            let defer = defers_readback(index, count);
            match self
                .forward_chunk(&rest[..rows], ChunkTail::Logits, &[], defer)?
                .0
            {
                ChunkOutcome::Logits(chunk_logits) => logits = chunk_logits,
                // A deferred chunk produces no outcome; the
                // final chunk's logits are the forward's.
                ChunkOutcome::Deferred => {}
                ChunkOutcome::Argmax(..) | ChunkOutcome::Block(..) => {
                    unreachable!("logits tail yields logits")
                }
            }
            rest = &rest[rows..];
            index += 1;
        }
        Ok(logits)
    }

    /// Run one forward over `tokens` and return the greedy
    /// next token and its finished logit, reduced ON THE
    /// DEVICE: the same forward as [`ResidentSession::forward`]
    /// (same chunks, same chain, same head), except each
    /// chunk's head segment appends K8's argmax reduction and
    /// the readback is the reduction's eight bytes (winning
    /// index, winning logit) instead of the logits vector.
    /// The winner is the host sampler's greedy pick (maximum
    /// logit, lowest index on an exact tie), so a generation
    /// driven by this method token-matches one driven by
    /// `forward` plus a host argmax, which is the property
    /// the resident-session test pins. Callers that need the
    /// full distribution (sampling, top-k, probabilities)
    /// keep using `forward`; this method exists for the
    /// greedy path, where the vector itself is waste. Like
    /// `forward`, a multi-chunk call defers every chunk's
    /// readback but the final chunk's (see
    /// [`defers_readback`]): an intermediate chunk's winner
    /// is overwritten unread by the next chunk, so its
    /// eight-byte map was the same pure serialization.
    ///
    /// # Errors
    ///
    /// The same validation and step failures as
    /// [`ResidentSession::forward`].
    pub fn forward_argmax(&mut self, tokens: &[u32]) -> Result<(u32, f32)> {
        self.validate_tokens(tokens)?;
        self.stats.forwards += 1;
        self.stats.tokens += tokens.len() as u64;
        // The same deferral as `forward`: the chunk count
        // is known up front, and every chunk but the last
        // submits without its readback.
        let mut count = 0usize;
        let mut remaining = tokens.len();
        while remaining > 0 {
            let rows = self.chunk_rows(remaining);
            count += 1;
            remaining -= rows;
        }
        let mut rest = tokens;
        let mut winner = (0, f32::NEG_INFINITY);
        let mut index = 0usize;
        while !rest.is_empty() {
            let rows = self.chunk_rows(rest.len());
            let defer = defers_readback(index, count);
            match self
                .forward_chunk(&rest[..rows], ChunkTail::Argmax, &[], defer)?
                .0
            {
                ChunkOutcome::Argmax(id, logit) => winner = (id, logit),
                // A deferred chunk's winner is never read;
                // the final chunk's winner is the call's.
                ChunkOutcome::Deferred => {}
                ChunkOutcome::Logits(_) | ChunkOutcome::Block(_) => {
                    unreachable!("argmax tail yields argmax")
                }
            }
            rest = &rest[rows..];
            index += 1;
        }
        Ok(winner)
    }

    /// Run one forward over `tokens` and return the finished
    /// logits of the last token plus, for each layer index in
    /// `capture_layers`, the residual rows that entered that
    /// layer ([`LayerInputCapture`], ascending layer order):
    /// the resident counterpart of
    /// `GlimmerModel::forward_capturing`, with the same
    /// additive-capture contract. Captures span the whole
    /// forward: a prompt that splits into chunks yields one
    /// capture per layer with the chunks' rows concatenated
    /// in position order.
    ///
    /// # Errors
    ///
    /// The same validation and step failures as
    /// [`ResidentSession::forward`], plus
    /// [`GlimmerError::CaptureLayerOutOfRange`] for a capture
    /// index at or past the layer count, refused before any
    /// chunk runs (the model's own rule).
    pub fn forward_capturing(
        &mut self,
        tokens: &[u32],
        capture_layers: &[usize],
    ) -> Result<(Vec<f32>, Vec<LayerInputCapture>)> {
        self.validate_tokens(tokens)?;
        let layers = self.validate_capture(capture_layers)?;
        self.stats.forwards += 1;
        self.stats.tokens += tokens.len() as u64;
        let mut rest = tokens;
        let mut logits = Vec::new();
        let mut captures: Vec<LayerInputCapture> = Vec::new();
        while !rest.is_empty() {
            let rows = self.chunk_rows(rest.len());
            let (outcome, chunk_captures) =
                self.forward_chunk(&rest[..rows], ChunkTail::Logits, &layers, false)?;
            match outcome {
                ChunkOutcome::Logits(chunk_logits) => logits = chunk_logits,
                ChunkOutcome::Argmax(..) | ChunkOutcome::Block(..) | ChunkOutcome::Deferred => {
                    unreachable!("logits tail yields logits")
                }
            }
            merge_captures(&mut captures, chunk_captures);
            rest = &rest[rows..];
        }
        Ok((logits, captures))
    }

    /// Run one forward over `tokens` and return the finished
    /// logits of EVERY token in the block, row-major
    /// `[tokens][vocab]`, plus the layer-input captures
    /// `capture_layers` names (possibly none): the resident
    /// counterpart of `GlimmerModel::forward_block_capturing`,
    /// and the verify-round forward of the speculative loop.
    /// Each row is finished exactly as [`ResidentSession::
    /// forward`] finishes its single row (the same final norm
    /// and head, run at the block's batch through the block
    /// tail), so row `i` is the distribution for the token
    /// following the prefix extended by `tokens[..=i]`.
    ///
    /// # Errors
    ///
    /// The same validation and step failures as
    /// [`ResidentSession::forward_capturing`].
    pub fn forward_block_capturing(
        &mut self,
        tokens: &[u32],
        capture_layers: &[usize],
    ) -> Result<(Vec<f32>, Vec<LayerInputCapture>)> {
        self.validate_tokens(tokens)?;
        let layers = self.validate_capture(capture_layers)?;
        self.stats.forwards += 1;
        self.stats.tokens += tokens.len() as u64;
        let mut rest = tokens;
        let mut logits: Vec<f32> = Vec::new();
        let mut captures: Vec<LayerInputCapture> = Vec::new();
        while !rest.is_empty() {
            let rows = self.chunk_rows(rest.len());
            let (outcome, chunk_captures) =
                self.forward_chunk(&rest[..rows], ChunkTail::BlockLogits, &layers, false)?;
            match outcome {
                ChunkOutcome::Block(chunk_logits) => logits.extend_from_slice(&chunk_logits),
                ChunkOutcome::Logits(_) | ChunkOutcome::Argmax(..) | ChunkOutcome::Deferred => {
                    unreachable!("block tail yields block logits")
                }
            }
            merge_captures(&mut captures, chunk_captures);
            rest = &rest[rows..];
        }
        Ok((logits, captures))
    }

    /// Rewind the session to an earlier `position`: the KV
    /// entries at positions `position..` are abandoned, and
    /// the next forward continues at `position`, overwriting
    /// them positionally.
    ///
    /// Soundness: KV slots are addressed by absolute position
    /// (global layers) or by `position % ring` (windowed
    /// layers), every attention view derives from the position
    /// a chunk ends at, and a chunk's appends land before its
    /// queries attend. The one hazard is the windowed ring: a
    /// discarded write can alias a committed key a full ring
    /// away. The ring spans `window + chunk_cap - 1`
    /// positions, so an alias lands inside a later query's
    /// window only when the discarded span exceeds the
    /// session's chunk cap (the algebra is the ring
    /// capacity comment's own). This method therefore refuses
    /// a rewind discarding more than `chunk_cap`
    /// positions; [`ResidentSession::reset`] is the path for
    /// a full rewind. A speculative round discards at most
    /// its block (depth + 1 positions), far inside the bound.
    ///
    /// # Errors
    ///
    /// [`GlimmerError::StepFailed`] under the `session` step
    /// when `position` is ahead of the current position or
    /// the discarded span exceeds the bound above.
    pub fn rewind_to(&mut self, position: usize) -> Result<()> {
        if position > self.position {
            return Err(GlimmerError::StepFailed {
                step: "session",
                detail: format!(
                    "rewind to {position} is ahead of the current position {}",
                    self.position
                ),
            });
        }
        let span = self.position - position;
        if span > self.chunk_cap {
            return Err(GlimmerError::StepFailed {
                step: "session",
                detail: format!(
                    "rewind from {} to {position} discards {span} positions, past the \
                     {}-position ring headroom; reset the session instead",
                    self.position, self.chunk_cap
                ),
            });
        }
        if span > 0 {
            dbg_log!(
                "session rewind: position {} -> {position} (span {span}); entries past the \
                 target are never read (views derive from position) and are overwritten \
                 positionally by the next forward",
                self.position
            );
            self.position = position;
        }
        Ok(())
    }

    /// Validate a capture layer list the model's way (an
    /// index at or past the layer count refuses with
    /// [`GlimmerError::CaptureLayerOutOfRange`]) and return
    /// it sorted ascending with duplicates removed: the
    /// chain segments and the returned captures both walk
    /// layers in ascending order.
    fn validate_capture(&self, capture_layers: &[usize]) -> Result<Vec<usize>> {
        let mut layers: Vec<usize> = capture_layers.to_vec();
        for &layer in &layers {
            if layer >= self.cfg.block_count as usize {
                return Err(GlimmerError::CaptureLayerOutOfRange {
                    layer,
                    block_count: self.cfg.block_count,
                });
            }
        }
        layers.sort_unstable();
        layers.dedup();
        Ok(layers)
    }

    /// The input validation both forward entry points share,
    /// matching the model's own: [`GlimmerError::EmptyInput`]
    /// on an empty slice and [`GlimmerError::TokenOutOfRange`]
    /// naming the first token past the vocabulary.
    fn validate_tokens(&self, tokens: &[u32]) -> Result<()> {
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
        Ok(())
    }

    /// Rows for the next chunk of an `n`-token remainder: the
    /// remainder under the session's resolved chunk cap.
    /// Flash attention materializes no scores, so no budget
    /// shrinks the chunk at long context (the two-pass form's
    /// scores scratch did, and chunks decayed to a handful of
    /// rows at full context; that decay is gone with it).
    fn chunk_rows(&self, n: usize) -> usize {
        chunk_rows_for(self.chunk_cap, n)
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
                let stored = end.min(ring_capacity(&self.cfg, self.chunk_cap));
                (stored, end - stored)
            }
        }
    }

    /// Run one chunk: ensure KV, refresh the plan's positions,
    /// upload the token ids, encode the whole chain into one
    /// command buffer, submit once, and read back per `tail`:
    /// the finished logits, the on-device argmax reduction
    /// over them (eight bytes), or every row's finished
    /// logits. When `capture_layers` is non-empty the chain
    /// encodes as one pass per layer range, with an encoder
    /// copy of stream A into the capture staging at each
    /// named layer's entry; the copies are not dispatches and
    /// the stats count none. When `defer_readback` is set (a
    /// non-final chunk of a multi-chunk `forward` or
    /// `forward_argmax`, see [`defers_readback`]) the chunk
    /// encodes and submits identically but skips its map and
    /// yields [`ChunkOutcome::Deferred`]: the tail's copy
    /// still lands in the plan's staging, unread. Deferral
    /// is honored only for the logits and argmax tails
    /// without captures; every other shape reads back as
    /// before.
    fn forward_chunk(
        &mut self,
        tokens: &[u32],
        tail: ChunkTail,
        capture_layers: &[usize],
        defer_readback: bool,
    ) -> Result<(ChunkOutcome, Vec<LayerInputCapture>)> {
        let ctx = self.backend.context();
        let rows = tokens.len();
        let start = self.position;
        let end = start + rows;
        let defer = defer_readback
            && matches!(tail, ChunkTail::Logits | ChunkTail::Argmax)
            && capture_layers.is_empty();

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

        // A plan stays current while it binds the session's
        // current KV buffers; flash attention keeps no scores
        // scratch, so no coverage condition rides along (the
        // two-pass form rebuilt plans as the scores capacity
        // grew with position).
        let current = self
            .plans
            .get(&rows)
            .is_some_and(|p| p.is_current(&self.kv));
        if !current {
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
                ring_capacity(&self.cfg, self.chunk_cap),
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
        let hidden = self.cfg.hidden as usize;
        // Capture staging for this chunk only: one
        // [rows][hidden] slab per named layer, created when a
        // capturing forward asked for it (a plain forward
        // never allocates it).
        let capture_staging: Option<wgpu::Buffer> = if capture_layers.is_empty() {
            None
        } else {
            Some(ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("session capture staging"),
                size: (capture_layers.len() * rows * hidden * 4) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }))
        };
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resident forward"),
            });
        if let Some(staging_buf) = &capture_staging {
            // Segmented chain: after the layers before a named
            // layer have run, stream A holds that layer's input
            // rows; copy them out, then resume the chain in a
            // fresh pass. The first segment also carries the
            // chain's opening (gather and embedding norm), so a
            // capture of layer 0 reads the post-norm stream.
            let mut from = 0usize;
            for (slot, &layer) in capture_layers.iter().enumerate() {
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("resident forward chain"),
                        timestamp_writes: None,
                    });
                    if slot == 0 {
                        plan.encode_open(&mut pass);
                    }
                    plan.encode_layers(&mut pass, from..layer);
                }
                encoder.copy_buffer_to_buffer(
                    &plan.scratch.stream_a,
                    0,
                    staging_buf,
                    (slot * rows * hidden * 4) as u64,
                    (rows * hidden * 4) as u64,
                );
                from = layer;
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("resident forward chain"),
                    timestamp_writes: None,
                });
                plan.encode_layers(&mut pass, from..plan.layers.len());
            }
        } else {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("resident forward chain"),
                timestamp_writes: None,
            });
            plan.encode_chain(&mut pass);
        }
        // The last-row copy feeds the standalone final norm;
        // a folded close already emitted the head input. A
        // block-tail chunk does not need it: its final norm
        // reads stream A directly.
        if tail != ChunkTail::BlockLogits && plan.final_norm.is_some() {
            plan.copy_last_row(&mut encoder, &self.cfg);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("resident forward head"),
                timestamp_writes: None,
            });
            match tail {
                ChunkTail::BlockLogits => plan.encode_block_head(&mut pass),
                _ => {
                    plan.encode_head(&mut pass);
                    if tail == ChunkTail::Argmax {
                        plan.encode_argmax(&mut pass);
                    }
                }
            }
        }
        match tail {
            ChunkTail::Logits => plan.copy_logits(&mut encoder, &self.cfg),
            ChunkTail::Argmax => plan.copy_argmax(&mut encoder),
            ChunkTail::BlockLogits => plan.copy_block_logits(&mut encoder, &self.cfg),
        }
        let dispatches = match tail {
            ChunkTail::Logits => plan.dispatch_count,
            ChunkTail::Argmax => plan.dispatch_count + 2,
            ChunkTail::BlockLogits => {
                plan.dispatch_count
                    - 1
                    - u64::from(plan.final_norm.is_some())
                    - u64::from(plan.finish.is_some())
                    + plan.block_dispatch_count
            }
        };
        let staging = plan.scratch.staging.clone();
        let argmax_staging = plan.scratch.argmax_staging.clone();
        let block_staging = plan.scratch.block_staging.clone();
        ctx.queue.submit([encoder.finish()]);
        self.stats.submissions += 1;
        self.stats.chunks += 1;
        self.stats.dispatches += dispatches;
        let outcome = match tail {
            // Deferred: the submission stands, the tail's
            // copy is in staging, and no map drains the
            // device for contents nobody will read. The
            // forward's final chunk reads back for all of
            // them.
            ChunkTail::Logits if defer => ChunkOutcome::Deferred,
            ChunkTail::Argmax if defer => ChunkOutcome::Deferred,
            ChunkTail::Logits => {
                let logits = ctx
                    .map_staging(&staging, self.cfg.vocab_size as usize)
                    .map_err(|e| we(e, "lm_head"))?;
                self.stats.readbacks += 1;
                ChunkOutcome::Logits(logits)
            }
            ChunkTail::Argmax => {
                // The staging's two words are the winning
                // index (as u32 bits in the first word's place)
                // and the winning logit itself.
                let words = ctx
                    .map_staging(&argmax_staging, 2)
                    .map_err(|e| we(e, "lm_head"))?;
                self.stats.argmax_readbacks += 1;
                ChunkOutcome::Argmax(words[0].to_bits(), words[1])
            }
            ChunkTail::BlockLogits => {
                let logits = ctx
                    .map_staging(&block_staging, rows * self.cfg.vocab_size as usize)
                    .map_err(|e| we(e, "lm_head"))?;
                self.stats.readbacks += 1;
                ChunkOutcome::Block(logits)
            }
        };
        let captures = match &capture_staging {
            Some(staging_buf) => {
                let flat = ctx
                    .map_staging(staging_buf, capture_layers.len() * rows * hidden)
                    .map_err(|e| we(e, "capture"))?;
                capture_layers
                    .iter()
                    .enumerate()
                    .map(|(slot, &layer)| LayerInputCapture {
                        layer,
                        rows: flat[slot * rows * hidden..(slot + 1) * rows * hidden].to_vec(),
                    })
                    .collect()
            }
            None => Vec::new(),
        };
        self.position = end;
        dbg_log!(
            "session forward: rows={rows} start={start} end={end} dispatches={dispatches} \
             submissions={}",
            self.stats.submissions
        );
        // The chunk's submission is complete and read back,
        // so its timestamp stamps are final: drain them into
        // the report before the next chunk starts filling
        // the window again. A no-op off the timing path.
        self.drain_diag();
        Ok((outcome, captures))
    }

    /// Drain the diag timestamp recorder into the session's
    /// accumulator. Called after every chunk submission (and
    /// from [`ResidentSession::finish_diag_report`]); with no
    /// recorder this is one branch and nothing else, which is
    /// what keeps the off path's encode and submission stream
    /// byte-identical.
    fn drain_diag(&mut self) {
        let Some(acc) = &mut self.diag_acc else {
            return;
        };
        let ctx = self.backend.context();
        if let Some(recorder) = ctx.diag_recorder() {
            acc.drain_and_fold(recorder, &ctx.device, &ctx.queue);
        }
    }

    /// Dispatch records the diag accumulator has folded so
    /// far (zero when timing is off or nothing has run).
    #[must_use]
    pub fn diag_dispatch_count(&self) -> u64 {
        self.diag_acc
            .as_ref()
            .map_or(0, DiagAccumulator::total_records)
    }

    /// Names of the kernels the diag accumulator has folded
    /// so far, in name order (empty when timing is off).
    #[must_use]
    pub fn diag_kernel_names(&self) -> Vec<&'static str> {
        self.diag_acc
            .as_ref()
            .map_or_else(Vec::new, DiagAccumulator::kernel_names)
    }

    /// The diag summary table over everything folded so far,
    /// in the `render_summary` CSV format, or `None` when the
    /// session carries no accumulator (timing off).
    #[must_use]
    pub fn diag_summary_csv(&self) -> Option<String> {
        self.diag_acc.as_ref().map(DiagAccumulator::render_csv)
    }

    /// Finish the diag report now: one last drain, then the
    /// tally line and the summary file (when
    /// `WGPU_COMPUTE_DIAG_OUT` named one). Idempotent; the
    /// session's drop calls it for callers that never do, so
    /// every timing run leaves its report behind.
    pub fn finish_diag_report(&mut self) {
        let Some(acc) = &mut self.diag_acc else {
            return;
        };
        let ctx = self.backend.context();
        if let Some(recorder) = ctx.diag_recorder() {
            acc.drain_and_fold(recorder, &ctx.device, &ctx.queue);
            acc.write_report(recorder.dropped());
        }
    }
}

impl Drop for ResidentSession<'_> {
    fn drop(&mut self) {
        // End of run: whatever the caller did not finish
        // explicitly, finish here. A no-op off the timing
        // path and after an explicit finish.
        self.finish_diag_report();
    }
}

/// What a forward chunk's head segment returns to the host:
/// the finished logits vector, the on-device argmax
/// reduction over it, or every row's finished logits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChunkTail {
    /// Read back the finished logits (`forward`).
    Logits,
    /// Reduce the finished logits on the device and read
    /// back the winner (`forward_argmax`).
    Argmax,
    /// Read back every row's finished logits through the
    /// block head segment (`forward_block_capturing`).
    BlockLogits,
}

/// A forward chunk's result, per its [`ChunkTail`].
enum ChunkOutcome {
    /// The finished logits of the chunk's last token.
    Logits(Vec<f32>),
    /// The winning token id and its finished logit.
    Argmax(u32, f32),
    /// Every row's finished logits, row-major `[rows][vocab]`.
    Block(Vec<f32>),
    /// No outcome: a deferred chunk of a multi-chunk
    /// `forward` or `forward_argmax`, submitted without
    /// its readback. Its tail copy landed in the plan's
    /// staging unread; the call's final chunk produces
    /// the only outcome the caller sees.
    Deferred,
}

#[cfg(test)]
mod tests {
    use super::*;
    use muse_glimmer::PatternSource;

    #[test]
    fn defers_readback_only_before_the_final_chunk() {
        // The pipelining contract, pinned: a single-chunk
        // forward (every decode step, and any prompt under
        // the cap) never defers; its only chunk carries the
        // return value.
        assert!(!defers_readback(0, 1));
        // Two chunks (the resident test's 70-token prompt
        // at the 64-row cap): the first defers, the second
        // reads back.
        assert!(defers_readback(0, 2));
        assert!(!defers_readback(1, 2));
        // The profiled shape (1,022 tokens at the 64-row
        // cap): sixteen chunks, the first fifteen defer.
        assert!(defers_readback(0, 16));
        assert!(defers_readback(14, 16));
        assert!(!defers_readback(15, 16));
    }

    #[test]
    fn pow2_chunks_decompose_the_verify_band() {
        // The exact decompositions the chunked GEMV form
        // dispatches at, largest power of two first.
        assert_eq!(pow2_chunks(2), vec![(0, 2)]);
        assert_eq!(pow2_chunks(3), vec![(0, 2), (2, 1)]);
        assert_eq!(pow2_chunks(4), vec![(0, 4)]);
        assert_eq!(pow2_chunks(5), vec![(0, 4), (4, 1)]);
        assert_eq!(pow2_chunks(6), vec![(0, 4), (4, 2)]);
        assert_eq!(pow2_chunks(7), vec![(0, 4), (4, 2), (6, 1)]);
        assert_eq!(pow2_chunks(8), vec![(0, 8)]);
        assert_eq!(pow2_chunks(15), vec![(0, 8), (8, 4), (12, 2), (14, 1)]);
    }

    #[test]
    fn pow2_chunks_cover_any_batch_contiguously() {
        for batch in 1..=64u32 {
            let chunks = pow2_chunks(batch);
            let covered: u32 = chunks.iter().map(|&(_, len)| len).sum();
            assert_eq!(covered, batch, "batch {batch} must be covered exactly");
            let mut next = 0;
            for &(start, len) in &chunks {
                assert_eq!(start, next, "batch {batch} chunks must be contiguous");
                assert!(
                    len.is_power_of_two(),
                    "batch {batch} chunk {len} is not a power of two"
                );
                next += len;
            }
        }
    }

    #[test]
    fn chunk_cap_resolution_takes_valid_counts() {
        assert_eq!(resolve_chunk_rows(None), DEFAULT_CHUNK_ROWS);
        assert_eq!(resolve_chunk_rows(Some("1")), 1);
        assert_eq!(resolve_chunk_rows(Some("32")), 32);
        assert_eq!(resolve_chunk_rows(Some("64")), MAX_CHUNK_ROWS);
        assert_eq!(resolve_chunk_rows(Some(" 16 ")), 16);
    }

    #[test]
    fn chunk_cap_resolution_falls_back_by_name() {
        // Unparsable, empty, zero, and past the structural
        // ceiling all land on the default, never on a cap the
        // ring algebra was not built around.
        assert_eq!(resolve_chunk_rows(Some("abc")), DEFAULT_CHUNK_ROWS);
        assert_eq!(resolve_chunk_rows(Some("")), DEFAULT_CHUNK_ROWS);
        assert_eq!(resolve_chunk_rows(Some("0")), DEFAULT_CHUNK_ROWS);
        assert_eq!(resolve_chunk_rows(Some("65")), DEFAULT_CHUNK_ROWS);
        assert_eq!(resolve_chunk_rows(Some("-4")), DEFAULT_CHUNK_ROWS);
    }

    #[test]
    fn chunk_rows_hold_the_cap_at_any_remainder() {
        assert_eq!(chunk_rows_for(64, 200), 64);
        assert_eq!(chunk_rows_for(64, 64), 64);
        assert_eq!(chunk_rows_for(64, 6), 6);
        assert_eq!(chunk_rows_for(64, 1), 1);
        // A lowered cap is honored exactly, including the
        // decode-shaped remainder of one.
        assert_eq!(chunk_rows_for(32, 200), 32);
        assert_eq!(chunk_rows_for(32, 1), 1);
        // The cap is the only sizing input: no stored-count
        // term exists anymore, so a chunk deep in a long
        // context sizes exactly like the first chunk. (The
        // two-pass budget this replaced shrank a 64-row chunk
        // to 4 rows at the full 131,072 context.)
        assert_eq!(chunk_rows_for(DEFAULT_CHUNK_ROWS, 1024), 64);
    }

    #[test]
    fn ring_capacity_is_window_plus_cap_headroom() {
        let mut cfg = GlimmerConfig {
            name: None,
            block_count: 1,
            context_length: 131_072,
            hidden: 6656,
            ffn_hidden: 19_968,
            q_heads: 32,
            kv_heads: 2,
            key_length: 128,
            value_length: 128,
            rope_freq_base: 500_000.0,
            rope_freq_base_swa: 500_000.0,
            rms_eps: 1e-5,
            sliding_window: 2048,
            logit_softcap: 20.0,
            logit_scale: 0.196_116_13,
            vocab_size: 202_048,
            layers: vec![LayerKind::SlidingWindow],
            pattern_source: PatternSource::MeasuredFallback,
        };
        // The recorded Glimmer geometry: window 2048, cap 64,
        // ring 2111 (the profile note's ring figure).
        assert_eq!(ring_capacity(&cfg, 64), 2111);
        assert_eq!(ring_capacity(&cfg, 32), 2079);
        assert_eq!(ring_capacity(&cfg, 1), 2048);
        cfg.sliding_window = 4;
        assert_eq!(ring_capacity(&cfg, 64), 67);
    }

    /// The resident fixture's layer shape: full QKV+gate
    /// batch, fused gate, paired FFN, chained norms.
    fn fixture_shape(first: bool) -> LayerFusionShape {
        LayerFusionShape {
            batched: true,
            full_batch_gate: true,
            gate_fused: true,
            ffn_paired: true,
            chained: true,
            first,
        }
    }

    #[test]
    fn dispatch_count_fixture_layers_match_the_resident_pin() {
        // Every projection at one dispatch (the GEMV and
        // GEMM forms quantize nothing, so no stage has a
        // share): layer 0 records 9 and each later layer 8,
        // the per-layer counts the resident session test's
        // 51-per-chunk pin is built from (gather + 9 + 5 x 8
        // + head).
        let one = [1; 8];
        assert_eq!(layer_dispatch_count(fixture_shape(true), one, 0), 9);
        assert_eq!(layer_dispatch_count(fixture_shape(false), one, 0), 8);
    }

    #[test]
    fn dispatch_count_sharing_is_neutral_on_full_batch_layers() {
        // A full-batch paired layer at MMQ rows: the batch
        // is the attention stage's only h projection and
        // the pair is the FFN stage's only one, so each
        // stage already had exactly one quantize. Sharing
        // moves those two quantizes off the products and
        // onto the layer (products 1, shared 1 per stage)
        // and the total does not move: quantize-once costs
        // nothing where there was no duplication.
        // Slots: Q(batch), K, V, gate, output (gated, own
        // quantize), FFN gate (pair), FFN up, FFN down
        // (SwiGLU, own quantize).
        let own = [2, 1, 1, 1, 2, 2, 1, 2];
        let shared = [1, 1, 1, 1, 2, 1, 1, 2];
        assert_eq!(layer_dispatch_count(fixture_shape(true), own, 0), 13);
        assert_eq!(layer_dispatch_count(fixture_shape(true), shared, 2), 13);
        assert_eq!(layer_dispatch_count(fixture_shape(false), shared, 2), 12);
    }

    #[test]
    fn dispatch_count_partial_batch_layer_drops_the_v_quantize() {
        // A partial batch (Q6_K V, the real model's other
        // layer class): the batch and the separate V both
        // quantized the attention stage's h. Sharing drops
        // exactly one dispatch per layer.
        let shape = LayerFusionShape {
            batched: true,
            full_batch_gate: false,
            gate_fused: true,
            ffn_paired: true,
            chained: true,
            first: true,
        };
        // Slots: batch, K (absent), V, gate (absent, rides
        // the batch), output, pair, up (absent), down.
        let own = [2, 1, 2, 1, 2, 2, 1, 2];
        let shared = [1, 1, 1, 1, 2, 1, 1, 2];
        assert_eq!(layer_dispatch_count(shape, own, 0), 15);
        assert_eq!(layer_dispatch_count(shape, shared, 2), 14);
    }

    #[test]
    fn dispatch_count_mixed_layer_shares_both_stages() {
        // Partial batch and an unpaired FFN (Q6_K up): the
        // attention stage shares across batch + V and the
        // FFN stage across gate + up, one dispatch saved
        // per stage. These are the per-layer counts the
        // mixed resident test's chunk pin (93) is built
        // from: 16 for layer 0, 15 for each later layer.
        let shape = LayerFusionShape {
            batched: true,
            full_batch_gate: false,
            gate_fused: true,
            ffn_paired: false,
            chained: true,
            first: true,
        };
        let own = [2, 1, 2, 1, 2, 2, 2, 2];
        let shared = [1, 1, 1, 1, 2, 1, 1, 2];
        assert_eq!(layer_dispatch_count(shape, own, 0), 18);
        assert_eq!(layer_dispatch_count(shape, shared, 2), 16);
        let later = LayerFusionShape {
            first: false,
            ..shape
        };
        assert_eq!(layer_dispatch_count(later, shared, 2), 15);
    }

    #[test]
    fn dispatch_count_unbatched_layer_shares_across_qkv_and_gate() {
        // No batch at all: Q, K, V, and the separate gate
        // projection all read the attention stage's h, and
        // an unpaired FFN reads the FFN stage's h twice.
        // Sharing drops three dispatches in the attention
        // stage and one in the FFN stage.
        let shape = LayerFusionShape {
            batched: false,
            full_batch_gate: false,
            gate_fused: false,
            ffn_paired: false,
            chained: true,
            first: true,
        };
        let own = [2, 2, 2, 2, 2, 2, 2, 2];
        let shared = [1, 1, 1, 1, 2, 1, 1, 2];
        assert_eq!(layer_dispatch_count(shape, own, 0), 23);
        assert_eq!(layer_dispatch_count(shape, shared, 2), 19);
    }
}
