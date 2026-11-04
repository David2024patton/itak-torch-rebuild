//! The DFlash draft model as a device-resident session:
//! the GPU counterpart of `dflash::DflashModel`, composed
//! in the resident pattern [`crate::session::ResidentSession`]
//! established for the Glimmer target (weights uploaded
//! once, position-addressed resident KV, one command chain
//! per pass, logits readback appended to the same encoder).
//!
//! The semantics are the CPU implementation's, step for
//! step; `crates/dflash` is the spec and the correctness
//! oracle:
//!
//! - **Injection** ([`DflashSession::inject`]): conditioning
//!   rows (the target's layer inputs at `dflash.target_layers`,
//!   concatenated per position) are fused by the `fc`
//!   projection and the encoder output RMSNorm; per layer,
//!   K/V projections of the fused rows, the per-head RMSNorm
//!   and NEOX RoPE on K, stored in the layer's resident KV
//!   buffers AT THEIR ABSOLUTE POSITIONS. The resident KV
//!   is a flat position-addressed space, not a ring: the
//!   CPU cache is a position-keyed map whose reads filter
//!   by mask, and the flat layout is that map's exact
//!   device shape (the sliding window is a read-side mask
//!   here, as it is in the CPU implementation).
//! - **Drafting** ([`DflashSession::forward_block`]): the
//!   noise block is gathered from the (borrowed) embedding
//!   table and run through the draft layers; attention
//!   reads the committed positions plus the block's own
//!   rows under the config's causal flag and per-layer
//!   sliding window (K15); the final RMSNorm and the
//!   (borrowed) output head produce the logits. No logit
//!   scale or softcap: the draft applies none. The pass
//!   never writes the resident KV, exactly like the CPU
//!   drafting pass never writes the cache.
//!
//! Composition: K1/K2/K3 GEMV projections at every batch
//! (the draft's blocks are at most `block_size` rows, and
//! staying on the GEMV family keeps the path free of the
//! activation quantization the MMQ prefill forms carry),
//! K4 for every norm and residual add (per-head norms fold
//! heads into rows), K7 SwiGLU, K8 embedding gather, and
//! K15 for the two primitives no Glimmer kernel provides
//! (NEOX rotate, two-source draft attention).
//!
//! Refusals speak draft vocabulary: every backend failure
//! is mapped to [`DflashError::StepFailed`] with the step
//! named, so draft callers (the speculative loop) never
//! see a raw backend error. The validation order of both
//! passes mirrors the CPU implementation's exactly; the
//! one deliberate difference is that the borrowed-table
//! resolution (missing table, wrong shape) happens at
//! session construction instead of at the first forward,
//! because the session must upload the tables once.
//!
//! Diagnostics ride the draft's own seam: the session
//! takes an `Arc<dyn dflash::Diag>` and emits the CPU
//! model's event names (`inject.begin`, `forward_block.end`,
//! and friends) plus `session.ready` at construction, and
//! this crate's `dbg_log` carries the device-side detail.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use dflash::{DflashConfig, DflashError, Diag, NullDiag};
use muse_glimmer::{Dtype, GlimmerError, TensorSource, TensorView};

use crate::compute::WgpuCompute;
use crate::diag::dbg_log;
use crate::error::WgpuError;
use crate::kernels::k15_dflash::{K15AttendCall, K15RotateCall};
use crate::kernels::k1_q4k_gemv::K1Call;
use crate::kernels::k2_q6k_gemv::K2Call;
use crate::kernels::k3_q5k_gemv::K3Call;
use crate::kernels::k4_rmsnorm::K4Call;
use crate::kernels::k7_swiglu::K7Call;
use crate::kernels::k8_head::K8GatherRowsCall;
use crate::session::{upload_norm, upload_proj, ProjWeights};

/// Result alias in draft vocabulary.
type Result<T> = std::result::Result<T, DflashError>;

/// The step name session-level failures carry.
const STEP_SESSION: &str = "dflash session";
/// The step name injection failures carry.
const STEP_INJECT: &str = "dflash inject";
/// The step name drafting failures carry.
const STEP_BLOCK: &str = "dflash forward_block";

/// Rows per injection segment inside one submission. The
/// speculative loop commits whole prompts in one call, so
/// injection segments internally (the resident session's
/// MAX_CHUNK_ROWS discipline) while staying one submission.
const INJECT_CHUNK_ROWS: usize = 64;

/// Map a Glimmer-vocabulary failure into draft vocabulary,
/// preserving a step failure's detail verbatim.
fn de(step: &'static str, e: GlimmerError) -> DflashError {
    match e {
        GlimmerError::StepFailed { detail, .. } => DflashError::StepFailed { step, detail },
        other => DflashError::StepFailed {
            step,
            detail: other.to_string(),
        },
    }
}

/// Map a backend failure into draft vocabulary.
fn we(step: &'static str, e: WgpuError) -> DflashError {
    DflashError::StepFailed {
        step,
        detail: e.to_string(),
    }
}

/// Map a tensor-source failure into the draft's load
/// vocabulary, name for name with the CPU load's mapping.
fn ge(name: &str, e: GlimmerError) -> DflashError {
    match e {
        GlimmerError::MissingTensor { .. } => DflashError::MissingTensor {
            name: name.to_string(),
        },
        // The glimmer source raises shape problems as step
        // failures in its own words; the draft's load does
        // its own shape checks in `checked_raw`, so anything
        // reaching here is a readability failure.
        other => DflashError::TensorUnreadable {
            name: name.to_string(),
            detail: other.to_string(),
        },
    }
}

/// The tensors of one draft layer as raw stored views, in
/// forward order: the mirror of `dflash::LayerWeights`
/// before dequantization.
#[derive(Clone, Debug)]
pub struct DflashRawLayer {
    /// `attn_norm.weight`: pre-attention RMSNorm.
    pub attn_norm: TensorView,
    /// `attn_q.weight`: query projection.
    pub attn_q: TensorView,
    /// `attn_k.weight`: key projection.
    pub attn_k: TensorView,
    /// `attn_v.weight`: value projection.
    pub attn_v: TensorView,
    /// `attn_q_norm.weight`: per-head query RMSNorm.
    pub attn_q_norm: TensorView,
    /// `attn_k_norm.weight`: per-head key RMSNorm.
    pub attn_k_norm: TensorView,
    /// `attn_output.weight`: attention output projection.
    pub attn_output: TensorView,
    /// `post_attention_norm.weight` when the file carries
    /// it (absent in the real file).
    pub post_attention_norm: Option<TensorView>,
    /// `ffn_norm.weight`: pre-FFN RMSNorm.
    pub ffn_norm: TensorView,
    /// `ffn_gate.weight`: SwiGLU gate projection.
    pub ffn_gate: TensorView,
    /// `ffn_up.weight`: SwiGLU up projection.
    pub ffn_up: TensorView,
    /// `ffn_down.weight`: FFN down projection.
    pub ffn_down: TensorView,
    /// `post_ffw_norm.weight` when the file carries it
    /// (absent in the real file).
    pub post_ffw_norm: Option<TensorView>,
}

/// The draft's full weight set as raw stored views: the
/// mirror of `dflash::Weights` before dequantization,
/// loaded from a [`muse_glimmer::TensorSource`] with the
/// CPU load's exact checks (same variant gates, same shape
/// requirements, same `fc` width derivation), so a file
/// the CPU load refuses is refused here with the same
/// error, and a file it accepts uploads here unchanged.
#[derive(Clone, Debug)]
pub struct DflashRawWeights {
    /// `fc.weight`: the conditioning fusion projection.
    pub fc: TensorView,
    /// `enc.output_norm.weight`: RMSNorm over fused rows.
    pub enc_output_norm: TensorView,
    /// `output_norm.weight`: final RMSNorm.
    pub output_norm: TensorView,
    /// The draft's own `token_embd.weight` when present;
    /// `None` is the borrowed state, as on the CPU load.
    pub embedding: Option<TensorView>,
    /// The draft's own `output.weight` when present.
    pub output: Option<TensorView>,
    /// The target hidden size derived from `fc` at load,
    /// exactly as the CPU load derives it.
    pub target_hidden: u32,
    /// Per-layer weights, in layer order.
    pub layers: Vec<DflashRawLayer>,
}

/// Per-layer tensor suffixes the raw load always requires,
/// in the CPU load's order.
const LAYER_SUFFIXES: [&str; 11] = [
    "attn_norm.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "attn_output.weight",
    "ffn_norm.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
];

/// Variant gates, with the CPU load's refusal text
/// verbatim: a file carrying any of these tensors is a
/// different draft than this module implements.
const VARIANT_GATES: [(&str, &str); 3] = [
    (
        "d2t",
        "a d2t tensor means the draft predicts a reduced vocabulary that maps to target ids; this module implements the plain full-vocabulary draft",
    ),
    (
        "markov_w1.weight",
        "a markov_w1 tensor marks a DSpark draft (Markov + confidence heads); this module implements plain DFlash",
    ),
    (
        "selector_hidden.weight",
        "a selector_hidden tensor marks a DFlash2 draft (conv + selector lattice); this module implements plain DFlash",
    ),
];

/// Probe a tensor's presence through the raw source: the
/// glimmer source has no `has_tensor`, so presence is a
/// fetch whose missing-tensor error means absent and
/// whose other errors propagate in draft vocabulary.
fn probe<S: TensorSource>(source: &S, name: &str) -> Result<Option<TensorView>> {
    match source.tensor(name) {
        Ok(view) => Ok(Some(view)),
        Err(GlimmerError::MissingTensor { .. }) => Ok(None),
        Err(e) => Err(ge(name, e)),
    }
}

/// Fetch one tensor and verify its shape, the CPU load's
/// `checked` in raw-view form.
fn checked_raw<S: TensorSource>(source: &S, name: &str, expected: &[u64]) -> Result<TensorView> {
    let view = source.tensor(name).map_err(|e| ge(name, e))?;
    if view.shape != expected {
        return Err(DflashError::ShapeMismatch {
            name: name.to_string(),
            expected: expected.to_vec(),
            actual: view.shape.clone(),
        });
    }
    Ok(view)
}

impl DflashRawWeights {
    /// Load the full raw weight set for `cfg` from
    /// `source`, mirroring `dflash::Weights::load` check
    /// for check. The first missing or misshaped tensor
    /// aborts the load with the CPU load's named error.
    ///
    /// # Errors
    ///
    /// The [`DflashError`] the CPU load raises for the
    /// same file: variant gates as
    /// [`DflashError::UnsupportedFeature`], absence as
    /// [`DflashError::MissingTensor`], wrong shapes as
    /// [`DflashError::ShapeMismatch`], an `fc` width that
    /// is not a whole multiple of the target layer count
    /// as [`DflashError::InvalidMetadata`].
    pub fn load<S: TensorSource>(source: &S, cfg: &DflashConfig) -> Result<Self> {
        for (gate, detail) in VARIANT_GATES {
            if probe(source, gate)?.is_some() {
                return Err(DflashError::UnsupportedFeature {
                    name: gate.to_string(),
                    detail: detail.to_string(),
                });
            }
        }

        let hidden = u64::from(cfg.hidden);
        let ffn = u64::from(cfg.ffn_hidden);
        let vocab = u64::from(cfg.vocab_size);
        let q_w = u64::from(cfg.q_width());
        let k_w = u64::from(cfg.k_width());
        let v_w = u64::from(cfg.v_width());
        let out_w = u64::from(cfg.attn_out_width());

        // The fc derivation, the CPU load's exact words.
        let fc = source.tensor("fc.weight").map_err(|e| ge("fc.weight", e))?;
        if fc.shape.len() != 2 || fc.shape[1] != hidden {
            return Err(DflashError::ShapeMismatch {
                name: fc.name.clone(),
                expected: vec![0, hidden],
                actual: fc.shape.clone(),
            });
        }
        let cond_in = fc.shape[0];
        let n_target = cfg.target_layers.len() as u64;
        if cond_in == 0 || cond_in % n_target != 0 {
            return Err(DflashError::InvalidMetadata {
                key: "dflash.target_layers".to_string(),
                detail: format!(
                    "fc.weight input width {cond_in} is not a whole multiple of the {} target layers, so the concatenated conditioning rows cannot be its input",
                    cfg.target_layers.len()
                ),
            });
        }
        let target_hidden =
            u32::try_from(cond_in / n_target).map_err(|_| DflashError::InvalidMetadata {
                key: "dflash.target_layers".to_string(),
                detail: "target hidden size derived from fc.weight does not fit in u32".to_string(),
            })?;

        let enc_output_norm = checked_raw(source, "enc.output_norm.weight", &[hidden])?;
        let output_norm = checked_raw(source, "output_norm.weight", &[hidden])?;
        let embedding = match probe(source, "token_embd.weight")? {
            Some(_) => Some(checked_raw(source, "token_embd.weight", &[hidden, vocab])?),
            None => None,
        };
        let output = match probe(source, "output.weight")? {
            Some(_) => Some(checked_raw(source, "output.weight", &[hidden, vocab])?),
            None => None,
        };

        let mut layers = Vec::with_capacity(cfg.block_count as usize);
        for index in 0..cfg.block_count {
            let get = |suffix: &str, shape: &[u64]| {
                checked_raw(source, &format!("blk.{index}.{suffix}"), shape)
            };
            let post = |suffix: &str| -> Result<Option<TensorView>> {
                let name = format!("blk.{index}.{suffix}");
                match probe(source, &name)? {
                    Some(_) => Ok(Some(checked_raw(source, &name, &[hidden])?)),
                    None => Ok(None),
                }
            };
            layers.push(DflashRawLayer {
                attn_norm: get(LAYER_SUFFIXES[0], &[hidden])?,
                attn_q: get(LAYER_SUFFIXES[1], &[hidden, q_w])?,
                attn_k: get(LAYER_SUFFIXES[2], &[hidden, k_w])?,
                attn_v: get(LAYER_SUFFIXES[3], &[hidden, v_w])?,
                attn_q_norm: get(LAYER_SUFFIXES[4], &[u64::from(cfg.key_length)])?,
                attn_k_norm: get(LAYER_SUFFIXES[5], &[u64::from(cfg.key_length)])?,
                attn_output: get(LAYER_SUFFIXES[6], &[out_w, hidden])?,
                post_attention_norm: post("post_attention_norm.weight")?,
                ffn_norm: get(LAYER_SUFFIXES[7], &[hidden])?,
                ffn_gate: get(LAYER_SUFFIXES[8], &[hidden, ffn])?,
                ffn_up: get(LAYER_SUFFIXES[9], &[hidden, ffn])?,
                ffn_down: get(LAYER_SUFFIXES[10], &[ffn, hidden])?,
                post_ffw_norm: post("post_ffw_norm.weight")?,
            });
        }
        Ok(DflashRawWeights {
            fc,
            enc_output_norm,
            output_norm,
            embedding,
            output,
            target_hidden,
            layers,
        })
    }

    /// The row width conditioning features must have, as
    /// loaded from `fc` (the CPU weights' own definition).
    pub fn conditioning_width(&self) -> usize {
        self.fc.shape[0] as usize
    }
}

/// Session construction knobs. Every operational limit of
/// the draft session lives here as config with a default,
/// never as a compiled constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DflashSessionConfig {
    /// Positions the resident KV space holds. The draft's
    /// KV is position-addressed, so capacity bounds the
    /// highest position an injection may write (and the
    /// resident bytes: `capacity x (k_width + v_width) x 4`
    /// per layer; at the real file's geometry the default
    /// costs about 320 MB across the five layers). An
    /// injection reaching past it refuses by name; it is
    /// a deployment choice (context length vs. memory),
    /// not a property of the model.
    pub kv_capacity: usize,
}

impl Default for DflashSessionConfig {
    fn default() -> Self {
        DflashSessionConfig { kv_capacity: 8192 }
    }
}

/// Counters for one draft session, in the resident
/// session's accounting spirit: passes, submissions, and
/// readbacks are separate numbers so the steady state is
/// never dressed up.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DflashSessionStats {
    /// Injection calls served.
    pub injects: u64,
    /// Drafting (block) calls served.
    pub blocks: u64,
    /// Queue submissions carrying a pass chain.
    pub submissions: u64,
    /// Logits/fused readbacks mapped.
    pub readbacks: u64,
    /// Positions the resident KV currently holds.
    pub committed_positions: usize,
}

/// One prepared device dispatch in a pass plan.
#[derive(Debug)]
enum DevCall {
    /// K1 (Q4_K GEMV) projection.
    Proj1(K1Call),
    /// K2 (Q6_K GEMV) projection.
    Proj2(K2Call),
    /// K3 (Q5_K GEMV) projection.
    Proj3(K3Call),
    /// K4 norm or residual add.
    Norm(K4Call),
    /// K7 SwiGLU.
    Swiglu(K7Call),
    /// K8 embedding gather.
    Gather(K8GatherRowsCall),
    /// K15 NEOX rotate.
    Rotate(K15RotateCall),
    /// K15 draft attention.
    Attend(K15AttendCall),
}

impl DevCall {
    fn encode_into(&self, pass: &mut wgpu::ComputePass<'_>) {
        match self {
            DevCall::Proj1(c) => c.encode_into(pass),
            DevCall::Proj2(c) => c.encode_into(pass),
            DevCall::Proj3(c) => c.encode_into(pass),
            DevCall::Norm(c) => c.encode_into(pass),
            DevCall::Swiglu(c) => c.encode_into(pass),
            DevCall::Gather(c) => c.encode_into(pass),
            DevCall::Rotate(c) => c.encode_into(pass),
            DevCall::Attend(c) => c.encode_into(pass),
        }
    }
}

/// Prepare one projection dispatch on the GEMV family,
/// whichever of K1/K2/K3 the weights' dtype selected at
/// upload (the resident session's own dispatch split,
/// minus its prefill matrix forms: the draft's row counts
/// never leave the GEMV regime, and the GEMV path carries
/// no activation quantization, so cross-backend parity is
/// by construction rather than by tolerance luck).
fn prepare_proj(
    backend: &WgpuCompute,
    weights: &ProjWeights,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    rows: usize,
    step: &'static str,
) -> Result<DevCall> {
    let ctx = backend.context();
    let batch = rows as u32;
    match weights {
        ProjWeights::Q4K(w) => backend
            .k1()
            .prepare(ctx, w, input, output, batch)
            .map(DevCall::Proj1),
        ProjWeights::Q6K(w) => backend
            .k2()
            .prepare(ctx, w, input, output, batch)
            .map(DevCall::Proj2),
        ProjWeights::Q5K(w) => backend
            .k3()
            .prepare(ctx, w, input, output, batch)
            .map(DevCall::Proj3),
    }
    .map_err(|e| we(step, e))
}

/// The width figure a conditioning-width refusal reports,
/// the CPU implementation's own helper.
fn reported_width(total: usize, rows: usize) -> usize {
    total.checked_div(rows).unwrap_or(total)
}

/// One draft layer's device residency: projection handles,
/// norm vectors as f32 buffers, and the position-addressed
/// KV space.
#[derive(Debug)]
struct DevLayer {
    attn_norm: wgpu::Buffer,
    attn_q_norm: wgpu::Buffer,
    attn_k_norm: wgpu::Buffer,
    ffn_norm: wgpu::Buffer,
    post_attention_norm: Option<wgpu::Buffer>,
    post_ffw_norm: Option<wgpu::Buffer>,
    attn_q: ProjWeights,
    attn_k: ProjWeights,
    attn_v: ProjWeights,
    attn_output: ProjWeights,
    ffn_gate: ProjWeights,
    ffn_up: ProjWeights,
    ffn_down: ProjWeights,
    /// Committed K rows, slot = absolute position:
    /// `[kv_capacity][k_width]` f32.
    kv_k: wgpu::Buffer,
    /// Committed V rows, same addressing: `[kv_capacity][v_width]`.
    kv_v: wgpu::Buffer,
}

/// The session's activation scratch, sized once for the
/// largest shapes the config allows (a full block, one
/// injection segment).
#[derive(Debug)]
struct Scratch {
    stream_a: wgpu::Buffer,
    stream_b: wgpu::Buffer,
    h: wgpu::Buffer,
    q: wgpu::Buffer,
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    qn: wgpu::Buffer,
    kn: wgpu::Buffer,
    attn: wgpu::Buffer,
    branch: wgpu::Buffer,
    gate: wgpu::Buffer,
    up: wgpu::Buffer,
    act: wgpu::Buffer,
    logits: wgpu::Buffer,
    /// Logits readback staging (`COPY_DST | MAP_READ`).
    staging: wgpu::Buffer,
    /// Block token ids, `[block_size]` u32.
    tokens: wgpu::Buffer,
    /// The committed position set, sorted ascending,
    /// `[kv_capacity]` u32; the attention dispatch reads
    /// its first `committed` entries.
    positions: wgpu::Buffer,
    /// The current block's positions, `[block_size]` u32.
    block_positions: wgpu::Buffer,
    /// Injection conditioning rows, one segment.
    features: wgpu::Buffer,
    fused: wgpu::Buffer,
    fused_n: wgpu::Buffer,
    ik: wgpu::Buffer,
    iv: wgpu::Buffer,
    ikn: wgpu::Buffer,
    /// One injection segment's positions, u32.
    chunk_positions: wgpu::Buffer,
    /// Fused-rows readback staging (`COPY_DST | MAP_READ`).
    fuse_staging: wgpu::Buffer,
}

impl Scratch {
    fn new(
        backend: &WgpuCompute,
        cfg: &DflashConfig,
        conditioning_width: usize,
        kv_capacity: usize,
    ) -> Result<Self> {
        let ctx = backend.context();
        let rows = cfg.block_size as usize;
        let hidden = cfg.hidden as usize;
        let buf = |label: &str, elems: usize| ctx.scratch_buffer(label, (elems * 4) as u64);
        let staging = |label: &str, elems: usize| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (elems * 4).max(4) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        Ok(Scratch {
            stream_a: buf("dflash stream a", rows * hidden),
            stream_b: buf("dflash stream b", rows * hidden),
            h: buf("dflash normed", rows * hidden),
            q: buf("dflash q", rows * cfg.q_width() as usize),
            k: buf("dflash block k", rows * cfg.k_width() as usize),
            v: buf("dflash block v", rows * cfg.v_width() as usize),
            qn: buf("dflash q normed", rows * cfg.q_width() as usize),
            kn: buf("dflash k normed", rows * cfg.k_width() as usize),
            attn: buf("dflash attn out", rows * cfg.attn_out_width() as usize),
            branch: buf("dflash branch", rows * hidden),
            gate: buf("dflash ffn gate", rows * cfg.ffn_hidden as usize),
            up: buf("dflash ffn up", rows * cfg.ffn_hidden as usize),
            act: buf("dflash ffn act", rows * cfg.ffn_hidden as usize),
            logits: buf("dflash logits", rows * cfg.vocab_size as usize),
            staging: staging("dflash logits staging", rows * cfg.vocab_size as usize),
            tokens: buf("dflash tokens", rows),
            positions: buf("dflash committed positions", kv_capacity),
            block_positions: buf("dflash block positions", rows),
            features: buf("dflash features", INJECT_CHUNK_ROWS * conditioning_width),
            fused: buf("dflash fused", INJECT_CHUNK_ROWS * hidden),
            fused_n: buf("dflash fused normed", INJECT_CHUNK_ROWS * hidden),
            ik: buf(
                "dflash inject k",
                INJECT_CHUNK_ROWS * cfg.k_width() as usize,
            ),
            iv: buf(
                "dflash inject v",
                INJECT_CHUNK_ROWS * cfg.v_width() as usize,
            ),
            ikn: buf(
                "dflash inject k normed",
                INJECT_CHUNK_ROWS * cfg.k_width() as usize,
            ),
            chunk_positions: buf("dflash chunk positions", INJECT_CHUNK_ROWS),
            fuse_staging: staging("dflash fuse staging", INJECT_CHUNK_ROWS * hidden),
        })
    }
}

/// A prepared drafting chain for one block row count,
/// cached across calls (the resident session's plan
/// discipline). The attention call's runtime params
/// (block start, committed count) are rewritten per call
/// through its params buffer.
#[derive(Debug)]
struct BlockPlan {
    calls: Vec<DevCall>,
    /// Index of the attention call in `calls`.
    attend: usize,
    /// Block rows this plan serves.
    rows: usize,
    /// Vocabulary width of the logits readback.
    vocab: usize,
}

impl BlockPlan {
    fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &crate::device::GpuContext,
        block_start: usize,
        committed: usize,
        logits: &wgpu::Buffer,
        staging: &wgpu::Buffer,
    ) {
        if let DevCall::Attend(call) = &self.calls[self.attend] {
            call.set_runtime(ctx, block_start as u32, committed as u32);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dflash block chain"),
                timestamp_writes: None,
            });
            for call in &self.calls {
                call.encode_into(&mut pass);
            }
        }
        encoder.copy_buffer_to_buffer(logits, 0, staging, 0, (self.rows * self.vocab * 4) as u64);
    }
}

/// A prepared injection chain for one segment row count:
/// the fusion prefix plus, per layer, the K/V projection
/// set. Encoding interleaves each layer's dispatch pass
/// with the encoder copies that land its rows in the
/// resident KV at their absolute positions (the scratch
/// K/V buffers are reused per layer, so the copies cannot
/// be deferred past the next layer's pass).
#[derive(Debug)]
struct InjectPlan {
    prefix: Vec<DevCall>,
    layers: Vec<[DevCall; 4]>,
}

impl InjectPlan {
    fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scratch: &Scratch,
        layers: &[DevLayer],
        positions: &[usize],
        k_width: usize,
        v_width: usize,
    ) {
        for (li, calls) in self.layers.iter().enumerate() {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("dflash inject segment"),
                    timestamp_writes: None,
                });
                if li == 0 {
                    for call in &self.prefix {
                        call.encode_into(&mut pass);
                    }
                }
                for call in calls {
                    call.encode_into(&mut pass);
                }
            }
            let layer = &layers[li];
            for (start, len) in contiguous_runs(positions) {
                let dst = (positions[start] * k_width * 4) as u64;
                encoder.copy_buffer_to_buffer(
                    &scratch.ikn,
                    (start * k_width * 4) as u64,
                    &layer.kv_k,
                    dst,
                    (len * k_width * 4) as u64,
                );
                let dst = (positions[start] * v_width * 4) as u64;
                encoder.copy_buffer_to_buffer(
                    &scratch.iv,
                    (start * v_width * 4) as u64,
                    &layer.kv_v,
                    dst,
                    (len * v_width * 4) as u64,
                );
            }
        }
    }
}

/// Maximal contiguous runs in a strictly increasing
/// position list, as `(first index, length)` pairs: one
/// encoder copy per run instead of one per row.
fn contiguous_runs(positions: &[usize]) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut i = 0;
    while i < positions.len() {
        let mut len = 1;
        while i + len < positions.len() && positions[i + len] == positions[i + len - 1] + 1 {
            len += 1;
        }
        runs.push((i, len));
        i += len;
    }
    runs
}

/// The DFlash draft model resident on the device: the
/// GPU counterpart of `dflash::DflashModel`, with the same
/// two passes and the same refusal vocabulary. Borrows
/// the backend like [`crate::session::ResidentSession`]
/// does; the speculative runner leaks the backend for the
/// process lifetime, as it already does for the target's
/// resident session.
pub struct DflashSession<'a> {
    backend: &'a WgpuCompute,
    cfg: DflashConfig,
    conditioning_width: usize,
    fc: ProjWeights,
    enc_output_norm: wgpu::Buffer,
    output_norm: wgpu::Buffer,
    /// The resolved embedding table's stored bytes
    /// (the draft's own when present, else the target's).
    embedding: wgpu::Buffer,
    embedding_dtype: Dtype,
    /// The resolved output head (own or borrowed).
    head: ProjWeights,
    layers: Vec<DevLayer>,
    inv_freq: wgpu::Buffer,
    scratch: Scratch,
    /// The committed position set, mirroring the CPU
    /// cache's key set exactly (ascending iteration).
    held: BTreeSet<usize>,
    block_plans: HashMap<usize, BlockPlan>,
    inject_plans: HashMap<usize, InjectPlan>,
    kv_capacity: usize,
    stats: DflashSessionStats,
    diag: Arc<dyn Diag>,
}

impl std::fmt::Debug for DflashSession<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The diag sink is a trait object without a Debug
        // impl; everything else prints.
        f.debug_struct("DflashSession")
            .field("conditioning_width", &self.conditioning_width)
            .field("layers", &self.layers.len())
            .field("held", &self.held)
            .field("kv_capacity", &self.kv_capacity)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

impl<'a> DflashSession<'a> {
    /// Build a session over `backend` from the raw
    /// weights, with a silent diagnostic sink. The
    /// borrowed target tables are the raw views of the
    /// target file's `token_embd.weight` and
    /// `output.weight`, consulted only where the draft
    /// file carries none of its own.
    ///
    /// # Errors
    ///
    /// [`DflashError::MissingTargetTensors`] when neither
    /// the draft nor the caller supplies a table;
    /// [`DflashError::ShapeMismatch`] when a resolved
    /// table is not `[hidden, vocab]`;
    /// [`DflashError::StepFailed`] for every device-side
    /// refusal (an unquantized projection dtype, a buffer
    /// past the device's limits, a zero KV capacity), with
    /// the cause in the detail.
    pub fn new(
        backend: &'a WgpuCompute,
        cfg: &DflashConfig,
        weights: &DflashRawWeights,
        target_embedding: Option<&TensorView>,
        target_output: Option<&TensorView>,
        session_cfg: DflashSessionConfig,
    ) -> Result<Self> {
        Self::with_diag(
            backend,
            cfg,
            weights,
            target_embedding,
            target_output,
            session_cfg,
            Arc::new(NullDiag),
        )
    }

    /// Build a session, reporting through `diag` (the
    /// draft's diagnostic seam).
    ///
    /// # Errors
    ///
    /// See [`DflashSession::new`].
    #[allow(clippy::too_many_arguments)]
    pub fn with_diag(
        backend: &'a WgpuCompute,
        cfg: &DflashConfig,
        weights: &DflashRawWeights,
        target_embedding: Option<&TensorView>,
        target_output: Option<&TensorView>,
        session_cfg: DflashSessionConfig,
        diag: Arc<dyn Diag>,
    ) -> Result<Self> {
        if session_cfg.kv_capacity == 0 {
            return Err(DflashError::StepFailed {
                step: STEP_SESSION,
                detail: "kv_capacity is zero; the resident KV must hold at least one position"
                    .to_string(),
            });
        }
        if weights.layers.len() != cfg.block_count as usize {
            return Err(DflashError::StepFailed {
                step: STEP_SESSION,
                detail: format!(
                    "raw weights carry {} layers for a config of {}",
                    weights.layers.len(),
                    cfg.block_count
                ),
            });
        }
        // Borrowed-table resolution: the CPU forward does
        // this per call; the session does it once, here,
        // because the tables upload once. Same names, same
        // refusal order (embedding before head), same
        // shape requirement.
        let emb_view = weights.embedding.as_ref().or(target_embedding).ok_or(
            DflashError::MissingTargetTensors {
                which: "token_embd.weight",
            },
        )?;
        let out_view =
            weights
                .output
                .as_ref()
                .or(target_output)
                .ok_or(DflashError::MissingTargetTensors {
                    which: "output.weight",
                })?;
        let table_shape = vec![u64::from(cfg.hidden), u64::from(cfg.vocab_size)];
        for view in [emb_view, out_view] {
            if view.shape != table_shape {
                return Err(DflashError::ShapeMismatch {
                    name: view.name.clone(),
                    expected: table_shape.clone(),
                    actual: view.shape.clone(),
                });
            }
        }
        if !matches!(
            emb_view.dtype,
            Dtype::F32
                | Dtype::F16
                | Dtype::Bf16
                | Dtype::Q4K
                | Dtype::Q5K
                | Dtype::Q6K
                | Dtype::Q8_0
        ) {
            return Err(DflashError::StepFailed {
                step: STEP_SESSION,
                detail: format!(
                    "embedding table {} has dtype {:?}; the gather kernel serves f32, f16, bf16, Q4_K, Q5_K, Q6_K, or Q8_0",
                    emb_view.name, emb_view.dtype
                ),
            });
        }

        let ctx = backend.context();
        ctx.check_tensor_fits(&emb_view.name, emb_view.data.len() as u64)
            .map_err(|e| we(STEP_SESSION, e))?;
        let embedding = ctx.upload_bytes(&emb_view.name, &emb_view.data);
        let embedding_dtype = emb_view.dtype;
        let head = upload_proj(backend, out_view, STEP_SESSION).map_err(|e| de(STEP_SESSION, e))?;
        let fc =
            upload_proj(backend, &weights.fc, STEP_SESSION).map_err(|e| de(STEP_SESSION, e))?;
        let enc_output_norm = upload_norm(backend, &weights.enc_output_norm, STEP_SESSION)
            .map_err(|e| de(STEP_SESSION, e))?;
        let output_norm = upload_norm(backend, &weights.output_norm, STEP_SESSION)
            .map_err(|e| de(STEP_SESSION, e))?;

        let k_bytes = (session_cfg.kv_capacity * cfg.k_width() as usize * 4) as u64;
        let v_bytes = (session_cfg.kv_capacity * cfg.v_width() as usize * 4) as u64;
        ctx.check_tensor_fits("dflash resident kv (k)", k_bytes)
            .map_err(|e| we(STEP_SESSION, e))?;
        ctx.check_tensor_fits("dflash resident kv (v)", v_bytes)
            .map_err(|e| we(STEP_SESSION, e))?;
        let mut layers = Vec::with_capacity(weights.layers.len());
        for w in &weights.layers {
            let norm = |view: &TensorView| {
                upload_norm(backend, view, STEP_SESSION).map_err(|e| de(STEP_SESSION, e))
            };
            let proj = |view: &TensorView| {
                upload_proj(backend, view, STEP_SESSION).map_err(|e| de(STEP_SESSION, e))
            };
            layers.push(DevLayer {
                attn_norm: norm(&w.attn_norm)?,
                attn_q_norm: norm(&w.attn_q_norm)?,
                attn_k_norm: norm(&w.attn_k_norm)?,
                ffn_norm: norm(&w.ffn_norm)?,
                post_attention_norm: match &w.post_attention_norm {
                    Some(view) => Some(norm(view)?),
                    None => None,
                },
                post_ffw_norm: match &w.post_ffw_norm {
                    Some(view) => Some(norm(view)?),
                    None => None,
                },
                attn_q: proj(&w.attn_q)?,
                attn_k: proj(&w.attn_k)?,
                attn_v: proj(&w.attn_v)?,
                attn_output: proj(&w.attn_output)?,
                ffn_gate: proj(&w.ffn_gate)?,
                ffn_up: proj(&w.ffn_up)?,
                ffn_down: proj(&w.ffn_down)?,
                kv_k: ctx.scratch_buffer("dflash kv k", k_bytes),
                kv_v: ctx.scratch_buffer("dflash kv v", v_bytes),
            });
        }
        let inv_freq =
            backend
                .k15()
                .upload_inv_freq(ctx, cfg.key_length as usize, cfg.rope_freq_base);
        let scratch = Scratch::new(
            backend,
            cfg,
            weights.conditioning_width(),
            session_cfg.kv_capacity,
        )?;
        dbg_log!(
            "dflash session built: layers={} capacity={} conditioning_width={}",
            layers.len(),
            session_cfg.kv_capacity,
            weights.conditioning_width()
        );
        diag.event(
            "session.ready",
            &format!(
                "layers={} kv_capacity={} conditioning_width={} target_hidden={}",
                layers.len(),
                session_cfg.kv_capacity,
                weights.conditioning_width(),
                weights.target_hidden
            ),
        );
        Ok(DflashSession {
            backend,
            cfg: cfg.clone(),
            conditioning_width: weights.conditioning_width(),
            fc,
            enc_output_norm,
            output_norm,
            embedding,
            embedding_dtype,
            head,
            layers,
            inv_freq,
            scratch,
            held: BTreeSet::new(),
            block_plans: HashMap::new(),
            inject_plans: HashMap::new(),
            kv_capacity: session_cfg.kv_capacity,
            stats: DflashSessionStats::default(),
            diag,
        })
    }

    /// The config this session serves.
    pub fn config(&self) -> &DflashConfig {
        &self.cfg
    }

    /// The conditioning row width (from the loaded `fc`).
    pub fn conditioning_width(&self) -> usize {
        self.conditioning_width
    }

    /// Positions currently held in the resident KV: the
    /// CPU cache's `len`.
    pub fn committed_len(&self) -> usize {
        self.held.len()
    }

    /// The held positions, ascending (the CPU cache's
    /// `positions`).
    pub fn positions(&self) -> Vec<usize> {
        self.held.iter().copied().collect()
    }

    /// The session's counters.
    pub fn stats(&self) -> DflashSessionStats {
        DflashSessionStats {
            committed_positions: self.held.len(),
            ..self.stats
        }
    }

    /// Abandon every committed position: the held set
    /// empties and the resident rows become unreachable
    /// (the attention dispatch reads only the positions
    /// array), exactly a fresh CPU cache.
    pub fn reset(&mut self) {
        self.held.clear();
        self.diag
            .event("session.reset", "committed positions cleared");
    }
}

/// The residual close of one sub-block: with a post norm,
/// the branch is RMSNormed before the add (one K4
/// weighted-residual dispatch, the CPU composition's exact
/// math); without one, a plain K4 add.
#[allow(clippy::too_many_arguments)]
fn close_call(
    backend: &WgpuCompute,
    residual: &wgpu::Buffer,
    branch: &wgpu::Buffer,
    post_norm: Option<&wgpu::Buffer>,
    out: &wgpu::Buffer,
    rows: usize,
    width: usize,
    eps: f32,
) -> DevCall {
    let ctx = backend.context();
    match post_norm {
        Some(weight) => DevCall::Norm(
            backend
                .k4()
                .prepare_norm_residual(ctx, residual, branch, weight, out, rows, width, eps),
        ),
        None => DevCall::Norm(
            backend
                .k4()
                .prepare_add(ctx, residual, branch, out, rows * width),
        ),
    }
}

impl<'a> DflashSession<'a> {
    /// Build (once per row count) the drafting chain.
    fn build_block_plan(&self, rows: usize) -> Result<BlockPlan> {
        let backend = self.backend;
        let ctx = backend.context();
        let cfg = &self.cfg;
        let s = &self.scratch;
        let hidden = cfg.hidden as usize;
        let ffn = cfg.ffn_hidden as usize;
        let q_heads = cfg.q_heads as usize;
        let kv_heads = cfg.kv_heads as usize;
        let key_len = cfg.key_length as usize;
        let value_len = cfg.value_length as usize;
        let eps = cfg.rms_eps;
        let scale = 1.0 / (key_len as f32).sqrt();
        let mut calls: Vec<DevCall> = Vec::new();
        let mut attend_idx = 0usize;
        calls.push(DevCall::Gather(
            backend
                .k8()
                .prepare_gather_rows(
                    ctx,
                    &self.embedding,
                    &s.tokens,
                    &s.stream_a,
                    rows,
                    hidden,
                    self.embedding_dtype,
                )
                .map_err(|e| de(STEP_BLOCK, e))?,
        ));
        let mut cur_is_a = true;
        for (li, layer) in self.layers.iter().enumerate() {
            let (cur, other) = if cur_is_a {
                (&s.stream_a, &s.stream_b)
            } else {
                (&s.stream_b, &s.stream_a)
            };
            // Attention sub-block: pre-norm, Q/K/V, per-head
            // norms (heads folded into rows), NEOX rotates,
            // K15 attention over committed + block, output
            // projection, close.
            calls.push(DevCall::Norm(backend.k4().prepare_rms_norm(
                ctx,
                cur,
                &layer.attn_norm,
                &s.h,
                rows,
                hidden,
                eps,
            )));
            calls.push(prepare_proj(
                backend,
                &layer.attn_q,
                &s.h,
                &s.q,
                rows,
                STEP_BLOCK,
            )?);
            calls.push(prepare_proj(
                backend,
                &layer.attn_k,
                &s.h,
                &s.k,
                rows,
                STEP_BLOCK,
            )?);
            calls.push(prepare_proj(
                backend,
                &layer.attn_v,
                &s.h,
                &s.v,
                rows,
                STEP_BLOCK,
            )?);
            calls.push(DevCall::Norm(backend.k4().prepare_rms_norm(
                ctx,
                &s.q,
                &layer.attn_q_norm,
                &s.qn,
                rows * q_heads,
                key_len,
                eps,
            )));
            calls.push(DevCall::Norm(backend.k4().prepare_rms_norm(
                ctx,
                &s.k,
                &layer.attn_k_norm,
                &s.kn,
                rows * kv_heads,
                key_len,
                eps,
            )));
            calls.push(DevCall::Rotate(
                backend
                    .k15()
                    .prepare_rotate(
                        ctx,
                        &s.qn,
                        &s.block_positions,
                        &self.inv_freq,
                        rows,
                        q_heads,
                        key_len,
                    )
                    .map_err(|e| we(STEP_BLOCK, e))?,
            ));
            calls.push(DevCall::Rotate(
                backend
                    .k15()
                    .prepare_rotate(
                        ctx,
                        &s.kn,
                        &s.block_positions,
                        &self.inv_freq,
                        rows,
                        kv_heads,
                        key_len,
                    )
                    .map_err(|e| we(STEP_BLOCK, e))?,
            ));
            attend_idx = calls.len();
            let sliding = if cfg.is_sliding(li) == Some(true) {
                Some(cfg.sliding_window)
            } else {
                None
            };
            calls.push(DevCall::Attend(
                backend
                    .k15()
                    .prepare_attend(
                        ctx,
                        &s.qn,
                        &layer.kv_k,
                        &layer.kv_v,
                        &s.kn,
                        &s.v,
                        &s.positions,
                        &s.attn,
                        rows,
                        q_heads,
                        kv_heads,
                        key_len,
                        value_len,
                        rows,
                        0,
                        0,
                        cfg.causal_attention,
                        sliding,
                        scale,
                    )
                    .map_err(|e| we(STEP_BLOCK, e))?,
            ));
            calls.push(prepare_proj(
                backend,
                &layer.attn_output,
                &s.attn,
                &s.branch,
                rows,
                STEP_BLOCK,
            )?);
            calls.push(close_call(
                backend,
                cur,
                &s.branch,
                layer.post_attention_norm.as_ref(),
                other,
                rows,
                hidden,
                eps,
            ));
            cur_is_a = !cur_is_a;
            // FFN sub-block: pre-norm, gate/up, SwiGLU,
            // down, close back into the other stream.
            let (cur, other) = if cur_is_a {
                (&s.stream_a, &s.stream_b)
            } else {
                (&s.stream_b, &s.stream_a)
            };
            calls.push(DevCall::Norm(backend.k4().prepare_rms_norm(
                ctx,
                cur,
                &layer.ffn_norm,
                &s.h,
                rows,
                hidden,
                eps,
            )));
            calls.push(prepare_proj(
                backend,
                &layer.ffn_gate,
                &s.h,
                &s.gate,
                rows,
                STEP_BLOCK,
            )?);
            calls.push(prepare_proj(
                backend,
                &layer.ffn_up,
                &s.h,
                &s.up,
                rows,
                STEP_BLOCK,
            )?);
            calls.push(DevCall::Swiglu(backend.k7().prepare_swiglu(
                ctx,
                &s.gate,
                &s.up,
                &s.act,
                rows * ffn,
            )));
            calls.push(prepare_proj(
                backend,
                &layer.ffn_down,
                &s.act,
                &s.branch,
                rows,
                STEP_BLOCK,
            )?);
            calls.push(close_call(
                backend,
                cur,
                &s.branch,
                layer.post_ffw_norm.as_ref(),
                other,
                rows,
                hidden,
                eps,
            ));
            cur_is_a = !cur_is_a;
        }
        let cur = if cur_is_a { &s.stream_a } else { &s.stream_b };
        calls.push(DevCall::Norm(backend.k4().prepare_rms_norm(
            ctx,
            cur,
            &self.output_norm,
            &s.h,
            rows,
            hidden,
            eps,
        )));
        calls.push(prepare_proj(
            backend, &self.head, &s.h, &s.logits, rows, STEP_BLOCK,
        )?);
        Ok(BlockPlan {
            calls,
            attend: attend_idx,
            rows,
            vocab: cfg.vocab_size as usize,
        })
    }

    /// Build (once per segment row count) the injection
    /// chain.
    fn build_inject_plan(&self, rows: usize) -> Result<InjectPlan> {
        let backend = self.backend;
        let ctx = backend.context();
        let cfg = &self.cfg;
        let s = &self.scratch;
        let hidden = cfg.hidden as usize;
        let kv_heads = cfg.kv_heads as usize;
        let key_len = cfg.key_length as usize;
        let eps = cfg.rms_eps;
        let prefix = vec![
            prepare_proj(backend, &self.fc, &s.features, &s.fused, rows, STEP_INJECT)?,
            DevCall::Norm(backend.k4().prepare_rms_norm(
                ctx,
                &s.fused,
                &self.enc_output_norm,
                &s.fused_n,
                rows,
                hidden,
                eps,
            )),
        ];
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            layers.push([
                prepare_proj(backend, &layer.attn_k, &s.fused_n, &s.ik, rows, STEP_INJECT)?,
                prepare_proj(backend, &layer.attn_v, &s.fused_n, &s.iv, rows, STEP_INJECT)?,
                DevCall::Norm(backend.k4().prepare_rms_norm(
                    ctx,
                    &s.ik,
                    &layer.attn_k_norm,
                    &s.ikn,
                    rows * kv_heads,
                    key_len,
                    eps,
                )),
                DevCall::Rotate(
                    backend
                        .k15()
                        .prepare_rotate(
                            ctx,
                            &s.ikn,
                            &s.chunk_positions,
                            &self.inv_freq,
                            rows,
                            kv_heads,
                            key_len,
                        )
                        .map_err(|e| we(STEP_INJECT, e))?,
                ),
            ]);
        }
        Ok(InjectPlan { prefix, layers })
    }

    /// Rewrite the committed positions array from the
    /// held set (ascending u32s). Queue writes are ordered
    /// before the next submission, so a plan prepared once
    /// reads the current set on every block.
    fn write_positions_array(&self) {
        if self.held.is_empty() {
            return;
        }
        let bytes: Vec<u8> = self
            .held
            .iter()
            .flat_map(|p| (*p as u32).to_le_bytes())
            .collect();
        self.backend
            .context()
            .queue
            .write_buffer(&self.scratch.positions, 0, &bytes);
    }

    /// Fuse conditioning features and write the resulting
    /// K/V into the resident KV at `positions`: the GPU
    /// form of `DflashModel::inject`, with the same
    /// validation order and the same refusals. Positions
    /// must be strictly increasing within the call;
    /// re-injecting a held position overwrites its K/V.
    ///
    /// # Errors
    ///
    /// The CPU pass's errors in its order
    /// ([`DflashError::EmptyInput`],
    /// [`DflashError::PositionConflict`],
    /// [`DflashError::SequenceTooLong`],
    /// [`DflashError::ConditioningWidth`]), plus one
    /// session-specific refusal: a position at or past the
    /// configured KV capacity is
    /// [`DflashError::StepFailed`] at step `"dflash inject"`,
    /// naming the capacity to raise.
    pub fn inject(&mut self, features: &[f32], positions: &[usize]) -> Result<()> {
        if positions.is_empty() {
            return Err(DflashError::EmptyInput);
        }
        for pair in positions.windows(2) {
            if pair[1] <= pair[0] {
                return Err(DflashError::PositionConflict {
                    position: pair[1],
                    detail: "injection positions must be strictly increasing within one call"
                        .to_string(),
                });
            }
        }
        let total = positions[positions.len() - 1] + 1;
        if total > self.cfg.context_length as usize {
            return Err(DflashError::SequenceTooLong {
                total,
                context_length: self.cfg.context_length,
            });
        }
        let width = self.conditioning_width;
        if features.len() != positions.len() * width {
            return Err(DflashError::ConditioningWidth {
                expected: width,
                actual: reported_width(features.len(), positions.len()),
            });
        }
        let last = positions[positions.len() - 1];
        if last >= self.kv_capacity {
            self.diag.event(
                "inject.refused",
                &format!(
                    "position {last} is past the session KV capacity of {} positions",
                    self.kv_capacity
                ),
            );
            return Err(DflashError::StepFailed {
                step: STEP_INJECT,
                detail: format!(
                    "position {last} is past the session's KV capacity of {} positions; raise DflashSessionConfig::kv_capacity for this deployment",
                    self.kv_capacity
                ),
            });
        }
        self.diag.event(
            "inject.begin",
            &format!(
                "rows={} positions={}..{} width={}",
                positions.len(),
                positions[0],
                last,
                width
            ),
        );
        let ctx = self.backend.context();
        let k_w = self.cfg.k_width() as usize;
        let v_w = self.cfg.v_width() as usize;
        let mut offset = 0;
        while offset < positions.len() {
            let rows = INJECT_CHUNK_ROWS.min(positions.len() - offset);
            let chunk_positions = &positions[offset..offset + rows];
            let chunk_features = &features[offset * width..(offset + rows) * width];
            let feature_bytes: Vec<u8> = chunk_features
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            ctx.queue
                .write_buffer(&self.scratch.features, 0, &feature_bytes);
            let position_bytes: Vec<u8> = chunk_positions
                .iter()
                .flat_map(|p| (*p as u32).to_le_bytes())
                .collect();
            ctx.queue
                .write_buffer(&self.scratch.chunk_positions, 0, &position_bytes);
            if !self.inject_plans.contains_key(&rows) {
                let plan = self.build_inject_plan(rows)?;
                self.inject_plans.insert(rows, plan);
            }
            let plan = &self.inject_plans[&rows];
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("dflash inject chain"),
                });
            plan.encode(
                &mut encoder,
                &self.scratch,
                &self.layers,
                chunk_positions,
                k_w,
                v_w,
            );
            ctx.queue.submit([encoder.finish()]);
            self.stats.submissions += 1;
            offset += rows;
        }
        self.held.extend(positions.iter().copied());
        self.write_positions_array();
        self.stats.injects += 1;
        self.diag.event(
            "inject.end",
            &format!("cached_positions={}", self.held.len()),
        );
        Ok(())
    }

    /// The conditioning fusion on its own (the CPU model's
    /// public seam): `fc` projection followed by the
    /// encoder output RMSNorm, read back to the host.
    ///
    /// # Errors
    ///
    /// [`DflashError::ConditioningWidth`] on a feature
    /// slice that is not `rows` rows of the conditioning
    /// width; [`DflashError::StepFailed`] for device-side
    /// failures.
    pub fn fuse(&mut self, features: &[f32], rows: usize) -> Result<Vec<f32>> {
        let width = self.conditioning_width;
        if features.len() != rows * width {
            return Err(DflashError::ConditioningWidth {
                expected: width,
                actual: reported_width(features.len(), rows),
            });
        }
        if rows == 0 {
            return Ok(Vec::new());
        }
        let hidden = self.cfg.hidden as usize;
        let ctx = self.backend.context();
        let mut out = Vec::with_capacity(rows * hidden);
        let mut offset = 0;
        while offset < rows {
            let crows = INJECT_CHUNK_ROWS.min(rows - offset);
            let chunk = &features[offset * width..(offset + crows) * width];
            let bytes: Vec<u8> = chunk.iter().flat_map(|v| v.to_le_bytes()).collect();
            ctx.queue.write_buffer(&self.scratch.features, 0, &bytes);
            if !self.inject_plans.contains_key(&crows) {
                let plan = self.build_inject_plan(crows)?;
                self.inject_plans.insert(crows, plan);
            }
            let plan = &self.inject_plans[&crows];
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("dflash fuse chain"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("dflash fuse prefix"),
                    timestamp_writes: None,
                });
                for call in &plan.prefix {
                    call.encode_into(&mut pass);
                }
            }
            encoder.copy_buffer_to_buffer(
                &self.scratch.fused_n,
                0,
                &self.scratch.fuse_staging,
                0,
                (crows * hidden * 4) as u64,
            );
            ctx.queue.submit([encoder.finish()]);
            self.stats.submissions += 1;
            let part = ctx
                .map_staging(&self.scratch.fuse_staging, crows * hidden)
                .map_err(|e| we(STEP_SESSION, e))?;
            out.extend_from_slice(&part);
            self.stats.readbacks += 1;
            offset += crows;
        }
        Ok(out)
    }

    /// Run one block through the draft and return
    /// per-position logits, row-major
    /// `[tokens.len()][vocab]`: the GPU form of
    /// `DflashModel::forward_block`, same validation order,
    /// same row meaning (row `i` is the distribution for
    /// the token following block position
    /// `start_position + i`). The pass never writes the
    /// resident KV.
    ///
    /// # Errors
    ///
    /// The CPU pass's errors in its order
    /// ([`DflashError::EmptyInput`],
    /// [`DflashError::BlockTooLong`],
    /// [`DflashError::TokenOutOfRange`],
    /// [`DflashError::SequenceTooLong`]);
    /// [`DflashError::StepFailed`] for device-side
    /// failures. (The borrowed-table refusals the CPU pass
    /// raises here were already settled at construction;
    /// see the module docs.)
    pub fn forward_block(&mut self, tokens: &[u32], start_position: usize) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        if tokens.is_empty() {
            return Err(DflashError::EmptyInput);
        }
        if tokens.len() > cfg.block_size as usize {
            return Err(DflashError::BlockTooLong {
                len: tokens.len(),
                block_size: cfg.block_size,
            });
        }
        for &token in tokens {
            if token >= cfg.vocab_size {
                return Err(DflashError::TokenOutOfRange {
                    token,
                    vocab_size: cfg.vocab_size,
                });
            }
        }
        let total = start_position + tokens.len();
        if total > cfg.context_length as usize {
            return Err(DflashError::SequenceTooLong {
                total,
                context_length: cfg.context_length,
            });
        }
        self.diag.event(
            "forward_block.begin",
            &format!(
                "tokens={} start={} cached={}",
                tokens.len(),
                start_position,
                self.held.len()
            ),
        );
        let rows = tokens.len();
        let vocab = cfg.vocab_size as usize;
        let ctx = self.backend.context();
        let token_bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        ctx.queue
            .write_buffer(&self.scratch.tokens, 0, &token_bytes);
        let position_bytes: Vec<u8> = (0..rows)
            .flat_map(|i| ((start_position + i) as u32).to_le_bytes())
            .collect();
        ctx.queue
            .write_buffer(&self.scratch.block_positions, 0, &position_bytes);
        if !self.block_plans.contains_key(&rows) {
            let plan = self.build_block_plan(rows)?;
            self.block_plans.insert(rows, plan);
        }
        let plan = &self.block_plans[&rows];
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dflash block forward"),
            });
        plan.encode(
            &mut encoder,
            ctx,
            start_position,
            self.held.len(),
            &self.scratch.logits,
            &self.scratch.staging,
        );
        ctx.queue.submit([encoder.finish()]);
        self.stats.submissions += 1;
        let logits = ctx
            .map_staging(&self.scratch.staging, rows * vocab)
            .map_err(|e| we(STEP_BLOCK, e))?;
        self.stats.blocks += 1;
        self.stats.readbacks += 1;
        self.diag
            .event("forward_block.end", &format!("logits={}", logits.len()));
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_runs_group_only_true_runs() {
        assert_eq!(contiguous_runs(&[]), Vec::<(usize, usize)>::new());
        assert_eq!(contiguous_runs(&[0, 1, 2, 3]), vec![(0, 4)]);
        assert_eq!(
            contiguous_runs(&[0, 1, 4, 5, 9]),
            vec![(0, 2), (2, 2), (4, 1)]
        );
        assert_eq!(contiguous_runs(&[7]), vec![(0, 1)]);
    }

    #[test]
    fn reported_width_matches_the_cpu_helper() {
        assert_eq!(reported_width(24, 4), 6);
        assert_eq!(reported_width(24, 0), 24);
        assert_eq!(reported_width(25, 4), 6);
    }

    #[test]
    fn session_config_default_is_the_documented_capacity() {
        assert_eq!(DflashSessionConfig::default().kv_capacity, 8192);
    }
}
