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
//!   sliding window (K17); the final RMSNorm and the
//!   (borrowed) output head produce the logits. No logit
//!   scale or softcap: the draft applies none. The pass
//!   never writes the resident KV, exactly like the CPU
//!   drafting pass never writes the cache.
//!
//! Composition: projections walk the same small-batch
//! ladder the resident session walks below its matrix
//! threshold (the register-streaming MMV products first
//! where the band routing table admits the batch, the
//! K18 small-batch products behind them, the chunked
//! GEMV form behind those, the plain K1/K2/K3 GEMV last;
//! see [`prepare_proj`]), K4 for every norm and residual
//! add (per-head norms fold heads into rows), K7 SwiGLU,
//! K8 embedding gather, and K17 for the two primitives
//! no Glimmer kernel provides (NEOX rotate, two-source
//! draft attention). The draft never takes the resident
//! ladder's matrix forms: its blocks are at most
//! `block_size` rows, and staying below the matrix
//! threshold keeps the path free of the activation
//! quantization the MMQ prefill forms carry except
//! where an operator opens the K18 rung, whose quantize
//! is the ladder's own.
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

use crate::compute::{mmv_shape_of, MmvBands, MmvFamily, WgpuCompute};
use crate::diag::dbg_log;
use crate::error::WgpuError;
use crate::kernels::k12_q8_quant::K12QuantCall;
use crate::kernels::k17_dflash::{K17AttendCall, K17RotateCall};
use crate::kernels::k18_small_batch::K18Call;
use crate::kernels::k19_q4k_mmv::K19Call;
use crate::kernels::k1_q4k_gemv::K1Call;
use crate::kernels::k20_q6k_mmv::K20Call;
use crate::kernels::k21_q5k_mmv::K21Call;
use crate::kernels::k2_q6k_gemv::K2Call;
use crate::kernels::k3_q5k_gemv::K3Call;
use crate::kernels::k4_rmsnorm::K4Call;
use crate::kernels::k7_swiglu::K7Call;
use crate::kernels::k8_head::K8GatherRowsCall;
use crate::session::{pow2_chunks, upload_norm, upload_proj, ProjWeights, GEMM_MIN_ROWS};
use crate::timestamps::{
    build_git, build_version, CaptureRecord, DiagAccumulator, DiagMode, RunManifest,
};

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
    /// Logit values the block forwards' readbacks carried
    /// to the host. A full block forward carries
    /// `block_size * vocab` values; the capped forward
    /// carries only its prefix, and this counter is where
    /// the difference shows.
    pub logits_values_read: u64,
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
    /// K19 (Q4_K register-streaming) projection: the
    /// ladder's MMV rung for the Q4_K family.
    Mmv1(K19Call),
    /// K20 (Q6_K register-streaming) projection: the
    /// ladder's MMV rung for the Q6_K family.
    Mmv2(K20Call),
    /// K21 (Q5_K register-streaming) projection: the
    /// ladder's MMV rung for the Q5_K family.
    Mmv3(K21Call),
    /// K12 quantize + K18 small-batch product: the
    /// ladder's small-batch rung, one variant for all
    /// three families (the call types are shared).
    Small {
        /// The activation quantize dispatch.
        quant: K12QuantCall,
        /// The integer product dispatch.
        mmq: K18Call,
    },
    /// K1 dispatches chunked at power-of-two row
    /// counts: the ladder's chunked GEMV rung.
    Chunked1(Vec<K1Call>),
    /// K2 dispatches chunked at power-of-two row
    /// counts: the ladder's chunked GEMV rung.
    Chunked2(Vec<K2Call>),
    /// K3 dispatches chunked at power-of-two row
    /// counts: the ladder's chunked GEMV rung.
    Chunked3(Vec<K3Call>),
    /// K4 norm or residual add.
    Norm(K4Call),
    /// K7 SwiGLU.
    Swiglu(K7Call),
    /// K8 embedding gather.
    Gather(K8GatherRowsCall),
    /// K17 NEOX rotate.
    Rotate(K17RotateCall),
    /// K17 draft attention.
    Attend(K17AttendCall),
}

impl DevCall {
    fn encode_into(&self, pass: &mut wgpu::ComputePass<'_>) {
        match self {
            DevCall::Proj1(c) => c.encode_into(pass),
            DevCall::Proj2(c) => c.encode_into(pass),
            DevCall::Proj3(c) => c.encode_into(pass),
            DevCall::Mmv1(c) => c.encode_into(pass),
            DevCall::Mmv2(c) => c.encode_into(pass),
            DevCall::Mmv3(c) => c.encode_into(pass),
            DevCall::Small { quant, mmq } => {
                quant.encode_into(pass);
                mmq.encode_into(pass);
            }
            DevCall::Chunked1(calls) => {
                for c in calls {
                    c.encode_into(pass);
                }
            }
            DevCall::Chunked2(calls) => {
                for c in calls {
                    c.encode_into(pass);
                }
            }
            DevCall::Chunked3(calls) => {
                for c in calls {
                    c.encode_into(pass);
                }
            }
            DevCall::Norm(c) => c.encode_into(pass),
            DevCall::Swiglu(c) => c.encode_into(pass),
            DevCall::Gather(c) => c.encode_into(pass),
            DevCall::Rotate(c) => c.encode_into(pass),
            DevCall::Attend(c) => c.encode_into(pass),
        }
    }
}

/// One rung of the draft projection ladder, in the
/// order [`prepare_proj`] tries them. The GEMV form is
/// not a rung: it is the unconditional last resort the
/// ladder falls back to, exactly as in the resident
/// session's ladder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProjForm {
    /// The register-streaming MMV products (K19/K20/K21).
    Mmv,
    /// The K18 small-batch products (K12 quantize + K18).
    Small,
    /// The chunked GEMV form (windowed GEMV dispatches
    /// at power-of-two row counts).
    Chunked,
}

/// The routing family and geometry of one projection:
/// the family's band cell and the shape class are what
/// the routing policy resolves on.
fn proj_dims(weights: &ProjWeights) -> (MmvFamily, usize, usize) {
    match weights {
        ProjWeights::Q4K(w) => (MmvFamily::Q4K, w.in_features, w.out_rows),
        ProjWeights::Q6K(w) => (MmvFamily::Q6K, w.in_features, w.out_rows),
        ProjWeights::Q5K(w) => (MmvFamily::Q5K, w.in_features, w.out_rows),
    }
}

/// The rungs the draft ladder tries for one projection,
/// in ladder order, resolved from the same policy the
/// resident session resolves: the band routing table
/// ([`MmvBands`], narrowed by the global MMV ceiling)
/// for the MMV rung, the small-batch ceiling for the
/// K18 rung, and the chunked ceiling plus the matrix
/// threshold bound for the chunked rung. Every
/// expression here is the resident predicate's own
/// (`session::mmv_applies`, `session::small_batch_applies`,
/// `session::chunked_gemv_applies`) over the resolved
/// policy values, and [`prepare_proj`] walks exactly
/// this list, so the unit tests below pin the prepare
/// path's resolution, not a copy of it. A pure function
/// of the resolved values so the pin needs no device.
///
/// At shipped defaults the MMV ceiling is 16 (the
/// small and chunked ceilings stay 1), so the list
/// carries the MMV rung exactly where the band table
/// admits the batch and nothing else. With
/// `WGPU_COMPUTE_MMV_MAX_ROWS=1` every ceiling is
/// closed and the list is empty at every batch: the
/// GEMV form serves, byte-identical to the pre-ladder
/// draft.
#[allow(clippy::too_many_arguments)]
fn proj_candidates(
    bands: &MmvBands,
    mmv_ceiling: u32,
    small_max: u32,
    chunked_max: u32,
    family: MmvFamily,
    in_features: usize,
    out_rows: usize,
    batch: u32,
) -> Vec<ProjForm> {
    let mut forms = Vec::new();
    if bands.routes(
        mmv_ceiling,
        family,
        mmv_shape_of(in_features, out_rows),
        batch,
    ) {
        forms.push(ProjForm::Mmv);
    }
    if batch >= 2 && batch <= small_max {
        forms.push(ProjForm::Small);
    }
    if batch >= 2 && batch <= chunked_max && (batch as usize) < GEMM_MIN_ROWS {
        forms.push(ProjForm::Chunked);
    }
    forms
}

/// Try the MMV rung for one projection: the family's
/// register-streaming product over the f32 input rows
/// themselves, no quantize in front of it. `None` when
/// the prepare refuses the shape (logged by name): the
/// ladder continues. Applicability is the caller's
/// decision ([`proj_candidates`]); this is the prepare
/// half only, mirroring the resident ladder's
/// `try_mmv_*` bodies.
fn try_mmv_draft(
    backend: &WgpuCompute,
    weights: &ProjWeights,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    batch: u32,
) -> Option<DevCall> {
    let ctx = backend.context();
    match weights {
        ProjWeights::Q4K(w) => match backend.k19().prepare(ctx, w, input, output, batch) {
            Ok(c) => Some(DevCall::Mmv1(c)),
            Err(e) => {
                dbg_log!("mmv q4k refused at the product prepare: {e}; the ladder continues");
                None
            }
        },
        ProjWeights::Q6K(w) => match backend.k20().prepare(ctx, w, input, output, batch) {
            Ok(c) => Some(DevCall::Mmv2(c)),
            Err(e) => {
                dbg_log!("mmv q6k refused at the product prepare: {e}; the ladder continues");
                None
            }
        },
        ProjWeights::Q5K(w) => match backend.k21().prepare(ctx, w, input, output, batch) {
            Ok(c) => Some(DevCall::Mmv3(c)),
            Err(e) => {
                dbg_log!("mmv q5k refused at the product prepare: {e}; the ladder continues");
                None
            }
        },
    }
}

/// Try the K18 rung for one projection: the K12 plain
/// quantize of `input` feeding the family's K18
/// product. `None` when the device built no K18
/// pipeline for the family or a prepare refuses
/// (logged by name): the ladder continues.
fn try_small_draft(
    backend: &WgpuCompute,
    weights: &ProjWeights,
    input: &wgpu::Buffer,
    q8_qs: &wgpu::Buffer,
    q8_ds: &wgpu::Buffer,
    output: &wgpu::Buffer,
    batch: u32,
) -> Option<DevCall> {
    let ctx = backend.context();
    let (family, in_features, _) = proj_dims(weights);
    let quant = match backend
        .k12()
        .prepare(ctx, input, q8_qs, q8_ds, batch, in_features)
    {
        Ok(q) => q,
        Err(e) => {
            dbg_log!(
                "small {} refused at the quantize prepare: {e}; the ladder continues",
                family.name()
            );
            return None;
        }
    };
    let product = match weights {
        ProjWeights::Q4K(w) => backend
            .k18_q4k()?
            .prepare(ctx, w, q8_qs, q8_ds, output, batch),
        ProjWeights::Q6K(w) => backend
            .k18_q6k()?
            .prepare(ctx, w, q8_qs, q8_ds, output, batch),
        ProjWeights::Q5K(w) => backend
            .k18_q5k()?
            .prepare(ctx, w, q8_qs, q8_ds, output, batch),
    };
    match product {
        Ok(mmq) => Some(DevCall::Small { quant, mmq }),
        Err(e) => {
            dbg_log!(
                "small {} refused at the product prepare: {e}; the ladder continues",
                family.name()
            );
            None
        }
    }
}

/// Try the chunked rung for one projection: the batch
/// served as one windowed GEMV dispatch per
/// power-of-two chunk ([`pow2_chunks`], the resident
/// ladder's own decomposition). `None` when a window
/// prepare refuses (logged by name): the ladder
/// continues to the whole-batch GEMV form.
fn try_chunked_draft(
    backend: &WgpuCompute,
    weights: &ProjWeights,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    batch: u32,
) -> Option<DevCall> {
    let ctx = backend.context();
    match weights {
        ProjWeights::Q4K(w) => {
            let mut calls = Vec::new();
            for (start, len) in pow2_chunks(batch) {
                match backend.k1().prepare_rows(ctx, w, input, output, start, len) {
                    Ok(c) => calls.push(c),
                    Err(e) => {
                        dbg_log!("chunked q4k refused at the window prepare: {e}; the gemv form serves it");
                        return None;
                    }
                }
            }
            Some(DevCall::Chunked1(calls))
        }
        ProjWeights::Q6K(w) => {
            let mut calls = Vec::new();
            for (start, len) in pow2_chunks(batch) {
                match backend.k2().prepare_rows(ctx, w, input, output, start, len) {
                    Ok(c) => calls.push(c),
                    Err(e) => {
                        dbg_log!("chunked q6k refused at the window prepare: {e}; the gemv form serves it");
                        return None;
                    }
                }
            }
            Some(DevCall::Chunked2(calls))
        }
        ProjWeights::Q5K(w) => {
            let mut calls = Vec::new();
            for (start, len) in pow2_chunks(batch) {
                match backend.k3().prepare_rows(ctx, w, input, output, start, len) {
                    Ok(c) => calls.push(c),
                    Err(e) => {
                        dbg_log!("chunked q5k refused at the window prepare: {e}; the gemv form serves it");
                        return None;
                    }
                }
            }
            Some(DevCall::Chunked3(calls))
        }
    }
}

/// Prepare one projection dispatch: the draft walks
/// the same ladder the resident session walks below
/// its matrix threshold. The register-streaming MMV
/// products serve first where the band routing table
/// admits the batch (the global
/// `WGPU_COMPUTE_MMV_MAX_ROWS` ceiling narrowed by the
/// per-(family, shape) cells of measured wins), the
/// K18 small-batch products behind them where the
/// small-batch ceiling admits it, the chunked GEMV
/// form behind those, and the plain K1/K2/K3 GEMV
/// last. The matrix forms are the one resident rung
/// the draft does not take: its row counts are block
/// and segment counts, and the MMQ forms' activation
/// quantization is a prefill tool (the module docs
/// carry the reasoning). Every form consumes the same
/// resident buffers, so the choice never changes the
/// chain's data layout, and a rung's prepare refusal
/// falls through to the next rung by name, never a
/// silent substitution inside a dispatch.
#[allow(clippy::too_many_arguments)]
fn prepare_proj(
    backend: &WgpuCompute,
    weights: &ProjWeights,
    input: &wgpu::Buffer,
    q8_qs: &wgpu::Buffer,
    q8_ds: &wgpu::Buffer,
    output: &wgpu::Buffer,
    rows: usize,
    step: &'static str,
) -> Result<DevCall> {
    let ctx = backend.context();
    let batch = rows as u32;
    let (family, in_features, out_rows) = proj_dims(weights);
    let candidates = proj_candidates(
        backend.mmv_bands(),
        backend.mmv_max_rows(),
        backend.small_batch_max_rows(),
        backend.chunked_gemv_max_rows(),
        family,
        in_features,
        out_rows,
        batch,
    );
    if !candidates.contains(&ProjForm::Mmv) && batch >= 2 && batch <= backend.mmv_max_rows() {
        // The ceiling admits the batch but the band
        // cell excludes it: the mid-band cliff the
        // table exists to remove, logged by name like
        // the resident ladder logs it, so an A/B can
        // see the fall-throughs it is made of.
        let shape = mmv_shape_of(in_features, out_rows);
        dbg_log!(
            "mmv {} {} excludes batch {batch} (in {in_features}, out {out_rows}); \
             the ladder continues",
            family.name(),
            shape.name()
        );
    }
    for form in candidates {
        let call = match form {
            ProjForm::Mmv => try_mmv_draft(backend, weights, input, output, batch),
            ProjForm::Small => {
                try_small_draft(backend, weights, input, q8_qs, q8_ds, output, batch)
            }
            ProjForm::Chunked => try_chunked_draft(backend, weights, input, output, batch),
        };
        if let Some(call) = call {
            return Ok(call);
        }
    }
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

/// The readback prefix a capped block forward resolves:
/// `row_cap` rows of the `rows`-row block when the cap is
/// a true prefix (`1..=rows`), else no resolution and the
/// caller refuses by name. A pure function so the cap
/// contract is testable without a device.
fn capped_rows(rows: usize, row_cap: usize) -> Option<usize> {
    (1..=rows).contains(&row_cap).then_some(row_cap)
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
    /// K12's quantized activations for the projection
    /// ladder's K18 rung: the packed int8 words,
    /// `[rows][blocks][8]` u32, blocks counted at the
    /// widest projection input a plan quantizes (the
    /// conditioning width, hidden, attention-out, or
    /// FFN width). One pair serves every projection:
    /// dispatches inside a pass are totally ordered,
    /// each quantize immediately precedes its product,
    /// and no two quantized forms are ever live at once
    /// (the resident scratch's own discipline).
    q8_qs: wgpu::Buffer,
    /// K12's quantized activation scales: one
    /// `vec4<f32>` per block, beside [`Scratch::q8_qs`].
    q8_ds: wgpu::Buffer,
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
        // Q8_1 blocks at the widest width any projection
        // in either plan quantizes (see the field docs),
        // rows at the largest plan either pass builds.
        let q8_rows = rows.max(INJECT_CHUNK_ROWS);
        let q8_blocks = conditioning_width
            .max(hidden)
            .max(cfg.attn_out_width() as usize)
            .max(cfg.ffn_hidden as usize)
            .div_ceil(32);
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
            q8_qs: buf("dflash q8 qs", q8_rows * q8_blocks * 8),
            q8_ds: buf("dflash q8 ds", q8_rows * q8_blocks * 4),
        })
    }
}

/// A prepared drafting chain for one block row count,
/// cached across calls (the resident session's plan
/// discipline). Every layer's attention call carries its
/// own params buffer, so the runtime params (block start,
/// committed count) are rewritten per call on each of
/// them through its params buffer.
#[derive(Debug)]
struct BlockPlan {
    calls: Vec<DevCall>,
    /// Indices of the attention calls in `calls`, one
    /// per layer, in layer order.
    attends: Vec<usize>,
    /// Block rows this plan serves.
    rows: usize,
    /// Vocabulary width of the logits readback.
    vocab: usize,
}

/// The readback half of a block encode: the buffer the
/// chain's logits land in, the staging they are copied to,
/// and how many of the block's rows the host takes.
struct BlockReadback<'a> {
    logits: &'a wgpu::Buffer,
    staging: &'a wgpu::Buffer,
    rows: usize,
}

impl BlockPlan {
    /// Encode the block chain, then copy the first
    /// `readback.rows` logit rows into the readback staging.
    /// The chain always computes every row (block attention
    /// is non-causal; a shorter block is a different draft);
    /// `readback.rows` sizes only the host transfer.
    fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &crate::device::GpuContext,
        block_start: usize,
        committed: usize,
        readback: BlockReadback<'_>,
    ) {
        for &attend in &self.attends {
            if let DevCall::Attend(call) = &self.calls[attend] {
                call.set_runtime(ctx, block_start as u32, committed as u32);
            }
        }
        // The caller refuses a cap outside `1..=rows` by
        // name before encoding; this assert is the seam's
        // own invariant, since a copy longer than the plan
        // would read past the logits the chain wrote.
        debug_assert!(
            readback.rows >= 1 && readback.rows <= self.rows,
            "readback rows {} outside the plan's 1..={} rows",
            readback.rows,
            self.rows
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dflash block chain"),
                timestamp_writes: None,
            });
            for call in &self.calls {
                call.encode_into(&mut pass);
            }
        }
        encoder.copy_buffer_to_buffer(
            readback.logits,
            0,
            readback.staging,
            0,
            (readback.rows * self.vocab * 4) as u64,
        );
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
    target_hidden: u32,
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
    /// The GPU-timing accumulator (crate::timestamps),
    /// present only when the backend's context carries a
    /// timestamp recorder; distinct from the `diag` event
    /// sink above. `None` is the off path, exactly as in
    /// [`crate::ResidentSession`].
    diag_acc: Option<DiagAccumulator>,
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
                .k17()
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
        // Capture level: stamp the run manifest now, while
        // the resolved configuration and the draft's shape
        // are all still in hand, mirroring
        // [`crate::ResidentSession`]. A no-op below capture
        // (begin_manifest itself is capture-gated), so the
        // timing and off paths construct exactly as before.
        let mut diag_acc = backend
            .context()
            .diag_recorder()
            .map(|_| DiagAccumulator::from_env());
        if let Some(acc) = &mut diag_acc {
            acc.begin_manifest(Self::capture_manifest(
                backend,
                cfg,
                weights.conditioning_width(),
                weights.target_hidden,
                session_cfg.kv_capacity,
            ));
        }
        Ok(DflashSession {
            backend,
            cfg: cfg.clone(),
            conditioning_width: weights.conditioning_width(),
            target_hidden: weights.target_hidden,
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
            diag_acc,
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

    /// The target hidden size derived from `fc` at load:
    /// each position's conditioning row is
    /// `target_layers.len()` rows of this width,
    /// concatenated (the layout capture staging assembles).
    pub fn target_hidden(&self) -> u32 {
        self.target_hidden
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

    /// Drain the diag timestamp recorder into the session's
    /// accumulator, mirroring [`crate::ResidentSession`]:
    /// called after every submitted chunk of the inject,
    /// fuse, and block-forward paths. A no-op off the timing
    /// path.
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

    /// The session section of the capture manifest, as
    /// ordered (name, value) pairs: the draft's own shape.
    /// The draft model's depth is its layer count
    /// (`block_count`); the speculative depth is the spec
    /// session's config, not this session's, and is not
    /// stated here. Pure, so the wiring is unit-testable
    /// without a device.
    fn capture_session_pairs(
        cfg: &DflashConfig,
        conditioning_width: usize,
        target_hidden: u32,
        kv_capacity: usize,
    ) -> Vec<(String, String)> {
        let pair = |name: &str, value: String| (name.to_string(), value);
        vec![
            pair("kind", "dflash".to_string()),
            pair("depth", cfg.block_count.to_string()),
            pair("block_size", cfg.block_size.to_string()),
            pair("hidden", cfg.hidden.to_string()),
            pair("vocab", cfg.vocab_size.to_string()),
            pair("context_length", cfg.context_length.to_string()),
            pair("target_hidden", target_hidden.to_string()),
            pair("conditioning_width", conditioning_width.to_string()),
            pair("kv_capacity", kv_capacity.to_string()),
        ]
    }

    /// The run manifest a draft session stamps at the
    /// capture diag level, mirroring
    /// [`crate::ResidentSession`]'s: build identity from
    /// this crate, adapter and driver identity from the
    /// backend, the resolved configuration the backend
    /// actually runs with (resolved values, never raw env
    /// text), and the draft's own shape
    /// ([`Self::capture_session_pairs`]). The model section
    /// is left empty here; the accumulator fills it from a
    /// noted or registered model file
    /// ([`DiagAccumulator::begin_manifest`]).
    fn capture_manifest(
        backend: &WgpuCompute,
        cfg: &DflashConfig,
        conditioning_width: usize,
        target_hidden: u32,
        kv_capacity: usize,
    ) -> RunManifest {
        let facts = &backend.context().adapter;
        let pair = |name: &str, value: String| (name.to_string(), value);
        RunManifest {
            diag_mode: DiagMode::Capture.name().to_string(),
            build_version: build_version().to_string(),
            git: build_git().to_string(),
            adapter_name: facts.name.clone(),
            adapter_backend: format!("{:?}", facts.backend),
            adapter_device_type: format!("{:?}", facts.device_type),
            adapter_driver: facts.driver.clone(),
            adapter_driver_info: facts.driver_info.clone(),
            resolved_config: vec![
                pair(
                    "small_batch_max_rows",
                    backend.small_batch_max_rows().to_string(),
                ),
                pair("mmv_max_rows", backend.mmv_max_rows().to_string()),
                pair("mmv_bands", backend.mmv_bands_describe()),
                pair(
                    "mmv_profile",
                    backend.mmv_profile_key().unwrap_or("none").to_string(),
                ),
                pair(
                    "chunked_gemv_max_rows",
                    backend.chunked_gemv_max_rows().to_string(),
                ),
                pair("k22_mmq", backend.k22_mmq().to_string()),
                pair("mmq_wide_min_rows", backend.wide_min_rows().to_string()),
                pair("k21f_head", backend.k21f_head().to_string()),
                pair("mmq_splitk", backend.mmq_splitk().describe()),
            ],
            model: None,
            session: Self::capture_session_pairs(
                cfg,
                conditioning_width,
                target_hidden,
                kv_capacity,
            ),
        }
    }

    /// Record a completed block forward's read-back logits
    /// with the capture accumulator. One branch off the
    /// capture level, like [`Self::drain_diag`], and the
    /// hash runs over exactly the rows the forward already
    /// read back (the whole block for
    /// [`Self::forward_block`], the returned prefix for
    /// [`Self::forward_block_capped`]): capture adds no GPU
    /// work and no readback.
    fn capture_block(&mut self, logits: &[f32]) {
        if let Some(acc) = &mut self.diag_acc {
            acc.record_forward(CaptureRecord::KIND_BLOCK, logits);
        }
    }

    /// Fingerprint the model file at `path` into this
    /// session's capture manifest (its path, size, and
    /// content fingerprint under
    /// [`crate::timestamps::MODEL_FINGERPRINT_METHOD`]),
    /// mirroring [`crate::ResidentSession::note_model_file`].
    /// With no accumulator (diag off) this is an `Ok(())`
    /// no-op.
    ///
    /// # Errors
    ///
    /// The `io` error when the file cannot be opened,
    /// stated, or read.
    pub fn note_model_file(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        match &mut self.diag_acc {
            Some(acc) => acc.note_model_file(path),
            None => Ok(()),
        }
    }

    /// Record one speculative round with the capture
    /// accumulator, mirroring
    /// [`crate::ResidentSession::note_spec_round`]: the
    /// draft's proposed token ids (fingerprinted) and how
    /// many of them the target accepted. Returns the record
    /// when one was made; off the capture level it records
    /// nothing.
    pub fn note_spec_round(&mut self, proposed: &[u32], accepted: usize) -> Option<CaptureRecord> {
        self.diag_acc
            .as_mut()?
            .record_spec_round(proposed, accepted)
    }

    /// The capture records this session has recorded so far,
    /// in order (empty off the capture level).
    #[must_use]
    pub fn diag_capture_records(&self) -> &[CaptureRecord] {
        self.diag_acc
            .as_ref()
            .map_or(&[], DiagAccumulator::capture_records)
    }

    /// This session's capture manifest rendered as JSON,
    /// when one was begun (capture level only).
    #[must_use]
    pub fn diag_manifest_json(&self) -> Option<String> {
        self.diag_acc
            .as_ref()
            .and_then(DiagAccumulator::manifest_json)
    }

    /// Finish the diag report now: one last drain, then the
    /// tally line and the summary file (when
    /// `WGPU_COMPUTE_DIAG_OUT` named one). Idempotent; the
    /// session's drop calls it for callers that never do.
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

impl Drop for DflashSession<'_> {
    fn drop(&mut self) {
        // End of run: finish the diag report the caller did
        // not finish explicitly. A no-op off the timing path
        // and after an explicit finish.
        self.finish_diag_report();
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
        let mut attends: Vec<usize> = Vec::new();
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
            // K17 attention over committed + block, output
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
                &s.q8_qs,
                &s.q8_ds,
                &s.q,
                rows,
                STEP_BLOCK,
            )?);
            calls.push(prepare_proj(
                backend,
                &layer.attn_k,
                &s.h,
                &s.q8_qs,
                &s.q8_ds,
                &s.k,
                rows,
                STEP_BLOCK,
            )?);
            calls.push(prepare_proj(
                backend,
                &layer.attn_v,
                &s.h,
                &s.q8_qs,
                &s.q8_ds,
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
                    .k17()
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
                    .k17()
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
            attends.push(calls.len());
            let sliding = if cfg.is_sliding(li) == Some(true) {
                Some(cfg.sliding_window)
            } else {
                None
            };
            calls.push(DevCall::Attend(
                backend
                    .k17()
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
                &s.q8_qs,
                &s.q8_ds,
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
                &s.q8_qs,
                &s.q8_ds,
                &s.gate,
                rows,
                STEP_BLOCK,
            )?);
            calls.push(prepare_proj(
                backend,
                &layer.ffn_up,
                &s.h,
                &s.q8_qs,
                &s.q8_ds,
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
                &s.q8_qs,
                &s.q8_ds,
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
            backend, &self.head, &s.h, &s.q8_qs, &s.q8_ds, &s.logits, rows, STEP_BLOCK,
        )?);
        Ok(BlockPlan {
            calls,
            attends,
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
            prepare_proj(
                backend,
                &self.fc,
                &s.features,
                &s.q8_qs,
                &s.q8_ds,
                &s.fused,
                rows,
                STEP_INJECT,
            )?,
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
                prepare_proj(
                    backend,
                    &layer.attn_k,
                    &s.fused_n,
                    &s.q8_qs,
                    &s.q8_ds,
                    &s.ik,
                    rows,
                    STEP_INJECT,
                )?,
                prepare_proj(
                    backend,
                    &layer.attn_v,
                    &s.fused_n,
                    &s.q8_qs,
                    &s.q8_ds,
                    &s.iv,
                    rows,
                    STEP_INJECT,
                )?,
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
                        .k17()
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
            self.drain_diag();
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
            self.drain_diag();
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
        self.run_block(tokens, start_position, None)
    }

    /// Run one block through the draft and read back only
    /// the first `row_cap` rows of its logits: the prefix
    /// variant of [`Self::forward_block`] for a caller that
    /// consumes a bounded prefix of the block (the
    /// speculative loop slices at most `depth + 1` rows:
    /// the anchor's row plus one per draft it may propose).
    ///
    /// The device still runs the full block: block
    /// attention is non-causal, so a truncated block would
    /// change the earlier rows' distributions. Only the
    /// staging copy and the host map shrink, so the
    /// returned rows are the full forward's first `row_cap`
    /// rows, bit for bit, in the same row-major layout.
    ///
    /// # Errors
    ///
    /// [`DflashError::StepFailed`] under the
    /// `dflash forward_block` step when `row_cap` is not a
    /// true prefix of the block (`1..=tokens.len()`); the
    /// CPU pass's errors, in its order, otherwise.
    pub fn forward_block_capped(
        &mut self,
        tokens: &[u32],
        start_position: usize,
        row_cap: usize,
    ) -> Result<Vec<f32>> {
        self.run_block(tokens, start_position, Some(row_cap))
    }

    /// The block pass behind both public forwards: the
    /// CPU pass's validations in its order, the one
    /// submission, and the readback of the resolved prefix
    /// (the whole block when uncapped).
    fn run_block(
        &mut self,
        tokens: &[u32],
        start_position: usize,
        row_cap: Option<usize>,
    ) -> Result<Vec<f32>> {
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
        let rows = tokens.len();
        let read_rows = match row_cap {
            None => rows,
            Some(cap) => capped_rows(rows, cap).ok_or_else(|| DflashError::StepFailed {
                step: STEP_BLOCK,
                detail: format!(
                    "row cap {cap} is not a prefix of the {rows}-row block; \
                     the cap sizes the readback only, never the block"
                ),
            })?,
        };
        self.diag.event(
            "forward_block.begin",
            &format!(
                "tokens={} start={} cached={}",
                tokens.len(),
                start_position,
                self.held.len()
            ),
        );
        if read_rows < rows {
            self.diag.event(
                "forward_block.capped",
                &format!("rows={rows} returned={read_rows}"),
            );
        }
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
            BlockReadback {
                logits: &self.scratch.logits,
                staging: &self.scratch.staging,
                rows: read_rows,
            },
        );
        ctx.queue.submit([encoder.finish()]);
        self.stats.submissions += 1;
        let logits = ctx
            .map_staging(&self.scratch.staging, read_rows * vocab)
            .map_err(|e| we(STEP_BLOCK, e))?;
        self.stats.blocks += 1;
        self.stats.readbacks += 1;
        self.stats.logits_values_read += logits.len() as u64;
        self.drain_diag();
        self.capture_block(&logits);
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
    fn capped_rows_accepts_only_true_prefixes() {
        assert_eq!(capped_rows(16, 1), Some(1));
        assert_eq!(capped_rows(16, 2), Some(2));
        assert_eq!(capped_rows(16, 16), Some(16));
        assert_eq!(capped_rows(16, 0), None);
        assert_eq!(capped_rows(16, 17), None);
        assert_eq!(capped_rows(0, 1), None);
        assert_eq!(capped_rows(1, 1), Some(1));
    }

    #[test]
    fn session_config_default_is_the_documented_capacity() {
        assert_eq!(DflashSessionConfig::default().kv_capacity, 8192);
    }

    /// A minimal draft config for the capture wiring pins:
    /// three layers deep, block size 4, the fixture's shape
    /// discipline in miniature.
    fn capture_test_cfg() -> DflashConfig {
        DflashConfig {
            name: None,
            block_count: 3,
            context_length: 128,
            hidden: 256,
            ffn_hidden: 512,
            q_heads: 4,
            kv_heads: 2,
            key_length: 64,
            value_length: 64,
            rope_freq_base: 10_000.0,
            rms_eps: 1e-5,
            block_size: 4,
            target_layers: vec![0, 2],
            sliding_window: 3,
            sliding_pattern: vec![true, false, true],
            causal_attention: false,
            mask_token_id: 511,
            vocab_size: 512,
        }
    }

    /// The manifest's session section states the draft's
    /// own shape: kind, depth (the layer count), and block
    /// size first, then the target shape the draft serves.
    #[test]
    fn capture_session_pairs_state_the_draft_shape() {
        let pairs = DflashSession::capture_session_pairs(&capture_test_cfg(), 512, 256, 64);
        let expected: Vec<(String, String)> = [
            ("kind", "dflash"),
            ("depth", "3"),
            ("block_size", "4"),
            ("hidden", "256"),
            ("vocab", "512"),
            ("context_length", "128"),
            ("target_hidden", "256"),
            ("conditioning_width", "512"),
            ("kv_capacity", "64"),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        assert_eq!(pairs, expected);
    }

    /// The exact calls the constructor and the block exit
    /// make, driven against a timing-mode accumulator (the
    /// resident path's pin, restated for the draft
    /// session): no manifest begins, the block record
    /// returns nothing, no capture rows exist, and the
    /// finish writes the timing report with NO capture
    /// sidecars beside it.
    #[test]
    fn capture_wiring_is_inert_in_timing_mode() {
        let dir = std::env::temp_dir().join(format!(
            "wgpu-compute-dflash-capture-timing-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let out = dir.join("timing.csv");
        let manifest = RunManifest {
            session: DflashSession::capture_session_pairs(&capture_test_cfg(), 512, 256, 64),
            ..RunManifest::default()
        };
        let mut acc = DiagAccumulator::with_mode(Some(out.clone()), DiagMode::Timing);
        acc.begin_manifest(manifest);
        assert!(acc.manifest_json().is_none(), "timing begins no manifest");
        assert!(acc
            .record_forward(CaptureRecord::KIND_BLOCK, &[1.0, 2.0])
            .is_none());
        assert!(acc.record_spec_round(&[7, 8], 1).is_none());
        assert!(acc.capture_records().is_empty());
        assert!(acc.write_report(0));
        assert!(!crate::timestamps::manifest_sidecar_path(&out).exists());
        assert!(!crate::timestamps::capture_sidecar_path(&out).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The same calls at capture level: the manifest
    /// begins with the draft's pairs inside it, and the
    /// block forward's logits land as one `block` record.
    #[test]
    fn capture_wiring_records_the_block_in_capture_mode() {
        let dir = std::env::temp_dir().join(format!(
            "wgpu-compute-dflash-capture-capture-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let out = dir.join("capture.csv");
        let manifest = RunManifest {
            session: DflashSession::capture_session_pairs(&capture_test_cfg(), 512, 256, 64),
            ..RunManifest::default()
        };
        let mut acc = DiagAccumulator::with_mode(Some(out.clone()), DiagMode::Capture);
        acc.begin_manifest(manifest);
        let json = acc.manifest_json().expect("capture begins a manifest");
        assert!(json.contains("\"kind\":\"dflash\""), "manifest: {json}");
        assert!(json.contains("\"depth\":\"3\""), "manifest: {json}");
        assert!(json.contains("\"block_size\":\"4\""), "manifest: {json}");
        let record = acc
            .record_forward(CaptureRecord::KIND_BLOCK, &[0.0, 1.0, 0.5])
            .expect("capture records the block");
        assert_eq!(record.kind, CaptureRecord::KIND_BLOCK);
        assert_eq!(record.values, 3);
        assert_eq!(acc.capture_record_count(), 1);
        assert!(acc.write_report(0));
        let capture_text = std::fs::read_to_string(crate::timestamps::capture_sidecar_path(&out))
            .expect("capture sidecar written");
        let parsed =
            crate::timestamps::parse_capture_csv(&capture_text).expect("capture sidecar parses");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].kind, CaptureRecord::KIND_BLOCK);
        assert!(crate::timestamps::manifest_sidecar_path(&out).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The draft ladder's resolution, pinned against
    /// the shipped band table at the draft's own
    /// geometries. The Glimmer draft is hidden-6656
    /// like its target, so the shape representatives
    /// are the sweep's: attention (6656 out), FFN
    /// (19968 out), KV (256 out), and the head
    /// (202048 out, FFN class by relative geometry).
    /// `cands` resolves with the shipped bands and the
    /// given ceilings in [`proj_candidates`] order
    /// (mmv ceiling, small ceiling, chunked ceiling).
    fn cands(
        ceilings: (u32, u32, u32),
        family: MmvFamily,
        in_features: usize,
        out_rows: usize,
        batch: u32,
    ) -> Vec<ProjForm> {
        proj_candidates(
            &crate::compute::default_mmv_bands(),
            ceilings.0,
            ceilings.1,
            ceilings.2,
            family,
            in_features,
            out_rows,
            batch,
        )
    }

    const DRAFT_ATTN: (usize, usize) = (6656, 6656);
    const DRAFT_FFN: (usize, usize) = (6656, 19968);
    const DRAFT_KV: (usize, usize) = (6656, 256);
    const DRAFT_HEAD: (usize, usize) = (6656, 202048);

    #[test]
    fn draft_ladder_ships_open_at_default_ceilings() {
        // The shipped defaults since the 2026-11-06
        // flip (bench/baseline-session-2026-11-06.md
        // leg (d)): the MMV ceiling resolves to 16,
        // the small and chunked ceilings stay 1. At
        // those ceilings the ladder carries exactly the
        // band table's MMV admissions and nothing else.
        // The ceilings are read from the constants, so
        // this pins the shipped resolution itself, not
        // a copy of it.
        let defaults = (
            crate::compute::DEFAULT_MMV_MAX_ROWS,
            crate::compute::DEFAULT_SMALL_BATCH_MAX_ROWS,
            crate::compute::DEFAULT_CHUNKED_GEMV_MAX_ROWS,
        );
        assert_eq!(defaults, (16, 1, 1));
        // The draft head at the block batch routes to
        // K21 through the Q5_K FFN cell: the route the
        // baseline session's leg (e) priced at 10 ms a
        // round of draft time.
        assert_eq!(
            cands(defaults, MmvFamily::Q5K, DRAFT_HEAD.0, DRAFT_HEAD.1, 16),
            vec![ProjForm::Mmv]
        );
        // Measured in-band admissions across families.
        assert_eq!(
            cands(defaults, MmvFamily::Q4K, DRAFT_ATTN.0, DRAFT_ATTN.1, 7),
            vec![ProjForm::Mmv]
        );
        assert_eq!(
            cands(defaults, MmvFamily::Q6K, DRAFT_KV.0, DRAFT_KV.1, 12),
            vec![ProjForm::Mmv]
        );
        assert_eq!(
            cands(defaults, MmvFamily::Q5K, DRAFT_KV.0, DRAFT_KV.1, 16),
            vec![ProjForm::Mmv]
        );
        // The table's exclusions stay exclusions at the
        // shipped default: Q4_K and Q6_K admit nothing
        // at the block batch, the Q4_K FFN hole at
        // row 3 stays a hole, the Q5_K KV cell admits
        // only row 16, and single rows and past-band
        // batches stay the GEMV's.
        for family in [MmvFamily::Q4K, MmvFamily::Q6K] {
            for (inf, out) in [DRAFT_ATTN, DRAFT_FFN, DRAFT_KV] {
                assert_eq!(
                    cands(defaults, family, inf, out, 16),
                    Vec::<ProjForm>::new(),
                    "{family:?} in {inf} out {out} batch 16"
                );
            }
        }
        assert_eq!(
            cands(defaults, MmvFamily::Q4K, DRAFT_FFN.0, DRAFT_FFN.1, 3),
            Vec::<ProjForm>::new()
        );
        assert_eq!(
            cands(defaults, MmvFamily::Q5K, DRAFT_KV.0, DRAFT_KV.1, 12),
            Vec::<ProjForm>::new()
        );
        for family in [MmvFamily::Q4K, MmvFamily::Q5K, MmvFamily::Q6K] {
            for (inf, out) in [DRAFT_ATTN, DRAFT_FFN, DRAFT_KV, DRAFT_HEAD] {
                for batch in [1, 64] {
                    assert_eq!(
                        cands(defaults, family, inf, out, batch),
                        Vec::<ProjForm>::new(),
                        "{family:?} in {inf} out {out} batch {batch}"
                    );
                }
            }
        }
        // The explicit close still closes everything:
        // with every ceiling at 1 no rung applies at
        // any batch, so the GEMV form serves exactly as
        // the pre-ladder draft served.
        for family in [MmvFamily::Q4K, MmvFamily::Q5K, MmvFamily::Q6K] {
            for (inf, out) in [DRAFT_ATTN, DRAFT_FFN, DRAFT_KV, DRAFT_HEAD] {
                for batch in [2, 3, 7, 12, 16, 64] {
                    assert_eq!(
                        cands((1, 1, 1), family, inf, out, batch),
                        Vec::<ProjForm>::new(),
                        "{family:?} in {inf} out {out} batch {batch}"
                    );
                }
            }
        }
    }

    #[test]
    fn draft_head_routes_to_mmv_at_block_batch() {
        // The cost map's fattest dispatch: the Q5_K
        // head at the block's 16 rows, ceiling open.
        // The Q5_K cells admit 16 in every shape class
        // (attn 2-16, ffn 3-16, kv exactly 16), so the
        // head takes K21.
        assert_eq!(
            cands((16, 1, 1), MmvFamily::Q5K, DRAFT_HEAD.0, DRAFT_HEAD.1, 16),
            vec![ProjForm::Mmv]
        );
        assert_eq!(
            cands((16, 1, 1), MmvFamily::Q5K, DRAFT_ATTN.0, DRAFT_ATTN.1, 16),
            vec![ProjForm::Mmv]
        );
        assert_eq!(
            cands((16, 1, 1), MmvFamily::Q5K, DRAFT_KV.0, DRAFT_KV.1, 16),
            vec![ProjForm::Mmv]
        );
        // The Q5_K KV cell is exactly row 16: at 12
        // nothing routes.
        assert_eq!(
            cands((16, 1, 1), MmvFamily::Q5K, DRAFT_KV.0, DRAFT_KV.1, 12),
            Vec::<ProjForm>::new()
        );
        // The global gate still rules: ceiling 1
        // closes the head like everything else.
        assert_eq!(
            cands((1, 1, 1), MmvFamily::Q5K, DRAFT_HEAD.0, DRAFT_HEAD.1, 16),
            Vec::<ProjForm>::new()
        );
    }

    #[test]
    fn draft_block_batch_leaves_q4k_q6k_on_the_lower_ladder() {
        // At the block's 16 rows the shipped Q4_K and
        // Q6_K cells admit nothing (their measured
        // cells end at 12; the sweep never measured
        // MMV at 16, where MMQ served both arms), so
        // with the lower ceilings also closed the GEMV
        // form serves, and the exclusion is policy, not
        // a kernel limit (the kernels serve 16).
        for family in [MmvFamily::Q4K, MmvFamily::Q6K] {
            for (inf, out) in [DRAFT_ATTN, DRAFT_FFN, DRAFT_KV] {
                assert_eq!(
                    cands((16, 1, 1), family, inf, out, 16),
                    Vec::<ProjForm>::new(),
                    "{family:?} in {inf} out {out}"
                );
            }
        }
        // Inside the measured band they route: Q4_K
        // attention at 7, Q6_K attention and KV at 12.
        assert_eq!(
            cands((16, 1, 1), MmvFamily::Q4K, DRAFT_ATTN.0, DRAFT_ATTN.1, 7),
            vec![ProjForm::Mmv]
        );
        assert_eq!(
            cands((16, 1, 1), MmvFamily::Q6K, DRAFT_ATTN.0, DRAFT_ATTN.1, 12),
            vec![ProjForm::Mmv]
        );
        assert_eq!(
            cands((16, 1, 1), MmvFamily::Q6K, DRAFT_KV.0, DRAFT_KV.1, 12),
            vec![ProjForm::Mmv]
        );
        // The Q4_K FFN cell has a measured hole at
        // row 3 ("2,4-12"): 3 falls through, 4 routes.
        assert_eq!(
            cands((16, 1, 1), MmvFamily::Q4K, DRAFT_FFN.0, DRAFT_FFN.1, 3),
            Vec::<ProjForm>::new()
        );
        assert_eq!(
            cands((16, 1, 1), MmvFamily::Q4K, DRAFT_FFN.0, DRAFT_FFN.1, 4),
            vec![ProjForm::Mmv]
        );
    }

    #[test]
    fn draft_ladder_orders_rungs_and_bounds_the_lower_ones() {
        // All ceilings open: the order is MMV, then
        // K18, then chunked, and a batch the MMV cell
        // excludes still reaches the lower rungs.
        assert_eq!(
            cands((16, 16, 15), MmvFamily::Q5K, DRAFT_ATTN.0, DRAFT_ATTN.1, 7),
            vec![ProjForm::Mmv, ProjForm::Small, ProjForm::Chunked]
        );
        assert_eq!(
            cands((16, 16, 15), MmvFamily::Q4K, DRAFT_KV.0, DRAFT_KV.1, 12),
            vec![ProjForm::Small, ProjForm::Chunked]
        );
        // The chunked rung keeps the resident bound:
        // at 16 rows the matrix threshold owns the
        // ladder, so even a wide-open chunked ceiling
        // admits nothing there.
        assert_eq!(
            cands((16, 1, 64), MmvFamily::Q4K, DRAFT_KV.0, DRAFT_KV.1, 16),
            Vec::<ProjForm>::new()
        );
        // Single rows are the GEMV's at every setting.
        for family in [MmvFamily::Q4K, MmvFamily::Q5K, MmvFamily::Q6K] {
            assert_eq!(
                cands((16, 16, 64), family, DRAFT_ATTN.0, DRAFT_ATTN.1, 1),
                Vec::<ProjForm>::new()
            );
        }
    }
}
