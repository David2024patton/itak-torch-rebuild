//! [`WgpuCompute`]: the wgpu backend behind `muse_glimmer::Compute`.
//!
//! What runs on the device: `proj` (and through it `lm_head`)
//! for Q4_K, Q6_K, and Q5_K tensors, through kernels K1, K2,
//! and K3 respectively, selected by the tensor's dtype; the
//! norms and the residual add through K4; `swiglu_ffn` end to
//! end (K1 gate and up, K7 activation, K2 down) with
//! `attn_gate` through K7; and the embedding gather through
//! K8. The one step that still refuses, by name, is
//! `attention`: the K5 (RoPE plus KV append) and K6 (scores
//! plus reduce) pipelines are built, but their composition
//! over this backend (device KV residency mirroring
//! `cpu-compute`'s backend-owned storage, behind the `Cache`
//! trait's position contract) is not wired yet. Nothing is
//! silently substituted: BACKEND.md Section 6 puts per-op CPU
//! fallback at load time as the caller's disclosed composition
//! decision, so this backend never detours a step to the CPU
//! behind the caller's back.
//!
//! Data path, stated plainly: weight tensors upload once and are
//! cached by name (per-tensor device buffers, BACKEND.md
//! Section 5); each `proj` call uploads its input rows and reads
//! its output rows back, because the trait's currency is host
//! slices. When the forward itself becomes device-resident, the
//! per-call transfers are what go away; the kernels and the cache
//! stay.

use std::cell::RefCell;
use std::collections::HashMap;

use muse_glimmer::{
    AttentionRequest, Cache, Compute, Dtype, GlimmerConfig, GlimmerError, LayerKind, LayerWeights,
    Result, TensorView,
};

use crate::detect::{detect, DetectPolicy};
use crate::device::GpuContext;
use crate::kernels::k10_q4k_gemm::K10Pipeline;
use crate::kernels::k11_q6k_gemm::K11Pipeline;
use crate::kernels::k1_q4k_gemv::{K1Pipeline, K1Weights, ACT_GATED, ACT_NONE, ACT_SWIGLU};
use crate::kernels::k2_q6k_gemv::{K2Pipeline, K2Weights};
use crate::kernels::k3_q5k_gemv::{K3Pipeline, K3Weights};
use crate::kernels::k4_rmsnorm::K4Pipeline;
use crate::kernels::k5_rope_kv::K5Pipeline;
use crate::kernels::k6_attention::{AttendGeom, GpuKvState, K6Pipeline};
use crate::kernels::k7_swiglu::K7Pipeline;
use crate::kernels::k8_head::K8Pipeline;
use crate::kernels::k9_attn_prep::K9Pipeline;

use crate::diag::dbg_log;

/// Build a [`GlimmerError::StepFailed`] for `step` with `detail`.
fn step_failed(step: &'static str, detail: impl Into<String>) -> GlimmerError {
    GlimmerError::StepFailed {
        step,
        detail: detail.into(),
    }
}

/// The mainline ggml type id of a [`Dtype`], for naming types in
/// refusals (the table is pinned in `muse-glimmer`'s weights
/// module; `Bf16` is 30).
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

/// A weight tensor resident on the device, in whichever kernel's
/// layout its dtype dictates. The cache keys these by tensor
/// name, and the name is an identity: a later tensor arriving
/// under a cached name with a different dtype or geometry is a
/// refusal, not a re-upload.
#[derive(Debug)]
enum CachedWeights {
    /// Q4_K, through kernel K1.
    Q4K(K1Weights),
    /// Q6_K, through kernel K2.
    Q6K(K2Weights),
    /// Q5_K, through kernel K3.
    Q5K(K3Weights),
}

impl CachedWeights {
    /// The (input width, output rows, raw byte length) the
    /// cached upload was made from.
    fn geometry(&self) -> (usize, usize, u64) {
        match self {
            CachedWeights::Q4K(w) => (w.in_features, w.out_rows, w.byte_len),
            CachedWeights::Q6K(w) => (w.in_features, w.out_rows, w.byte_len),
            CachedWeights::Q5K(w) => (w.in_features, w.out_rows, w.byte_len),
        }
    }

    /// The kernel name this entry's dtype runs through, for
    /// refusal messages.
    fn kernel(&self) -> &'static str {
        match self {
            CachedWeights::Q4K(_) => "K1 (Q4_K)",
            CachedWeights::Q6K(_) => "K2 (Q6_K)",
            CachedWeights::Q5K(_) => "K3 (Q5_K)",
        }
    }
}

/// The wgpu [`Compute`] backend. Constructing one selects an
/// adapter and opens a device (see [`crate::detect`]), so
/// construction is the fallible step; a constructed backend has a
/// live device and every kernel pipeline compiled. Stateful in
/// exactly one way, like `cpu-compute`: it owns the KV cache of
/// the one sequence it is serving (device buffers plus a host
/// mirror, in [`GpuKvState`]). Construct one per sequence, and
/// call [`WgpuCompute::reset`] to abandon a sequence and start
/// over.
#[derive(Debug)]
pub struct WgpuCompute {
    ctx: GpuContext,
    k1: K1Pipeline,
    k1_gated: K1Pipeline,
    k1_swiglu: K1Pipeline,
    k2: K2Pipeline,
    k2_swiglu: K2Pipeline,
    k3: K3Pipeline,
    k4: K4Pipeline,
    k5: K5Pipeline,
    k6: K6Pipeline,
    k7: K7Pipeline,
    k8: K8Pipeline,
    k9: K9Pipeline,
    k10: Option<K10Pipeline>,
    k10_gated: Option<K10Pipeline>,
    k10_swiglu: Option<K10Pipeline>,
    k11: Option<K11Pipeline>,
    k11_swiglu: Option<K11Pipeline>,
    weights: RefCell<HashMap<String, CachedWeights>>,
    kv: RefCell<Option<GpuKvState>>,
}

impl WgpuCompute {
    /// Open a backend under the default selection policy (discrete
    /// Vulkan or refusal).
    ///
    /// # Errors
    ///
    /// [`crate::error::WgpuError::NoSuitableAdapter`] when no adapter qualifies,
    /// plus the device and pipeline failures of
    /// [`WgpuCompute::with_policy`].
    pub fn new() -> crate::error::Result<Self> {
        Self::with_policy(&DetectPolicy::default())
    }

    /// Open a backend under an explicit selection policy.
    ///
    /// # Errors
    ///
    /// [`crate::error::WgpuError::NoSuitableAdapter`], [`crate::error::WgpuError::DeviceRequest`],
    /// [`crate::error::WgpuError::LimitShortfall`], or [`crate::error::WgpuError::Pipeline`]
    /// when device opening or pipeline compilation fails.
    pub fn with_policy(policy: &DetectPolicy) -> crate::error::Result<Self> {
        Self::from_context(detect(policy)?)
    }

    /// Build a backend over an already-opened context.
    ///
    /// # Errors
    ///
    /// [`crate::error::WgpuError::Pipeline`] when any kernel
    /// shader fails to compile.
    pub fn from_context(ctx: GpuContext) -> crate::error::Result<Self> {
        let k1 = K1Pipeline::new(&ctx.device)?;
        let k1_gated =
            K1Pipeline::with_shape_act(&ctx.device, crate::kernels::k1_q4k_gemv::shipped_shape(ctx.device.features()), ACT_GATED)?;
        let k1_swiglu =
            K1Pipeline::with_shape_act(&ctx.device, crate::kernels::k1_q4k_gemv::shipped_shape(ctx.device.features()), ACT_SWIGLU)?;
        let k2 = K2Pipeline::new(&ctx.device)?;
        let k2_swiglu = K2Pipeline::with_shape_act(
            &ctx.device,
            crate::kernels::k2_q6k_gemv::shipped_shape(ctx.device.features()),
            ACT_SWIGLU,
        )?;
        let k3 = K3Pipeline::new(&ctx.device)?;
        let k4 = K4Pipeline::new(&ctx.device)?;
        let k5 = K5Pipeline::new(&ctx.device)?;
        let k6 = K6Pipeline::new(&ctx.device)?;
        let k7 = K7Pipeline::new(&ctx.device)?;
        let k8 = K8Pipeline::new(&ctx.device)?;
        let k9 = K9Pipeline::new(&ctx.device)?;
        // The prefill GEMMs are the one optional pipeline set:
        // a device granting too little workgroup storage for
        // the smallest staged tile gets `None` here, and the
        // resident session keeps its GEMV chunk plans on it
        // (the K10 module docs own the fallback contract).
        let k10 = K10Pipeline::for_device(&ctx.device, ACT_NONE)?;
        let k10_gated = K10Pipeline::for_device(&ctx.device, ACT_GATED)?;
        let k10_swiglu = K10Pipeline::for_device(&ctx.device, ACT_SWIGLU)?;
        let k11 = K11Pipeline::for_device(&ctx.device, ACT_NONE)?;
        let k11_swiglu = K11Pipeline::for_device(&ctx.device, ACT_SWIGLU)?;
        dbg_log!("backend ready on {}", ctx.adapter.name);
        Ok(WgpuCompute {
            ctx,
            k1,
            k1_gated,
            k1_swiglu,
            k2,
            k2_swiglu,
            k3,
            k4,
            k5,
            k6,
            k7,
            k8,
            k9,
            k10,
            k10_gated,
            k10_swiglu,
            k11,
            k11_swiglu,
            weights: RefCell::new(HashMap::new()),
            kv: RefCell::new(None),
        })
    }

    /// The device context this backend runs on.
    #[must_use]
    pub fn context(&self) -> &GpuContext {
        &self.ctx
    }

    /// The K1 (Q4_K GEMV) pipeline. Crate-internal: the
    /// device-resident session ([`crate::session`]) composes the
    /// same pipelines this backend's trait implementation uses,
    /// so the two paths can never drift onto different kernels.
    pub(crate) fn k1(&self) -> &K1Pipeline {
        &self.k1
    }

    /// The K1 pipeline with the attention-gate activation
    /// fused in (the resident session's output projection).
    /// See [`WgpuCompute::k1`].
    pub(crate) fn k1_gated(&self) -> &K1Pipeline {
        &self.k1_gated
    }

    /// The K1 pipeline with the SwiGLU activation fused in
    /// (the resident session's down projection, when the down
    /// tensor is Q4_K). See [`WgpuCompute::k1`].
    pub(crate) fn k1_swiglu(&self) -> &K1Pipeline {
        &self.k1_swiglu
    }

    /// The K2 (Q6_K GEMV) pipeline. See [`WgpuCompute::k1`].
    pub(crate) fn k2(&self) -> &K2Pipeline {
        &self.k2
    }

    /// The K2 pipeline with the SwiGLU activation fused in
    /// (the resident session's down projection, when the down
    /// tensor is Q6_K). See [`WgpuCompute::k1`].
    pub(crate) fn k2_swiglu(&self) -> &K2Pipeline {
        &self.k2_swiglu
    }

    /// The K3 (Q5_K GEMV) pipeline. See [`WgpuCompute::k1`].
    pub(crate) fn k3(&self) -> &K3Pipeline {
        &self.k3
    }

    /// The K4 (RMSNorm and residual add) pipeline. See
    /// [`WgpuCompute::k1`].
    pub(crate) fn k4(&self) -> &K4Pipeline {
        &self.k4
    }

    /// The K6 (grouped-query attention) pipeline. See
    /// [`WgpuCompute::k1`].
    pub(crate) fn k6(&self) -> &K6Pipeline {
        &self.k6
    }

    /// The K7 (SwiGLU and attention gate) pipeline. See
    /// [`WgpuCompute::k1`].
    pub(crate) fn k7(&self) -> &K7Pipeline {
        &self.k7
    }

    /// The K8 (embedding gather and head finishing) pipeline.
    /// See [`WgpuCompute::k1`].
    pub(crate) fn k8(&self) -> &K8Pipeline {
        &self.k8
    }

    /// The K9 (fused attention preparation) pipeline. Only the
    /// device-resident session prepares its calls; the trait
    /// implementation composes K4 and K5 instead. See
    /// [`WgpuCompute::k1`].
    pub(crate) fn k9(&self) -> &K9Pipeline {
        &self.k9
    }

    /// The K10 (Q4_K GEMM) pipeline, when this device grants
    /// the workgroup storage its smallest tile needs. Only
    /// the device-resident session prepares its calls, for
    /// chunk plans at or past the GEMM row threshold; `None`
    /// is the session's signal to keep the GEMV form. See
    /// [`WgpuCompute::k1`].
    pub(crate) fn k10(&self) -> Option<&K10Pipeline> {
        self.k10.as_ref()
    }

    /// The K10 pipeline with the attention-gate activation
    /// fused in. See [`WgpuCompute::k10`].
    pub(crate) fn k10_gated(&self) -> Option<&K10Pipeline> {
        self.k10_gated.as_ref()
    }

    /// The K10 pipeline with the SwiGLU activation fused in.
    /// See [`WgpuCompute::k10`].
    pub(crate) fn k10_swiglu(&self) -> Option<&K10Pipeline> {
        self.k10_swiglu.as_ref()
    }

    /// The K11 (Q6_K GEMM) pipeline, when this device grants
    /// the workgroup storage. See [`WgpuCompute::k10`].
    pub(crate) fn k11(&self) -> Option<&K11Pipeline> {
        self.k11.as_ref()
    }

    /// The K11 pipeline with the SwiGLU activation fused in.
    /// See [`WgpuCompute::k10`].
    pub(crate) fn k11_swiglu(&self) -> Option<&K11Pipeline> {
        self.k11_swiglu.as_ref()
    }

    /// Drop all stored KV state, device buffers and host mirror.
    /// The next attention call rebuilds the state from its
    /// config and expects the caller's sequence position to be 0
    /// again. Mirrors `CpuCompute::reset`.
    pub fn reset(&self) {
        dbg_log!("gpu kv state reset");
        *self.kv.borrow_mut() = None;
    }

    /// Ensure a device copy of `view`'s tensor exists in the
    /// cache, uploading it through its dtype's kernel on first
    /// use. A later tensor arriving under a cached name with a
    /// different dtype or geometry is a refusal, not a
    /// re-upload: the name is the identity the loader gave it.
    fn ensure_uploaded(&self, view: &TensorView, step: &'static str) -> Result<()> {
        let (inn, out) = (view.shape[0] as usize, view.shape[1] as usize);
        {
            let cache = self.weights.borrow();
            if let Some(existing) = cache.get(&view.name) {
                let (e_in, e_out, e_bytes) = existing.geometry();
                let kind_matches = matches!(
                    (view.dtype, existing),
                    (Dtype::Q4K, CachedWeights::Q4K(_))
                        | (Dtype::Q6K, CachedWeights::Q6K(_))
                        | (Dtype::Q5K, CachedWeights::Q5K(_))
                );
                if !kind_matches || e_in != inn || e_out != out || e_bytes != view.data.len() as u64
                {
                    return Err(step_failed(
                        step,
                        format!(
                            "tensor {} was already uploaded through {} as [{} x {}] ({} bytes) \
                             but now arrives as [{inn} x {out}] ({} bytes) of another kind; \
                             a tensor name is an identity in this backend, not a slot",
                            view.name,
                            existing.kernel(),
                            e_in,
                            e_out,
                            e_bytes,
                            view.data.len()
                        ),
                    ));
                }
                return Ok(());
            }
        }
        let uploaded = match view.dtype {
            Dtype::Q4K => CachedWeights::Q4K(
                self.k1
                    .upload(&self.ctx, &view.name, &view.data, inn, out)
                    .map_err(|e| step_failed(step, e.to_string()))?,
            ),
            Dtype::Q6K => CachedWeights::Q6K(
                self.k2
                    .upload(&self.ctx, &view.name, &view.data, inn, out)
                    .map_err(|e| step_failed(step, e.to_string()))?,
            ),
            Dtype::Q5K => CachedWeights::Q5K(
                self.k3
                    .upload(&self.ctx, &view.name, &view.data, inn, out)
                    .map_err(|e| step_failed(step, e.to_string()))?,
            ),
            other => {
                return Err(step_failed(
                    step,
                    format!(
                        "tensor {} has dtype {} (ggml type id {}), which has no GPU kernel in \
                         this backend yet (K1 covers Q4_K, K2 covers Q6_K, K3 covers Q5_K); \
                         BACKEND.md Section 6 level L2 runs such types on the CPU backend \
                         as a load-time composition decision, not inside this step",
                        view.name,
                        gguf::ggml::type_name(dtype_id(other)),
                        dtype_id(other)
                    ),
                ))
            }
        };
        self.weights
            .borrow_mut()
            .insert(view.name.clone(), uploaded);
        Ok(())
    }

    /// Shared projection body, mirroring `cpu-compute`'s: the FFN
    /// calls it under `swiglu_ffn`, the head under `lm_head`, so
    /// refusals name the step the forward thinks it is in.
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
        let inn = weight.shape[0] as usize;
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
        self.ensure_uploaded(weight, step)?;
        let cache = self.weights.borrow();
        let cached = cache.get(&weight.name).ok_or_else(|| {
            step_failed(
                step,
                format!(
                    "tensor {} was uploaded but is missing from the cache",
                    weight.name
                ),
            )
        })?;
        match (weight.dtype, cached) {
            (Dtype::Q4K, CachedWeights::Q4K(w)) => self
                .k1
                .run(&self.ctx, w, x)
                .map_err(|e| step_failed(step, e.to_string())),
            (Dtype::Q6K, CachedWeights::Q6K(w)) => self
                .k2
                .run(&self.ctx, w, x)
                .map_err(|e| step_failed(step, e.to_string())),
            (Dtype::Q5K, CachedWeights::Q5K(w)) => self
                .k3
                .run(&self.ctx, w, x)
                .map_err(|e| step_failed(step, e.to_string())),
            _ => Err(step_failed(
                step,
                format!(
                    "tensor {} changed kind between upload and run; the cache holds it \
                     through {}, which its dtype does not select",
                    weight.name,
                    cached.kernel()
                ),
            )),
        }
    }
}

impl Compute for WgpuCompute {
    fn embed(&self, table: &TensorView, tokens: &[u32], cfg: &GlimmerConfig) -> Result<Vec<f32>> {
        self.k8.embed(&self.ctx, table, tokens, cfg)
    }

    fn rms_norm(&self, x: &[f32], weight: &TensorView, rows: usize, eps: f32) -> Result<Vec<f32>> {
        let step = "rms_norm";
        // The norm weight is a small F32 vector in the real file;
        // dequantize it here (the same view-to-values step, with
        // the same byte-length validation, that the CPU backend
        // takes in its dequantize_view) and let K4 run the
        // normalization itself on the device.
        let ty = match weight.dtype {
            Dtype::F32 => gguf::GgmlType::F32,
            Dtype::F16 => gguf::GgmlType::F16,
            Dtype::Q8_0 => gguf::GgmlType::Q8_0,
            Dtype::Q4K => gguf::GgmlType::Q4K,
            Dtype::Q5K => gguf::GgmlType::Q5K,
            Dtype::Q6K => gguf::GgmlType::Q6K,
            Dtype::Bf16 => gguf::GgmlType::Bf16,
            other => {
                return Err(step_failed(
                    step,
                    format!(
                        "norm weight {} has dtype {} (ggml type id {}), which this backend \
                         cannot dequantize for K4",
                        weight.name,
                        gguf::ggml::type_name(dtype_id(other)),
                        dtype_id(other)
                    ),
                ))
            }
        };
        let elements = weight.elements() as usize;
        let expected = ty.byte_len(elements as u64).ok_or_else(|| {
            step_failed(
                step,
                format!(
                    "norm weight {} has {elements} elements, not a whole number of blocks",
                    weight.name
                ),
            )
        })?;
        if expected != weight.data.len() as u64 {
            return Err(step_failed(
                step,
                format!(
                    "norm weight {} holds {} bytes but {elements} elements need {expected}",
                    weight.name,
                    weight.data.len()
                ),
            ));
        }
        let w = gguf::dequant::dequantize(ty, &weight.data, elements);
        self.k4
            .rms_norm(&self.ctx, x, &w, rows, eps)
            .map_err(|e| step_failed(step, e.to_string()))
    }

    fn rms_norm_plain(&self, x: &[f32], rows: usize, eps: f32) -> Result<Vec<f32>> {
        self.k4
            .rms_norm_plain(&self.ctx, x, rows, eps)
            .map_err(|e| step_failed("rms_norm_plain", e.to_string()))
    }

    fn proj(&self, x: &[f32], weight: &TensorView, rows: usize) -> Result<Vec<f32>> {
        self.proj_impl(x, weight, rows, "proj")
    }

    fn attention(&self, req: &AttentionRequest<'_>, cache: &mut dyn Cache) -> Result<Vec<f32>> {
        let step = "attention";
        let cfg = req.cfg;
        // The validation mirrors cpu-compute's, the spec: the
        // attention crate stores keys and values at one head
        // dimension, the request's kind must be the config's, and
        // the projections must be whole rows.
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
            *guard = Some(GpuKvState::build(cfg, step)?);
        }
        let state = guard.as_mut().expect("kv state was just built");
        state.check_shape(cfg, step)?;
        let appended = state
            .host()
            .next_position(req.layer)
            .map_err(|e| step_failed(step, e.to_string()))?;
        if appended != start {
            return Err(step_failed(
                step,
                format!(
                    "KV state for layer {} is at position {appended} but the sequence cache is at position {start}: \
                     the two disagree, so attending would read the wrong history. \
                     Pair one Cache with one WgpuCompute per sequence, or call WgpuCompute::reset \
                     when a sequence is abandoned (a forward that failed midway leaves the KV state \
                     ahead of the cache and also needs a reset)",
                    req.layer
                ),
            ));
        }

        let head_dim = cfg.key_length as usize;
        let q_heads = cfg.q_heads as usize;
        let kv_heads = cfg.kv_heads as usize;
        let window = match req.kind {
            LayerKind::Global => None,
            LayerKind::SlidingWindow => Some(cfg.sliding_window as usize),
        };

        // Rotate Q and K in place on the device, sliding layers
        // only, at the SWA base; global layers are NoPE and pass
        // through unrotated (src/models/muse-glimmer.cpp).
        let q_buf = self.ctx.upload_f32("attention q", req.q);
        let k_buf = self.ctx.upload_f32("attention k", req.k);
        if req.kind == LayerKind::SlidingWindow {
            self.k5
                .rotate(
                    &self.ctx,
                    &q_buf,
                    rows,
                    q_heads,
                    head_dim,
                    start,
                    cfg.rope_freq_base_swa,
                )
                .map_err(|e| step_failed(step, e.to_string()))?;
            self.k5
                .rotate(
                    &self.ctx,
                    &k_buf,
                    rows,
                    kv_heads,
                    head_dim,
                    start,
                    cfg.rope_freq_base_swa,
                )
                .map_err(|e| step_failed(step, e.to_string()))?;
        }
        // The rotated keys come back exactly once: the host
        // mirror appends what the device computed, so the mirror
        // (positions, cross-checks, tests) and the device buffers
        // can never drift apart in what they hold.
        let k_rot = self
            .ctx
            .read_f32(&k_buf, req.k.len())
            .map_err(|e| step_failed(step, e.to_string()))?;
        for i in 0..rows {
            state
                .host_mut()
                .append(
                    req.layer,
                    &k_rot[i * k_width..(i + 1) * k_width],
                    &req.v[i * v_width..(i + 1) * v_width],
                    start + i,
                )
                .map_err(|e| step_failed(step, e.to_string()))?;
        }
        let (stored, first_pos) = {
            let view = state
                .host()
                .view(req.layer)
                .map_err(|e| step_failed(step, e.to_string()))?;
            let first = view.position_at(0).ok_or_else(|| {
                step_failed(
                    step,
                    "the layer's cache view is empty right after an append",
                )
            })?;
            (view.len(), first)
        };
        let appended_total = state
            .host()
            .next_position(req.layer)
            .map_err(|e| step_failed(step, e.to_string()))?;
        let layer_kv = state.ensure_layer(&self.ctx, req.layer, k_width, appended_total, step)?;
        // K5's second half: append the rotated K rows and the V
        // rows into the layer's resident buffers, at the slots
        // the cache semantics assign.
        let v_buf = self.ctx.upload_f32("attention v", req.v);
        self.k5
            .append(
                &self.ctx,
                &k_buf,
                &v_buf,
                &layer_kv.k,
                &layer_kv.v,
                rows,
                k_width,
                start,
                window,
            )
            .map_err(|e| step_failed(step, e.to_string()))?;
        // K6: scores plus the online-softmax reduce over the
        // resident buffers.
        let geom = AttendGeom {
            rows,
            q_heads,
            kv_heads,
            head_dim,
            stored,
            first_pos,
            start_pos: start,
            window,
            // The trait state's ring holds exactly one window,
            // so its slot modulus is the window itself.
            ring: window,
        };
        let scores = self
            .k6
            .scores(&self.ctx, &q_buf, layer_kv, &geom)
            .map_err(|e| step_failed(step, e.to_string()))?;
        let out = self
            .k6
            .reduce(&self.ctx, &scores, layer_kv, &geom)
            .map_err(|e| step_failed(step, e.to_string()))?;
        dbg_log!(
            "attention layer={} kind={:?} start={} rows={} stored={}",
            req.layer,
            req.kind,
            start,
            rows,
            stored
        );
        Ok(out)
    }

    fn attn_gate(&self, attn_out: &[f32], gate: &[f32]) -> Result<Vec<f32>> {
        self.k7
            .attn_gate(&self.ctx, attn_out, gate)
            .map_err(|e| step_failed("attn_gate", e.to_string()))
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.k4
            .add(&self.ctx, a, b)
            .map_err(|e| step_failed("add", e.to_string()))
    }

    fn swiglu_ffn(
        &self,
        x: &[f32],
        w: &LayerWeights,
        rows: usize,
        _cfg: &GlimmerConfig,
    ) -> Result<Vec<f32>> {
        let step = "swiglu_ffn";
        // The FFN end to end: gate and up through K1 (Q4_K in the
        // real file), the activation through K7, and the down
        // projection through K2 (Q6_K).
        let gate = self.proj_impl(x, &w.ffn_gate, rows, step)?;
        let up = self.proj_impl(x, &w.ffn_up, rows, step)?;
        let act = self
            .k7
            .swiglu(&self.ctx, &gate, &up)
            .map_err(|e| step_failed(step, e.to_string()))?;
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
        self.proj_impl(x, head, 1, step)
    }
}
