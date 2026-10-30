//! [`WgpuCompute`]: the wgpu backend behind `muse_glimmer::Compute`.
//!
//! What runs today: `proj` for Q4_K tensors, through kernel K1.
//! Q6_K and Q5_K projections refuse through K2's and K3's modules;
//! every other step refuses through its own kernel module (norms
//! through K4, attention through K6, the FFN at its activation
//! through K7, the embedding through K8). Nothing is silently
//! substituted: BACKEND.md Section 6 puts per-op CPU fallback at
//! load time as the caller's disclosed composition decision, so
//! this backend never detours a step to the CPU behind the
//! caller's back.
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
    AttentionRequest, Cache, Compute, Dtype, GlimmerConfig, GlimmerError, LayerWeights, Result,
    TensorView,
};

use crate::detect::{detect, DetectPolicy};
use crate::device::GpuContext;
use crate::kernels::k1_q4k_gemv::{K1Pipeline, K1Weights};
use crate::kernels::{k2_q6k_gemv, k3_q5k_gemv, k4_rmsnorm, k6_attention, k7_swiglu, k8_head};

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

/// The wgpu [`Compute`] backend. Constructing one selects an
/// adapter and opens a device (see [`crate::detect`]), so
/// construction is the fallible step; a constructed backend has a
/// live device and a compiled K1 pipeline.
#[derive(Debug)]
pub struct WgpuCompute {
    ctx: GpuContext,
    k1: K1Pipeline,
    weights: RefCell<HashMap<String, K1Weights>>,
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
    /// when device opening or K1 compilation fails.
    pub fn with_policy(policy: &DetectPolicy) -> crate::error::Result<Self> {
        Self::from_context(detect(policy)?)
    }

    /// Build a backend over an already-opened context.
    ///
    /// # Errors
    ///
    /// [`crate::error::WgpuError::Pipeline`] when the K1 shader fails to compile.
    pub fn from_context(ctx: GpuContext) -> crate::error::Result<Self> {
        let k1 = K1Pipeline::new(&ctx.device)?;
        dbg_log!("backend ready on {}", ctx.adapter.name);
        Ok(WgpuCompute {
            ctx,
            k1,
            weights: RefCell::new(HashMap::new()),
        })
    }

    /// The device context this backend runs on.
    #[must_use]
    pub fn context(&self) -> &GpuContext {
        &self.ctx
    }

    /// Fetch the cached device copy of a Q4_K tensor, uploading it
    /// on first use. A later tensor arriving under a cached name
    /// with different geometry is a refusal, not a re-upload: the
    /// name is the identity the loader gave it.
    fn weights_for(&self, view: &TensorView, step: &'static str) -> Result<K1WeightsRef<'_>> {
        let (inn, out) = (view.shape[0] as usize, view.shape[1] as usize);
        let mut cache = self.weights.borrow_mut();
        if let Some(existing) = cache.get(&view.name) {
            if existing.in_features != inn
                || existing.out_rows != out
                || existing.byte_len != view.data.len() as u64
            {
                return Err(step_failed(
                    step,
                    format!(
                        "tensor {} was already uploaded as [{} x {}] ({} bytes) but now \
                         arrives as [{inn} x {out}] ({} bytes); a tensor name is an \
                         identity in this backend, not a slot",
                        view.name,
                        existing.in_features,
                        existing.out_rows,
                        existing.byte_len,
                        view.data.len()
                    ),
                ));
            }
            // The cache owns the buffer; the caller borrows it
            // through the returned guard.
            return Ok(K1WeightsRef {
                weights: cache,
                name: view.name.clone(),
                _step: step,
            });
        }
        let uploaded = self
            .k1
            .upload(&self.ctx, &view.name, &view.data, inn, out)
            .map_err(|e| step_failed(step, e.to_string()))?;
        let name = view.name.clone();
        cache.insert(view.name.clone(), uploaded);
        Ok(K1WeightsRef {
            weights: cache,
            name,
            _step: step,
        })
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
        match weight.dtype {
            Dtype::Q4K => {
                let guard = self.weights_for(weight, step)?;
                let weights = guard.get();
                self.k1
                    .run(&self.ctx, weights, x)
                    .map_err(|e| step_failed(step, e.to_string()))
            }
            Dtype::Q6K => k2_q6k_gemv::refuse(step),
            Dtype::Q5K => k3_q5k_gemv::refuse(step),
            other => Err(step_failed(
                step,
                format!(
                    "tensor {} has dtype {} (ggml type id {}), which has no GPU kernel in \
                     this backend yet (K1 covers Q4_K); BACKEND.md Section 6 level L2 \
                     runs such types on the CPU backend as a load-time composition \
                     decision, not inside this step",
                    weight.name,
                    gguf::ggml::type_name(dtype_id(other)),
                    dtype_id(other)
                ),
            )),
        }
    }
}

/// A borrow of a cached [`K1Weights`] that keeps the cache's
/// `RefCell` borrow alive for the duration of one projection. The
/// guard exists because `run` needs `&K1Weights` while the cache
/// map owns it.
#[derive(Debug)]
struct K1WeightsRef<'a> {
    weights: std::cell::RefMut<'a, HashMap<String, K1Weights>>,
    name: String,
    _step: &'static str,
}

impl K1WeightsRef<'_> {
    fn get(&self) -> &K1Weights {
        self.weights
            .get(&self.name)
            .expect("the entry was inserted or found above")
    }
}

impl Compute for WgpuCompute {
    fn embed(
        &self,
        _table: &TensorView,
        _tokens: &[u32],
        _cfg: &GlimmerConfig,
    ) -> Result<Vec<f32>> {
        k8_head::refuse("embed")
    }

    fn rms_norm(
        &self,
        _x: &[f32],
        _weight: &TensorView,
        _rows: usize,
        _eps: f32,
    ) -> Result<Vec<f32>> {
        k4_rmsnorm::refuse("rms_norm")
    }

    fn rms_norm_plain(&self, _x: &[f32], _rows: usize, _eps: f32) -> Result<Vec<f32>> {
        k4_rmsnorm::refuse("rms_norm_plain")
    }

    fn proj(&self, x: &[f32], weight: &TensorView, rows: usize) -> Result<Vec<f32>> {
        self.proj_impl(x, weight, rows, "proj")
    }

    fn attention(&self, _req: &AttentionRequest<'_>, _cache: &mut dyn Cache) -> Result<Vec<f32>> {
        k6_attention::refuse("attention")
    }

    fn attn_gate(&self, _attn_out: &[f32], _gate: &[f32]) -> Result<Vec<f32>> {
        Err(step_failed(
            "attn_gate",
            "no GPU kernel for the attention gate elementwise step yet \
             (sigmoid times attention output; it is not one of BACKEND.md \
             Section 3's K1 to K8 kernels and lands with the elementwise family)",
        ))
    }

    fn add(&self, _a: &[f32], _b: &[f32]) -> Result<Vec<f32>> {
        Err(step_failed(
            "add",
            "no standalone GPU kernel for the residual add yet; BACKEND.md \
             Section 3 fuses it into K4 (RMSNorm plus residual), which has \
             not landed",
        ))
    }

    fn swiglu_ffn(
        &self,
        x: &[f32],
        w: &LayerWeights,
        rows: usize,
        _cfg: &GlimmerConfig,
    ) -> Result<Vec<f32>> {
        let step = "swiglu_ffn";
        // Gate and up are Q4_K in the real file and run through K1
        // here; the activation is where this step currently ends.
        let _gate = self.proj_impl(x, &w.ffn_gate, rows, step)?;
        let _up = self.proj_impl(x, &w.ffn_up, rows, step)?;
        k7_swiglu::refuse(step)
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
