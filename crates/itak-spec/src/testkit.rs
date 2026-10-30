//! Synthetic fixtures for this crate's tests: a small random-weight
//! Glimmer target (4 layers, hidden 32, vocab 128, window 16) and a small
//! random-weight DFlash draft with compatible geometry (hidden 32, the
//! same width as the target, exactly as the real draft file declares;
//! target layers [1, 3] of the target, conditioning width 64), wired the
//! real files wire: the draft borrows the target's embedding table and
//! head. Plus a sabotage compute that fails one attention call on command,
//! for the verify-refusal paths. Test code only.

use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use cpu_compute::CpuCompute;
use dflash::{
    DflashConfig, DflashModel, TensorData, TensorSource as DraftSource, Weights as DraftWeights,
};
use muse_glimmer::{
    Cache, Dtype, GlimmerConfig, GlimmerError, GlimmerModel, LayerKind, PatternSource,
    PositionCache, TensorSource as TargetSource, TensorView, Weights as TargetWeights,
};

use crate::diag::NullDiag;
use crate::sampler;
use crate::session::SpecSession;
use crate::speculator::{DflashSpeculator, SharedCompute};
use crate::SpecConfig;

/// Target hidden size in the fixtures.
pub const TARGET_HIDDEN: usize = 32;
/// Target vocabulary in the fixtures.
pub const VOCAB: usize = 128;
/// Draft mask token in the fixtures.
pub const MASK_TOKEN: u32 = 126;

/// Deterministic pseudo-random values in [-0.25, 0.25] (the same LCG the
/// cpu-compute forward tests use, so fixtures are stable everywhere).
pub struct Lcg(pub u64);

impl Lcg {
    /// Next pseudo-random value.
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let v = ((self.0 >> 40) & 0xFF_FFFF) as f32 / (1u64 << 24) as f32;
        (v - 0.5) * 0.5
    }
}

fn rand_vec(rng: &mut Lcg, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.next_f32()).collect()
}

fn ones(n: usize) -> Vec<f32> {
    vec![1.0; n]
}

// ---------------------------------------------------------------- target

/// The synthetic target's configuration.
pub fn target_config() -> GlimmerConfig {
    GlimmerConfig {
        name: Some("synthetic spec target".to_string()),
        block_count: 4,
        context_length: 256,
        hidden: TARGET_HIDDEN as u32,
        ffn_hidden: 64,
        q_heads: 4,
        kv_heads: 2,
        key_length: 8,
        value_length: 8,
        rope_freq_base: 10_000.0,
        rope_freq_base_swa: 10_000.0,
        rms_eps: 1e-6,
        sliding_window: 16,
        logit_softcap: 20.0,
        logit_scale: 0.5,
        vocab_size: VOCAB as u32,
        layers: vec![
            LayerKind::SlidingWindow,
            LayerKind::Global,
            LayerKind::SlidingWindow,
            LayerKind::Global,
        ],
        pattern_source: PatternSource::MeasuredFallback,
    }
}

struct TargetMapSource {
    tensors: HashMap<String, TensorView>,
}

impl TargetSource for TargetMapSource {
    fn tensor(&self, name: &str) -> muse_glimmer::Result<TensorView> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| GlimmerError::MissingTensor {
                name: name.to_string(),
            })
    }
}

fn f32_tensor(name: &str, shape: &[u64], values: Vec<f32>) -> TensorView {
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    TensorView {
        name: name.to_string(),
        dtype: Dtype::F32,
        shape: shape.to_vec(),
        data: Arc::from(data.into_boxed_slice()),
    }
}

/// A built target plus the borrowed tensors the draft speculator needs,
/// in `dflash::TensorData` form (the same values the target loaded).
pub struct TargetFixture {
    /// The target model, forwarding through the given compute.
    pub model: GlimmerModel,
    /// The target's embedding table, token-major rows.
    pub embedding: TensorData,
    /// The target's LM head, token-major rows.
    pub output: TensorData,
}

/// Build the synthetic target over `compute`.
pub fn build_target(compute: Box<dyn muse_glimmer::Compute>) -> TargetFixture {
    let cfg = target_config();
    let hidden = cfg.hidden as usize;
    let ffn = cfg.ffn_hidden as usize;
    let qw = cfg.q_width() as usize;
    let kw = cfg.k_width() as usize;
    let vw = cfg.v_width() as usize;
    let vocab = cfg.vocab_size as usize;
    let head_dim = cfg.key_length as usize;
    let mut rng = Lcg(0x5eed_0001);
    let mut tensors: HashMap<String, TensorView> = HashMap::new();
    let mut put = |name: String, shape: &[u64], values: Vec<f32>| {
        tensors.insert(name.clone(), f32_tensor(&name, shape, values));
    };
    let emb_values = rand_vec(&mut rng, vocab * hidden);
    put(
        "token_embd.weight".to_string(),
        &[hidden as u64, vocab as u64],
        emb_values.clone(),
    );
    put("output_norm.weight".to_string(), &[hidden as u64], ones(hidden));
    let head_values = rand_vec(&mut rng, vocab * hidden);
    put(
        "output.weight".to_string(),
        &[hidden as u64, vocab as u64],
        head_values.clone(),
    );
    for layer in 0..cfg.block_count as usize {
        let name = |suffix: &str| format!("blk.{layer}.{suffix}");
        put(name("attn_norm.weight"), &[hidden as u64], ones(hidden));
        put(name("attn_q.weight"), &[hidden as u64, qw as u64], rand_vec(&mut rng, hidden * qw));
        put(name("attn_k.weight"), &[hidden as u64, kw as u64], rand_vec(&mut rng, hidden * kw));
        put(name("attn_v.weight"), &[hidden as u64, vw as u64], rand_vec(&mut rng, hidden * vw));
        put(name("attn_q_norm.weight"), &[head_dim as u64], ones(head_dim));
        put(name("attn_k_norm.weight"), &[head_dim as u64], ones(head_dim));
        put(name("attn_gate.weight"), &[hidden as u64, qw as u64], rand_vec(&mut rng, hidden * qw));
        put(name("attn_output.weight"), &[qw as u64, hidden as u64], rand_vec(&mut rng, qw * hidden));
        put(name("post_attention_norm.weight"), &[hidden as u64], ones(hidden));
        put(name("ffn_norm.weight"), &[hidden as u64], ones(hidden));
        put(name("ffn_gate.weight"), &[hidden as u64, ffn as u64], rand_vec(&mut rng, hidden * ffn));
        put(name("ffn_up.weight"), &[hidden as u64, ffn as u64], rand_vec(&mut rng, hidden * ffn));
        put(name("ffn_down.weight"), &[ffn as u64, hidden as u64], rand_vec(&mut rng, ffn * hidden));
        put(name("post_ffw_norm.weight"), &[hidden as u64], ones(hidden));
    }
    let source = TargetMapSource { tensors };
    let weights = TargetWeights::load(&source, &cfg).expect("synthetic target weights load");
    TargetFixture {
        model: GlimmerModel::new(cfg, weights, compute),
        embedding: TensorData {
            name: "token_embd.weight".to_string(),
            shape: vec![hidden as u64, vocab as u64],
            data: emb_values,
        },
        output: TensorData {
            name: "output.weight".to_string(),
            shape: vec![hidden as u64, vocab as u64],
            data: head_values,
        },
    }
}

/// The plain greedy reference: prompt through a fresh target, then
/// single-token argmax steps. This is what speculative output must equal
/// at temperature 0.
pub fn plain_greedy(prompt: &[u32], count: usize) -> Vec<u32> {
    let fixture = build_target(Box::new(CpuCompute::new()));
    let model = fixture.model;
    let mut cache = PositionCache::new();
    let mut logits = model.forward(prompt, &mut cache).expect("plain prefill");
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let token = sampler::argmax(&logits);
        out.push(token);
        if i + 1 < count {
            logits = model.forward(&[token], &mut cache).expect("plain step");
        }
    }
    out
}

// ----------------------------------------------------------------- draft

/// The synthetic draft's configuration at a given block size.
pub fn draft_config(block_size: u32) -> DflashConfig {
    DflashConfig {
        name: Some("synthetic spec draft".to_string()),
        block_count: 2,
        context_length: 256,
        hidden: 32,
        ffn_hidden: 64,
        q_heads: 2,
        kv_heads: 1,
        key_length: 16,
        value_length: 16,
        rope_freq_base: 10_000.0,
        rms_eps: 1e-5,
        block_size,
        target_layers: vec![1, 3],
        sliding_window: 16,
        sliding_pattern: vec![true, false],
        causal_attention: false,
        mask_token_id: MASK_TOKEN,
        vocab_size: VOCAB as u32,
    }
}

struct DraftMapSource {
    tensors: HashMap<String, TensorData>,
}

impl DraftSource for DraftMapSource {
    fn tensor(&self, name: &str) -> dflash::Result<TensorData> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| dflash::DflashError::MissingTensor {
                name: name.to_string(),
            })
    }

    fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

/// Build the synthetic draft speculator borrowing `fixture`'s tensors.
pub fn build_speculator(fixture: &TargetFixture, block_size: u32) -> DflashSpeculator {
    let cfg = draft_config(block_size);
    let hidden = cfg.hidden as usize;
    let ffn = cfg.ffn_hidden as usize;
    let qw = cfg.q_width() as usize;
    let kw = cfg.k_width() as usize;
    let vw = cfg.v_width() as usize;
    let out_w = cfg.attn_out_width() as usize;
    let head_dim = cfg.key_length as usize;
    let cond_in = cfg.target_layers.len() * TARGET_HIDDEN;
    let mut rng = Lcg(0x5eed_0002);
    let mut tensors: HashMap<String, TensorData> = HashMap::new();
    let mut put = |name: String, shape: &[u64], values: Vec<f32>| {
        tensors.insert(
            name.clone(),
            TensorData {
                name,
                shape: shape.to_vec(),
                data: values,
            },
        );
    };
    put("fc.weight".to_string(), &[cond_in as u64, hidden as u64], rand_vec(&mut rng, cond_in * hidden));
    put("enc.output_norm.weight".to_string(), &[hidden as u64], ones(hidden));
    put("output_norm.weight".to_string(), &[hidden as u64], ones(hidden));
    for layer in 0..cfg.block_count as usize {
        let name = |suffix: &str| format!("blk.{layer}.{suffix}");
        put(name("attn_norm.weight"), &[hidden as u64], ones(hidden));
        put(name("attn_q.weight"), &[hidden as u64, qw as u64], rand_vec(&mut rng, hidden * qw));
        put(name("attn_k.weight"), &[hidden as u64, kw as u64], rand_vec(&mut rng, hidden * kw));
        put(name("attn_v.weight"), &[hidden as u64, vw as u64], rand_vec(&mut rng, hidden * vw));
        put(name("attn_q_norm.weight"), &[head_dim as u64], ones(head_dim));
        put(name("attn_k_norm.weight"), &[head_dim as u64], ones(head_dim));
        put(name("attn_output.weight"), &[out_w as u64, hidden as u64], rand_vec(&mut rng, out_w * hidden));
        put(name("ffn_norm.weight"), &[hidden as u64], ones(hidden));
        put(name("ffn_gate.weight"), &[hidden as u64, ffn as u64], rand_vec(&mut rng, hidden * ffn));
        put(name("ffn_up.weight"), &[hidden as u64, ffn as u64], rand_vec(&mut rng, hidden * ffn));
        put(name("ffn_down.weight"), &[ffn as u64, hidden as u64], rand_vec(&mut rng, ffn * hidden));
    }
    let source = DraftMapSource { tensors };
    let weights = DraftWeights::load(&source, &cfg).expect("synthetic draft weights load");
    let model = DflashModel::new(cfg, weights);
    DflashSpeculator::new(model, fixture.embedding.clone(), fixture.output.clone())
}

// --------------------------------------------------------------- session

/// A session over the standard fixtures at the given draft block size.
pub fn build_session(config: SpecConfig, block_size: u32) -> SpecSession {
    let compute = Rc::new(CpuCompute::new());
    let fixture = build_target(Box::new(SharedCompute(compute.clone())));
    let speculator = build_speculator(&fixture, block_size);
    SpecSession::new(
        fixture.model,
        compute,
        Some(speculator),
        config,
        Arc::new(NullDiag),
    )
    .expect("fixture session builds")
}

/// A compute that delegates to a shared [`CpuCompute`] but, while armed,
/// fails exactly the next attention call and then disarms itself. Used to
/// force a one-shot verify refusal mid-round.
pub struct SabotageCompute {
    inner: Rc<CpuCompute>,
    armed: Rc<Cell<bool>>,
}

impl muse_glimmer::Compute for SabotageCompute {
    fn embed(
        &self,
        table: &muse_glimmer::TensorView,
        tokens: &[u32],
        cfg: &GlimmerConfig,
    ) -> muse_glimmer::Result<Vec<f32>> {
        self.inner.embed(table, tokens, cfg)
    }
    fn rms_norm(
        &self,
        x: &[f32],
        weight: &muse_glimmer::TensorView,
        rows: usize,
        eps: f32,
    ) -> muse_glimmer::Result<Vec<f32>> {
        self.inner.rms_norm(x, weight, rows, eps)
    }
    fn rms_norm_plain(&self, x: &[f32], rows: usize, eps: f32) -> muse_glimmer::Result<Vec<f32>> {
        self.inner.rms_norm_plain(x, rows, eps)
    }
    fn proj(
        &self,
        x: &[f32],
        weight: &muse_glimmer::TensorView,
        rows: usize,
    ) -> muse_glimmer::Result<Vec<f32>> {
        self.inner.proj(x, weight, rows)
    }
    fn attention(
        &self,
        req: &muse_glimmer::AttentionRequest,
        cache: &mut dyn Cache,
    ) -> muse_glimmer::Result<Vec<f32>> {
        if self.armed.get() {
            self.armed.set(false);
            return Err(GlimmerError::StepFailed {
                step: "attention",
                detail: "sabotaged attention for the refusal test".to_string(),
            });
        }
        self.inner.attention(req, cache)
    }
    fn attn_gate(&self, attn_out: &[f32], gate: &[f32]) -> muse_glimmer::Result<Vec<f32>> {
        self.inner.attn_gate(attn_out, gate)
    }
    fn add(&self, a: &[f32], b: &[f32]) -> muse_glimmer::Result<Vec<f32>> {
        self.inner.add(a, b)
    }
    fn swiglu_ffn(
        &self,
        x: &[f32],
        w: &muse_glimmer::LayerWeights,
        rows: usize,
        cfg: &GlimmerConfig,
    ) -> muse_glimmer::Result<Vec<f32>> {
        self.inner.swiglu_ffn(x, w, rows, cfg)
    }
    fn lm_head(
        &self,
        x: &[f32],
        head: &muse_glimmer::TensorView,
        cfg: &GlimmerConfig,
    ) -> muse_glimmer::Result<Vec<f32>> {
        self.inner.lm_head(x, head, cfg)
    }
}

/// A session whose target compute can be armed to refuse one attention
/// call. Returns the session and the arming switch.
pub fn build_sabotage_session(config: SpecConfig) -> (SpecSession, Rc<Cell<bool>>) {
    let compute = Rc::new(CpuCompute::new());
    let armed = Rc::new(Cell::new(false));
    let sabotage = SabotageCompute {
        inner: compute.clone(),
        armed: armed.clone(),
    };
    let fixture = build_target(Box::new(sabotage));
    let speculator = build_speculator(&fixture, 5);
    let session = SpecSession::new(
        fixture.model,
        compute,
        Some(speculator),
        config,
        Arc::new(NullDiag),
    )
    .expect("sabotage fixture session builds");
    (session, armed)
}
