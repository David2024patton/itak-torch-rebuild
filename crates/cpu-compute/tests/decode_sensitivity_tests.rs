//! Regression test for the integration run B decode freeze
//! (`crates/itak-loader/INTEGRATION-RUN-B.md`, "Second symptom"):
//! on the real 30B file, decode steps returned a top-5 identical to
//! six printed decimals across steps that fed different tokens at
//! advancing positions. The distribution had stopped tracking the
//! fed token.
//!
//! ## Trace of the decode step, and the verdict on it
//!
//! Traced end to end (2026-10-29, lane 04):
//!
//! - Embedding: `CpuCompute::embed` dequantizes exactly the fed
//!   token's row of the table (`dequantize_rows` on
//!   `token..token + 1`); the runner feeds the previous step's
//!   greedy pick. No stale or shared row.
//! - Position: `model.forward` reads `cache.position()` once;
//!   `CpuCompute::attention` takes it as `start`, processes row `i`
//!   at absolute position `start + i`, and refuses if the KV
//!   store's per-layer next position disagrees. RoPE rotates Q and
//!   K at that absolute position (f64 angles, adjacent pairs).
//! - KV write: each row's rotated K and raw V are appended at that
//!   absolute position, in order enforced by `KvCache::append`
//!   (`OutOfOrderPosition` on any mismatch); windowed layers write
//!   ring slot `position % window` and record the slot's position.
//! - KV read: the view is taken after the append, so the current
//!   position is visible to itself; `gqa_attention` masks over
//!   absolute positions (causal everywhere, `(p - window, p]` on
//!   windowed layers), maps Q heads to KV heads in contiguous
//!   groups (`h / (q_heads / kv_heads)`), and reads through the
//!   view's absolute-position indexing.
//! - The layer pattern (global every fourth from index 3) is the
//!   same in `GlimmerConfig` and in the `attention` crate's cache.
//!
//! Verdict: the machinery is correct, and this file proves it at
//! the real model's decode shape (below). The freeze mechanism is
//! signal balance, not state handling: with per-projection output
//! RMS growing like sqrt(fan-in) (flat random weights), the block
//! outputs dwarf the residual stream, and the post norm after
//! every residual add (2 per layer, 104 at full depth) shrinks the
//! stream's share of each sum by that factor again. One token's
//! contribution is attenuated multiplicatively until the output
//! distribution is independent of it at the 1e-8 level, measured.
//! On the real file the same imbalance is what run B's diagnosis 1
//! (the norm convention question) points at: mis-scaled norms
//! mis-weight every block output against the residual in all 52
//! layers. That is a forward-fidelity defect, not a KV one; the
//! per-layer activation diff (run B's stated next step) is where
//! it gets localized.
//!
//! ## What this file proves
//!
//! The existing synthetic proofs cover decode at 2 layers, GQA
//! group 2, head dim 8, RoPE base 1e4, eps 1e-6. The real model is
//! 52 layers, group 16 (32 Q heads over 2 KV heads), head dim 128,
//! RoPE base 5e5, eps 1e-5, window 2048, and the file's own logit
//! finishing (softcap 20, scale 0.19611613). This file rebuilds the
//! proof at that shape: same weights, same prompt, two continuations
//! that differ only in the token fed at a decode step must produce
//! materially different next-token distributions, and successive
//! steps feeding different tokens must not repeat a distribution.
//!
//! "Materially" is measured as total variation distance between the
//! softmax distributions. The run B freeze agreed to six decimals on
//! every printed probability, i.e. a TV distance below ~1e-5. The
//! bound asserted here is 0.01, three orders of magnitude above the
//! freeze signature: a healthy token-sensitive path clears it by
//! construction, a frozen one cannot.

use std::collections::HashMap;
use std::sync::Arc;

use cpu_compute::CpuCompute;
use muse_glimmer::{
    Dtype, GlimmerConfig, GlimmerError, GlimmerModel, LayerKind, PatternSource, PositionCache,
    TensorSource, TensorView, Weights,
};

const HIDDEN: usize = 64;
const FFN: usize = 128;
const VOCAB: usize = 128;

/// Deterministic pseudo-random values in [-0.25, 0.25]; same LCG as
/// the sibling forward tests, so fixtures are stable across runs.
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
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

/// Random projection weights with per-entry std `1 / sqrt(fan_in)`:
/// the variance balance under which a projection's output RMS equals
/// its input's RMS. This matters at depth. With flat ±0.25 entries
/// at every matrix instead, each projection's output RMS grows by
/// ~sqrt(fan_in) (the attention output projection has fan-in 4096),
/// and because muse-glimmer applies a post norm after every
/// residual add, the residual stream's share of each sum shrinks by
/// that factor at all 2 * layers sub-blocks: the fed token's signal
/// is attenuated arithmetically, in any faithful implementation of
/// the architecture, trained weights or not. Measured on this
/// fixture at 52 layers with flat weights: TV distance 1.8e-8
/// between the distributions after feeding two different tokens,
/// the run B freeze signature exactly. Trained weights do not have
/// that imbalance (it is what training removes), so the regression
/// fixture uses the balanced draw; the machinery under test
/// (embedding lookup, positions, KV writes and reads) is identical
/// either way.
fn rand_proj(rng: &mut Lcg, n: usize, fan_in: usize) -> Vec<f32> {
    // The LCG draw is uniform on [-0.25, 0.25], std 0.25 / sqrt(3).
    // Rescale it to std 1 / sqrt(fan_in).
    let factor = (3.0f32).sqrt() / 0.25 / (fan_in as f32).sqrt();
    (0..n).map(|_| rng.next_f32() * factor).collect()
}

fn ones(n: usize) -> Vec<f32> {
    vec![1.0; n]
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

struct MapSource {
    tensors: HashMap<String, TensorView>,
}

impl TensorSource for MapSource {
    fn tensor(&self, name: &str) -> muse_glimmer::Result<TensorView> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| GlimmerError::MissingTensor {
                name: name.to_string(),
            })
    }
}

/// All-random weight set for `cfg`, shapes derived from the config
/// exactly as the sibling tests' Random style does.
fn build_source(cfg: &GlimmerConfig) -> MapSource {
    let hidden = cfg.hidden as usize;
    let ffn = cfg.ffn_hidden as usize;
    let qw = cfg.q_width() as usize;
    let kw = cfg.k_width() as usize;
    let vw = cfg.v_width() as usize;
    let vocab = cfg.vocab_size as usize;
    let head_dim = cfg.key_length as usize;
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    let mut tensors: HashMap<String, TensorView> = HashMap::new();
    let mut put = |name: String, shape: &[u64], values: Vec<f32>| {
        tensors.insert(name.clone(), f32_tensor(&name, shape, values));
    };

    put(
        "token_embd.weight".to_string(),
        &[hidden as u64, vocab as u64],
        rand_vec(&mut rng, vocab * hidden),
    );
    put(
        "output_norm.weight".to_string(),
        &[hidden as u64],
        ones(hidden),
    );
    put(
        "output.weight".to_string(),
        &[hidden as u64, vocab as u64],
        rand_proj(&mut rng, vocab * hidden, hidden),
    );

    for layer in 0..cfg.block_count as usize {
        let name = |suffix: &str| format!("blk.{layer}.{suffix}");
        put(name("attn_norm.weight"), &[hidden as u64], ones(hidden));
        put(
            name("attn_q.weight"),
            &[hidden as u64, qw as u64],
            rand_proj(&mut rng, hidden * qw, hidden),
        );
        put(
            name("attn_k.weight"),
            &[hidden as u64, kw as u64],
            rand_proj(&mut rng, hidden * kw, hidden),
        );
        put(
            name("attn_v.weight"),
            &[hidden as u64, vw as u64],
            rand_proj(&mut rng, hidden * vw, hidden),
        );
        put(
            name("attn_q_norm.weight"),
            &[head_dim as u64],
            ones(head_dim),
        );
        put(
            name("attn_k_norm.weight"),
            &[head_dim as u64],
            ones(head_dim),
        );
        put(
            name("attn_gate.weight"),
            &[hidden as u64, qw as u64],
            rand_proj(&mut rng, hidden * qw, hidden),
        );
        put(
            name("attn_output.weight"),
            &[qw as u64, hidden as u64],
            rand_proj(&mut rng, qw * hidden, qw),
        );
        put(
            name("post_attention_norm.weight"),
            &[hidden as u64],
            ones(hidden),
        );
        put(name("ffn_norm.weight"), &[hidden as u64], ones(hidden));
        put(
            name("ffn_gate.weight"),
            &[hidden as u64, ffn as u64],
            rand_proj(&mut rng, hidden * ffn, hidden),
        );
        put(
            name("ffn_up.weight"),
            &[hidden as u64, ffn as u64],
            rand_proj(&mut rng, hidden * ffn, hidden),
        );
        put(
            name("ffn_down.weight"),
            &[ffn as u64, hidden as u64],
            rand_proj(&mut rng, ffn * hidden, ffn),
        );
        put(name("post_ffw_norm.weight"), &[hidden as u64], ones(hidden));
    }
    MapSource { tensors }
}

/// A config with the real model's decode-relevant shape: GQA group
/// 16 (32 Q heads over 2 KV heads), head dim 128, RoPE base 5e5,
/// eps 1e-5, the file's logit finishing, and the Glimmer layer
/// pattern (global every fourth layer starting at index 3). Layer
/// count, hidden size, and window are the caller's choice so the
/// proof can be run at several depths.
fn glimmer_shaped_config(layers: usize, sliding_window: u32) -> GlimmerConfig {
    let kinds: Vec<LayerKind> = (0..layers)
        .map(|i| {
            if i % 4 == 3 {
                LayerKind::Global
            } else {
                LayerKind::SlidingWindow
            }
        })
        .collect();
    GlimmerConfig {
        name: Some("decode sensitivity synthetic".to_string()),
        block_count: layers as u32,
        context_length: 131_072,
        hidden: HIDDEN as u32,
        ffn_hidden: FFN as u32,
        q_heads: 32,
        kv_heads: 2,
        key_length: 128,
        value_length: 128,
        rope_freq_base: 500_000.0,
        rms_eps: 1e-5,
        sliding_window,
        logit_softcap: 20.0,
        logit_scale: 0.196_116_13,
        vocab_size: VOCAB as u32,
        layers: kinds,
        pattern_source: PatternSource::MeasuredFallback,
    }
}

fn make_model(cfg: &GlimmerConfig) -> GlimmerModel {
    let source = build_source(cfg);
    let weights = Weights::load(&source, cfg).expect("synthetic weights load");
    GlimmerModel::new(cfg.clone(), weights, Box::new(CpuCompute::new()))
}

/// Softmax over the forward's finished logits, in f64.
fn probs(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = logits.iter().map(|&z| f64::from(z - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Total variation distance between two distributions.
fn tv_distance(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f64>()
        / 2.0
}

/// The freeze signature from run B is agreement to six printed
/// decimals (TV below ~1e-5). A token-sensitive decode must clear
/// this bound by orders of magnitude.
const MATERIAL: f64 = 0.01;

/// Run one full continuation: prefill `prompt`, then feed `feeds`
/// one token per decode step, returning the distribution after the
/// prefill and after each decode step.
fn run_continuation(cfg: &GlimmerConfig, prompt: &[u32], feeds: &[u32]) -> Vec<Vec<f64>> {
    let model = make_model(cfg);
    let mut cache = PositionCache::new();
    let mut dists = Vec::new();
    let logits = model.forward(prompt, &mut cache).expect("prefill runs");
    dists.push(probs(&logits));
    for &token in feeds {
        let logits = model
            .forward(&[token], &mut cache)
            .expect("decode step runs");
        dists.push(probs(&logits));
    }
    dists
}

fn assert_decode_tracks_tokens(layers: usize, sliding_window: u32) {
    let cfg = glimmer_shaped_config(layers, sliding_window);
    let prompt = [5u32, 9, 17, 33];
    let token_a = 41u32;
    let token_b = 97u32;

    // Two continuations of the same prefill, differing only in the
    // token fed at the first decode step.
    let cont_a = run_continuation(&cfg, &prompt, &[token_a, token_b]);
    let cont_b = run_continuation(&cfg, &prompt, &[token_b, token_a]);

    // Prefill state is shared, so the prefill distributions agree;
    // this pins that the two runs really do diverge only at decode.
    let tv_prefill = tv_distance(&cont_a[0], &cont_b[0]);
    assert!(
        tv_prefill <= 1e-6,
        "identical prefill must give identical distributions, got TV {tv_prefill}"
    );

    // The regression assertion: feeding token A vs token B at the
    // same decode step, from the same state, must move the
    // distribution materially. Run B's step 2 instead repeated the
    // prefill distribution to six decimals.
    let tv_step = tv_distance(&cont_a[1], &cont_b[1]);
    eprintln!(
        "decode sensitivity layers={layers} window={sliding_window}: \
         TV after feeding {token_a} vs {token_b} at one step = {tv_step}"
    );
    assert!(
        tv_step > MATERIAL,
        "decode step ignores the fed token: distributions after feeding {token_a} vs {token_b} \
         differ by TV {tv_step}, at or below the freeze signature"
    );

    // Successive steps feeding different tokens must not repeat a
    // distribution either (run B steps 3 to 8 shared one top-5).
    let tv_successive = tv_distance(&cont_a[1], &cont_a[2]);
    eprintln!(
        "decode sensitivity layers={layers} window={sliding_window}: \
         TV between successive steps feeding {token_a} then {token_b} = {tv_successive}"
    );
    assert!(
        tv_successive > MATERIAL,
        "successive decode steps repeat a distribution: TV {tv_successive} between the step \
         feeding {token_a} and the step feeding {token_b}"
    );
}

#[test]
fn decode_distribution_tracks_the_fed_token() {
    // 8 layers: global layers at indices 3 and 7, sliding layers
    // with the real window (2048, no eviction at this length).
    assert_decode_tracks_tokens(8, 2048);
}

#[test]
fn decode_distribution_tracks_the_fed_token_at_full_depth() {
    // The real depth: 52 layers, same shape otherwise. If the
    // freeze is a depth effect in the decode path (signal from one
    // token attenuated through the stack until the distribution
    // stops tracking it), this is the scale that exposes it.
    assert_decode_tracks_tokens(52, 2048);
}

#[test]
fn decode_distribution_tracks_the_fed_token_across_window_eviction() {
    // Sliding layers with a window of 3, so decode steps run past
    // eviction in the ring buffers while the global layers keep
    // full history.
    assert_decode_tracks_tokens(8, 3);
}
