//! End-to-end tests of `CpuCompute` driving `GlimmerModel::forward`
//! on a tiny synthetic model (2 layers, hidden 32, vocab 128) whose
//! weights are set by hand in this file. The analytic tests compute
//! their expected logits with test-local f64 math written from the
//! formulas, sharing no code with the backend, so agreement is a
//! real check and not the implementation grading itself.

use std::collections::HashMap;
use std::sync::Arc;

use cpu_compute::CpuCompute;
use muse_glimmer::{
    Cache, Dtype, GlimmerConfig, GlimmerError, GlimmerModel, LayerKind, PatternSource,
    PositionCache, TensorSource, TensorView, Weights,
};

const HIDDEN: usize = 32;
const FFN: usize = 64;
const VOCAB: usize = 128;
const HEAD_DIM: usize = 8;

/// Deterministic pseudo-random values in [-0.25, 0.25], so the
/// "random" fixtures are identical on every run and every machine.
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

fn ones(n: usize) -> Vec<f32> {
    vec![1.0; n]
}

fn zeros(n: usize) -> Vec<f32> {
    vec![0.0; n]
}

/// Identity matrix as stored GGUF data: shape [n, n], row o is e_o.
fn identity(n: usize) -> Vec<f32> {
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    v
}

fn scaled_identity(n: usize, s: f32) -> Vec<f32> {
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = s;
    }
    v
}

/// Selector matrix: shape [inn, out] with out <= inn, row o is e_o,
/// so the projection keeps the first `out` channels.
fn selector(inn: usize, out: usize) -> Vec<f32> {
    let mut v = vec![0.0; inn * out];
    for o in 0..out {
        v[o * inn + o] = 1.0;
    }
    v
}

/// Output head whose row j is e_{j mod hidden}: logit j reads
/// channel j mod hidden of the normed hidden state.
fn head_repeat(hidden: usize, vocab: usize) -> Vec<f32> {
    let mut v = vec![0.0; hidden * vocab];
    for j in 0..vocab {
        v[j * hidden + (j % hidden)] = 1.0;
    }
    v
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

#[derive(Clone, Copy, PartialEq)]
enum Style {
    /// Attention output and FFN zeroed: the residual stream is only
    /// ever renormalized, so the expected logits are a closed form.
    Analytic,
    /// Identity Q, selector K/V, quarter-identity gate, identity
    /// output projection, zero FFN: attention really mixes positions
    /// and the naive route below can follow it exactly.
    Attention,
    /// Every projection pseudo-random: the full path exercised with
    /// no structure to hide behind.
    Random,
}

fn emb_pattern_a(token: usize, i: usize) -> f32 {
    ((token * 31 + i * 17) % 13) as f32 - 6.0
}

fn emb_pattern_b(token: usize, i: usize) -> f32 {
    ((token * 7 + i * 11) % 9) as f32 - 4.0
}

/// Build the full synthetic weight set for `cfg` in one of the three
/// styles. All shapes are derived from the config, so configs with
/// different head lengths still load.
fn build_source(cfg: &GlimmerConfig, style: Style) -> MapSource {
    let hidden = cfg.hidden as usize;
    let ffn = cfg.ffn_hidden as usize;
    let qw = cfg.q_width() as usize;
    let kw = cfg.k_width() as usize;
    let vw = cfg.v_width() as usize;
    let vocab = cfg.vocab_size as usize;
    let head_dim = cfg.key_length as usize;
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    let mut tensors: HashMap<String, TensorView> = HashMap::new();
    let put = |tensors: &mut HashMap<String, TensorView>,
               name: String,
               shape: &[u64],
               values: Vec<f32>| {
        tensors.insert(name.clone(), f32_tensor(&name, shape, values));
    };

    let emb_values = match style {
        Style::Analytic => (0..vocab * hidden)
            .map(|e| emb_pattern_a(e / hidden, e % hidden))
            .collect(),
        Style::Attention => (0..vocab * hidden)
            .map(|e| emb_pattern_b(e / hidden, e % hidden))
            .collect(),
        Style::Random => rand_vec(&mut rng, vocab * hidden),
    };
    put(
        &mut tensors,
        "token_embd.weight".to_string(),
        &[hidden as u64, vocab as u64],
        emb_values,
    );
    put(
        &mut tensors,
        "output_norm.weight".to_string(),
        &[hidden as u64],
        ones(hidden),
    );
    let head_values = match style {
        Style::Random => rand_vec(&mut rng, vocab * hidden),
        _ => head_repeat(hidden, vocab),
    };
    put(
        &mut tensors,
        "output.weight".to_string(),
        &[hidden as u64, vocab as u64],
        head_values,
    );

    for layer in 0..cfg.block_count as usize {
        let name = |suffix: &str| format!("blk.{layer}.{suffix}");
        put(
            &mut tensors,
            name("attn_norm.weight"),
            &[hidden as u64],
            ones(hidden),
        );
        let q_values = match style {
            Style::Random => rand_vec(&mut rng, hidden * qw),
            _ => identity(hidden),
        };
        put(
            &mut tensors,
            name("attn_q.weight"),
            &[hidden as u64, qw as u64],
            q_values,
        );
        let k_values = match style {
            Style::Random => rand_vec(&mut rng, hidden * kw),
            _ => selector(hidden, kw),
        };
        put(
            &mut tensors,
            name("attn_k.weight"),
            &[hidden as u64, kw as u64],
            k_values,
        );
        let v_values = match style {
            Style::Random => rand_vec(&mut rng, hidden * vw),
            _ => selector(hidden, vw),
        };
        put(
            &mut tensors,
            name("attn_v.weight"),
            &[hidden as u64, vw as u64],
            v_values,
        );
        put(
            &mut tensors,
            name("attn_q_norm.weight"),
            &[head_dim as u64],
            ones(head_dim),
        );
        put(
            &mut tensors,
            name("attn_k_norm.weight"),
            &[head_dim as u64],
            ones(head_dim),
        );
        let gate_values = match style {
            Style::Analytic => rand_vec(&mut rng, hidden * qw),
            Style::Attention => scaled_identity(hidden, 0.25),
            Style::Random => rand_vec(&mut rng, hidden * qw),
        };
        put(
            &mut tensors,
            name("attn_gate.weight"),
            &[hidden as u64, qw as u64],
            gate_values,
        );
        let out_values = match style {
            Style::Analytic => zeros(qw * hidden),
            Style::Attention => identity(hidden),
            Style::Random => rand_vec(&mut rng, qw * hidden),
        };
        put(
            &mut tensors,
            name("attn_output.weight"),
            &[qw as u64, hidden as u64],
            out_values,
        );
        put(
            &mut tensors,
            name("post_attention_norm.weight"),
            &[hidden as u64],
            ones(hidden),
        );
        put(
            &mut tensors,
            name("ffn_norm.weight"),
            &[hidden as u64],
            ones(hidden),
        );
        let ffn_gate_values = match style {
            Style::Attention => zeros(hidden * ffn),
            _ => rand_vec(&mut rng, hidden * ffn),
        };
        put(
            &mut tensors,
            name("ffn_gate.weight"),
            &[hidden as u64, ffn as u64],
            ffn_gate_values,
        );
        let ffn_up_values = match style {
            Style::Attention => zeros(hidden * ffn),
            _ => rand_vec(&mut rng, hidden * ffn),
        };
        put(
            &mut tensors,
            name("ffn_up.weight"),
            &[hidden as u64, ffn as u64],
            ffn_up_values,
        );
        let ffn_down_values = match style {
            Style::Random => rand_vec(&mut rng, ffn * hidden),
            _ => zeros(ffn * hidden),
        };
        put(
            &mut tensors,
            name("ffn_down.weight"),
            &[ffn as u64, hidden as u64],
            ffn_down_values,
        );
        put(
            &mut tensors,
            name("post_ffw_norm.weight"),
            &[hidden as u64],
            ones(hidden),
        );
    }
    MapSource { tensors }
}

fn test_config(layers: Vec<LayerKind>, sliding_window: u32) -> GlimmerConfig {
    test_config_kv(layers, sliding_window, 8, 8)
}

fn test_config_kv(
    layers: Vec<LayerKind>,
    sliding_window: u32,
    key_length: u32,
    value_length: u32,
) -> GlimmerConfig {
    GlimmerConfig {
        name: Some("synthetic test model".to_string()),
        block_count: layers.len() as u32,
        context_length: 128,
        hidden: HIDDEN as u32,
        ffn_hidden: FFN as u32,
        q_heads: 4,
        kv_heads: 2,
        key_length,
        value_length,
        rope_freq_base: 10_000.0,
        rms_eps: 1e-6,
        sliding_window,
        logit_softcap: 20.0,
        logit_scale: 0.5,
        vocab_size: VOCAB as u32,
        layers,
        pattern_source: PatternSource::MeasuredFallback,
    }
}

fn make_model(cfg: &GlimmerConfig, style: Style) -> GlimmerModel {
    let source = build_source(cfg, style);
    let weights = Weights::load(&source, cfg).expect("synthetic weights load");
    GlimmerModel::new(cfg.clone(), weights, Box::new(CpuCompute::new()))
}

/// Test-local RMSNorm (weight 1) in f64, from the formula. Shares no
/// code with the backend or with `torch-math`.
fn rms64(v: &[f64], eps: f64) -> Vec<f64> {
    let ms = v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64;
    let inv = 1.0 / (ms + eps).sqrt();
    v.iter().map(|x| x * inv).collect()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn zero_attention_and_ffn_match_hand_computed_logits() {
    // ASSUMPTION PINNED, one of the two placements muse-glimmer's
    // model docs flag for verification: the post norms are applied
    // AFTER each residual add (the Gemma-style pre/post pattern).
    // With the attention output projection and the FFN down
    // projection zeroed, neither sub-block can change the residual
    // stream, so under that reading each layer maps the stream x to
    // rms(rms(x)): one post-attention norm, one post-FFN norm. The
    // expected logits below are computed by exactly that reading
    // with test-local math. If the post norms ever move ahead of the
    // adds, the count changes and this test fails visibly, which is
    // what pinning an assumption means.
    let cfg = test_config(vec![LayerKind::SlidingWindow, LayerKind::SlidingWindow], 16);
    let model = make_model(&cfg, Style::Analytic);
    let mut cache = PositionCache::new();
    let logits = model
        .forward(&[7, 42, 3], &mut cache)
        .expect("forward runs");
    assert_eq!(cache.position(), 3);
    assert_eq!(logits.len(), VOCAB);

    // Last token is 3; its embedding row passes 2 norms per layer
    // (4 total) plus the final norm, then the repeat-identity head,
    // then the file's softcap and scale (cap 20, scale 0.5).
    let mut x: Vec<f64> = (0..HIDDEN)
        .map(|i| f64::from(emb_pattern_a(3, i)))
        .collect();
    for _ in 0..5 {
        x = rms64(&x, 1e-6);
    }
    for (j, got) in logits.iter().enumerate() {
        let z = x[j % HIDDEN];
        let want = 0.5 * 20.0 * (z / 20.0).tanh();
        assert!(
            (f64::from(*got) - want).abs() < 1e-4,
            "logit {j}: got {got}, want {want}"
        );
    }
}

/// Test-local RoPE in f64: NEOX split-half pairs, pair i joining
/// channels i and i + dim/2 and turning by position * base^(-2i /
/// dim), the convention llama.cpp applies to Glimmer and the one
/// `torch-math`'s `rope_neox` implements. Written from the
/// definition here so the naive reference shares no code with the
/// backend.
fn rope64(v: &[f64], position: usize, base: f64) -> Vec<f64> {
    let mut out = v.to_vec();
    let dim = v.len() as f64;
    let half = v.len() / 2;
    for pair in 0..half {
        let angle = position as f64 * base.powf(-((2 * pair) as f64) / dim);
        let (s, c) = angle.sin_cos();
        let (a, b) = (out[pair], out[pair + half]);
        out[pair] = a * c - b * s;
        out[pair + half] = a * s + b * c;
    }
    out
}

#[test]
fn attention_mixing_matches_a_naive_reference() {
    // ASSUMPTIONS PINNED, both flagged in muse-glimmer's model docs:
    // (1) the attention gate projection reads the same normed input
    // as the Q/K/V projections. Here attn_gate is 0.25 * identity,
    // so the gate values are 0.25 * h and the naive route below uses
    // h (the normed input), never the raw residual; the opposite
    // reading produces different numbers and would fail. (2) The
    // post norms sit after the residual adds, as in the sibling
    // test. Everything else (QK norms per head, RoPE at absolute
    // positions, GQA grouping 2:1, causal masking, the gate's
    // sigmoid, the final norm, the softcap) is computed here from
    // the formulas, in f64, sharing no code with the backend.
    let cfg = test_config(vec![LayerKind::Global], 16);
    let model = make_model(&cfg, Style::Attention);
    let mut cache = PositionCache::new();
    let logits = model.forward(&[5, 9], &mut cache).expect("forward runs");
    assert_eq!(cache.position(), 2);

    let eps = 1e-6f64;
    let toks = [5usize, 9usize];
    let emb: Vec<Vec<f64>> = toks
        .iter()
        .map(|&t| {
            (0..HIDDEN)
                .map(|i| f64::from(emb_pattern_b(t, i)))
                .collect()
        })
        .collect();
    let h: Vec<Vec<f64>> = emb.iter().map(|x| rms64(x, eps)).collect();
    let q_of = |p: usize, head: usize| -> Vec<f64> {
        rope64(
            &rms64(&h[p][head * HEAD_DIM..(head + 1) * HEAD_DIM], eps),
            p,
            10_000.0,
        )
    };
    let k_of = |p: usize, kv: usize| -> Vec<f64> {
        rope64(
            &rms64(&h[p][kv * HEAD_DIM..(kv + 1) * HEAD_DIM], eps),
            p,
            10_000.0,
        )
    };
    let v_of =
        |p: usize, kv: usize| -> Vec<f64> { h[p][kv * HEAD_DIM..(kv + 1) * HEAD_DIM].to_vec() };

    // Attention for the last position (p = 1) only; its output is
    // what the returned logits are computed from.
    let p = 1;
    let mut attn = vec![0.0f64; HIDDEN];
    for head in 0..4 {
        let kv = head / 2; // GQA: heads pair up over 2 KV heads.
        let q = q_of(p, head);
        let mut scores = Vec::new();
        for j in 0..=p {
            let k = k_of(j, kv);
            let s: f64 =
                q.iter().zip(&k).map(|(a, b)| a * b).sum::<f64>() / (HEAD_DIM as f64).sqrt();
            scores.push(s);
        }
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        for d in 0..HEAD_DIM {
            let mut acc = 0.0;
            for (j, e) in exps.iter().enumerate() {
                acc += (e / sum) * v_of(j, kv)[d];
            }
            attn[head * HEAD_DIM + d] = acc;
        }
    }
    // Gate: attn_gate = 0.25 * identity over the normed input h.
    let gated: Vec<f64> = attn
        .iter()
        .zip(&h[p])
        .map(|(a, hv)| a / (1.0 + (-(0.25 * hv)).exp()))
        .collect();
    // Output projection is identity; residual add; post norms.
    let after: Vec<f64> = emb[p].iter().zip(&gated).map(|(x, g)| x + g).collect();
    let a = rms64(&after, eps); // post-attention norm
    let b = rms64(&a, eps); // post-FFN norm (FFN output is zero)
    let n = rms64(&b, eps); // final norm
    for (j, got) in logits.iter().enumerate() {
        let z = n[j % HIDDEN];
        let want = 0.5 * 20.0 * (z / 20.0).tanh();
        assert!(
            (f64::from(*got) - want).abs() < 1e-3,
            "logit {j}: got {got}, want {want}"
        );
    }
}

#[test]
fn forward_is_deterministic() {
    let cfg = test_config(vec![LayerKind::SlidingWindow, LayerKind::Global], 16);
    let run = || {
        let model = make_model(&cfg, Style::Random);
        let mut cache = PositionCache::new();
        model
            .forward(&[11, 22, 33, 44], &mut cache)
            .expect("forward runs")
    };
    let a = run();
    let b = run();
    assert_eq!(a, b, "same weights and input must give identical logits");
}

#[test]
fn incremental_decode_matches_one_shot_prefill() {
    // THE KV correctness proof. One model consumes the sequence in
    // three calls (2 tokens, 1 token, 2 tokens) against one cache;
    // separate models consume each prefix in a single call. Because
    // CpuCompute's attention appends and attends one position at a
    // time, both routes execute the same operations in the same
    // order, and the logits must agree to float exactness (the 1e-6
    // bound leaves room only for reassociation noise, of which there
    // should be none).
    let cfg = test_config(vec![LayerKind::SlidingWindow, LayerKind::Global], 16);
    let seq = [3u32, 17, 42, 90, 5];

    let mut reference: Vec<Vec<f32>> = Vec::new();
    for len in [2usize, 3, 5] {
        let model = make_model(&cfg, Style::Random);
        let mut cache = PositionCache::new();
        reference.push(
            model
                .forward(&seq[..len], &mut cache)
                .expect("one-shot forward runs"),
        );
    }

    let model = make_model(&cfg, Style::Random);
    let mut cache = PositionCache::new();
    let first = model.forward(&seq[0..2], &mut cache).expect("chunk 1");
    let second = model.forward(&seq[2..3], &mut cache).expect("chunk 2");
    let third = model.forward(&seq[3..5], &mut cache).expect("chunk 3");
    let got = [first, second, third];
    assert_eq!(cache.position(), 5);

    for (i, (g, r)) in got.iter().zip(reference.iter()).enumerate() {
        let d = max_diff(g, r);
        assert!(
            d <= 1e-6,
            "step {i}: incremental logits differ from one-shot prefill by {d}"
        );
    }
}

#[test]
fn incremental_matches_prefill_across_window_eviction() {
    // Same proof, but the sliding layer's window (3) is smaller than
    // the sequence (6), so the ring buffer evicts mid-sequence on
    // both routes. The global layer rides along in the same model.
    let cfg = test_config(vec![LayerKind::SlidingWindow, LayerKind::Global], 3);
    let seq = [9u32, 8, 7, 6, 5, 4];

    let reference = {
        let model = make_model(&cfg, Style::Random);
        let mut cache = PositionCache::new();
        model
            .forward(&seq, &mut cache)
            .expect("one-shot forward runs")
    };

    let model = make_model(&cfg, Style::Random);
    let mut cache = PositionCache::new();
    model.forward(&seq[0..4], &mut cache).expect("chunk 1");
    let got = model.forward(&seq[4..6], &mut cache).expect("chunk 2");
    assert_eq!(cache.position(), 6);

    let d = max_diff(&got, &reference);
    assert!(
        d <= 1e-6,
        "incremental logits differ from one-shot prefill by {d} across window eviction"
    );
}

#[test]
fn unsupported_dtype_refuses_by_name() {
    // A Q4_0 tensor in the FFN down projection: the gguf crate has
    // no Q4_0 dequantizer, so the forward must refuse naming the
    // tensor and the type, never substitute anything.
    let cfg = test_config(vec![LayerKind::SlidingWindow, LayerKind::Global], 16);
    let mut source = build_source(&cfg, Style::Random);
    source.tensors.insert(
        "blk.0.ffn_down.weight".to_string(),
        TensorView {
            name: "blk.0.ffn_down.weight".to_string(),
            dtype: Dtype::Q4_0,
            // A plausible [ffn, hidden] shape, so the shape checks
            // pass and it is the dtype refusal that fires.
            shape: vec![FFN as u64, HIDDEN as u64],
            data: Arc::from(vec![0u8; 4].into_boxed_slice()),
        },
    );
    let weights = Weights::load(&source, &cfg).expect("weights load");
    let model = GlimmerModel::new(cfg, weights, Box::new(CpuCompute::new()));
    let mut cache = PositionCache::new();
    let err = model
        .forward(&[1, 2], &mut cache)
        .expect_err("Q4_0 must refuse");
    match err {
        GlimmerError::StepFailed { step, detail } => {
            assert_eq!(step, "swiglu_ffn");
            assert!(detail.contains("blk.0.ffn_down.weight"), "{detail}");
            assert!(detail.contains("Q4_0"), "{detail}");
        }
        other => panic!("expected StepFailed, got {other:?}"),
    }
    // A refused pass must not advance the caller's cache.
    assert_eq!(cache.position(), 0);
}

#[test]
fn fresh_cache_against_used_compute_refuses() {
    // The position contract: the backend's KV state is at position 2
    // after a 2-token forward, so handing the same model a fresh
    // cache (position 0) must refuse, not attend to a history the
    // caller does not know about.
    let cfg = test_config(vec![LayerKind::SlidingWindow, LayerKind::Global], 16);
    let model = make_model(&cfg, Style::Random);
    let mut cache_a = PositionCache::new();
    model
        .forward(&[1, 2], &mut cache_a)
        .expect("first forward runs");
    assert_eq!(cache_a.position(), 2);

    let mut cache_b = PositionCache::new();
    let err = model
        .forward(&[3], &mut cache_b)
        .expect_err("diverged caches must refuse");
    match err {
        GlimmerError::StepFailed { step, detail } => {
            assert_eq!(step, "attention");
            assert!(detail.contains("position 2"), "{detail}");
            assert!(detail.contains("position 0"), "{detail}");
        }
        other => panic!("expected StepFailed, got {other:?}"),
    }
}

#[test]
fn mismatched_key_value_length_refuses() {
    // The attention crate stores keys and values at one head
    // dimension; a config with key_length 8 and value_length 4 is
    // refused by name at the attention step.
    let cfg = test_config_kv(vec![LayerKind::Global], 16, 8, 4);
    let model = make_model(&cfg, Style::Random);
    let mut cache = PositionCache::new();
    let err = model
        .forward(&[1], &mut cache)
        .expect_err("mismatched head lengths must refuse");
    match err {
        GlimmerError::StepFailed { step, detail } => {
            assert_eq!(step, "attention");
            assert!(detail.contains("key_length 8"), "{detail}");
            assert!(detail.contains("value_length 4"), "{detail}");
        }
        other => panic!("expected StepFailed, got {other:?}"),
    }
}
