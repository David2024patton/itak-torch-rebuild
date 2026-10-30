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
        rope_freq_base_swa: 10_000.0,
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
    // CONVENTIONS PINNED, settled from llama.cpp source in the Run B
    // follow-up (src/models/muse-glimmer.cpp): (1) the embedding
    // output passes a weightless RMSNorm before layer 0; (2) each
    // post norm normalizes the sub-block's OUTPUT and the result is
    // added to the untouched residual, so the residual stream itself
    // is never renormalized; (3) the head scales first, then
    // softcaps. With the attention output projection and the FFN
    // down projection zeroed, both branch outputs are the zero
    // vector, their post norms are zero too, and each layer leaves
    // the stream exactly as it found it. The expected logits below
    // are computed by exactly that reading with test-local math:
    // the embedding row passes TWO norms in total (embedding norm,
    // final norm). The pre-fix conventions (post norms after the
    // adds, no embedding norm, cap before scale) would give five
    // norms and a different finishing curve, and fail here.
    let cfg = test_config(vec![LayerKind::SlidingWindow, LayerKind::SlidingWindow], 16);
    let model = make_model(&cfg, Style::Analytic);
    let mut cache = PositionCache::new();
    let logits = model
        .forward(&[7, 42, 3], &mut cache)
        .expect("forward runs");
    assert_eq!(cache.position(), 3);
    assert_eq!(logits.len(), VOCAB);

    // Last token is 3; its embedding row passes the embedding norm
    // and the final norm, then the repeat-identity head, then the
    // file's scale and softcap (scale 0.5, cap 20, in that order).
    let mut x: Vec<f64> = (0..HIDDEN)
        .map(|i| f64::from(emb_pattern_a(3, i)))
        .collect();
    for _ in 0..2 {
        x = rms64(&x, 1e-6);
    }
    for (j, got) in logits.iter().enumerate() {
        let z = x[j % HIDDEN];
        let want = 20.0 * (0.5 * z / 20.0).tanh();
        assert!(
            (f64::from(*got) - want).abs() < 1e-4,
            "logit {j}: got {got}, want {want}"
        );
    }
}

/// Test-local RoPE in f64: adjacent pairs, pair i turning by
/// position * base^(-2i / dim). This is ggml's NORM pairing, the
/// one llama.cpp selects for muse-glimmer; torch-math's default
/// `rope` is the half-split NEOX pairing the llama/qwen2 families
/// use, and the Glimmer backend must not call that one.
fn rope64(v: &[f64], position: usize, base: f64) -> Vec<f64> {
    let mut out = v.to_vec();
    let dim = v.len() as f64;
    for pair in 0..v.len() / 2 {
        let angle = position as f64 * base.powf(-((2 * pair) as f64) / dim);
        let (s, c) = angle.sin_cos();
        let (a, b) = (out[2 * pair], out[2 * pair + 1]);
        out[2 * pair] = a * c - b * s;
        out[2 * pair + 1] = a * s + b * c;
    }
    out
}

#[test]
fn attention_mixing_matches_a_naive_reference() {
    // CONVENTIONS PINNED, settled from llama.cpp source in the Run B
    // follow-up (src/models/muse-glimmer.cpp): (1) the attention
    // gate projection reads the same normed input as the Q/K/V
    // projections. Here attn_gate is 0.25 * identity, so the gate
    // values are 0.25 * h and the naive route below uses h (the
    // normed input), never the raw stream; the opposite reading
    // produces different numbers and would fail. (2) The embedding
    // output is RMS-normed (weightless) before the layer, and that
    // normed stream, not the raw embedding, is the residual base.
    // (3) The post-attention norm normalizes the branch output
    // (with the reference graph's hardcoded eps 1e-8) before the
    // residual add; the FFN output is zero in this style, so its
    // post norm contributes nothing. (4) RoPE runs on the sliding
    // layer and NOT on the global layer (NoPE), so the naive route
    // runs twice, once per kind, rotating only for the sliding one:
    // a backend that rotates the global layer, or fails to rotate
    // the sliding one, disagrees with one of the two runs.
    // Everything else (QK norms per head, GQA grouping 2:1, causal
    // masking, the gate's sigmoid, the final norm, scale-then-cap)
    // is computed here from the formulas, in f64, sharing no code
    // with the backend.
    let eps = 1e-6f64;
    let post_eps = 1e-8f64; // the reference graph's hardcoded post-norm eps
    let toks = [5usize, 9usize];
    let emb: Vec<Vec<f64>> = toks
        .iter()
        .map(|&t| {
            (0..HIDDEN)
                .map(|i| f64::from(emb_pattern_b(t, i)))
                .collect()
        })
        .collect();
    // The stream entering the layer: embedding rows after the
    // weightless embedding norm.
    let stream: Vec<Vec<f64>> = emb.iter().map(|x| rms64(x, eps)).collect();
    let h: Vec<Vec<f64>> = stream.iter().map(|x| rms64(x, eps)).collect();

    for (kind, rotate) in [
        (LayerKind::Global, false),
        (LayerKind::SlidingWindow, true),
    ] {
        let cfg = test_config(vec![kind], 16);
        let model = make_model(&cfg, Style::Attention);
        let mut cache = PositionCache::new();
        let logits = model.forward(&[5, 9], &mut cache).expect("forward runs");
        assert_eq!(cache.position(), 2);

        let q_of = |p: usize, head: usize| -> Vec<f64> {
            let q = rms64(&h[p][head * HEAD_DIM..(head + 1) * HEAD_DIM], eps);
            if rotate {
                rope64(&q, p, 10_000.0)
            } else {
                q
            }
        };
        let k_of = |p: usize, kv: usize| -> Vec<f64> {
            let k = rms64(&h[p][kv * HEAD_DIM..(kv + 1) * HEAD_DIM], eps);
            if rotate {
                rope64(&k, p, 10_000.0)
            } else {
                k
            }
        };
        let v_of = |p: usize, kv: usize| -> Vec<f64> {
            h[p][kv * HEAD_DIM..(kv + 1) * HEAD_DIM].to_vec()
        };

        // Attention for the last position (p = 1) only; its output
        // is what the returned logits are computed from.
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
        // Output projection is identity. Post norm on the branch,
        // then the residual add against the embedding-normed stream.
        // The FFN branch is zero, so the stream passes it unchanged.
        let post = rms64(&gated, post_eps);
        let after: Vec<f64> = stream[p]
            .iter()
            .zip(&post)
            .map(|(x, g)| x + g)
            .collect();
        let n = rms64(&after, eps); // final norm
        for (j, got) in logits.iter().enumerate() {
            let z = n[j % HIDDEN];
            let want = 20.0 * (0.5 * z / 20.0).tanh();
            assert!(
                (f64::from(*got) - want).abs() < 1e-3,
                "{kind:?} logit {j}: got {got}, want {want}"
            );
        }
    }
}

#[test]
fn post_norm_eps_and_placement_are_pinned_by_a_tiny_branch() {
    // The post-norm convention has two halves, both settled from
    // src/models/muse-glimmer.cpp: the norm applies to the branch
    // output BEFORE the residual add, with eps hardcoded to 1e-8
    // (not the file's 1e-5). With an ordinary-size branch the eps
    // half is numerically invisible, so this test shrinks the
    // attention output projection to 1e-4 * identity: the branch RMS
    // lands near sqrt(1e-8), where eps 1e-8 vs eps 1e-6 changes the
    // normalized branch by more than an order of magnitude, and the
    // placement half matters just as much (normalizing the branch
    // rescales it to unit RMS; normalizing the sum would leave it a
    // 1e-4 ripple on the stream). The naive route below computes
    // the expected logits under both eps readings and asserts the
    // backend lands on the 1e-8 one decisively.
    let cfg = test_config(vec![LayerKind::Global], 16);
    let mut source = build_source(&cfg, Style::Attention);
    source.tensors.insert(
        "blk.0.attn_output.weight".to_string(),
        f32_tensor(
            "blk.0.attn_output.weight",
            &[HIDDEN as u64, HIDDEN as u64],
            scaled_identity(HIDDEN, 1e-4),
        ),
    );
    let weights = Weights::load(&source, &cfg).expect("weights load");
    let model = GlimmerModel::new(cfg, weights, Box::new(CpuCompute::new()));
    let mut cache = PositionCache::new();
    let logits = model.forward(&[5, 9], &mut cache).expect("forward runs");

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
    let stream: Vec<Vec<f64>> = emb.iter().map(|x| rms64(x, eps)).collect();
    let h: Vec<Vec<f64>> = stream.iter().map(|x| rms64(x, eps)).collect();
    let p = 1;
    let mut attn = vec![0.0f64; HIDDEN];
    for head in 0..4 {
        let kv = head / 2;
        let q = rms64(&h[p][head * HEAD_DIM..(head + 1) * HEAD_DIM], eps);
        let mut scores = Vec::new();
        for hj in h.iter().take(p + 1) {
            let k = rms64(&hj[kv * HEAD_DIM..(kv + 1) * HEAD_DIM], eps);
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
                acc += (e / sum) * h[j][kv * HEAD_DIM + d];
            }
            attn[head * HEAD_DIM + d] = acc;
        }
    }
    let branch: Vec<f64> = attn
        .iter()
        .zip(&h[p])
        .map(|(a, hv)| 1e-4 * a / (1.0 + (-(0.25 * hv)).exp()))
        .collect();
    let expected = |post_eps: f64| -> Vec<f64> {
        let post = rms64(&branch, post_eps);
        let after: Vec<f64> = stream[p].iter().zip(&post).map(|(x, g)| x + g).collect();
        let n = rms64(&after, eps);
        (0..VOCAB)
            .map(|j| 20.0 * (0.5 * n[j % HIDDEN] / 20.0).tanh())
            .collect()
    };
    let to_f32 = |v: &[f64]| v.iter().map(|x| *x as f32).collect::<Vec<f32>>();
    let d_right = max_diff(&logits, &to_f32(&expected(1e-8)));
    let d_wrong = max_diff(&logits, &to_f32(&expected(1e-6)));
    assert!(
        d_right < 2e-3 && d_right * 10.0 < d_wrong,
        "post-norm eps/placement not decisive: distance to the eps-1e-8 reading \
         is {d_right}, to the eps-1e-6 reading is {d_wrong}"
    );
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

/// Intermediate-scale config for the decode-sensitivity test: the
/// real model's head geometry (32 Q heads over 2 KV heads, a 16 to
/// 1 grouping), its eps and RoPE base, an every-fourth global layer
/// pattern like the real 52, but only 8 layers and hidden 512 so
/// the test stays inside a debug-build time budget (the reference
/// backend dequantizes per call; at hidden 1024 this test took
/// ~300 s in debug). The existing proofs are 2 layers, 4 heads,
/// head dim 8; run B's freeze appeared at real scale, so this test
/// exists to find which scale axis, if any, kills decode
/// sensitivity in isolation.
fn intermediate_config() -> GlimmerConfig {
    let layers: Vec<LayerKind> = (0..8)
        .map(|i| {
            if i % 4 == 3 {
                LayerKind::Global
            } else {
                LayerKind::SlidingWindow
            }
        })
        .collect();
    GlimmerConfig {
        name: Some("intermediate synthetic".to_string()),
        block_count: 8,
        context_length: 4096,
        hidden: 512,
        ffn_hidden: 1024,
        q_heads: 32,
        kv_heads: 2,
        key_length: 16,
        value_length: 16,
        rope_freq_base: 500_000.0,
        rope_freq_base_swa: 500_000.0,
        rms_eps: 1e-5,
        sliding_window: 2048,
        logit_softcap: 20.0,
        logit_scale: 0.196_116_13,
        vocab_size: 96,
        layers,
        pattern_source: PatternSource::MeasuredFallback,
    }
}

#[test]
fn decode_step_tracks_the_fed_token_at_intermediate_scale() {
    // Run B's second symptom: at decode time the returned top-5 was
    // identical to about 1e-6 regardless of the token fed. This test
    // measures exactly that sensitivity at intermediate scale: same
    // prompt, two different decode tokens, and the max logit
    // difference between the two resulting distributions must be
    // orders of magnitude above the freeze signature. The one-shot
    // cross-check separates "the decode path is broken" from "the
    // model's dynamics are insensitive": a correct decode step must
    // also reproduce the one-shot logits for prompt + token.
    let cfg = intermediate_config();
    let prompt: [u32; 5] = [1, 2, 3, 4, 5];

    let decode_after = |token: u32| -> Vec<f32> {
        let model = make_model(&cfg, Style::Random);
        let mut cache = PositionCache::new();
        model.forward(&prompt, &mut cache).expect("prefill");
        model.forward(&[token], &mut cache).expect("decode step")
    };
    let logits_a = decode_after(7);
    let logits_b = decode_after(42);

    let sensitivity = max_diff(&logits_a, &logits_b);
    eprintln!("intermediate-scale decode sensitivity (7 vs 42): {sensitivity}");
    assert!(
        sensitivity > 1e-3,
        "decode sensitivity collapsed: feeding token 7 vs token 42 after the \
         same prompt changed the logits by only {sensitivity} (freeze signature \
         in run B was ~1e-6)"
    );

    // The decode step must equal a one-shot run over prompt + token.
    let one_shot = {
        let model = make_model(&cfg, Style::Random);
        let mut cache = PositionCache::new();
        let mut seq = prompt.to_vec();
        seq.push(7);
        model.forward(&seq, &mut cache).expect("one-shot")
    };
    let lag = max_diff(&logits_a, &one_shot);
    assert!(
        lag <= 1e-5,
        "decode step disagrees with one-shot prompt+token by {lag}: the decode \
         path is not computing the distribution for the position it claims"
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

/// Block forward (E1) and the speculative round trip (E1 + E2),
/// appended as their own module so the convention proofs above stay
/// untouched. These tests reuse the fixtures above: same weight
/// builder, same configs, same max-diff helper.
mod block_and_spec {
    use super::*;
    use std::rc::Rc;

    use muse_glimmer::{AttentionRequest, Compute, LayerWeights};

    #[test]
    fn block_forward_matches_sequential_single_token_forwards() {
        // E1's core proof: teacher-forcing a block yields, at every
        // row, the distribution a sequential decode produces at that
        // position. Causality is what makes this exact rather than
        // approximate: row i may only depend on tokens 0..=i, and
        // both routes compute it that way.
        let cfg = test_config(vec![LayerKind::SlidingWindow, LayerKind::Global], 16);
        let vocab = cfg.vocab_size as usize;
        let seq = [3u32, 17, 42, 90, 5];

        let model = make_model(&cfg, Style::Random);
        let mut cache = PositionCache::new();
        let block = model
            .forward_block(&seq, &mut cache)
            .expect("block forward runs");
        assert_eq!(block.len(), seq.len() * vocab);
        assert_eq!(cache.position(), seq.len());

        let stepped = make_model(&cfg, Style::Random);
        let mut stepped_cache = PositionCache::new();
        for (i, &token) in seq.iter().enumerate() {
            let logits = stepped
                .forward(&[token], &mut stepped_cache)
                .expect("sequential step runs");
            let row = &block[i * vocab..(i + 1) * vocab];
            let d = max_diff(row, &logits);
            assert!(
                d <= 1e-6,
                "position {i}: block row differs from the sequential forward by {d}"
            );
        }

        // forward is the block's last row: one implementation, two
        // entry points.
        let whole = make_model(&cfg, Style::Random);
        let mut whole_cache = PositionCache::new();
        let last = whole
            .forward(&seq, &mut whole_cache)
            .expect("one-shot forward runs");
        let d = max_diff(&block[(seq.len() - 1) * vocab..], &last);
        assert!(d <= 1e-6, "forward disagrees with forward_block's last row by {d}");
    }

    #[test]
    fn block_forward_crossing_a_sliding_window_boundary_matches_sequential() {
        // Window 3, sequence 6: the block crosses the ring boundary
        // mid-block, and rows after the crossing were computed
        // against a window the earlier rows never saw. The per-row
        // agreement with the sequential route must survive that.
        let cfg = test_config(vec![LayerKind::SlidingWindow, LayerKind::Global], 3);
        let vocab = cfg.vocab_size as usize;
        let seq = [9u32, 8, 7, 6, 5, 4];

        let model = make_model(&cfg, Style::Random);
        let mut cache = PositionCache::new();
        let block = model
            .forward_block(&seq, &mut cache)
            .expect("block forward runs");

        let stepped = make_model(&cfg, Style::Random);
        let mut stepped_cache = PositionCache::new();
        for (i, &token) in seq.iter().enumerate() {
            let logits = stepped
                .forward(&[token], &mut stepped_cache)
                .expect("sequential step runs");
            let row = &block[i * vocab..(i + 1) * vocab];
            let d = max_diff(row, &logits);
            assert!(
                d <= 1e-6,
                "position {i}: block row differs across the window boundary by {d}"
            );
        }
    }

    #[test]
    fn block_forward_matches_sequential_at_intermediate_scale() {
        // Same proof at the convention tests' intermediate scale:
        // 8 layers in the real every-fourth pattern, 32 Q heads over
        // 2 KV heads, the real eps and softcap/scale finishing.
        let cfg = intermediate_config();
        let vocab = cfg.vocab_size as usize;
        let seq = [3u32, 17, 42, 90, 5, 23];

        let model = make_model(&cfg, Style::Random);
        let mut cache = PositionCache::new();
        let block = model
            .forward_block(&seq, &mut cache)
            .expect("block forward runs");
        assert_eq!(block.len(), seq.len() * vocab);

        let stepped = make_model(&cfg, Style::Random);
        let mut stepped_cache = PositionCache::new();
        for (i, &token) in seq.iter().enumerate() {
            let logits = stepped
                .forward(&[token], &mut stepped_cache)
                .expect("sequential step runs");
            let row = &block[i * vocab..(i + 1) * vocab];
            let d = max_diff(row, &logits);
            assert!(
                d <= 1e-5,
                "position {i}: block row differs from the sequential forward by {d} at intermediate scale"
            );
        }
    }

    /// A `Compute` that delegates to a shared `CpuCompute`, so a
    /// test can hold the backend next to the model that owns it and
    /// drive the checkpoint API the model cannot see. Test
    /// scaffolding only: every method is a one-line forward.
    struct SharedCompute(Rc<CpuCompute>);

    impl Compute for SharedCompute {
        fn embed(
            &self,
            table: &TensorView,
            tokens: &[u32],
            cfg: &GlimmerConfig,
        ) -> muse_glimmer::Result<Vec<f32>> {
            self.0.embed(table, tokens, cfg)
        }

        fn rms_norm(
            &self,
            x: &[f32],
            weight: &TensorView,
            rows: usize,
            eps: f32,
        ) -> muse_glimmer::Result<Vec<f32>> {
            self.0.rms_norm(x, weight, rows, eps)
        }

        fn rms_norm_plain(
            &self,
            x: &[f32],
            rows: usize,
            eps: f32,
        ) -> muse_glimmer::Result<Vec<f32>> {
            self.0.rms_norm_plain(x, rows, eps)
        }

        fn proj(
            &self,
            x: &[f32],
            weight: &TensorView,
            rows: usize,
        ) -> muse_glimmer::Result<Vec<f32>> {
            self.0.proj(x, weight, rows)
        }

        fn attention(
            &self,
            req: &AttentionRequest<'_>,
            cache: &mut dyn Cache,
        ) -> muse_glimmer::Result<Vec<f32>> {
            self.0.attention(req, cache)
        }

        fn attn_gate(&self, attn_out: &[f32], gate: &[f32]) -> muse_glimmer::Result<Vec<f32>> {
            self.0.attn_gate(attn_out, gate)
        }

        fn add(&self, a: &[f32], b: &[f32]) -> muse_glimmer::Result<Vec<f32>> {
            self.0.add(a, b)
        }

        fn swiglu_ffn(
            &self,
            x: &[f32],
            w: &LayerWeights,
            rows: usize,
            cfg: &GlimmerConfig,
        ) -> muse_glimmer::Result<Vec<f32>> {
            self.0.swiglu_ffn(x, w, rows, cfg)
        }

        fn lm_head(
            &self,
            x: &[f32],
            head: &TensorView,
            cfg: &GlimmerConfig,
        ) -> muse_glimmer::Result<Vec<f32>> {
            self.0.lm_head(x, head, cfg)
        }
    }

    fn make_shared_model(cfg: &GlimmerConfig) -> (GlimmerModel, Rc<CpuCompute>) {
        let compute = Rc::new(CpuCompute::new());
        let source = build_source(cfg, Style::Random);
        let weights = Weights::load(&source, cfg).expect("synthetic weights load");
        let model = GlimmerModel::new(
            cfg.clone(),
            weights,
            Box::new(SharedCompute(compute.clone())),
        );
        (model, compute)
    }

    #[test]
    fn speculative_round_trip_rewind_restores_exact_continuation() {
        // The whole E1 + E2 loop at model scale, on a window-3
        // config: the prompt alone overflows the ring, and the
        // block's appends evict rows held for the committed prefix
        // (SPEC Section 5.1's corruption case, proven byte-exact at
        // the cache level in the attention crate's tests). The round
        // bound equals the window here, the largest round the ring
        // admits exactly. Checkpoint, block forward over three
        // drafted tokens, rewind to keep the first two, release,
        // then continue: the continuation logits must match a route
        // that never speculated, because the rewind restored the
        // cache exactly, kept speculative rows included.
        let cfg = test_config(vec![LayerKind::SlidingWindow, LayerKind::Global], 3);
        let prompt = [3u32, 17, 42, 90];
        let block_tokens = [5u32, 23, 8];
        let continuation = [11u32];

        let reference = {
            let model = make_model(&cfg, Style::Random);
            let mut cache = PositionCache::new();
            model.forward(&prompt, &mut cache).expect("prefill");
            model
                .forward(&block_tokens[..2], &mut cache)
                .expect("kept tokens");
            model
                .forward(&continuation, &mut cache)
                .expect("continuation")
        };

        let (model, compute) = make_shared_model(&cfg);
        let mut cache = PositionCache::new();
        model.forward(&prompt, &mut cache).expect("prefill");
        assert_eq!(cache.position(), 4);
        let checkpoint = compute.cache_checkpoint(3).expect("checkpoint");
        let block = model
            .forward_block(&block_tokens, &mut cache)
            .expect("verify forward");
        assert_eq!(block.len(), block_tokens.len() * VOCAB);
        assert_eq!(cache.position(), 7);
        // Commit: base 4, pending + 1 accepted = keep 2 of the block.
        compute.cache_rewind(&checkpoint, 6).expect("kv rewind");
        cache.rewind_to(6);
        compute.cache_release(&checkpoint).expect("release");
        let got = model
            .forward(&continuation, &mut cache)
            .expect("continuation");
        assert_eq!(cache.position(), 7);

        let d = max_diff(&got, &reference);
        assert!(
            d <= 1e-6,
            "post-rewind continuation differs from the never-speculated route by {d}"
        );
    }

    #[test]
    fn rewound_kv_with_unrewound_position_cache_refuses() {
        // The two-truths hazard of SPEC Section 5.4: the KV store is
        // rewound but the position cache is not, so they disagree
        // (2 vs 4). The next forward must refuse by name at the
        // attention step's cross-check, not attend across the gap.
        let cfg = test_config(vec![LayerKind::SlidingWindow, LayerKind::Global], 16);
        let (model, compute) = make_shared_model(&cfg);
        let mut cache = PositionCache::new();
        model.forward(&[3, 17], &mut cache).expect("prefill");
        let checkpoint = compute.cache_checkpoint(2).expect("checkpoint");
        model
            .forward_block(&[42, 90], &mut cache)
            .expect("block forward");
        compute.cache_rewind(&checkpoint, 2).expect("kv rewind");
        compute.cache_release(&checkpoint).expect("release");
        // Deliberately no cache.rewind_to(2) here.
        let err = model
            .forward(&[5], &mut cache)
            .expect_err("diverged position truths must refuse");
        match err {
            GlimmerError::StepFailed { step, detail } => {
                assert_eq!(step, "attention");
                assert!(detail.contains("position 2"), "{detail}");
                assert!(detail.contains("position 4"), "{detail}");
            }
            other => panic!("expected StepFailed, got {other:?}"),
        }
    }
}
