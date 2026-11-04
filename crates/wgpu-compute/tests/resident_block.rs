//! The resident block pin (TODO T-9.6): the speculative
//! loop's three new session shapes against the CPU reference
//! on the cross-backend fixture (hidden 512, FFN 1024, vocab
//! 1024, six layers in a sliding/global mix, window 4).
//!
//! - Block logits: `forward_block_capturing` over a verify
//!   block must return, per row, the distribution sequential
//!   forwards produce (the CPU model's
//!   `forward_block_capturing` is the reference; both are
//!   held to the cross-backend bar: per-row argmax agreement
//!   and max-abs logit difference under 0.5).
//! - Captures: the resident layer-input rows must match the
//!   CPU model's captures at the same bar, on the block
//!   forward and on the capturing prefill.
//! - Rewind: after `rewind_to`, re-forwarding different
//!   tokens must reproduce a fresh session's block exactly
//!   (same bar), and the two refusal shapes (ahead of the
//!   position, past the ring headroom) refuse by name.
//!
//! Needs a discrete Vulkan GPU under the backend's default
//! policy; skips with a printed reason where none qualifies.

use std::collections::HashMap;
use std::sync::Arc;

use cpu_compute::CpuCompute;
use muse_glimmer::{
    expected_tensor_names, Dtype, GlimmerConfig, GlimmerError, GlimmerModel, LayerKind,
    PatternSource, PositionCache, Result, TensorSource, TensorView, Weights,
};
use wgpu_compute::kernels::k1_q4k_gemv::f32_to_f16_bits;
use wgpu_compute::{ResidentSession, WgpuCompute, WgpuError};

/// A tiny deterministic byte/float source (the cross-backend
/// fixture's generator, same seed discipline).
struct Rng(u64);

impl Rng {
    fn next_u8(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 56) as u8
    }

    /// A value in [-0.5, 0.5).
    fn next_f32(&mut self) -> f32 {
        self.next_u8() as f32 / 256.0 - 0.5
    }
}

/// A Q4_K tensor of shape [inn, out] from by-construction
/// blocks, exactly the cross-backend fixture's construction.
fn q4k_tensor(rng: &mut Rng, name: &str, inn: usize, out: usize) -> TensorView {
    let blocks = inn * out / 256;
    let mut data = Vec::with_capacity(blocks * 144);
    for _ in 0..blocks {
        data.extend_from_slice(&f32_to_f16_bits(0.005).to_le_bytes());
        data.extend_from_slice(&f32_to_f16_bits(0.002).to_le_bytes());
        for _ in 0..12 {
            data.push(rng.next_u8() & 0x07);
        }
        for _ in 0..128 {
            data.push(rng.next_u8());
        }
    }
    TensorView {
        name: name.to_string(),
        dtype: Dtype::Q4K,
        shape: vec![inn as u64, out as u64],
        data: Arc::from(data.into_boxed_slice()),
    }
}

/// A Q5_K tensor of shape [inn, out], same construction.
fn q5k_tensor(rng: &mut Rng, name: &str, inn: usize, out: usize) -> TensorView {
    let blocks = inn * out / 256;
    let mut data = Vec::with_capacity(blocks * 176);
    for _ in 0..blocks {
        data.extend_from_slice(&f32_to_f16_bits(0.005).to_le_bytes());
        data.extend_from_slice(&f32_to_f16_bits(0.002).to_le_bytes());
        for _ in 0..12 {
            data.push(rng.next_u8() & 0x07);
        }
        for _ in 0..32 {
            data.push(rng.next_u8());
        }
        for _ in 0..128 {
            data.push(rng.next_u8());
        }
    }
    TensorView {
        name: name.to_string(),
        dtype: Dtype::Q5K,
        shape: vec![inn as u64, out as u64],
        data: Arc::from(data.into_boxed_slice()),
    }
}

/// An F32 tensor whose values cluster around `center`.
fn f32_tensor(rng: &mut Rng, name: &str, shape: &[u64], center: f32) -> TensorView {
    let count: u64 = shape.iter().product();
    let data: Vec<u8> = (0..count)
        .flat_map(|_| (center + rng.next_f32()).to_le_bytes())
        .collect();
    TensorView {
        name: name.to_string(),
        dtype: Dtype::F32,
        shape: shape.to_vec(),
        data: Arc::from(data.into_boxed_slice()),
    }
}

/// The synthetic config: real head geometry, scaled-down
/// widths, a sliding/global layer mix, window 4.
fn cross_cfg() -> GlimmerConfig {
    GlimmerConfig {
        name: None,
        block_count: 6,
        context_length: 256,
        hidden: 512,
        ffn_hidden: 1024,
        q_heads: 32,
        kv_heads: 2,
        key_length: 128,
        value_length: 128,
        rope_freq_base: 500_000.0,
        rope_freq_base_swa: 500_000.0,
        rms_eps: 1e-5,
        sliding_window: 4,
        logit_softcap: 20.0,
        logit_scale: 0.196_116_13,
        vocab_size: 1024,
        layers: vec![
            LayerKind::SlidingWindow,
            LayerKind::SlidingWindow,
            LayerKind::SlidingWindow,
            LayerKind::Global,
            LayerKind::SlidingWindow,
            LayerKind::Global,
        ],
        pattern_source: PatternSource::MeasuredFallback,
    }
}

/// Build the full weight set for `cfg`, keyed by the exact
/// names `Weights::load` will ask for (the cross-backend
/// fixture's rules, tensor for tensor).
fn cross_weights(cfg: &GlimmerConfig) -> Weights {
    struct MapSource {
        tensors: HashMap<String, TensorView>,
    }
    impl TensorSource for MapSource {
        fn tensor(&self, name: &str) -> Result<TensorView> {
            self.tensors
                .get(name)
                .cloned()
                .ok_or_else(|| GlimmerError::MissingTensor {
                    name: name.to_string(),
                })
        }
    }

    let hidden = cfg.hidden as usize;
    let ffn = cfg.ffn_hidden as usize;
    let q_width = cfg.q_width() as usize;
    let kv_width = cfg.k_width() as usize;
    let vocab = cfg.vocab_size as usize;
    let head_dim = cfg.key_length as usize;

    let mut rng = Rng(0x5eed_c0de);
    let mut tensors: HashMap<String, TensorView> = HashMap::new();
    for name in expected_tensor_names(cfg) {
        let view = if name == "token_embd.weight" {
            q4k_tensor(&mut rng, &name, hidden, vocab)
        } else if name == "output.weight" {
            q5k_tensor(&mut rng, &name, hidden, vocab)
        } else if name == "output_norm.weight" {
            f32_tensor(&mut rng, &name, &[hidden as u64], 0.0)
        } else if name.ends_with("attn_q_norm.weight") || name.ends_with("attn_k_norm.weight") {
            f32_tensor(&mut rng, &name, &[head_dim as u64], 1.0)
        } else if name.ends_with("attn_norm.weight")
            || name.ends_with("post_attention_norm.weight")
            || name.ends_with("ffn_norm.weight")
            || name.ends_with("post_ffw_norm.weight")
        {
            f32_tensor(&mut rng, &name, &[hidden as u64], 1.0)
        } else if name.ends_with("attn_q.weight") || name.ends_with("attn_gate.weight") {
            q4k_tensor(&mut rng, &name, hidden, q_width)
        } else if name.ends_with("attn_k.weight") || name.ends_with("attn_v.weight") {
            q4k_tensor(&mut rng, &name, hidden, kv_width)
        } else if name.ends_with("attn_output.weight") {
            q4k_tensor(&mut rng, &name, q_width, hidden)
        } else if name.ends_with("ffn_gate.weight") || name.ends_with("ffn_up.weight") {
            q4k_tensor(&mut rng, &name, hidden, ffn)
        } else if name.ends_with("ffn_down.weight") {
            q4k_tensor(&mut rng, &name, ffn, hidden)
        } else {
            panic!("no fixture rule for tensor {name}");
        };
        tensors.insert(name, view);
    }
    Weights::load(&MapSource { tensors }, cfg).expect("fixture weights load")
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("logits are finite"))
        .map(|(i, _)| i)
        .expect("logits are non-empty")
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Build a backend or skip the test: the resident_session.rs
/// discipline (a missing adapter is an environment fact, not
/// a failure).
fn backend_or_skip(test: &str) -> Option<WgpuCompute> {
    match WgpuCompute::new() {
        Ok(b) => Some(b),
        Err(WgpuError::NoSuitableAdapter { seen, .. }) => {
            eprintln!(
                "skipping {test}: no suitable adapter (enumerated: [{}])",
                seen.join("; ")
            );
            None
        }
        Err(e) => panic!("backend construction failed unexpectedly: {e}"),
    }
}

const PROMPT: &[u32] = &[3, 100, 57, 900, 12, 640];
const CAPTURE_LAYERS: &[usize] = &[1, 4];

/// Block logits and captures against the CPU reference, on
/// the capturing prefill and on a verify block.
#[test]
fn resident_block_logits_and_captures_match_cpu() {
    let Some(backend) = backend_or_skip("resident block test") else {
        return;
    };
    let cfg = cross_cfg();
    let weights = cross_weights(&cfg);
    let cpu_model = GlimmerModel::new(cfg.clone(), weights.clone(), Box::new(CpuCompute::new()));
    let mut session =
        ResidentSession::new(&backend, &cfg, &weights).expect("resident session builds");
    let mut cpu_cache = PositionCache::new();
    let vocab = cfg.vocab_size as usize;
    let hidden = cfg.hidden as usize;

    // Capturing prefill: last-row logits and the prompt's
    // layer-input rows against the CPU model's own capturing
    // forward.
    let (cpu_logits, cpu_captures) = cpu_model
        .forward_capturing(PROMPT, &mut cpu_cache, CAPTURE_LAYERS)
        .expect("cpu capturing forward");
    let (res_logits, res_captures) = session
        .forward_capturing(PROMPT, CAPTURE_LAYERS)
        .expect("resident capturing forward");
    assert_eq!(argmax(&cpu_logits), argmax(&res_logits), "prefill top-1");
    let d = max_abs(&cpu_logits, &res_logits);
    assert!(d < 0.5, "prefill logit drift beyond the bar: {d}");
    assert_eq!(res_captures.len(), CAPTURE_LAYERS.len());
    for (cpu_cap, res_cap) in cpu_captures.iter().zip(res_captures.iter()) {
        assert_eq!(cpu_cap.layer, res_cap.layer);
        assert_eq!(res_cap.rows.len(), PROMPT.len() * hidden);
        let d = max_abs(&cpu_cap.rows, &res_cap.rows);
        assert!(
            d < 0.5,
            "prefill capture drift at layer {} beyond the bar: {d}",
            cpu_cap.layer
        );
    }
    assert_eq!(session.position(), PROMPT.len());

    // The verify block: every row against the CPU model's
    // block forward over the same tokens.
    let block: Vec<u32> = vec![11, 222, 5, 999, 3, 77];
    let (cpu_block, cpu_block_captures) = cpu_model
        .forward_block_capturing(&block, &mut cpu_cache, CAPTURE_LAYERS)
        .expect("cpu block forward");
    let before = session.stats();
    let (res_block, res_block_captures) = session
        .forward_block_capturing(&block, CAPTURE_LAYERS)
        .expect("resident block forward");
    assert_eq!(res_block.len(), block.len() * vocab);
    for row in 0..block.len() {
        let cpu_row = &cpu_block[row * vocab..(row + 1) * vocab];
        let res_row = &res_block[row * vocab..(row + 1) * vocab];
        assert_eq!(
            argmax(cpu_row),
            argmax(res_row),
            "block row {row} top-1 disagreement"
        );
        let d = max_abs(cpu_row, res_row);
        assert!(d < 0.5, "block row {row} logit drift beyond the bar: {d}");
        assert!(
            res_row.iter().all(|v| v.is_finite()),
            "block row {row} produced non-finite logits"
        );
    }
    for (cpu_cap, res_cap) in cpu_block_captures.iter().zip(res_block_captures.iter()) {
        assert_eq!(cpu_cap.layer, res_cap.layer);
        assert_eq!(res_cap.rows.len(), block.len() * hidden);
        let d = max_abs(&cpu_cap.rows, &res_cap.rows);
        assert!(
            d < 0.5,
            "block capture drift at layer {} beyond the bar: {d}",
            cpu_cap.layer
        );
    }
    // The block was one chunk: one submission, one readback.
    let after = session.stats();
    assert_eq!(after.chunks, before.chunks + 1);
    assert_eq!(after.submissions, before.submissions + 1);
    assert_eq!(after.readbacks, before.readbacks + 1);
    assert_eq!(session.position(), PROMPT.len() + block.len());

    // A capture index at the layer count refuses before any
    // forward, the model's own error.
    let err = session
        .forward_block_capturing(&[1], &[cfg.block_count as usize])
        .expect_err("out-of-range capture must refuse");
    assert!(
        matches!(err, GlimmerError::CaptureLayerOutOfRange { .. }),
        "got {err}"
    );
    assert_eq!(
        session.position(),
        PROMPT.len() + block.len(),
        "the refused forward advanced nothing"
    );
}

/// Rewind: a discarded block leaves no trace. After rewinding
/// it away and forwarding different tokens, the block logits
/// equal a fresh session's over the same final prefix; the
/// refusal shapes refuse by name.
#[test]
fn resident_rewind_restores_the_committed_prefix() {
    let Some(backend_a) = backend_or_skip("resident rewind test") else {
        return;
    };
    let backend_b = WgpuCompute::new().expect("second backend instance builds");
    let cfg = cross_cfg();
    let weights = cross_weights(&cfg);
    let vocab = cfg.vocab_size as usize;

    let block_a: Vec<u32> = vec![11, 222, 5, 999];
    let block_b: Vec<u32> = vec![640, 3, 88, 451];

    // Session A: prompt, block A, rewind it away, block B.
    let mut a = ResidentSession::new(&backend_a, &cfg, &weights).expect("session A builds");
    a.forward(PROMPT).expect("A prompt");
    a.forward_block_capturing(&block_a, &[]).expect("A block A");
    assert_eq!(a.position(), PROMPT.len() + block_a.len());
    a.rewind_to(PROMPT.len()).expect("A rewind");
    assert_eq!(a.position(), PROMPT.len());
    let (a_logits, _) = a.forward_block_capturing(&block_b, &[]).expect("A block B");

    // Session B: prompt, block B directly.
    let mut b = ResidentSession::new(&backend_b, &cfg, &weights).expect("session B builds");
    b.forward(PROMPT).expect("B prompt");
    let (b_logits, _) = b.forward_block_capturing(&block_b, &[]).expect("B block B");

    for row in 0..block_b.len() {
        let a_row = &a_logits[row * vocab..(row + 1) * vocab];
        let b_row = &b_logits[row * vocab..(row + 1) * vocab];
        assert_eq!(
            argmax(a_row),
            argmax(b_row),
            "post-rewind row {row} top-1 disagreement"
        );
        let d = max_abs(a_row, b_row);
        assert!(d < 0.5, "post-rewind row {row} drift beyond the bar: {d}");
    }
    assert_eq!(a.position(), b.position());

    // Refusals: ahead of the position, and a span past the
    // ring headroom (the fixture's context is 256, so a
    // 70-token forward then a rewind to 0 discards 70).
    let err = a
        .rewind_to(a.position() + 1)
        .expect_err("rewind ahead of the position must refuse");
    assert!(
        matches!(
            err,
            GlimmerError::StepFailed {
                step: "session",
                ..
            }
        ),
        "got {err}"
    );
    a.forward(&vec![7u32; 60]).expect("A extends to 70");
    assert_eq!(a.position(), 70);
    let err = a
        .rewind_to(0)
        .expect_err("a 70-position rewind must refuse");
    assert!(
        matches!(
            err,
            GlimmerError::StepFailed {
                step: "session",
                ..
            }
        ),
        "got {err}"
    );
    a.rewind_to(6)
        .expect("a 64-position rewind is inside the bound");
    assert_eq!(a.position(), 6);
    // Reset remains the full-rewind path.
    a.reset();
    assert_eq!(a.position(), 0);
}
