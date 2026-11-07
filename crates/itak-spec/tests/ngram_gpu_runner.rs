//! The GPU runner's n-gram construction path, pinned at the
//! level the runner uses it: a `SpecSession` built through
//! `SpecSession::new_ngram` over a `ResidentTarget` (the
//! resident GPU driver `spec_generate_gpu` phase B wraps),
//! with NO draft file opened, no `DflashSpeculator` built, and
//! no draft weights loaded anywhere in the test. That is the
//! whole `--drafter=ngram` claim: the proposer conditions on
//! tokens only, so the resident session plus the proposer is
//! a complete speculative session.
//!
//! The fixture is the wgpu-compute resident suite's synthetic
//! model (real head geometry, hidden 512, FFN 1024, vocab
//! 1024, six layers in a sliding/global mix, window 4), the
//! smallest geometry the resident kernels accept, built here
//! tensor for tensor the same way. Adapter-guarded like every
//! GPU test in the workspace: skips with a printed reason
//! where no adapter qualifies (this VM enumerates none), and
//! exercises construction, `begin`, and a short generation
//! where one does.

use std::collections::HashMap;
use std::sync::Arc;

use itak_spec::{
    NgramConfig, NgramProposer, ResidentTarget, SpecConfig, SpecMode, SpecSession, StderrDiag,
};
use muse_glimmer::{
    expected_tensor_names, Dtype, GlimmerConfig, GlimmerError, LayerKind, PatternSource, Result,
    TensorSource, TensorView, Weights,
};
use wgpu_compute::kernels::k1_q4k_gemv::f32_to_f16_bits;
use wgpu_compute::{ResidentSession, WgpuCompute, WgpuError};

/// A tiny deterministic byte source (the resident fixture's
/// generator, same seed discipline).
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
/// blocks (the resident fixture's construction).
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

/// The resident fixture's synthetic config: real head
/// geometry, scaled-down widths, a sliding/global layer mix,
/// window 4.
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
/// names `Weights::load` asks for (the resident fixture's
/// rules, tensor for tensor; every projection Q4_K, the head
/// Q5_K).
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

#[test]
fn ngram_session_constructs_over_resident_target_without_a_draft_file() {
    let backend = match WgpuCompute::new() {
        Ok(b) => b,
        Err(WgpuError::NoSuitableAdapter { seen, .. }) => {
            eprintln!(
                "skipping n-gram GPU runner construction test: no suitable adapter \
                 (enumerated: [{}])",
                seen.join("; ")
            );
            return;
        }
        Err(e) => panic!("backend construction failed unexpectedly: {e}"),
    };
    let cfg = cross_cfg();
    let weights = cross_weights(&cfg);
    let resident = ResidentSession::new(&backend, &cfg, &weights).expect("resident session builds");

    // The runner's n-gram phase B, exactly: proposer from the
    // default config at the target's vocabulary, session
    // through `new_ngram` over the resident driver. No draft
    // file, speculator, or draft config exists in this test.
    let proposer = NgramProposer::new(NgramConfig::default(), cfg.vocab_size as usize)
        .expect("proposer builds at the target vocabulary");
    let config = SpecConfig {
        mode: SpecMode::Ngram,
        depth: 4,
        ..SpecConfig::default()
    };
    let mut session = SpecSession::new_ngram(
        ResidentTarget::new(resident),
        Some(proposer),
        config,
        Arc::new(StderrDiag),
    )
    .expect("n-gram session constructs over the resident target");

    // Begin is the plain resident prefill plus the proposer
    // commit, and a short generation runs real rounds through
    // the loop: the stream is full-length (no stop tokens are
    // configured) and in-vocabulary, and the round accounting
    // the runner prints is populated and self-consistent.
    let pending = session.begin(&[3, 100, 57, 900]).expect("begin runs");
    assert!(pending < cfg.vocab_size);
    let stream = session.generate(8, &[]).expect("generation runs");
    assert_eq!(stream.len(), 8);
    assert!(stream.iter().all(|&t| t < cfg.vocab_size));
    let stats = session.finish();
    assert!(stats.rounds >= 1, "rounds ran: {stats:?}");
    assert!(
        stats.accepted <= stats.proposed,
        "accounting is consistent: {stats:?}"
    );
}
