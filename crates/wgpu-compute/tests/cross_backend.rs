//! The cross-backend proof: one synthetic multi-layer Glimmer
//! model, the same weights and token sequence through
//! `cpu-compute` (the verified reference) and `wgpu-compute`,
//! comparing per-token logits.
//!
//! The fixture keeps the real model's head geometry (32 Q heads,
//! 2 KV heads, head dim 128) and scales only what is
//! size-invariant: hidden 512, FFN 1024, vocab 1024, six layers
//! in a sliding/global mix, sliding window 4 so a 9-token
//! sequence forces ring eviction on the sliding layers. Every
//! projection, the embedding, and the head are K-quantized
//! (Q4_K, the head Q5_K like the real file's family mix), built
//! by-construction from deterministic block payloads, so both
//! backends consume byte-identical weights; the norms are F32.
//!
//! Pass bar, stated up front: top-1 (argmax) agreement at every
//! step, and max-abs logit difference under 0.5 (the logits are
//! softcapped to [-20, 20]; the two backends sum in different
//! orders, so exact equality is not the honest expectation).
//! Both numbers print at every step.
//!
//! Needs a discrete Vulkan GPU under the backend's default
//! policy; skips with a printed reason where none qualifies.

use std::collections::HashMap;
use std::sync::Arc;

use cpu_compute::CpuCompute;
use muse_glimmer::{
    expected_tensor_names, Cache, Dtype, GlimmerConfig, GlimmerError, GlimmerModel, LayerKind,
    PatternSource, PositionCache, Result, TensorSource, TensorView, Weights,
};
use wgpu_compute::kernels::k1_q4k_gemv::f32_to_f16_bits;
use wgpu_compute::{WgpuCompute, WgpuError};

/// A tiny deterministic byte/float source.
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

/// A Q4_K tensor of shape [inn, out] (GGUF stored order) from
/// by-construction blocks: small f16 header scales, packed 6-bit
/// sub-block scales held to 0..=7, random nibbles. The values
/// this encodes stay near +-0.5, so a six-layer random network
/// produces well-scaled logits on both backends.
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

/// A Q5_K tensor of shape [inn, out], same construction
/// discipline as [`q4k_tensor`] (the head's family in the real
/// file).
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

/// The synthetic config: real head geometry, scaled-down widths,
/// a sliding/global layer mix, window 4.
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

/// Build the full weight set for `cfg`, keyed by the exact names
/// `Weights::load` will ask for.
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
            // The real file's final norm is deliberately
            // unshifted, so its weights center on zero; mirror
            // that here.
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

#[test]
fn cross_backend_greedy_generation_agrees() {
    let gpu_backend = match WgpuCompute::new() {
        Ok(b) => b,
        Err(WgpuError::NoSuitableAdapter { seen, .. }) => {
            eprintln!(
                "skipping cross-backend test: no suitable adapter (enumerated: [{}])",
                seen.join("; ")
            );
            return;
        }
        Err(e) => panic!("backend construction failed unexpectedly: {e}"),
    };
    let cfg = cross_cfg();
    let weights = cross_weights(&cfg);
    let cpu_model = GlimmerModel::new(cfg.clone(), weights.clone(), Box::new(CpuCompute::new()));
    let gpu_model = GlimmerModel::new(cfg, weights, Box::new(gpu_backend));

    let mut cpu_cache = PositionCache::new();
    let mut gpu_cache = PositionCache::new();

    // Prompt of 4, then 5 greedy decode steps: 9 positions
    // through a window of 4.
    let mut next = vec![3u32, 100, 57, 900];
    let mut worst = 0.0f32;
    for step in 0..6 {
        let cpu_logits = cpu_model
            .forward(&next, &mut cpu_cache)
            .expect("cpu forward");
        let gpu_logits = gpu_model
            .forward(&next, &mut gpu_cache)
            .expect("gpu forward");
        let d = max_abs(&cpu_logits, &gpu_logits);
        worst = worst.max(d);
        let cpu_top = argmax(&cpu_logits);
        let gpu_top = argmax(&gpu_logits);
        eprintln!(
            "step {step}: fed {} token(s) at position {}, max abs logit diff {d:.4e}, \
             top-1 cpu {cpu_top} gpu {gpu_top}",
            next.len(),
            cpu_cache.position() - next.len(),
        );
        assert_eq!(
            cpu_top, gpu_top,
            "top-1 disagreement at step {step} (max abs diff {d})"
        );
        assert!(d < 0.5, "logit drift beyond the stated bound at step {step}: {d}");
        assert_eq!(cpu_cache.position(), gpu_cache.position());
        next = vec![cpu_top as u32];
    }
    eprintln!("cross-backend: 6 forwards agreed on top-1 throughout; worst max abs diff {worst:.4e}");
}
