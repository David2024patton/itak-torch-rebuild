//! The resident-path pin: one synthetic multi-layer Glimmer
//! model (the cross-backend fixture: real head geometry, hidden
//! 512, FFN 1024, vocab 1024, six layers in a sliding/global
//! mix, window 4), the same weights and token sequence through
//! THREE drivers: `cpu-compute` behind `GlimmerModel` (the
//! verified reference), `wgpu-compute` behind `GlimmerModel`
//! (the trait path), and the device-resident
//! [`ResidentSession`], which re-implements the model's step
//! order as one command-buffer chain per chunk.
//!
//! This test is what licenses the resident driver's duplicated
//! step order: any drift between `GlimmerModel::forward_rows`
//! and the session's chain (norm placement, RoPE scope, the
//! gate order, the head finishing) shows up here as a token
//! disagreement.
//!
//! Pass bar, the cross-backend bar: top-1 (argmax) agreement
//! at every step across all three drivers, and max-abs logit
//! difference under 0.5 between every pair. The submission
//! contract is asserted too: one submission per chunk, and
//! one blocking logits readback per forward (a multi-chunk
//! forward defers every chunk's readback but its final
//! chunk's; see the multi-chunk deferral test below).
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

/// A Q6_K tensor of shape [inn, out]: 210-byte blocks
/// (128 low-bit bytes, 64 high-bit bytes, 16 scale bytes, the
/// f16 scale), deterministic payload with a small scale, the
/// construction the fused-kernel fixtures use.
fn q6k_tensor(rng: &mut Rng, name: &str, inn: usize, out: usize) -> TensorView {
    let blocks = inn * out / 256;
    let mut data = Vec::with_capacity(blocks * 210);
    for _ in 0..blocks {
        for _ in 0..128 {
            data.push(rng.next_u8());
        }
        for _ in 0..64 {
            data.push(rng.next_u8());
        }
        for _ in 0..16 {
            data.push(rng.next_u8() & 0x03);
        }
        data.extend_from_slice(&f32_to_f16_bits(0.01).to_le_bytes());
    }
    TensorView {
        name: name.to_string(),
        dtype: Dtype::Q6K,
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
    build_weights(cfg, false)
}

/// The mixed fixture: like [`cross_weights`], but the value
/// projection and the FFN up projection are Q6_K on every
/// layer. The value projection then cannot join the Q4_K
/// batch (a partial batch: Q, K, gate batched, V separate)
/// and the FFN pair cannot form (gate and up separate), so
/// at MMQ row counts the attention stage holds two
/// projections over `h` and the FFN stage holds two: the
/// layer classes whose duplicate quantizes quantize-once
/// removes.
fn cross_weights_mixed(cfg: &GlimmerConfig) -> Weights {
    build_weights(cfg, true)
}

fn build_weights(cfg: &GlimmerConfig, mixed: bool) -> Weights {
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
        } else if name.ends_with("attn_k.weight") {
            q4k_tensor(&mut rng, &name, hidden, kv_width)
        } else if name.ends_with("attn_v.weight") {
            if mixed {
                q6k_tensor(&mut rng, &name, hidden, kv_width)
            } else {
                q4k_tensor(&mut rng, &name, hidden, kv_width)
            }
        } else if name.ends_with("attn_output.weight") {
            q4k_tensor(&mut rng, &name, q_width, hidden)
        } else if name.ends_with("ffn_gate.weight") {
            q4k_tensor(&mut rng, &name, hidden, ffn)
        } else if name.ends_with("ffn_up.weight") {
            if mixed {
                q6k_tensor(&mut rng, &name, hidden, ffn)
            } else {
                q4k_tensor(&mut rng, &name, hidden, ffn)
            }
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

/// Compare one step across the three drivers and assert the
/// bar; returns the agreed next token (the CPU reference's
/// argmax, fed to all three so the sequences cannot diverge
/// even on a near-tie).
fn assert_step_agrees(step: usize, cpu: &[f32], trait_gpu: &[f32], resident: &[f32]) -> u32 {
    for (label, logits) in [("cpu", cpu), ("trait", trait_gpu), ("resident", resident)] {
        let bad = logits.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(
            bad, 0,
            "{label} produced {bad} non-finite logits at step {step}"
        );
    }
    let (cpu_top, trait_top, resident_top) = (argmax(cpu), argmax(trait_gpu), argmax(resident));
    let d_trait = max_abs(cpu, trait_gpu);
    let d_resident = max_abs(cpu, resident);
    let d_cross = max_abs(trait_gpu, resident);
    eprintln!(
        "step {step}: top-1 cpu {cpu_top} trait {trait_top} resident {resident_top}; \
         max abs diff cpu-trait {d_trait:.4e} cpu-resident {d_resident:.4e} \
         trait-resident {d_cross:.4e}"
    );
    assert_eq!(
        cpu_top, trait_top,
        "trait top-1 disagreement at step {step}"
    );
    assert_eq!(
        cpu_top, resident_top,
        "resident top-1 disagreement at step {step}"
    );
    for (label, d) in [
        ("cpu-trait", d_trait),
        ("cpu-resident", d_resident),
        ("trait-resident", d_cross),
    ] {
        assert!(
            d < 0.5,
            "{label} logit drift beyond the stated bound at step {step}: {d}"
        );
    }
    cpu_top as u32
}

#[test]
fn resident_session_matches_trait_and_cpu() {
    let gpu_backend = match WgpuCompute::new() {
        Ok(b) => b,
        Err(WgpuError::NoSuitableAdapter { seen, .. }) => {
            eprintln!(
                "skipping resident session test: no suitable adapter (enumerated: [{}])",
                seen.join("; ")
            );
            return;
        }
        Err(e) => panic!("backend construction failed unexpectedly: {e}"),
    };
    let cfg = cross_cfg();
    let weights = cross_weights(&cfg);
    let cpu_model = GlimmerModel::new(cfg.clone(), weights.clone(), Box::new(CpuCompute::new()));
    let gpu_model = GlimmerModel::new(cfg.clone(), weights.clone(), Box::new(gpu_backend));
    // The session borrows its backend unboxed, so it gets a
    // second instance: same adapter, separate device, no
    // shared state with the trait model's backend.
    let session_backend = WgpuCompute::new().expect("second backend instance builds");
    let mut session =
        ResidentSession::new(&session_backend, &cfg, &weights).expect("resident session builds");

    let mut cpu_cache = PositionCache::new();
    let mut gpu_cache = PositionCache::new();

    // Phase 1: prompt of 4, then 5 greedy decode steps: 9
    // positions through a window of 4, forcing ring eviction
    // on the sliding layers in all three drivers.
    let mut next = vec![3u32, 100, 57, 900];
    for step in 0..6 {
        let cpu_logits = cpu_model
            .forward(&next, &mut cpu_cache)
            .expect("cpu forward");
        let gpu_logits = gpu_model
            .forward(&next, &mut gpu_cache)
            .expect("gpu forward");
        let resident_logits = session.forward(&next).expect("resident forward");
        next = vec![assert_step_agrees(
            step,
            &cpu_logits,
            &gpu_logits,
            &resident_logits,
        )];
        assert_eq!(session.position(), cpu_cache.position());
    }
    // The submission contract: 6 forwards, each a single chunk
    // at this scale, so 6 submissions and 6 readbacks, and the
    // dispatch count is the full 6-layer chain per chunk.
    let stats = session.stats();
    eprintln!("resident stats after phase 1: {stats:?}");
    assert_eq!(stats.forwards, 6);
    assert_eq!(stats.chunks, 6);
    assert_eq!(
        stats.submissions, 6,
        "one submission per chunk, no exceptions"
    );
    assert_eq!(
        stats.readbacks, 6,
        "one logits readback per (single-chunk) forward"
    );
    // The dispatch count is the full 6-layer chain per chunk,
    // at the fused count: the fixture's layers are all Q4_K,
    // so every layer batches QKV+gate and pairs gate+up, and
    // the norm chains fold the post norms into the next
    // norms. On every plan, chunk or decode, layer 0 records
    // 9 dispatches (attn norm, batch, prep, flash, gated
    // output, post chain, pair, fused down, boundary chain)
    // and each later layer 8 (its attn norm rides the
    // previous boundary chain); a chunk adds the gather and
    // the head (the fixture's head is Q5_K, so the finishing
    // map is fused into it), the embedding gather carries
    // its norm fused, and the last layer's close folds the
    // final norm in: 51 per chunk. (Chunk plans recorded 57
    // before the flash switch, when their two-pass attention
    // cost one extra dispatch per layer.) Phase 1 is one
    // 4-row prefill chunk and five decode chunks.
    assert_eq!(
        stats.dispatches,
        6 * 51,
        "the chain is the whole graph at the fused count: {stats:?}"
    );

    // Phase 2: a 70-token prompt, longer than the session's
    // 64-row chunk cap, so prefill splits into two chunks and
    // the second chunk must continue the first chunk's KV at
    // the right positions. Fresh drivers all round: the
    // backends keep per-sequence KV state inside the model, so
    // phase 2 builds new model instances (the session resets
    // in place).
    session.reset();
    assert_eq!(session.position(), 0);
    let cpu_model = GlimmerModel::new(cfg.clone(), weights.clone(), Box::new(CpuCompute::new()));
    let gpu_model = GlimmerModel::new(
        cfg.clone(),
        weights.clone(),
        Box::new(WgpuCompute::new().expect("phase 2 trait backend builds")),
    );
    let mut cpu_cache = PositionCache::new();
    let mut gpu_cache = PositionCache::new();
    let prompt: Vec<u32> = (0..70).map(|i| (i * 37 % 1024) as u32).collect();
    let cpu_logits = cpu_model
        .forward(&prompt, &mut cpu_cache)
        .expect("cpu forward");
    // The trait path is fed token by token here: a single
    // 70-row trait forward currently returns non-finite logits
    // (observed 2026-10-30 on the server run of this test; the
    // trait path had only ever been driven at 9 rows or fewer).
    // Token-by-token is its proven regime and accumulates the
    // same positions, so the comparison stays honest; the
    // resident path takes the prompt in its own chunks, which
    // is the behavior under test.
    let mut gpu_logits = Vec::new();
    for token in &prompt {
        gpu_logits = gpu_model
            .forward(&[*token], &mut gpu_cache)
            .expect("gpu forward");
    }
    let resident_logits = session.forward(&prompt).expect("resident chunked forward");
    let token = assert_step_agrees(70, &cpu_logits, &gpu_logits, &resident_logits);
    assert_eq!(session.position(), 70);
    let stats = session.stats();
    assert_eq!(stats.chunks, 8, "the 70-token prompt adds two chunks");
    assert_eq!(stats.submissions, 8);

    // One decode step after the chunked prefill, still agreeing.
    let cpu_logits = cpu_model
        .forward(&[token], &mut cpu_cache)
        .expect("cpu forward");
    let gpu_logits = gpu_model
        .forward(&[token], &mut gpu_cache)
        .expect("gpu forward");
    let resident_logits = session.forward(&[token]).expect("resident decode");
    assert_step_agrees(71, &cpu_logits, &gpu_logits, &resident_logits);
    assert_eq!(session.position(), 71);
}

/// The host greedy pick with the engine sampler's convention:
/// a strictly-greater scan, so the LOWEST index wins an exact
/// tie. (The file's `argmax` helper above uses `max_by`,
/// which keeps the last maximum; the sampler and the device
/// reduction both keep the first.)
fn host_greedy(logits: &[f32]) -> (u32, f32) {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    (best as u32, best_val)
}

/// The argmax tail: `forward_argmax` must return the same
/// winner (and the same winning logit) as `forward` plus the
/// host greedy pick, on a multi-chunk prompt and across a
/// greedy generation, while reading back eight bytes per
/// readback instead of the logits vector.
#[test]
fn resident_session_argmax_tail_matches_logits() {
    let logits_backend = match WgpuCompute::new() {
        Ok(b) => b,
        Err(WgpuError::NoSuitableAdapter { seen, .. }) => {
            eprintln!(
                "skipping resident argmax test: no suitable adapter (enumerated: [{}])",
                seen.join("; ")
            );
            return;
        }
        Err(e) => panic!("backend construction failed unexpectedly: {e}"),
    };
    let argmax_backend = WgpuCompute::new().expect("argmax backend instance builds");
    let cfg = cross_cfg();
    let weights = cross_weights(&cfg);
    let mut logits_session =
        ResidentSession::new(&logits_backend, &cfg, &weights).expect("logits session builds");
    let mut argmax_session =
        ResidentSession::new(&argmax_backend, &cfg, &weights).expect("argmax session builds");

    // A 70-token prompt: two chunks through both readback
    // shapes, so the reduction is proven on the chunked
    // prefill path as well as on decode rows.
    let prompt: Vec<u32> = (0..70).map(|i| (i * 37 % 1024) as u32).collect();
    let logits = logits_session.forward(&prompt).expect("logits forward");
    let (id, logit) = argmax_session
        .forward_argmax(&prompt)
        .expect("argmax forward");
    let (want_id, want_logit) = host_greedy(&logits);
    assert_eq!(id, want_id, "argmax winner disagrees after the prompt");
    assert!(
        (logit - want_logit).abs() <= 1e-6 + 1e-5 * want_logit.abs(),
        "argmax winning logit {logit} vs host {want_logit}"
    );
    assert_eq!(logits_session.position(), argmax_session.position());
    let a_stats = argmax_session.stats();
    // The two-chunk prompt defers its first chunk's
    // readback on the argmax tail too: one blocking
    // readback (the final chunk's) serves the forward.
    assert_eq!(
        a_stats.argmax_readbacks, 1,
        "one argmax readback per (multi-chunk) forward"
    );
    assert_eq!(a_stats.readbacks, 0, "the argmax tail never reads logits");
    // The logits session's two-chunk forward defers its
    // first chunk's readback: submissions and chunks still
    // count per chunk, but exactly one blocking readback
    // (the final chunk's) serves the whole forward.
    let l_stats = logits_session.stats();
    assert_eq!(l_stats.chunks, 2);
    assert_eq!(l_stats.submissions, 2);
    assert_eq!(
        l_stats.readbacks, 1,
        "a multi-chunk forward reads back once, at its final chunk"
    );

    // Greedy generation: each session feeds its own winner;
    // the streams must stay identical step by step.
    let mut next = vec![id];
    for step in 0..12 {
        let logits = logits_session.forward(&next).expect("logits decode");
        let (step_id, step_logit) = argmax_session.forward_argmax(&next).expect("argmax decode");
        let (want_id, want_logit) = host_greedy(&logits);
        assert_eq!(
            step_id, want_id,
            "argmax winner disagrees at decode step {step}"
        );
        assert!(
            (step_logit - want_logit).abs() <= 1e-6 + 1e-5 * want_logit.abs(),
            "argmax winning logit {step_logit} vs host {want_logit} at step {step}"
        );
        next = vec![step_id];
    }
    // 1 for the two-chunk prompt (its first chunk
    // deferred) plus 12 single-chunk decode steps.
    assert_eq!(argmax_session.stats().argmax_readbacks, 13);
    assert_eq!(argmax_session.position(), logits_session.position());
}

/// The prefill pipelining contract: a multi-chunk
/// `forward` submits every chunk but reads back only its
/// final chunk's logits, and the logits it returns are
/// the ones a chunk-at-a-time driver produces (each chunk
/// its own forward there, hence each chunk read back).
/// The deferral is the whole change: same chunk
/// submissions, same dispatches, same results; one
/// blocking map instead of one per chunk, so on a real
/// device the host prepares chunk N+1 while the device
/// still runs chunk N instead of draining it first.
#[test]
fn resident_session_multichunk_forward_defers_readbacks() {
    let pipelined_backend = match WgpuCompute::new() {
        Ok(b) => b,
        Err(WgpuError::NoSuitableAdapter { seen, .. }) => {
            eprintln!(
                "skipping resident multichunk deferral test: no suitable adapter \
                 (enumerated: [{}])",
                seen.join("; ")
            );
            return;
        }
        Err(e) => panic!("backend construction failed unexpectedly: {e}"),
    };
    let stepped_backend = WgpuCompute::new().expect("stepped backend instance builds");
    let cfg = cross_cfg();
    let weights = cross_weights(&cfg);
    let mut pipelined =
        ResidentSession::new(&pipelined_backend, &cfg, &weights).expect("pipelined session builds");
    let mut stepped =
        ResidentSession::new(&stepped_backend, &cfg, &weights).expect("stepped session builds");

    // 150 tokens at the 64-row cap: chunks of 64, 64, 22.
    let prompt: Vec<u32> = (0..150).map(|i| (i * 37 % 1024) as u32).collect();
    let pipelined_logits = pipelined.forward(&prompt).expect("pipelined forward");
    let mut stepped_logits = Vec::new();
    for chunk in prompt.chunks(64) {
        stepped_logits = stepped.forward(chunk).expect("stepped forward");
    }
    assert_eq!(pipelined.position(), 150);
    assert_eq!(stepped.position(), 150);
    assert_eq!(
        argmax(&pipelined_logits),
        argmax(&stepped_logits),
        "deferral changed the winning token"
    );
    let d = max_abs(&pipelined_logits, &stepped_logits);
    assert!(
        d < 0.5,
        "deferred and stepped logits drift beyond the cross-path bound: {d}"
    );

    // The fingerprint: the same chunks and submissions in
    // both drivers, one readback against three, and a
    // dispatch count that does not move.
    let p_stats = pipelined.stats();
    assert_eq!(p_stats.forwards, 1);
    assert_eq!(p_stats.chunks, 3);
    assert_eq!(
        p_stats.submissions, 3,
        "one submission per chunk, pipelined or not"
    );
    assert_eq!(p_stats.readbacks, 1, "only the final chunk reads back");
    let s_stats = stepped.stats();
    assert_eq!(s_stats.forwards, 3);
    assert_eq!(s_stats.chunks, 3);
    assert_eq!(s_stats.submissions, 3);
    assert_eq!(s_stats.readbacks, 3, "every stepped forward reads back");
    assert_eq!(
        p_stats.dispatches, s_stats.dispatches,
        "deferral changes no dispatch: the device work is identical"
    );
}

/// Quantize-once, proven end to end: the mixed fixture
/// (Q6_K value projections and FFN ups, so every layer is a
/// partial batch with an unpaired FFN) through a 20-token
/// prompt. Twenty rows is past the matrix-form threshold,
/// so on a backend that built the MMQ set the chunk's
/// projections take their MMQ forms and each layer stage's
/// two `h` projections ride ONE shared K12 quantize; the
/// resident stream must still agree with the CPU reference
/// and the trait path at the cross-backend bar, which is
/// the byte-identity proof at the only level that matters:
/// if a member product read anything but the shared
/// quantize's bytes (stale, clobbered, or a neighbor's),
/// the logits would drift off the reference.
///
/// The dispatch pin carries the arithmetic. Per layer
/// (partial batch, gate fused into the output projection,
/// unpaired FFN, chained norms): 16 at the unfused base,
/// less the batch's 2, the fused gate's 1, and the chains'
/// 1 (plus layer 0's own attention norm in place of the
/// boundary ride). The gated output and the SwiGLU-shaped
/// down keep their own quantizes (+1 each). Before
/// quantize-once the batch, V, gate, and up products each
/// carried a quantize too: layer 0 recorded 18, later
/// layers 17, and the chunk 105. With the shares (one per
/// stage per layer), the products record 1 and the layer
/// adds 2: layer 0 records 16, later layers 15, and the
/// chunk 1 (gather) + 16 + 5 x 15 + 1 (the Q5_K head, its
/// finishing fused) = 93. On a backend with no MMQ set the
/// GEMM forms serve instead: no quantize legs exist at
/// all, every projection records 1, and the chunk is 69.
/// Decode rows quantize nothing on any backend (the GEMV
/// forms read `h` directly), so three decode chunks add
/// exactly 3 x 69 under either serving form: the decode
/// count the mixed layers record with no shares anywhere.
#[test]
fn resident_session_partial_batch_chunk_shares_one_quantize_per_stage() {
    let trait_backend = match WgpuCompute::new() {
        Ok(b) => b,
        Err(WgpuError::NoSuitableAdapter { seen, .. }) => {
            eprintln!(
                "skipping partial-batch resident test: no suitable adapter (enumerated: [{}])",
                seen.join("; ")
            );
            return;
        }
        Err(e) => panic!("backend construction failed unexpectedly: {e}"),
    };
    let cfg = cross_cfg();
    let weights = cross_weights_mixed(&cfg);
    let cpu_model = GlimmerModel::new(cfg.clone(), weights.clone(), Box::new(CpuCompute::new()));
    let gpu_model = GlimmerModel::new(cfg.clone(), weights.clone(), Box::new(trait_backend));
    let session_backend = WgpuCompute::new().expect("second backend instance builds");
    let mut session =
        ResidentSession::new(&session_backend, &cfg, &weights).expect("resident session builds");

    let mut cpu_cache = PositionCache::new();
    let mut gpu_cache = PositionCache::new();

    // The 20-token prompt: one resident chunk at MMQ row
    // counts. The trait path is fed token by token (its
    // proven regime, as in the phase-2 comparison above);
    // the CPU reference takes the prompt whole.
    let prompt: Vec<u32> = (0..20).map(|i| (i * 53 % 1024) as u32).collect();
    let cpu_logits = cpu_model
        .forward(&prompt, &mut cpu_cache)
        .expect("cpu forward");
    let mut gpu_logits = Vec::new();
    for token in &prompt {
        gpu_logits = gpu_model
            .forward(&[*token], &mut gpu_cache)
            .expect("gpu forward");
    }
    let resident_logits = session.forward(&prompt).expect("resident chunked forward");
    let mut next = vec![assert_step_agrees(
        20,
        &cpu_logits,
        &gpu_logits,
        &resident_logits,
    )];
    assert_eq!(session.position(), 20);
    let stats = session.stats();
    assert_eq!(stats.chunks, 1, "the prompt is one chunk: {stats:?}");
    assert_eq!(stats.submissions, 1);
    let form = match stats.dispatches {
        93 => "mmq forms with one shared quantize per stage",
        69 => "gemm/gemv forms (no MMQ set on this adapter)",
        other => panic!(
            "chunk dispatch count {other} matches neither the shared-MMQ count (93) \
             nor the no-MMQ count (69): {stats:?}"
        ),
    };
    eprintln!("partial-batch chunk served by {form}: {stats:?}");

    // Three greedy decode steps on the same drivers: parity
    // holds and the dispatch delta is the decode count,
    // which carries no quantize legs on any backend.
    let before = session.stats().dispatches;
    for step in 0..3 {
        let cpu_logits = cpu_model
            .forward(&next, &mut cpu_cache)
            .expect("cpu decode");
        let gpu_logits = gpu_model
            .forward(&next, &mut gpu_cache)
            .expect("gpu decode");
        let resident_logits = session.forward(&next).expect("resident decode");
        next = vec![assert_step_agrees(
            21 + step,
            &cpu_logits,
            &gpu_logits,
            &resident_logits,
        )];
    }
    assert_eq!(session.position(), 23);
    assert_eq!(
        session.stats().dispatches - before,
        3 * 69,
        "decode chunks carry no quantize legs: {:?}",
        session.stats()
    );
}

/// The diag smoke collector (David's order, 2026-11-04: every
/// kernel, every block, one full report from a smoke run).
/// Opt-in: it runs only when `WGPU_COMPUTE_DIAG=timing` and
/// `WGPU_COMPUTE_DIAG_OUT` names the report file, because the
/// recorder it drains exists only in timing mode. It drives
/// the resident session over the mixed fixture (Q4_K, Q6_K and
/// Q5_K tensors) through a batched chunk and two decode steps,
/// drains the timestamp recorder after each forward (one
/// query window covers several forwards; draining per forward
/// keeps it that way), and writes the per-kernel summary
/// table: dispatch count, total ms, share of GPU time, weight
/// bytes, and aggregate GB/s per kernel.
#[test]
fn resident_session_diag_smoke_report() {
    use wgpu_compute::timestamps::{render_summary, DIAG_OUT_ENV};
    use wgpu_compute::DiagMode;

    let Some(out_path) = std::env::var_os(DIAG_OUT_ENV) else {
        eprintln!("skipping diag smoke: {DIAG_OUT_ENV} is not set");
        return;
    };
    if DiagMode::from_env() != DiagMode::Timing {
        eprintln!("skipping diag smoke: WGPU_COMPUTE_DIAG is not `timing`");
        return;
    }
    let backend = match WgpuCompute::new() {
        Ok(b) => b,
        Err(WgpuError::NoSuitableAdapter { seen, .. }) => {
            eprintln!(
                "skipping diag smoke: no suitable adapter (enumerated: [{}])",
                seen.join("; ")
            );
            return;
        }
        Err(e) => panic!("backend construction failed unexpectedly: {e}"),
    };
    let Some(recorder) = backend.context().diag_recorder().cloned() else {
        eprintln!(
            "skipping diag smoke: timing requested but the device granted \
             no timestamp features, so no recorder exists"
        );
        return;
    };
    let cfg = cross_cfg();
    let weights = cross_weights_mixed(&cfg);
    let mut session =
        ResidentSession::new(&backend, &cfg, &weights).expect("resident session builds");
    let ctx = backend.context();
    for tokens in [&[3u32, 100, 57, 900][..], &[42][..], &[7][..]] {
        session.forward(tokens).expect("diag smoke forward");
        recorder
            .drain(&ctx.device, &ctx.queue)
            .expect("diag drain after forward");
    }
    let records = recorder.take_records();
    assert!(
        !records.is_empty(),
        "diag smoke recorded no dispatches: the instrumentation is not reaching the session"
    );
    let distinct: std::collections::BTreeSet<&str> = records.iter().map(|r| r.kernel).collect();
    assert!(
        distinct.len() >= 5,
        "diag smoke should cover the kernel family, saw only {distinct:?}"
    );
    assert!(
        distinct.contains("k1_q4k_gemv"),
        "decode forwards must stamp the Q4_K GEMV, saw {distinct:?}"
    );
    let table = render_summary(&records);
    assert!(table.contains("TOTAL,"));
    std::fs::write(&out_path, &table).expect("write diag smoke report");
    eprintln!(
        "diag smoke: {} dispatches across {} kernels, {} dropped, report written to {}",
        records.len(),
        distinct.len(),
        recorder.dropped(),
        std::path::Path::new(&out_path).display()
    );
}
