//! K4 through K8 against the CPU references, kernel by kernel:
//! K4's norms and add against `cpu-compute` (and the fused
//! norm+residual form against the trait composition), K5's
//! rotate against `torch-math`'s `rope_interleaved` and its
//! append against the cache slot law, K6's attention against
//! `cpu-compute`'s whole attention step over a mixed
//! sliding/global pattern with ring eviction, K7's activations
//! against `torch-math` and `cpu-compute`, K8's gather against
//! `cpu-compute`'s embed for every dtype both backends accept,
//! and K8's finishing kernel against
//! `muse_glimmer::softcap_scale` at the real 202,048 vocab
//! width.
//!
//! These tests need a GPU. The pipeline-compile test takes any
//! adapter wgpu enumerates (shader validity is not a selection
//! policy question); the numeric tests take the backend's
//! default policy, a discrete Vulkan GPU, and skip with a
//! printed reason where none qualifies (this workstation, CI
//! boxes), exactly like the K1 tests. On David's server the
//! 7900 XTX answers and every test runs for real.

use std::sync::Arc;

use cpu_compute::CpuCompute;
use muse_glimmer::{
    AttentionRequest, Cache, Compute, Dtype, GlimmerConfig, GlimmerError, LayerKind, PatternSource,
    PositionCache, TensorView,
};
use wgpu_compute::kernels::k1_q4k_gemv::f32_to_f16_bits;
use wgpu_compute::kernels::k4_rmsnorm::K4Pipeline;
use wgpu_compute::kernels::k5_rope_kv::K5Pipeline;
use wgpu_compute::kernels::k8_head::K8Pipeline;
use wgpu_compute::{WgpuCompute, WgpuError};

/// A backend, or `None` after printing why this machine cannot
/// provide one (the K1 tests' pattern).
fn backend_or_skip() -> Option<WgpuCompute> {
    match WgpuCompute::new() {
        Ok(b) => Some(b),
        Err(WgpuError::NoSuitableAdapter { seen, .. }) => {
            eprintln!(
                "skipping GPU test: no suitable adapter (enumerated: [{}])",
                seen.join("; ")
            );
            None
        }
        Err(e) => panic!("backend construction failed unexpectedly: {e}"),
    }
}

/// Deterministic pseudo-random values in [-0.5, 0.5).
fn lcg(seed: u64, count: usize) -> Vec<f32> {
    let mut state = seed.max(1);
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as u32) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

/// The largest absolute difference between two equal-length
/// vectors (panics on a length mismatch: a shape bug must not
/// hide inside a tolerance).
fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch in comparison");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// A config literal with the fields a given test needs set;
/// everything else holds the tiny values the CPU backend's own
/// tests use.
fn test_cfg() -> GlimmerConfig {
    GlimmerConfig {
        name: None,
        block_count: 3,
        context_length: 256,
        hidden: 256,
        ffn_hidden: 512,
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
        vocab_size: 8,
        layers: vec![
            LayerKind::SlidingWindow,
            LayerKind::Global,
            LayerKind::SlidingWindow,
        ],
        pattern_source: PatternSource::MeasuredFallback,
    }
}

fn f32_view(name: &str, shape: &[u64], values: &[f32]) -> TensorView {
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    TensorView {
        name: name.to_string(),
        dtype: Dtype::F32,
        shape: shape.to_vec(),
        data: Arc::from(data.into_boxed_slice()),
    }
}

#[test]
fn k4_to_k8_shaders_compile_on_any_available_adapter() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    let Some(adapter) = adapters.into_iter().next() else {
        eprintln!("skipping K4-K8 shader compile test: wgpu enumerates no adapters here");
        return;
    };
    let info = adapter.get_info();
    let (device, _queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("wgpu-compute K4-K8 shader compile test"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .expect("device for shader compile test");
    K4Pipeline::new(&device).unwrap_or_else(|e| panic!("K4 must build on {}: {e}", info.name));
    K5Pipeline::new(&device).unwrap_or_else(|e| panic!("K5 must build on {}: {e}", info.name));
    wgpu_compute::K6Pipeline::new(&device)
        .unwrap_or_else(|e| panic!("K6 must build on {}: {e}", info.name));
    wgpu_compute::K7Pipeline::new(&device)
        .unwrap_or_else(|e| panic!("K7 must build on {}: {e}", info.name));
    K8Pipeline::new(&device).unwrap_or_else(|e| panic!("K8 must build on {}: {e}", info.name));
    eprintln!(
        "K4-K8 shipped shaders compiled on {} ({:?})",
        info.name, info.backend
    );
}

#[test]
fn k4_norms_and_add_match_cpu() {
    let Some(gpu) = backend_or_skip() else {
        return;
    };
    let cpu = CpuCompute::new();
    let rows = 3;
    let width = 512;
    let x = lcg(41, rows * width);
    let w_values = lcg(42, width);
    let w = f32_view("norm.weight", &[width as u64], &w_values);

    for eps in [1e-5f32, 1e-8] {
        let want = cpu.rms_norm(&x, &w, rows, eps).expect("cpu norm");
        let got = gpu.rms_norm(&x, &w, rows, eps).expect("gpu norm");
        let d = max_abs(&got, &want);
        eprintln!("k4 rms_norm eps={eps}: max abs diff {d:.3e}");
        assert!(d < 1e-4, "rms_norm drifted: {d}");
    }
    let want = cpu.rms_norm_plain(&x, rows, 1e-5).expect("cpu plain");
    let got = gpu.rms_norm_plain(&x, rows, 1e-5).expect("gpu plain");
    let d = max_abs(&got, &want);
    eprintln!("k4 rms_norm_plain: max abs diff {d:.3e}");
    assert!(d < 1e-4, "rms_norm_plain drifted: {d}");

    let b = lcg(43, rows * width);
    let want = cpu.add(&x, &b).expect("cpu add");
    let got = gpu.add(&x, &b).expect("gpu add");
    assert_eq!(got, want, "the residual add is exact");

    // The fused norm+residual form against the trait composition
    // (norm the branch, then add): this is the settled post-norm
    // order, so the fused kernel must equal it.
    let ctx = gpu.context();
    let k4 = K4Pipeline::new(&ctx.device).expect("k4 pipeline");
    let branch = lcg(44, rows * width);
    let composed = {
        let normed = gpu
            .rms_norm(&branch, &w, rows, 1e-8)
            .expect("gpu branch norm");
        gpu.add(&x, &normed).expect("gpu add")
    };
    let fused = k4
        .norm_residual(ctx, &x, &branch, &w_values, rows, 1e-8)
        .expect("fused norm+residual");
    let d = max_abs(&fused, &composed);
    eprintln!("k4 norm_residual fused vs composed: max abs diff {d:.3e}");
    assert!(d < 1e-5, "fused norm+residual drifted: {d}");
}

#[test]
fn k7_activations_match_references() {
    let Some(gpu) = backend_or_skip() else {
        return;
    };
    let cpu = CpuCompute::new();
    let gate = lcg(51, 4096);
    let up = lcg(52, 4096);

    // The attention gate through the trait on both backends.
    let want = cpu.attn_gate(&up, &gate).expect("cpu gate");
    let got = gpu.attn_gate(&up, &gate).expect("gpu gate");
    let d = max_abs(&got, &want);
    eprintln!("k7 attn_gate: max abs diff {d:.3e}");
    assert!(d < 1e-5, "attn_gate drifted: {d}");

    // SwiGLU through the K7 pipeline directly (the trait's
    // swiglu_ffn wraps it in projections the sibling tests cover).
    let ctx = gpu.context();
    let k7 = wgpu_compute::K7Pipeline::new(&ctx.device).expect("k7 pipeline");
    let want = torch_math::swiglu(&gate, &up).expect("torch-math swiglu");
    let got = k7.swiglu(ctx, &gate, &up).expect("gpu swiglu");
    let d = max_abs(&got, &want);
    eprintln!("k7 swiglu: max abs diff {d:.3e}");
    assert!(d < 1e-5, "swiglu drifted: {d}");
}

#[test]
fn k5_rotate_matches_torch_math_and_append_hits_cache_slots() {
    let Some(gpu) = backend_or_skip() else {
        return;
    };
    let ctx = gpu.context();
    let k5 = K5Pipeline::new(&ctx.device).expect("k5 pipeline");
    let heads = 32usize;
    let head_dim = 128usize;
    let width = heads * head_dim;
    let rows = 3usize;
    let start = 5usize;
    let x = lcg(61, rows * width);

    let buf = ctx.upload_f32("k5 test q", &x);
    k5.rotate(ctx, &buf, rows, heads, head_dim, start, 500_000.0)
        .expect("rotate");
    let got = ctx.read_f32(&buf, x.len()).expect("readback");
    let mut want = Vec::with_capacity(x.len());
    for r in 0..rows {
        for h in 0..heads {
            let rotated = torch_math::rope_interleaved(
                &x[r * width + h * head_dim..r * width + (h + 1) * head_dim],
                start + r,
                500_000.0,
                head_dim,
            )
            .expect("torch-math rope");
            want.extend_from_slice(&rotated);
        }
    }
    let d = max_abs(&got, &want);
    eprintln!("k5 rotate (start {start}, base 500000): max abs diff {d:.3e}");
    assert!(d < 1e-3, "rotate drifted: {d}");

    // Append: windowed layer, window 4, positions 6 and 7 land in
    // ring slots 2 and 3; a global layer's positions 6 and 7 land
    // in slots 6 and 7.
    let kv_width = 256usize;
    let k_rows = lcg(62, 2 * kv_width);
    let v_rows = lcg(63, 2 * kv_width);
    let k_src = ctx.upload_f32("k5 test k", &k_rows);
    let v_src = ctx.upload_f32("k5 test v", &v_rows);
    for (window, slots) in [(Some(4usize), [2usize, 3]), (None, [6usize, 7])] {
        let capacity = window.unwrap_or(8);
        let k_dst = ctx.scratch_buffer("k5 test k dst", (capacity * kv_width * 4) as u64);
        let v_dst = ctx.scratch_buffer("k5 test v dst", (capacity * kv_width * 4) as u64);
        k5.append(ctx, &k_src, &v_src, &k_dst, &v_dst, 2, kv_width, 6, window)
            .expect("append");
        let got_k = ctx.read_f32(&k_dst, capacity * kv_width).expect("k readback");
        let got_v = ctx.read_f32(&v_dst, capacity * kv_width).expect("v readback");
        for (i, slot) in slots.iter().enumerate() {
            assert_eq!(
                &got_k[slot * kv_width..(slot + 1) * kv_width],
                &k_rows[i * kv_width..(i + 1) * kv_width],
                "window {window:?}: K row {i} must sit in slot {slot}"
            );
            assert_eq!(
                &got_v[slot * kv_width..(slot + 1) * kv_width],
                &v_rows[i * kv_width..(i + 1) * kv_width],
                "window {window:?}: V row {i} must sit in slot {slot}"
            );
        }
        eprintln!("k5 append window {window:?}: rows landed in slots {slots:?}");
    }
}

#[test]
fn k6_attention_matches_cpu_step_for_step() {
    let Some(gpu) = backend_or_skip() else {
        return;
    };
    let cpu = CpuCompute::new();
    let cfg = test_cfg();
    let q_width = cfg.q_width() as usize;
    let k_width = cfg.k_width() as usize;
    let v_width = cfg.v_width() as usize;

    let mut cpu_cache = PositionCache::new();
    let mut gpu_cache = PositionCache::new();
    // A 3-token prefill, then decode steps out to position 6:
    // the sliding window is 4, so the ring evicts and the window
    // mask's edge is exercised on the sliding layers.
    let chunks = [3usize, 1, 1, 1, 1];
    let mut step = 0usize;
    for (chunk_idx, &rows) in chunks.iter().enumerate() {
        let q = lcg(100 + chunk_idx as u64, rows * q_width);
        let k = lcg(200 + chunk_idx as u64, rows * k_width);
        let v = lcg(300 + chunk_idx as u64, rows * v_width);
        for layer in 0..cfg.layers.len() {
            let req_cpu = AttentionRequest {
                q: &q,
                k: &k,
                v: &v,
                layer,
                kind: cfg.layers[layer],
                cfg: &cfg,
            };
            let want = cpu
                .attention(&req_cpu, &mut cpu_cache)
                .expect("cpu attention");
            let req_gpu = AttentionRequest {
                q: &q,
                k: &k,
                v: &v,
                layer,
                kind: cfg.layers[layer],
                cfg: &cfg,
            };
            let got = gpu
                .attention(&req_gpu, &mut gpu_cache)
                .expect("gpu attention");
            let d = max_abs(&got, &want);
            eprintln!(
                "k6 attention chunk {chunk_idx} layer {layer} ({:?}): max abs diff {d:.3e}",
                cfg.layers[layer]
            );
            assert!(d < 2e-3, "attention drifted at chunk {chunk_idx} layer {layer}: {d}");
            step += 1;
        }
        cpu_cache.advance(rows);
        gpu_cache.advance(rows);
    }
    assert_eq!(step, 15);
    eprintln!("k6 attention: prefill + 4 decode steps x 3 layers all matched");
}

/// One quantized embedding table of `vocab` rows x `hidden`
/// channels for `dtype`, built from deterministic payload bytes
/// with small scale headers, plus the config that describes it.
fn embed_table(dtype: Dtype, hidden: usize, vocab: usize) -> (TensorView, GlimmerConfig) {
    let mut cfg = test_cfg();
    cfg.hidden = hidden as u32;
    cfg.vocab_size = vocab as u32;
    let bytes: Vec<u8> = match dtype {
        Dtype::F32 => lcg(71, vocab * hidden)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        Dtype::F16 => lcg(72, vocab * hidden)
            .iter()
            .flat_map(|v| f32_to_f16_bits(*v).to_le_bytes())
            .collect(),
        Dtype::Bf16 => lcg(73, vocab * hidden)
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect(),
        Dtype::Q8_0 => {
            // 34-byte blocks of 32: f16 scale 0.05, signed bytes.
            let mut out = Vec::new();
            let mut payload = lcg(74, vocab * hidden).into_iter();
            for _ in 0..(vocab * hidden / 32) {
                out.extend_from_slice(&f32_to_f16_bits(0.05).to_le_bytes());
                for _ in 0..32 {
                    let val = payload.next().expect("payload");
                    out.push((val * 100.0) as i8 as u8);
                }
            }
            out
        }
        Dtype::Q4K | Dtype::Q5K | Dtype::Q6K => {
            let block_bytes = match dtype {
                Dtype::Q4K => 144,
                Dtype::Q5K => 176,
                _ => 210,
            };
            let blocks = vocab * hidden / 256;
            let mut out = Vec::with_capacity(blocks * block_bytes);
            let mut state = 88_172_645_463_325_252_u64;
            let mut next = move || {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 56) as u8
            };
            for _ in 0..blocks {
                // Header: d and dmin as small f16s at the offsets
                // each format reads them from. Q6_K carries only
                // d, at the block's end; the K formats lead with
                // the pair.
                let mut block = vec![0u8; block_bytes];
                if dtype == Dtype::Q6K {
                    for b in block.iter_mut() {
                        *b = next() & 0x3f;
                    }
                    block[208..210].copy_from_slice(&f32_to_f16_bits(0.02).to_le_bytes());
                } else {
                    block[0..2].copy_from_slice(&f32_to_f16_bits(0.02).to_le_bytes());
                    block[2..4].copy_from_slice(&f32_to_f16_bits(0.005).to_le_bytes());
                    for b in block.iter_mut().skip(4) {
                        *b = next();
                    }
                    // Keep the packed 6-bit scales small.
                    for b in block.iter_mut().take(16).skip(4) {
                        *b &= 0x07;
                    }
                }
                out.extend_from_slice(&block);
            }
            out
        }
        other => panic!("embed_table does not build {other:?}"),
    };
    let view = TensorView {
        name: "token_embd.weight".to_string(),
        dtype,
        shape: vec![hidden as u64, vocab as u64],
        data: Arc::from(bytes.into_boxed_slice()),
    };
    (view, cfg)
}

#[test]
fn k8_embed_matches_cpu_for_every_shared_dtype() {
    let Some(gpu) = backend_or_skip() else {
        return;
    };
    let cpu = CpuCompute::new();
    let tokens = [0u32, 3, 7, 5];
    for dtype in [
        Dtype::F32,
        Dtype::F16,
        Dtype::Bf16,
        Dtype::Q8_0,
        Dtype::Q4K,
        Dtype::Q5K,
        Dtype::Q6K,
    ] {
        let (table, cfg) = embed_table(dtype, 256, 8);
        let want = cpu.embed(&table, &tokens, &cfg).expect("cpu embed");
        let got = gpu.embed(&table, &tokens, &cfg).expect("gpu embed");
        let d = max_abs(&got, &want);
        eprintln!("k8 embed {dtype:?}: max abs diff {d:.3e}");
        assert!(d < 1e-4, "embed drifted for {dtype:?}: {d}");
    }

    // Refusals keep cpu-compute's boundary: a dtype outside the
    // shared set refuses by name.
    let (mut table, cfg) = embed_table(Dtype::F32, 256, 8);
    table.dtype = Dtype::Q4_0;
    let err = gpu
        .embed(&table, &[0], &cfg)
        .expect_err("Q4_0 embedding must refuse");
    assert!(
        matches!(err, GlimmerError::StepFailed { step: "embed", .. }),
        "unexpected refusal: {err:?}"
    );
}

#[test]
fn k8_finish_matches_softcap_scale_at_real_vocab_width() {
    let Some(gpu) = backend_or_skip() else {
        return;
    };
    let ctx = gpu.context();
    let k8 = K8Pipeline::new(&ctx.device).expect("k8 pipeline");
    let logits = lcg(81, 202_048)
        .iter()
        .map(|v| v * 80.0)
        .collect::<Vec<f32>>();
    let mut want = logits.clone();
    muse_glimmer::softcap_scale(&mut want, 20.0, 0.196_116_13);
    let got = k8
        .finish(ctx, &logits, 0.196_116_13, 20.0)
        .expect("gpu finish");
    let d = max_abs(&got, &want);
    eprintln!("k8 finish over 202048 logits: max abs diff {d:.3e}");
    assert!(d < 1e-3, "finish drifted: {d}");
}
