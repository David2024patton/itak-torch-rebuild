//! The DFlash GPU session parity pin: `DflashSession`
//! against the CPU implementation (`dflash::DflashModel`,
//! the arch's correctness oracle) on a synthetic fixture,
//! at the cross-backend bar the resident tests use
//! (per-row argmax agreement and max-abs logit difference
//! under 0.5).
//!
//! The fixture is built so both paths consume THE SAME
//! values: every tensor is synthesized once as stored
//! GGUF bytes (the real file's dtype families: Q4_K
//! projections, Q6_K value/down, F32 norms, a Q5_K
//! borrowed head); the GPU session uploads those bytes,
//! and the CPU model is fed their dequantization. Any
//! divergence is therefore the device's arithmetic, not a
//! fixture mismatch.
//!
//! The fixture's geometry exercises the draft-specific
//! paths the Glimmer fixture cannot: three layers with a
//! sliding/global/sliding pattern and window 3 (so the
//! mask genuinely bites at the block positions used), a
//! post-norm pair on the middle layer, a sparse committed
//! set (gaps in the injected positions), and the borrowed
//! embedding/head state (the fixture draft carries
//! neither of its own).
//!
//! Adapter-bound tests skip with a printed reason where
//! no adapter qualifies, the resident tests' discipline;
//! the raw-load refusal tests are pure CPU and run
//! everywhere.

use std::collections::HashMap;
use std::sync::Arc;

use dflash::{
    DflashCache, DflashConfig, DflashError, DflashModel, TargetTensors, TensorData,
    Weights as CpuWeights,
};
use gguf::ggml::GgmlType;
use muse_glimmer::{Dtype, GlimmerError, Result as GlimmerResult, TensorSource, TensorView};
use wgpu_compute::kernels::k1_q4k_gemv::f32_to_f16_bits;
use wgpu_compute::{DflashRawWeights, DflashSession, DflashSessionConfig, WgpuCompute, WgpuError};

/// A tiny deterministic byte/float source (the
/// cross-backend fixture's generator, same seed
/// discipline as the resident tests).
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
/// blocks, exactly the resident fixture's construction.
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
/// (ql, qh, i8 scales, f16 d), sane magnitudes by
/// construction.
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
            data.push(rng.next_u8() % 5);
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

/// A raw view source over a name map (the loader's shape,
/// minus the file).
struct MapSource {
    tensors: HashMap<String, TensorView>,
}

impl TensorSource for MapSource {
    fn tensor(&self, name: &str) -> GlimmerResult<TensorView> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| GlimmerError::MissingTensor {
                name: name.to_string(),
            })
    }
}

/// The CPU source over the same map: every tensor
/// dequantized from the very bytes the GPU uploads.
struct CpuSource {
    tensors: HashMap<String, TensorView>,
}

fn dequantized(view: &TensorView) -> TensorData {
    let ty = match view.dtype {
        Dtype::F32 => GgmlType::F32,
        Dtype::F16 => GgmlType::F16,
        Dtype::Bf16 => GgmlType::Bf16,
        Dtype::Q4K => GgmlType::Q4K,
        Dtype::Q5K => GgmlType::Q5K,
        Dtype::Q6K => GgmlType::Q6K,
        Dtype::Q8_0 => GgmlType::Q8_0,
        other => panic!("fixture dtype {other:?} has no dequantizer mapping"),
    };
    let count: usize = view.shape.iter().map(|d| *d as usize).product();
    TensorData {
        name: view.name.clone(),
        shape: view.shape.clone(),
        data: gguf::dequant::dequantize(ty, &view.data, count),
    }
}

impl dflash::TensorSource for CpuSource {
    fn tensor(&self, name: &str) -> dflash::Result<TensorData> {
        self.tensors
            .get(name)
            .map(dequantized)
            .ok_or_else(|| DflashError::MissingTensor {
                name: name.to_string(),
            })
    }

    fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

/// The fixture config: real head geometry scaled down to
/// kernel-legal widths (every Q4_K/Q6_K input a whole
/// number of 256-blocks), a sliding/global/sliding
/// pattern with window 3, two target layers over a
/// 256-wide target.
fn fixture_cfg() -> DflashConfig {
    DflashConfig {
        name: None,
        block_count: 3,
        context_length: 128,
        hidden: 256,
        ffn_hidden: 512,
        q_heads: 4,
        kv_heads: 2,
        key_length: 64,
        value_length: 64,
        rope_freq_base: 10_000.0,
        rms_eps: 1e-5,
        block_size: 4,
        target_layers: vec![0, 2],
        sliding_window: 3,
        sliding_pattern: vec![true, false, true],
        causal_attention: false,
        mask_token_id: 511,
        vocab_size: 512,
    }
}

/// Build the fixture's draft tensor map (no embedding,
/// no head: the borrowed state), plus the target's two
/// tables as separate views.
fn fixture_tensors(cfg: &DflashConfig) -> (HashMap<String, TensorView>, TensorView, TensorView) {
    let hidden = cfg.hidden as usize;
    let ffn = cfg.ffn_hidden as usize;
    let q_w = cfg.q_width() as usize;
    let k_w = cfg.k_width() as usize;
    let out_w = cfg.attn_out_width() as usize;
    let vocab = cfg.vocab_size as usize;
    let head_dim = cfg.key_length as usize;
    let cond = cfg.target_layers.len() * hidden;

    let mut rng = Rng(0xd15c_0de5);
    let mut tensors: HashMap<String, TensorView> = HashMap::new();
    let mut put = |view: TensorView| {
        tensors.insert(view.name.clone(), view);
    };
    put(q4k_tensor(&mut rng, "fc.weight", cond, hidden));
    put(f32_tensor(
        &mut rng,
        "enc.output_norm.weight",
        &[hidden as u64],
        1.0,
    ));
    put(f32_tensor(
        &mut rng,
        "output_norm.weight",
        &[hidden as u64],
        1.0,
    ));
    for layer in 0..cfg.block_count {
        let p = |suffix: &str| format!("blk.{layer}.{suffix}");
        put(f32_tensor(
            &mut rng,
            &p("attn_norm.weight"),
            &[hidden as u64],
            1.0,
        ));
        put(q4k_tensor(&mut rng, &p("attn_q.weight"), hidden, q_w));
        put(q4k_tensor(&mut rng, &p("attn_k.weight"), hidden, k_w));
        put(q6k_tensor(&mut rng, &p("attn_v.weight"), hidden, k_w));
        put(f32_tensor(
            &mut rng,
            &p("attn_q_norm.weight"),
            &[head_dim as u64],
            1.0,
        ));
        put(f32_tensor(
            &mut rng,
            &p("attn_k_norm.weight"),
            &[head_dim as u64],
            1.0,
        ));
        put(q4k_tensor(
            &mut rng,
            &p("attn_output.weight"),
            out_w,
            hidden,
        ));
        if layer == 1 {
            put(f32_tensor(
                &mut rng,
                &p("post_attention_norm.weight"),
                &[hidden as u64],
                1.0,
            ));
            put(f32_tensor(
                &mut rng,
                &p("post_ffw_norm.weight"),
                &[hidden as u64],
                1.0,
            ));
        }
        put(f32_tensor(
            &mut rng,
            &p("ffn_norm.weight"),
            &[hidden as u64],
            1.0,
        ));
        put(q4k_tensor(&mut rng, &p("ffn_gate.weight"), hidden, ffn));
        put(q4k_tensor(&mut rng, &p("ffn_up.weight"), hidden, ffn));
        put(q6k_tensor(&mut rng, &p("ffn_down.weight"), ffn, hidden));
    }
    let embedding = q4k_tensor(&mut rng, "token_embd.weight", hidden, vocab);
    let output = q5k_tensor(&mut rng, "output.weight", hidden, vocab);
    (tensors, embedding, output)
}

/// Deterministic conditioning features for `positions`:
/// values in a tame range, distinct per row.
fn features_for(cfg: &DflashConfig, positions: &[usize]) -> Vec<f32> {
    let width = cfg.target_layers.len() * cfg.hidden as usize;
    let mut rng = Rng(0xfeed_0001 + positions.len() as u64);
    (0..positions.len() * width)
        .map(|_| rng.next_f32())
        .collect()
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

/// Build a backend or skip the test: the resident tests'
/// discipline (a missing adapter is an environment fact,
/// not a failure).
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

/// The raw load's refusal shapes, CPU-only: they must
/// name the same errors the CPU load names for the same
/// broken files.
#[test]
fn raw_load_refusals_match_the_cpu_load() {
    let cfg = fixture_cfg();
    let (tensors, _, _) = fixture_tensors(&cfg);

    // Missing a required tensor.
    let mut broken = tensors.clone();
    broken.remove("blk.1.ffn_up.weight");
    let err = DflashRawWeights::load(&MapSource { tensors: broken }, &cfg)
        .expect_err("a missing tensor must refuse");
    assert!(
        matches!(&err, DflashError::MissingTensor { name } if name == "blk.1.ffn_up.weight"),
        "got {err}"
    );

    // A variant gate tensor present.
    let mut variant = tensors.clone();
    variant.insert(
        "markov_w1.weight".to_string(),
        f32_tensor(&mut Rng(7), "markov_w1.weight", &[4], 0.0),
    );
    let err = DflashRawWeights::load(&MapSource { tensors: variant }, &cfg)
        .expect_err("a variant file must refuse");
    assert!(
        matches!(&err, DflashError::UnsupportedFeature { name, .. } if name == "markov_w1.weight"),
        "got {err}"
    );

    // A wrong shape.
    let mut misshaped = tensors.clone();
    misshaped.insert(
        "blk.0.attn_q.weight".to_string(),
        q4k_tensor(&mut Rng(9), "blk.0.attn_q.weight", 256, 512),
    );
    let err = DflashRawWeights::load(&MapSource { tensors: misshaped }, &cfg)
        .expect_err("a misshaped tensor must refuse");
    assert!(
        matches!(&err, DflashError::ShapeMismatch { name, .. } if name == "blk.0.attn_q.weight"),
        "got {err}"
    );

    // fc width not a whole multiple of the target layer
    // count (3 target layers against a 512-wide fc).
    let mut cfg3 = fixture_cfg();
    cfg3.target_layers = vec![0, 1, 2];
    let err = DflashRawWeights::load(
        &MapSource {
            tensors: tensors.clone(),
        },
        &cfg3,
    )
    .expect_err("a non-multiple fc width must refuse");
    assert!(
        matches!(&err, DflashError::InvalidMetadata { key, .. } if key == "dflash.target_layers"),
        "got {err}"
    );

    // The intact fixture loads, and derives the same
    // scalars the CPU load derives.
    let raw = DflashRawWeights::load(
        &MapSource {
            tensors: tensors.clone(),
        },
        &cfg,
    )
    .expect("the intact fixture loads");
    assert_eq!(raw.conditioning_width(), 512);
    assert_eq!(raw.target_hidden, 256);
    assert!(raw.embedding.is_none() && raw.output.is_none());
    let cpu = CpuWeights::load(&CpuSource { tensors }, &cfg)
        .expect("the CPU load accepts the same fixture");
    assert_eq!(cpu.conditioning_width(), raw.conditioning_width());
    assert_eq!(cpu.target_hidden, raw.target_hidden);
}

/// The full parity run: fuse, inject, and block logits
/// against the CPU model, over a contiguous commit and
/// over a sparse one, with the session's refusal shapes
/// asserted at the end.
#[test]
fn dflash_session_matches_the_cpu_model() {
    let Some(backend) = backend_or_skip("dflash session parity test") else {
        return;
    };
    let cfg = fixture_cfg();
    let (tensors, emb_view, out_view) = fixture_tensors(&cfg);
    let raw = DflashRawWeights::load(
        &MapSource {
            tensors: tensors.clone(),
        },
        &cfg,
    )
    .expect("raw weights load");
    let cpu_weights = CpuWeights::load(&CpuSource { tensors }, &cfg).expect("cpu weights load");
    let cpu_model = DflashModel::new(cfg.clone(), cpu_weights);
    let emb_cpu = dequantized(&emb_view);
    let out_cpu = dequantized(&out_view);
    let target = TargetTensors {
        embedding: Some(&emb_cpu),
        output: Some(&out_cpu),
    };
    let vocab = cfg.vocab_size as usize;

    let mut session = DflashSession::new(
        &backend,
        &cfg,
        &raw,
        Some(&emb_view),
        Some(&out_view),
        DflashSessionConfig { kv_capacity: 64 },
    )
    .expect("draft session builds");
    assert_eq!(session.committed_len(), 0);
    assert_eq!(session.conditioning_width(), 512);

    // Round 1: a contiguous commit of eight positions,
    // then a noise block at 8.
    let positions: Vec<usize> = (0..8).collect();
    let features = features_for(&cfg, &positions);
    let cpu_fused = cpu_model
        .fuse(&features, positions.len())
        .expect("cpu fuse");
    let gpu_fused = session.fuse(&features, positions.len()).expect("gpu fuse");
    let d = max_abs(&cpu_fused, &gpu_fused);
    assert!(d < 0.5, "fuse drift beyond the cross-backend bar: {d}");

    let mut cache = DflashCache::new(&cfg);
    cpu_model
        .inject(&mut cache, &features, &positions)
        .expect("cpu inject");
    session.inject(&features, &positions).expect("gpu inject");
    assert_eq!(session.committed_len(), cache.len());
    assert_eq!(session.positions(), positions);

    let block = cpu_model.noise_block(42);
    let cpu_logits = cpu_model
        .forward_block(&cache, &block, 8, &target)
        .expect("cpu block");
    let gpu_logits = session.forward_block(&block, 8).expect("gpu block");
    assert_eq!(gpu_logits.len(), block.len() * vocab);
    for row in 0..block.len() {
        let cpu_row = &cpu_logits[row * vocab..(row + 1) * vocab];
        let gpu_row = &gpu_logits[row * vocab..(row + 1) * vocab];
        assert_eq!(
            argmax(cpu_row),
            argmax(gpu_row),
            "block row {row} top-1 disagreement"
        );
        let d = max_abs(cpu_row, gpu_row);
        assert!(d < 0.5, "block row {row} logit drift beyond the bar: {d}");
        assert!(
            gpu_row.iter().all(|v| v.is_finite()),
            "block row {row} produced non-finite logits"
        );
    }
    // The drafting pass wrote nothing: both sides still
    // hold exactly the injected set.
    assert_eq!(session.committed_len(), 8);

    // Round 2: a sparse commit (positions 16, 19, 23)
    // into a fresh pair, then a block at 24. The gaps are
    // the case a ring-addressed cache cannot state and
    // the position-addressed one must get right.
    let mut session2 = DflashSession::new(
        &backend,
        &cfg,
        &raw,
        Some(&emb_view),
        Some(&out_view),
        DflashSessionConfig { kv_capacity: 64 },
    )
    .expect("second session builds");
    let sparse = vec![16usize, 19, 23];
    let features2 = features_for(&cfg, &sparse);
    let mut cache2 = DflashCache::new(&cfg);
    cpu_model
        .inject(&mut cache2, &features2, &sparse)
        .expect("cpu sparse inject");
    session2
        .inject(&features2, &sparse)
        .expect("gpu sparse inject");
    assert_eq!(session2.positions(), sparse);
    let cpu_logits2 = cpu_model
        .forward_block(&cache2, &block, 24, &target)
        .expect("cpu sparse block");
    let gpu_logits2 = session2
        .forward_block(&block, 24)
        .expect("gpu sparse block");
    for row in 0..block.len() {
        let cpu_row = &cpu_logits2[row * vocab..(row + 1) * vocab];
        let gpu_row = &gpu_logits2[row * vocab..(row + 1) * vocab];
        assert_eq!(
            argmax(cpu_row),
            argmax(gpu_row),
            "sparse block row {row} top-1 disagreement"
        );
        let d = max_abs(cpu_row, gpu_row);
        assert!(d < 0.5, "sparse block row {row} drift beyond the bar: {d}");
    }

    // Refusals, in the CPU pass's own names and order.
    let err = session
        .forward_block(&[], 0)
        .expect_err("empty block refuses");
    assert!(matches!(err, DflashError::EmptyInput), "got {err}");
    let err = session
        .forward_block(&[1, 2, 3, 4, 5], 0)
        .expect_err("an over-long block refuses");
    assert!(matches!(err, DflashError::BlockTooLong { .. }), "got {err}");
    let err = session
        .forward_block(&[cfg.vocab_size], 0)
        .expect_err("an out-of-range token refuses");
    assert!(
        matches!(err, DflashError::TokenOutOfRange { .. }),
        "got {err}"
    );
    let err = session
        .inject(&features, &[3, 3])
        .expect_err("non-increasing positions refuse");
    assert!(
        matches!(err, DflashError::PositionConflict { .. }),
        "got {err}"
    );
    let err = session
        .inject(&features_for(&cfg, &[100]), &[100])
        .expect_err("a position past capacity refuses by name");
    assert!(
        matches!(&err, DflashError::StepFailed { step, .. } if *step == "dflash inject"),
        "got {err}"
    );

    // Reset returns the session to the empty-cache state.
    session.reset();
    assert_eq!(session.committed_len(), 0);
    assert!(session.positions().is_empty());
}

/// Session construction refusals that precede adapter
/// work where the code allows: a zero capacity and a
/// missing borrowed table refuse by name (both need the
/// backend handle, so they ride the same skip).
#[test]
fn dflash_session_construction_refusals() {
    let Some(backend) = backend_or_skip("dflash session construction test") else {
        return;
    };
    let cfg = fixture_cfg();
    let (tensors, emb_view, out_view) = fixture_tensors(&cfg);
    let raw = DflashRawWeights::load(&MapSource { tensors }, &cfg).expect("raw weights load");

    let err = DflashSession::new(
        &backend,
        &cfg,
        &raw,
        Some(&emb_view),
        Some(&out_view),
        DflashSessionConfig { kv_capacity: 0 },
    )
    .expect_err("zero capacity must refuse");
    assert!(matches!(err, DflashError::StepFailed { .. }), "got {err}");

    let err = DflashSession::new(
        &backend,
        &cfg,
        &raw,
        None,
        Some(&out_view),
        DflashSessionConfig::default(),
    )
    .expect_err("a missing borrowed embedding must refuse by name");
    assert!(
        matches!(
            &err,
            DflashError::MissingTargetTensors {
                which: "token_embd.weight"
            }
        ),
        "got {err}"
    );
}

/// The K15 rotate against `torch_math::rope` directly:
/// NEOX pairing, per-row positions from the array,
/// in place.
#[test]
fn k15_rotate_matches_torch_math() {
    let Some(backend) = backend_or_skip("k15 rotate test") else {
        return;
    };
    let ctx = backend.context();
    let pipeline = wgpu_compute::K15Pipeline::new(&ctx.device).expect("k15 pipelines build");
    let (rows, heads, head_dim) = (3usize, 2usize, 8usize);
    let positions = [0usize, 5, 127];
    let mut rng = Rng(0x207a7e);
    let input: Vec<f32> = (0..rows * heads * head_dim)
        .map(|_| rng.next_f32())
        .collect();
    let x = ctx.upload_f32("k15 rotate input", &input);
    let pos_bytes: Vec<u8> = positions
        .iter()
        .flat_map(|p| (*p as u32).to_le_bytes())
        .collect();
    let pos_buf = ctx.upload_bytes("k15 rotate positions", &pos_bytes);
    let inv_freq = pipeline.upload_inv_freq(ctx, head_dim, 10_000.0);
    let call = pipeline
        .prepare_rotate(ctx, &x, &pos_buf, &inv_freq, rows, heads, head_dim)
        .expect("rotate prepares");
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("k15 rotate test"),
        });
    call.encode(&mut encoder);
    ctx.queue.submit([encoder.finish()]);
    let got = ctx
        .read_f32(&x, input.len())
        .expect("rotate output reads back");
    for (r, &pos) in positions.iter().enumerate() {
        for h in 0..heads {
            let row_in = &input[(r * heads + h) * head_dim..(r * heads + h + 1) * head_dim];
            let want = torch_math::rope(row_in, pos, 10_000.0, head_dim).expect("cpu rope");
            let got_row = &got[(r * heads + h) * head_dim..(r * heads + h + 1) * head_dim];
            let d = max_abs(&want, got_row);
            assert!(d < 1e-4, "rotate row {r} head {h} drift: {d}");
        }
    }
}
