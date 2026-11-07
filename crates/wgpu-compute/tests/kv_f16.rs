//! The f16 KV storage pins (bench/f16-kv-2026-11-07.md):
//! the resident session can store its per-layer K and V
//! buffers as packed f16 instead of f32, behind
//! `WGPU_COMPUTE_KV_F16` (exact "1" only, default off).
//!
//! Three legs:
//!
//! - **Round-trip bounds (host, runs everywhere).** The
//!   conversion the shaders perform (`pack2x16float` on
//!   write, `unpack2x16float` on read) is mirrored on
//!   the host by the crate's own `f32_to_f16_bits` and
//!   gguf's `f16_to_f32`, and the per-element error is
//!   measured over a deterministic sweep: every
//!   representable f16 value round-trips exactly, and
//!   arbitrary f32 inputs stay inside the half-ulp bound
//!   (relative error at most 2^-11 for normals, absolute
//!   error at most 2^-25 for subnormal results). The
//!   packed word layout (even element in the low half)
//!   is pinned the same way.
//! - **Memory math (host, runs everywhere).** The
//!   full-context KV bytes are computed FROM the real
//!   Glimmer config shape by
//!   `full_context_kv_bytes`, never hardcoded in the
//!   formula: 3,653,238,784 bytes in f32 (K and V, 13
//!   global layers at 131,072 rows plus 39 windowed at
//!   2,048, 2 KV heads at head dim 128 per side), and
//!   exactly half, 1,826,619,392, in f16.
//! - **Session fixture (device; skips where no adapter
//!   qualifies).** The resident-session cross fixture
//!   (hidden 512, six layers in a sliding/global mix,
//!   window 4, both weight mixes) runs twice on the
//!   same weights: once on a backend built with the
//!   switch unset (f32 KV) and once on a backend built
//!   with the switch at "1" (f16 KV). Greedy feeding is
//!   shared (the f32 session's argmax feeds both, so
//!   the sequences cannot diverge), top-1 must agree at
//!   every step, and the measured max logit delta is
//!   printed. The assertion ceiling is the crate's
//!   cross-backend bar of 0.5, used here as a breakage
//!   tripwire and disclosed as such: the delta the card
//!   run prints replaces it (see the bench note). A
//!   second f32 session must reproduce the first
//!   session's logits bit for bit, proving the harness
//!   itself is deterministic before any cross-format
//!   delta is believed. The same test also pins the
//!   two-pass form's named refusal of an f16 layer.

use std::collections::HashMap;
use std::sync::Arc;

use muse_glimmer::{
    expected_tensor_names, Dtype, GlimmerConfig, GlimmerError, LayerKind, PatternSource, Result,
    TensorSource, TensorView, Weights,
};
use wgpu_compute::kernels::k1_q4k_gemv::f32_to_f16_bits;
use wgpu_compute::kernels::k6_attention::{
    full_context_kv_bytes, AttendGeom, DeviceKv, K6Pipeline, KvStorage,
};
use wgpu_compute::{ResidentSession, WgpuCompute, WgpuError};

// ----------------------------------------------------------------
// Host conversion mirror
// ----------------------------------------------------------------

/// One packed word the way WGSL's `pack2x16float`
/// builds it: the even element (the vec's x) in the low
/// half, the odd element (y) in the high half.
fn pack_pair(lo: f32, hi: f32) -> u32 {
    u32::from(f32_to_f16_bits(lo)) | (u32::from(f32_to_f16_bits(hi)) << 16)
}

/// The two elements of a packed word, the way WGSL's
/// `unpack2x16float` reads them back.
fn unpack_pair(word: u32) -> (f32, f32) {
    (
        gguf::dequant::f16_to_f32(word as u16),
        gguf::dequant::f16_to_f32((word >> 16) as u16),
    )
}

#[test]
fn f16_round_trip_stays_inside_the_half_ulp_bound() {
    // Every representable finite f16 value decodes to an
    // exact f32 and re-encodes to the identical bits:
    // the stored cache holds exactly the representable
    // values, so this half of the round trip is lossless
    // by construction, and any encoder drift fails here.
    let mut exact = 0u32;
    for bits in (0x0000u32..=0x7bff).chain(0x8000..=0xfbff) {
        let value = gguf::dequant::f16_to_f32(bits as u16);
        assert!(value.is_finite(), "bits {bits:#06x} should be finite");
        assert_eq!(
            f32_to_f16_bits(value),
            bits as u16,
            "representable value at bits {bits:#06x} must round-trip exactly"
        );
        exact += 1;
    }

    // Arbitrary f32 inputs round to the nearest f16:
    // the error is at most half an f16 ulp, which is a
    // relative 2^-11 for normal results and an absolute
    // 2^-25 once the result is subnormal (or zero). The
    // sweep is a deterministic LCG stream across
    // magnitudes, plus the exact binade edges.
    let mut rng = Rng(0x0f16_0bad);
    let mut max_rel = 0.0f32;
    let mut max_abs_sub = 0.0f32;
    let mut checked = 0u32;
    let check = |x: f32, max_rel: &mut f32, max_abs_sub: &mut f32, checked: &mut u32| {
        if !x.is_finite() || x.abs() >= 65_504.0 {
            return;
        }
        let rt = gguf::dequant::f16_to_f32(f32_to_f16_bits(x));
        let err = (rt - x).abs();
        if x.abs() >= f32::from_bits(0x3880_0000) {
            // |x| >= 2^-14: the result is a normal f16
            // (or rounds up to one), so the bound is
            // relative.
            let rel = err / x.abs();
            assert!(
                rel <= 0.000_488_281_25,
                "round-trip relative error {rel:e} past 2^-11 at x={x:e}"
            );
            *max_rel = (*max_rel).max(rel);
        } else {
            assert!(
                err <= 0.000_000_029_802_322,
                "round-trip absolute error {err:e} past 2^-25 at x={x:e}"
            );
            *max_abs_sub = (*max_abs_sub).max(err);
        }
        *checked += 1;
    };
    for e in -20..=15i32 {
        for m in [1.0f32, 1.000_976_6, 1.5, 1.999_023_4, 0.5, 0.75] {
            let x = m * 2.0f32.powi(e);
            check(x, &mut max_rel, &mut max_abs_sub, &mut checked);
            check(-x, &mut max_rel, &mut max_abs_sub, &mut checked);
        }
    }
    for _ in 0..200_000 {
        // Random bit patterns biased to the K/V
        // magnitude range: exponents from 2^-20 to 2^15.
        let sign = if rng.next_u8() & 1 == 0 { 1.0 } else { -1.0 };
        let e = (rng.next_u8() % 36) as i32 - 20;
        let mant = 1.0 + (f32::from(rng.next_u8()) / 256.0 + f32::from(rng.next_u8()) / 65_536.0);
        check(
            sign * mant * 2.0f32.powi(e),
            &mut max_rel,
            &mut max_abs_sub,
            &mut checked,
        );
    }
    eprintln!(
        "f16 round trip: {exact} representable values exact; {checked} arbitrary inputs, \
         max relative error {max_rel:e} (bound 2^-11 = 4.8828125e-4), max subnormal \
         absolute error {max_abs_sub:e} (bound 2^-25 = 2.9802322e-8)"
    );
    assert!(checked > 200_000);
    assert!(max_rel > 0.0, "the sweep must actually measure error");
}

#[test]
fn packed_word_layout_is_even_low_odd_high() {
    // The layout both shader sides assume: element 2i in
    // the low half of word i, element 2i+1 in the high
    // half (WGSL pack2x16float's x/y order).
    let (lo, hi) = unpack_pair(pack_pair(1.5, -0.25));
    assert_eq!((lo, hi), (1.5, -0.25));
    let (lo, hi) = unpack_pair(pack_pair(0.0, 65_504.0));
    assert_eq!((lo, hi), (0.0, 65_504.0));
}

// ----------------------------------------------------------------
// Memory math from the real config shape
// ----------------------------------------------------------------

/// The real Glimmer config shape (the values the KV
/// byte math reads: layer pattern, context, window,
/// head counts and lengths; the rest rides along so
/// the literal is a complete config).
fn glimmer_cfg() -> GlimmerConfig {
    GlimmerConfig {
        name: None,
        block_count: 52,
        context_length: 131_072,
        hidden: 6656,
        ffn_hidden: 19_968,
        q_heads: 32,
        kv_heads: 2,
        key_length: 128,
        value_length: 128,
        rope_freq_base: 500_000.0,
        rope_freq_base_swa: 500_000.0,
        rms_eps: 1e-5,
        sliding_window: 2048,
        logit_softcap: 20.0,
        logit_scale: 0.196_116_13,
        vocab_size: 202_048,
        layers: (0..52)
            .map(|i| {
                if i % 4 == 3 {
                    LayerKind::Global
                } else {
                    LayerKind::SlidingWindow
                }
            })
            .collect(),
        pattern_source: PatternSource::MeasuredFallback,
    }
}

#[test]
fn glimmer_full_context_kv_bytes_from_the_config() {
    let cfg = glimmer_cfg();
    // The row census the formula sums: 13 global layers
    // at the full context plus 39 windowed layers at
    // the window.
    let rows: u64 = cfg
        .layers
        .iter()
        .map(|k| match k {
            LayerKind::Global => u64::from(cfg.context_length),
            LayerKind::SlidingWindow => u64::from(cfg.sliding_window),
        })
        .sum();
    assert_eq!(rows, 13 * 131_072 + 39 * 2048);
    let f32_bytes = full_context_kv_bytes(&cfg, KvStorage::F32);
    let f16_bytes = full_context_kv_bytes(&cfg, KvStorage::F16);
    eprintln!("glimmer full-context KV: {rows} rows, f32 {f32_bytes} bytes, f16 {f16_bytes} bytes");
    // K and V both: 1,783,808 rows x 512 elements x 4 B.
    assert_eq!(f32_bytes, 3_653_238_784);
    // Exactly half in f16: this is also the figure the
    // lane tasking quoted as the f32 total; it is one
    // side (K or V) in f32, equivalently K+V in f16.
    // The halving is exact either way (see the note).
    assert_eq!(f16_bytes, 1_826_619_392);
    assert_eq!(f16_bytes * 2, f32_bytes);
}

// ----------------------------------------------------------------
// The resident-session fixture (device)
// ----------------------------------------------------------------

/// A tiny deterministic byte/float source (the
/// cross-backend fixture's generator, same seed
/// discipline).
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
/// widths, a sliding/global layer mix, window 4 (the
/// cross-backend fixture's config, as in
/// `resident_session.rs`).
fn fixture_cfg() -> GlimmerConfig {
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

/// Build the fixture weights (the cross-backend
/// fixture's construction; `mixed` puts the value
/// projection and FFN up on Q6_K, forcing the partial
/// batch so both of the session's preparation call
/// shapes run).
fn fixture_weights(cfg: &GlimmerConfig, mixed: bool) -> Weights {
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

/// The comparison driver: run the first session
/// greedily (a 4-token prompt, a multi-row chunk on the
/// staged flash form, then 7 single-token decode steps
/// on the shipped form; 11 positions through a window
/// of 4, so the sliding layers evict ring slots and
/// both K write paths, rotated sliding and plain
/// global, run), record its token script, then replay
/// the identical script through the second session so
/// the sequences cannot diverge and every step's
/// logits compare position for position. Returns the
/// worst max-abs logit delta across the steps.
fn run_pair(
    cfg: &GlimmerConfig,
    weights: &Weights,
    backend32: &WgpuCompute,
    backend16: &WgpuCompute,
    label: &str,
) -> f32 {
    let mut s32 = ResidentSession::new(backend32, cfg, weights).expect("f32 session builds");
    let mut script: Vec<u32> = vec![3, 100, 57, 900];
    let mut logits32: Vec<Vec<f32>> = Vec::new();
    let mut next: Vec<u32> = script.clone();
    for _ in 0..8 {
        let logits = s32.forward(&next).expect("f32 forward");
        assert!(logits.iter().all(|v| v.is_finite()), "f32 logits finite");
        let pick = argmax(&logits) as u32;
        logits32.push(logits);
        script.push(pick);
        next = vec![pick];
    }

    let mut s16 = ResidentSession::new(backend16, cfg, weights).expect("f16 session builds");
    let mut worst = 0.0f32;
    let mut feed: Vec<u32> = script[..4].to_vec();
    for step in 0..8 {
        let logits = s16.forward(&feed).expect("f16 forward");
        assert!(logits.iter().all(|v| v.is_finite()), "f16 logits finite");
        let reference = &logits32[step];
        assert_eq!(
            argmax(&logits),
            argmax(reference),
            "{label}: f16 top-1 disagrees with f32 at step {step}"
        );
        let delta = max_abs(&logits, reference);
        eprintln!(
            "{label} step {step}: f16-vs-f32 max logit delta {delta:.4e} \
             (top-1 {})",
            argmax(reference)
        );
        worst = worst.max(delta);
        feed = vec![script[4 + step]];
    }
    worst
}

#[test]
fn kv_f16_session_fixture_matches_f32_top1() {
    // The switch is read at backend construction, so
    // the env choreography is: remove for the f32
    // backends, set to "1" for the f16 one, remove
    // again on the way out. This test binary's other
    // tests never construct a backend, so nothing else
    // can observe the mutation.
    std::env::remove_var("WGPU_COMPUTE_KV_F16");
    let backend32 = match WgpuCompute::new() {
        Ok(b) => b,
        Err(WgpuError::NoSuitableAdapter { seen, .. }) => {
            eprintln!(
                "skipping kv f16 session fixture: no suitable adapter (enumerated: [{}])",
                seen.join("; ")
            );
            return;
        }
        Err(e) => panic!("backend construction failed unexpectedly: {e}"),
    };
    let cfg = fixture_cfg();

    // Harness determinism first: a second f32 backend
    // and session must reproduce the first session's
    // logits bit for bit under the shared script,
    // or no cross-format delta measured below means
    // anything.
    let weights = fixture_weights(&cfg, false);
    let backend32_b = WgpuCompute::new().expect("second f32 backend builds");
    let det = run_pair(&cfg, &weights, &backend32, &backend32_b, "f32-vs-f32");
    assert_eq!(det, 0.0, "two f32 sessions must agree bit for bit");

    std::env::set_var("WGPU_COMPUTE_KV_F16", "1");
    let backend16 = WgpuCompute::new().expect("f16 backend builds");
    let mut worst = 0.0f32;
    for mixed in [false, true] {
        let weights = fixture_weights(&cfg, mixed);
        let label = if mixed { "mixed" } else { "uniform" };
        worst = worst.max(run_pair(&cfg, &weights, &backend32, &backend16, label));
    }

    // The two-pass form refuses an f16 layer by name:
    // build one layer of f16 KV on the f16 backend's
    // context and ask for the scores form.
    let ctx = backend16.context();
    let pattern = [attention::LayerKind::Global];
    let mut kv = DeviceKv::new_with_storage(&pattern, 0, KvStorage::F16);
    let layer = kv
        .ensure_layer(ctx, 0, 256, 4, "attention")
        .expect("f16 layer ensures");
    let k6 = K6Pipeline::new(&ctx.device).expect("k6 pipelines build");
    let geom = AttendGeom {
        rows: 1,
        q_heads: 32,
        kv_heads: 2,
        head_dim: 128,
        stored: 4,
        first_pos: 0,
        start_pos: 3,
        window: None,
        ring: None,
    };
    let q = ctx.scratch_buffer("kv f16 refusal q", (32 * 128 * 4) as u64);
    let scores = ctx.scratch_buffer("kv f16 refusal scores", (32 * 4 * 4) as u64);
    let refused = k6.prepare_scores(ctx, &q, layer, &scores, &geom);
    assert!(
        refused.is_err(),
        "prepare_scores over f16 KV must refuse by name, got {refused:?}"
    );
    let out = ctx.scratch_buffer("kv f16 refusal out", (32 * 128 * 4) as u64);
    let refused = k6.prepare_reduce(ctx, &scores, layer, &out, &geom);
    assert!(
        refused.is_err(),
        "prepare_reduce over f16 KV must refuse by name, got {refused:?}"
    );

    std::env::remove_var("WGPU_COMPUTE_KV_F16");
    eprintln!("kv f16 session fixture: worst f16-vs-f32 logit delta {worst:.4e}");
    // The ceiling is the crate's cross-backend bar
    // (0.5), used as a breakage tripwire and disclosed
    // as such in the bench note: the expected delta
    // from the precision math is orders of magnitude
    // smaller, and the value the card run prints is
    // the measurement this ceiling gets replaced by.
    assert!(
        worst < 0.5,
        "f16-vs-f32 fixture logit delta {worst} past the documenting ceiling"
    );
}
