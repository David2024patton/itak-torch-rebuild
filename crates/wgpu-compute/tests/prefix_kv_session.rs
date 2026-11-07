//! The prefix-KV session seam, end to end on the resident
//! fixture (the cross-backend shape: hidden 512, FFN 1024,
//! vocab 1024, six layers in a sliding/global mix, window 4).
//!
//! The prompt is 70 tokens: longer than the window (4) by
//! design, and at the session's ring capacity
//! (window + chunk cap - 1 = 4 + 64 - 1 = 67) the stored
//! windowed rows (positions 66..70) wrap the ring's end
//! (slots 66, 0, 1, 2). That wrap is the corruption case the
//! foundation note names: a capture or install that copies
//! slot order instead of absolute-position order scrambles
//! exactly those rows, and the continuation comparison below
//! is what catches it.
//!
//! One test function runs the phases sequentially because the
//! seam's master switch is process environment: phases set
//! and clear it explicitly rather than racing sibling tests.
//! Needs a discrete Vulkan GPU under the backend's default
//! policy; skips with a printed reason where none qualifies
//! (the VM has no adapter, so there this file is a compile
//! and skip check; the card run is the executing proof).
//!
//! The second test composes the seam with the OTHER storage
//! switch (`WGPU_COMPUTE_KV_F16`): an f16-storage session
//! must refuse capture and install by name, leave its state
//! untouched, and never write a snapshot to the store (the
//! critic panel's Category A composition finding; see
//! bench/prefix-f16-guard-2026-11-07.md). The guard itself is
//! also pinned device-free in the session module's unit
//! tests (`prefix_storage_refusal_gates_the_f16_composition`),
//! which run everywhere.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use itak_kv::{KvGeometry, KvSnapshot, PrefixStore, StoreConfig};
use muse_glimmer::{
    expected_tensor_names, Dtype, GlimmerConfig, GlimmerError, LayerKind, PatternSource, Result,
    TensorSource, TensorView, Weights,
};
use wgpu_compute::kernels::k1_q4k_gemv::f32_to_f16_bits;
use wgpu_compute::{ResidentSession, WgpuCompute, WgpuError};

const FINGERPRINT: &str = "fixture-gguf-sha256-0000";
/// Greedy continuation steps compared past the restore point
/// (the brief's floor is 32; this runs 40).
const CONTINUATION: usize = 40;

/// A tiny deterministic byte/float source (the resident
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

/// The resident fixture's config: window 4, so a 70-token
/// prompt's windowed rows wrap the 67-slot ring.
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

fn fixture_weights(cfg: &GlimmerConfig) -> Weights {
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

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best as u32
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

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

fn temp_root(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "itak-prefix-session-test-{}-{tag}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    root
}

/// Run `steps` greedy continuation steps on `session`,
/// teacher-forced on `inputs` (the cold run's picks), checking
/// each step's argmax against `want_picks` and the logits
/// against `want_logits` at the cross-path bar (max abs diff
/// under 0.5, the resident session test's bound). Returns the
/// worst logit difference seen.
fn check_continuation(
    session: &mut ResidentSession,
    inputs: &[u32],
    want_logits: &[Vec<f32>],
    want_picks: &[u32],
) -> f32 {
    let mut worst = 0.0f32;
    for (step, &input) in inputs.iter().enumerate() {
        let logits = session.forward(&[input]).expect("continuation forward");
        let d = max_abs(&logits, &want_logits[step]);
        worst = worst.max(d);
        assert!(
            d < 0.5,
            "continuation logits drift past the cross-path bar at step {step}: {d}"
        );
        assert_eq!(
            argmax(&logits),
            want_picks[step],
            "continuation greedy pick diverges at step {step}"
        );
    }
    worst
}

#[test]
fn resident_session_prefix_store_round_trip() {
    // Phase 0: the switch starts OFF for this process,
    // whatever the launching environment carried; the store
    // directory is pointed at a fresh temp root.
    std::env::remove_var(itak_kv::ENABLE_ENV);
    let root = temp_root("main");
    std::env::set_var(itak_kv::DIR_ENV, &root);

    let Some(backend_a) = backend_or_skip("prefix session round trip") else {
        return;
    };
    let cfg = fixture_cfg();
    let weights = fixture_weights(&cfg);
    // The 70-token prompt: two resident chunks, windowed rows
    // wrapping the ring (see the module docs).
    let prompt: Vec<u32> = (0..70).map(|i| (i * 37 % 1024) as u32).collect();

    // --- Phase 1: switch off. No store opens, no directory
    // is created, capture and install refuse by name, and the
    // session is untouched by the refusals.
    assert!(
        ResidentSession::prefix_store_open().is_none(),
        "switch off: no store may open"
    );
    assert!(
        !root.exists(),
        "switch off: the store directory must not be created"
    );
    let mut off_session =
        ResidentSession::new(&backend_a, &cfg, &weights).expect("off session builds");
    let off_logits = off_session.forward(&prompt).expect("off forward");
    assert_eq!(off_session.position(), 70);
    let err = off_session
        .capture_prefix_snapshot(FINGERPRINT)
        .expect_err("switch off: capture must refuse");
    assert!(
        err.to_string().contains(itak_kv::ENABLE_ENV),
        "capture refusal names the switch: {err}"
    );
    let dummy_snapshot = {
        // A structurally valid snapshot for the geometry, built
        // by hand: the install refusal must come from the
        // switch, not from validation.
        let geometry = KvGeometry {
            layers: cfg
                .layers
                .iter()
                .map(|kind| match kind {
                    LayerKind::Global => attention::LayerKind::Global,
                    LayerKind::SlidingWindow => attention::LayerKind::Windowed { window: 4 },
                })
                .collect(),
            num_kv_heads: 2,
            head_dim: 128,
        };
        let row = geometry.row_floats();
        let layers = geometry
            .layers
            .iter()
            .map(|kind| {
                let (first, rows) = match kind {
                    attention::LayerKind::Global => (0, 5),
                    attention::LayerKind::Windowed { window } => {
                        let rows = 5usize.min(*window);
                        (5 - rows, rows)
                    }
                };
                itak_kv::LayerSnapshot {
                    first_position: first,
                    keys: vec![0.0; rows * row],
                    values: vec![0.0; rows * row],
                }
            })
            .collect();
        KvSnapshot {
            geometry,
            model_fingerprint: FINGERPRINT.to_string(),
            token_count: 5,
            layers,
        }
    };
    let err = off_session
        .install_prefix_snapshot(&dummy_snapshot, FINGERPRINT)
        .expect_err("switch off: install must refuse");
    assert!(
        err.to_string().contains(itak_kv::ENABLE_ENV),
        "install refusal names the switch: {err}"
    );
    assert_eq!(
        off_session.position(),
        70,
        "switch-off refusals leave the session untouched"
    );
    assert!(off_session.prefix_match().is_none());
    drop(off_session);

    // --- Phase 2: switch on. Cold run: prefill, capture and
    // store through the session flow, then a 40-step greedy
    // continuation recorded as the reference.
    std::env::set_var(itak_kv::ENABLE_ENV, "1");
    let mut store = ResidentSession::prefix_store_open()
        .expect("switch on: the store opens")
        .expect("store opens cleanly");
    let backend_b = WgpuCompute::new().expect("cold backend builds");
    let mut cold = ResidentSession::new(&backend_b, &cfg, &weights).expect("cold session builds");
    let cold_prompt_logits = cold.forward(&prompt).expect("cold prefill");
    assert!(
        max_abs(&cold_prompt_logits, &off_logits) < 0.5,
        "the switch alone changes no forward math"
    );
    cold.capture_and_store_prefix(&mut store, &prompt, FINGERPRINT)
        .expect("capture and store after the cold prefill");
    assert_eq!(store.stats().entries, 1, "one prefix stored");
    // The captured snapshot's shape, checked through a fresh
    // lookup: windowed layers carry 4 rows starting at
    // position 66 (the wrap case), global layers all 70.
    let query: Vec<u32> = prompt.iter().chain([1_u32, 2, 3].iter()).copied().collect();
    let hit = store
        .lookup(&query, None, None, itak_kv::now_unix())
        .expect("lookup runs")
        .expect("the stored prefix hits");
    assert_eq!(hit.token_count, 70);
    for (index, layer) in hit.snapshot.layers.iter().enumerate() {
        match cfg.layers[index] {
            LayerKind::Global => {
                assert_eq!(layer.first_position, 0, "layer {index}");
                assert_eq!(layer.keys.len(), 70 * 256, "layer {index}");
            }
            LayerKind::SlidingWindow => {
                assert_eq!(layer.first_position, 66, "layer {index}");
                assert_eq!(layer.keys.len(), 4 * 256, "layer {index}");
            }
        }
    }
    // Cold continuation: greedy picks fed back as inputs.
    let mut inputs: Vec<u32> = Vec::new();
    let mut want_logits: Vec<Vec<f32>> = Vec::new();
    let mut want_picks: Vec<u32> = Vec::new();
    let mut next = argmax(&cold_prompt_logits);
    for _ in 0..CONTINUATION {
        inputs.push(next);
        let logits = cold.forward(&[next]).expect("cold continuation");
        next = argmax(&logits);
        want_picks.push(next);
        want_logits.push(logits);
    }
    assert_eq!(cold.position(), 70 + CONTINUATION);
    let cold_stats = cold.stats();
    assert_eq!(cold_stats.prefix_captures, 1);
    drop(cold);

    // --- Phase 3: warm run. A fresh session restores from
    // the store and continues: same picks, logits at the
    // cross-path bar, for the full continuation.
    let backend_c = WgpuCompute::new().expect("warm backend builds");
    let mut warm = ResidentSession::new(&backend_c, &cfg, &weights).expect("warm session builds");
    let matched = warm
        .restore_prefix_from_store(&mut store, &query, FINGERPRINT)
        .expect("restore flow runs")
        .expect("the store serves the warm session");
    assert_eq!(matched, 70);
    assert_eq!(warm.position(), 70);
    assert_eq!(warm.prefix_match(), Some(70));
    assert_eq!(warm.stats().prefix_restores, 1);
    assert_eq!(warm.stats().prefix_restored_tokens, 70);
    let worst = check_continuation(&mut warm, &inputs, &want_logits, &want_picks);
    eprintln!(
        "prefix round trip: worst continuation logit diff {worst:.4e} over {CONTINUATION} steps"
    );
    assert_eq!(warm.position(), 70 + CONTINUATION);
    drop(warm);

    // --- Phase 4: the one-call flow. A fresh store: the
    // first session misses and stores; the second hits the
    // full prompt (install + one-token re-forward) and its
    // returned logits agree with the cold prompt logits.
    let root2 = temp_root("flow");
    let mut store2 = PrefixStore::open(&StoreConfig {
        root: root2.clone(),
        max_bytes: itak_kv::DEFAULT_MAX_BYTES,
    })
    .expect("flow store opens");
    let backend_d = WgpuCompute::new().expect("flow backend builds");
    let mut flow_cold =
        ResidentSession::new(&backend_d, &cfg, &weights).expect("flow cold session builds");
    let flow_logits = flow_cold
        .forward_with_prefix_store(&mut store2, &prompt, FINGERPRINT)
        .expect("flow cold forward");
    assert_eq!(argmax(&flow_logits), argmax(&cold_prompt_logits));
    assert!(max_abs(&flow_logits, &cold_prompt_logits) < 0.5);
    assert_eq!(flow_cold.prefix_match(), None, "a miss installs nothing");
    assert_eq!(store2.stats().entries, 1, "the miss stored the prefix");
    drop(flow_cold);
    let backend_e = WgpuCompute::new().expect("flow warm backend builds");
    let mut flow_warm =
        ResidentSession::new(&backend_e, &cfg, &weights).expect("flow warm session builds");
    let flow_warm_logits = flow_warm
        .forward_with_prefix_store(&mut store2, &prompt, FINGERPRINT)
        .expect("flow warm forward");
    assert_eq!(
        flow_warm.prefix_match(),
        Some(70),
        "the full prompt was restored"
    );
    assert_eq!(flow_warm.position(), 70);
    assert_eq!(argmax(&flow_warm_logits), argmax(&cold_prompt_logits));
    assert!(
        max_abs(&flow_warm_logits, &cold_prompt_logits) < 0.5,
        "full-hit re-forward logits agree with the cold run"
    );
    drop(flow_warm);

    // --- Phase 5: refusals leave a live session untouched.
    // Session R runs 3 tokens, then takes three refused
    // installs (geometry, fingerprint, oversized); its
    // position and its next forward must be exactly the
    // reference session's.
    let backend_f = WgpuCompute::new().expect("refusal backend builds");
    let mut refused =
        ResidentSession::new(&backend_f, &cfg, &weights).expect("refusal session builds");
    let backend_g = WgpuCompute::new().expect("reference backend builds");
    let mut reference =
        ResidentSession::new(&backend_g, &cfg, &weights).expect("reference session builds");
    let head: Vec<u32> = prompt[..3].to_vec();
    refused.forward(&head).expect("refusal head forward");
    reference.forward(&head).expect("reference head forward");
    // Geometry: a self-consistent snapshot at head dim 64.
    let foreign = {
        let geometry = KvGeometry {
            layers: hit.snapshot.geometry.layers.clone(),
            num_kv_heads: 2,
            head_dim: 64,
        };
        let row = geometry.row_floats();
        let layers = geometry
            .layers
            .iter()
            .map(|kind| {
                let (first, rows) = match kind {
                    attention::LayerKind::Global => (0, 10),
                    attention::LayerKind::Windowed { window } => {
                        let rows = 10usize.min(*window);
                        (10 - rows, rows)
                    }
                };
                itak_kv::LayerSnapshot {
                    first_position: first,
                    keys: vec![0.0; rows * row],
                    values: vec![0.0; rows * row],
                }
            })
            .collect();
        KvSnapshot {
            geometry,
            model_fingerprint: FINGERPRINT.to_string(),
            token_count: 10,
            layers,
        }
    };
    let err = refused
        .install_prefix_snapshot(&foreign, FINGERPRINT)
        .expect_err("foreign geometry must refuse");
    assert!(err.to_string().contains("geometry"), "{err}");
    assert_eq!(refused.position(), 3, "geometry refusal: session untouched");
    // Fingerprint: the real stored snapshot, wrong expectation.
    let err = refused
        .install_prefix_snapshot(&hit.snapshot, "some-other-model")
        .expect_err("fingerprint mismatch must refuse");
    assert!(err.to_string().contains("fingerprint"), "{err}");
    assert_eq!(
        refused.position(),
        3,
        "fingerprint refusal: session untouched"
    );
    // Oversized: the session's own geometry at 300 tokens,
    // past the 256-position context.
    let oversized = {
        let geometry = hit.snapshot.geometry.clone();
        let row = geometry.row_floats();
        let layers = geometry
            .layers
            .iter()
            .map(|kind| {
                let (first, rows) = match kind {
                    attention::LayerKind::Global => (0, 300),
                    attention::LayerKind::Windowed { window } => {
                        let rows = 300usize.min(*window);
                        (300 - rows, rows)
                    }
                };
                itak_kv::LayerSnapshot {
                    first_position: first,
                    keys: vec![0.0; rows * row],
                    values: vec![0.0; rows * row],
                }
            })
            .collect();
        KvSnapshot {
            geometry,
            model_fingerprint: FINGERPRINT.to_string(),
            token_count: 300,
            layers,
        }
    };
    let err = refused
        .install_prefix_snapshot(&oversized, FINGERPRINT)
        .expect_err("oversized prefix must refuse");
    assert!(err.to_string().contains("context capacity"), "{err}");
    assert_eq!(
        refused.position(),
        3,
        "oversized refusal: session untouched"
    );
    assert!(refused.prefix_match().is_none(), "no refusal installs");
    // The session still forwards exactly like the reference.
    let refused_logits = refused.forward(&[prompt[3]]).expect("post-refusal forward");
    let reference_logits = reference.forward(&[prompt[3]]).expect("reference forward");
    assert_eq!(argmax(&refused_logits), argmax(&reference_logits));
    assert!(max_abs(&refused_logits, &reference_logits) < 0.5);
    assert_eq!(refused.position(), 4);

    std::env::remove_var(itak_kv::ENABLE_ENV);
    std::env::remove_var(itak_kv::DIR_ENV);
    let _ = std::fs::remove_dir_all(&root);
    let _ = std::fs::remove_dir_all(&root2);
}

/// A hand-built, structurally valid snapshot for the fixture
/// geometry at `tokens` tokens (zeros; its contents never
/// reach a device buffer in the tests that use it, because
/// the refusal under test fires before any byte moves).
fn fixture_snapshot(cfg: &GlimmerConfig, tokens: usize) -> KvSnapshot {
    let geometry = KvGeometry {
        layers: cfg
            .layers
            .iter()
            .map(|kind| match kind {
                LayerKind::Global => attention::LayerKind::Global,
                LayerKind::SlidingWindow => attention::LayerKind::Windowed { window: 4 },
            })
            .collect(),
        num_kv_heads: 2,
        head_dim: 128,
    };
    let row = geometry.row_floats();
    let layers = geometry
        .layers
        .iter()
        .map(|kind| {
            let (first, rows) = match kind {
                attention::LayerKind::Global => (0, tokens),
                attention::LayerKind::Windowed { window } => {
                    let rows = tokens.min(*window);
                    (tokens - rows, rows)
                }
            };
            itak_kv::LayerSnapshot {
                first_position: first,
                keys: vec![0.0; rows * row],
                values: vec![0.0; rows * row],
            }
        })
        .collect();
    KvSnapshot {
        geometry,
        model_fingerprint: FINGERPRINT.to_string(),
        token_count: tokens,
        layers,
    }
}

/// The two switches composed: prefix store ON and KV storage
/// f16. Snapshots are f32 and the seam does not convert, so
/// capture and install must refuse by name (naming the f16
/// storage and the remedy), the refusals must leave the
/// session byte-identical (proven by an exact continuation
/// match against an untouched twin), the restore flow must
/// fall back to `None` rather than erroring, and the store
/// flow must never write a snapshot. Needs a discrete GPU;
/// skips like the round-trip test where none qualifies.
#[test]
fn resident_session_prefix_seam_refuses_f16_storage() {
    // Both switches are read from the process environment
    // (the prefix switch per call, the f16 switch at backend
    // construction), so the choreography is explicit: f16
    // off while the fixture snapshot is captured, on for the
    // sessions under test, off again on the way out.
    std::env::remove_var("WGPU_COMPUTE_KV_F16");
    std::env::set_var(itak_kv::ENABLE_ENV, "1");
    let root = temp_root("f16");
    std::env::set_var(itak_kv::DIR_ENV, &root);

    let Some(_backend32) = backend_or_skip("prefix f16 guard") else {
        std::env::remove_var(itak_kv::ENABLE_ENV);
        std::env::remove_var(itak_kv::DIR_ENV);
        return;
    };
    let cfg = fixture_cfg();
    let weights = fixture_weights(&cfg);
    let prompt: Vec<u32> = (0..12).map(|i| (i * 37 % 1024) as u32).collect();

    // The valid snapshot the install arm is refused: built
    // by hand for the fixture geometry (the same shape the
    // round-trip test's switch-off arm builds), so on an f32
    // session it would install; only the storage format can
    // refuse it below. Its contents never reach a device
    // buffer here.
    let snapshot = fixture_snapshot(&cfg, 12);
    std::env::set_var("WGPU_COMPUTE_KV_F16", "1");
    let backend16 = WgpuCompute::new().expect("f16 backend builds");
    let mut guarded = ResidentSession::new(&backend16, &cfg, &weights).expect("f16 session builds");
    let mut twin = ResidentSession::new(&backend16, &cfg, &weights).expect("f16 twin builds");
    guarded.forward(&prompt).expect("guarded forward");
    twin.forward(&prompt).expect("twin forward");
    assert_eq!(guarded.position(), 12);

    // Capture refuses by name: the f16 storage, the switch
    // that selected it, and the remedy.
    let err = guarded
        .capture_prefix_snapshot(FINGERPRINT)
        .expect_err("f16 capture must refuse");
    let text = err.to_string();
    assert!(
        text.contains("f16"),
        "capture refusal names the storage: {err}"
    );
    assert!(
        text.contains("WGPU_COMPUTE_KV_F16"),
        "capture refusal names the switch: {err}"
    );
    assert!(
        text.contains("unset or off"),
        "capture refusal names the remedy: {err}"
    );
    assert!(
        matches!(&err, GlimmerError::StepFailed { step, .. } if *step == "prefix_capture"),
        "capture refusal is raised under prefix_capture: {err:?}"
    );
    assert_eq!(
        guarded.position(),
        12,
        "capture refusal: position untouched"
    );
    assert_eq!(
        guarded.stats().prefix_captures,
        0,
        "capture refusal: no capture counted"
    );

    // The store flow refuses through the capture and writes
    // nothing to the store.
    let mut store = ResidentSession::prefix_store_open()
        .expect("switch on: the store opens")
        .expect("store opens cleanly");
    let err = guarded
        .capture_and_store_prefix(&mut store, &prompt, FINGERPRINT)
        .expect_err("f16 store flow must refuse");
    assert!(err.to_string().contains("f16"), "{err}");
    assert_eq!(store.stats().entries, 0, "no snapshot reaches the store");

    // Install refuses by name BEFORE its reset: a successful
    // install would rewind the session to the snapshot's 12
    // tokens and stamp the restore counters; the refusal
    // must leave position, match, and counters alone.
    let err = guarded
        .install_prefix_snapshot(&snapshot, FINGERPRINT)
        .expect_err("f16 install must refuse");
    let text = err.to_string();
    assert!(
        text.contains("f16"),
        "install refusal names the storage: {err}"
    );
    assert!(
        text.contains("unset or off"),
        "install refusal names the remedy: {err}"
    );
    assert!(
        matches!(&err, GlimmerError::StepFailed { step, .. } if *step == "prefix_install"),
        "install refusal is raised under prefix_install: {err:?}"
    );
    assert_eq!(guarded.position(), 12, "install refusal: no reset happened");
    assert!(
        guarded.prefix_match().is_none(),
        "install refusal: no match"
    );
    assert_eq!(guarded.stats().prefix_restores, 0);

    // The restore flow keeps its fallback contract under
    // f16: even with a real hit in the store it returns
    // `None` (its diag line names the refusal) instead of
    // surfacing the install refusal as a hard error.
    store
        .put(&prompt, &snapshot, itak_kv::now_unix())
        .expect("the f32 snapshot is stored directly");
    assert_eq!(store.stats().entries, 1);
    let restored = guarded
        .restore_prefix_from_store(&mut store, &prompt, FINGERPRINT)
        .expect("restore flow must not hard-fail under f16");
    assert_eq!(restored, None, "f16 restore falls back to a cold prefill");
    assert_eq!(guarded.position(), 12);
    drop(store);

    // Byte-identical state after every refusal: the next
    // forward matches the untouched twin exactly (same
    // backend configuration, deterministic fixture, so the
    // cross-session comparison is exact, the bar the f16
    // fixture test's determinism arm establishes).
    let guarded_logits = guarded.forward(&[prompt[0]]).expect("post-refusal forward");
    let twin_logits = twin.forward(&[prompt[0]]).expect("twin forward");
    assert_eq!(
        guarded_logits, twin_logits,
        "refusals leave the session byte-identical"
    );
    assert_eq!(guarded.position(), 13);

    std::env::remove_var("WGPU_COMPUTE_KV_F16");
    std::env::remove_var(itak_kv::ENABLE_ENV);
    std::env::remove_var(itak_kv::DIR_ENV);
    let _ = std::fs::remove_dir_all(&root);
}
