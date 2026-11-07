//! The K21F head-routing pins (bench/k21-finish-2026-11-06.md):
//! the resident block head's Q5_K product routes to the
//! finish-carrying MMV form (`k21_q5k_mmv_finish`) when the
//! shared `WGPU_COMPUTE_MMV_MAX_ROWS` gate admits the block
//! batch, and stays on the K3 fused call otherwise, with the
//! finish (the fixture config's logit scale and tanh
//! softcap, Glimmer's own values) applied identically on
//! both paths.
//!
//! Two arms, selected by the environment the test binary is
//! launched with (the diag smoke / capture-pin discipline:
//! each test self-skips with a printed reason when its
//! arm's conditions are not met):
//!
//! - Gate closed (the shipped default): launch with
//!   `WGPU_COMPUTE_DIAG=timing` and
//!   `WGPU_COMPUTE_MMV_MAX_ROWS` unset (or 1). The block
//!   head must stamp `k3_q5k_gemv` and never
//!   `k21_q5k_mmv_finish`.
//! - Gate open: launch with `WGPU_COMPUTE_DIAG=timing` and
//!   `WGPU_COMPUTE_MMV_MAX_ROWS=16`. The block head must
//!   stamp `k21_q5k_mmv_finish` at the block batch, while a
//!   batch-1 forward's head still stamps `k3_q5k_gemv`
//!   (the standard head stays K3 by construction).
//!
//! Both arms hold the block logits to the CPU model's
//! block forward at the cross-backend bar (per-row argmax
//! agreement, max-abs under 0.5): the CPU head applies the
//! true finish, so a finish the routing dropped, swapped,
//! or re-ordered on its way into the K21F call cannot pass
//! the open arm. Both arms need a discrete Vulkan GPU and
//! skip where none qualifies.

use std::collections::HashMap;
use std::sync::Arc;

use cpu_compute::CpuCompute;
use muse_glimmer::{
    expected_tensor_names, Dtype, GlimmerConfig, GlimmerError, GlimmerModel, LayerKind,
    PatternSource, PositionCache, Result, TensorSource, TensorView, Weights,
};
use wgpu_compute::kernels::k1_q4k_gemv::f32_to_f16_bits;
use wgpu_compute::timestamps::DiagMode;
use wgpu_compute::{ResidentSession, WgpuCompute, WgpuError};

/// The shared MMV gate's env name (see
/// `wgpu_compute::compute::MMV_MAX_ROWS_ENV`; not
/// re-exported at the crate root).
const MMV_MAX_ROWS_ENV: &str = "WGPU_COMPUTE_MMV_MAX_ROWS";

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

/// The synthetic config: the resident fixture's (hidden
/// 512, FFN 1024, vocab 1024, six layers in a sliding/
/// global mix, window 4), with Glimmer's real finish
/// values (logit scale, softcap 20.0) so the head finish
/// the routing forwards is the production finish shape.
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
/// fixture's rules, tensor for tensor; the head,
/// `output.weight`, is Q5_K).
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

/// The resolved MMV gate this process was launched with:
/// the block batch (16) is admitted exactly when the
/// parsed threshold reaches it. Unparsable values resolve
/// to the shipped default in the backend, which admits
/// nothing, so they read as closed here too.
fn gate_admits_block_batch() -> bool {
    std::env::var(MMV_MAX_ROWS_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .is_some_and(|max_rows| max_rows >= 16)
}

/// The diag conditions both arms read their exposure from:
/// timing mode on, and a recorder on the device (a device
/// that grants no timestamp features records nothing, and
/// the kernel-name assertions would read an empty table).
fn diag_ready(test: &str, backend: &WgpuCompute) -> bool {
    if DiagMode::from_env() != DiagMode::Timing {
        eprintln!("skipping {test}: WGPU_COMPUTE_DIAG is not `timing`");
        return false;
    }
    if backend.context().diag_recorder().is_none() {
        eprintln!(
            "skipping {test}: timing requested but the device granted \
             no timestamp features, so no recorder exists"
        );
        return false;
    }
    true
}

const PROMPT: &[u32] = &[3, 100, 57, 900, 12, 640];
/// The spec block case: a verify block at the DFlash
/// block width, the batch the routing exists for.
const BLOCK: &[u32] = &[
    11, 222, 5, 999, 3, 77, 640, 12, 900, 57, 100, 3, 42, 8, 731, 256,
];

/// Run the fixture prompt and block on a fresh resident
/// session and the CPU model in lockstep, returning the
/// session (for its diag table) and both block logit sets.
fn run_block(backend: &WgpuCompute) -> (ResidentSession<'_>, Vec<f32>, Vec<f32>) {
    let cfg = cross_cfg();
    let weights = cross_weights(&cfg);
    let cpu_model = GlimmerModel::new(cfg.clone(), weights.clone(), Box::new(CpuCompute::new()));
    let mut session =
        ResidentSession::new(backend, &cfg, &weights).expect("resident session builds");
    let mut cpu_cache = PositionCache::new();
    cpu_model
        .forward_capturing(PROMPT, &mut cpu_cache, &[])
        .expect("cpu prompt forward");
    session.forward(PROMPT).expect("resident prompt forward");
    let (cpu_block, _) = cpu_model
        .forward_block_capturing(BLOCK, &mut cpu_cache, &[])
        .expect("cpu block forward");
    let (res_block, _) = session
        .forward_block_capturing(BLOCK, &[])
        .expect("resident block forward");
    session.finish_diag_report();
    (session, res_block, cpu_block)
}

/// The finish-agreement half of both arms: the resident
/// block logits against the CPU model's, per row, at the
/// cross-backend bar. The CPU head applies the fixture's
/// finish (scale, softcap) itself, so this is the pin
/// that the routed path's finish is that finish.
fn assert_block_matches_cpu(res_block: &[f32], cpu_block: &[f32], vocab: usize) {
    assert_eq!(res_block.len(), BLOCK.len() * vocab);
    assert_eq!(cpu_block.len(), BLOCK.len() * vocab);
    for row in 0..BLOCK.len() {
        let res_row = &res_block[row * vocab..(row + 1) * vocab];
        let cpu_row = &cpu_block[row * vocab..(row + 1) * vocab];
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
}

/// Gate closed (the shipped default): the block head is
/// the K3 fused call, exactly as shipped, and no
/// `k21_q5k_mmv_finish` stamp appears anywhere.
#[test]
fn block_head_stays_k3_fused_with_the_gate_closed() {
    let test = "k21f closed-gate routing pin";
    if gate_admits_block_batch() {
        eprintln!(
            "skipping {test}: {MMV_MAX_ROWS_ENV} admits the block batch; \
             this is the closed arm"
        );
        return;
    }
    let Some(backend) = backend_or_skip(test) else {
        return;
    };
    if !diag_ready(test, &backend) {
        return;
    }
    let (session, res_block, cpu_block) = run_block(&backend);
    assert_block_matches_cpu(&res_block, &cpu_block, 1024);
    let names = session.diag_kernel_names();
    assert!(
        names.contains(&"k3_q5k_gemv"),
        "the closed gate's block head must stamp the K3 GEMV, saw {names:?}"
    );
    assert!(
        !names.contains(&"k21_q5k_mmv_finish"),
        "the closed gate must never stamp the finish form, saw {names:?}"
    );
}

/// Gate open at the block batch: the block head routes to
/// the finish form, the batch-1 head stays K3, and the
/// routed block still carries the fixture's finish
/// (agreement with the CPU model at the bar).
#[test]
fn block_head_routes_to_k21f_with_the_gate_open() {
    let test = "k21f open-gate routing pin";
    if !gate_admits_block_batch() {
        eprintln!(
            "skipping {test}: {MMV_MAX_ROWS_ENV} does not admit the block \
             batch (unset, unparsable, or below 16); this is the open arm"
        );
        return;
    }
    let Some(backend) = backend_or_skip(test) else {
        return;
    };
    if !diag_ready(test, &backend) {
        return;
    }
    // The batch-1 head first, on its own session: even
    // with the gate open, a single-row forward's head is
    // the K3 fused call (the band floor refuses batch 1),
    // and no finish stamp appears.
    let cfg = cross_cfg();
    let weights = cross_weights(&cfg);
    let mut decode = ResidentSession::new(&backend, &cfg, &weights).expect("session builds");
    decode.forward(&[7]).expect("decode forward");
    decode.finish_diag_report();
    let names = decode.diag_kernel_names();
    assert!(
        names.contains(&"k3_q5k_gemv"),
        "the batch-1 head must stamp the K3 GEMV, saw {names:?}"
    );
    assert!(
        !names.contains(&"k21_q5k_mmv_finish"),
        "batch 1 must never route to the finish form, saw {names:?}"
    );
    drop(decode);

    let (session, res_block, cpu_block) = run_block(&backend);
    assert_block_matches_cpu(&res_block, &cpu_block, 1024);
    let names = session.diag_kernel_names();
    assert!(
        names.contains(&"k21_q5k_mmv_finish"),
        "the open gate's block head must stamp the finish form, saw {names:?}"
    );
}
