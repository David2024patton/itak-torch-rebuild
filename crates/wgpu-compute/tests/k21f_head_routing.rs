//! The K21F head-routing pins (bench/k21-finish-2026-11-06.md,
//! guard in bench/k21f-default-guard-2026-11-07.md, default
//! flip in bench/k21f-default-flip-2026-11-07.md): the
//! resident block head's Q5_K product routes to the
//! finish-carrying MMV form (`k21_q5k_mmv_finish`) only
//! when BOTH gates admit: the route's own switch
//! (`WGPU_COMPUTE_K21F_HEAD`: on when unset or `1`, off
//! when `0`, fail-closed off on any other value) and
//! the shared `WGPU_COMPUTE_MMV_MAX_ROWS` band at the
//! block batch. Otherwise the head stays on the K3 fused
//! call, with the finish (the fixture config's logit
//! scale and tanh softcap, Glimmer's own values) applied
//! identically on both paths.
//!
//! Arms, selected by the environment the test binary is
//! launched with (the diag smoke / capture-pin discipline:
//! each test self-skips with a printed reason when its
//! arm's conditions are not met):
//!
//! - Head switch off (explicit): launch with
//!   `WGPU_COMPUTE_DIAG=timing`,
//!   `WGPU_COMPUTE_K21F_HEAD=0`, and
//!   `WGPU_COMPUTE_MMV_MAX_ROWS` unset (the backend's
//!   shipped ceiling, 16, admits the block batch) or
//!   explicitly 16. The block head must stamp
//!   `k3_q5k_gemv` and never `k21_q5k_mmv_finish`: the
//!   open shared ceiling alone must not open the head
//!   route. An unrecognized switch value (anything but
//!   `1` or `0`, e.g. `true`) also lands in this arm by
//!   construction, which is the fail-closed pin. An
//!   unset switch does NOT land here: unset resolves
//!   on since the default flip, and the next arm pins
//!   that position.
//! - Default construction: launch with
//!   `WGPU_COMPUTE_DIAG=timing` and both
//!   `WGPU_COMPUTE_K21F_HEAD` and
//!   `WGPU_COMPUTE_MMV_MAX_ROWS` unset. The block head
//!   must stamp `k21_q5k_mmv_finish` at the block
//!   batch: the shipped defaults route the head, which
//!   is the flip's central pin.
//! - Band closed: launch with
//!   `WGPU_COMPUTE_MMV_MAX_ROWS=1` (either switch
//!   position). The block head must stamp `k3_q5k_gemv`
//!   and never `k21_q5k_mmv_finish`.
//! - Both gates open (explicit): launch with
//!   `WGPU_COMPUTE_DIAG=timing`,
//!   `WGPU_COMPUTE_MMV_MAX_ROWS=16`, and
//!   `WGPU_COMPUTE_K21F_HEAD=1`. The block head must
//!   stamp `k21_q5k_mmv_finish` at the block batch,
//!   while a batch-1 forward's head still stamps
//!   `k3_q5k_gemv` (the standard head stays K3 by
//!   construction).
//!
//! All arms hold the block logits to the CPU model's
//! block forward at the cross-backend bar (per-row argmax
//! agreement, max-abs under 0.5): the CPU head applies the
//! true finish, so a finish the routing dropped, swapped,
//! or re-ordered on its way into the K21F call cannot pass
//! the open arms. All arms need a discrete Vulkan GPU and
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

/// The K21F head switch's env name (see
/// `wgpu_compute::compute::K21F_HEAD_ENV`; not
/// re-exported at the crate root).
const K21F_HEAD_ENV: &str = "WGPU_COMPUTE_K21F_HEAD";

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

/// A Q5_K tensor of shape [inn, out], same construction,
/// with one deliberate divergence from the cross-backend
/// fixture: the twelve scale bytes are full-range, not
/// masked to `& 0x07`. This tensor is the head, the one
/// tensor the K21F route dequantizes; the masked fixture
/// shrank a dropped-nibble dequant error (the qc mask
/// bug, bench/k21f-mask-fix-2026-11-07.md) far under this
/// file's 0.5 bar by keeping every 6-bit scale field
/// tiny. Full-range scales (the k21_finish fixture's
/// discipline) make a wrong-nibble error land at full
/// size. The finish still bounds the comparison's
/// exposure: the logit scale (0.196) and the tanh
/// softcap compress the logits the bar reads, so the
/// wider weights do not widen honest reassociation
/// drift toward the bar.
fn q5k_tensor(rng: &mut Rng, name: &str, inn: usize, out: usize) -> TensorView {
    let blocks = inn * out / 256;
    let mut data = Vec::with_capacity(blocks * 176);
    for _ in 0..blocks {
        data.extend_from_slice(&f32_to_f16_bits(0.005).to_le_bytes());
        data.extend_from_slice(&f32_to_f16_bits(0.002).to_le_bytes());
        for _ in 0..12 {
            data.push(rng.next_u8());
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
/// parsed threshold reaches it. Unset and unparsable
/// values read as not-admitting here; the arms that
/// need those positions select on the raw env state
/// instead (see `mmv_ceiling_unset`,
/// `gate_explicitly_closed`).
fn gate_admits_block_batch() -> bool {
    std::env::var(MMV_MAX_ROWS_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .is_some_and(|max_rows| max_rows >= 16)
}

/// Whether the shared ceiling env is set to a position
/// that closes the band at the block batch: set, and
/// either unparsable (the backend resolves those to
/// its fail-closed fallback, 1) or parsed below 16.
/// Unset is NOT closed: the shipped default is 16,
/// which admits the block batch. The closed-gate arm
/// selects on this so it can never proceed at the
/// shipped defaults and mistake the default-open
/// ceiling for a closed one.
fn gate_explicitly_closed() -> bool {
    match std::env::var(MMV_MAX_ROWS_ENV) {
        Err(_) => false,
        Ok(v) => v
            .trim()
            .parse::<u32>()
            .map_or(true, |max_rows| max_rows < 16),
    }
}

/// The resolved head switch this process was launched
/// with, mirroring the backend's resolver exactly:
/// unset and the exact value `1` (after trimming) are
/// on; the exact value `0` and every unrecognized
/// value read as off (the explicit off and the
/// fail-closed position, which is what places them in
/// the switch-off arm).
fn head_switch_on() -> bool {
    match std::env::var(K21F_HEAD_ENV) {
        Err(_) => true,
        Ok(v) => v.trim() == "1",
    }
}

/// Whether the head switch env is entirely unset: the
/// default-construction arm's selection condition.
fn head_switch_unset() -> bool {
    std::env::var_os(K21F_HEAD_ENV).is_none()
}

/// Whether the head switch was set to the exact `1`
/// explicitly: the both-gates-open arm's selection
/// condition (unset resolves on too since the default
/// flip, but that position belongs to the
/// default-construction arm).
fn head_switch_explicitly_on() -> bool {
    std::env::var(K21F_HEAD_ENV).is_ok_and(|v| v.trim() == "1")
}

/// Whether the shared ceiling env is entirely unset:
/// at the shipped default (16) the backend admits the
/// block batch, which is half of the default-
/// construction arm's selection condition.
fn mmv_ceiling_unset() -> bool {
    std::env::var_os(MMV_MAX_ROWS_ENV).is_none()
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

/// Head switch off, exercised explicitly
/// (`WGPU_COMPUTE_K21F_HEAD=0`, the one-variable
/// rollback; every unrecognized switch value lands
/// here too, fail-closed): even with the shared
/// ceiling at its shipped open position, the block head
/// is the K3 fused call and no `k21_q5k_mmv_finish`
/// stamp appears anywhere. The off position must be
/// reachable and exact at every point after the
/// default flip.
#[test]
fn block_head_stays_k3_when_the_head_switch_is_off() {
    let test = "k21f switch-off routing pin";
    if head_switch_on() {
        if head_switch_unset() {
            eprintln!(
                "skipping {test}: {K21F_HEAD_ENV} is unset, which resolves \
                 the head switch ON at the shipped default; this arm needs \
                 {K21F_HEAD_ENV}=0 (or an unrecognized value)"
            );
        } else {
            eprintln!(
                "skipping {test}: {K21F_HEAD_ENV}=1 resolves the head switch \
                 on; this arm needs {K21F_HEAD_ENV}=0 (or an unrecognized value)"
            );
        }
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
        "the switch-off block head must stamp the K3 GEMV, saw {names:?}"
    );
    assert!(
        !names.contains(&"k21_q5k_mmv_finish"),
        "the switch-off position must never stamp the finish form, saw {names:?}"
    );
}

/// Default construction (both envs unset: the head
/// switch at its shipped default ON, the shared
/// ceiling at its shipped default 16): the block head
/// routes to the finish form. This is the default
/// flip's central pin (bench/k21f-default-flip-2026-11-07.md):
/// it proves the position a run lands in when nobody
/// sets anything, the position the wave18 retest
/// proved on the real file
/// (bench/k21f-retest-2026-11-07.md).
#[test]
fn block_head_routes_to_k21f_at_default_construction() {
    let test = "k21f default-construction routing pin";
    if !head_switch_unset() {
        eprintln!(
            "skipping {test}: {K21F_HEAD_ENV} is set; this arm needs it \
             unset (the shipped default position)"
        );
        return;
    }
    if !mmv_ceiling_unset() {
        eprintln!(
            "skipping {test}: {MMV_MAX_ROWS_ENV} is set; this arm needs it \
             unset (the shipped ceiling, 16)"
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
        names.contains(&"k21_q5k_mmv_finish"),
        "default construction must stamp the finish form at the block \
         batch, saw {names:?}"
    );
}

/// Band closed: whatever the head switch says, a shared
/// ceiling that does not admit the block batch keeps the
/// block head on the K3 fused call, and no
/// `k21_q5k_mmv_finish` stamp appears anywhere.
#[test]
fn block_head_stays_k3_fused_with_the_gate_closed() {
    let test = "k21f closed-gate routing pin";
    if !gate_explicitly_closed() {
        eprintln!(
            "skipping {test}: {MMV_MAX_ROWS_ENV} is not set to a closing \
             value (unset resolves to the shipped ceiling 16, which admits \
             the block batch); this arm needs {MMV_MAX_ROWS_ENV}=1"
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

/// Both gates open at the block batch (head switch on,
/// shared band admitting): the block head routes to the
/// finish form, the batch-1 head stays K3, and the
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
    if !head_switch_explicitly_on() {
        eprintln!(
            "skipping {test}: {K21F_HEAD_ENV} is not the exact \"1\"; this \
             arm needs the switch set explicitly (unset resolves on too, \
             but that position is the default-construction arm's)"
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
