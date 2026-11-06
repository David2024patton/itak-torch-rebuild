//! K21F (the finish-carrying Q5_K register-streaming
//! small-batch product) against two references on the
//! same inputs: the CPU f32 reference composed with
//! the head finishing map on the host, and K3's GEMV
//! with its fused finish (the shipped head's exact
//! composition). The band is K21's: 2, 3, 5, 7, 9, 16.
//!
//! The claim these gates exist to prove: swapping
//! the block head's product engine from K3 to K21F
//! changes no output beyond the family's
//! reassociation bound, shrunk by the finish. The
//! raw products agree inside the sibling files'
//! bound (`2e-3 + 5e-5 * set scale`); the finishing
//! map is Lipschitz with constant `scale` in both
//! modes (the scale mode exactly, the softcap
//! because `cap * tanh(scale * v / cap)` has
//! derivative `scale * sech^2(..) <= scale`), so the
//! finished bound is `scale` times the raw bound,
//! plus a small absolute slack for the one
//! evaluation that is not shared arithmetic, the
//! driver's `tanh` against the host's. Against K3's
//! fused finish even that slack is principled
//! headroom only: both kernels evaluate the same
//! WGSL `tanh` text on the same driver.
//!
//! The finish values are Glimmer's real ones (the
//! file's logit scale 0.19611613 and softcap 20.0):
//! the gates run the head's actual map, not a
//! convenient one. Both finish modes are gated, and
//! every reduction shape the device can build (the
//! workgroup-tree form always, the subgroup form
//! where the device grants SUBGROUP), because the
//! finish sits in the reduction's store in each.
//!
//! These tests need a discrete Vulkan GPU under the
//! default selection policy. On a machine without
//! one they print the refusal and pass; shader
//! validity on any adapter is gated separately in
//! `shader_compile.rs`, and the real-file proof runs
//! on the server through the verdict protocol in
//! bench/k21-finish-2026-11-06.md.

use gguf::GgmlType;
use wgpu_compute::kernels::k21_q5k_mmv::K21Shape;
use wgpu_compute::kernels::k21_q5k_mmv_finish::K21FinishPipeline;
use wgpu_compute::kernels::k3_q5k_gemv::{
    K3Pipeline, K3Weights, FINISH_NONE, FINISH_SCALE, FINISH_SOFTCAP,
};
use wgpu_compute::{GpuContext, WgpuCompute, WgpuError};

/// Glimmer's logit scale, the file value
/// (0.1961161345243454 in f64; the head consumes the
/// f32 rounding of it).
const GLIMMER_LOGIT_SCALE: f32 = 0.19611613;
/// Glimmer's logit softcap, the file value.
const GLIMMER_LOGIT_SOFTCAP: f32 = 20.0;

/// A backend, or `None` after printing why this
/// machine cannot provide one.
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

/// Deterministic pseudo-random f32s in [-0.5, 0.5)
/// via a 64-bit LCG, matching the other kernel
/// tests.
fn lcg_input(seed: u64, count: usize) -> Vec<f32> {
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

/// Deterministic pseudo-random bytes via a small
/// LCG.
fn lcg_bytes(seed: u8, count: usize) -> Vec<u8> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(31).wrapping_add(17);
            state
        })
        .collect()
}

/// A synthetic Q5_K tensor's raw bytes (the sibling
/// files' fixture: 176-byte blocks, rotating (d,
/// dmin) headers over LCG payloads; the scales, qh,
/// and ql planes are payload).
fn synthetic_q5k(in_elems: usize, out_rows: usize) -> Vec<u8> {
    let blocks_per_row = in_elems / 256;
    let mut data = Vec::with_capacity(out_rows * blocks_per_row * 176);
    let headers: [(u16, u16); 4] = [
        (0x3c00, 0x3400),
        (0x3800, 0x0000),
        (0x4000, 0x2e66),
        (0x3e00, 0x3800),
    ];
    for row in 0..out_rows {
        for blk in 0..blocks_per_row {
            let (d, dmin) = headers[(row + blk) % headers.len()];
            let mut block = [0u8; 176];
            block[0..2].copy_from_slice(&d.to_le_bytes());
            block[2..4].copy_from_slice(&dmin.to_le_bytes());
            block[4..176].copy_from_slice(&lcg_bytes((row * 7 + blk * 13 + 5) as u8, 172)[..]);
            data.extend_from_slice(&block);
        }
    }
    data
}

/// The CPU f32 reference: dequantize the tensor
/// once, then matvec per batch row (the plain
/// product, no finish).
fn reference_f32(bytes: &[u8], inn: usize, out_rows: usize, x: &[f32], batch: usize) -> Vec<f32> {
    let weights = gguf::dequant::dequantize(GgmlType::Q5K, bytes, inn * out_rows);
    let mut out = Vec::with_capacity(batch * out_rows);
    for b in 0..batch {
        out.extend_from_slice(
            &torch_math::matvec(&weights, out_rows, inn, &x[b * inn..(b + 1) * inn])
                .expect("cpu reference"),
        );
    }
    out
}

/// The head finishing map on the host, in K8's
/// expression order: the CPU composition the fused
/// kernels must match.
fn finish_host(v: f32, mode: u32, scale: f32, cap: f32) -> f32 {
    match mode {
        FINISH_SCALE => v * scale,
        FINISH_SOFTCAP => {
            let x = v * scale;
            cap * (x / cap).tanh()
        }
        _ => unreachable!("tests only ask for finishing modes"),
    }
}

/// Assert two finished f32 sets agree within the
/// Lipschitz-shrunk bound: `scale` times the
/// family's raw reassociation bound at the raw
/// set's scale, plus the tanh slack. Reports the
/// worst element.
fn check_close_finished(label: &str, got: &[f32], want: &[f32], raw_scale: f32, scale: f32) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    let tol = scale * (2e-3 + 5e-5 * raw_scale) + 2e-5;
    let mut max_diff = 0.0f32;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        max_diff = max_diff.max(diff);
        assert!(
            diff <= tol,
            "{label}: element {i}: got {g}, want {w} (diff {diff}, tol {tol} at raw set scale {raw_scale})"
        );
    }
    eprintln!("{label}: max abs diff {max_diff:.3e} (tol {tol:.3e})");
}

/// K3's GEMV with its fused finish over the same f32
/// rows: the shipped head composition.
#[allow(clippy::too_many_arguments)]
fn run_gemv_q5k_finish(
    ctx: &GpuContext,
    k3: &K3Pipeline,
    weights: &K3Weights,
    x: &[f32],
    batch: usize,
    out_rows: usize,
    mode: u32,
    scale: f32,
    cap: f32,
) -> Vec<f32> {
    let in_buf = ctx.upload_f32("k21 finish test gemv input", x);
    let out_buf = ctx.scratch_buffer("k21 finish test gemv out", (batch * out_rows * 4) as u64);
    let call = k3
        .prepare_with_finish(
            ctx,
            weights,
            &in_buf,
            &out_buf,
            batch as u32,
            mode,
            scale,
            cap,
        )
        .expect("k3 finish prepare");
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("k21 finish test gemv run"),
        });
    call.encode(&mut encoder);
    ctx.queue.submit([encoder.finish()]);
    ctx.read_f32(&out_buf, batch * out_rows)
        .expect("gemv finish readback")
}

/// The reduction shapes to gate: the tree form
/// always, the subgroup form when the device grants
/// it.
fn shape_candidates(features: wgpu::Features) -> Vec<K21Shape> {
    let mut shapes = vec![K21Shape::new(2, 128)];
    if features.contains(wgpu::Features::SUBGROUP) {
        shapes.push(K21Shape::new(2, 128).with_subgroup());
    }
    shapes
}

/// The batch counts the gates sweep (K21's band
/// points, including both of llama.cpp's
/// dose-response cliffs).
const BAND: [usize; 6] = [2, 3, 5, 7, 9, 16];

/// The finishes the gates run: Glimmer's scale-only
/// position and its shipped scale-plus-softcap
/// position.
const FINISHES: [(u32, f32, f32); 2] = [
    (FINISH_SCALE, GLIMMER_LOGIT_SCALE, 0.0),
    (FINISH_SOFTCAP, GLIMMER_LOGIT_SCALE, GLIMMER_LOGIT_SOFTCAP),
];

#[test]
fn k21_finish_matches_cpu_composition_and_k3_finish() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    // Width 512 = two super-blocks per row; 100
    // output rows = 50 two-row workgroups exactly.
    let (inn, out_rows) = (512usize, 100usize);
    let bytes = synthetic_q5k(inn, out_rows);
    let k3 = K3Pipeline::new(&ctx.device).expect("k3 pipeline");
    let weights = k3
        .upload(ctx, "synthetic.q5k.mmv.finish", &bytes, inn, out_rows)
        .expect("k3 upload");
    for shape in shape_candidates(ctx.device.features()) {
        let mmv = K21FinishPipeline::with_shape(&ctx.device, shape).expect("k21 finish pipeline");
        for batch in BAND {
            let x = lcg_input(700 + batch as u64, batch * inn);
            let raw = reference_f32(&bytes, inn, out_rows, &x, batch);
            let raw_scale = raw.iter().fold(0.0f32, |a, w| a.max(w.abs()));
            for (mode, scale, cap) in FINISHES {
                let got = mmv
                    .run_with_finish(ctx, &weights, &x, mode, scale, cap)
                    .expect("k21 finish run");
                let want: Vec<f32> = raw
                    .iter()
                    .map(|&v| finish_host(v, mode, scale, cap))
                    .collect();
                check_close_finished(
                    &format!(
                        "k21 finish {} batch {batch} mode {mode} vs cpu composition",
                        shape.name()
                    ),
                    &got,
                    &want,
                    raw_scale,
                    scale,
                );
                let gemv =
                    run_gemv_q5k_finish(ctx, &k3, &weights, &x, batch, out_rows, mode, scale, cap);
                check_close_finished(
                    &format!(
                        "k21 finish {} batch {batch} mode {mode} vs k3 finish",
                        shape.name()
                    ),
                    &got,
                    &gemv,
                    raw_scale,
                    scale,
                );
            }
        }
    }
}

#[test]
fn k21_finish_deep_geometry_and_odd_row_count() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    // Width 768 = three super-blocks per row; 101
    // output rows: the last workgroup covers one row
    // of its two-row tile, exercising the row guard
    // and the guarded finish store (a finish applied
    // to a row that does not exist would corrupt the
    // neighbor's logit, which the comparison sees).
    let (inn, out_rows) = (768usize, 101usize);
    let bytes = synthetic_q5k(inn, out_rows);
    let k3 = K3Pipeline::new(&ctx.device).expect("k3 pipeline");
    let weights = k3
        .upload(ctx, "synthetic.q5k.mmv.finish.deep", &bytes, inn, out_rows)
        .expect("k3 upload");
    let mmv = K21FinishPipeline::new(&ctx.device).expect("k21 finish pipeline");
    for batch in [3usize, 16] {
        let x = lcg_input(900 + batch as u64, batch * inn);
        let raw = reference_f32(&bytes, inn, out_rows, &x, batch);
        let raw_scale = raw.iter().fold(0.0f32, |a, w| a.max(w.abs()));
        for (mode, scale, cap) in FINISHES {
            let got = mmv
                .run_with_finish(ctx, &weights, &x, mode, scale, cap)
                .expect("k21 finish run");
            let want: Vec<f32> = raw
                .iter()
                .map(|&v| finish_host(v, mode, scale, cap))
                .collect();
            check_close_finished(
                &format!("k21 finish deep batch {batch} mode {mode} vs cpu composition"),
                &got,
                &want,
                raw_scale,
                scale,
            );
        }
    }
}

#[test]
fn k21_finish_at_the_heads_own_width() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    // The head's own geometry: 6,656 in (26
    // super-blocks per row, Glimmer's hidden width).
    // The output is cut to 101 rows to keep the
    // fixture small; the per-row math the finish
    // rides on is the head's exactly.
    let (inn, out_rows) = (6656usize, 101usize);
    let bytes = synthetic_q5k(inn, out_rows);
    let k3 = K3Pipeline::new(&ctx.device).expect("k3 pipeline");
    let weights = k3
        .upload(ctx, "synthetic.q5k.mmv.finish.head", &bytes, inn, out_rows)
        .expect("k3 upload");
    let mmv = K21FinishPipeline::new(&ctx.device).expect("k21 finish pipeline");
    for batch in [2usize, 16] {
        let x = lcg_input(1100 + batch as u64, batch * inn);
        let raw = reference_f32(&bytes, inn, out_rows, &x, batch);
        let raw_scale = raw.iter().fold(0.0f32, |a, w| a.max(w.abs()));
        for (mode, scale, cap) in FINISHES {
            let got = mmv
                .run_with_finish(ctx, &weights, &x, mode, scale, cap)
                .expect("k21 finish run");
            let want: Vec<f32> = raw
                .iter()
                .map(|&v| finish_host(v, mode, scale, cap))
                .collect();
            check_close_finished(
                &format!("k21 finish head-width batch {batch} mode {mode} vs cpu composition"),
                &got,
                &want,
                raw_scale,
                scale,
            );
            let gemv =
                run_gemv_q5k_finish(ctx, &k3, &weights, &x, batch, out_rows, mode, scale, cap);
            check_close_finished(
                &format!("k21 finish head-width batch {batch} mode {mode} vs k3 finish"),
                &got,
                &gemv,
                raw_scale,
                scale,
            );
        }
    }
}

#[test]
fn k21_finish_refusals_are_named() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    let (inn, out_rows) = (512usize, 100usize);
    let bytes = synthetic_q5k(inn, out_rows);
    let k3 = K3Pipeline::new(&ctx.device).expect("k3 pipeline");
    let weights = k3
        .upload(ctx, "synthetic.q5k.mmv.finish.edges", &bytes, inn, out_rows)
        .expect("k3 upload");
    let mmv = K21FinishPipeline::new(&ctx.device).expect("k21 finish pipeline");
    let in_buf = ctx.scratch_buffer("k21 finish test in", 4096);
    let out_buf = ctx.scratch_buffer("k21 finish test out", 4096);
    // The plain product's mode is not a finishing
    // mode here, and neither is an unknown one: both
    // refuse with K3's words, at a batch inside the
    // band, so the finish check is the one that
    // fires.
    for mode in [FINISH_NONE, 7] {
        let err = mmv
            .prepare_with_finish(
                ctx,
                &weights,
                &in_buf,
                &out_buf,
                2,
                mode,
                GLIMMER_LOGIT_SCALE,
                GLIMMER_LOGIT_SOFTCAP,
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("is not a finishing mode"),
            "mode {mode} refusal names the rule: {err}"
        );
    }
    // A softcap with no positive cap refuses; the
    // scale mode never reads the cap and prepares
    // fine with a zero one (K3's contract, kept).
    for cap in [0.0, -1.0] {
        let err = mmv
            .prepare_with_finish(
                ctx,
                &weights,
                &in_buf,
                &out_buf,
                2,
                FINISH_SOFTCAP,
                GLIMMER_LOGIT_SCALE,
                cap,
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("is not positive"),
            "cap {cap} refusal names the rule: {err}"
        );
    }
    assert!(mmv
        .prepare_with_finish(
            ctx,
            &weights,
            &in_buf,
            &out_buf,
            2,
            FINISH_SCALE,
            GLIMMER_LOGIT_SCALE,
            0.0,
        )
        .is_ok());
    // The band still bounds the finish form: batch 1
    // (the standard head's size) and batch 17 refuse
    // by name, with a valid finish requested, so the
    // band check is the one that fires.
    for batch in [1u32, 17] {
        let err = mmv
            .prepare_with_finish(
                ctx,
                &weights,
                &in_buf,
                &out_buf,
                batch,
                FINISH_SOFTCAP,
                GLIMMER_LOGIT_SCALE,
                GLIMMER_LOGIT_SOFTCAP,
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("outside 2..=16"),
            "batch {batch} refusal names the band: {err}"
        );
    }
    // The one-shot path derives the batch from the
    // input, so a single row refuses the same way.
    let x = lcg_input(42, inn);
    let err = mmv
        .run_with_finish(
            ctx,
            &weights,
            &x,
            FINISH_SOFTCAP,
            GLIMMER_LOGIT_SCALE,
            GLIMMER_LOGIT_SOFTCAP,
        )
        .unwrap_err();
    assert!(
        err.to_string().contains("outside 2..=16"),
        "one-row run refusal names the band: {err}"
    );
}
