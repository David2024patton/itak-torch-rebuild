//! K20 (the Q6_K register-streaming small-batch
//! product) against the CPU f32 reference and against
//! K2's GEMV on the same inputs, at the row band the
//! kernel exists for: 2, 3, 5, 7, 9, 16.
//!
//! K20 computes the GEMV's own math (f32 activations,
//! weights dequantized in registers, f32 dots), so the
//! gates are tighter than K18's: the reference is the
//! crate's standard one (`gguf` dequantize plus
//! `torch-math` matvec per batch row), and the bound is
//! the sibling files' reassociation bound. The K2
//! cross-check runs the same inputs through the proven
//! GEMV: the two must agree inside the same bound,
//! which is the parity claim the routing relies on
//! (a plan can swap the forms without moving a byte or
//! changing the math).
//!
//! Every reduction shape the device can build is
//! gated (the workgroup-tree form always, the subgroup
//! form where the device grants SUBGROUP): the two
//! reductions sum the same partials in different
//! orders, and a slice-indexing defect in either is
//! exactly what this file exists to catch.
//!
//! These tests need a discrete Vulkan GPU under the
//! default selection policy. On a machine without one
//! they print the refusal and pass; shader validity on
//! any adapter is gated separately in
//! `shader_compile.rs`, and the real-file proof runs on
//! the server through the spec verdict runs
//! (bench/mmv-q6k-2026-11-05.md).

use gguf::GgmlType;
use wgpu_compute::kernels::k20_q6k_mmv::{K20Pipeline, MmvShape};
use wgpu_compute::kernels::k2_q6k_gemv::{K2Pipeline, K2Weights};
use wgpu_compute::{GpuContext, WgpuCompute, WgpuError};

/// A backend, or `None` after printing why this machine
/// cannot provide one.
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

/// Deterministic pseudo-random f32s in [-0.5, 0.5) via
/// a 64-bit LCG, matching the other kernel tests.
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

/// Deterministic pseudo-random bytes via a small LCG.
fn lcg_bytes(seed: u8, count: usize) -> Vec<u8> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(31).wrapping_add(17);
            state
        })
        .collect()
}

/// A synthetic Q6_K tensor's raw bytes (the sibling
/// files' fixture: 210-byte blocks of LCG payload with
/// a rotating f16 super-scale at offset 208).
fn synthetic_q6k(in_elems: usize, out_rows: usize) -> Vec<u8> {
    let blocks_per_row = in_elems / 256;
    let mut data = Vec::with_capacity(out_rows * blocks_per_row * 210);
    let ds: [u16; 3] = [0x3c00, 0x3800, 0x4000]; // 1.0, 0.5, 2.0
    for row in 0..out_rows {
        for blk in 0..blocks_per_row {
            let mut block = lcg_bytes((row * 11 + blk * 7 + 3) as u8, 210);
            block[208..210].copy_from_slice(&ds[(row + blk) % ds.len()].to_le_bytes());
            data.extend_from_slice(&block);
        }
    }
    data
}

/// Assert two f32 sets agree within the sibling files'
/// reassociation bound, reporting the worst element.
fn check_close(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    let scale = want.iter().fold(0.0f32, |a, w| a.max(w.abs()));
    let tol = 2e-3 + 5e-5 * scale;
    let mut max_diff = 0.0f32;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        max_diff = max_diff.max(diff);
        assert!(
            diff <= tol,
            "{label}: element {i}: got {g}, want {w} (diff {diff}, tol {tol} at set scale {scale})"
        );
    }
    eprintln!(
        "{label}: max abs diff {max_diff:.3e} (set scale {scale:.3e}, rel {:.3e})",
        max_diff / scale.max(1.0)
    );
}

/// The CPU f32 reference: dequantize the tensor once,
/// then matvec per batch row.
fn reference_f32(bytes: &[u8], inn: usize, out_rows: usize, x: &[f32], batch: usize) -> Vec<f32> {
    let weights = gguf::dequant::dequantize(GgmlType::Q6K, bytes, inn * out_rows);
    let mut out = Vec::with_capacity(batch * out_rows);
    for b in 0..batch {
        out.extend_from_slice(
            &torch_math::matvec(&weights, out_rows, inn, &x[b * inn..(b + 1) * inn])
                .expect("cpu reference"),
        );
    }
    out
}

/// K2's GEMV over the same f32 rows.
fn run_gemv_q6k(
    ctx: &GpuContext,
    k2: &K2Pipeline,
    weights: &K2Weights,
    x: &[f32],
    batch: usize,
    out_rows: usize,
) -> Vec<f32> {
    let in_buf = ctx.upload_f32("k20 test gemv input", x);
    let out_buf = ctx.scratch_buffer("k20 test gemv out", (batch * out_rows * 4) as u64);
    let call = k2
        .prepare(ctx, weights, &in_buf, &out_buf, batch as u32)
        .expect("k2 prepare");
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("k20 test gemv run"),
        });
    call.encode(&mut encoder);
    ctx.queue.submit([encoder.finish()]);
    ctx.read_f32(&out_buf, batch * out_rows)
        .expect("gemv readback")
}

/// The reduction shapes to gate: the tree form always,
/// the subgroup form when the device grants it (the
/// shipped shape is whichever the device takes; both
/// are exercised so the shipped choice is never the
/// only proven one).
fn shape_candidates(features: wgpu::Features) -> Vec<MmvShape> {
    let mut shapes = vec![MmvShape::new(2, 128)];
    if features.contains(wgpu::Features::SUBGROUP) {
        shapes.push(MmvShape::new(2, 128).with_subgroup());
    }
    shapes
}

/// The batch counts the gates sweep: the task's named
/// rows across the band, including both cliff points
/// of llama.cpp's dose-response (9, where the stock
/// cap breaks, and 16, the cap this kernel ships).
const BAND: [usize; 6] = [2, 3, 5, 7, 9, 16];

#[test]
fn k20_matches_cpu_reference_and_gemv_at_small_batches() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    // Width 512 = two super-blocks per row; 100 output
    // rows = 50 two-row workgroups exactly.
    let (inn, out_rows) = (512usize, 100usize);
    let bytes = synthetic_q6k(inn, out_rows);
    let k2 = K2Pipeline::new(&ctx.device).expect("k2 pipeline");
    let weights = k2
        .upload(ctx, "synthetic.q6k.mmv", &bytes, inn, out_rows)
        .expect("k2 upload");
    for shape in shape_candidates(ctx.device.features()) {
        let mmv = K20Pipeline::with_shape(&ctx.device, shape).expect("k20 pipeline");
        for batch in BAND {
            let x = lcg_input(700 + batch as u64, batch * inn);
            let got = mmv.run(ctx, &weights, &x).expect("k20 run");
            let want = reference_f32(&bytes, inn, out_rows, &x, batch);
            check_close(
                &format!("k20 {} batch {batch} vs cpu", shape.name()),
                &got,
                &want,
            );
            let gemv = run_gemv_q6k(ctx, &k2, &weights, &x, batch, out_rows);
            check_close(
                &format!("k20 {} batch {batch} vs gemv", shape.name()),
                &got,
                &gemv,
            );
        }
    }
}

#[test]
fn k20_deep_geometry_and_odd_row_count() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    // Width 768 = three super-blocks per row; 101
    // output rows: the last workgroup covers one row
    // of its two-row tile, exercising the row guard and
    // the guarded finish store.
    let (inn, out_rows) = (768usize, 101usize);
    let bytes = synthetic_q6k(inn, out_rows);
    let k2 = K2Pipeline::new(&ctx.device).expect("k2 pipeline");
    let weights = k2
        .upload(ctx, "synthetic.q6k.mmv.deep", &bytes, inn, out_rows)
        .expect("k2 upload");
    let mmv = K20Pipeline::new(&ctx.device).expect("k20 pipeline");
    for batch in [3usize, 16] {
        let x = lcg_input(900 + batch as u64, batch * inn);
        let got = mmv.run(ctx, &weights, &x).expect("k20 run");
        let want = reference_f32(&bytes, inn, out_rows, &x, batch);
        check_close(&format!("k20 deep batch {batch} vs cpu"), &got, &want);
    }
}

#[test]
fn k20_band_edges_refuse_by_name() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    let (inn, out_rows) = (512usize, 100usize);
    let bytes = synthetic_q6k(inn, out_rows);
    let k2 = K2Pipeline::new(&ctx.device).expect("k2 pipeline");
    let weights = k2
        .upload(ctx, "synthetic.q6k.mmv.edges", &bytes, inn, out_rows)
        .expect("k2 upload");
    let mmv = K20Pipeline::new(&ctx.device).expect("k20 pipeline");
    let in_buf = ctx.scratch_buffer("k20 test in", 4096);
    let out_buf = ctx.scratch_buffer("k20 test out", 4096);
    // Below the floor: batch 1 is decode, the GEMV's
    // job; batch 0 is nothing at all.
    for batch in [0u32, 1] {
        let err = mmv
            .prepare(ctx, &weights, &in_buf, &out_buf, batch)
            .unwrap_err();
        assert!(
            err.to_string().contains("outside 2..=16"),
            "batch {batch} refusal names the band: {err}"
        );
    }
    // Past the cap: the GEMM ladder's territory.
    let err = mmv
        .prepare(ctx, &weights, &in_buf, &out_buf, 17)
        .unwrap_err();
    assert!(
        err.to_string().contains("outside 2..=16"),
        "batch 17 refusal names the band: {err}"
    );
    // The one-shot path derives the batch from the
    // input, so a single row refuses the same way.
    let x = lcg_input(42, inn);
    let err = mmv.run(ctx, &weights, &x).unwrap_err();
    assert!(
        err.to_string().contains("outside 2..=16"),
        "one-row run refusal names the band: {err}"
    );
}
