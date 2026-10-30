//! K10/K11 (the Q4_K and Q6_K prefill GEMMs) against synthetic
//! tensors with known contents: the GPU results must match the
//! CPU reference (the `gguf` crate's dequantizer plus a naive
//! host matmul) within float reassociation noise, across batch
//! sizes that cross the GEMM row threshold's neighborhood and
//! output widths that are not tile multiples, and through the
//! activation forms the resident session's chunk plans use
//! (the gated output projection and the SwiGLU-fused down
//! projection, at the paired gate/up geometry).
//!
//! These tests need a discrete Vulkan GPU under the default
//! selection policy. On a machine without one they print the
//! refusal and pass; the real-file proof runs on the server
//! through the resident session (the GEMM's only consumer).

use wgpu_compute::kernels::k10_q4k_gemm::{GemmShape, K10Pipeline};
use wgpu_compute::kernels::k11_q6k_gemm::K11Pipeline;
use wgpu_compute::kernels::k1_q4k_gemv::{K1ActGeom, K1Pipeline, ACT_GATED, ACT_NONE, ACT_SWIGLU};
use wgpu_compute::kernels::k2_q6k_gemv::K2Pipeline;
use wgpu_compute::{WgpuCompute, WgpuError};

/// A backend, or `None` after printing why this machine cannot
/// provide one.
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

/// Deterministic pseudo-random f32s in [-0.5, 0.5) via a 64-bit
/// LCG, matching the other kernel tests.
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

/// A synthetic Q4_K tensor's raw bytes: `[out_rows][in_blocks]`
/// super-blocks with pseudo-random payloads and a rotating set
/// of (d, dmin) headers so rows exercise different scales.
fn synthetic_q4k(in_elems: usize, out_rows: usize) -> Vec<u8> {
    let blocks_per_row = in_elems / 256;
    let mut data = Vec::with_capacity(out_rows * blocks_per_row * 144);
    let headers: [(u16, u16); 4] = [
        (0x3c00, 0x3400), // d 1.0, dmin 0.25
        (0x3800, 0x0000), // d 0.5, dmin 0
        (0x4000, 0x2e66), // d 2.0, dmin ~0.1
        (0x3e00, 0x3800), // d 0.75, dmin 0.5
    ];
    for row in 0..out_rows {
        for blk in 0..blocks_per_row {
            let (d, dmin) = headers[(row + blk) % headers.len()];
            let mut block = [0u8; 144];
            block[0..2].copy_from_slice(&d.to_le_bytes());
            block[2..4].copy_from_slice(&dmin.to_le_bytes());
            block[4..144].copy_from_slice(&lcg_bytes((row * 7 + blk * 13 + 5) as u8, 140)[..]);
            data.extend_from_slice(&block);
        }
    }
    data
}

/// A synthetic Q6_K tensor's raw bytes (210-byte blocks, the
/// f16 super-scale at offset 208).
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

/// The naive host GEMM over dequantized weights:
/// `out[m][n] = sum_k w[n][k] * x[m][k]`, with the optional
/// elementwise transform the fused forms apply to `x` first
/// (gate: `a * sigmoid(b)` in K7's branch-stable form; SwiGLU:
/// `silu(a) * b` in K7's quotient form), `b` addressed by the
/// same stride/offset geometry the kernels take.
fn reference_gemm(
    w: &[f32],
    in_elems: usize,
    out_rows: usize,
    x_a: &[f32],
    act: Option<(&[f32], usize, usize, u32)>,
    batch: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * out_rows];
    for m in 0..batch {
        for n in 0..out_rows {
            let mut acc = 0.0f32;
            for k in 0..in_elems {
                let mut a = x_a[m * in_elems + k];
                if let Some((x_b, b_stride, b_off, mode)) = act {
                    let b = x_b[m * b_stride + b_off + k];
                    a = if mode == ACT_GATED {
                        a * sigmoid(b)
                    } else {
                        silu(a) * b
                    };
                }
                acc += w[n * in_elems + k] * a;
            }
            out[m * out_rows + n] = acc;
        }
    }
    out
}

/// K7's branch-stable sigmoid, mirrored on the host.
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// K7's SiLU quotient form, mirrored on the host.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Assert two result sets agree within reassociation noise,
/// scaled to the reference magnitude.
fn check_close(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    let mut max_diff = 0.0f32;
    let mut max_rel = 0.0f32;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        let rel = diff / w.abs().max(1.0);
        max_diff = max_diff.max(diff);
        max_rel = max_rel.max(rel);
        assert!(
            diff <= 2e-3 + 1e-4 * w.abs(),
            "{label}: element {i}: got {g}, want {w} (diff {diff})"
        );
    }
    eprintln!("{label}: max abs diff {max_diff:.3e}, max scaled rel {max_rel:.3e}");
}

/// The shipped GEMM shape on this device, plus the narrow
/// tile as a second configuration: the correctness story must
/// hold for every shape the fallback ladder can pick.
fn shapes(backend: &WgpuCompute) -> Vec<GemmShape> {
    let mut out = Vec::new();
    if let Some(s) = wgpu_compute::kernels::k10_q4k_gemm::shipped_shape(
        &backend.context().device.limits(),
    ) {
        out.push(s);
    }
    let narrow = GemmShape::new(32, 128);
    if !out.contains(&narrow) {
        out.push(narrow);
    }
    out
}

#[test]
fn k10_gemm_matches_cpu_reference_across_batches() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    let (inn, out_rows) = (512usize, 100usize); // 100 is not a tile multiple
    let bytes = synthetic_q4k(inn, out_rows);
    let w_ref = gguf::dequant::dequantize(gguf::GgmlType::Q4K, &bytes, inn * out_rows);
    let k1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let weights = k1
        .upload(ctx, "synthetic.q4k", &bytes, inn, out_rows)
        .expect("k1 upload");
    for shape in shapes(&backend) {
        let gemm = K10Pipeline::with_shape_act(&ctx.device, shape, ACT_NONE)
            .expect("k10 pipeline builds");
        for batch in [1usize, 5, 16, 37, 64] {
            let x = lcg_input(100 + batch as u64, batch * inn);
            let got = gemm.run(ctx, &weights, &x).expect("k10 run");
            let want = reference_gemm(&w_ref, inn, out_rows, &x, None, batch);
            check_close(
                &format!("k10 {} batch {batch}", shape.name()),
                &got,
                &want,
            );
        }
    }
}

#[test]
fn k10_gemm_act_forms_match_cpu_reference() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    // The paired SwiGLU geometry: the down projection reads
    // [rows][2 * ffn] pair rows, gate segment first; and the
    // gated geometry: attention rows against a gate stream at
    // its own stride.
    let (inn, out_rows, batch) = (512usize, 96usize, 29usize);
    let bytes = synthetic_q4k(inn, out_rows);
    let w_ref = gguf::dequant::dequantize(gguf::GgmlType::Q4K, &bytes, inn * out_rows);
    let k1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let weights = k1
        .upload(ctx, "synthetic.q4k.act", &bytes, inn, out_rows)
        .expect("k1 upload");
    for shape in shapes(&backend) {
        for act in [ACT_GATED, ACT_SWIGLU] {
            let gemm = K10Pipeline::with_shape_act(&ctx.device, shape, act).expect("k10 act");
            // Stream A rows are 2 * inn apart (the pair
            // layout); stream B is the same buffer at offset
            // inn for SwiGLU, a standalone buffer for gated.
            let a_stride = 2 * inn;
            let x_a = lcg_input(55, batch * a_stride);
            let (x_b_vec, b_stride, b_off) = if act == ACT_SWIGLU {
                (x_a.clone(), a_stride, inn)
            } else {
                (lcg_input(77, batch * inn), inn, 0)
            };
            let a_buf = ctx.upload_f32("k10 test a", &x_a);
            let b_buf = ctx.upload_f32("k10 test b", &x_b_vec);
            let out_buf = ctx.scratch_buffer("k10 test out", (batch * out_rows * 4) as u64);
            let geom = K1ActGeom {
                a_stride_elems: a_stride,
                b_stride_elems: b_stride,
                b_off_elems: b_off,
            };
            let call = gemm
                .prepare_act(ctx, &weights, &a_buf, &b_buf, &out_buf, batch as u32, &geom)
                .expect("k10 prepare_act");
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("k10 act test"),
                });
            call.encode(&mut encoder);
            ctx.queue.submit([encoder.finish()]);
            let got = ctx
                .read_f32(&out_buf, batch * out_rows)
                .expect("k10 readback");
            // The reference reads stream A at its own stride;
            // rebuild the compacted A rows it addresses.
            let mut a_rows = vec![0.0f32; batch * inn];
            for m in 0..batch {
                a_rows[m * inn..(m + 1) * inn]
                    .copy_from_slice(&x_a[m * a_stride..m * a_stride + inn]);
            }
            let want = reference_gemm(
                &w_ref,
                inn,
                out_rows,
                &a_rows,
                Some((&x_b_vec, b_stride, b_off, act)),
                batch,
            );
            check_close(
                &format!("k10 {} act {act}", shape.name()),
                &got,
                &want,
            );
        }
    }
}

#[test]
fn k11_gemm_matches_cpu_reference_across_batches() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    let (inn, out_rows) = (512usize, 100usize);
    let bytes = synthetic_q6k(inn, out_rows);
    let w_ref = gguf::dequant::dequantize(gguf::GgmlType::Q6K, &bytes, inn * out_rows);
    let k2 = K2Pipeline::new(&ctx.device).expect("k2 pipeline");
    let weights = k2
        .upload(ctx, "synthetic.q6k", &bytes, inn, out_rows)
        .expect("k2 upload");
    for shape in shapes(&backend) {
        let gemm = K11Pipeline::with_shape_act(&ctx.device, shape, ACT_NONE)
            .expect("k11 pipeline builds");
        for batch in [1usize, 5, 16, 37, 64] {
            let x = lcg_input(200 + batch as u64, batch * inn);
            let got = gemm.run(ctx, &weights, &x).expect("k11 run");
            let want = reference_gemm(&w_ref, inn, out_rows, &x, None, batch);
            check_close(
                &format!("k11 {} batch {batch}", shape.name()),
                &got,
                &want,
            );
        }
    }
}

#[test]
fn k11_gemm_swiglu_matches_cpu_reference() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    let (inn, out_rows, batch) = (512usize, 96usize, 29usize);
    let bytes = synthetic_q6k(inn, out_rows);
    let w_ref = gguf::dequant::dequantize(gguf::GgmlType::Q6K, &bytes, inn * out_rows);
    let k2 = K2Pipeline::new(&ctx.device).expect("k2 pipeline");
    let weights = k2
        .upload(ctx, "synthetic.q6k.act", &bytes, inn, out_rows)
        .expect("k2 upload");
    for shape in shapes(&backend) {
        let gemm = K11Pipeline::with_shape_act(&ctx.device, shape, ACT_SWIGLU)
            .expect("k11 swiglu pipeline builds");
        let a_stride = 2 * inn;
        let x_a = lcg_input(99, batch * a_stride);
        let a_buf = ctx.upload_f32("k11 test a", &x_a);
        let out_buf = ctx.scratch_buffer("k11 test out", (batch * out_rows * 4) as u64);
        let geom = K1ActGeom {
            a_stride_elems: a_stride,
            b_stride_elems: a_stride,
            b_off_elems: inn,
        };
        let call = gemm
            .prepare_act(ctx, &weights, &a_buf, &a_buf, &out_buf, batch as u32, &geom)
            .expect("k11 prepare_act");
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k11 act test"),
            });
        call.encode(&mut encoder);
        ctx.queue.submit([encoder.finish()]);
        let got = ctx
            .read_f32(&out_buf, batch * out_rows)
            .expect("k11 readback");
        let mut a_rows = vec![0.0f32; batch * inn];
        for m in 0..batch {
            a_rows[m * inn..(m + 1) * inn].copy_from_slice(&x_a[m * a_stride..m * a_stride + inn]);
        }
        let want = reference_gemm(
            &w_ref,
            inn,
            out_rows,
            &a_rows,
            Some((&x_a, a_stride, inn, ACT_SWIGLU)),
            batch,
        );
        check_close(&format!("k11 {} swiglu", shape.name()), &got, &want);
    }
}

#[test]
fn gemm_prepare_refuses_out_of_range_batches() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    let (inn, out_rows) = (256usize, 64usize);
    let bytes = synthetic_q4k(inn, out_rows);
    let k1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let weights = k1
        .upload(ctx, "synthetic.q4k.refusal", &bytes, inn, out_rows)
        .expect("k1 upload");
    let gemm = K10Pipeline::with_shape_act(&ctx.device, GemmShape::new(32, 128), ACT_NONE)
        .expect("k10 pipeline builds");
    let input = ctx.scratch_buffer("k10 refusal in", (65 * inn * 4) as u64);
    let output = ctx.scratch_buffer("k10 refusal out", (65 * out_rows * 4) as u64);
    assert!(gemm.prepare(ctx, &weights, &input, &output, 0).is_err());
    assert!(gemm.prepare(ctx, &weights, &input, &output, 65).is_err());
    assert!(gemm.prepare(ctx, &weights, &input, &output, 64).is_ok());
}
