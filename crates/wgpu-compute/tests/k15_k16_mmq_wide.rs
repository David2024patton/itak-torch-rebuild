//! K15/K16 (the wide-tile double-buffered Q4_K / Q6_K MMQ
//! product kernels) against the same synthetic tensors and
//! host references as the K13/K14 gates in
//! `k12_k14_mmq.rs`, which this file mirrors on purpose:
//! the wide kernels change the tile geometry and the
//! staging pipeline, never the math, so the gates are the
//! same two layers at the same tolerances.
//!
//! - The product must match a host reference running the
//!   SAME quantized math (exact integer dots plus the
//!   per-sub-block scale fold) within float reassociation
//!   noise. The plain forms build the reference from K12's
//!   host quantizer; the activation forms build it from the
//!   GPU quantizer's own read-back output, for the
//!   rounding-tie reason the K13/K14 file documents.
//! - Against the f32 reference the results must land
//!   inside the activation-quantization band (asserted at
//!   the same loose 5% bound, measured value printed).
//!
//! Every wide tile candidate the device can build is
//! gated across the batch sweep, not just the shipped
//! shape: the candidates differ only in a tile-width
//! literal, and a staging index that is wrong at one
//! width is exactly the defect class this file exists to
//! catch. The double-buffered pipeline gets its own
//! targeted geometry: widths with several super-blocks
//! per row (so the parity pipeline runs multiple
//! iterations, including an odd step count) and output
//! row counts that straddle tile boundaries.
//!
//! These tests need a discrete Vulkan GPU under the
//! default selection policy. On a machine without one
//! they print the refusal and pass; shader validity on
//! any adapter is gated separately in `shader_compile.rs`,
//! and the real-file proof runs on the server through the
//! resident session and the bench script.

use wgpu_compute::kernels::k12_q8_quant::{quantize_row_q8_1, K12QuantPipeline};
use wgpu_compute::kernels::k15_q4k_mmq_wide::{
    shipped_wide_shape as shipped_wide_q4k, K15Pipeline, WideMmqShape, WIDE_TILE_CANDIDATES,
};
use wgpu_compute::kernels::k16_q6k_mmq_wide::{
    shipped_wide_shape as shipped_wide_q6k, K16Pipeline,
};
use wgpu_compute::kernels::k1_q4k_gemv::{K1ActGeom, K1Pipeline, ACT_GATED, ACT_SWIGLU};
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

/// A synthetic Q4_K tensor's raw bytes (the GEMM test's
/// fixture: rotating (d, dmin) headers over LCG payloads).
fn synthetic_q4k(in_elems: usize, out_rows: usize) -> Vec<u8> {
    let blocks_per_row = in_elems / 256;
    let mut data = Vec::with_capacity(out_rows * blocks_per_row * 144);
    let headers: [(u16, u16); 4] = [
        (0x3c00, 0x3400),
        (0x3800, 0x0000),
        (0x4000, 0x2e66),
        (0x3e00, 0x3800),
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
    let ds: [u16; 3] = [0x3c00, 0x3800, 0x4000];
    for row in 0..out_rows {
        for blk in 0..blocks_per_row {
            let mut block = lcg_bytes((row * 11 + blk * 7 + 3) as u8, 210);
            block[208..210].copy_from_slice(&ds[(row + blk) % ds.len()].to_le_bytes());
            data.extend_from_slice(&block);
        }
    }
    data
}

/// Unpack a quantized row's words into per-element quants.
fn unpack_qs(words: &[u32]) -> Vec<i32> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for w in words {
        for byte in w.to_le_bytes() {
            out.push(i32::from(byte as i8));
        }
    }
    out
}

/// ggml's `get_scale_min_k4`, mirrored from the gguf crate.
fn scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

/// The host MMQ reference for Q4_K: exact integer dots of the
/// weight nibbles against the host-quantized activations,
/// folded per sub-block in super-block order with the same
/// f32 expression the kernel uses
/// (`vec_dot_q4_K_q8_1`'s shape).
fn reference_mmq_q4k(
    raw: &[u8],
    in_elems: usize,
    out_rows: usize,
    qs: &[u32],
    ds: &[f32],
    batch: usize,
) -> Vec<f32> {
    let blocks_per_row = in_elems / 256;
    let blocks32 = in_elems / 32;
    let mut out = vec![0.0f32; batch * out_rows];
    for m in 0..batch {
        let qx = unpack_qs(&qs[m * blocks32 * 8..(m + 1) * blocks32 * 8]);
        let row_ds = &ds[m * blocks32 * 4..(m + 1) * blocks32 * 4];
        for n in 0..out_rows {
            let mut acc = 0.0f32;
            for b in 0..blocks_per_row {
                let block = &raw[(n * blocks_per_row + b) * 144..][..144];
                let d = gguf::dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let dmin = gguf::dequant::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
                let scales = &block[4..16];
                let payload = &block[16..144];
                for j in 0..8usize {
                    let (sc, mn) = scale_min_k4(j, scales);
                    let g = j / 2;
                    let hi = j % 2 == 1;
                    let mut sumi: i64 = 0;
                    for l in 0..32 {
                        let byte = payload[g * 32 + l];
                        let nib = if hi { byte >> 4 } else { byte & 0xF };
                        sumi += i64::from(nib) * i64::from(qx[b * 256 + j * 32 + l]);
                    }
                    let blk32 = b * 8 + j;
                    let (dx, s) = (row_ds[blk32 * 4], row_ds[blk32 * 4 + 1]);
                    acc += (d * f32::from(sc)) * (dx * sumi as f32) - (dmin * f32::from(mn)) * s;
                }
            }
            out[m * out_rows + n] = acc;
        }
    }
    out
}

/// The host MMQ reference for Q6_K: exact integer dots of
/// the assembled 6-bit quants against the host-quantized
/// activations, folded per 16-element half
/// (`vec_dot_q6_K_q8_1`'s shape).
fn reference_mmq_q6k(
    raw: &[u8],
    in_elems: usize,
    out_rows: usize,
    qs: &[u32],
    ds: &[f32],
    batch: usize,
) -> Vec<f32> {
    let blocks_per_row = in_elems / 256;
    let blocks32 = in_elems / 32;
    let mut out = vec![0.0f32; batch * out_rows];
    for m in 0..batch {
        let qx = unpack_qs(&qs[m * blocks32 * 8..(m + 1) * blocks32 * 8]);
        let row_ds = &ds[m * blocks32 * 4..(m + 1) * blocks32 * 4];
        for n in 0..out_rows {
            let mut acc = 0.0f32;
            for b in 0..blocks_per_row {
                let block = &raw[(n * blocks_per_row + b) * 210..][..210];
                let ql = &block[0..128];
                let qh = &block[128..192];
                let sc = &block[192..208];
                let d = gguf::dequant::f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
                for j in 0..8usize {
                    let h = j / 4;
                    let s = j % 4;
                    let ql_h = &ql[h * 64..h * 64 + 64];
                    let qh_h = &qh[h * 32..h * 32 + 32];
                    // The sub-block's 32 quants, in element
                    // order (dequant_q6_k's interleave).
                    let mut qw = [0i64; 32];
                    for l in 0..32 {
                        qw[l] = match s {
                            0 => i64::from((ql_h[l] & 0xF) | ((qh_h[l] & 3) << 4)),
                            1 => i64::from((ql_h[l + 32] & 0xF) | (((qh_h[l] >> 2) & 3) << 4)),
                            2 => i64::from((ql_h[l] >> 4) | (((qh_h[l] >> 4) & 3) << 4)),
                            _ => i64::from((ql_h[l + 32] >> 4) | (((qh_h[l] >> 6) & 3) << 4)),
                        };
                    }
                    let blk32 = b * 8 + j;
                    let dx = row_ds[blk32 * 4];
                    for half in 0..2usize {
                        let sci = sc[h * 8 + half + 2 * s] as i8;
                        let mut sumi: i64 = 0;
                        for l in 0..16 {
                            sumi +=
                                qw[half * 16 + l] * i64::from(qx[b * 256 + j * 32 + half * 16 + l]);
                        }
                        let sh = row_ds[blk32 * 4 + 2 + half];
                        acc += (d * f32::from(sci)) * (dx * sumi as f32 - 32.0 * sh);
                    }
                }
            }
            out[m * out_rows + n] = acc;
        }
    }
    out
}

/// The naive f32 GEMM over dequantized weights (the GEMM
/// test's reference): the band reference for quantization
/// error, not the correctness gate.
fn reference_gemm_f32(
    w: &[f32],
    in_elems: usize,
    out_rows: usize,
    x: &[f32],
    batch: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * out_rows];
    for m in 0..batch {
        for n in 0..out_rows {
            let mut acc = 0.0f32;
            for k in 0..in_elems {
                acc += w[n * in_elems + k] * x[m * in_elems + k];
            }
            out[m * out_rows + n] = acc;
        }
    }
    out
}

/// Assert two result sets agree within reassociation noise
/// (the GEMM test's bound: relative to the result set's own
/// magnitude scale, plus a small floor).
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

/// Report (and loosely bound) the gap between the MMQ
/// result and the exact f32 result: the activation
/// quantization band.
fn check_quant_band(label: &str, got: &[f32], exact: &[f32]) {
    let scale = exact.iter().fold(0.0f32, |a, w| a.max(w.abs()));
    let mut max_diff = 0.0f32;
    for (g, w) in got.iter().zip(exact.iter()) {
        max_diff = max_diff.max((g - w).abs());
    }
    let rel = max_diff / scale.max(1.0);
    eprintln!("{label}: quantization band max rel diff {rel:.4} (set scale {scale:.3e})");
    assert!(
        rel <= 0.05,
        "{label}: quantization band {rel} exceeds 5% of the result scale"
    );
}

/// Quantize every row of `x` on the host: the reference the
/// GPU pipeline runs are checked against.
fn host_quantize(x: &[f32], width: usize) -> (Vec<u32>, Vec<f32>) {
    let mut qs = Vec::new();
    let mut ds = Vec::new();
    for row in x.chunks_exact(width) {
        let (q, d) = quantize_row_q8_1(row);
        qs.extend_from_slice(&q);
        ds.extend_from_slice(&d);
    }
    (qs, ds)
}

/// The wide Q4_K shapes this device can build: every
/// candidate whose double-buffered staging set the
/// workgroup storage grant covers.
fn wide_shapes_q4k(backend: &WgpuCompute) -> Vec<WideMmqShape> {
    let device = &backend.context().device;
    WIDE_TILE_CANDIDATES
        .iter()
        .copied()
        .map(WideMmqShape::new)
        .filter(|s| K15Pipeline::with_shape(device, *s).is_ok())
        .collect()
}

/// The wide Q6_K shapes this device can build. See
/// [`wide_shapes_q4k`].
fn wide_shapes_q6k(backend: &WgpuCompute) -> Vec<WideMmqShape> {
    let device = &backend.context().device;
    WIDE_TILE_CANDIDATES
        .iter()
        .copied()
        .map(WideMmqShape::new)
        .filter(|s| K16Pipeline::with_shape(device, *s).is_ok())
        .collect()
}

#[test]
fn k15_wide_mmq_matches_quantized_reference_across_batches() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    // Width 512 = two super-blocks per row = four pipeline
    // steps: the parity pipeline runs two full iterations.
    let (inn, out_rows) = (512usize, 100usize);
    let bytes = synthetic_q4k(inn, out_rows);
    let w_ref = gguf::dequant::dequantize(gguf::GgmlType::Q4K, &bytes, inn * out_rows);
    let k1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let weights = k1
        .upload(ctx, "synthetic.q4k.wide", &bytes, inn, out_rows)
        .expect("k1 upload");
    let shapes = wide_shapes_q4k(&backend);
    assert!(
        !shapes.is_empty(),
        "a backend device must build at least the narrowest wide shape"
    );
    for shape in shapes {
        let mmq = K15Pipeline::with_shape(&ctx.device, shape).expect("k15 pipeline builds");
        for batch in [1usize, 5, 16, 37, 64] {
            let x = lcg_input(300 + batch as u64, batch * inn);
            let (qs, ds) = host_quantize(&x, inn);
            let got = mmq
                .run(ctx, &weights, &qs, &ds, batch as u32)
                .expect("k15 run");
            let want = reference_mmq_q4k(&bytes, inn, out_rows, &qs, &ds, batch);
            check_close(&format!("k15 {} batch {batch}", shape.name()), &got, &want);
            if batch == 64 {
                let exact = reference_gemm_f32(&w_ref, inn, out_rows, &x, batch);
                check_quant_band(&format!("k15 {} batch 64", shape.name()), &got, &exact);
            }
        }
    }
}

#[test]
fn k15_wide_mmq_pipelines_across_deep_rows_and_tile_edges() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    let shape = shipped_wide_q4k(&ctx.device.limits()).expect("k15 shipped shape exists");
    let mmq = K15Pipeline::with_shape(&ctx.device, shape).expect("k15 pipeline builds");
    // Width 768 = three super-blocks per row = six steps,
    // and 230 output rows straddle the tile boundary at
    // every candidate width (two tiles at w128, with a
    // partially covered second tile). Batch 33 keeps row
    // guards and the activation staging loops off their
    // round numbers.
    let (inn, out_rows, batch) = (768usize, 230usize, 33usize);
    let bytes = synthetic_q4k(inn, out_rows);
    let k1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let weights = k1
        .upload(ctx, "synthetic.q4k.wide.deep", &bytes, inn, out_rows)
        .expect("k1 upload");
    let x = lcg_input(733, batch * inn);
    let (qs, ds) = host_quantize(&x, inn);
    let got = mmq
        .run(ctx, &weights, &qs, &ds, batch as u32)
        .expect("k15 deep run");
    let want = reference_mmq_q4k(&bytes, inn, out_rows, &qs, &ds, batch);
    check_close(
        &format!("k15 {} deep width {inn} rows {out_rows}", shape.name()),
        &got,
        &want,
    );
    // Width 256 = one super-block per row = two steps: the
    // pipeline's shortest run (prologue plus one pair
    // iteration whose second half must not fire).
    let (inn2, out_rows2, batch2) = (256usize, 65usize, 17usize);
    let bytes2 = synthetic_q4k(inn2, out_rows2);
    let weights2 = k1
        .upload(ctx, "synthetic.q4k.wide.short", &bytes2, inn2, out_rows2)
        .expect("k1 upload");
    let x2 = lcg_input(734, batch2 * inn2);
    let (qs2, ds2) = host_quantize(&x2, inn2);
    let got2 = mmq
        .run(ctx, &weights2, &qs2, &ds2, batch2 as u32)
        .expect("k15 short run");
    let want2 = reference_mmq_q4k(&bytes2, inn2, out_rows2, &qs2, &ds2, batch2);
    check_close(
        &format!("k15 {} short width {inn2}", shape.name()),
        &got2,
        &want2,
    );
}

#[test]
fn k15_wide_mmq_act_forms_match_quantized_reference() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    let (inn, out_rows, batch) = (512usize, 96usize, 29usize);
    let bytes = synthetic_q4k(inn, out_rows);
    let k1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let weights = k1
        .upload(ctx, "synthetic.q4k.wide.act", &bytes, inn, out_rows)
        .expect("k1 upload");
    let shape = shipped_wide_q4k(&ctx.device.limits()).expect("k15 shipped shape exists");
    let mmq = K15Pipeline::with_shape(&ctx.device, shape).expect("k15 pipeline builds");
    let a_stride = 2 * inn;
    let x_a = lcg_input(55, batch * a_stride);
    for act in [ACT_GATED, ACT_SWIGLU] {
        let quant = K12QuantPipeline::new(&ctx.device, act).expect("k12 act pipeline");
        let (x_b_vec, b_stride, b_off) = if act == ACT_SWIGLU {
            (x_a.clone(), a_stride, inn)
        } else {
            (lcg_input(77, batch * inn), inn, 0)
        };
        let a_buf = ctx.upload_f32("k15 test a", &x_a);
        let b_buf = ctx.upload_f32("k15 test b", &x_b_vec);
        let qs_buf = ctx.scratch_buffer(
            "k15 test qs",
            (wgpu_compute::kernels::k12_q8_quant::qs_words(batch, inn) * 4) as u64,
        );
        let ds_buf = ctx.scratch_buffer(
            "k15 test ds",
            (wgpu_compute::kernels::k12_q8_quant::ds_vec4s(batch, inn) * 16) as u64,
        );
        let out_buf = ctx.scratch_buffer("k15 test out", (batch * out_rows * 4) as u64);
        let geom = K1ActGeom {
            a_stride_elems: a_stride,
            b_stride_elems: b_stride,
            b_off_elems: b_off,
        };
        let qcall = quant
            .prepare_act(
                ctx,
                &a_buf,
                &b_buf,
                &qs_buf,
                &ds_buf,
                batch as u32,
                inn,
                &geom,
            )
            .expect("k12 prepare_act");
        let mcall = mmq
            .prepare(ctx, &weights, &qs_buf, &ds_buf, &out_buf, batch as u32)
            .expect("k15 prepare");
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k15 act test"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("k15 act test"),
                timestamp_writes: None,
            });
            qcall.encode_into(&mut pass);
            mcall.encode_into(&mut pass);
        }
        ctx.queue.submit([encoder.finish()]);
        let got = ctx
            .read_f32(&out_buf, batch * out_rows)
            .expect("k15 readback");
        // The reference runs on the GPU quantizer's OWN
        // output, read back: the product kernel is what is
        // under test here, and the quantizer's fidelity is
        // gated separately by the k12 test (see the K13 act
        // test for the rounding-tie reasoning).
        let qs: Vec<u32> = ctx
            .read_f32(
                &qs_buf,
                wgpu_compute::kernels::k12_q8_quant::qs_words(batch, inn),
            )
            .expect("k15 qs readback")
            .iter()
            .map(|f| f.to_bits())
            .collect();
        let ds = ctx
            .read_f32(
                &ds_buf,
                wgpu_compute::kernels::k12_q8_quant::ds_vec4s(batch, inn) * 4,
            )
            .expect("k15 ds readback");
        let want = reference_mmq_q4k(&bytes, inn, out_rows, &qs, &ds, batch);
        check_close(&format!("k15 {} act {act}", shape.name()), &got, &want);
    }
}

#[test]
fn k16_wide_mmq_matches_quantized_reference_across_batches() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    let (inn, out_rows) = (512usize, 100usize);
    let bytes = synthetic_q6k(inn, out_rows);
    let w_ref = gguf::dequant::dequantize(gguf::GgmlType::Q6K, &bytes, inn * out_rows);
    let k2 = K2Pipeline::new(&ctx.device).expect("k2 pipeline");
    let weights = k2
        .upload(ctx, "synthetic.q6k.wide", &bytes, inn, out_rows)
        .expect("k2 upload");
    let shapes = wide_shapes_q6k(&backend);
    assert!(
        !shapes.is_empty(),
        "a backend device must build at least the narrowest wide shape"
    );
    for shape in shapes {
        let mmq = K16Pipeline::with_shape(&ctx.device, shape).expect("k16 pipeline builds");
        for batch in [1usize, 5, 16, 37, 64] {
            let x = lcg_input(400 + batch as u64, batch * inn);
            let (qs, ds) = host_quantize(&x, inn);
            let got = mmq
                .run(ctx, &weights, &qs, &ds, batch as u32)
                .expect("k16 run");
            let want = reference_mmq_q6k(&bytes, inn, out_rows, &qs, &ds, batch);
            check_close(&format!("k16 {} batch {batch}", shape.name()), &got, &want);
            if batch == 64 {
                let exact = reference_gemm_f32(&w_ref, inn, out_rows, &x, batch);
                check_quant_band(&format!("k16 {} batch 64", shape.name()), &got, &exact);
            }
        }
    }
}

#[test]
fn k16_wide_mmq_swiglu_matches_quantized_reference() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    let (inn, out_rows, batch) = (512usize, 96usize, 29usize);
    let bytes = synthetic_q6k(inn, out_rows);
    let k2 = K2Pipeline::new(&ctx.device).expect("k2 pipeline");
    let weights = k2
        .upload(ctx, "synthetic.q6k.wide.act", &bytes, inn, out_rows)
        .expect("k2 upload");
    let shape = shipped_wide_q6k(&ctx.device.limits()).expect("k16 shipped shape exists");
    let mmq = K16Pipeline::with_shape(&ctx.device, shape).expect("k16 pipeline builds");
    let a_stride = 2 * inn;
    let x_a = lcg_input(99, batch * a_stride);
    let quant = K12QuantPipeline::new(&ctx.device, ACT_SWIGLU).expect("k12 swiglu pipeline");
    let a_buf = ctx.upload_f32("k16 test a", &x_a);
    let qs_buf = ctx.scratch_buffer(
        "k16 test qs",
        (wgpu_compute::kernels::k12_q8_quant::qs_words(batch, inn) * 4) as u64,
    );
    let ds_buf = ctx.scratch_buffer(
        "k16 test ds",
        (wgpu_compute::kernels::k12_q8_quant::ds_vec4s(batch, inn) * 16) as u64,
    );
    let out_buf = ctx.scratch_buffer("k16 test out", (batch * out_rows * 4) as u64);
    let geom = K1ActGeom {
        a_stride_elems: a_stride,
        b_stride_elems: a_stride,
        b_off_elems: inn,
    };
    let qcall = quant
        .prepare_act(
            ctx,
            &a_buf,
            &a_buf,
            &qs_buf,
            &ds_buf,
            batch as u32,
            inn,
            &geom,
        )
        .expect("k12 prepare_act");
    let mcall = mmq
        .prepare(ctx, &weights, &qs_buf, &ds_buf, &out_buf, batch as u32)
        .expect("k16 prepare");
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("k16 act test"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k16 act test"),
            timestamp_writes: None,
        });
        qcall.encode_into(&mut pass);
        mcall.encode_into(&mut pass);
    }
    ctx.queue.submit([encoder.finish()]);
    let got = ctx
        .read_f32(&out_buf, batch * out_rows)
        .expect("k16 readback");
    // As in the k15 act test: the reference runs on the GPU
    // quantizer's own output (the k12 test gates the
    // quantizer itself).
    let qs: Vec<u32> = ctx
        .read_f32(
            &qs_buf,
            wgpu_compute::kernels::k12_q8_quant::qs_words(batch, inn),
        )
        .expect("k16 qs readback")
        .iter()
        .map(|f| f.to_bits())
        .collect();
    let ds = ctx
        .read_f32(
            &ds_buf,
            wgpu_compute::kernels::k12_q8_quant::ds_vec4s(batch, inn) * 4,
        )
        .expect("k16 ds readback");
    let want = reference_mmq_q6k(&bytes, inn, out_rows, &qs, &ds, batch);
    check_close(&format!("k16 {} swiglu", shape.name()), &got, &want);
}

#[test]
fn wide_mmq_prepare_refuses_out_of_range_geometry() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    let (inn, out_rows) = (256usize, 64usize);
    let qs = ctx.scratch_buffer(
        "k15 refusal qs",
        (wgpu_compute::kernels::k12_q8_quant::qs_words(65, inn) * 4) as u64,
    );
    let ds = ctx.scratch_buffer(
        "k15 refusal ds",
        (wgpu_compute::kernels::k12_q8_quant::ds_vec4s(65, inn) * 16) as u64,
    );
    let output = ctx.scratch_buffer("k15 refusal out", (65 * out_rows * 4) as u64);
    // Q4_K family.
    let bytes4 = synthetic_q4k(inn, out_rows);
    let k1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let w4 = k1
        .upload(ctx, "synthetic.q4k.wide.refusal", &bytes4, inn, out_rows)
        .expect("k1 upload");
    let shape4 = shipped_wide_q4k(&ctx.device.limits()).expect("k15 shipped shape exists");
    let mmq15 = K15Pipeline::with_shape(&ctx.device, shape4).expect("k15 pipeline");
    assert!(mmq15.prepare(ctx, &w4, &qs, &ds, &output, 0).is_err());
    assert!(mmq15.prepare(ctx, &w4, &qs, &ds, &output, 65).is_err());
    assert!(mmq15.prepare(ctx, &w4, &qs, &ds, &output, 64).is_ok());
    // A non-candidate tile width refuses at construction.
    assert!(K15Pipeline::with_shape(&ctx.device, WideMmqShape::new(100)).is_err());
    // Q6_K family.
    let bytes6 = synthetic_q6k(inn, out_rows);
    let k2 = K2Pipeline::new(&ctx.device).expect("k2 pipeline");
    let w6 = k2
        .upload(ctx, "synthetic.q6k.wide.refusal", &bytes6, inn, out_rows)
        .expect("k2 upload");
    let shape6 = shipped_wide_q6k(&ctx.device.limits()).expect("k16 shipped shape exists");
    let mmq16 = K16Pipeline::with_shape(&ctx.device, shape6).expect("k16 pipeline");
    assert!(mmq16.prepare(ctx, &w6, &qs, &ds, &output, 0).is_err());
    assert!(mmq16.prepare(ctx, &w6, &qs, &ds, &output, 65).is_err());
    assert!(mmq16.prepare(ctx, &w6, &qs, &ds, &output, 64).is_ok());
}
