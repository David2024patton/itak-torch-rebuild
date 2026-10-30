//! K12/K13/K14 (the Q8_1 quantizer and the Q4_K / Q6_K MMQ
//! product kernels) against synthetic tensors with known
//! contents. The gates are two-layered, on purpose:
//!
//! - The product kernels must match a host reference running
//!   the SAME quantized math (exact integer dots plus the
//!   per-sub-block scale fold, mirroring ggml's
//!   `vec_dot_q4_K_q8_1` / `vec_dot_q6_K_q8_1`) within float
//!   reassociation noise: this is the correctness gate, and
//!   it is as tight as the GEMM gates. The plain forms build
//!   the reference from K12's host quantizer; the activation
//!   forms build it from the GPU quantizer's own read-back
//!   output, because the quantizer's rounding ties are its
//!   own contract (gated by the k12 test: quants within one
//!   step, scales within ulps) and a single tie flip moves a
//!   product by a full weight quant, which is not a product
//!   defect.
//! - Against the f32 reference (dequantized weights, exact
//!   activations) the same results must land inside the
//!   activation-quantization band, which is a different and
//!   larger error: Q8_1 keeps 8 bits of each activation, so
//!   per-element relative error ~2^-8 is the design, not a
//!   defect. The test prints the measured band and asserts a
//!   loose bound on it, so a future change that widens the
//!   band shows up as a failure with the number attached.
//!
//! These tests need a discrete Vulkan GPU under the default
//! selection policy. On a machine without one they print the
//! refusal and pass; the real-file proof runs on the server
//! through the resident session (the MMQ's consumer).

use wgpu_compute::kernels::k12_q8_quant::{quantize_row_q8_1, K12QuantPipeline};
use wgpu_compute::kernels::k13_q4k_mmq::{K13Pipeline, MmqShape};
use wgpu_compute::kernels::k14_q6k_mmq::K14Pipeline;
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

/// Apply an activation transform to one row, host-side, in
/// K7's exact expressions (the reference the act quantizers
/// are gated against). `b` is addressed at the row's own
/// segment of the paired buffer the caller built.
fn apply_act(mode: u32, a: &[f32], b: &[f32]) -> Vec<f32> {
    match mode {
        ACT_GATED => a
            .iter()
            .zip(b.iter())
            .map(|(x, g)| x * sigmoid(*g))
            .collect(),
        ACT_SWIGLU => a.iter().zip(b.iter()).map(|(x, u)| silu(*x) * u).collect(),
        _ => a.to_vec(),
    }
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

/// The MMQ shapes this device can build (the shipped one
/// plus the wide tile when its storage grants it).
fn shapes(backend: &WgpuCompute) -> Vec<MmqShape> {
    let mut out = Vec::new();
    if let Some(s) =
        wgpu_compute::kernels::k13_q4k_mmq::shipped_shape(&backend.context().device.limits())
    {
        out.push(s);
    }
    let wide = MmqShape::new(64);
    if !out.contains(&wide) {
        // Include the wide tile when the device grants its
        // staging; with_shape refuses by name otherwise.
        if K13Pipeline::with_shape(&backend.context().device, wide).is_ok() {
            out.push(wide);
        }
    }
    out
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

#[test]
fn k12_quantize_matches_host_reference() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    let (width, batch) = (512usize, 5usize);
    let x = lcg_input(42, batch * width);
    for act in [ACT_NONE, ACT_GATED, ACT_SWIGLU] {
        let pipe = K12QuantPipeline::new(&ctx.device, act).expect("k12 pipeline");
        // Stream B at its own stride for gated, the paired
        // segment for swiglu (the session's two geometries).
        let (gpu_qs, gpu_ds, want_x) = if act == ACT_NONE {
            let (qs, ds) = pipe.run(ctx, &x, width).expect("k12 run");
            (qs, ds, x.clone())
        } else {
            let b_stride = if act == ACT_GATED { width } else { 2 * width };
            let b_off = if act == ACT_GATED { 0 } else { width };
            let x_b = lcg_input(43, batch * b_stride);
            let a_buf = ctx.upload_f32("k12 test a", &x);
            let b_buf = ctx.upload_f32("k12 test b", &x_b);
            let qs_buf = ctx.scratch_buffer(
                "k12 test qs",
                (wgpu_compute::kernels::k12_q8_quant::qs_words(batch, width) * 4) as u64,
            );
            let ds_buf = ctx.scratch_buffer(
                "k12 test ds",
                (wgpu_compute::kernels::k12_q8_quant::ds_vec4s(batch, width) * 16) as u64,
            );
            let geom = K1ActGeom {
                a_stride_elems: width,
                b_stride_elems: b_stride,
                b_off_elems: b_off,
            };
            let call = pipe
                .prepare_act(
                    ctx,
                    &a_buf,
                    &b_buf,
                    &qs_buf,
                    &ds_buf,
                    batch as u32,
                    width,
                    &geom,
                )
                .expect("k12 prepare_act");
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("k12 act test"),
                });
            call.encode(&mut encoder);
            ctx.queue.submit([encoder.finish()]);
            let qs = ctx
                .read_f32(
                    &qs_buf,
                    wgpu_compute::kernels::k12_q8_quant::qs_words(batch, width),
                )
                .expect("k12 qs readback")
                .iter()
                .map(|f| f.to_bits())
                .collect::<Vec<u32>>();
            let ds = ctx
                .read_f32(
                    &ds_buf,
                    wgpu_compute::kernels::k12_q8_quant::ds_vec4s(batch, width) * 4,
                )
                .expect("k12 ds readback");
            // The host transform reads B at the same
            // stride/offset per row.
            let mut transformed = Vec::with_capacity(batch * width);
            for m in 0..batch {
                let a = &x[m * width..(m + 1) * width];
                let b = &x_b[m * b_stride + b_off..m * b_stride + b_off + width];
                transformed.extend_from_slice(&apply_act(act, a, b));
            }
            (qs, ds, transformed)
        };
        let (want_qs, want_ds) = host_quantize(&want_x, width);
        // Quants: exact except for possible rounding-tie
        // flips (the shader `round` tie rule vs the host's);
        // a flip moves one quant by one step.
        let got_q = unpack_qs(&gpu_qs);
        let want_q = unpack_qs(&want_qs);
        let mut exact = 0usize;
        for (i, (g, w)) in got_q.iter().zip(want_q.iter()).enumerate() {
            let diff = (g - w).abs();
            assert!(
                diff <= 1,
                "k12 act {act}: quant {i}: got {g}, want {w} (diff {diff})"
            );
            if diff == 0 {
                exact += 1;
            }
        }
        eprintln!("k12 act {act}: {exact}/{} quants exact", got_q.len());
        // Scales: the amax reduction is exact f32 ops, but
        // the final division is not correctly rounded on
        // every backend (RADV's reciprocal refinement
        // differs from IEEE division by 1 ulp, measured on
        // the server card), so d is gated at a few ulps,
        // not bitwise. The sums move by at most a couple of
        // quanta under tie flips.
        for b in 0..gpu_ds.len() / 4 {
            let d = want_ds[b * 4];
            let d_diff = (gpu_ds[b * 4] - d).abs();
            assert!(
                d_diff <= d * 1e-6,
                "k12 act {act}: block {b} scale: got {}, want {d}",
                gpu_ds[b * 4]
            );
            for c in 1..4 {
                let diff = (gpu_ds[b * 4 + c] - want_ds[b * 4 + c]).abs();
                assert!(
                    diff <= 2.0 * d + 1e-12,
                    "k12 act {act}: block {b} sum {c}: got {}, want {} (d {d})",
                    gpu_ds[b * 4 + c],
                    want_ds[b * 4 + c]
                );
            }
        }
    }
}

#[test]
fn k13_mmq_matches_quantized_reference_across_batches() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    let (inn, out_rows) = (512usize, 100usize);
    let bytes = synthetic_q4k(inn, out_rows);
    let w_ref = gguf::dequant::dequantize(gguf::GgmlType::Q4K, &bytes, inn * out_rows);
    let k1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let weights = k1
        .upload(ctx, "synthetic.q4k", &bytes, inn, out_rows)
        .expect("k1 upload");
    for shape in shapes(&backend) {
        let mmq = K13Pipeline::with_shape(&ctx.device, shape).expect("k13 pipeline builds");
        for batch in [1usize, 5, 16, 37, 64] {
            let x = lcg_input(300 + batch as u64, batch * inn);
            let (qs, ds) = host_quantize(&x, inn);
            let got = mmq
                .run(ctx, &weights, &qs, &ds, batch as u32)
                .expect("k13 run");
            let want = reference_mmq_q4k(&bytes, inn, out_rows, &qs, &ds, batch);
            check_close(&format!("k13 {} batch {batch}", shape.name()), &got, &want);
            if batch == 64 {
                let exact = reference_gemm_f32(&w_ref, inn, out_rows, &x, batch);
                check_quant_band(&format!("k13 {} batch 64", shape.name()), &got, &exact);
            }
        }
    }
}

#[test]
fn k13_mmq_act_forms_match_quantized_reference() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    let (inn, out_rows, batch) = (512usize, 96usize, 29usize);
    let bytes = synthetic_q4k(inn, out_rows);
    let k1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let weights = k1
        .upload(ctx, "synthetic.q4k.act", &bytes, inn, out_rows)
        .expect("k1 upload");
    let a_stride = 2 * inn;
    let x_a = lcg_input(55, batch * a_stride);
    for shape in shapes(&backend) {
        let mmq = K13Pipeline::with_shape(&ctx.device, shape).expect("k13 pipeline builds");
        for act in [ACT_GATED, ACT_SWIGLU] {
            let quant = K12QuantPipeline::new(&ctx.device, act).expect("k12 act pipeline");
            let (x_b_vec, b_stride, b_off) = if act == ACT_SWIGLU {
                (x_a.clone(), a_stride, inn)
            } else {
                (lcg_input(77, batch * inn), inn, 0)
            };
            let a_buf = ctx.upload_f32("k13 test a", &x_a);
            let b_buf = ctx.upload_f32("k13 test b", &x_b_vec);
            let qs_buf = ctx.scratch_buffer(
                "k13 test qs",
                (wgpu_compute::kernels::k12_q8_quant::qs_words(batch, inn) * 4) as u64,
            );
            let ds_buf = ctx.scratch_buffer(
                "k13 test ds",
                (wgpu_compute::kernels::k12_q8_quant::ds_vec4s(batch, inn) * 16) as u64,
            );
            let out_buf = ctx.scratch_buffer("k13 test out", (batch * out_rows * 4) as u64);
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
                .expect("k13 prepare");
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("k13 act test"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("k13 act test"),
                    timestamp_writes: None,
                });
                qcall.encode_into(&mut pass);
                mcall.encode_into(&mut pass);
            }
            ctx.queue.submit([encoder.finish()]);
            let got = ctx
                .read_f32(&out_buf, batch * out_rows)
                .expect("k13 readback");
            // The reference runs on the GPU quantizer's OWN
            // output, read back: the product kernel is what
            // is under test here, and the quantizer's
            // fidelity (including its rounding-tie behavior)
            // is gated separately by the k12 test. Building
            // the reference from the host quantizer instead
            // would fail this test on a single tie flip,
            // whose product effect is a full weight quant.
            let qs: Vec<u32> = ctx
                .read_f32(
                    &qs_buf,
                    wgpu_compute::kernels::k12_q8_quant::qs_words(batch, inn),
                )
                .expect("k13 qs readback")
                .iter()
                .map(|f| f.to_bits())
                .collect();
            let ds = ctx
                .read_f32(
                    &ds_buf,
                    wgpu_compute::kernels::k12_q8_quant::ds_vec4s(batch, inn) * 4,
                )
                .expect("k13 ds readback");
            let want = reference_mmq_q4k(&bytes, inn, out_rows, &qs, &ds, batch);
            check_close(&format!("k13 {} act {act}", shape.name()), &got, &want);
        }
    }
}

#[test]
fn k14_mmq_matches_quantized_reference_across_batches() {
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
        let mmq = K14Pipeline::with_shape(&ctx.device, shape).expect("k14 pipeline builds");
        for batch in [1usize, 5, 16, 37, 64] {
            let x = lcg_input(400 + batch as u64, batch * inn);
            let (qs, ds) = host_quantize(&x, inn);
            let got = mmq
                .run(ctx, &weights, &qs, &ds, batch as u32)
                .expect("k14 run");
            let want = reference_mmq_q6k(&bytes, inn, out_rows, &qs, &ds, batch);
            check_close(&format!("k14 {} batch {batch}", shape.name()), &got, &want);
            if batch == 64 {
                let exact = reference_gemm_f32(&w_ref, inn, out_rows, &x, batch);
                check_quant_band(&format!("k14 {} batch 64", shape.name()), &got, &exact);
            }
        }
    }
}

#[test]
fn k14_mmq_swiglu_matches_quantized_reference() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    let (inn, out_rows, batch) = (512usize, 96usize, 29usize);
    let bytes = synthetic_q6k(inn, out_rows);
    let k2 = K2Pipeline::new(&ctx.device).expect("k2 pipeline");
    let weights = k2
        .upload(ctx, "synthetic.q6k.act", &bytes, inn, out_rows)
        .expect("k2 upload");
    let a_stride = 2 * inn;
    let x_a = lcg_input(99, batch * a_stride);
    for shape in shapes(&backend) {
        let mmq = K14Pipeline::with_shape(&ctx.device, shape).expect("k14 pipeline builds");
        let quant = K12QuantPipeline::new(&ctx.device, ACT_SWIGLU).expect("k12 swiglu pipeline");
        let a_buf = ctx.upload_f32("k14 test a", &x_a);
        let qs_buf = ctx.scratch_buffer(
            "k14 test qs",
            (wgpu_compute::kernels::k12_q8_quant::qs_words(batch, inn) * 4) as u64,
        );
        let ds_buf = ctx.scratch_buffer(
            "k14 test ds",
            (wgpu_compute::kernels::k12_q8_quant::ds_vec4s(batch, inn) * 16) as u64,
        );
        let out_buf = ctx.scratch_buffer("k14 test out", (batch * out_rows * 4) as u64);
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
            .expect("k14 prepare");
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k14 act test"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("k14 act test"),
                timestamp_writes: None,
            });
            qcall.encode_into(&mut pass);
            mcall.encode_into(&mut pass);
        }
        ctx.queue.submit([encoder.finish()]);
        let got = ctx
            .read_f32(&out_buf, batch * out_rows)
            .expect("k14 readback");
        // As in the k13 act test: the reference runs on the
        // GPU quantizer's own output (the k12 test gates the
        // quantizer itself), so a host/GPU rounding-tie flip
        // in one activation cannot fail the product gate.
        let qs: Vec<u32> = ctx
            .read_f32(
                &qs_buf,
                wgpu_compute::kernels::k12_q8_quant::qs_words(batch, inn),
            )
            .expect("k14 qs readback")
            .iter()
            .map(|f| f.to_bits())
            .collect();
        let ds = ctx
            .read_f32(
                &ds_buf,
                wgpu_compute::kernels::k12_q8_quant::ds_vec4s(batch, inn) * 4,
            )
            .expect("k14 ds readback");
        let want = reference_mmq_q6k(&bytes, inn, out_rows, &qs, &ds, batch);
        check_close(&format!("k14 {} swiglu", shape.name()), &got, &want);
    }
}

#[test]
fn mmq_prepare_refuses_out_of_range_geometry() {
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
    let mmq = K13Pipeline::with_shape(&ctx.device, MmqShape::new(32)).expect("k13 pipeline");
    let qs = ctx.scratch_buffer(
        "k13 refusal qs",
        (wgpu_compute::kernels::k12_q8_quant::qs_words(65, inn) * 4) as u64,
    );
    let ds = ctx.scratch_buffer(
        "k13 refusal ds",
        (wgpu_compute::kernels::k12_q8_quant::ds_vec4s(65, inn) * 16) as u64,
    );
    let output = ctx.scratch_buffer("k13 refusal out", (65 * out_rows * 4) as u64);
    assert!(mmq.prepare(ctx, &weights, &qs, &ds, &output, 0).is_err());
    assert!(mmq.prepare(ctx, &weights, &qs, &ds, &output, 65).is_err());
    assert!(mmq.prepare(ctx, &weights, &qs, &ds, &output, 64).is_ok());
    // The quantizer refuses a width that is not a whole
    // number of 256-element quanta, by name.
    let quant = K12QuantPipeline::new(&ctx.device, ACT_NONE).expect("k12 pipeline");
    let input = ctx.scratch_buffer("k12 refusal in", (384 * 4) as u64);
    assert!(quant.prepare(ctx, &input, &qs, &ds, 1, 384).is_err());
    assert!(quant.prepare(ctx, &input, &qs, &ds, 1, 256).is_ok());
}
