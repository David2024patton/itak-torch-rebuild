//! K18 (the small-batch Q4_K / Q5_K / Q6_K products)
//! against the same synthetic tensors and host
//! references as the K13/K14/K15/K16 gates, at the row
//! band K18 exists for: 2, 3, 7, 15, 16 (plus 1 as the
//! band floor sanity).
//!
//! Two gates per family, mirroring the sibling files:
//!
//! - The product must match a host reference running
//!   the SAME quantized math (exact integer dots plus
//!   the per-sub-block scale fold) within float
//!   reassociation noise: the correctness gate.
//! - Against the family's GEMV product on the same f32
//!   inputs (the form the session runs today at these
//!   rows) the results must land inside the
//!   activation-quantization band (the sibling files'
//!   loose 5% bound), and the per-row argmax must agree
//!   wherever the GEMV's own top-2 margin clears the
//!   row's observed quantization noise: the parity the
//!   task names, at the established bar.
//!
//! Every tile candidate the device can build is gated,
//! not just the shipped shape: the candidates differ
//! only in a tile-width literal, and a staging index
//! that is wrong at one width is exactly the defect
//! class this file exists to catch.
//!
//! These tests need a discrete Vulkan GPU under the
//! default selection policy. On a machine without one
//! they print the refusal and pass; shader validity on
//! any adapter is gated separately in
//! `shader_compile.rs`, and the real-file proof runs on
//! the server through the bench example and the spec
//! verdict runs (bench/small-batch-2026-11-04.md).

use wgpu_compute::kernels::k12_q8_quant::quantize_row_q8_1;
use wgpu_compute::kernels::k18_small_batch::{
    shipped_shape_q4k, shipped_shape_q5k, shipped_shape_q6k, K18Q4KPipeline, K18Q5KPipeline,
    K18Q6KPipeline, SmallShape,
};
use wgpu_compute::kernels::k1_q4k_gemv::{K1Pipeline, K1Weights};
use wgpu_compute::kernels::k2_q6k_gemv::{K2Pipeline, K2Weights};
use wgpu_compute::kernels::k3_q5k_gemv::{K3Pipeline, K3Weights};
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

/// A synthetic Q4_K tensor's raw bytes (the sibling
/// files' fixture: rotating (d, dmin) headers over LCG
/// payloads).
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

/// A synthetic Q5_K tensor's raw bytes (176-byte
/// blocks, the same rotating headers; scales, qh, and
/// ql planes are LCG payload).
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

/// A synthetic Q6_K tensor's raw bytes (210-byte
/// blocks, the f16 super-scale at offset 208).
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

/// Unpack a quantized row's words into per-element
/// quants.
fn unpack_qs(words: &[u32]) -> Vec<i32> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for w in words {
        for byte in w.to_le_bytes() {
            out.push(i32::from(byte as i8));
        }
    }
    out
}

/// ggml's `get_scale_min_k4`, mirrored from the gguf
/// crate.
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

/// The host MMQ reference for Q4_K: exact integer dots
/// of the weight nibbles against the host-quantized
/// activations, folded per sub-block in super-block
/// order with the same f32 expression the kernel uses
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

/// The host MMQ reference for Q5_K: exact integer dots
/// of the assembled 5-bit quants (ql nibble plus the
/// qh plane's bit `j` for sub-block `j`) against the
/// host-quantized activations, folded per sub-block
/// with Q4_K's expression (`vec_dot_q5_K_q8_1`'s
/// shape).
fn reference_mmq_q5k(
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
                let block = &raw[(n * blocks_per_row + b) * 176..][..176];
                let d = gguf::dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let dmin = gguf::dequant::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
                let scales = &block[4..16];
                let qh = &block[16..48];
                let ql = &block[48..176];
                for j in 0..8usize {
                    let (sc, mn) = scale_min_k4(j, scales);
                    let g = j / 2;
                    let hi = j % 2 == 1;
                    let mut sumi: i64 = 0;
                    for l in 0..32 {
                        let byte = ql[g * 32 + l];
                        let nib = if hi { byte >> 4 } else { byte & 0xF };
                        let q = i64::from(nib) + 16 * i64::from((qh[l] >> j) & 1);
                        sumi += q * i64::from(qx[b * 256 + j * 32 + l]);
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

/// The host MMQ reference for Q6_K: exact integer dots
/// of the assembled 6-bit quants against the
/// host-quantized activations, folded per 16-element
/// half (`vec_dot_q6_K_q8_1`'s shape).
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
                    // The sub-block's 32 quants, in
                    // element order (dequant_q6_k's
                    // interleave).
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

/// Assert two result sets agree within reassociation
/// noise (the sibling files' bound: relative to the
/// result set's own magnitude scale, plus a small
/// floor).
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

/// Report (and loosely bound) the gap between the
/// quantized product and an f32 reference product: the
/// activation-quantization band (the sibling files'
/// 5% bound).
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

/// Assert per-row argmax parity between the quantized
/// product and the GEMV product, with the honest tie
/// rule: a flip is excused only where the GEMV row's
/// own top-2 margin is within twice the row's observed
/// max abs diff (the two candidates can genuinely swap
/// under that much quantization noise). Raw agreement
/// is printed as evidence.
fn check_argmax_band(label: &str, got: &[f32], gemv: &[f32], out_rows: usize) {
    let rows = got.len() / out_rows;
    let mut agree = 0usize;
    for m in 0..rows {
        let g = &got[m * out_rows..(m + 1) * out_rows];
        let v = &gemv[m * out_rows..(m + 1) * out_rows];
        let argmax = |row: &[f32]| {
            row.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                    if x > bv {
                        (i, x)
                    } else {
                        (bi, bv)
                    }
                })
                .0
        };
        let (ga, va) = (argmax(g), argmax(v));
        if ga == va {
            agree += 1;
            continue;
        }
        let row_noise = g
            .iter()
            .zip(v.iter())
            .fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
        let mut top2 = v.to_vec();
        top2.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let margin = top2[0] - top2[1];
        assert!(
            margin <= 2.0 * row_noise,
            "{label}: row {m} argmax flip is not a tie: gemv top-2 margin {margin} \
             against row noise {row_noise} (got argmax {ga}, gemv argmax {va})"
        );
    }
    eprintln!("{label}: argmax agreement {agree}/{rows} rows raw, the rest inside the tie rule");
}

/// Quantize every row of `x` on the host: the
/// reference the GPU pipeline runs are checked
/// against.
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

/// Run the K1 GEMV over `x` and read the product back:
/// the form the session runs at these rows today.
fn run_gemv_q4k(
    ctx: &GpuContext,
    k1: &K1Pipeline,
    weights: &K1Weights,
    x: &[f32],
    batch: usize,
    out_rows: usize,
) -> Vec<f32> {
    let in_buf = ctx.upload_f32("k18 test gemv input", x);
    let out_buf = ctx.scratch_buffer("k18 test gemv out", (batch * out_rows * 4) as u64);
    let call = k1
        .prepare(ctx, weights, &in_buf, &out_buf, batch as u32)
        .expect("k1 prepare");
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("k18 gemv run"),
        });
    call.encode(&mut encoder);
    ctx.queue.submit([encoder.finish()]);
    ctx.read_f32(&out_buf, batch * out_rows)
        .expect("k1 readback")
}

/// Run the K2 GEMV over `x` and read the product back.
fn run_gemv_q6k(
    ctx: &GpuContext,
    k2: &K2Pipeline,
    weights: &K2Weights,
    x: &[f32],
    batch: usize,
    out_rows: usize,
) -> Vec<f32> {
    let in_buf = ctx.upload_f32("k18 test gemv input", x);
    let out_buf = ctx.scratch_buffer("k18 test gemv out", (batch * out_rows * 4) as u64);
    let call = k2
        .prepare(ctx, weights, &in_buf, &out_buf, batch as u32)
        .expect("k2 prepare");
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("k18 gemv run"),
        });
    call.encode(&mut encoder);
    ctx.queue.submit([encoder.finish()]);
    ctx.read_f32(&out_buf, batch * out_rows)
        .expect("k2 readback")
}

/// Run the K3 GEMV over `x` and read the product back.
fn run_gemv_q5k(
    ctx: &GpuContext,
    k3: &K3Pipeline,
    weights: &K3Weights,
    x: &[f32],
    batch: usize,
    out_rows: usize,
) -> Vec<f32> {
    let in_buf = ctx.upload_f32("k18 test gemv input", x);
    let out_buf = ctx.scratch_buffer("k18 test gemv out", (batch * out_rows * 4) as u64);
    let call = k3
        .prepare(ctx, weights, &in_buf, &out_buf, batch as u32)
        .expect("k3 prepare");
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("k18 gemv run"),
        });
    call.encode(&mut encoder);
    ctx.queue.submit([encoder.finish()]);
    ctx.read_f32(&out_buf, batch * out_rows)
        .expect("k3 readback")
}

/// The K18 tile candidates, widest first.
fn tile_candidates() -> [SmallShape; 2] {
    [SmallShape::new(64), SmallShape::new(32)]
}

/// The row counts the gates sweep: K18's named band.
const BAND: [usize; 6] = [1, 2, 3, 7, 15, 16];

#[test]
fn k18_q4k_matches_gemv_and_quantized_reference_at_small_batches() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    // Width 512 = two super-blocks per row; 100 output
    // rows straddle both tiles' boundaries (64 + 36,
    // 3 x 32 + 4).
    let (inn, out_rows) = (512usize, 100usize);
    let bytes = synthetic_q4k(inn, out_rows);
    let k1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let weights = k1
        .upload(ctx, "synthetic.q4k.small", &bytes, inn, out_rows)
        .expect("k1 upload");
    let mut built = 0usize;
    for shape in tile_candidates() {
        let Ok(small) = K18Q4KPipeline::with_shape(&ctx.device, shape) else {
            continue;
        };
        built += 1;
        for batch in BAND {
            let x = lcg_input(400 + batch as u64, batch * inn);
            let (qs, ds) = host_quantize(&x, inn);
            let got = small
                .run(ctx, &weights, &qs, &ds, batch as u32)
                .expect("k18 q4k run");
            let want = reference_mmq_q4k(&bytes, inn, out_rows, &qs, &ds, batch);
            check_close(
                &format!("k18 q4k {} batch {batch}", shape.name()),
                &got,
                &want,
            );
            let gemv = run_gemv_q4k(ctx, &k1, &weights, &x, batch, out_rows);
            check_quant_band(
                &format!("k18 q4k {} batch {batch} vs gemv", shape.name()),
                &got,
                &gemv,
            );
            check_argmax_band(
                &format!("k18 q4k {} batch {batch} vs gemv", shape.name()),
                &got,
                &gemv,
                out_rows,
            );
        }
    }
    assert!(
        built > 0,
        "a backend device must build at least one k18 q4k shape"
    );
    assert_eq!(
        shipped_shape_q4k(&ctx.device.limits()),
        Some(tile_candidates()[if built == 2 { 0 } else { 1 }]),
        "the shipped shape is the widest buildable candidate"
    );
    // The band's edges refuse by name.
    let small = K18Q4KPipeline::for_device(&ctx.device)
        .expect("k18 q4k for_device")
        .expect("k18 q4k exists on a backend device");
    let qs_buf = ctx.scratch_buffer("k18 test qs", 4096);
    let ds_buf = ctx.scratch_buffer("k18 test ds", 4096);
    let out_buf = ctx.scratch_buffer("k18 test out", 4096);
    let err = small
        .prepare(ctx, &weights, &qs_buf, &ds_buf, &out_buf, 0)
        .unwrap_err();
    assert!(
        err.to_string().contains("outside 1..=16"),
        "batch 0 refusal names the band: {err}"
    );
    let err = small
        .prepare(ctx, &weights, &qs_buf, &ds_buf, &out_buf, 17)
        .unwrap_err();
    assert!(
        err.to_string().contains("outside 1..=16"),
        "batch 17 refusal names the band: {err}"
    );
}

#[test]
fn k18_q6k_matches_gemv_and_quantized_reference_at_small_batches() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    let (inn, out_rows) = (512usize, 100usize);
    let bytes = synthetic_q6k(inn, out_rows);
    let k2 = K2Pipeline::new(&ctx.device).expect("k2 pipeline");
    let weights = k2
        .upload(ctx, "synthetic.q6k.small", &bytes, inn, out_rows)
        .expect("k2 upload");
    let mut built = 0usize;
    for shape in tile_candidates() {
        let Ok(small) = K18Q6KPipeline::with_shape(&ctx.device, shape) else {
            continue;
        };
        built += 1;
        for batch in BAND {
            let x = lcg_input(500 + batch as u64, batch * inn);
            let (qs, ds) = host_quantize(&x, inn);
            let got = small
                .run(ctx, &weights, &qs, &ds, batch as u32)
                .expect("k18 q6k run");
            let want = reference_mmq_q6k(&bytes, inn, out_rows, &qs, &ds, batch);
            check_close(
                &format!("k18 q6k {} batch {batch}", shape.name()),
                &got,
                &want,
            );
            let gemv = run_gemv_q6k(ctx, &k2, &weights, &x, batch, out_rows);
            check_quant_band(
                &format!("k18 q6k {} batch {batch} vs gemv", shape.name()),
                &got,
                &gemv,
            );
            check_argmax_band(
                &format!("k18 q6k {} batch {batch} vs gemv", shape.name()),
                &got,
                &gemv,
                out_rows,
            );
        }
    }
    assert!(
        built > 0,
        "a backend device must build at least one k18 q6k shape"
    );
    assert_eq!(
        shipped_shape_q6k(&ctx.device.limits()),
        Some(tile_candidates()[if built == 2 { 0 } else { 1 }]),
        "the shipped shape is the widest buildable candidate"
    );
}

#[test]
fn k18_q5k_matches_gemv_and_quantized_reference_at_small_batches() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    let (inn, out_rows) = (512usize, 100usize);
    let bytes = synthetic_q5k(inn, out_rows);
    let k3 = K3Pipeline::new(&ctx.device).expect("k3 pipeline");
    let weights = k3
        .upload(ctx, "synthetic.q5k.small", &bytes, inn, out_rows)
        .expect("k3 upload");
    let mut built = 0usize;
    for shape in tile_candidates() {
        let Ok(small) = K18Q5KPipeline::with_shape(&ctx.device, shape) else {
            continue;
        };
        built += 1;
        for batch in BAND {
            let x = lcg_input(600 + batch as u64, batch * inn);
            let (qs, ds) = host_quantize(&x, inn);
            let got = small
                .run(ctx, &weights, &qs, &ds, batch as u32)
                .expect("k18 q5k run");
            let want = reference_mmq_q5k(&bytes, inn, out_rows, &qs, &ds, batch);
            check_close(
                &format!("k18 q5k {} batch {batch}", shape.name()),
                &got,
                &want,
            );
            let gemv = run_gemv_q5k(ctx, &k3, &weights, &x, batch, out_rows);
            check_quant_band(
                &format!("k18 q5k {} batch {batch} vs gemv", shape.name()),
                &got,
                &gemv,
            );
            check_argmax_band(
                &format!("k18 q5k {} batch {batch} vs gemv", shape.name()),
                &got,
                &gemv,
                out_rows,
            );
        }
    }
    assert!(
        built > 0,
        "a backend device must build at least one k18 q5k shape"
    );
    assert_eq!(
        shipped_shape_q5k(&ctx.device.limits()),
        Some(tile_candidates()[if built == 2 { 0 } else { 1 }]),
        "the shipped shape is the widest buildable candidate"
    );
}

#[test]
fn k18_deep_width_and_tile_edge_geometry() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    // Width 768 = three super-blocks per row and 230
    // output rows (tile edges at every candidate), at
    // the band's working depths: the staging loops and
    // the column loop run off their round numbers.
    let (inn, out_rows) = (768usize, 230usize);
    let bytes4 = synthetic_q4k(inn, out_rows);
    let k1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let w4 = k1
        .upload(ctx, "synthetic.q4k.small.deep", &bytes4, inn, out_rows)
        .expect("k1 upload");
    let bytes5 = synthetic_q5k(inn, out_rows);
    let k3 = K3Pipeline::new(&ctx.device).expect("k3 pipeline");
    let w5 = k3
        .upload(ctx, "synthetic.q5k.small.deep", &bytes5, inn, out_rows)
        .expect("k3 upload");
    let bytes6 = synthetic_q6k(inn, out_rows);
    let k2 = K2Pipeline::new(&ctx.device).expect("k2 pipeline");
    let w6 = k2
        .upload(ctx, "synthetic.q6k.small.deep", &bytes6, inn, out_rows)
        .expect("k2 upload");
    for shape in tile_candidates() {
        for batch in [7usize, 16] {
            let x = lcg_input(900 + batch as u64, batch * inn);
            let (qs, ds) = host_quantize(&x, inn);
            if let Ok(small) = K18Q4KPipeline::with_shape(&ctx.device, shape) {
                let got = small
                    .run(ctx, &w4, &qs, &ds, batch as u32)
                    .expect("k18 q4k deep run");
                let want = reference_mmq_q4k(&bytes4, inn, out_rows, &qs, &ds, batch);
                check_close(
                    &format!("k18 q4k {} deep batch {batch}", shape.name()),
                    &got,
                    &want,
                );
            }
            if let Ok(small) = K18Q5KPipeline::with_shape(&ctx.device, shape) {
                let got = small
                    .run(ctx, &w5, &qs, &ds, batch as u32)
                    .expect("k18 q5k deep run");
                let want = reference_mmq_q5k(&bytes5, inn, out_rows, &qs, &ds, batch);
                check_close(
                    &format!("k18 q5k {} deep batch {batch}", shape.name()),
                    &got,
                    &want,
                );
            }
            if let Ok(small) = K18Q6KPipeline::with_shape(&ctx.device, shape) {
                let got = small
                    .run(ctx, &w6, &qs, &ds, batch as u32)
                    .expect("k18 q6k deep run");
                let want = reference_mmq_q6k(&bytes6, inn, out_rows, &qs, &ds, batch);
                check_close(
                    &format!("k18 q6k {} deep batch {batch}", shape.name()),
                    &got,
                    &want,
                );
            }
        }
    }
}
