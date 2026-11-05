//! K22 (the Q4_K MMQ product in the register
//! organization) against the host reference running the
//! same quantized math, and against K13 (the shipped Q4_K
//! MMQ) on the same inputs, at the prefill band the kernel
//! exists for: batches 16, 32, 64, plus a 62-row chunk
//! tail (the 1,022-token prefill's last chunk shape).
//!
//! K22 runs K13's exact math (the same integer dots and
//! per-sub-block scale fold, in the same order), so both
//! gates use the crate's MMQ reassociation bound
//! (`check_close`, the GEMM gates' bound): the reference
//! gate is the correctness claim, and the K13 gate is the
//! parity claim the routing relies on (a plan can swap
//! the forms without moving a byte or changing the math
//! beyond float reassociation).
//!
//! These tests need a discrete Vulkan GPU under the
//! default selection policy. On a machine without one
//! they print the refusal and pass; shader validity on
//! any adapter is gated separately in
//! `shader_compile.rs`, and the real-file proof runs on
//! the server through the prefill A/B
//! (bench/k22-mmq-2026-11-05.md).

use wgpu_compute::kernels::k12_q8_quant::quantize_row_q8_1;
use wgpu_compute::kernels::k13_q4k_mmq::{K13Pipeline, MmqShape};
use wgpu_compute::kernels::k1_q4k_gemv::K1Pipeline;
use wgpu_compute::kernels::k22_q4k_mmq::K22Pipeline;
use wgpu_compute::{WgpuCompute, WgpuError};

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

/// Deterministic pseudo-random f32s in [-0.5, 0.5) via a
/// 64-bit LCG, matching the other kernel tests.
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

/// A synthetic Q4_K tensor's raw bytes (the MMQ test's
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

/// The host MMQ reference for Q4_K: exact integer dots of
/// the weight nibbles against the host-quantized
/// activations, folded per sub-block in super-block order
/// with the same f32 expression the kernels use
/// (`vec_dot_q4_K_q8_1`'s shape). Copied from the K13
/// test's reference: K22's gate is against the same math.
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

/// Assert two result sets agree within reassociation
/// noise (the GEMM test's bound: relative to the result
/// set's own magnitude scale, plus a small floor).
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

/// Quantize every row of `x` on the host: the reference
/// the GPU pipeline runs are checked against.
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

/// The K13 pipeline this device can build for the parity
/// leg: the shipped shape, or the m64 tile directly when
/// the grant covers it (same math either way; the tile
/// only changes which workgroup computes a row).
fn k13_or_skip(backend: &WgpuCompute) -> Option<K13Pipeline> {
    let ctx = backend.context();
    if let Some(shape) = wgpu_compute::kernels::k13_q4k_mmq::shipped_shape(&ctx.device.limits()) {
        return K13Pipeline::with_shape(&ctx.device, shape).ok();
    }
    K13Pipeline::with_shape(&ctx.device, MmqShape::new(64)).ok()
}

#[test]
fn k22_mmq_matches_reference_and_k13_at_prefill_batches() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    let k22 = K22Pipeline::new(&ctx.device).expect("k22 pipeline builds");
    let k13 = k13_or_skip(&backend);
    let (inn, out_rows) = (512usize, 100usize);
    let bytes = synthetic_q4k(inn, out_rows);
    let k1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let weights = k1
        .upload(ctx, "synthetic.q4k.k22", &bytes, inn, out_rows)
        .expect("k1 upload");
    for batch in [16usize, 32, 64] {
        let x = lcg_input(400 + batch as u64, batch * inn);
        let (qs, ds) = host_quantize(&x, inn);
        let got = k22
            .run(ctx, &weights, &qs, &ds, batch as u32)
            .expect("k22 run");
        let want = reference_mmq_q4k(&bytes, inn, out_rows, &qs, &ds, batch);
        check_close(&format!("k22 vs reference batch {batch}"), &got, &want);
        if let Some(k13) = &k13 {
            let shipped = k13
                .run(ctx, &weights, &qs, &ds, batch as u32)
                .expect("k13 run");
            check_close(&format!("k22 vs k13 batch {batch}"), &got, &shipped);
        }
    }
}

#[test]
fn k22_mmq_matches_reference_and_k13_on_a_chunk_tail() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    let k22 = K22Pipeline::new(&ctx.device).expect("k22 pipeline builds");
    let k13 = k13_or_skip(&backend);
    // The 1,022-token prefill runs fifteen 64-row chunks
    // and one 62-row tail; the tail's partial batch tile
    // (and an out_rows that is not a tile multiple) is
    // the shape the guards exist for.
    let (inn, out_rows, batch) = (512usize, 100usize, 62usize);
    let bytes = synthetic_q4k(inn, out_rows);
    let k1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let weights = k1
        .upload(ctx, "synthetic.q4k.k22tail", &bytes, inn, out_rows)
        .expect("k1 upload");
    let x = lcg_input(462, batch * inn);
    let (qs, ds) = host_quantize(&x, inn);
    let got = k22
        .run(ctx, &weights, &qs, &ds, batch as u32)
        .expect("k22 run");
    let want = reference_mmq_q4k(&bytes, inn, out_rows, &qs, &ds, batch);
    check_close("k22 vs reference batch 62", &got, &want);
    if let Some(k13) = &k13 {
        let shipped = k13
            .run(ctx, &weights, &qs, &ds, batch as u32)
            .expect("k13 run");
        check_close("k22 vs k13 batch 62", &got, &shipped);
    }
}

#[test]
fn k22_mmq_refuses_out_of_band_batches_by_name() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    let k22 = K22Pipeline::new(&ctx.device).expect("k22 pipeline builds");
    let (inn, out_rows) = (512usize, 64usize);
    let bytes = synthetic_q4k(inn, out_rows);
    let k1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let weights = k1
        .upload(ctx, "synthetic.q4k.k22band", &bytes, inn, out_rows)
        .expect("k1 upload");
    let qs_buf = ctx.scratch_buffer("k22 band qs", 4096);
    let ds_buf = ctx.scratch_buffer("k22 band ds", 4096);
    let out_buf = ctx.scratch_buffer("k22 band out", 4096);
    for batch in [0u32, 65] {
        let err = k22
            .prepare(ctx, &weights, &qs_buf, &ds_buf, &out_buf, batch)
            .expect_err("out-of-band batch must refuse");
        assert!(
            err.to_string().contains("outside 1..=64"),
            "batch {batch}: refusal names the band: {err}"
        );
    }
}
