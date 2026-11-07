//! K24 (the Q4_K coopmat MMQ v2, register-resident)
//! against the host reference running the same
//! quantized math, against K13 (the shipped Q4_K MMQ),
//! and against K23 (the v1 coopmat kernel) on the same
//! inputs, at the shape the lane protocol pins: batch
//! 64 at the model's hidden width 6656, plus a 62-row
//! chunk tail (the 1,022-token prefill's last chunk
//! shape).
//!
//! K24 runs K13's and K23's exact math (the same
//! integer dots and per-sub-block scale fold; the
//! products issue to the cooperative matrix units and
//! the fold reads the accumulator's elements directly
//! instead of round-tripping through a shared tile),
//! so all gates use the crate's MMQ reassociation bound
//! (`check_close`): the reference gate is the
//! correctness claim, the K13 gate is the parity claim
//! the routing relies on, and the K23 gate is the A/B
//! claim against the v1 kernel it succeeds. The
//! reference gate is also the proof of the wave64
//! accumulator element mapping the register fold
//! stands on (Mesa's GFX11 lowering: element e at row
//! (inv / 16) + 4e, column inv % 16; see the kernel
//! module's docs): a wrong mapping scrambles the
//! output rows and columns and fails this gate by a
//! wide margin, not by reassociation noise.
//!
//! These tests need a discrete Vulkan GPU whose device
//! was opened through the coopmat bridge: run them with
//! WGPU_COMPUTE_COOPMAT_BRIDGE=1. On a machine without
//! a suitable adapter, or with a stock device (no
//! bridge), they print the reason and pass: the kernel
//! cannot exist there by design, and the bridge smoke
//! plus the prefill A/B run on the server through the
//! resident suite and the profile tool
//! (bench/coopmat-v2-2026-11-07.md). Naga cannot
//! validate the passthrough module, so this file's
//! on-card numeric gates are the shader's correctness
//! evidence, as the feasibility note disclosed.

use wgpu_compute::kernels::k12_q8_quant::quantize_row_q8_1;
use wgpu_compute::kernels::k13_q4k_mmq::{K13Pipeline, MmqShape};
use wgpu_compute::kernels::k1_q4k_gemv::K1Pipeline;
use wgpu_compute::kernels::k23_q4k_coopmat::K23Pipeline;
use wgpu_compute::kernels::k24_q4k_coopmat::K24Pipeline;
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

/// The K24 pipeline on this backend's device, or `None`
/// (with the reason printed) when the device is a stock
/// device or fails the kernel's gates: the pipeline
/// exists only behind the coopmat bridge.
fn k24_or_skip(backend: &WgpuCompute) -> Option<K24Pipeline> {
    let ctx = backend.context();
    let Some(info) = ctx.coopmat else {
        eprintln!(
            "skipping k24 test: stock device (no coopmat bridge); \
             run with WGPU_COMPUTE_COOPMAT_BRIDGE=1 on a coopmat device"
        );
        return None;
    };
    match K24Pipeline::for_device(&ctx.device, &info) {
        Ok(Some(p)) => Some(p),
        Ok(None) => {
            eprintln!("skipping k24 test: device fails the kernel's gates (see diag log)");
            None
        }
        Err(e) => panic!("k24 pipeline construction failed unexpectedly: {e}"),
    }
}

/// The K23 (v1) pipeline on this backend's device, for
/// the A/B parity leg, or `None` (with the reason
/// printed) on the same terms as [`k24_or_skip`].
fn k23_or_skip(backend: &WgpuCompute) -> Option<K23Pipeline> {
    let ctx = backend.context();
    let Some(info) = ctx.coopmat else {
        eprintln!("skipping k24 v1-parity leg: stock device (no coopmat bridge)");
        return None;
    };
    match K23Pipeline::for_device(&ctx.device, &info) {
        Ok(Some(p)) => Some(p),
        Ok(None) => {
            eprintln!("skipping k24 v1-parity leg: device fails the v1 kernel's gates");
            None
        }
        Err(e) => panic!("k23 pipeline construction failed unexpectedly: {e}"),
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
/// test's reference: K24's gate is against the same math.
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
fn k24_coopmat_matches_reference_k13_and_k23_at_hidden_width() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let Some(k24) = k24_or_skip(&backend) else {
        return;
    };
    let ctx = backend.context();
    let k13 = k13_or_skip(&backend);
    let k23 = k23_or_skip(&backend);
    // The protocol's pinned shape: the model's hidden
    // width, a full 64-row prefill chunk. The output
    // width is cut to 512 rows so the host reference
    // stays tractable; the width that stresses the
    // kernel's chunk loop is the input width, kept whole.
    let (inn, out_rows) = (6656usize, 512usize);
    let bytes = synthetic_q4k(inn, out_rows);
    let k1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let weights = k1
        .upload(ctx, "synthetic.q4k.k24", &bytes, inn, out_rows)
        .expect("k1 upload");
    for batch in [64usize, 62] {
        let x = lcg_input(400 + batch as u64, batch * inn);
        let (qs, ds) = host_quantize(&x, inn);
        let got = k24
            .run(ctx, &weights, &qs, &ds, batch as u32)
            .expect("k24 run");
        let want = reference_mmq_q4k(&bytes, inn, out_rows, &qs, &ds, batch);
        check_close(&format!("k24 vs reference batch {batch}"), &got, &want);
        if let Some(k13) = &k13 {
            let shipped = k13
                .run(ctx, &weights, &qs, &ds, batch as u32)
                .expect("k13 run");
            check_close(&format!("k24 vs k13 batch {batch}"), &got, &shipped);
        }
        if let Some(k23) = &k23 {
            let v1 = k23
                .run(ctx, &weights, &qs, &ds, batch as u32)
                .expect("k23 run");
            check_close(&format!("k24 vs k23 batch {batch}"), &got, &v1);
        }
    }
}

#[test]
fn k24_coopmat_matches_k13_on_the_full_square() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let Some(k24) = k24_or_skip(&backend) else {
        return;
    };
    let ctx = backend.context();
    let Some(k13) = k13_or_skip(&backend) else {
        eprintln!("skipping k24 full-square parity: no k13 pipeline on this device");
        return;
    };
    // The full attention-projection shape: 6656 out x
    // 6656 in at batch 64, K13 parity only (the host
    // reference at this size belongs to the test above).
    let (inn, out_rows, batch) = (6656usize, 6656usize, 64usize);
    let bytes = synthetic_q4k(inn, out_rows);
    let k1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let weights = k1
        .upload(ctx, "synthetic.q4k.k24full", &bytes, inn, out_rows)
        .expect("k1 upload");
    let x = lcg_input(723, batch * inn);
    let (qs, ds) = host_quantize(&x, inn);
    let got = k24
        .run(ctx, &weights, &qs, &ds, batch as u32)
        .expect("k24 run");
    let shipped = k13
        .run(ctx, &weights, &qs, &ds, batch as u32)
        .expect("k13 run");
    check_close("k24 vs k13 batch 64 full square", &got, &shipped);
}

#[test]
fn k24_coopmat_refuses_out_of_band_batches_by_name() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let Some(k24) = k24_or_skip(&backend) else {
        return;
    };
    let ctx = backend.context();
    let (inn, out_rows) = (512usize, 64usize);
    let bytes = synthetic_q4k(inn, out_rows);
    let k1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let weights = k1
        .upload(ctx, "synthetic.q4k.k24band", &bytes, inn, out_rows)
        .expect("k1 upload");
    let qs_buf = ctx.scratch_buffer("k24 band qs", 4096);
    let ds_buf = ctx.scratch_buffer("k24 band ds", 4096);
    let out_buf = ctx.scratch_buffer("k24 band out", 4096);
    for batch in [0u32, 65] {
        let err = k24
            .prepare(ctx, &weights, &qs_buf, &ds_buf, &out_buf, batch)
            .expect_err("out-of-band batch must refuse");
        assert!(
            err.to_string().contains("outside 1..=64"),
            "batch {batch}: refusal names the band: {err}"
        );
    }
}
