//! Split-K forms of the K13/K14 MMQ products against
//! synthetic tensors with known contents
//! (bench/splitk-mmq-2026-11-07.md is the design record).
//!
//! The gate is the plain MMQ gate, unchanged: the split
//! product must match a host reference running the SAME
//! quantized math (exact integer dots plus the
//! per-sub-block scale fold) within float reassociation
//! noise, the bound `k12_k14_mmq`'s `check_close` uses.
//! The split form is not bit-identical to the unsplit
//! product by construction (the f32 accumulation is
//! regrouped: one chain per slice, then the slice sums
//! added in ascending slice order), so split-vs-unsplit
//! is gated at the same reassociation bound and the
//! measured drift is printed, not hidden.
//!
//! Shapes covered: the 256-wide KV projection shape
//! (6656 in, 256 out) that motivates the split, widths
//! whose super-block count the factor does not divide
//! (5 blocks at factor 4: ranges 2/2/1/empty), and a
//! factor past the block count (8 over 3 blocks: the
//! trailing slices are empty and must contribute exact
//! zeros). These tests need a discrete Vulkan GPU under
//! the default selection policy; on a machine without
//! one they print the refusal and pass (the VM-side
//! gates are the shader text pins and the Naga
//! validation in the kernel modules).

use wgpu_compute::kernels::k12_q8_quant::quantize_row_q8_1;
use wgpu_compute::kernels::k13_q4k_mmq::{K13Pipeline, MmqShape};
use wgpu_compute::kernels::k14_q6k_mmq::K14Pipeline;
use wgpu_compute::kernels::k1_q4k_gemv::K1Pipeline;
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

/// The host MMQ reference for Q4_K (the k12_k14_mmq
/// reference, mirrored): exact integer dots folded per
/// sub-block in super-block order.
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

/// The host MMQ reference for Q6_K (the k12_k14_mmq
/// reference, mirrored): exact integer dots folded per
/// 16-element half.
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

/// Assert two result sets agree within reassociation noise
/// (the MMQ gate's bound: relative to the result set's own
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
    if !out.contains(&wide) && K13Pipeline::with_shape(&backend.context().device, wide).is_ok() {
        out.push(wide);
    }
    out
}

#[test]
fn k13_split_matches_quantized_reference() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    // (in, out, batch, splits): the KV projection width at
    // the model's real K length; a ragged block count the
    // factor does not divide; a factor past the block
    // count, where trailing slices are empty.
    let cases: [(usize, usize, usize, u32); 5] = [
        (6656, 256, 4, 2),
        (6656, 256, 4, 8),
        (1280, 100, 3, 4),
        (768, 96, 5, 2),
        (768, 96, 5, 8),
    ];
    for shape in shapes(&backend) {
        let mmq = K13Pipeline::with_shape(&ctx.device, shape).expect("k13 pipeline builds");
        for (inn, out_rows, batch, splits) in cases {
            let bytes = synthetic_q4k(inn, out_rows);
            let k1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
            let weights = k1
                .upload(ctx, "synthetic.q4k.split", &bytes, inn, out_rows)
                .expect("k1 upload");
            let x = lcg_input(700 + batch as u64 + splits as u64, batch * inn);
            let (qs, ds) = host_quantize(&x, inn);
            let want = reference_mmq_q4k(&bytes, inn, out_rows, &qs, &ds, batch);
            let got = mmq
                .run_split(ctx, &weights, &qs, &ds, batch as u32, splits)
                .expect("k13 split run");
            check_close(
                &format!(
                    "k13 split {} in {inn} out {out_rows} batch {batch} splits {splits}",
                    shape.name()
                ),
                &got,
                &want,
            );
            if (inn, out_rows) == (6656, 256) {
                // The drift against the unsplit product
                // itself: reassociation class, printed.
                let plain = mmq
                    .run(ctx, &weights, &qs, &ds, batch as u32)
                    .expect("k13 run");
                check_close(
                    &format!("k13 split-vs-unsplit {} splits {splits}", shape.name()),
                    &got,
                    &plain,
                );
            }
        }
    }
}

#[test]
fn k14_split_matches_quantized_reference() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    // The 256-wide KV projection shape first: it is the
    // shape the split exists for (6656 in, 256 out).
    let cases: [(usize, usize, usize, u32); 5] = [
        (6656, 256, 4, 2),
        (6656, 256, 4, 4),
        (6656, 256, 4, 8),
        (768, 96, 5, 2),
        (768, 96, 5, 8),
    ];
    for shape in shapes(&backend) {
        let mmq = K14Pipeline::with_shape(&ctx.device, shape).expect("k14 pipeline builds");
        for (inn, out_rows, batch, splits) in cases {
            let bytes = synthetic_q6k(inn, out_rows);
            let k2 = K2Pipeline::new(&ctx.device).expect("k2 pipeline");
            let weights = k2
                .upload(ctx, "synthetic.q6k.split", &bytes, inn, out_rows)
                .expect("k2 upload");
            let x = lcg_input(900 + batch as u64 + splits as u64, batch * inn);
            let (qs, ds) = host_quantize(&x, inn);
            let want = reference_mmq_q6k(&bytes, inn, out_rows, &qs, &ds, batch);
            let got = mmq
                .run_split(ctx, &weights, &qs, &ds, batch as u32, splits)
                .expect("k14 split run");
            check_close(
                &format!(
                    "k14 split {} in {inn} out {out_rows} batch {batch} splits {splits}",
                    shape.name()
                ),
                &got,
                &want,
            );
            if (inn, out_rows) == (6656, 256) {
                let plain = mmq
                    .run(ctx, &weights, &qs, &ds, batch as u32)
                    .expect("k14 run");
                check_close(
                    &format!("k14 split-vs-unsplit {} splits {splits}", shape.name()),
                    &got,
                    &plain,
                );
            }
        }
    }
}

#[test]
fn split_factor_refusals_are_named() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    let (inn, out_rows, batch) = (512usize, 64usize, 2usize);
    let x = lcg_input(11, batch * inn);
    let (qs, ds) = host_quantize(&x, inn);
    let bytes4 = synthetic_q4k(inn, out_rows);
    let k1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let w4 = k1
        .upload(ctx, "synthetic.q4k.refuse", &bytes4, inn, out_rows)
        .expect("k1 upload");
    let mmq13 = K13Pipeline::with_shape(&ctx.device, MmqShape::new(64)).expect("k13 builds");
    // A split of 1 is the unsplit product, 0 is nonsense,
    // and 9 is past the shared ceiling: all refuse.
    for splits in [0u32, 1, 9] {
        assert!(
            mmq13
                .run_split(ctx, &w4, &qs, &ds, batch as u32, splits)
                .is_err(),
            "k13 splits {splits} must refuse"
        );
    }
    let bytes6 = synthetic_q6k(inn, out_rows);
    let k2 = K2Pipeline::new(&ctx.device).expect("k2 pipeline");
    let w6 = k2
        .upload(ctx, "synthetic.q6k.refuse", &bytes6, inn, out_rows)
        .expect("k2 upload");
    let mmq14 = K14Pipeline::with_shape(&ctx.device, MmqShape::new(64)).expect("k14 builds");
    for splits in [0u32, 1, 9] {
        assert!(
            mmq14
                .run_split(ctx, &w6, &qs, &ds, batch as u32, splits)
                .is_err(),
            "k14 splits {splits} must refuse"
        );
    }
}
