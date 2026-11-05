//! The chunked GEMV form: a small batch served as
//! several windowed GEMV dispatches at power-of-two row
//! counts ([`pow2_chunks`] in the session, mirrored here)
//! instead of one dispatch at the batch's own count.
//!
//! The gate is exactness, not closeness: every output
//! element's reduction runs over the same weight row and
//! the same activation row in both forms, in the same
//! shader, so the chunked product must equal the
//! whole-batch GEMV product bit for bit. Anything less
//! would mean the windowed binding changed the math,
//! which is the defect this file exists to catch.
//!
//! The window prepares' refusals are gated too: an empty
//! window, a window past either buffer, and a window
//! whose offsets break the device's storage-binding
//! alignment all refuse by name, so the session's
//! fallback composition can rely on the refusal instead
//! of discovering a bad binding at encode time.
//!
//! These tests need a discrete Vulkan GPU under the
//! default selection policy. On a machine without one
//! they print the refusal and pass; the session-level
//! routing is gated by the resident session tests, and
//! the head-to-head speed proof runs on the server
//! (bench/chunked-gemv-2026-11-05.md).

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
/// blocks; the scale word sits at the block tail).
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

/// Decompose `batch` into power-of-two row chunks,
/// largest first: the session's decomposition, mirrored
/// here so the test drives the prepares the way the
/// session composes them (the decomposition itself is
/// unit-gated in the session module).
fn pow2_chunks(batch: u32) -> Vec<(u32, u32)> {
    let mut chunks = Vec::new();
    let mut start = 0u32;
    let mut rest = batch;
    while rest > 0 {
        let len = 1u32 << (31 - rest.leading_zeros());
        chunks.push((start, len));
        start += len;
        rest -= len;
    }
    chunks
}

/// The row counts the gates sweep: the verify band's
/// shapes, including the power-of-two counts (one chunk,
/// windowed whole) and the counts the strix report found
/// pathological through a shared path (3, 5, 6, 7).
const BAND: [usize; 7] = [2, 3, 4, 5, 6, 7, 15];

/// Assert two products are bit-identical, element by
/// element: the chunked form's contract.
fn check_bit_exact(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{label}: element {i} differs: chunked {g} vs whole-batch {w}"
        );
    }
}

#[test]
fn chunked_q4k_matches_whole_batch_gemv_bit_exactly() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    // Width 512 = two super-blocks per row; 100 output
    // rows, the sibling fixtures' geometry.
    let (inn, out_rows) = (512usize, 100usize);
    let bytes = synthetic_q4k(inn, out_rows);
    let k1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let weights: K1Weights = k1
        .upload(ctx, "synthetic.q4k.chunked", &bytes, inn, out_rows)
        .expect("k1 upload");
    for batch in BAND {
        let x = lcg_input(900 + batch as u64, batch * inn);
        let in_buf = ctx.upload_f32("chunked test q4k input", &x);
        // Whole-batch form.
        let whole_buf = ctx.scratch_buffer("chunked test q4k whole", (batch * out_rows * 4) as u64);
        let whole = k1
            .prepare(ctx, &weights, &in_buf, &whole_buf, batch as u32)
            .expect("k1 whole prepare");
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("chunked q4k whole run"),
            });
        whole.encode(&mut encoder);
        ctx.queue.submit([encoder.finish()]);
        let want = ctx
            .read_f32(&whole_buf, batch * out_rows)
            .expect("k1 whole readback");
        // Chunked form: one windowed dispatch per chunk,
        // all into one output buffer.
        let chunked_buf =
            ctx.scratch_buffer("chunked test q4k chunked", (batch * out_rows * 4) as u64);
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("chunked q4k chunked run"),
            });
        for (start, len) in pow2_chunks(batch as u32) {
            let call = k1
                .prepare_rows(ctx, &weights, &in_buf, &chunked_buf, start, len)
                .expect("k1 window prepare");
            call.encode(&mut encoder);
        }
        ctx.queue.submit([encoder.finish()]);
        let got = ctx
            .read_f32(&chunked_buf, batch * out_rows)
            .expect("k1 chunked readback");
        check_bit_exact(&format!("chunked q4k batch {batch}"), &got, &want);
    }
}

#[test]
fn chunked_q6k_matches_whole_batch_gemv_bit_exactly() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    let (inn, out_rows) = (512usize, 100usize);
    let bytes = synthetic_q6k(inn, out_rows);
    let k2 = K2Pipeline::new(&ctx.device).expect("k2 pipeline");
    let weights: K2Weights = k2
        .upload(ctx, "synthetic.q6k.chunked", &bytes, inn, out_rows)
        .expect("k2 upload");
    for batch in BAND {
        let x = lcg_input(1200 + batch as u64, batch * inn);
        let in_buf = ctx.upload_f32("chunked test q6k input", &x);
        let whole_buf = ctx.scratch_buffer("chunked test q6k whole", (batch * out_rows * 4) as u64);
        let whole = k2
            .prepare(ctx, &weights, &in_buf, &whole_buf, batch as u32)
            .expect("k2 whole prepare");
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("chunked q6k whole run"),
            });
        whole.encode(&mut encoder);
        ctx.queue.submit([encoder.finish()]);
        let want = ctx
            .read_f32(&whole_buf, batch * out_rows)
            .expect("k2 whole readback");
        let chunked_buf =
            ctx.scratch_buffer("chunked test q6k chunked", (batch * out_rows * 4) as u64);
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("chunked q6k chunked run"),
            });
        for (start, len) in pow2_chunks(batch as u32) {
            let call = k2
                .prepare_rows(ctx, &weights, &in_buf, &chunked_buf, start, len)
                .expect("k2 window prepare");
            call.encode(&mut encoder);
        }
        ctx.queue.submit([encoder.finish()]);
        let got = ctx
            .read_f32(&chunked_buf, batch * out_rows)
            .expect("k2 chunked readback");
        check_bit_exact(&format!("chunked q6k batch {batch}"), &got, &want);
    }
}

#[test]
fn chunked_q5k_matches_whole_batch_gemv_bit_exactly() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    let (inn, out_rows) = (512usize, 100usize);
    let bytes = synthetic_q5k(inn, out_rows);
    let k3 = K3Pipeline::new(&ctx.device).expect("k3 pipeline");
    let weights: K3Weights = k3
        .upload(ctx, "synthetic.q5k.chunked", &bytes, inn, out_rows)
        .expect("k3 upload");
    for batch in BAND {
        let x = lcg_input(1500 + batch as u64, batch * inn);
        let in_buf = ctx.upload_f32("chunked test q5k input", &x);
        let whole_buf = ctx.scratch_buffer("chunked test q5k whole", (batch * out_rows * 4) as u64);
        let whole = k3
            .prepare(ctx, &weights, &in_buf, &whole_buf, batch as u32)
            .expect("k3 whole prepare");
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("chunked q5k whole run"),
            });
        whole.encode(&mut encoder);
        ctx.queue.submit([encoder.finish()]);
        let want = ctx
            .read_f32(&whole_buf, batch * out_rows)
            .expect("k3 whole readback");
        let chunked_buf =
            ctx.scratch_buffer("chunked test q5k chunked", (batch * out_rows * 4) as u64);
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("chunked q5k chunked run"),
            });
        for (start, len) in pow2_chunks(batch as u32) {
            let call = k3
                .prepare_rows(ctx, &weights, &in_buf, &chunked_buf, start, len)
                .expect("k3 window prepare");
            call.encode(&mut encoder);
        }
        ctx.queue.submit([encoder.finish()]);
        let got = ctx
            .read_f32(&chunked_buf, batch * out_rows)
            .expect("k3 chunked readback");
        check_bit_exact(&format!("chunked q5k batch {batch}"), &got, &want);
    }
}

#[test]
fn windowed_prepare_refuses_empty_past_end_and_misaligned() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx: &GpuContext = backend.context();
    let (inn, out_rows) = (512usize, 100usize);
    let bytes = synthetic_q4k(inn, out_rows);
    let k1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let weights = k1
        .upload(ctx, "synthetic.q4k.chunked", &bytes, inn, out_rows)
        .expect("k1 upload");
    // Buffers sized for exactly 4 rows.
    let x = lcg_input(42, 4 * inn);
    let in_buf = ctx.upload_f32("chunked test refusal input", &x);
    let out_buf = ctx.scratch_buffer("chunked test refusal out", (4 * out_rows * 4) as u64);
    // An empty window refuses.
    assert!(
        k1.prepare_rows(ctx, &weights, &in_buf, &out_buf, 0, 0)
            .is_err(),
        "an empty window must refuse"
    );
    // A window running past the buffers refuses.
    assert!(
        k1.prepare_rows(ctx, &weights, &in_buf, &out_buf, 3, 2)
            .is_err(),
        "a window past the buffer end must refuse"
    );
    assert!(
        k1.prepare_rows(ctx, &weights, &in_buf, &out_buf, 5, 1)
            .is_err(),
        "a window starting past the buffer end must refuse"
    );
    // A window whose output offset (row 1 = 100 f32 =
    // 400 bytes in) breaks the device's storage-binding
    // alignment refuses; on a device whose alignment
    // divides 400 the same window is legal and prepares.
    let align = u64::from(ctx.device.limits().min_storage_buffer_offset_alignment);
    let row_out_bytes = (out_rows * 4) as u64;
    let windowed = k1.prepare_rows(ctx, &weights, &in_buf, &out_buf, 1, 1);
    if row_out_bytes % align != 0 {
        assert!(
            windowed.is_err(),
            "a misaligned window (output offset {row_out_bytes}, alignment {align}) must refuse"
        );
    } else {
        assert!(
            windowed.is_ok(),
            "an aligned window must prepare (output offset {row_out_bytes}, alignment {align})"
        );
    }
    // A full-window prepare at row 0 always prepares:
    // the chunked form's power-of-two batches ride it.
    assert!(
        k1.prepare_rows(ctx, &weights, &in_buf, &out_buf, 0, 4)
            .is_ok(),
        "a whole-buffer window must prepare"
    );
}
