//! The sweep harness's routed handle
//! ([`ProjBench`]): an upload from raw parts prepares
//! through the session's own routing ladder, names the
//! form the ladder chose, and produces the GEMV
//! form's numbers at the shipped defaults. Small
//! synthetic shapes; the harness itself sweeps the
//! real ones.

use muse_glimmer::Dtype;
use wgpu_compute::kernels::k12_q8_quant::{ds_vec4s, qs_words};
use wgpu_compute::kernels::k1_q4k_gemv::K1Pipeline;
use wgpu_compute::{ProjBench, WgpuCompute, WgpuError};

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

fn lcg_bytes(seed: u8, count: usize) -> Vec<u8> {
    let mut state = seed.max(1);
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(31).wrapping_add(17);
            state
        })
        .collect()
}

/// A synthetic Q4_K tensor's raw bytes, the kernel
/// tests' fixture shape (rotating f16 headers over
/// LCG payloads).
fn synthetic_q4k(in_elems: usize, out_rows: usize) -> Vec<u8> {
    let blocks_per_row = in_elems / 256;
    let mut data = Vec::with_capacity(out_rows * blocks_per_row * 144);
    let headers: [(u16, u16); 2] = [(0x3c00, 0x3400), (0x3800, 0x0000)];
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

fn run(backend: &WgpuCompute, call: &wgpu_compute::ProjBenchCall) {
    let ctx = backend.context();
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("proj bench test pass"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("proj bench test"),
            timestamp_writes: None,
        });
        call.encode_into(&mut pass);
    }
    ctx.queue.submit([encoder.finish()]);
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
}

#[test]
fn proj_bench_routes_to_gemv_at_shipped_defaults_and_matches_it() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let ctx = backend.context();
    // Two super-blocks in, four rows out: small enough
    // to run anywhere, shaped like the real thing.
    let (inn, out) = (512usize, 4usize);
    let raw = synthetic_q4k(inn, out);
    let bench = ProjBench::upload(&backend, "test q4k", Dtype::Q4K, &raw, inn, out)
        .expect("proj bench upload");
    assert_eq!(bench.byte_len(), raw.len());
    assert_eq!(bench.in_features(), inn);
    assert_eq!(bench.out_features(), out);

    let rows = 2usize;
    let input: Vec<f32> = (0..rows * inn)
        .map(|i| (i % 7) as f32 * 0.05 - 0.1)
        .collect();
    let in_buf = ctx.upload_f32("proj bench input", &input);
    let qs_buf = ctx.scratch_buffer("proj bench qs", (qs_words(rows, inn) * 4) as u64);
    let ds_buf = ctx.scratch_buffer("proj bench ds", (ds_vec4s(rows, inn) * 16) as u64);
    let out_buf = ctx.scratch_buffer("proj bench out", (rows * out * 4) as u64);

    let call = bench
        .prepare(&backend, &in_buf, &qs_buf, &ds_buf, &out_buf, rows)
        .expect("proj bench prepare");
    // At the shipped defaults (small-batch ladder off,
    // below the matrix threshold) the ladder serves
    // the GEMV form, and the handle names it.
    assert_eq!(call.kernel_name(), "k1_q4k_gemv");
    assert!(!call.quantized());
    assert_eq!(call.dispatches(), 1);
    run(&backend, &call);
    let got = ctx.read_f32(&out_buf, rows * out).expect("readback");

    // The direct GEMV form over the same bytes must
    // produce the identical stream.
    let k1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let w = k1
        .upload(ctx, "test q4k direct", &raw, inn, out)
        .expect("k1 upload");
    let direct = k1
        .prepare(ctx, &w, &in_buf, &out_buf, rows as u32)
        .expect("k1 prepare");
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("proj bench direct pass"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("proj bench direct"),
            timestamp_writes: None,
        });
        direct.encode_into(&mut pass);
    }
    ctx.queue.submit([encoder.finish()]);
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let want = ctx.read_f32(&out_buf, rows * out).expect("readback");
    assert_eq!(got, want, "routed GEMV and direct GEMV disagree");
}

#[test]
fn proj_bench_refuses_an_uncovered_dtype_by_name() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let err = match ProjBench::upload(&backend, "test f32", Dtype::F32, &[0u8; 64], 16, 4) {
        Ok(_) => panic!("an F32 tensor has no GEMV family and must be refused"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("the gemv kernels cover"),
        "refusal names the covered families: {err}"
    );
}
