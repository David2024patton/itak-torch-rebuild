//! Small-batch bench (K18 against the GEMV forms it
//! would replace, and the MMQ forms at the band's top):
//! the largest real-file tensor of each quant family,
//! swept over the row counts the speculative loop runs
//! at (1..=16), with the K12 quantize pass timed both
//! alone and composed into the K18 total.
//!
//! The question is the round economics one
//! (bench/spec-gpu-port-2026-11-03.md): below 16 rows
//! the session's projections stream the weights once
//! per row in GEMV form. The GEMV rows in this table
//! are that cost, per tensor; the K18 rows are one
//! weight stream per dispatch however many rows ride
//! it. The stream column is tensor bytes per second
//! (so a GEMV line at R rows reads up to R times the
//! card's DRAM rate: it is re-reading), and the ms
//! column is the verdict.
//!
//! Usage:
//!   small_batch_bench [--model PATH] [--iters N]
//!                     [--warmup N] [--budget-mib N]

use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use gguf::{GgmlType, GgufFile};
use wgpu_compute::kernels::k12_q8_quant::{K12QuantCall, K12QuantPipeline};
use wgpu_compute::kernels::k13_q4k_mmq::{K13Call, K13Pipeline, MmqShape};
use wgpu_compute::kernels::k14_q6k_mmq::{K14Call, K14Pipeline};
use wgpu_compute::kernels::k18_small_batch::{
    K18Call, K18Q4KPipeline, K18Q5KPipeline, K18Q6KPipeline,
};
use wgpu_compute::kernels::k1_q4k_gemv::{K1Call, K1Pipeline, K1Weights, ACT_NONE};
use wgpu_compute::kernels::k2_q6k_gemv::{K2Call, K2Pipeline, K2Weights};
use wgpu_compute::kernels::k3_q5k_gemv::{K3Call, K3Pipeline, K3Weights};
use wgpu_compute::{DetectPolicy, GpuContext, WgpuCompute, WgpuError};

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

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

/// The row sweep: the band K18 serves, with the
/// working points of the speculative loop (2 for a
/// depth-1 round's verify, 7 for depth 6, 16 for the
/// draft block) called out in the set.
const ROWS: [u32; 10] = [1, 2, 3, 4, 6, 7, 8, 12, 15, 16];

/// One encodable dispatch, whichever kernel built it;
/// the bench times compositions, not types.
enum BenchCall {
    K1(K1Call),
    K2(K2Call),
    K3(K3Call),
    K12(K12QuantCall),
    K13(K13Call),
    K14(K14Call),
    K18(K18Call),
}

impl BenchCall {
    fn encode_into(&self, pass: &mut wgpu::ComputePass<'_>) {
        match self {
            BenchCall::K1(c) => c.encode_into(pass),
            BenchCall::K2(c) => c.encode_into(pass),
            BenchCall::K3(c) => c.encode_into(pass),
            BenchCall::K12(c) => c.encode_into(pass),
            BenchCall::K13(c) => c.encode_into(pass),
            BenchCall::K14(c) => c.encode_into(pass),
            BenchCall::K18(c) => c.encode_into(pass),
        }
    }
}

fn time_calls(ctx: &GpuContext, calls: &[BenchCall], warmup: u32, iters: u32) -> f64 {
    let run_pass = |ctx: &GpuContext| {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("small batch bench pass"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("small batch bench"),
                timestamp_writes: None,
            });
            for c in calls {
                c.encode_into(&mut pass);
            }
        }
        ctx.queue.submit([encoder.finish()]);
    };
    for _ in 0..warmup {
        run_pass(ctx);
    }
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll after warmup");
    let start = Instant::now();
    for _ in 0..iters {
        run_pass(ctx);
    }
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll after timed loop");
    start.elapsed().as_secs_f64() / f64::from(iters)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let model = arg(&args, "--model").unwrap_or_else(|| {
        "/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf".to_string()
    });
    let iters: u32 = arg(&args, "--iters")
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let warmup: u32 = arg(&args, "--warmup")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let budget: u64 = arg(&args, "--budget-mib")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(1536)
        * 1024
        * 1024;

    let file = match GgufFile::open(Path::new(&model)) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("cannot open {model}: {e}");
            return ExitCode::from(2);
        }
    };
    // The largest tensor of each family, under the
    // budget: the Q5_K pick is the untied output head,
    // the single biggest small-batch stream in a
    // verify forward.
    let mut by_size: Vec<(String, GgmlType, u64, usize, usize)> = file
        .tensors()
        .iter()
        .filter(|t| {
            matches!(
                t.ggml_type(),
                Some(GgmlType::Q4K | GgmlType::Q6K | GgmlType::Q5K)
            )
        })
        .filter_map(|t| {
            let bytes = t.byte_len()?;
            let dims = t.dims();
            Some((
                t.name().to_string(),
                t.ggml_type()?,
                bytes,
                dims[0] as usize,
                dims[1] as usize,
            ))
        })
        .collect();
    by_size.sort_by_key(|c| std::cmp::Reverse(c.2));
    let mut chosen: Vec<(String, GgmlType, u64, usize, usize)> = Vec::new();
    let mut seen_families: Vec<GgmlType> = Vec::new();
    let mut total: u64 = 0;
    for cand in &by_size {
        if seen_families.contains(&cand.1) {
            continue;
        }
        if total + cand.2 > budget {
            continue;
        }
        total += cand.2;
        seen_families.push(cand.1);
        chosen.push(cand.clone());
        if chosen.len() >= 3 {
            break;
        }
    }
    if chosen.is_empty() {
        eprintln!("no Q4_K/Q6_K/Q5_K tensor fits the budget");
        return ExitCode::from(2);
    }

    let backend = match WgpuCompute::with_policy(&DetectPolicy::default()) {
        Ok(b) => b,
        Err(WgpuError::NoSuitableAdapter { seen, .. }) => {
            println!(
                "SKIP: no suitable GPU on this machine (enumerated: [{}])",
                seen.join("; ")
            );
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("backend construction failed: {e}");
            return ExitCode::from(2);
        }
    };
    let ctx = backend.context();
    println!("adapter: {}", ctx.adapter.name);
    println!("rows: {ROWS:?}, iters {iters} (+{warmup} warmup)");

    let k1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let k2 = K2Pipeline::new(&ctx.device).expect("k2 pipeline");
    let k3 = K3Pipeline::new(&ctx.device).expect("k3 pipeline");
    let k12 = K12QuantPipeline::new(&ctx.device, ACT_NONE).expect("k12 pipeline");
    let small4 = K18Q4KPipeline::for_device(&ctx.device)
        .expect("k18 q4k probe")
        .expect("k18 q4k pipeline on the bench device");
    let small6 = K18Q6KPipeline::for_device(&ctx.device)
        .expect("k18 q6k probe")
        .expect("k18 q6k pipeline on the bench device");
    let small5 = K18Q5KPipeline::for_device(&ctx.device)
        .expect("k18 q5k probe")
        .expect("k18 q5k pipeline on the bench device");
    println!(
        "k18 shipped shapes: q4k {} q6k {} q5k {}",
        small4.shape().name(),
        small6.shape().name(),
        small5.shape().name()
    );
    // The MMQ comparison at the band's top (16 rows):
    // the standard tiles, whose showing at small rows
    // is the reason K18 exists.
    let mmq4 = K13Pipeline::with_shape(&ctx.device, MmqShape::new(64))
        .ok()
        .or_else(|| K13Pipeline::with_shape(&ctx.device, MmqShape::new(32)).ok());
    let mmq6 = K14Pipeline::with_shape(&ctx.device, MmqShape::new(64))
        .ok()
        .or_else(|| K14Pipeline::with_shape(&ctx.device, MmqShape::new(32)).ok());

    let mut keep_q4: Vec<K1Weights> = Vec::new();
    let mut keep_q6: Vec<K2Weights> = Vec::new();
    let mut keep_q5: Vec<K3Weights> = Vec::new();
    for (name, ty, bytes, inn, out) in &chosen {
        let raw = file.tensor_bytes(name).expect("tensor bytes");
        println!(
            "\n{name} ({ty:?}, {inn}x{out}, {:.1} MB)",
            *bytes as f64 / 1e6
        );
        let report = |label: &str, rows: u32, secs: f64| {
            let rows_s = f64::from(rows) / secs;
            let stream = *bytes as f64 / secs / 1e9;
            println!(
                "  rows {rows:>2}  {label:<22} {:>9.3} ms  {:>9.0} tok/s  {:>7.1} GB/s of tensor",
                secs * 1e3,
                rows_s,
                stream
            );
        };
        // Buffers are sized at the sweep's top and the
        // prepares slice them per row count (every
        // prepare validates sizes as minimums).
        let max_rows = 16usize;
        let input = lcg_input(7, max_rows * inn);
        let in_buf = ctx.upload_f32("bench input", &input);
        let out_buf = ctx.scratch_buffer("bench output", (max_rows * out * 4) as u64);
        let qs_buf = ctx.scratch_buffer(
            "bench qs",
            (wgpu_compute::kernels::k12_q8_quant::qs_words(max_rows, *inn) * 4) as u64,
        );
        let ds_buf = ctx.scratch_buffer(
            "bench ds",
            (wgpu_compute::kernels::k12_q8_quant::ds_vec4s(max_rows, *inn) * 16) as u64,
        );
        match ty {
            GgmlType::Q4K => {
                let w = k1.upload(ctx, name, raw, *inn, *out).expect("k1 upload");
                for rows in ROWS {
                    let gemv = [BenchCall::K1(
                        k1.prepare(ctx, &w, &in_buf, &out_buf, rows)
                            .expect("k1 prepare"),
                    )];
                    report("gemv", rows, time_calls(ctx, &gemv, warmup, iters));
                    let quant = BenchCall::K12(
                        k12.prepare(ctx, &in_buf, &qs_buf, &ds_buf, rows, *inn)
                            .expect("k12 prepare"),
                    );
                    let prod = [BenchCall::K18(
                        small4
                            .prepare(ctx, &w, &qs_buf, &ds_buf, &out_buf, rows)
                            .expect("k18 q4k prepare"),
                    )];
                    let prod_s = time_calls(ctx, &prod, warmup, iters);
                    report(
                        &format!("k18 {} product", small4.shape().name()),
                        rows,
                        prod_s,
                    );
                    let total = [
                        quant,
                        BenchCall::K18(
                            small4
                                .prepare(ctx, &w, &qs_buf, &ds_buf, &out_buf, rows)
                                .expect("k18 q4k prepare"),
                        ),
                    ];
                    report(
                        &format!("k18 {} + quantize", small4.shape().name()),
                        rows,
                        time_calls(ctx, &total, warmup, iters),
                    );
                    if rows == 16 {
                        if let Some(mmq) = &mmq4 {
                            let prod = [BenchCall::K13(
                                mmq.prepare(ctx, &w, &qs_buf, &ds_buf, &out_buf, rows)
                                    .expect("k13 prepare"),
                            )];
                            report(
                                &format!("mmq {} product", mmq.shape().name()),
                                rows,
                                time_calls(ctx, &prod, warmup, iters),
                            );
                        }
                    }
                }
                keep_q4.push(w);
            }
            GgmlType::Q6K => {
                let w = k2.upload(ctx, name, raw, *inn, *out).expect("k2 upload");
                for rows in ROWS {
                    let gemv = [BenchCall::K2(
                        k2.prepare(ctx, &w, &in_buf, &out_buf, rows)
                            .expect("k2 prepare"),
                    )];
                    report("gemv", rows, time_calls(ctx, &gemv, warmup, iters));
                    let quant = BenchCall::K12(
                        k12.prepare(ctx, &in_buf, &qs_buf, &ds_buf, rows, *inn)
                            .expect("k12 prepare"),
                    );
                    let prod = [BenchCall::K18(
                        small6
                            .prepare(ctx, &w, &qs_buf, &ds_buf, &out_buf, rows)
                            .expect("k18 q6k prepare"),
                    )];
                    let prod_s = time_calls(ctx, &prod, warmup, iters);
                    report(
                        &format!("k18 {} product", small6.shape().name()),
                        rows,
                        prod_s,
                    );
                    let total = [
                        quant,
                        BenchCall::K18(
                            small6
                                .prepare(ctx, &w, &qs_buf, &ds_buf, &out_buf, rows)
                                .expect("k18 q6k prepare"),
                        ),
                    ];
                    report(
                        &format!("k18 {} + quantize", small6.shape().name()),
                        rows,
                        time_calls(ctx, &total, warmup, iters),
                    );
                    if rows == 16 {
                        if let Some(mmq) = &mmq6 {
                            let prod = [BenchCall::K14(
                                mmq.prepare(ctx, &w, &qs_buf, &ds_buf, &out_buf, rows)
                                    .expect("k14 prepare"),
                            )];
                            report(
                                &format!("mmq {} product", mmq.shape().name()),
                                rows,
                                time_calls(ctx, &prod, warmup, iters),
                            );
                        }
                    }
                }
                keep_q6.push(w);
            }
            GgmlType::Q5K => {
                let w = k3.upload(ctx, name, raw, *inn, *out).expect("k3 upload");
                for rows in ROWS {
                    let gemv = [BenchCall::K3(
                        k3.prepare(ctx, &w, &in_buf, &out_buf, rows)
                            .expect("k3 prepare"),
                    )];
                    report("gemv", rows, time_calls(ctx, &gemv, warmup, iters));
                    let quant = BenchCall::K12(
                        k12.prepare(ctx, &in_buf, &qs_buf, &ds_buf, rows, *inn)
                            .expect("k12 prepare"),
                    );
                    let prod = [BenchCall::K18(
                        small5
                            .prepare(ctx, &w, &qs_buf, &ds_buf, &out_buf, rows)
                            .expect("k18 q5k prepare"),
                    )];
                    let prod_s = time_calls(ctx, &prod, warmup, iters);
                    report(
                        &format!("k18 {} product", small5.shape().name()),
                        rows,
                        prod_s,
                    );
                    let total = [
                        quant,
                        BenchCall::K18(
                            small5
                                .prepare(ctx, &w, &qs_buf, &ds_buf, &out_buf, rows)
                                .expect("k18 q5k prepare"),
                        ),
                    ];
                    report(
                        &format!("k18 {} + quantize", small5.shape().name()),
                        rows,
                        time_calls(ctx, &total, warmup, iters),
                    );
                }
                keep_q5.push(w);
            }
            _ => unreachable!("selection filtered to Q4_K/Q6_K/Q5_K"),
        }
    }
    let _ = (keep_q4, keep_q6, keep_q5);
    ExitCode::SUCCESS
}
