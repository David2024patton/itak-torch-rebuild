//! MMQ prefill bench (K12 quantizer + K13/K14 integer-dot
//! product kernels) against the same real-file tensors and
//! the same discipline as `k10_bench`: the largest Q4_K and
//! Q6_K tensors resident, each projection shape run through
//! the GEMV replay, the shipped f32 GEMM (t32x128), and the
//! MMQ at both tile shapes, with the K12 quantize pass timed
//! both alone and composed into the MMQ total. The GEMV at
//! batch 1 doubles as the window canary (~816 GB/s Q4_K,
//! ~832 GB/s Q6_K in a healthy window).
//!
//! Usage:
//!   mmq_bench [--model PATH] [--batch N] [--iters N]
//!             [--warmup N] [--budget-mib N]

use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use gguf::{GgmlType, GgufFile};
use wgpu_compute::kernels::k10_q4k_gemm::{GemmShape, K10Call, K10Pipeline};
use wgpu_compute::kernels::k11_q6k_gemm::{K11Call, K11Pipeline};
use wgpu_compute::kernels::k12_q8_quant::{K12QuantCall, K12QuantPipeline};
use wgpu_compute::kernels::k13_q4k_mmq::{K13Call, K13Pipeline, MmqShape};
use wgpu_compute::kernels::k14_q6k_mmq::{K14Call, K14Pipeline};
use wgpu_compute::kernels::k1_q4k_gemv::{K1Call, K1Pipeline, K1Weights, ACT_NONE};
use wgpu_compute::kernels::k2_q6k_gemv::{K2Call, K2Pipeline, K2Weights};
use wgpu_compute::{DetectPolicy, WgpuCompute, WgpuError};

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

/// One encodable dispatch, whichever kernel built it; the
/// bench times compositions, not types.
enum BenchCall {
    K1(K1Call),
    K2(K2Call),
    K10(K10Call),
    K11(K11Call),
    K12(K12QuantCall),
    K13(K13Call),
    K14(K14Call),
}

impl BenchCall {
    fn encode_into(&self, pass: &mut wgpu::ComputePass<'_>) {
        match self {
            BenchCall::K1(c) => c.encode_into(pass),
            BenchCall::K2(c) => c.encode_into(pass),
            BenchCall::K10(c) => c.encode_into(pass),
            BenchCall::K11(c) => c.encode_into(pass),
            BenchCall::K12(c) => c.encode_into(pass),
            BenchCall::K13(c) => c.encode_into(pass),
            BenchCall::K14(c) => c.encode_into(pass),
        }
    }
}

fn time_calls(ctx: &wgpu_compute::GpuContext, calls: &[BenchCall], warmup: u32, iters: u32) -> f64 {
    let run_pass = |ctx: &wgpu_compute::GpuContext| {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mmq bench pass"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mmq bench"),
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
    let batch: u32 = arg(&args, "--batch")
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let iters: u32 = arg(&args, "--iters")
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let warmup: u32 = arg(&args, "--warmup")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let budget: u64 = arg(&args, "--budget-mib")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(512)
        * 1024
        * 1024;

    let file = match GgufFile::open(Path::new(&model)) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("cannot open {model}: {e}");
            return ExitCode::from(2);
        }
    };
    // Same selection as k10_bench: the largest tensor of
    // each family first, then more under the budget.
    let mut by_size: Vec<(String, GgmlType, u64, usize, usize)> = file
        .tensors()
        .iter()
        .filter(|t| matches!(t.ggml_type(), Some(GgmlType::Q4K | GgmlType::Q6K)))
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
        let new_family = !seen_families.contains(&cand.1);
        if !new_family && chosen.len() >= 3 {
            continue;
        }
        if total + cand.2 > budget {
            continue;
        }
        total += cand.2;
        seen_families.push(cand.1);
        chosen.push(cand.clone());
        if chosen.len() >= 4 {
            break;
        }
    }
    if chosen.is_empty() {
        eprintln!("no Q4_K/Q6_K tensor fits the budget");
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
    println!("batch: {batch}, iters {iters} (+{warmup} warmup)");

    let k1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let k2 = K2Pipeline::new(&ctx.device).expect("k2 pipeline");
    let k10 = K10Pipeline::with_shape_act(&ctx.device, GemmShape::new(32, 128), ACT_NONE)
        .expect("k10 t32x128");
    let k11 = K11Pipeline::with_shape_act(&ctx.device, GemmShape::new(32, 128), ACT_NONE)
        .expect("k11 t32x128");
    let k12 = K12QuantPipeline::new(&ctx.device, ACT_NONE).expect("k12 pipeline");
    let mut mmq4: Vec<(MmqShape, K13Pipeline)> = Vec::new();
    let mut mmq6: Vec<(MmqShape, K14Pipeline)> = Vec::new();
    for shape in [MmqShape::new(32), MmqShape::new(64)] {
        match K13Pipeline::with_shape(&ctx.device, shape) {
            Ok(p) => mmq4.push((shape, p)),
            Err(e) => println!("k13 {} unavailable: {e}", shape.name()),
        }
        match K14Pipeline::with_shape(&ctx.device, shape) {
            Ok(p) => mmq6.push((shape, p)),
            Err(e) => println!("k14 {} unavailable: {e}", shape.name()),
        }
    }

    let mut keep_q4: Vec<K1Weights> = Vec::new();
    let mut keep_q6: Vec<K2Weights> = Vec::new();
    for (name, ty, bytes, inn, out) in &chosen {
        let raw = file.tensor_bytes(name).expect("tensor bytes");
        let input = lcg_input(7, batch as usize * inn);
        let in_buf = ctx.upload_f32("bench input", &input);
        let out_buf = ctx.scratch_buffer("bench output", (batch as usize * out * 4) as u64);
        let qs_buf = ctx.scratch_buffer(
            "bench qs",
            (wgpu_compute::kernels::k12_q8_quant::qs_words(batch as usize, *inn) * 4) as u64,
        );
        let ds_buf = ctx.scratch_buffer(
            "bench ds",
            (wgpu_compute::kernels::k12_q8_quant::ds_vec4s(batch as usize, *inn) * 16) as u64,
        );
        let quant = BenchCall::K12(
            k12.prepare(ctx, &in_buf, &qs_buf, &ds_buf, batch, *inn)
                .expect("k12 prepare"),
        );
        println!(
            "\n{name} ({ty:?}, {inn}x{out}, {:.1} MB)",
            *bytes as f64 / 1e6
        );
        let report = |label: &str, secs: f64| {
            let rows_s = f64::from(batch) / secs;
            let stream = *bytes as f64 / secs / 1e9;
            println!(
                "  {label:<22} {:>9.3} ms  {:>9.0} tok/s  {:>7.1} GB/s stream",
                secs * 1e3,
                rows_s,
                stream
            );
        };
        match ty {
            GgmlType::Q4K => {
                let w = k1.upload(ctx, name, raw, *inn, *out).expect("k1 upload");
                let gemv = [BenchCall::K1(
                    k1.prepare(ctx, &w, &in_buf, &out_buf, batch)
                        .expect("k1 prepare"),
                )];
                report("gemv replay", time_calls(ctx, &gemv, warmup, iters));
                let gemm = [BenchCall::K10(
                    k10.prepare(ctx, &w, &in_buf, &out_buf, batch)
                        .expect("k10 prepare"),
                )];
                report("gemm t32x128", time_calls(ctx, &gemm, warmup, iters));
                report("k12 quantize", time_calls(ctx, &[quant], warmup, iters));
                for (shape, mmq) in &mmq4 {
                    let prod = [BenchCall::K13(
                        mmq.prepare(ctx, &w, &qs_buf, &ds_buf, &out_buf, batch)
                            .expect("k13 prepare"),
                    )];
                    let prod_s = time_calls(ctx, &prod, warmup, iters);
                    report(&format!("mmq {} product", shape.name()), prod_s);
                    let quant2 = BenchCall::K12(
                        k12.prepare(ctx, &in_buf, &qs_buf, &ds_buf, batch, *inn)
                            .expect("k12 prepare"),
                    );
                    let total = [
                        quant2,
                        BenchCall::K13(
                            mmq.prepare(ctx, &w, &qs_buf, &ds_buf, &out_buf, batch)
                                .expect("k13 prepare"),
                        ),
                    ];
                    let total_s = time_calls(ctx, &total, warmup, iters);
                    report(&format!("mmq {} + quantize", shape.name()), total_s);
                }
                keep_q4.push(w);
            }
            GgmlType::Q6K => {
                let w = k2.upload(ctx, name, raw, *inn, *out).expect("k2 upload");
                let gemv = [BenchCall::K2(
                    k2.prepare(ctx, &w, &in_buf, &out_buf, batch)
                        .expect("k2 prepare"),
                )];
                report("gemv replay", time_calls(ctx, &gemv, warmup, iters));
                let gemm = [BenchCall::K11(
                    k11.prepare(ctx, &w, &in_buf, &out_buf, batch)
                        .expect("k11 prepare"),
                )];
                report("gemm t32x128", time_calls(ctx, &gemm, warmup, iters));
                report("k12 quantize", time_calls(ctx, &[quant], warmup, iters));
                for (shape, mmq) in &mmq6 {
                    let prod = [BenchCall::K14(
                        mmq.prepare(ctx, &w, &qs_buf, &ds_buf, &out_buf, batch)
                            .expect("k14 prepare"),
                    )];
                    let prod_s = time_calls(ctx, &prod, warmup, iters);
                    report(&format!("mmq {} product", shape.name()), prod_s);
                    let quant2 = BenchCall::K12(
                        k12.prepare(ctx, &in_buf, &qs_buf, &ds_buf, batch, *inn)
                            .expect("k12 prepare"),
                    );
                    let total = [
                        quant2,
                        BenchCall::K14(
                            mmq.prepare(ctx, &w, &qs_buf, &ds_buf, &out_buf, batch)
                                .expect("k14 prepare"),
                        ),
                    ];
                    let total_s = time_calls(ctx, &total, warmup, iters);
                    report(&format!("mmq {} + quantize", shape.name()), total_s);
                }
                keep_q6.push(w);
            }
            _ => unreachable!("selection filtered to Q4_K/Q6_K"),
        }
    }
    let _ = (keep_q4, keep_q6);
    ExitCode::SUCCESS
}
