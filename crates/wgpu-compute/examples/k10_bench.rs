//! K10/K11 prefill GEMM bench: hold the real file's largest
//! Q4_K and Q6_K tensors resident, run each projection shape
//! at a prefill batch through the GEMV kernels and through
//! the GEMM variants, and report per-tensor pass time,
//! tokens/s (batch rows per second), and the effective weight
//! stream. The GEMV at batch 1 doubles as the window canary:
//! its GB/s is the number BACKEND.md's healthy windows put at
//! ~816 (Q4_K) and ~832 (Q6_K), so a slow window shows itself
//! before any GEMM verdict is read from it.
//!
//! Usage:
//!   k10_bench [--model PATH] [--batch N] [--iters N]
//!             [--warmup N] [--budget-mib N]
//!
//! The resident set is capped by --budget-mib for the same
//! coexistence reason as k1_bench; the default 512 MiB fits
//! the largest tensors of both families (an FFN gate at
//! 74.8 MB, an FFN down at 109.4 MB) with room to spare.

use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use gguf::{GgmlType, GgufFile};
use wgpu_compute::kernels::k10_q4k_gemm::{GemmShape, K10Call, K10Pipeline};
use wgpu_compute::kernels::k11_q6k_gemm::{K11Call, K11Pipeline};
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

/// One encodable projection dispatch, whichever kernel built
/// it; the bench times compositions, not types.
enum BenchCall {
    K1(K1Call),
    K2(K2Call),
    K10(K10Call),
    K11(K11Call),
}

impl BenchCall {
    fn encode_into(&self, pass: &mut wgpu::ComputePass<'_>) {
        match self {
            BenchCall::K1(c) => c.encode_into(pass),
            BenchCall::K2(c) => c.encode_into(pass),
            BenchCall::K10(c) => c.encode_into(pass),
            BenchCall::K11(c) => c.encode_into(pass),
        }
    }
}

struct TensorBench {
    name: String,
    family: &'static str,
    bytes: u64,
    inn: usize,
    out: usize,
    calls: Vec<(String, BenchCall)>,
}

fn time_calls(
    ctx: &wgpu_compute::GpuContext,
    calls: &[BenchCall],
    warmup: u32,
    iters: u32,
) -> f64 {
    let run_pass = |ctx: &wgpu_compute::GpuContext| {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k10 bench pass"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("k10 bench"),
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
    // The largest tensor of each family, plus the largest
    // Q4_K tensor whose output is attention-width (the QKV
    // batch's shape class), greedily under the budget.
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
    let gemm_shapes = [
        GemmShape::new(64, 256),
        GemmShape::new(64, 128),
        GemmShape::new(32, 128),
    ];
    let mut k10s: Vec<(String, K10Pipeline)> = Vec::new();
    let mut k11s: Vec<(String, K11Pipeline)> = Vec::new();
    for shape in gemm_shapes {
        match K10Pipeline::with_shape_act(&ctx.device, shape, ACT_NONE) {
            Ok(p) => k10s.push((shape.name(), p)),
            Err(e) => println!("k10 {} unavailable: {e}", shape.name()),
        }
        match K11Pipeline::with_shape_act(&ctx.device, shape, ACT_NONE) {
            Ok(p) => k11s.push((shape.name(), p)),
            Err(e) => println!("k11 {} unavailable: {e}", shape.name()),
        }
    }

    let mut benches: Vec<TensorBench> = Vec::new();
    // Keep the weight handles alive beside their calls.
    let mut keep_q4: Vec<K1Weights> = Vec::new();
    let mut keep_q6: Vec<K2Weights> = Vec::new();
    for (name, ty, bytes, inn, out) in &chosen {
        let raw = file.tensor_bytes(name).expect("tensor bytes");
        let input = lcg_input(7, batch as usize * inn);
        let in_buf = ctx.upload_f32("bench input", &input);
        let out_buf = ctx.scratch_buffer("bench output", (batch as usize * out * 4) as u64);
        let mut calls: Vec<(String, BenchCall)> = Vec::new();
        match ty {
            GgmlType::Q4K => {
                let w = k1.upload(ctx, name, raw, *inn, *out).expect("k1 upload");
                calls.push((
                    "gemv".to_string(),
                    BenchCall::K1(
                        k1.prepare(ctx, &w, &in_buf, &out_buf, batch)
                            .expect("k1 prepare"),
                    ),
                ));
                for (label, g) in &k10s {
                    calls.push((
                        format!("gemm {label}"),
                        BenchCall::K10(
                            g.prepare(ctx, &w, &in_buf, &out_buf, batch)
                                .expect("k10 prepare"),
                        ),
                    ));
                }
                keep_q4.push(w);
            }
            GgmlType::Q6K => {
                let w = k2.upload(ctx, name, raw, *inn, *out).expect("k2 upload");
                calls.push((
                    "gemv".to_string(),
                    BenchCall::K2(
                        k2.prepare(ctx, &w, &in_buf, &out_buf, batch)
                            .expect("k2 prepare"),
                    ),
                ));
                for (label, g) in &k11s {
                    calls.push((
                        format!("gemm {label}"),
                        BenchCall::K11(
                            g.prepare(ctx, &w, &in_buf, &out_buf, batch)
                                .expect("k11 prepare"),
                        ),
                    ));
                }
                keep_q6.push(w);
            }
            _ => unreachable!(),
        }
        benches.push(TensorBench {
            name: name.clone(),
            family: if *ty == GgmlType::Q4K { "Q4_K" } else { "Q6_K" },
            bytes: *bytes,
            inn: *inn,
            out: *out,
            calls,
        });
    }
    // The weight handles stay in scope until every timed pass
    // has run (their buffers back the calls' bind groups).
    let _residency = (keep_q4, keep_q6);

    for bench in &benches {
        println!(
            "\n{} [{} x {}] {}: {} bytes",
            bench.name, bench.inn, bench.out, bench.family, bench.bytes
        );
        for (label, call) in &bench.calls {
            let secs = time_calls(ctx, std::slice::from_ref(call), warmup, iters);
            let ms = secs * 1e3;
            let tok_s = f64::from(batch) / secs;
            let stream_once = bench.bytes as f64 / secs / 1e9;
            let stream_per_row = stream_once * f64::from(batch);
            println!(
                "  {label:<12} {ms:9.3} ms/pass  {tok_s:9.1} tok/s  stream {stream_once:8.1} GB/s (x1)  {stream_per_row:9.1} GB/s (x{batch})"
            );
        }
    }
    ExitCode::SUCCESS
}
