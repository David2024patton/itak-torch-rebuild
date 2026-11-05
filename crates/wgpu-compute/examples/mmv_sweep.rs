//! Small-batch sweep harness: every projection
//! variant the session's routing ladder can choose,
//! measured head to head over the verify band at the
//! real Glimmer shapes, one CSV per run.
//!
//! Each run drives single projections through
//! [`ProjBench`], which prepares through the session's
//! own `prepare_proj` ladder, so the variant measured
//! is the variant a forward would dispatch under the
//! process environment as it stands: shipped GEMV/MM
//! forms as control, K18 under
//! `WGPU_COMPUTE_SMALL_BATCH_MAX_ROWS`, the MMV
//! register-streaming forms under
//! `WGPU_COMPUTE_MMV_MAX_ROWS`, the chunked forms
//! under their env, wide MMQ under
//! `WGPU_COMPUTE_WIDE_MIN_ROWS`. The harness sets no
//! routing env itself; it records the active routing
//! env into every CSV row's `variant_env` column.
//! `bench/mmv-sweep.sh` runs the arm matrix.
//!
//! The cell matrix is the autotune table the research
//! plan calls for: quant family (Q4_K, Q5_K, Q6_K) x
//! shape (attention-sized 6656 out, FFN-up 19968 out,
//! KV-sized 256 out, all from hidden 6656; the KV
//! width is `GlimmerConfig::k_width()`, KV heads times
//! key length, 2 x 128) x rows {1..=9, 12, 16}.
//!
//! Measurement discipline: warmup dispatches are
//! excluded, every cell gets at least 20 timed
//! repetitions at the default settings, and the CSV
//! reports median and min. GPU time comes from the
//! diag timestamps system (one stamp around the whole
//! projection composition) when the context carries a
//! recorder (`WGPU_COMPUTE_DIAG=timing` and a device
//! that granted the timestamp features); otherwise
//! each repetition is bracketed on the wall clock
//! around submit plus poll, and the run says which
//! method it used on stdout and in its log. A cell
//! whose stamp records do not line up with its
//! repetition count is re-timed on the wall clock,
//! with the substitution printed.
//!
//! Correctness guard: every cell's variant output is
//! compared against the GEMV form's output for the
//! same inputs, prepared directly from the kernel
//! pipelines, before anything is timed. Agreement uses
//! the crate's own kernel-test rules
//! (tests/k18_small_batch.rs): a form that quantizes
//! its activation answers to the 5% quantization band
//! (`check_quant_band`), an f32 form to the tight
//! reassociation bound (`check_close`). A disagreeing
//! cell is marked `agree=0` in the CSV, never silently
//! timed as if it agreed.
//!
//! Usage:
//!   mmv_sweep --out PATH [--label NAME] [--reps N]
//!             [--warmup N] [--rows 1,2,7] [--quants Q4_K,Q6_K]
//!             [--outs 6656,19968,256]
//!   mmv_sweep --enumerate   (print the cell matrix, no GPU)

use std::fmt::Write as _;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use muse_glimmer::Dtype;
use wgpu_compute::kernels::k12_q8_quant::{ds_vec4s, qs_words};
use wgpu_compute::kernels::k1_q4k_gemv::{K1Pipeline, K1Weights};
use wgpu_compute::kernels::k2_q6k_gemv::{K2Pipeline, K2Weights};
use wgpu_compute::kernels::k3_q5k_gemv::{K3Pipeline, K3Weights};
use wgpu_compute::timestamps::DispatchMeta;
use wgpu_compute::{DetectPolicy, GpuContext, ProjBench, ProjBenchCall, WgpuCompute, WgpuError};

/// Input width of every swept projection: the model's
/// hidden size.
const IN: usize = 6656;

/// The swept shapes as (label, output width). The KV
/// width is the config's `k_width()` (KV heads times
/// key length, 2 x 128 = 256), read from
/// crates/muse-glimmer/src/config.rs, whose real-file
/// test pins it.
const SHAPES: [(&str, usize); 3] = [("attn", 6656), ("ffn", 19968), ("kv", 256)];

/// The row sweep: the whole verify band (a depth-1
/// round verifies 2 rows, depth 6 verifies 7), the
/// batch-9 cliff point from the NUM_COLS dose-response,
/// and the matrix threshold at 16.
const ROWS: [u32; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16];

/// Timed repetitions per cell at the default: the
/// floor the measurement discipline sets. A run below
/// it is a smoke, and says so.
const DEFAULT_REPS: u32 = 25;
const MIN_VERDICT_REPS: u32 = 20;
const DEFAULT_WARMUP: u32 = 3;

/// Routing env the harness watches and stamps into
/// its output. It sets none of them.
const ROUTING_VARS: [&str; 5] = [
    "WGPU_COMPUTE_SMALL_BATCH_MAX_ROWS",
    "WGPU_COMPUTE_MMV_MAX_ROWS",
    "WGPU_COMPUTE_WIDE_MIN_ROWS",
    "WGPU_COMPUTE_CHUNK_ROWS",
    "WGPU_COMPUTE_CHUNK_MAX_ROWS",
];

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

fn lcg_bytes(seed: u8, count: usize) -> Vec<u8> {
    let mut state = seed.max(1);
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(31).wrapping_add(17);
            state
        })
        .collect()
}

/// A synthetic Q4_K tensor's raw bytes (144-byte
/// blocks), the kernel tests' fixture shape: rotating
/// (d, dmin) f16 headers over LCG payloads
/// (tests/k18_small_batch.rs).
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
/// blocks, same fixture discipline as Q4_K).
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

/// Bytes of one synthetic tensor of `quant` at
/// `IN` x `out`: block geometry only, for the
/// enumeration listing. Run-mode weight bytes come
/// from the uploaded handle, not from here.
fn geometry_bytes(quant: Dtype, out: usize) -> usize {
    let block_bytes = match quant {
        Dtype::Q4K => 144,
        Dtype::Q5K => 176,
        Dtype::Q6K => 210,
        _ => 0,
    };
    IN / 256 * out * block_bytes
}

fn quant_name(quant: Dtype) -> &'static str {
    match quant {
        Dtype::Q4K => "Q4_K",
        Dtype::Q5K => "Q5_K",
        Dtype::Q6K => "Q6_K",
        _ => "other",
    }
}

/// The routing env stamp: the run's label plus every
/// watched routing variable that is set, or `env:none`
/// when none is. Semicolon-joined so the CSV column
/// never grows a comma.
fn variant_env_stamp(label: &str) -> String {
    let mut set: Vec<String> = Vec::new();
    for var in ROUTING_VARS {
        if let Ok(v) = std::env::var(var) {
            set.push(format!("{var}={v}"));
        }
    }
    if set.is_empty() {
        format!("{label}|env:none")
    } else {
        format!("{label}|{}", set.join(";"))
    }
}

/// The agreement rule between a variant output and
/// the GEMV control output, mirroring the crate's
/// kernel tests (tests/k18_small_batch.rs): a form
/// that quantizes its activation answers to the 5%
/// quantization band (`check_quant_band`), an f32 form
/// to the tight reassociation bound (`check_close`).
/// Returns (agrees, max abs diff).
fn agree(got: &[f32], want: &[f32], quantized: bool) -> (bool, f32) {
    let scale = want.iter().fold(0.0f32, |a, w| a.max(w.abs()));
    let max_diff = got
        .iter()
        .zip(want.iter())
        .fold(0.0f32, |a, (g, w)| a.max((g - w).abs()));
    if quantized {
        (max_diff / scale.max(1.0) <= 0.05, max_diff)
    } else {
        (max_diff <= 2e-3 + 5e-5 * scale, max_diff)
    }
}

fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// The GEMV control weights, one family at a time,
/// uploaded from the same synthetic bytes the routed
/// handle holds.
enum GemvWeights {
    Q4K(K1Weights),
    Q6K(K2Weights),
    Q5K(K3Weights),
}

/// Time one cell on the wall clock: warmup passes
/// untimed, then one submit-plus-poll per repetition.
/// Returns (median ms, min ms).
fn time_wall(ctx: &GpuContext, call: &ProjBenchCall, warmup: u32, reps: u32) -> (f64, f64) {
    let run_one = |ctx: &GpuContext| {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mmv sweep pass"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mmv sweep"),
                timestamp_writes: None,
            });
            call.encode_into(&mut pass);
        }
        ctx.queue.submit([encoder.finish()]);
    };
    for _ in 0..warmup {
        run_one(ctx);
    }
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll after warmup");
    let mut times: Vec<f64> = Vec::with_capacity(reps as usize);
    for _ in 0..reps {
        let start = Instant::now();
        run_one(ctx);
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll in timed loop");
        times.push(start.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (median(&times), times[0])
}

/// Time one cell with the diag timestamps system: one
/// stamp around the whole projection composition per
/// repetition, warmup passes unstamped. Returns
/// (median ms, min ms), or `None` when the records do
/// not line up with the repetition count (the caller
/// re-times on the wall clock and says so).
fn time_diag(
    ctx: &GpuContext,
    call: &ProjBenchCall,
    meta: DispatchMeta,
    warmup: u32,
    reps: u32,
) -> Option<(f64, f64)> {
    let recorder = ctx.diag_recorder()?;
    let dropped_before = recorder.dropped();
    let run_one = |ctx: &GpuContext, stamp: bool| {
        let stamp = if stamp {
            ctx.diag_stamp(meta.clone())
        } else {
            None
        };
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mmv sweep pass"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mmv sweep"),
                timestamp_writes: None,
            });
            let slot = stamp.as_ref().and_then(|s| s.begin(&mut pass));
            call.encode_into(&mut pass);
            if let (Some(s), Some(slot)) = (stamp.as_ref(), slot) {
                s.end(&mut pass, slot);
            }
        }
        ctx.queue.submit([encoder.finish()]);
    };
    for _ in 0..warmup {
        run_one(ctx, false);
    }
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll after warmup");
    for _ in 0..reps {
        run_one(ctx, true);
    }
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll after timed loop");
    recorder.drain(&ctx.device, &ctx.queue).expect("diag drain");
    if recorder.dropped() != dropped_before {
        return None;
    }
    let records = recorder.take_records();
    if records.len() != reps as usize {
        return None;
    }
    let mut times: Vec<f64> = records.iter().map(|r| r.duration_ns as f64 / 1e6).collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some((median(&times), times[0]))
}

fn parse_filter<T: std::str::FromStr>(raw: Option<String>) -> Option<Vec<T>> {
    raw.map(|s| {
        s.split(',')
            .filter_map(|p| p.trim().parse::<T>().ok())
            .collect()
    })
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let quants_all = [Dtype::Q4K, Dtype::Q5K, Dtype::Q6K];

    if args.iter().any(|a| a == "--enumerate") {
        println!("cell matrix: in={IN}, shapes (label, out) = {SHAPES:?}");
        println!("variant_env,kernel,quant,rows,out_cols,median_ms,min_ms,weight_bytes,gbps_median,agree");
        for quant in quants_all {
            for (_, out) in SHAPES {
                for rows in ROWS {
                    println!(
                        "enumerate,enumerate,{},{rows},{out},,,{},,",
                        quant_name(quant),
                        geometry_bytes(quant, out),
                    );
                }
            }
        }
        println!(
            "cells per run: {}",
            quants_all.len() * SHAPES.len() * ROWS.len()
        );
        return ExitCode::SUCCESS;
    }

    let out_path = match arg(&args, "--out") {
        Some(p) => p,
        None => {
            eprintln!("--out PATH is required in run mode (or pass --enumerate)");
            return ExitCode::from(2);
        }
    };
    let label = arg(&args, "--label").unwrap_or_else(|| "unlabeled".to_string());
    let reps: u32 = arg(&args, "--reps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_REPS);
    let warmup: u32 = arg(&args, "--warmup")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_WARMUP);
    let rows_filter = parse_filter::<u32>(arg(&args, "--rows"));
    let outs_filter = parse_filter::<usize>(arg(&args, "--outs"));
    let quants_filter: Option<Vec<String>> = arg(&args, "--quants").map(|s| {
        s.split(',')
            .map(|p| p.trim().to_string())
            .collect::<Vec<_>>()
    });
    let rows: Vec<u32> = ROWS
        .iter()
        .copied()
        .filter(|r| rows_filter.as_ref().is_none_or(|f| f.contains(r)))
        .collect();
    let shapes: Vec<(&str, usize)> = SHAPES
        .iter()
        .copied()
        .filter(|(_, o)| outs_filter.as_ref().is_none_or(|f| f.contains(o)))
        .collect();
    let quants: Vec<Dtype> = quants_all
        .iter()
        .copied()
        .filter(|q| {
            quants_filter
                .as_ref()
                .is_none_or(|f| f.iter().any(|n| n == quant_name(*q)))
        })
        .collect();
    if rows.is_empty() || shapes.is_empty() || quants.is_empty() {
        eprintln!("filters selected no cells");
        return ExitCode::from(2);
    }
    if reps < MIN_VERDICT_REPS {
        println!(
            "note: reps {reps} is below the {MIN_VERDICT_REPS}-repetition verdict floor; this run is a smoke, not a verdict run"
        );
    }

    let stamp = variant_env_stamp(&label);
    println!("variant_env: {stamp}");
    println!("reps {reps} (+{warmup} warmup), rows {rows:?}");

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
    let timing_method = if ctx.diag_recorder().is_some() {
        "diag_timestamps"
    } else {
        "wall_clock"
    };
    println!("timing_method: {timing_method}");

    let k1 = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    let k2 = K2Pipeline::new(&ctx.device).expect("k2 pipeline");
    let k3 = K3Pipeline::new(&ctx.device).expect("k3 pipeline");

    let max_rows = *rows.iter().max().unwrap() as usize;
    let input = lcg_input(7, max_rows * IN);
    let in_buf = ctx.upload_f32("sweep input", &input);
    let qs_buf = ctx.scratch_buffer("sweep qs", (qs_words(max_rows, IN) * 4) as u64);
    let ds_buf = ctx.scratch_buffer("sweep ds", (ds_vec4s(max_rows, IN) * 16) as u64);

    let mut csv = String::from(
        "variant_env,kernel,quant,rows,out_cols,median_ms,min_ms,weight_bytes,gbps_median,agree\n",
    );
    let mut disagreements = 0u32;

    for quant in &quants {
        for (shape_label, out) in &shapes {
            let name = format!("sweep {} {shape_label}", quant_name(*quant));
            let raw = match quant {
                Dtype::Q4K => synthetic_q4k(IN, *out),
                Dtype::Q5K => synthetic_q5k(IN, *out),
                Dtype::Q6K => synthetic_q6k(IN, *out),
                _ => unreachable!("quants filtered to the three K families"),
            };
            let bench = ProjBench::upload(&backend, &name, *quant, &raw, IN, *out)
                .expect("proj bench upload");
            let gemv_w = match quant {
                Dtype::Q4K => {
                    GemvWeights::Q4K(k1.upload(ctx, &name, &raw, IN, *out).expect("k1 upload"))
                }
                Dtype::Q6K => {
                    GemvWeights::Q6K(k2.upload(ctx, &name, &raw, IN, *out).expect("k2 upload"))
                }
                Dtype::Q5K => {
                    GemvWeights::Q5K(k3.upload(ctx, &name, &raw, IN, *out).expect("k3 upload"))
                }
                _ => unreachable!("quants filtered to the three K families"),
            };
            let weight_bytes = bench.byte_len() as u64;
            let out_variant = ctx.scratch_buffer("sweep out variant", (max_rows * out * 4) as u64);
            let out_control = ctx.scratch_buffer("sweep out control", (max_rows * out * 4) as u64);

            for rows_n in &rows {
                // The GEMV control output for this cell,
                // prepared directly from the kernel
                // pipelines, before anything is timed.
                let run_control = |ctx: &GpuContext| {
                    let mut encoder =
                        ctx.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("mmv sweep control pass"),
                            });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("mmv sweep control"),
                            timestamp_writes: None,
                        });
                        match &gemv_w {
                            GemvWeights::Q4K(w) => k1
                                .prepare(ctx, w, &in_buf, &out_control, *rows_n)
                                .expect("k1 prepare")
                                .encode_into(&mut pass),
                            GemvWeights::Q6K(w) => k2
                                .prepare(ctx, w, &in_buf, &out_control, *rows_n)
                                .expect("k2 prepare")
                                .encode_into(&mut pass),
                            GemvWeights::Q5K(w) => k3
                                .prepare(ctx, w, &in_buf, &out_control, *rows_n)
                                .expect("k3 prepare")
                                .encode_into(&mut pass),
                        }
                    }
                    ctx.queue.submit([encoder.finish()]);
                };
                run_control(ctx);
                ctx.device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .expect("poll after control");
                let control = ctx
                    .read_f32(&out_control, *rows_n as usize * out)
                    .expect("control readback");

                // The routed variant for this cell.
                let call = bench
                    .prepare(
                        &backend,
                        &in_buf,
                        &qs_buf,
                        &ds_buf,
                        &out_variant,
                        *rows_n as usize,
                    )
                    .expect("proj bench prepare");
                {
                    let mut encoder =
                        ctx.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("mmv sweep check pass"),
                            });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("mmv sweep check"),
                            timestamp_writes: None,
                        });
                        call.encode_into(&mut pass);
                    }
                    ctx.queue.submit([encoder.finish()]);
                }
                ctx.device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .expect("poll after check");
                let got = ctx
                    .read_f32(&out_variant, *rows_n as usize * out)
                    .expect("variant readback");
                let (agrees, max_diff) = agree(&got, &control, call.quantized());
                if !agrees {
                    disagreements += 1;
                    eprintln!(
                        "DISAGREE {} {shape_label} rows {rows_n}: kernel {} max abs diff {max_diff:.4} (quantized form: {})",
                        quant_name(*quant),
                        call.kernel_name(),
                        call.quantized(),
                    );
                }

                let meta = DispatchMeta {
                    kernel: call.kernel_name(),
                    rows: u64::from(*rows_n),
                    cols: IN as u64,
                    quant: quant_name(*quant),
                    weight_bytes,
                    shape: format!("sweep {shape_label}"),
                };
                let (median_ms, min_ms) = if timing_method == "diag_timestamps" {
                    match time_diag(ctx, &call, meta, warmup, reps) {
                        Some(t) => t,
                        None => {
                            eprintln!(
                                "note: diag records did not line up for {} {shape_label} rows {rows_n}; re-timed on the wall clock",
                                quant_name(*quant),
                            );
                            time_wall(ctx, &call, warmup, reps)
                        }
                    }
                } else {
                    time_wall(ctx, &call, warmup, reps)
                };
                let gbps = if median_ms > 0.0 {
                    weight_bytes as f64 / (median_ms / 1e3) / 1e9
                } else {
                    0.0
                };
                writeln!(
                    csv,
                    "{stamp},{},{},{rows_n},{out},{median_ms:.4},{min_ms:.4},{weight_bytes},{gbps:.2},{}",
                    call.kernel_name(),
                    quant_name(*quant),
                    u8::from(agrees),
                )
                .expect("csv line");
                println!(
                    "{} {shape_label} rows {rows_n:>2}: {:<18} median {median_ms:>9.4} ms  min {min_ms:>9.4} ms  {gbps:>8.2} GB/s  agree {}",
                    quant_name(*quant),
                    call.kernel_name(),
                    u8::from(agrees),
                );
            }
        }
    }

    std::fs::write(Path::new(&out_path), &csv).expect("write csv");
    println!("wrote {out_path} ({disagreements} disagreeing cells)");
    if disagreements > 0 {
        eprintln!(
            "warning: {disagreements} cells disagreed with the GEMV control and are marked agree=0"
        );
    }
    ExitCode::SUCCESS
}
