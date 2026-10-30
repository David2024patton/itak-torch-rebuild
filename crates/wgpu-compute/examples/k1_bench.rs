//! K1 streaming bench (BACKEND.md Section 7, tier 1 shape):
//! hold a set of the real file's largest Q4_K tensors resident,
//! dispatch the GEMV over all of them per iteration, and report
//! the effective streaming bandwidth against llama.cpp's measured
//! 708 GB/s and the 637 GB/s ash decision trigger.
//!
//! Usage:
//!   k1_bench [--model PATH] [--budget-mib N] [--iters N] [--warmup N]
//!
//! The resident set is capped by --budget-mib because this bench
//! runs beside whatever already holds VRAM on the card (on the
//! server, the production llama-server). The default 160 MiB is
//! sized for that coexistence and is roughly twice the card's
//! 96 MiB Infinity Cache, so the stream genuinely comes from
//! VRAM; the tool prints the card's VRAM state it found so the
//! report carries its own constraint. A full-model residency run
//! (14.89 GiB) needs the card to itself and is a separate,
//! scheduled measurement.

use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use gguf::{GgmlType, GgufFile};
use wgpu_compute::kernels::k1_q4k_gemv::{K1Call, K1Pipeline, K1Weights};
use wgpu_compute::{DetectPolicy, WgpuCompute, WgpuError};

/// llama.cpp Vulkan's measured effective bandwidth on this card
/// (BACKEND.md Section 1): the number to match.
const LLAMA_CPP_GBS: f64 = 708.0;
/// The ash decision trigger (BACKEND.md Section 8): below 90
/// percent of llama.cpp's number after tuning, evaluate raw
/// Vulkan.
const ASH_TRIGGER_GBS: f64 = 637.0;

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

/// Deterministic pseudo-random f32s in [-0.5, 0.5) via a 64-bit
/// LCG, matching `k1_check`.
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

/// Best-effort VRAM accounting from sysfs, printed so the bench
/// report states the card's occupancy it ran under.
fn vram_state() -> Option<String> {
    // Only the discrete card exposes mem_info files; skip every
    // card that does not rather than giving up at the first one.
    for card in 0..8 {
        let base = format!("/sys/class/drm/card{card}/device");
        let Ok(total) = std::fs::read_to_string(format!("{base}/mem_info_vram_total")) else {
            continue;
        };
        let Ok(used) = std::fs::read_to_string(format!("{base}/mem_info_vram_used")) else {
            continue;
        };
        let (Ok(total), Ok(used)) = (total.trim().parse::<u64>(), used.trim().parse::<u64>())
        else {
            continue;
        };
        return Some(format!(
            "card{card} VRAM: {used} of {total} bytes in use ({} free)",
            total - used
        ));
    }
    None
}

struct Resident {
    // The weights handle is never read after setup; it is kept so
    // the buffer's residency is explicit in this tool's own
    // accounting, beside the bind group that references it.
    #[allow(dead_code)]
    weights: K1Weights,
    call: K1Call,
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let model = arg(&args, "--model").unwrap_or_else(|| {
        "/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf".to_string()
    });
    let budget_mib: u64 = arg(&args, "--budget-mib")
        .and_then(|s| s.parse().ok())
        .unwrap_or(160);
    let iters: u32 = arg(&args, "--iters")
        .and_then(|s| s.parse().ok())
        .unwrap_or(25);
    let warmup: u32 = arg(&args, "--warmup")
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let budget = budget_mib * 1024 * 1024;

    let file = match GgufFile::open(Path::new(&model)) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("cannot open {model}: {e}");
            return ExitCode::from(2);
        }
    };
    // Largest Q4_K tensors first, greedily packed under the
    // budget. Glimmer's Q4_K population is dominated by the
    // 74,760,192-byte FFN gate/up tensors.
    let mut candidates: Vec<(String, u64, usize, usize)> = file
        .tensors()
        .iter()
        .filter(|t| t.ggml_type() == Some(GgmlType::Q4K))
        .filter_map(|t| {
            let bytes = t.byte_len()?;
            let dims = t.dims();
            Some((
                t.name().to_string(),
                bytes,
                dims[0] as usize,
                dims[1] as usize,
            ))
        })
        .collect();
    candidates.sort_by_key(|c| std::cmp::Reverse(c.1));
    let mut chosen: Vec<(String, u64, usize, usize)> = Vec::new();
    let mut total_bytes: u64 = 0;
    for candidate in candidates {
        if total_bytes + candidate.1 > budget {
            continue;
        }
        total_bytes += candidate.1;
        chosen.push(candidate);
    }
    if chosen.is_empty() {
        eprintln!("no Q4_K tensor fits the {budget_mib} MiB budget");
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
    let pipeline = K1Pipeline::new(&ctx.device).expect("k1 pipeline");
    println!("adapter: {}", ctx.adapter.name);
    if let Some(state) = vram_state() {
        println!("{state}");
    }
    println!(
        "resident set: {} tensors, {} bytes streamed per iteration",
        chosen.len(),
        total_bytes
    );

    let mut resident: Vec<Resident> = Vec::new();
    for (name, _bytes, inn, out) in &chosen {
        let bytes = file.tensor_bytes(name).expect("tensor bytes");
        let weights = match pipeline.upload(ctx, name, bytes, *inn, *out) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("upload of {name} failed: {e}");
                return ExitCode::from(2);
            }
        };
        let input = lcg_input(7, *inn);
        let in_bytes: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bench input"),
            size: in_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&input_buffer, 0, &in_bytes);
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bench output"),
            size: (*out * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let call = match pipeline.prepare(ctx, &weights, &input_buffer, &output_buffer, 1) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("prepare for {name} failed: {e}");
                return ExitCode::from(2);
            }
        };
        println!("  {name}: {} bytes [{inn} x {out}]", weights.byte_len);
        resident.push(Resident { weights, call });
    }

    let run_pass = |ctx: &wgpu_compute::GpuContext| {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k1 bench pass"),
            });
        for r in &resident {
            r.call.encode(&mut encoder);
        }
        ctx.queue.submit([encoder.finish()]);
    };

    for _ in 0..warmup {
        run_pass(ctx);
    }
    if ctx
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .is_err()
    {
        eprintln!("device poll failed after warmup");
        return ExitCode::from(2);
    }
    let start = Instant::now();
    for _ in 0..iters {
        run_pass(ctx);
    }
    if ctx
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .is_err()
    {
        eprintln!("device poll failed after timed loop");
        return ExitCode::from(2);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let streamed = total_bytes as f64 * f64::from(iters);
    let gbs = streamed / elapsed / 1e9;
    println!(
        "iters {iters} (+{warmup} warmup): {streamed:.0} bytes in {elapsed:.3} s = {gbs:.1} GB/s"
    );
    println!("reference: llama.cpp Vulkan {LLAMA_CPP_GBS:.0} GB/s; ash trigger {ASH_TRIGGER_GBS:.0} GB/s");
    if gbs >= LLAMA_CPP_GBS {
        println!(
            "verdict: at or above llama.cpp's measured bandwidth; wgpu stays, no ash question"
        );
    } else if gbs >= ASH_TRIGGER_GBS {
        println!(
            "verdict: above the ash trigger ({:.1}% of llama.cpp); wgpu stays per BACKEND.md Section 8",
            100.0 * gbs / LLAMA_CPP_GBS
        );
    } else {
        println!(
            "verdict: BELOW the ash trigger ({:.1}% of llama.cpp); BACKEND.md Section 8's ash evaluation trigger fires",
            100.0 * gbs / LLAMA_CPP_GBS
        );
    }
    ExitCode::SUCCESS
}
