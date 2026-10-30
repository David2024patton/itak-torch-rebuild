//! K3 streaming bench (BACKEND.md Section 7, tier 1 shape):
//! hold the real file's Q5_K output head resident, dispatch the
//! GEMV per iteration, and report the effective streaming
//! bandwidth against llama.cpp's measured 708 GB/s and the
//! 637 GB/s ash decision trigger.
//!
//! Usage:
//!   k3_bench [--model PATH] [--budget-mib N] [--iters N] [--warmup N]
//!            [--variant NAME] [--wg N]
//!
//! `--variant` selects the K3 shape (see `K3Shape::from_name`:
//! `r<rows>[sg]`; `shipped` is whatever `K3Pipeline::new` would
//! build on this device). `--wg` overrides the workgroup size.
//!
//! Glimmer has exactly one Q5_K tensor, the 202,048-row head at
//! 924,571,648 bytes. When the budget cannot hold it whole, the
//! bench streams the leading whole rows that fit (rows are
//! contiguous: 26 blocks of 176 bytes each) and prints the slice
//! geometry loudly, so a sliced number is never mistaken for a
//! whole-head number. The default 160 MiB budget slices; run
//! with --budget-mib 1024 or more to hold the whole head when
//! the card has the room.

use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use gguf::{GgmlType, GgufFile};
use wgpu_compute::kernels::k3_q5k_gemv::{K3Call, K3Pipeline, K3Shape, K3Weights};
use wgpu_compute::{DetectPolicy, WgpuCompute, WgpuError};

/// llama.cpp Vulkan's measured effective bandwidth on this card
/// (BACKEND.md Section 1): the number to match.
const LLAMA_CPP_GBS: f64 = 708.0;
/// The ash decision trigger (BACKEND.md Section 8): below 90
/// percent of llama.cpp's number after tuning, evaluate raw
/// Vulkan.
const ASH_TRIGGER_GBS: f64 = 637.0;
/// Bytes per head row: 6,656 inputs is 26 Q5_K blocks of 176.
const ROW_BYTES: u64 = 26 * 176;

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

/// Deterministic pseudo-random f32s in [-0.5, 0.5) via a 64-bit
/// LCG, matching `k3_check`.
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
    weights: K3Weights,
    call: K3Call,
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
    let variant = arg(&args, "--variant").unwrap_or_else(|| "shipped".to_string());
    let wg_override: Option<u32> = arg(&args, "--wg").and_then(|s| s.parse().ok());
    let budget = budget_mib * 1024 * 1024;

    let file = match GgufFile::open(Path::new(&model)) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("cannot open {model}: {e}");
            return ExitCode::from(2);
        }
    };
    // The one Q5_K tensor: the output head.
    let head = file
        .tensors()
        .iter()
        .find(|t| t.ggml_type() == Some(GgmlType::Q5K));
    let Some(head) = head else {
        eprintln!("no Q5_K tensor in {model}");
        return ExitCode::from(2);
    };
    let name = head.name().to_string();
    let dims = head.dims();
    let (inn, full_rows) = (dims[0] as usize, dims[1] as usize);
    let full_bytes = match head.byte_len() {
        Some(b) => b,
        None => {
            eprintln!("tensor {name} has no byte length");
            return ExitCode::from(2);
        }
    };
    if full_bytes != full_rows as u64 * ROW_BYTES || inn != 6_656 {
        eprintln!(
            "tensor {name} geometry [{inn} x {full_rows}] at {full_bytes} bytes is not the expected head shape"
        );
        return ExitCode::from(2);
    }
    let rows = if full_bytes <= budget {
        full_rows
    } else {
        (budget / ROW_BYTES) as usize
    };
    if rows == 0 {
        eprintln!("budget {budget_mib} MiB cannot hold even one head row");
        return ExitCode::from(2);
    }
    let total_bytes = rows as u64 * ROW_BYTES;

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
    let mut shape = if variant == "shipped" {
        wgpu_compute::kernels::k3_q5k_gemv::shipped_shape(ctx.device.features())
    } else {
        match K3Shape::from_name(&variant) {
            Some(s) => s,
            None => {
                eprintln!("unknown variant {variant:?} (expected r<rows>[sg] or shipped)");
                return ExitCode::from(2);
            }
        }
    };
    if let Some(wg) = wg_override {
        shape.workgroup = wg;
    }
    let pipeline = match K3Pipeline::with_shape(&ctx.device, shape) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("k3 pipeline for shape {} failed: {e}", shape.name());
            return ExitCode::from(2);
        }
    };
    println!("adapter: {}", ctx.adapter.name);
    println!(
        "shape: {} (rows {}, workgroup {}, subgroup {})",
        shape.name(),
        shape.rows,
        shape.workgroup,
        shape.subgroup
    );
    if let Some(state) = vram_state() {
        println!("{state}");
    }
    if rows == full_rows {
        println!("resident set: the whole head {name}, {total_bytes} bytes streamed per iteration [{inn} x {rows}]");
    } else {
        println!(
            "resident set: SLICE of {name}: leading {rows} of {full_rows} rows, {total_bytes} bytes streamed per iteration [{inn} x {rows}] (budget {budget_mib} MiB cannot hold the whole {full_bytes}-byte head)"
        );
    }

    let all_bytes = match file.tensor_bytes(&name) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("cannot read bytes of {name}: {e}");
            return ExitCode::from(2);
        }
    };
    let slice = &all_bytes[..total_bytes as usize];
    let weights = match pipeline.upload(ctx, &name, slice, inn, rows) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("upload of {name} failed: {e}");
            return ExitCode::from(2);
        }
    };
    let input = lcg_input(7, inn);
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
        size: (rows * 4) as u64,
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
    let resident = Resident { weights, call };

    let run_pass = |ctx: &wgpu_compute::GpuContext| {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k3 bench pass"),
            });
        resident.call.encode(&mut encoder);
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
