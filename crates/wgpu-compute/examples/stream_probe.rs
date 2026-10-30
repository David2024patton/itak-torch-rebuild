//! Tier 1 stream probe (BACKEND.md Section 7): what does the
//! wgpu stack itself stream at on this card, before any
//! dequantization math? It runs over the largest Q4_K tensor of
//! the real file, held resident.
//!
//! The probe uses K1's grid shape (one workgroup per output
//! row, lanes striding the row's words, LDS reduction) but sums
//! words instead of dequantizing, which isolates the grid and
//! the access pattern from the dequant arithmetic: on the server
//! it streamed at 746.9 GB/s where the first K1 shader managed
//! 148.6 GB/s, which is how the K1 gap was attributed to load
//! instructions inside the kernel rather than to the stack.
//!
//! Usage: stream_probe [--model PATH] [--iters N]
//!
//! This is a measurement tool for the backend's bench record, not
//! a kernel: its outputs are checksums nobody reads.

use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use gguf::{GgmlType, GgufFile};
use wgpu_compute::{DetectPolicy, WgpuCompute, WgpuError};

const ROW_SHADER: &str = r#"
struct Params {
    words_per_row: u32,
    out_rows: u32,
    grid_x: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> partials: array<u32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let flat = wid.y * params.grid_x + wid.x;
    if (flat >= params.out_rows) {
        return;
    }
    let lane = lid.x;
    let row0 = flat * params.words_per_row;
    var acc = 0u;
    for (var i = lane; i < params.words_per_row; i += 128u) {
        acc += weights[row0 + i];
    }
    partials[lane] = acc;
    workgroupBarrier();
    var stride = 64u;
    while (stride > 0u) {
        if (lane < stride) {
            partials[lane] += partials[lane + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    if (lane == 0u) {
        output[flat] = partials[0];
    }
}
"#;

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn grid(total: u64) -> (u32, u32) {
    let x = total.min(65_535) as u32;
    (x, total.div_ceil(u64::from(x)) as u32)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let model = arg(&args, "--model").unwrap_or_else(|| {
        "/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf".to_string()
    });
    let iters: u32 = arg(&args, "--iters")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);

    let file = match GgufFile::open(Path::new(&model)) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("cannot open {model}: {e}");
            return ExitCode::from(2);
        }
    };
    let mut tensors: Vec<(String, u64, usize, usize)> = file
        .tensors()
        .iter()
        .filter(|t| t.ggml_type() == Some(GgmlType::Q4K))
        .filter_map(|t| {
            Some((
                t.name().to_string(),
                t.byte_len()?,
                t.dims()[0] as usize,
                t.dims()[1] as usize,
            ))
        })
        .collect();
    tensors.sort_by_key(|c| std::cmp::Reverse(c.1));
    let (name, bytes_len, inn, out) = tensors[0].clone();
    let bytes = file.tensor_bytes(&name).expect("tensor bytes").to_vec();

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
    println!("tensor {name}: {bytes_len} bytes [{inn} x {out}]");

    let weights = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe weights"),
        size: bytes_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&weights, 0, &bytes);
    let output = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe output"),
        size: 202_048 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let shader = ROW_SHADER;
    let words_per_row = (bytes_len / 4 / out as u64) as u32;
    let mut params_bytes = Vec::new();
    params_bytes.extend_from_slice(&words_per_row.to_le_bytes());
    params_bytes.extend_from_slice(&(out as u32).to_le_bytes());
    let dispatch = grid(out as u64);
    params_bytes.extend_from_slice(&dispatch.0.to_le_bytes());
    params_bytes.extend_from_slice(&1u32.to_le_bytes()); // pad to 16 bytes

    let params = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe params"),
        size: params_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&params, 0, &params_bytes);
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stream probe shader"),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("stream probe pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("probe bind group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: weights.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params.as_entire_binding(),
            },
        ],
    });

    let run_pass = |ctx: &wgpu_compute::GpuContext| {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("probe pass"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("probe"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch.0, dispatch.1, 1);
        }
        ctx.queue.submit([encoder.finish()]);
    };

    for _ in 0..50 {
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
    let streamed = bytes_len as f64 * f64::from(iters);
    println!(
        "row mode: {iters} iters, {streamed:.0} bytes in {elapsed:.3} s = {:.1} GB/s",
        streamed / elapsed / 1e9
    );
    ExitCode::SUCCESS
}
