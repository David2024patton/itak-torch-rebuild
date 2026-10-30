//! Integer-dot throughput probe: the evidence for whether
//! WGSL's packed i8 dot actually reaches the hardware's
//! integer dot instruction on this backend (RADV's
//! `v_dot4_i32_i8`), rather than naga decomposing it into
//! scalar integer multiplies.
//!
//! Three register-loop kernels, identical launch geometry
//! and trip counts, so their rates are directly comparable:
//!
//! - `packed`: one `dot4I8Packed` per step (4 int MACs).
//! - `unpack`: `unpack4xI8` into a `vec4<i32>` and a vector
//!   dot per step (the same 4 int MACs, if the compiler keeps
//!   them vectorized; if it scalarizes them, this collapses).
//! - `fp32`: one fused multiply-add per step (1 FMA = 2
//!   FLOPs), the FP32 reference rate.
//!
//! A packed rate near 4x the fp32 FMA rate (per step) means
//! the integer dot instruction is doing the work: the
//! 7900 XTX issues one DP4A per lane per clock alongside the
//! FP32 pipes. A packed rate near the fp32 rate means the
//! "integer path" is scalar multiplies in disguise, and the
//! MMQ kernels' measured rate must be read with that in
//! mind. The loop bodies mix the step index into the inputs
//! so nothing constant-folds, and each thread writes one
//! word at the end so nothing dead-code-eliminates.
//!
//! Usage: dp4a_probe [--iters N] [--steps N]

use std::process::ExitCode;
use std::time::Instant;

use wgpu_compute::{DetectPolicy, WgpuCompute, WgpuError};

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

const SHADER: &str = "
@group(0) @binding(0) var<storage, read_write> sink: array<i32, 1048576>;

const STEPS: u32 = __STEPS__u;

@compute @workgroup_size(256)
fn packed_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    var a: u32 = 0x01020304u + idx;
    var b: u32 = 0x05060708u;
    var acc: i32 = 0;
    for (var i = 0u; i < STEPS; i++) {
        a = a * 1664525u + 1013904223u;
        acc += dot4I8Packed(a, b);
    }
    sink[idx] = acc;
}

@compute @workgroup_size(256)
fn unpack_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    var a: u32 = 0x01020304u + idx;
    let b = vec4<i32>(5, 6, 7, 8);
    var acc: i32 = 0;
    for (var i = 0u; i < STEPS; i++) {
        a = a * 1664525u + 1013904223u;
        acc += dot(unpack4xI8(a), b);
    }
    sink[idx] = acc;
}

@compute @workgroup_size(256)
fn fp32_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    var a: f32 = 1.0009765625 + f32(idx) * 0.0000001;
    var acc: f32 = 0.0;
    for (var i = 0u; i < STEPS; i++) {
        acc = a * acc + 1.0000001;
        a = a * 1.0000001 + 0.0000001;
    }
    sink[idx] = i32(acc);
}
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let iters: u32 = arg(&args, "--iters")
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let steps: u32 = arg(&args, "--steps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
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
    let threads: u64 = 1_048_576;
    let workgroups = (threads / 256) as u32;
    let src = SHADER.replace("__STEPS__", &steps.to_string());
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dp4a probe"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
    let sink = ctx.scratch_buffer("dp4a sink", threads * 4);
    let layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dp4a probe layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dp4a probe bind group"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: sink.as_entire_binding(),
        }],
    });
    let pipe_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dp4a probe pipeline layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
    for (name, entry) in [
        ("packed dot4I8Packed", "packed_main"),
        ("unpack vec4<i32> dot", "unpack_main"),
        ("fp32 fma", "fp32_main"),
    ] {
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pipe_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let run = |ctx: &wgpu_compute::GpuContext| {
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("dp4a probe pass"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("dp4a probe"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            ctx.queue.submit([encoder.finish()]);
        };
        for _ in 0..3 {
            run(ctx);
        }
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll after warmup");
        let start = Instant::now();
        for _ in 0..iters {
            run(ctx);
        }
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll after timed loop");
        let secs = start.elapsed().as_secs_f64() / f64::from(iters);
        let steps_done = threads * u64::from(steps);
        if name.starts_with("fp32") {
            let tflops = steps_done as f64 * 2.0 / secs / 1e12;
            println!("{name:<22} {:>8.3} ms  {tflops:>7.2} TFLOPS", secs * 1e3);
        } else {
            let tops = steps_done as f64 * 4.0 / secs / 1e12;
            println!(
                "{name:<22} {:>8.3} ms  {tops:>7.2} TOPS (int8 MAC)",
                secs * 1e3
            );
        }
    }
    ExitCode::SUCCESS
}
