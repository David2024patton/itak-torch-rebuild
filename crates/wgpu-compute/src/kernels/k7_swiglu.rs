//! K7: the elementwise activation family (BACKEND.md Section 3).
//!
//! Two kernels, one shader module:
//!
//! - **SwiGLU**: `silu(gate) * up` over the FFN intermediate
//!   (19,968 wide on the real model), the activation between the
//!   gate/up projections (K1, and K2 when it lands) and the down
//!   projection. `silu` is `torch-math`'s exact form,
//!   `x / (1 + exp(-x))`.
//! - **The attention gate**: `attn_out * sigmoid(gate)`, the
//!   gated-attention multiply the forward applies between the
//!   attention step and the output projection. It is a separate
//!   small op in the trait (`Compute::attn_gate`), not one of
//!   BACKEND.md's numbered kernels; it lives in this module
//!   because it is the same elementwise family and the same
//!   sigmoid, and `cpu-compute` (the spec) computes it with the
//!   branch-stable sigmoid this shader mirrors term for term:
//!   for `x >= 0`, `1 / (1 + exp(-x))`; below zero,
//!   `exp(x) / (1 + exp(x))`, so neither tail overflows.
//!
//! Both kernels are one thread per element over a two-dimensional
//! grid (the FFN intermediate at prefill sizes and the 202,048
//! class of widths both exceed one dispatch dimension's thread
//! count once rows multiply in).

use crate::device::GpuContext;
use crate::error::{Result, WgpuError};

use crate::diag::dbg_log;

/// Threads per workgroup for the elementwise kernels.
const WORKGROUP: u32 = 256;

/// The elementwise shader: two entry points sharing the params
/// struct and the sigmoid.
const SHADER: &str = r#"
struct Params {
    n: u32,
    grid_x: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

// Logistic sigmoid, cpu-compute's branch-stable form: the
// exponential's argument is never positive, so neither tail
// overflows.
fn sigmoid(x: f32) -> f32 {
    if (x >= 0.0) {
        return 1.0 / (1.0 + exp(-x));
    }
    let e = exp(x);
    return e / (1.0 + e);
}

// SiLU, torch-math's form: x * sigmoid(x) written as the single
// quotient the reference computes.
fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

fn index(wid: vec3<u32>, lid: vec3<u32>) -> u32 {
    return (wid.y * params.grid_x + wid.x) * 256u + lid.x;
}

@compute @workgroup_size(256)
fn swiglu_main(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = index(wid, lid);
    if (i < params.n) {
        out[i] = silu(a[i]) * b[i];
    }
}

@compute @workgroup_size(256)
fn gate_main(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = index(wid, lid);
    if (i < params.n) {
        out[i] = a[i] * sigmoid(b[i]);
    }
}
"#;

/// The compiled K7 pipelines (SwiGLU and the attention gate).
#[derive(Debug)]
pub struct K7Pipeline {
    swiglu: wgpu::ComputePipeline,
    gate: wgpu::ComputePipeline,
}

impl K7Pipeline {
    /// Compile both K7 pipelines on `device`.
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] when wgpu rejects the shader.
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("k7 elementwise shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let build = |entry: &'static str, label: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let swiglu = build("swiglu_main", "k7 swiglu pipeline");
        let gate = build("gate_main", "k7 attention gate pipeline");
        dbg_log!("k7 pipelines built");
        Ok(K7Pipeline { swiglu, gate })
    }

    /// Shared elementwise body: upload both operands, dispatch,
    /// read back.
    fn run(
        &self,
        ctx: &GpuContext,
        pipeline: &wgpu::ComputePipeline,
        a: &[f32],
        b: &[f32],
        what: &'static str,
    ) -> Result<Vec<f32>> {
        if a.len() != b.len() {
            return Err(WgpuError::InvalidTensor {
                name: "k7 input".to_string(),
                detail: format!("{what}: operands hold {} and {} elements", a.len(), b.len()),
            });
        }
        if a.is_empty() {
            return Ok(Vec::new());
        }
        let a_buf = ctx.upload_f32("k7 a", a);
        let b_buf = ctx.upload_f32("k7 b", b);
        let out_buf = ctx.scratch_buffer("k7 output", (a.len() * 4) as u64);
        let total = (a.len() as u64).div_ceil(u64::from(WORKGROUP));
        let grid = crate::kernels::k1_q4k_gemv::grid_dims(total);
        let params: [u32; 2] = [a.len() as u32, grid.0];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buf = ctx.uniform_buffer("k7 params", &param_bytes);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k7 bind group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k7 run"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(what),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(grid.0, grid.1, 1);
        }
        ctx.queue.submit([encoder.finish()]);
        dbg_log!("k7 {what}: n={}", a.len());
        ctx.read_f32(&out_buf, a.len())
    }

    /// SwiGLU activation: `silu(gate) * up`, elementwise.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when the halves differ in
    /// length; [`WgpuError::Readback`] on map failure.
    pub fn swiglu(&self, ctx: &GpuContext, gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
        self.run(ctx, &self.swiglu, gate, up, "swiglu")
    }

    /// The attention gate: `attn_out * sigmoid(gate)`, elementwise.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when the operands differ in
    /// length; [`WgpuError::Readback`] on map failure.
    pub fn attn_gate(&self, ctx: &GpuContext, attn_out: &[f32], gate: &[f32]) -> Result<Vec<f32>> {
        self.run(ctx, &self.gate, attn_out, gate, "attn_gate")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_carries_both_entry_points() {
        assert!(SHADER.contains("fn swiglu_main"));
        assert!(SHADER.contains("fn gate_main"));
        // The branch-stable sigmoid is a pinned convention from
        // cpu-compute; the shader must carry both branches.
        assert!(SHADER.contains("if (x >= 0.0)"));
    }
}
