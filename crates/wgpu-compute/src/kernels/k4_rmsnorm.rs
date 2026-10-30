//! K4: RMSNorm fused with the residual add (BACKEND.md Section 3).
//!
//! The bytes are trivial (the 6,656-wide hidden vector, twice per
//! layer); the point is fusion, so the hidden vector is not
//! re-read from VRAM between the residual add and the norm. One
//! shader, mode-selected by a uniform, covers every norm the
//! forward calls:
//!
//! - **Weighted** ([`K4Pipeline::rms_norm`]): `x / sqrt(mean(x^2) +
//!   eps) * w`, plain `x * w` scaling. This is the pre norms (at
//!   the file's eps), the per-head Q and K norms (width = head
//!   dim, rows = positions times heads), and the final norm.
//! - **Plain** ([`K4Pipeline::rms_norm_plain`]): the weightless
//!   form the reference graph applies to the embedding output
//!   before layer 0 (src/models/muse-glimmer.cpp: build_norm with
//!   a null weight). A distinct mode, not a ones weight the
//!   caller would have to invent, mirroring the trait.
//! - **Weighted + residual** ([`K4Pipeline::norm_residual`]): the
//!   fused post-norm step, `residual + norm(branch)`, the shape
//!   BACKEND.md's fusion note describes: the branch output is
//!   normalized (at the reference graph's hardcoded post-norm eps
//!   1e-8, passed in as `eps` like every other value here) and the
//!   untouched residual is added in the same pass. The `Compute`
//!   trait hands the forward separate `rms_norm` and `add` steps,
//!   so the trait path composes the two halves of this module;
//!   the fused entry point is the one a device-resident forward
//!   calls, and it is proven against that composition in the
//!   crate's tests.
//!
//! The residual add also lives here as its own tiny kernel
//! ([`K4Pipeline::add`]): BACKEND.md names K4 "RMSNorm fused with
//! residual add", so the add is this module's second half, not a
//! kernel of its own.
//!
//! Numerical note: the CPU reference accumulates the mean of
//! squares in f64; this kernel accumulates in f32 across the
//! workgroup. The difference is reassociation-class noise, far
//! below the cross-backend tolerance the crate's tests assert,
//! and it is stated here rather than hidden.

use crate::device::GpuContext;
use crate::error::{Result, WgpuError};

use crate::diag::dbg_log;

/// Threads per workgroup: one workgroup normalizes one row,
/// lanes striding over the row's channels.
const WORKGROUP: u32 = 256;

/// The norm shader. `mode`: 0 weighted, 1 plain (weightless),
/// 2 weighted with the residual added. The weight and residual
/// bindings are always present; modes that do not read them bind
/// a one-element dummy buffer instead.
const NORM_SHADER: &str = r#"
struct Params {
    width: u32,
    total_rows: u32,
    grid_x: u32,
    mode: u32,
    eps: f32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> residual: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> partials: array<f32, 256>;
var<workgroup> inv_rms_shared: f32;

@compute @workgroup_size(256)
fn norm_main(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.y * params.grid_x + wid.x;
    // Whole-workgroup early out: every lane shares `row`, so the
    // exit is uniform and the barriers below stay legal.
    if (row >= params.total_rows) {
        return;
    }
    let lane = lid.x;
    let base = row * params.width;

    var ss = 0.0;
    for (var i = lane; i < params.width; i += 256u) {
        let v = x[base + i];
        ss += v * v;
    }
    partials[lane] = ss;
    workgroupBarrier();
    var stride = 128u;
    while (stride > 0u) {
        if (lane < stride) {
            partials[lane] += partials[lane + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    if (lane == 0u) {
        inv_rms_shared = inverseSqrt(partials[0] / f32(params.width) + params.eps);
    }
    workgroupBarrier();
    let inv = inv_rms_shared;

    for (var i = lane; i < params.width; i += 256u) {
        var y = x[base + i] * inv;
        if (params.mode != 1u) {
            y = y * weight[i];
        }
        if (params.mode == 2u) {
            y = residual[base + i] + y;
        }
        out[base + i] = y;
    }
}
"#;

/// The residual add shader: one thread per element.
const ADD_SHADER: &str = r#"
struct AddParams {
    n: u32,
    grid_x: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: AddParams;

@compute @workgroup_size(256)
fn add_main(@builtin(workgroup_id) wid: vec3<u32>,
            @builtin(local_invocation_id) lid: vec3<u32>) {
    let flat = (wid.y * params.grid_x + wid.x) * 256u + lid.x;
    if (flat < params.n) {
        out[flat] = a[flat] + b[flat];
    }
}
"#;

/// Mode values for the norm shader's `mode` uniform.
mod mode {
    pub const WEIGHTED: u32 = 0;
    pub const PLAIN: u32 = 1;
    pub const WEIGHTED_RESIDUAL: u32 = 2;
}

/// One prepared K4 dispatch: a bind group plus grid dimensions,
/// built once for a fixed (buffers, rows, width, mode, eps)
/// combination and encodable any number of times into any
/// command encoder. This is the device-resident forward's unit
/// of composition (see [`crate::session`]): the one-shot entry
/// points below prepare, encode, submit, and read back; the
/// resident path prepares once per shape and encodes the whole
/// forward into a single submission.
#[derive(Debug)]
pub struct K4Call {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    grid: (u32, u32),
}

impl K4Call {
    /// Record this call's dispatch into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k4 dispatch"),
            timestamp_writes: None,
        });
        self.encode_into(&mut pass);
    }

    /// Record this dispatch into an already-open compute
    /// `pass`: the composition form the device-resident
    /// session uses to chain a whole forward into one pass
    /// per segment (wgpu inserts the hazard barriers
    /// between dispatches inside the pass).
    pub fn encode_into(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.grid.0, self.grid.1, 1);
    }
}

/// The compiled K4 pipelines (norm and add), plus the dummy
/// buffer the norm shader binds for inputs a mode does not read.
#[derive(Debug)]
pub struct K4Pipeline {
    norm: wgpu::ComputePipeline,
    add: wgpu::ComputePipeline,
    dummy: wgpu::Buffer,
}

impl K4Pipeline {
    /// Compile both K4 pipelines on `device`.
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] when wgpu rejects a shader.
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        let norm_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("k4 rmsnorm shader"),
            source: wgpu::ShaderSource::Wgsl(NORM_SHADER.into()),
        });
        let norm = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k4 rmsnorm pipeline"),
            layout: None,
            module: &norm_module,
            entry_point: Some("norm_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let add_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("k4 residual add shader"),
            source: wgpu::ShaderSource::Wgsl(ADD_SHADER.into()),
        });
        let add = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k4 residual add pipeline"),
            layout: None,
            module: &add_module,
            entry_point: Some("add_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k4 dummy binding"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        dbg_log!("k4 pipelines built");
        Ok(K4Pipeline { norm, add, dummy })
    }

    /// Prepare a norm dispatch over device buffers: `x` read,
    /// `out` written, `weight` and `residual` bound when the mode
    /// reads them (a dummy binding otherwise). `rows` rows of
    /// `width` channels; the caller validates shapes, exactly as
    /// the one-shot entry points do before they call this.
    // One argument per bound input plus the mode and eps: the
    // parameter list is the bind group's shape, kept flat on
    // purpose for this private helper.
    #[allow(clippy::too_many_arguments)]
    fn prepare_norm(
        &self,
        ctx: &GpuContext,
        x: &wgpu::Buffer,
        weight: Option<&wgpu::Buffer>,
        residual: Option<&wgpu::Buffer>,
        out: &wgpu::Buffer,
        rows: usize,
        width: usize,
        mode_value: u32,
        eps: f32,
    ) -> K4Call {
        let grid = crate::kernels::k1_q4k_gemv::grid_dims(rows as u64);
        let params: [u32; 5] = [width as u32, rows as u32, grid.0, mode_value, eps.to_bits()];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buf = ctx.uniform_buffer("k4 norm params", &param_bytes);
        let dummy = self.dummy.clone();
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k4 norm bind group"),
            layout: &self.norm.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight.unwrap_or(&dummy).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: residual.unwrap_or(&dummy).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        K4Call {
            pipeline: self.norm.clone(),
            bind_group,
            grid,
        }
    }

    /// Prepare a weighted RMSNorm over device buffers:
    /// `x / sqrt(mean(x^2) + eps) * w` per row, `x` holding
    /// `rows` rows of the weight's width, written to `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_rms_norm(
        &self,
        ctx: &GpuContext,
        x: &wgpu::Buffer,
        weight: &wgpu::Buffer,
        out: &wgpu::Buffer,
        rows: usize,
        width: usize,
        eps: f32,
    ) -> K4Call {
        self.prepare_norm(
            ctx,
            x,
            Some(weight),
            None,
            out,
            rows,
            width,
            mode::WEIGHTED,
            eps,
        )
    }

    /// Prepare a weightless RMSNorm over device buffers (the
    /// post-embedding form): `x / sqrt(mean(x^2) + eps)` per row.
    pub fn prepare_rms_norm_plain(
        &self,
        ctx: &GpuContext,
        x: &wgpu::Buffer,
        out: &wgpu::Buffer,
        rows: usize,
        width: usize,
        eps: f32,
    ) -> K4Call {
        self.prepare_norm(ctx, x, None, None, out, rows, width, mode::PLAIN, eps)
    }

    /// Prepare the fused post-norm step over device buffers:
    /// `residual + rms_norm(branch) * w` per row in one pass,
    /// written to `out` (which aliases neither input).
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_norm_residual(
        &self,
        ctx: &GpuContext,
        residual: &wgpu::Buffer,
        branch: &wgpu::Buffer,
        weight: &wgpu::Buffer,
        out: &wgpu::Buffer,
        rows: usize,
        width: usize,
        eps: f32,
    ) -> K4Call {
        self.prepare_norm(
            ctx,
            branch,
            Some(weight),
            Some(residual),
            out,
            rows,
            width,
            mode::WEIGHTED_RESIDUAL,
            eps,
        )
    }

    /// Prepare a residual add over device buffers: `a + b`
    /// elementwise into `out`, `n` elements.
    pub fn prepare_add(
        &self,
        ctx: &GpuContext,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        out: &wgpu::Buffer,
        n: usize,
    ) -> K4Call {
        let total = (n as u64).div_ceil(u64::from(WORKGROUP));
        let grid = crate::kernels::k1_q4k_gemv::grid_dims(total);
        let params: [u32; 2] = [n as u32, grid.0];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buf = ctx.uniform_buffer("k4 add params", &param_bytes);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k4 add bind group"),
            layout: &self.add.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        K4Call {
            pipeline: self.add.clone(),
            bind_group,
            grid,
        }
    }

    /// Shared norm body: upload, dispatch one workgroup per row,
    /// read back. `weight` and `residual` are bound only by the
    /// modes that read them; the caller validates shapes.
    #[allow(clippy::too_many_arguments)]
    fn run_norm(
        &self,
        ctx: &GpuContext,
        x: &[f32],
        weight: Option<&[f32]>,
        residual: Option<&[f32]>,
        rows: usize,
        width: usize,
        mode_value: u32,
        eps: f32,
    ) -> Result<Vec<f32>> {
        let x_buf = ctx.upload_f32("k4 norm input", x);
        let w_buf = weight.map(|w| ctx.upload_f32("k4 norm weight", w));
        let r_buf = residual.map(|r| ctx.upload_f32("k4 norm residual", r));
        let out_buf = ctx.scratch_buffer("k4 norm output", (x.len() * 4) as u64);
        let call = self.prepare_norm(
            ctx,
            &x_buf,
            w_buf.as_ref(),
            r_buf.as_ref(),
            &out_buf,
            rows,
            width,
            mode_value,
            eps,
        );
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k4 norm run"),
            });
        call.encode(&mut encoder);
        ctx.queue.submit([encoder.finish()]);
        dbg_log!("k4 norm: rows={rows} width={width} mode={mode_value} eps={eps}");
        ctx.read_f32(&out_buf, x.len())
    }

    /// Weighted RMSNorm: `x / sqrt(mean(x^2) + eps) * w` per row.
    /// `x` holds `rows` rows of `weight.len()` channels.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when `x` is not a whole number
    /// of weight-wide rows; [`WgpuError::Readback`] on map failure.
    pub fn rms_norm(
        &self,
        ctx: &GpuContext,
        x: &[f32],
        weight: &[f32],
        rows: usize,
        eps: f32,
    ) -> Result<Vec<f32>> {
        let width = check_rows("rms_norm", x, rows, weight.len())?;
        self.run_norm(ctx, x, Some(weight), None, rows, width, mode::WEIGHTED, eps)
    }

    /// Weightless RMSNorm: `x / sqrt(mean(x^2) + eps)` per row.
    /// `rows` divides `x` evenly; the width is `x.len() / rows`.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when `rows` does not divide
    /// `x`; [`WgpuError::Readback`] on map failure.
    pub fn rms_norm_plain(
        &self,
        ctx: &GpuContext,
        x: &[f32],
        rows: usize,
        eps: f32,
    ) -> Result<Vec<f32>> {
        if rows == 0 || x.is_empty() || x.len() % rows != 0 {
            return Err(WgpuError::InvalidTensor {
                name: "rms_norm_plain input".to_string(),
                detail: format!(
                    "input holds {} elements, not a whole number of {rows} rows",
                    x.len()
                ),
            });
        }
        let width = x.len() / rows;
        self.run_norm(ctx, x, None, None, rows, width, mode::PLAIN, eps)
    }

    /// The fused post-norm step: `residual + rms_norm(branch) * w`
    /// per row, in one pass. `eps` is the caller's: the reference
    /// graph's post norms pass its hardcoded 1e-8.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] on any shape disagreement;
    /// [`WgpuError::Readback`] on map failure.
    pub fn norm_residual(
        &self,
        ctx: &GpuContext,
        residual: &[f32],
        branch: &[f32],
        weight: &[f32],
        rows: usize,
        eps: f32,
    ) -> Result<Vec<f32>> {
        if residual.len() != branch.len() {
            return Err(WgpuError::InvalidTensor {
                name: "norm_residual input".to_string(),
                detail: format!(
                    "residual holds {} elements but the branch holds {}",
                    residual.len(),
                    branch.len()
                ),
            });
        }
        let width = check_rows("norm_residual", branch, rows, weight.len())?;
        self.run_norm(
            ctx,
            branch,
            Some(weight),
            Some(residual),
            rows,
            width,
            mode::WEIGHTED_RESIDUAL,
            eps,
        )
    }

    /// Residual add of two equal-length activations.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when the lengths differ;
    /// [`WgpuError::Readback`] on map failure.
    pub fn add(&self, ctx: &GpuContext, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if a.len() != b.len() {
            return Err(WgpuError::InvalidTensor {
                name: "add input".to_string(),
                detail: format!(
                    "residual operands hold {} and {} elements",
                    a.len(),
                    b.len()
                ),
            });
        }
        if a.is_empty() {
            return Ok(Vec::new());
        }
        let a_buf = ctx.upload_f32("k4 add a", a);
        let b_buf = ctx.upload_f32("k4 add b", b);
        let out_buf = ctx.scratch_buffer("k4 add output", (a.len() * 4) as u64);
        let call = self.prepare_add(ctx, &a_buf, &b_buf, &out_buf, a.len());
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k4 add run"),
            });
        call.encode(&mut encoder);
        ctx.queue.submit([encoder.finish()]);
        dbg_log!("k4 add: n={}", a.len());
        ctx.read_f32(&out_buf, a.len())
    }
}

/// Validate that `x` is exactly `rows` rows of `width`, returning
/// the width. Shared by the weighted entry points.
fn check_rows(step: &'static str, x: &[f32], rows: usize, width: usize) -> Result<usize> {
    if rows == 0 || width == 0 || x.len() != rows * width {
        return Err(WgpuError::InvalidTensor {
            name: "k4 input".to_string(),
            detail: format!(
                "{step}: input holds {} elements but {rows} rows of width {width} need {}",
                x.len(),
                rows * width
            ),
        });
    }
    Ok(width)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_shader_names_its_entry_point_and_modes() {
        // CPU-side guard: the shader carries both entry points the
        // pipelines bind, and the mode uniform the host selects.
        assert!(NORM_SHADER.contains("fn norm_main"));
        assert!(NORM_SHADER.contains("mode: u32"));
        assert!(ADD_SHADER.contains("fn add_main"));
    }

    #[test]
    fn check_rows_rejects_mismatched_shapes() {
        assert!(check_rows("rms_norm", &[0.0; 6], 2, 3).is_ok());
        assert!(check_rows("rms_norm", &[0.0; 6], 2, 4).is_err());
        assert!(check_rows("rms_norm", &[0.0; 6], 0, 3).is_err());
        assert!(check_rows("rms_norm", &[0.0; 6], 2, 0).is_err());
    }
}
