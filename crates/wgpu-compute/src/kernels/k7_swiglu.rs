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
//!
//! Each kernel also has a **strided** entry point
//! (`gate_strided_main`, `swiglu_strided_main`): the same
//! elementwise math over rows whose two operand streams sit at
//! [`K1ActGeom`] strides inside larger buffers, writing tightly
//! packed output rows. That is the activation stage of the
//! fused gated/SwiGLU projection forms (K1's gated and SwiGLU
//! forms read their operands through exactly that geometry),
//! materialized on its own so the projection behind it can be
//! any plain product the routing ladder chooses; see
//! `prepare_gated_proj` / `prepare_swiglu_proj` in
//! `crate::session` and bench/gated-swiglu-routing-2026-11-06.md.

use crate::device::GpuContext;
use crate::error::{Result, WgpuError};
use crate::kernels::k1_q4k_gemv::K1ActGeom;

use crate::diag::dbg_log;

/// Threads per workgroup for the elementwise kernels.
const WORKGROUP: u32 = 256;

/// The elementwise shader: two entry points sharing the params
/// struct and the sigmoid.
const SHADER: &str = r#"
struct Params {
    n: u32,
    grid_x: u32,
    // Strided-form fields (the flat entries ignore them; the
    // flat prepares fill them with the flat-consistent values
    // n / n / n / 0, under which the strided index math reduces
    // to the flat one exactly).
    width: u32,
    a_stride: u32,
    b_stride: u32,
    b_off: u32,
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

// The strided position of output element i (row r, column c of
// the packed output): stream A at r * a_stride + c, stream B at
// r * b_stride + b_off + c. The geometry is K1's K1ActGeom, the
// same one the fused gated/SwiGLU product forms read through.
fn strided_pos(i: u32) -> vec3<u32> {
    let r = i / params.width;
    let c = i - r * params.width;
    return vec3<u32>(
        r * params.a_stride + c,
        r * params.b_stride + params.b_off + c,
        i,
    );
}

@compute @workgroup_size(256)
fn swiglu_strided_main(@builtin(workgroup_id) wid: vec3<u32>,
                       @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = index(wid, lid);
    if (i < params.n) {
        let p = strided_pos(i);
        out[p.z] = silu(a[p.x]) * b[p.y];
    }
}

@compute @workgroup_size(256)
fn gate_strided_main(@builtin(workgroup_id) wid: vec3<u32>,
                     @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = index(wid, lid);
    if (i < params.n) {
        let p = strided_pos(i);
        out[p.z] = a[p.x] * sigmoid(b[p.y]);
    }
}
"#;

/// One prepared K7 dispatch: a bind group plus grid dimensions,
/// built once for a fixed (buffers, length) combination and
/// encodable any number of times into any command encoder. The
/// device-resident forward's unit of composition (see
/// [`crate::session`]); the one-shot entry points below prepare,
/// encode, submit, and read back.
#[derive(Debug)]
pub struct K7Call {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    grid: (u32, u32),
    /// Diag timestamp stamp (see `crate::timestamps`); `None`
    /// unless the context carries a recorder.
    diag: Option<crate::timestamps::DiagStamp>,
}

impl K7Call {
    /// Record this call's dispatch into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k7 dispatch"),
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
        let slot = self.diag.as_ref().and_then(|d| d.begin(pass));
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.grid.0, self.grid.1, 1);
        if let (Some(d), Some(s)) = (&self.diag, slot) {
            d.end(pass, s);
        }
    }
}

/// The compiled K7 pipelines (SwiGLU and the attention gate,
/// flat and strided forms).
#[derive(Debug)]
pub struct K7Pipeline {
    swiglu: wgpu::ComputePipeline,
    gate: wgpu::ComputePipeline,
    swiglu_strided: wgpu::ComputePipeline,
    gate_strided: wgpu::ComputePipeline,
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
        let swiglu_strided = build("swiglu_strided_main", "k7 swiglu strided pipeline");
        let gate_strided = build("gate_strided_main", "k7 attention gate strided pipeline");
        dbg_log!("k7 pipelines built");
        Ok(K7Pipeline {
            swiglu,
            gate,
            swiglu_strided,
            gate_strided,
        })
    }

    /// Prepare an elementwise dispatch over device buffers:
    /// `out[i] = op(a[i], b[i])` for `n` elements, through
    /// `pipeline`. The mode tag rides along so the diag stamp
    /// can name the operation (swiglu vs gate).
    #[allow(clippy::too_many_arguments)]
    fn prepare(
        &self,
        ctx: &GpuContext,
        pipeline: &wgpu::ComputePipeline,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        out: &wgpu::Buffer,
        n: usize,
        mode: &'static str,
    ) -> K7Call {
        let total = (n as u64).div_ceil(u64::from(WORKGROUP));
        let grid = crate::kernels::k1_q4k_gemv::grid_dims(total);
        // The strided fields ride along at their flat-consistent
        // values (width = strides = n, offset 0): the flat entry
        // points never read them, and a strided entry pointed at
        // this buffer would compute the flat map exactly.
        let params: [u32; 6] = [n as u32, grid.0, n as u32, n as u32, n as u32, 0];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buf = ctx.uniform_buffer("k7 params", &param_bytes);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k7 bind group"),
            layout: &pipeline.get_bind_group_layout(0),
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
        K7Call {
            pipeline: pipeline.clone(),
            bind_group,
            grid,
            diag: ctx.diag_stamp(crate::timestamps::DispatchMeta {
                kernel: "k7_swiglu_gate",
                rows: n as u64,
                cols: 1,
                quant: "",
                weight_bytes: 0,
                shape: format!("mode={mode} n={n}"),
            }),
        }
    }

    /// Prepare a SwiGLU dispatch over device buffers:
    /// `out = silu(gate) * up` elementwise, `n` elements.
    pub fn prepare_swiglu(
        &self,
        ctx: &GpuContext,
        gate: &wgpu::Buffer,
        up: &wgpu::Buffer,
        out: &wgpu::Buffer,
        n: usize,
    ) -> K7Call {
        self.prepare(ctx, &self.swiglu, gate, up, out, n, "swiglu")
    }

    /// Prepare an attention-gate dispatch over device buffers:
    /// `out = attn_out * sigmoid(gate)` elementwise, `n`
    /// elements.
    pub fn prepare_gate(
        &self,
        ctx: &GpuContext,
        attn_out: &wgpu::Buffer,
        gate: &wgpu::Buffer,
        out: &wgpu::Buffer,
        n: usize,
    ) -> K7Call {
        self.prepare(ctx, &self.gate, attn_out, gate, out, n, "gate")
    }

    /// Shared strided prepare: `rows` packed output rows of
    /// `width` elements each, stream A read at
    /// `geom.a_stride_elems` per row, stream B at
    /// `geom.b_stride_elems` per row plus `geom.b_off_elems`.
    /// The caller's contract (the fused forms' own geometry
    /// contract): `width <= a_stride_elems` and
    /// `b_off_elems + width <= b_stride_elems`, and the output
    /// buffer holds `rows * width` elements. The elementwise
    /// math is the flat form's, term for term; only the index
    /// map differs.
    #[allow(clippy::too_many_arguments)]
    fn prepare_strided(
        &self,
        ctx: &GpuContext,
        pipeline: &wgpu::ComputePipeline,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        out: &wgpu::Buffer,
        rows: usize,
        width: usize,
        geom: &K1ActGeom,
        mode: &'static str,
    ) -> K7Call {
        let n = rows * width;
        let total = (n as u64).div_ceil(u64::from(WORKGROUP));
        let grid = crate::kernels::k1_q4k_gemv::grid_dims(total);
        let params: [u32; 6] = [
            n as u32,
            grid.0,
            width as u32,
            geom.a_stride_elems as u32,
            geom.b_stride_elems as u32,
            geom.b_off_elems as u32,
        ];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buf = ctx.uniform_buffer("k7 strided params", &param_bytes);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k7 strided bind group"),
            layout: &pipeline.get_bind_group_layout(0),
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
        K7Call {
            pipeline: pipeline.clone(),
            bind_group,
            grid,
            diag: ctx.diag_stamp(crate::timestamps::DispatchMeta {
                kernel: "k7_swiglu_gate",
                rows: n as u64,
                cols: 1,
                quant: "",
                weight_bytes: 0,
                shape: format!(
                    "mode={mode} rows={rows} width={width} a_stride={} b_stride={} b_off={}",
                    geom.a_stride_elems, geom.b_stride_elems, geom.b_off_elems
                ),
            }),
        }
    }

    /// Prepare a strided SwiGLU dispatch: `out[r][c] =
    /// silu(a[r * a_stride + c]) * b[r * b_stride + b_off + c]`
    /// over `rows` rows of `width`, writing packed rows. This
    /// is the FFN pair's activation stage (gate and up riding
    /// one pair buffer at `geom`) with the product behind it
    /// left to the caller.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_swiglu_strided(
        &self,
        ctx: &GpuContext,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        out: &wgpu::Buffer,
        rows: usize,
        width: usize,
        geom: &K1ActGeom,
    ) -> K7Call {
        self.prepare_strided(
            ctx,
            &self.swiglu_strided,
            a,
            b,
            out,
            rows,
            width,
            geom,
            "swiglu_strided",
        )
    }

    /// Prepare a strided attention-gate dispatch: `out[r][c] =
    /// a[r * a_stride + c] * sigmoid(b[r * b_stride + b_off +
    /// c])` over `rows` rows of `width`, writing packed rows.
    /// This is the gated output projection's activation stage
    /// (the gate riding the QKV batch buffer at `geom`, or a
    /// gate buffer of its own) with the product behind it left
    /// to the caller.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_gate_strided(
        &self,
        ctx: &GpuContext,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        out: &wgpu::Buffer,
        rows: usize,
        width: usize,
        geom: &K1ActGeom,
    ) -> K7Call {
        self.prepare_strided(
            ctx,
            &self.gate_strided,
            a,
            b,
            out,
            rows,
            width,
            geom,
            "gate_strided",
        )
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
        let call = self.prepare(ctx, pipeline, &a_buf, &b_buf, &out_buf, a.len(), what);
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k7 run"),
            });
        call.encode(&mut encoder);
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

    #[test]
    fn shader_carries_the_strided_entry_points() {
        assert!(SHADER.contains("fn swiglu_strided_main"));
        assert!(SHADER.contains("fn gate_strided_main"));
        // The strided index map is the K1ActGeom read, pinned
        // term for term: stream A at row * a_stride + col,
        // stream B at row * b_stride + b_off + col.
        assert!(SHADER.contains("r * params.a_stride + c"));
        assert!(SHADER.contains("r * params.b_stride + params.b_off + c"));
        // The strided entries apply the same two functions as
        // the flat ones: the quotient silu and the
        // branch-stable sigmoid, never a second formula.
        assert!(SHADER.contains("out[p.z] = silu(a[p.x]) * b[p.y];"));
        assert!(SHADER.contains("out[p.z] = a[p.x] * sigmoid(b[p.y]);"));
    }

    #[test]
    fn shader_validates_under_naga() {
        // The strongest shader gate a machine with no GPU
        // adapter at all can run: the whole module (flat and
        // strided entries) must parse and validate under
        // Naga, the same front end wgpu feeds the driver
        // with.
        let module = naga::front::wgsl::parse_str(SHADER)
            .unwrap_or_else(|e| panic!("WGSL parse failed: {}", e.emit_to_string(SHADER)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("Naga validation failed: {e:?}"));
    }

    /// cpu-compute's branch-stable sigmoid, the reference the
    /// shader mirrors term for term.
    fn ref_sigmoid(x: f32) -> f32 {
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let e = x.exp();
            e / (1.0 + e)
        }
    }

    /// torch-math's silu quotient, the shader's `silu`.
    fn ref_silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// The strided index map, restated on the host: output
    /// element (r, c) reads A at `r * a_stride + c` and B at
    /// `r * b_stride + b_off + c`. This is the layout
    /// contract `strided_pos` encodes in the shader; the
    /// tests below pin its two load-bearing properties.
    fn strided_sources(rows: usize, width: usize, geom: &K1ActGeom) -> Vec<(usize, usize)> {
        let mut out = Vec::with_capacity(rows * width);
        for r in 0..rows {
            for c in 0..width {
                out.push((
                    r * geom.a_stride_elems + c,
                    r * geom.b_stride_elems + geom.b_off_elems + c,
                ));
            }
        }
        out
    }

    #[test]
    fn strided_map_with_flat_geometry_is_the_flat_map() {
        // Under the flat-consistent geometry (width = strides
        // = n over one row, offset 0) the strided map is the
        // identity: the flat prepares' params values are
        // therefore a special case of the strided form, not a
        // second convention.
        let n = 37usize;
        let geom = K1ActGeom {
            a_stride_elems: n,
            b_stride_elems: n,
            b_off_elems: 0,
        };
        let map = strided_sources(1, n, &geom);
        for (i, &(ai, bi)) in map.iter().enumerate() {
            assert_eq!(ai, i);
            assert_eq!(bi, i);
        }
    }

    #[test]
    fn strided_map_reads_exactly_the_geometric_segments() {
        // The two session geometries, with sentinels outside
        // the segments: the gate riding a QKV batch row
        // (b segment at an offset inside a wider row) and
        // the FFN pair (both streams inside one row of twice
        // the width, B offset by the width).
        let gate_geom = K1ActGeom {
            a_stride_elems: 8,
            b_stride_elems: 32,
            b_off_elems: 24,
        };
        let map = strided_sources(3, 8, &gate_geom);
        assert_eq!(map.len(), 24);
        assert_eq!(map[0], (0, 24));
        assert_eq!(map[7], (7, 31));
        assert_eq!(map[8], (8, 56));
        assert_eq!(map[23], (23, 95));

        let pair_geom = K1ActGeom {
            a_stride_elems: 16,
            b_stride_elems: 16,
            b_off_elems: 8,
        };
        let map = strided_sources(2, 8, &pair_geom);
        assert_eq!(map[0], (0, 8));
        assert_eq!(map[7], (7, 15));
        assert_eq!(map[8], (16, 24));
        assert_eq!(map[15], (23, 31));

        // And the math over a strided layout equals the flat
        // math over the extracted segments, element for
        // element: the decomposition the routed session path
        // relies on. The segments are extracted by hand here,
        // not through the map under test.
        let a: Vec<f32> = (0..96).map(|i| (i as f32 - 40.0) * 0.05).collect();
        let b: Vec<f32> = (0..96).map(|i| (40.0 - i as f32) * 0.03).collect();
        let map = strided_sources(3, 8, &gate_geom);
        for (r, row) in map.chunks(8).enumerate() {
            for (c, &(ai, bi)) in row.iter().enumerate() {
                assert_eq!(
                    a[ai] * ref_sigmoid(b[bi]),
                    a[r * 8 + c] * ref_sigmoid(b[r * 32 + 24 + c])
                );
                assert_eq!(
                    ref_silu(a[ai]) * b[bi],
                    ref_silu(a[r * 8 + c]) * b[r * 32 + 24 + c]
                );
            }
        }
    }
}
