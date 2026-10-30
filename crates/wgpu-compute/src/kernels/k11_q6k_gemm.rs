//! K11: Q6_K matrix-matrix GEMM for prefill, K10's sibling.
//!
//! The tiling, the batch contract, and the fallback contract
//! are K10's (see [`crate::kernels::k10_q4k_gemm`]'s module
//! docs); what differs is the staged dequantization, which is
//! Q6_K's: element value `d * scale * (q - 32)` with the
//! 6-bit `q` assembled from a `ql` nibble and two `qh` bits,
//! the signed scale chosen by the element's stream and
//! 16-element half, mirroring `gguf::dequant::dequant_q6_k`
//! term for term. The kernel reads the SAME padded device
//! layout K2's upload writes (212-byte super-block stride:
//! 128 bytes of `ql`, 64 of `qh`, 16 signed scales, the f16
//! super-scale, two pad bytes), so a Q6_K tensor lives on the
//! device exactly once and both the GEMV and this GEMM bind
//! that one buffer.
//!
//! The activation vocabulary is the backend's K2 set: the
//! plain form and [`ACT_SWIGLU`] (the FFN down projection,
//! Q6_K on half the real model's layers, consuming the
//! gate+up pair with the activation fused). There is no
//! gated pipeline: the attention output projection whose
//! gate fusion exists is Q4_K in every file this backend
//! serves, and K2 offers no gated pipeline either.

use crate::device::GpuContext;
use crate::error::{Result, WgpuError};
use crate::kernels::k10_q4k_gemm::{shipped_shape, GemmShape, GEMM_MAX_BATCH};
use crate::kernels::k1_q4k_gemv::{K1ActGeom, ACT_NONE, ACT_SWIGLU};
use crate::kernels::k2_q6k_gemv::K2Weights;

use crate::diag::dbg_log;

/// Words per Q6_K super-block in the device buffer (the
/// padded 212-byte stride; see K2's module docs).
const BLOCK_WORDS_DEV: u32 = 53;
/// Elements per Q6_K super-block.
const BLOCK_ELEMS: usize = 256;

/// The Q6_K GEMM shader for `shape` and `act`: K10's tiling
/// body with the Q6_K stage. The stage assignment matches
/// K10's (one 32-element sub-block per task); within a
/// super-block, sub-block `j` is half `j / 4`, stream
/// `j % 4` of `dequant_q6_k`'s interleave: stream `s` reads
/// its nibbles from `ql` bytes at half offset `(s % 2) * 32`
/// (low nibbles for streams 0 and 1, high for 2 and 3), its
/// `qh` bit-pair at bit shift `(s % 2) * 2 + (s / 2) * 4`,
/// and its scale at `half * 8 + (l / 16) + 2 * s`.
fn gemm_shader(shape: &GemmShape, act: u32) -> String {
    let tile_n = shape.tile_n;
    let tile_k = shape.tile_k;
    let wg = shape.workgroup;
    let tile_vec4s = tile_n * tile_k / 4;
    let subs = tile_k / 32;
    let hb_count = 256 / tile_k;
    let reps = (tile_n * subs) / wg;
    let decls = if act == ACT_SWIGLU {
        r#"
@group(0) @binding(4) var<storage, read> input_b: array<vec4<f32>>;

// SiLU, K7's quotient form: x * sigmoid(x) as the single
// quotient the reference computes.
fn silu1(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

fn act_apply(g: vec4<f32>, u: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(silu1(g.x) * u.x, silu1(g.y) * u.y,
                     silu1(g.z) * u.z, silu1(g.w) * u.w);
}
"#
    } else {
        ""
    };

    let mut x_loads = String::new();
    for j in 0..4 {
        x_loads.push_str(&format!(
            "                var xv{j} = vec4<f32>(0.0);\n                if (m{j} < params.batch) {{\n                    xv{j} = input[(m{j} * params.a_stride_elems + e_base + kk) / 4u];\n"
        ));
        if act != ACT_NONE {
            x_loads.push_str(&format!(
                "                    xv{j} = act_apply(xv{j}, input_b[(m{j} * params.b_stride_elems + params.b_off_elems + e_base + kk) / 4u]);\n"
            ));
        }
        x_loads.push_str("                }\n");
    }
    let mut dots = String::new();
    for i in 0..4 {
        dots.push_str(&format!(
            "                let wv{i} = tile[((n_sub * 4u + {i}u) * {tile_k}u + kk) / 4u];\n"
        ));
        for j in 0..4 {
            dots.push_str(&format!(
                "                acc[{}] += dot(wv{i}, xv{j});\n",
                i * 4 + j
            ));
        }
    }
    let mut stores = String::new();
    for i in 0..4 {
        for j in 0..4 {
            stores.push_str(&format!(
                "    if (n{i} < params.out_rows && m{j} < params.batch) {{\n        output[m{j} * params.out_rows + n{i}] = acc[{}];\n    }}\n",
                i * 4 + j
            ));
        }
    }

    format!(
        r#"struct Params {{
    blocks_per_row: u32,
    out_rows: u32,
    batch: u32,
    a_stride_elems: u32,
    b_stride_elems: u32,
    b_off_elems: u32,
}}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
{decls}
var<workgroup> tile: array<vec4<f32>, {tile_vec4s}>;

// IEEE 754 half to f32, mirroring gguf::dequant::f16_to_f32 for
// every value a weight file can hold.
fn f16_to_f32(h: u32) -> f32 {{
    let sign = (h >> 15u) & 1u;
    let exp = (h >> 10u) & 31u;
    let frac = h & 1023u;
    if (exp == 31u) {{
        return bitcast<f32>((sign << 31u) | 0x7f800000u | (frac << 13u));
    }}
    if (exp == 0u) {{
        let v = f32(frac) * 5.960464477539063e-08;
        if (sign == 1u) {{ return -v; }}
        return v;
    }}
    return bitcast<f32>((sign << 31u) | ((exp + 112u) << 23u) | (frac << 13u));
}}

// One signed sub-block scale: byte `j` of the block's 16-byte
// scale plane, whose four words the caller loaded. The shift
// through the top byte sign-extends the int8.
fn scale_at(scw: array<u32, 4>, j: u32) -> f32 {{
    let b = (scw[j >> 2u] >> (8u * (j & 3u))) & 0xffu;
    return f32(bitcast<i32>(b << 24u) >> 24);
}}

@compute @workgroup_size({wg})
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
    let tile_n0 = wid.x * {tile_n}u;
    let lane = lid.x;
    let n_sub = lane / 16u;
    let m_sub = lane % 16u;
    let m0 = m_sub * 4u;
    let m1 = m0 + 1u;
    let m2 = m0 + 2u;
    let m3 = m0 + 3u;
    let n0 = tile_n0 + n_sub * 4u;
    let n1 = n0 + 1u;
    let n2 = n0 + 2u;
    let n3 = n0 + 3u;
    var acc = array<f32, 16>();
    for (var blk = 0u; blk < params.blocks_per_row; blk++) {{
        for (var hb = 0u; hb < {hb_count}u; hb++) {{
            // Stage: dequantize this column step's sub-blocks
            // into the tile (the assignment is K10's; the
            // per-sub-block math is dequant_q6_k's). Rows past
            // the tensor's end stage zeros.
            for (var rep = 0u; rep < {reps}u; rep++) {{
                let task = lane + rep * {wg}u;
                let n = task / {subs}u;
                let sub = task % {subs}u;
                let tbase = (n * {tile_k}u + sub * 32u) / 4u;
                let gn = tile_n0 + n;
                if (gn < params.out_rows) {{
                    let j = hb * {subs}u + sub;
                    let h = j / 4u;
                    let s = j % 4u;
                    let bw = (gn * params.blocks_per_row + blk) * {BLOCK_WORDS_DEV}u;
                    let d = f16_to_f32(weights[bw + 52u] & 0xffffu);
                    let scw = array<u32, 4>(
                        weights[bw + 48u], weights[bw + 49u],
                        weights[bw + 50u], weights[bw + 51u],
                    );
                    let ql_base = bw + h * 16u + (s & 1u) * 8u;
                    let qh_base = bw + 32u + h * 8u;
                    let shift = (s & 1u) * 2u + (s >> 1u) * 4u;
                    let high = s >= 2u;
                    for (var w = 0u; w < 8u; w++) {{
                        let qlb = unpack4xU8(weights[ql_base + w]);
                        let qhb = unpack4xU8(weights[qh_base + w]);
                        let nibs = select(qlb & vec4<u32>(0xfu), (qlb >> vec4<u32>(4u)) & vec4<u32>(0xfu), vec4<bool>(high));
                        let q = nibs | (((qhb >> vec4<u32>(shift)) & vec4<u32>(3u)) << vec4<u32>(4u));
                        let sc = scale_at(scw, h * 8u + w / 4u + 2u * s);
                        tile[tbase + w] = (d * sc) * (vec4<f32>(q) - vec4<f32>(32.0));
                    }}
                }} else {{
                    for (var w = 0u; w < 8u; w++) {{
                        tile[tbase + w] = vec4<f32>(0.0);
                    }}
                }}
            }}
            workgroupBarrier();
            let e_base = blk * 256u + hb * {tile_k}u;
            for (var kk = 0u; kk < {tile_k}u; kk += 4u) {{
{x_loads}{dots}            }}
            workgroupBarrier();
        }}
    }}
{stores}}}
"#
    )
}

/// One prepared K11 dispatch: a bind group plus grid
/// dimensions, built once for a (weights, input, output,
/// batch) combination and encodable any number of times.
#[derive(Debug)]
pub struct K11Call {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    grid: (u32, u32),
}

impl K11Call {
    /// Record this call's dispatch into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k11 q6_k gemm"),
            timestamp_writes: None,
        });
        self.encode_into(&mut pass);
    }

    /// Record this dispatch into an already-open compute
    /// `pass`: the composition form the device-resident
    /// session uses to chain a whole forward into one pass.
    pub fn encode_into(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.grid.0, self.grid.1, 1);
    }
}

/// The compiled K11 pipeline, in one [`GemmShape`] and one
/// activation mode ([`ACT_NONE`] or [`ACT_SWIGLU`]).
#[derive(Debug)]
pub struct K11Pipeline {
    pipeline: wgpu::ComputePipeline,
    shape: GemmShape,
    act: u32,
}

impl K11Pipeline {
    /// Compile the shipped shape for `device` (see
    /// [`crate::kernels::k10_q4k_gemm::shipped_shape`]), or
    /// `None` when the device grants less workgroup storage
    /// than the smallest tile needs. See
    /// [`crate::kernels::k10_q4k_gemm::K10Pipeline::for_device`]
    /// for the fallback contract.
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] when a device that grants the
    /// storage rejects the shader.
    pub fn for_device(device: &wgpu::Device, act: u32) -> Result<Option<Self>> {
        match shipped_shape(&device.limits()) {
            Some(shape) => Ok(Some(Self::with_shape_act(device, shape, act)?)),
            None => {
                dbg_log!(
                    "k11 gemm unavailable: device grants {} bytes of workgroup storage, \
                     below the smallest tile",
                    device.limits().max_compute_workgroup_storage_size
                );
                Ok(None)
            }
        }
    }

    /// Compile an explicit [`GemmShape`] with an activation
    /// mode on `device`.
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] on a bad shape, an activation
    /// mode this kernel does not define, or insufficient
    /// workgroup storage.
    pub fn with_shape_act(device: &wgpu::Device, shape: GemmShape, act: u32) -> Result<Self> {
        shape.validate()?;
        if !matches!(act, ACT_NONE | ACT_SWIGLU) {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "k11 activation mode {act} is not one this kernel defines (plain and swiglu only)"
                ),
            });
        }
        let granted = device.limits().max_compute_workgroup_storage_size;
        if granted < shape.tile_bytes() {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "k11 shape {} stages {} bytes in workgroup memory but the device grants {granted}",
                    shape.name(),
                    shape.tile_bytes()
                ),
            });
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("k11 q6_k gemm shader"),
            source: wgpu::ShaderSource::Wgsl(gemm_shader(&shape, act).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k11 q6_k gemm pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        dbg_log!("k11 pipeline built: shape {} act {act}", shape.name());
        Ok(K11Pipeline {
            pipeline,
            shape,
            act,
        })
    }

    /// The shape this pipeline runs.
    #[must_use]
    pub fn shape(&self) -> GemmShape {
        self.shape
    }

    /// The activation mode this pipeline runs.
    #[must_use]
    pub fn act(&self) -> u32 {
        self.act
    }

    /// Prepare a plain GEMM dispatch of `weights` over a
    /// batch of input rows, against the same `[batch][width]`
    /// buffer layout K2's prepares use. See
    /// [`crate::kernels::k10_q4k_gemm::K10Pipeline::prepare`]
    /// for the contract.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] on an activation-carrying
    /// pipeline, an out-of-range batch, or undersized buffers.
    pub fn prepare(
        &self,
        ctx: &GpuContext,
        weights: &K2Weights,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        batch: u32,
    ) -> Result<K11Call> {
        if self.act != ACT_NONE {
            return Err(WgpuError::InvalidTensor {
                name: "k11 call".to_string(),
                detail: format!(
                    "this pipeline carries activation mode {}; prepare through prepare_act",
                    self.act
                ),
            });
        }
        self.prepare_inner(ctx, weights, input, None, output, batch, None)
    }

    /// Prepare an activation GEMM dispatch (the SwiGLU form):
    /// stream A transformed against stream B before any dot
    /// is taken, with K10's geometry contract (see
    /// [`crate::kernels::k10_q4k_gemm::K10Pipeline::prepare_act`]).
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when this pipeline carries
    /// no activation mode, a stride or offset is not a whole
    /// vec4 or does not cover the width, or a buffer is
    /// smaller than the geometry addresses.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_act(
        &self,
        ctx: &GpuContext,
        weights: &K2Weights,
        input_a: &wgpu::Buffer,
        input_b: &wgpu::Buffer,
        output: &wgpu::Buffer,
        batch: u32,
        geom: &K1ActGeom,
    ) -> Result<K11Call> {
        if self.act == ACT_NONE {
            return Err(WgpuError::InvalidTensor {
                name: "k11 call".to_string(),
                detail: "this pipeline carries no activation mode; prepare through prepare"
                    .to_string(),
            });
        }
        let width = weights.in_features;
        let quad = |v: usize| v != 0 && v % 4 == 0;
        if !quad(geom.a_stride_elems) || geom.a_stride_elems < width {
            return Err(WgpuError::InvalidTensor {
                name: "k11 act call".to_string(),
                detail: format!(
                    "stream A stride {} is not a whole number of vec4s covering the width {width}",
                    geom.a_stride_elems
                ),
            });
        }
        if !quad(geom.b_stride_elems)
            || geom.b_off_elems % 4 != 0
            || geom.b_off_elems + width > geom.b_stride_elems
        {
            return Err(WgpuError::InvalidTensor {
                name: "k11 act call".to_string(),
                detail: format!(
                    "stream B segment (stride {}, offset {}) does not address a whole \
                     {width}-wide segment in whole vec4s",
                    geom.b_stride_elems, geom.b_off_elems
                ),
            });
        }
        let rows = batch as usize;
        let need_a = ((rows - 1) * geom.a_stride_elems + width) as u64 * 4;
        let need_b = ((rows - 1) * geom.b_stride_elems + geom.b_off_elems + width) as u64 * 4;
        let need_out = rows as u64 * weights.out_rows as u64 * 4;
        if input_a.size() < need_a || input_b.size() < need_b || output.size() < need_out {
            return Err(WgpuError::InvalidTensor {
                name: "k11 act call".to_string(),
                detail: format!(
                    "buffer sizes (a {}, b {}, out {}) are below the geometry's needs \
                     ({need_a}, {need_b}, {need_out})",
                    input_a.size(),
                    input_b.size(),
                    output.size()
                ),
            });
        }
        self.prepare_inner(
            ctx,
            weights,
            input_a,
            Some(input_b),
            output,
            batch,
            Some(geom),
        )
    }

    /// The shared prepare: bind the weights (the K2 upload's
    /// one resident buffer), the streams, and the output, and
    /// write the uniform block.
    #[allow(clippy::too_many_arguments)]
    fn prepare_inner(
        &self,
        ctx: &GpuContext,
        weights: &K2Weights,
        input: &wgpu::Buffer,
        input_b: Option<&wgpu::Buffer>,
        output: &wgpu::Buffer,
        batch: u32,
        geom: Option<&K1ActGeom>,
    ) -> Result<K11Call> {
        if batch == 0 || batch > GEMM_MAX_BATCH {
            return Err(WgpuError::InvalidTensor {
                name: "k11 call".to_string(),
                detail: format!(
                    "batch {batch} is outside 1..={GEMM_MAX_BATCH} (the batch tile's row count)"
                ),
            });
        }
        let tiles = (weights.out_rows as u64).div_ceil(u64::from(self.shape.tile_n));
        if tiles > 65_535 {
            return Err(WgpuError::InvalidTensor {
                name: "k11 call".to_string(),
                detail: format!(
                    "{} output rows need {tiles} tile workgroups, past one dispatch dimension",
                    weights.out_rows
                ),
            });
        }
        let grid = (tiles as u32, 1);
        let blocks_per_row = (weights.in_features / BLOCK_ELEMS) as u32;
        let (a_stride_elems, b_stride_elems, b_off_elems) = match geom {
            Some(g) => (
                g.a_stride_elems as u32,
                g.b_stride_elems as u32,
                g.b_off_elems as u32,
            ),
            None => (weights.in_features as u32, 0, 0),
        };
        let params: [u32; 6] = [
            blocks_per_row,
            weights.out_rows as u32,
            batch,
            a_stride_elems,
            b_stride_elems,
            b_off_elems,
        ];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k11 params"),
            size: param_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&params_buffer, 0, &param_bytes);
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: weights.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ];
        if let Some(b) = input_b {
            entries.push(wgpu::BindGroupEntry {
                binding: 4,
                resource: b.as_entire_binding(),
            });
        }
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k11 bind group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        Ok(K11Call {
            pipeline: self.pipeline.clone(),
            bind_group,
            grid,
        })
    }

    /// Convenience one-shot: upload `input` ([`batch`, in]
    /// f32 rows), run the GEMM, read the [`batch`, out] result
    /// back. See
    /// [`crate::kernels::k10_q4k_gemm::K10Pipeline::run`].
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when `input` is not a whole
    /// number of input rows or the batch is out of range;
    /// [`WgpuError::Readback`] when the result cannot be mapped
    /// back.
    pub fn run(&self, ctx: &GpuContext, weights: &K2Weights, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() || input.len() % weights.in_features != 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k11 input".to_string(),
                detail: format!(
                    "input holds {} elements, not a whole number of {}-wide rows",
                    input.len(),
                    weights.in_features
                ),
            });
        }
        let batch = (input.len() / weights.in_features) as u32;
        let in_bytes: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k11 input"),
            size: in_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&input_buffer, 0, &in_bytes);
        let out_elems = batch as usize * weights.out_rows;
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k11 output"),
            size: (out_elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let call = self.prepare(ctx, weights, &input_buffer, &output_buffer, batch)?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k11 run"),
            });
        call.encode(&mut encoder);
        let out = ctx.read_f32(&output_buffer, out_elems)?;
        dbg_log!(
            "k11 run: batch={batch} out_rows={} bytes_streamed={}",
            weights.out_rows,
            weights.byte_len
        );
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_carries_the_shape_and_act() {
        let plain = gemm_shader(&GemmShape::new(64, 128), ACT_NONE);
        assert!(plain.contains("@workgroup_size(256)"));
        assert!(plain.contains("array<vec4<f32>, 2048>"));
        assert!(!plain.contains("input_b"));
        assert!(plain.contains("* 53u"));
        let swiglu = gemm_shader(&GemmShape::new(64, 256), ACT_SWIGLU);
        assert!(swiglu.contains("silu1"));
        assert!(swiglu.contains("input_b"));
    }
}
