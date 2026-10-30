//! K12: the Q8_1 activation quantizer for the MMQ prefill
//! path (K13/K14), the on-the-fly half of llama.cpp's MMQ
//! design (ggml's `quantize_row_q8_1` plus the MMQ block
//! layout of ggml-cuda's `block_q8_1_mmq`).
//!
//! The f32-tile GEMM (K10/K11) is bound inside the workgroup:
//! every staged f32 element is re-read from workgroup memory
//! at roughly one FMA per byte, which priced the prefill
//! projections at ~5.7 TFLOPS effective against the card's
//! ~61 TFLOPS of FP32 peak (bench/gpu-resident-2026-10-30.md,
//! GEMM section). llama.cpp's answer is to never run the
//! product in f32 at all: activations are quantized to Q8_1
//! blocks of 32 (one f32 scale `d = amax / 127`, 32 int8
//! values, and the block sum precomputed), and the inner
//! product runs as integer dot products (DP4A) against the
//! weights' own integer payloads, with the scales folded back
//! per block at the store. This kernel is the quantization
//! half of that pair; K13 and K14 are the product halves.
//!
//! The quantized form is device-only (no file format carries
//! it), so the layout is this crate's: per (row, 32-element
//! block), eight u32 words of packed int8 quants (byte `i` of
//! word `w` is element `4w + i` of the block, exactly the lane
//! order `dot4I8Packed` pairs) and one `vec4<f32>` of
//! `(d, s, sh0, sh1)`: the block scale, the scaled whole-block
//! sum `d * sum(q)`, and the scaled half sums `d * sum(q)` over
//! elements 0..16 and 16..32. Q4_K's consumer folds the whole
//! sum against the super-block's mins (the `- dmin * min *
//! sum(x)` term of `vec_dot_q4_K_q8_1`); Q6_K's scales change
//! every 16 elements, so its consumer needs the half sums for
//! the `(q - 32)` offset term. ggml stores `d` and `s` as f16;
//! this layout keeps f32 (the values are produced and consumed
//! on the device within one chunk, and the extra bytes cost
//! nothing next to the weight stream).
//!
//! The activation vocabulary is K1's ([`ACT_GATED`],
//! [`ACT_SWIGLU`], same [`K1ActGeom`] addressing, K7's exact
//! expressions applied per element before quantization): the
//! MMQ consumers carry no activation forms of their own,
//! because the transform is finished here, upstream of the
//! product. That is the one structural difference from the
//! GEMM pair, whose activation stage lives inside the product
//! kernel.
//!
//! Contract: the width must be a whole number of 256-element
//! quanta (eight blocks) and the batch within
//! [`GEMM_MAX_BATCH`]; anything else refuses by name at
//! prepare, and the session's composition falls back to the
//! GEMM form (the fallback is the caller's, never a silent
//! substitution inside a dispatch). Rounding follows
//! `quantize_row_q8_1_ref`'s `roundf` (nearest, ties away from
//! zero) up to the shader `round` builtin's tie rule; a tie
//! disagreement moves one quant by one step, deep inside the
//! band the MMQ gates carry.

use crate::device::GpuContext;
use crate::error::{Result, WgpuError};
use crate::kernels::k10_q4k_gemm::GEMM_MAX_BATCH;
use crate::kernels::k1_q4k_gemv::{K1ActGeom, ACT_GATED, ACT_NONE, ACT_SWIGLU};

use crate::diag::dbg_log;

/// Elements per Q8_1 block (ggml's QK8_1).
pub const Q8_BLOCK_ELEMS: u32 = 32;
/// Packed u32 words per Q8_1 block (32 int8 values).
pub const Q8_BLOCK_WORDS: u32 = 8;
/// Elements one quantize workgroup covers: eight blocks, one
/// super-block's worth of a row.
pub const QUANT_CHUNK_ELEMS: u32 = 256;

/// Words of packed quants a quantized `[batch][width]` tensor
/// occupies.
#[must_use]
pub fn qs_words(batch: usize, width: usize) -> usize {
    batch * (width / Q8_BLOCK_ELEMS as usize) * Q8_BLOCK_WORDS as usize
}

/// `vec4<f32>` scale entries a quantized `[batch][width]`
/// tensor occupies (one per block).
#[must_use]
pub fn ds_vec4s(batch: usize, width: usize) -> usize {
    batch * (width / Q8_BLOCK_ELEMS as usize)
}

/// Quantize one row of f32 activations to this module's
/// device layout on the host: the reference the GPU kernel is
/// gated against, and ggml's `quantize_row_q8_1_ref` math
/// (amax over each 32-block, `d = amax / 127`, round-half-away
/// via `f32::round`, sums over the rounded quants) with the
/// half sums this layout adds. Returns the packed words and
/// the flat `(d, s, sh0, sh1)` quads in row order.
///
/// # Panics
///
/// When `x` is not a whole number of 32-element blocks.
#[must_use]
pub fn quantize_row_q8_1(x: &[f32]) -> (Vec<u32>, Vec<f32>) {
    assert!(
        x.len() % Q8_BLOCK_ELEMS as usize == 0,
        "quantize_row_q8_1: {} elements is not a whole number of blocks",
        x.len()
    );
    let blocks = x.len() / Q8_BLOCK_ELEMS as usize;
    let mut qs = vec![0u32; blocks * Q8_BLOCK_WORDS as usize];
    let mut ds = vec![0.0f32; blocks * 4];
    for (b, block) in x.chunks_exact(Q8_BLOCK_ELEMS as usize).enumerate() {
        let amax = block.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let mut q = [0i32; 32];
        let (mut sh0, mut sh1) = (0i32, 0i32);
        for (i, v) in block.iter().enumerate() {
            q[i] = (v * id).round() as i32;
            if i < 16 {
                sh0 += q[i];
            } else {
                sh1 += q[i];
            }
        }
        for w in 0..Q8_BLOCK_WORDS as usize {
            let bytes: [u8; 4] = std::array::from_fn(|i| (q[w * 4 + i] as i8) as u8);
            qs[b * Q8_BLOCK_WORDS as usize + w] = u32::from_le_bytes(bytes);
        }
        ds[b * 4] = d;
        ds[b * 4 + 1] = d * (sh0 + sh1) as f32;
        ds[b * 4 + 2] = d * sh0 as f32;
        ds[b * 4 + 3] = d * sh1 as f32;
    }
    (qs, ds)
}

/// The activation declarations for `act`, textually K10's
/// with a scalar apply: the second input stream at binding 4
/// and K7's sigmoid / SiLU in their exact forms. Empty for
/// the plain quantizer.
fn act_decls(act: u32) -> &'static str {
    match act {
        ACT_GATED => {
            r#"
@group(0) @binding(4) var<storage, read> input_b: array<f32>;

// Logistic sigmoid, K7's branch-stable form (cpu-compute's):
// the exponential's argument is never positive.
fn sigmoid1(x: f32) -> f32 {
    if (x >= 0.0) {
        return 1.0 / (1.0 + exp(-x));
    }
    let e = exp(x);
    return e / (1.0 + e);
}

fn act_apply(a: f32, b: f32) -> f32 {
    return a * sigmoid1(b);
}
"#
        }
        ACT_SWIGLU => {
            r#"
@group(0) @binding(4) var<storage, read> input_b: array<f32>;

// SiLU, K7's quotient form: x * sigmoid(x) as the single
// quotient the reference computes.
fn silu1(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

fn act_apply(g: f32, u: f32) -> f32 {
    return silu1(g) * u;
}
"#
        }
        _ => "",
    }
}

/// The quantizer shader for `act`. One workgroup of 256
/// threads owns one 256-element quantum of one row: thread
/// `lane` loads element `lane`, the per-32-block amax reduces
/// through workgroup memory inside each 32-thread group, and
/// the block's first thread writes the scale quad while every
/// fourth thread packs its four quants into one word.
fn quant_shader(act: u32) -> String {
    let decls = act_decls(act);
    let apply = if act == ACT_NONE {
        String::new()
    } else {
        "    x = act_apply(x, input_b[row * params.b_stride_elems + params.b_off_elems + e]);\n"
            .to_string()
    };
    format!(
        r#"struct Params {{
    width: u32,
    a_stride_elems: u32,
    b_stride_elems: u32,
    b_off_elems: u32,
}}

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> qs_out: array<u32>;
@group(0) @binding(2) var<storage, read_write> ds_out: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: Params;
{decls}
var<workgroup> shm_max: array<f32, 256>;
var<workgroup> shm_q: array<i32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
    let row = wid.y;
    let lane = lid.x;
    let block = lane / 32u;
    let l = lane % 32u;
    let e = wid.x * 256u + lane;
    var x = input_a[row * params.a_stride_elems + e];
{apply}    shm_max[lane] = abs(x);
    workgroupBarrier();
    // The block amax: a tree reduction inside each group of
    // 32 lanes (every lane reads its own group's slots, so
    // the strided writes stay inside the group).
    for (var s = 16u; s > 0u; s >>= 1u) {{
        if (l < s) {{
            shm_max[lane] = max(shm_max[lane], shm_max[lane + s]);
        }}
        workgroupBarrier();
    }}
    let amax = shm_max[block * 32u];
    let d = amax / 127.0;
    let id = select(0.0, 1.0 / d, d > 0.0);
    shm_q[lane] = i32(round(x * id));
    workgroupBarrier();
    let blocks_per_row = params.width / 32u;
    let blk = row * blocks_per_row + wid.x * 8u + block;
    if (l == 0u) {{
        var sh0 = 0i;
        var sh1 = 0i;
        for (var t = 0u; t < 16u; t++) {{ sh0 += shm_q[block * 32u + t]; }}
        for (var t = 16u; t < 32u; t++) {{ sh1 += shm_q[block * 32u + t]; }}
        ds_out[blk] = vec4<f32>(d, d * f32(sh0 + sh1), d * f32(sh0), d * f32(sh1));
    }}
    if (l % 4u == 0u) {{
        qs_out[blk * 8u + l / 4u] = pack4xI8(vec4<i32>(
            shm_q[lane], shm_q[lane + 1u], shm_q[lane + 2u], shm_q[lane + 3u]));
    }}
}}
"#
    )
}

/// One prepared K12 dispatch: a bind group plus grid
/// dimensions, built once for a (streams, quantized output,
/// batch, width) combination and encodable any number of
/// times.
#[derive(Debug)]
pub struct K12QuantCall {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    grid: (u32, u32),
}

impl K12QuantCall {
    /// Record this call's dispatch into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k12 q8_1 quantize"),
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

/// The compiled K12 pipeline, in one activation mode (K1's
/// vocabulary). Compilation happens at backend construction,
/// like every kernel in this crate; the quantizer's workgroup
/// footprint is two 256-entry arrays, so no device class
/// refuses it for storage.
#[derive(Debug)]
pub struct K12QuantPipeline {
    pipeline: wgpu::ComputePipeline,
    act: u32,
}

impl K12QuantPipeline {
    /// Compile the quantizer in activation mode `act`.
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] on an unknown activation mode
    /// or a shader the device rejects.
    pub fn new(device: &wgpu::Device, act: u32) -> Result<Self> {
        if act > ACT_SWIGLU {
            return Err(WgpuError::Pipeline {
                detail: format!("k12 activation mode {act} is not one this kernel defines"),
            });
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("k12 q8_1 quantize shader"),
            source: wgpu::ShaderSource::Wgsl(quant_shader(act).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k12 q8_1 quantize pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        dbg_log!("k12 quantize pipeline built: act {act}");
        Ok(K12QuantPipeline { pipeline, act })
    }

    /// The activation mode this pipeline runs.
    #[must_use]
    pub fn act(&self) -> u32 {
        self.act
    }

    /// Prepare a plain quantization dispatch: `input` holds
    /// `batch` rows of `width` f32 (row stride `width`), and
    /// `qs` / `ds` receive the quantized form (see the module
    /// docs for the layout and [`qs_words`] / [`ds_vec4s`]
    /// for the sizes).
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when this pipeline
    /// carries an activation mode, the batch is out of
    /// range, the width is not a whole number of quanta, or
    /// a buffer is smaller than the geometry addresses.
    pub fn prepare(
        &self,
        ctx: &GpuContext,
        input: &wgpu::Buffer,
        qs: &wgpu::Buffer,
        ds: &wgpu::Buffer,
        batch: u32,
        width: usize,
    ) -> Result<K12QuantCall> {
        if self.act != ACT_NONE {
            return Err(WgpuError::InvalidTensor {
                name: "k12 call".to_string(),
                detail: format!(
                    "this pipeline carries activation mode {}; prepare through prepare_act",
                    self.act
                ),
            });
        }
        let geom = K1ActGeom {
            a_stride_elems: width,
            b_stride_elems: 0,
            b_off_elems: 0,
        };
        self.prepare_inner(ctx, input, None, qs, ds, batch, width, &geom)
    }

    /// Prepare an activation quantization dispatch (see
    /// [`ACT_GATED`] and [`ACT_SWIGLU`]): stream A's elements
    /// are transformed against stream B, with K1's
    /// [`K1ActGeom`] addressing, before quantization. The
    /// geometry contract is K10's (strides and offset whole
    /// vec4s, the B segment covering the width).
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when this pipeline
    /// carries no activation mode, the batch is out of
    /// range, the width is not a whole number of quanta, a
    /// stride or offset is not a whole vec4 or does not
    /// cover the width, or a buffer is smaller than the
    /// geometry addresses.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_act(
        &self,
        ctx: &GpuContext,
        input_a: &wgpu::Buffer,
        input_b: &wgpu::Buffer,
        qs: &wgpu::Buffer,
        ds: &wgpu::Buffer,
        batch: u32,
        width: usize,
        geom: &K1ActGeom,
    ) -> Result<K12QuantCall> {
        if self.act == ACT_NONE {
            return Err(WgpuError::InvalidTensor {
                name: "k12 call".to_string(),
                detail: "this pipeline carries no activation mode; prepare through prepare"
                    .to_string(),
            });
        }
        let quad = |v: usize| v != 0 && v % 4 == 0;
        if !quad(geom.a_stride_elems) || geom.a_stride_elems < width {
            return Err(WgpuError::InvalidTensor {
                name: "k12 act call".to_string(),
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
                name: "k12 act call".to_string(),
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
        if input_a.size() < need_a || input_b.size() < need_b {
            return Err(WgpuError::InvalidTensor {
                name: "k12 act call".to_string(),
                detail: format!(
                    "input buffers (a {}, b {}) are below the geometry's needs ({need_a}, {need_b})",
                    input_a.size(),
                    input_b.size()
                ),
            });
        }
        self.prepare_inner(ctx, input_a, Some(input_b), qs, ds, batch, width, geom)
    }

    /// The shared prepare: validate the batch/width contract,
    /// bind the streams and the quantized outputs, and write
    /// the uniform block.
    #[allow(clippy::too_many_arguments)]
    fn prepare_inner(
        &self,
        ctx: &GpuContext,
        input: &wgpu::Buffer,
        input_b: Option<&wgpu::Buffer>,
        qs: &wgpu::Buffer,
        ds: &wgpu::Buffer,
        batch: u32,
        width: usize,
        geom: &K1ActGeom,
    ) -> Result<K12QuantCall> {
        if batch == 0 || batch > GEMM_MAX_BATCH {
            return Err(WgpuError::InvalidTensor {
                name: "k12 call".to_string(),
                detail: format!(
                    "batch {batch} is outside 1..={GEMM_MAX_BATCH} (the MMQ batch tile's row count)"
                ),
            });
        }
        if width == 0 || width % QUANT_CHUNK_ELEMS as usize != 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k12 call".to_string(),
                detail: format!(
                    "width {width} is not a whole number of {QUANT_CHUNK_ELEMS}-element quanta"
                ),
            });
        }
        let need_qs = (qs_words(batch as usize, width) * 4) as u64;
        let need_ds = (ds_vec4s(batch as usize, width) * 16) as u64;
        if qs.size() < need_qs || ds.size() < need_ds {
            return Err(WgpuError::InvalidTensor {
                name: "k12 call".to_string(),
                detail: format!(
                    "quantized buffers (qs {}, ds {}) are below the geometry's needs \
                     ({need_qs}, {need_ds})",
                    qs.size(),
                    ds.size()
                ),
            });
        }
        let grid = ((width / QUANT_CHUNK_ELEMS as usize) as u32, batch);
        let params: [u32; 4] = [
            width as u32,
            geom.a_stride_elems as u32,
            geom.b_stride_elems as u32,
            geom.b_off_elems as u32,
        ];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k12 params"),
            size: param_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&params_buffer, 0, &param_bytes);
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: qs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: ds.as_entire_binding(),
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
            label: Some("k12 bind group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        Ok(K12QuantCall {
            pipeline: self.pipeline.clone(),
            bind_group,
            grid,
        })
    }

    /// Convenience one-shot: quantize `input` ([`batch`,
    /// width] f32 rows) and read the quantized form back as
    /// (packed words, flat scale quads). The check/bench data
    /// path; the resident session prepares against its own
    /// buffers instead.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when `input` is not a
    /// whole number of rows or the geometry is out of range;
    /// [`WgpuError::Readback`] when a result cannot be mapped
    /// back.
    pub fn run(
        &self,
        ctx: &GpuContext,
        input: &[f32],
        width: usize,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        if width == 0 || input.is_empty() || input.len() % width != 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k12 input".to_string(),
                detail: format!(
                    "input holds {} elements, not a whole number of {width}-wide rows",
                    input.len()
                ),
            });
        }
        let batch = input.len() / width;
        let in_bytes: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k12 input"),
            size: in_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&input_buffer, 0, &in_bytes);
        let qs_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k12 qs"),
            size: (qs_words(batch, width) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let ds_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k12 ds"),
            size: (ds_vec4s(batch, width) * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let call = self.prepare(
            ctx,
            &input_buffer,
            &qs_buffer,
            &ds_buffer,
            batch as u32,
            width,
        )?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k12 run"),
            });
        call.encode(&mut encoder);
        ctx.queue.submit([encoder.finish()]);
        let qs = read_u32(ctx, &qs_buffer, qs_words(batch, width))?;
        let ds = ctx.read_f32(&ds_buffer, ds_vec4s(batch, width) * 4)?;
        dbg_log!("k12 run: batch={batch} width={width}");
        Ok((qs, ds))
    }
}

/// Read `count` u32 values back from a storage buffer (the
/// `GpuContext` readback is f32-shaped; the words here are
/// bit patterns, so read them as words).
fn read_u32(ctx: &GpuContext, buffer: &wgpu::Buffer, count: usize) -> Result<Vec<u32>> {
    let floats = ctx.read_f32(buffer, count)?;
    Ok(floats.iter().map(|f| f.to_bits()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_reference_quantizes_like_ggml() {
        // A block whose amax is 127 * 2^-7, so d = 2^-7 and
        // the reciprocal is exact in f32: every value below
        // lands on an exact quantum.
        let mut x = vec![0.0f32; 256];
        x[0] = 0.992_187_5; // 127 quanta
        x[1] = -0.5; // -64 quanta
        x[2] = 0.25; // 32 quanta
        x[16] = -0.992_187_5; // -127 quanta, second half
        let (qs, ds) = quantize_row_q8_1(&x);
        assert_eq!(ds[0], 0.007_812_5);
        // Whole-block sum: 127 - 64 + 32 - 127 = -32; first
        // half: 127 - 64 + 32 = 95; second half: -127.
        assert_eq!(ds[1], 0.007_812_5 * -32.0);
        assert_eq!(ds[2], 0.007_812_5 * 95.0);
        assert_eq!(ds[3], 0.007_812_5 * -127.0);
        let word0 = qs[0].to_le_bytes();
        assert_eq!(word0, [127, (-64i8) as u8, 32, 0]);
    }

    #[test]
    fn host_reference_zero_block_is_zero() {
        let (qs, ds) = quantize_row_q8_1(&vec![0.0f32; 256]);
        assert!(qs.iter().all(|w| *w == 0));
        assert!(ds.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn shader_carries_the_act_vocabulary() {
        let plain = quant_shader(ACT_NONE);
        assert!(plain.contains("@workgroup_size(256)"));
        assert!(!plain.contains("input_b"));
        assert!(!plain.contains("dot4I8Packed"));
        let gated = quant_shader(ACT_GATED);
        assert!(gated.contains("input_b"));
        assert!(gated.contains("sigmoid1"));
        let swiglu = quant_shader(ACT_SWIGLU);
        assert!(swiglu.contains("silu1"));
    }
}
