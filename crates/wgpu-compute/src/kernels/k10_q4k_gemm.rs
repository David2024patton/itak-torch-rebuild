//! K10: Q4_K matrix-matrix GEMM for prefill (the MMQ follow-on
//! bench/gpu-resident-2026-10-30.md names).
//!
//! The resident session's chunk plans used to replay the K1
//! GEMV chain per prompt row: the grid folds the batch into
//! the workgroup index, so a 64-row chunk streams every weight
//! tensor up to 64 times (L2 catches some of it; the measured
//! warm prefill was 69 to 78 tok/s against llama.cpp's 998).
//! This kernel is the tiled form: one workgroup owns a tile of
//! [`GemmShape::tile_n`] output rows and ALL batch rows of the
//! chunk, walks the input in super-block columns, and for each
//! column cooperatively dequantizes the tile's weights ONCE
//! into workgroup memory (f32), then applies the staged tile
//! to every batch row from registers. Weight DRAM traffic per
//! chunk drops from `batch` tensor streams to one; the
//! dequantization arithmetic (nibble unpacks, the packed
//! scale/min extraction) is paid once per tile instead of once
//! per batch row; and the inner loop is a dense f32 FMA stream
//! over staged weights and L2-resident activations.
//!
//! The math is K1's, unchanged: the staged values are exactly
//! `gguf::dequant::dequant_q4_k`'s (`d * scale * q - dmin *
//! min` per element, the scale/min pair from
//! `get_scale_min_k4`), and each output element is one f32 dot
//! over the row in super-block order, so results check against
//! the GEMV path within float reassociation noise (the dot's
//! grouping differs: staged-tile order is the same block order
//! K1 strides, but the four-wide partials combine differently,
//! which is precisely the reassociation the cross-path gates
//! already tolerate).
//!
//! The activation modes are K1's vocabulary ([`ACT_GATED`],
//! [`ACT_SWIGLU`], same second-stream addressing through
//! [`K1ActGeom`]): the transform applies to the activation
//! quads as they are loaded, per element, with K7's exact
//! expressions, before any dot is taken, so a chunk plan's
//! fused projections (the gated output projection, the
//! SwiGLU-fused down projection) dispatch through this kernel
//! with the same composition the GEMV plan used.
//!
//! Tile geometry is a [`GemmShape`] chosen from the device's
//! granted workgroup storage (see [`shipped_shape`]): the
//! staged tile is `tile_n` x `tile_k` f32 values, the batch
//! tile is a fixed 64 rows (the resident session's chunk cap),
//! and each thread owns a 4x4 accumulator sub-tile. A device
//! granting less than the smallest tile's storage gets no
//! GEMM pipeline at all ([`K10Pipeline::for_device`] returns
//! `None`), and the session keeps its GEMV chunk plans there:
//! the fallback is a composition fact the caller reads, never
//! a silent substitution inside a dispatch.

use crate::device::GpuContext;
use crate::error::{Result, WgpuError};
use crate::kernels::k1_q4k_gemv::{K1ActGeom, K1Weights, ACT_GATED, ACT_NONE, ACT_SWIGLU};

use crate::diag::dbg_log;

/// Elements per Q4_K super-block.
const BLOCK_ELEMS: usize = 256;
/// Words per Q4_K super-block (144 bytes).
const BLOCK_WORDS: u32 = 36;

/// Batch rows one workgroup covers: every row of the chunk.
/// The resident session caps chunks at 64 rows
/// (`session::MAX_CHUNK_ROWS`); a prepare past this refuses by
/// name rather than silently truncating the batch.
pub const GEMM_MAX_BATCH: u32 = 64;

/// The GEMM tile geometry: output rows per workgroup, staged
/// input elements per dequant step, and the workgroup size
/// the first two imply (each thread owns a 4x4 sub-tile of
/// the 64-row batch tile, so the thread count is
/// `(tile_n / 4) * 16`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmShape {
    /// Output rows one workgroup computes (the staged tile's
    /// height).
    pub tile_n: u32,
    /// Input elements staged per dequant step (the staged
    /// tile's width): one half or one whole super-block
    /// column.
    pub tile_k: u32,
    /// Threads per workgroup: `(tile_n / 4) * (64 / 4)`.
    pub workgroup: u32,
}

impl GemmShape {
    /// A shape with the given tile; the workgroup size is
    /// derived (see the type docs).
    #[must_use]
    pub const fn new(tile_n: u32, tile_k: u32) -> GemmShape {
        GemmShape {
            tile_n,
            tile_k,
            workgroup: tile_n * 4,
        }
    }

    /// Workgroup storage the staged tile occupies, in bytes.
    #[must_use]
    pub const fn tile_bytes(&self) -> u32 {
        self.tile_n * self.tile_k * 4
    }

    /// The bench/check name: `t<tile_n>x<tile_k>`.
    #[must_use]
    pub fn name(&self) -> String {
        format!("t{}x{}", self.tile_n, self.tile_k)
    }

    /// Parse a bench/check tile name (see [`GemmShape::name`]).
    #[must_use]
    pub fn from_name(name: &str) -> Option<GemmShape> {
        let rest = name.strip_prefix('t')?;
        let (n, k) = rest.split_once('x')?;
        Some(GemmShape::new(n.parse().ok()?, k.parse().ok()?))
    }

    /// Check the shape's geometry rules (the storage check is
    /// the caller's, against the device it will run on).
    pub(crate) fn validate(&self) -> Result<()> {
        if !matches!(self.tile_n, 32 | 64) {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "gemm shape {}: tile_n {} is not 32 or 64",
                    self.name(),
                    self.tile_n
                ),
            });
        }
        if !matches!(self.tile_k, 128 | 256) {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "gemm shape {}: tile_k {} is not 128 or 256",
                    self.name(),
                    self.tile_k
                ),
            });
        }
        Ok(())
    }
}

/// The tile geometry a device with these limits runs: the
/// widest staged tile its granted workgroup storage holds.
/// The candidates, widest first: a 64x256 tile (64 KiB, the
/// RADV grant), a 64x128 tile (32 KiB), and a 32x128 tile
/// (16 KiB, the floor two of the three session shapes need).
/// `None` below the floor: no GEMM on that device (see the
/// module docs for the fallback contract).
#[must_use]
pub fn shipped_shape(limits: &wgpu::Limits) -> Option<GemmShape> {
    let storage = limits.max_compute_workgroup_storage_size;
    for shape in [
        GemmShape::new(64, 256),
        GemmShape::new(64, 128),
        GemmShape::new(32, 128),
    ] {
        if storage >= shape.tile_bytes() {
            return Some(shape);
        }
    }
    None
}

/// The activation-fusion declarations for `act`, textually
/// K1's (the second input stream at binding 4, K7's sigmoid
/// and SiLU in their exact forms, and the per-quad apply the
/// loader calls). Empty for the plain GEMM.
fn act_decls(act: u32) -> &'static str {
    match act {
        ACT_GATED => {
            r#"
@group(0) @binding(4) var<storage, read> input_b: array<vec4<f32>>;

// Logistic sigmoid, K7's branch-stable form (cpu-compute's):
// the exponential's argument is never positive.
fn sigmoid1(x: f32) -> f32 {
    if (x >= 0.0) {
        return 1.0 / (1.0 + exp(-x));
    }
    let e = exp(x);
    return e / (1.0 + e);
}

fn act_apply(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.x * sigmoid1(b.x), a.y * sigmoid1(b.y),
                     a.z * sigmoid1(b.z), a.w * sigmoid1(b.w));
}
"#
        }
        ACT_SWIGLU => {
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
        }
        _ => "",
    }
}

/// The Q4_K GEMM shader for `shape` and `act`, generated with
/// the tile geometry as literals so the compiler unrolls the
/// sub-tile loops and sizes the staged tile exactly. The body
/// has three stages per super-block column (see the module
/// docs): cooperative dequant into the tile, a barrier, and
/// the register inner loop over the tile; a second barrier
/// keeps the next column's dequant from overwriting a tile
/// still being read.
fn gemm_shader(shape: &GemmShape, act: u32) -> String {
    let tile_n = shape.tile_n;
    let tile_k = shape.tile_k;
    let wg = shape.workgroup;
    let tile_vec4s = tile_n * tile_k / 4;
    // Sub-blocks (32 elements) staged per column step, and
    // how many column steps one super-block takes.
    let subs = tile_k / 32;
    let hb_count = 256 / tile_k;
    // Dequant tasks per step: one per (tile row, sub-block);
    // each thread takes this many, strided by the workgroup.
    let reps = (tile_n * subs) / wg;
    let decls = act_decls(act);

    // The activation loads for the thread's four batch rows,
    // unrolled: a guarded vec4 load per row, with the fused
    // transform applied when this pipeline carries one.
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
    // The inner product: for each of the thread's four output
    // rows, one staged vec4 load and four dots.
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
    // The output store, guarded per element.
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

// One of the 12 packed scale bytes of a super-block.
fn scale_byte(sw: array<u32, 3>, i: u32) -> u32 {{
    return (sw[i / 4u] >> (8u * (i % 4u))) & 0xffu;
}}

// ggml get_scale_min_k4: sub-block j's (scale, min) 6-bit pair.
fn scale_min_k4(j: u32, sw: array<u32, 3>) -> vec2<u32> {{
    if (j < 4u) {{
        return vec2<u32>(scale_byte(sw, j) & 63u, scale_byte(sw, j + 4u) & 63u);
    }}
    return vec2<u32>(
        (scale_byte(sw, j + 4u) & 0xfu) | ((scale_byte(sw, j - 4u) >> 6u) << 4u),
        (scale_byte(sw, j + 4u) >> 4u) | ((scale_byte(sw, j) >> 6u) << 4u),
    );
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
            // into the tile. Task `lane + rep * wg` owns tile
            // row `task / subs` and sub-block `task % subs`
            // (element offset sub * 32 within the step); the
            // global sub-block index within the super-block is
            // hb * subs + sub. Rows past the tensor's end
            // stage zeros (their outputs are never stored).
            for (var rep = 0u; rep < {reps}u; rep++) {{
                let task = lane + rep * {wg}u;
                let n = task / {subs}u;
                let sub = task % {subs}u;
                let tbase = (n * {tile_k}u + sub * 32u) / 4u;
                let gn = tile_n0 + n;
                if (gn < params.out_rows) {{
                    let j = hb * {subs}u + sub;
                    let g = j / 2u;
                    let bw = (gn * params.blocks_per_row + blk) * {BLOCK_WORDS}u;
                    let hw = weights[bw];
                    let d = f16_to_f32(hw & 0xffffu);
                    let dmin = f16_to_f32(hw >> 16u);
                    let sw = array<u32, 3>(weights[bw + 1u], weights[bw + 2u], weights[bw + 3u]);
                    let sm = scale_min_k4(j, sw);
                    let dsc = d * f32(sm.x);
                    let dmn = dmin * f32(sm.y);
                    let qbase = bw + 4u + g * 8u;
                    let hi = (j & 1u) == 1u;
                    for (var w = 0u; w < 8u; w++) {{
                        let word = weights[qbase + w];
                        let nibs = unpack4xU8(select(word & 0x0f0f0f0fu, (word >> 4u) & 0x0f0f0f0fu, hi));
                        tile[tbase + w] = dsc * vec4<f32>(nibs) - vec4<f32>(dmn);
                    }}
                }} else {{
                    for (var w = 0u; w < 8u; w++) {{
                        tile[tbase + w] = vec4<f32>(0.0);
                    }}
                }}
            }}
            workgroupBarrier();
            // Inner: every thread applies the staged tile to
            // its 4x4 sub-tile of outputs.
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

/// One prepared K10 dispatch: a bind group plus grid
/// dimensions, built once for a (weights, input, output,
/// batch) combination and encodable any number of times.
#[derive(Debug)]
pub struct K10Call {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    grid: (u32, u32),
}

impl K10Call {
    /// Record this call's dispatch into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k10 q4_k gemm"),
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

/// The compiled K10 pipeline, in one [`GemmShape`] and one
/// activation mode (K1's vocabulary). Compilation happens at
/// backend construction, like every kernel in this crate.
#[derive(Debug)]
pub struct K10Pipeline {
    pipeline: wgpu::ComputePipeline,
    shape: GemmShape,
    act: u32,
}

impl K10Pipeline {
    /// Compile the shipped shape for `device` (see
    /// [`shipped_shape`]), or `None` when the device grants
    /// less workgroup storage than the smallest tile needs:
    /// the caller (the backend) records the absence and the
    /// session keeps its GEMV chunk plans on such a device.
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
                    "k10 gemm unavailable: device grants {} bytes of workgroup storage, \
                     below the smallest tile",
                    device.limits().max_compute_workgroup_storage_size
                );
                Ok(None)
            }
        }
    }

    /// Compile an explicit [`GemmShape`] with an activation
    /// mode on `device`. The shape is validated, and the
    /// device's granted workgroup storage must hold the tile:
    /// a shortfall is a named refusal here, not a shader
    /// validation failure later.
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] on a bad shape, an unknown
    /// activation mode, or insufficient workgroup storage.
    pub fn with_shape_act(device: &wgpu::Device, shape: GemmShape, act: u32) -> Result<Self> {
        shape.validate()?;
        if act > ACT_SWIGLU {
            return Err(WgpuError::Pipeline {
                detail: format!("k10 activation mode {act} is not one this kernel defines"),
            });
        }
        let granted = device.limits().max_compute_workgroup_storage_size;
        if granted < shape.tile_bytes() {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "k10 shape {} stages {} bytes in workgroup memory but the device grants {granted}",
                    shape.name(),
                    shape.tile_bytes()
                ),
            });
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("k10 q4_k gemm shader"),
            source: wgpu::ShaderSource::Wgsl(gemm_shader(&shape, act).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k10 q4_k gemm pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        dbg_log!("k10 pipeline built: shape {} act {act}", shape.name());
        Ok(K10Pipeline {
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
    /// batch of input rows: `input` holds `batch *
    /// weights.in_features` f32 and `output` holds `batch *
    /// weights.out_rows` f32, both caller-owned device buffers
    /// in the same `[batch][width]` layout the GEMV prepares
    /// against, so a plan can swap one kernel for the other
    /// without moving a byte.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when this pipeline carries
    /// an activation mode, the batch is zero or past
    /// [`GEMM_MAX_BATCH`], the output rows need more
    /// workgroups than one dispatch dimension holds, or a
    /// buffer is smaller than the geometry addresses.
    pub fn prepare(
        &self,
        ctx: &GpuContext,
        weights: &K1Weights,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        batch: u32,
    ) -> Result<K10Call> {
        if self.act != ACT_NONE {
            return Err(WgpuError::InvalidTensor {
                name: "k10 call".to_string(),
                detail: format!(
                    "this pipeline carries activation mode {}; prepare through prepare_act",
                    self.act
                ),
            });
        }
        self.prepare_inner(ctx, weights, input, None, output, batch, None)
    }

    /// Prepare an activation GEMM dispatch (see [`ACT_GATED`]
    /// and [`ACT_SWIGLU`]): the GEMM over stream A with K7's
    /// elementwise math applied to A's quads against stream B
    /// before any dot is taken. The geometry contract is
    /// K1's [`K1ActGeom`], with one relaxation: the strides
    /// and the offset need only be whole vec4s here (the GEMV
    /// addresses activations in whole super-blocks; the GEMM
    /// stages weights, so its activation addressing is
    /// element-precise to a vec4).
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when this pipeline carries
    /// no activation mode, the batch is zero or past
    /// [`GEMM_MAX_BATCH`], a stride or offset is not a whole
    /// vec4 or does not cover the width, or a buffer is
    /// smaller than the geometry addresses.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_act(
        &self,
        ctx: &GpuContext,
        weights: &K1Weights,
        input_a: &wgpu::Buffer,
        input_b: &wgpu::Buffer,
        output: &wgpu::Buffer,
        batch: u32,
        geom: &K1ActGeom,
    ) -> Result<K10Call> {
        if self.act == ACT_NONE {
            return Err(WgpuError::InvalidTensor {
                name: "k10 call".to_string(),
                detail: "this pipeline carries no activation mode; prepare through prepare"
                    .to_string(),
            });
        }
        let width = weights.in_features;
        let quad = |v: usize| v != 0 && v % 4 == 0;
        if !quad(geom.a_stride_elems) || geom.a_stride_elems < width {
            return Err(WgpuError::InvalidTensor {
                name: "k10 act call".to_string(),
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
                name: "k10 act call".to_string(),
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
                name: "k10 act call".to_string(),
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

    /// The shared prepare: bind the weights (the K1 upload's
    /// one resident buffer), the streams, and the output, and
    /// write the uniform block.
    #[allow(clippy::too_many_arguments)]
    fn prepare_inner(
        &self,
        ctx: &GpuContext,
        weights: &K1Weights,
        input: &wgpu::Buffer,
        input_b: Option<&wgpu::Buffer>,
        output: &wgpu::Buffer,
        batch: u32,
        geom: Option<&K1ActGeom>,
    ) -> Result<K10Call> {
        if batch == 0 || batch > GEMM_MAX_BATCH {
            return Err(WgpuError::InvalidTensor {
                name: "k10 call".to_string(),
                detail: format!(
                    "batch {batch} is outside 1..={GEMM_MAX_BATCH} (the batch tile's row count)"
                ),
            });
        }
        let tiles = (weights.out_rows as u64).div_ceil(u64::from(self.shape.tile_n));
        if tiles > u64::from(u32::MAX) || tiles > 65_535 {
            return Err(WgpuError::InvalidTensor {
                name: "k10 call".to_string(),
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
            label: Some("k10 params"),
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
            label: Some("k10 bind group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        Ok(K10Call {
            pipeline: self.pipeline.clone(),
            bind_group,
            grid,
        })
    }

    /// Convenience one-shot: upload `input` ([`batch`, in]
    /// f32 rows), run the GEMM, read the [`batch`, out] result
    /// back. The check/bench data path; the resident session
    /// prepares against its own buffers instead.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when `input` is not a whole
    /// number of input rows or the batch is out of range;
    /// [`WgpuError::Readback`] when the result cannot be mapped
    /// back.
    pub fn run(&self, ctx: &GpuContext, weights: &K1Weights, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() || input.len() % weights.in_features != 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k10 input".to_string(),
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
            label: Some("k10 input"),
            size: in_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&input_buffer, 0, &in_bytes);
        let out_elems = batch as usize * weights.out_rows;
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k10 output"),
            size: (out_elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let call = self.prepare(ctx, weights, &input_buffer, &output_buffer, batch)?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k10 run"),
            });
        call.encode(&mut encoder);
        let out = ctx.read_f32(&output_buffer, out_elems)?;
        dbg_log!(
            "k10 run: batch={batch} out_rows={} bytes_streamed={}",
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
    fn shape_names_round_trip() {
        for shape in [
            GemmShape::new(64, 256),
            GemmShape::new(64, 128),
            GemmShape::new(32, 128),
        ] {
            assert_eq!(GemmShape::from_name(&shape.name()), Some(shape));
        }
        assert_eq!(GemmShape::from_name("bogus"), None);
        assert_eq!(GemmShape::from_name("t64"), None);
    }

    #[test]
    fn shape_workgroups_follow_the_tile() {
        assert_eq!(GemmShape::new(64, 256).workgroup, 256);
        assert_eq!(GemmShape::new(64, 128).workgroup, 256);
        assert_eq!(GemmShape::new(32, 128).workgroup, 128);
        assert_eq!(GemmShape::new(64, 256).tile_bytes(), 65_536);
        assert_eq!(GemmShape::new(32, 128).tile_bytes(), 16_384);
    }

    #[test]
    fn shader_carries_the_shape_and_act() {
        let plain = gemm_shader(&GemmShape::new(64, 128), ACT_NONE);
        assert!(plain.contains("@workgroup_size(256)"));
        assert!(plain.contains("array<vec4<f32>, 2048>"));
        assert!(!plain.contains("input_b"));
        let gated = gemm_shader(&GemmShape::new(32, 128), ACT_GATED);
        assert!(gated.contains("@workgroup_size(128)"));
        assert!(gated.contains("input_b"));
        assert!(gated.contains("sigmoid1"));
        let swiglu = gemm_shader(&GemmShape::new(64, 256), ACT_SWIGLU);
        assert!(swiglu.contains("silu1"));
        assert!(swiglu.contains("array<vec4<f32>, 4096>"));
    }
}
