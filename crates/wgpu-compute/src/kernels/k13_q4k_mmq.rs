//! K13: Q4_K matrix-matrix MMQ for prefill, the integer-dot
//! product half of llama.cpp's MMQ design (K12 is the
//! quantization half; bench/gpu-resident-2026-10-30.md's GEMM
//! section names this kernel as the lever).
//!
//! Where K10 stages each super-block column's weights
//! DEQUANTIZED into an f32 tile and runs the product as f32
//! FMAs over it (one FMA per staged byte re-read, ~5.7 TFLOPS
//! effective, workgroup-bound), K13 stages the weights as
//! their packed integer payloads and runs the product as
//! DP4A integer dots against K12's Q8_1 activations, the
//! design ggml-vulkan's `mul_mmq` runs through
//! `dotPacked4x8EXT` and ggml-cuda's MMQ through the tensor
//! cores. The math is `vec_dot_q4_K_q8_1`, term for term: per
//! 32-element sub-block `j` of a super-block,
//! `d * scale_j * (d_x * sumi_j) - dmin * min_j * s_j`, where
//! `sumi_j` is the exact integer dot of the weight nibbles
//! with the activation quants and `s_j` is K12's precomputed
//! scaled block sum. The integer dots are exact in i32 (a
//! sub-block dot is bounded by 32 * 15 * 127); only the
//! per-sub-block scale fold runs in f32, in super-block
//! order, so results check against a host reference running
//! the same quantized math within float reassociation noise
//! (and against the f32 GEMM within activation-quantization
//! error, a different and larger band the gates keep
//! separate).
//!
//! Staging, per super-block column: the tile's weight words
//! are packed once into workgroup memory (each raw qs word
//! yields the low-nibble word for sub-block `2g` and the
//! high-nibble word for `2g + 1`, which is exactly the packed
//! form `dot4I8Packed` consumes: `pack4xU8(unpack4xU8(w &
//! 0x0f0f0f0f))` is `w & 0x0f0f0f0f` itself), the tile's
//! `(d * scale, dmin * min)` pairs beside them, and the
//! chunk's activation words and `(d_x, s)` pairs for the
//! column next to those. The inner loop is then pure LDS
//! reads and DP4A: each thread owns a 4x4 sub-tile of
//! (batch row, output row), each staged weight word serves
//! its thread's four batch rows, and each staged activation
//! word its four output rows.
//!
//! The split-K form ([`K13Pipeline::prepare_split`]) runs
//! this same generated body with the reduction partitioned
//! across slices: on narrow outputs the plain grid is a
//! handful of workgroups (a 256-row output is 4 tiles of
//! 64) and the card idles while each workgroup walks the
//! whole K range serially. The split multiplies the grid
//! by the factor and folds f32 partials through
//! [`crate::kernels::splitk_reduce`]; routing and the
//! factor rule live in the backend's `WGPU_COMPUTE_MMQ_SPLITK`
//! resolution (default off; bench/splitk-mmq-2026-11-07.md).
//!
//! The activation transform (the gated output projection,
//! the SwiGLU-fused down projection) is NOT here: it is
//! finished in K12's quantizer, upstream of the product, so
//! this kernel has one form. The session selects K13 over
//! the f32 GEMM (K10) per projection when this pipeline
//! exists; a device granting less workgroup storage than the
//! smallest staging set gets no pipeline
//! ([`K13Pipeline::for_device`] returns `None`) and keeps
//! the GEMM form: the fallback is a composition fact the
//! caller reads, never a silent substitution inside a
//! dispatch.

use crate::device::GpuContext;
use crate::error::{Result, WgpuError};
use crate::kernels::k12_q8_quant::{ds_vec4s, qs_words};
use crate::kernels::k1_q4k_gemv::K1Weights;
use crate::kernels::splitk_reduce::{SplitkReduceCall, SplitkReducer, SPLITK_MAX};

use crate::diag::dbg_log;

/// Elements per Q4_K super-block.
const BLOCK_ELEMS: usize = 256;
/// Words per Q4_K super-block (144 bytes).
const BLOCK_WORDS: u32 = 36;
/// Batch rows one workgroup covers: every row of the chunk
/// (the resident session's chunk cap, K10's contract).
pub const MMQ_MAX_BATCH: u32 = 64;

/// The MMQ tile geometry: output rows per workgroup. Each
/// thread owns a 4x4 sub-tile, so the workgroup is
/// `(tile_n / 4) * 16 = tile_n * 4` threads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MmqShape {
    /// Output rows one workgroup computes (the staged weight
    /// tile's height).
    pub tile_n: u32,
    /// Threads per workgroup: `tile_n * 4`.
    pub workgroup: u32,
}

impl MmqShape {
    /// A shape with the given tile; the workgroup size is
    /// derived (see the type docs).
    #[must_use]
    pub const fn new(tile_n: u32) -> MmqShape {
        MmqShape {
            tile_n,
            workgroup: tile_n * 4,
        }
    }

    /// The bench/check name: `m<tile_n>`.
    #[must_use]
    pub fn name(&self) -> String {
        format!("m{}", self.tile_n)
    }

    /// Check the shape's geometry rules (the storage check
    /// is the caller's, against the device it will run on).
    pub(crate) fn validate(&self) -> Result<()> {
        if !matches!(self.tile_n, 32 | 64) {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "mmq shape {}: tile_n {} is not 32 or 64",
                    self.name(),
                    self.tile_n
                ),
            });
        }
        Ok(())
    }
}

/// Workgroup storage K13's staging occupies at `shape`, in
/// bytes: the packed weight words (`tile_n` x 64 words), the
/// activation words (batch cap x 64 words), the weight scale
/// pairs (`tile_n` x 8 vec2), and the activation scale pairs
/// (batch cap x 8 vec2).
#[must_use]
pub const fn q4k_staging_bytes(shape: &MmqShape) -> u32 {
    shape.tile_n * 64 * 4 + MMQ_MAX_BATCH * 65 * 4 + shape.tile_n * 8 * 8 + MMQ_MAX_BATCH * 8 * 8
}

/// The tile geometry a device with these limits runs: the
/// first candidate, in preference order, whose staging its
/// granted workgroup storage holds. The wide tile is
/// preferred on this kernel's own bench record (opposite
/// of the f32 GEMM's lesson): with the activation tile's
/// LDS bank skew in place, m64 beats the shipped GEMM by
/// 1.38x on the real file's largest Q4_K tensor and m32 by
/// a further margin, so m32 remains only as the shape for
/// devices whose grants fit it but not m64. `None` below
/// the narrow tile's need: no MMQ on that device (see the
/// module docs for the fallback contract).
#[must_use]
pub fn shipped_shape(limits: &wgpu::Limits) -> Option<MmqShape> {
    let storage = limits.max_compute_workgroup_storage_size;
    [MmqShape::new(64), MmqShape::new(32)]
        .into_iter()
        .find(|shape| storage >= q4k_staging_bytes(shape))
}

/// The Q4_K MMQ shader for `shape`, generated with the tile
/// geometry as literals so the compiler unrolls the sub-tile
/// loops and sizes the staging arrays exactly.
fn mmq_shader(shape: &MmqShape) -> String {
    mmq_shader_split(shape, false)
}

/// The split-K form of the Q4_K MMQ shader: the same
/// generated body with the reduction partitioned across
/// `wid.y` slices. Slice `s` walks a contiguous range of
/// super-block columns and writes its partial sums to slice
/// `s` of the output binding (a `[splits][batch][out_rows]`
/// partials buffer), which [`crate::kernels::splitk_reduce`]
/// folds into the real output. The split factor is a
/// runtime parameter, not a generated literal, so one
/// compiled pipeline serves every factor 2..=[`SPLITK_MAX`].
/// The drift contract is the reducer's module docs: same
/// exact integer dots, same per-sub-block fold order inside
/// each slice, reassociation-class drift against the
/// unsplit product from the regrouped f32 accumulation.
fn mmq_split_shader(shape: &MmqShape) -> String {
    mmq_shader_split(shape, true)
}

fn mmq_shader_split(shape: &MmqShape, split: bool) -> String {
    let tile_n = shape.tile_n;
    let wg = shape.workgroup;
    let w_words = tile_n * 64;
    let w_pairs = tile_n * 8;

    // The inner product, unrolled over the eight sub-blocks
    // at generation time: every staged-memory offset in the
    // sub-block body is then a compile-time constant, so the
    // loads fold into the LDS instructions' immediate
    // offsets and the compiler schedules the whole
    // super-block's dots as one block. (The first version
    // ran the sub-block loop at runtime and measured at the
    // f32 GEMM's rate; the per-load address arithmetic on a
    // runtime index is the difference.) Per sub-block: the
    // thread's four activation words and four weight words
    // per packed word, sixteen DP4A dots, then the
    // per-sub-block scale fold.
    let mut inner = String::new();
    for sub in 0..8u32 {
        let so = sub * 8;
        inner.push_str("        {\n            var isum = array<i32, 16>();\n");
        for w in 0..8 {
            for j in 0..4 {
                inner.push_str(&format!(
                    "            let xv{j}w{w} = x_qs[m{j} * 65u + {so}u + {w}u];\n"
                ));
            }
            for i in 0..4 {
                inner.push_str(&format!(
                    "            let wv{i}w{w} = w_qs[(n_sub * 4u + {i}u) * 64u + {so}u + {w}u];\n"
                ));
            }
            for i in 0..4 {
                for j in 0..4 {
                    inner.push_str(&format!(
                        "            isum[{}] += dot4I8Packed(wv{i}w{w}, xv{j}w{w});\n",
                        i * 4 + j
                    ));
                }
            }
        }
        for i in 0..4 {
            inner.push_str(&format!(
                "            let c{i} = w_sm[(n_sub * 4u + {i}u) * 8u + {sub}u];\n"
            ));
        }
        for j in 0..4 {
            inner.push_str(&format!(
                "            let dm{j} = x_ds[m{j} * 8u + {sub}u];\n"
            ));
        }
        for i in 0..4 {
            for j in 0..4 {
                inner.push_str(&format!(
                    "            acc[{}] += c{i}.x * (dm{j}.x * f32(isum[{}])) - c{i}.y * dm{j}.y;\n",
                    i * 4 + j,
                    i * 4 + j
                ));
            }
        }
        inner.push_str("        }\n");
    }
    // The output store, guarded per element (K10's form).
    // The split form indexes its partials buffer by slice
    // (wid.y) first: [splits][batch][out_rows].
    let mut stores = String::new();
    for i in 0..4 {
        for j in 0..4 {
            let index = if split {
                format!("(wid.y * params.batch + m{j}) * params.out_rows + n{i}")
            } else {
                format!("m{j} * params.out_rows + n{i}")
            };
            stores.push_str(&format!(
                "    if (n{i} < params.out_rows && m{j} < params.batch) {{\n        output[{index}] = acc[{}];\n    }}\n",
                i * 4 + j
            ));
        }
    }
    let params_extra = if split { "\n    splits: u32," } else { "" };
    // The split form walks only its slice's contiguous
    // super-block range: slice s covers
    // [s * per_split, min((s + 1) * per_split, blocks)).
    // A slice past the end (more splits than super-blocks)
    // walks an empty range and stores its zero
    // accumulators, which the reduction folds in as exact
    // zeros: any factor in range is correct for any width.
    let loop_open = if split {
        "    let per_split = (params.blocks_per_row + params.splits - 1u) / params.splits;\n    let blk_first = wid.y * per_split;\n    let blk_last = min(blk_first + per_split, params.blocks_per_row);\n    for (var blk = blk_first; blk < blk_last; blk++) {"
    } else {
        "    for (var blk = 0u; blk < params.blocks_per_row; blk++) {"
    };

    format!(
        r#"struct Params {{
    blocks_per_row: u32,
    out_rows: u32,
    batch: u32,
    blocks32: u32,{params_extra}
}}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> qs_in: array<u32>;
@group(0) @binding(2) var<storage, read> ds_in: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> w_qs: array<u32, {w_words}>;
// The activation tile is row-skewed (65 words per row, not
// 64): a wavefront's sixteen batch lanes read the same word
// of sixteen different rows, and at a 64-word row stride
// those addresses all land in the same LDS bank (a 16-way
// conflict on every activation load). The odd stride puts
// each lane's word in its own bank.
var<workgroup> x_qs: array<u32, 4160>;
var<workgroup> w_sm: array<vec2<f32>, {w_pairs}>;
var<workgroup> x_ds: array<vec2<f32>, 512>;

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
{loop_open}
        // Stage the tile's packed weight words and scale
        // pairs for this super-block column. Task (n, g)
        // owns output row n's four word-groups: group g's
        // eight raw words pack into sub-blocks 2g (low
        // nibbles) and 2g + 1 (high nibbles). Rows past the
        // tensor's end stage zeros.
        for (var task = lane; task < {tile_n}u * 4u; task += {wg}u) {{
            let n = task / 4u;
            let g = task % 4u;
            let gn = tile_n0 + n;
            if (gn < params.out_rows) {{
                let bw = (gn * params.blocks_per_row + blk) * {BLOCK_WORDS}u;
                let hw = weights[bw];
                let dd = f16_to_f32(hw & 0xffffu);
                let ddm = f16_to_f32(hw >> 16u);
                let sw = array<u32, 3>(weights[bw + 1u], weights[bw + 2u], weights[bw + 3u]);
                for (var h2 = 0u; h2 < 2u; h2++) {{
                    let j = g * 2u + h2;
                    let sm = scale_min_k4(j, sw);
                    w_sm[n * 8u + j] = vec2<f32>(dd * f32(sm.x), ddm * f32(sm.y));
                }}
                for (var w = 0u; w < 8u; w++) {{
                    let raw = weights[bw + 4u + g * 8u + w];
                    w_qs[(n * 8u + g * 2u) * 8u + w] = raw & 0x0f0f0f0fu;
                    w_qs[(n * 8u + g * 2u + 1u) * 8u + w] = (raw >> 4u) & 0x0f0f0f0fu;
                }}
            }} else {{
                for (var h2 = 0u; h2 < 2u; h2++) {{
                    w_sm[n * 8u + g * 2u + h2] = vec2<f32>(0.0);
                }}
                for (var w = 0u; w < 8u; w++) {{
                    w_qs[(n * 8u + g * 2u) * 8u + w] = 0u;
                    w_qs[(n * 8u + g * 2u + 1u) * 8u + w] = 0u;
                }}
            }}
        }}
        // Stage the chunk's activation words and (d, s)
        // pairs for this column: plain copies, coalesced.
        for (var flat = lane; flat < params.batch * 64u; flat += {wg}u) {{
            let m = flat / 64u;
            let jw = flat % 64u;
            x_qs[m * 65u + jw] = qs_in[(m * params.blocks32 + blk * 8u + jw / 8u) * 8u + jw % 8u];
        }}
        for (var flat = lane; flat < params.batch * 8u; flat += {wg}u) {{
            let m = flat / 8u;
            let j = flat % 8u;
            let v = ds_in[m * params.blocks32 + blk * 8u + j];
            x_ds[flat] = vec2<f32>(v.x, v.y);
        }}
        workgroupBarrier();
        // Inner: the eight sub-blocks, unrolled at
        // generation time (constant staged offsets).
{inner}        workgroupBarrier();
    }}
{stores}}}
"#
    )
}

/// One prepared K13 dispatch: a bind group plus grid
/// dimensions, built once for a (weights, quantized input,
/// output, batch) combination and encodable any number of
/// times.
#[derive(Debug)]
pub struct K13Call {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    grid: (u32, u32),
    /// Diag timestamp stamp (see `crate::timestamps`); `None`
    /// unless the context carries a recorder.
    diag: Option<crate::timestamps::DiagStamp>,
}

impl K13Call {
    /// Record this call's dispatch into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k13 q4_k mmq"),
            timestamp_writes: None,
        });
        self.encode_into(&mut pass);
    }

    /// Record this dispatch into an already-open compute
    /// `pass`: the composition form the device-resident
    /// session uses to chain a whole forward into one pass.
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

/// One prepared split-K K13 dispatch pair: the split
/// product (writing f32 partial sums to a scratch buffer
/// the prepare allocated) followed by the split-K
/// reduction into the caller's output. Encoded as one unit
/// so the session's plan sees a single projection call; the
/// two dispatches carry separate diag stamps
/// (`k13_q4k_mmq_split` and `splitk_reduce`) so a profile
/// run shows the split engaged.
#[derive(Debug)]
pub struct K13SplitCall {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    grid: (u32, u32),
    reduce: SplitkReduceCall,
    /// Diag timestamp stamp for the product dispatch (the
    /// reduction carries its own, inside `reduce`).
    diag: Option<crate::timestamps::DiagStamp>,
}

impl K13SplitCall {
    /// Record this call's dispatches into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k13 q4_k mmq split"),
            timestamp_writes: None,
        });
        self.encode_into(&mut pass);
    }

    /// Record the product and its reduction into an
    /// already-open compute `pass`, in that order (the
    /// reduction reads the product's partials; dispatches
    /// within one pass are ordered, the same composition
    /// the K12 quantize + product chain relies on).
    pub fn encode_into(&self, pass: &mut wgpu::ComputePass<'_>) {
        let slot = self.diag.as_ref().and_then(|d| d.begin(pass));
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.grid.0, self.grid.1, 1);
        if let (Some(d), Some(s)) = (&self.diag, slot) {
            d.end(pass, s);
        }
        self.reduce.encode_into(pass);
    }

    /// The split factor this call runs (its grid's slice
    /// count).
    #[must_use]
    pub fn splits(&self) -> u32 {
        self.grid.1
    }
}

/// The compiled K13 pipeline, in one [`MmqShape`].
/// Compilation happens at backend construction, like every
/// kernel in this crate. The split-K form's pipeline and
/// the shared reducer compile alongside the plain pipeline
/// regardless of the split switch, so a shader defect in
/// either form fails at construction, never at first use.
#[derive(Debug)]
pub struct K13Pipeline {
    pipeline: wgpu::ComputePipeline,
    split_pipeline: wgpu::ComputePipeline,
    reducer: SplitkReducer,
    shape: MmqShape,
}

impl K13Pipeline {
    /// Compile the shipped shape for `device` (see
    /// [`shipped_shape`]), or `None` when the device grants
    /// less workgroup storage than the smallest staging set
    /// needs: the caller (the backend) records the absence
    /// and the session keeps its GEMM chunk plans on such a
    /// device.
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] when a device that grants the
    /// storage rejects the shader.
    pub fn for_device(device: &wgpu::Device) -> Result<Option<Self>> {
        match shipped_shape(&device.limits()) {
            Some(shape) => Ok(Some(Self::with_shape(device, shape)?)),
            None => {
                dbg_log!(
                    "k13 mmq unavailable: device grants {} bytes of workgroup storage, \
                     below the smallest staging set",
                    device.limits().max_compute_workgroup_storage_size
                );
                Ok(None)
            }
        }
    }

    /// Compile an explicit [`MmqShape`] on `device`. The
    /// shape is validated, and the device's granted
    /// workgroup storage must hold the staging set: a
    /// shortfall is a named refusal here, not a shader
    /// validation failure later.
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] on a bad shape or
    /// insufficient workgroup storage.
    pub fn with_shape(device: &wgpu::Device, shape: MmqShape) -> Result<Self> {
        shape.validate()?;
        let granted = device.limits().max_compute_workgroup_storage_size;
        if granted < q4k_staging_bytes(&shape) {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "k13 shape {} stages {} bytes in workgroup memory but the device grants {granted}",
                    shape.name(),
                    q4k_staging_bytes(&shape)
                ),
            });
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("k13 q4_k mmq shader"),
            source: wgpu::ShaderSource::Wgsl(mmq_shader(&shape).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k13 q4_k mmq pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let split_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("k13 q4_k mmq split shader"),
            source: wgpu::ShaderSource::Wgsl(mmq_split_shader(&shape).into()),
        });
        let split_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k13 q4_k mmq split pipeline"),
            layout: None,
            module: &split_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let reducer = SplitkReducer::new(device)?;
        dbg_log!(
            "k13 mmq pipelines built: shape {} (plain and split-K forms)",
            shape.name()
        );
        Ok(K13Pipeline {
            pipeline,
            split_pipeline,
            reducer,
            shape,
        })
    }

    /// The shape this pipeline runs.
    #[must_use]
    pub fn shape(&self) -> MmqShape {
        self.shape
    }

    /// Prepare an MMQ dispatch of `weights` over a batch of
    /// K12-quantized input rows: `qs` / `ds` hold the
    /// quantized form of `batch` rows of
    /// `weights.in_features` f32 (K12's layout), and
    /// `output` holds `batch * weights.out_rows` f32, in the
    /// same `[batch][width]` layout the GEMV prepares
    /// against, so a plan can swap one form for the other
    /// without moving a byte.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when the batch is zero
    /// or past [`MMQ_MAX_BATCH`], the input width is not a
    /// whole number of super-blocks, the output rows need
    /// more workgroups than one dispatch dimension holds,
    /// or a buffer is smaller than the geometry addresses.
    pub fn prepare(
        &self,
        ctx: &GpuContext,
        weights: &K1Weights,
        qs: &wgpu::Buffer,
        ds: &wgpu::Buffer,
        output: &wgpu::Buffer,
        batch: u32,
    ) -> Result<K13Call> {
        if batch == 0 || batch > MMQ_MAX_BATCH {
            return Err(WgpuError::InvalidTensor {
                name: "k13 call".to_string(),
                detail: format!(
                    "batch {batch} is outside 1..={MMQ_MAX_BATCH} (the batch tile's row count)"
                ),
            });
        }
        if weights.in_features == 0 || weights.in_features % BLOCK_ELEMS != 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k13 call".to_string(),
                detail: format!(
                    "input width {} is not a whole number of {BLOCK_ELEMS}-element super-blocks",
                    weights.in_features
                ),
            });
        }
        let tiles = (weights.out_rows as u64).div_ceil(u64::from(self.shape.tile_n));
        if tiles > 65_535 {
            return Err(WgpuError::InvalidTensor {
                name: "k13 call".to_string(),
                detail: format!(
                    "{} output rows need {tiles} tile workgroups, past one dispatch dimension",
                    weights.out_rows
                ),
            });
        }
        let need_qs = (qs_words(batch as usize, weights.in_features) * 4) as u64;
        let need_ds = (ds_vec4s(batch as usize, weights.in_features) * 16) as u64;
        let need_out = batch as u64 * weights.out_rows as u64 * 4;
        if qs.size() < need_qs || ds.size() < need_ds || output.size() < need_out {
            return Err(WgpuError::InvalidTensor {
                name: "k13 call".to_string(),
                detail: format!(
                    "buffer sizes (qs {}, ds {}, out {}) are below the geometry's needs \
                     ({need_qs}, {need_ds}, {need_out})",
                    qs.size(),
                    ds.size(),
                    output.size()
                ),
            });
        }
        let grid = (tiles as u32, 1);
        let params: [u32; 4] = [
            (weights.in_features / BLOCK_ELEMS) as u32,
            weights.out_rows as u32,
            batch,
            (weights.in_features / 32) as u32,
        ];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k13 params"),
            size: param_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&params_buffer, 0, &param_bytes);
        let entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: weights.buffer().as_entire_binding(),
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
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buffer.as_entire_binding(),
            },
        ];
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k13 bind group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        Ok(K13Call {
            pipeline: self.pipeline.clone(),
            bind_group,
            grid,
            diag: ctx.diag_stamp(crate::timestamps::DispatchMeta {
                kernel: "k13_q4k_mmq",
                rows: weights.out_rows as u64,
                cols: weights.in_features as u64,
                quant: "Q4_K",
                weight_bytes: weights.byte_len,
                shape: format!("batch={batch} grid={}x{}", grid.0, grid.1),
            }),
        })
    }

    /// Prepare a split-K MMQ dispatch: [`Self::prepare`]'s
    /// geometry and buffers, with the reduction partitioned
    /// across `splits` workgroups per output tile. The
    /// partial sums land in a scratch buffer this prepare
    /// allocates (`splits * batch * out_rows` f32, owned by
    /// the returned call's bind groups) and the paired
    /// reduction writes the caller's `output`. Any factor
    /// in 2..=[`SPLITK_MAX`] is correct for any width: a
    /// factor past the super-block count leaves trailing
    /// slices empty (they store zeros), so callers may cap
    /// the factor by width for efficiency but never need to
    /// for correctness.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] on the same geometry
    /// refusals as [`Self::prepare`], or a split factor
    /// outside 2..=[`SPLITK_MAX`].
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_split(
        &self,
        ctx: &GpuContext,
        weights: &K1Weights,
        qs: &wgpu::Buffer,
        ds: &wgpu::Buffer,
        output: &wgpu::Buffer,
        batch: u32,
        splits: u32,
    ) -> Result<K13SplitCall> {
        if !(2..=SPLITK_MAX).contains(&splits) {
            return Err(WgpuError::InvalidTensor {
                name: "k13 split call".to_string(),
                detail: format!(
                    "splits {splits} is outside 2..={SPLITK_MAX} (a split of 1 is the unsplit product)"
                ),
            });
        }
        if batch == 0 || batch > MMQ_MAX_BATCH {
            return Err(WgpuError::InvalidTensor {
                name: "k13 split call".to_string(),
                detail: format!(
                    "batch {batch} is outside 1..={MMQ_MAX_BATCH} (the batch tile's row count)"
                ),
            });
        }
        if weights.in_features == 0 || weights.in_features % BLOCK_ELEMS != 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k13 split call".to_string(),
                detail: format!(
                    "input width {} is not a whole number of {BLOCK_ELEMS}-element super-blocks",
                    weights.in_features
                ),
            });
        }
        let tiles = (weights.out_rows as u64).div_ceil(u64::from(self.shape.tile_n));
        if tiles > 65_535 {
            return Err(WgpuError::InvalidTensor {
                name: "k13 split call".to_string(),
                detail: format!(
                    "{} output rows need {tiles} tile workgroups, past one dispatch dimension",
                    weights.out_rows
                ),
            });
        }
        let need_qs = (qs_words(batch as usize, weights.in_features) * 4) as u64;
        let need_ds = (ds_vec4s(batch as usize, weights.in_features) * 16) as u64;
        let need_out = batch as u64 * weights.out_rows as u64 * 4;
        if qs.size() < need_qs || ds.size() < need_ds || output.size() < need_out {
            return Err(WgpuError::InvalidTensor {
                name: "k13 split call".to_string(),
                detail: format!(
                    "buffer sizes (qs {}, ds {}, out {}) are below the geometry's needs \
                     ({need_qs}, {need_ds}, {need_out})",
                    qs.size(),
                    ds.size(),
                    output.size()
                ),
            });
        }
        let partials = ctx.scratch_buffer("k13 split partials", u64::from(splits) * need_out);
        let grid = (tiles as u32, splits);
        let params: [u32; 5] = [
            (weights.in_features / BLOCK_ELEMS) as u32,
            weights.out_rows as u32,
            batch,
            (weights.in_features / 32) as u32,
            splits,
        ];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k13 split params"),
            size: param_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&params_buffer, 0, &param_bytes);
        let entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: weights.buffer().as_entire_binding(),
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
                resource: partials.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buffer.as_entire_binding(),
            },
        ];
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k13 split bind group"),
            layout: &self.split_pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let reduce = self.reducer.prepare(
            ctx,
            &partials,
            output,
            batch as u64 * weights.out_rows as u64,
            splits,
            "Q4_K",
        )?;
        Ok(K13SplitCall {
            pipeline: self.split_pipeline.clone(),
            bind_group,
            grid,
            reduce,
            diag: ctx.diag_stamp(crate::timestamps::DispatchMeta {
                kernel: "k13_q4k_mmq_split",
                rows: weights.out_rows as u64,
                cols: weights.in_features as u64,
                quant: "Q4_K",
                weight_bytes: weights.byte_len,
                shape: format!("batch={batch} grid={}x{} splits={splits}", grid.0, grid.1),
            }),
        })
    }

    /// Convenience one-shot: run the MMQ over host-side
    /// quantized rows (`qs` packed words and flat `ds`
    /// quads, K12's layout for `batch` rows of the tensor's
    /// input width) and read the [`batch`, out] result back.
    /// The check/bench data path; the resident session
    /// prepares against its own buffers instead.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when the quantized
    /// inputs are not sized for the batch and width, or the
    /// batch is out of range; [`WgpuError::Readback`] when
    /// the result cannot be mapped back.
    pub fn run(
        &self,
        ctx: &GpuContext,
        weights: &K1Weights,
        qs: &[u32],
        ds: &[f32],
        batch: u32,
    ) -> Result<Vec<f32>> {
        if qs.len() != qs_words(batch as usize, weights.in_features)
            || ds.len() != ds_vec4s(batch as usize, weights.in_features) * 4
        {
            return Err(WgpuError::InvalidTensor {
                name: "k13 input".to_string(),
                detail: format!(
                    "quantized inputs hold {} words and {} scale values, not the \
                     {} words and {} values {batch} rows of width {} need",
                    qs.len(),
                    ds.len(),
                    qs_words(batch as usize, weights.in_features),
                    ds_vec4s(batch as usize, weights.in_features) * 4,
                    weights.in_features
                ),
            });
        }
        let qs_bytes: Vec<u8> = qs.iter().flat_map(|v| v.to_le_bytes()).collect();
        let qs_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k13 qs"),
            size: qs_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&qs_buffer, 0, &qs_bytes);
        let ds_bytes: Vec<u8> = ds.iter().flat_map(|v| v.to_le_bytes()).collect();
        let ds_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k13 ds"),
            size: ds_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&ds_buffer, 0, &ds_bytes);
        let out_elems = batch as usize * weights.out_rows;
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k13 output"),
            size: (out_elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let call = self.prepare(ctx, weights, &qs_buffer, &ds_buffer, &output_buffer, batch)?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k13 run"),
            });
        call.encode(&mut encoder);
        ctx.queue.submit([encoder.finish()]);
        let out = ctx.read_f32(&output_buffer, out_elems)?;
        dbg_log!(
            "k13 run: batch={batch} out_rows={} bytes_streamed={}",
            weights.out_rows,
            weights.byte_len
        );
        Ok(out)
    }

    /// Convenience one-shot for the split-K form: like
    /// [`Self::run`], with the reduction partitioned across
    /// `splits` workgroups per output tile. The check/bench
    /// data path for the split form.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] on the same conditions
    /// as [`Self::prepare_split`]; [`WgpuError::Readback`]
    /// when the result cannot be mapped back.
    pub fn run_split(
        &self,
        ctx: &GpuContext,
        weights: &K1Weights,
        qs: &[u32],
        ds: &[f32],
        batch: u32,
        splits: u32,
    ) -> Result<Vec<f32>> {
        if qs.len() != qs_words(batch as usize, weights.in_features)
            || ds.len() != ds_vec4s(batch as usize, weights.in_features) * 4
        {
            return Err(WgpuError::InvalidTensor {
                name: "k13 split input".to_string(),
                detail: format!(
                    "quantized inputs hold {} words and {} scale values, not the \
                     {} words and {} values {batch} rows of width {} need",
                    qs.len(),
                    ds.len(),
                    qs_words(batch as usize, weights.in_features),
                    ds_vec4s(batch as usize, weights.in_features) * 4,
                    weights.in_features
                ),
            });
        }
        let qs_bytes: Vec<u8> = qs.iter().flat_map(|v| v.to_le_bytes()).collect();
        let qs_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k13 split qs"),
            size: qs_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&qs_buffer, 0, &qs_bytes);
        let ds_bytes: Vec<u8> = ds.iter().flat_map(|v| v.to_le_bytes()).collect();
        let ds_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k13 split ds"),
            size: ds_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&ds_buffer, 0, &ds_bytes);
        let out_elems = batch as usize * weights.out_rows;
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k13 split output"),
            size: (out_elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let call = self.prepare_split(
            ctx,
            weights,
            &qs_buffer,
            &ds_buffer,
            &output_buffer,
            batch,
            splits,
        )?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k13 split run"),
            });
        call.encode(&mut encoder);
        ctx.queue.submit([encoder.finish()]);
        let out = ctx.read_f32(&output_buffer, out_elems)?;
        dbg_log!(
            "k13 split run: batch={batch} splits={splits} out_rows={} bytes_streamed={}",
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
    fn shape_names_and_staging() {
        assert_eq!(MmqShape::new(32).name(), "m32");
        assert_eq!(MmqShape::new(32).workgroup, 128);
        assert_eq!(MmqShape::new(64).workgroup, 256);
        assert_eq!(q4k_staging_bytes(&MmqShape::new(32)), 30_976);
        assert_eq!(q4k_staging_bytes(&MmqShape::new(64)), 41_216);
    }

    #[test]
    fn shipped_shape_respects_storage() {
        let at = |storage: u32| wgpu::Limits {
            max_compute_workgroup_storage_size: storage,
            ..Default::default()
        };
        assert_eq!(shipped_shape(&at(16_384)), None);
        assert_eq!(shipped_shape(&at(32_768)), Some(MmqShape::new(32)));
        assert_eq!(shipped_shape(&at(65_536)), Some(MmqShape::new(64)));
    }

    #[test]
    fn shader_carries_the_shape_and_dp4a() {
        let s32 = mmq_shader(&MmqShape::new(32));
        assert!(s32.contains("@workgroup_size(128)"));
        assert!(s32.contains("array<u32, 2048>"));
        assert!(s32.contains("dot4I8Packed"));
        let s64 = mmq_shader(&MmqShape::new(64));
        assert!(s64.contains("@workgroup_size(256)"));
        assert!(s64.contains("array<u32, 4096>"));
    }

    #[test]
    fn split_shader_partitions_the_reduction() {
        let s = mmq_split_shader(&MmqShape::new(64));
        // The split factor arrives as a runtime parameter.
        assert!(s.contains("splits: u32,"));
        // Each slice walks its contiguous super-block range.
        assert!(s.contains(
            "let per_split = (params.blocks_per_row + params.splits - 1u) / params.splits;"
        ));
        assert!(s.contains("let blk_first = wid.y * per_split;"));
        assert!(s.contains("for (var blk = blk_first; blk < blk_last; blk++)"));
        // Partials are indexed [slice][batch][out_row].
        assert!(s.contains("output[(wid.y * params.batch + m0) * params.out_rows + n0]"));
        // The staging and inner product are the plain
        // kernel's, unchanged.
        assert!(s.contains("dot4I8Packed"));
        assert!(s.contains("@workgroup_size(256)"));
        // The plain shader carries none of the split
        // machinery: the unsplit product is untouched.
        let plain = mmq_shader(&MmqShape::new(64));
        assert!(!plain.contains("per_split"));
        assert!(!plain.contains("splits"));
        assert!(plain.contains("for (var blk = 0u; blk < params.blocks_per_row; blk++)"));
    }

    #[test]
    fn plain_and_split_shaders_validate_under_naga() {
        // The strongest shader gate a machine with no GPU
        // adapter at all can run (K19's reasoning): every
        // generated variant must parse and validate under
        // Naga, the same front end wgpu feeds the driver
        // with. Driver acceptance and register behavior
        // are the card session's checks, listed in
        // bench/splitk-mmq-2026-11-07.md.
        for shape in [MmqShape::new(32), MmqShape::new(64)] {
            for src in [mmq_shader(&shape), mmq_split_shader(&shape)] {
                // A format-escape leak (a doubled brace from
                // a substituted value) is a syntax error the
                // parser reports at end of file; name it here
                // instead.
                assert!(
                    !src.contains("{{") && !src.contains("}}"),
                    "shape {}: format escape leaked into the generated shader",
                    shape.name()
                );
                let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| {
                    panic!(
                        "shape {}: WGSL parse failed: {}",
                        shape.name(),
                        e.emit_to_string(&src)
                    )
                });
                naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap_or_else(|e| {
                    panic!("shape {}: Naga validation failed: {e:?}", shape.name())
                });
            }
        }
    }
}
