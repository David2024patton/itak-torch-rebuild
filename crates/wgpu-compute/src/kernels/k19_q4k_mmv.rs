//! K19: the Q4_K small-batch product in llama.cpp's
//! `mul_mat_vec` register-streaming shape, built for
//! the same row band K18 serves (2..=16 batch rows) as
//! a structural counter-proposal to it.
//!
//! Why this kernel exists is a measurement, stated in
//! bench/shader-autopsy-2026-11-04.md,
//! bench/small-batch-2026-11-04.md, and
//! bench/mmv-prototype-2026-11-05.md: both of this
//! crate's earlier attempts
//! at one weight stream per dispatch (the wide MMQ,
//! K15/K16, and the small-batch MMQ, K18) stage weight
//! tiles in workgroup memory, and both stream far
//! below the GEMV rate (K18 at about 70 to 130 GB/s
//! against GEMV's up to 676 GB/s on the same tensors).
//! The shader autopsy (bench/shader-autopsy-2026-11-
//! 04.md) cleared the GEMV's compiled form and the
//! compiler with it; what the two slow kernels share
//! and the fast one lacks is the LDS weight staging.
//! llama.cpp's fast small-batch Vulkan path makes the
//! opposite choice: `mul_mat_vec` dequantizes weights
//! in registers and never stages them (sources read
//! from the server's llama.cpp tree,
//! `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.
//! comp` and `mul_mat_vec_base.glsl`), and its column
//! count scales with the batch (llama.cpp issue
//! #25356's dose-response: the stock NUM_COLS cap of 8
//! cliffs at batch 9, 125.0 down to 19.7 t/s; a cap of
//! 16 gives 131.5 at batch 9 and 158.5 at batch 16).
//!
//! The shape, term for term with that source:
//!
//! - The batch lives INSIDE the workgroup (llama's
//!   `NUM_COLS`), specialized as a literal in the
//!   generated shader per batch count, so the
//!   accumulator array is fully unrolled at generation
//!   time and every index is a compile-time constant:
//!   `temp[NUM_COLS][NUM_ROWS]` in the source is
//!   `acc[rows * batch]` here, one register accumulator
//!   per (output row, batch row) pair.
//! - A workgroup covers [`MmvShape::rows`] output rows
//!   (llama's `NUM_ROWS`; 2, the value ggml-vulkan
//!   picks for K-quants on RDNA3 and the tile K1 ships)
//!   times the whole batch. The weight tile a workgroup
//!   touches is those rows only, so the tensor streams
//!   from device memory once per dispatch however many
//!   batch rows ride it: the property K18 was built
//!   for, reached without its staging.
//! - Per super-block, the sixteen-thread group
//!   assignment, the packed scale extraction, and the
//!   dot math are K1's llama-family text verbatim (K1
//!   is itself a port of `mul_mat_vec_q4_k.comp`): each
//!   thread owns sixteen elements, dequantizes them
//!   into registers (four `vec4`s), and dots them
//!   against every batch row's activation quads in
//!   turn. The activations are re-read per output row
//!   from the input buffer (they are the small side:
//!   the whole batch's activations are hundreds of KB
//!   against the weight stream's GBs, and they ride
//!   the cache); the weights are loaded and
//!   dequantized exactly once per workgroup.
//! - The reduction is K1's: a segmented subgroup
//!   butterfly inside each sixteen-thread group plus a
//!   workgroup-memory finish (ggml's HYBRID mode), or
//!   a pure workgroup-memory tree where the device
//!   grants no subgroup feature. The only workgroup
//!   memory in the shader is those reduction partials,
//!   exactly the exception llama.cpp's own reduction
//!   makes (`tmpsh` in `mul_mat_vec_base.glsl`).
//! - The input is the resident f32 activation buffer
//!   itself, the GEMV's currency: no K12 quantize
//!   dispatch rides in front (K18 and the MMQ forms
//!   pay one), and the output layout is the GEMV's
//!   `[batch][out_rows]`, so a plan can swap this form
//!   for K1's without moving a byte.
//!
//! What WGSL could not carry from the GLSL source,
//! named precisely (details in the bench note):
//! llama's `temp` is indexed by specialization
//! constants (`constant_id`), which WGSL expresses
//! only as pipeline-overridable `override` values or,
//! as here, generation-time literals; and llama's K
//! loop reads its input vector through a second
//! register file of pre-loaded quads shared across
//! rows, where this port re-reads the quads per row
//! from the storage buffer and relies on the cache,
//! the same trade K1's port already made. Neither
//! changes the streaming structure; both are noted so
//! the card verdict can be read against them.
//!
//! Routing is the session's, and it is config: the
//! session prefers K19 inside the band only up to the
//! backend's `mmv_max_rows` threshold (env
//! `WGPU_COMPUTE_MMV_MAX_ROWS`, shipped default 16,
//! the full band, narrowed per shape by the band
//! routing table). The default shipped at 1 (off: no
//! batch qualified) under the wide-MMQ and K18
//! precedent, that an unproven kernel never enters the
//! ladder by default; the proof sequence in
//! bench/mmv-prototype-2026-11-05.md and the baseline
//! session's card verdict
//! (bench/baseline-session-2026-11-06.md leg (d))
//! discharged it, and the default moved on 2026-11-06.
//! Setting the env to 1 restores the old posture
//! exactly.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::device::GpuContext;
use crate::error::{Result, WgpuError};
use crate::kernels::k1_q4k_gemv::{grid_dims, K1Weights};

use crate::diag::dbg_log;

/// Elements per Q4_K super-block.
const BLOCK_ELEMS: usize = 256;

/// Batch rows one K19 dispatch covers at most: the
/// top of the small-batch band, and the column cap
/// llama.cpp's dose-response argues for (issue
/// #25356, cited in the module docs). The kernel is
/// specialized per batch count from [`MMV_MIN_BATCH`]
/// to this cap; batches past it are the GEMM ladder's,
/// not this kernel's.
pub const MMV_MAX_BATCH: u32 = 16;

/// The band's floor: one row is decode, and decode is
/// the GEMV's (llama.cpp dispatches a single column
/// through the same shader family; this crate already
/// has a proven kernel there, and the prototype's
/// question starts where the batch does).
pub const MMV_MIN_BATCH: u32 = 2;

/// The parallel shape a K19 pipeline runs: K1's
/// llama-family geometry (sixteen-thread groups over
/// super-blocks, [`Self::workgroup`] threads), with the
/// batch specialized per variant rather than carried
/// in the shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MmvShape {
    /// Output rows one workgroup covers (llama.cpp's
    /// NUM_ROWS).
    pub rows: u32,
    /// Threads per workgroup; a multiple of sixteen
    /// (one sixteen-thread group per super-block in
    /// flight).
    pub workgroup: u32,
    /// Reduce with the segmented subgroup butterfly
    /// (needs the device's SUBGROUP feature) instead of
    /// the pure workgroup-memory tree.
    pub subgroup: bool,
}

impl MmvShape {
    /// A shape with the given tile geometry and the
    /// workgroup-memory reduction.
    #[must_use]
    pub const fn new(rows: u32, workgroup: u32) -> MmvShape {
        MmvShape {
            rows,
            workgroup,
            subgroup: false,
        }
    }

    /// This shape with the subgroup reduction.
    #[must_use]
    pub const fn with_subgroup(self) -> MmvShape {
        MmvShape {
            subgroup: true,
            ..self
        }
    }

    /// The bench/check name: `r<rows>` with an `sg`
    /// suffix for the subgroup reduction.
    #[must_use]
    pub fn name(&self) -> String {
        let mut name = format!("r{}", self.rows);
        if self.subgroup {
            name.push_str("sg");
        }
        name
    }

    /// Check the shape against a device's granted
    /// features and the geometry rules, naming what is
    /// missing.
    fn validate(&self, features: wgpu::Features) -> Result<()> {
        if !(1..=4).contains(&self.rows) {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "k19 shape {}: rows per workgroup {} is outside 1..=4",
                    self.name(),
                    self.rows
                ),
            });
        }
        if self.workgroup == 0 || self.workgroup % 16 != 0 || self.workgroup > 256 {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "k19 shape {}: workgroup size {} is not a multiple of 16 in 16..=256",
                    self.name(),
                    self.workgroup
                ),
            });
        }
        if self.subgroup && !features.contains(wgpu::Features::SUBGROUP) {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "k19 shape {} needs the SUBGROUP device feature, which this device was not granted",
                    self.name()
                ),
            });
        }
        Ok(())
    }
}

/// The shape [`K19Pipeline::new`] builds for a device
/// with the given granted features: two rows per
/// workgroup of 128 (K1's shipped geometry and
/// ggml-vulkan's K-quant choice on RDNA3), with the
/// subgroup reduction when the device grants it.
#[must_use]
pub fn shipped_shape(features: wgpu::Features) -> MmvShape {
    let base = MmvShape::new(2, 128);
    if features.contains(wgpu::Features::SUBGROUP) {
        base.with_subgroup()
    } else {
        base
    }
}

/// IEEE 754 half to f32, shared WGSL text (K1's,
/// mirroring gguf::dequant::f16_to_f32 for every value
/// a weight file can hold).
const F16_TO_F32_WGSL: &str = r#"
// IEEE 754 half to f32, mirroring gguf::dequant::f16_to_f32 for
// every value a weight file can hold (normals, subnormals, zero;
// inf and NaN payloads keep their class).
fn f16_to_f32(h: u32) -> f32 {
    let sign = (h >> 15u) & 1u;
    let exp = (h >> 10u) & 31u;
    let frac = h & 1023u;
    if (exp == 31u) {
        return bitcast<f32>((sign << 31u) | 0x7f800000u | (frac << 13u));
    }
    if (exp == 0u) {
        let v = f32(frac) * 5.960464477539063e-08;
        if (sign == 1u) { return -v; }
        return v;
    }
    return bitcast<f32>((sign << 31u) | ((exp + 112u) << 23u) | (frac << 13u));
}
"#;

/// The K19 shader for `shape` at `batch`, generated
/// with the tile geometry and the batch as literals:
/// the (row, batch) accumulator array is fully
/// unrolled at generation time, so every accumulator
/// index is a compile-time constant and the array can
/// live in registers (the structural point of the
/// prototype; a runtime-indexed local array in WGSL
/// is free to spill to function memory, which would
/// quietly rebuild the staging this kernel exists to
/// avoid). The per-thread element assignment, packed
/// scale extraction, and dot math are K1's llama-family
/// text (see the module docs).
fn mmv_shader(shape: &MmvShape, batch: u32) -> String {
    let rows = shape.rows;
    let wg = shape.workgroup;
    let groups = wg / 16;
    let slices = rows * batch;
    let partials_len = if shape.subgroup { slices * groups } else { wg };

    // The main loop's body, unrolled per output row of
    // the tile and per batch row: per row, the header,
    // the packed scale extraction, and the dequantized
    // register quads; per (row, batch), the activation
    // quads and K1's dot math into the pair's own
    // accumulator.
    let mut body = String::new();
    for r in 0..rows {
        body.push_str(&format!(
            "        {{\n            let row = row0 + {r}u;\n            if (row < params.out_rows) {{\n"
        ));
        body.push_str(
            "                let bw = (row * params.blocks_per_row + blk) * 36u;\n                let hw = weights[bw];\n                let d = f16_to_f32(hw & 0xffffu);\n                let dmin = f16_to_f32(hw >> 16u);\n                let s16 = (weights[bw + 1u] >> (16u * half)) & 0xffffu;\n                let m16 = (weights[bw + 2u] >> (16u * half)) & 0xffffu;\n                let h16 = (weights[bw + 3u] >> (16u * half)) & 0xffffu;\n                let lo = (m16 << 16u) | s16;\n                let sc_lo = vec4<f32>(unpack4xU8(lo & 0x3f3f3f3fu));\n                let hi = (((h16 << 12u) | h16) & 0x0f0f0f0fu) | ((lo & 0xc0c0c0c0u) >> 2u);\n                let sc_hi = vec4<f32>(unpack4xU8(hi));\n                let w0 = weights[bw + w0_off];\n                let w1 = weights[bw + w0_off + 16u];\n                let qa = vec4<f32>(unpack4xU8(w0 & 0x0f0f0f0fu));\n                let qb = vec4<f32>(unpack4xU8((w0 >> 4u) & 0x0f0f0f0fu));\n                let qc = vec4<f32>(unpack4xU8(w1 & 0x0f0f0f0fu));\n                let qd = vec4<f32>(unpack4xU8((w1 >> 4u) & 0x0f0f0f0fu));\n",
        );
        for b in 0..batch {
            let acc_idx = r * batch + b;
            body.push_str(&format!(
                "                {{\n                    let xb0 = ({b}u * params.blocks_per_row + blk) * 64u;\n                    let xa = input[xb0 + yv];\n                    let xbq = input[xb0 + yv + 8u];\n                    let xc = input[xb0 + yv + 32u];\n                    let xd = input[xb0 + yv + 40u];\n                    let sa = dot(xa, vec4<f32>(1.0));\n                    let sb = dot(xbq, vec4<f32>(1.0));\n                    let sc_ = dot(xc, vec4<f32>(1.0));\n                    let sd = dot(xd, vec4<f32>(1.0));\n                    acc[{acc_idx}] += d * (sc_lo.x * dot(qa, xa) + sc_lo.y * dot(qb, xbq)\n                            + sc_hi.x * dot(qc, xc) + sc_hi.y * dot(qd, xd))\n                        - dmin * (sc_lo.z * sa + sc_lo.w * sb + sc_hi.z * sc_ + sc_hi.w * sd);\n                }}\n"
            ));
        }
        body.push_str("            }\n        }\n");
    }

    // The reduction, generated per accumulator slice
    // for the same reason as the body: constant
    // indices throughout.
    let reduction = if shape.subgroup {
        let mut red = String::new();
        for s in 0..slices {
            red.push_str(&format!(
                "    {{ var v = acc[{s}]; v += subgroupShuffleXor(v, 1u); v += subgroupShuffleXor(v, 2u); v += subgroupShuffleXor(v, 4u); v += subgroupShuffleXor(v, 8u); acc[{s}] = v; }}\n"
            ));
        }
        red.push_str("    if (t == 0u) {\n");
        for s in 0..slices {
            red.push_str(&format!(
                "        partials[{idx} * {groups}u + group] = acc[{s}];\n",
                idx = s
            ));
        }
        red.push_str("    }\n    workgroupBarrier();\n    if (lane == 0u) {\n");
        for s in 0..slices {
            let r = s / batch;
            let b = s % batch;
            red.push_str(&format!(
                "        {{ var sum = 0.0; for (var g = 0u; g < {groups}u; g++) {{ sum += partials[{s} * {groups}u + g]; }} if (row0 + {r}u < params.out_rows) {{ output[{b}u * params.out_rows + row0 + {r}u] = sum; }} }}\n"
            ));
        }
        red.push_str("    }\n");
        red
    } else {
        let half_wg = wg / 2;
        let mut red = String::new();
        for s in 0..slices {
            let r = s / batch;
            let b = s % batch;
            red.push_str(&format!(
                "    partials[lane] = acc[{s}];\n    workgroupBarrier();\n    {{ var stride = {half_wg}u; while (stride > 0u) {{ if (lane < stride) {{ partials[lane] += partials[lane + stride]; }} workgroupBarrier(); stride >>= 1u; }} }}\n    if (lane == 0u && row0 + {r}u < params.out_rows) {{ output[{b}u * params.out_rows + row0 + {r}u] = partials[0]; }}\n"
            ));
        }
        red
    };

    format!(
        r#"struct Params {{
    blocks_per_row: u32,
    out_rows: u32,
    grid_x: u32,
}}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

// Reduction partials only: the one workgroup array in
// the shader, carrying per-group (or per-lane) sums to
// the finish. No weight or activation tile is staged
// anywhere; weights dequantize in registers and the
// activations are read from the input buffer per row.
var<workgroup> partials: array<f32, {partials_len}>;
{F16_TO_F32_WGSL}
@compute @workgroup_size({wg})
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
    let row_groups = (params.out_rows + {rows}u - 1u) / {rows}u;
    let flat = wid.y * params.grid_x + wid.x;
    // Whole-workgroup early out: every lane shares
    // `flat`, so the exit is uniform and the barriers
    // in the reduction stay legal.
    if (flat >= row_groups) {{
        return;
    }}
    let row0 = flat * {rows}u;
    let lane = lid.x;
    let group = lane / 16u;
    let t = lane % 16u;
    let half = t / 8u;
    let l0 = 4u * (t % 8u);
    // This thread's qs words within a super-block (the
    // qs plane starts at word 4, after the four header
    // words; the second word sits 64 bytes later) and
    // its first activation quad within the block's
    // 64-quad span: K1's assignment, verbatim.
    let w0_off = 4u + 8u * half + l0 / 4u;
    let yv = 16u * half + l0 / 4u;

    var acc = array<f32, {slices}>();
    for (var blk = group; blk < params.blocks_per_row; blk += {groups}u) {{
{body}    }}
{reduction}}}
"#
    )
}

/// One prepared K19 dispatch: a bind group plus grid
/// dimensions, built once for a (weights, input,
/// output, batch) combination and encodable any number
/// of times.
#[derive(Debug)]
pub struct K19Call {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    grid: (u32, u32),
    /// Diag timestamp stamp (see `crate::timestamps`);
    /// `None` unless the context carries a recorder.
    diag: Option<crate::timestamps::DiagStamp>,
}

impl K19Call {
    /// Record this call's dispatch into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k19 q4_k mmv"),
            timestamp_writes: None,
        });
        self.encode_into(&mut pass);
    }

    /// Record this dispatch into an already-open compute
    /// `pass`: the composition form the device-resident
    /// session uses to chain a whole forward into one
    /// pass.
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

/// The compiled K19 pipeline set: one shader variant
/// per batch count in the band, compiled lazily and
/// cached (the batch is a generation literal, so each
/// batch count is its own module; a session warms the
/// counts it will run through [`K19Pipeline::
/// precompile`] at plan time). Compilation of the
/// band-floor variant happens at construction, so a
/// WGSL defect in the family fails at backend
/// construction like every other kernel's.
#[derive(Debug)]
pub struct K19Pipeline {
    shape: MmvShape,
    variants: RefCell<HashMap<u32, wgpu::ComputePipeline>>,
}

impl K19Pipeline {
    /// Compile the shipped shape for `device` (see
    /// [`shipped_shape`]).
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] on a shape the device
    /// cannot run or a shader wgpu rejects.
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        Self::with_shape(device, shipped_shape(device.features()))
    }

    /// Compile an explicit [`MmvShape`] on `device`.
    /// The shape is validated against the device's
    /// granted features first: a shape the device
    /// cannot run is a named refusal here, not a shader
    /// validation failure later.
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] on a shape the device
    /// cannot run or a shader wgpu rejects.
    pub fn with_shape(device: &wgpu::Device, shape: MmvShape) -> Result<Self> {
        shape.validate(device.features())?;
        let this = K19Pipeline {
            shape,
            variants: RefCell::new(HashMap::new()),
        };
        // The construction smoke check: the band-floor
        // variant compiles now, so the family's WGSL is
        // proven on this device before any plan routes
        // here.
        this.precompile(device, MMV_MIN_BATCH)?;
        dbg_log!("k19 pipeline built: shape {}", shape.name());
        Ok(this)
    }

    /// The shape this pipeline runs.
    #[must_use]
    pub fn shape(&self) -> MmvShape {
        self.shape
    }

    /// Compile (or fetch from the cache) the variant
    /// for `batch`: the session's plan builder calls
    /// this ahead of the first prepare at a batch count
    /// so Naga's translation cost stays out of the
    /// token loop, and `prepare` calls it on demand
    /// regardless.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when `batch` is
    /// outside the band; [`WgpuError::Pipeline`] when
    /// wgpu rejects the variant's shader.
    pub fn precompile(&self, device: &wgpu::Device, batch: u32) -> Result<()> {
        self.variant(device, batch).map(|_| ())
    }

    /// The cached variant for `batch`, compiling it on
    /// first use.
    fn variant(&self, device: &wgpu::Device, batch: u32) -> Result<wgpu::ComputePipeline> {
        if !(MMV_MIN_BATCH..=MMV_MAX_BATCH).contains(&batch) {
            return Err(WgpuError::InvalidTensor {
                name: "k19 call".to_string(),
                detail: format!(
                    "batch {batch} is outside {MMV_MIN_BATCH}..={MMV_MAX_BATCH} \
                     (the register-streaming small-batch band)"
                ),
            });
        }
        if let Some(p) = self.variants.borrow().get(&batch) {
            return Ok(p.clone());
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("k19 q4_k mmv shader"),
            source: wgpu::ShaderSource::Wgsl(mmv_shader(&self.shape, batch).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k19 q4_k mmv pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        dbg_log!(
            "k19 variant compiled: shape {} batch {batch}",
            self.shape.name()
        );
        self.variants.borrow_mut().insert(batch, pipeline.clone());
        Ok(pipeline)
    }

    /// Prepare a dispatch of `weights` over a batch of
    /// f32 input rows: `input` holds `batch *
    /// weights.in_features` f32 and `output` holds
    /// `batch * weights.out_rows` f32, both in the GEMV's
    /// `[batch][width]` layout, so a plan can swap this
    /// form for K1's without moving a byte.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when the batch is
    /// outside the band or a buffer is smaller than the
    /// geometry addresses.
    pub fn prepare(
        &self,
        ctx: &GpuContext,
        weights: &K1Weights,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        batch: u32,
    ) -> Result<K19Call> {
        let pipeline = self.variant(&ctx.device, batch)?;
        let need_in = u64::from(batch) * weights.in_features as u64 * 4;
        let need_out = u64::from(batch) * weights.out_rows as u64 * 4;
        if input.size() < need_in || output.size() < need_out {
            return Err(WgpuError::InvalidTensor {
                name: "k19 call".to_string(),
                detail: format!(
                    "buffer sizes (in {}, out {}) are below the geometry's needs \
                     ({need_in}, {need_out})",
                    input.size(),
                    output.size()
                ),
            });
        }
        let total = (weights.out_rows as u64).div_ceil(u64::from(self.shape.rows));
        let grid = grid_dims(total);
        let blocks_per_row = (weights.in_features / BLOCK_ELEMS) as u32;
        let params: [u32; 3] = [blocks_per_row, weights.out_rows as u32, grid.0];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k19 params"),
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
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k19 bind group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        Ok(K19Call {
            pipeline,
            bind_group,
            grid,
            diag: ctx.diag_stamp(crate::timestamps::DispatchMeta {
                kernel: "k19_q4k_mmv",
                rows: weights.out_rows as u64,
                cols: weights.in_features as u64,
                quant: "Q4_K",
                weight_bytes: weights.byte_len,
                shape: format!("batch={batch} grid={}x{}", grid.0, grid.1),
            }),
        })
    }

    /// Convenience one-shot: run the product over host
    /// f32 rows and read the result back. The batch is
    /// the input's row count. The check/bench data path;
    /// the resident session prepares against its own
    /// buffers instead.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when the input is
    /// not a whole number of rows or its row count is
    /// outside the band; [`WgpuError::Readback`] when
    /// the result cannot be mapped back.
    pub fn run(&self, ctx: &GpuContext, weights: &K1Weights, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() || input.len() % weights.in_features != 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k19 input".to_string(),
                detail: format!(
                    "input holds {} elements, not a whole number of {}-wide rows",
                    input.len(),
                    weights.in_features
                ),
            });
        }
        let batch = (input.len() / weights.in_features) as u32;
        let input_buffer = ctx.upload_f32("k19 input", input);
        let out_elems = batch as usize * weights.out_rows;
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k19 output"),
            size: (out_elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let call = self.prepare(ctx, weights, &input_buffer, &output_buffer, batch)?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k19 run"),
            });
        call.encode(&mut encoder);
        ctx.queue.submit([encoder.finish()]);
        let out = ctx.read_f32(&output_buffer, out_elems)?;
        dbg_log!(
            "k19 run: batch={batch} out_rows={} bytes_streamed={}",
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
    fn shape_names_and_validation() {
        assert_eq!(MmvShape::new(2, 128).name(), "r2");
        assert_eq!(MmvShape::new(2, 128).with_subgroup().name(), "r2sg");
        assert_eq!(
            shipped_shape(wgpu::Features::empty()),
            MmvShape::new(2, 128)
        );
        assert_eq!(
            shipped_shape(wgpu::Features::SUBGROUP),
            MmvShape::new(2, 128).with_subgroup()
        );
        let err = MmvShape::new(9, 128)
            .validate(wgpu::Features::all())
            .unwrap_err();
        assert!(
            err.to_string().contains("outside 1..=4"),
            "refusal names the row range: {err}"
        );
        let err = MmvShape::new(2, 100)
            .validate(wgpu::Features::all())
            .unwrap_err();
        assert!(
            err.to_string().contains("not a multiple of 16"),
            "refusal names the workgroup rule: {err}"
        );
        let err = MmvShape::new(2, 128)
            .with_subgroup()
            .validate(wgpu::Features::empty())
            .unwrap_err();
        assert!(
            err.to_string().contains("SUBGROUP"),
            "refusal names the missing feature: {err}"
        );
    }

    #[test]
    fn band_edges_are_named_constants() {
        assert_eq!(MMV_MIN_BATCH, 2);
        assert_eq!(MMV_MAX_BATCH, 16);
    }

    #[test]
    fn shader_specializes_the_batch_and_stages_no_weights() {
        for batch in [2, 7, 16] {
            for shape in [MmvShape::new(2, 128), MmvShape::new(2, 128).with_subgroup()] {
                let shader = mmv_shader(&shape, batch);
                // The accumulator array is sized by the
                // literal slice count, and every dot
                // lands in a constant-indexed slot.
                let slices = 2 * batch;
                assert!(
                    shader.contains(&format!("array<f32, {slices}>()")),
                    "batch {batch}: accumulator array is specialized"
                );
                assert!(shader.contains(&format!("acc[{}]", slices - 1)));
                assert!(shader.contains("@workgroup_size(128)"));
                // The one workgroup array is the
                // reduction partials: no weight tile,
                // no activation tile, no staged array
                // sized by the tensor geometry.
                assert_eq!(shader.matches("var<workgroup>").count(), 1);
                assert!(shader.contains("var<workgroup> partials: array<f32,"));
                assert!(!shader.contains("w_qs"));
                assert!(!shader.contains("x_qs"));
                // K1's packed scale extraction and word
                // assignment ride along verbatim.
                assert!(shader.contains("0x3f3f3f3fu"));
                assert!(shader.contains("let w0_off = 4u + 8u * half + l0 / 4u;"));
                // The dequantized quads exist as named
                // register values.
                assert!(shader.contains("let qa = vec4<f32>(unpack4xU8(w0 & 0x0f0f0f0fu));"));
            }
        }
        // The subgroup variant shuffles inside the
        // sixteen; the tree variant never names a
        // subgroup builtin.
        let sg = mmv_shader(&MmvShape::new(2, 128).with_subgroup(), 4);
        assert!(sg.contains("subgroupShuffleXor"));
        let tree = mmv_shader(&MmvShape::new(2, 128), 4);
        assert!(!tree.contains("subgroupShuffleXor"));
        assert!(tree.contains("partials[lane] = acc[0];"));
    }

    #[test]
    fn every_variant_validates_under_naga() {
        // The strongest shader gate a machine with no
        // GPU adapter at all can run: every generated
        // variant (the full band, both reductions) must
        // parse and validate under Naga, the same
        // front end wgpu feeds the driver with. This
        // proves the WGSL is well-formed and well-typed;
        // it does not prove a driver's compiler accepts
        // it or keeps the accumulators in registers,
        // which is the server protocol's step 0 and
        // step 1 (bench/mmv-prototype-2026-11-05.md).
        for batch in MMV_MIN_BATCH..=MMV_MAX_BATCH {
            for shape in [MmvShape::new(2, 128), MmvShape::new(2, 128).with_subgroup()] {
                let src = mmv_shader(&shape, batch);
                let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| {
                    panic!(
                        "shape {} batch {batch}: WGSL parse failed: {}",
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
                    panic!(
                        "shape {} batch {batch}: Naga validation failed: {e:?}",
                        shape.name()
                    )
                });
            }
        }
    }
}
