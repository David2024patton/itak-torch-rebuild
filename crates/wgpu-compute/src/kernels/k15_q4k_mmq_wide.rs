//! K15: Q4_K wide-tile double-buffered MMQ for prefill,
//! K13's successor for the projection GEMMs at chunk row
//! counts.
//!
//! K13's own bench record (bench/gpu-resident-2026-10-30.md)
//! names what binds it: the m64 tile stages 41 KB of
//! workgroup memory, so one workgroup is resident per CU;
//! the tile is one super-block deep with a barrier on each
//! side of the inner product, so the stage phase and the
//! compute phase alternate with the DP4A units idle during
//! staging and the load units idle during compute; latency
//! hiding rests on in-thread ILP alone. The named next
//! step was "the wide-tile (128x128-class) double-buffered
//! MMQ". This kernel is that step, shaped by what the
//! device actually grants:
//!
//! - The tile is 128 output rows by the full 64-row batch
//!   (the 128-class width), with each thread owning a 4x8
//!   sub-tile (4 output rows, 8 batch rows): 256 threads,
//!   and each staged weight word serves eight batch rows
//!   instead of K13's four, halving the weight tile's LDS
//!   read traffic per dot.
//! - The K direction stages in half-super-block steps
//!   (128 elements, 4 sub-blocks). A full-super-block step
//!   at this width cannot be double-buffered inside the
//!   64 KiB workgroup storage grant: two steps of weight
//!   words alone are 64 KiB before activations, scale
//!   pairs, or the batch tile are counted. At half depth
//!   the whole ping-pong set is 61,952 bytes (see
//!   [`q4k_wide_staging_bytes`]), which fits with the
//!   activation tile and both scale sets beside it.
//! - Double buffering is the software pipeline the WGSL
//!   barrier model allows: staging for step s+1 is issued
//!   (into the parity buffer compute is not reading)
//!   before step s's inner product, and one
//!   `workgroupBarrier` per step both publishes the staged
//!   step and guarantees the parity buffer's previous
//!   reader is done before its next overwrite, two steps
//!   later. One barrier per step instead of K13's two per
//!   super-block, and the step's global loads are in flight
//!   across the compute they precede. The step loop is
//!   unrolled by parity at generation time, so every
//!   staged-memory offset in the inner product is a
//!   compile-time constant plus the lane's own indices,
//!   the addressing discipline K13's record credits for
//!   its rate (its first, runtime-indexed build measured
//!   at the f32 GEMM's rate).
//!
//! The math is K13's, term for term: the same
//! `vec_dot_q4_K_q8_1` fold against K12's Q8_1 activations,
//! exact i32 integer dots, per-sub-block scale folds in
//! f32 in super-block order. Only the order in which staged
//! sub-blocks reach the accumulators changes (half a
//! super-block at a time, same sequence), so results check
//! against the same host reference within the same
//! reassociation noise. The activation transform stays in
//! K12, upstream, exactly as in K13.
//!
//! Selection: the session tries this kernel first for
//! chunk projections at or past the wide row threshold
//! (the backend's `WGPU_COMPUTE_MMQ_WIDE_MIN_ROWS` config,
//! default 32 rows; see `crate::compute`), then K13, then
//! the GEMM: the same fallback ladder, one rung longer,
//! every refusal logged by name. Single-row decode never
//! reaches any of them.

use crate::device::GpuContext;
use crate::error::{Result, WgpuError};
use crate::kernels::k12_q8_quant::{ds_vec4s, qs_words};
use crate::kernels::k13_q4k_mmq::MMQ_MAX_BATCH;
use crate::kernels::k1_q4k_gemv::K1Weights;

use crate::diag::dbg_log;

/// Elements per Q4_K super-block.
const BLOCK_ELEMS: usize = 256;
/// Words per Q4_K super-block (144 bytes).
const BLOCK_WORDS: u32 = 36;
/// Sub-blocks staged per step: half a super-block. The
/// step is the double-buffering unit; see the module docs
/// for the storage arithmetic that fixes it at half depth.
pub const WIDE_SUBS_PER_STEP: u32 = 4;
/// Batch rows each thread owns in the wide sub-tile (the
/// output side owns 4; see [`WideMmqShape`]).
const SUB_M: u32 = 8;
/// Activation words one batch row contributes per step
/// (128 int8 elements), before the bank-skew word.
const STEP_WORDS_PER_ROW: u32 = 32;
/// The staged activation tile's row stride in words: the
/// step's 32 words plus one skew word, K13's bank-conflict
/// fix at the step's width.
const X_ROW_STRIDE: u32 = STEP_WORDS_PER_ROW + 1;

/// The wide MMQ tile geometry: output rows per workgroup.
/// Each thread owns a 4x[`SUB_M`] sub-tile, so the
/// workgroup is `(tile_n / 4) * (MMQ_MAX_BATCH / SUB_M)`
/// = `tile_n * 2` threads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WideMmqShape {
    /// Output rows one workgroup computes (the staged
    /// weight tile's height).
    pub tile_n: u32,
    /// Threads per workgroup: `tile_n * 2`.
    pub workgroup: u32,
}

/// Tile widths the wide kernels consider, widest first.
/// The shipped width on a device is the first whose
/// double-buffered staging set its granted workgroup
/// storage holds (see [`shipped_wide_shape`]).
pub const WIDE_TILE_CANDIDATES: [u32; 4] = [128, 112, 96, 64];

impl WideMmqShape {
    /// A shape with the given tile; the workgroup size is
    /// derived (see the type docs).
    #[must_use]
    pub const fn new(tile_n: u32) -> WideMmqShape {
        WideMmqShape {
            tile_n,
            // (tile_n / 4 output sub-tiles) x (batch cap /
            // SUB_M batch sub-tiles) threads.
            workgroup: tile_n / 4 * (MMQ_MAX_BATCH / SUB_M),
        }
    }

    /// The bench/check name: `w<tile_n>`.
    #[must_use]
    pub fn name(&self) -> String {
        format!("w{}", self.tile_n)
    }

    /// Check the shape's geometry rules (the storage check
    /// is the caller's, against the device it will run on).
    pub(crate) fn validate(&self) -> Result<()> {
        if !WIDE_TILE_CANDIDATES.contains(&self.tile_n) {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "mmq wide shape {}: tile_n {} is not one of {WIDE_TILE_CANDIDATES:?}",
                    self.name(),
                    self.tile_n
                ),
            });
        }
        Ok(())
    }
}

/// Workgroup storage one parity buffer of K15's staging
/// occupies, in bytes: the packed weight words
/// (`tile_n` rows x [`STEP_WORDS_PER_ROW`] words), the
/// activation words (batch cap x the skewed stride), the
/// weight scale pairs (`tile_n` x [`WIDE_SUBS_PER_STEP`]
/// vec2), and the activation scale pairs (batch cap x
/// [`WIDE_SUBS_PER_STEP`] vec2).
#[must_use]
pub const fn q4k_wide_buffer_bytes(shape: &WideMmqShape) -> u32 {
    shape.tile_n * STEP_WORDS_PER_ROW * 4
        + MMQ_MAX_BATCH * X_ROW_STRIDE * 4
        + shape.tile_n * WIDE_SUBS_PER_STEP * 8
        + MMQ_MAX_BATCH * WIDE_SUBS_PER_STEP * 8
}

/// Workgroup storage K15's staging occupies at `shape`,
/// in bytes: both parity buffers (the double buffering is
/// the point; see the module docs).
#[must_use]
pub const fn q4k_wide_staging_bytes(shape: &WideMmqShape) -> u32 {
    2 * q4k_wide_buffer_bytes(shape)
}

/// The wide tile geometry a device with these limits
/// runs: the first candidate, in preference order, whose
/// double-buffered staging set its granted workgroup
/// storage holds. `None` below the narrowest candidate's
/// need: no wide MMQ on that device, and the session
/// keeps K13 (see the module docs for the ladder).
#[must_use]
pub fn shipped_wide_shape(limits: &wgpu::Limits) -> Option<WideMmqShape> {
    let storage = limits.max_compute_workgroup_storage_size;
    WIDE_TILE_CANDIDATES
        .into_iter()
        .map(WideMmqShape::new)
        .find(|shape| storage >= q4k_wide_staging_bytes(shape))
}

/// The staging statements for one step of K15's pipeline:
/// super-block `({step}) >> 1`, half `({step}) & 1`, staged
/// into parity buffer `buf`. Global addresses take the
/// step as a runtime expression (they are global loads;
/// K13's constant-offset discipline binds the LDS side,
/// which is all in the compute body).
fn stage_body(shape: &WideMmqShape, buf: usize, step: &str) -> String {
    let tile_n = shape.tile_n;
    let wg = shape.workgroup;
    let tasks = tile_n * 2;
    format!(
        r#"        {{
            let s_blk = ({step}) >> 1u;
            let s_half = ({step}) & 1u;
            // Stage this step's weight words and scale
            // pairs. Task (n, gl) owns output row n's word
            // group gl of the step's half: group g = the
            // half's two groups, each packing into
            // sub-blocks 2g (low nibbles) and 2g + 1 (high).
            for (var task = lane; task < {tasks}u; task += {wg}u) {{
                let n = task / 2u;
                let gl = task % 2u;
                let g = s_half * 2u + gl;
                let gn = tile_n0 + n;
                if (gn < params.out_rows) {{
                    let bw = (gn * params.blocks_per_row + s_blk) * {BLOCK_WORDS}u;
                    let hw = weights[bw];
                    let dd = f16_to_f32(hw & 0xffffu);
                    let ddm = f16_to_f32(hw >> 16u);
                    let sw = array<u32, 3>(weights[bw + 1u], weights[bw + 2u], weights[bw + 3u]);
                    for (var h2 = 0u; h2 < 2u; h2++) {{
                        let j = g * 2u + h2;
                        let sm = scale_min_k4(j, sw);
                        w_sm{buf}[n * 4u + gl * 2u + h2] = vec2<f32>(dd * f32(sm.x), ddm * f32(sm.y));
                    }}
                    for (var w = 0u; w < 8u; w++) {{
                        let raw = weights[bw + 4u + g * 8u + w];
                        w_qs{buf}[(n * 4u + gl * 2u) * 8u + w] = raw & 0x0f0f0f0fu;
                        w_qs{buf}[(n * 4u + gl * 2u + 1u) * 8u + w] = (raw >> 4u) & 0x0f0f0f0fu;
                    }}
                }} else {{
                    for (var h2 = 0u; h2 < 2u; h2++) {{
                        w_sm{buf}[n * 4u + gl * 2u + h2] = vec2<f32>(0.0);
                    }}
                    for (var w = 0u; w < 8u; w++) {{
                        w_qs{buf}[(n * 4u + gl * 2u) * 8u + w] = 0u;
                        w_qs{buf}[(n * 4u + gl * 2u + 1u) * 8u + w] = 0u;
                    }}
                }}
            }}
            // Stage the chunk's activation words and (d, s)
            // pairs for this step's sub-blocks: plain
            // copies, coalesced.
            for (var flat = lane; flat < params.batch * 32u; flat += {wg}u) {{
                let m = flat / 32u;
                let jw = flat % 32u;
                x_qs{buf}[m * 33u + jw] = qs_in[(m * params.blocks32 + s_blk * 8u + s_half * 4u + jw / 8u) * 8u + jw % 8u];
            }}
            for (var flat = lane; flat < params.batch * 4u; flat += {wg}u) {{
                let m = flat / 4u;
                let j = flat % 4u;
                let v = ds_in[m * params.blocks32 + s_blk * 8u + s_half * 4u + j];
                x_ds{buf}[flat] = vec2<f32>(v.x, v.y);
            }}
        }}
"#
    )
}

/// The inner product for one staged step of K15: the
/// step's four sub-blocks, unrolled at generation time so
/// every staged offset is a compile-time constant (K13's
/// reasoning), reading parity buffer `buf`.
fn compute_body(buf: usize) -> String {
    let mut inner = String::new();
    for sub in 0..WIDE_SUBS_PER_STEP {
        let so = sub * 8;
        inner.push_str("        {\n            var isum = array<i32, 32>();\n");
        for w in 0..8 {
            for j in 0..8 {
                inner.push_str(&format!(
                    "            let xv{j}w{w} = x_qs{buf}[m{j} * 33u + {so}u + {w}u];\n"
                ));
            }
            for i in 0..4 {
                inner.push_str(&format!(
                    "            let wv{i}w{w} = w_qs{buf}[(n_sub * 4u + {i}u) * 32u + {so}u + {w}u];\n"
                ));
            }
            for i in 0..4 {
                for j in 0..8 {
                    inner.push_str(&format!(
                        "            isum[{}] += dot4I8Packed(wv{i}w{w}, xv{j}w{w});\n",
                        i * 8 + j
                    ));
                }
            }
        }
        for i in 0..4 {
            inner.push_str(&format!(
                "            let c{i} = w_sm{buf}[(n_sub * 4u + {i}u) * 4u + {sub}u];\n"
            ));
        }
        for j in 0..8 {
            inner.push_str(&format!(
                "            let dm{j} = x_ds{buf}[m{j} * 4u + {sub}u];\n"
            ));
        }
        for i in 0..4 {
            for j in 0..8 {
                let idx = i * 8 + j;
                inner.push_str(&format!(
                    "            acc[{idx}] += c{i}.x * (dm{j}.x * f32(isum[{idx}])) - c{i}.y * dm{j}.y;\n"
                ));
            }
        }
        inner.push_str("        }\n");
    }
    inner
}

/// The Q4_K wide MMQ shader for `shape`, generated with
/// the tile geometry as literals (see
/// [`stage_body`]/[`compute_body`] for the two halves of
/// the pipeline and the module docs for the design).
fn mmq_wide_shader(shape: &WideMmqShape) -> String {
    let tile_n = shape.tile_n;
    let wg = shape.workgroup;
    let w_words = tile_n * STEP_WORDS_PER_ROW;
    let x_words = MMQ_MAX_BATCH * X_ROW_STRIDE;
    let w_pairs = tile_n * WIDE_SUBS_PER_STEP;
    let x_pairs = MMQ_MAX_BATCH * WIDE_SUBS_PER_STEP;

    let stage0 = stage_body(shape, 0, "0u");
    let stage1_next = stage_body(shape, 1, "s + 1u");
    let stage0_next = stage_body(shape, 0, "s + 2u");
    let compute0 = compute_body(0);
    let compute1 = compute_body(1);

    // The output store, guarded per element (K13's form).
    let mut stores = String::new();
    for i in 0..4 {
        for j in 0..8 {
            stores.push_str(&format!(
                "    if (n{i} < params.out_rows && m{j} < params.batch) {{\n        output[m{j} * params.out_rows + n{i}] = acc[{}];\n    }}\n",
                i * 8 + j
            ));
        }
    }
    let mut m_defs = String::new();
    for j in 0..8 {
        m_defs.push_str(&format!("    let m{j} = m_sub * 8u + {j}u;\n"));
    }
    // n0 itself is declared in the template; only the
    // sub-tile's other three output rows are derived here.
    let mut n_defs = String::new();
    for i in 1..4 {
        n_defs.push_str(&format!("    let n{i} = n0 + {i}u;\n"));
    }

    format!(
        r#"struct Params {{
    blocks_per_row: u32,
    out_rows: u32,
    batch: u32,
    blocks32: u32,
}}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> qs_in: array<u32>;
@group(0) @binding(2) var<storage, read> ds_in: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

// The double-buffered staging set: parity 0 and parity 1
// hold alternate steps, so the step being computed and
// the step being staged never share an array. Per step a
// buffer holds one half-super-block column: the tile's
// packed weight words, the batch's activation words
// (row-skewed by one word against LDS bank conflicts,
// K13's fix at this width), the weight scale pairs, and
// the activation scale pairs.
var<workgroup> w_qs0: array<u32, {w_words}>;
var<workgroup> w_qs1: array<u32, {w_words}>;
var<workgroup> x_qs0: array<u32, {x_words}>;
var<workgroup> x_qs1: array<u32, {x_words}>;
var<workgroup> w_sm0: array<vec2<f32>, {w_pairs}>;
var<workgroup> w_sm1: array<vec2<f32>, {w_pairs}>;
var<workgroup> x_ds0: array<vec2<f32>, {x_pairs}>;
var<workgroup> x_ds1: array<vec2<f32>, {x_pairs}>;

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
    let n_sub = lane / 8u;
    let m_sub = lane % 8u;
{m_defs}    let n0 = tile_n0 + n_sub * 4u;
{n_defs}    var acc = array<f32, 32>();
    let steps = params.blocks_per_row * 2u;
    // Prologue: stage step 0 into parity 0 and publish it.
{stage0}    workgroupBarrier();
    // Software pipeline, unrolled by parity so the inner
    // product's staged offsets stay compile-time constants:
    // staging for the next step is issued into the parity
    // buffer this step's compute does not read, then the
    // step computes, then one barrier publishes the staged
    // step and retires the computed step's buffer for its
    // overwrite two steps later.
    for (var s = 0u; s < steps; s += 2u) {{
        if (s + 1u < steps) {{
{stage1_next}        }}
{compute0}        workgroupBarrier();
        if (s + 1u < steps) {{
            if (s + 2u < steps) {{
{stage0_next}            }}
{compute1}            workgroupBarrier();
        }}
    }}
{stores}}}
"#
    )
}

/// One prepared K15 dispatch: a bind group plus grid
/// dimensions, built once for a (weights, quantized input,
/// output, batch) combination and encodable any number of
/// times.
#[derive(Debug)]
pub struct K15Call {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    grid: (u32, u32),
    /// Diag timestamp stamp (see `crate::timestamps`); `None`
    /// unless the context carries a recorder.
    diag: Option<crate::timestamps::DiagStamp>,
}

impl K15Call {
    /// Record this call's dispatch into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k15 q4_k mmq wide"),
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

/// The compiled K15 pipeline, in one [`WideMmqShape`].
/// Compilation happens at backend construction, like every
/// kernel in this crate.
#[derive(Debug)]
pub struct K15Pipeline {
    pipeline: wgpu::ComputePipeline,
    shape: WideMmqShape,
}

impl K15Pipeline {
    /// Compile the shipped wide shape for `device` (see
    /// [`shipped_wide_shape`]), or `None` when the device
    /// grants less workgroup storage than the narrowest
    /// candidate's staging set needs: the caller (the
    /// backend) records the absence and the session keeps
    /// its K13 form on such a device.
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] when a device that grants
    /// the storage rejects the shader.
    pub fn for_device(device: &wgpu::Device) -> Result<Option<Self>> {
        match shipped_wide_shape(&device.limits()) {
            Some(shape) => Ok(Some(Self::with_shape(device, shape)?)),
            None => {
                dbg_log!(
                    "k15 mmq wide unavailable: device grants {} bytes of workgroup storage, \
                     below the narrowest wide staging set",
                    device.limits().max_compute_workgroup_storage_size
                );
                Ok(None)
            }
        }
    }

    /// Compile an explicit [`WideMmqShape`] on `device`.
    /// The shape is validated, and the device's granted
    /// workgroup storage must hold the double-buffered
    /// staging set: a shortfall is a named refusal here,
    /// not a shader validation failure later.
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] on a bad shape or
    /// insufficient workgroup storage.
    pub fn with_shape(device: &wgpu::Device, shape: WideMmqShape) -> Result<Self> {
        shape.validate()?;
        let granted = device.limits().max_compute_workgroup_storage_size;
        if granted < q4k_wide_staging_bytes(&shape) {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "k15 wide shape {} stages {} bytes in workgroup memory but the device grants {granted}",
                    shape.name(),
                    q4k_wide_staging_bytes(&shape)
                ),
            });
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("k15 q4_k mmq wide shader"),
            source: wgpu::ShaderSource::Wgsl(mmq_wide_shader(&shape).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k15 q4_k mmq wide pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        dbg_log!("k15 mmq wide pipeline built: shape {}", shape.name());
        Ok(K15Pipeline { pipeline, shape })
    }

    /// The shape this pipeline runs.
    #[must_use]
    pub fn shape(&self) -> WideMmqShape {
        self.shape
    }

    /// Prepare a wide MMQ dispatch of `weights` over a
    /// batch of K12-quantized input rows. The buffer
    /// contract is K13's, byte for byte, so a plan can swap
    /// one product form for the other without moving a
    /// byte: `qs` / `ds` hold the quantized form of `batch`
    /// rows of `weights.in_features` f32 (K12's layout), and
    /// `output` holds `batch * weights.out_rows` f32.
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
    ) -> Result<K15Call> {
        if batch == 0 || batch > MMQ_MAX_BATCH {
            return Err(WgpuError::InvalidTensor {
                name: "k15 call".to_string(),
                detail: format!(
                    "batch {batch} is outside 1..={MMQ_MAX_BATCH} (the batch tile's row count)"
                ),
            });
        }
        if weights.in_features == 0 || weights.in_features % BLOCK_ELEMS != 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k15 call".to_string(),
                detail: format!(
                    "input width {} is not a whole number of {BLOCK_ELEMS}-element super-blocks",
                    weights.in_features
                ),
            });
        }
        let tiles = (weights.out_rows as u64).div_ceil(u64::from(self.shape.tile_n));
        if tiles > 65_535 {
            return Err(WgpuError::InvalidTensor {
                name: "k15 call".to_string(),
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
                name: "k15 call".to_string(),
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
            label: Some("k15 params"),
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
            label: Some("k15 bind group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        Ok(K15Call {
            pipeline: self.pipeline.clone(),
            bind_group,
            grid,
            diag: ctx.diag_stamp(crate::timestamps::DispatchMeta {
                kernel: "k15_q4k_mmq_wide",
                rows: weights.out_rows as u64,
                cols: weights.in_features as u64,
                quant: "Q4_K",
                weight_bytes: weights.byte_len,
                shape: format!("batch={batch} grid={}x{}", grid.0, grid.1),
            }),
        })
    }

    /// Convenience one-shot: run the wide MMQ over
    /// host-side quantized rows and read the
    /// [`batch`, out] result back. The check/bench data
    /// path; the resident session prepares against its own
    /// buffers instead.
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
                name: "k15 input".to_string(),
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
            label: Some("k15 qs"),
            size: qs_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&qs_buffer, 0, &qs_bytes);
        let ds_bytes: Vec<u8> = ds.iter().flat_map(|v| v.to_le_bytes()).collect();
        let ds_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k15 ds"),
            size: ds_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&ds_buffer, 0, &ds_bytes);
        let out_elems = batch as usize * weights.out_rows;
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k15 output"),
            size: (out_elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let call = self.prepare(ctx, weights, &qs_buffer, &ds_buffer, &output_buffer, batch)?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k15 run"),
            });
        call.encode(&mut encoder);
        ctx.queue.submit([encoder.finish()]);
        let out = ctx.read_f32(&output_buffer, out_elems)?;
        dbg_log!(
            "k15 run: batch={batch} out_rows={} bytes_streamed={}",
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
        assert_eq!(WideMmqShape::new(128).name(), "w128");
        assert_eq!(WideMmqShape::new(128).workgroup, 256);
        assert_eq!(WideMmqShape::new(112).workgroup, 224);
        assert_eq!(WideMmqShape::new(96).workgroup, 192);
        assert_eq!(WideMmqShape::new(64).workgroup, 128);
        // The module docs' headline number: the wide tile
        // double-buffered fits the 64 KiB grant, a
        // full-super-block step would not.
        assert_eq!(q4k_wide_staging_bytes(&WideMmqShape::new(128)), 61_952);
        assert_eq!(q4k_wide_staging_bytes(&WideMmqShape::new(112)), 56_832);
        assert_eq!(q4k_wide_staging_bytes(&WideMmqShape::new(96)), 51_712);
        assert_eq!(q4k_wide_staging_bytes(&WideMmqShape::new(64)), 41_472);
    }

    #[test]
    fn shipped_wide_shape_respects_storage() {
        let at = |storage: u32| wgpu::Limits {
            max_compute_workgroup_storage_size: storage,
            ..Default::default()
        };
        assert_eq!(shipped_wide_shape(&at(16_384)), None);
        assert_eq!(shipped_wide_shape(&at(41_472)), Some(WideMmqShape::new(64)));
        assert_eq!(
            shipped_wide_shape(&at(65_536)),
            Some(WideMmqShape::new(128))
        );
    }

    #[test]
    fn shape_validation_refuses_off_list_tiles() {
        assert!(WideMmqShape::new(128).validate().is_ok());
        assert!(WideMmqShape::new(100).validate().is_err());
        assert!(WideMmqShape::new(32).validate().is_err());
    }

    #[test]
    fn shader_is_double_buffered_and_carries_dp4a() {
        let s = mmq_wide_shader(&WideMmqShape::new(128));
        assert!(s.contains("@workgroup_size(256)"));
        assert!(s.contains("dot4I8Packed"));
        // Both parities of every staged array exist.
        for name in ["w_qs0", "w_qs1", "x_qs0", "x_qs1", "w_sm0", "w_sm1"] {
            assert!(s.contains(name), "shader missing {name}");
        }
        // Staged sizes: weights 128 rows x 32 words, the
        // skewed activation tile 64 x 33.
        assert!(s.contains("array<u32, 4096>"));
        assert!(s.contains("array<u32, 2112>"));
        // One barrier per step in the pipeline loop (plus
        // the prologue's publish): the loop body carries
        // exactly two barrier statements, one per parity
        // half.
        assert_eq!(s.matches("workgroupBarrier()").count(), 3);
    }
}
