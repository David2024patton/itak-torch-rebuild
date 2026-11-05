//! K16: Q6_K wide-tile double-buffered MMQ for prefill,
//! K15's sibling (see [`crate::kernels::k15_q4k_mmq_wide`]
//! for the design: the 128-class tile, half-super-block
//! steps, the parity-unrolled software pipeline, and the
//! fallback ladder this kernel joins).
//!
//! What differs from K15 is what differs between K14 and
//! K13: the staged packing (each element's 6-bit quant
//! assembled into its own byte at staging, K14's
//! extraction term for term) and the fold (per 16-element
//! half, against K12's precomputed scaled half sums). And
//! one geometry consequence: K16's activation record is a
//! vec4 per sub-block (the `(d, s)` pair plus the two half
//! sums) where K15's is a vec2, so at the 64 KiB grant the
//! widest double-buffered tile that fits is w112
//! (60,928 bytes) where K15 fits w128; w128 here would
//! stage 66,048 bytes and refuse by name. The candidate
//! machinery (widest first, first fit ships) is shared
//! with K15, so the difference is arithmetic, not policy.
//!
//! The kernel reads the SAME padded device layout K2's
//! upload writes (212-byte super-block stride), so a Q6_K
//! tensor lives on the device exactly once and the GEMV,
//! the GEMM, K14, and this kernel all bind that one
//! buffer.

use crate::device::GpuContext;
use crate::error::{Result, WgpuError};
use crate::kernels::k12_q8_quant::{ds_vec4s, qs_words};
use crate::kernels::k13_q4k_mmq::MMQ_MAX_BATCH;
use crate::kernels::k15_q4k_mmq_wide::{WideMmqShape, WIDE_SUBS_PER_STEP, WIDE_TILE_CANDIDATES};
use crate::kernels::k2_q6k_gemv::K2Weights;

use crate::diag::dbg_log;

/// Elements per Q6_K super-block.
const BLOCK_ELEMS: usize = 256;
/// Words per Q6_K super-block in the device buffer (the
/// padded 212-byte stride; see K2's module docs).
const BLOCK_WORDS_DEV: u32 = 53;
/// Activation words one batch row contributes per step
/// (four sub-blocks x eight words), before the skew word.
const STEP_WORDS_PER_ROW: u32 = WIDE_SUBS_PER_STEP * 8;
/// The staged activation tile's row stride in words: the
/// step's words plus one skew word (K15's docs).
const X_ROW_STRIDE: u32 = STEP_WORDS_PER_ROW + 1;

/// Workgroup storage one parity buffer of K16's staging
/// occupies, in bytes: the packed weight words
/// (`tile_n` rows x [`STEP_WORDS_PER_ROW`] words), the
/// activation words (batch cap x the skewed stride), the
/// weight scale pairs (`tile_n` x [`WIDE_SUBS_PER_STEP`]
/// vec2), and the activation scale quads (batch cap x
/// [`WIDE_SUBS_PER_STEP`] vec4: this kernel reads the
/// half sums, which is the 512 bytes that push w128 past
/// the grant; see the module docs).
#[must_use]
pub const fn q6k_wide_buffer_bytes(shape: &WideMmqShape) -> u32 {
    shape.tile_n * STEP_WORDS_PER_ROW * 4
        + MMQ_MAX_BATCH * X_ROW_STRIDE * 4
        + shape.tile_n * WIDE_SUBS_PER_STEP * 8
        + MMQ_MAX_BATCH * WIDE_SUBS_PER_STEP * 16
}

/// Workgroup storage K16's staging occupies at `shape`,
/// in bytes: both parity buffers.
#[must_use]
pub const fn q6k_wide_staging_bytes(shape: &WideMmqShape) -> u32 {
    2 * q6k_wide_buffer_bytes(shape)
}

/// The wide tile geometry a device with these limits runs
/// for Q6_K (see
/// [`crate::kernels::k15_q4k_mmq_wide::shipped_wide_shape`]
/// for the preference reasoning). `None` below the
/// narrowest candidate's need.
#[must_use]
pub fn shipped_wide_shape(limits: &wgpu::Limits) -> Option<WideMmqShape> {
    let storage = limits.max_compute_workgroup_storage_size;
    WIDE_TILE_CANDIDATES
        .into_iter()
        .map(WideMmqShape::new)
        .find(|shape| storage >= q6k_wide_staging_bytes(shape))
}

/// The staging statements for one step of K16's pipeline:
/// super-block `({step}) >> 1`, half `({step}) & 1`,
/// staged into parity buffer `buf`. The step split is
/// also Q6_K's own half split (sub-blocks 0..3 are half
/// h=0, 4..7 are h=1 in `dequant_q6_k`'s interleave), so
/// each step stages exactly one `h` per row.
fn stage_body(shape: &WideMmqShape, buf: usize, step: &str) -> String {
    let tile_n = shape.tile_n;
    let wg = shape.workgroup;
    let tasks = tile_n * WIDE_SUBS_PER_STEP;
    format!(
        r#"        {{
            let s_blk = ({step}) >> 1u;
            let s_half = ({step}) & 1u;
            // Stage this step's packed weight words and
            // scale pairs. Task (n, jl) owns output row
            // n's sub-block jl of the step's half: it
            // assembles the 6-bit quants into bytes (K14's
            // extraction) and packs four per word.
            for (var task = lane; task < {tasks}u; task += {wg}u) {{
                let n = task / 4u;
                let jl = task % 4u;
                let j = s_half * 4u + jl;
                let gn = tile_n0 + n;
                if (gn < params.out_rows) {{
                    let bw = (gn * params.blocks_per_row + s_blk) * {BLOCK_WORDS_DEV}u;
                    let d = f16_to_f32(weights[bw + 52u] & 0xffffu);
                    let scw = array<u32, 4>(
                        weights[bw + 48u], weights[bw + 49u],
                        weights[bw + 50u], weights[bw + 51u],
                    );
                    let h = j / 4u;
                    let st = j % 4u;
                    let ql_base = bw + h * 16u + (st & 1u) * 8u;
                    let qh_base = bw + 32u + h * 8u;
                    let shift = (st & 1u) * 2u + (st >> 1u) * 4u;
                    let high = st >= 2u;
                    for (var w = 0u; w < 8u; w++) {{
                        let qlb = unpack4xU8(weights[ql_base + w]);
                        let qhb = unpack4xU8(weights[qh_base + w]);
                        let nibs = select(qlb & vec4<u32>(0xfu), (qlb >> vec4<u32>(4u)) & vec4<u32>(0xfu), vec4<bool>(high));
                        let q = nibs | (((qhb >> vec4<u32>(shift)) & vec4<u32>(3u)) << vec4<u32>(4u));
                        w_qs{buf}[(n * 4u + jl) * 8u + w] = pack4xU8(q);
                    }}
                    w_sc{buf}[n * 4u + jl] = vec2<f32>(
                        d * scale_at(scw, h * 8u + 2u * st),
                        d * scale_at(scw, h * 8u + 1u + 2u * st));
                }} else {{
                    for (var w = 0u; w < 8u; w++) {{
                        w_qs{buf}[(n * 4u + jl) * 8u + w] = 0u;
                    }}
                    w_sc{buf}[n * 4u + jl] = vec2<f32>(0.0);
                }}
            }}
            // Stage the chunk's activation words and scale
            // quads for this step's sub-blocks: plain
            // copies, coalesced.
            for (var flat = lane; flat < params.batch * 32u; flat += {wg}u) {{
                let m = flat / 32u;
                let jw = flat % 32u;
                x_qs{buf}[m * 33u + jw] = qs_in[(m * params.blocks32 + s_blk * 8u + s_half * 4u + jw / 8u) * 8u + jw % 8u];
            }}
            for (var flat = lane; flat < params.batch * 4u; flat += {wg}u) {{
                let m = flat / 4u;
                let j = flat % 4u;
                x_ds{buf}[flat] = ds_in[m * params.blocks32 + s_blk * 8u + s_half * 4u + j];
            }}
        }}
"#
    )
}

/// The inner product for one staged step of K16: exact
/// integer dots over the two 16-element halves of each
/// sub-block, then the per-half fold (K14's math at the
/// wide geometry), reading parity buffer `buf`.
fn compute_body(buf: usize) -> String {
    let mut inner = String::new();
    for sub in 0..WIDE_SUBS_PER_STEP {
        let so = sub * 8;
        inner.push_str(
            "        {\n            var isum_lo = array<i32, 32>();\n            var isum_hi = array<i32, 32>();\n",
        );
        for w in 0..8 {
            let acc_name = if w < 4 { "isum_lo" } else { "isum_hi" };
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
                        "            {acc_name}[{}] += dot4I8Packed(wv{i}w{w}, xv{j}w{w});\n",
                        i * 8 + j
                    ));
                }
            }
        }
        for i in 0..4 {
            inner.push_str(&format!(
                "            let sc{i} = w_sc{buf}[(n_sub * 4u + {i}u) * 4u + {sub}u];\n"
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
                    "            acc[{idx}] += sc{i}.x * (dm{j}.x * f32(isum_lo[{idx}]) - 32.0 * dm{j}.z)\n                         + sc{i}.y * (dm{j}.x * f32(isum_hi[{idx}]) - 32.0 * dm{j}.w);\n"
                ));
            }
        }
        inner.push_str("        }\n");
    }
    inner
}

/// The Q6_K wide MMQ shader for `shape`: K15's pipeline
/// structure with the Q6_K stage and fold.
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

// The double-buffered staging set (K15's docs own the
// design): parity 0 and parity 1 hold alternate steps.
// The activation tile is row-skewed by one word against
// LDS bank conflicts, as in K14.
var<workgroup> w_qs0: array<u32, {w_words}>;
var<workgroup> w_qs1: array<u32, {w_words}>;
var<workgroup> x_qs0: array<u32, {x_words}>;
var<workgroup> x_qs1: array<u32, {x_words}>;
var<workgroup> w_sc0: array<vec2<f32>, {w_pairs}>;
var<workgroup> w_sc1: array<vec2<f32>, {w_pairs}>;
var<workgroup> x_ds0: array<vec4<f32>, {x_pairs}>;
var<workgroup> x_ds1: array<vec4<f32>, {x_pairs}>;

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
    let n_sub = lane / 8u;
    let m_sub = lane % 8u;
{m_defs}    let n0 = tile_n0 + n_sub * 4u;
{n_defs}    var acc = array<f32, 32>();
    let steps = params.blocks_per_row * 2u;
    // Prologue: stage step 0 into parity 0 and publish it.
{stage0}    workgroupBarrier();
    // Software pipeline, unrolled by parity (K15's docs):
    // staging for the next step is issued into the parity
    // buffer this step's compute does not read, then the
    // step computes, then one barrier publishes the staged
    // step and retires the computed step's buffer.
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

/// One prepared K16 dispatch: a bind group plus grid
/// dimensions, built once for a (weights, quantized input,
/// output, batch) combination and encodable any number of
/// times.
#[derive(Debug)]
pub struct K16Call {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    grid: (u32, u32),
    /// Diag timestamp stamp (see `crate::timestamps`); `None`
    /// unless the context carries a recorder.
    diag: Option<crate::timestamps::DiagStamp>,
}

impl K16Call {
    /// Record this call's dispatch into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k16 q6_k mmq wide"),
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

/// The compiled K16 pipeline, in one [`WideMmqShape`].
/// Compilation happens at backend construction, like every
/// kernel in this crate.
#[derive(Debug)]
pub struct K16Pipeline {
    pipeline: wgpu::ComputePipeline,
    shape: WideMmqShape,
}

impl K16Pipeline {
    /// Compile the shipped wide shape for `device` (see
    /// [`shipped_wide_shape`]), or `None` when the device
    /// grants less workgroup storage than the narrowest
    /// candidate's staging set needs. See
    /// [`crate::kernels::k15_q4k_mmq_wide::K15Pipeline::for_device`]
    /// for the fallback contract.
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
                    "k16 mmq wide unavailable: device grants {} bytes of workgroup storage, \
                     below the narrowest wide staging set",
                    device.limits().max_compute_workgroup_storage_size
                );
                Ok(None)
            }
        }
    }

    /// Compile an explicit [`WideMmqShape`] on `device`.
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] on a bad shape or
    /// insufficient workgroup storage.
    pub fn with_shape(device: &wgpu::Device, shape: WideMmqShape) -> Result<Self> {
        shape.validate()?;
        let granted = device.limits().max_compute_workgroup_storage_size;
        if granted < q6k_wide_staging_bytes(&shape) {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "k16 wide shape {} stages {} bytes in workgroup memory but the device grants {granted}",
                    shape.name(),
                    q6k_wide_staging_bytes(&shape)
                ),
            });
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("k16 q6_k mmq wide shader"),
            source: wgpu::ShaderSource::Wgsl(mmq_wide_shader(&shape).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k16 q6_k mmq wide pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        dbg_log!("k16 mmq wide pipeline built: shape {}", shape.name());
        Ok(K16Pipeline { pipeline, shape })
    }

    /// The shape this pipeline runs.
    #[must_use]
    pub fn shape(&self) -> WideMmqShape {
        self.shape
    }

    /// Prepare a wide MMQ dispatch of `weights` over a
    /// batch of K12-quantized input rows. See
    /// [`crate::kernels::k15_q4k_mmq_wide::K15Pipeline::prepare`]
    /// for the contract (identical, byte for byte).
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] on an out-of-range
    /// batch, a width that is not a whole number of
    /// super-blocks, or undersized buffers.
    pub fn prepare(
        &self,
        ctx: &GpuContext,
        weights: &K2Weights,
        qs: &wgpu::Buffer,
        ds: &wgpu::Buffer,
        output: &wgpu::Buffer,
        batch: u32,
    ) -> Result<K16Call> {
        if batch == 0 || batch > MMQ_MAX_BATCH {
            return Err(WgpuError::InvalidTensor {
                name: "k16 call".to_string(),
                detail: format!(
                    "batch {batch} is outside 1..={MMQ_MAX_BATCH} (the batch tile's row count)"
                ),
            });
        }
        if weights.in_features == 0 || weights.in_features % BLOCK_ELEMS != 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k16 call".to_string(),
                detail: format!(
                    "input width {} is not a whole number of {BLOCK_ELEMS}-element super-blocks",
                    weights.in_features
                ),
            });
        }
        let tiles = (weights.out_rows as u64).div_ceil(u64::from(self.shape.tile_n));
        if tiles > 65_535 {
            return Err(WgpuError::InvalidTensor {
                name: "k16 call".to_string(),
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
                name: "k16 call".to_string(),
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
            label: Some("k16 params"),
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
            label: Some("k16 bind group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        Ok(K16Call {
            pipeline: self.pipeline.clone(),
            bind_group,
            grid,
            diag: ctx.diag_stamp(crate::timestamps::DispatchMeta {
                kernel: "k16_q6k_mmq_wide",
                rows: weights.out_rows as u64,
                cols: weights.in_features as u64,
                quant: "Q6_K",
                weight_bytes: weights.byte_len * 212 / 210,
                shape: format!("batch={batch} grid={}x{}", grid.0, grid.1),
            }),
        })
    }

    /// Convenience one-shot: run the wide MMQ over
    /// host-side quantized rows and read the result back.
    /// See [`crate::kernels::k15_q4k_mmq_wide::K15Pipeline::run`].
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
        weights: &K2Weights,
        qs: &[u32],
        ds: &[f32],
        batch: u32,
    ) -> Result<Vec<f32>> {
        if qs.len() != qs_words(batch as usize, weights.in_features)
            || ds.len() != ds_vec4s(batch as usize, weights.in_features) * 4
        {
            return Err(WgpuError::InvalidTensor {
                name: "k16 input".to_string(),
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
            label: Some("k16 qs"),
            size: qs_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&qs_buffer, 0, &qs_bytes);
        let ds_bytes: Vec<u8> = ds.iter().flat_map(|v| v.to_le_bytes()).collect();
        let ds_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k16 ds"),
            size: ds_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&ds_buffer, 0, &ds_bytes);
        let out_elems = batch as usize * weights.out_rows;
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k16 output"),
            size: (out_elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let call = self.prepare(ctx, weights, &qs_buffer, &ds_buffer, &output_buffer, batch)?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k16 run"),
            });
        call.encode(&mut encoder);
        ctx.queue.submit([encoder.finish()]);
        let out = ctx.read_f32(&output_buffer, out_elems)?;
        dbg_log!(
            "k16 run: batch={batch} out_rows={} bytes_streamed={}",
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
    fn staging_sizes() {
        // The module docs' headline numbers: w128 does
        // not fit the 64 KiB grant for Q6_K (66,048), so
        // the shipped wide shape there is w112 (60,928).
        assert_eq!(q6k_wide_staging_bytes(&WideMmqShape::new(128)), 66_048);
        assert_eq!(q6k_wide_staging_bytes(&WideMmqShape::new(112)), 60_928);
        assert_eq!(q6k_wide_staging_bytes(&WideMmqShape::new(96)), 55_808);
        assert_eq!(q6k_wide_staging_bytes(&WideMmqShape::new(64)), 45_568);
    }

    #[test]
    fn shipped_wide_shape_respects_storage() {
        let at = |storage: u32| wgpu::Limits {
            max_compute_workgroup_storage_size: storage,
            ..Default::default()
        };
        assert_eq!(shipped_wide_shape(&at(16_384)), None);
        assert_eq!(shipped_wide_shape(&at(45_568)), Some(WideMmqShape::new(64)));
        assert_eq!(
            shipped_wide_shape(&at(65_536)),
            Some(WideMmqShape::new(112))
        );
    }

    #[test]
    fn shader_is_double_buffered_and_carries_dp4a() {
        let s = mmq_wide_shader(&WideMmqShape::new(112));
        assert!(s.contains("@workgroup_size(224)"));
        assert!(s.contains("dot4I8Packed"));
        assert!(s.contains("isum_lo"));
        for name in ["w_qs0", "w_qs1", "x_qs0", "x_qs1", "w_sc0", "w_sc1"] {
            assert!(s.contains(name), "shader missing {name}");
        }
        // Staged sizes at w112: weights 112 rows x 32
        // words, the skewed activation tile 64 x 33.
        assert!(s.contains("array<u32, 3584>"));
        assert!(s.contains("array<u32, 2112>"));
        assert!(s.contains("* 53u"));
        assert_eq!(s.matches("workgroupBarrier()").count(), 3);
    }
}
