//! K14: Q6_K matrix-matrix MMQ for prefill, K13's sibling.
//!
//! The tiling, the staging discipline, and the fallback
//! contract are K13's (see [`crate::kernels::k13_q4k_mmq`]'s
//! module docs); what differs is the staged packing and the
//! fold, which are Q6_K's. The kernel reads the SAME padded
//! device layout K2's upload writes (212-byte super-block
//! stride), so a Q6_K tensor lives on the device exactly
//! once and the GEMV, the GEMM, and this MMQ all bind that
//! one buffer.
//!
//! Per 32-element sub-block `j` of a super-block (half
//! `h = j / 4`, stream `s = j % 4` of `dequant_q6_k`'s
//! interleave), the stage assembles each element's 6-bit
//! quant into its own byte (a `ql` nibble plus two `qh`
//! bits, K11's extraction term for term) and packs four per
//! word for `dot4I8Packed`. The math is
//! `vec_dot_q6_K_q8_1`: scales change every 16 elements, so
//! the fold runs per half,
//! `d * sc_h * (d_x * sumi_h - 32 * sh_h)`, where `sumi_h`
//! is the exact integer dot over the half and `sh_h` is
//! K12's precomputed scaled half sum: the `-32` is the
//! quant's offset, folded against the activation sum
//! instead of subtracted per element. There is no gated
//! form (K13's note applies: the activation transform is
//! finished in K12), and the SwiGLU down projection is the
//! one activation consumer, through the quantizer's
//! SwiGLU variant.
//!
//! The split-K form ([`K14Pipeline::prepare_split`]) is
//! K13's split contract on this kernel's body; it exists
//! for this family's narrow projections (the 256-wide KV
//! tensors whose plain grid starves the card), routed by
//! the backend's `WGPU_COMPUTE_MMQ_SPLITK` resolution.

use crate::device::GpuContext;
use crate::error::{Result, WgpuError};
use crate::kernels::k12_q8_quant::{ds_vec4s, qs_words};
use crate::kernels::k13_q4k_mmq::{MmqShape, MMQ_MAX_BATCH};
use crate::kernels::k2_q6k_gemv::K2Weights;
use crate::kernels::splitk_reduce::{SplitkReduceCall, SplitkReducer, SPLITK_MAX};

use crate::diag::dbg_log;

/// Elements per Q6_K super-block.
const BLOCK_ELEMS: usize = 256;
/// Words per Q6_K super-block in the device buffer (the
/// padded 212-byte stride; see K2's module docs).
const BLOCK_WORDS_DEV: u32 = 53;

/// Workgroup storage K14's staging occupies at `shape`, in
/// bytes: the packed weight words (`tile_n` x 64 words), the
/// activation words (batch cap x 64 words), the weight scale
/// pairs (`tile_n` x 8 vec2), and the activation scale quads
/// (batch cap x 8 vec4: this kernel reads the half sums).
#[must_use]
pub const fn q6k_staging_bytes(shape: &MmqShape) -> u32 {
    shape.tile_n * 64 * 4 + MMQ_MAX_BATCH * 65 * 4 + shape.tile_n * 8 * 8 + MMQ_MAX_BATCH * 8 * 16
}

/// The tile geometry a device with these limits runs (see
/// [`crate::kernels::k13_q4k_mmq::shipped_shape`] for the
/// preference reasoning). `None` below the narrow tile's
/// need: no MMQ on that device.
#[must_use]
pub fn shipped_shape(limits: &wgpu::Limits) -> Option<MmqShape> {
    let storage = limits.max_compute_workgroup_storage_size;
    [MmqShape::new(64), MmqShape::new(32)]
        .into_iter()
        .find(|shape| storage >= q6k_staging_bytes(shape))
}

/// The Q6_K MMQ shader for `shape`: K13's tiling body with
/// the Q6_K stage and the per-half fold.
fn mmq_shader(shape: &MmqShape) -> String {
    mmq_shader_split(shape, false)
}

/// The split-K form of the Q6_K MMQ shader: the same
/// generated body with the reduction partitioned across
/// `wid.y` slices, exactly K13's split contract (see
/// [`crate::kernels::k13_q4k_mmq`] and
/// [`crate::kernels::splitk_reduce`]). This is the form the
/// 256-wide KV projections need: their plain grid is 4
/// workgroups walking 26 super-blocks each.
fn mmq_split_shader(shape: &MmqShape) -> String {
    mmq_shader_split(shape, true)
}

fn mmq_shader_split(shape: &MmqShape, split: bool) -> String {
    let tile_n = shape.tile_n;
    let wg = shape.workgroup;
    let w_words = tile_n * 64;
    let w_pairs = tile_n * 8;

    // The inner product, unrolled over the eight
    // sub-blocks at generation time (K13's reasoning: every
    // staged offset becomes a compile-time constant). Per
    // sub-block: exact integer dots over the two 16-element
    // halves, then the per-half fold.
    let mut inner = String::new();
    for sub in 0..8u32 {
        let so = sub * 8;
        inner.push_str(
            "        {\n            var isum_lo = array<i32, 16>();\n            var isum_hi = array<i32, 16>();\n",
        );
        for w in 0..8 {
            let acc_name = if w < 4 { "isum_lo" } else { "isum_hi" };
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
                        "            {acc_name}[{}] += dot4I8Packed(wv{i}w{w}, xv{j}w{w});\n",
                        i * 4 + j
                    ));
                }
            }
        }
        for i in 0..4 {
            inner.push_str(&format!(
                "            let sc{i} = w_sc[(n_sub * 4u + {i}u) * 8u + {sub}u];\n"
            ));
        }
        for j in 0..4 {
            inner.push_str(&format!(
                "            let dm{j} = x_ds[m{j} * 8u + {sub}u];\n"
            ));
        }
        for i in 0..4 {
            for j in 0..4 {
                let idx = i * 4 + j;
                inner.push_str(&format!(
                    "            acc[{idx}] += sc{i}.x * (dm{j}.x * f32(isum_lo[{idx}]) - 32.0 * dm{j}.z)\n                         + sc{i}.y * (dm{j}.x * f32(isum_hi[{idx}]) - 32.0 * dm{j}.w);\n"
                ));
            }
        }
        inner.push_str("        }\n");
    }
    // The output store, guarded per element (K13's form).
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
    // super-block range (K13's split contract): a slice
    // past the end walks an empty range and stores zeros.
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
var<workgroup> w_sc: array<vec2<f32>, {w_pairs}>;
var<workgroup> x_ds: array<vec4<f32>, 512>;

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
{loop_open}
        // Stage the tile's packed weight words and scale
        // pairs for this super-block column. Task (n, j)
        // owns output row n's sub-block j: it assembles the
        // 6-bit quants into bytes (K11's extraction) and
        // packs four per word. Rows past the tensor's end
        // stage zeros.
        for (var task = lane; task < {tile_n}u * 8u; task += {wg}u) {{
            let n = task / 8u;
            let j = task % 8u;
            let gn = tile_n0 + n;
            if (gn < params.out_rows) {{
                let bw = (gn * params.blocks_per_row + blk) * {BLOCK_WORDS_DEV}u;
                let d = f16_to_f32(weights[bw + 52u] & 0xffffu);
                let scw = array<u32, 4>(
                    weights[bw + 48u], weights[bw + 49u],
                    weights[bw + 50u], weights[bw + 51u],
                );
                let h = j / 4u;
                let s = j % 4u;
                let ql_base = bw + h * 16u + (s & 1u) * 8u;
                let qh_base = bw + 32u + h * 8u;
                let shift = (s & 1u) * 2u + (s >> 1u) * 4u;
                let high = s >= 2u;
                for (var w = 0u; w < 8u; w++) {{
                    let qlb = unpack4xU8(weights[ql_base + w]);
                    let qhb = unpack4xU8(weights[qh_base + w]);
                    let nibs = select(qlb & vec4<u32>(0xfu), (qlb >> vec4<u32>(4u)) & vec4<u32>(0xfu), vec4<bool>(high));
                    let q = nibs | (((qhb >> vec4<u32>(shift)) & vec4<u32>(3u)) << vec4<u32>(4u));
                    w_qs[(n * 8u + j) * 8u + w] = pack4xU8(q);
                }}
                w_sc[n * 8u + j] = vec2<f32>(
                    d * scale_at(scw, h * 8u + 2u * s),
                    d * scale_at(scw, h * 8u + 1u + 2u * s));
            }} else {{
                for (var w = 0u; w < 8u; w++) {{
                    w_qs[(n * 8u + j) * 8u + w] = 0u;
                }}
                w_sc[n * 8u + j] = vec2<f32>(0.0);
            }}
        }}
        // Stage the chunk's activation words and scale
        // quads for this column: plain copies, coalesced.
        for (var flat = lane; flat < params.batch * 64u; flat += {wg}u) {{
            let m = flat / 64u;
            let jw = flat % 64u;
            x_qs[m * 65u + jw] = qs_in[(m * params.blocks32 + blk * 8u + jw / 8u) * 8u + jw % 8u];
        }}
        for (var flat = lane; flat < params.batch * 8u; flat += {wg}u) {{
            let m = flat / 8u;
            let j = flat % 8u;
            x_ds[flat] = ds_in[m * params.blocks32 + blk * 8u + j];
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

/// One prepared K14 dispatch: a bind group plus grid
/// dimensions, built once for a (weights, quantized input,
/// output, batch) combination and encodable any number of
/// times.
#[derive(Debug)]
pub struct K14Call {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    grid: (u32, u32),
    /// Diag timestamp stamp (see `crate::timestamps`); `None`
    /// unless the context carries a recorder.
    diag: Option<crate::timestamps::DiagStamp>,
}

impl K14Call {
    /// Record this call's dispatch into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k14 q6_k mmq"),
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

/// One prepared split-K K14 dispatch pair: the split
/// product (writing f32 partial sums to a scratch buffer
/// the prepare allocated) followed by the split-K
/// reduction into the caller's output. K13's split call
/// contract, Q6_K edition; the two dispatches carry
/// separate diag stamps (`k14_q6k_mmq_split` and
/// `splitk_reduce`).
#[derive(Debug)]
pub struct K14SplitCall {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    grid: (u32, u32),
    reduce: SplitkReduceCall,
    /// Diag timestamp stamp for the product dispatch (the
    /// reduction carries its own, inside `reduce`).
    diag: Option<crate::timestamps::DiagStamp>,
}

impl K14SplitCall {
    /// Record this call's dispatches into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k14 q6_k mmq split"),
            timestamp_writes: None,
        });
        self.encode_into(&mut pass);
    }

    /// Record the product and its reduction into an
    /// already-open compute `pass`, in that order.
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

/// The compiled K14 pipeline, in one [`MmqShape`].
/// Compilation happens at backend construction, like every
/// kernel in this crate. The split-K form's pipeline and
/// the shared reducer compile alongside the plain pipeline
/// regardless of the split switch (K13's contract).
#[derive(Debug)]
pub struct K14Pipeline {
    pipeline: wgpu::ComputePipeline,
    split_pipeline: wgpu::ComputePipeline,
    reducer: SplitkReducer,
    shape: MmqShape,
}

impl K14Pipeline {
    /// Compile the shipped shape for `device` (see
    /// [`shipped_shape`]), or `None` when the device grants
    /// less workgroup storage than the smallest staging set
    /// needs. See [`crate::kernels::k13_q4k_mmq::K13Pipeline::for_device`]
    /// for the fallback contract.
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
                    "k14 mmq unavailable: device grants {} bytes of workgroup storage, \
                     below the smallest staging set",
                    device.limits().max_compute_workgroup_storage_size
                );
                Ok(None)
            }
        }
    }

    /// Compile an explicit [`MmqShape`] on `device`.
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] on a bad shape or
    /// insufficient workgroup storage.
    pub fn with_shape(device: &wgpu::Device, shape: MmqShape) -> Result<Self> {
        shape.validate()?;
        let granted = device.limits().max_compute_workgroup_storage_size;
        if granted < q6k_staging_bytes(&shape) {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "k14 shape {} stages {} bytes in workgroup memory but the device grants {granted}",
                    shape.name(),
                    q6k_staging_bytes(&shape)
                ),
            });
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("k14 q6_k mmq shader"),
            source: wgpu::ShaderSource::Wgsl(mmq_shader(&shape).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k14 q6_k mmq pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let split_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("k14 q6_k mmq split shader"),
            source: wgpu::ShaderSource::Wgsl(mmq_split_shader(&shape).into()),
        });
        let split_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k14 q6_k mmq split pipeline"),
            layout: None,
            module: &split_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let reducer = SplitkReducer::new(device)?;
        dbg_log!(
            "k14 mmq pipelines built: shape {} (plain and split-K forms)",
            shape.name()
        );
        Ok(K14Pipeline {
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
    /// K12-quantized input rows. See
    /// [`crate::kernels::k13_q4k_mmq::K13Pipeline::prepare`]
    /// for the contract.
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
    ) -> Result<K14Call> {
        if batch == 0 || batch > MMQ_MAX_BATCH {
            return Err(WgpuError::InvalidTensor {
                name: "k14 call".to_string(),
                detail: format!(
                    "batch {batch} is outside 1..={MMQ_MAX_BATCH} (the batch tile's row count)"
                ),
            });
        }
        if weights.in_features == 0 || weights.in_features % BLOCK_ELEMS != 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k14 call".to_string(),
                detail: format!(
                    "input width {} is not a whole number of {BLOCK_ELEMS}-element super-blocks",
                    weights.in_features
                ),
            });
        }
        let tiles = (weights.out_rows as u64).div_ceil(u64::from(self.shape.tile_n));
        if tiles > 65_535 {
            return Err(WgpuError::InvalidTensor {
                name: "k14 call".to_string(),
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
                name: "k14 call".to_string(),
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
            label: Some("k14 params"),
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
            label: Some("k14 bind group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        Ok(K14Call {
            pipeline: self.pipeline.clone(),
            bind_group,
            grid,
            diag: ctx.diag_stamp(crate::timestamps::DispatchMeta {
                kernel: "k14_q6k_mmq",
                rows: weights.out_rows as u64,
                cols: weights.in_features as u64,
                quant: "Q6_K",
                weight_bytes: weights.byte_len * 212 / 210,
                shape: format!("batch={batch} grid={}x{}", grid.0, grid.1),
            }),
        })
    }

    /// Prepare a split-K MMQ dispatch: [`Self::prepare`]'s
    /// geometry and buffers, with the reduction partitioned
    /// across `splits` workgroups per output tile. The
    /// contract (partials scratch ownership, empty trailing
    /// slices, factor range) is
    /// [`crate::kernels::k13_q4k_mmq::K13Pipeline::prepare_split`]'s.
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
        weights: &K2Weights,
        qs: &wgpu::Buffer,
        ds: &wgpu::Buffer,
        output: &wgpu::Buffer,
        batch: u32,
        splits: u32,
    ) -> Result<K14SplitCall> {
        if !(2..=SPLITK_MAX).contains(&splits) {
            return Err(WgpuError::InvalidTensor {
                name: "k14 split call".to_string(),
                detail: format!(
                    "splits {splits} is outside 2..={SPLITK_MAX} (a split of 1 is the unsplit product)"
                ),
            });
        }
        if batch == 0 || batch > MMQ_MAX_BATCH {
            return Err(WgpuError::InvalidTensor {
                name: "k14 split call".to_string(),
                detail: format!(
                    "batch {batch} is outside 1..={MMQ_MAX_BATCH} (the batch tile's row count)"
                ),
            });
        }
        if weights.in_features == 0 || weights.in_features % BLOCK_ELEMS != 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k14 split call".to_string(),
                detail: format!(
                    "input width {} is not a whole number of {BLOCK_ELEMS}-element super-blocks",
                    weights.in_features
                ),
            });
        }
        let tiles = (weights.out_rows as u64).div_ceil(u64::from(self.shape.tile_n));
        if tiles > 65_535 {
            return Err(WgpuError::InvalidTensor {
                name: "k14 split call".to_string(),
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
                name: "k14 split call".to_string(),
                detail: format!(
                    "buffer sizes (qs {}, ds {}, out {}) are below the geometry's needs \
                     ({need_qs}, {need_ds}, {need_out})",
                    qs.size(),
                    ds.size(),
                    output.size()
                ),
            });
        }
        let partials = ctx.scratch_buffer("k14 split partials", u64::from(splits) * need_out);
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
            label: Some("k14 split params"),
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
            label: Some("k14 split bind group"),
            layout: &self.split_pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let reduce = self.reducer.prepare(
            ctx,
            &partials,
            output,
            batch as u64 * weights.out_rows as u64,
            splits,
            "Q6_K",
        )?;
        Ok(K14SplitCall {
            pipeline: self.split_pipeline.clone(),
            bind_group,
            grid,
            reduce,
            diag: ctx.diag_stamp(crate::timestamps::DispatchMeta {
                kernel: "k14_q6k_mmq_split",
                rows: weights.out_rows as u64,
                cols: weights.in_features as u64,
                quant: "Q6_K",
                weight_bytes: weights.byte_len * 212 / 210,
                shape: format!("batch={batch} grid={}x{} splits={splits}", grid.0, grid.1),
            }),
        })
    }

    /// Convenience one-shot: run the MMQ over host-side
    /// quantized rows and read the result back. See
    /// [`crate::kernels::k13_q4k_mmq::K13Pipeline::run`].
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
                name: "k14 input".to_string(),
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
            label: Some("k14 qs"),
            size: qs_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&qs_buffer, 0, &qs_bytes);
        let ds_bytes: Vec<u8> = ds.iter().flat_map(|v| v.to_le_bytes()).collect();
        let ds_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k14 ds"),
            size: ds_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&ds_buffer, 0, &ds_bytes);
        let out_elems = batch as usize * weights.out_rows;
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k14 output"),
            size: (out_elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let call = self.prepare(ctx, weights, &qs_buffer, &ds_buffer, &output_buffer, batch)?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k14 run"),
            });
        call.encode(&mut encoder);
        ctx.queue.submit([encoder.finish()]);
        let out = ctx.read_f32(&output_buffer, out_elems)?;
        dbg_log!(
            "k14 run: batch={batch} out_rows={} bytes_streamed={}",
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
        weights: &K2Weights,
        qs: &[u32],
        ds: &[f32],
        batch: u32,
        splits: u32,
    ) -> Result<Vec<f32>> {
        if qs.len() != qs_words(batch as usize, weights.in_features)
            || ds.len() != ds_vec4s(batch as usize, weights.in_features) * 4
        {
            return Err(WgpuError::InvalidTensor {
                name: "k14 split input".to_string(),
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
            label: Some("k14 split qs"),
            size: qs_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&qs_buffer, 0, &qs_bytes);
        let ds_bytes: Vec<u8> = ds.iter().flat_map(|v| v.to_le_bytes()).collect();
        let ds_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k14 split ds"),
            size: ds_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&ds_buffer, 0, &ds_bytes);
        let out_elems = batch as usize * weights.out_rows;
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k14 split output"),
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
                label: Some("k14 split run"),
            });
        call.encode(&mut encoder);
        ctx.queue.submit([encoder.finish()]);
        let out = ctx.read_f32(&output_buffer, out_elems)?;
        dbg_log!(
            "k14 split run: batch={batch} splits={splits} out_rows={} bytes_streamed={}",
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
        assert_eq!(q6k_staging_bytes(&MmqShape::new(32)), 35_072);
        assert_eq!(q6k_staging_bytes(&MmqShape::new(64)), 45_312);
    }

    #[test]
    fn shipped_shape_respects_storage() {
        let at = |storage: u32| wgpu::Limits {
            max_compute_workgroup_storage_size: storage,
            ..Default::default()
        };
        assert_eq!(shipped_shape(&at(32_768)), None);
        assert_eq!(shipped_shape(&at(45_312)), Some(MmqShape::new(64)));
    }

    #[test]
    fn shader_carries_the_shape_and_dp4a() {
        let s32 = mmq_shader(&MmqShape::new(32));
        assert!(s32.contains("@workgroup_size(128)"));
        assert!(s32.contains("dot4I8Packed"));
        assert!(s32.contains("* 53u"));
        assert!(s32.contains("isum_lo"));
        let s64 = mmq_shader(&MmqShape::new(64));
        assert!(s64.contains("@workgroup_size(256)"));
    }

    #[test]
    fn split_shader_partitions_the_reduction() {
        let s = mmq_split_shader(&MmqShape::new(64));
        assert!(s.contains("splits: u32,"));
        assert!(s.contains(
            "let per_split = (params.blocks_per_row + params.splits - 1u) / params.splits;"
        ));
        assert!(s.contains("let blk_first = wid.y * per_split;"));
        assert!(s.contains("for (var blk = blk_first; blk < blk_last; blk++)"));
        assert!(s.contains("output[(wid.y * params.batch + m0) * params.out_rows + n0]"));
        assert!(s.contains("dot4I8Packed"));
        assert!(s.contains("isum_lo"));
        let plain = mmq_shader(&MmqShape::new(64));
        assert!(!plain.contains("per_split"));
        assert!(!plain.contains("splits"));
        assert!(plain.contains("for (var blk = 0u; blk < params.blocks_per_row; blk++)"));
    }

    #[test]
    fn plain_and_split_shaders_validate_under_naga() {
        // K13's gate, mirrored: every generated variant
        // parses and validates under Naga on adapter-less
        // machines; driver acceptance is the card
        // session's check (bench/splitk-mmq-2026-11-07.md).
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
