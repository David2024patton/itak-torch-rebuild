//! K22: the Q4_K MMQ product reorganized for occupancy, a
//! prototype alternative to K13 for the prefill band.
//!
//! The evidence it answers (bench/prefill-profile-2026-11-05.md,
//! bench/shader-autopsy-2026-11-04.md): on the real Glimmer
//! file's 1,022-token prefill, K13 takes 63.1% of GPU time
//! while streaming its weights at 46.4 GB/s (4.8% of the
//! card's 960 GB/s ceiling) and reaching 17% of FP32 peak,
//! and the autopsy shows why: K13 compiles to VGPR 256, the
//! architectural per-wave maximum, so occupancy sits at the
//! floor and the kernel cannot hide its own dequant-and-dot
//! latency. The lm_head GEMV in the same run streams at
//! 649.5 GB/s, so the memory system is not the suspect; the
//! kernel's register demand is. K13's demand is structural:
//! its generated inner body unrolls all eight sub-blocks of
//! a super-block back to back, holds a 16-wide i32 `isum`
//! array per sub-block across all eight word steps, and
//! folds once per sub-block, so the compiler's scheduler is
//! free (and latency hiding rewards it) to keep several
//! sub-blocks' operand windows in flight at once.
//!
//! K22 runs K13's exact math (the same `vec_dot_q4_K_q8_1`
//! terms, the same K12 Q8_1 input layout, the same staged
//! values, the same bindings and `[batch][out]` output
//! layout, so the session can swap the forms without moving
//! a byte) in llama.cpp `mul_mmq`'s register organization
//! instead:
//!
//! - The workgroup tile is the same 64 output rows x 64
//!   batch rows K13's m64 covers, but 128 threads cover it
//!   (llama's warp tiling: four warps, each on a 16-row x
//!   64-column warp tile), and each thread owns a 4x8
//!   sub-tile: 4 output rows x 8 batch rows, 32 f32
//!   accumulators.
//! - Per 32-element sub-block, the thread first loads its
//!   four rows' operands into a named register cache (the
//!   eight packed words and the `(d * scale, dmin * min)`
//!   pair per row: 40 values), then walks its eight batch
//!   columns one at a time: load that column's eight words
//!   and `(d, s)` pair (10 values), run four scalar i32
//!   dots (one per row, eight DP4A each), and fold each
//!   into its accumulator immediately. There is no `isum`
//!   array and no per-sub-block fold phase; at most one
//!   column's operands and one scalar dot accumulator per
//!   row are live at a time.
//!
//! Source-level live-value arithmetic for the inner peak,
//! against K13's: accumulators 32 f32 (K13: 16), register
//! cache 40 (K13: its operand windows are scheduler-held,
//! not named), the column operand 10 (K13: 8 per word step),
//! one i32 dot accumulator per row in flight, 4 total (K13:
//! the 16-wide `isum`), fold pairs ride inside the cache
//! and the column operand (K13 loads 16 more f32 at its
//! fold), plus a dozen row/column index values that stay
//! live across the K loop in both kernels. K22's named peak
//! is about 100 values; the compiled band the server
//! autopsy is asked to confirm is 96-128 VGPR, off the 256
//! cap and into doubled resident waves. That is a
//! prediction, not a result: register residence is a
//! compiler outcome, and bench/k22-mmq-2026-11-05.md's
//! protocol measures it before any speed number is read.
//!
//! Deliberately NOT copied from llama's `mul_mmq`: its
//! Q4_K staging repack (interleaving two 16-element halves
//! into one word and permuting the activation order to
//! match, with the dot extracting nibbles by shifts). It
//! would halve the register cache's word count (llama
//! holds 4 words per row entry to K22's 8), but it changes
//! the staged byte layout and the B-side order, a second
//! variable and a new correctness surface; K22 keeps
//! K13's staged layout so the prototype's only variable
//! is the register organization. Also not copied: the
//! coopmat form (ranked below the plain rework by the
//! profile note, same ceiling, more risk) and llama's
//! per-device specialization constants (WGSL has none;
//! the geometry here is fixed at generation time).
//!
//! One staging change IS made, and it is K13's own trick:
//! the weight tile's row stride is padded from 64 to 65
//! words, the skew K13 already applies to its activation
//! tile. K22's thread mapping puts four distinct output
//! rows on one wavefront load instruction (K13's mapping
//! puts two), and at a 64-word stride those addresses all
//! land in the same LDS bank; the odd stride puts each in
//! its own bank. The staged values are K13's, byte for
//! byte; only the padding differs (+256 bytes).
//!
//! Routing: the session tries K22 ahead of K13 only when
//! the backend's K22 switch is on, which is config read at
//! backend construction from `WGPU_COMPUTE_K22_MMQ`
//! (see `crate::compute`); the default is OFF, and with it
//! off the session's ladder is the shipped ladder exactly.
//! A device granting less workgroup storage than the
//! staging set gets no pipeline ([`K22Pipeline::for_device`]
//! returns `None`) and keeps the standard form: the
//! fallback is a composition fact the caller reads, never
//! a silent substitution inside a dispatch.

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
/// Output rows one workgroup covers (the tile's height,
/// K13 m64's coverage, so the dispatch grid is unchanged).
pub const K22_TILE_ROWS: u32 = 64;
/// Batch rows one workgroup covers: every row of the
/// chunk, K13's contract (the batch cap is
/// [`MMQ_MAX_BATCH`]).
pub const K22_TILE_BATCH: u32 = 64;
/// Threads per workgroup: four warps, llama's warp tiling
/// (each warp on a 16-row x 64-column warp tile).
pub const K22_WORKGROUP: u32 = 128;
/// Staged words per weight-tile row: the super-block's 64
/// packed words at a padded 65-word row stride (see the
/// module docs for the bank-skew reasoning).
const W_ROW_WORDS: u32 = 65;
/// Staged words per activation-tile row: 64 packed words
/// at K13's own 65-word skewed stride.
const X_ROW_WORDS: u32 = 65;

/// Workgroup storage K22's staging occupies, in bytes: the
/// packed weight words (`64 x 65` words), the activation
/// words (`64 x 65` words), the weight scale pairs
/// (`64 x 8` vec2), and the activation scale pairs
/// (`64 x 8` vec2). 41,472 bytes: K13 m64's set plus the
/// weight tile's row padding.
#[must_use]
pub const fn k22_staging_bytes() -> u32 {
    K22_TILE_ROWS * W_ROW_WORDS * 4
        + K22_TILE_BATCH * X_ROW_WORDS * 4
        + K22_TILE_ROWS * 8 * 8
        + K22_TILE_BATCH * 8 * 8
}

/// The Q4_K MMQ shader in K22's register organization,
/// generated with the tile geometry as literals so the
/// sub-block bodies unroll with constant staged offsets
/// (K13's generation idiom; a runtime-indexed sub-block
/// loop is the form K13's module records as measuring at
/// the f32 GEMM's rate).
fn mmq22_shader() -> String {
    // The inner product, unrolled over the eight
    // sub-blocks at generation time. Per sub-block: the
    // register cache loads (this thread's four rows), then
    // the eight batch columns in turn, each braced so its
    // operand names die with it. Every staged offset is a
    // compile-time constant; the consumption order is the
    // program order, which is the prototype's whole point.
    let mut inner = String::new();
    for sub in 0..8u32 {
        let so = sub * 8;
        inner.push_str("        {\n");
        for r in 0..4 {
            for w in 0..8 {
                inner.push_str(&format!(
                    "            let aq{r}w{w} = w_qs[(rn + {r}u) * 65u + {so}u + {w}u];\n"
                ));
            }
            inner.push_str(&format!(
                "            let as{r} = w_sm[(rn + {r}u) * 8u + {sub}u];\n"
            ));
        }
        for c in 0..8u32 {
            inner.push_str("            {\n");
            for w in 0..8 {
                inner.push_str(&format!(
                    "                let bqw{w} = x_qs[mm{c} * 65u + {so}u + {w}u];\n"
                ));
            }
            inner.push_str(&format!(
                "                let bd = x_ds[mm{c} * 8u + {sub}u];\n"
            ));
            for r in 0..4 {
                inner.push_str(&format!("                var q{r}: i32 = 0;\n"));
                for w in 0..8 {
                    inner.push_str(&format!(
                        "                q{r} += dot4I8Packed(aq{r}w{w}, bqw{w});\n"
                    ));
                }
                inner.push_str(&format!(
                    "                acc[{}] += as{r}.x * (bd.x * f32(q{r})) - as{r}.y * bd.y;\n",
                    r * 8 + c
                ));
            }
            inner.push_str("            }\n");
        }
        inner.push_str("        }\n");
    }
    // The output store, guarded per element (K13's form):
    // rows from the thread mapping, columns enumerated in
    // the same (wsic, cc) order the inner loop walks.
    let mut stores = String::new();
    for r in 0..4 {
        for c in 0..8u32 {
            stores.push_str(&format!(
                "    if (gn{r} < params.out_rows && mm{c} < params.batch) {{\n        output[mm{c} * params.out_rows + gn{r}] = acc[{}];\n    }}\n",
                r * 8 + c
            ));
        }
    }
    // The thread's eight batch columns as lets, in llama's
    // (wsic, cc) decomposition: four 16-wide subtile spans,
    // two columns per span from the thread's column group.
    let mut cols = String::new();
    for c in 0..8u32 {
        let wsic = c / 2;
        let cc = c % 2;
        cols.push_str(&format!(
            "    let mm{c} = {}u + tiwc * 2u + {cc}u;\n",
            wsic * 16
        ));
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

// The weight tile is row-skewed (65 words per row, not
// 64): a wavefront's four row lanes read the same word of
// four different rows, and at a 64-word row stride those
// addresses all land in the same LDS bank. The odd stride
// puts each lane's word in its own bank (the same trick
// K13's activation tile already uses).
var<workgroup> w_qs: array<u32, 4160>;
var<workgroup> x_qs: array<u32, 4160>;
var<workgroup> w_sm: array<vec2<f32>, 512>;
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

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
    let tile_n0 = wid.x * 64u;
    let lane = lid.x;
    let warp = lane / 32u;
    let wl = lane % 32u;
    let tiwr = wl % 4u;
    let tiwc = wl / 4u;
    // This thread's tile-local first output row (its warp's
    // 16-row span, its row group's four rows) and its four
    // global output rows.
    let rn = warp * 16u + tiwr * 4u;
    let gn0 = tile_n0 + rn + 0u;
    let gn1 = tile_n0 + rn + 1u;
    let gn2 = tile_n0 + rn + 2u;
    let gn3 = tile_n0 + rn + 3u;
{cols}    var acc = array<f32, 32>();
    for (var blk = 0u; blk < params.blocks_per_row; blk++) {{
        // Stage the tile's packed weight words and scale
        // pairs for this super-block column. Task (n, g)
        // owns output row n's four word-groups: group g's
        // eight raw words pack into sub-blocks 2g (low
        // nibbles) and 2g + 1 (high nibbles), at the
        // weight tile's skewed row stride. Rows past the
        // tensor's end stage zeros. (K13's staging, with
        // the stride change the module docs describe.)
        for (var task = lane; task < 256u; task += 128u) {{
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
                    w_qs[n * 65u + g * 16u + w] = raw & 0x0f0f0f0fu;
                    w_qs[n * 65u + g * 16u + 8u + w] = (raw >> 4u) & 0x0f0f0f0fu;
                }}
            }} else {{
                for (var h2 = 0u; h2 < 2u; h2++) {{
                    w_sm[n * 8u + g * 2u + h2] = vec2<f32>(0.0);
                }}
                for (var w = 0u; w < 8u; w++) {{
                    w_qs[n * 65u + g * 16u + w] = 0u;
                    w_qs[n * 65u + g * 16u + 8u + w] = 0u;
                }}
            }}
        }}
        // Stage the chunk's activation words and (d, s)
        // pairs for this column: plain copies, coalesced
        // (K13's loops, verbatim).
        for (var flat = lane; flat < params.batch * 64u; flat += 128u) {{
            let m = flat / 64u;
            let jw = flat % 64u;
            x_qs[m * 65u + jw] = qs_in[(m * params.blocks32 + blk * 8u + jw / 8u) * 8u + jw % 8u];
        }}
        for (var flat = lane; flat < params.batch * 8u; flat += 128u) {{
            let m = flat / 8u;
            let j = flat % 8u;
            let v = ds_in[m * params.blocks32 + blk * 8u + j];
            x_ds[flat] = vec2<f32>(v.x, v.y);
        }}
        workgroupBarrier();
        // Inner: the eight sub-blocks, unrolled at
        // generation time. Per sub-block the register
        // cache loads first, then the columns consume it
        // one at a time (see the module docs).
{inner}        workgroupBarrier();
    }}
{stores}}}
"#
    )
}

/// One prepared K22 dispatch: a bind group plus grid
/// dimensions, built once for a (weights, quantized input,
/// output, batch) combination and encodable any number of
/// times. The K13 call's twin: same buffers, same layout.
#[derive(Debug)]
pub struct K22Call {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    grid: (u32, u32),
    /// Diag timestamp stamp (see `crate::timestamps`); `None`
    /// unless the context carries a recorder.
    diag: Option<crate::timestamps::DiagStamp>,
}

impl K22Call {
    /// Record this call's dispatch into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k22 q4_k mmq"),
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

/// The compiled K22 pipeline, in the one shipped geometry
/// (see the module docs). Compilation happens at backend
/// construction, like every kernel in this crate.
#[derive(Debug)]
pub struct K22Pipeline {
    pipeline: wgpu::ComputePipeline,
}

impl K22Pipeline {
    /// Compile K22 for `device`, or `None` when the device
    /// grants less workgroup storage than the staging set
    /// needs: the caller (the backend) records the absence
    /// and the session keeps its standard MMQ form there
    /// (K13's fallback contract).
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] when a device that grants
    /// the storage rejects the shader.
    pub fn for_device(device: &wgpu::Device) -> Result<Option<Self>> {
        let granted = device.limits().max_compute_workgroup_storage_size;
        if granted < k22_staging_bytes() {
            dbg_log!(
                "k22 mmq unavailable: device grants {granted} bytes of workgroup storage, \
                 below the {} its staging set needs",
                k22_staging_bytes()
            );
            return Ok(None);
        }
        Ok(Some(Self::new(device)?))
    }

    /// Compile K22 on `device`. The device's granted
    /// workgroup storage must hold the staging set: a
    /// shortfall is a named refusal here, not a shader
    /// validation failure later.
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] on insufficient workgroup
    /// storage or a rejected shader.
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        let granted = device.limits().max_compute_workgroup_storage_size;
        if granted < k22_staging_bytes() {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "k22 stages {} bytes in workgroup memory but the device grants {granted}",
                    k22_staging_bytes()
                ),
            });
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("k22 q4_k mmq shader"),
            source: wgpu::ShaderSource::Wgsl(mmq22_shader().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k22 q4_k mmq pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        dbg_log!("k22 mmq pipeline built: tile 64x64, workgroup {K22_WORKGROUP}");
        Ok(K22Pipeline { pipeline })
    }

    /// Prepare an MMQ dispatch of `weights` over a batch of
    /// K12-quantized input rows: `qs` / `ds` hold the
    /// quantized form of `batch` rows of
    /// `weights.in_features` f32 (K12's layout), and
    /// `output` holds `batch * weights.out_rows` f32, in
    /// the same `[batch][width]` layout K13 prepares
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
    ) -> Result<K22Call> {
        if batch == 0 || batch > MMQ_MAX_BATCH {
            return Err(WgpuError::InvalidTensor {
                name: "k22 call".to_string(),
                detail: format!(
                    "batch {batch} is outside 1..={MMQ_MAX_BATCH} (the batch tile's row count)"
                ),
            });
        }
        if weights.in_features == 0 || weights.in_features % BLOCK_ELEMS != 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k22 call".to_string(),
                detail: format!(
                    "input width {} is not a whole number of {BLOCK_ELEMS}-element super-blocks",
                    weights.in_features
                ),
            });
        }
        let tiles = (weights.out_rows as u64).div_ceil(u64::from(K22_TILE_ROWS));
        if tiles > 65_535 {
            return Err(WgpuError::InvalidTensor {
                name: "k22 call".to_string(),
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
                name: "k22 call".to_string(),
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
            label: Some("k22 params"),
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
            label: Some("k22 bind group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        Ok(K22Call {
            pipeline: self.pipeline.clone(),
            bind_group,
            grid,
            diag: ctx.diag_stamp(crate::timestamps::DispatchMeta {
                kernel: "k22_q4k_mmq",
                rows: weights.out_rows as u64,
                cols: weights.in_features as u64,
                quant: "Q4_K",
                weight_bytes: weights.byte_len,
                shape: format!(
                    "tile=64x64 wg={K22_WORKGROUP} batch={batch} grid={}x{}",
                    grid.0, grid.1
                ),
            }),
        })
    }

    /// Convenience one-shot: run the product over host-side
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
                name: "k22 input".to_string(),
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
            label: Some("k22 qs"),
            size: qs_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&qs_buffer, 0, &qs_bytes);
        let ds_bytes: Vec<u8> = ds.iter().flat_map(|v| v.to_le_bytes()).collect();
        let ds_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k22 ds"),
            size: ds_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&ds_buffer, 0, &ds_bytes);
        let out_elems = batch as usize * weights.out_rows;
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k22 output"),
            size: (out_elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let call = self.prepare(ctx, weights, &qs_buffer, &ds_buffer, &output_buffer, batch)?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k22 run"),
            });
        call.encode(&mut encoder);
        ctx.queue.submit([encoder.finish()]);
        let out = ctx.read_f32(&output_buffer, out_elems)?;
        dbg_log!(
            "k22 run: batch={batch} out_rows={} bytes_streamed={}",
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
    fn staging_bytes_cover_the_skewed_tiles() {
        // The declared set: two 64x65-word tiles plus the
        // two 64x8 vec2 pair sets.
        assert_eq!(k22_staging_bytes(), 41_472);
        // K13 m64's set is 41,216; the whole difference is
        // the weight tile's 64 rows of 4-byte row padding.
        assert_eq!(
            k22_staging_bytes()
                - crate::kernels::k13_q4k_mmq::q4k_staging_bytes(
                    &crate::kernels::k13_q4k_mmq::MmqShape::new(64)
                ),
            256
        );
        assert_eq!(K22_WORKGROUP, 128);
        assert_eq!(K22_TILE_ROWS * K22_TILE_BATCH, 64 * 64);
    }

    #[test]
    fn shader_carries_the_register_organization() {
        let s = mmq22_shader();
        assert!(s.contains("@workgroup_size(128)"));
        assert!(s.contains("var<workgroup> w_qs: array<u32, 4160>;"));
        assert!(s.contains("var<workgroup> x_qs: array<u32, 4160>;"));
        assert!(s.contains("var acc = array<f32, 32>();"));
        // The whole product's dot count, pinned: 8
        // sub-blocks x 8 columns x 4 rows x 8 words. A
        // generation slip that drops or duplicates a dot
        // block fails here before any GPU runs it.
        assert_eq!(s.matches("dot4I8Packed").count(), 2048);
        // The structural claim of the prototype: no
        // sub-block-wide integer accumulator array (K13's
        // isum) exists; each (row, column) dot accumulates
        // in one scalar and folds immediately.
        assert!(!s.contains("isum"));
        // The register cache loads are present at the
        // skewed weight stride, and the columns enumerate
        // in (wsic, cc) order.
        assert!(s.contains("let aq3w7 = w_qs[(rn + 3u) * 65u + 56u + 7u];"));
        assert!(s.contains("let mm7 = 48u + tiwc * 2u + 1u;"));
        assert!(s.contains("let mm0 = 0u + tiwc * 2u + 0u;"));
    }

    #[test]
    fn shader_validates_under_naga() {
        // The strongest shader gate a machine with no GPU
        // adapter at all can run: the one generated shader
        // (the geometry is fixed, so there is exactly one
        // variant) must parse and validate under Naga, the
        // same front end wgpu feeds the driver with. This
        // proves the WGSL is well-formed and well-typed; it
        // does not prove a driver's compiler accepts it or
        // keeps the cache in registers, which is the server
        // protocol's steps 0 and 1
        // (bench/k22-mmq-2026-11-05.md).
        let src = mmq22_shader();
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("k22 WGSL parse failed: {}", e.emit_to_string(&src)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("k22 Naga validation failed: {e:?}"));
    }
}
