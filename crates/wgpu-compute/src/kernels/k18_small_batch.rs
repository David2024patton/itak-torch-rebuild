//! K18: the small-batch matrix products for Q4_K, Q5_K,
//! and Q6_K, built for the row band the speculative loop
//! actually runs at: 2..=16 rows.
//!
//! The gap this fills is measured, not hypothetical
//! (bench/spec-gpu-port-2026-11-03.md, round economics).
//! Below [`crate::session`]'s GEMM_MIN_ROWS (16) the
//! resident session's projections run in GEMV form: one
//! full weight stream per row, so a 7-row verify forward
//! streams the 16 GB target seven times (about 140 ms at
//! the measured 815 GB/s) and the DFlash draft's 16-row
//! block pays the same disease at 57 ms. Forcing the
//! existing matrix forms down to those rows was tested
//! and made it worse (7 rows: 140 to 253 ms, dispatch
//! count doubled): the GEMM and MMQ tiles are built for
//! 32 and 64 rows. What is missing is a product whose
//! tile is sized for the band: stage each weight tile
//! ONCE per workgroup and apply it to every row in the
//! pass, so the weight traffic per dispatch is one
//! stream of the tensor however many rows ride it.
//!
//! The math is the MMQ math, term for term: K12's Q8_1
//! activations, exact i32 DP4A dots per sub-block, and
//! the per-sub-block scale fold of `vec_dot_*_q8_1`
//! (K13's for Q4_K, K14's per-half fold for Q6_K; Q5_K
//! folds like Q4_K once its 5-bit quants are assembled).
//! Only the tile geometry and the batch cap differ from
//! K13/K14: the batch tile is 16 rows, so the workgroup
//! is `tile_n` threads (four batch lanes of four rows,
//! `tile_n / 4` output lanes of four rows, the same 4x4
//! per-thread sub-tile), and the activation staging
//! shrinks to the band. At tile 32 the Q4_K and Q5_K
//! staging sets fit even a default-limits device
//! (15,424 of 16,384 bytes); the Q6_K set is 64 bytes
//! over that grant at tile 32, so on such a device Q6_K
//! gets no pipeline and keeps its GEMV form, the same
//! named-absence contract as K13/K14.
//!
//! Routing is the session's, and it is config: the
//! session prefers K18 inside the band only up to the
//! backend's `small_batch_max_rows` threshold (env
//! `WGPU_COMPUTE_SMALL_BATCH_MAX_ROWS`, default 1, which
//! is off: no batch qualifies). The wide-MMQ lesson is
//! the reason for the shipped default: an unproven
//! kernel never enters the ladder by default, however
//! good its streaming math looks. The server proof
//! sequence that can move the default is in
//! bench/small-batch-2026-11-04.md.
//!
//! There is no activation form here (K13's note
//! applies: the transform is finished in K12's
//! quantizer, upstream of the product), and no fused
//! finish: the Q5_K head's softcap path stays inside
//! K3, whose call the block-logits path prepares
//! directly rather than through the projection ladder
//! this kernel serves.

use crate::device::GpuContext;
use crate::error::{Result, WgpuError};
use crate::kernels::k12_q8_quant::{ds_vec4s, qs_words};
use crate::kernels::k1_q4k_gemv::K1Weights;
use crate::kernels::k2_q6k_gemv::K2Weights;
use crate::kernels::k3_q5k_gemv::K3Weights;

use crate::diag::dbg_log;

/// Elements per super-block (all three families).
const BLOCK_ELEMS: usize = 256;
/// Words per Q4_K super-block (144 bytes).
const BLOCK_WORDS_Q4K: u32 = 36;
/// Words per Q5_K super-block (176 bytes).
const BLOCK_WORDS_Q5K: u32 = 44;
/// Words per Q6_K super-block in the device buffer (the
/// padded 212-byte stride; see K2's module docs).
const BLOCK_WORDS_Q6K_DEV: u32 = 53;

/// Batch rows one K18 workgroup covers at most: the top
/// of the small-batch band. The band's floor (2) is a
/// routing fact, not a kernel limit: `prepare` accepts
/// 1..=[`SMALL_MAX_BATCH`] like K13 accepts 1..=64, and
/// the session never routes a single row here (decode
/// stays GEMV).
pub const SMALL_MAX_BATCH: u32 = 16;

/// The K18 tile geometry: output rows per workgroup.
/// Each thread owns a 4x4 sub-tile (four output rows,
/// four batch rows), so with four batch lanes the
/// workgroup is `tile_n` threads: `(tile_n / 4) * 4`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmallShape {
    /// Output rows one workgroup computes (the staged
    /// weight tile's height).
    pub tile_n: u32,
    /// Threads per workgroup: `tile_n` (see the type
    /// docs).
    pub workgroup: u32,
}

impl SmallShape {
    /// A shape with the given tile; the workgroup size
    /// is derived (see the type docs).
    #[must_use]
    pub const fn new(tile_n: u32) -> SmallShape {
        SmallShape {
            tile_n,
            workgroup: tile_n,
        }
    }

    /// The bench/check name: `s<tile_n>`.
    #[must_use]
    pub fn name(&self) -> String {
        format!("s{}", self.tile_n)
    }

    /// Check the shape's geometry rules (the storage
    /// check is the caller's, against the device it will
    /// run on).
    pub(crate) fn validate(&self) -> Result<()> {
        if !matches!(self.tile_n, 32 | 64) {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "small shape {}: tile_n {} is not 32 or 64",
                    self.name(),
                    self.tile_n
                ),
            });
        }
        Ok(())
    }
}

/// Workgroup storage the Q4_K staging occupies at
/// `shape`, in bytes: the packed weight words
/// (`tile_n` x 64 words), the activation words (batch
/// cap x 65 words, the skewed stride K13 introduced),
/// the weight scale pairs (`tile_n` x 8 vec2), and the
/// activation scale pairs (batch cap x 8 vec2).
#[must_use]
pub const fn q4k_small_staging_bytes(shape: &SmallShape) -> u32 {
    shape.tile_n * 64 * 4
        + SMALL_MAX_BATCH * 65 * 4
        + shape.tile_n * 8 * 8
        + SMALL_MAX_BATCH * 8 * 8
}

/// Workgroup storage the Q5_K staging occupies at
/// `shape`: the same layout as Q4_K's (packed assembled
/// bytes, vec2 scale pairs), so the same bytes.
#[must_use]
pub const fn q5k_small_staging_bytes(shape: &SmallShape) -> u32 {
    q4k_small_staging_bytes(shape)
}

/// Workgroup storage the Q6_K staging occupies at
/// `shape`, in bytes: like Q4_K's but the activation
/// scale entries are vec4 (this family reads the half
/// sums), which is what pushes tile 32 sixty-four bytes
/// past a default-limits grant.
#[must_use]
pub const fn q6k_small_staging_bytes(shape: &SmallShape) -> u32 {
    shape.tile_n * 64 * 4
        + SMALL_MAX_BATCH * 65 * 4
        + shape.tile_n * 8 * 8
        + SMALL_MAX_BATCH * 8 * 16
}

/// The tile geometry a device with these limits runs
/// for the staging set `staging` sizes: the first
/// candidate, in preference order (wide first, K13's
/// bench-backed order), whose staging the granted
/// workgroup storage holds. `None` below the narrow
/// tile's need: no K18 pipeline for that family on that
/// device (see the module docs for the fallback
/// contract).
fn shipped_shape_with(
    limits: &wgpu::Limits,
    staging: fn(&SmallShape) -> u32,
) -> Option<SmallShape> {
    let storage = limits.max_compute_workgroup_storage_size;
    [SmallShape::new(64), SmallShape::new(32)]
        .into_iter()
        .find(|shape| storage >= staging(shape))
}

/// The shipped Q4_K tile for these limits (see
/// [`shipped_shape_with`]).
#[must_use]
pub fn shipped_shape_q4k(limits: &wgpu::Limits) -> Option<SmallShape> {
    shipped_shape_with(limits, q4k_small_staging_bytes)
}

/// The shipped Q5_K tile for these limits (see
/// [`shipped_shape_with`]).
#[must_use]
pub fn shipped_shape_q5k(limits: &wgpu::Limits) -> Option<SmallShape> {
    shipped_shape_with(limits, q5k_small_staging_bytes)
}

/// The shipped Q6_K tile for these limits (see
/// [`shipped_shape_with`]).
#[must_use]
pub fn shipped_shape_q6k(limits: &wgpu::Limits) -> Option<SmallShape> {
    shipped_shape_with(limits, q6k_small_staging_bytes)
}

/// IEEE 754 half to f32, shared WGSL text (K13's,
/// mirroring gguf::dequant::f16_to_f32 for every value
/// a weight file can hold).
const F16_TO_F32_WGSL: &str = r#"
// IEEE 754 half to f32, mirroring gguf::dequant::f16_to_f32 for
// every value a weight file can hold.
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

/// ggml's `get_scale_min_k4` extraction, shared WGSL
/// text for the Q4_K and Q5_K forms (their 12 packed
/// scale bytes are packed identically).
const SCALE_MIN_K4_WGSL: &str = r#"
// One of the 12 packed scale bytes of a super-block.
fn scale_byte(sw: array<u32, 3>, i: u32) -> u32 {
    return (sw[i / 4u] >> (8u * (i % 4u))) & 0xffu;
}

// ggml get_scale_min_k4: sub-block j's (scale, min) 6-bit pair.
fn scale_min_k4(j: u32, sw: array<u32, 3>) -> vec2<u32> {
    if (j < 4u) {
        return vec2<u32>(scale_byte(sw, j) & 63u, scale_byte(sw, j + 4u) & 63u);
    }
    return vec2<u32>(
        (scale_byte(sw, j + 4u) & 0xfu) | ((scale_byte(sw, j - 4u) >> 6u) << 4u),
        (scale_byte(sw, j + 4u) >> 4u) | ((scale_byte(sw, j) >> 6u) << 4u),
    );
}
"#;

/// The activation staging block, identical in all
/// three forms (K13's text): the chunk's activation
/// words and scale entries for this super-block
/// column, plain coalesced copies into the skewed
/// workgroup tile. `x_ds` element type differs per
/// family and is part of each shader's declarations;
/// the copy is the same because Q4_K/Q5_K read a vec4
/// and keep its (d, s) pair, which is exactly what
/// their staged vec2 holds.
const ACT_STAGE_WGSL: &str = r#"        // Stage the chunk's activation words and (d, s)
        // pairs for this column: plain copies, coalesced.
        for (var flat = lane; flat < params.batch * 64u; flat += WG_PLACEHOLDER) {
            let m = flat / 64u;
            let jw = flat % 64u;
            x_qs[m * 65u + jw] = qs_in[(m * params.blocks32 + blk * 8u + jw / 8u) * 8u + jw % 8u];
        }
        for (var flat = lane; flat < params.batch * 8u; flat += WG_PLACEHOLDER) {
            let m = flat / 8u;
            let j = flat % 8u;
            let v = ds_in[m * params.blocks32 + blk * 8u + j];
            x_ds[flat] = DS_EXPR_PLACEHOLDER;
        }
        workgroupBarrier();
"#;

/// The output stores, identical in all three forms
/// (K13's text): sixteen guarded stores per thread.
fn stores_wgsl() -> String {
    let mut stores = String::new();
    for i in 0..4 {
        for j in 0..4 {
            stores.push_str(&format!(
                "    if (n{i} < params.out_rows && m{j} < params.batch) {{\n        output[m{j} * params.out_rows + n{i}] = acc[{}];\n    }}\n",
                i * 4 + j
            ));
        }
    }
    stores
}

/// The lane setup shared by all three forms: K13's
/// 4x4 sub-tile assignment with four batch lanes
/// instead of sixteen, which is the whole geometric
/// difference from the MMQ pair.
const LANE_SETUP_WGSL: &str = r#"    let tile_n0 = wid.x * TILE_PLACEHOLDER;
    let lane = lid.x;
    let n_sub = lane / 4u;
    let m_sub = lane % 4u;
    let m0 = m_sub * 4u;
    let m1 = m0 + 1u;
    let m2 = m0 + 2u;
    let m3 = m0 + 3u;
    let n0 = tile_n0 + n_sub * 4u;
    let n1 = n0 + 1u;
    let n2 = n0 + 2u;
    let n3 = n0 + 3u;
    var acc = array<f32, 16>();
"#;

/// Substitute the shape literals into a shared WGSL
/// chunk: workgroup size, tile width, and the staged
/// activation expression.
fn fill_template(text: &str, shape: &SmallShape, ds_expr: &str) -> String {
    text.replace("WG_PLACEHOLDER", &format!("{}u", shape.workgroup))
        .replace("TILE_PLACEHOLDER", &format!("{}u", shape.tile_n))
        .replace("DS_EXPR_PLACEHOLDER", ds_expr)
}

/// The inner product for the Q4_K-fold families
/// (Q4_K, Q5_K): K13's generation, unrolled over the
/// eight sub-blocks at generation time so every staged
/// offset is a compile-time constant. Per sub-block:
/// sixteen DP4A dots per staged word position over the
/// thread's 4x4 sub-tile, then the per-sub-block scale
/// fold `c * (d_x * sumi) - cmin * s`.
fn inner_q4k_fold() -> String {
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
    inner
}

/// The inner product for Q6_K: K14's generation. Per
/// sub-block: exact integer dots over the two
/// 16-element halves, then the per-half fold
/// `d * sc_h * (d_x * sumi_h - 32 * sh_h)`.
fn inner_q6k_fold() -> String {
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
    inner
}

/// The Q4_K small-batch shader for `shape`: K13's
/// stage and fold at the small-batch geometry (see
/// the module docs).
fn q4k_shader(shape: &SmallShape) -> String {
    let tile_n = shape.tile_n;
    let wg = shape.workgroup;
    let w_words = tile_n * 64;
    let w_pairs = tile_n * 8;
    let x_words = SMALL_MAX_BATCH * 65;
    let x_pairs = SMALL_MAX_BATCH * 8;
    let lane_setup = fill_template(LANE_SETUP_WGSL, shape, "");
    let act_stage = fill_template(ACT_STAGE_WGSL, shape, "vec2<f32>(v.x, v.y)");
    let inner = inner_q4k_fold();
    let stores = stores_wgsl();
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

var<workgroup> w_qs: array<u32, {w_words}>;
// The activation tile is row-skewed (65 words per row,
// K13's bank-conflict reasoning: a wavefront's batch
// lanes read the same word of different rows, and the
// odd stride puts each lane's word in its own bank).
var<workgroup> x_qs: array<u32, {x_words}>;
var<workgroup> w_sm: array<vec2<f32>, {w_pairs}>;
var<workgroup> x_ds: array<vec2<f32>, {x_pairs}>;
{F16_TO_F32_WGSL}{SCALE_MIN_K4_WGSL}
@compute @workgroup_size({wg})
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
{lane_setup}    for (var blk = 0u; blk < params.blocks_per_row; blk++) {{
        // Stage the tile's packed weight words and scale
        // pairs for this super-block column (K13's stage,
        // verbatim): task (n, g) owns output row n's four
        // word-groups; group g's eight raw words pack into
        // sub-blocks 2g (low nibbles) and 2g + 1 (high
        // nibbles). Rows past the tensor's end stage
        // zeros.
        for (var task = lane; task < {tile_n}u * 4u; task += {wg}u) {{
            let n = task / 4u;
            let g = task % 4u;
            let gn = tile_n0 + n;
            if (gn < params.out_rows) {{
                let bw = (gn * params.blocks_per_row + blk) * {BLOCK_WORDS_Q4K}u;
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
{act_stage}        // Inner: the eight sub-blocks, unrolled at
        // generation time (constant staged offsets).
{inner}        workgroupBarrier();
    }}
{stores}}}
"#
    )
}

/// The Q5_K small-batch shader for `shape`: the Q4_K
/// form's tiling and fold, with the stage assembling
/// each element's 5-bit quant into its own byte (a
/// `ql` nibble plus one bit of the shared `qh` plane:
/// bit `j` of `qh[l]` belongs to sub-block `j`), packed
/// four per word for `dot4I8Packed`, K14's assembly
/// idiom.
fn q5k_shader(shape: &SmallShape) -> String {
    let tile_n = shape.tile_n;
    let wg = shape.workgroup;
    let w_words = tile_n * 64;
    let w_pairs = tile_n * 8;
    let x_words = SMALL_MAX_BATCH * 65;
    let x_pairs = SMALL_MAX_BATCH * 8;
    let lane_setup = fill_template(LANE_SETUP_WGSL, shape, "");
    let act_stage = fill_template(ACT_STAGE_WGSL, shape, "vec2<f32>(v.x, v.y)");
    let inner = inner_q4k_fold();
    let stores = stores_wgsl();
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

var<workgroup> w_qs: array<u32, {w_words}>;
// The activation tile is row-skewed (65 words per row,
// K13's bank-conflict reasoning).
var<workgroup> x_qs: array<u32, {x_words}>;
var<workgroup> w_sm: array<vec2<f32>, {w_pairs}>;
var<workgroup> x_ds: array<vec2<f32>, {x_pairs}>;
{F16_TO_F32_WGSL}{SCALE_MIN_K4_WGSL}
@compute @workgroup_size({wg})
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
{lane_setup}    for (var blk = 0u; blk < params.blocks_per_row; blk++) {{
        // Stage the tile's assembled weight words and
        // scale pairs for this super-block column. Task
        // (n, j) owns output row n's sub-block j: the
        // ql plane starts at word 12 of the block, the
        // qh plane at word 4; sub-block j reads group
        // g = j / 2's ql words (low nibbles for even j,
        // high for odd) and bit j of each qh byte. Rows
        // past the tensor's end stage zeros.
        for (var task = lane; task < {tile_n}u * 8u; task += {wg}u) {{
            let n = task / 8u;
            let j = task % 8u;
            let gn = tile_n0 + n;
            if (gn < params.out_rows) {{
                let bw = (gn * params.blocks_per_row + blk) * {BLOCK_WORDS_Q5K}u;
                let hw = weights[bw];
                let dd = f16_to_f32(hw & 0xffffu);
                let ddm = f16_to_f32(hw >> 16u);
                let sw = array<u32, 3>(weights[bw + 1u], weights[bw + 2u], weights[bw + 3u]);
                let sm = scale_min_k4(j, sw);
                w_sm[n * 8u + j] = vec2<f32>(dd * f32(sm.x), ddm * f32(sm.y));
                let g = j / 2u;
                let high = (j & 1u) == 1u;
                for (var w = 0u; w < 8u; w++) {{
                    let qlb = unpack4xU8(weights[bw + 12u + g * 8u + w]);
                    let qhb = unpack4xU8(weights[bw + 4u + w]);
                    let nibs = select(qlb & vec4<u32>(0xfu), (qlb >> vec4<u32>(4u)) & vec4<u32>(0xfu), vec4<bool>(high));
                    let q = nibs | (((qhb >> vec4<u32>(j)) & vec4<u32>(1u)) << vec4<u32>(4u));
                    w_qs[(n * 8u + j) * 8u + w] = pack4xU8(q);
                }}
            }} else {{
                w_sm[n * 8u + j] = vec2<f32>(0.0);
                for (var w = 0u; w < 8u; w++) {{
                    w_qs[(n * 8u + j) * 8u + w] = 0u;
                }}
            }}
        }}
{act_stage}        // Inner: the eight sub-blocks, unrolled at
        // generation time (constant staged offsets).
{inner}        workgroupBarrier();
    }}
{stores}}}
"#
    )
}

/// The Q6_K small-batch shader for `shape`: K14's
/// stage and per-half fold at the small-batch
/// geometry.
fn q6k_shader(shape: &SmallShape) -> String {
    let tile_n = shape.tile_n;
    let wg = shape.workgroup;
    let w_words = tile_n * 64;
    let w_pairs = tile_n * 8;
    let x_words = SMALL_MAX_BATCH * 65;
    let x_pairs = SMALL_MAX_BATCH * 8;
    let lane_setup = fill_template(LANE_SETUP_WGSL, shape, "");
    let act_stage = fill_template(ACT_STAGE_WGSL, shape, "v");
    let inner = inner_q6k_fold();
    let stores = stores_wgsl();
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

var<workgroup> w_qs: array<u32, {w_words}>;
// The activation tile is row-skewed (65 words per row,
// K13's bank-conflict reasoning).
var<workgroup> x_qs: array<u32, {x_words}>;
var<workgroup> w_sc: array<vec2<f32>, {w_pairs}>;
var<workgroup> x_ds: array<vec4<f32>, {x_pairs}>;
{F16_TO_F32_WGSL}
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
{lane_setup}    for (var blk = 0u; blk < params.blocks_per_row; blk++) {{
        // Stage the tile's packed weight words and scale
        // pairs for this super-block column (K14's stage,
        // verbatim): task (n, j) owns output row n's
        // sub-block j; it assembles the 6-bit quants into
        // bytes (a ql nibble plus two qh bits) and packs
        // four per word. Rows past the tensor's end stage
        // zeros.
        for (var task = lane; task < {tile_n}u * 8u; task += {wg}u) {{
            let n = task / 8u;
            let j = task % 8u;
            let gn = tile_n0 + n;
            if (gn < params.out_rows) {{
                let bw = (gn * params.blocks_per_row + blk) * {BLOCK_WORDS_Q6K_DEV}u;
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
{act_stage}        // Inner: the eight sub-blocks, unrolled at
        // generation time (constant staged offsets).
{inner}        workgroupBarrier();
    }}
{stores}}}
"#
    )
}

/// One prepared K18 dispatch: a bind group plus grid
/// dimensions, built once for a (weights, quantized
/// input, output, batch) combination and encodable any
/// number of times. One call type serves all three
/// dtype forms: the pipeline inside already encodes
/// which stage and fold the dispatch runs.
#[derive(Debug)]
pub struct K18Call {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    grid: (u32, u32),
    label: &'static str,
    /// Diag timestamp stamp (see `crate::timestamps`); `None`
    /// unless the context carries a recorder.
    diag: Option<crate::timestamps::DiagStamp>,
}

impl K18Call {
    /// Record this call's dispatch into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(self.label),
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

/// Compile one family's pipeline at `shape`, with the
/// family's staging accounting and shader. Shared by
/// the three forms so their construction terms cannot
/// drift: validate the shape, refuse by name when the
/// device's granted workgroup storage cannot hold the
/// staging set, then compile.
fn build_pipeline(
    device: &wgpu::Device,
    shape: SmallShape,
    staging: fn(&SmallShape) -> u32,
    shader: fn(&SmallShape) -> String,
    tag: &'static str,
    shader_label: &'static str,
    pipeline_label: &'static str,
) -> Result<wgpu::ComputePipeline> {
    shape.validate()?;
    let granted = device.limits().max_compute_workgroup_storage_size;
    if granted < staging(&shape) {
        return Err(WgpuError::Pipeline {
            detail: format!(
                "k18 {tag} shape {} stages {} bytes in workgroup memory but the device grants {granted}",
                shape.name(),
                staging(&shape)
            ),
        });
    }
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(shader_label),
        source: wgpu::ShaderSource::Wgsl(shader(&shape).into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(pipeline_label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    dbg_log!("k18 {tag} small pipeline built: shape {}", shape.name());
    Ok(pipeline)
}

/// Compile the shipped shape for one family (see
/// [`shipped_shape_with`]), or `None` when the device
/// grants less workgroup storage than the smallest
/// staging set needs: the caller (the backend) records
/// the absence and the session keeps its GEMV form on
/// such a device.
fn for_device(
    device: &wgpu::Device,
    staging: fn(&SmallShape) -> u32,
    shader: fn(&SmallShape) -> String,
    tag: &'static str,
    shader_label: &'static str,
    pipeline_label: &'static str,
) -> Result<Option<wgpu::ComputePipeline>> {
    match shipped_shape_with(&device.limits(), staging) {
        Some(shape) => Ok(Some(build_pipeline(
            device,
            shape,
            staging,
            shader,
            tag,
            shader_label,
            pipeline_label,
        )?)),
        None => {
            dbg_log!(
                "k18 {tag} small unavailable: device grants {} bytes of workgroup storage, \
                 below the smallest staging set",
                device.limits().max_compute_workgroup_storage_size
            );
            Ok(None)
        }
    }
}

/// The shipped shape a built pipeline of `tag` runs:
/// recomputed from the device's limits at construction
/// (the pipeline stores it alongside).
fn shape_of(limits: &wgpu::Limits, staging: fn(&SmallShape) -> u32) -> Option<SmallShape> {
    shipped_shape_with(limits, staging)
}

/// Prepare a K18 dispatch (shared by the three forms;
/// the geometry contract is K13's with the band cap in
/// place of the MMQ batch cap).
#[allow(clippy::too_many_arguments)]
fn prepare_call(
    pipeline: &wgpu::ComputePipeline,
    shape: SmallShape,
    ctx: &GpuContext,
    tag: &'static str,
    call_label: &'static str,
    weights_buffer: &wgpu::Buffer,
    in_features: usize,
    out_rows: usize,
    qs: &wgpu::Buffer,
    ds: &wgpu::Buffer,
    output: &wgpu::Buffer,
    batch: u32,
) -> Result<K18Call> {
    let name = format!("k18 {tag} call");
    if batch == 0 || batch > SMALL_MAX_BATCH {
        return Err(WgpuError::InvalidTensor {
            name,
            detail: format!(
                "batch {batch} is outside 1..={SMALL_MAX_BATCH} (the small-batch row band)"
            ),
        });
    }
    if in_features == 0 || in_features % BLOCK_ELEMS != 0 {
        return Err(WgpuError::InvalidTensor {
            name,
            detail: format!(
                "input width {in_features} is not a whole number of {BLOCK_ELEMS}-element super-blocks"
            ),
        });
    }
    let tiles = (out_rows as u64).div_ceil(u64::from(shape.tile_n));
    if tiles > 65_535 {
        return Err(WgpuError::InvalidTensor {
            name,
            detail: format!(
                "{out_rows} output rows need {tiles} tile workgroups, past one dispatch dimension"
            ),
        });
    }
    let need_qs = (qs_words(batch as usize, in_features) * 4) as u64;
    let need_ds = (ds_vec4s(batch as usize, in_features) * 16) as u64;
    let need_out = batch as u64 * out_rows as u64 * 4;
    if qs.size() < need_qs || ds.size() < need_ds || output.size() < need_out {
        return Err(WgpuError::InvalidTensor {
            name,
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
        (in_features / BLOCK_ELEMS) as u32,
        out_rows as u32,
        batch,
        (in_features / 32) as u32,
    ];
    let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
    let params_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("k18 params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&params_buffer, 0, &param_bytes);
    let entries = vec![
        wgpu::BindGroupEntry {
            binding: 0,
            resource: weights_buffer.as_entire_binding(),
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
        label: Some("k18 bind group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    Ok(K18Call {
        pipeline: pipeline.clone(),
        bind_group,
        grid,
        label: call_label,
        diag: ctx.diag_stamp(crate::timestamps::DispatchMeta {
            kernel: "k18_small_batch",
            rows: out_rows as u64,
            cols: in_features as u64,
            quant: match tag {
                "q4k" => "Q4_K",
                "q5k" => "Q5_K",
                "q6k" => "Q6_K",
                _ => "",
            },
            // The bound buffer's own size: for Q6_K that is
            // the padded device layout the kernel streams.
            weight_bytes: weights_buffer.size(),
            shape: format!("batch={batch} tag={tag} grid={}x{}", grid.0, grid.1),
        }),
    })
}

/// The one-shot run behind each family's `run` (K13's
/// flow): validate the host quantized inputs against
/// the batch and width, upload, prepare, dispatch,
/// read back.
#[allow(clippy::too_many_arguments)]
fn run_product(
    pipeline: &wgpu::ComputePipeline,
    shape: SmallShape,
    ctx: &GpuContext,
    tag: &'static str,
    call_label: &'static str,
    weights_buffer: &wgpu::Buffer,
    in_features: usize,
    out_rows: usize,
    byte_len: u64,
    qs: &[u32],
    ds: &[f32],
    batch: u32,
) -> Result<Vec<f32>> {
    if qs.len() != qs_words(batch as usize, in_features)
        || ds.len() != ds_vec4s(batch as usize, in_features) * 4
    {
        return Err(WgpuError::InvalidTensor {
            name: format!("k18 {tag} input"),
            detail: format!(
                "quantized inputs hold {} words and {} scale values, not the \
                 {} words and {} values {batch} rows of width {in_features} need",
                qs.len(),
                ds.len(),
                qs_words(batch as usize, in_features),
                ds_vec4s(batch as usize, in_features) * 4,
            ),
        });
    }
    let qs_bytes: Vec<u8> = qs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let qs_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("k18 qs"),
        size: qs_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&qs_buffer, 0, &qs_bytes);
    let ds_bytes: Vec<u8> = ds.iter().flat_map(|v| v.to_le_bytes()).collect();
    let ds_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("k18 ds"),
        size: ds_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&ds_buffer, 0, &ds_bytes);
    let out_elems = batch as usize * out_rows;
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("k18 output"),
        size: (out_elems * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let call = prepare_call(
        pipeline,
        shape,
        ctx,
        tag,
        call_label,
        weights_buffer,
        in_features,
        out_rows,
        &qs_buffer,
        &ds_buffer,
        &output_buffer,
        batch,
    )?;
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("k18 run"),
        });
    call.encode(&mut encoder);
    ctx.queue.submit([encoder.finish()]);
    let out = ctx.read_f32(&output_buffer, out_elems)?;
    dbg_log!("k18 {tag} run: batch={batch} out_rows={out_rows} bytes_streamed={byte_len}");
    Ok(out)
}

/// Generate the three family pipeline structs, which
/// differ only in their weights type, staging
/// accounting, shader, and labels. Each carries the
/// same construction contract (see [`for_device`]) and
/// the same prepare/run surface as K13/K14.
macro_rules! small_pipeline {
    (
        $(#[$meta:meta])*
        $name:ident, $weights:ty, $tag:literal,
        $staging:ident, $shader:ident,
        $shader_label:literal, $pipeline_label:literal, $call_label:literal
    ) => {
        $(#[$meta])*
        #[derive(Debug)]
        pub struct $name {
            pipeline: wgpu::ComputePipeline,
            shape: SmallShape,
        }

        impl $name {
            /// Compile the shipped shape for `device`
            /// (see [`shipped_shape_with`]), or `None`
            /// when the device grants less workgroup
            /// storage than the smallest staging set
            /// needs: the caller (the backend) records
            /// the absence and the session keeps its GEMV
            /// form on such a device.
            ///
            /// # Errors
            ///
            /// [`WgpuError::Pipeline`] when a device that
            /// grants the storage rejects the shader.
            pub fn for_device(device: &wgpu::Device) -> Result<Option<Self>> {
                match for_device(
                    device,
                    $staging,
                    $shader,
                    $tag,
                    $shader_label,
                    $pipeline_label,
                )? {
                    Some(pipeline) => Ok(Some(Self {
                        pipeline,
                        shape: shape_of(&device.limits(), $staging)
                            .expect("a built pipeline has a shipped shape"),
                    })),
                    None => Ok(None),
                }
            }

            /// Compile an explicit [`SmallShape`] on
            /// `device`. The shape is validated, and the
            /// device's granted workgroup storage must
            /// hold the staging set: a shortfall is a
            /// named refusal here, not a shader
            /// validation failure later.
            ///
            /// # Errors
            ///
            /// [`WgpuError::Pipeline`] on a bad shape or
            /// insufficient workgroup storage.
            pub fn with_shape(device: &wgpu::Device, shape: SmallShape) -> Result<Self> {
                let pipeline = build_pipeline(
                    device,
                    shape,
                    $staging,
                    $shader,
                    $tag,
                    $shader_label,
                    $pipeline_label,
                )?;
                Ok(Self { pipeline, shape })
            }

            /// The shape this pipeline runs.
            #[must_use]
            pub fn shape(&self) -> SmallShape {
                self.shape
            }

            /// Prepare a small-batch dispatch of
            /// `weights` over a batch of K12-quantized
            /// input rows: `qs` / `ds` hold the quantized
            /// form of `batch` rows of the tensor's input
            /// width (K12's layout), and `output` holds
            /// `batch * weights.out_rows` f32 in the same
            /// `[batch][width]` layout the GEMV prepares
            /// against, so a plan can swap one form for
            /// the other without moving a byte.
            ///
            /// # Errors
            ///
            /// [`WgpuError::InvalidTensor`] when the
            /// batch is outside the small-batch band, the
            /// input width is not a whole number of
            /// super-blocks, the output rows need more
            /// workgroups than one dispatch dimension
            /// holds, or a buffer is smaller than the
            /// geometry addresses.
            pub fn prepare(
                &self,
                ctx: &GpuContext,
                weights: &$weights,
                qs: &wgpu::Buffer,
                ds: &wgpu::Buffer,
                output: &wgpu::Buffer,
                batch: u32,
            ) -> Result<K18Call> {
                prepare_call(
                    &self.pipeline,
                    self.shape,
                    ctx,
                    $tag,
                    $call_label,
                    weights.buffer(),
                    weights.in_features,
                    weights.out_rows,
                    qs,
                    ds,
                    output,
                    batch,
                )
            }

            /// Convenience one-shot: run the product over
            /// host-side quantized rows and read the
            /// result back. The check/bench data path;
            /// the resident session prepares against its
            /// own buffers instead.
            ///
            /// # Errors
            ///
            /// [`WgpuError::InvalidTensor`] when the
            /// quantized inputs are not sized for the
            /// batch and width, or the batch is out of
            /// band; [`WgpuError::Readback`] when the
            /// result cannot be mapped back.
            pub fn run(
                &self,
                ctx: &GpuContext,
                weights: &$weights,
                qs: &[u32],
                ds: &[f32],
                batch: u32,
            ) -> Result<Vec<f32>> {
                run_product(
                    &self.pipeline,
                    self.shape,
                    ctx,
                    $tag,
                    $call_label,
                    weights.buffer(),
                    weights.in_features,
                    weights.out_rows,
                    weights.byte_len,
                    qs,
                    ds,
                    batch,
                )
            }
        }
    };
}

small_pipeline!(
    /// The compiled K18 Q4_K small-batch pipeline.
    /// Compilation happens at backend construction,
    /// like every kernel in this crate.
    K18Q4KPipeline, K1Weights, "q4k",
    q4k_small_staging_bytes, q4k_shader,
    "k18 q4_k small shader", "k18 q4_k small pipeline", "k18 q4_k small"
);

small_pipeline!(
    /// The compiled K18 Q5_K small-batch pipeline.
    /// Compilation happens at backend construction,
    /// like every kernel in this crate.
    K18Q5KPipeline, K3Weights, "q5k",
    q5k_small_staging_bytes, q5k_shader,
    "k18 q5_k small shader", "k18 q5_k small pipeline", "k18 q5_k small"
);

small_pipeline!(
    /// The compiled K18 Q6_K small-batch pipeline.
    /// Compilation happens at backend construction,
    /// like every kernel in this crate.
    K18Q6KPipeline, K2Weights, "q6k",
    q6k_small_staging_bytes, q6k_shader,
    "k18 q6_k small shader", "k18 q6_k small pipeline", "k18 q6_k small"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_names_and_staging() {
        assert_eq!(SmallShape::new(32).name(), "s32");
        assert_eq!(SmallShape::new(32).workgroup, 32);
        assert_eq!(SmallShape::new(64).name(), "s64");
        assert_eq!(SmallShape::new(64).workgroup, 64);
        // The module docs' arithmetic: tile 32 Q4_K /
        // Q5_K staging fits a default-limits grant;
        // Q6_K at tile 32 is 64 bytes over it.
        assert_eq!(q4k_small_staging_bytes(&SmallShape::new(32)), 15_424);
        assert_eq!(q4k_small_staging_bytes(&SmallShape::new(64)), 25_664);
        assert_eq!(q5k_small_staging_bytes(&SmallShape::new(32)), 15_424);
        assert_eq!(q6k_small_staging_bytes(&SmallShape::new(32)), 16_448);
        assert_eq!(q6k_small_staging_bytes(&SmallShape::new(64)), 26_688);
    }

    #[test]
    fn shipped_shape_respects_storage() {
        let at = |storage: u32| wgpu::Limits {
            max_compute_workgroup_storage_size: storage,
            ..Default::default()
        };
        assert_eq!(shipped_shape_q4k(&at(16_384)), Some(SmallShape::new(32)));
        assert_eq!(shipped_shape_q5k(&at(16_384)), Some(SmallShape::new(32)));
        assert_eq!(shipped_shape_q6k(&at(16_384)), None);
        assert_eq!(shipped_shape_q6k(&at(16_448)), Some(SmallShape::new(32)));
        assert_eq!(shipped_shape_q4k(&at(32_768)), Some(SmallShape::new(64)));
        assert_eq!(shipped_shape_q6k(&at(32_768)), Some(SmallShape::new(64)));
        assert_eq!(shipped_shape_q4k(&at(8_192)), None);
    }

    #[test]
    fn shape_validation_refuses_by_name() {
        let err = SmallShape::new(100).validate().unwrap_err();
        assert!(
            err.to_string().contains("tile_n 100 is not 32 or 64"),
            "refusal names the tile set: {err}"
        );
        assert!(SmallShape::new(32).validate().is_ok());
        assert!(SmallShape::new(64).validate().is_ok());
    }

    #[test]
    fn shaders_carry_the_shape_and_dp4a() {
        for shader in [
            q4k_shader(&SmallShape::new(32)),
            q5k_shader(&SmallShape::new(32)),
            q6k_shader(&SmallShape::new(32)),
        ] {
            assert!(shader.contains("@workgroup_size(32)"));
            assert!(shader.contains("array<u32, 2048>"));
            assert!(shader.contains("array<u32, 1040>"));
            assert!(shader.contains("dot4I8Packed"));
            assert!(shader.contains("let n_sub = lane / 4u;"));
        }
        let s64 = q4k_shader(&SmallShape::new(64));
        assert!(s64.contains("@workgroup_size(64)"));
        assert!(s64.contains("array<u32, 4096>"));
        // The family stages: Q4_K splits nibbles in
        // place, Q5_K assembles 5-bit bytes from the qh
        // plane, Q6_K assembles 6-bit bytes.
        assert!(q4k_shader(&SmallShape::new(32)).contains("raw & 0x0f0f0f0fu"));
        assert!(q5k_shader(&SmallShape::new(32)).contains("qhb >> vec4<u32>(j)"));
        assert!(q6k_shader(&SmallShape::new(32)).contains("scale_at"));
    }
}
