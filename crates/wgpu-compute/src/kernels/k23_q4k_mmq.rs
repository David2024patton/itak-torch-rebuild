//! K23: the Q4_K MMQ (prompt-path) product with llama.cpp's
//! staging repack - the one structural piece K22 deliberately
//! left behind.
//!
//! Logger: `wgpu-compute::kernels::k23_q4k_mmq`. Verbosity is
//! the crate diag channel: set `WGPU_COMPUTE_DIAG=timing` for
//! per-dispatch stamps (kernel name `k23_q4k_mmq`), and the
//! backend prints `mmq k23 config: enabled=...` at
//! construction through the crate `dbg_log!` channel. Main
//! failure modes, all named refusals through [`WgpuError`]:
//! a batch outside `1..=MMQ_MAX_BATCH`, an input width that
//! is not a whole number of Q4_K super-blocks, a tile grid
//! past one dispatch dimension, undersized buffers, and a
//! device whose workgroup-storage grant is below the staged
//! set.
//!
//! # Why this kernel exists
//!
//! The parity extraction (`bench/llama-parity-specs-2026-11-06.md`,
//! gap 2) named exactly one structural difference left between
//! our prompt-path MMQ and llama.cpp's: the staging. K13 stages
//! one whole 256-element super-block column per barrier round,
//! with the weight tile's scale pairs interleaved beside the
//! data they scale (weights in `w_sm[n][sub]`, activations in
//! `x_ds[m][sub]`) and a single skew, the activation rows
//! pitched at 65 words. K22 changed only the register
//! organization and compiled to the same VGPR 256 cap as K13
//! (autopsy in `bench/k22-mmq-2026-11-05.md`: ACO re-inflated
//! the register file and the occupancy premise died before any
//! speed test). If the staging shape is what pins the compiled
//! form, the staging is what has to change. K23 is that
//! experiment, and it changes nothing else: K13's tile
//! coverage (64 output rows by 64 batch rows), K13's thread
//! organization (256 threads, a 4x4 sub-tile per thread, the
//! `isum` array, scalar `dot4I8Packed`, immediate folds in
//! K13's exact order and expression), K13's bindings, params,
//! prepare contract, and `[batch][out]` output layout. The
//! session can swap K13, K22, and K23 without moving a byte;
//! only the staging differs.
//!
//! # The staging, following llama's arithmetic
//!
//! llama's coopmat MMQ (mul_mmq_cm1) stages in rounds of
//! `BK x BK_STEP` K elements, not whole super-block columns:
//! `BK = 32` (mul_mmq_cm1.comp:89) and `BK_STEP = 4`
//! (mul_mmq_cm1.comp:93, the non-`MUL_MAT_ID` form), so one
//! round covers 128 elements and a 256-element Q4_K
//! super-block column takes two rounds. The staged arrays are
//! split and pitched (mul_mmq_cm1.comp:97-110):
//!
//! - `buf_a_qs[BM * QPITCH]` with
//!   `QPITCH = BK_STEP * (BK / 4) + 4 = 36`
//!   (mul_mmq_cm1.comp:97): each row's round segment is 32
//!   words of qs data plus a 4-word skew against bank
//!   conflicts on the consumption loads.
//! - `buf_a_dm[BM * BK_STEP]` (mul_mmq_cm1.comp:101): the
//!   weight scale pairs live in their own array, indexed
//!   `buf_a_dm[ks * BM + buf_ib]` - sub-block-major, not
//!   interleaved with the qs words.
//! - `buf_b_qs[BN * QPITCH]` (mul_mmq_cm1.comp:106),
//!   `buf_b_d[BN * BK_STEP]` and `buf_b_s[BN * BK_STEP]`
//!   (mul_mmq_cm1.comp:107,110): the activation side mirrors
//!   the split, with `d` and the folded sum `s` in separate
//!   scalar arrays, indexed `buf_b_d[ks * BN + buf_ib]`.
//!
//! K23 transcribes those addresses with `BM = BN = 64` (the
//! tile):
//!
//! - `w_qs[n * 36 + ks * 8 + loadr * 2 + half]`: llama's Q4_K
//!   `block_a_to_shmem` writes exactly
//!   `buf_ib * QPITCH + ks * (BK / 4) + loadr * 2 + {0,1}`
//!   (mul_mmq_cm1_funcs.glsl, the `DATA_A_Q4_K` branch at
//!   line 246, `block_a_to_shmem` at line 274): `loadr` in
//!   `0..4` walks the four word-pairs of a sub-block's
//!   8-word group, and the staged word is the raw Q4_K word
//!   masked to the low nibbles for an even sub-block or
//!   shifted down for an odd one - llama's `block_a_load`
//!   nibble select on `ib % 2`. Weight rows past `out_rows`
//!   stage zeros in qs and dm alike, as K13 stages them.
//! - `w_dm[ks * 64 + n]`: written by the `loadr == 0` task of
//!   each (row, sub-block), llama's own placement
//!   (`block_a_to_shmem` writes `buf_a_dm[ks * BM + buf_ib]`
//!   on `loadr == 0`). The VALUE keeps K13's convention,
//!   `(d * scale, dmin * min)`; cm1 stages
//!   `(d * scale, -(dmin * min))` and subtracts in its fold
//!   where K13's fold text carries the negation. Same
//!   product, same order, only the array and the address
//!   are new - K23's fold text is K13's, byte for byte.
//! - `x_qs[m * 36 + ks * 8 + loadr * 4 + c]`: llama's
//!   `block_b_to_shmem` (mul_mmq_cm1_funcs.glsl:529) writes
//!   the four components of one loaded `ivec4` at
//!   `buf_ib * QPITCH + ks * (BK / 4) + loadr * 4 + 0..3`,
//!   with `loadr` in `0..2` covering the block's two
//!   4-word halves. The loads come from `block_b_load`
//!   (same file, just above): `qs = data_b[ib_outer].qs[
//!   ib_inner * 2 + loadr]`, one vector out of a
//!   `block_q8_1_x4` group.
//! - `x_d[ks * 64 + m]` and `x_s[ks * 64 + m]`: the
//!   `loadr == 0` task also splits the block's scale pair
//!   out of its group into the separate arrays, llama's
//!   `buf_b_d[ks * BN + buf_ib] = d` and
//!   `buf_b_s[ks * BN + buf_ib] = s`.
//!
//! # What the de-interleave means on our buffers
//!
//! Honesty about the B side, because the gap note's wording
//! ("de-interleaves the x4-packed Q8_1 layout into
//! consumption order") describes llama's DRAM format, not
//! ours. llama's activation buffer is `block_q8_1_x4`
//! (types.glsl:267): four Q8_1 blocks grouped, their four
//! `(d, s)` pairs collected in a header array `ds[4]` ahead
//! of the qs words, each block's 8 words contiguous at
//! `qs[8 * ib_inner .. 8 * ib_inner + 7]` (the producer
//! writes exactly that, quantize_q8_1.comp:87 and :116).
//! `block_b_to_shmem` (scalar form at
//! mul_mmq_funcs.glsl:651, cm1 form at
//! mul_mmq_cm1_funcs.glsl:529) pulls each block's pair out
//! of the group header into the block's own staged slot and
//! scatters its words to the pitched consumption addresses.
//! Our K12 output is two flat streams instead - `qs` words
//! in plain (row, block, word) order and one `vec4` of
//! scale data per block - so there is no group header to
//! undo and the within-block permutation is the identity by
//! construction. What K23 copies is the structural content:
//! the staged addresses, the split scale arrays, the pitch,
//! and the 128-element staging round. The bench note
//! (`bench/k23-mmq-repack-2026-11-06.md`) says the same and
//! prices what the transcription costs.
//!
//! # Deliberately not copied
//!
//! - K22's register organization. One design variable per
//!   experiment: K23 keeps K13's organization so a compiled
//!   or measured difference is attributable to the staging.
//! - The scalar mul_mmq A-side repack: llama's scalar
//!   `block_a_to_shmem` (mul_mmq_funcs.glsl:530, the Q4_K
//!   branch) interleaves the two nibble halves into one
//!   staged word (`vals0 | vals1 << 4`) and its dot loop
//!   extracts with shifts per dot. That repack serves the
//!   scalar path's register-file consumption; K23's spec is
//!   the cm1 split layout, whose A side stages nibbles
//!   already separated, so the interleave has no consumer
//!   here. It remains unbuilt, named so a later lane can
//!   take it deliberately.
//! - Coopmat itself, and llama's per-device tile
//!   specialization: out of scope for a staging experiment,
//!   and the coopmat feasibility note
//!   (`bench/coopmat-feasibility-2026-11-06.md`) owns that
//!   question.
//!
//! # The staged set
//!
//! Per workgroup: `w_qs` 64 x 36 words (9,216 B), `x_qs`
//! 64 x 36 words (9,216 B), `w_dm` 4 x 64 vec2 pairs
//! (2,048 B), `x_d` and `x_s` 4 x 64 floats each (1,024 B
//! apiece): 22,528 bytes total ([`k23_staging_bytes`]).
//! Against K13's 41,216 B at the same tile: a round stages
//! half the K span, so the qs arrays halve and the pitch
//! adds 4 words per row per array. Whether the smaller set
//! buys occupancy depends on the register count the
//! compiler produces, which is exactly what step 0 of the
//! card protocol in `bench/k23-mmq-repack-2026-11-06.md`
//! reads before any speed claim.

use crate::device::GpuContext;
use crate::error::{Result, WgpuError};
use crate::kernels::k12_q8_quant::{ds_vec4s, qs_words};
use crate::kernels::k13_q4k_mmq::MMQ_MAX_BATCH;
use crate::kernels::k1_q4k_gemv::K1Weights;

use crate::diag::dbg_log;

/// Elements per Q4_K super-block (and per staged column).
const BLOCK_ELEMS: usize = 256;

/// Words per Q4_K super-block in the packed weight words.
const BLOCK_WORDS: u32 = 36;

/// Output rows per K23 workgroup: K13's/K22's tile.
pub const K23_TILE_ROWS: u32 = 64;

/// Batch rows per K23 workgroup: K13's/K22's tile.
pub const K23_TILE_BATCH: u32 = 64;

/// Threads per K23 workgroup: K13's organization (sixteen
/// output-row groups by sixteen batch-row groups, a 4x4
/// sub-tile per thread).
pub const K23_WORKGROUP: u32 = 256;

/// llama's staging block width in K elements
/// (mul_mmq_cm1.comp:89, `#define BK 32`): one Q8_1 block.
pub const K23_BK: u32 = 32;

/// Blocks staged per round (mul_mmq_cm1.comp:93,
/// `#define BK_STEP 4`, the non-`MUL_MAT_ID` form). A
/// 256-element super-block column is two rounds.
pub const K23_BK_STEP: u32 = 4;

/// Row pitch of each staged qs array, in words:
/// `BK_STEP * (BK / 4) + 4 = 36` (mul_mmq_cm1.comp:97).
/// The `+ 4` is llama's bank-conflict skew.
pub const K23_QPITCH: u32 = K23_BK_STEP * (K23_BK / 4) + 4;

/// Total workgroup bytes the K23 shader stages: the two
/// pitched qs arrays, the split weight scale-pair array,
/// and the two split activation scale arrays.
pub const fn k23_staging_bytes() -> u32 {
    K23_TILE_ROWS * K23_QPITCH * 4 // w_qs
        + K23_TILE_BATCH * K23_QPITCH * 4 // x_qs
        + K23_BK_STEP * K23_TILE_ROWS * 8 // w_dm (vec2<f32>)
        + K23_BK_STEP * K23_TILE_BATCH * 4 // x_d
        + K23_BK_STEP * K23_TILE_BATCH * 4 // x_s
}

/// The 4 sub-block bodies of one staging round, generated
/// with every staged offset a compile-time constant (the
/// pitch arithmetic is folded at generation time; `ks` is
/// the sub-block within the round). The math text - the
/// `isum` array, the dot order, the fold expression - is
/// K13's inner body, with the staged addresses rewritten
/// to the split layout: qs reads at row pitch
/// [`K23_QPITCH`], weight pairs from `w_dm[ks * 64 + row]`,
/// activation pairs assembled from `x_d` / `x_s` at
/// `[ks * 64 + row]`.
fn inner_round() -> String {
    let mut inner = String::new();
    for ks in 0..K23_BK_STEP {
        let so = ks * 8; // staged qs offset of this sub-block
        let dm_base = ks * K23_TILE_BATCH; // scale-array row base
        inner.push_str("        {\n            var isum = array<i32, 16>();\n");
        for w in 0..8 {
            for j in 0..4 {
                inner.push_str(&format!(
                    "            let xv{j}w{w} = x_qs[m{j} * 36u + {so}u + {w}u];\n"
                ));
            }
            for i in 0..4 {
                inner.push_str(&format!(
                    "            let wv{i}w{w} = w_qs[(n_sub * 4u + {i}u) * 36u + {so}u + {w}u];\n"
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
                "            let c{i} = w_dm[{dm_base}u + n_sub * 4u + {i}u];\n"
            ));
        }
        for j in 0..4 {
            inner.push_str(&format!(
                "            let dm{j} = vec2<f32>(x_d[{dm_base}u + m{j}], x_s[{dm_base}u + m{j}]);\n"
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

/// The staging text for one round, in llama's task shapes.
/// The A side runs one task per (tile row, sub-block,
/// loader quarter) - cm1's `block_a_load` /
/// `block_a_to_shmem` decomposition, with `loadr == 0`
/// depositing the row's scale pair into `w_dm`. The B side
/// runs one task per (batch row, sub-block, loader half),
/// scattering one `vec4` per task in `block_b_to_shmem`'s
/// order, with `loadr == 0` splitting the `(d, s)` pair
/// into `x_d` / `x_s`. `round` enters only the SOURCE
/// addresses (which sub-blocks of the column this round
/// covers); the staged addresses are round-invariant
/// because each round reuses the same staged set.
fn stage_round(round: u32) -> String {
    let sub_base = round * K23_BK_STEP;
    let last = sub_base + K23_BK_STEP - 1;
    let mut s = String::new();
    s.push_str(&format!(
        "        // Stage round {round} of the weight tile (sub-blocks\n"
    ));
    s.push_str(&format!(
        "        // {sub_base}..{last} of this column) in cm1's task shape:\n"
    ));
    s.push_str("        // one task per (row, sub-block, loader quarter).\n");
    s.push_str("        for (var task = lane; task < 1024u; task += 256u) {\n");
    s.push_str("            let n = task / 16u;\n");
    s.push_str("            let ks = (task % 16u) / 4u;\n");
    s.push_str("            let loadr = task % 4u;\n");
    s.push_str(&format!("            let sub = {sub_base}u + ks;\n"));
    s.push_str("            let g = sub / 2u;\n");
    s.push_str("            let gn = tile_n0 + n;\n");
    s.push_str("            let dst = n * 36u + ks * 8u + loadr * 2u;\n");
    s.push_str("            if (gn < params.out_rows) {\n");
    s.push_str(&format!(
        "                let bw = (gn * params.blocks_per_row + blk) * {BLOCK_WORDS}u;\n"
    ));
    s.push_str("                let raw0 = weights[bw + 4u + g * 8u + loadr * 2u];\n");
    s.push_str("                let raw1 = weights[bw + 4u + g * 8u + loadr * 2u + 1u];\n");
    s.push_str("                if (sub % 2u == 0u) {\n");
    s.push_str("                    w_qs[dst] = raw0 & 0x0f0f0f0fu;\n");
    s.push_str("                    w_qs[dst + 1u] = raw1 & 0x0f0f0f0fu;\n");
    s.push_str("                } else {\n");
    s.push_str("                    w_qs[dst] = (raw0 >> 4u) & 0x0f0f0f0fu;\n");
    s.push_str("                    w_qs[dst + 1u] = (raw1 >> 4u) & 0x0f0f0f0fu;\n");
    s.push_str("                }\n");
    s.push_str("                if (loadr == 0u) {\n");
    s.push_str("                    let hw = weights[bw];\n");
    s.push_str("                    let dd = f16_to_f32(hw & 0xffffu);\n");
    s.push_str("                    let ddm = f16_to_f32(hw >> 16u);\n");
    s.push_str("                    let sw = array<u32, 3>(weights[bw + 1u], weights[bw + 2u], weights[bw + 3u]);\n");
    s.push_str("                    let sm = scale_min_k4(sub, sw);\n");
    s.push_str(
        "                    w_dm[ks * 64u + n] = vec2<f32>(dd * f32(sm.x), ddm * f32(sm.y));\n",
    );
    s.push_str("                }\n");
    s.push_str("            } else {\n");
    s.push_str("                w_qs[dst] = 0u;\n");
    s.push_str("                w_qs[dst + 1u] = 0u;\n");
    s.push_str("                if (loadr == 0u) {\n");
    s.push_str("                    w_dm[ks * 64u + n] = vec2<f32>(0.0);\n");
    s.push_str("                }\n");
    s.push_str("            }\n");
    s.push_str("        }\n");
    s.push_str(&format!(
        "        // Stage round {round} of the activation tile in\n"
    ));
    s.push_str("        // block_b_to_shmem's order: one ivec4 scattered per\n");
    s.push_str("        // (row, block, loader half); loader half 0 also splits\n");
    s.push_str("        // the (d, s) pair into x_d / x_s.\n");
    s.push_str("        for (var task = lane; task < params.batch * 8u; task += 256u) {\n");
    s.push_str("            let m = task / 8u;\n");
    s.push_str("            let ks = (task % 8u) / 2u;\n");
    s.push_str("            let loadr = task % 2u;\n");
    s.push_str(&format!(
        "            let b32 = blk * 8u + {sub_base}u + ks;\n"
    ));
    s.push_str("            let src = (m * params.blocks32 + b32) * 8u + loadr * 4u;\n");
    s.push_str("            let v = vec4<u32>(qs_in[src], qs_in[src + 1u], qs_in[src + 2u], qs_in[src + 3u]);\n");
    s.push_str("            let dst = m * 36u + ks * 8u + loadr * 4u;\n");
    s.push_str("            x_qs[dst] = v.x;\n");
    s.push_str("            x_qs[dst + 1u] = v.y;\n");
    s.push_str("            x_qs[dst + 2u] = v.z;\n");
    s.push_str("            x_qs[dst + 3u] = v.w;\n");
    s.push_str("            if (loadr == 0u) {\n");
    s.push_str("                let dv = ds_in[m * params.blocks32 + b32];\n");
    s.push_str("                x_d[ks * 64u + m] = dv.x;\n");
    s.push_str("                x_s[ks * 64u + m] = dv.y;\n");
    s.push_str("            }\n");
    s.push_str("        }\n");
    s
}

/// The K23 shader: K13's thread organization and math with
/// the cm1 staging. Generated because the sub-block bodies
/// are unrolled at generation time (K13's stated reason:
/// every staged offset a compile-time constant), and the
/// two staging rounds are emitted as text for the same
/// reason - the round's source sub-blocks fold into the
/// staging addresses.
fn mmq23_shader() -> String {
    let inner = inner_round();
    let stage0 = stage_round(0);
    let stage1 = stage_round(1);
    // The output store, guarded per element (K13's form).
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
    blocks32: u32,
}}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> qs_in: array<u32>;
@group(0) @binding(2) var<storage, read> ds_in: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

// The staged set is llama mul_mmq_cm1's split layout at
// this tile: each qs array holds one 128-element staging
// round per row at QPITCH = 36 words (32 data + 4 skew),
// and the scale pairs live apart from the qs words -
// weight pairs in w_dm, activation d and s in their own
// arrays - all indexed [sub-block][row]. See the module
// docs for the arithmetic and the citations.
var<workgroup> w_qs: array<u32, 2304>;
var<workgroup> x_qs: array<u32, 2304>;
var<workgroup> w_dm: array<vec2<f32>, 256>;
var<workgroup> x_d: array<f32, 256>;
var<workgroup> x_s: array<f32, 256>;

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

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
    let tile_n0 = wid.x * 64u;
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
{stage0}        workgroupBarrier();
        // Inner: this round's four sub-blocks, unrolled at
        // generation time (constant staged offsets).
{inner}        workgroupBarrier();
{stage1}        workgroupBarrier();
{inner}        workgroupBarrier();
    }}
{stores}}}
"#
    )
}

/// One prepared K23 dispatch: a bind group plus grid
/// dimensions, built once for a (weights, quantized input,
/// output, batch) combination and encodable any number of
/// times. The K13/K22 call's twin: same buffers, same
/// layout.
#[derive(Debug)]
pub struct K23Call {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    grid: (u32, u32),
    /// Diag timestamp stamp (see `crate::timestamps`); `None`
    /// unless the context carries a recorder.
    diag: Option<crate::timestamps::DiagStamp>,
}

impl K23Call {
    /// Record this call's dispatch into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k23 q4_k mmq"),
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

/// The compiled K23 pipeline, in the one shipped geometry
/// (see the module docs). Compilation happens at backend
/// construction, like every kernel in this crate.
#[derive(Debug)]
pub struct K23Pipeline {
    pipeline: wgpu::ComputePipeline,
}

impl K23Pipeline {
    /// Compile K23 for `device`, or `None` when the device
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
        if granted < k23_staging_bytes() {
            dbg_log!(
                "k23 mmq unavailable: device grants {granted} bytes of workgroup storage, \
                 below the {} its staging set needs",
                k23_staging_bytes()
            );
            return Ok(None);
        }
        Ok(Some(Self::new(device)?))
    }

    /// Compile K23 on `device`. The device's granted
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
        if granted < k23_staging_bytes() {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "k23 stages {} bytes in workgroup memory but the device grants {granted}",
                    k23_staging_bytes()
                ),
            });
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("k23 q4_k mmq shader"),
            source: wgpu::ShaderSource::Wgsl(mmq23_shader().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k23 q4_k mmq pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        dbg_log!("k23 mmq pipeline built: tile 64x64, workgroup {K23_WORKGROUP}");
        Ok(K23Pipeline { pipeline })
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
    ) -> Result<K23Call> {
        if batch == 0 || batch > MMQ_MAX_BATCH {
            return Err(WgpuError::InvalidTensor {
                name: "k23 call".to_string(),
                detail: format!(
                    "batch {batch} is outside 1..={MMQ_MAX_BATCH} (the batch tile's row count)"
                ),
            });
        }
        if weights.in_features == 0 || weights.in_features % BLOCK_ELEMS != 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k23 call".to_string(),
                detail: format!(
                    "input width {} is not a whole number of {BLOCK_ELEMS}-element super-blocks",
                    weights.in_features
                ),
            });
        }
        let tiles = (weights.out_rows as u64).div_ceil(u64::from(K23_TILE_ROWS));
        if tiles > 65_535 {
            return Err(WgpuError::InvalidTensor {
                name: "k23 call".to_string(),
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
                name: "k23 call".to_string(),
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
            label: Some("k23 params"),
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
            label: Some("k23 bind group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        Ok(K23Call {
            pipeline: self.pipeline.clone(),
            bind_group,
            grid,
            diag: ctx.diag_stamp(crate::timestamps::DispatchMeta {
                kernel: "k23_q4k_mmq",
                rows: weights.out_rows as u64,
                cols: weights.in_features as u64,
                quant: "Q4_K",
                weight_bytes: weights.byte_len,
                shape: format!(
                    "tile=64x64 wg={K23_WORKGROUP} batch={batch} grid={}x{}",
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
                name: "k23 input".to_string(),
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
            label: Some("k23 qs"),
            size: qs_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&qs_buffer, 0, &qs_bytes);
        let ds_bytes: Vec<u8> = ds.iter().flat_map(|v| v.to_le_bytes()).collect();
        let ds_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k23 ds"),
            size: ds_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&ds_buffer, 0, &ds_bytes);
        let out_elems = batch as usize * weights.out_rows;
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k23 output"),
            size: (out_elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let call = self.prepare(ctx, weights, &qs_buffer, &ds_buffer, &output_buffer, batch)?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k23 run"),
            });
        call.encode(&mut encoder);
        ctx.queue.submit([encoder.finish()]);
        let out = ctx.read_f32(&output_buffer, out_elems)?;
        dbg_log!(
            "k23 run: batch={batch} out_rows={} bytes_streamed={}",
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
    fn qpitch_is_llamas_arithmetic() {
        // mul_mmq_cm1.comp:97: QPITCH = BK_STEP * (BK / 4) + 4.
        assert_eq!(K23_BK, 32);
        assert_eq!(K23_BK_STEP, 4);
        assert_eq!(K23_QPITCH, 36);
        assert_eq!(K23_QPITCH, K23_BK_STEP * (K23_BK / 4) + 4);
        // One round stages BK * BK_STEP = 128 elements, so a
        // 256-element super-block column is exactly two
        // rounds; the generated shader emits both.
        assert_eq!(K23_BK * K23_BK_STEP * 2, BLOCK_ELEMS as u32);
    }

    #[test]
    fn staging_bytes_are_the_split_layouts_set() {
        // w_qs 64*36*4 = 9216; x_qs 64*36*4 = 9216;
        // w_dm 4*64*8 = 2048; x_d 4*64*4 = 1024;
        // x_s 4*64*4 = 1024. Total 22,528.
        assert_eq!(k23_staging_bytes(), 22_528);
        // Materially smaller than K13's whole-column set at
        // the same tile (41,216 B): the rounds stage half
        // the K span.
        assert!(
            k23_staging_bytes()
                < crate::kernels::k13_q4k_mmq::q4k_staging_bytes(
                    &crate::kernels::k13_q4k_mmq::MmqShape::new(64)
                )
        );
        assert_eq!(K23_WORKGROUP, 256);
        assert_eq!(K23_TILE_ROWS * K23_TILE_BATCH, 64 * 64);
    }

    #[test]
    fn shader_uses_the_split_pitched_layout() {
        let s = mmq23_shader();
        // K13's workgroup size and thread organization.
        assert!(s.contains("@workgroup_size(256)"));
        // The staged set, at llama's pitch: two 64-row qs
        // arrays of 36-word pitch, the split scale arrays.
        assert!(s.contains("var<workgroup> w_qs: array<u32, 2304>;"));
        assert!(s.contains("var<workgroup> x_qs: array<u32, 2304>;"));
        assert!(s.contains("var<workgroup> w_dm: array<vec2<f32>, 256>;"));
        assert!(s.contains("var<workgroup> x_d: array<f32, 256>;"));
        assert!(s.contains("var<workgroup> x_s: array<f32, 256>;"));
        // Pitched consumption addresses: row pitch 36, one
        // sub-block's segment 8 words wide.
        assert!(s.contains("let xv0w3 = x_qs[m0 * 36u + 8u + 3u];"));
        assert!(s.contains("let wv3w7 = w_qs[(n_sub * 4u + 3u) * 36u + 24u + 7u];"));
        // Split scale reads: weight pairs sub-block-major,
        // the activation pair assembled from two arrays.
        assert!(s.contains("let c0 = w_dm[64u + n_sub * 4u + 0u];"));
        assert!(s.contains("let dm2 = vec2<f32>(x_d[128u + m2], x_s[128u + m2]);"));
        // The staging scatter in block_b_to_shmem's shape,
        // and the A side's pitched deposit.
        assert!(s.contains("let dst = m * 36u + ks * 8u + loadr * 4u;"));
        assert!(s.contains("x_qs[dst + 3u] = v.w;"));
        assert!(s.contains("w_dm[ks * 64u + n] = vec2<f32>(dd * f32(sm.x), ddm * f32(sm.y));"));
        // K13's 65-word skew is gone from this shader: the
        // pitch carries the skew instead.
        assert!(!s.contains("* 65u"));
        // Both rounds are emitted (source sub-blocks 0..3
        // and 4..7 of the column).
        assert!(s.contains("let sub = 0u + ks;"));
        assert!(s.contains("let sub = 4u + ks;"));
        // The dot count is K13's total: 8 sub-blocks x 8
        // word steps x 16 dots = 1,024 per column pass
        // (4 sub-blocks per round, two rounds).
        assert_eq!(s.matches("dot4I8Packed").count(), 1_024);
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
        // what the compiled form costs, which is the server
        // protocol's steps 0 and 1
        // (bench/k23-mmq-repack-2026-11-06.md).
        let src = mmq23_shader();
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("k23 WGSL parse failed: {}", e.emit_to_string(&src)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("k23 Naga validation failed: {e:?}"));
    }
}
