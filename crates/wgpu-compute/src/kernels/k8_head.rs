//! K8: the bookends of the forward (BACKEND.md Section 3): the
//! embedding gather at the front and the head finishing math at
//! the back.
//!
//! **Embedding gather.** The trait feeds the embedding table as
//! a [`TensorView`], and this module accepts exactly the dtypes
//! `cpu-compute` (the spec) accepts there, refusing the rest by
//! name with the tensor and its type: F32, F16, BF16, Q8_0,
//! Q4_K, Q5_K, Q6_K. One kernel per storage family:
//!
//! - F32 rows: a plain indexed copy.
//! - F16 / BF16 rows: a two-byte element conversion (f16 via the
//!   WGSL unpack intrinsic, bf16 by the shift-into-place the
//!   format is).
//! - Quantized rows: the element is dequantized in-kernel from
//!   its block bytes. Every decoder restates `gguf`'s reference
//!   dequantizer instruction for instruction (the K4_K sub-block
//!   scale unpacking shared by Q5_K, Q5_K's shared fifth-bit
//!   array, Q6_K's four interleaved streams, Q8_0's signed
//!   bytes), the same discipline K1's shader keeps for Q4_K
//!   matrices. Elements are addressed globally (token row start
//!   plus channel), so a row that is not a whole number of
//!   blocks gathers exactly the way cpu-compute's whole-tensor
//!   fallback dequantizes it.
//!
//! A gather is one token row per dispatch; a chunk of tokens is
//! one call per row, matching how `cpu-compute`'s `embed` fills
//! its output row by row through `dequantize_rows`.
//!
//! **Head finishing.** The forward's last arithmetic after the
//! head projection: scale every logit by `cfg.logit_scale`, then
//! the tanh softcap `cap * tanh(x / cap)`, in that order (the
//! `muse_glimmer::softcap_scale` law). The kernel here is the GPU
//! form of that math for the device-resident pipeline it will
//! serve: one thread per logit over a two-dimensional grid,
//! because the 202,048 wide vocab exceeds a single dispatch
//! dimension's comfortable reach once rows multiply. The trait's
//! `lm_head` deliberately does NOT apply this finishing (the
//! model applies `softcap_scale` itself after `lm_head`
//! returns, so applying it twice would be a bug); the kernel is
//! proven against `softcap_scale` directly in this crate's
//! tests, which is the seam it will be wired into later.

use muse_glimmer::{Dtype, GlimmerConfig, GlimmerError, Result as GlimmerResult, TensorView};

use crate::device::GpuContext;
use crate::error::{Result, WgpuError};
use crate::kernels::k1_q4k_gemv::grid_dims;

use crate::diag::dbg_log;

/// Build a [`GlimmerError::StepFailed`] for `step` with `detail`.
fn step_failed(step: &'static str, detail: impl Into<String>) -> GlimmerError {
    GlimmerError::StepFailed {
        step,
        detail: detail.into(),
    }
}

/// The mainline ggml type id of a [`Dtype`], for naming types in
/// refusals (mirrors the table in the sibling compute modules).
fn dtype_id(dtype: Dtype) -> u32 {
    match dtype {
        Dtype::F32 => 0,
        Dtype::F16 => 1,
        Dtype::Q4_0 => 2,
        Dtype::Q8_0 => 8,
        Dtype::Q4K => 12,
        Dtype::Q5K => 13,
        Dtype::Q6K => 14,
        Dtype::Bf16 => 30,
        Dtype::Unknown(id) => id,
    }
}

/// Threads per workgroup for the gather and finish kernels.
const WORKGROUP: u32 = 256;

/// Workgroups in the argmax reduction's first stage: each
/// reduces its strided share of the logits to one (value,
/// index) pair in the partials buffer, and the finish stage
/// reduces those pairs to the single winner. 128 groups x
/// 256 threads keeps the per-thread stride short at the real
/// vocabulary (202,048 logits: about seven elements per
/// thread) while the finish stage stays one workgroup.
pub const ARGMAX_GROUPS: usize = 128;

/// The argmax shader: two entry points sharing the combine
/// rule and the workgroup reduction arrays. The rule is the
/// host sampler's (`itak-engine`'s greedy pick): the maximum
/// value wins, and an exact tie goes to the LOWEST index. The
/// rule is a total order (value descending, index ascending),
/// so the staged reduction returns exactly what a sequential
/// scan returns on any finite input. A thread that visits no
/// element (and a group whose partial is empty) carries
/// (-inf, u32::MAX), which can never win a tie against a real
/// element; NaN logits would poison the comparisons the same
/// way they poison the host scan's, and the head's finishing
/// guarantees finite logits for the models this backend
/// serves (the softcap bounds them outright).
const ARGMAX_SHADER: &str = r#"
struct Params {
    n: u32,
}

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> result: array<u32>;

var<workgroup> red_v: array<f32, 256>;
var<workgroup> red_i: array<u32, 256>;

// The greedy combine: (ov, oi) replaces (v, i) when it is
// strictly greater, or equal with a lower index.
fn better(v: f32, i: u32, ov: f32, oi: u32) -> bool {
    return ov > v || (ov == v && oi < i);
}

// Reduce this workgroup's per-thread pairs (already in the
// reduction arrays) to lane 0's pair.
fn reduce(lane: u32) {
    var stride = 128u;
    while (stride > 0u) {
        if (lane < stride) {
            if (better(red_v[lane], red_i[lane], red_v[lane + stride], red_i[lane + stride])) {
                red_v[lane] = red_v[lane + stride];
                red_i[lane] = red_i[lane + stride];
            }
        }
        workgroupBarrier();
        stride >>= 1u;
    }
}

@compute @workgroup_size(256)
fn argmax_stage_main(@builtin(workgroup_id) wid: vec3<u32>,
                     @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let tid = wid.x * 256u + lane;
    var bv = bitcast<f32>(0xff800000u);
    var bi = 0xffffffffu;
    for (var i = tid; i < params.n; i += 32768u) {
        let v = logits[i];
        if (better(bv, bi, v, i)) {
            bv = v;
            bi = i;
        }
    }
    red_v[lane] = bv;
    red_i[lane] = bi;
    workgroupBarrier();
    reduce(lane);
    if (lane == 0u) {
        partials[wid.x * 2u] = bitcast<u32>(red_v[0]);
        partials[wid.x * 2u + 1u] = red_i[0];
    }
}

@compute @workgroup_size(256)
fn argmax_finish_main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    var bv = bitcast<f32>(0xff800000u);
    var bi = 0xffffffffu;
    if (lane < 128u) {
        bv = bitcast<f32>(partials[lane * 2u]);
        bi = partials[lane * 2u + 1u];
    }
    red_v[lane] = bv;
    red_i[lane] = bi;
    workgroupBarrier();
    reduce(lane);
    if (lane == 0u) {
        // The result buffer is the caller's: word 0 the
        // winning index, word 1 the winning logit's bits.
        result[0] = red_i[0];
        result[1] = bitcast<u32>(red_v[0]);
    }
}
"#;

/// The F32 gather shader: one thread per output channel.
const GATHER_F32_SHADER: &str = r#"
struct GatherParams {
    width: u32,
    row_start: u32,
}

@group(0) @binding(0) var<storage, read> table: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: GatherParams;

@compute @workgroup_size(256)
fn gather_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c < params.width) {
        out[c] = table[params.row_start + c];
    }
}
"#;

/// The two-byte gather shader (F16 and BF16 element conversion).
const GATHER_HALF_SHADER: &str = r#"
struct GatherParams {
    width: u32,
    row_start: u32,
    dtype: u32,
}

@group(0) @binding(0) var<storage, read> table: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: GatherParams;

@compute @workgroup_size(256)
fn gather_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= params.width) {
        return;
    }
    let g = params.row_start + c;
    let bits = (table[g / 2u] >> (16u * (g % 2u))) & 0xFFFFu;
    if (params.dtype == 0u) {
        // F16: widen through the packed unpack intrinsic (the
        // second lane is padding).
        out[c] = unpack2x16float(bits).x;
    } else {
        // BF16: the value is the top half of an f32.
        out[c] = bitcast<f32>(bits << 16u);
    }
}
"#;

/// The quantized gather shader: one thread per output channel,
/// dequantizing its own element from the block bytes. Block
/// layouts and decoders restate `gguf`'s reference
/// implementations; dtype codes: 0 = Q4_K, 1 = Q6_K, 2 = Q5_K,
/// 3 = Q8_0. Composed from the params/bindings head, the shared
/// dequantizer helpers ([`QUANT_DEQUANT_HELPERS`], which the
/// multi-row gather below also uses), and the single-row main.
fn gather_quant_shader() -> String {
    format!("{GATHER_QUANT_HEAD}{QUANT_DEQUANT_HELPERS}{GATHER_QUANT_MAIN}")
}

/// The quantized multi-row gather shader, composed the same
/// way from its own head and main around the shared helpers.
fn gather_rows_quant_shader() -> String {
    format!("{GATHER_ROWS_QUANT_HEAD}{QUANT_DEQUANT_HELPERS}{GATHER_ROWS_QUANT_MAIN}")
}

/// The single-row gather's params and bindings head.
const GATHER_QUANT_HEAD: &str = r#"
struct GatherParams {
    width: u32,
    row_start: u32,
    dtype: u32,
}

@group(0) @binding(0) var<storage, read> table: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: GatherParams;
"#;

/// The dequantizer helpers shared by both quantized gathers:
/// byte access over the u32 table view and one dequantizer per
/// dtype code (0 = Q4_K, 1 = Q6_K, 2 = Q5_K, 3 = Q8_0), each
/// restating `gguf`'s reference decoder instruction for
/// instruction.
const QUANT_DEQUANT_HELPERS: &str = r#"
// One byte of the table. The u32 view's word and lane are
// host-endian: the host uploads little-endian bytes and reads
// them back on the same machine class (see the K1 shader's note
// on the same pattern).
fn byte_at(i: u32) -> u32 {
    return (table[i / 4u] >> (8u * (i % 4u))) & 0xFFu;
}

fn f16_at(i: u32) -> f32 {
    let bits = byte_at(i) | (byte_at(i + 1u) << 8u);
    return unpack2x16float(bits).x;
}

// K4_K scale/min unpacking, gguf's get_scale_min_k4 restated.
// `b` is the block's byte base; scales start at b + 4.
fn k4_scale(b: u32, j: u32) -> vec2<u32> {
    if (j < 4u) {
        return vec2<u32>(byte_at(b + 4u + j) & 63u, byte_at(b + 4u + j + 4u) & 63u);
    }
    let sc = (byte_at(b + 4u + j + 4u) & 0xFu) | ((byte_at(b + 4u + j - 4u) >> 6u) << 4u);
    let mn = (byte_at(b + 4u + j + 4u) >> 4u) | ((byte_at(b + 4u + j) >> 6u) << 4u);
    return vec2<u32>(sc, mn);
}

fn dequant_q4k(g: u32) -> f32 {
    let b = (g / 256u) * 144u;
    let within = g % 256u;
    let d = f16_at(b);
    let dmin = f16_at(b + 2u);
    let group = within / 64u;
    let lane = within % 64u;
    let sm = k4_scale(b, 2u * group + lane / 32u);
    let packed = byte_at(b + 16u + group * 32u + lane % 32u);
    var q = packed & 0xFu;
    if (lane >= 32u) {
        q = packed >> 4u;
    }
    return d * f32(sm.x) * f32(q) - dmin * f32(sm.y);
}

fn dequant_q5k(g: u32) -> f32 {
    let b = (g / 256u) * 176u;
    let within = g % 256u;
    let d = f16_at(b);
    let dmin = f16_at(b + 2u);
    let group = within / 64u;
    let lane = within % 64u;
    let sub = lane / 32u;
    let sm = k4_scale(b, 2u * group + sub);
    let packed = byte_at(b + 48u + group * 32u + lane % 32u);
    var q = packed & 0xFu;
    if (sub == 1u) {
        q = packed >> 4u;
    }
    // The fifth bits share one 32-byte array: element lane l of
    // group g reads bit 2g (first sub-block) or 2g+1 (second) of
    // qh[l].
    let qh = byte_at(b + 16u + lane % 32u);
    if ((qh >> (2u * group + sub)) & 1u) == 1u {
        q += 16u;
    }
    return d * f32(sm.x) * f32(q) - dmin * f32(sm.y);
}

fn dequant_q6k(g: u32) -> f32 {
    let b = (g / 256u) * 210u;
    let within = g % 256u;
    let half = within / 128u;
    let l = within % 128u;
    // The reference's four 32-wide streams within a half: stream
    // n reads byte (n % 2) * 32 + e of the half's 64 ql bytes,
    // the low nibble for n < 2 and the high nibble above, and
    // bits 2n, 2n+1 of the qh byte at index e.
    let n = l / 32u;
    let e = l % 32u;
    let packed = byte_at(b + half * 64u + (n % 2u) * 32u + e);
    var q = packed & 0xFu;
    if (n >= 2u) {
        q = packed >> 4u;
    }
    let qbits = byte_at(b + 128u + half * 32u + e);
    q = q | (((qbits >> (2u * n)) & 1u) << 4u) | (((qbits >> (2u * n + 1u)) & 1u) << 5u);
    var sc = i32(byte_at(b + 192u + half * 8u + l / 16u));
    if (sc >= 128) {
        sc -= 256;
    }
    let d = f16_at(b + 208u);
    return d * f32(sc) * (f32(q) - 32.0);
}

fn dequant_q8_0(g: u32) -> f32 {
    let b = (g / 32u) * 34u;
    let within = g % 32u;
    let d = f16_at(b);
    var q = i32(byte_at(b + 2u + within));
    if (q >= 128) {
        q -= 256;
    }
    return d * f32(q);
}
"#;

/// The single-row gather's main: one thread per output channel
/// of the one row `row_start` selects.
const GATHER_QUANT_MAIN: &str = r#"
@compute @workgroup_size(256)
fn gather_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= params.width) {
        return;
    }
    let g = params.row_start + c;
    if (params.dtype == 0u) {
        out[c] = dequant_q4k(g);
    } else if (params.dtype == 1u) {
        out[c] = dequant_q6k(g);
    } else if (params.dtype == 2u) {
        out[c] = dequant_q5k(g);
    } else {
        out[c] = dequant_q8_0(g);
    }
}
"#;

/// The multi-row gathers for the device-resident forward: one
/// dispatch gathers a whole chunk of token rows, the token ids
/// read from a device buffer (the only per-forward upload the
/// resident path makes, a few bytes per token). One thread per
/// output element of the `[rows][width]` block; the row's token
/// comes from `tokens[row]`. Three storage families, mirroring
/// the single-row gathers above.
const GATHER_ROWS_F32_SHADER: &str = r#"
struct GatherRowsParams {
    width: u32,
    rows: u32,
    grid_x: u32,
}

@group(0) @binding(0) var<storage, read> table: array<f32>;
@group(0) @binding(1) var<storage, read> tokens: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: GatherRowsParams;

@compute @workgroup_size(256)
fn gather_rows_main(@builtin(workgroup_id) wid: vec3<u32>,
                    @builtin(local_invocation_id) lid: vec3<u32>) {
    let flat = (wid.y * params.grid_x + wid.x) * 256u + lid.x;
    if (flat >= params.rows * params.width) {
        return;
    }
    let row = flat / params.width;
    out[flat] = table[tokens[row] * params.width + flat % params.width];
}
"#;

/// The two-byte (F16 / BF16) multi-row gather; dtype codes as
/// in the single-row shader (0 = F16, 1 = BF16).
const GATHER_ROWS_HALF_SHADER: &str = r#"
struct GatherRowsParams {
    width: u32,
    rows: u32,
    dtype: u32,
    grid_x: u32,
}

@group(0) @binding(0) var<storage, read> table: array<u32>;
@group(0) @binding(1) var<storage, read> tokens: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: GatherRowsParams;

@compute @workgroup_size(256)
fn gather_rows_main(@builtin(workgroup_id) wid: vec3<u32>,
                    @builtin(local_invocation_id) lid: vec3<u32>) {
    let flat = (wid.y * params.grid_x + wid.x) * 256u + lid.x;
    if (flat >= params.rows * params.width) {
        return;
    }
    let row = flat / params.width;
    let g = tokens[row] * params.width + flat % params.width;
    let bits = (table[g / 2u] >> (16u * (g % 2u))) & 0xFFFFu;
    if (params.dtype == 0u) {
        out[flat] = unpack2x16float(bits).x;
    } else {
        out[flat] = bitcast<f32>(bits << 16u);
    }
}
"#;

/// The quantized multi-row gather's params and bindings head;
/// the shader is this head plus [`QUANT_DEQUANT_HELPERS`] plus
/// [`GATHER_ROWS_QUANT_MAIN`]. Dtype codes as in the single-row
/// shader (0 = Q4_K, 1 = Q6_K, 2 = Q5_K, 3 = Q8_0).
const GATHER_ROWS_QUANT_HEAD: &str = r#"
struct GatherRowsParams {
    width: u32,
    rows: u32,
    dtype: u32,
    grid_x: u32,
}

@group(0) @binding(0) var<storage, read> table: array<u32>;
@group(0) @binding(1) var<storage, read> tokens: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: GatherRowsParams;
"#;

/// The quantized multi-row gather's main.
const GATHER_ROWS_QUANT_MAIN: &str = r#"
@compute @workgroup_size(256)
fn gather_rows_main(@builtin(workgroup_id) wid: vec3<u32>,
                    @builtin(local_invocation_id) lid: vec3<u32>) {
    let flat = (wid.y * params.grid_x + wid.x) * 256u + lid.x;
    if (flat >= params.rows * params.width) {
        return;
    }
    let row = flat / params.width;
    let g = tokens[row] * params.width + flat % params.width;
    if (params.dtype == 0u) {
        out[flat] = dequant_q4k(g);
    } else if (params.dtype == 1u) {
        out[flat] = dequant_q6k(g);
    } else if (params.dtype == 2u) {
        out[flat] = dequant_q5k(g);
    } else {
        out[flat] = dequant_q8_0(g);
    }
}
"#;

/// Largest row width the fused gather+norm serves: 256
/// lanes holding at most 32 strided elements each (the
/// register-residency bound, the same shape as K4's chain).
pub const GATHER_NORM_MAX_WIDTH: usize = 256 * 32;

/// The fused gather+norm shaders: the multi-row gather and
/// the weightless embedding norm in ONE dispatch (the
/// resident forward chains them with nothing between). One
/// workgroup per row; each lane dequantizes its strided
/// share of the gathered row into registers, the row's
/// sum of squares reduces through K4's partials tree, and
/// the normed row is written from the registers: precisely
/// the two-dispatch composition's values (the norm's
/// reduction distribution is K4's, and the gather's element
/// decoders are the gather shaders' own). Three storage
/// families, mirroring the gathers.
const GATHER_NORM_HEAD: &str = r#"
struct GatherNormParams {
    width: u32,
    rows: u32,
    dtype: u32,
    grid_x: u32,
    eps: f32,
}

@group(0) @binding(1) var<storage, read> tokens: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: GatherNormParams;

var<workgroup> partials: array<f32, 256>;
var<workgroup> inv_shared: f32;
"#;

/// The shared gather+norm main, parameterized by the
/// family's element load (an expression over `g`, the
/// element's global index in the table).
fn gather_norm_main(load: &str) -> String {
    format!(
        r#"
@compute @workgroup_size(256)
fn gather_norm_main(@builtin(workgroup_id) wid: vec3<u32>,
                    @builtin(local_invocation_id) lid: vec3<u32>) {{
    let row = wid.y * params.grid_x + wid.x;
    // Whole-workgroup early out: every lane shares `row`.
    if (row >= params.rows) {{
        return;
    }}
    let lane = lid.x;
    let base = tokens[row] * params.width;
    var vals: array<f32, 32>;
    var ss = 0.0;
    var n = 0u;
    for (var i = lane; i < params.width; i += 256u) {{
        let g = base + i;
        let x = {load};
        vals[n] = x;
        ss += x * x;
        n += 1u;
    }}
    partials[lane] = ss;
    workgroupBarrier();
    var stride = 128u;
    while (stride > 0u) {{
        if (lane < stride) {{
            partials[lane] += partials[lane + stride];
        }}
        workgroupBarrier();
        stride >>= 1u;
    }}
    if (lane == 0u) {{
        inv_shared = inverseSqrt(partials[0] / f32(params.width) + params.eps);
    }}
    workgroupBarrier();
    let inv = inv_shared;
    var m = 0u;
    for (var i = lane; i < params.width; i += 256u) {{
        out[row * params.width + i] = vals[m] * inv;
        m += 1u;
    }}
}}
"#
    )
}

/// The F32 fused gather+norm shader.
fn gather_norm_f32_shader() -> String {
    format!(
        "@group(0) @binding(0) var<storage, read> table: array<f32>;\n{GATHER_NORM_HEAD}{}",
        gather_norm_main("table[g]")
    )
}

/// The two-byte (F16 / BF16) fused gather+norm shader;
/// dtype codes as in the gathers (0 = F16, 1 = BF16).
fn gather_norm_half_shader() -> String {
    format!(
        "@group(0) @binding(0) var<storage, read> table: array<u32>;\n{GATHER_NORM_HEAD}\n\
         fn half_at(g: u32) -> f32 {{\n    \
             let bits = (table[g / 2u] >> (16u * (g % 2u))) & 0xFFFFu;\n    \
             if (params.dtype == 0u) {{ return unpack2x16float(bits).x; }}\n    \
             return bitcast<f32>(bits << 16u);\n}}\n{}",
        gather_norm_main("half_at(g)")
    )
}

/// The quantized fused gather+norm shader: the shared head,
/// the gather's dequantizer helpers, and a dtype dispatch
/// over them.
fn gather_norm_quant_shader() -> String {
    format!(
        "@group(0) @binding(0) var<storage, read> table: array<u32>;\n{GATHER_NORM_HEAD}{QUANT_DEQUANT_HELPERS}\n\
         fn dequant_any(g: u32) -> f32 {{\n    \
             if (params.dtype == 0u) {{ return dequant_q4k(g); }}\n    \
             if (params.dtype == 1u) {{ return dequant_q6k(g); }}\n    \
             if (params.dtype == 2u) {{ return dequant_q5k(g); }}\n    \
             return dequant_q8_0(g);\n}}\n{}",
        gather_norm_main("dequant_any(g)")
    )
}

/// The head finishing shader: scale, then tanh softcap.
const FINISH_SHADER: &str = r#"
struct FinishParams {
    n: u32,
    scale: f32,
    cap: f32,
    grid_x: u32,
}

@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<uniform> params: FinishParams;

@compute @workgroup_size(256)
fn finish_main(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = (wid.y * params.grid_x + wid.x) * 256u + lid.x;
    if (i < params.n) {
        let x = logits[i] * params.scale;
        logits[i] = params.cap * tanh(x / params.cap);
    }
}
"#;

/// The scale-only finishing shader: `muse_glimmer::softcap_scale`
/// with no positive cap multiplies by the scale and stops, so
/// the resident path needs the same half of the finishing
/// without the tanh (the finish kernel above refuses a
/// non-positive cap by design, keeping its one job). Same
/// params layout as the finish shader, `cap` unused.
const SCALE_SHADER: &str = r#"
struct FinishParams {
    n: u32,
    scale: f32,
    cap: f32,
    grid_x: u32,
}

@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<uniform> params: FinishParams;

@compute @workgroup_size(256)
fn scale_main(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = (wid.y * params.grid_x + wid.x) * 256u + lid.x;
    if (i < params.n) {
        logits[i] = logits[i] * params.scale;
    }
}
"#;

/// Dtype codes for the quantized gather's uniform.
const CODE_Q4_K: u32 = 0;
const CODE_Q6_K: u32 = 1;
const CODE_Q5_K: u32 = 2;
const CODE_Q8_0: u32 = 3;

/// Dtype codes for the two-byte gather's uniform.
const CODE_F16: u32 = 0;
const CODE_BF16: u32 = 1;

/// One prepared multi-row gather dispatch: token ids from a
/// device buffer, rows written to a device output. Encodable
/// any number of times (the token buffer's contents are the
/// caller's to rewrite between forwards). See
/// [`K8Pipeline::prepare_gather_rows`].
#[derive(Debug)]
pub struct K8GatherRowsCall {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    grid: (u32, u32),
}

impl K8GatherRowsCall {
    /// Record this call's dispatch into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k8 embedding gather rows"),
            timestamp_writes: None,
        });
        self.encode_into(&mut pass);
    }

    /// Record this dispatch into an already-open compute
    /// `pass`: the composition form the device-resident
    /// session uses to chain a whole forward into one pass
    /// per segment (wgpu inserts the hazard barriers
    /// between dispatches inside the pass).
    pub fn encode_into(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.grid.0, self.grid.1, 1);
    }
}

/// One prepared head-finishing dispatch (scale-then-softcap,
/// or scale-only through the sibling pipeline): in place over
/// a logits buffer, encodable any number of times. See
/// [`K8Pipeline::prepare_finish`] and
/// [`K8Pipeline::prepare_scale`].
#[derive(Debug)]
pub struct K8FinishCall {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    grid: (u32, u32),
}

impl K8FinishCall {
    /// Record this call's dispatch into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k8 head finish"),
            timestamp_writes: None,
        });
        self.encode_into(&mut pass);
    }

    /// Record this dispatch into an already-open compute
    /// `pass`: the composition form the device-resident
    /// session uses to chain a whole forward into one pass
    /// per segment (wgpu inserts the hazard barriers
    /// between dispatches inside the pass).
    pub fn encode_into(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.grid.0, self.grid.1, 1);
    }
}

/// One prepared argmax reduction over a device logits
/// buffer: the stage dispatch (one workgroup per
/// [`ARGMAX_GROUPS`] slice, writing (value, index) pairs to
/// the caller's partials buffer) and the finish dispatch
/// (one workgroup reducing the pairs into the caller's
/// two-word result buffer: winning index, then the winning
/// logit's bits). Encodable any number of times against the
/// same buffers; the logits buffer's contents are the
/// caller's to rewrite between forwards. See
/// [`K8Pipeline::prepare_argmax`].
#[derive(Debug)]
pub struct K8ArgmaxCall {
    stage_pipeline: wgpu::ComputePipeline,
    stage_bind_group: wgpu::BindGroup,
    finish_pipeline: wgpu::ComputePipeline,
    finish_bind_group: wgpu::BindGroup,
}

impl K8ArgmaxCall {
    /// Record both dispatches into `encoder` (the stage, then
    /// the finish; wgpu's hazard tracking orders the second
    /// after the first's writes).
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k8 argmax"),
            timestamp_writes: None,
        });
        self.encode_into(&mut pass);
    }

    /// Record both dispatches into an already-open compute
    /// `pass`: the composition form the device-resident
    /// session uses to append the reduction to the head
    /// segment of a forward chunk.
    pub fn encode_into(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_pipeline(&self.stage_pipeline);
        pass.set_bind_group(0, &self.stage_bind_group, &[]);
        pass.dispatch_workgroups(ARGMAX_GROUPS as u32, 1, 1);
        pass.set_pipeline(&self.finish_pipeline);
        pass.set_bind_group(0, &self.finish_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
}

/// The compiled K8 pipelines (F32 gather, two-byte gather,
/// quantized gather, their multi-row forms, head finish, and
/// the scale-only finish).
#[derive(Debug)]
pub struct K8Pipeline {
    gather_f32: wgpu::ComputePipeline,
    gather_half: wgpu::ComputePipeline,
    gather_quant: wgpu::ComputePipeline,
    gather_rows_f32: wgpu::ComputePipeline,
    gather_rows_half: wgpu::ComputePipeline,
    gather_rows_quant: wgpu::ComputePipeline,
    gather_norm_f32: wgpu::ComputePipeline,
    gather_norm_half: wgpu::ComputePipeline,
    gather_norm_quant: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
    scale: wgpu::ComputePipeline,
    argmax_stage: wgpu::ComputePipeline,
    argmax_finish: wgpu::ComputePipeline,
}

impl K8Pipeline {
    /// Compile all K8 pipelines on `device`.
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] when wgpu rejects a shader.
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        let build = |source: &str, entry: &'static str, label: &'static str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let gather_f32 = build(GATHER_F32_SHADER, "gather_main", "k8 gather f32");
        let gather_half = build(GATHER_HALF_SHADER, "gather_main", "k8 gather half");
        let gather_quant = build(&gather_quant_shader(), "gather_main", "k8 gather quant");
        let gather_rows_f32 = build(
            GATHER_ROWS_F32_SHADER,
            "gather_rows_main",
            "k8 gather rows f32",
        );
        let gather_rows_half = build(
            GATHER_ROWS_HALF_SHADER,
            "gather_rows_main",
            "k8 gather rows half",
        );
        let gather_rows_quant = build(
            &gather_rows_quant_shader(),
            "gather_rows_main",
            "k8 gather rows quant",
        );
        let gather_norm_f32 = build(
            &gather_norm_f32_shader(),
            "gather_norm_main",
            "k8 gather norm f32",
        );
        let gather_norm_half = build(
            &gather_norm_half_shader(),
            "gather_norm_main",
            "k8 gather norm half",
        );
        let gather_norm_quant = build(
            &gather_norm_quant_shader(),
            "gather_norm_main",
            "k8 gather norm quant",
        );
        let finish = build(FINISH_SHADER, "finish_main", "k8 head finish");
        let scale = build(SCALE_SHADER, "scale_main", "k8 head scale");
        let argmax_stage = build(ARGMAX_SHADER, "argmax_stage_main", "k8 argmax stage");
        let argmax_finish = build(ARGMAX_SHADER, "argmax_finish_main", "k8 argmax finish");
        dbg_log!("k8 pipelines built");
        Ok(K8Pipeline {
            gather_f32,
            gather_half,
            gather_quant,
            gather_rows_f32,
            gather_rows_half,
            gather_rows_quant,
            gather_norm_f32,
            gather_norm_half,
            gather_norm_quant,
            finish,
            scale,
            argmax_stage,
            argmax_finish,
        })
    }

    /// Prepare a multi-row embedding gather over device
    /// buffers: `rows` token ids read from `tokens` (rewritable
    /// between forwards), rows of the resident `table` written
    /// to `out` as `[rows][width]`. The table's dtype selects
    /// the storage family, accepting exactly the set
    /// [`K8Pipeline::embed`] accepts and refusing the rest by
    /// name.
    ///
    /// # Errors
    ///
    /// [`GlimmerError::StepFailed`] under `"embed"` on a dtype
    /// with no gather family or a zero geometry.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_gather_rows(
        &self,
        ctx: &GpuContext,
        table: &wgpu::Buffer,
        tokens: &wgpu::Buffer,
        out: &wgpu::Buffer,
        rows: usize,
        width: usize,
        dtype: Dtype,
    ) -> GlimmerResult<K8GatherRowsCall> {
        let step = "embed";
        if rows == 0 || width == 0 {
            return Err(step_failed(
                step,
                format!("gather of {rows} rows of width {width}: nothing to gather"),
            ));
        }
        let (pipeline, code): (&wgpu::ComputePipeline, Option<u32>) = match dtype {
            Dtype::F32 => (&self.gather_rows_f32, None),
            Dtype::F16 => (&self.gather_rows_half, Some(CODE_F16)),
            Dtype::Bf16 => (&self.gather_rows_half, Some(CODE_BF16)),
            Dtype::Q4K => (&self.gather_rows_quant, Some(CODE_Q4_K)),
            Dtype::Q6K => (&self.gather_rows_quant, Some(CODE_Q6_K)),
            Dtype::Q5K => (&self.gather_rows_quant, Some(CODE_Q5_K)),
            Dtype::Q8_0 => (&self.gather_rows_quant, Some(CODE_Q8_0)),
            other => {
                return Err(step_failed(
                    step,
                    format!(
                        "embedding dtype {} (ggml type id {}) has no gather family in this backend",
                        gguf::ggml::type_name(dtype_id(other)),
                        dtype_id(other)
                    ),
                ))
            }
        };
        let total_wg = ((rows * width) as u64).div_ceil(u64::from(WORKGROUP));
        let grid = grid_dims(total_wg);
        let param_bytes: Vec<u8> = match code {
            Some(code) => [width as u32, rows as u32, code, grid.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            None => [width as u32, rows as u32, grid.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        };
        let params_buf = ctx.uniform_buffer("k8 gather rows params", &param_bytes);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k8 gather rows bind group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: table.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tokens.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        dbg_log!("k8 gather rows prepared: rows={rows} width={width} dtype={dtype:?}");
        Ok(K8GatherRowsCall {
            pipeline: pipeline.clone(),
            bind_group,
            grid,
        })
    }

    /// Prepare a fused gather+norm dispatch over device
    /// buffers: `rows` token ids read from `tokens`, each
    /// gathered row of the resident `table` RMS-normalized
    /// (weightless, at `eps`) as it is gathered, written to
    /// `out` as `[rows][width]`. The fused form of
    /// [`K8Pipeline::prepare_gather_rows`] followed by the
    /// embedding norm; the dtype acceptance matches the
    /// gather's.
    ///
    /// # Errors
    ///
    /// [`GlimmerError::StepFailed`] under `"embed"` on a dtype
    /// with no gather family, a zero geometry, or a width
    /// past [`GATHER_NORM_MAX_WIDTH`] (the register-residency
    /// bound; wider rows keep the two dispatches).
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_gather_rows_norm(
        &self,
        ctx: &GpuContext,
        table: &wgpu::Buffer,
        tokens: &wgpu::Buffer,
        out: &wgpu::Buffer,
        rows: usize,
        width: usize,
        eps: f32,
        dtype: Dtype,
    ) -> GlimmerResult<K8GatherRowsCall> {
        let step = "embed";
        if rows == 0 || width == 0 || width > GATHER_NORM_MAX_WIDTH {
            return Err(step_failed(
                step,
                format!(
                    "gather+norm of {rows} rows of width {width}: nothing to gather, or the \
                     width is past the fused kernel's {GATHER_NORM_MAX_WIDTH}-element bound"
                ),
            ));
        }
        let (pipeline, code): (&wgpu::ComputePipeline, u32) = match dtype {
            Dtype::F32 => (&self.gather_norm_f32, 0),
            Dtype::F16 => (&self.gather_norm_half, CODE_F16),
            Dtype::Bf16 => (&self.gather_norm_half, CODE_BF16),
            Dtype::Q4K => (&self.gather_norm_quant, CODE_Q4_K),
            Dtype::Q6K => (&self.gather_norm_quant, CODE_Q6_K),
            Dtype::Q5K => (&self.gather_norm_quant, CODE_Q5_K),
            Dtype::Q8_0 => (&self.gather_norm_quant, CODE_Q8_0),
            other => {
                return Err(step_failed(
                    step,
                    format!(
                        "embedding dtype {} (ggml type id {}) has no gather family in this backend",
                        gguf::ggml::type_name(dtype_id(other)),
                        dtype_id(other)
                    ),
                ))
            }
        };
        let grid = grid_dims(rows as u64);
        let params: [u32; 5] = [width as u32, rows as u32, code, grid.0, eps.to_bits()];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buf = ctx.uniform_buffer("k8 gather norm params", &param_bytes);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k8 gather norm bind group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: table.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tokens.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        dbg_log!("k8 gather+norm prepared: rows={rows} width={width} dtype={dtype:?}");
        Ok(K8GatherRowsCall {
            pipeline: pipeline.clone(),
            bind_group,
            grid,
        })
    }

    /// Prepare a head-finishing dispatch over a device logits
    /// buffer: scale, then tanh softcap, in place. The resident
    /// form of [`K8Pipeline::finish`].
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when `cap` is not positive
    /// and finite (the no-cap case is [`K8Pipeline::prepare_scale`]).
    pub fn prepare_finish(
        &self,
        ctx: &GpuContext,
        logits: &wgpu::Buffer,
        n: usize,
        scale: f32,
        cap: f32,
    ) -> Result<K8FinishCall> {
        if !cap.is_finite() || cap <= 0.0 {
            return Err(WgpuError::InvalidTensor {
                name: "k8 finish".to_string(),
                detail: format!("softcap {cap} is not positive and finite"),
            });
        }
        Ok(self.prepare_finish_impl(ctx, &self.finish, logits, n, scale, cap))
    }

    /// Prepare a scale-only finishing dispatch over a device
    /// logits buffer, for a model whose softcap is not positive
    /// (`muse_glimmer::softcap_scale` applies the scale and
    /// stops there in that case).
    pub fn prepare_scale(
        &self,
        ctx: &GpuContext,
        logits: &wgpu::Buffer,
        n: usize,
        scale: f32,
    ) -> K8FinishCall {
        self.prepare_finish_impl(ctx, &self.scale, logits, n, scale, 0.0)
    }

    /// Shared body of the two finishing prepares: the pipelines
    /// share the params layout and the bind group shape.
    fn prepare_finish_impl(
        &self,
        ctx: &GpuContext,
        pipeline: &wgpu::ComputePipeline,
        logits: &wgpu::Buffer,
        n: usize,
        scale: f32,
        cap: f32,
    ) -> K8FinishCall {
        let total_wg = (n as u64).div_ceil(u64::from(WORKGROUP));
        let grid = grid_dims(total_wg);
        let params: [u32; 4] = [n as u32, scale.to_bits(), cap.to_bits(), grid.0];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buf = ctx.uniform_buffer("k8 finish params", &param_bytes);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k8 finish bind group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: logits.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        K8FinishCall {
            pipeline: pipeline.clone(),
            bind_group,
            grid,
        }
    }

    /// Prepare an on-device argmax over a logits buffer of
    /// `n` elements: the two-stage reduction whose winner is
    /// the host sampler's greedy pick (maximum value, lowest
    /// index on an exact tie; see the shader's docs).
    /// `partials` must hold at least [`ARGMAX_GROUPS`] (value
    /// bits, index) u32 pairs and `result` at least two u32
    /// words; both are caller-owned scratch the call writes
    /// and the caller reads after submission.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when `n` is zero or a
    /// scratch buffer is undersized.
    pub fn prepare_argmax(
        &self,
        ctx: &GpuContext,
        logits: &wgpu::Buffer,
        n: usize,
        partials: &wgpu::Buffer,
        result: &wgpu::Buffer,
    ) -> Result<K8ArgmaxCall> {
        if n == 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k8 argmax".to_string(),
                detail: "argmax over zero logits".to_string(),
            });
        }
        if partials.size() < (ARGMAX_GROUPS * 2 * 4) as u64 || result.size() < 8 {
            return Err(WgpuError::InvalidTensor {
                name: "k8 argmax".to_string(),
                detail: format!(
                    "partials ({} bytes) or result ({} bytes) undersized for the reduction \
                     ({} pairs and 2 words needed)",
                    partials.size(),
                    result.size(),
                    ARGMAX_GROUPS
                ),
            });
        }
        let params: [u32; 4] = [n as u32, 0, 0, 0];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buf = ctx.uniform_buffer("k8 argmax params", &param_bytes);
        let stage_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k8 argmax stage bind group"),
            layout: &self.argmax_stage.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: logits.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: partials.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let finish_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k8 argmax finish bind group"),
            layout: &self.argmax_finish.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: partials.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: result.as_entire_binding(),
                },
            ],
        });
        dbg_log!("k8 argmax prepared: n={n}");
        Ok(K8ArgmaxCall {
            stage_pipeline: self.argmax_stage.clone(),
            stage_bind_group,
            finish_pipeline: self.argmax_finish.clone(),
            finish_bind_group,
        })
    }

    /// One-shot argmax over host logits: upload, reduce on
    /// the device, read back the eight result bytes. Returns
    /// the winning index and the winning logit. The check and
    /// bench data path; the resident session prepares its
    /// reduction against its own buffers instead.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] on empty input;
    /// [`WgpuError::Readback`] when the result cannot be
    /// mapped back.
    pub fn argmax(&self, ctx: &GpuContext, logits: &[f32]) -> Result<(u32, f32)> {
        if logits.is_empty() {
            return Err(WgpuError::InvalidTensor {
                name: "k8 argmax".to_string(),
                detail: "argmax over zero logits".to_string(),
            });
        }
        let logits_buf = ctx.upload_f32("k8 argmax logits", logits);
        let partials = ctx.scratch_buffer("k8 argmax partials", (ARGMAX_GROUPS * 2 * 4) as u64);
        let result = ctx.scratch_buffer("k8 argmax result", 8);
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k8 argmax staging"),
            size: 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let call = self.prepare_argmax(ctx, &logits_buf, logits.len(), &partials, &result)?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k8 argmax run"),
            });
        call.encode(&mut encoder);
        encoder.copy_buffer_to_buffer(&result, 0, &staging, 0, 8);
        ctx.queue.submit([encoder.finish()]);
        let words = ctx.map_staging(&staging, 2)?;
        Ok((words[0].to_bits(), words[1]))
    }

    /// Dispatch one gather pipeline for one token row and read
    /// the row back. `table` is the already-uploaded table
    /// buffer, `code` the family's dtype code (unused by the F32
    /// pipeline, whose uniform has no code field; callers pass
    /// the right pipeline for the family).
    fn gather_row(
        &self,
        ctx: &GpuContext,
        pipeline: &wgpu::ComputePipeline,
        table: &wgpu::Buffer,
        row_start: usize,
        width: usize,
        code: Option<u32>,
    ) -> Result<Vec<f32>> {
        let out_buf = ctx.scratch_buffer("k8 embedding row", (width * 4) as u64);
        let param_bytes: Vec<u8> = match code {
            Some(code) => [width as u32, row_start as u32, code]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            None => [width as u32, row_start as u32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        };
        let params_buf = ctx.uniform_buffer("k8 gather params", &param_bytes);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k8 gather bind group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: table.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let total_wg = (width as u64).div_ceil(u64::from(WORKGROUP));
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k8 gather run"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("k8 embedding gather"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(total_wg as u32, 1, 1);
        }
        ctx.queue.submit([encoder.finish()]);
        ctx.read_f32(&out_buf, width)
    }

    /// Gather the embedding rows for `tokens`, row-major
    /// `[tokens][hidden]`, validating exactly what `cpu-compute`
    /// validates (table shape against the config, byte length
    /// against the dtype's block geometry, token range) and
    /// accepting exactly its dtype set.
    ///
    /// # Errors
    ///
    /// [`GlimmerError::StepFailed`] under `"embed"` on any
    /// refusal; device failures are wrapped under the same step.
    pub fn embed(
        &self,
        ctx: &GpuContext,
        table: &TensorView,
        tokens: &[u32],
        cfg: &GlimmerConfig,
    ) -> GlimmerResult<Vec<f32>> {
        let step = "embed";
        let hidden = cfg.hidden as usize;
        let vocab = cfg.vocab_size as usize;
        if table.shape.len() != 2
            || table.shape[0] as usize != hidden
            || table.shape[1] as usize != vocab
        {
            return Err(step_failed(
                step,
                format!(
                    "embedding table {} has shape {:?}, expected [{}, {}] in GGUF stored order",
                    table.name, table.shape, hidden, cfg.vocab_size
                ),
            ));
        }
        let ty = match table.dtype {
            Dtype::F32 => gguf::GgmlType::F32,
            Dtype::F16 => gguf::GgmlType::F16,
            Dtype::Q8_0 => gguf::GgmlType::Q8_0,
            Dtype::Q4K => gguf::GgmlType::Q4K,
            Dtype::Q5K => gguf::GgmlType::Q5K,
            Dtype::Q6K => gguf::GgmlType::Q6K,
            Dtype::Bf16 => gguf::GgmlType::Bf16,
            other => {
                return Err(step_failed(
                    step,
                    format!(
                        "tensor {} has dtype {} (ggml type id {}), which the wgpu backend cannot dequantize for the embedding gather",
                        table.name,
                        gguf::ggml::type_name(dtype_id(other)),
                        dtype_id(other)
                    ),
                ))
            }
        };
        let elements = hidden * vocab;
        let expected = ty.byte_len(elements as u64).ok_or_else(|| {
            step_failed(
                step,
                format!(
                    "tensor {} has {elements} elements, not a whole number of {} blocks",
                    table.name,
                    gguf::ggml::type_name(ty.id())
                ),
            )
        })?;
        if expected != table.data.len() as u64 {
            return Err(step_failed(
                step,
                format!(
                    "tensor {} holds {} bytes but {elements} elements of {} need {expected}",
                    table.name,
                    table.data.len(),
                    gguf::ggml::type_name(ty.id())
                ),
            ));
        }
        for &token in tokens {
            if token as usize >= vocab {
                return Err(step_failed(
                    step,
                    format!(
                        "tensor {} has {vocab} rows, too few for the requested token {token}",
                        table.name
                    ),
                ));
            }
        }

        // One upload of the whole table, one dispatch per token
        // row. (The real table is 1.3 billion elements in Q4_K;
        // uploading it per embed call is the trait-seam cost this
        // landing accepts, the same one-shot shape as K1's run
        // before the weight cache. The gather kernels themselves
        // are what this module proves.)
        let as_err = |e: WgpuError| step_failed(step, e.to_string());
        let mut out = Vec::with_capacity(tokens.len() * hidden);
        match table.dtype {
            Dtype::F32 => {
                let values: Vec<f32> = table
                    .data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let buf = ctx.upload_f32("k8 embedding table", &values);
                for &token in tokens {
                    let row = self
                        .gather_row(
                            ctx,
                            &self.gather_f32,
                            &buf,
                            token as usize * hidden,
                            hidden,
                            None,
                        )
                        .map_err(as_err)?;
                    out.extend_from_slice(&row);
                }
            }
            Dtype::F16 | Dtype::Bf16 => {
                let code = if table.dtype == Dtype::F16 {
                    CODE_F16
                } else {
                    CODE_BF16
                };
                let buf = ctx.upload_bytes("k8 embedding table", &table.data);
                for &token in tokens {
                    let row = self
                        .gather_row(
                            ctx,
                            &self.gather_half,
                            &buf,
                            token as usize * hidden,
                            hidden,
                            Some(code),
                        )
                        .map_err(as_err)?;
                    out.extend_from_slice(&row);
                }
            }
            Dtype::Q4K | Dtype::Q5K | Dtype::Q6K | Dtype::Q8_0 => {
                let code = match table.dtype {
                    Dtype::Q4K => CODE_Q4_K,
                    Dtype::Q6K => CODE_Q6_K,
                    Dtype::Q5K => CODE_Q5_K,
                    _ => CODE_Q8_0,
                };
                let buf = ctx.upload_bytes("k8 embedding table", &table.data);
                for &token in tokens {
                    let row = self
                        .gather_row(
                            ctx,
                            &self.gather_quant,
                            &buf,
                            token as usize * hidden,
                            hidden,
                            Some(code),
                        )
                        .map_err(as_err)?;
                    out.extend_from_slice(&row);
                }
            }
            _ => unreachable!("the dtype was matched above"),
        }
        dbg_log!("k8 embed: {} tokens, hidden {hidden}", tokens.len());
        Ok(out)
    }

    /// The head finishing math over a logits buffer: upload,
    /// scale-then-softcap, read back. The GPU form of
    /// `muse_glimmer::softcap_scale`; see the module docs for
    /// why the trait does not call this today.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when `cap` is not positive
    /// and finite; [`WgpuError::Readback`] on map failure.
    pub fn finish(
        &self,
        ctx: &GpuContext,
        logits: &[f32],
        scale: f32,
        cap: f32,
    ) -> Result<Vec<f32>> {
        if !cap.is_finite() || cap <= 0.0 {
            return Err(WgpuError::InvalidTensor {
                name: "k8 finish".to_string(),
                detail: format!("softcap {cap} is not positive and finite"),
            });
        }
        if logits.is_empty() {
            return Ok(Vec::new());
        }
        let buf = ctx.upload_f32("k8 logits", logits);
        let call = self.prepare_finish(ctx, &buf, logits.len(), scale, cap)?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k8 finish run"),
            });
        call.encode(&mut encoder);
        ctx.queue.submit([encoder.finish()]);
        dbg_log!("k8 finish: n={} scale={scale} cap={cap}", logits.len());
        ctx.read_f32(&buf, logits.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaders_carry_their_entry_points() {
        assert!(GATHER_F32_SHADER.contains("fn gather_main"));
        assert!(GATHER_HALF_SHADER.contains("fn gather_main"));
        assert!(gather_quant_shader().contains("fn dequant_q4k"));
        assert!(gather_quant_shader().contains("fn dequant_q5k"));
        assert!(gather_quant_shader().contains("fn dequant_q6k"));
        assert!(gather_quant_shader().contains("fn dequant_q8_0"));
        assert!(FINISH_SHADER.contains("fn finish_main"));
        assert!(gather_norm_f32_shader().contains("fn gather_norm_main"));
        assert!(gather_norm_half_shader().contains("fn half_at"));
        assert!(gather_norm_quant_shader().contains("fn dequant_any"));
        // The finish order is pinned: scale first, then the cap.
        assert!(FINISH_SHADER.contains("logits[i] * params.scale"));
        assert!(FINISH_SHADER.contains("params.cap * tanh"));
        // The multi-row gathers read their token ids from a
        // device buffer and share the one dequantizer source.
        assert!(GATHER_ROWS_F32_SHADER.contains("fn gather_rows_main"));
        assert!(GATHER_ROWS_F32_SHADER.contains("tokens[row]"));
        assert!(GATHER_ROWS_HALF_SHADER.contains("fn gather_rows_main"));
        assert!(gather_rows_quant_shader().contains("fn gather_rows_main"));
        assert!(gather_rows_quant_shader().contains("fn dequant_q4k"));
        assert!(gather_rows_quant_shader().contains("tokens[row]"));
        assert!(SCALE_SHADER.contains("fn scale_main"));
        assert!(!SCALE_SHADER.contains("tanh"));
    }
}
