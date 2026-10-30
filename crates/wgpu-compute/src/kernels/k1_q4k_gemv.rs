//! K1: Q4_K dequantize-and-multiply GEMV (BACKEND.md Section 3).
//!
//! The bulk of the decode stream: 77.3 percent of Glimmer's bytes
//! per token are Q4_K (FFN gate and up, attention Q and K
//! projections, the embedding table's rows through `proj`-shaped
//! calls). This kernel's GB/s is effectively the backend's GB/s.
//!
//! The shader reads the ggml block layout directly from the
//! tensor's raw bytes (144 bytes per 256-element super-block: f16
//! `d`, f16 `dmin`, 12 packed scale/min bytes, 128 nibble bytes),
//! expanding values in registers and never materializing a
//! dequantized weight anywhere, per BACKEND.md Section 4. The
//! dequantization arithmetic mirrors `gguf::dequant::dequant_q4_k`
//! term for term, including `get_scale_min_k4`'s 6-bit unpacking, so
//! GPU results can be checked against the CPU reference within
//! float reassociation noise.
//!
//! Two parallel shapes live here behind [`K1Shape`]:
//!
//! - The legacy shape (the first landing's v2): one workgroup of
//!   32 lanes per output row, one lane per super-block, everything
//!   as `vec4`. It measured 216.5 GB/s on the server (BACKEND.md's
//!   results section), against a 746.9 GB/s stream probe through
//!   the same stack: correct, and the baseline the tuning pass
//!   started from. The tuned llama shape below replaced it as
//!   the shipped kernel (816.5 GB/s in the tuning pass's healthy
//!   window; the full record is BACKEND.md's second
//!   BACKEND-RESULTS section).
//! - The llama shape, a port of llama.cpp's Vulkan
//!   `mul_mat_vec_q4_k.comp` structure (the shader ggml-vulkan
//!   compiles for Q4_K GEMV; sources read from the server's
//!   llama.cpp tree, `ggml/src/ggml-vulkan/vulkan-shaders/`):
//!   sixteen threads cooperate on one super-block, each thread
//!   owning sixteen elements (two u32 words of nibbles, unpacked
//!   with the byte-unpack trick instead of per-element shifts);
//!   the eight sub-block scale/min pairs are extracted with
//!   llama.cpp's packed construction (three u16 lanes and two
//!   `unpack4xU8`s per thread-half) instead of twelve byte
//!   extractions; a workgroup covers several output rows so the
//!   activation quads each thread loads serve every row in the
//!   tile (llama.cpp's NUM_ROWS, which is 2 for K-quants on RDNA3
//!   per `ggml-vulkan.cpp`'s `rm_kq`); and the reduction is
//!   either a workgroup-memory tree or, when the device granted
//!   the subgroup feature, a segmented shuffle butterfly inside
//!   each sixteen-thread group plus a short shared-memory finish
//!   (ggml's HYBRID reduction mode, with the subgroup forced to
//!   the block-group width there by construction and here by the
//!   xor masks staying inside the aligned sixteen).
//!
//!   The f16 variant additionally rides the activations as
//!   `vec4<f16>` and takes the sixteen-element dot products in
//!   f16 (the packed form llama.cpp's `mul_mat_vec_q4_k_f16_f32`
//!   pipeline uses for the activation side), converting each
//!   per-sub-block dot to f32 before the scale combine, so the
//!   accumulation across blocks and the row reduction stay f32.
//!   That is strictly tighter than llama.cpp's f16 pipeline,
//!   whose running `temp` is f16 throughout; whether it holds
//!   this crate's check tolerance is a measured question, and
//!   the answer is recorded in BACKEND.md's results.
//!
//! What is deliberately NOT here: shared-memory dequant staging.
//! A GEMV consumes every dequantized value exactly once, so
//! staging weights in workgroup memory buys a barrier and a
//! round trip for zero reuse; the values that ARE reused (the
//! activation quads and their sums, across the rows of a tile)
//! already live in registers. llama.cpp's GEMV stages nothing
//! either (its shared memory carries reduction partials, as
//! here); its GEMM path is the one that stages, for reuse
//! GEMV does not have. Scale staging fails the same test: all
//! sixteen lanes extract their scale bytes in parallel in the
//! same instruction slots a single staging lane would spend
//! serially, so the packed extraction above is the cooperative
//! form of the same idea.
//!
//! The grid is two-dimensional because one dispatch dimension
//! caps at 65,535 workgroups (BACKEND.md Section 2): the flat
//! workgroup index wraps over `grid_x`, which matters for the
//! 202,048-row head class of shapes even though K1's own widest
//! tensor (FFN at 19,968 rows) fits in one dimension.

use crate::device::GpuContext;
use crate::error::{Result, WgpuError};

use crate::diag::dbg_log;

/// Elements per Q4_K super-block.
const BLOCK_ELEMS: usize = 256;
/// Bytes per Q4_K super-block.
const BLOCK_BYTES: usize = 144;
/// wgpu's per-dimension dispatch cap on this stack.
const MAX_WORKGROUPS_PER_DIM: u32 = 65_535;

/// The parallel shape a K1 pipeline runs.
///
/// The legacy shape is the first landing's kernel, kept verbatim
/// for A/B measurement. The llama shapes share one generated
/// shader family, parameterized by rows per workgroup, workgroup
/// size, the reduction, and the activation precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct K1Shape {
    /// Output rows one workgroup covers (llama.cpp's NUM_ROWS).
    /// Ignored by the legacy shape, which covers one row.
    pub rows: u32,
    /// Threads per workgroup; a multiple of sixteen (one
    /// sixteen-thread group per super-block in flight). Ignored
    /// by the legacy shape, which runs 32.
    pub workgroup: u32,
    /// Reduce with the segmented subgroup butterfly (needs the
    /// device's SUBGROUP feature) instead of the pure
    /// workgroup-memory tree.
    pub subgroup: bool,
    /// Ride activations as f16 and take the per-sub-block dots
    /// in f16 (needs the device's SHADER_F16 feature).
    pub f16_activations: bool,
    /// The first landing's shader, unchanged.
    pub legacy: bool,
}

impl K1Shape {
    /// The first landing's shipped shape (v2, vec4 traffic).
    pub const LEGACY: K1Shape = K1Shape {
        rows: 1,
        workgroup: 32,
        subgroup: false,
        f16_activations: false,
        legacy: true,
    };

    /// A llama-family shape with the given tile geometry and the
    /// workgroup-memory reduction, f32 activations.
    #[must_use]
    pub const fn llama(rows: u32, workgroup: u32) -> K1Shape {
        K1Shape {
            rows,
            workgroup,
            subgroup: false,
            f16_activations: false,
            legacy: false,
        }
    }

    /// This shape with the subgroup reduction.
    #[must_use]
    pub const fn with_subgroup(self) -> K1Shape {
        K1Shape {
            subgroup: true,
            ..self
        }
    }

    /// This shape with f16 activations.
    #[must_use]
    pub const fn with_f16(self) -> K1Shape {
        K1Shape {
            f16_activations: true,
            ..self
        }
    }

    /// Whether the input buffer this shape reads holds packed
    /// f16 (two bytes per element) instead of f32.
    #[must_use]
    pub const fn input_is_f16(&self) -> bool {
        self.f16_activations && !self.legacy
    }

    /// The bench/check name of this shape: `v2` for the legacy
    /// kernel, else `r<rows>` with `sg` and `f16` suffixes for
    /// the reduction and the activation precision (workgroup
    /// size is a separate bench flag, defaulting to 128).
    #[must_use]
    pub fn name(&self) -> String {
        if self.legacy {
            return "v2".to_string();
        }
        let mut name = format!("r{}", self.rows);
        if self.subgroup {
            name.push_str("sg");
        }
        if self.f16_activations {
            name.push_str("f16");
        }
        name
    }

    /// Parse a bench/check shape name (see [`K1Shape::name`]),
    /// with workgroup size 128 (llama.cpp's large-workgroup
    /// variant on this card: subgroup16 times four).
    #[must_use]
    pub fn from_name(name: &str) -> Option<K1Shape> {
        if name == "v2" {
            return Some(K1Shape::LEGACY);
        }
        let rest = name.strip_prefix('r')?;
        let (digits, suffix) = rest.split_at(
            rest.find(|c: char| !c.is_ascii_digit())
                .unwrap_or(rest.len()),
        );
        let rows: u32 = digits.parse().ok()?;
        let mut shape = K1Shape::llama(rows, 128);
        match suffix {
            "" => {}
            "sg" => shape.subgroup = true,
            "f16" => shape.f16_activations = true,
            "sgf16" => {
                shape.subgroup = true;
                shape.f16_activations = true;
            }
            _ => return None,
        }
        Some(shape)
    }

    /// Check the shape against a device's granted features and
    /// the geometry rules, naming what is missing.
    fn validate(&self, features: wgpu::Features) -> Result<()> {
        if self.legacy {
            return Ok(());
        }
        if !(1..=8).contains(&self.rows) {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "k1 shape {}: rows per workgroup {} is outside 1..=8",
                    self.name(),
                    self.rows
                ),
            });
        }
        if self.workgroup == 0 || self.workgroup % 16 != 0 || self.workgroup > 256 {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "k1 shape {}: workgroup size {} is not a multiple of 16 in 16..=256",
                    self.name(),
                    self.workgroup
                ),
            });
        }
        if self.subgroup && !features.contains(wgpu::Features::SUBGROUP) {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "k1 shape {} needs the SUBGROUP device feature, which this device was not granted",
                    self.name()
                ),
            });
        }
        if self.f16_activations && !features.contains(wgpu::Features::SHADER_F16) {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "k1 shape {} needs the SHADER_F16 device feature, which this device was not granted",
                    self.name()
                ),
            });
        }
        Ok(())
    }
}

/// The shape [`K1Pipeline::new`] builds for a device with the
/// given granted features: the best measured shape the features
/// allow, falling back step by step to shapes that need less.
/// The measurements behind the choice are recorded in
/// BACKEND.md's BACKEND-RESULTS sections: the winner is two
/// rows per workgroup of 128 with the subgroup reduction
/// (816.5 GB/s on the server, 115 percent of llama.cpp's
/// measured 708), which is also the configuration ggml-vulkan
/// itself picks for K-quants on RDNA3 (`rm_kq` 2, large
/// workgroup, hybrid reduction). The f16 shapes are never
/// shipped: they measured slower than their f32 twins and
/// their real-tensor error (rel 5.9e-4) breaks this crate's
/// check tolerance (rel 1e-4), so they remain bench-only
/// variants behind explicit names.
#[must_use]
pub fn shipped_shape(features: wgpu::Features) -> K1Shape {
    if features.contains(wgpu::Features::SUBGROUP) {
        K1Shape::llama(2, 128).with_subgroup()
    } else {
        K1Shape::llama(2, 128)
    }
}

/// Round-to-nearest-even f32 to f16 bit pattern, for packing
/// activations for the f16 shapes. Exact inverse of
/// `gguf::dequant::f16_to_f32` up to the rounding it names.
#[must_use]
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7f_ffff;
    if exp == 0xff {
        // Inf keeps its class; NaN becomes a quiet NaN half.
        return sign | if frac == 0 { 0x7c00 } else { 0x7e00 };
    }
    let mut e = exp - 112; // the f16 exponent field if normal
    if e >= 0x1f {
        return sign | 0x7c00; // overflow rounds to inf
    }
    if e <= 0 {
        // Subnormal half (or zero): value is m * 2^-24 with m a
        // 10-bit field; shift the 24-bit significand down with
        // round-to-nearest-even on the dropped bits.
        if e < -10 {
            return sign;
        }
        let m = frac | 0x80_0000;
        let shift = (14 - e) as u32;
        let rem = m & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        let mut out = m >> shift;
        if rem > halfway || (rem == halfway && (out & 1) == 1) {
            out += 1;
        }
        return sign | (out as u16);
    }
    // Normal half: drop 13 mantissa bits, round to nearest even.
    let mut kept = frac >> 13;
    let rem = frac & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (kept & 1) == 1) {
        kept += 1;
        if kept == 0x400 {
            kept = 0;
            e += 1;
            if e >= 0x1f {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((e as u16) << 10) | (kept as u16)
}

/// Pack f32 activations as little-endian f16 bytes, the input
/// buffer layout the f16 shapes read.
#[must_use]
pub fn pack_input_f16(input: &[f32]) -> Vec<u8> {
    input
        .iter()
        .flat_map(|v| f32_to_f16_bits(*v).to_le_bytes())
        .collect()
}

/// The legacy K1 shader (v2), in WGSL. See BACKEND.md's results
/// section for its tuning history; it is kept verbatim so the
/// tuned shapes bench against the exact kernel that shipped.
const SHADER_V2: &str = r#"
struct Params {
    blocks_per_row: u32,
    out_rows: u32,
    grid_x: u32,
    batch: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partials: array<f32, 32>;

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
        // Subnormal half: value is frac * 2^-24, exact in f32.
        let v = f32(frac) * 5.960464477539063e-08;
        if (sign == 1u) { return -v; }
        return v;
    }
    return bitcast<f32>((sign << 31u) | ((exp + 112u) << 23u) | (frac << 13u));
}

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

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let flat = wid.y * params.grid_x + wid.x;
    // Whole-workgroup early out: every lane shares `flat`, so the
    // exit is uniform and the barriers below stay legal.
    if (flat >= params.out_rows * params.batch) {
        return;
    }
    let out_row = flat % params.out_rows;
    let batch_row = flat / params.out_rows;
    let lane = lid.x;

    // One lane per super-block (striding when a row has more
    // blocks than lanes). A block is nine vec4 words: header,
    // then eight of nibbles. The activation span of a block is
    // 64 vec4s, and every output row of a batch row reads the
    // same input vector.
    let row_block0 = out_row * params.blocks_per_row;
    let x_row_base = batch_row * params.blocks_per_row * 64u;

    var acc = 0.0;
    for (var blk = lane; blk < params.blocks_per_row; blk += 32u) {
        let bw = (row_block0 + blk) * 9u;
        let hdr = weights[bw];
        let d = f16_to_f32(hdr.x & 0xffffu);
        let dmin = f16_to_f32((hdr.x >> 16u) & 0xffffu);
        let sw = array<u32, 3>(hdr.y, hdr.z, hdr.w);
        let x_base = x_row_base + blk * 64u;
        // Four groups of 64 elements. Group g's nibble bytes are
        // vec4s 2g and 2g+1 after the header; within a byte, the
        // low nibble is element l and the high nibble element
        // l + 32 of the group, exactly ggml's grouping.
        for (var grp = 0u; grp < 4u; grp++) {
            let sm1 = scale_min_k4(2u * grp, sw);
            let sm2 = scale_min_k4(2u * grp + 1u, sw);
            let d1 = d * f32(sm1.x);
            let dm1 = dmin * f32(sm1.y);
            let d2 = d * f32(sm2.x);
            let dm2 = dmin * f32(sm2.y);
            var sum_qx_lo = 0.0;
            var sum_x_lo = 0.0;
            var sum_qx_hi = 0.0;
            var sum_x_hi = 0.0;
            // A group is 32 nibble bytes in two vec4s. Byte l of
            // the group pairs with activation l (low nibble) and
            // l + 32 (high nibble). The low span of a half is 16
            // activations = 4 vec4s; the high span sits 8 vec4s
            // later. All loop bounds are literal so the compiler
            // unrolls and the vec4 loads stay single instructions.
            for (var half = 0u; half < 2u; half++) {
                let qv = weights[bw + 1u + grp * 2u + half];
                for (var c = 0u; c < 4u; c++) {
                    let word = qv[c];
                    let xv_lo = input[x_base + grp * 16u + half * 4u + c];
                    let xv_hi = input[x_base + grp * 16u + 8u + half * 4u + c];
                    for (var j = 0u; j < 4u; j++) {
                        let qbyte = (word >> (8u * j)) & 0xffu;
                        let x_lo = xv_lo[j];
                        let x_hi = xv_hi[j];
                        sum_qx_lo += f32(qbyte & 0xfu) * x_lo;
                        sum_x_lo += x_lo;
                        sum_qx_hi += f32(qbyte >> 4u) * x_hi;
                        sum_x_hi += x_hi;
                    }
                }
            }
            acc += d1 * sum_qx_lo - dm1 * sum_x_lo + d2 * sum_qx_hi - dm2 * sum_x_hi;
        }
    }

    partials[lane] = acc;
    workgroupBarrier();
    var stride = 16u;
    while (stride > 0u) {
        if (lane < stride) {
            partials[lane] += partials[lane + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    if (lane == 0u) {
        output[batch_row * params.out_rows + out_row] = partials[0];
    }
}
"#;

/// The llama-family K1 shader for `shape`, generated with the
/// tile geometry as literals so the compiler unrolls the row
/// loop and sizes the reduction arrays exactly. The body follows
/// llama.cpp's `mul_mat_vec_q4_k.comp` (cited in the module
/// docs): thread `t` of a sixteen-thread group owns elements
/// `l0..l0+3` of sub-blocks `2h`, `2h+1`, `2h+4`, `2h+5` of each
/// super-block it strides, where `h = t / 8` is the block half
/// and `l0 = 4 * (t % 8)` the byte offset within the half's
/// 64-byte span (the qs plane starts at byte 16 of the block,
/// word 4). (llama.cpp interleaves the same assignment
/// differently across its lanes; the set of (word, activation)
/// pairs per thread is identical, and consecutive lanes here
/// read consecutive words.)
fn llama_shader(shape: &K1Shape) -> String {
    let rows = shape.rows;
    let wg = shape.workgroup;
    let groups = wg / 16;
    // Reduction scratch: the tree variant keeps one partial per
    // lane per row; the subgroup variant only the per-group sums.
    let partials_len = if shape.subgroup {
        rows * groups
    } else {
        rows * wg
    };
    let enable_f16 = if shape.f16_activations {
        "enable f16;\n"
    } else {
        ""
    };
    let input_decl = if shape.f16_activations {
        "array<vec4<f16>>"
    } else {
        "array<vec4<f32>>"
    };
    // Per-sub-block dot products and activation sums, in the
    // activation precision, named for the four sub-blocks this
    // thread owns (A and B from the first qs word's low and high
    // nibbles, C and D from the second word's).
    let dot_math = if shape.f16_activations {
        r#"
            let qa = vec4<f16>(unpack4xU8(w0 & 0x0f0f0f0fu));
            let qb = vec4<f16>(unpack4xU8((w0 >> 4u) & 0x0f0f0f0fu));
            let qc = vec4<f16>(unpack4xU8(w1 & 0x0f0f0f0fu));
            let qd = vec4<f16>(unpack4xU8((w1 >> 4u) & 0x0f0f0f0fu));
            acc[r] += d * (sc_lo.x * f32(dot(qa, xa)) + sc_lo.y * f32(dot(qb, xbq))
                         + sc_hi.x * f32(dot(qc, xc)) + sc_hi.y * f32(dot(qd, xd)))
                    - dmin * (sc_lo.z * f32(sa) + sc_lo.w * f32(sb)
                            + sc_hi.z * f32(sc_) + sc_hi.w * f32(sd));"#
    } else {
        r#"
            let qa = vec4<f32>(unpack4xU8(w0 & 0x0f0f0f0fu));
            let qb = vec4<f32>(unpack4xU8((w0 >> 4u) & 0x0f0f0f0fu));
            let qc = vec4<f32>(unpack4xU8(w1 & 0x0f0f0f0fu));
            let qd = vec4<f32>(unpack4xU8((w1 >> 4u) & 0x0f0f0f0fu));
            acc[r] += d * (sc_lo.x * dot(qa, xa) + sc_lo.y * dot(qb, xbq)
                         + sc_hi.x * dot(qc, xc) + sc_hi.y * dot(qd, xd))
                    - dmin * (sc_lo.z * sa + sc_lo.w * sb
                            + sc_hi.z * sc_ + sc_hi.w * sd);"#
    };
    let one = if shape.f16_activations {
        "vec4<f16>(1.0h)"
    } else {
        "vec4<f32>(1.0)"
    };
    let reduction = if shape.subgroup {
        format!(
            r#"
    // Segmented butterfly inside each sixteen-thread group
    // (the xor masks never cross the aligned sixteen, so each
    // group reduces its own partials inside whatever hardware
    // subgroup width the device runs), then one shared-memory
    // finish across groups: ggml's HYBRID reduction.
    for (var r = 0u; r < {rows}u; r++) {{
        var v = acc[r];
        v += subgroupShuffleXor(v, 1u);
        v += subgroupShuffleXor(v, 2u);
        v += subgroupShuffleXor(v, 4u);
        v += subgroupShuffleXor(v, 8u);
        acc[r] = v;
    }}
    if (t == 0u) {{
        for (var r = 0u; r < {rows}u; r++) {{
            partials[r * {groups}u + group] = acc[r];
        }}
    }}
    workgroupBarrier();
    if (lane == 0u) {{
        for (var r = 0u; r < {rows}u; r++) {{
            if (row0 + r < params.out_rows) {{
                var s = 0.0;
                for (var g = 0u; g < {groups}u; g++) {{
                    s += partials[r * {groups}u + g];
                }}
                output[batch_row * params.out_rows + row0 + r] = s;
            }}
        }}
    }}"#
        )
    } else {
        format!(
            r#"
    for (var r = 0u; r < {rows}u; r++) {{
        partials[r * {wg}u + lane] = acc[r];
    }}
    workgroupBarrier();
    var stride = {half_wg}u;
    while (stride > 0u) {{
        if (lane < stride) {{
            for (var r = 0u; r < {rows}u; r++) {{
                partials[r * {wg}u + lane] += partials[r * {wg}u + lane + stride];
            }}
        }}
        workgroupBarrier();
        stride >>= 1u;
    }}
    if (lane == 0u) {{
        for (var r = 0u; r < {rows}u; r++) {{
            if (row0 + r < params.out_rows) {{
                output[batch_row * params.out_rows + row0 + r] = partials[r * {wg}u];
            }}
        }}
    }}"#,
            half_wg = wg / 2
        )
    };

    format!(
        r#"{enable_f16}struct Params {{
    blocks_per_row: u32,
    out_rows: u32,
    grid_x: u32,
    batch: u32,
}}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input: {input_decl};
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partials: array<f32, {partials_len}>;

// IEEE 754 half to f32, mirroring gguf::dequant::f16_to_f32 for
// every value a weight file can hold (normals, subnormals, zero;
// inf and NaN payloads keep their class).
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

@compute @workgroup_size({wg})
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
    let row_groups = (params.out_rows + {rows}u - 1u) / {rows}u;
    let flat = wid.y * params.grid_x + wid.x;
    // Whole-workgroup early out: every lane shares `flat`, so
    // the exit is uniform and the barriers below stay legal.
    if (flat >= params.batch * row_groups) {{
        return;
    }}
    let batch_row = flat / row_groups;
    let row0 = (flat % row_groups) * {rows}u;
    let lane = lid.x;
    let group = lane / 16u;
    let t = lane % 16u;
    let half = t / 8u;
    let l0 = 4u * (t % 8u);
    // This thread's qs words within a super-block (the qs plane
    // starts at word 4, after the four header words; the second
    // word sits 64 bytes later) and its first activation quad
    // within the block's 64-quad span.
    let w0_off = 4u + 8u * half + l0 / 4u;
    let yv = 16u * half + l0 / 4u;

    var acc = array<f32, {rows}>();
    for (var blk = group; blk < params.blocks_per_row; blk += {groups}u) {{
        // Activation quads and their sums are per-block, shared
        // by every row of the tile: loaded once, used {rows}
        // times. A pairs with sub-block 2h, B with 2h+1 (the
        // first word's high nibbles, 32 elements later), C and
        // D with 2h+4 and 2h+5 (the second word, 128 later).
        let xb0 = (batch_row * params.blocks_per_row + blk) * 64u;
        let xa = input[xb0 + yv];
        let xbq = input[xb0 + yv + 8u];
        let xc = input[xb0 + yv + 32u];
        let xd = input[xb0 + yv + 40u];
        let sa = dot(xa, {one});
        let sb = dot(xbq, {one});
        let sc_ = dot(xc, {one});
        let sd = dot(xd, {one});
        for (var r = 0u; r < {rows}u; r++) {{
            let row = row0 + r;
            if (row >= params.out_rows) {{
                continue;
            }}
            let bw = (row * params.blocks_per_row + blk) * 36u;
            let hw = weights[bw];
            let d = f16_to_f32(hw & 0xffffu);
            let dmin = f16_to_f32(hw >> 16u);
            // llama.cpp's packed scale extraction
            // (mul_mat_vec_q4_k.comp): the three u16 lanes of
            // the scale bytes for this thread's half, combined
            // into (scaleA, scaleB, minA, minB) and (scaleC,
            // scaleD, minC, minD) with two byte unpacks. The
            // values equal get_scale_min_k4(2h), (2h+1), (2h+4),
            // (2h+5) exactly; the packing is the point.
            let s16 = (weights[bw + 1u] >> (16u * half)) & 0xffffu;
            let m16 = (weights[bw + 2u] >> (16u * half)) & 0xffffu;
            let h16 = (weights[bw + 3u] >> (16u * half)) & 0xffffu;
            let lo = (m16 << 16u) | s16;
            let sc_lo = vec4<f32>(unpack4xU8(lo & 0x3f3f3f3fu));
            let hi = (((h16 << 12u) | h16) & 0x0f0f0f0fu) | ((lo & 0xc0c0c0c0u) >> 2u);
            let sc_hi = vec4<f32>(unpack4xU8(hi));
            let w0 = weights[bw + w0_off];
            let w1 = weights[bw + w0_off + 16u];
            {dot_math}
        }}
    }}
{reduction}
}}
"#
    )
}

/// Grid dimensions `(x, y)` for `total` workgroups under the
/// per-dimension cap: x takes as many as it may, y takes the rest,
/// and the shader recovers the flat index as `y * grid_x + x`.
#[must_use]
pub fn grid_dims(total: u64) -> (u32, u32) {
    if total == 0 {
        return (0, 0);
    }
    let x = total.min(u64::from(MAX_WORKGROUPS_PER_DIM)) as u32;
    let y = total.div_ceil(u64::from(x));
    (x, y.min(u64::from(u32::MAX)) as u32)
}

/// A Q4_K weight tensor resident on the device: one storage buffer
/// holding the tensor's raw GGUF bytes, bound whole per dispatch
/// (BACKEND.md Section 5's per-tensor buffer strategy).
#[derive(Debug)]
pub struct K1Weights {
    /// Whole-tensor byte length, for the bench's stream accounting.
    pub byte_len: u64,
    /// Input width (elements per row); a whole number of blocks.
    pub in_features: usize,
    /// Output rows.
    pub out_rows: usize,
    buffer: wgpu::Buffer,
}

/// One prepared K1 dispatch set: a bind group plus grid dimensions,
/// built once for a (weights, input, output, batch) combination and
/// encodable any number of times. The bench encodes the same call
/// repeatedly; `run` encodes it once.
#[derive(Debug)]
pub struct K1Call {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    grid: (u32, u32),
}

impl K1Call {
    /// Record this call's dispatch into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k1 q4_k gemv"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.grid.0, self.grid.1, 1);
    }
}

/// The compiled K1 pipeline, in one [`K1Shape`]. Compilation
/// happens once, at backend construction, so Naga translation
/// cost stays out of the token loop (BACKEND.md Section 4).
#[derive(Debug)]
pub struct K1Pipeline {
    pipeline: wgpu::ComputePipeline,
    shape: K1Shape,
}

impl K1Pipeline {
    /// Compile the shipped shape for `device` (see
    /// [`shipped_shape`]: the best measured shape the device's
    /// granted features allow).
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] if wgpu rejects the module or the
    /// pipeline. (wgpu reports shader errors through its error
    /// handling rather than a return value on some paths; the
    /// crate's conformance tests compile this pipeline on whatever
    /// adapter a machine offers, which is where a WGSL defect
    /// would surface first.)
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        Self::with_shape(device, shipped_shape(device.features()))
    }

    /// Compile an explicit [`K1Shape`] on `device`. The shape is
    /// validated against the device's granted features first: a
    /// shape the device cannot run is a named refusal here, not
    /// a shader validation failure later.
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] on a shape the device cannot run
    /// or a shader wgpu rejects.
    pub fn with_shape(device: &wgpu::Device, shape: K1Shape) -> Result<Self> {
        shape.validate(device.features())?;
        let source: String = if shape.legacy {
            SHADER_V2.to_string()
        } else {
            llama_shader(&shape)
        };
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("k1 q4_k gemv shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k1 q4_k gemv pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        dbg_log!("k1 pipeline built: shape {}", shape.name());
        Ok(K1Pipeline { pipeline, shape })
    }

    /// The shape this pipeline runs.
    #[must_use]
    pub fn shape(&self) -> K1Shape {
        self.shape
    }

    /// Upload a Q4_K tensor's raw bytes into a fresh device buffer.
    /// The geometry is validated against the bytes before anything
    /// is uploaded: a mismatch is a refusal naming the tensor,
    /// never a truncated or padded upload.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] on any geometry or byte-count
    /// disagreement; [`WgpuError::LimitShortfall`] through
    /// [`GpuContext::check_tensor_fits`] when the device cannot
    /// hold or bind the tensor whole.
    pub fn upload(
        &self,
        ctx: &GpuContext,
        name: &str,
        bytes: &[u8],
        in_features: usize,
        out_rows: usize,
    ) -> Result<K1Weights> {
        if in_features == 0 || in_features % BLOCK_ELEMS != 0 {
            return Err(WgpuError::InvalidTensor {
                name: name.to_string(),
                detail: format!(
                    "input width {in_features} is not a positive whole number of {BLOCK_ELEMS}-element Q4_K blocks"
                ),
            });
        }
        if out_rows == 0 {
            return Err(WgpuError::InvalidTensor {
                name: name.to_string(),
                detail: "output row count is zero".to_string(),
            });
        }
        let expected = (in_features / BLOCK_ELEMS) * BLOCK_BYTES * out_rows;
        if bytes.len() != expected {
            return Err(WgpuError::InvalidTensor {
                name: name.to_string(),
                detail: format!(
                    "tensor holds {} bytes but [{in_features}, {out_rows}] of Q4_K needs {expected}",
                    bytes.len()
                ),
            });
        }
        ctx.check_tensor_fits(name, bytes.len() as u64)?;
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&buffer, 0, bytes);
        dbg_log!(
            "k1 upload {name}: {} bytes [{in_features} x {out_rows}]",
            bytes.len()
        );
        Ok(K1Weights {
            byte_len: bytes.len() as u64,
            in_features,
            out_rows,
            buffer,
        })
    }

    /// Prepare a dispatch of `weights` over a batch of input rows.
    /// `input` must hold `batch * weights.in_features` elements in
    /// this pipeline's input precision (f32, or packed f16 when
    /// [`K1Shape::input_is_f16`]) and `output` must hold
    /// `batch * weights.out_rows` f32; both are caller-owned
    /// device buffers so the bench can hold them resident across
    /// iterations.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when the buffer sizes disagree
    /// with the batch the caller declares.
    pub fn prepare(
        &self,
        ctx: &GpuContext,
        weights: &K1Weights,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        batch: u32,
    ) -> Result<K1Call> {
        if batch == 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k1 call".to_string(),
                detail: "batch is zero".to_string(),
            });
        }
        let total = if self.shape.legacy {
            u64::from(batch) * weights.out_rows as u64
        } else {
            u64::from(batch) * (weights.out_rows as u64).div_ceil(u64::from(self.shape.rows))
        };
        let grid = grid_dims(total);
        let params: [u32; 4] = [
            (weights.in_features / BLOCK_ELEMS) as u32,
            weights.out_rows as u32,
            grid.0,
            batch,
        ];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k1 params"),
            size: param_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&params_buffer, 0, &param_bytes);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k1 bind group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weights.buffer.as_entire_binding(),
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
            ],
        });
        Ok(K1Call {
            pipeline: self.pipeline.clone(),
            bind_group,
            grid,
        })
    }

    /// Convenience one-shot: upload `input` ([`batch`, in] rows,
    /// always given as f32; packed to f16 on the host when the
    /// shape reads f16), run the GEMV, read the [`batch`, out]
    /// result back to host. This is the `Compute::proj` data
    /// path; it pays an upload and a readback per call, which the
    /// crate docs state plainly.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when `input` is not a whole
    /// number of input rows; [`WgpuError::Readback`] when the
    /// result cannot be mapped back.
    pub fn run(&self, ctx: &GpuContext, weights: &K1Weights, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() || input.len() % weights.in_features != 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k1 input".to_string(),
                detail: format!(
                    "input holds {} elements, not a whole number of {}-wide rows",
                    input.len(),
                    weights.in_features
                ),
            });
        }
        let batch = (input.len() / weights.in_features) as u32;
        let in_bytes: Vec<u8> = if self.shape.input_is_f16() {
            pack_input_f16(input)
        } else {
            input.iter().flat_map(|v| v.to_le_bytes()).collect()
        };
        let input_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k1 input"),
            size: in_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&input_buffer, 0, &in_bytes);
        let out_elems = batch as usize * weights.out_rows;
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k1 output"),
            size: (out_elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k1 readback"),
            size: (out_elems * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let call = self.prepare(ctx, weights, &input_buffer, &output_buffer, batch)?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k1 run"),
            });
        call.encode(&mut encoder);
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, (out_elems * 4) as u64);
        ctx.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| WgpuError::Readback {
                detail: e.to_string(),
            })?;
        rx.recv()
            .map_err(|_| WgpuError::Readback {
                detail: "map callback never fired".to_string(),
            })?
            .map_err(|e| WgpuError::Readback {
                detail: e.to_string(),
            })?;
        let mapped = slice.get_mapped_range();
        let out: Vec<f32> = mapped
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        drop(mapped);
        staging.unmap();
        dbg_log!(
            "k1 run: batch={batch} out_rows={} bytes_streamed={}",
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
    fn grid_dims_single_dimension_when_rows_fit() {
        // The widest K1 tensor: FFN gate/up at 19,968 rows.
        assert_eq!(grid_dims(19_968), (19_968, 1));
        assert_eq!(grid_dims(65_535), (65_535, 1));
    }

    #[test]
    fn grid_dims_wraps_to_two_dimensions_past_the_cap() {
        // The output head's 202,048 rows, the case the second
        // dimension exists for.
        let (x, y) = grid_dims(202_048);
        assert_eq!(x, 65_535);
        assert_eq!(y, 4);
        assert!(u64::from(x) * u64::from(y) >= 202_048);
        assert_eq!(grid_dims(65_536), (65_535, 2));
    }

    #[test]
    fn grid_dims_zero_is_zero() {
        assert_eq!(grid_dims(0), (0, 0));
    }

    #[test]
    fn shape_names_round_trip() {
        for shape in [
            K1Shape::LEGACY,
            K1Shape::llama(1, 128),
            K1Shape::llama(2, 128),
            K1Shape::llama(4, 128),
            K1Shape::llama(2, 128).with_subgroup(),
            K1Shape::llama(4, 128).with_subgroup(),
            K1Shape::llama(2, 128).with_f16(),
            K1Shape::llama(4, 128).with_subgroup().with_f16(),
        ] {
            assert_eq!(K1Shape::from_name(&shape.name()), Some(shape));
        }
        assert_eq!(K1Shape::from_name("bogus"), None);
        assert_eq!(K1Shape::from_name("r"), None);
        assert_eq!(K1Shape::from_name("r2zz"), None);
    }

    #[test]
    fn f16_packing_round_trips_within_half_precision() {
        // Exact values first.
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-2.0), 0xc000);
        assert_eq!(f32_to_f16_bits(65_504.0), 0x7bff);
        assert_eq!(f32_to_f16_bits(f32::from_bits(0x3380_0000)), 0x0001); // 2^-24
                                                                          // Round trip through the gguf crate's exact decoder: the
                                                                          // error must stay within half-precision rounding.
        let mut state = 7u64;
        for _ in 0..4096 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let v = ((state >> 40) as u32) as f32 / (1u32 << 24) as f32 - 0.5;
            let back = gguf::dequant::f16_to_f32(f32_to_f16_bits(v));
            let rel = (back - v).abs() / v.abs().max(1e-6);
            assert!(rel <= 0.001, "f16 round trip of {v} gave {back}");
        }
    }

    #[test]
    fn llama_shader_carries_the_shape() {
        let shader = llama_shader(&K1Shape::llama(4, 128).with_subgroup());
        assert!(shader.contains("@workgroup_size(128)"));
        assert!(shader.contains("subgroupShuffleXor"));
        assert!(shader.contains("array<f32, 32>")); // 4 rows x 8 groups of partials
        let f16 = llama_shader(&K1Shape::llama(2, 64).with_f16());
        assert!(f16.starts_with("enable f16;"));
        assert!(f16.contains("array<vec4<f16>>"));
        assert!(f16.contains("@workgroup_size(64)"));
    }
}
