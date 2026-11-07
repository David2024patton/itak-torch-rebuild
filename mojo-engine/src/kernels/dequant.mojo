# dequant.mojo - shared dequant helpers for the GEMV kernel family
# (mojo-engine/src/kernels/). Lane A3 (Q6_K, Q5_K, F32/F16).
#
# The byte layouts and arithmetic mirror the Rust semantic
# reference, crates/gguf/src/dequant.rs, term for term:
#   Q6_K: 210-byte super-block of 256 elements: ql[0..128],
#         qh[128..192], signed scales sc[192..208], f16 d at 208.
#         Value = d * scale * (q - 32), q a 6-bit value assembled
#         from a ql nibble and two qh bits.
#   Q5_K: 176-byte super-block of 256 elements: f16 d, f16 dmin,
#         12 scale bytes, qh[16..48], ql[48..176]. Same grouping
#         as Q4_K; the fifth bit of each quant lives in qh.
#   F16:  exact half -> float conversion (same code as K1's).
#
# Two flavours of each helper: a device-pointer flavour for the
# GPU kernels and a host-List flavour for the CPU twins, exactly
# the split k1_gemv.mojo established.

from diag import log_debug


def f16_to_f32(bits: UInt16) -> Float32:
    # Exact f16 -> f32, identical to k1_gemv.mojo / gguf::dequant.
    var sign: Float32 = -1.0 if (bits >> 15) & 1 == 1 else 1.0
    var exp = Int((bits >> 10) & 0x1F)
    var frac = Int(bits & 0x3FF)
    if exp == 0:
        return sign * Float32(frac) * 5.9604644775390625e-08
    if exp == 31:
        return Float32(0.0) / Float32(0.0)
    var v = Float32(1.0) + Float32(frac) / Float32(1024.0)
    var e = exp - 15
    var scale = Float32(1.0)
    if e >= 0:
        for _ in range(e):
            scale *= Float32(2.0)
    else:
        for _ in range(-e):
            scale /= Float32(2.0)
    return sign * v * scale


def read_f16_ptr(w: Pointer[UInt8, MutAnyOrigin], off: Int) -> Float32:
    var bits = UInt16(w[unsafe_offset=off]) | (
        UInt16(w[unsafe_offset=off + 1]) << 8
    )
    return f16_to_f32(bits)


def read_f16_list(w: List[UInt8], off: Int) -> Float32:
    var bits = UInt16(w[off]) | (UInt16(w[off + 1]) << 8)
    return f16_to_f32(bits)


def signed_byte(b: UInt8) -> Int:
    var v = Int(b)
    return v - 256 if v >= 128 else v


# ---- Q4_K-style scale/min unpacking (Q5_K shares it) ----


def scale_min_k4_ptr(
    j: Int, w: Pointer[UInt8, MutAnyOrigin], sb_off: Int
) -> Tuple[Float32, Float32]:
    if j < 4:
        var sc = Int(w[unsafe_offset=sb_off + 4 + j]) & 63
        var m = Int(w[unsafe_offset=sb_off + 8 + j]) & 63
        return (Float32(sc), Float32(m))
    else:
        var hi = Int(w[unsafe_offset=sb_off + 4 + (j + 4)])
        var sc = (hi & 0xF) | (
            (Int(w[unsafe_offset=sb_off + 4 + (j - 4)]) >> 6) << 4
        )
        var m = (hi >> 4) | ((Int(w[unsafe_offset=sb_off + 4 + j]) >> 6) << 4)
        return (Float32(sc), Float32(m))


def scale_min_k4_list(
    j: Int, w: List[UInt8], sb_off: Int
) -> Tuple[Float32, Float32]:
    if j < 4:
        var sc = Int(w[sb_off + 4 + j]) & 63
        var m = Int(w[sb_off + 8 + j]) & 63
        return (Float32(sc), Float32(m))
    else:
        var hi = Int(w[sb_off + 4 + (j + 4)])
        var sc = (hi & 0xF) | ((Int(w[sb_off + 4 + (j - 4)]) >> 6) << 4)
        var m = (hi >> 4) | ((Int(w[sb_off + 4 + j]) >> 6) << 4)
        return (Float32(sc), Float32(m))


# ---- Q6_K single-element dequant (element `within`, 0..255) ----


def q6k_element_ptr(
    w: Pointer[UInt8, MutAnyOrigin], sb_off: Int, within: Int
) -> Float32:
    var half = within >> 7
    var r = within & 127
    var stream = r >> 5
    var l = r & 31
    var is_ = l >> 4
    var ql_base = sb_off + half * 64
    var qh_byte = Int(w[unsafe_offset=sb_off + 128 + half * 32 + l])
    var d = read_f16_ptr(w, sb_off + 208)
    var ql_byte: Int
    var hq: Int
    var sc_idx: Int
    if stream == 0:
        ql_byte = Int(w[unsafe_offset=ql_base + l])
        hq = qh_byte & 3
        sc_idx = is_
    elif stream == 1:
        ql_byte = Int(w[unsafe_offset=ql_base + 32 + l])
        hq = (qh_byte >> 2) & 3
        sc_idx = is_ + 2
    elif stream == 2:
        ql_byte = Int(w[unsafe_offset=ql_base + l])
        hq = (qh_byte >> 4) & 3
        sc_idx = is_ + 4
    else:
        ql_byte = Int(w[unsafe_offset=ql_base + 32 + l])
        hq = (qh_byte >> 6) & 3
        sc_idx = is_ + 6
    var nib = (ql_byte & 0xF) if stream < 2 else (ql_byte >> 4)
    var q = nib | (hq << 4)
    var sc = Float32(
        signed_byte(w[unsafe_offset=sb_off + 192 + half * 8 + sc_idx])
    )
    return d * sc * Float32(q - 32)


def q6k_element_list(w: List[UInt8], sb_off: Int, within: Int) -> Float32:
    var half = within >> 7
    var r = within & 127
    var stream = r >> 5
    var l = r & 31
    var is_ = l >> 4
    var ql_base = sb_off + half * 64
    var qh_byte = Int(w[sb_off + 128 + half * 32 + l])
    var d = read_f16_list(w, sb_off + 208)
    var ql_byte: Int
    var hq: Int
    var sc_idx: Int
    if stream == 0:
        ql_byte = Int(w[ql_base + l])
        hq = qh_byte & 3
        sc_idx = is_
    elif stream == 1:
        ql_byte = Int(w[ql_base + 32 + l])
        hq = (qh_byte >> 2) & 3
        sc_idx = is_ + 2
    elif stream == 2:
        ql_byte = Int(w[ql_base + l])
        hq = (qh_byte >> 4) & 3
        sc_idx = is_ + 4
    else:
        ql_byte = Int(w[ql_base + 32 + l])
        hq = (qh_byte >> 6) & 3
        sc_idx = is_ + 6
    var nib = (ql_byte & 0xF) if stream < 2 else (ql_byte >> 4)
    var q = nib | (hq << 4)
    var sc = Float32(signed_byte(w[sb_off + 192 + half * 8 + sc_idx]))
    return d * sc * Float32(q - 32)


# ---- Q5_K single-element dequant (element `within`, 0..255) ----


def q5k_element_ptr(
    w: Pointer[UInt8, MutAnyOrigin], sb_off: Int, within: Int
) -> Float32:
    var g = within >> 6
    var p = within & 63
    var l = p & 31
    var byte = Int(w[unsafe_offset=sb_off + 48 + g * 32 + l])
    var qh = Int(w[unsafe_offset=sb_off + 16 + l])
    var d = read_f16_ptr(w, sb_off)
    var dmin = read_f16_ptr(w, sb_off + 2)
    if p < 32:
        var sm = scale_min_k4_ptr(2 * g, w, sb_off)
        var u1 = 1 << (2 * g)
        var q = (byte & 0xF) + (16 if qh & u1 != 0 else 0)
        return d * sm[0] * Float32(q) - dmin * sm[1]
    else:
        var sm = scale_min_k4_ptr(2 * g + 1, w, sb_off)
        var u2 = 1 << (2 * g + 1)
        var q = (byte >> 4) + (16 if qh & u2 != 0 else 0)
        return d * sm[0] * Float32(q) - dmin * sm[1]


def q5k_element_list(w: List[UInt8], sb_off: Int, within: Int) -> Float32:
    var g = within >> 6
    var p = within & 63
    var l = p & 31
    var byte = Int(w[sb_off + 48 + g * 32 + l])
    var qh = Int(w[sb_off + 16 + l])
    var d = read_f16_list(w, sb_off)
    var dmin = read_f16_list(w, sb_off + 2)
    if p < 32:
        var sm = scale_min_k4_list(2 * g, w, sb_off)
        var u1 = 1 << (2 * g)
        var q = (byte & 0xF) + (16 if qh & u1 != 0 else 0)
        return d * sm[0] * Float32(q) - dmin * sm[1]
    else:
        var sm = scale_min_k4_list(2 * g + 1, w, sb_off)
        var u2 = 1 << (2 * g + 1)
        var q = (byte >> 4) + (16 if qh & u2 != 0 else 0)
        return d * sm[0] * Float32(q) - dmin * sm[1]


def note_loaded(module: String, rows: Int, cols: Int) -> String:
    return log_debug(
        module, "shape " + String(rows) + " rows x " + String(cols) + " cols"
    )
