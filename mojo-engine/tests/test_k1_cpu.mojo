# test_k1_cpu.mojo - CPU fixture test for the K1 kernel module.
#
# Fixture: SYNTHETIC Q4_K data, named as such (the bench note
# records why: the real Glimmer fixture lives on the server and
# is exercised there through the GPU path). Deterministic
# LCG-generated weight bytes and inputs at rows=8, cols=512
# (2 super-blocks per row). The reference is computed inside this
# file by a structurally different route: it materializes each
# super-block's 256 dequantized values in layout order and dots
# in Float64. Bars for this CPU pass: cosine >= 0.999999999 and
# max relative error <= 1e-5 against that reference (the real
# fixture's recorded GPU bars are cosine 1.000000000, rel
# 8.941e-08; see bench/mojo-kernel-test-2026-11-07.md).

from std.math import sqrt
from std.testing import TestSuite, assert_true, assert_equal

from kernels.k1_gemv import k1_gemv_cpu


def lcg_next(state: UInt32) -> UInt32:
    return state * 1664525 + 1013904223


def f16_ref(bits: UInt16) -> Float64:
    # Reference f16 decode via Float64 arithmetic.
    var sign: Float64 = -1.0 if (bits >> 15) & 1 == 1 else 1.0
    var exp = Int((bits >> 10) & 0x1F)
    var frac = Int(bits & 0x3FF)
    if exp == 0:
        return sign * Float64(frac) * 5.9604644775390625e-08
    var v = 1.0 + Float64(frac) / 1024.0
    var e = exp - 15
    var scale = Float64(1.0)
    if e >= 0:
        for _ in range(e):
            scale *= 2.0
    else:
        for _ in range(-e):
            scale /= 2.0
    return sign * v * scale


def scale_min_ref(
    j: Int, w: List[UInt8], sb_off: Int
) -> Tuple[Float64, Float64]:
    if j < 4:
        return (
            Float64(Int(w[sb_off + 4 + j]) & 63),
            Float64(Int(w[sb_off + 8 + j]) & 63),
        )
    var hi = Int(w[sb_off + 4 + (j + 4)])
    var sc = (hi & 0xF) | ((Int(w[sb_off + 4 + (j - 4)]) >> 6) << 4)
    var m = (hi >> 4) | ((Int(w[sb_off + 4 + j]) >> 6) << 4)
    return (Float64(sc), Float64(m))


def reference_gemv(
    w: List[UInt8], x: List[Float32], rows: Int, cols: Int
) -> List[Float64]:
    var row_bytes = (cols // 256) * 144
    var out = List[Float64](capacity=rows)
    out.resize(rows, Float64(0.0))
    for row in range(rows):
        var acc = Float64(0.0)
        for sb in range(cols // 256):
            var sb_off = row * row_bytes + sb * 144
            var d = f16_ref(UInt16(w[sb_off]) | (UInt16(w[sb_off + 1]) << 8))
            var dmin = f16_ref(
                UInt16(w[sb_off + 2]) | (UInt16(w[sb_off + 3]) << 8)
            )
            # Materialize the super-block's 256 values in order:
            # 8 groups of 32, low nibbles then high nibbles.
            var vals = List[Float64](capacity=256)
            vals.resize(256, Float64(0.0))
            for g in range(4):
                var lo = scale_min_ref(2 * g, w, sb_off)
                var hi = scale_min_ref(2 * g + 1, w, sb_off)
                for i in range(32):
                    var byte_lo = Int(w[sb_off + 16 + g * 32 + i])
                    vals[g * 64 + i] = (
                        d * lo[0] * Float64(byte_lo & 0xF) - dmin * lo[1]
                    )
                    vals[g * 64 + 32 + i] = (
                        d * hi[0] * Float64(byte_lo >> 4) - dmin * hi[1]
                    )
            for i in range(256):
                acc += vals[i] * Float64(x[sb * 256 + i])
        out[row] = acc
    return out^


def test_k1_cpu_synthetic_fixture() raises:
    comptime ROWS = 8
    comptime COLS = 512
    comptime ROW_BYTES = (COLS // 256) * 144
    var w = List[UInt8](capacity=ROWS * ROW_BYTES)
    w.resize(ROWS * ROW_BYTES, UInt8(0))
    var state: UInt32 = 42
    for i in range(len(w)):
        state = lcg_next(state)
        w[i] = UInt8((state >> 16) & 0xFF)
    # Keep the f16 scale fields finite and tame: force the two
    # f16 headers of each super-block to a small positive value
    # (d = 0.01-ish pattern) so the fixture exercises real math
    # without overflow noise. Byte 0..3 := f16(0.0078125)=0x1C00
    # for d and f16(0.00390625)=0x1800 for dmin, little-endian.
    for row in range(ROWS):
        for sb in range(COLS // 256):
            var off = row * ROW_BYTES + sb * 144
            w[off] = 0x00
            w[off + 1] = 0x1C
            w[off + 2] = 0x00
            w[off + 3] = 0x18
    var x = List[Float32](capacity=COLS)
    x.resize(COLS, Float32(0.0))
    for i in range(COLS):
        state = lcg_next(state)
        x[i] = Float32(Int((state >> 16) & 0xFF) - 128) / Float32(64.0)

    var got = k1_gemv_cpu(w, x, ROWS, COLS)
    var expected = reference_gemv(w, x, ROWS, COLS)

    var dot = Float64(0.0)
    var ng = Float64(0.0)
    var nr = Float64(0.0)
    var max_rel = Float64(0.0)
    for r in range(ROWS):
        var g = Float64(got[r])
        dot += g * expected[r]
        ng += g * g
        nr += expected[r] * expected[r]
        var denom = expected[r] if expected[r] >= 0 else -expected[r]
        if denom > 1.0e-6:
            var rel = g - expected[r]
            if rel < 0:
                rel = -rel
            rel /= denom
            if rel > max_rel:
                max_rel = rel
    var cosine = dot / (sqrt(ng) * sqrt(nr))
    print("k1 cpu synthetic: cosine", cosine, "max_rel", max_rel)
    assert_true(cosine >= 0.999999999, "cosine below the CPU bar")
    assert_true(max_rel <= 1.0e-5, "max rel error above the CPU bar")


def test_k1_cpu_refuses_bad_shape_by_name() raises:
    var w = List[UInt8](capacity=16)
    w.resize(16, UInt8(0))
    var x = List[Float32](capacity=100)
    x.resize(100, Float32(0.0))
    var refused = False
    try:
        _ = k1_gemv_cpu(w, x, 1, 100)
    except e:
        refused = String(e).find("k1_gemv_cpu_bad_shape") != -1
    assert_true(refused, "bad shape must refuse by name")
    assert_equal(len(w), 16)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
