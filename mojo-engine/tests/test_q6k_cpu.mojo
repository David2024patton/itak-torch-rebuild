# test_q6k_cpu.mojo - CPU fixture test for the Q6_K kernel.
# Lane A3.
#
# Fixture: SYNTHETIC Q6_K data, named as such. No recorded
# on-disk fixture for Q6_K exists in the repo (the Rust check,
# crates/wgpu-compute/examples/k2_check.rs, reads the real
# tensor from the server file at run time; its recorded bars
# are rel < 1e-4 and cosine > 0.999999 against the gguf CPU
# reference). This test follows the K1 CPU pattern: deterministic
# LCG bytes at rows=8, cols=512 (2 super-blocks per row), and an
# in-test independent Float64 reference that materializes each
# super-block's 256 values in layout order, following
# crates/gguf/src/dequant.rs dequant_q6_k's own loop structure
# (structurally different from the kernel's per-element route).
# Bars: cosine >= 0.999999999 and max relative error <= 1e-5,
# the same class as the K1 CPU test and tighter than the
# recorded k2_check bars.

from std.math import sqrt
from std.testing import TestSuite, assert_true, assert_equal

from kernels.q6k_gemv import q6k_gemv_cpu


def lcg_next(state: UInt32) -> UInt32:
    return state * 1664525 + 1013904223


def f16_ref(bits: UInt16) -> Float64:
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


def signed_byte64(b: UInt8) -> Float64:
    var v = Int(b)
    return Float64(v - 256 if v >= 128 else v)


def reference_gemv(
    w: List[UInt8], x: List[Float32], rows: Int, cols: Int
) -> List[Float64]:
    var row_bytes = (cols // 256) * 210
    var out = List[Float64](capacity=rows)
    out.resize(rows, Float64(0.0))
    for row in range(rows):
        var acc = Float64(0.0)
        for sb in range(cols // 256):
            var off = row * row_bytes + sb * 210
            var d = f16_ref(UInt16(w[off + 208]) | (UInt16(w[off + 209]) << 8))
            # Materialize the block's 256 values in layout order,
            # exactly dequant_q6_k's two-half loop.
            var vals = List[Float64](capacity=256)
            vals.resize(256, Float64(0.0))
            for half in range(2):
                var base = half * 128
                for l in range(32):
                    var is_ = l // 16
                    var ql0 = Int(w[off + half * 64 + l])
                    var ql1 = Int(w[off + half * 64 + 32 + l])
                    var qh = Int(w[off + 128 + half * 32 + l])
                    var q1 = Float64(((ql0 & 0xF) | ((qh & 3) << 4)) - 32)
                    var q2 = Float64(
                        ((ql1 & 0xF) | (((qh >> 2) & 3) << 4)) - 32
                    )
                    var q3 = Float64(((ql0 >> 4) | (((qh >> 4) & 3) << 4)) - 32)
                    var q4 = Float64(((ql1 >> 4) | (((qh >> 6) & 3) << 4)) - 32)
                    vals[base + l] = (
                        d * signed_byte64(w[off + 192 + half * 8 + is_]) * q1
                    )
                    vals[base + 32 + l] = (
                        d
                        * signed_byte64(w[off + 192 + half * 8 + is_ + 2])
                        * q2
                    )
                    vals[base + 64 + l] = (
                        d
                        * signed_byte64(w[off + 192 + half * 8 + is_ + 4])
                        * q3
                    )
                    vals[base + 96 + l] = (
                        d
                        * signed_byte64(w[off + 192 + half * 8 + is_ + 6])
                        * q4
                    )
            for i in range(256):
                acc += vals[i] * Float64(x[sb * 256 + i])
        out[row] = acc
    return out^


def check_bars(
    got: List[Float32], expected: List[Float64], rows: Int, label: String
) raises:
    var dot = Float64(0.0)
    var ng = Float64(0.0)
    var nr = Float64(0.0)
    var max_rel = Float64(0.0)
    for r in range(rows):
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
    print(label, "cosine", cosine, "max_rel", max_rel)
    assert_true(cosine >= 0.999999999, "cosine below the CPU bar")
    assert_true(max_rel <= 1.0e-5, "max rel error above the CPU bar")


def test_q6k_cpu_synthetic_fixture() raises:
    comptime ROWS = 8
    comptime COLS = 512
    comptime ROW_BYTES = (COLS // 256) * 210
    var w = List[UInt8](capacity=ROWS * ROW_BYTES)
    w.resize(ROWS * ROW_BYTES, UInt8(0))
    var state: UInt32 = 42
    for i in range(len(w)):
        state = lcg_next(state)
        w[i] = UInt8((state >> 16) & 0xFF)
    # Tame the f16 super-scale of each block: d = 0.0078125.
    for row in range(ROWS):
        for sb in range(COLS // 256):
            var off = row * ROW_BYTES + sb * 210
            w[off + 208] = 0x00
            w[off + 209] = 0x1C
    var x = List[Float32](capacity=COLS)
    x.resize(COLS, Float32(0.0))
    for i in range(COLS):
        state = lcg_next(state)
        x[i] = Float32(Int((state >> 16) & 0xFF) - 128) / Float32(64.0)
    var got = q6k_gemv_cpu(w, x, ROWS, COLS)
    var expected = reference_gemv(w, x, ROWS, COLS)
    check_bars(got, expected, ROWS, "q6k cpu synthetic:")


def test_q6k_cpu_refuses_bad_shape_by_name() raises:
    var w = List[UInt8](capacity=16)
    w.resize(16, UInt8(0))
    var x = List[Float32](capacity=100)
    x.resize(100, Float32(0.0))
    var refused = False
    try:
        _ = q6k_gemv_cpu(w, x, 1, 100)
    except e:
        refused = String(e).find("q6k_gemv_cpu_bad_shape") != -1
    assert_true(refused, "bad shape must refuse by name")
    assert_equal(len(w), 16)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
