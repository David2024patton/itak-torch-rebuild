# test_f32_cpu.mojo - CPU fixture tests for the F32 and F16
# GEMV twins. Lane A3.
#
# Fixtures: SYNTHETIC, named as such. No recorded on-disk
# fixture exists for plain-float GEMV in the repo; the
# reference here is an in-test independent Float64 dot (for
# F16, an independent Float64 f16 decode, same pattern as the
# K1 CPU test's f16_ref). Bars: cosine >= 0.999999999 and max
# relative error <= 1e-5, the same class as the K1 CPU test.
# The F16 fixture stores values as exact halves built from
# small dyadic rationals, so the decode is exercised on real
# bit patterns (normals and a subnormal) with no rounding
# ambiguity in the reference.

from std.math import sqrt
from std.testing import TestSuite, assert_true, assert_equal

from kernels.f32_gemv import f32_gemv_cpu, f16_gemv_cpu


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


def test_f32_cpu_synthetic_fixture() raises:
    comptime ROWS = 8
    comptime COLS = 512
    var w = List[Float32](capacity=ROWS * COLS)
    w.resize(ROWS * COLS, Float32(0.0))
    var state: UInt32 = 42
    for i in range(ROWS * COLS):
        state = lcg_next(state)
        w[i] = Float32(Int((state >> 16) & 0xFF) - 128) / Float32(32.0)
    var x = List[Float32](capacity=COLS)
    x.resize(COLS, Float32(0.0))
    for i in range(COLS):
        state = lcg_next(state)
        x[i] = Float32(Int((state >> 16) & 0xFF) - 128) / Float32(64.0)
    var expected = List[Float64](capacity=ROWS)
    expected.resize(ROWS, Float64(0.0))
    for r in range(ROWS):
        var acc = Float64(0.0)
        for c in range(COLS):
            acc += Float64(w[r * COLS + c]) * Float64(x[c])
        expected[r] = acc
    var got = f32_gemv_cpu(w, x, ROWS, COLS)
    check_bars(got, expected, ROWS, "f32 cpu synthetic:")


def test_f16_cpu_synthetic_fixture() raises:
    comptime ROWS = 8
    comptime COLS = 512
    # A spread of exact half bit patterns: 1.0, -2.0, 0.5,
    # -0.25, 0.0078125, smallest subnormal, 0.0, 100.0.
    var patterns = List[UInt16](capacity=8)
    patterns.append(0x3C00)
    patterns.append(0xC000)
    patterns.append(0x3800)
    patterns.append(0xB400)
    patterns.append(0x1C00)
    patterns.append(0x0001)
    patterns.append(0x0000)
    patterns.append(0x5640)
    var w = List[UInt8](capacity=ROWS * COLS * 2)
    w.resize(ROWS * COLS * 2, UInt8(0))
    var state: UInt32 = 7
    for i in range(ROWS * COLS):
        state = lcg_next(state)
        var bits = patterns[Int((state >> 16) & 7)]
        w[i * 2] = UInt8(bits & 0xFF)
        w[i * 2 + 1] = UInt8(bits >> 8)
    var x = List[Float32](capacity=COLS)
    x.resize(COLS, Float32(0.0))
    for i in range(COLS):
        state = lcg_next(state)
        x[i] = Float32(Int((state >> 16) & 0xFF) - 128) / Float32(64.0)
    var expected = List[Float64](capacity=ROWS)
    expected.resize(ROWS, Float64(0.0))
    for r in range(ROWS):
        var acc = Float64(0.0)
        for c in range(COLS):
            var idx = r * COLS + c
            var bits = UInt16(w[idx * 2]) | (UInt16(w[idx * 2 + 1]) << 8)
            acc += f16_ref(bits) * Float64(x[c])
        expected[r] = acc
    var got = f16_gemv_cpu(w, x, ROWS, COLS)
    check_bars(got, expected, ROWS, "f16 cpu synthetic:")


def test_f32_cpu_refuses_bad_shape_by_name() raises:
    var w = List[Float32](capacity=16)
    w.resize(16, Float32(0.0))
    var x = List[Float32](capacity=100)
    x.resize(100, Float32(0.0))
    var refused = False
    try:
        _ = f32_gemv_cpu(w, x, 3, 100)
    except e:
        refused = String(e).find("f32_gemv_cpu_bad_shape") != -1
    assert_true(refused, "bad shape must refuse by name")
    assert_equal(len(w), 16)


def test_f16_cpu_refuses_bad_shape_by_name() raises:
    var w = List[UInt8](capacity=16)
    w.resize(16, UInt8(0))
    var x = List[Float32](capacity=100)
    x.resize(100, Float32(0.0))
    var refused = False
    try:
        _ = f16_gemv_cpu(w, x, 3, 100)
    except e:
        refused = String(e).find("f16_gemv_cpu_bad_shape") != -1
    assert_true(refused, "bad shape must refuse by name")
    assert_equal(len(w), 16)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
