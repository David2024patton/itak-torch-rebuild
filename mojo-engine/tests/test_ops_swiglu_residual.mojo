# test_ops_swiglu_residual.mojo - Lane A4 proof for swiglu + add.
#
# References: Float64 silu(gate)*up and a+b inside this file.
# Tolerances: swiglu max rel <= 1e-5 (f32 exp/divide chain vs
# the Float64 evaluation of the same formula; Rust's own spot
# checks use 1e-6 on single elements, this bar covers a
# 19968-wide-style accumulation-free elementwise op at f32);
# residual add EXACT (f32 addition is the operation itself, the
# reference in Float64 of two f32 inputs rounded back to f32 is
# the identical value for these fixtures, whose values are
# chosen exactly representable with exact sums).

from std.math import exp
from std.testing import TestSuite, assert_true

from ops.residual import residual_add_cpu
from ops.swiglu import silu_cpu, swiglu_cpu


def test_swiglu_fixture() raises:
    comptime N = 256
    var gate = List[Float32](capacity=N)
    gate.resize(N, Float32(0.0))
    var up = List[Float32](capacity=N)
    up.resize(N, Float32(0.0))
    for i in range(N):
        gate[i] = Float32(((i * 13) % 21) - 10) / Float32(3.0)
        up[i] = Float32(((i * 7) % 13) - 6) / Float32(2.0)
    var got = swiglu_cpu(gate, up)
    var worst = Float64(0.0)
    for i in range(N):
        var g = Float64(gate[i])
        var expected = g / (Float64(1.0) + exp(-g)) * Float64(up[i])
        var d = Float64(got[i]) - expected
        if d < 0:
            d = -d
        var denom = expected if expected >= 0 else -expected
        if denom > Float64(1.0e-4):
            d /= denom
            if d > worst:
                worst = d
    print("swiglu 256: max_rel", worst)
    assert_true(worst <= 1.0e-5, "swiglu above bar")
    # Rust spot values: silu(1) = 0.7310585786300049, swiglu
    # ([1],[2]) = 1.4621171572600098 (activation.rs tests).
    var d1 = Float64(silu_cpu(Float32(1.0))) - 0.7310585786300049
    if d1 < 0:
        d1 = -d1
    assert_true(d1 <= 1.0e-6, "silu(1) spot value off")
    # Tails: no NaN, limits are 0 and x.
    var lo = silu_cpu(Float32(-100.0))
    var hi = silu_cpu(Float32(100.0))
    assert_true(
        lo <= Float32(1.0e-6) and lo >= Float32(-1.0e-6), "silu tail lo"
    )
    var dh = Float64(hi) - 100.0
    if dh < 0:
        dh = -dh
    assert_true(dh <= 1.0e-3, "silu tail hi")


def test_swiglu_refusal_by_name() raises:
    var a = List[Float32](capacity=2)
    a.resize(2, Float32(1.0))
    var b = List[Float32](capacity=1)
    b.resize(1, Float32(1.0))
    var refused = False
    try:
        _ = swiglu_cpu(a, b)
    except e:
        refused = String(e).find("swiglu_cpu_bad_shape") != -1
    assert_true(refused, "mismatch must refuse by name")


def test_residual_add_exact() raises:
    comptime N = 128
    var a = List[Float32](capacity=N)
    a.resize(N, Float32(0.0))
    var b = List[Float32](capacity=N)
    b.resize(N, Float32(0.0))
    for i in range(N):
        # Quarters and halves: exactly representable, sums exact.
        a[i] = Float32(((i * 5) % 17) - 8) / Float32(4.0)
        b[i] = Float32(((i * 3) % 11) - 5) / Float32(2.0)
    var got = residual_add_cpu(a, b)
    for i in range(N):
        assert_true(got[i] == a[i] + b[i], "residual add must be exact")
    print("residual_add 128: exact, max_abs 0.0")
    var short = List[Float32](capacity=1)
    short.resize(1, Float32(0.0))
    var refused = False
    try:
        _ = residual_add_cpu(a, short)
    except e:
        refused = String(e).find("residual_add_cpu_bad_shape") != -1
    assert_true(refused, "mismatch must refuse by name")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
