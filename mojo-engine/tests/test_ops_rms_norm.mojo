# test_ops_rms_norm.mojo - Lane A4 proof for ops/rms_norm.
#
# Reference: independent Float64 evaluation of
# x / sqrt(mean(x^2) + eps) * w inside this file.
# Tolerance: max rel error <= 1e-5. Justification: the CPU twin
# accumulates in Float64 exactly like the Rust reference
# (crates/torch-math/src/norm.rs), so the only error left is the
# final f32 store (about 6e-8 relative); 1e-5 is the same bar the
# Rust-side known-vector test uses, with two orders of headroom.
# Also proven against the Rust crate's own recorded values:
# rms_norm([1,2,3,4], ones, 1e-5) ~= [0.36515, 0.73030, 1.09545,
# 1.46061] (norm.rs test known_vector, tol 1e-4 there).

from std.math import sqrt
from std.testing import TestSuite, assert_true

from config import DEFAULT_RMS_EPS
from ops.rms_norm import rms_norm_cpu, rms_norm_plain_cpu


def _fixture(n: Int, seed: Int) -> List[Float32]:
    var out = List[Float32](capacity=n)
    out.resize(n, Float32(0.0))
    for i in range(n):
        out[i] = Float32(((i * 37 + seed * 11) % 23) - 11) / Float32(4.0)
    return out^


def _ref(x: List[Float32], w: List[Float32], eps: Float32) -> List[Float64]:
    var ms = Float64(0.0)
    for i in range(len(x)):
        ms += Float64(x[i]) * Float64(x[i])
    ms /= Float64(len(x))
    var inv = Float64(1.0) / sqrt(ms + Float64(eps))
    var out = List[Float64](capacity=len(x))
    out.resize(len(x), Float64(0.0))
    for i in range(len(x)):
        out[i] = Float64(x[i]) * inv * Float64(w[i])
    return out^


def _max_rel(got: List[Float32], expected: List[Float64]) -> Float64:
    var m = Float64(0.0)
    for i in range(len(got)):
        var d = Float64(got[i]) - expected[i]
        if d < 0:
            d = -d
        var denom = expected[i] if expected[i] >= 0 else -expected[i]
        if denom > Float64(1.0e-6):
            d /= denom
            if d > m:
                m = d
    return m


def test_rms_norm_glimmer_eps_fixture() raises:
    # Width 128 fixture (one head-dim slice) at the Glimmer eps.
    var x = _fixture(128, 3)
    var w = _fixture(128, 7)
    for i in range(len(w)):
        w[i] = Float32(1.0) + w[i] / Float32(8.0)
    var got = rms_norm_cpu(x, w, DEFAULT_RMS_EPS)
    var err = _max_rel(got, _ref(x, w, DEFAULT_RMS_EPS))
    print("rms_norm 128 eps=1e-5: max_rel", err)
    assert_true(err <= 1.0e-5, "rms_norm max rel above 1e-5 bar")


def test_rms_norm_rust_recorded_vector() raises:
    var x = List[Float32](capacity=4)
    x.resize(4, Float32(0.0))
    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0
    x[3] = 4.0
    var w = List[Float32](capacity=4)
    w.resize(4, Float32(1.0))
    var got = rms_norm_cpu(x, w, Float32(1.0e-5))
    var expected = List[Float64](capacity=4)
    expected.resize(4, Float64(0.0))
    expected[0] = 0.36515
    expected[1] = 0.73030
    expected[2] = 1.09545
    expected[3] = 1.46061
    var worst = Float64(0.0)
    for i in range(4):
        var d = Float64(got[i]) - expected[i]
        if d < 0:
            d = -d
        if d > worst:
            worst = d
    print("rms_norm rust recorded vector: max_abs vs recorded", worst)
    assert_true(worst <= 1.0e-4, "disagrees with Rust recorded values")


def test_rms_norm_plain_and_eps_is_parameter() raises:
    var x = _fixture(64, 5)
    var ones = List[Float32](capacity=64)
    ones.resize(64, Float32(1.0))
    var plain = rms_norm_plain_cpu(x, DEFAULT_RMS_EPS)
    var err = _max_rel(plain, _ref(x, ones, DEFAULT_RMS_EPS))
    print("rms_norm_plain 64: max_rel", err)
    assert_true(err <= 1.0e-5, "plain form above bar")
    # eps is a parameter: the Rust large-eps case, eps=4 on
    # [2,2,2,2] gives 2/sqrt(8) = 0.70710678 per element.
    var c = List[Float32](capacity=4)
    c.resize(4, Float32(2.0))
    var w4 = List[Float32](capacity=4)
    w4.resize(4, Float32(1.0))
    var got = rms_norm_cpu(c, w4, Float32(4.0))
    var d = Float64(got[0]) - 0.7071067811865476
    if d < 0:
        d = -d
    assert_true(d <= 1.0e-6, "large-eps case must follow the parameter")


def test_rms_norm_refusals_by_name() raises:
    var x = _fixture(4, 1)
    var short_w = List[Float32](capacity=2)
    short_w.resize(2, Float32(1.0))
    var saw = 0
    try:
        _ = rms_norm_cpu(x, short_w, DEFAULT_RMS_EPS)
    except e:
        if String(e).find("rms_norm_cpu_bad_shape") != -1:
            saw += 1
    try:
        _ = rms_norm_cpu(x, x, Float32(0.0))
    except e:
        if String(e).find("rms_norm_cpu_bad_eps") != -1:
            saw += 1
    var empty = List[Float32]()
    try:
        _ = rms_norm_plain_cpu(empty, DEFAULT_RMS_EPS)
    except e:
        if String(e).find("rms_norm_plain_cpu_bad_shape") != -1:
            saw += 1
    assert_true(saw == 3, "all three refusals must be named")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
