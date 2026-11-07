# test_ops_rope.mojo - Lane A4 proof for ops/rope.
#
# Reference: independent Float64 evaluation of the INTERLEAVED
# pairing (pair i = channels 2i, 2i+1; angle = pos *
# base^(-2i/head_dim)) inside this file - the Glimmer pairing
# (crates/torch-math/src/rope_interleaved.rs). The known-rotation
# case also distinguishes it from the half-split pairing, which
# gives (-0.301, 0, 1.382, 0) on the same input.
# Tolerance: max abs error <= 1e-5. Justification: angles are
# Float64 in both the twin and this reference (as in Rust), so
# the only error is the f32 store; 1e-5 matches the Rust test's
# known-rotation tolerance.

from std.math import sin, cos, pow, sqrt
from std.testing import TestSuite, assert_true

from ops.rope import GLIMMER_ROPE_BASE, rope_cpu


def _ref(
    x: List[Float32], positions: List[Int], head_dim: Int, base: Float32
) -> List[Float64]:
    var out = List[Float64](capacity=len(x))
    out.resize(len(x), Float64(0.0))
    var vectors = len(x) // head_dim
    for v in range(vectors):
        var off = v * head_dim
        for i in range(head_dim // 2):
            var angle = Float64(positions[v]) * pow(
                Float64(base), -(Float64(2 * i) / Float64(head_dim))
            )
            var a = Float64(x[off + 2 * i])
            var b = Float64(x[off + 2 * i + 1])
            out[off + 2 * i] = a * cos(angle) - b * sin(angle)
            out[off + 2 * i + 1] = a * sin(angle) + b * cos(angle)
    return out^


def _max_abs(got: List[Float32], expected: List[Float64]) -> Float64:
    var m = Float64(0.0)
    for i in range(len(got)):
        var d = Float64(got[i]) - expected[i]
        if d < 0:
            d = -d
        if d > m:
            m = d
    return m


def test_rope_known_rotation_interleaved() raises:
    var x = List[Float32](capacity=4)
    x.resize(4, Float32(0.0))
    x[0] = 1.0
    x[2] = 1.0
    var pos = List[Int](capacity=1)
    pos.resize(1, 1)
    var got = rope_cpu(x, pos, 4, Float32(10000.0))
    var expected = List[Float64](capacity=4)
    expected.resize(4, Float64(0.0))
    expected[0] = 0.5403023058681398
    expected[1] = 0.8414709848078965
    expected[2] = 0.9999500004166665
    expected[3] = 0.009999833334166664
    var err = _max_abs(got, expected)
    print("rope known rotation: max_abs", err)
    assert_true(err <= 1.0e-5, "interleaved known rotation off")


def test_rope_glimmer_shape_positions() raises:
    # Glimmer head dim 128, base 500000, three vectors at
    # positions 0, 3, 17 (position 0 must be an exact identity).
    comptime HD = 128
    var x = List[Float32](capacity=3 * HD)
    x.resize(3 * HD, Float32(0.0))
    for i in range(len(x)):
        x[i] = Float32(((i * 29) % 19) - 9) / Float32(5.0)
    var pos = List[Int](capacity=3)
    pos.resize(3, 0)
    pos[1] = 3
    pos[2] = 17
    var got = rope_cpu(x, pos, HD, GLIMMER_ROPE_BASE)
    var err = _max_abs(got, _ref(x, pos, HD, GLIMMER_ROPE_BASE))
    print("rope glimmer 3x128 base=500000: max_abs", err)
    assert_true(err <= 1.0e-5, "glimmer-shape rope above bar")
    for i in range(HD):
        assert_true(got[i] == x[i], "position 0 must be exact identity")
    # Rotation preserves each vector's sum of squares.
    var before = Float64(0.0)
    var after = Float64(0.0)
    for i in range(HD, 2 * HD):
        before += Float64(x[i]) * Float64(x[i])
        after += Float64(got[i]) * Float64(got[i])
    var drift = after - before
    if drift < 0:
        drift = -drift
    assert_true(drift <= 1.0e-3, "rope must preserve norms")


def test_rope_refusals_by_name() raises:
    var x = List[Float32](capacity=4)
    x.resize(4, Float32(1.0))
    var pos = List[Int](capacity=1)
    pos.resize(1, 0)
    var saw = 0
    try:
        _ = rope_cpu(x, pos, 3, Float32(10000.0))
    except e:
        if String(e).find("rope_cpu_bad_shape") != -1:
            saw += 1
    try:
        _ = rope_cpu(x, pos, 4, Float32(1.0))
    except e:
        if String(e).find("rope_cpu_bad_base") != -1:
            saw += 1
    var no_pos = List[Int]()
    try:
        _ = rope_cpu(x, no_pos, 4, Float32(10000.0))
    except e:
        if String(e).find("rope_cpu_bad_shape") != -1:
            saw += 1
    assert_true(saw == 3, "all three refusals must be named")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
