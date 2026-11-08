# test_attention_direct_cpu.mojo - Wave E lane E2 proofs for
# the direct-output attention CPU twin in
# attention/attention.mojo (attention_direct_cpu, the twin
# of attention_tiled_direct_kernel).
#
# The direct path consumes PRE-ROTATED Q/K (the GPU
# kernels' contract), so the independent Float64 reference
# works directly on the given stores, exactly as in
# test_attention_split_cpu.
#
# Bars: (1) direct twin vs attention_split_cpu at
# splits == 1 must be EXACT (max diff 0.0): the direct
# float path is the single-split partials path plus the
# merge's final divide, operation for operation; (2)
# direct twin vs the Float64 reference <= 1e-5 max abs
# error, the Wave B attention bar class; (3) the split
# path's refusal names propagate (bad tile, bad shape).
#
# For the coordinator: wire this file into run_tests.sh's
# TestSuite list (lane files are named in the lane report;
# run_tests.sh is coordinator-only).

from std.math import sqrt, exp
from std.testing import TestSuite, assert_true, assert_equal

from attention.attention import (
    attention_direct_cpu,
    attention_split_cpu,
)


def _det(i: Int) -> Float32:
    return Float32(((i * 37 + 13) % 31) - 15) / Float32(8.0)


def _make(n: Int, seed: Int) -> List[Float32]:
    var x = List[Float32](capacity=n)
    x.resize(n, Float32(0.0))
    for i in range(n):
        x[i] = _det(i + seed * 101)
    return x^


def _ref_prerotated(
    q: List[Float32],
    keys: List[Float32],
    values: List[Float32],
    num_q_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
) -> List[Float64]:
    """Float64 single-position GQA over PRE-ROTATED stores,
    all stored rows visible."""
    var stored = len(keys) // (num_kv_heads * head_dim)
    var group = num_q_heads // num_kv_heads
    var row = num_kv_heads * head_dim
    var scale = Float64(1.0) / sqrt(Float64(head_dim))
    var out = List[Float64](capacity=num_q_heads * head_dim)
    out.resize(num_q_heads * head_dim, Float64(0.0))
    for h in range(num_q_heads):
        var kv_head = h // group
        var scores = List[Float64](capacity=stored)
        scores.resize(stored, Float64(0.0))
        var mx = Float64(-1.0e30)
        for i in range(stored):
            var k_off = i * row + kv_head * head_dim
            var dot = Float64(0.0)
            for d in range(head_dim):
                dot += Float64(q[h * head_dim + d]) * Float64(keys[k_off + d])
            scores[i] = dot * scale
            if scores[i] > mx:
                mx = scores[i]
        var sum = Float64(0.0)
        for i in range(stored):
            scores[i] = exp(scores[i] - mx)
            sum += scores[i]
        for d in range(head_dim):
            var acc = Float64(0.0)
            for i in range(stored):
                var v_off = i * row + kv_head * head_dim
                acc += (scores[i] / sum) * Float64(values[v_off + d])
            out[h * head_dim + d] = acc
    return out^


def _max_err(got: List[Float32], expected: List[Float64]) -> Float64:
    var m = Float64(0.0)
    for i in range(len(got)):
        var d = Float64(got[i]) - expected[i]
        if d < 0:
            d = -d
        if d > m:
            m = d
    return m


def _max_diff(a: List[Float32], b: List[Float32]) -> Float64:
    var m = Float64(0.0)
    for i in range(len(a)):
        var d = Float64(a[i]) - Float64(b[i])
        if d < 0:
            d = -d
        if d > m:
            m = d
    return m


def _check_case(
    stored: Int,
    num_q_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    tile: Int,
) raises:
    var q = _make(num_q_heads * head_dim, 1)
    var keys = _make(stored * num_kv_heads * head_dim, 2)
    var values = _make(stored * num_kv_heads * head_dim, 3)
    var direct = attention_direct_cpu(
        q, keys, values, stored, num_q_heads, num_kv_heads, head_dim, tile
    )
    var split1 = attention_split_cpu(
        q, keys, values, stored, num_q_heads, num_kv_heads, head_dim, 1, tile
    )
    assert_equal(_max_diff(direct, split1), Float64(0.0))
    var refv = _ref_prerotated(
        q, keys, values, num_q_heads, num_kv_heads, head_dim
    )
    assert_true(_max_err(direct, refv) <= Float64(1.0e-5))


def test_direct_matches_split1_exact_and_ref() raises:
    # Small GQA, one tile exactly and across tile edges.
    _check_case(8, 4, 2, 16, 8)
    _check_case(9, 4, 2, 16, 8)
    _check_case(37, 4, 2, 16, 8)
    # Glimmer geometry class: 32 Q / 2 KV, head_dim 128,
    # tile 32; stored below, at, and past one tile and well
    # into the chooser's single-split regime (span 512).
    _check_case(16, 32, 2, 128, 32)
    _check_case(32, 32, 2, 128, 32)
    _check_case(33, 32, 2, 128, 32)
    _check_case(128, 32, 2, 128, 32)
    _check_case(512, 32, 2, 128, 32)
    # Tile invariance of the twin pair (both in Float64 bar,
    # cross-diff at the split suite's recorded scale).
    var q = _make(32 * 128, 4)
    var keys = _make(64 * 2 * 128, 5)
    var values = _make(64 * 2 * 128, 6)
    var d8 = attention_direct_cpu(q, keys, values, 64, 32, 2, 128, 8)
    var d64 = attention_direct_cpu(q, keys, values, 64, 32, 2, 128, 64)
    assert_true(_max_diff(d8, d64) <= Float64(1.0e-5))


def test_direct_refusals() raises:
    var q = _make(4 * 16, 1)
    var keys = _make(8 * 2 * 16, 2)
    var values = _make(8 * 2 * 16, 3)
    # Bad tile: the tiled path's named refusal.
    var saw_tile = False
    try:
        var _x = attention_direct_cpu(q, keys, values, 8, 4, 2, 16, 24)
    except e:
        saw_tile = String(e).find("attention_tiled_bad_tile") != -1
    assert_true(saw_tile)
    # Bad shape: q length against the head geometry.
    var saw_shape = False
    try:
        var _y = attention_direct_cpu(q, keys, values, 8, 8, 2, 16, 8)
    except e:
        saw_shape = String(e).find("attention_split_bad_shape") != -1
    assert_true(saw_shape)
    # Zero stored: the split shape gate refuses by name.
    var saw_stored = False
    try:
        var _z = attention_direct_cpu(q, keys, values, 0, 4, 2, 16, 8)
    except e:
        saw_stored = String(e).find("attention_split_bad_shape") != -1
    assert_true(saw_stored)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
