# test_attention_split_cpu.mojo - Lane D2 (Wave D) proofs for
# the tiled split-KV CPU twins in attention/attention.mojo.
#
# The split path consumes PRE-ROTATED Q/K (the GPU kernels'
# contract), so the independent Float64 reference here works
# directly on the given (already rotated) stores: no rotation
# inside the reference. Where a test starts from raw rows it
# rotates with Wave A's rope_cpu first, exactly the device
# path's construction, and cross-checks against Wave B's
# attention_cpu_rotated_cache (a different code path over the
# same stores).
# Bars: max abs error <= 1e-5 against Float64 references,
# the Wave B attention bar class.
#
# For the coordinator: wire this file into run_tests.sh's
# TestSuite list (lane files are named in the lane report;
# run_tests.sh is coordinator-only).

from std.math import sqrt, exp
from std.testing import TestSuite, assert_true, assert_equal

from attention.attention import (
    ROPE_NONE,
    attention_cpu_rotated_cache,
    attention_merge_partials_cpu,
    attn_split_count,
    attention_split_cpu,
    attention_split_partials_cpu,
    attn_split_range,
    attn_validate_tile,
)
from ops.rope import rope_cpu


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


def test_split_ranges_partition() raises:
    # (10, 3) -> (0,4), (4,3), (7,3): earlier splits take the
    # remainder, lengths differ by at most one.
    var r0 = attn_split_range(10, 3, 0)
    var r1 = attn_split_range(10, 3, 1)
    var r2 = attn_split_range(10, 3, 2)
    assert_equal(r0[0], 0)
    assert_equal(r0[1], 4)
    assert_equal(r1[0], 4)
    assert_equal(r1[1], 3)
    assert_equal(r2[0], 7)
    assert_equal(r2[1], 3)
    # General partition sweep: contiguous, complete.
    for stored in [1, 5, 64, 100, 1025]:
        for splits in [1, 2, 3, 4, 7, 8, 16]:
            var pos = 0
            for s in range(splits):
                var r = attn_split_range(stored, splits, s)
                assert_equal(r[0], pos)
                pos += r[1]
            assert_equal(pos, stored)
    # splits > stored: trailing splits are empty.
    var e = attn_split_range(5, 8, 7)
    assert_equal(e[0], 5)
    assert_equal(e[1], 0)
    # Bad requests refuse by name.
    var saw = 0
    try:
        _ = attn_split_range(10, 0, 0)
    except err:
        if String(err).find("attention_split_bad_range") != -1:
            saw += 1
    try:
        _ = attn_split_range(10, 3, 3)
    except err:
        if String(err).find("attention_split_bad_range") != -1:
            saw += 1
    assert_equal(saw, 2)


def test_split_count_chooser() raises:
    assert_equal(attn_split_count(64, 1024, 8), 1)
    assert_equal(attn_split_count(1024, 1024, 8), 1)
    assert_equal(attn_split_count(1025, 1024, 8), 2)
    assert_equal(attn_split_count(4096, 1024, 8), 4)
    assert_equal(attn_split_count(4097, 1024, 8), 5)
    assert_equal(attn_split_count(131072, 1024, 8), 8)
    assert_equal(attn_split_count(4096, 512, 4), 4)
    assert_equal(attn_split_count(4096, 1024, 1), 1)
    var refused = False
    try:
        _ = attn_split_count(0, 1024, 8)
    except err:
        refused = String(err).find("attention_split_bad_config") != -1
    assert_true(refused, "zero stored must refuse by name")


def test_tile_validator() raises:
    for t in [8, 16, 32, 64]:
        attn_validate_tile(t)
    var saw = 0
    for t in [0, 7, 24, 128, -16]:
        try:
            attn_validate_tile(t)
            assert_true(False, "uncompiled tile must refuse")
        except err:
            if String(err).find("attention_tiled_bad_tile") != -1:
                saw += 1
    assert_equal(saw, 5)


def test_merged_matches_f64_small() raises:
    comptime NQ = 4
    comptime NKV = 2
    comptime HD = 8
    comptime STORED = 5
    var q = _make(NQ * HD, 51)
    var k = _make(STORED * NKV * HD, 52)
    var v = _make(STORED * NKV * HD, 53)
    var expected = _ref_prerotated(q, k, v, NQ, NKV, HD)
    for splits in [1, 2, 3, 5]:
        var got = attention_split_cpu(q, k, v, STORED, NQ, NKV, HD, splits, 8)
        var err = _max_err(got, expected)
        print("split merged small splits", splits, "max_err", err)
        assert_true(err <= 1.0e-5, "merged split output above 1e-5 bar")


def test_merged_matches_rotated_cache_hd128() raises:
    # Glimmer geometry, sliding RoPE: raw rows rotated with
    # rope_cpu (Q at position 63, each K row at its own
    # position), then the split path over the rotated stores
    # must match Wave B's rotated-cache twin over the same
    # stores, for split counts that divide, miss, and exceed
    # the tile boundaries.
    comptime NQ = 4
    comptime NKV = 2
    comptime HD = 128
    comptime STORED = 64
    comptime POS = 63
    comptime BASE = Float32(500000.0)
    var q = _make(NQ * HD, 61)
    var k = _make(STORED * NKV * HD, 62)
    var v = _make(STORED * NKV * HD, 63)
    var q_pos = List[Int](capacity=NQ)
    q_pos.resize(NQ, POS)
    var q_rot = rope_cpu(q, q_pos, HD, BASE)
    var k_pos = List[Int](capacity=STORED * NKV)
    k_pos.resize(STORED * NKV, 0)
    for i in range(STORED):
        for kv in range(NKV):
            k_pos[i * NKV + kv] = i
    var k_rot = rope_cpu(k, k_pos, HD, BASE)
    var twin = attention_cpu_rotated_cache(
        q, k_rot, v, 0, STORED, NQ, NKV, HD, POS, BASE
    )
    for splits in [1, 2, 4, 8]:
        for tile in [8, 16, 32, 64]:
            var got = attention_split_cpu(
                q_rot, k_rot, v, STORED, NQ, NKV, HD, splits, tile
            )
            var diff = _max_diff(got, twin)
            print(
                "split vs rotated-cache splits",
                splits,
                "tile",
                tile,
                "max_diff",
                diff,
            )
            assert_true(diff <= 1.0e-5, "split path diverged from twin")


def test_empty_splits_merge_as_noops() raises:
    comptime NQ = 2
    comptime NKV = 1
    comptime HD = 8
    comptime STORED = 5
    var q = _make(NQ * HD, 71)
    var k = _make(STORED * NKV * HD, 72)
    var v = _make(STORED * NKV * HD, 73)
    var expected = _ref_prerotated(q, k, v, NQ, NKV, HD)
    var got = attention_split_cpu(q, k, v, STORED, NQ, NKV, HD, 8, 16)
    var err = _max_err(got, expected)
    print("split empty-splits max_err", err)
    assert_true(err <= 1.0e-5, "empty splits must merge as no-ops")


def test_merge_hand_computed_logsumexp() raises:
    # One head, head_dim 2, two partials with known states:
    # split0 (max 1, sum 2, acc [2, 4]), split1 (max 3, sum
    # 1, acc [3, -3]). Combined max 3: split0 rescales by
    # exp(-2), split1 by 1 - the exact combine, checked in
    # Float64 here.
    var p_out = List[Float32]()
    for x in [Float32(2.0), Float32(4.0), Float32(3.0), Float32(-3.0)]:
        p_out.append(x)
    var p_max = List[Float32]()
    p_max.append(Float32(1.0))
    p_max.append(Float32(3.0))
    var p_sum = List[Float32]()
    p_sum.append(Float32(2.0))
    p_sum.append(Float32(1.0))
    var got = attention_merge_partials_cpu(p_out, p_max, p_sum, 2, 1, 2)
    var a1 = exp(Float64(-2.0))
    var esum = 2.0 * a1 + 1.0
    var e0 = (2.0 * a1 + 3.0) / esum
    var e1 = (4.0 * a1 - 3.0) / esum
    var d0 = Float64(got[0]) - e0
    if d0 < 0:
        d0 = -d0
    var d1 = Float64(got[1]) - e1
    if d1 < 0:
        d1 = -d1
    print("merge hand-check diffs", d0, d1)
    assert_true(d0 <= 1.0e-6 and d1 <= 1.0e-6, "merge combine inexact")
    # An empty partial between two live ones is skipped.
    var p_out3 = List[Float32]()
    for x in [
        Float32(2.0),
        Float32(4.0),
        Float32(0.0),
        Float32(0.0),
        Float32(3.0),
        Float32(-3.0),
    ]:
        p_out3.append(x)
    var p_max3 = List[Float32]()
    p_max3.append(Float32(1.0))
    p_max3.append(Float32(-3.4028235e38))
    p_max3.append(Float32(3.0))
    var p_sum3 = List[Float32]()
    p_sum3.append(Float32(2.0))
    p_sum3.append(Float32(0.0))
    p_sum3.append(Float32(1.0))
    var got3 = attention_merge_partials_cpu(p_out3, p_max3, p_sum3, 3, 1, 2)
    assert_true(
        _max_diff(got3, got) <= 1.0e-6,
        "empty middle partial must not change the merge",
    )


def test_tile_choice_stays_in_bar() raises:
    # The tile sets the rescale points, so two tiles round
    # differently; both must stay inside the bar against the
    # Float64 reference (they are the same math).
    comptime NQ = 4
    comptime NKV = 2
    comptime HD = 32
    comptime STORED = 100
    var q = _make(NQ * HD, 81)
    var k = _make(STORED * NKV * HD, 82)
    var v = _make(STORED * NKV * HD, 83)
    var expected = _ref_prerotated(q, k, v, NQ, NKV, HD)
    var a = attention_split_cpu(q, k, v, STORED, NQ, NKV, HD, 3, 8)
    var b = attention_split_cpu(q, k, v, STORED, NQ, NKV, HD, 3, 64)
    print(
        "tile invariance: err8",
        _max_err(a, expected),
        "err64",
        _max_err(b, expected),
        "cross",
        _max_diff(a, b),
    )
    assert_true(_max_err(a, expected) <= 1.0e-5, "tile 8 above bar")
    assert_true(_max_err(b, expected) <= 1.0e-5, "tile 64 above bar")
    assert_true(_max_diff(a, b) <= 1.0e-5, "tiles disagree beyond bar")


def test_long_cpu_1025_positions() raises:
    # Past the 1,024 mark on the CPU twin (the card proof
    # carries the 4,096-position fixture): stored 1025,
    # splits engage, output inside the bar vs Float64.
    comptime NQ = 2
    comptime NKV = 1
    comptime HD = 32
    comptime STORED = 1025
    var q = _make(NQ * HD, 91)
    var k = _make(STORED * NKV * HD, 92)
    var v = _make(STORED * NKV * HD, 93)
    var expected = _ref_prerotated(q, k, v, NQ, NKV, HD)
    var splits = attn_split_count(STORED, 1024, 8)
    assert_equal(splits, 2)
    var got = attention_split_cpu(q, k, v, STORED, NQ, NKV, HD, splits, 32)
    var err = _max_err(got, expected)
    print("split long-1025 max_err", err)
    assert_true(err <= 1.0e-5, "long split output above bar")


def test_split_refusals_by_name() raises:
    var q = _make(2 * 8, 95)
    var k = _make(2 * 1 * 8, 96)
    var v = _make(2 * 1 * 8, 97)
    var saw = 0
    try:
        _ = attention_split_cpu(q, k, v, 2, 2, 1, 8, 0, 16)
    except err:
        if String(err).find("attention_split_bad_splits") != -1:
            saw += 1
    try:
        _ = attention_split_cpu(q, k, v, 2, 2, 1, 8, 2, 7)
    except err:
        if String(err).find("attention_tiled_bad_tile") != -1:
            saw += 1
    var qbad = _make(3, 98)
    try:
        _ = attention_split_cpu(qbad, k, v, 2, 2, 1, 8, 1, 16)
    except err:
        if String(err).find("attention_split_bad_shape") != -1:
            saw += 1
    try:
        _ = attention_split_cpu(q, k, v, 0, 2, 1, 8, 1, 16)
    except err:
        if String(err).find("attention_split_bad_shape") != -1:
            saw += 1
    # Merge over all-empty partials refuses by name.
    var p_out = List[Float32]()
    p_out.resize(2 * 1 * 2, Float32(0.0))
    var p_max = List[Float32]()
    p_max.resize(2, Float32(-3.4028235e38))
    var p_sum = List[Float32]()
    p_sum.resize(2, Float32(0.0))
    try:
        _ = attention_merge_partials_cpu(p_out, p_max, p_sum, 2, 1, 2)
    except err:
        if String(err).find("attention_split_empty_partials") != -1:
            saw += 1
    print("split refusals named:", saw, "of 5")
    assert_equal(saw, 5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
