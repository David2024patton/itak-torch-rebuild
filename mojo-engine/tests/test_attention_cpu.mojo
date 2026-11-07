# test_attention_cpu.mojo - Lane B2 proof for attention/attention.
#
# References are INDEPENDENT Float64 evaluations inside this
# file (rotation, scores, softmax, weighted sum all recomputed
# in Float64 from the same synthetic inputs); no module helper
# is used to build an expected value. The one exception is the
# task-mandated cross-check in the sliding/RoPE test: Wave A's
# rope_cpu (a different module, already proven in Wave A) is
# compared against this file's Float64 rotation for one head,
# and the pre-rotated-cache variant is fed rope_cpu-rotated
# keys, mirroring the Rust write-time storage.
# Bar: max abs error <= 1e-5 against the Float64 references.

from std.math import sin, cos, pow, sqrt, exp
from std.testing import TestSuite, assert_true

from attention.attention import (
    ROPE_NONE,
    attention_cpu,
    attention_cpu_rotated_cache,
    attention_prefill_cpu,
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


def _rot64(
    x: List[Float32], pos_of_vec: List[Int], head_dim: Int, base: Float32
) -> List[Float64]:
    """Float64 interleaved rotation, vector by vector."""
    var out = List[Float64](capacity=len(x))
    out.resize(len(x), Float64(0.0))
    var vectors = len(x) // head_dim
    for v in range(vectors):
        var off = v * head_dim
        for i in range(head_dim // 2):
            var angle = Float64(pos_of_vec[v]) * pow(
                Float64(base), -(Float64(2 * i) / Float64(head_dim))
            )
            var a = Float64(x[off + 2 * i])
            var b = Float64(x[off + 2 * i + 1])
            out[off + 2 * i] = a * cos(angle) - b * sin(angle)
            out[off + 2 * i + 1] = a * sin(angle) + b * cos(angle)
    return out^


def _ref(
    q: List[Float32],
    keys: List[Float32],
    values: List[Float32],
    first_visible: Int,
    num_q_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    position: Int,
    rope_base: Float32,
    keys_prerotated: Bool,
) -> List[Float64]:
    """Float64 single-position GQA reference over the causal,
    visible rows only. Keys are raw unless keys_prerotated."""
    var stored = len(keys) // (num_kv_heads * head_dim)
    var effective = stored
    if position - first_visible + 1 < effective:
        effective = position - first_visible + 1
    var group = num_q_heads // num_kv_heads
    var row = num_kv_heads * head_dim
    var q_pos = List[Int](capacity=num_q_heads)
    q_pos.resize(num_q_heads, position)
    var k_pos = List[Int](capacity=stored * num_kv_heads)
    k_pos.resize(stored * num_kv_heads, 0)
    for i in range(stored):
        for kv in range(num_kv_heads):
            k_pos[i * num_kv_heads + kv] = first_visible + i
    var q64: List[Float64]
    var k64: List[Float64]
    if rope_base != ROPE_NONE:
        q64 = _rot64(q, q_pos, head_dim, rope_base)
        if keys_prerotated:
            k64 = List[Float64](capacity=len(keys))
            k64.resize(len(keys), Float64(0.0))
            for i in range(len(keys)):
                k64[i] = Float64(keys[i])
        else:
            k64 = _rot64(keys, k_pos, head_dim, rope_base)
    else:
        q64 = List[Float64](capacity=len(q))
        q64.resize(len(q), Float64(0.0))
        for i in range(len(q)):
            q64[i] = Float64(q[i])
        k64 = List[Float64](capacity=len(keys))
        k64.resize(len(keys), Float64(0.0))
        for i in range(len(keys)):
            k64[i] = Float64(keys[i])
    var scale = Float64(1.0) / sqrt(Float64(head_dim))
    var out = List[Float64](capacity=num_q_heads * head_dim)
    out.resize(num_q_heads * head_dim, Float64(0.0))
    for h in range(num_q_heads):
        var kv_head = h // group
        var scores = List[Float64](capacity=effective)
        scores.resize(effective, Float64(0.0))
        var mx = Float64(-1.0e30)
        for i in range(effective):
            var k_off = i * row + kv_head * head_dim
            var dot = Float64(0.0)
            for d in range(head_dim):
                dot += q64[h * head_dim + d] * k64[k_off + d]
            scores[i] = dot * scale
            if scores[i] > mx:
                mx = scores[i]
        var sum = Float64(0.0)
        for i in range(effective):
            scores[i] = exp(scores[i] - mx)
            sum += scores[i]
        for d in range(head_dim):
            var acc = Float64(0.0)
            for i in range(effective):
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


def test_tiny_gqa_matches_f64() raises:
    # (a) 2 KV heads, 4 Q heads, head_dim 8, 5 visible rows.
    comptime NQ = 4
    comptime NKV = 2
    comptime HD = 8
    comptime STORED = 5
    var q = _make(NQ * HD, 1)
    var k = _make(STORED * NKV * HD, 2)
    var v = _make(STORED * NKV * HD, 3)
    var got = attention_cpu(q, k, v, 0, STORED, NQ, NKV, HD, 4, ROPE_NONE)
    var expected = _ref(q, k, v, 0, NQ, NKV, HD, 4, ROPE_NONE, False)
    var err = _max_err(got, expected)
    print("attention tiny GQA 4q/2kv hd8: max_err", err)
    assert_true(err <= 1.0e-5, "tiny GQA above 1e-5 bar")


def test_causal_future_row_never_seen() raises:
    # (b) Row 2 carries a dominating key and a huge value; a
    # query at position 1 handed all 3 rows must ignore row 2.
    comptime HD = 4
    var q = List[Float32](capacity=HD)
    q.resize(HD, Float32(0.0))
    q[0] = 1.0
    var k = List[Float32](capacity=3 * HD)
    k.resize(3 * HD, Float32(0.0))
    k[0] = 0.0
    k[HD] = 1.0
    k[2 * HD] = 50.0
    var v = List[Float32](capacity=3 * HD)
    v.resize(3 * HD, Float32(0.0))
    v[0] = 1.0
    v[HD] = 100.0
    v[2 * HD] = 100000.0
    var got = attention_cpu(q, k, v, 0, 3, 1, 1, HD, 1, ROPE_NONE)
    var expected = _ref(q, k, v, 0, 1, 1, HD, 1, ROPE_NONE, False)
    var err = _max_err(got, expected)
    print("attention causal: max_err vs causal ref", err, "out0", got[0])
    assert_true(err <= 1.0e-5, "causal output must match rows 0..1 only")
    assert_true(got[0] < 1000.0, "future value 100000 leaked into output")


def test_window_boundary_first_visible() raises:
    # (c) Visible range starts mid-sequence at position 3.
    comptime NQ = 2
    comptime NKV = 1
    comptime HD = 8
    comptime STORED = 4
    var q = _make(NQ * HD, 5)
    var k = _make(STORED * NKV * HD, 6)
    var v = _make(STORED * NKV * HD, 7)
    var got = attention_cpu(q, k, v, 3, STORED, NQ, NKV, HD, 6, ROPE_NONE)
    var expected = _ref(q, k, v, 3, NQ, NKV, HD, 6, ROPE_NONE, False)
    var err = _max_err(got, expected)
    print("attention window boundary first_visible=3: max_err", err)
    assert_true(err <= 1.0e-5, "window-boundary output above bar")


def test_global_nope_no_rotation() raises:
    # (d1) Global layer: rope off, positions large; output is
    # the plain unrotated attention.
    comptime NQ = 4
    comptime NKV = 2
    comptime HD = 16
    comptime STORED = 6
    var q = _make(NQ * HD, 8)
    var k = _make(STORED * NKV * HD, 9)
    var v = _make(STORED * NKV * HD, 10)
    var got = attention_cpu(q, k, v, 10, STORED, NQ, NKV, HD, 15, ROPE_NONE)
    var expected = _ref(q, k, v, 10, NQ, NKV, HD, 15, ROPE_NONE, False)
    var err = _max_err(got, expected)
    print("attention global NoPE: max_err", err)
    assert_true(err <= 1.0e-5, "NoPE output above bar")


def test_sliding_rope_matches_f64_and_rope_cpu() raises:
    # (d2) Sliding layer at Glimmer geometry: head_dim 128,
    # base 500000, visible range starting at position 2, query
    # at position 7.
    comptime NQ = 4
    comptime NKV = 2
    comptime HD = 128
    comptime STORED = 6
    comptime BASE = Float32(500000.0)
    var q = _make(NQ * HD, 11)
    var k = _make(STORED * NKV * HD, 12)
    var v = _make(STORED * NKV * HD, 13)
    # Cross-check: rope_cpu on the Q row vs this file's Float64
    # rotation, head 0 (the Rust path rotates Q via rope_row).
    var q_pos = List[Int](capacity=NQ)
    q_pos.resize(NQ, 7)
    var q_rope_cpu = rope_cpu(q, q_pos, HD, BASE)
    var q_rot64 = _rot64(q, q_pos, HD, BASE)
    var rot_err = Float64(0.0)
    for i in range(HD):
        var d = Float64(q_rope_cpu[i]) - q_rot64[i]
        if d < 0:
            d = -d
        if d > rot_err:
            rot_err = d
    print("rope cross-check head0 (rope_cpu vs f64): max_err", rot_err)
    assert_true(rot_err <= 1.0e-5, "rope_cpu disagrees with f64 rotation")
    var got = attention_cpu(q, k, v, 2, STORED, NQ, NKV, HD, 7, BASE)
    var expected = _ref(q, k, v, 2, NQ, NKV, HD, 7, BASE, False)
    var err = _max_err(got, expected)
    print("attention sliding RoPE hd128 base500k: max_err", err)
    assert_true(err <= 1.0e-5, "sliding RoPE output above bar")
    # The rotated-cache variant: keys pre-rotated with rope_cpu
    # per row position (the Rust write-time storage) must give
    # the same output through attention_cpu_rotated_cache.
    var k_pos = List[Int](capacity=STORED * NKV)
    k_pos.resize(STORED * NKV, 0)
    for i in range(STORED):
        for kv in range(NKV):
            k_pos[i * NKV + kv] = 2 + i
    var k_rot = rope_cpu(k, k_pos, HD, BASE)
    var got2 = attention_cpu_rotated_cache(
        q, k_rot, v, 2, STORED, NQ, NKV, HD, 7, BASE
    )
    var err2 = _max_err(got2, expected)
    print("attention rotated-cache variant: max_err", err2)
    assert_true(err2 <= 1.0e-5, "rotated-cache variant above bar")


def test_prefill_windowed_causal_loop() raises:
    # Prefill helper: 5 positions, window 3, rope off. Each
    # output row must equal the Float64 reference for that
    # position restricted to its causal, windowed range.
    comptime NQ = 2
    comptime NKV = 1
    comptime HD = 4
    comptime NP = 5
    comptime WINDOW = 3
    var q = _make(NP * NQ * HD, 21)
    var k = _make(NP * NKV * HD, 22)
    var v = _make(NP * NKV * HD, 23)
    var got = attention_prefill_cpu(
        q, k, v, NP, NQ, NKV, HD, 0, WINDOW, ROPE_NONE
    )
    var worst = Float64(0.0)
    for idx in range(NP):
        var start = 0
        if idx - WINDOW + 1 > 0:
            start = idx - WINDOW + 1
        var count = idx - start + 1
        var q_row = List[Float32](capacity=NQ * HD)
        q_row.resize(NQ * HD, Float32(0.0))
        for i in range(NQ * HD):
            q_row[i] = q[idx * NQ * HD + i]
        var k_sub = List[Float32](capacity=count * NKV * HD)
        k_sub.resize(count * NKV * HD, Float32(0.0))
        var v_sub = List[Float32](capacity=count * NKV * HD)
        v_sub.resize(count * NKV * HD, Float32(0.0))
        for r in range(count):
            for i in range(NKV * HD):
                k_sub[r * NKV * HD + i] = k[(start + r) * NKV * HD + i]
                v_sub[r * NKV * HD + i] = v[(start + r) * NKV * HD + i]
        var expected = _ref(
            q_row, k_sub, v_sub, start, NQ, NKV, HD, idx, ROPE_NONE, False
        )
        var row_got = List[Float32](capacity=NQ * HD)
        row_got.resize(NQ * HD, Float32(0.0))
        for i in range(NQ * HD):
            row_got[i] = got[idx * NQ * HD + i]
        var err = _max_err(row_got, expected)
        if err > worst:
            worst = err
    print("attention prefill window3 causal: worst max_err", worst)
    assert_true(worst <= 1.0e-5, "prefill output above bar")


def test_refusals_by_name() raises:
    var saw = 0
    # Bad head_dim (odd).
    var q7 = _make(2 * 7, 31)
    var k7 = _make(1 * 1 * 7, 32)
    try:
        _ = attention_cpu(q7, k7, k7, 0, 1, 2, 1, 7, 0, ROPE_NONE)
    except e:
        if String(e).find("attention_cpu_bad_shape") != -1:
            saw += 1
    # Empty visible range.
    var q = _make(2 * 8, 33)
    var empty = List[Float32]()
    try:
        _ = attention_cpu(q, empty, empty, 0, 0, 2, 1, 8, 0, ROPE_NONE)
    except e:
        if String(e).find("attention_cpu_no_visible_keys") != -1:
            saw += 1
    # q shape wrong.
    var k = _make(2 * 1 * 8, 34)
    var v = _make(2 * 1 * 8, 35)
    var qbad = _make(3, 36)
    try:
        _ = attention_cpu(qbad, k, v, 0, 2, 2, 1, 8, 1, ROPE_NONE)
    except e:
        if String(e).find("attention_cpu_bad_shape") != -1:
            saw += 1
    # Nondivisible heads: 3 Q over 2 KV.
    var q3 = _make(3 * 8, 37)
    var k2 = _make(2 * 2 * 8, 38)
    try:
        _ = attention_cpu(q3, k2, k2, 0, 2, 3, 2, 8, 1, ROPE_NONE)
    except e:
        if String(e).find("attention_cpu_nondivisible_heads") != -1:
            saw += 1
    # Bad rope base (0 < base <= 1, and not ROPE_NONE).
    try:
        _ = attention_cpu(q, k, v, 0, 2, 2, 1, 8, 1, Float32(0.5))
    except e:
        if String(e).find("attention_cpu_bad_rope_base") != -1:
            saw += 1
    print("attention refusals named:", saw, "of 5")
    assert_true(saw == 5, "all five refusals must be named")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
