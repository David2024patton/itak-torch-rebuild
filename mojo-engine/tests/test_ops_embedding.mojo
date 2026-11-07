# test_ops_embedding.mojo - Lane A4 proof for ops/embedding.
#
# Tolerance: EXACT. A row gather copies f32 values; any
# difference is a wrong row, not rounding. The reference is the
# table itself, indexed by hand in this file (a structurally
# trivial reference, which is the point: gather has no math to
# hide behind). Quantized dtypes must refuse by name.

from std.testing import TestSuite, assert_true

from ops.embedding import embedding_cpu, embedding_cpu_for_dtype


def _table(vocab: Int, hidden: Int) -> List[Float32]:
    var t = List[Float32](capacity=vocab * hidden)
    t.resize(vocab * hidden, Float32(0.0))
    for i in range(len(t)):
        t[i] = Float32(i % 251) / Float32(8.0) - Float32(16.0)
    return t^


def test_embedding_gather_exact() raises:
    comptime VOCAB = 37
    comptime HIDDEN = 32
    var table = _table(VOCAB, HIDDEN)
    var tokens = List[Int](capacity=4)
    tokens.resize(4, 0)
    tokens[0] = 0
    tokens[1] = 36
    tokens[2] = 5
    tokens[3] = 5
    var got = embedding_cpu(table, tokens, HIDDEN)
    for t in range(4):
        for j in range(HIDDEN):
            assert_true(
                got[t * HIDDEN + j] == table[tokens[t] * HIDDEN + j],
                "gathered row must be the table row, exactly",
            )
    print("embedding 4 tokens x 32: exact, max_abs 0.0")
    var via_dtype = embedding_cpu_for_dtype(table, tokens, HIDDEN, "F32")
    for i in range(len(got)):
        assert_true(via_dtype[i] == got[i], "F32 dtype path is the gather")


def test_embedding_refusals_by_name() raises:
    var table = _table(8, 4)
    var bad = List[Int](capacity=1)
    bad.resize(1, 8)
    var neg = List[Int](capacity=1)
    neg.resize(1, -1)
    var ok = List[Int](capacity=1)
    ok.resize(1, 3)
    var saw = 0
    try:
        _ = embedding_cpu(table, bad, 4)
    except e:
        if String(e).find("embedding_cpu_bad_token") != -1:
            saw += 1
    try:
        _ = embedding_cpu(table, neg, 4)
    except e:
        if String(e).find("embedding_cpu_bad_token") != -1:
            saw += 1
    try:
        _ = embedding_cpu(table, ok, 3)
    except e:
        if String(e).find("embedding_cpu_bad_shape") != -1:
            saw += 1
    try:
        _ = embedding_cpu_for_dtype(table, ok, 4, "Q8_0")
    except e:
        if String(e).find("embedding_cpu_quantized_unsupported") != -1:
            saw += 1
    assert_true(saw == 4, "all four refusals must be named")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
