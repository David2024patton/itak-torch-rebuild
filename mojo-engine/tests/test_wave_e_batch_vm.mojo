# test_wave_e_batch_vm.mojo - VM tests for Wave E Lane
# E1's pure parts (no device):
#   - config.mojo's prefill routing knob: default, parser
#     accept/refuse-by-name, and the prefill_uses_batch
#     routing rule;
#   - gpu_layer_batch.mojo's causal stored-count helper;
#   - attention_batch_causal_cpu, the CPU twin of the
#     batched tiled attention, against (a) per-element
#     attention_split_cpu over the same causal prefixes
#     (the twin's construction, checked independently)
#     and (b) attention_cpu_rotated_cache at ROPE_NONE
#     (the single-position math core) inside the Wave B
#     bar class.
# The device launches are proven on the card by
# tests/server_proof_wave_e_batch_layer.mojo.
# NOTE for the coordinator: this file is NOT wired into
# tests/run_tests.sh (lane file-ownership rule); the lane
# note lists it for wiring.

from std.math import sqrt
from std.testing import TestSuite, assert_true, assert_equal

from config import (
    DEFAULT_PREFILL_BATCH_MIN_TOKENS,
    parse_prefill_batch_min_tokens,
    prefill_uses_batch,
)
from attention.attention import (
    attention_cpu_rotated_cache,
    attention_split_cpu,
    ROPE_NONE,
)
from model.gpu_layer_batch import (
    attention_batch_causal_cpu,
    batch_attention_stored,
    batch_row_slice,
)


def test_routing_knob_default_and_parser() raises:
    assert_equal(DEFAULT_PREFILL_BATCH_MIN_TOKENS, 16)
    assert_equal(parse_prefill_batch_min_tokens("1"), 1)
    assert_equal(parse_prefill_batch_min_tokens("64"), 64)
    assert_equal(parse_prefill_batch_min_tokens("1022"), 1022)
    for raw in [String("abc"), String("0"), String("-3"), String("1.5")]:
        var refused = False
        try:
            _ = parse_prefill_batch_min_tokens(raw)
        except e:
            var msg = String(e)
            refused = (
                msg.find("mojo_engine_config") != -1
                and msg.find("MOJO_ENGINE_PREFILL_BATCH_MIN_TOKENS") != -1
            )
        assert_true(refused, "routing knob must refuse by name: " + raw)


def test_prefill_uses_batch_rule() raises:
    # Both thresholds must admit the prompt (the Wave D
    # A/B control still forces the per-row arm alone).
    assert_true(prefill_uses_batch(64, 2, 64), "at both minima: batch")
    assert_true(prefill_uses_batch(1022, 2, 64), "long prompt: batch")
    assert_true(not prefill_uses_batch(63, 2, 64), "below route knob: rows")
    assert_true(not prefill_uses_batch(6, 2, 64), "short prompt: rows")
    assert_true(
        not prefill_uses_batch(1022, 1000000, 64),
        "GEMM A/B control raised: rows even for long prompts",
    )
    assert_true(prefill_uses_batch(2, 2, 1), "both at minimum: batch")
    assert_true(not prefill_uses_batch(1, 2, 1), "below GEMM minimum: rows")


def test_batch_attention_stored_counts() raises:
    assert_equal(batch_attention_stored(0, 0), 1)
    assert_equal(batch_attention_stored(0, 5), 6)
    assert_equal(batch_attention_stored(10, 0), 11)
    assert_equal(batch_attention_stored(10, 63), 74)
    var refused = False
    try:
        _ = batch_attention_stored(-1, 0)
    except e:
        refused = String(e).find("gpu_layer_bad_shape") != -1
    assert_true(refused, "negative start must refuse by name")


def _lcg_list(n: Int, seed: UInt32, scale: Float32) -> List[Float32]:
    var out = List[Float32](capacity=n)
    out.resize(n, Float32(0.0))
    var state = seed
    for i in range(n):
        state = state * 1664525 + 1013904223
        out[i] = (
            Float32(Int((state >> 16) & 0xFF) - 128) / Float32(128.0)
        ) * scale
    return out^


def _cosine_rel(
    got: List[Float32], expected: List[Float32]
) -> Tuple[Float64, Float64]:
    var dot = Float64(0.0)
    var ng = Float64(0.0)
    var ne = Float64(0.0)
    var max_abs = Float64(0.0)
    var max_e = Float64(0.0)
    for i in range(len(expected)):
        var g = Float64(got[i])
        var e = Float64(expected[i])
        dot += g * e
        ng += g * g
        ne += e * e
        var d = g - e
        if d < 0:
            d = -d
        if d > max_abs:
            max_abs = d
        var ae = e
        if ae < 0:
            ae = -ae
        if ae > max_e:
            max_e = ae
    var denom = max_e if max_e > 1.0 else 1.0
    return (dot / (sqrt(ng) * sqrt(ne)), max_abs / denom)


def test_batch_causal_cpu_matches_references() raises:
    comptime Q_HEADS = 4
    comptime KV_HEADS = 2
    comptime HEAD_DIM = 8
    comptime FIRST = 3
    comptime BATCH = 4
    comptime SPLITS = 2
    comptime TILE = 8
    var q_width = Q_HEADS * HEAD_DIM
    var kv_width = KV_HEADS * HEAD_DIM
    var total_stored = FIRST + BATCH - 1
    var q_rows = _lcg_list(BATCH * q_width, UInt32(11), Float32(0.5))
    var keys = _lcg_list(total_stored * kv_width, UInt32(23), Float32(0.5))
    var values = _lcg_list(total_stored * kv_width, UInt32(37), Float32(0.5))
    var got = attention_batch_causal_cpu(
        q_rows,
        keys,
        values,
        FIRST,
        Q_HEADS,
        KV_HEADS,
        HEAD_DIM,
        SPLITS,
        TILE,
    )
    assert_equal(len(got), BATCH * q_width)
    for i in range(BATCH):
        var stored = batch_attention_stored(FIRST - 1, i)
        var q_row = batch_row_slice(q_rows, i, q_width)
        var k_sub = List[Float32](capacity=stored * kv_width)
        k_sub.resize(stored * kv_width, Float32(0.0))
        var v_sub = List[Float32](capacity=stored * kv_width)
        v_sub.resize(stored * kv_width, Float32(0.0))
        for j in range(stored * kv_width):
            k_sub[j] = keys[j]
            v_sub[j] = values[j]
        var got_row = batch_row_slice(got, i, q_width)
        # (a) per-element split twin over the same prefix:
        # the float path is identical, so this is exact.
        var ref_split = attention_split_cpu(
            q_row,
            k_sub,
            v_sub,
            stored,
            Q_HEADS,
            KV_HEADS,
            HEAD_DIM,
            SPLITS,
            TILE,
        )
        for j in range(q_width):
            assert_true(
                got_row[j] == ref_split[j],
                "batch twin row differs from per-element split twin",
            )
        # (b) the single-position math core (Float64
        # accumulation): Wave B bar class.
        var ref_core = attention_cpu_rotated_cache(
            q_row,
            k_sub,
            v_sub,
            0,
            stored,
            Q_HEADS,
            KV_HEADS,
            HEAD_DIM,
            stored - 1,
            ROPE_NONE,
        )
        var cmp = _cosine_rel(got_row, ref_core)
        assert_true(
            cmp[0] >= 0.999999 and cmp[1] <= 1.0e-4,
            "batch twin vs math core outside bar: cos "
            + String(cmp[0])
            + " rel "
            + String(cmp[1]),
        )


def test_batch_causal_cpu_refuses_bad_shapes() raises:
    var q = _lcg_list(8, UInt32(5), Float32(0.5))
    var k = _lcg_list(8, UInt32(6), Float32(0.5))
    var saw = 0
    try:
        _ = attention_batch_causal_cpu(q, k, k, 0, 1, 1, 8, 1, 8)
        assert_true(False, "first_stored 0 must refuse")
    except e:
        assert_true(
            String(e).find("gpu_layer_bad_shape") != -1,
            "refusal must be named",
        )
        saw += 1
    try:
        _ = attention_batch_causal_cpu(q, k, k, 2, 1, 1, 8, 1, 8)
        assert_true(False, "short stores must refuse")
    except e:
        assert_true(
            String(e).find("gpu_layer_bad_shape") != -1,
            "refusal must be named",
        )
        saw += 1
    assert_equal(saw, 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
