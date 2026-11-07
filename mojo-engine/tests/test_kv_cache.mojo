# test_kv_cache.mojo - Lane B1 proof for kv/cache.
#
# CPU, synthetic small shapes (plus one real-window shape).
# Every value expectation is computed HERE from the position /
# head / dim formula below, never from the cache's own layout:
# a read is proven when the value the formula predicts for that
# absolute position comes back, so a ring slot landing in the
# wrong place or an evicted row leaking through fails loudly.
#
# Formula (exact small integers in f32, distinct per position):
#   k element i of position p: (p * 131 + i * 17 + 11) % 997
#   v element i of position p: (p * 57 + i * 29 + 3) % 991
# where i is the flat row index (kv_head * head_dim + dim).

from std.testing import TestSuite, assert_true, assert_equal

from kv.cache import (
    KvCache,
    LayerKind,
    glimmer_layer_pattern,
    kv_size_accounting,
)


def _k_expected(position: Int, i: Int) -> Float32:
    return Float32((position * 131 + i * 17 + 11) % 997)


def _v_expected(position: Int, i: Int) -> Float32:
    return Float32((position * 57 + i * 29 + 3) % 991)


def _rows(
    kv_heads: Int, head_dim: Int, position: Int
) -> Tuple[List[Float32], List[Float32]]:
    var row = kv_heads * head_dim
    var k = List[Float32](capacity=row)
    var v = List[Float32](capacity=row)
    for i in range(row):
        k.append(_k_expected(position, i))
        v.append(_v_expected(position, i))
    return (k^, v^)


def _append_pos(
    mut cache: KvCache, layer: Int, kv_heads: Int, head_dim: Int, position: Int
) raises:
    var rows = _rows(kv_heads, head_dim, position)
    cache.append(layer, rows[0], rows[1], position)


def _assert_head(
    cache: KvCache, layer: Int, position: Int, kv_head: Int, head_dim: Int
) raises:
    var k = cache.key(layer, position, kv_head)
    var v = cache.value(layer, position, kv_head)
    assert_equal(len(k), head_dim)
    assert_equal(len(v), head_dim)
    for dim in range(head_dim):
        var i = kv_head * head_dim + dim
        assert_true(
            k[dim] == _k_expected(position, i),
            "key must be the value the formula predicts for this position",
        )
        assert_true(
            v[dim] == _v_expected(position, i),
            "value must be the value the formula predicts for this position",
        )


def _refusal_name(action: String, name: String) -> Bool:
    return action.find(name) != -1


# (a) global layer write/read round-trip, several positions,
# both KV heads, values exact.
def test_global_round_trip() raises:
    var pattern = List[LayerKind]()
    pattern.append(LayerKind.make_global())
    var cache = KvCache.new(pattern^, 2, 4)
    for pos in range(6):
        _append_pos(cache, 0, 2, 4, pos)
    assert_equal(cache.stored_len(0), 6)
    assert_equal(cache.next_position(0), 6)
    assert_equal(cache.first_visible_position(0), 0)
    for pos in range(6):
        for head in range(2):
            _assert_head(cache, 0, pos, head, 4)
    # Full-row accessors agree with the same formula.
    var k_row = cache.key_row(0, 3)
    var v_row = cache.value_row(0, 3)
    for i in range(8):
        assert_true(k_row[i] == _k_expected(3, i), "key_row formula value")
        assert_true(v_row[i] == _v_expected(3, i), "value_row formula value")
    print("kv global round-trip: 6 positions x 2 heads x 4 dims exact")


# (b) sliding window wraparound, small window 4: append 0..6,
# stored caps at 4, first visible 3, position 2 refuses as
# evicted, 3..6 read back exact, and slot reuse is right
# (position 6 sits in slot 2 = 6 % 4 and returns its own
# values, which differ from position 2's by the formula).
def test_window_wraparound_small() raises:
    var pattern = List[LayerKind]()
    pattern.append(LayerKind.make_windowed(4))
    var cache = KvCache.new(pattern^, 2, 3)
    for pos in range(7):
        _append_pos(cache, 0, 2, 3, pos)
    assert_equal(cache.stored_len(0), 4)
    assert_equal(cache.next_position(0), 7)
    assert_equal(cache.first_visible_position(0), 3)
    var saw_evicted = 0
    try:
        _ = cache.key(0, 2, 0)
    except e:
        if _refusal_name(String(e), "kv_position_evicted"):
            saw_evicted += 1
    try:
        _ = cache.value(0, 0, 1)
    except e:
        if _refusal_name(String(e), "kv_position_evicted"):
            saw_evicted += 1
    assert_equal(saw_evicted, 2)
    # The formula separates position 6 from position 2, so the
    # exact reads below prove slot 2 now holds position 6.
    assert_true(
        _k_expected(6, 0) != _k_expected(2, 0),
        "formula must separate positions 6 and 2 for the slot proof",
    )
    for pos in range(3, 7):
        for head in range(2):
            _assert_head(cache, 0, pos, head, 3)
    print("kv window 4 wrap: stored 4, first visible 3, 3..6 exact")


# (b, real shape) the Glimmer window 2048 at kv_heads 2,
# head_dim 128: append 0..2049, stored caps at 2048, first
# visible 2, boundary positions 2047/2048/2049 exact.
def test_window_real_2048_boundary() raises:
    var pattern = List[LayerKind]()
    pattern.append(LayerKind.make_windowed(2048))
    var cache = KvCache.new(pattern^, 2, 128)
    for pos in range(2050):
        _append_pos(cache, 0, 2, 128, pos)
    assert_equal(cache.stored_len(0), 2048)
    assert_equal(cache.next_position(0), 2050)
    assert_equal(cache.first_visible_position(0), 2)
    for pos in [2047, 2048, 2049]:
        for head in range(2):
            _assert_head(cache, 0, pos, head, 128)
    _assert_head(cache, 0, 2, 0, 128)
    var saw = 0
    try:
        _ = cache.key(0, 1, 0)
    except e:
        if _refusal_name(String(e), "kv_position_evicted"):
            saw += 1
    assert_equal(saw, 1)
    print("kv window 2048: 2050 appends, boundary positions exact")


# (c) a global layer in the same cache is unaffected by the
# windowed layer wrapping: its position 0 is still readable
# after the windowed layer has evicted its own position 0.
def test_global_unaffected_by_window_wrap() raises:
    var pattern = List[LayerKind]()
    pattern.append(LayerKind.make_windowed(4))
    pattern.append(LayerKind.make_global())
    var cache = KvCache.new(pattern^, 2, 4)
    for pos in range(7):
        _append_pos(cache, 0, 2, 4, pos)
        _append_pos(cache, 1, 2, 4, pos)
    assert_equal(cache.stored_len(0), 4)
    assert_equal(cache.stored_len(1), 7)
    assert_equal(cache.first_visible_position(1), 0)
    for head in range(2):
        _assert_head(cache, 1, 0, head, 4)
        _assert_head(cache, 1, 6, head, 4)
    print("kv mixed cache: global position 0 survives window wrap")


# Accessors and kinds on a fresh mixed cache.
def test_accessors_and_kinds() raises:
    var pattern = List[LayerKind]()
    pattern.append(LayerKind.make_windowed(8))
    pattern.append(LayerKind.make_global())
    var cache = KvCache.new(pattern^, 2, 4)
    assert_equal(cache.layer_count(), 2)
    assert_equal(cache.row_len(), 8)
    assert_equal(cache.stored_len(0), 0)
    assert_equal(cache.next_position(1), 0)
    assert_equal(cache.first_visible_position(0), 0)
    var w = cache.layer_kind(0)
    assert_true(not w.is_global, "layer 0 is windowed")
    assert_equal(w.window_size(), 8)
    var g = cache.layer_kind(1)
    assert_true(g == LayerKind.make_global(), "layer 1 is global")
    # A not-yet-written read refuses by name.
    var saw = 0
    try:
        _ = cache.key(1, 0, 0)
    except e:
        if _refusal_name(String(e), "kv_position_not_written"):
            saw += 1
    assert_equal(saw, 1)


# (d) named refusals: out-of-order append, layer out of range,
# wrong row length (k and v).
def test_refusals_append_and_layer() raises:
    var pattern = List[LayerKind]()
    pattern.append(LayerKind.make_global())
    var cache = KvCache.new(pattern^, 2, 4)
    var rows = _rows(2, 4, 0)
    var short = List[Float32](capacity=4)
    short.resize(4, Float32(0.0))
    var saw = 0
    # Skipping position 0.
    try:
        cache.append(0, rows[0], rows[1], 1)
    except e:
        if _refusal_name(String(e), "kv_out_of_order_position"):
            saw += 1
    # Layer id out of range on append and on a read accessor.
    try:
        cache.append(5, rows[0], rows[1], 0)
    except e:
        if _refusal_name(String(e), "kv_invalid_layer"):
            saw += 1
    try:
        _ = cache.stored_len(5)
    except e:
        if _refusal_name(String(e), "kv_invalid_layer"):
            saw += 1
    # Wrong row lengths, k then v.
    try:
        cache.append(0, short, rows[1], 0)
    except e:
        if _refusal_name(String(e), "kv_shape_mismatch"):
            saw += 1
    try:
        cache.append(0, rows[0], short, 0)
    except e:
        if _refusal_name(String(e), "kv_shape_mismatch"):
            saw += 1
    # A bad KV head on a read.
    _append_pos(cache, 0, 2, 4, 0)
    try:
        _ = cache.key(0, 0, 2)
    except e:
        if _refusal_name(String(e), "kv_invalid_kv_head"):
            saw += 1
    assert_equal(saw, 6)
    # Nothing from the refused appends was stored out of order.
    assert_equal(cache.stored_len(0), 1)
    assert_equal(cache.next_position(0), 1)


# (d) named refusals at construction: zero window (through the
# constructor AND smuggled through the raw fields), empty
# pattern, zero head count, zero head dim.
def test_refusals_construction() raises:
    var saw = 0
    try:
        _ = LayerKind.make_windowed(0)
    except e:
        if _refusal_name(String(e), "kv_invalid_window"):
            saw += 1
    var smuggled = List[LayerKind]()
    smuggled.append(LayerKind(False, 0))
    try:
        _ = KvCache.new(smuggled^, 2, 4)
    except e:
        if _refusal_name(String(e), "kv_invalid_window"):
            saw += 1
    var empty = List[LayerKind]()
    try:
        _ = KvCache.new(empty^, 2, 4)
    except e:
        if _refusal_name(String(e), "kv_zero_dimension"):
            saw += 1
    var one = List[LayerKind]()
    one.append(LayerKind.make_global())
    try:
        _ = KvCache.new(one^, 0, 4)
    except e:
        if _refusal_name(String(e), "kv_zero_dimension"):
            saw += 1
    var one_more = List[LayerKind]()
    one_more.append(LayerKind.make_global())
    try:
        _ = KvCache.new(one_more^, 2, 0)
    except e:
        if _refusal_name(String(e), "kv_zero_dimension"):
            saw += 1
    assert_equal(saw, 5)


# Checked size arithmetic: sizes that overflow refuse by name,
# never wrap into a small allocation.
def test_refusals_size_overflow() raises:
    var saw = 0
    # window * row overflows.
    var big_window = List[LayerKind]()
    big_window.append(LayerKind.make_windowed(9223372036854775807))
    try:
        _ = KvCache.new(big_window^, 2, 128)
    except e:
        if _refusal_name(String(e), "kv_size_overflow"):
            saw += 1
    # num_kv_heads * head_dim itself overflows, even on a global
    # layer (the row length is checked before any storage call).
    var global_only = List[LayerKind]()
    global_only.append(LayerKind.make_global())
    try:
        _ = KvCache.new(global_only^, 4611686018427387904, 2)
    except e:
        if _refusal_name(String(e), "kv_size_overflow"):
            saw += 1
    assert_equal(saw, 2)


# (e) Glimmer pattern and full-context f32 size accounting.
#
# Which parts the recorded total covers, plainly: windowed
# layers store WINDOW rows, not context rows, so the real
# full-context cost charges each global layer 131072 rows and
# each windowed layer 2048 rows:
#   (13 * 131072 + 39 * 2048) * 256 * 4 * 2 = 3,653,238,784
# That capped sum IS the recorded 3,653,238,784. It is NOT the
# all-global upper bound; the upper bound charges every one of
# the 52 layers the full context:
#   52 * 131072 * 256 * 4 * 2 = 13,958,643,712
# The accounting exposes both figures and this test asserts
# both, each computed from the formula here as well as against
# the recorded constant. (The lane brief's sentence calling
# the recorded total the all-global upper bound does not
# survive its own formula; the numbers above are what the
# formula gives, and both are asserted so the discrepancy is
# visible rather than papered over.)
def test_glimmer_pattern_and_size_accounting() raises:
    var pattern = glimmer_layer_pattern()
    assert_equal(len(pattern), 52)
    var globals = List[Int]()
    var windowed = 0
    for i in range(len(pattern)):
        if pattern[i].is_global:
            globals.append(i)
        else:
            windowed += 1
            assert_equal(pattern[i].window, 2048)
    assert_equal(windowed, 39)
    assert_equal(len(globals), 13)
    var expected_globals = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51]
    for i in range(13):
        assert_equal(globals[i], expected_globals[i])

    var acct = kv_size_accounting(pattern, 2, 128, 131072)
    # Bytes per position per layer: 256 floats * 4 bytes * 2 (K+V).
    assert_equal(acct.bytes_per_position, 256 * 4 * 2)
    # Capped total, from the formula and the recorded constant.
    var expected_capped = (13 * 131072 + 39 * 2048) * 256 * 4 * 2
    assert_equal(expected_capped, 3653238784)
    assert_equal(acct.total_bytes, expected_capped)
    # The global-only portion of the capped total.
    var expected_global_portion = 13 * 131072 * 256 * 4 * 2
    assert_equal(acct.global_bytes, expected_global_portion)
    assert_equal(acct.windowed_bytes, 39 * 2048 * 256 * 4 * 2)
    assert_equal(acct.global_bytes + acct.windowed_bytes, acct.total_bytes)
    # The all-global upper bound, from the formula.
    var expected_all_global = 52 * 131072 * 256 * 4 * 2
    assert_equal(expected_all_global, 13958643712)
    assert_equal(acct.all_global_bytes, expected_all_global)
    # The method form on a built Glimmer cache agrees (windowed
    # storage for 39 layers is allocated; globals start empty).
    var cache = KvCache.new(glimmer_layer_pattern(), 2, 128)
    assert_equal(cache.layer_count(), 52)
    var via_cache = cache.size_accounting(131072)
    assert_equal(via_cache.total_bytes, 3653238784)
    assert_equal(via_cache.all_global_bytes, 13958643712)
    print(
        "kv glimmer: 52 layers, 13 global, capped bytes 3653238784,"
        " all-global bound 13958643712"
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
