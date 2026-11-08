# test_gpu_plumbing.mojo - Lane C1 proof for the pure parts
# of src/gpu/ (Wave C). NO device is touched: budget
# arithmetic, KV slot/layout arithmetic, weights slot
# arithmetic, and the refusal names on every pure path. The
# device paths themselves are proven on the card by
# tests/server_proof_gpu_plumbing.mojo.
#
# Every number below is a Wave C contract number stated as a
# literal AND computed from the formula, so a drift in either
# the code or the contract figures fails here by name.

from std.testing import TestSuite, assert_true, assert_equal

from gpu.context import (
    GPU_SCRATCH_ENVELOPE_BYTES,
    budget_release,
    check_gpu_budget,
    gpu_budget_total,
)
from gpu.device_kv import (
    kv_append_refusal,
    kv_buffer_bytes,
    kv_capacity_positions,
    kv_gather_needed,
    kv_slot_of,
    kv_total_bytes,
    kv_visible_slot,
)
from gpu.weights import meta_index, slot_resident_bytes
from kv.cache import LayerKind, glimmer_layer_pattern


def _refuses(name: String, err: String) -> Bool:
    return err.find(name) != -1


# Budget arithmetic with the contract's planning figures:
# weights planning bound 16,756,683,904 B (the whole Glimmer
# file), device KV 3,653,238,784 B at context 131,072, scratch
# envelope 536,870,912 B, planning total 20,946,793,600 B vs
# the cap 24,696,061,952 B.
def test_budget_arithmetic_contract_numbers() raises:
    assert_equal(GPU_SCRATCH_ENVELOPE_BYTES, 536870912)
    var total = gpu_budget_total(16756683904, 3653238784, 536870912)
    assert_equal(total, 20946793600)
    # Under the cap passes; exactly at a cap passes.
    check_gpu_budget(16756683904, 3653238784, 536870912, 24696061952)
    check_gpu_budget(16756683904, 3653238784, 536870912, 20946793600)
    print("gpu budget: planning total 20946793600 under cap 24696061952")


def test_budget_refusal_names() raises:
    var saw = 0
    # One byte over the planning total at that exact cap.
    try:
        check_gpu_budget(16756683905, 3653238784, 536870912, 20946793600)
        assert_true(False, "over-cap total must refuse")
    except e:
        assert_true(
            _refuses("gpu_budget_exceeded", String(e)),
            "over-cap refusal must name gpu_budget_exceeded",
        )
        saw += 1
    # A negative component refuses by the same name.
    try:
        _ = gpu_budget_total(-1, 0, 0)
        assert_true(False, "negative component must refuse")
    except e:
        assert_true(
            _refuses("gpu_budget_exceeded", String(e)),
            "negative refusal must name gpu_budget_exceeded",
        )
        saw += 1
    # A non-positive cap refuses by the same name.
    try:
        check_gpu_budget(0, 0, 536870912, 0)
        assert_true(False, "zero cap must refuse")
    except e:
        assert_true(
            _refuses("gpu_budget_exceeded", String(e)),
            "zero-cap refusal must name gpu_budget_exceeded",
        )
        saw += 1
    assert_equal(saw, 3)


# KV slot / layout arithmetic: windowed slot = position %
# window, global slot = position; capacity = window for
# windowed layers, the configured context for global layers;
# per-layer byte sizes and the recorded full total.
# Budget release / unreserve (Wave E, owed since Wave C): the
# ledger state is simulated here as plain ints driven through
# the SAME pure functions the runtime methods delegate to -
# check_gpu_budget is reserve_weights/reserve_kv's check and
# budget_release is unreserve_weights/unreserve_kv's whole
# arithmetic (the methods themselves need a DeviceContext, so
# they cannot be constructed on this device-less VM; their
# delegation is one line each in src/gpu/context.mojo).
# Scenario, at a small exact cap: reserve weights to near the
# cap, reserve KV, prove a second identical KV reservation
# refuses (the BEFORE behavior: with no release, a destroyed
# cache's bytes stayed counted, so the rebuild refused on a
# card/process that had the room - the Wave C layer proof
# reset rt.kv_bytes directly to work around exactly this),
# then release both reservations and prove the identical
# reservations succeed again, with the totals back at the
# scratch envelope alone.
def test_budget_release_reserve_cycle() raises:
    var cap = 1000
    var scratch = 100
    # Reserve to near the cap: weights 800 + kv 50 + scratch.
    check_gpu_budget(800, 0, scratch, cap)
    var weights = 800
    check_gpu_budget(weights, 50, scratch, cap)
    var kv = 50
    assert_equal(gpu_budget_total(weights, kv, scratch), 950)
    # BEFORE the release: a second KV build of the same size
    # in the same runtime refuses, even though the first cache
    # is conceptually destroyed - its bytes are still counted.
    var refused = False
    try:
        check_gpu_budget(weights, kv + 100, scratch, cap)
    except e:
        refused = _refuses("gpu_budget_exceeded", String(e))
    assert_true(refused, "second reserve without release must refuse")
    # Release both reservations (the teardown path).
    kv = budget_release(kv, 50, "kv")
    weights = budget_release(weights, 800, "weights")
    assert_equal(kv, 0)
    assert_equal(weights, 0)
    assert_equal(gpu_budget_total(weights, kv, scratch), scratch)
    # AFTER the release: the identical reservations succeed.
    check_gpu_budget(800, 0, scratch, cap)
    weights = 800
    check_gpu_budget(weights, 50, scratch, cap)
    kv = 50
    assert_equal(gpu_budget_total(weights, kv, scratch), 950)
    # Partial release frees exactly its amount.
    kv = budget_release(kv, 20, "kv")
    assert_equal(kv, 30)
    print("gpu budget release: reserve/release/reserve cycle exact")


def test_budget_release_refusal_names() raises:
    var saw = 0
    # Releasing more than the bucket holds (double release).
    try:
        _ = budget_release(50, 51, "kv")
        assert_true(False, "over-release must refuse")
    except e:
        assert_true(
            _refuses("gpu_budget_exceeded", String(e)),
            "over-release refusal must name gpu_budget_exceeded",
        )
        saw += 1
    # Releasing from an empty bucket (never reserved).
    try:
        _ = budget_release(0, 1, "weights")
        assert_true(False, "release from empty bucket must refuse")
    except e:
        assert_true(
            _refuses("gpu_budget_exceeded", String(e)),
            "empty-bucket refusal must name gpu_budget_exceeded",
        )
        saw += 1
    # A negative release amount.
    try:
        _ = budget_release(50, -1, "kv")
        assert_true(False, "negative release must refuse")
    except e:
        assert_true(
            _refuses("gpu_budget_exceeded", String(e)),
            "negative refusal must name gpu_budget_exceeded",
        )
        saw += 1
    # Exact and zero releases are legal.
    assert_equal(budget_release(50, 50, "kv"), 0)
    assert_equal(budget_release(50, 0, "kv"), 50)
    assert_equal(saw, 3)


def test_kv_layout_arithmetic() raises:
    var windowed = LayerKind.make_windowed(2048)
    var glob = LayerKind.make_global()
    assert_equal(kv_capacity_positions(windowed, 131072), 2048)
    assert_equal(kv_capacity_positions(glob, 131072), 131072)
    assert_equal(kv_capacity_positions(glob, 4096), 4096)
    assert_equal(kv_slot_of(windowed, 0), 0)
    assert_equal(kv_slot_of(windowed, 2047), 2047)
    assert_equal(kv_slot_of(windowed, 2048), 0)
    assert_equal(kv_slot_of(windowed, 2049), 1)
    assert_equal(kv_slot_of(glob, 12345), 12345)
    # One buffer (K or V): capacity x kv_heads x head_dim x 4 B.
    assert_equal(kv_buffer_bytes(2048, 2, 128), 2048 * 256 * 4)
    assert_equal(kv_buffer_bytes(2048, 2, 128), 2097152)
    assert_equal(kv_buffer_bytes(131072, 2, 128), 134217728)
    # Totals (K and V) for single-layer patterns.
    var one_window = List[LayerKind]()
    one_window.append(LayerKind.make_windowed(2048))
    assert_equal(kv_total_bytes(one_window, 2, 128, 131072), 4194304)
    var one_global = List[LayerKind]()
    one_global.append(LayerKind.make_global())
    assert_equal(kv_total_bytes(one_global, 2, 128, 131072), 268435456)
    print("gpu kv layout: slots, capacities, per-layer bytes exact")


def test_kv_total_glimmer_contract_number() raises:
    # (13 x 131,072 + 39 x 2,048) positions x 2,048 B = the
    # contract's 3,653,238,784 B at context 131,072.
    var expected = (13 * 131072 + 39 * 2048) * 2048
    assert_equal(expected, 3653238784)
    assert_equal(
        kv_total_bytes(glimmer_layer_pattern(), 2, 128, 131072), expected
    )
    print("gpu kv glimmer total: 3653238784 at context 131072")


# Refusal names on the pure KV paths: shape before order
# (B1's order), the global capacity refusal, and the
# construction-side names. Wave D: windowed wraps earn NO
# refusal (the ring evicts, B1-exact); the Wave C name
# gpu_kv_window_wrapped is retired from this cache.
def test_kv_refusal_names_pure() raises:
    var windowed = LayerKind.make_windowed(4)
    var glob = LayerKind.make_global()
    # Legal appends earn no refusal.
    assert_equal(kv_append_refusal(windowed, 4, 3, 3, 8, 8, 8), "")
    assert_equal(kv_append_refusal(glob, 16, 15, 15, 8, 8, 8), "")
    # Wrong row lengths (k or v) refuse shape, ahead of order.
    assert_equal(
        kv_append_refusal(windowed, 4, 0, 0, 7, 8, 8),
        "gpu_kv_shape_mismatch",
    )
    assert_equal(
        kv_append_refusal(windowed, 4, 0, 1, 8, 7, 8),
        "gpu_kv_shape_mismatch",
    )
    # Out-of-order append.
    assert_equal(
        kv_append_refusal(windowed, 4, 2, 3, 8, 8, 8),
        "gpu_kv_out_of_order_position",
    )
    # Wave D: a windowed append at/past position == window
    # (in order) wraps the ring and is LEGAL, B1-exact.
    assert_equal(kv_append_refusal(windowed, 4, 4, 4, 8, 8, 8), "")
    assert_equal(kv_append_refusal(windowed, 4, 9, 9, 8, 8, 8), "")
    var real_window = LayerKind.make_windowed(2048)
    assert_equal(
        kv_append_refusal(real_window, 2048, 2048, 2048, 256, 256, 256), ""
    )
    assert_equal(
        kv_append_refusal(real_window, 2048, 2049, 2049, 256, 256, 256), ""
    )
    assert_equal(
        kv_append_refusal(real_window, 2048, 2047, 2047, 256, 256, 256), ""
    )
    # A global append at the configured capacity refuses.
    assert_equal(
        kv_append_refusal(glob, 16, 16, 16, 8, 8, 8),
        "gpu_kv_capacity_exceeded",
    )
    # Construction-side names.
    var saw = 0
    var smuggled = LayerKind(False, 0)
    try:
        _ = kv_capacity_positions(smuggled, 131072)
        assert_true(False, "zero window must refuse")
    except e:
        assert_true(
            _refuses("gpu_kv_invalid_window", String(e)),
            "zero window must name gpu_kv_invalid_window",
        )
        saw += 1
    try:
        _ = kv_capacity_positions(glob, 0)
        assert_true(False, "zero context must refuse")
    except e:
        assert_true(
            _refuses("gpu_kv_zero_dimension", String(e)),
            "zero context must name gpu_kv_zero_dimension",
        )
        saw += 1
    var empty = List[LayerKind]()
    try:
        _ = kv_total_bytes(empty, 2, 128, 131072)
        assert_true(False, "empty pattern must refuse")
    except e:
        assert_true(
            _refuses("gpu_kv_zero_dimension", String(e)),
            "empty pattern must name gpu_kv_zero_dimension",
        )
        saw += 1
    var one = List[LayerKind]()
    one.append(LayerKind.make_global())
    try:
        _ = kv_total_bytes(one, 0, 128, 131072)
        assert_true(False, "zero kv heads must refuse")
    except e:
        assert_true(
            _refuses("gpu_kv_zero_dimension", String(e)),
            "zero heads must name gpu_kv_zero_dimension",
        )
        saw += 1
    assert_equal(saw, 4)


# Weights slot arithmetic: F32 slots reside as values x 4 B,
# every other slot as its raw byte length; meta indices
# mirror GlimmerModel.slot order (embd 0, output 1, then
# 2 + layer x 14 + suffix).
def test_weights_slot_arithmetic() raises:
    assert_equal(slot_resident_bytes(0, 999, 10), 40)
    assert_equal(slot_resident_bytes(12, 999, 10), 999)
    assert_equal(slot_resident_bytes(1, 999, 10), 999)
    assert_equal(slot_resident_bytes(13, 176, 0), 176)
    assert_equal(meta_index(0, 0), 2)
    assert_equal(meta_index(0, 10), 12)
    assert_equal(meta_index(3, 13), 57)
    assert_equal(meta_index(51, 13), 729)
    # 730 metas cover 730 of the file's 731 tensors; the 731st
    # (output_norm.weight) is the separate final-norm buffer.
    assert_equal(meta_index(51, 13) + 1, 730)
    print("gpu weights: slot bytes and meta indices exact")


# Visible-order arithmetic (Wave D): the gather's slot map.
# Window 4, first_visible 3 (positions 3,4,5,6 visible after
# appends 0..6): slots are 3,0,1,2. A first_visible landing
# on slot 0 needs no gather; globals never do.
def test_kv_visible_order_arithmetic() raises:
    var w4 = LayerKind.make_windowed(4)
    var glob = LayerKind.make_global()
    assert_equal(kv_visible_slot(w4, 3, 0), 3)
    assert_equal(kv_visible_slot(w4, 3, 1), 0)
    assert_equal(kv_visible_slot(w4, 3, 2), 1)
    assert_equal(kv_visible_slot(w4, 3, 3), 2)
    assert_equal(kv_visible_slot(w4, 0, 2), 2)
    assert_equal(kv_visible_slot(w4, 2048, 5), 1)
    var w2048 = LayerKind.make_windowed(2048)
    assert_equal(kv_visible_slot(w2048, 2048, 5), 5)
    assert_equal(kv_visible_slot(w2048, 1, 2047), 0)
    assert_equal(kv_visible_slot(glob, 0, 777), 777)
    assert_true(kv_gather_needed(w4, 3), "wrapped window must gather")
    assert_true(kv_gather_needed(w4, 2049), "first_visible 2049 % 4 = 1")
    assert_true(not kv_gather_needed(w4, 0), "pre-wrap needs no gather")
    assert_true(
        not kv_gather_needed(w4, 2048), "first_visible on slot 0: raw order"
    )
    assert_true(not kv_gather_needed(glob, 0), "global never gathers")
    print("gpu kv visible order: slot map + gather-needed exact")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
