# test_gemm_cpu.mojo - CPU fixture tests for the Wave D
# prefill GEMM twins (src/kernels/gemm.mojo).
#
# Fixtures: SYNTHETIC, named as such, in the Wave A CPU test
# pattern (deterministic LCG bytes, tamed f16 super-scales).
# The GEMM CPU twins run the GEMV CPU twins per batch row by
# construction, so the property under test here is the BATCH
# LAYOUT and the shape discipline: for batch counts 1, 3 and 6
# (6 covers a partial structure-(3) tile of 4 + 2), every
# output row of every shape's GEMM twin must be BIT-IDENTICAL
# to the GEMV twin's row for the same input row, and the grid
# helpers must return the recorded workgroup counts. The GPU
# kernels are measured against these twins on the card under
# the Wave A bars (tests/server_proof_gemm.mojo).

from std.testing import TestSuite, assert_true, assert_equal

from kernels.gemm import (
    q4k_gemm_cpu,
    q6k_gemm_cpu,
    q5k_gemm_cpu,
    f32_gemm_cpu,
    gemm_plain_grid,
    gemm_tiled_grid,
    gemm_k1_grid,
    gemm_k1_tiled_grid,
    check_gemm_dims,
    BATCH_TILE,
)
from kernels.k1_gemv import k1_gemv_cpu
from kernels.q6k_gemv import q6k_gemv_cpu
from kernels.q5k_gemv import q5k_gemv_cpu
from kernels.f32_gemv import f32_gemv_cpu


def lcg_next(state: UInt32) -> UInt32:
    return state * 1664525 + 1013904223


def make_x(batch: Int, cols: Int, seed: UInt32) -> List[Float32]:
    var x = List[Float32](capacity=batch * cols)
    x.resize(batch * cols, Float32(0.0))
    var state = seed
    for i in range(len(x)):
        state = lcg_next(state)
        x[i] = Float32(Int((state >> 16) & 0xFF) - 128) / Float32(64.0)
    return x^


def slice_row(x: List[Float32], b: Int, cols: Int) -> List[Float32]:
    var out = List[Float32](capacity=cols)
    out.resize(cols, Float32(0.0))
    for i in range(cols):
        out[i] = x[b * cols + i]
    return out^


def assert_rows_exact(
    got: List[Float32], want_row: List[Float32], b: Int, rows: Int, label: String
) raises:
    for r in range(rows):
        assert_true(
            got[b * rows + r] == want_row[r],
            label + " row mismatch at batch " + String(b) + " row " + String(r),
        )


def test_gemm_grid_arithmetic() raises:
    assert_equal(gemm_plain_grid(8, 6), 48)
    assert_equal(gemm_plain_grid(6656, 1), 6656)
    assert_equal(gemm_tiled_grid(8, 6), 16)
    assert_equal(gemm_tiled_grid(8, 4), 8)
    assert_equal(gemm_tiled_grid(8, 1), 8)
    assert_equal(BATCH_TILE, 4)
    assert_equal(gemm_k1_grid(8, 6), 24)
    assert_equal(gemm_k1_tiled_grid(8, 6), 8)
    assert_equal(gemm_k1_tiled_grid(8, 256), 256)
    var refused = False
    try:
        _ = gemm_k1_grid(7, 2)
    except e:
        refused = String(e).find("gemm_bad_shape") != -1
    assert_true(refused, "odd k1 rows must refuse by name")


def test_q4k_gemm_matches_gemv_rows() raises:
    comptime ROWS = 8
    comptime COLS = 512
    comptime ROW_BYTES = (COLS // 256) * 144
    var w = List[UInt8](capacity=ROWS * ROW_BYTES)
    w.resize(ROWS * ROW_BYTES, UInt8(0))
    var state: UInt32 = 42
    for i in range(len(w)):
        state = lcg_next(state)
        w[i] = UInt8((state >> 16) & 0xFF)
    for row in range(ROWS):
        for sb in range(COLS // 256):
            var off = row * ROW_BYTES + sb * 144
            w[off] = 0x00
            w[off + 1] = 0x1C
            w[off + 2] = 0x00
            w[off + 3] = 0x18
    for batch in [1, 3, 6]:
        var x = make_x(batch, COLS, UInt32(1000 + batch))
        var got = q4k_gemm_cpu(w, x, batch, ROWS, COLS)
        assert_equal(len(got), batch * ROWS)
        for b in range(batch):
            var want = k1_gemv_cpu(w, slice_row(x, b, COLS), ROWS, COLS)
            assert_rows_exact(got, want, b, ROWS, "q4k gemm")


def test_q6k_gemm_matches_gemv_rows() raises:
    comptime ROWS = 8
    comptime COLS = 512
    comptime ROW_BYTES = (COLS // 256) * 210
    var w = List[UInt8](capacity=ROWS * ROW_BYTES)
    w.resize(ROWS * ROW_BYTES, UInt8(0))
    var state: UInt32 = 43
    for i in range(len(w)):
        state = lcg_next(state)
        w[i] = UInt8((state >> 16) & 0xFF)
    for row in range(ROWS):
        for sb in range(COLS // 256):
            var off = row * ROW_BYTES + sb * 210
            w[off + 208] = 0x00
            w[off + 209] = 0x1C
    for batch in [1, 3, 6]:
        var x = make_x(batch, COLS, UInt32(2000 + batch))
        var got = q6k_gemm_cpu(w, x, batch, ROWS, COLS)
        assert_equal(len(got), batch * ROWS)
        for b in range(batch):
            var want = q6k_gemv_cpu(w, slice_row(x, b, COLS), ROWS, COLS)
            assert_rows_exact(got, want, b, ROWS, "q6k gemm")


def test_q5k_gemm_matches_gemv_rows() raises:
    comptime ROWS = 8
    comptime COLS = 512
    comptime ROW_BYTES = (COLS // 256) * 176
    var w = List[UInt8](capacity=ROWS * ROW_BYTES)
    w.resize(ROWS * ROW_BYTES, UInt8(0))
    var state: UInt32 = 44
    for i in range(len(w)):
        state = lcg_next(state)
        w[i] = UInt8((state >> 16) & 0xFF)
    for row in range(ROWS):
        for sb in range(COLS // 256):
            var off = row * ROW_BYTES + sb * 176
            w[off] = 0x00
            w[off + 1] = 0x1C
            w[off + 2] = 0x00
            w[off + 3] = 0x18
    for batch in [1, 3, 6]:
        var x = make_x(batch, COLS, UInt32(3000 + batch))
        var got = q5k_gemm_cpu(w, x, batch, ROWS, COLS)
        assert_equal(len(got), batch * ROWS)
        for b in range(batch):
            var want = q5k_gemv_cpu(w, slice_row(x, b, COLS), ROWS, COLS)
            assert_rows_exact(got, want, b, ROWS, "q5k gemm")


def test_f32_gemm_matches_gemv_rows() raises:
    comptime ROWS = 8
    comptime COLS = 512
    var w = List[Float32](capacity=ROWS * COLS)
    w.resize(ROWS * COLS, Float32(0.0))
    var state: UInt32 = 45
    for i in range(len(w)):
        state = lcg_next(state)
        w[i] = Float32(Int((state >> 16) & 0xFF) - 128) / Float32(256.0)
    for batch in [1, 3, 6]:
        var x = make_x(batch, COLS, UInt32(4000 + batch))
        var got = f32_gemm_cpu(w, x, batch, ROWS, COLS)
        assert_equal(len(got), batch * ROWS)
        for b in range(batch):
            var want = f32_gemv_cpu(w, slice_row(x, b, COLS), ROWS, COLS)
            assert_rows_exact(got, want, b, ROWS, "f32 gemm")


def test_gemm_refuses_bad_shapes_by_name() raises:
    var x = List[Float32](capacity=100)
    x.resize(100, Float32(0.0))
    var w = List[UInt8](capacity=16)
    w.resize(16, UInt8(0))
    var refused = False
    try:
        _ = q6k_gemm_cpu(w, x, 0, 8, 512)
    except e:
        refused = String(e).find("gemm_bad_shape") != -1
    assert_true(refused, "zero batch must refuse by name")
    refused = False
    try:
        _ = q4k_gemm_cpu(w, x, 2, 8, 100)
    except e:
        refused = String(e).find("gemm_bad_shape") != -1
    assert_true(refused, "cols not a multiple of 256 must refuse by name")
    refused = False
    try:
        check_gemm_dims("probe", 1, -1, 256)
    except e:
        refused = String(e).find("gemm_bad_shape") != -1
    assert_true(refused, "negative rows must refuse by name")
    assert_equal(len(w), 16)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
