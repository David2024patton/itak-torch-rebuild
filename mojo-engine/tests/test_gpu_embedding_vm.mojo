# test_gpu_embedding_vm.mojo - VM tests for Lane C2's device
# embedding pure parts (src/ops/gpu_embedding.mojo).
#
# The device dequant kernel is proven on the card by
# tests/server_proof_gpu_layer.mojo against
# GlimmerModel.embed_token; here, with no device: the
# super-block / row byte arithmetic for every served dtype
# and the named refusals for the unserved ones.
#
# For the coordinator: wire this file into run_tests.sh's
# TestSuite list (lane files are named in the lane report;
# run_tests.sh is coordinator-only).

from std.testing import TestSuite, assert_true, assert_equal

from ops.gpu_embedding import (
    embedding_sb_bytes,
    embedding_row_bytes,
    DT_Q4_K,
    DT_Q5_K,
    DT_Q6_K,
    DT_F16,
    DT_F32,
)


def test_sb_and_row_bytes() raises:
    assert_equal(embedding_sb_bytes(DT_Q4_K), 144, "Q4_K super-block")
    assert_equal(embedding_sb_bytes(DT_Q6_K), 210, "Q6_K super-block")
    assert_equal(embedding_sb_bytes(DT_Q5_K), 176, "Q5_K super-block")
    assert_equal(embedding_sb_bytes(DT_F16), 0, "F16 has no super-blocks")
    # Glimmer hidden 6656 = 26 super-blocks.
    assert_equal(embedding_row_bytes(DT_Q4_K, 6656), 26 * 144, "Q4_K row")
    assert_equal(embedding_row_bytes(DT_Q6_K, 6656), 26 * 210, "Q6_K row")
    assert_equal(embedding_row_bytes(DT_Q5_K, 6656), 26 * 176, "Q5_K row")
    assert_equal(embedding_row_bytes(DT_F16, 6656), 6656 * 2, "F16 row")
    assert_equal(embedding_row_bytes(DT_F32, 6656), 6656 * 4, "F32 row")


def test_refusals_by_name() raises:
    var refused = False
    try:
        _ = embedding_sb_bytes(7)
    except e:
        refused = String(e).find("gpu_embedding_unsupported_dtype") != -1
    assert_true(refused, "unknown dtype must refuse by name")
    refused = False
    try:
        _ = embedding_row_bytes(DT_Q4_K, 6650)
    except e:
        refused = String(e).find("gpu_embedding_bad_shape") != -1
    assert_true(refused, "non-block hidden must refuse by name")
    refused = False
    try:
        _ = embedding_row_bytes(DT_Q4_K, 0)
    except e:
        refused = String(e).find("gpu_embedding_bad_shape") != -1
    assert_true(refused, "zero hidden must refuse by name")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
