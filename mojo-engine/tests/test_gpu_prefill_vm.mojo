# test_gpu_prefill_vm.mojo - VM tests for Wave D Lane D1's
# batched-prefill pure parts (src/model/gpu_layer_batch.mojo
# shape validators and row slicing, src/model/gpu_model.mojo's
# prefill row helpers).
#
# The device launches themselves are proven on the card by
# tests/server_proof_gemm.mojo, tests/server_proof_prefill.mojo
# and the golden window; what runs here, with no device, is
# the host-side shape discipline that can be wrong before a
# launch: the batched projection validator and its named
# refusals (including the F16 batch refusal), and the flat
# batch row slice/place round-trip the batch step is built on.

from std.testing import TestSuite, assert_true, assert_equal

from model.gpu_layer import DT_Q4_K, DT_Q6_K, DT_Q5_K, DT_F32, DT_F16
from model.gpu_layer_batch import (
    gpu_layer_check_proj_batch,
    batch_row_slice,
    batch_row_place,
)


def test_check_proj_batch_accepts_glimmer_shapes() raises:
    gpu_layer_check_proj_batch("attn_q", DT_Q4_K, 4096, 6656, 6, 6 * 6656)
    gpu_layer_check_proj_batch("ffn_down", DT_Q6_K, 6656, 19968, 256, 256 * 19968)
    gpu_layer_check_proj_batch("output", DT_Q5_K, 202048, 6656, 1, 6656)
    gpu_layer_check_proj_batch("normlike", DT_F32, 6656, 6656, 64, 64 * 6656)


def test_check_proj_batch_refuses_by_name() raises:
    var names = List[String]()
    try:
        gpu_layer_check_proj_batch("x", DT_Q4_K, 4096, 6656, 6, 123)
        names.append("none")
    except e:
        names.append(String(e))
    try:
        gpu_layer_check_proj_batch("x", DT_F16, 4096, 6656, 6, 6 * 6656)
        names.append("none")
    except e:
        names.append(String(e))
    try:
        gpu_layer_check_proj_batch("x", DT_Q4_K, 4095, 6656, 6, 6 * 6656)
        names.append("none")
    except e:
        names.append(String(e))
    try:
        gpu_layer_check_proj_batch("x", DT_Q6_K, 4096, 6656, 0, 0)
        names.append("none")
    except e:
        names.append(String(e))
    assert_true(
        names[0].find("gpu_layer_bad_shape") != -1, "flat length refusal misnamed"
    )
    assert_true(
        names[1].find("gpu_layer_unsupported_dtype") != -1,
        "F16 batch refusal misnamed",
    )
    assert_true(
        names[2].find("gpu_layer_bad_shape") != -1, "odd Q4 rows refusal misnamed"
    )
    assert_true(
        names[3].find("gpu_layer_bad_shape") != -1, "zero batch refusal misnamed"
    )


def test_batch_row_slice_place_roundtrip() raises:
    var width = 4
    var batch = 3
    var x = List[Float32](capacity=batch * width)
    x.resize(batch * width, Float32(0.0))
    for i in range(len(x)):
        x[i] = Float32(i)
    var row = batch_row_slice(x, 1, width)
    assert_equal(len(row), width)
    for j in range(width):
        assert_true(row[j] == Float32(width + j), "slice row wrong")
    var newrow = List[Float32](capacity=width)
    newrow.resize(width, Float32(-1.0))
    batch_row_place(x, 2, width, newrow)
    for j in range(width):
        assert_true(x[2 * width + j] == Float32(-1.0), "place row wrong")
        assert_true(x[j] == Float32(j), "place touched another row")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
