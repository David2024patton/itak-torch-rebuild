# test_gpu_layer_vm.mojo - VM tests for Lane C2's GPU layer
# assembly pure parts (src/model/gpu_layer.mojo).
#
# The device launches themselves are proven on the card by
# tests/server_proof_gpu_layer.mojo; what runs here, with no
# device, is everything host-side that can be wrong before a
# launch: the projection/norm shape validators and their
# named refusals, the rope launch-shape arithmetic (the
# kernel has no bounds guard, so the launch must cover the
# pair count EXACTLY), and the host twin of the new gate
# assembly kernel against the CPU model's sigmoid shape.
#
# For the coordinator: wire this file into run_tests.sh's
# TestSuite list (lane files are named in the lane report;
# run_tests.sh is coordinator-only).

from std.testing import TestSuite, assert_true, assert_equal
from std.math import exp

from model.gpu_layer import (
    gpu_layer_check_proj,
    gpu_layer_check_norm,
    rope_launch_shape,
    attn_gate_cpu,
    DT_Q4_K,
    DT_Q6_K,
    DT_F32,
)


def test_check_proj_accepts_glimmer_shapes() raises:
    # Q4_K: rows even, cols a 256 multiple.
    gpu_layer_check_proj("attn_q", DT_Q4_K, 4096, 6656, 6656)
    gpu_layer_check_proj("attn_k", DT_Q4_K, 256, 6656, 6656)
    # Q6_K / F32 shapes.
    gpu_layer_check_proj("ffn_down", DT_Q6_K, 6656, 19968, 19968)
    gpu_layer_check_proj("normlike", DT_F32, 6656, 6656, 6656)


def test_check_proj_refuses_by_name() raises:
    var names = List[String]()
    try:
        gpu_layer_check_proj("x", DT_Q4_K, 4096, 6656, 123)
        names.append("none")
    except e:
        names.append(String(e))
    try:
        gpu_layer_check_proj("x", 99, 4096, 6656, 6656)
        names.append("none")
    except e:
        names.append(String(e))
    try:
        gpu_layer_check_proj("x", DT_Q4_K, 4095, 6656, 6656)
        names.append("none")
    except e:
        names.append(String(e))
    try:
        gpu_layer_check_proj("x", DT_Q6_K, 4096, 6650, 6650)
        names.append("none")
    except e:
        names.append(String(e))
    assert_true(
        names[0].find("gpu_layer_bad_shape") != -1, "input length refusal misnamed"
    )
    assert_true(
        names[1].find("gpu_layer_unsupported_dtype") != -1, "dtype refusal misnamed"
    )
    assert_true(
        names[2].find("gpu_layer_bad_shape") != -1, "odd Q4 rows refusal misnamed"
    )
    assert_true(
        names[3].find("gpu_layer_bad_shape") != -1, "cols multiple refusal misnamed"
    )


def test_check_norm() raises:
    gpu_layer_check_norm(6656, 6656, 6656)
    gpu_layer_check_norm(4096, 128, 128)
    var refused = False
    try:
        gpu_layer_check_norm(4096, 6656, 128)
    except e:
        refused = String(e).find("gpu_layer_bad_shape") != -1
    assert_true(refused, "norm weight width must refuse by name")
    refused = False
    try:
        gpu_layer_check_norm(4100, 128, 128)
    except e:
        refused = String(e).find("gpu_layer_bad_shape") != -1
    assert_true(refused, "non-whole rows must refuse by name")


def test_rope_launch_shape_exact_coverage() raises:
    # Glimmer Q row: 32 heads x 64 pairs = 2048 -> (8, 256).
    var q = rope_launch_shape(2048)
    assert_equal(q[0] * q[1], 2048, "Q rope launch must cover exactly")
    assert_equal(q[1], 256, "Q rope block")
    # Glimmer K row: 2 heads x 64 pairs = 128 -> (1, 128).
    var k = rope_launch_shape(128)
    assert_equal(k[0] * k[1], 128, "K rope launch must cover exactly")
    assert_equal(k[0], 1, "K rope grid")
    # A pair count no exact launch covers refuses by name.
    var refused = False
    try:
        _ = rope_launch_shape(1031)
    except e:
        refused = String(e).find("gpu_layer_bad_shape") != -1
    assert_true(refused, "uncoverable rope shape must refuse by name")
    refused = False
    try:
        _ = rope_launch_shape(0)
    except e:
        refused = String(e).find("gpu_layer_bad_shape") != -1
    assert_true(refused, "zero pairs must refuse by name")


def test_attn_gate_cpu_matches_model_sigmoid() raises:
    var a = List[Float32]()
    var g = List[Float32]()
    for v in [Float32(-30.0), Float32(-1.5), Float32(0.0), Float32(0.75), Float32(25.0)]:
        a.append(Float32(2.0))
        g.append(v)
    var got = attn_gate_cpu(a, g)
    for i in range(len(g)):
        var gv = g[i]
        var sig: Float32
        if gv >= Float32(0.0):
            sig = Float32(1.0) / (Float32(1.0) + exp(-gv))
        else:
            var e = exp(gv)
            sig = e / (Float32(1.0) + e)
        var expect = Float32(2.0) * sig
        var d = got[i] - expect
        if d < 0:
            d = -d
        assert_true(d <= Float32(1.0e-6), "gate assembly element off")
    var refused = False
    try:
        _ = attn_gate_cpu(a, List[Float32]())
    except e:
        refused = String(e).find("gpu_layer_bad_shape") != -1
    assert_true(refused, "gate length mismatch must refuse by name")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
