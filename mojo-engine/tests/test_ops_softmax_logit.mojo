# test_ops_softmax_logit.mojo - Lane A4 proof for softmax and
# the (separate) final-logit transform.
#
# Softmax reference: Float64 max-subtracted softmax in this
# file. Tolerance max abs <= 1e-6: probabilities lie in [0, 1],
# both sides evaluate exp in Float64 and the twin stores f32
# (rounding about 6e-8), so 1e-6 is tight with headroom. Each
# row must also sum to 1 within 1e-6.
# Final-logit reference: Float64 scale-then-tanh-softcap in this
# file, tolerance max abs <= 1e-5 (f32 tanh chain), PLUS the
# Rust model's own recorded case (model.rs
# softcap_scale_matches_the_formula): [0, 20, -20, 1000] at
# cap 20, scale 0.5 gives [0, 9.242343, -9.242343, ~20], a value
# that distinguishes scale-first from cap-first.

from std.math import exp, tanh
from std.testing import TestSuite, assert_true

from config import DEFAULT_LOGIT_SCALE, DEFAULT_LOGIT_SOFTCAP
from ops.final_logit import final_logit_cpu
from ops.softmax import softmax_cpu


def test_softmax_rows() raises:
    comptime W = 16
    var x = List[Float32](capacity=3 * W)
    x.resize(3 * W, Float32(0.0))
    for i in range(len(x)):
        x[i] = Float32(((i * 11) % 23) - 11) / Float32(2.0)
    # Row 2 is the stability case: large scores that would
    # overflow a non-max-subtracted exp.
    for i in range(W):
        x[2 * W + i] = Float32(1000.0) + Float32(i % 5)
    var got = softmax_cpu(x, W)
    var worst = Float64(0.0)
    for r in range(3):
        var mx = Float64(x[r * W])
        for i in range(1, W):
            if Float64(x[r * W + i]) > mx:
                mx = Float64(x[r * W + i])
        var sum = Float64(0.0)
        for i in range(W):
            sum += exp(Float64(x[r * W + i]) - mx)
        var row_sum = Float64(0.0)
        for i in range(W):
            var expected = exp(Float64(x[r * W + i]) - mx) / sum
            row_sum += Float64(got[r * W + i])
            var d = Float64(got[r * W + i]) - expected
            if d < 0:
                d = -d
            if d > worst:
                worst = d
        var ds = row_sum - Float64(1.0)
        if ds < 0:
            ds = -ds
        assert_true(ds <= 1.0e-6, "softmax row must sum to 1")
    print("softmax 3x16 (incl 1000-row): max_abs", worst)
    assert_true(worst <= 1.0e-6, "softmax above bar")


def test_softmax_refusal_by_name() raises:
    var x = List[Float32](capacity=5)
    x.resize(5, Float32(1.0))
    var refused = False
    try:
        _ = softmax_cpu(x, 4)
    except e:
        refused = String(e).find("softmax_cpu_bad_shape") != -1
    assert_true(refused, "ragged row must refuse by name")


def test_final_logit_rust_recorded_case() raises:
    var x = List[Float32](capacity=4)
    x.resize(4, Float32(0.0))
    x[1] = 20.0
    x[2] = -20.0
    x[3] = 1000.0
    var got = final_logit_cpu(x, Float32(0.5), Float32(20.0))
    assert_true(got[0] == Float32(0.0), "zero stays zero")
    var d = Float64(got[1]) - 9.242343
    if d < 0:
        d = -d
    print("final_logit rust recorded: |got-9.242343|", d)
    assert_true(d <= 1.0e-5, "scale-first value must match Rust record")
    assert_true(got[2] == -got[1], "softcap is odd")
    var dsat = Float64(got[3]) - 20.0
    if dsat < 0:
        dsat = -dsat
    assert_true(dsat <= 1.0e-4, "large logit saturates at the cap")


def test_final_logit_glimmer_values_and_no_cap() raises:
    comptime N = 64
    var x = List[Float32](capacity=N)
    x.resize(N, Float32(0.0))
    for i in range(N):
        x[i] = Float32(((i * 7) % 29) - 14) * Float32(3.0)
    var got = final_logit_cpu(x, DEFAULT_LOGIT_SCALE, DEFAULT_LOGIT_SOFTCAP)
    var worst = Float64(0.0)
    for i in range(N):
        var z = Float64(x[i]) * Float64(DEFAULT_LOGIT_SCALE)
        var expected = Float64(DEFAULT_LOGIT_SOFTCAP) * tanh(
            z / Float64(DEFAULT_LOGIT_SOFTCAP)
        )
        var d = Float64(got[i]) - expected
        if d < 0:
            d = -d
        if d > worst:
            worst = d
        assert_true(
            got[i] <= DEFAULT_LOGIT_SOFTCAP
            and got[i] >= -DEFAULT_LOGIT_SOFTCAP,
            "softcap bounds every logit",
        )
    print("final_logit glimmer scale/cap 64: max_abs", worst)
    assert_true(worst <= 1.0e-5, "final logit above bar")
    # Cap 0 = no cap (Rust softcap_scale): [2, -4] * 0.25 exact.
    var v = List[Float32](capacity=2)
    v.resize(2, Float32(0.0))
    v[0] = 2.0
    v[1] = -4.0
    var plain = final_logit_cpu(v, Float32(0.25), Float32(0.0))
    assert_true(
        plain[0] == Float32(0.5) and plain[1] == Float32(-1.0), "no-cap"
    )
    var saw = 0
    try:
        _ = final_logit_cpu(v, Float32(0.0) / Float32(0.0), Float32(20.0))
    except e:
        if String(e).find("final_logit_cpu_bad_scale") != -1:
            saw += 1
    try:
        _ = final_logit_cpu(v, Float32(0.5), Float32(-1.0))
    except e:
        if String(e).find("final_logit_cpu_bad_cap") != -1:
            saw += 1
    assert_true(saw == 2, "bad scale and bad cap refuse by name")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
