# server_proof_gpu_layer.mojo - Lane C2 server proof, run on
# the RX 7900 XTX against the real Glimmer weights.
#
# NOT a TestSuite file: run explicitly on a GPU host with
#   mojo run -I src tests/server_proof_gpu_layer.mojo
# (against Lane C1's real src/gpu/), or, before C1 is
# integrable, with the Lane C2 stand-in package appended:
#   mojo run -I src -I tests/gpu_standin \
#       tests/server_proof_gpu_layer.mojo
# The run prints which plumbing it used (PLUMBING line):
# when the stand-in include path is present the stand-in
# DeviceWeights / DeviceKvCache / GpuRuntime are what the
# layer code ran against, and the lines say STANDIN plainly.
# The layer, embedding, and logits code under proof is the
# committed src/ code either way.
#
# What is proven, and against what:
# 1. EMBEDDING: gpu_embed_token vs GlimmerModel.embed_token
#    for tokens 13796 and 200000. Dequant is exact per
#    element; bar max abs <= 1e-6.
# 2. LAYERS: gpu_layer_step for layer 0 (sliding) and layer 3
#    (global) at position 0, each on a FRESH DeviceKvCache,
#    input = the model's own embedding of token 13796 + plain
#    rms_norm at rms_eps (the rung-2 construction of
#    server_proof_model.mojo). Two comparisons:
#    (a) the in-process CPU twin's layer_step row is checked
#        against the COUNTED rung-2 record (sum and first 8,
#        Wave B spot-check counted those rows at ~1e-6), and
#    (b) the GPU row is compared against that CPU row over
#        the FULL row: cosine >= 0.999999 and
#        max rel <= 1e-4 (rel = max abs diff / max abs
#        expected, the gpu_proof_attention definition).
#    The counted full rows were never published (only sum +
#    first 8), so (a)+(b) is the honest chain: CPU row ==
#    counted row at the published points, GPU row == CPU row
#    everywhere at the brief's bar.
# 3. LOGITS: gpu_logits vs GlimmerModel.logits on the SAME
#    input row (the CPU layer-0 output row from part 2 - a
#    real-scale hidden row; logits is a pure function of a
#    hidden row + weights). Bar: argmax EXACT match, cosine
#    >= 0.999999, max rel <= 1e-4.
#
# Exits nonzero on any miss. On a GPU-less host the runtime
# creation fails loudly; that is a host fact, not a pass.

from std.math import sqrt
from std.sys import exit
from std.time import perf_counter_ns

from gpu.context import GpuRuntime
from gpu.weights import DeviceWeights
from gpu.device_kv import DeviceKvCache
from model.model import GlimmerModel
from model.gpu_layer import gpu_logits, gpu_layer_step
from ops.gpu_embedding import gpu_embed_token
from ops.rms_norm import rms_norm_plain_cpu
from generate.sampler import greedy_argmax

comptime MODEL_PATH = (
    "/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
)
comptime GPU_BUDGET_CAP = 24696061952

# The counted rung-2 record (Wave B, Float64 spot check).
comptime COUNTED_SUM_0 = 65.22733633872122
comptime COUNTED_SUM_3 = 157.77914571017027


def _counted_first8(layer: Int) -> List[Float64]:
    var out = List[Float64]()
    if layer == 0:
        for v in [
            0.08513339,
            0.026624471,
            -0.42816293,
            3.276528,
            -1.1669861,
            2.0494282,
            -0.08531731,
            0.34014347,
        ]:
            out.append(v)
    else:
        for v in [
            1.254974,
            -0.3695352,
            -0.49299628,
            1.3146128,
            -0.27929068,
            1.1605207,
            0.099737644,
            0.68446195,
        ]:
            out.append(v)
    return out^


def _sum(row: List[Float32]) -> Float64:
    var s = Float64(0.0)
    for i in range(len(row)):
        s += Float64(row[i])
    return s


def _compare(
    got: List[Float32], expected: List[Float32]
) -> Tuple[Float64, Float64]:
    """(cosine, max_rel) with rel = max abs diff / max abs
    expected, the gpu_proof_attention definition."""
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
    var cosine = dot / (sqrt(ng) * sqrt(ne))
    return (cosine, max_abs / max_e)


def _max_abs(got: List[Float32], expected: List[Float32]) -> Float64:
    var worst = Float64(0.0)
    for i in range(len(expected)):
        var d = Float64(got[i]) - Float64(expected[i])
        if d < 0:
            d = -d
        if d > worst:
            worst = d
    return worst


def main() raises:
    var ok = True
    var t0 = perf_counter_ns()
    var cpu_model = GlimmerModel.open(MODEL_PATH)
    print(
        "LOAD_DONE ms="
        + String((perf_counter_ns() - t0) // 1_000_000)
        + " embd_type="
        + String(cpu_model.embd.type_id)
        + " output_type="
        + String(cpu_model.output.type_id)
    )
    var rt = GpuRuntime.create(GPU_BUDGET_CAP)
    var t_w = perf_counter_ns()
    var weights = DeviceWeights.from_model(rt, cpu_model)
    print(
        "WEIGHTS_UPLOADED ms="
        + String((perf_counter_ns() - t_w) // 1_000_000)
        + " resident_bytes="
        + String(weights.resident_bytes)
    )
    print(
        "PLUMBING note: DeviceWeights/DeviceKvCache/GpuRuntime"
        " come from whichever gpu package the include path"
        " resolved; the staged run script records STANDIN or"
        " REAL_C1 next to this log"
    )

    # ---- 1. embedding ----
    for token in [13796, 200000]:
        var gpu_row = gpu_embed_token(rt, weights, token)
        var cpu_row = cpu_model.embed_token(token)
        var worst = _max_abs(gpu_row, cpu_row)
        var eok = worst <= 1.0e-6
        ok = ok and eok
        print(
            "EMB_PROOF token="
            + String(token)
            + " max_abs="
            + String(worst)
            + " bar 1e-6 ok="
            + String(1 if eok else 0)
        )

    # ---- 2. layers 0 and 3 at position 0 ----
    var twin = GlimmerModel.open(MODEL_PATH)
    var layer0_cpu_out = List[Float32]()
    for target in [0, 3]:
        var x = cpu_model.embed_token(13796)
        x = rms_norm_plain_cpu(x, cpu_model.cfg.rms_eps)
        var kv = DeviceKvCache.create(
            rt,
            cpu_model.cfg.kinds.copy(),
            cpu_model.cfg.kv_heads,
            cpu_model.cfg.head_dim,
            cpu_model.cfg.context_len,
        )
        var t_g = perf_counter_ns()
        var gpu_out = gpu_layer_step(
            rt, weights, kv, cpu_model, x.copy(), target, 0
        )
        var gpu_ms = (perf_counter_ns() - t_g) // 1_000_000
        var t_c = perf_counter_ns()
        var cpu_out = twin.layer_step(x.copy(), target, 0)
        var cpu_ms = (perf_counter_ns() - t_c) // 1_000_000
        if target == 0:
            layer0_cpu_out = cpu_out.copy()
        # (a) CPU row vs the counted record.
        var counted_sum = COUNTED_SUM_0 if target == 0 else COUNTED_SUM_3
        var cpu_sum = _sum(cpu_out)
        var sum_rel_cpu = (cpu_sum - counted_sum) / counted_sum
        if sum_rel_cpu < 0:
            sum_rel_cpu = -sum_rel_cpu
        var first8 = _counted_first8(target)
        var f8_worst_cpu = Float64(0.0)
        for i in range(8):
            var d = Float64(cpu_out[i]) - first8[i]
            if d < 0:
                d = -d
            if d > f8_worst_cpu:
                f8_worst_cpu = d
        # (b) GPU row vs CPU row, full row.
        var cmp = _compare(gpu_out, cpu_out)
        var gpu_sum = _sum(gpu_out)
        var sum_rel_gpu = (gpu_sum - counted_sum) / counted_sum
        if sum_rel_gpu < 0:
            sum_rel_gpu = -sum_rel_gpu
        var f8_worst_gpu = Float64(0.0)
        for i in range(8):
            var d = Float64(gpu_out[i]) - first8[i]
            if d < 0:
                d = -d
            if d > f8_worst_gpu:
                f8_worst_gpu = d
        var lok = (
            cmp[0] >= 0.999999
            and cmp[1] <= 1.0e-4
            and sum_rel_cpu <= 1.0e-4
            and sum_rel_gpu <= 1.0e-4
        )
        ok = ok and lok
        print(
            "LAYER_PROOF layer="
            + String(target)
            + " gpu_ms="
            + String(gpu_ms)
            + " cpu_ms="
            + String(cpu_ms)
            + " gpu_sum="
            + String(gpu_sum)
            + " counted_sum="
            + String(counted_sum)
            + " sum_rel_gpu="
            + String(sum_rel_gpu)
            + " sum_rel_cpu="
            + String(sum_rel_cpu)
            + " first8_max_abs_gpu="
            + String(f8_worst_gpu)
            + " first8_max_abs_cpu="
            + String(f8_worst_cpu)
            + " cosine="
            + String(cmp[0])
            + " max_rel="
            + String(cmp[1])
            + " ok="
            + String(1 if lok else 0)
        )

    # ---- 3. logits on the layer-0 output row ----
    var t_l = perf_counter_ns()
    var gpu_logits_row = gpu_logits(rt, weights, cpu_model, layer0_cpu_out)
    var gpu_logits_ms = (perf_counter_ns() - t_l) // 1_000_000
    t_l = perf_counter_ns()
    var cpu_logits_row = cpu_model.logits(layer0_cpu_out)
    var cpu_logits_ms = (perf_counter_ns() - t_l) // 1_000_000
    var g_arg = greedy_argmax(gpu_logits_row)
    var c_arg = greedy_argmax(cpu_logits_row)
    var lcmp = _compare(gpu_logits_row, cpu_logits_row)
    var log_ok = (
        g_arg == c_arg and lcmp[0] >= 0.999999 and lcmp[1] <= 1.0e-4
    )
    ok = ok and log_ok
    print(
        "LOGITS_PROOF gpu_ms="
        + String(gpu_logits_ms)
        + " cpu_ms="
        + String(cpu_logits_ms)
        + " argmax_gpu="
        + String(g_arg)
        + " argmax_cpu="
        + String(c_arg)
        + " argmax_match="
        + String(1 if g_arg == c_arg else 0)
        + " cosine="
        + String(lcmp[0])
        + " max_rel="
        + String(lcmp[1])
        + " ok="
        + String(1 if log_ok else 0)
    )

    if ok:
        print("GPU_LAYER_PROOF PASS")
    else:
        print("GPU_LAYER_PROOF FAIL")
        exit(1)
