# server_proof_gpu_residency.mojo - Lane D4 (Wave D) server
# proof for activation residency. NOT a TestSuite file: run
# explicitly on a GPU host with
#   mojo run -I src tests/server_proof_gpu_residency.mojo
#
# Shape: the C2 layer harness exactly (tests/
# server_proof_gpu_layer.mojo): layer 0 (sliding) and layer 3
# (global) at position 0 on the real weights, input = the
# model's own embedding of token 13796 + plain rms_norm (the
# rung-2 construction), a FRESH DeviceKvCache per step.
#
# Protocol: one warmup round per path per layer (untimed),
# then 4 timed rounds INTERLEAVED hostchain/resident so card
# state (clocks, cache warmth) hits both paths alike. The
# timed region is the layer step only; the KV build is
# outside it (small context_len 64: a position-0 step needs
# capacity 1 on global layers and the window on sliding
# layers, so the proof's caches are honest and cheap).
#
# Correctness (every resident round, full row): the
# in-process CPU twin's layer_step row is checked against
# the COUNTED rung-2 record (sum 65.22733633872122 layer 0,
# 157.77914571017027 layer 3), then the resident GPU row
# against that CPU row at cosine >= 0.999999 and
# max rel <= 1e-4, plus the counted sums. The hostchain row
# is checked the same way in round 0 (the baseline must be
# the proven Wave C path, not a strawman).
#
# Reported: per-layer median step ms for both paths and the
# plain ratio. Exits nonzero on any correctness miss.

from std.math import sqrt
from std.sys import exit
from std.time import perf_counter_ns

from gpu.context import GpuRuntime
from gpu.device_kv import DeviceKvCache
from gpu.weights import DeviceWeights
from model.gpu_layer import gpu_layer_step_hostchain, gpu_layer_step_resident
from model.model import GlimmerModel
from ops.rms_norm import rms_norm_plain_cpu

comptime MODEL_PATH = (
    "/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
)
comptime GPU_BUDGET_CAP = 24696061952
comptime COUNTED_SUM_0 = 65.22733633872122
comptime COUNTED_SUM_3 = 157.77914571017027
comptime ROUNDS = 4


def _sum(row: List[Float32]) -> Float64:
    var s = Float64(0.0)
    for i in range(len(row)):
        s += Float64(row[i])
    return s


def _compare(
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
    return (dot / (sqrt(ng) * sqrt(ne)), max_abs / max_e)


def _median(mut values: List[Int]) -> Int:
    # Insertion sort, tiny lists.
    for i in range(1, len(values)):
        var j = i
        while j > 0 and values[j - 1] > values[j]:
            var tmp = values[j - 1]
            values[j - 1] = values[j]
            values[j] = tmp
            j -= 1
    return values[len(values) // 2]


def main() raises:
    var ok = True
    var cpu_model = GlimmerModel.open(MODEL_PATH)
    var twin = GlimmerModel.open(MODEL_PATH)
    var rt = GpuRuntime.create(GPU_BUDGET_CAP)
    var weights = DeviceWeights.build(rt, cpu_model)
    print(
        "RESIDENCY weights resident_bytes=" + String(weights.resident_bytes())
    )

    for target in [0, 3]:
        var x = cpu_model.embed_token(13796)
        x = rms_norm_plain_cpu(x, cpu_model.cfg.rms_eps)
        var cpu_out = twin.layer_step(x.copy(), target, 0)
        var counted = COUNTED_SUM_0 if target == 0 else COUNTED_SUM_3
        var cpu_sum = _sum(cpu_out)
        var cpu_sum_rel = (cpu_sum - counted) / counted
        if cpu_sum_rel < 0:
            cpu_sum_rel = -cpu_sum_rel
        print(
            "RESIDENCY layer="
            + String(target)
            + " cpu_sum="
            + String(cpu_sum)
            + " counted_sum="
            + String(counted)
            + " cpu_sum_rel="
            + String(cpu_sum_rel)
        )
        if cpu_sum_rel > 1.0e-4:
            print("RESIDENCY FAIL cpu twin disagrees with counted record")
            ok = False

        var host_times = List[Int]()
        var res_times = List[Int]()
        # Warmup round (untimed), then interleaved timed rounds.
        for round in range(ROUNDS + 1):
            var timed = round > 0
            rt.kv_bytes = 0
            var kv_h = DeviceKvCache.build(
                rt,
                cpu_model.cfg.kinds.copy(),
                cpu_model.cfg.kv_heads,
                cpu_model.cfg.head_dim,
                64,
            )
            var t0 = perf_counter_ns()
            var host_out = gpu_layer_step_hostchain(
                rt, weights, kv_h, cpu_model, x.copy(), target, 0
            )
            var host_ns = Int(perf_counter_ns() - t0)
            var host_ms = host_ns // 1_000_000
            rt.kv_bytes = 0
            var kv_r = DeviceKvCache.build(
                rt,
                cpu_model.cfg.kinds.copy(),
                cpu_model.cfg.kv_heads,
                cpu_model.cfg.head_dim,
                64,
            )
            var t1 = perf_counter_ns()
            var res_out = gpu_layer_step_resident(
                rt, weights, kv_r, cpu_model, x.copy(), target, 0
            )
            var res_ns = Int(perf_counter_ns() - t1)
            var res_ms = res_ns // 1_000_000
            if timed:
                host_times.append(host_ns)
                res_times.append(res_ns)
            # Correctness: resident every round, hostchain in
            # the warmup round (baseline sanity).
            var cmp = _compare(res_out, cpu_out)
            var res_sum = _sum(res_out)
            var res_sum_rel = (res_sum - counted) / counted
            if res_sum_rel < 0:
                res_sum_rel = -res_sum_rel
            var rok = (
                cmp[0] >= 0.999999
                and cmp[1] <= 1.0e-4
                and res_sum_rel <= 1.0e-4
            )
            ok = ok and rok
            print(
                "RESIDENCY layer="
                + String(target)
                + " round="
                + String(round)
                + " hostchain_ms="
                + String(host_ms)
                + " resident_ms="
                + String(res_ms)
                + " hostchain_us="
                + String(host_ns // 1000)
                + " resident_us="
                + String(res_ns // 1000)
                + " resident_cosine="
                + String(cmp[0])
                + " resident_max_rel="
                + String(cmp[1])
                + " resident_sum="
                + String(res_sum)
                + " resident_sum_rel="
                + String(res_sum_rel)
                + " ok="
                + String(1 if rok else 0)
            )
            if round == 0:
                var hcmp = _compare(host_out, cpu_out)
                var hok = hcmp[0] >= 0.999999 and hcmp[1] <= 1.0e-4
                ok = ok and hok
                print(
                    "RESIDENCY layer="
                    + String(target)
                    + " hostchain_cosine="
                    + String(hcmp[0])
                    + " hostchain_max_rel="
                    + String(hcmp[1])
                    + " ok="
                    + String(1 if hok else 0)
                )
        var host_med = _median(host_times)
        var res_med = _median(res_times)
        var ratio = Float64(host_med) / Float64(
            res_med
        ) if res_med > 0 else Float64(0)
        print(
            "RESIDENCY layer="
            + String(target)
            + " hostchain_median_ms="
            + String(host_med // 1_000_000)
            + " resident_median_ms="
            + String(res_med // 1_000_000)
            + " hostchain_median_us="
            + String(host_med // 1000)
            + " resident_median_us="
            + String(res_med // 1000)
            + " speedup_x="
            + String(ratio)
        )

    if ok:
        print("RESIDENCY PROOF PASS")
    else:
        print("RESIDENCY PROOF FAIL")
        exit(1)
