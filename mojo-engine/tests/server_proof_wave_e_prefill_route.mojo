# server_proof_wave_e_prefill_route.mojo - Wave E Lane E1
# prefill routing measurement: batch route vs per-row
# resident route on the real model, interleaved, at prompt
# lengths 6..1022, plus the 1,022-token integrated-arms
# comparison. NOT a TestSuite file: run explicitly on a GPU
# host with
#   mojo run -I src tests/server_proof_wave_e_prefill_route.mojo
#
# The two arms are the SAME GpuGlimmerModel code on the
# SAME staged commit; the only differences are the
# constructor knobs handed in: the BATCH arm gets
# prefill_gemm_min_batch=2 and prefill_batch_min_tokens=1
# (every measured prompt takes the batched route), the
# ROWS arm gets prefill_gemm_min_batch=1,000,000 (no
# prompt reaches the GEMM minimum, so prefill is the
# per-row resident step loop - the Wave D A/B control).
# Arms rotate across two rounds (BATCH/ROWS, then
# ROWS/BATCH), each arm on a FRESH model (fresh KV,
# positions 0..n-1), load time excluded and printed.
#
# Correctness inside the measurement: each arm's last
# hidden row is compared against the other arm's
# (k1_check metrics) and the finished-logits argmax must
# agree exactly. Verdict bar (the Wave C/D full-forward
# class): cosine >= 0.999999, rel <= 1e-4, argmax exact.
#
# The routing output: per length, both arms' min-of-rounds
# ms; WAVE_E_CROSSOVER names the smallest measured length
# at which the batch arm's min beats the per-row arm's
# min and stays ahead at every larger measured length
# (0 when the batch arm never takes the lead).
#
# Prompt: deterministic pseudo-random token ids (LCG, ids
# in [0, 200000)), as server_proof_prefill - this measures
# the forward, not the tokenizer. Context 4096 as there.

from std.math import sqrt
from std.sys import exit
from std.time import perf_counter_ns

from config import EngineConfig
from generate.sampler import greedy_argmax
from model.gpu_model import GpuGlimmerModel

comptime PROOF_CONTEXT_LEN = 4096
comptime BATCH_GEMM_MIN = 2
comptime ROWS_GEMM_MIN = 1000000
comptime ROUTE_MIN_TOKENS = 1


def lcg_next(state: UInt32) -> UInt32:
    return state * 1664525 + 1013904223


def make_tokens(n: Int) -> List[Int]:
    var out = List[Int](capacity=n)
    var state: UInt32 = 987654321
    for _ in range(n):
        state = lcg_next(state)
        out.append(Int((state >> 8) % 200000))
    return out^


def arm_prefill(
    path: String,
    budget: Int,
    tiled_min: Int,
    gemm_min: Int,
    tokens: List[Int],
) raises -> Tuple[Int, Int, List[Float32], Int]:
    """One fresh model, one timed forward_rows over the
    prompt. Returns (load_ms, prefill_ms, last_hidden,
    argmax)."""
    var t0 = perf_counter_ns()
    var model = GpuGlimmerModel.open(
        path, PROOF_CONTEXT_LEN, budget, gemm_min, tiled_min, ROUTE_MIN_TOKENS
    )
    var load_ms = Int((perf_counter_ns() - t0) // 1_000_000)
    var t1 = perf_counter_ns()
    var hidden = model.forward_rows(tokens.copy())
    var prefill_ms = Int((perf_counter_ns() - t1) // 1_000_000)
    var logits = model.logits(hidden.copy())
    var argmax = greedy_argmax(logits)
    return (load_ms, prefill_ms, hidden^, argmax)


def compare_rows(a: List[Float32], b: List[Float32]) -> Tuple[Float64, Float64]:
    var max_abs = Float64(0.0)
    var scale = Float64(0.0)
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(b)):
        var x = Float64(a[i])
        var y = Float64(b[i])
        var d = x - y
        if d < 0:
            d = -d
        if d > max_abs:
            max_abs = d
        var ay = y
        if ay < 0:
            ay = -ay
        if ay > scale:
            scale = ay
        dot += x * y
        na += x * x
        nb += y * y
    var denom = scale if scale > 1.0 else 1.0
    return (max_abs / denom, dot / (sqrt(na) * sqrt(nb)))


def run() raises:
    var config = EngineConfig.load()
    var path = config.model_path
    print("WAVE_E_ROUTE_PROOF model_path", path)
    var sizes = List[Int]()
    for n in [6, 16, 32, 64, 128, 256, 512, 1022]:
        sizes.append(n)
    var ok = True
    var crossover = 0
    var batch_ahead_from = -1
    for si in range(len(sizes)):
        var n = sizes[si]
        var tokens = make_tokens(n)
        var batch_ms = List[Int]()
        var rows_ms = List[Int]()
        var ref_hidden = List[Float32]()
        var ref_argmax = -1
        for round in range(2):
            for arm in range(2):
                var is_batch = (arm + round) % 2 == 0
                var gemm_min = BATCH_GEMM_MIN if is_batch else ROWS_GEMM_MIN
                var label = "batch" if is_batch else "rows"
                var res = arm_prefill(
                    path,
                    config.gpu_budget_bytes,
                    config.gemm_tiled_min_batch,
                    gemm_min,
                    tokens,
                )
                print(
                    "WAVE_E_ROUTE n="
                    + String(n)
                    + " arm="
                    + label
                    + " round="
                    + String(round)
                    + " load_ms="
                    + String(res[0])
                    + " prefill_ms="
                    + String(res[1])
                    + " tok_s="
                    + String(Float64(n) / (Float64(res[1]) / 1000.0))
                    + " argmax="
                    + String(res[3])
                )
                if is_batch:
                    batch_ms.append(res[1])
                else:
                    rows_ms.append(res[1])
                if ref_argmax == -1:
                    ref_hidden = res[2].copy()
                    ref_argmax = res[3]
                else:
                    var cmp = compare_rows(res[2], ref_hidden)
                    print(
                        "WAVE_E_ROUTE_COMPARE n="
                        + String(n)
                        + " arm="
                        + label
                        + " round="
                        + String(round)
                        + " rel="
                        + String(cmp[0])
                        + " cosine="
                        + String(cmp[1])
                        + " argmax_match="
                        + String(1 if res[3] == ref_argmax else 0)
                    )
                    if (
                        cmp[0] > 1.0e-4
                        or cmp[1] < 0.999999
                        or res[3] != ref_argmax
                    ):
                        ok = False
        var b_min = batch_ms[0] if batch_ms[0] < batch_ms[1] else batch_ms[1]
        var r_min = rows_ms[0] if rows_ms[0] < rows_ms[1] else rows_ms[1]
        print(
            "WAVE_E_ROUTE_SUMMARY n="
            + String(n)
            + " batch_ms_min="
            + String(b_min)
            + " rows_ms_min="
            + String(r_min)
            + " batch_speedup="
            + String(Float64(r_min) / Float64(b_min))
        )
        if b_min < r_min:
            if batch_ahead_from == -1:
                batch_ahead_from = n
        else:
            batch_ahead_from = -1
        if n == 1022:
            print(
                "WAVE_E_ARMS1022 batch_ms_min="
                + String(b_min)
                + " rows_ms_min="
                + String(r_min)
                + " speedup="
                + String(Float64(r_min) / Float64(b_min))
            )
    if batch_ahead_from != -1:
        crossover = batch_ahead_from
    print("WAVE_E_CROSSOVER first_batch_wins_n=" + String(crossover))
    print("WAVE_E_ROUTE_VERDICT", "PASS" if ok else "FAIL")
    if not ok:
        exit(1)


def main():
    try:
        run()
    except e:
        print("REFUSAL", e)
        exit(2)
