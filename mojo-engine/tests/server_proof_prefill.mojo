# server_proof_prefill.mojo - Wave D Lane D1 real-model
# prefill measurement: a 1,022-token prefill on the card,
# interleaved NEW (batched GEMM prefill) vs Wave-C path
# (per-row GEMV prefill), same window, same staged tree.
# NOT a TestSuite file: run explicitly on a GPU host with
#   mojo run -I src tests/server_proof_prefill.mojo
#
# The two arms are the SAME GpuGlimmerModel code on the SAME
# staged commit; the only difference is the configured
# MOJO_ENGINE_PREFILL_GEMM_MIN_BATCH value handed to the
# constructor: 2 for the NEW arm (every projection of the
# 1,022-row prompt runs as GEMM launches) and 1,000,000 for
# the OLD arm (no prompt ever reaches the GEMM minimum, so
# prefill is the Wave C step loop, GEMV per row). That is the
# honest in-tree A/B: one binary, one window, arms rotated
# across two rounds (NEW/OLD, then OLD/NEW), each arm on a
# FRESH model (fresh KV, positions 0..1021), load time
# excluded from the prefill timing and printed separately.
#
# Correctness inside the measurement: each arm's last hidden
# row is compared against the other arm's (k1_check metrics:
# rel = max_abs_err / max(1, max|ref|), Float64 cosine) and
# the argmax of the finished logits must agree exactly.
# Verdict bar (the Wave C layer-proof class for full-forward
# device comparisons): cosine >= 0.999999, rel <= 1e-4,
# argmax exact.
#
# Prompt: 1,022 deterministic pseudo-random token ids
# (LCG, ids in [0, 200000)), stated plainly - this measures
# the forward, not the tokenizer; token VALUES do not change
# the per-shape work. 1,022 < 2,048 keeps every sliding layer
# inside the Wave C pre-wrap limit, and the proof context is
# 4,096 like the teacher-forced proof (residency is the
# runner window's question, not this measurement's).

from std.math import sqrt
from std.sys import exit
from std.time import perf_counter_ns

from config import EngineConfig
from generate.sampler import greedy_argmax
from model.gpu_model import GpuGlimmerModel

comptime PROOF_CONTEXT_LEN = 4096
comptime N_TOKENS = 1022
comptime NEW_MIN_BATCH = 2
comptime OLD_MIN_BATCH = 1000000


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
    min_batch: Int,
    tokens: List[Int],
) raises -> Tuple[Int, Int, List[Float32], Int]:
    """One fresh model, one timed forward_rows over the
    prompt. Returns (load_ms, prefill_ms, last_hidden,
    argmax)."""
    var t0 = perf_counter_ns()
    var model = GpuGlimmerModel.open(
        path, PROOF_CONTEXT_LEN, budget, min_batch, tiled_min
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
    print("PREFILL_PROOF model_path", path)
    print(
        "PREFILL_PROOF tokens",
        String(N_TOKENS),
        "context_len",
        String(PROOF_CONTEXT_LEN),
    )
    var tokens = make_tokens(N_TOKENS)
    var new_ms = List[Int]()
    var old_ms = List[Int]()
    var ref_hidden = List[Float32]()
    var ref_argmax = -1
    var ok = True
    for round in range(2):
        for arm in range(2):
            var is_new = (arm + round) % 2 == 0
            var min_batch = NEW_MIN_BATCH if is_new else OLD_MIN_BATCH
            var label = "new" if is_new else "old"
            var res = arm_prefill(
                path,
                config.gpu_budget_bytes,
                config.gemm_tiled_min_batch,
                min_batch,
                tokens,
            )
            print(
                "PREFILL1022 arm="
                + label
                + " round="
                + String(round)
                + " load_ms="
                + String(res[0])
                + " prefill_ms="
                + String(res[1])
                + " tok_s="
                + String(Float64(N_TOKENS) / (Float64(res[1]) / 1000.0))
                + " argmax="
                + String(res[3])
            )
            if is_new:
                new_ms.append(res[1])
            else:
                old_ms.append(res[1])
            if ref_argmax == -1:
                ref_hidden = res[2].copy()
                ref_argmax = res[3]
            else:
                var cmp = compare_rows(res[2], ref_hidden)
                print(
                    "PREFILL1022_COMPARE arm="
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
                if cmp[0] > 1.0e-4 or cmp[1] < 0.999999 or res[3] != ref_argmax:
                    ok = False
    var new_med = new_ms[0] if new_ms[0] < new_ms[1] else new_ms[1]
    var old_med = old_ms[0] if old_ms[0] < old_ms[1] else old_ms[1]
    # Two rounds each: report the faster (min) and both values;
    # with two samples the min is the cleaner machine figure,
    # stated as such.
    print(
        "PREFILL1022_SUMMARY new_ms_min="
        + String(new_med)
        + " old_ms_min="
        + String(old_med)
        + " new_tok_s="
        + String(Float64(N_TOKENS) / (Float64(new_med) / 1000.0))
        + " old_tok_s="
        + String(Float64(N_TOKENS) / (Float64(old_med) / 1000.0))
        + " speedup="
        + String(Float64(old_med) / Float64(new_med))
    )
    print("PREFILL1022_VERDICT", "PASS" if ok else "FAIL")
    if not ok:
        exit(1)


def main():
    try:
        run()
    except e:
        print("REFUSAL", e)
        exit(2)
