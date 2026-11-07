# server_proof_model.mojo - Lane B3 rungs 2 and 3, run on the
# server against the real Glimmer file.
#
# RUNG 2: layer 0 (sliding) and layer 3 (global) alone. Input =
# this model's own embedding + weightless plain norm for token
# 13796 at position 0, fed through layer_step on a FRESH model
# (so the cache state is exactly position 0 for every layer;
# only the named layer's output is read). Prints the first 8
# output elements plus a Float64 checksum-style sum for the
# independent Python spot check to compare against.
#
# RUNG 3: teacher-forced golden. A fresh model steps the
# prompt [200000, 954, 7963, 323, 11698, 373] (positions 0..5),
# then predicts after each step; the predictions must walk the
# golden stream [13796, 26, 1573, 373, 290, 2167, 26899,
# 1212]: after the prompt, argmax must be 13796; then the
# GOLDEN ids are fed one at a time and each next argmax must
# equal the next golden id. Prints one line per prediction:
# RUNG3 pos=<consumed position> pred=<id> golden=<id> ok=<0/1>
# plus the winning logit, and RUNG3 DONE pass=<count>/8.

from std.time import perf_counter_ns

from model.model import GlimmerModel
from generate.sampler import greedy_argmax
from ops.rms_norm import rms_norm_plain_cpu

comptime MODEL_PATH = (
    "/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
)


def _top5(logits: List[Float32]) -> String:
    var ids = List[Int]()
    var vals = List[Float32]()
    for _ in range(5):
        ids.append(-1)
        vals.append(Float32(-1.0e30))
    for i in range(len(logits)):
        var v = logits[i]
        for r in range(5):
            if v > vals[r]:
                for s in range(4, r, -1):
                    vals[s] = vals[s - 1]
                    ids[s] = ids[s - 1]
                vals[r] = v
                ids[r] = i
                break
    var out = String("")
    for r in range(5):
        out += String(ids[r]) + ":" + String(vals[r]) + " "
    return out


def _row_stats(row: List[Float32]) -> String:
    var s = Float64(0.0)
    var mx = Float64(-1.0e30)
    for i in range(len(row)):
        s += Float64(row[i])
        if Float64(row[i]) > mx:
            mx = Float64(row[i])
    return (
        "sum="
        + String(s)
        + " max="
        + String(mx)
        + " first8="
        + String(row[0])
        + ","
        + String(row[1])
        + ","
        + String(row[2])
        + ","
        + String(row[3])
        + ","
        + String(row[4])
        + ","
        + String(row[5])
        + ","
        + String(row[6])
        + ","
        + String(row[7])
    )


def main() raises:
    var t0 = perf_counter_ns()
    var model = GlimmerModel.open(MODEL_PATH)
    var t_load = perf_counter_ns()
    print(
        "LOAD_DONE ms="
        + String((t_load - t0) // 1_000_000)
        + " embd_type="
        + String(model.embd.type_id)
        + " output_type="
        + String(model.output.type_id)
    )

    # ---- rung 2: single real layers at position 0 ----
    for target in [0, 3]:
        var fresh = GlimmerModel.open(MODEL_PATH)
        var x = fresh.embed_token(13796)
        x = rms_norm_plain_cpu(x, fresh.cfg.rms_eps)
        var t1 = perf_counter_ns()
        var out = fresh.layer_step(x, target, 0)
        var t2 = perf_counter_ns()
        print(
            "RUNG2 layer="
            + String(target)
            + " ms="
            + String((t2 - t1) // 1_000_000)
            + " "
            + _row_stats(out)
        )

    # ---- rung 3: teacher-forced golden ----
    var gen = GlimmerModel.open(MODEL_PATH)
    var prompt = List[Int]()
    for t in [200000, 954, 7963, 323, 11698, 373]:
        prompt.append(t)
    var golden = List[Int]()
    for t in [13796, 26, 1573, 373, 290, 2167, 26899, 1212]:
        golden.append(t)
    var hidden = gen.forward_rows(prompt.copy())
    var passed = 0
    var t3 = perf_counter_ns()
    for gi in range(len(golden)):
        var logits = gen.logits(hidden)
        var pred = greedy_argmax(logits)
        var ok = 1 if pred == golden[gi] else 0
        passed += ok
        print(
            "RUNG3 idx="
            + String(gi)
            + " pred="
            + String(pred)
            + " golden="
            + String(golden[gi])
            + " ok="
            + String(ok)
            + " win_logit="
            + String(logits[pred])
        )
        if gi == 0:
            print("RUNG3 top5 " + _top5(logits))
        if gi < len(golden) - 1:
            var tstep = perf_counter_ns()
            hidden = gen.step(golden[gi], len(prompt) + gi)
            print(
                "RUNG3 step_ms="
                + String((perf_counter_ns() - tstep) // 1_000_000)
            )
    print(
        "RUNG3 DONE pass="
        + String(passed)
        + "/8 total_ms="
        + String((perf_counter_ns() - t3) // 1_000_000)
    )
