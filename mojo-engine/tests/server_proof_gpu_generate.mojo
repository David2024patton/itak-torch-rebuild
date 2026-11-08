# server_proof_gpu_generate.mojo - Lane C3 teacher-forced
# device check, run on the server against the real Glimmer
# file. NOT a TestSuite file: run explicitly on a GPU host with
#   mojo run -I src tests/server_proof_gpu_generate.mojo
#
# The teacher-forced check ONLY (the end-to-end free run is the
# coordinator's integration window, per WAVE_C_INTERFACES.md
# Lane C3): build the GPU model on the real file, step the
# prompt [200000, 954, 7963, 323, 11698, 373] at positions
# 0..5, then predict after each step; the predictions must
# walk the golden stream [13796, 26, 1573, 373, 290, 2167,
# 26899, 1212]. After the prompt, argmax must be 13796; then
# the GOLDEN ids are fed one at a time (positions 6, 7, ...)
# and each next argmax must equal the next golden id - so
# forcing golden 2167 at position 11 predicts 26899, and
# forcing golden 26899 at position 12 predicts 1212.
#
# This mirrors EXACTLY how Wave B's rung 3 teacher-forced walk
# was constructed in tests/server_proof_model.mojo: rung 3 fed
# the prompt with forward_rows (the step loop over positions
# 0..5), predicted 8 times from the running hidden row, and
# fed golden[gi] at position len(prompt) + gi between
# predictions - 8 predictions, prompt 6 + forced goldens.
# Same construction here, with GpuGlimmerModel's forward_rows
# and step in place of the CPU model's.
#
# Prints one line per prediction:
#   TEACHER_FORCED_GPU idx=<i> pos=<consumed position>
#       pred=<id> expected=<id> ok=<0/1> win_logit=<value>
# and a final verdict line:
#   TEACHER_FORCED_GPU 8/8        (or the actual count, n/8)
# Exits nonzero on any mismatch. On a GPU-less host the GPU
# model construction refuses by name (gpu_no_device) and this
# proof exits nonzero with that REFUSAL line: a host fact, not
# a pass.
#
# Context length: the proof builds the device KV cache at
# PROOF_CONTEXT_LEN positions for global layers, NOT the
# configured 131,072 - the teacher-forced walk touches 14
# positions, and a full-context device cache is the runner /
# integration window's residency question (the contract's
# memory budget), not this correctness check's. Stated plainly
# so the smaller residency of this proof is never mistaken for
# the budget figure.

from std.sys import exit
from std.time import perf_counter_ns

from config import EngineConfig
from generate.sampler import greedy_argmax
from model.gpu_model import GpuGlimmerModel

comptime PROOF_CONTEXT_LEN = 4096


def run() raises:
    var config = EngineConfig.load()
    var path = config.model_path
    print("GPU_GENERATE_PROOF model_path", path)
    print("GPU_GENERATE_PROOF context_len", String(PROOF_CONTEXT_LEN))

    var t0 = perf_counter_ns()
    var model = GpuGlimmerModel.open(path, PROOF_CONTEXT_LEN)
    var t_load = perf_counter_ns()
    print("LOAD_MS", String((t_load - t0) // 1_000_000))

    var prompt = List[Int]()
    for t in [200000, 954, 7963, 323, 11698, 373]:
        prompt.append(t)
    var expected = List[Int]()
    for t in [13796, 26, 1573, 373, 290, 2167, 26899, 1212]:
        expected.append(t)

    var hidden = model.forward_rows(prompt.copy())
    var passed = 0
    var t3 = perf_counter_ns()
    for gi in range(len(expected)):
        var logits = model.logits(hidden)
        var pred = greedy_argmax(logits)
        var ok = 1 if pred == expected[gi] else 0
        passed += ok
        print(
            "TEACHER_FORCED_GPU idx="
            + String(gi)
            + " pos="
            + String(len(prompt) + gi - 1)
            + " pred="
            + String(pred)
            + " expected="
            + String(expected[gi])
            + " ok="
            + String(ok)
            + " win_logit="
            + String(logits[pred])
        )
        if gi < len(expected) - 1:
            var tstep = perf_counter_ns()
            hidden = model.step(expected[gi], len(prompt) + gi)
            print(
                "TEACHER_FORCED_GPU step_ms="
                + String((perf_counter_ns() - tstep) // 1_000_000)
            )
    print(
        "TEACHER_FORCED_GPU "
        + String(passed)
        + "/8 total_ms="
        + String((perf_counter_ns() - t3) // 1_000_000)
    )
    if passed != len(expected):
        exit(1)


def main():
    try:
        run()
    except e:
        print("REFUSAL", e)
        exit(2)
