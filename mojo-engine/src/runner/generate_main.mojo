# generate_main.mojo - the end-to-end generation runner.
#
# Opens the real Glimmer model through the GGUF loader,
# tokenizes the golden prompt "The capital of France is" with
# the Wave A tokenizer (BOS prepended by the caller, exactly
# as the Rust golden harness builds its prompt ids: encode()
# adds no BOS, the harness does), wraps the model in
# GlimmerForwardAdapter, and runs B4's greedy generate for 16
# tokens at eos 200001. Prints the prompt ids, every produced
# token id, the decoded text, a first-8 golden comparison,
# and first-light timing:
#
#   PREFILL_MS    measured on a SECOND fresh model stepping
#                 the same prompt ids (forward_rows), so the
#                 timed pass cannot pollute the run's cache.
#   GENERATE_MS   the whole generate() call (6 prefill steps
#                 + 15 decode steps + 16 logits evaluations).
#   DECODE_TOK_S  approximate: 15 decode steps over
#                 (GENERATE_MS - PREFILL_MS); the spread of
#                 logits evaluations across that window makes
#                 this a first-light figure, not a benchmark.
#
# Usage (from the mojo-engine/ directory):
#   mojo run -I src src/runner/generate_main.mojo
# Model path comes from EngineConfig (MOJO_ENGINE_MODEL_PATH
# over the compiled default), as everywhere else in the
# engine. Exit 0 on a completed run (the GOLDEN line carries
# the verdict), exit 2 with a REFUSAL line on any refusal.

from std.sys import exit
from std.time import perf_counter_ns

from config import ENGINE_NAME, ENGINE_VERSION, EngineConfig
from diag import log_debug
from generate.generate import generate
from model.model import GlimmerForwardAdapter, GlimmerModel
from tokenizer.gguf_vocab import load_tokenizer_data
from tokenizer.tokenizer import Tokenizer

comptime PROMPT_TEXT = "The capital of France is"
comptime MAX_TOKENS = 16
comptime EOS_ID = 200001


def _strict_ok(tok: Tokenizer, ids: List[Int]) -> Bool:
    try:
        _ = tok.decode(ids)
        return True
    except:
        return False


def _golden() -> List[Int]:
    var g = List[Int]()
    for t in [13796, 26, 1573, 373, 290, 2167, 26899, 1212]:
        g.append(t)
    return g^


def _ids_string(ids: List[Int]) -> String:
    var out = String("[")
    for i in range(len(ids)):
        if i > 0:
            out += ", "
        out += String(ids[i])
    out += "]"
    return out


def run() raises:
    print("RUNNER", ENGINE_NAME, ENGINE_VERSION)
    var config = EngineConfig.load()
    var path = config.model_path
    print("MODEL_PATH", path)

    var data = load_tokenizer_data(path)
    var tok = Tokenizer.build(data^)
    print(
        "TOKENIZER family="
        + tok.family_name()
        + " vocab="
        + String(tok.vocab_size())
        + " bos="
        + String(tok.bos_id())
        + " eos="
        + String(tok.eos_id())
    )

    var prompt = List[Int]()
    prompt.append(tok.bos_id())
    var body = tok.encode(PROMPT_TEXT)
    for i in range(len(body)):
        prompt.append(body[i])
    print("PROMPT_TEXT", PROMPT_TEXT)
    print("PROMPT_IDS", _ids_string(prompt))

    var t0 = perf_counter_ns()
    var model = GlimmerModel.open(path)
    var t_load = perf_counter_ns()
    print("LOAD_MS", String((t_load - t0) // 1_000_000))

    var adapter = GlimmerForwardAdapter(model^)
    var t_gen0 = perf_counter_ns()
    var out = generate(adapter, prompt.copy(), MAX_TOKENS, EOS_ID)
    var t_gen1 = perf_counter_ns()
    var generate_ms = (t_gen1 - t_gen0) // 1_000_000
    print("GENERATE_MS", String(generate_ms))
    print("TOKENS_COUNT", String(len(out)))
    print("TOKENS", _ids_string(out))

    var text = tok.decode_lossy(out.copy())
    print("TEXT", text)
    print("TEXT_STRICT_OK", String(1 if _strict_ok(tok, out.copy()) else 0))

    var golden = _golden()
    var gmatch = 1
    if len(out) < len(golden):
        gmatch = 0
    else:
        for i in range(len(golden)):
            if out[i] != golden[i]:
                gmatch = 0
    print("GOLDEN_FIRST8_MATCH", String(gmatch))
    _ = log_debug(
        "runner",
        "generated " + String(len(out)) + " tokens, match " + String(gmatch),
    )

    # Prefill timing on a second fresh model (see header).
    var fresh = GlimmerModel.open(path)
    var t_pf0 = perf_counter_ns()
    _ = fresh.forward_rows(prompt.copy())
    var t_pf1 = perf_counter_ns()
    var prefill_ms = (t_pf1 - t_pf0) // 1_000_000
    print("PREFILL_MS", String(prefill_ms))
    var decode_ms = generate_ms - prefill_ms
    if decode_ms > 0:
        var step_ms = Float64(decode_ms) / 15.0
        print("DECODE_STEP_MS_APPROX", String(step_ms))
        print("DECODE_TOK_S_APPROX", String(1000.0 / step_ms))
    print("RUNNER_DONE")


def main():
    try:
        run()
    except e:
        print("REFUSAL", e)
        exit(2)
