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
#
# Wave C (Lane C3): when the environment variable
# MOJO_ENGINE_GPU is set to a value other than "" and "0"
# (the diag seam's convention, diag.debug_value_enabled), the
# run drives GpuGlimmerModel instead of the CPU model, the
# same prints, and the same prefill-measurement construction
# (a second fresh model stepping the prompt ids). Wave F:
# the GPU drive is generate_device (the resident-seam loop,
# generate()'s exact semantics with decode steps resident on
# device); the CPU path still runs B4's generate(). The
# default is the CPU path,
# exactly as Wave B shipped. One printed line names the path
# taken: MODEL_PATH_KIND gpu|cpu. Every other printed line
# keeps its Wave B format.

from std.os import getenv
from std.sys import exit
from std.time import perf_counter_ns

from config import ENGINE_NAME, ENGINE_VERSION, EngineConfig
from diag import debug_value_enabled, log_debug
from generate.generate import generate, generate_device
from generate.sampler import golden_first_match
from model.gpu_model import GpuGlimmerModel
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


def _use_gpu() -> Bool:
    """True when MOJO_ENGINE_GPU selects the GPU path: any
    value other than "" and "0", the diag seam's convention."""
    return debug_value_enabled(getenv("MOJO_ENGINE_GPU"))


def _report_results(
    tok: Tokenizer, tokens_out: List[Int], generate_ms: Int
) raises:
    """The result prints shared by both paths, in Wave B
    order and formats: GENERATE_MS, TOKENS_COUNT, TOKENS,
    TEXT, TEXT_STRICT_OK, GOLDEN_FIRST8_MATCH."""
    print("GENERATE_MS", String(generate_ms))
    print("TOKENS_COUNT", String(len(tokens_out)))
    print("TOKENS", _ids_string(tokens_out))

    var text = tok.decode_lossy(tokens_out.copy())
    print("TEXT", text)
    print(
        "TEXT_STRICT_OK",
        String(1 if _strict_ok(tok, tokens_out.copy()) else 0),
    )

    var golden = _golden()
    # The comparison itself is generate.sampler's
    # golden_first_match, the same function the suite's
    # poison gates exercise with a flipped 8th token.
    var gmatch = 1 if golden_first_match(tokens_out, golden) else 0
    print("GOLDEN_FIRST8_MATCH", String(gmatch))
    _ = log_debug(
        "runner",
        "generated "
        + String(len(tokens_out))
        + " tokens, match "
        + String(gmatch),
    )


def _report_prefill(prefill_ms: Int, generate_ms: Int):
    """The prefill timing prints, Wave B order and formats."""
    print("PREFILL_MS", String(prefill_ms))
    var decode_ms = generate_ms - prefill_ms
    if decode_ms > 0:
        var step_ms = Float64(decode_ms) / 15.0
        print("DECODE_STEP_MS_APPROX", String(step_ms))
        print("DECODE_TOK_S_APPROX", String(1000.0 / step_ms))
    print("RUNNER_DONE")


def _gpu_generate_phase(
    path: String,
    context_len: Int,
    budget_bytes: Int,
    attn_tile: Int,
    attn_max_splits: Int,
    attn_span: Int,
    prompt: List[Int],
    prefill_gemm_min_batch: Int,
    gemm_tiled_min_batch: Int,
    prefill_batch_min_tokens: Int,
) raises -> Tuple[List[Int], Int, Int]:
    """Build the GPU model, time its load, generate, and
    DESTROY the model on return. Wave C integration fix: the
    CPU path holds its generate model and its fresh prefill
    model alive at the same time (host RAM shrugs); on device
    two resident models are about 41 GB against a 24 GB card,
    so the GPU path phases the two constructions - the
    prefill model is built only after this one is gone. The
    printed lines and their order are unchanged."""
    var t0 = perf_counter_ns()
    var model = GpuGlimmerModel(
        GlimmerModel.open(path)^,
        context_len,
        budget_bytes,
        attn_tile,
        attn_max_splits,
        attn_span,
        prefill_gemm_min_batch,
        gemm_tiled_min_batch,
        prefill_batch_min_tokens,
    )
    var t_load = perf_counter_ns()
    var load_ms = Int((t_load - t0) // 1_000_000)
    var t_gen0 = perf_counter_ns()
    var out = generate_device(model, prompt.copy(), MAX_TOKENS, EOS_ID)
    var t_gen1 = perf_counter_ns()
    var generate_ms = Int((t_gen1 - t_gen0) // 1_000_000)
    return (out^, load_ms, generate_ms)


def _gpu_prefill_phase(
    path: String,
    context_len: Int,
    budget_bytes: Int,
    attn_tile: Int,
    attn_max_splits: Int,
    attn_span: Int,
    prompt: List[Int],
    prefill_gemm_min_batch: Int,
    gemm_tiled_min_batch: Int,
    prefill_batch_min_tokens: Int,
) raises -> Int:
    """Build a fresh GPU model (the generate model is already
    destroyed), time forward_rows over the prompt ids, and
    destroy it on return. The Wave B prefill construction,
    phased for device memory."""
    var fresh = GpuGlimmerModel(
        GlimmerModel.open(path)^,
        context_len,
        budget_bytes,
        attn_tile,
        attn_max_splits,
        attn_span,
        prefill_gemm_min_batch,
        gemm_tiled_min_batch,
        prefill_batch_min_tokens,
    )
    var t_pf0 = perf_counter_ns()
    _ = fresh.forward_rows(prompt.copy())
    var t_pf1 = perf_counter_ns()
    return Int((t_pf1 - t_pf0) // 1_000_000)


def _run_gpu(
    path: String,
    context_len: Int,
    budget_bytes: Int,
    attn_tile: Int,
    attn_max_splits: Int,
    attn_span: Int,
    tok: Tokenizer,
    prompt: List[Int],
    prefill_gemm_min_batch: Int,
    gemm_tiled_min_batch: Int,
    prefill_batch_min_tokens: Int,
) raises:
    """The Wave C GPU path: same run, same prints, driven by
    GpuGlimmerModel. Prefill is measured on a second fresh GPU
    model stepping the same prompt ids (forward_rows), the
    identical construction to the CPU path below, phased so
    only one device model is resident at a time. The Wave D
    attention knobs arrive from run()'s EngineConfig."""
    var phase = _gpu_generate_phase(
        path,
        context_len,
        budget_bytes,
        attn_tile,
        attn_max_splits,
        attn_span,
        prompt,
        prefill_gemm_min_batch,
        gemm_tiled_min_batch,
        prefill_batch_min_tokens,
    )
    var out = phase[0].copy()
    print("LOAD_MS", String(phase[1]))
    _report_results(tok, out, phase[2])

    var prefill_ms = _gpu_prefill_phase(
        path,
        context_len,
        budget_bytes,
        attn_tile,
        attn_max_splits,
        attn_span,
        prompt,
        prefill_gemm_min_batch,
        gemm_tiled_min_batch,
        prefill_batch_min_tokens,
    )
    _report_prefill(prefill_ms, phase[2])


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

    var use_gpu = _use_gpu()
    print("MODEL_PATH_KIND", "gpu" if use_gpu else "cpu")
    if use_gpu:
        _run_gpu(
            path,
            config.context_len,
            config.gpu_budget_bytes,
            config.attn_tile_positions,
            config.attn_max_splits,
            config.attn_split_span_positions,
            tok,
            prompt,
            config.prefill_gemm_min_batch,
            config.gemm_tiled_min_batch,
            config.prefill_batch_min_tokens,
        )
        return

    var t0 = perf_counter_ns()
    var model = GlimmerModel.open(path)
    var t_load = perf_counter_ns()
    print("LOAD_MS", String((t_load - t0) // 1_000_000))

    var adapter = GlimmerForwardAdapter(model^)
    var t_gen0 = perf_counter_ns()
    var out = generate(adapter, prompt.copy(), MAX_TOKENS, EOS_ID)
    var t_gen1 = perf_counter_ns()
    var generate_ms = (t_gen1 - t_gen0) // 1_000_000
    _report_results(tok, out, generate_ms)

    # Prefill timing on a second fresh model (see header).
    var fresh = GlimmerModel.open(path)
    var t_pf0 = perf_counter_ns()
    _ = fresh.forward_rows(prompt.copy())
    var t_pf1 = perf_counter_ns()
    var prefill_ms = (t_pf1 - t_pf0) // 1_000_000
    _report_prefill(prefill_ms, generate_ms)


def main():
    try:
        run()
    except e:
        print("REFUSAL", e)
        exit(2)
