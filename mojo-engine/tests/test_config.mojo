# test_config.mojo - config defaults and named refusals.
#
# The runner executes this suite in a clean environment (no
# MOJO_ENGINE_* vars set), so load() must return the recorded
# defaults. Env override and bad-value behavior at process level
# are covered by tests/config_probe.mojo driven from
# tests/run_tests.sh; the parse-level refusals are covered here
# directly through the same code path load() uses.

from std.testing import TestSuite, assert_true, assert_equal

from config import (
    DEFAULT_ATTN_MAX_SPLITS,
    DEFAULT_ATTN_SPLIT_SPAN_POSITIONS,
    DEFAULT_ATTN_TILE_POSITIONS,
    DEFAULT_CONTEXT_LEN,
    DEFAULT_GPU_BUDGET_BYTES,
    DEFAULT_MODEL_PATH,
    EngineConfig,
    GPU_CARD_TOTAL_BYTES,
    GPU_RESERVE_BYTES,
    DEFAULT_GEMM_TILED_MIN_BATCH,
    DEFAULT_PREFILL_GEMM_MIN_BATCH,
    parse_attn_max_splits,
    parse_attn_split_span,
    parse_attn_tile_positions,
    parse_context_len,
    parse_gemm_tiled_min_batch,
    parse_gpu_budget_bytes,
    parse_prefill_gemm_min_batch,
    # Wave E Lane E3: Glimmer op defaults folded from src/ops/.
    DEFAULT_HEAD_DIM,
    DEFAULT_HIDDEN,
    DEFAULT_LOGIT_SCALE,
    DEFAULT_LOGIT_SOFTCAP,
    DEFAULT_RMS_EPS,
    DEFAULT_ROPE_BASE,
    DEFAULT_VOCAB,
    parse_head_dim,
    parse_hidden,
    parse_logit_scale,
    parse_logit_softcap,
    parse_rms_eps,
    parse_rope_base,
    parse_vocab,
)


def test_defaults_in_clean_env() raises:
    var cfg = EngineConfig.load()
    assert_equal(cfg.context_len, DEFAULT_CONTEXT_LEN)
    assert_equal(cfg.model_path, DEFAULT_MODEL_PATH)
    assert_equal(cfg.context_len, 131072)
    assert_equal(cfg.gpu_budget_bytes, DEFAULT_GPU_BUDGET_BYTES)
    assert_equal(cfg.attn_tile_positions, DEFAULT_ATTN_TILE_POSITIONS)
    assert_equal(cfg.attn_max_splits, DEFAULT_ATTN_MAX_SPLITS)
    assert_equal(
        cfg.attn_split_span_positions, DEFAULT_ATTN_SPLIT_SPAN_POSITIONS
    )
    assert_equal(cfg.prefill_gemm_min_batch, DEFAULT_PREFILL_GEMM_MIN_BATCH)
    assert_equal(cfg.gemm_tiled_min_batch, DEFAULT_GEMM_TILED_MIN_BATCH)
    assert_equal(cfg.prefill_gemm_min_batch, 2)
    assert_equal(cfg.gemm_tiled_min_batch, 8)


def test_attn_knob_defaults_are_the_recorded_ones() raises:
    # Wave D Lane D2 starting defaults; a change here must
    # come with the trial record in the wave bench note.
    assert_equal(DEFAULT_ATTN_TILE_POSITIONS, 32)
    assert_equal(DEFAULT_ATTN_MAX_SPLITS, 8)
    assert_equal(DEFAULT_ATTN_SPLIT_SPAN_POSITIONS, 512)


def test_parse_attn_knobs_accept_and_refuse() raises:
    assert_equal(parse_attn_tile_positions("16"), 16)
    assert_equal(parse_attn_max_splits("4"), 4)
    assert_equal(parse_attn_split_span("512"), 512)
    var parsers = List[String]()
    parsers.append("tile")
    parsers.append("splits")
    parsers.append("span")
    var saw = 0
    for raw in [String("abc"), String("0"), String("-2")]:
        for which in range(3):
            try:
                if which == 0:
                    _ = parse_attn_tile_positions(raw)
                elif which == 1:
                    _ = parse_attn_max_splits(raw)
                else:
                    _ = parse_attn_split_span(raw)
                assert_true(False, "bad attn knob must refuse")
            except e:
                assert_true(
                    String(e).find("mojo_engine_config") != -1,
                    "refusal must name mojo_engine_config",
                )
                saw += 1
    assert_equal(saw, 9)


def test_gpu_budget_default_arithmetic() raises:
    # Card total minus the 1 GiB reserve, the Wave C contract
    # numbers stated as literals so a drift in either constant
    # fails here by name.
    assert_equal(GPU_CARD_TOTAL_BYTES, 25769803776)
    assert_equal(GPU_RESERVE_BYTES, 1073741824)
    assert_equal(DEFAULT_GPU_BUDGET_BYTES, 24696061952)
    assert_equal(
        DEFAULT_GPU_BUDGET_BYTES, GPU_CARD_TOTAL_BYTES - GPU_RESERVE_BYTES
    )


def test_parse_gpu_budget_accepts_and_refuses() raises:
    assert_equal(parse_gpu_budget_bytes("4096"), 4096)
    assert_equal(parse_gpu_budget_bytes("24696061952"), 24696061952)
    var saw = 0
    for raw in [String("abc"), String("0"), String("-5")]:
        try:
            _ = parse_gpu_budget_bytes(raw)
            assert_true(False, "bad GPU budget must refuse")
        except e:
            assert_true(
                String(e).find("mojo_engine_config") != -1,
                "refusal must name mojo_engine_config",
            )
            assert_true(
                String(e).find("MOJO_ENGINE_GPU_BUDGET_BYTES") != -1,
                "refusal must name the env var",
            )
            saw += 1
    assert_equal(saw, 3)


def test_parse_accepts_plain_numbers() raises:
    assert_equal(parse_context_len("4096"), 4096)
    assert_equal(parse_context_len("131072"), 131072)


def test_parse_refuses_by_name() raises:
    var saw = 0
    try:
        _ = parse_context_len("abc")
        assert_true(False, "non-numeric context length must refuse")
    except e:
        assert_true(
            String(e).find("mojo_engine_config") != -1,
            "refusal must name mojo_engine_config",
        )
        saw += 1
    try:
        _ = parse_context_len("-5")
        assert_true(False, "negative context length must refuse")
    except e:
        assert_true(
            String(e).find("mojo_engine_config") != -1,
            "refusal must name mojo_engine_config",
        )
        saw += 1
    try:
        _ = parse_context_len("0")
        assert_true(False, "zero context length must refuse")
    except e:
        assert_true(
            String(e).find("mojo_engine_config") != -1,
            "refusal must name mojo_engine_config",
        )
        saw += 1
    assert_equal(saw, 3)


def test_parse_gemm_knobs_accept_and_refuse() raises:
    assert_equal(parse_prefill_gemm_min_batch("2"), 2)
    assert_equal(parse_prefill_gemm_min_batch("4096"), 4096)
    assert_equal(parse_gemm_tiled_min_batch("8"), 8)
    for raw in ["abc", "0", "-3", "1.5"]:
        var refused = False
        try:
            _ = parse_prefill_gemm_min_batch(raw)
        except e:
            refused = String(e).find("mojo_engine_config") != -1
        assert_true(refused, "min batch must refuse by name: " + raw)
        refused = False
        try:
            _ = parse_gemm_tiled_min_batch(raw)
        except e:
            refused = String(e).find("mojo_engine_config") != -1
        assert_true(refused, "tiled min batch must refuse by name: " + raw)


# Wave E Lane E3: the Glimmer op defaults folded in from
# src/ops/ (GLIMMER_RMS_EPS, GLIMMER_ROPE_BASE,
# GLIMMER_LOGIT_SCALE / GLIMMER_LOGIT_SOFTCAP, GLIMMER_HEAD_DIM,
# GLIMMER_HIDDEN, GLIMMER_VOCAB) keep their exact recorded
# values as EngineConfig defaults, stated here as literals so
# a drift in either the constant or the value fails by name.
def test_glimmer_op_defaults_are_the_recorded_ones() raises:
    assert_true(DEFAULT_RMS_EPS == Float32(1.0e-5), "rms eps default")
    assert_true(DEFAULT_ROPE_BASE == Float32(500000.0), "rope base default")
    assert_true(
        DEFAULT_LOGIT_SCALE == Float32(0.1961161345243454),
        "logit scale default",
    )
    assert_true(DEFAULT_LOGIT_SOFTCAP == Float32(20.0), "softcap default")
    assert_equal(DEFAULT_HEAD_DIM, 128)
    assert_equal(DEFAULT_HIDDEN, 6656)
    assert_equal(DEFAULT_VOCAB, 202048)
    var cfg = EngineConfig.load()
    assert_true(cfg.rms_eps == DEFAULT_RMS_EPS, "cfg rms_eps default")
    assert_true(cfg.rope_base == DEFAULT_ROPE_BASE, "cfg rope_base default")
    assert_true(
        cfg.logit_scale == DEFAULT_LOGIT_SCALE, "cfg logit_scale default"
    )
    assert_true(
        cfg.logit_softcap == DEFAULT_LOGIT_SOFTCAP, "cfg softcap default"
    )
    assert_equal(cfg.head_dim, DEFAULT_HEAD_DIM)
    assert_equal(cfg.hidden, DEFAULT_HIDDEN)
    assert_equal(cfg.vocab, DEFAULT_VOCAB)


def test_parse_glimmer_op_knobs_accept() raises:
    assert_true(parse_rms_eps("0.001") == Float32(0.001), "rms eps parse")
    assert_true(parse_rms_eps("1e-5") == Float32(1.0e-5), "rms eps sci")
    assert_true(
        parse_rope_base("500000") == Float32(500000.0), "rope base parse"
    )
    assert_true(parse_logit_scale("0.5") == Float32(0.5), "scale parse")
    assert_true(
        parse_logit_softcap("20") == Float32(20.0), "softcap parse"
    )
    assert_true(
        parse_logit_softcap("0") == Float32(0.0), "softcap 0 = no cap"
    )
    assert_equal(parse_head_dim("128"), 128)
    assert_equal(parse_hidden("6656"), 6656)
    assert_equal(parse_vocab("202048"), 202048)


def test_parse_glimmer_op_knobs_refuse_by_name() raises:
    # (which parser, raw) pairs; every one must refuse naming
    # mojo_engine_config AND its own env var.
    var cases = List[Tuple[Int, String, String]]()
    cases.append((0, String("abc"), String("MOJO_ENGINE_RMS_EPS")))
    cases.append((0, String("0"), String("MOJO_ENGINE_RMS_EPS")))
    cases.append((0, String("-1"), String("MOJO_ENGINE_RMS_EPS")))
    cases.append((0, String("nan"), String("MOJO_ENGINE_RMS_EPS")))
    cases.append((0, String("inf"), String("MOJO_ENGINE_RMS_EPS")))
    cases.append((1, String("abc"), String("MOJO_ENGINE_ROPE_BASE")))
    cases.append((1, String("1"), String("MOJO_ENGINE_ROPE_BASE")))
    cases.append((1, String("0.5"), String("MOJO_ENGINE_ROPE_BASE")))
    cases.append((2, String("abc"), String("MOJO_ENGINE_LOGIT_SCALE")))
    cases.append((2, String("nan"), String("MOJO_ENGINE_LOGIT_SCALE")))
    cases.append((3, String("-1"), String("MOJO_ENGINE_LOGIT_SOFTCAP")))
    cases.append((3, String("inf"), String("MOJO_ENGINE_LOGIT_SOFTCAP")))
    cases.append((4, String("0"), String("MOJO_ENGINE_HEAD_DIM")))
    cases.append((5, String("abc"), String("MOJO_ENGINE_HIDDEN")))
    cases.append((6, String("-2"), String("MOJO_ENGINE_VOCAB")))
    var saw = 0
    for i in range(len(cases)):
        var which = cases[i][0]
        var raw = cases[i][1]
        var env_name = cases[i][2]
        try:
            if which == 0:
                _ = parse_rms_eps(raw)
            elif which == 1:
                _ = parse_rope_base(raw)
            elif which == 2:
                _ = parse_logit_scale(raw)
            elif which == 3:
                _ = parse_logit_softcap(raw)
            elif which == 4:
                _ = parse_head_dim(raw)
            elif which == 5:
                _ = parse_hidden(raw)
            else:
                _ = parse_vocab(raw)
            assert_true(False, "bad op knob must refuse: " + raw)
        except e:
            assert_true(
                String(e).find("mojo_engine_config") != -1,
                "refusal must name mojo_engine_config",
            )
            assert_true(
                String(e).find(env_name) != -1,
                "refusal must name its env var " + env_name,
            )
            saw += 1
    assert_equal(saw, len(cases))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
