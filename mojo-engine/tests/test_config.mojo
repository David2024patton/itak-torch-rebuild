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
    DEFAULT_CONTEXT_LEN,
    DEFAULT_GPU_BUDGET_BYTES,
    DEFAULT_MODEL_PATH,
    EngineConfig,
    GPU_CARD_TOTAL_BYTES,
    GPU_RESERVE_BYTES,
    parse_context_len,
    parse_gpu_budget_bytes,
)


def test_defaults_in_clean_env() raises:
    var cfg = EngineConfig.load()
    assert_equal(cfg.context_len, DEFAULT_CONTEXT_LEN)
    assert_equal(cfg.model_path, DEFAULT_MODEL_PATH)
    assert_equal(cfg.context_len, 131072)
    assert_equal(cfg.gpu_budget_bytes, DEFAULT_GPU_BUDGET_BYTES)


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
