# config.mojo - engine configuration.
#
# David's law: operational limits are config with defaults, never
# compiled-in constants scattered through the code. Every setting
# below lives in exactly one place (the DEFAULT_* block), each one
# overridable by an env var at launch, and a bad value is refused
# BY NAME at load time instead of being silently clamped or
# ignored. Nothing else in the engine reads env vars directly.

from std.os import getenv

# The new engine line starts at 1.1 under the repo's versioning
# scheme (1.<release number>, counting up, never resets).
# 1.2: Wave A landed (GGUF loader, tuned K1, the GEMV shape
# family, the elementwise ops, the tokenizer).
comptime ENGINE_NAME = "mojo-engine"
comptime ENGINE_VERSION = "1.4"

# Defaults. These are configuration defaults, stated once here;
# env vars of the same name override them per launch.
comptime DEFAULT_MODEL_PATH = (
    "/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
)
comptime DEFAULT_CONTEXT_LEN = 131072

# GPU memory budget (Wave C): the RX 7900 XTX carries
# 25,769,803,776 B total; the engine keeps a 1 GiB reserve for
# the rest of the system, so the default budget cap is the
# card total minus the reserve = 24,696,061,952 B. Device
# construction refuses by name (gpu_budget_exceeded, in
# src/gpu/context.mojo) when the computed resident total
# (weights + KV + scratch envelope) exceeds this cap.
comptime GPU_CARD_TOTAL_BYTES = 25769803776
comptime GPU_RESERVE_BYTES = 1073741824
comptime DEFAULT_GPU_BUDGET_BYTES = GPU_CARD_TOTAL_BYTES - GPU_RESERVE_BYTES


def parse_context_len(raw: String) raises -> Int:
    """Parse and validate a context length value.

    Split out of load() so the refusal names are directly
    testable. Raises mojo_engine_config on anything that is not
    a positive whole number.
    """
    var parsed: Int
    try:
        parsed = Int(raw)
    except:
        raise Error(
            "mojo_engine_config: MOJO_ENGINE_CONTEXT_LEN is not a"
            " whole number: '"
            + raw
            + "'"
        )
    if parsed <= 0:
        raise Error(
            "mojo_engine_config: MOJO_ENGINE_CONTEXT_LEN must be positive, got "
            + String(parsed)
        )
    return parsed


def parse_gpu_budget_bytes(raw: String) raises -> Int:
    """Parse and validate a GPU budget byte count.

    Split out of load() so the refusal names are directly
    testable, exactly like parse_context_len. Raises
    mojo_engine_config on anything that is not a positive
    whole number.
    """
    var parsed: Int
    try:
        parsed = Int(raw)
    except:
        raise Error(
            "mojo_engine_config: MOJO_ENGINE_GPU_BUDGET_BYTES is not a"
            " whole number: '"
            + raw
            + "'"
        )
    if parsed <= 0:
        raise Error(
            "mojo_engine_config: MOJO_ENGINE_GPU_BUDGET_BYTES must be"
            " positive, got "
            + String(parsed)
        )
    return parsed


struct EngineConfig(Copyable, Movable):
    """One loaded configuration; build via EngineConfig.load()."""

    var model_path: String
    var context_len: Int
    var gpu_budget_bytes: Int

    def __init__(
        out self, model_path: String, context_len: Int, gpu_budget_bytes: Int
    ):
        self.model_path = model_path
        self.context_len = context_len
        self.gpu_budget_bytes = gpu_budget_bytes

    @staticmethod
    def load() raises -> EngineConfig:
        """Load from the environment over the defaults.

        Raises a named error (mojo_engine_config: ...) on any bad
        value; the caller decides how to present the refusal.
        """
        var model_path = getenv("MOJO_ENGINE_MODEL_PATH")
        if model_path == "":
            model_path = DEFAULT_MODEL_PATH

        var context_len = DEFAULT_CONTEXT_LEN
        var raw_ctx = getenv("MOJO_ENGINE_CONTEXT_LEN")
        if raw_ctx != "":
            context_len = parse_context_len(raw_ctx)

        var gpu_budget_bytes = DEFAULT_GPU_BUDGET_BYTES
        var raw_budget = getenv("MOJO_ENGINE_GPU_BUDGET_BYTES")
        if raw_budget != "":
            gpu_budget_bytes = parse_gpu_budget_bytes(raw_budget)

        return EngineConfig(model_path, context_len, gpu_budget_bytes)
