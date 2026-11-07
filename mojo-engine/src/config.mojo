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
comptime ENGINE_NAME = "mojo-engine"
comptime ENGINE_VERSION = "1.1"

# Defaults. These are configuration defaults, stated once here;
# env vars of the same name override them per launch.
comptime DEFAULT_MODEL_PATH = (
    "/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
)
comptime DEFAULT_CONTEXT_LEN = 131072


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


struct EngineConfig(Copyable, Movable):
    """One loaded configuration; build via EngineConfig.load()."""

    var model_path: String
    var context_len: Int

    def __init__(out self, model_path: String, context_len: Int):
        self.model_path = model_path
        self.context_len = context_len

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

        return EngineConfig(model_path, context_len)
