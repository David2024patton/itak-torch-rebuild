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
comptime ENGINE_VERSION = "1.5"

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

# Attention occupancy knobs (Wave D, Lane D2): the GPU
# attention path stages K/V in shared-memory tiles of
# DEFAULT_ATTN_TILE_POSITIONS positions and, once the stored
# sequence grows past one split span, splits the KV sequence
# across workgroups (split-KV) up to DEFAULT_ATTN_MAX_SPLITS
# partials merged by exact logsumexp rescaling. These are the
# operator-tunable defaults; the attention module validates
# the tile against its compiled instantiations by name
# (attention_tiled_bad_tile) and the split chooser's inputs
# by name (attention_split_bad_config). Trial record:
# bench/mojo-engine-wave-d-2026-11-07.md.
comptime DEFAULT_ATTN_TILE_POSITIONS = 32
comptime DEFAULT_ATTN_MAX_SPLITS = 8
# Span 512 (not the 1024 first cut): the recorded card
# trials (bench/mojo-engine-wave-d-2026-11-07.md) show the
# split path still gaining at 8 splits on the 4,096-position
# fixture (tile 32: 512 us at 8 splits vs 896 us at 4), so
# the chooser reaches 8 splits at 4,096 stored; at or below
# 512 stored the launch stays single-split.
comptime DEFAULT_ATTN_SPLIT_SPAN_POSITIONS = 512

# Prefill GEMM knobs (Wave D, Lane D1): a prompt batch at or
# above PREFILL_GEMM_MIN_BATCH rows takes the batched GEMM
# projection path on device (below it the per-row GEMV path is
# kept - at one or two rows the GEMV launches are already the
# cheap shape, and decode always uses GEMV); at or above
# GEMM_TILED_MIN_BATCH rows the register-tiled GEMM structure
# is used instead of the straightforward batched one. Setting
# PREFILL_GEMM_MIN_BATCH above any real prompt length disables
# the GEMM path entirely (the A/B control the Wave D timing
# harness uses). Both are positive whole numbers, refused by
# name like every other value here.
comptime DEFAULT_PREFILL_GEMM_MIN_BATCH = 2
comptime DEFAULT_GEMM_TILED_MIN_BATCH = 8


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


def _parse_positive(raw: String, env_name: String) raises -> Int:
    """Shared positive-whole-number parse for the attention
    and prefill GEMM knobs; refuses by name like the other
    config values."""
    var parsed: Int
    try:
        parsed = Int(raw)
    except:
        raise Error(
            "mojo_engine_config: "
            + env_name
            + " is not a whole number: '"
            + raw
            + "'"
        )
    if parsed <= 0:
        raise Error(
            "mojo_engine_config: "
            + env_name
            + " must be positive, got "
            + String(parsed)
        )
    return parsed


def parse_attn_tile_positions(raw: String) raises -> Int:
    """MOJO_ENGINE_ATTN_TILE_POSITIONS: positions per shared-
    memory K/V tile in the GPU attention kernels."""
    return _parse_positive(raw, "MOJO_ENGINE_ATTN_TILE_POSITIONS")


def parse_attn_max_splits(raw: String) raises -> Int:
    """MOJO_ENGINE_ATTN_MAX_SPLITS: the split-KV cap."""
    return _parse_positive(raw, "MOJO_ENGINE_ATTN_MAX_SPLITS")


def parse_attn_split_span(raw: String) raises -> Int:
    """MOJO_ENGINE_ATTN_SPLIT_SPAN_POSITIONS: stored positions
    per split-KV partial (splits engage past one span)."""
    return _parse_positive(raw, "MOJO_ENGINE_ATTN_SPLIT_SPAN_POSITIONS")


def parse_prefill_gemm_min_batch(raw: String) raises -> Int:
    """Parse and validate a prefill GEMM minimum batch value.

    Split out of load() so the refusal names are directly
    testable, exactly like parse_context_len. Raises
    mojo_engine_config on anything that is not a positive
    whole number.
    """
    return _parse_positive(raw, "MOJO_ENGINE_PREFILL_GEMM_MIN_BATCH")


def parse_gemm_tiled_min_batch(raw: String) raises -> Int:
    """Parse and validate a tiled-GEMM minimum batch value.

    Split out of load() so the refusal names are directly
    testable, exactly like parse_context_len. Raises
    mojo_engine_config on anything that is not a positive
    whole number.
    """
    return _parse_positive(raw, "MOJO_ENGINE_GEMM_TILED_MIN_BATCH")


struct EngineConfig(Copyable, Movable):
    """One loaded configuration; build via EngineConfig.load()."""

    var model_path: String
    var context_len: Int
    var gpu_budget_bytes: Int
    var attn_tile_positions: Int
    var attn_max_splits: Int
    var attn_split_span_positions: Int
    var prefill_gemm_min_batch: Int
    var gemm_tiled_min_batch: Int

    def __init__(
        out self,
        model_path: String,
        context_len: Int,
        gpu_budget_bytes: Int,
        attn_tile_positions: Int = DEFAULT_ATTN_TILE_POSITIONS,
        attn_max_splits: Int = DEFAULT_ATTN_MAX_SPLITS,
        attn_split_span_positions: Int = DEFAULT_ATTN_SPLIT_SPAN_POSITIONS,
        prefill_gemm_min_batch: Int = DEFAULT_PREFILL_GEMM_MIN_BATCH,
        gemm_tiled_min_batch: Int = DEFAULT_GEMM_TILED_MIN_BATCH,
    ):
        self.model_path = model_path
        self.context_len = context_len
        self.gpu_budget_bytes = gpu_budget_bytes
        self.attn_tile_positions = attn_tile_positions
        self.attn_max_splits = attn_max_splits
        self.attn_split_span_positions = attn_split_span_positions
        self.prefill_gemm_min_batch = prefill_gemm_min_batch
        self.gemm_tiled_min_batch = gemm_tiled_min_batch

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

        var attn_tile = DEFAULT_ATTN_TILE_POSITIONS
        var raw_tile = getenv("MOJO_ENGINE_ATTN_TILE_POSITIONS")
        if raw_tile != "":
            attn_tile = parse_attn_tile_positions(raw_tile)

        var attn_splits = DEFAULT_ATTN_MAX_SPLITS
        var raw_splits = getenv("MOJO_ENGINE_ATTN_MAX_SPLITS")
        if raw_splits != "":
            attn_splits = parse_attn_max_splits(raw_splits)

        var attn_span = DEFAULT_ATTN_SPLIT_SPAN_POSITIONS
        var raw_span = getenv("MOJO_ENGINE_ATTN_SPLIT_SPAN_POSITIONS")
        if raw_span != "":
            attn_span = parse_attn_split_span(raw_span)

        var prefill_gemm_min_batch = DEFAULT_PREFILL_GEMM_MIN_BATCH
        var raw_min = getenv("MOJO_ENGINE_PREFILL_GEMM_MIN_BATCH")
        if raw_min != "":
            prefill_gemm_min_batch = parse_prefill_gemm_min_batch(raw_min)

        var gemm_tiled_min_batch = DEFAULT_GEMM_TILED_MIN_BATCH
        var raw_tiled = getenv("MOJO_ENGINE_GEMM_TILED_MIN_BATCH")
        if raw_tiled != "":
            gemm_tiled_min_batch = parse_gemm_tiled_min_batch(raw_tiled)

        return EngineConfig(
            model_path,
            context_len,
            gpu_budget_bytes,
            attn_tile,
            attn_splits,
            attn_span,
            prefill_gemm_min_batch,
            gemm_tiled_min_batch,
        )
