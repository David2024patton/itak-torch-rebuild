# config.mojo - engine configuration.
#
# David's law: operational limits are config with defaults, never
# compiled-in constants scattered through the code. Every setting
# below lives in exactly one place (the DEFAULT_* block), each one
# overridable by an env var at launch, and a bad value is refused
# BY NAME at load time instead of being silently clamped or
# ignored. Nothing else in the engine reads env vars directly.

from std.math import isnan, isinf
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

# =====================================================================
# BEGIN Wave E Lane E3 block: Glimmer op defaults, folded in
# from src/ops/ (Wave A open item). Keep this block together:
# lane E1 also edits this file and the merge is meant to stay
# mechanical.
#
# Before this fold, the ops modules each carried their own
# recorded-Glimmer defaults as GLIMMER_* comptime constants
# (rms_norm.mojo GLIMMER_RMS_EPS, rope.mojo GLIMMER_ROPE_BASE /
# GLIMMER_HEAD_DIM, final_logit.mojo GLIMMER_LOGIT_SCALE /
# GLIMMER_LOGIT_SOFTCAP, embedding.mojo GLIMMER_HIDDEN /
# GLIMMER_VOCAB). The defaults now live ONLY here, as
# EngineConfig fields with MOJO_ENGINE_* env overrides, and
# the ops take these values as parameters as they always did -
# callers that name the recorded model pass the config values.
#
# Precedence, stated plainly so nobody misreads this fold as
# a behavior change: for a LOADED model, the values in effect
# are the ones GlimmerModel reads from the GGUF file itself
# (ModelConfig in src/model/model.mojo: rms_eps, rope_base,
# logit_scale, logit_softcap, hidden, head_dim, vocab), and
# those are unchanged by this fold and do NOT read these
# fields. These EngineConfig fields are the recorded defaults
# for callers/tests that name the Glimmer model without a
# file in hand - exactly what the GLIMMER_* constants served.
# The numeric values are byte-for-byte the old constants.
# =====================================================================
comptime DEFAULT_RMS_EPS = Float32(1.0e-5)
comptime DEFAULT_ROPE_BASE = Float32(500000.0)
comptime DEFAULT_LOGIT_SCALE = Float32(0.1961161345243454)
comptime DEFAULT_LOGIT_SOFTCAP = Float32(20.0)
comptime DEFAULT_HEAD_DIM = 128
comptime DEFAULT_HIDDEN = 6656
comptime DEFAULT_VOCAB = 202048
# ===== END Wave E Lane E3 block (defaults) =====

# Prefill routing knob (Wave E, Lane E1): a prompt at or
# above PREFILL_BATCH_MIN_TOKENS tokens takes the batched
# prefill route (GEMM projections + resident batch layer
# step + batched tiled attention); below it, prefill runs
# the per-row resident step loop. The route is taken only
# when BOTH this knob and PREFILL_GEMM_MIN_BATCH admit the
# prompt, so the Wave D A/B control (raising the GEMM
# minimum above any prompt length) still forces the
# per-row arm on its own. The default is set from the
# measured batch-vs-per-row crossover on the real model;
# the trial record is
# bench/mojo-engine-wave-e1-batch-2026-11-08.md.
comptime DEFAULT_PREFILL_BATCH_MIN_TOKENS = 64


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


def parse_prefill_batch_min_tokens(raw: String) raises -> Int:
    """Parse and validate a prefill batch-route minimum
    token count (Wave E): prompts at or above it take the
    batched prefill route."""
    return _parse_positive(raw, "MOJO_ENGINE_PREFILL_BATCH_MIN_TOKENS")


def prefill_uses_batch(
    prompt_len: Int, gemm_min_batch: Int, batch_min_tokens: Int
) -> Bool:
    """The prefill routing rule (pure, VM-testable): the
    batched route serves a prompt only when its length
    reaches BOTH the GEMM minimum batch (the Wave D A/B
    control) and the batch-route minimum token count (the
    Wave E crossover knob)."""
    return prompt_len >= gemm_min_batch and prompt_len >= batch_min_tokens


def parse_gemm_tiled_min_batch(raw: String) raises -> Int:
    """Parse and validate a tiled-GEMM minimum batch value.

    Split out of load() so the refusal names are directly
    testable, exactly like parse_context_len. Raises
    mojo_engine_config on anything that is not a positive
    whole number.
    """
    return _parse_positive(raw, "MOJO_ENGINE_GEMM_TILED_MIN_BATCH")


# =====================================================================
# BEGIN Wave E Lane E3 block: parsers for the Glimmer op
# defaults above. Same discipline as the other values here:
# split out of load() so the refusal names are directly
# testable, every bad value refused as mojo_engine_config
# naming its env var.
# =====================================================================


def _parse_float(raw: String, env_name: String) raises -> Float32:
    """Shared finite-float parse for the op-default knobs.

    Float64(raw) refuses non-numeric text itself; NaN and the
    infinities PARSE in Mojo, so they are refused here by
    name, matching the ops' own bad-value rules (their CPU
    twins refuse non-finite eps/base/scale/cap by name too).
    """
    var parsed: Float64
    try:
        parsed = Float64(raw)
    except:
        raise Error(
            "mojo_engine_config: "
            + env_name
            + " is not a number: '"
            + raw
            + "'"
        )
    if isnan(parsed) or isinf(parsed):
        raise Error(
            "mojo_engine_config: "
            + env_name
            + " must be finite, got '"
            + raw
            + "'"
        )
    return Float32(parsed)


def parse_rms_eps(raw: String) raises -> Float32:
    """MOJO_ENGINE_RMS_EPS: RMSNorm epsilon; the op refuses
    eps <= 0 by name, so config does too."""
    var v = _parse_float(raw, "MOJO_ENGINE_RMS_EPS")
    if v <= Float32(0.0):
        raise Error(
            "mojo_engine_config: MOJO_ENGINE_RMS_EPS must be positive, got "
            + String(v)
        )
    return v


def parse_rope_base(raw: String) raises -> Float32:
    """MOJO_ENGINE_ROPE_BASE: RoPE frequency base; the op
    refuses base <= 1 by name, so config does too."""
    var v = _parse_float(raw, "MOJO_ENGINE_ROPE_BASE")
    if v <= Float32(1.0):
        raise Error(
            "mojo_engine_config: MOJO_ENGINE_ROPE_BASE must be > 1, got "
            + String(v)
        )
    return v


def parse_logit_scale(raw: String) raises -> Float32:
    """MOJO_ENGINE_LOGIT_SCALE: final-logit scale; any finite
    value is legal (the op refuses only non-finite)."""
    return _parse_float(raw, "MOJO_ENGINE_LOGIT_SCALE")


def parse_logit_softcap(raw: String) raises -> Float32:
    """MOJO_ENGINE_LOGIT_SOFTCAP: final-logit softcap; the op
    refuses cap < 0 by name (0 = no cap), so config does too.
    """
    var v = _parse_float(raw, "MOJO_ENGINE_LOGIT_SOFTCAP")
    if v < Float32(0.0):
        raise Error(
            "mojo_engine_config: MOJO_ENGINE_LOGIT_SOFTCAP must be"
            " >= 0 (0 = no cap), got "
            + String(v)
        )
    return v


def parse_head_dim(raw: String) raises -> Int:
    """MOJO_ENGINE_HEAD_DIM: attention head dimension."""
    return _parse_positive(raw, "MOJO_ENGINE_HEAD_DIM")


def parse_hidden(raw: String) raises -> Int:
    """MOJO_ENGINE_HIDDEN: model hidden width."""
    return _parse_positive(raw, "MOJO_ENGINE_HIDDEN")


def parse_vocab(raw: String) raises -> Int:
    """MOJO_ENGINE_VOCAB: tokenizer vocabulary size."""
    return _parse_positive(raw, "MOJO_ENGINE_VOCAB")


# ===== END Wave E Lane E3 block (parsers) =====


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
    # -- BEGIN Wave E Lane E3 fields: Glimmer op defaults --
    var rms_eps: Float32
    var rope_base: Float32
    var logit_scale: Float32
    var logit_softcap: Float32
    var head_dim: Int
    var hidden: Int
    var vocab: Int
    # -- END Wave E Lane E3 fields --
    var prefill_batch_min_tokens: Int

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
        # -- BEGIN Wave E Lane E3 init params --
        rms_eps: Float32 = DEFAULT_RMS_EPS,
        rope_base: Float32 = DEFAULT_ROPE_BASE,
        logit_scale: Float32 = DEFAULT_LOGIT_SCALE,
        logit_softcap: Float32 = DEFAULT_LOGIT_SOFTCAP,
        head_dim: Int = DEFAULT_HEAD_DIM,
        hidden: Int = DEFAULT_HIDDEN,
        vocab: Int = DEFAULT_VOCAB,
        # -- END Wave E Lane E3 init params --
        prefill_batch_min_tokens: Int = DEFAULT_PREFILL_BATCH_MIN_TOKENS,
    ):
        self.model_path = model_path
        self.context_len = context_len
        self.gpu_budget_bytes = gpu_budget_bytes
        self.attn_tile_positions = attn_tile_positions
        self.attn_max_splits = attn_max_splits
        self.attn_split_span_positions = attn_split_span_positions
        self.prefill_gemm_min_batch = prefill_gemm_min_batch
        self.gemm_tiled_min_batch = gemm_tiled_min_batch
        # -- BEGIN Wave E Lane E3 init assignments --
        self.rms_eps = rms_eps
        self.rope_base = rope_base
        self.logit_scale = logit_scale
        self.logit_softcap = logit_softcap
        self.head_dim = head_dim
        self.hidden = hidden
        self.vocab = vocab
        # -- END Wave E Lane E3 init assignments --
        self.prefill_batch_min_tokens = prefill_batch_min_tokens

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

        # -- BEGIN Wave E Lane E3 load: Glimmer op defaults --
        var rms_eps = DEFAULT_RMS_EPS
        var raw_eps = getenv("MOJO_ENGINE_RMS_EPS")
        if raw_eps != "":
            rms_eps = parse_rms_eps(raw_eps)

        var rope_base = DEFAULT_ROPE_BASE
        var raw_base = getenv("MOJO_ENGINE_ROPE_BASE")
        if raw_base != "":
            rope_base = parse_rope_base(raw_base)

        var logit_scale = DEFAULT_LOGIT_SCALE
        var raw_scale = getenv("MOJO_ENGINE_LOGIT_SCALE")
        if raw_scale != "":
            logit_scale = parse_logit_scale(raw_scale)

        var logit_softcap = DEFAULT_LOGIT_SOFTCAP
        var raw_cap = getenv("MOJO_ENGINE_LOGIT_SOFTCAP")
        if raw_cap != "":
            logit_softcap = parse_logit_softcap(raw_cap)

        var head_dim = DEFAULT_HEAD_DIM
        var raw_hd = getenv("MOJO_ENGINE_HEAD_DIM")
        if raw_hd != "":
            head_dim = parse_head_dim(raw_hd)

        var hidden = DEFAULT_HIDDEN
        var raw_hidden = getenv("MOJO_ENGINE_HIDDEN")
        if raw_hidden != "":
            hidden = parse_hidden(raw_hidden)

        var vocab = DEFAULT_VOCAB
        var raw_vocab = getenv("MOJO_ENGINE_VOCAB")
        if raw_vocab != "":
            vocab = parse_vocab(raw_vocab)
        # -- END Wave E Lane E3 load --

        var prefill_batch_min_tokens = DEFAULT_PREFILL_BATCH_MIN_TOKENS
        var raw_route = getenv("MOJO_ENGINE_PREFILL_BATCH_MIN_TOKENS")
        if raw_route != "":
            prefill_batch_min_tokens = parse_prefill_batch_min_tokens(raw_route)

        return EngineConfig(
            model_path,
            context_len,
            gpu_budget_bytes,
            attn_tile,
            attn_splits,
            attn_span,
            prefill_gemm_min_batch,
            gemm_tiled_min_batch,
            # -- BEGIN Wave E Lane E3 load return values --
            rms_eps,
            rope_base,
            logit_scale,
            logit_softcap,
            head_dim,
            hidden,
            vocab,
            # -- END Wave E Lane E3 load return values --
            prefill_batch_min_tokens,
        )
