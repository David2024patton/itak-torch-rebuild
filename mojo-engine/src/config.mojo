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
comptime ENGINE_VERSION = "1.7"

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
# per-row arm on its own. The default is the measured
# batch-vs-per-row crossover on the real model
# (2026-11-08, RX 7900 XTX, interleaved fresh-model
# arms): at 6 tokens the batch arm loses (286 ms vs the
# per-row arm's 218 ms); at 16 tokens it wins (364 ms vs
# 557 ms) and it stays ahead at every larger measured
# length (32/64/128/256/512/1022). So prompts below 16
# tokens prefill per-row and prompts at or above 16 take
# the batch route. The trial record is
# bench/mojo-engine-wave-e1-batch-2026-11-08.md.
comptime DEFAULT_PREFILL_BATCH_MIN_TOKENS = 16

# Serving seam knobs (DESIGN.md step 7, decided 2026-11-08:
# the endpoint is Mojo-only, an HTTP server inside the engine
# process; see src/serve/). The default port is deliberately
# NOT 8080: production Glimmer stays down and port 8080 stays
# clear. The host default is loopback only; serving beyond the
# host is an operator's explicit env choice. Body and token
# limits are config for the same reason every other limit is:
# a request past them refuses BY NAME (serve_body_too_large /
# serve_max_tokens_exceeded), never silently truncates.
comptime DEFAULT_SERVE_HOST = "127.0.0.1"
comptime DEFAULT_SERVE_PORT = 18767
comptime DEFAULT_SERVE_MAX_BODY_BYTES = 65536
comptime DEFAULT_SERVE_DEFAULT_MAX_TOKENS = 64
comptime DEFAULT_SERVE_MAX_TOKENS = 512
comptime DEFAULT_SERVE_DEBUG_RING_LINES = 200
comptime DEFAULT_SERVE_MAX_CONTEXT_BYTES = 32768
# History / fact-sheet / fingerprint knobs (amendments,
# David 2026-11-08): history persists as JSON lines under
# serve_history_dir (relative paths are under the process
# working directory, which is mojo-engine/ in the bench
# harness); the fact sheet registry dir is scanned for
# *.gguf files whose fact sheets /models serves (the served
# model is always listed first, read from its own file).
# backend_target is a config STRING on purpose: the chip +
# runtime the measurements were taken on is an operator fact
# (gfx1100 / MAX 26.7 on David's card, an sm_XX / MAX string
# on Chris's), recorded onto every run record, never guessed
# or probed. attach_warn_tokens is the dashboard's client-side
# archive budget pre-check threshold (Piece 3 guardrail): the
# page enforces it, the value is published so the page and the
# operator agree on one number.
comptime DEFAULT_SERVE_HISTORY_DIR = "serve_history"
comptime DEFAULT_SERVE_MODELS_DIR = "/mnt/models"
comptime DEFAULT_SERVE_BACKEND_TARGET = "gfx1100 / MAX 26.7"
comptime DEFAULT_SERVE_ATTACH_WARN_TOKENS = 4096


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


def parse_serve_port(raw: String) raises -> Int:
    """MOJO_ENGINE_SERVE_PORT: TCP port, 1..65535."""
    var v = _parse_positive(raw, "MOJO_ENGINE_SERVE_PORT")
    if v > 65535:
        raise Error(
            "mojo_engine_config: MOJO_ENGINE_SERVE_PORT must be <= 65535, got "
            + String(v)
        )
    return v


def parse_serve_max_body_bytes(raw: String) raises -> Int:
    """MOJO_ENGINE_SERVE_MAX_BODY_BYTES: request body cap."""
    return _parse_positive(raw, "MOJO_ENGINE_SERVE_MAX_BODY_BYTES")


def parse_serve_default_max_tokens(raw: String) raises -> Int:
    """MOJO_ENGINE_SERVE_DEFAULT_MAX_TOKENS: chat completions
    without an explicit max_tokens generate this many."""
    return _parse_positive(raw, "MOJO_ENGINE_SERVE_DEFAULT_MAX_TOKENS")


def parse_serve_max_tokens(raw: String) raises -> Int:
    """MOJO_ENGINE_SERVE_MAX_TOKENS: hard cap on a chat
    request's max_tokens; past it the request refuses."""
    return _parse_positive(raw, "MOJO_ENGINE_SERVE_MAX_TOKENS")


def parse_ipv4(host: String) raises -> List[Int]:
    """Parse a dotted-quad IPv4 host into its 4 octets.

    The serve socket binds an IPv4 address; hostnames are
    refused BY NAME as mojo_engine_config instead of being
    resolved by a hidden DNS call. Pure and testable: digits
    accumulate numerically, so no substring surgery is needed.
    """
    var bytes = host.as_bytes()
    var parts = List[Int]()
    var value = 0
    var digits = 0
    for i in range(len(bytes) + 1):
        if i == len(bytes) or bytes[i] == 46:  # '.'
            if digits == 0 or digits > 3 or value > 255:
                raise Error(
                    "mojo_engine_config: MOJO_ENGINE_SERVE_HOST is not"
                    " a dotted-quad IPv4 address: '"
                    + host
                    + "'"
                )
            parts.append(value)
            value = 0
            digits = 0
        else:
            var b = Int(bytes[i])
            if b < 48 or b > 57:
                raise Error(
                    "mojo_engine_config: MOJO_ENGINE_SERVE_HOST is not"
                    " a dotted-quad IPv4 address: '"
                    + host
                    + "'"
                )
            value = value * 10 + (b - 48)
            digits += 1
    if len(parts) != 4:
        raise Error(
            "mojo_engine_config: MOJO_ENGINE_SERVE_HOST is not"
            " a dotted-quad IPv4 address: '"
            + host
            + "'"
        )
    return parts^


def parse_serve_debug_ring_lines(raw: String) raises -> Int:
    """MOJO_ENGINE_SERVE_DEBUG_RING_LINES: /debug/log ring size."""
    return _parse_positive(raw, "MOJO_ENGINE_SERVE_DEBUG_RING_LINES")


def parse_serve_max_context_bytes(raw: String) raises -> Int:
    """MOJO_ENGINE_SERVE_MAX_CONTEXT_BYTES: cap on attachment
    context text accepted with a chat request."""
    return _parse_positive(raw, "MOJO_ENGINE_SERVE_MAX_CONTEXT_BYTES")


def parse_serve_attach_warn_tokens(raw: String) raises -> Int:
    """MOJO_ENGINE_SERVE_ATTACH_WARN_TOKENS: the dashboard's
    client-side token-budget pre-check threshold for archive /
    document context (Piece 3 guardrail, amendments)."""
    return _parse_positive(raw, "MOJO_ENGINE_SERVE_ATTACH_WARN_TOKENS")


def parse_serve_dir(raw: String, env_name: String) raises -> String:
    """A directory path knob (history dir, models dir): any
    non-empty path is accepted as text; existence is checked
    by the consumer at use time with its own named refusal."""
    if raw == "":
        raise Error(
            "mojo_engine_config: " + env_name + " must not be empty"
        )
    return raw


def parse_serve_backend_target(raw: String) raises -> String:
    """MOJO_ENGINE_SERVE_BACKEND_TARGET: the chip + runtime
    string stamped onto every run record (an operator fact,
    never probed)."""
    if raw == "":
        raise Error(
            "mojo_engine_config: MOJO_ENGINE_SERVE_BACKEND_TARGET"
            " must not be empty"
        )
    return raw


def parse_serve_host(raw: String) raises -> String:
    """MOJO_ENGINE_SERVE_HOST: dotted-quad IPv4 only (v1)."""
    _ = parse_ipv4(raw)
    return raw


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
    # -- Serving seam fields (step 7) --
    var serve_host: String
    var serve_port: Int
    var serve_max_body_bytes: Int
    var serve_default_max_tokens: Int
    var serve_max_tokens: Int
    var serve_debug_ring_lines: Int
    var serve_max_context_bytes: Int
    var serve_history_dir: String
    var serve_models_dir: String
    var serve_backend_target: String
    var serve_attach_warn_tokens: Int

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
        serve_host: String = DEFAULT_SERVE_HOST,
        serve_port: Int = DEFAULT_SERVE_PORT,
        serve_max_body_bytes: Int = DEFAULT_SERVE_MAX_BODY_BYTES,
        serve_default_max_tokens: Int = DEFAULT_SERVE_DEFAULT_MAX_TOKENS,
        serve_max_tokens: Int = DEFAULT_SERVE_MAX_TOKENS,
        serve_debug_ring_lines: Int = DEFAULT_SERVE_DEBUG_RING_LINES,
        serve_max_context_bytes: Int = DEFAULT_SERVE_MAX_CONTEXT_BYTES,
        serve_history_dir: String = DEFAULT_SERVE_HISTORY_DIR,
        serve_models_dir: String = DEFAULT_SERVE_MODELS_DIR,
        serve_backend_target: String = DEFAULT_SERVE_BACKEND_TARGET,
        serve_attach_warn_tokens: Int = DEFAULT_SERVE_ATTACH_WARN_TOKENS,
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
        self.serve_host = serve_host
        self.serve_port = serve_port
        self.serve_max_body_bytes = serve_max_body_bytes
        self.serve_default_max_tokens = serve_default_max_tokens
        self.serve_max_tokens = serve_max_tokens
        self.serve_debug_ring_lines = serve_debug_ring_lines
        self.serve_max_context_bytes = serve_max_context_bytes
        self.serve_history_dir = serve_history_dir
        self.serve_models_dir = serve_models_dir
        self.serve_backend_target = serve_backend_target
        self.serve_attach_warn_tokens = serve_attach_warn_tokens

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

        var serve_host = DEFAULT_SERVE_HOST
        var raw_host = getenv("MOJO_ENGINE_SERVE_HOST")
        if raw_host != "":
            serve_host = parse_serve_host(raw_host)

        var serve_port = DEFAULT_SERVE_PORT
        var raw_port = getenv("MOJO_ENGINE_SERVE_PORT")
        if raw_port != "":
            serve_port = parse_serve_port(raw_port)

        var serve_max_body = DEFAULT_SERVE_MAX_BODY_BYTES
        var raw_body = getenv("MOJO_ENGINE_SERVE_MAX_BODY_BYTES")
        if raw_body != "":
            serve_max_body = parse_serve_max_body_bytes(raw_body)

        var serve_default_max_tokens = DEFAULT_SERVE_DEFAULT_MAX_TOKENS
        var raw_dmt = getenv("MOJO_ENGINE_SERVE_DEFAULT_MAX_TOKENS")
        if raw_dmt != "":
            serve_default_max_tokens = parse_serve_default_max_tokens(raw_dmt)

        var serve_max_tokens = DEFAULT_SERVE_MAX_TOKENS
        var raw_mt = getenv("MOJO_ENGINE_SERVE_MAX_TOKENS")
        if raw_mt != "":
            serve_max_tokens = parse_serve_max_tokens(raw_mt)
        var serve_debug_ring_lines = DEFAULT_SERVE_DEBUG_RING_LINES
        var raw_ring = getenv("MOJO_ENGINE_SERVE_DEBUG_RING_LINES")
        if raw_ring != "":
            serve_debug_ring_lines = parse_serve_debug_ring_lines(raw_ring)

        var serve_max_context_bytes = DEFAULT_SERVE_MAX_CONTEXT_BYTES
        var raw_ctx_bytes = getenv("MOJO_ENGINE_SERVE_MAX_CONTEXT_BYTES")
        if raw_ctx_bytes != "":
            serve_max_context_bytes = parse_serve_max_context_bytes(
                raw_ctx_bytes
            )

        var serve_history_dir = DEFAULT_SERVE_HISTORY_DIR
        var raw_hist = getenv("MOJO_ENGINE_SERVE_HISTORY_DIR")
        if raw_hist != "":
            serve_history_dir = parse_serve_dir(
                raw_hist, "MOJO_ENGINE_SERVE_HISTORY_DIR"
            )

        var serve_models_dir = DEFAULT_SERVE_MODELS_DIR
        var raw_models = getenv("MOJO_ENGINE_SERVE_MODELS_DIR")
        if raw_models != "":
            serve_models_dir = parse_serve_dir(
                raw_models, "MOJO_ENGINE_SERVE_MODELS_DIR"
            )

        var serve_backend_target = DEFAULT_SERVE_BACKEND_TARGET
        var raw_backend = getenv("MOJO_ENGINE_SERVE_BACKEND_TARGET")
        if raw_backend != "":
            serve_backend_target = parse_serve_backend_target(raw_backend)

        var serve_attach_warn_tokens = DEFAULT_SERVE_ATTACH_WARN_TOKENS
        var raw_warn = getenv("MOJO_ENGINE_SERVE_ATTACH_WARN_TOKENS")
        if raw_warn != "":
            serve_attach_warn_tokens = parse_serve_attach_warn_tokens(raw_warn)

        if serve_default_max_tokens > serve_max_tokens:
            raise Error(
                "mojo_engine_config: MOJO_ENGINE_SERVE_DEFAULT_MAX_TOKENS"
                " exceeds MOJO_ENGINE_SERVE_MAX_TOKENS"
            )

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
            serve_host,
            serve_port,
            serve_max_body,
            serve_default_max_tokens,
            serve_max_tokens,
            serve_debug_ring_lines,
            serve_max_context_bytes,
            serve_history_dir,
            serve_models_dir,
            serve_backend_target,
            serve_attach_warn_tokens,
        )
