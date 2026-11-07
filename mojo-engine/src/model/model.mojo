# model.mojo - the Glimmer transformer forward, Lane B3 (CPU path).
#
# Semantic reference (Rust, branch main), read before this file:
#   crates/muse-glimmer/src/model.rs forward_rows / finish_last
#   (the ONLY correct step order; mirrored statement by statement
#   below), crates/muse-glimmer/src/config.rs (metadata keys),
#   crates/muse-glimmer/src/layer.rs derive_layer_pattern (the
#   file's sliding_window_pattern bool array wins; the %4==3
#   pattern is the fallback), crates/muse-glimmer/src/weights.rs
#   (the 14 per-layer tensor suffixes), crates/cpu-compute (RoPE
#   at write time: the cache holds ROTATED K; attn_gate =
#   attn_out * sigmoid(gate); POST_NORM_EPS 1e-8).
#
# Shape of the computation: the Wave A GEMV CPU twins are
# single-vector GEMVs, so a forward over N positions is N
# per-position steps. step() runs ONE token at ONE absolute
# position through embedding, all layers, and returns the final
# hidden row; forward_rows() is the step loop. B4's generate()
# drives prefill through the ForwardModel adapter at the bottom
# of this file. There is no batched prefill in Wave B.
#
# Weight bytes are CACHED in the model at construction (every
# tensor read once through the Wave A loader's staged reads).
# The loader serves bytes per call; re-reading a 74 MB FFN
# tensor from disk for every token would dominate even this
# correctness wave's runtimes. Memory cost is the file's own
# size (~17 GB for Glimmer) held as host lists; stated plainly
# so the serving wave can revisit it.
#
# RoPE decision (coordinator, mirroring the Rust CPU path):
# sliding layers rotate Q and K at the position BEFORE the K
# row is appended; the cache holds rotated K; attention runs
# through attention_cpu_rotated_cache. Globals never rotate.
#
# RUNG-1 FINDING (2026-11-07, coordinator), RESOLVED same
# day: the sliding tiny fixture (window 4, positions 0..2,
# never wraps) diverged from the Float64 reference at
# max_rel 4.37 while the global fixture matched at 5.8e-05.
# Root cause: this file rotated q itself AND passed it to
# attention_cpu_rotated_cache, which rotates the raw q row
# internally (its docstring says so) -- a double rotation.
# Fix: q passes raw; only K rotates here, at write time.
#
# Named refusals: model_missing_metadata, model_bad_metadata,
# model_pattern_mismatch, model_missing_tensor,
# model_unsupported_dtype, model_bad_shape, model_bad_token.

from std.math import exp

from diag import log_debug
from gguf.loader import GgufFile, parse_gguf, f32_from_bits
from kv.cache import KvCache, LayerKind, glimmer_layer_pattern
from attention.attention import attention_cpu_rotated_cache, ROPE_NONE
from kernels.k1_gemv import k1_gemv_cpu
from kernels.q6k_gemv import q6k_gemv_cpu
from kernels.q5k_gemv import q5k_gemv_cpu
from kernels.f32_gemv import f32_gemv_cpu, f16_gemv_cpu
from kernels.dequant import q6k_element_list, q5k_element_list, read_f16_list
from ops.rms_norm import rms_norm_cpu, rms_norm_plain_cpu
from ops.rope import rope_cpu
from ops.swiglu import swiglu_cpu
from ops.final_logit import final_logit_cpu
from ops.residual import residual_add_cpu
from generate.generate import ForwardModel

comptime POST_NORM_EPS = Float32(1.0e-8)

# ggml type ids (see gguf/loader.mojo ggml_type_name).
comptime DT_F32 = 0
comptime DT_F16 = 1
comptime DT_Q4_K = 12
comptime DT_Q5_K = 13
comptime DT_Q6_K = 14

comptime LAYER_SUFFIX_COUNT = 14


def layer_suffix(i: Int) raises -> String:
    """The 14 per-layer tensor suffixes, weights.rs order."""
    if i == 0:
        return "attn_norm.weight"
    if i == 1:
        return "attn_q.weight"
    if i == 2:
        return "attn_k.weight"
    if i == 3:
        return "attn_v.weight"
    if i == 4:
        return "attn_q_norm.weight"
    if i == 5:
        return "attn_k_norm.weight"
    if i == 6:
        return "attn_output.weight"
    if i == 7:
        return "attn_gate.weight"
    if i == 8:
        return "post_attention_norm.weight"
    if i == 9:
        return "ffn_norm.weight"
    if i == 10:
        return "ffn_gate.weight"
    if i == 11:
        return "ffn_up.weight"
    if i == 12:
        return "ffn_down.weight"
    if i == 13:
        return "post_ffw_norm.weight"
    raise Error("model_bad_shape: layer suffix index " + String(i))


def _sigmoid(v: Float32) -> Float32:
    """Branch-stable sigmoid, the cpu-compute shape."""
    if v >= Float32(0.0):
        return Float32(1.0) / (Float32(1.0) + exp(-v))
    var e = exp(v)
    return e / (Float32(1.0) + e)


def _bytes_to_f32(w: List[UInt8]) raises -> List[Float32]:
    if len(w) % 4 != 0:
        raise Error(
            "model_bad_shape: F32 tensor byte length "
            + String(len(w))
            + " is not a multiple of 4"
        )
    var out = List[Float32](capacity=len(w) // 4)
    out.resize(len(w) // 4, Float32(0.0))
    for i in range(len(out)):
        var bits = (
            UInt32(w[4 * i])
            | (UInt32(w[4 * i + 1]) << 8)
            | (UInt32(w[4 * i + 2]) << 16)
            | (UInt32(w[4 * i + 3]) << 24)
        )
        out[i] = f32_from_bits(bits)
    return out^


def _q4k_element(w: List[UInt8], sb_off: Int, within: Int) -> Float32:
    """One Q4_K element, k1_gemv_cpu's inner math copied term for
    term (dequant.mojo's scale_min_k4_list takes the group index
    as its first argument): value = d * sc * q - dmin * m."""
    from kernels.dequant import scale_min_k4_list, f16_to_f32

    var d_bits = UInt16(w[sb_off]) | (UInt16(w[sb_off + 1]) << 8)
    var dm_bits = UInt16(w[sb_off + 2]) | (UInt16(w[sb_off + 3]) << 8)
    var d = f16_to_f32(d_bits)
    var dmin = f16_to_f32(dm_bits)
    var g = within >> 6
    var p = within & 63
    var sc: Float32
    var m: Float32
    var q: Float32
    if p < 32:
        var sm = scale_min_k4_list(2 * g, w, sb_off)
        sc = sm[0]
        m = sm[1]
        q = Float32(Int(w[sb_off + 16 + g * 32 + p]) & 0xF)
    else:
        var sm2 = scale_min_k4_list(2 * g + 1, w, sb_off)
        sc = sm2[0]
        m = sm2[1]
        q = Float32(Int(w[sb_off + 16 + g * 32 + (p - 32)]) >> 4)
    return d * sc * q - dmin * m


struct ModelConfig(Copyable, Movable):
    var hidden: Int
    var ffn: Int
    var layers: Int
    var q_heads: Int
    var kv_heads: Int
    var head_dim: Int
    var vocab: Int
    var context_len: Int
    var window: Int
    var rms_eps: Float32
    var rope_base: Float32
    var rope_base_swa: Float32
    var logit_scale: Float32
    var logit_softcap: Float32
    var kinds: List[LayerKind]

    def __init__(
        out self,
        hidden: Int,
        ffn: Int,
        layers: Int,
        q_heads: Int,
        kv_heads: Int,
        head_dim: Int,
        vocab: Int,
        context_len: Int,
        window: Int,
        rms_eps: Float32,
        rope_base: Float32,
        rope_base_swa: Float32,
        logit_scale: Float32,
        logit_softcap: Float32,
        var kinds: List[LayerKind],
    ):
        self.hidden = hidden
        self.ffn = ffn
        self.layers = layers
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.vocab = vocab
        self.context_len = context_len
        self.window = window
        self.rms_eps = rms_eps
        self.rope_base = rope_base
        self.rope_base_swa = rope_base_swa
        self.logit_scale = logit_scale
        self.logit_softcap = logit_softcap
        self.kinds = kinds^


def load_model_config(gf: GgufFile) raises -> ModelConfig:
    """Read the model config from GGUF metadata, GlimmerConfig's
    mirror. Refuses by name on missing or implausible values."""
    var arch = gf.architecture()
    if arch != "muse-glimmer":
        raise Error(
            "model_bad_metadata: general.architecture is '"
            + arch
            + "', this engine serves muse-glimmer"
        )
    var layers = Int(gf.get_u32("muse-glimmer.block_count"))
    var context_len = Int(gf.get_u32("muse-glimmer.context_length"))
    var hidden = Int(gf.get_u32("muse-glimmer.embedding_length"))
    var ffn = Int(gf.get_u32("muse-glimmer.feed_forward_length"))
    var q_heads = Int(gf.get_u32("muse-glimmer.attention.head_count"))
    var kv_heads = Int(gf.get_u32("muse-glimmer.attention.head_count_kv"))
    var key_len = Int(gf.get_u32("muse-glimmer.attention.key_length"))
    var val_len = Int(gf.get_u32("muse-glimmer.attention.value_length"))
    if key_len != val_len:
        raise Error(
            "model_bad_metadata: key_length "
            + String(key_len)
            + " != value_length "
            + String(val_len)
            + "; one head_dim cannot serve both"
        )
    if layers <= 0 or layers > 4096:
        raise Error(
            "model_bad_metadata: block_count " + String(layers) + " implausible"
        )
    if hidden <= 0 or ffn <= 0 or q_heads <= 0 or kv_heads <= 0 or key_len <= 0:
        raise Error("model_bad_metadata: a model dimension is not positive")
    if q_heads % kv_heads != 0:
        raise Error(
            "model_bad_metadata: head_count not divisible by head_count_kv"
        )
    var window = Int(gf.get_u32("muse-glimmer.attention.sliding_window"))
    if window <= 0 or window > context_len:
        raise Error(
            "model_bad_metadata: sliding_window "
            + String(window)
            + " outside 1..context_length"
        )
    var rms_eps = gf.get_f32("muse-glimmer.attention.layer_norm_rms_epsilon")
    var rope_base = gf.get_f32("muse-glimmer.rope.freq_base")
    var rope_base_swa = rope_base
    if gf.has_metadata("muse-glimmer.rope.freq_base_swa"):
        rope_base_swa = gf.get_f32("muse-glimmer.rope.freq_base_swa")
    var logit_softcap = gf.get_f32("muse-glimmer.final_logit_softcapping")
    var logit_scale = gf.get_f32("muse-glimmer.logit_scale")
    var vocab = gf.metadata_array_len("tokenizer.ggml.tokens")

    # Layer pattern: the file's bool array wins (true = sliding),
    # exactly derive_layer_pattern; absent means the %4==3 rule.
    var kinds = List[LayerKind]()
    if gf.has_metadata("muse-glimmer.attention.sliding_window_pattern"):
        var n = gf.metadata_array_len(
            "muse-glimmer.attention.sliding_window_pattern"
        )
        if n != layers:
            raise Error(
                "model_pattern_mismatch: sliding_window_pattern has "
                + String(n)
                + " entries, block_count is "
                + String(layers)
            )
        for i in range(layers):
            var is_swa = gf.metadata_array_bool(
                "muse-glimmer.attention.sliding_window_pattern", i
            )
            if is_swa:
                kinds.append(LayerKind.make_windowed(window))
            else:
                kinds.append(LayerKind.make_global())
        # Cross-check against the derived Glimmer pattern: the
        # real file's array and the rule agree; a disagreement on
        # a 52-layer file names a file this engine has not
        # verified, so refuse rather than guess.
        if layers == 52:
            var derived = glimmer_layer_pattern()
            for i in range(layers):
                if kinds[i].is_global != derived[i].is_global:
                    raise Error(
                        "model_pattern_mismatch: file pattern layer "
                        + String(i)
                        + " disagrees with the derived Glimmer pattern"
                    )
    else:
        for i in range(layers):
            if i % 4 == 3:
                kinds.append(LayerKind.make_global())
            else:
                kinds.append(LayerKind.make_windowed(window))
    return ModelConfig(
        hidden,
        ffn,
        layers,
        q_heads,
        kv_heads,
        key_len,
        vocab,
        context_len,
        window,
        rms_eps,
        rope_base,
        rope_base_swa,
        logit_scale,
        logit_softcap,
        kinds^,
    )


struct TensorSlot(Copyable, Movable):
    var name: String
    var type_id: Int
    var rows: Int
    var cols: Int
    var bytes: List[UInt8]
    var values: List[Float32]

    def __init__(
        out self,
        name: String,
        type_id: Int,
        rows: Int,
        cols: Int,
        var bytes: List[UInt8],
        var values: List[Float32],
    ):
        self.name = name
        self.type_id = type_id
        self.rows = rows
        self.cols = cols
        self.bytes = bytes^
        self.values = values^


struct GlimmerModel(Movable):
    var cfg: ModelConfig
    var embd: TensorSlot
    var final_norm: List[Float32]
    var output: TensorSlot
    var layer_weights: List[TensorSlot]
    var cache: KvCache
    var capture_enabled: Bool
    var capture_layer: Int
    var capture_in: List[Float32]
    var capture_out: List[Float32]
    var capture_embed: List[Float32]

    def __init__(out self, var cfg: ModelConfig, var gf: GgufFile) raises:
        self.cfg = cfg^
        self.layer_weights = List[TensorSlot]()
        # Embedding + head first (also validates their presence).
        self.embd = Self._load_slot(gf, "token_embd.weight")
        var out_info = gf.tensor_info("output.weight")
        _ = out_info
        self.output = Self._load_slot(gf, "output.weight")
        var norm_bytes = gf.tensor_bytes("output_norm.weight")
        self.final_norm = _bytes_to_f32(norm_bytes)
        if len(self.final_norm) != self.cfg.hidden:
            raise Error(
                "model_bad_shape: output_norm.weight has "
                + String(len(self.final_norm))
                + " elements, hidden is "
                + String(self.cfg.hidden)
            )
        for layer in range(self.cfg.layers):
            for s in range(LAYER_SUFFIX_COUNT):
                var name = "blk." + String(layer) + "." + layer_suffix(s)
                self.layer_weights.append(Self._load_slot(gf, name))
        self.cache = KvCache.new(
            self.cfg.kinds.copy(), self.cfg.kv_heads, self.cfg.head_dim
        )
        self.capture_enabled = False
        self.capture_layer = -1
        self.capture_in = List[Float32]()
        self.capture_out = List[Float32]()
        self.capture_embed = List[Float32]()
        _ = log_debug(
            "model",
            "loaded layers="
            + String(self.cfg.layers)
            + " hidden="
            + String(self.cfg.hidden),
        )

    @staticmethod
    def open(path: String) raises -> GlimmerModel:
        var gf = parse_gguf(path)
        var cfg = load_model_config(gf)
        return GlimmerModel(cfg^, gf^)

    @staticmethod
    def _load_slot(gf: GgufFile, name: String) raises -> TensorSlot:
        if gf.find_tensor(name) == -1:
            raise Error("model_missing_tensor: " + name + " not in file")
        var info = gf.tensor_info(name)
        var bytes = gf.tensor_bytes(name)
        var rows = 1
        var cols = info.elements
        if len(info.dims) == 2:
            # GGUF dims are [cols, rows] (innermost first).
            cols = info.dims[0]
            rows = info.dims[1]
        elif len(info.dims) != 1:
            raise Error(
                "model_bad_shape: tensor "
                + name
                + " has "
                + String(len(info.dims))
                + " dims"
            )
        var values = List[Float32]()
        if info.type_id == DT_F32:
            values = _bytes_to_f32(bytes.copy())
        return TensorSlot(name, info.type_id, rows, cols, bytes^, values^)

    def slot(self, layer: Int, suffix_index: Int) -> TensorSlot:
        return self.layer_weights[
            layer * LAYER_SUFFIX_COUNT + suffix_index
        ].copy()

    def proj(self, slot: TensorSlot, x: List[Float32]) raises -> List[Float32]:
        """One GEMV by the slot's dtype: y = W x, W is [rows][cols]."""
        if len(x) != slot.cols:
            raise Error(
                "model_bad_shape: projection input has "
                + String(len(x))
                + " elements, "
                + slot.name
                + " expects "
                + String(slot.cols)
            )
        if slot.type_id == DT_Q4_K:
            return k1_gemv_cpu(slot.bytes, x, slot.rows, slot.cols)
        if slot.type_id == DT_Q6_K:
            return q6k_gemv_cpu(slot.bytes, x, slot.rows, slot.cols)
        if slot.type_id == DT_Q5_K:
            return q5k_gemv_cpu(slot.bytes, x, slot.rows, slot.cols)
        if slot.type_id == DT_F32:
            return f32_gemv_cpu(slot.values, x, slot.rows, slot.cols)
        if slot.type_id == DT_F16:
            return f16_gemv_cpu(slot.bytes, x, slot.rows, slot.cols)
        raise Error(
            "model_unsupported_dtype: "
            + slot.name
            + " has ggml type id "
            + String(slot.type_id)
        )

    def embed_token(self, token: Int) raises -> List[Float32]:
        """One embedding row for token, any storage dtype."""
        if token < 0 or token >= self.cfg.vocab:
            raise Error(
                "model_bad_token: token "
                + String(token)
                + " outside vocab "
                + String(self.cfg.vocab)
            )
        var h = self.cfg.hidden
        var out = List[Float32](capacity=h)
        out.resize(h, Float32(0.0))
        if self.embd.type_id == DT_F32:
            for j in range(h):
                out[j] = self.embd.values[token * h + j]
            return out^
        if self.embd.type_id == DT_F16:
            for j in range(h):
                out[j] = read_f16_list(self.embd.bytes, (token * h + j) * 2)
            return out^
        if self.embd.type_id == DT_Q4_K:
            var sb_bytes = 144
            var row_blocks = h // 256
            var row_bytes = row_blocks * sb_bytes
            var base = token * row_bytes
            for b in range(row_blocks):
                var sb_off = base + b * sb_bytes
                for w in range(256):
                    out[b * 256 + w] = _q4k_element(self.embd.bytes, sb_off, w)
            return out^
        if self.embd.type_id == DT_Q6_K:
            var row_blocks6 = h // 256
            var row_bytes6 = row_blocks6 * 210
            var base6 = token * row_bytes6
            for b in range(row_blocks6):
                var sb6 = base6 + b * 210
                for w in range(256):
                    out[b * 256 + w] = q6k_element_list(self.embd.bytes, sb6, w)
            return out^
        if self.embd.type_id == DT_Q5_K:
            var row_blocks5 = h // 256
            var row_bytes5 = row_blocks5 * 176
            var base5 = token * row_bytes5
            for b in range(row_blocks5):
                var sb5 = base5 + b * 176
                for w in range(256):
                    out[b * 256 + w] = q5k_element_list(self.embd.bytes, sb5, w)
            return out^
        raise Error(
            "model_unsupported_dtype: token_embd.weight has ggml type id "
            + String(self.embd.type_id)
        )

    def step(mut self, token_id: Int, position: Int) raises -> List[Float32]:
        """One token at one absolute position through the whole
        stack; appends to the KV cache; returns the final
        hidden row. Mirrors forward_rows for rows == 1."""
        var cfg = self.cfg.copy()
        var x = self.embed_token(token_id)
        x = rms_norm_plain_cpu(x, cfg.rms_eps)
        if self.capture_enabled:
            self.capture_embed = x.copy()
        for layer in range(cfg.layers):
            if self.capture_enabled and layer == self.capture_layer:
                self.capture_in = x.copy()
            x = self.layer_step(x, layer, position)
            if self.capture_enabled and layer == self.capture_layer:
                self.capture_out = x.copy()
        return x^

    def layer_step(
        mut self, x_in: List[Float32], layer: Int, position: Int
    ) raises -> List[Float32]:
        var cfg = self.cfg.copy()
        var kind = cfg.kinds[layer].copy()
        var residual = x_in.copy()
        var h = rms_norm_cpu(residual, self.slot(layer, 0).values, cfg.rms_eps)
        var q = self.proj(self.slot(layer, 1), h)
        var k = self.proj(self.slot(layer, 2), h)
        var v = self.proj(self.slot(layer, 3), h)
        # Per-head Q/K norms: each head_dim row against the
        # norm weight (Rust applies rms_norm over rows*heads
        # rows of head_dim; the norm weights are [head_dim]).
        q = Self._norm_rows(
            q, self.slot(layer, 4).values, cfg.head_dim, cfg.rms_eps
        )
        k = Self._norm_rows(
            k, self.slot(layer, 5).values, cfg.head_dim, cfg.rms_eps
        )
        var sliding = not kind.is_global
        var base = cfg.rope_base_swa if sliding else ROPE_NONE
        if sliding:
            # K rotates HERE (write time; the cache holds
            # rotated K, Rust-exact). Q does NOT rotate here:
            # attention_cpu_rotated_cache takes the RAW q row
            # and rotates it internally, exactly as its
            # docstring states. (Rung 1 caught this: rotating
            # q here too double-rotated it and the sliding
            # fixture diverged at max_rel 4.37.)
            var kpos = List[Int]()
            for _ in range(cfg.kv_heads):
                kpos.append(position)
            k = rope_cpu(k, kpos, cfg.head_dim, base)
        self.cache.append(layer, k, v, position)
        # Gather the visible rows in position order, flat
        # [stored][kv_heads][head_dim] per head-major layout of
        # the cache rows (row = [kv_head][head_dim]).
        var first_vis = self.cache.first_visible_position(layer)
        var stored = self.cache.stored_len(layer)
        var keys = List[Float32](capacity=stored * cfg.kv_heads * cfg.head_dim)
        var values = List[Float32](
            capacity=stored * cfg.kv_heads * cfg.head_dim
        )
        for i in range(stored):
            var row_k = self.cache.key_row(layer, first_vis + i)
            var row_v = self.cache.value_row(layer, first_vis + i)
            for j in range(len(row_k)):
                keys.append(row_k[j])
            for j in range(len(row_v)):
                values.append(row_v[j])
        var attn = attention_cpu_rotated_cache(
            q,
            keys,
            values,
            first_vis,
            stored,
            cfg.q_heads,
            cfg.kv_heads,
            cfg.head_dim,
            position,
            base,
        )
        var gate = self.proj(self.slot(layer, 7), h)
        if len(gate) != len(attn):
            raise Error(
                "model_bad_shape: attn_gate row has "
                + String(len(gate))
                + " elements, attention output has "
                + String(len(attn))
            )
        for i in range(len(attn)):
            attn[i] = attn[i] * _sigmoid(gate[i])
        var out = self.proj(self.slot(layer, 6), attn)
        out = rms_norm_cpu(out, self.slot(layer, 8).values, POST_NORM_EPS)
        var after_attn = residual_add_cpu(residual, out)
        var residual2 = after_attn.copy()
        var h2 = rms_norm_cpu(
            residual2, self.slot(layer, 9).values, cfg.rms_eps
        )
        var gate_ffn = self.proj(self.slot(layer, 10), h2)
        var up_ffn = self.proj(self.slot(layer, 11), h2)
        var act = swiglu_cpu(gate_ffn, up_ffn)
        var ffn = self.proj(self.slot(layer, 12), act)
        ffn = rms_norm_cpu(ffn, self.slot(layer, 13).values, POST_NORM_EPS)
        return residual_add_cpu(residual2, ffn)

    @staticmethod
    def _norm_rows(
        x: List[Float32], weight: List[Float32], row_len: Int, eps: Float32
    ) raises -> List[Float32]:
        if len(x) % row_len != 0 or len(weight) != row_len:
            raise Error(
                "model_bad_shape: per-head norm over "
                + String(len(x))
                + " elements with row "
                + String(row_len)
                + " and weight "
                + String(len(weight))
            )
        var out = List[Float32](capacity=len(x))
        out.resize(len(x), Float32(0.0))
        var rows = len(x) // row_len
        for r in range(rows):
            var row = List[Float32](capacity=row_len)
            for j in range(row_len):
                row.append(x[r * row_len + j])
            var normed = rms_norm_cpu(row, weight, eps)
            for j in range(row_len):
                out[r * row_len + j] = normed[j]
        return out^

    def logits(self, hidden: List[Float32]) raises -> List[Float32]:
        """finish_last: final norm, lm_head, scale-then-softcap."""
        var cfg = self.cfg.copy()
        if len(hidden) != cfg.hidden:
            raise Error(
                "model_bad_shape: logits got hidden row of "
                + String(len(hidden))
                + ", hidden is "
                + String(cfg.hidden)
            )
        var normed = rms_norm_cpu(hidden, self.final_norm, cfg.rms_eps)
        var raw = self.proj(self.output.copy(), normed)
        return final_logit_cpu(raw, cfg.logit_scale, cfg.logit_softcap)

    def forward_rows(mut self, tokens: List[Int]) raises -> List[Float32]:
        """The step loop over a token list starting at the
        cache's next position; returns the LAST hidden row."""
        var pos = self.cache.next_position(0)
        var hidden = List[Float32]()
        for i in range(len(tokens)):
            hidden = self.step(tokens[i], pos + i)
        return hidden^


struct GlimmerForwardAdapter(ForwardModel, Movable):
    """B4's ForwardModel over GlimmerModel. Holds the model by
    reference semantics through an explicit pointer-free
    design: the adapter OWNS the model (integration constructs
    one adapter and hands it to generate())."""

    var model: GlimmerModel

    def __init__(out self, var model: GlimmerModel):
        self.model = model^

    def step(mut self, token_id: Int, position: Int) raises -> List[Float32]:
        return self.model.step(token_id, position)

    def logits(self, hidden: List[Float32]) raises -> List[Float32]:
        return self.model.logits(hidden)
