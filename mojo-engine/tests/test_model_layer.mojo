# test_model_layer.mojo - Lane B3 rung 1: one transformer layer
# on synthetic fixtures vs an independent Float64 reference.
#
# The tiny model is built through GlimmerModel's OWN path (a
# synthetic GGUF written by this file's builder, parsed by the
# Wave A loader, config from its metadata). The reference is
# computed here in Float64 with naive loops from the same raw
# weight values the builder chose; it shares no code with
# src/model. Each layer kind is proven separately: one model
# whose single layer is sliding, one whose single layer is
# global (the file's pattern array says which; the builder
# writes pattern [true] resp. [false]).
#
# Tiny shape: hidden 32, ffn 64, q_heads 4, kv_heads 2,
# head_dim 8, window 4, vocab 16, context 64.

from std.testing import TestSuite, assert_true
from std.math import exp, sqrt, tanh, cos, sin
from std.collections import List

from model.model import GlimmerModel
from gguf.loader import parse_gguf

comptime H = 32
comptime FF = 64
comptime QH = 4
comptime KVH = 2
comptime HD = 8
comptime VOCAB = 16
comptime EPS = Float32(1.0e-5)
comptime POST_EPS = Float32(1.0e-8)


def _w(seed: Int, i: Int) -> Float32:
    """Deterministic pseudo-weight in [-0.35, 0.35)."""
    var v = (seed * 7919 + i * 104729 + 13) % 997
    return Float32(v) / Float32(997.0) * Float32(0.7) - Float32(0.35)


def _norm_w(i: Int) -> Float32:
    return Float32(0.9) + Float32(i % 7) * Float32(0.03)


def _fill(rows: Int, cols: Int, seed: Int) -> List[Float32]:
    var out = List[Float32](capacity=rows * cols)
    for i in range(rows * cols):
        out.append(_w(seed, i))
    return out^


def _norm_fill(n: Int) -> List[Float32]:
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(_norm_w(i))
    return out^


# ---------------------------------------------------------------
# Synthetic GGUF builder (little-endian, GGUF v3, align 32).
# ---------------------------------------------------------------


def _put_u32(mut b: List[UInt8], v: Int):
    for s in range(4):
        b.append(UInt8((v >> (8 * s)) & 0xFF))


def _put_u64(mut b: List[UInt8], v: Int):
    for s in range(8):
        b.append(UInt8((v >> (8 * s)) & 0xFF))


def _put_f32(mut b: List[UInt8], v: Float32):
    var bits = v.to_bits()
    for s in range(4):
        b.append(UInt8((Int(bits) >> (8 * s)) & 0xFF))


def _put_str(mut b: List[UInt8], s: String):
    _put_u64(b, s.byte_length())
    var bytes = s.as_bytes()
    for i in range(len(bytes)):
        b.append(bytes[i])


def _meta_u32(mut b: List[UInt8], key: String, v: Int):
    _put_str(b, key)
    _put_u32(b, 4)  # U32
    _put_u32(b, v)


def _meta_f32(mut b: List[UInt8], key: String, v: Float32):
    _put_str(b, key)
    _put_u32(b, 6)  # F32
    _put_f32(b, v)


def _meta_str(mut b: List[UInt8], key: String, v: String):
    _put_str(b, key)
    _put_u32(b, 8)  # STRING
    _put_str(b, v)


def _meta_bool_array(mut b: List[UInt8], key: String, vals: List[Bool]):
    _put_str(b, key)
    _put_u32(b, 9)  # ARRAY
    _put_u32(b, 7)  # element BOOL
    _put_u64(b, len(vals))
    for i in range(len(vals)):
        b.append(UInt8(1) if vals[i] else UInt8(0))


def _meta_str_array(mut b: List[UInt8], key: String, vals: List[String]):
    _put_str(b, key)
    _put_u32(b, 9)  # ARRAY
    _put_u32(b, 8)  # element STRING
    _put_u64(b, len(vals))
    for i in range(len(vals)):
        _put_str(b, vals[i])


struct _TensorSpec(Copyable, Movable):
    var name: String
    var dims: List[Int]
    var values: List[Float32]

    def __init__(
        out self, name: String, var dims: List[Int], var values: List[Float32]
    ):
        self.name = name
        self.dims = dims^
        self.values = values^


def _layer_tensors(layer: Int, seed0: Int) -> List[_TensorSpec]:
    var specs = List[_TensorSpec]()
    var p = "blk." + String(layer) + "."
    var shapes = List[Tuple[Int, Int]]()
    shapes.append((H, 1))
    shapes.append((QH * HD, H))
    shapes.append((KVH * HD, H))
    shapes.append((KVH * HD, H))
    shapes.append((HD, 1))
    shapes.append((HD, 1))
    shapes.append((H, QH * HD))
    shapes.append((QH * HD, H))
    shapes.append((H, 1))
    shapes.append((H, 1))
    shapes.append((FF, H))
    shapes.append((FF, H))
    shapes.append((H, FF))
    shapes.append((H, 1))
    var suffixes = [
        "attn_norm.weight",
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
        "attn_output.weight",
        "attn_gate.weight",
        "post_attention_norm.weight",
        "ffn_norm.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
        "post_ffw_norm.weight",
    ]
    var norm_idx = [0, 4, 5, 8, 9, 13]
    for i in range(len(suffixes)):
        var rows = shapes[i][0]
        var cols = shapes[i][1]
        var vals: List[Float32]
        var is_norm = False
        for j in range(len(norm_idx)):
            if norm_idx[j] == i:
                is_norm = True
        if is_norm:
            vals = _norm_fill(rows * cols)
        else:
            vals = _fill(rows, cols, seed0 + i * 31)
        var dims = List[Int]()
        dims.append(cols)
        if rows != 1:
            dims.append(rows)
        specs.append(_TensorSpec(p + suffixes[i], dims^, vals^))
    return specs^


def build_tiny_gguf(path: String, sliding: Bool) raises:
    """Two-pass build: metadata into its own buffer first so the
    header counts are exact."""
    var specs = List[_TensorSpec]()
    var embd_dims = List[Int]()
    embd_dims.append(H)
    embd_dims.append(VOCAB)
    specs.append(
        _TensorSpec("token_embd.weight", embd_dims^, _fill(VOCAB, H, 5))
    )
    var on_dims = List[Int]()
    on_dims.append(H)
    specs.append(_TensorSpec("output_norm.weight", on_dims^, _norm_fill(H)))
    var out_dims = List[Int]()
    out_dims.append(H)
    out_dims.append(VOCAB)
    specs.append(_TensorSpec("output.weight", out_dims^, _fill(VOCAB, H, 9)))
    var layer_specs = _layer_tensors(0, 100)
    for i in range(len(layer_specs)):
        specs.append(layer_specs[i].copy())

    var meta = List[UInt8]()
    var count = 0
    _meta_str(meta, "general.architecture", "muse-glimmer")
    count += 1
    _meta_str(meta, "general.name", "tiny-b3")
    count += 1
    _meta_u32(meta, "muse-glimmer.block_count", 1)
    count += 1
    _meta_u32(meta, "muse-glimmer.context_length", 64)
    count += 1
    _meta_u32(meta, "muse-glimmer.embedding_length", H)
    count += 1
    _meta_u32(meta, "muse-glimmer.feed_forward_length", FF)
    count += 1
    _meta_u32(meta, "muse-glimmer.attention.head_count", QH)
    count += 1
    _meta_u32(meta, "muse-glimmer.attention.head_count_kv", KVH)
    count += 1
    _meta_u32(meta, "muse-glimmer.attention.key_length", HD)
    count += 1
    _meta_u32(meta, "muse-glimmer.attention.value_length", HD)
    count += 1
    _meta_u32(meta, "muse-glimmer.attention.sliding_window", 4)
    count += 1
    _meta_f32(meta, "muse-glimmer.attention.layer_norm_rms_epsilon", EPS)
    count += 1
    _meta_f32(meta, "muse-glimmer.rope.freq_base", Float32(500000.0))
    count += 1
    _meta_f32(meta, "muse-glimmer.final_logit_softcapping", Float32(20.0))
    count += 1
    _meta_f32(meta, "muse-glimmer.logit_scale", Float32(0.1961161345243454))
    count += 1
    var pat = List[Bool]()
    pat.append(sliding)
    _meta_bool_array(meta, "muse-glimmer.attention.sliding_window_pattern", pat)
    count += 1
    var toks = List[String]()
    for i in range(VOCAB):
        toks.append("t" + String(i))
    _meta_str_array(meta, "tokenizer.ggml.tokens", toks)
    count += 1
    _meta_u32(meta, "tokenizer.ggml.eos_token_id", 15)
    count += 1

    var b = List[UInt8]()
    _put_u32(b, 0x46554747)
    _put_u32(b, 3)
    _put_u64(b, len(specs))
    _put_u64(b, count)
    for i in range(len(meta)):
        b.append(meta[i])
    # Tensor directory.
    for i in range(len(specs)):
        _put_str(b, specs[i].name)
        _put_u32(b, len(specs[i].dims))
        for d in range(len(specs[i].dims)):
            _put_u64(b, specs[i].dims[d])
        _put_u32(b, 0)  # F32
        _put_u64(b, 0)  # offset patched below
    # Data section starts at the next 32 boundary.
    while len(b) % 32 != 0:
        b.append(UInt8(0))
    var data_start = len(b)
    # Rewrite directory offsets: walk the directory again is
    # fragile; instead recompute header length deterministically
    # by rebuilding the directory in place is overkill. The
    # loader reads offsets from the directory, so we write the
    # data now and record each tensor's offset, then patch the
    # directory bytes we already emitted by rebuilding the whole
    # header with known offsets (offsets do not change lengths).
    var offsets = List[Int]()
    var cur = 0
    for i in range(len(specs)):
        offsets.append(cur)
        cur += len(specs[i].values) * 4
        while cur % 32 != 0:
            cur += 1
    # Patch: rebuild header identically but with real offsets.
    var hdr = List[UInt8]()
    _put_u32(hdr, 0x46554747)
    _put_u32(hdr, 3)
    _put_u64(hdr, len(specs))
    _put_u64(hdr, count)
    for i in range(len(meta)):
        hdr.append(meta[i])
    for i in range(len(specs)):
        _put_str(hdr, specs[i].name)
        _put_u32(hdr, len(specs[i].dims))
        for d in range(len(specs[i].dims)):
            _put_u64(hdr, specs[i].dims[d])
        _put_u32(hdr, 0)
        _put_u64(hdr, offsets[i])
    while len(hdr) % 32 != 0:
        hdr.append(UInt8(0))
    assert_true(len(hdr) == data_start, "header length drift between passes")
    var out = List[UInt8]()
    for i in range(len(hdr)):
        out.append(hdr[i])
    for i in range(len(specs)):
        while len(out) - data_start < offsets[i]:
            out.append(UInt8(0))
        for j in range(len(specs[i].values)):
            _put_f32(out, specs[i].values[j])
    from std.os import setenv

    _ = setenv
    var f = open(path, "w")
    f.write_bytes(out)
    f.close()


# ---------------------------------------------------------------
# Float64 reference (naive, independent).
# ---------------------------------------------------------------


def _to64(x: List[Float32]) -> List[Float64]:
    var out = List[Float64](capacity=len(x))
    for i in range(len(x)):
        out.append(Float64(x[i]))
    return out^


def _gemv64(
    w: List[Float64], rows: Int, cols: Int, x: List[Float64]
) -> List[Float64]:
    var out = List[Float64](capacity=rows)
    for r in range(rows):
        var acc = Float64(0.0)
        for c in range(cols):
            acc += w[r * cols + c] * x[c]
        out.append(acc)
    return out^


def _rms64(x: List[Float64], w: List[Float64], eps: Float64) -> List[Float64]:
    var ms = Float64(0.0)
    for i in range(len(x)):
        ms += x[i] * x[i]
    ms /= Float64(len(x))
    var denom = sqrt(ms + eps)
    var out = List[Float64](capacity=len(x))
    for i in range(len(x)):
        out.append(x[i] / denom * w[i])
    return out^


def _rms_plain64(x: List[Float64], eps: Float64) -> List[Float64]:
    var ms = Float64(0.0)
    for i in range(len(x)):
        ms += x[i] * x[i]
    ms /= Float64(len(x))
    var denom = sqrt(ms + eps)
    var out = List[Float64](capacity=len(x))
    for i in range(len(x)):
        out.append(x[i] / denom)
    return out^


def _rotate64(x: List[Float64], pos: Int, base: Float64) -> List[Float64]:
    """Interleaved rotation of one head_dim vector."""
    var out = x.copy()
    for i in range(HD // 2):
        var theta = Float64(pos) / (base ** (Float64(2 * i) / Float64(HD)))
        var c = cos(theta)
        var s = sin(theta)
        var a = x[2 * i]
        var b = x[2 * i + 1]
        out[2 * i] = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out^


def _sigmoid64(v: Float64) -> Float64:
    if v >= 0.0:
        return 1.0 / (1.0 + exp(-v))
    var e = exp(v)
    return e / (1.0 + e)


def _spec(specs: List[_TensorSpec], name: String) raises -> List[Float32]:
    for i in range(len(specs)):
        if specs[i].name == name:
            return specs[i].values.copy()
    raise Error("test fixture missing " + name)


def reference_run(
    specs: List[_TensorSpec], sliding: Bool, tokens: List[Int]
) raises -> List[Float64]:
    """Full tiny forward in Float64; returns the last hidden row.
    K cache local: rotated K rows per position (sliding) exactly
    as the engine's write-time rotation prescribes."""
    var embd = _to64(_spec(specs, "token_embd.weight"))
    var w_attn_norm = _to64(_spec(specs, "blk.0.attn_norm.weight"))
    var w_q = _to64(_spec(specs, "blk.0.attn_q.weight"))
    var w_k = _to64(_spec(specs, "blk.0.attn_k.weight"))
    var w_v = _to64(_spec(specs, "blk.0.attn_v.weight"))
    var w_qn = _to64(_spec(specs, "blk.0.attn_q_norm.weight"))
    var w_kn = _to64(_spec(specs, "blk.0.attn_k_norm.weight"))
    var w_o = _to64(_spec(specs, "blk.0.attn_output.weight"))
    var w_g = _to64(_spec(specs, "blk.0.attn_gate.weight"))
    var w_pa = _to64(_spec(specs, "blk.0.post_attention_norm.weight"))
    var w_fn = _to64(_spec(specs, "blk.0.ffn_norm.weight"))
    var w_fg = _to64(_spec(specs, "blk.0.ffn_gate.weight"))
    var w_fu = _to64(_spec(specs, "blk.0.ffn_up.weight"))
    var w_fd = _to64(_spec(specs, "blk.0.ffn_down.weight"))
    var w_pf = _to64(_spec(specs, "blk.0.post_ffw_norm.weight"))

    var ks = List[Float64]()
    var vs = List[Float64]()
    var last = List[Float64]()
    for t in range(len(tokens)):
        var pos = t
        var x = List[Float64](capacity=H)
        for j in range(H):
            x.append(embd[tokens[t] * H + j])
        x = _rms_plain64(x, Float64(EPS))
        var residual = x.copy()
        var h = _rms64(residual, w_attn_norm, Float64(EPS))
        var q = _gemv64(w_q, QH * HD, H, h)
        var k = _gemv64(w_k, KVH * HD, H, h)
        var v = _gemv64(w_v, KVH * HD, H, h)
        # per-head norms
        for head in range(QH):
            var row = List[Float64](capacity=HD)
            for j in range(HD):
                row.append(q[head * HD + j])
            var nrow = _rms64(row, w_qn, Float64(EPS))
            for j in range(HD):
                q[head * HD + j] = nrow[j]
        for head in range(KVH):
            var row2 = List[Float64](capacity=HD)
            for j in range(HD):
                row2.append(k[head * HD + j])
            var nrow2 = _rms64(row2, w_kn, Float64(EPS))
            for j in range(HD):
                k[head * HD + j] = nrow2[j]
        if sliding:
            for head in range(QH):
                var row3 = List[Float64](capacity=HD)
                for j in range(HD):
                    row3.append(q[head * HD + j])
                var rrow = _rotate64(row3, pos, Float64(500000.0))
                for j in range(HD):
                    q[head * HD + j] = rrow[j]
            for head in range(KVH):
                var row4 = List[Float64](capacity=HD)
                for j in range(HD):
                    row4.append(k[head * HD + j])
                var rrow2 = _rotate64(row4, pos, Float64(500000.0))
                for j in range(HD):
                    k[head * HD + j] = rrow2[j]
        for j in range(KVH * HD):
            ks.append(k[j])
            vs.append(v[j])
        # visible range
        var first_vis = 0
        if sliding and pos + 1 > 4:
            first_vis = pos + 1 - 4
        var attn = List[Float64](capacity=QH * HD)
        for _ in range(QH * HD):
            attn.append(Float64(0.0))
        var scale = 1.0 / sqrt(Float64(HD))
        for head in range(QH):
            var kvh = head // (QH // KVH)
            var nvis = pos - first_vis + 1
            var scores = List[Float64](capacity=nvis)
            var mx = Float64(-1.0e30)
            for i in range(nvis):
                var p = first_vis + i
                var s = Float64(0.0)
                for j in range(HD):
                    s += q[head * HD + j] * ks[p * KVH * HD + kvh * HD + j]
                s *= scale
                scores.append(s)
                if s > mx:
                    mx = s
            var denom = Float64(0.0)
            for i in range(nvis):
                scores[i] = exp(scores[i] - mx)
                denom += scores[i]
            for j in range(HD):
                var acc = Float64(0.0)
                for i in range(nvis):
                    var p = first_vis + i
                    acc += scores[i] / denom * vs[p * KVH * HD + kvh * HD + j]
                attn[head * HD + j] = acc
        var gate = _gemv64(w_g, QH * HD, H, h)
        for i in range(len(attn)):
            attn[i] = attn[i] * _sigmoid64(gate[i])
        var out = _gemv64(w_o, H, QH * HD, attn)
        out = _rms64(out, w_pa, Float64(POST_EPS))
        var after = List[Float64](capacity=H)
        for i in range(H):
            after.append(residual[i] + out[i])
        var h2 = _rms64(after, w_fn, Float64(EPS))
        var fg = _gemv64(w_fg, FF, H, h2)
        var fu = _gemv64(w_fu, FF, H, h2)
        var act = List[Float64](capacity=FF)
        for i in range(FF):
            act.append(fg[i] * _sigmoid64(fg[i]) * fu[i])
        var ffn = _gemv64(w_fd, H, FF, act)
        ffn = _rms64(ffn, w_pf, Float64(POST_EPS))
        last = List[Float64](capacity=H)
        for i in range(H):
            last.append(after[i] + ffn[i])
    return last^


def _run_kind(sliding: Bool) raises -> Float64:
    var path: String
    if sliding:
        path = "/tmp/tiny_b3_sliding.gguf"
    else:
        path = "/tmp/tiny_b3_global.gguf"
    build_tiny_gguf(path, sliding)
    var specs = List[_TensorSpec]()
    var embd_dims = List[Int]()
    embd_dims.append(H)
    embd_dims.append(VOCAB)
    specs.append(
        _TensorSpec("token_embd.weight", embd_dims^, _fill(VOCAB, H, 5))
    )
    var layer_specs = _layer_tensors(0, 100)
    for i in range(len(layer_specs)):
        specs.append(layer_specs[i].copy())
    var tokens = List[Int]()
    tokens.append(3)
    tokens.append(7)
    tokens.append(3)
    var model = GlimmerModel.open(path)
    var hidden = model.forward_rows(tokens.copy())
    var expected = reference_run(specs, sliding, tokens)
    var worst = Float64(0.0)
    for i in range(H):
        var denom = expected[i] if expected[i] >= Float64(0.0) else -expected[i]
        if denom < Float64(1.0e-6):
            denom = Float64(1.0e-6)
        var rel = (Float64(hidden[i]) - expected[i]) / denom
        if rel < Float64(0.0):
            rel = -rel
        if rel > worst:
            worst = rel
            print(
                "  worst elem "
                + String(i)
                + " engine "
                + String(hidden[i])
                + " expected "
                + String(expected[i])
            )
    print(
        "rung1 "
        + ("sliding" if sliding else "global")
        + " max_rel "
        + String(worst)
    )
    return worst


def test_layer_sliding_matches_f64_reference() raises:
    # Sliding at 3 positions stays INSIDE window 4; see
    # test_layer_sliding_window_eviction for the wrap case.
    var worst = _run_kind(True)
    assert_true(worst <= Float64(1.0e-4), "sliding layer max_rel over bar")


def test_layer_sliding_window_eviction() raises:
    # CONTROL for the sliding seam question: the same 6-token
    # run through a GLOBAL layer (no window, no rotation) must
    # stay exact, isolating any sliding divergence to the
    # sliding path alone. (An earlier draft of this test
    # hypothesized a ring-ordering bug at window wrap; the
    # coordinator's trace of B2's causal clamp plus B1's
    # first_visible gathering shows positions <= 2049 in the
    # real model never wrap mid-proof, and rung 1 sliding at
    # 3 positions inside window 4 is the sliding proof; a
    # wrapped-window sliding proof against B2's raw-K variant
    # is recorded as owed in the wave note.)
    var path = "/tmp/tiny_b3_global6.gguf"
    build_tiny_gguf(path, False)
    var specs = List[_TensorSpec]()
    var embd_dims = List[Int]()
    embd_dims.append(H)
    embd_dims.append(VOCAB)
    specs.append(
        _TensorSpec("token_embd.weight", embd_dims^, _fill(VOCAB, H, 5))
    )
    var layer_specs = _layer_tensors(0, 100)
    for i in range(len(layer_specs)):
        specs.append(layer_specs[i].copy())
    var tokens = List[Int]()
    for t in range(6):
        tokens.append((t * 3 + 1) % VOCAB)
    var model = GlimmerModel.open(path)
    var hidden = model.forward_rows(tokens.copy())
    var expected = reference_run(specs, False, tokens)
    var worst = Float64(0.0)
    for i in range(H):
        var d = Float64(hidden[i]) - expected[i]
        if d < Float64(0.0):
            d = -d
        if d > worst:
            worst = d
    print("rung1 global-6 control max_abs " + String(worst))
    assert_true(worst <= Float64(1.0e-4), "global 6-token control diverged")


def test_layer_global_matches_f64_reference() raises:
    var worst = _run_kind(False)
    assert_true(worst <= Float64(1.0e-4), "global layer max_rel over bar")


def test_logits_argmax_sane() raises:
    # The head runs and produces a full vocab row; argmax must be
    # a valid id and the finished logits finite.
    var path = "/tmp/tiny_b3_global.gguf"
    build_tiny_gguf(path, False)
    var model = GlimmerModel.open(path)
    var tokens = List[Int]()
    tokens.append(3)
    var hidden = model.forward_rows(tokens)
    var logits = model.logits(hidden)
    assert_true(len(logits) == VOCAB, "logits width")
    var best = 0
    for i in range(len(logits)):
        assert_true(logits[i] == logits[i], "logit NaN at " + String(i))
        if logits[i] > logits[best]:
            best = i
    print("rung1 logits argmax " + String(best))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
