# embedding.mojo - embedding lookup (row gather), Lane A4.
#
# Semantic reference (Rust, branch main):
#   crates/muse-glimmer/src/compute.rs, Compute::embed - "Look up
#   the embedding rows for tokens, returning [tokens.len(),
#   hidden]".
#   crates/cpu-compute/src/cpu.rs::embed - the stored table is in
#   GGUF stored order [hidden, vocab]; each token's row is
#   dequantized on demand. Glimmer: hidden 6656, vocab 202048.
#
# DTYPE HANDLING (the split, stated plainly): this module does a
# plain F32 row gather. A QUANTIZED embedding table (Q8_0, Q4_K,
# ...) is NOT gathered here: dequantization belongs to the
# kernels family (K1 and its siblings, Lane A3's shapes), and
# until those shapes land for the embedding path, a quantized
# request is REFUSED BY NAME (embedding_cpu_quantized_unsupported)
# instead of being silently misread as f32 bytes. Callers holding
# an already-dequantized f32 table use embedding_cpu directly.
#
# Layout here: the f32 table is row-major [vocab, hidden], one
# row per token (the dequantized layout; the Rust stored-order
# [hidden, vocab] table reaches this op only after the loader /
# dequant family has produced rows).
#
# Named refusals: embedding_cpu_bad_shape /
# embedding_cpu_bad_token / embedding_cpu_quantized_unsupported.

from max.gpu import thread_idx, block_idx, block_dim

from diag import log_debug

# Wave E (Lane E3): the recorded Glimmer dimensions that used
# to live here as GLIMMER_HIDDEN / GLIMMER_VOCAB now live in
# src/config.mojo as DEFAULT_HIDDEN / DEFAULT_VOCAB
# (EngineConfig.hidden / .vocab, env MOJO_ENGINE_HIDDEN /
# MOJO_ENGINE_VOCAB). They had no consumer in this module:
# hidden is a parameter of the gathers below, as it always
# was.


def embedding_cpu(
    table: List[Float32], tokens: List[Int], hidden: Int
) raises -> List[Float32]:
    """Gather one f32 row per token: out[t * hidden + j]."""
    if hidden <= 0:
        raise Error(
            "embedding_cpu_bad_shape: hidden must be > 0, got " + String(hidden)
        )
    if len(table) == 0 or len(table) % hidden != 0:
        raise Error(
            "embedding_cpu_bad_shape: table has "
            + String(len(table))
            + " elements, not a whole number of hidden "
            + String(hidden)
            + " rows"
        )
    var vocab = len(table) // hidden
    var out = List[Float32](capacity=len(tokens) * hidden)
    out.resize(len(tokens) * hidden, Float32(0.0))
    for t in range(len(tokens)):
        var tok = tokens[t]
        if tok < 0 or tok >= vocab:
            raise Error(
                "embedding_cpu_bad_token: token "
                + String(tok)
                + " outside vocab 0.."
                + String(vocab)
            )
        var src = tok * hidden
        var dst = t * hidden
        for j in range(hidden):
            out[dst + j] = table[src + j]
    _ = log_debug(
        "ops.embedding.cpu",
        "tokens " + String(len(tokens)) + " hidden " + String(hidden),
    )
    return out^


def embedding_cpu_for_dtype(
    table: List[Float32], tokens: List[Int], hidden: Int, dtype: String
) raises -> List[Float32]:
    """Dtype-gated entry: only the plain F32 gather is served.

    Any quantized dtype refuses by name (see the module doc):
    dequant is the kernels family's job, not a silent reinterpret
    of quantized bytes as f32.
    """
    if dtype != "F32":
        raise Error(
            "embedding_cpu_quantized_unsupported: dtype "
            + dtype
            + " needs the kernels-family dequant path (Lane A3"
            " shapes); only a plain F32 row gather is served here"
        )
    return embedding_cpu(table, tokens, hidden)


def embedding_kernel(
    table: Pointer[Float32, MutAnyOrigin],
    tokens: Pointer[Int32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    hidden: Int32,
):
    """GPU F32 row gather: one thread per output element."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var h = Int(hidden)
    var row = idx // h
    var col = idx % h
    var tok = Int(tokens[unsafe_offset=row])
    y[unsafe_offset=idx] = table[unsafe_offset=tok * h + col]
