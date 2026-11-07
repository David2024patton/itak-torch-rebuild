# rope.mojo - rotary position embedding (RoPE), Lane A4.
#
# Semantic reference (Rust, branch main):
#   crates/torch-math/src/rope_interleaved.rs::rope_interleaved
#     Glimmer uses the INTERLEAVED (adjacent) pairing, ggml
#     GGML_ROPE_TYPE_NORM: pair i joins channels (2i, 2i+1) and
#     rotates by position * base^(-2i / head_dim). This is NOT
#     the half-split NEOX pairing of crates/torch-math/src/rope.rs
#     (that one is qwen2's); the two pair different channels at
#     the same rates and are not interchangeable. Glimmer's
#     converter unpermutes Q/K into the NORM layout, so the
#     stored weights assume this pairing.
#   crates/muse-glimmer/src/config.rs: RoPE base 500000, read
#     from muse-glimmer.rope.freq_base (the SWA key falls back to
#     it; the real file carries no separate SWA key, so the SWA
#     base also equals 500000). Head dim 128.
#
# WHERE ROPE APPLIES (by design, no flags): RoPE runs on the
# SLIDING-window layers only, at the SWA base; global layers are
# NoPE (crates/muse-glimmer/src/compute.rs, Compute::attention:
# "RoPE runs on the SWA layers, NoPE on full ones"). That split
# is expressed by the CALLER simply not calling this op for a
# global layer. This module deliberately has no skip flag, no
# layer-kind argument, and no identity shortcut for global
# layers: absence is the mechanism.
#
# base is a PARAMETER sourced from config / GGUF metadata, never
# a literal in the math. Positions are an input (one per vector),
# because prefill and decode assign positions per position, not
# per call. Angles are computed in Float64 on the CPU twin, as
# the Rust reference does (long contexts lose f32 precision).
#
# Named refusals: rope_cpu_bad_shape / rope_cpu_bad_base.

from std.math import sin, cos, pow, isnan, isinf
from max.gpu import thread_idx, block_idx, block_dim

from diag import log_debug

comptime GLIMMER_ROPE_BASE = Float32(500000.0)
comptime GLIMMER_HEAD_DIM = 128


def rope_cpu(
    x: List[Float32], positions: List[Int], head_dim: Int, base: Float32
) raises -> List[Float32]:
    """Rotate every head vector in x by its position.

    x is flat [vectors * head_dim]; positions holds one position
    per vector (a Q or K projection flattened head-by-head is the
    intended input). Interleaved pairing, see the module doc.
    """
    if head_dim < 2 or head_dim % 2 != 0:
        raise Error(
            "rope_cpu_bad_shape: head_dim must be even and >= 2, got "
            + String(head_dim)
        )
    if len(x) == 0 or len(x) % head_dim != 0:
        raise Error(
            "rope_cpu_bad_shape: input has "
            + String(len(x))
            + " elements, not a whole number of head_dim "
            + String(head_dim)
            + " vectors"
        )
    var vectors = len(x) // head_dim
    if len(positions) != vectors:
        raise Error(
            "rope_cpu_bad_shape: positions has "
            + String(len(positions))
            + " entries for "
            + String(vectors)
            + " vectors"
        )
    if isnan(base) or isinf(base) or base <= Float32(1.0):
        raise Error(
            "rope_cpu_bad_base: base must be finite and > 1, got "
            + String(base)
        )
    for v in range(vectors):
        if positions[v] < 0:
            raise Error(
                "rope_cpu_bad_shape: position must be >= 0, got "
                + String(positions[v])
            )
    var out = List[Float32](capacity=len(x))
    out.resize(len(x), Float32(0.0))
    var half = head_dim // 2
    var base64 = Float64(base)
    for v in range(vectors):
        var off = v * head_dim
        var pos = Float64(positions[v])
        for i in range(half):
            var exponent = -(Float64(2 * i) / Float64(head_dim))
            var angle = pos * pow(base64, exponent)
            var c = cos(angle)
            var s = sin(angle)
            var a = Float64(x[off + 2 * i])
            var b = Float64(x[off + 2 * i + 1])
            out[off + 2 * i] = Float32(a * c - b * s)
            out[off + 2 * i + 1] = Float32(a * s + b * c)
    _ = log_debug(
        "ops.rope.cpu",
        "vectors " + String(vectors) + " head_dim " + String(head_dim),
    )
    return out^


def rope_kernel(
    x: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    positions: Pointer[Int32, MutAnyOrigin],
    head_dim: Int32,
    base: Float32,
):
    """GPU RoPE, interleaved pairing: one thread per pair.

    Host validates shapes before launch (CPU twin refusals are
    the named surface). Angles in f32 on device; CPU twin (f64
    angles) is the parity reference.
    """
    var pair = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var half = Int(head_dim) // 2
    var vec = pair // half
    var i = pair % half
    var exponent = -(Float32(2 * i) / Float32(head_dim))
    var angle = Float32(positions[unsafe_offset=vec]) * pow(base, exponent)
    var c = cos(angle)
    var s = sin(angle)
    var off = vec * Int(head_dim) + 2 * i
    var a = x[unsafe_offset=off]
    var b = x[unsafe_offset=off + 1]
    y[unsafe_offset=off] = a * c - b * s
    y[unsafe_offset=off + 1] = a * s + b * c
