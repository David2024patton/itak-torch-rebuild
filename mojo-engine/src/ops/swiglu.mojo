# swiglu.mojo - SiLU and the gated multiply (SwiGLU), Lane A4.
#
# Semantic reference (Rust, branch main):
#   crates/torch-math/src/activation.rs: silu(x) = x * sigmoid(x);
#   swiglu(gate, up)[i] = silu(gate[i]) * up[i]. Glimmer's FFN is
#   SwiGLU with intermediate size 19968 (config.rs census).
# The CPU twin uses the branch-stable sigmoid form of
# crates/cpu-compute/src/cpu.rs::sigmoid (same function, the
# exponential stays finite at both tails); it is mathematically
# identical to the torch-math form and matches it within f32
# rounding, which the test tolerance states.
#
# Named refusal: swiglu_cpu_bad_shape (length mismatch).

from std.math import exp
from max.gpu import thread_idx, block_idx, block_dim

from diag import log_debug


def silu_cpu(v: Float32) -> Float32:
    """SiLU of one scalar, branch-stable at both tails."""
    var sig: Float32
    if v >= Float32(0.0):
        sig = Float32(1.0) / (Float32(1.0) + exp(-v))
    else:
        var e = exp(v)
        sig = e / (Float32(1.0) + e)
    return v * sig


def swiglu_cpu(gate: List[Float32], up: List[Float32]) raises -> List[Float32]:
    """SwiGLU: silu(gate) * up, elementwise over the FFN halves."""
    if len(gate) != len(up):
        raise Error(
            "swiglu_cpu_bad_shape: gate has "
            + String(len(gate))
            + " elements, up has "
            + String(len(up))
        )
    var out = List[Float32](capacity=len(gate))
    out.resize(len(gate), Float32(0.0))
    for i in range(len(gate)):
        out[i] = silu_cpu(gate[i]) * up[i]
    _ = log_debug("ops.swiglu.cpu", "len " + String(len(gate)))
    return out^


def swiglu_kernel(
    gate: Pointer[Float32, MutAnyOrigin],
    up: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    n: Int32,
):
    """GPU SwiGLU: one thread per element. Host validates first."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        var g = gate[unsafe_offset=i]
        var sig: Float32
        if g >= Float32(0.0):
            sig = Float32(1.0) / (Float32(1.0) + exp(-g))
        else:
            var e = exp(g)
            sig = e / (Float32(1.0) + e)
        y[unsafe_offset=i] = g * sig * up[unsafe_offset=i]
