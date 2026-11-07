# softmax.mojo - plain attention softmax, Lane A4.
#
# Semantic reference (Rust, branch main):
#   crates/attention/src/attention.rs: scores are max-subtracted
#   for stability, exponentiated, summed, and divided
#   ("Softmax in place, max-subtracted for stability").
# This module is ONLY that: a row-wise softmax over plain score
# rows. It is not the sampler's temperature/top-k distribution
# and it is not the final-logit transform (see final_logit.mojo;
# the two are separate modules so they are never conflated).
#
# The CPU twin accumulates the exp sum in Float64 (the Rust
# attention loop sums in f32; the formulas are identical and the
# f64 sum is the more accurate evaluation of the same formula,
# which is why this twin can serve as the parity reference at a
# tight tolerance). Empty rows refuse by name: a softmax over no
# keys is the attention layer's NoVisibleKeys case upstream, and
# silently returning an empty row here would hide it.
#
# Named refusal: softmax_cpu_bad_shape.

from std.math import exp
from max.gpu import thread_idx, block_idx, block_dim, barrier
from max.gpu.memory import AddressSpace
from std.memory import stack_allocation

from diag import log_debug

comptime SOFTMAX_THREADS = 256


def softmax_cpu(x: List[Float32], width: Int) raises -> List[Float32]:
    """Row-wise softmax over flat [rows * width] scores."""
    if width <= 0:
        raise Error(
            "softmax_cpu_bad_shape: width must be > 0, got " + String(width)
        )
    if len(x) == 0 or len(x) % width != 0:
        raise Error(
            "softmax_cpu_bad_shape: input has "
            + String(len(x))
            + " elements, not a whole number of width "
            + String(width)
            + " rows"
        )
    var rows = len(x) // width
    var out = List[Float32](capacity=len(x))
    out.resize(len(x), Float32(0.0))
    for r in range(rows):
        var base = r * width
        var mx = x[base]
        for i in range(1, width):
            if x[base + i] > mx:
                mx = x[base + i]
        var sum = Float64(0.0)
        for i in range(width):
            sum += exp(Float64(x[base + i]) - Float64(mx))
        for i in range(width):
            var e = exp(Float64(x[base + i]) - Float64(mx))
            out[base + i] = Float32(e / sum)
    _ = log_debug(
        "ops.softmax.cpu", "rows " + String(rows) + " width " + String(width)
    )
    return out^


def softmax_kernel(
    x: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    width: Int32,
):
    """GPU softmax: one block per row, shared-memory reductions.

    Pass 1 reduces the row max, pass 2 the exp sum (recomputing
    exp per element rather than staging the row), pass 3 writes
    exp(x-max)/sum. Host validates shapes before launch.
    """
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var n = Int(width)
    var base = row * n
    var shared = stack_allocation[
        SOFTMAX_THREADS, Float32, address_space=AddressSpace.SHARED
    ]()
    var local_max = (
        x[unsafe_offset=base + tid] if tid < n else x[unsafe_offset=base]
    )
    var i = tid + SOFTMAX_THREADS
    while i < n:
        if x[unsafe_offset=base + i] > local_max:
            local_max = x[unsafe_offset=base + i]
        i += SOFTMAX_THREADS
    shared[unsafe_offset=tid] = local_max
    barrier()
    var stride = SOFTMAX_THREADS // 2
    while stride > 0:
        if tid < stride:
            if shared[unsafe_offset=tid + stride] > shared[unsafe_offset=tid]:
                shared[unsafe_offset=tid] = shared[unsafe_offset=tid + stride]
        barrier()
        stride //= 2
    var mx = shared[unsafe_offset=0]
    var local_sum = Float32(0.0)
    i = tid
    while i < n:
        local_sum += exp(x[unsafe_offset=base + i] - mx)
        i += SOFTMAX_THREADS
    shared[unsafe_offset=tid] = local_sum
    barrier()
    stride = SOFTMAX_THREADS // 2
    while stride > 0:
        if tid < stride:
            shared[unsafe_offset=tid] = (
                shared[unsafe_offset=tid] + shared[unsafe_offset=tid + stride]
            )
        barrier()
        stride //= 2
    var inv_sum = Float32(1.0) / shared[unsafe_offset=0]
    i = tid
    while i < n:
        y[unsafe_offset=base + i] = (
            exp(x[unsafe_offset=base + i] - mx) * inv_sum
        )
        i += SOFTMAX_THREADS
