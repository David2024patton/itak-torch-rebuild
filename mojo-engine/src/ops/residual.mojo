# residual.mojo - residual add, Lane A4.
#
# Semantic reference: crates/muse-glimmer/src/compute.rs,
# Compute::add - "Residual add of two equal-length activations."
# out[i] = a[i] + b[i], plain f32 addition, no scaling, no norm:
# in the Glimmer graph any post-add norm is a separate rms_norm
# step, never folded in here.
#
# Named refusal: residual_add_cpu_bad_shape.

from max.gpu import thread_idx, block_idx, block_dim

from diag import log_debug


def residual_add_cpu(
    a: List[Float32], b: List[Float32]
) raises -> List[Float32]:
    if len(a) != len(b):
        raise Error(
            "residual_add_cpu_bad_shape: a has "
            + String(len(a))
            + " elements, b has "
            + String(len(b))
        )
    var out = List[Float32](capacity=len(a))
    out.resize(len(a), Float32(0.0))
    for i in range(len(a)):
        out[i] = a[i] + b[i]
    _ = log_debug("ops.residual.cpu", "len " + String(len(a)))
    return out^


def residual_add_kernel(
    a: Pointer[Float32, MutAnyOrigin],
    b: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    n: Int32,
):
    """GPU residual add: one thread per element."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        y[unsafe_offset=i] = a[unsafe_offset=i] + b[unsafe_offset=i]
