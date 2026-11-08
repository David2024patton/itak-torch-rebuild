# rms_norm.mojo - RMSNorm (weight-applied and weightless), Lane A4.
#
# Semantic reference (Rust, branch main):
#   crates/torch-math/src/norm.rs::rms_norm
#     out[i] = input[i] / sqrt(mean(input^2) + eps) * weight[i]
#     (mean of squares accumulated in f64, stored f32)
#   crates/muse-glimmer/src/config.rs (measured Glimmer values):
#     RMS eps 1e-5, read from the GGUF key
#     muse-glimmer.attention.layer_norm_rms_epsilon; the config
#     test asserts (cfg.rms_eps - 1e-5).abs() < 1e-9.
#   crates/cpu-compute/src/cpu.rs::rms_norm / rms_norm_plain:
#     the plain (weightless) form is a distinct step in the
#     Glimmer graph (embedding output norm), not a ones-weight
#     caller trick, so both forms live here.
#
# eps is a PARAMETER, never a literal inside the math: the caller
# sources it from config / GGUF metadata (src/config.mojo is the
# engine's config home; the loader lane supplies the file value).
# Wave E (Lane E3): the recorded Glimmer default that used to
# live here as GLIMMER_RMS_EPS now lives in src/config.mojo as
# DEFAULT_RMS_EPS / EngineConfig.rms_eps (env
# MOJO_ENGINE_RMS_EPS), the one home for defaults; callers and
# tests that name the real model take it from there.
#
# Named refusals: rms_norm_cpu_bad_shape / rms_norm_cpu_bad_eps /
# rms_norm_plain_cpu_bad_shape / rms_norm_plain_cpu_bad_eps.

from std.math import sqrt, isnan, isinf
from max.gpu import thread_idx, block_idx, block_dim, barrier
from max.gpu.memory import AddressSpace
from std.memory import stack_allocation

from diag import log_debug

comptime RMS_THREADS = 256


def _check_eps(eps: Float32, op: String) raises:
    if isnan(eps) or isinf(eps) or eps <= Float32(0.0):
        raise Error(
            op + "_bad_eps: eps must be finite and > 0, got " + String(eps)
        )


def rms_norm_cpu(
    x: List[Float32], weight: List[Float32], eps: Float32
) raises -> List[Float32]:
    """One row: out[i] = x[i] / sqrt(mean(x^2)+eps) * weight[i].

    Mean of squares accumulated in Float64, exactly as the Rust
    reference does, so wide rows (Glimmer hidden 6656) do not
    drift. For multi-row buffers call once per row (the row-wise
    driver in the layer lane composes rows, as Rust rms_rows
    does). Raises a named error on empty/mismatched input or a
    non-finite / non-positive eps.
    """
    if len(x) == 0:
        raise Error("rms_norm_cpu_bad_shape: input must not be empty")
    if len(weight) != len(x):
        raise Error(
            "rms_norm_cpu_bad_shape: weight has "
            + String(len(weight))
            + " elements, input has "
            + String(len(x))
        )
    _check_eps(eps, "rms_norm_cpu")
    var mean_sq = Float64(0.0)
    for i in range(len(x)):
        var v = Float64(x[i])
        mean_sq += v * v
    mean_sq /= Float64(len(x))
    var inv = Float64(1.0) / sqrt(mean_sq + Float64(eps))
    var out = List[Float32](capacity=len(x))
    out.resize(len(x), Float32(0.0))
    for i in range(len(x)):
        out[i] = Float32(Float64(x[i]) * inv * Float64(weight[i]))
    _ = log_debug("ops.rms_norm.cpu", "len " + String(len(x)))
    return out^


def rms_norm_plain_cpu(x: List[Float32], eps: Float32) raises -> List[Float32]:
    """Weightless RMSNorm for one row (Glimmer embedding norm)."""
    if len(x) == 0:
        raise Error("rms_norm_plain_cpu_bad_shape: input must not be empty")
    _check_eps(eps, "rms_norm_plain_cpu")
    var ones = List[Float32](capacity=len(x))
    ones.resize(len(x), Float32(1.0))
    # Same arithmetic path as the weighted form, per the Rust
    # cpu-compute comment (ones handed to the shared formula).
    return rms_norm_cpu(x, ones, eps)


def rms_norm_kernel(
    x: Pointer[Float32, MutAnyOrigin],
    w: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    width: Int32,
    eps: Float32,
):
    """GPU RMSNorm: one block per row, shared-memory reduction.

    Shapes are validated by the host before launch (the CPU twin
    is the named-refusal surface). Accumulation is f32 on device;
    the CPU twin (f64 accumulation) is the parity reference, so
    device bars are looser than CPU bars by design.
    """
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var n = Int(width)
    var base = row * n
    var shared = stack_allocation[
        RMS_THREADS, Float32, address_space=AddressSpace.SHARED
    ]()
    var acc = Float32(0.0)
    var i = tid
    while i < n:
        var v = x[unsafe_offset=base + i]
        acc += v * v
        i += RMS_THREADS
    shared[unsafe_offset=tid] = acc
    barrier()
    var stride = RMS_THREADS // 2
    while stride > 0:
        if tid < stride:
            shared[unsafe_offset=tid] = (
                shared[unsafe_offset=tid] + shared[unsafe_offset=tid + stride]
            )
        barrier()
        stride //= 2
    if tid == 0:
        shared[unsafe_offset=0] = Float32(1.0) / sqrt(
            shared[unsafe_offset=0] / Float32(n) + eps
        )
    barrier()
    var inv = shared[unsafe_offset=0]
    i = tid
    while i < n:
        y[unsafe_offset=base + i] = (
            x[unsafe_offset=base + i] * inv * w[unsafe_offset=i]
        )
        i += RMS_THREADS
