# gpu_proof_attention_direct.mojo - Wave E lane E2 device
# proof for attention_tiled_direct_kernel. NOT a TestSuite
# file: run explicitly on a GPU host with
#   mojo run -I src tests/gpu_proof_attention_direct.mojo
#
# Fixtures: PRE-ROTATED stores (the kernels' contract) at
# the Glimmer geometry (32 Q heads / 2 KV heads / head_dim
# 128), stored 16 / 128 / 512 - the chooser's whole
# single-split regime (span 512) including its top edge.
# References: attention_direct_cpu (the direct twin) and
# attention_split_cpu at splits == 1 (the production path
# this replaces in the short regime), both on the same
# fixture. Bars (the Wave B gpu_proof_attention bars):
# cosine >= 0.999999999 and rel (max abs diff / max abs
# expected) <= 1e-5 for the GPU output against BOTH
# references; the two CPU references must agree EXACTLY
# (max diff 0.0, the VM suite's result on device-independent
# code, re-asserted here so the fixture is the same one).
#
# Timing: interleaved (split-pair, direct) launch pairs at
# each fixture, per-launch averages from perf_counter_ns
# around enqueue+synchronize, the D2 proof's pattern;
# buffers preallocated once per fixture, so the figures
# are kernel+launch time, not allocator time. The direct
# arm allocates no partial buffers in this proof either -
# that saving belongs to the production wiring and shows
# in the end-to-end A/B, not here.

from std.math import sqrt
from std.sys import exit
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer

from attention.attention import (
    ATTN_THREADS,
    attention_direct_cpu,
    attention_merge_kernel,
    attention_split_cpu,
    attention_tiled_direct_kernel,
    attention_tiled_kernel,
)

comptime NQ = 32
comptime NKV = 2
comptime HD = 128
comptime TILE = 32
comptime REPS = 30


def _det(i: Int) -> Float32:
    return Float32(((i * 37 + 13) % 31) - 15) / Float32(8.0)


def _make(n: Int, seed: Int) -> List[Float32]:
    var x = List[Float32](capacity=n)
    x.resize(n, Float32(0.0))
    for i in range(n):
        x[i] = _det(i + seed * 101)
    return x^


def _metrics(
    got: List[Float32], expected: List[Float32]
) -> Tuple[Float64, Float64]:
    var dot = Float64(0.0)
    var norm_g = Float64(0.0)
    var norm_e = Float64(0.0)
    var max_abs = Float64(0.0)
    var max_e = Float64(0.0)
    for i in range(len(expected)):
        var g = Float64(got[i])
        var e = Float64(expected[i])
        dot += g * e
        norm_g += g * g
        norm_e += e * e
        var d = g - e
        if d < 0:
            d = -d
        if d > max_abs:
            max_abs = d
        var ae = e
        if ae < 0:
            ae = -ae
        if ae > max_e:
            max_e = ae
    var cosine = dot / (sqrt(norm_g) * sqrt(norm_e))
    return (cosine, max_abs / max_e)


def _bar(cosine: Float64, rel: Float64) -> Bool:
    return cosine >= 0.999999999 and rel <= 1.0e-5


def _upload(
    mut ctx: DeviceContext, data: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    var dev = ctx.enqueue_create_buffer[DType.float32](len(data))
    var host = ctx.enqueue_create_host_buffer[DType.float32](len(data))
    for i in range(len(data)):
        host[i] = data[i]
    ctx.enqueue_copy(src_buf=host, dst_buf=dev)
    return dev^


def _download(
    mut ctx: DeviceContext, dev: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(src_buf=dev, dst_buf=host)
    ctx.synchronize()
    var out = List[Float32](capacity=n)
    out.resize(n, Float32(0.0))
    for i in range(n):
        out[i] = host[i]
    return out^


def run() raises:
    var ctx = DeviceContext()
    var ok = True
    var scale = Float32(1.0) / sqrt(Float32(HD))
    for stored in [16, 128, 512]:
        var q = _make(NQ * HD, 1)
        var keys = _make(stored * NKV * HD, 2)
        var values = _make(stored * NKV * HD, 3)
        var ref_direct = attention_direct_cpu(
            q, keys, values, stored, NQ, NKV, HD, TILE
        )
        var ref_split = attention_split_cpu(
            q, keys, values, stored, NQ, NKV, HD, 1, TILE
        )
        var twin_diff = Float64(0.0)
        for i in range(len(ref_direct)):
            var d = Float64(ref_direct[i]) - Float64(ref_split[i])
            if d < 0:
                d = -d
            if d > twin_diff:
                twin_diff = d
        var q_dev = _upload(ctx, q)
        var k_dev = _upload(ctx, keys)
        var v_dev = _upload(ctx, values)
        var y_dev = ctx.enqueue_create_buffer[DType.float32](NQ * HD)
        var p_out = ctx.enqueue_create_buffer[DType.float32](NQ * HD)
        var p_max = ctx.enqueue_create_buffer[DType.float32](NQ)
        var p_sum = ctx.enqueue_create_buffer[DType.float32](NQ)
        # Direct arm correctness.
        ctx.enqueue_function[attention_tiled_direct_kernel[TILE]](
            q_dev,
            k_dev,
            v_dev,
            y_dev,
            Int32(stored),
            Int32(NKV),
            Int32(NQ),
            Int32(HD),
            Int32(NQ // NKV),
            scale,
            grid_dim=NQ,
            block_dim=ATTN_THREADS,
        )
        ctx.synchronize()
        var got_direct = _download(ctx, y_dev, NQ * HD)
        var m1 = _metrics(got_direct, ref_direct)
        var m2 = _metrics(got_direct, ref_split)
        var cok = _bar(m1[0], m1[1]) and _bar(m2[0], m2[1]) and twin_diff == 0.0
        ok = ok and cok
        print(
            "DIRECT_PROOF stored="
            + String(stored)
            + " cosine_vs_direct_twin="
            + String(m1[0])
            + " rel_vs_direct_twin="
            + String(m1[1])
            + " cosine_vs_split1="
            + String(m2[0])
            + " rel_vs_split1="
            + String(m2[1])
            + " twin_exact_diff="
            + String(twin_diff)
            + " ok="
            + String(1 if cok else 0)
        )
        # Interleaved timing: split pair vs direct.
        var pair_ns = 0
        var direct_ns = 0
        for rep in range(REPS):
            var t0 = perf_counter_ns()
            if rep % 2 == 0:
                ctx.enqueue_function[attention_tiled_kernel[TILE]](
                    q_dev,
                    k_dev,
                    v_dev,
                    p_out,
                    p_max,
                    p_sum,
                    Int32(stored),
                    Int32(NKV),
                    Int32(NQ),
                    Int32(HD),
                    Int32(NQ // NKV),
                    scale,
                    Int32(1),
                    grid_dim=NQ,
                    block_dim=ATTN_THREADS,
                )
                ctx.enqueue_function[attention_merge_kernel](
                    p_out,
                    p_max,
                    p_sum,
                    y_dev,
                    Int32(NQ),
                    Int32(HD),
                    Int32(1),
                    grid_dim=NQ,
                    block_dim=ATTN_THREADS,
                )
                ctx.synchronize()
                pair_ns += Int(perf_counter_ns() - t0)
                var t1 = perf_counter_ns()
                ctx.enqueue_function[attention_tiled_direct_kernel[TILE]](
                    q_dev,
                    k_dev,
                    v_dev,
                    y_dev,
                    Int32(stored),
                    Int32(NKV),
                    Int32(NQ),
                    Int32(HD),
                    Int32(NQ // NKV),
                    scale,
                    grid_dim=NQ,
                    block_dim=ATTN_THREADS,
                )
                ctx.synchronize()
                direct_ns += Int(perf_counter_ns() - t1)
            else:
                ctx.enqueue_function[attention_tiled_direct_kernel[TILE]](
                    q_dev,
                    k_dev,
                    v_dev,
                    y_dev,
                    Int32(stored),
                    Int32(NKV),
                    Int32(NQ),
                    Int32(HD),
                    Int32(NQ // NKV),
                    scale,
                    grid_dim=NQ,
                    block_dim=ATTN_THREADS,
                )
                ctx.synchronize()
                direct_ns += Int(perf_counter_ns() - t0)
                var t1 = perf_counter_ns()
                ctx.enqueue_function[attention_tiled_kernel[TILE]](
                    q_dev,
                    k_dev,
                    v_dev,
                    p_out,
                    p_max,
                    p_sum,
                    Int32(stored),
                    Int32(NKV),
                    Int32(NQ),
                    Int32(HD),
                    Int32(NQ // NKV),
                    scale,
                    Int32(1),
                    grid_dim=NQ,
                    block_dim=ATTN_THREADS,
                )
                ctx.enqueue_function[attention_merge_kernel](
                    p_out,
                    p_max,
                    p_sum,
                    y_dev,
                    Int32(NQ),
                    Int32(HD),
                    Int32(1),
                    grid_dim=NQ,
                    block_dim=ATTN_THREADS,
                )
                ctx.synchronize()
                pair_ns += Int(perf_counter_ns() - t1)
        var pair_us = Float64(pair_ns) / Float64(REPS) / 1000.0
        var direct_us = Float64(direct_ns) / Float64(REPS) / 1000.0
        print(
            "DIRECT_TIMING stored="
            + String(stored)
            + " pair_us="
            + String(pair_us)
            + " direct_us="
            + String(direct_us)
            + " ratio_pair_over_direct="
            + String(pair_us / direct_us)
        )
    if ok:
        print("GPU_PROOF_ATTENTION_DIRECT PASS")
    else:
        print("GPU_PROOF_ATTENTION_DIRECT FAIL")
        exit(1)


def main():
    try:
        run()
    except e:
        print("REFUSAL", e)
        exit(2)
