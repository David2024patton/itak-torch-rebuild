# K1 Q4_K GEMV port in Mojo (bench code for the Mojo kernel test;
# see bench/mojo-kernel-test-2026-11-07.md). NOT part of the
# shipped engine: a standalone bench program that runs the same
# Q4_K GEMV math as crates/wgpu-compute's K1 kernel (the r2sg
# shape's math: ggml Q4_K super-block dequant plus a dot product
# per output row) on the fixture k1_fixture_dump writes.
#
# Debug law: structured log lines behind MOJO_KERNEL_DEBUG (any
# value other than "0"/empty enables), every line named
# [k1_gemv_mojo][phase]; every bad input shape gets a named
# REFUSAL line and exit code 2.
#
# Usage:
#   k1_gemv <fixture_dir> [--rows N] [--cols N] [--iters N]
#           [--warmup N] [--out PATH]
# Defaults: rows 19968, cols 6656 (the fixture tensor's shape),
# iters 0 (correctness only), warmup 3, out <fixture_dir>/mojo_out_f32.bin.
# With --iters N > 0 the process additionally runs a timed loop
# of N launches after warmup and prints a MOJO_TIMING line whose
# GB/s counts the same streamed weight bytes the wgpu k1_bench
# counts (rows * row_bytes * iters / elapsed).

from std.os import getenv
from std.time import perf_counter_ns
from std.sys import argv, exit
from max.gpu import thread_idx, block_idx, block_dim, barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from std.memory import stack_allocation

comptime THREADS = 256
comptime SB_ELEMS = 256
comptime SB_BYTES = 144


def log_debug(phase: String, msg: String):
    var flag = getenv("MOJO_KERNEL_DEBUG")
    if flag != "" and flag != "0":
        print("[k1_gemv_mojo][", phase, "] ", msg, sep="")


def refuse(what: String, detail: String) -> None:
    print("REFUSAL k1_gemv_mojo: ", what, ": ", detail, sep="")
    exit(2)


def f16_to_f32(bits: UInt16) -> Float32:
    # Exact f16 -> f32: normals via (1 + frac/1024) * 2^(exp-15)
    # (both factors exact in f32, power-of-two scaling exact);
    # subnormals via frac * 2^-24 (exact); zero signed.
    var sign: Float32 = -1.0 if (bits >> 15) & 1 == 1 else 1.0
    var exp = Int((bits >> 10) & 0x1F)
    var frac = Int(bits & 0x3FF)
    if exp == 0:
        return sign * Float32(frac) * 5.9604644775390625e-08
    if exp == 31:
        # Weights never carry inf/NaN; refuse silently wrong math
        # by returning a NaN the comparison will catch.
        return Float32(0.0) / Float32(0.0)
    var v = Float32(1.0) + Float32(frac) / Float32(1024.0)
    var e = exp - 15
    var scale = Float32(1.0)
    if e >= 0:
        for _ in range(e):
            scale *= Float32(2.0)
    else:
        for _ in range(-e):
            scale /= Float32(2.0)
    return sign * v * scale


def scale_min_k4(
    j: Int, w: Pointer[UInt8, MutAnyOrigin], sb_off: Int
) -> Tuple[Float32, Float32]:
    # Mirrors gguf::dequant::scale_min_k4 over the 12 scale bytes
    # at sb_off + 4: returns (scale, min) as plain counts.
    if j < 4:
        var sc = Int(w[unsafe_offset=sb_off + 4 + j]) & 63
        var m = Int(w[unsafe_offset=sb_off + 8 + j]) & 63
        return (Float32(sc), Float32(m))
    else:
        var hi = Int(w[unsafe_offset=sb_off + 4 + (j + 4)])
        var sc = (hi & 0xF) | (
            (Int(w[unsafe_offset=sb_off + 4 + (j - 4)]) >> 6) << 4
        )
        var m = (hi >> 4) | ((Int(w[unsafe_offset=sb_off + 4 + j]) >> 6) << 4)
        return (Float32(sc), Float32(m))


def k1_gemv_kernel(
    w: Pointer[UInt8, MutAnyOrigin],
    x: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    cols: Int32,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var row_bytes = (Int(cols) // SB_ELEMS) * SB_BYTES
    var base = row * row_bytes
    var shared = stack_allocation[
        THREADS, Float32, address_space=AddressSpace.SHARED
    ]()
    var acc = Float32(0.0)
    var steps = Int(cols) // THREADS
    for k in range(steps):
        var col = tid + k * THREADS
        var sb = col >> 8
        var within = col & 255
        var sb_off = base + sb * SB_BYTES
        var d_bits = UInt16(w[unsafe_offset=sb_off]) | (
            UInt16(w[unsafe_offset=sb_off + 1]) << 8
        )
        var dm_bits = UInt16(w[unsafe_offset=sb_off + 2]) | (
            UInt16(w[unsafe_offset=sb_off + 3]) << 8
        )
        var d = f16_to_f32(d_bits)
        var dmin = f16_to_f32(dm_bits)
        var g = within >> 6
        var p = within & 63
        var sc: Float32
        var m: Float32
        var q: Float32
        if p < 32:
            var sm = scale_min_k4(2 * g, w, sb_off)
            sc = sm[0]
            m = sm[1]
            q = Float32(Int(w[unsafe_offset=sb_off + 16 + g * 32 + p]) & 0xF)
        else:
            var sm = scale_min_k4(2 * g + 1, w, sb_off)
            sc = sm[0]
            m = sm[1]
            q = Float32(
                Int(w[unsafe_offset=sb_off + 16 + g * 32 + (p - 32)]) >> 4
            )
        var val = d * sc * q - dmin * m
        acc += val * x[unsafe_offset=col]
    shared[unsafe_offset=tid] = acc
    barrier()
    var stride = THREADS // 2
    while stride > 0:
        if tid < stride:
            shared[unsafe_offset=tid] = (
                shared[unsafe_offset=tid] + shared[unsafe_offset=tid + stride]
            )
        barrier()
        stride //= 2
    if tid == 0:
        y[unsafe_offset=row] = shared[unsafe_offset=0]


def read_file_exact(path: String, expected: Int) raises -> List[UInt8]:
    with open(path, "r") as f:
        var data = f.read_bytes(expected + 1)
        if len(data) != expected:
            print(
                "REFUSAL k1_gemv_mojo: fixture size mismatch: ",
                path,
                " has ",
                len(data),
                " readable bytes, expected ",
                expected,
                sep="",
            )
            exit(2)
        return data^


def arg_value(args: List[String], flag: String, default: String) -> String:
    for i in range(len(args) - 1):
        if args[i] == flag:
            return args[i + 1]
    return default


def main() raises:
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    if len(args) < 2:
        refuse(
            "usage",
            (
                "k1_gemv <fixture_dir> [--rows N] [--cols N] [--iters N]"
                " [--warmup N] [--out PATH]"
            ),
        )
    var fixture_dir = args[1]
    var rows = Int(arg_value(args, "--rows", "19968"))
    var cols = Int(arg_value(args, "--cols", "6656"))
    var iters = Int(arg_value(args, "--iters", "0"))
    var warmup = Int(arg_value(args, "--warmup", "3"))
    var out_path = arg_value(args, "--out", fixture_dir + "/mojo_out_f32.bin")
    if cols % SB_ELEMS != 0:
        refuse(
            "bad shape",
            "cols must be a multiple of 256 (Q4_K super-block), got "
            + String(cols),
        )
    if cols % THREADS != 0:
        refuse(
            "bad shape",
            "cols must be a multiple of the 256-thread block, got "
            + String(cols),
        )
    var row_bytes = (cols // SB_ELEMS) * SB_BYTES
    var weights_bytes = rows * row_bytes
    log_debug("load", "fixture dir " + fixture_dir)
    var w_host = read_file_exact(
        fixture_dir + "/weights_q4k.bin", weights_bytes
    )
    var x_host = read_file_exact(fixture_dir + "/input_f32.bin", cols * 4)
    log_debug(
        "load",
        "weights "
        + String(weights_bytes)
        + " bytes, input "
        + String(cols)
        + " f32",
    )

    var ctx = DeviceContext()
    log_debug("device", "DeviceContext created")
    var w_dev = ctx.enqueue_create_buffer[DType.uint8](weights_bytes)
    var x_dev = ctx.enqueue_create_buffer[DType.float32](cols)
    var o_dev = ctx.enqueue_create_buffer[DType.float32](rows)
    var w_src = ctx.enqueue_create_host_buffer[DType.uint8](weights_bytes)
    var x_src = ctx.enqueue_create_host_buffer[DType.float32](cols)
    for i in range(weights_bytes):
        w_src[i] = w_host[i]
    for i in range(cols):
        var p = x_host.unsafe_ptr().unsafe_bitcast[Float32]()
        x_src[i] = p[unsafe_offset=i]
    ctx.enqueue_copy(src_buf=w_src, dst_buf=w_dev)
    ctx.enqueue_copy(src_buf=x_src, dst_buf=x_dev)
    log_debug("device", "fixture uploaded")

    ctx.enqueue_function[k1_gemv_kernel](
        w_dev, x_dev, o_dev, Int32(cols), grid_dim=rows, block_dim=THREADS
    )
    ctx.synchronize()
    log_debug("run", "correctness launch done")

    var o_host = ctx.enqueue_create_host_buffer[DType.float32](rows)
    ctx.enqueue_copy(src_buf=o_dev, dst_buf=o_host)
    ctx.synchronize()
    var out_bytes = List[UInt8](capacity=rows * 4)
    out_bytes.resize(rows * 4, UInt8(0))
    var op = out_bytes.unsafe_ptr().unsafe_bitcast[Float32]()
    for i in range(rows):
        op[unsafe_offset=i] = o_host[i]
    with open(out_path, "w") as f:
        f.write_bytes(out_bytes)
    print("MOJO_OUT wrote", out_path, rows, "f32")

    if iters > 0:
        for _ in range(warmup):
            ctx.enqueue_function[k1_gemv_kernel](
                w_dev,
                x_dev,
                o_dev,
                Int32(cols),
                grid_dim=rows,
                block_dim=THREADS,
            )
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(iters):
            ctx.enqueue_function[k1_gemv_kernel](
                w_dev,
                x_dev,
                o_dev,
                Int32(cols),
                grid_dim=rows,
                block_dim=THREADS,
            )
        ctx.synchronize()
        var elapsed = Float64(perf_counter_ns() - t0) / 1.0e9
        var streamed = Float64(weights_bytes) * Float64(iters)
        var gbs = streamed / elapsed / 1.0e9
        print(
            "MOJO_TIMING iters",
            iters,
            "(+" + String(warmup) + " warmup):",
            Int(streamed),
            "bytes in",
            elapsed,
            "s =",
            gbs,
            "GB/s",
        )
        log_debug("timing", "loop done")
