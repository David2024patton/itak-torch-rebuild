# q5k_gemv.mojo - the Q5_K GEMV kernel (K3 family) for the
# Mojo/MAX engine. Lane A3.
#
# Real Glimmer shape served (census 2026-10-29, see
# crates/muse-glimmer/src/weights.rs: "output.weight: the
# untied output head (Q5_K in the real file)", and
# crates/wgpu-compute/examples/k3_check.rs):
#   output.weight  [6656 in, 202048 out] Q5_K
# The math mirrors crates/gguf/src/dequant.rs dequant_q5_k term
# for term via kernels/dequant.mojo (176-byte super-block:
# d, dmin, 12 scale bytes, qh[16..48], ql[48..176]).
# Straight port for correctness (DESIGN.md speed law).
#
from std.time import perf_counter_ns
from std.sys import argv, exit
from max.gpu import thread_idx, block_idx, barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from std.memory import stack_allocation

from diag import log_debug
from kernels.dequant import q5k_element_ptr, q5k_element_list, note_loaded

comptime THREADS = 256
comptime SB_ELEMS = 256
comptime SB_BYTES = 176


def refuse(what: String, detail: String) -> None:
    print("REFUSAL q5k_gemv_mojo: ", what, ": ", detail, sep="")
    exit(2)


def q5k_gemv_kernel(
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
        var sb_off = base + (col >> 8) * SB_BYTES
        var val = q5k_element_ptr(w, sb_off, col & 255)
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


def q5k_gemv_cpu(
    w: List[UInt8], x: List[Float32], rows: Int, cols: Int
) raises -> List[Float32]:
    """The Q5_K math on the host: dequant + dot per row.

    Identical math to q5k_gemv_kernel, sequential accumulation.
    Raises a named error on a bad shape.
    """
    if cols % SB_ELEMS != 0:
        raise Error(
            "q5k_gemv_cpu_bad_shape: cols must be a multiple of 256"
            " (Q5_K super-block), got "
            + String(cols)
        )
    var row_bytes = (cols // SB_ELEMS) * SB_BYTES
    if len(w) != rows * row_bytes:
        raise Error(
            "q5k_gemv_cpu_bad_shape: weights have "
            + String(len(w))
            + " bytes, expected "
            + String(rows * row_bytes)
        )
    if len(x) != cols:
        raise Error(
            "q5k_gemv_cpu_bad_shape: input has "
            + String(len(x))
            + " f32, expected "
            + String(cols)
        )
    var y = List[Float32](capacity=rows)
    y.resize(rows, Float32(0.0))
    for row in range(rows):
        var base = row * row_bytes
        var acc = Float32(0.0)
        for col in range(cols):
            var sb_off = base + (col >> 8) * SB_BYTES
            acc += q5k_element_list(w, sb_off, col & 255) * x[col]
        y[row] = acc
    _ = note_loaded("q5k_gemv.cpu", rows, cols)
    return y^


def read_file_exact(path: String, expected: Int) raises -> List[UInt8]:
    with open(path, "r") as f:
        var data = f.read_bytes(expected + 1)
        if len(data) != expected:
            print(
                "REFUSAL q5k_gemv_mojo: fixture size mismatch: ",
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
                "q5k_gemv <fixture_dir> [--rows N] [--cols N] [--iters N]"
                " [--warmup N] [--out PATH]"
            ),
        )
    var fixture_dir = args[1]
    var rows = Int(arg_value(args, "--rows", "202048"))
    var cols = Int(arg_value(args, "--cols", "6656"))
    var iters = Int(arg_value(args, "--iters", "0"))
    var warmup = Int(arg_value(args, "--warmup", "3"))
    var out_path = arg_value(args, "--out", fixture_dir + "/mojo_out_f32.bin")
    if cols % SB_ELEMS != 0:
        refuse(
            "bad shape",
            "cols must be a multiple of 256 (Q5_K super-block), got "
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
    _ = log_debug("q5k_gemv.load", "fixture dir " + fixture_dir)
    var w_host = read_file_exact(
        fixture_dir + "/weights_q5k.bin", weights_bytes
    )
    var x_host = read_file_exact(fixture_dir + "/input_f32.bin", cols * 4)
    _ = log_debug(
        "q5k_gemv.load",
        "weights "
        + String(weights_bytes)
        + " bytes, input "
        + String(cols)
        + " f32",
    )

    var ctx = DeviceContext()
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
    _ = log_debug("q5k_gemv.device", "fixture uploaded")

    ctx.enqueue_function[q5k_gemv_kernel](
        w_dev, x_dev, o_dev, Int32(cols), grid_dim=rows, block_dim=THREADS
    )
    ctx.synchronize()
    _ = log_debug("q5k_gemv.run", "correctness launch done")

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
            ctx.enqueue_function[q5k_gemv_kernel](
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
            ctx.enqueue_function[q5k_gemv_kernel](
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
