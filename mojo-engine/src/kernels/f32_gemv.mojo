# f32_gemv.mojo - the plain-float GEMV kernels (F32 and F16
# weight storage) for the Mojo/MAX engine. Lane A3.
#
# Real Glimmer tensors in this family (census 2026-10-29): the
# norm weights are F32 vectors (attn_norm, attn_q_norm and
# attn_k_norm [128], post_attention_norm, ffn_norm,
# post_ffw_norm, output_norm [6656]). Norms are consumed as
# vectors, not GEMV matrices, in the forward; this module is the
# family's GEMV citizen for any plain-float matrix the loader
# hands the engine (and the parity baseline the quantized
# kernels are checked against), plus the F16 storage variant:
# F16 appears in the file family the gguf loader serves
# (crates/gguf/src/dequant.rs F16 arm) and dequantizes exactly
# via kernels/dequant.f16_to_f32.
#
# F32 weights arrive as a Float32 list (host) / pointer (device);
# F16 weights arrive as raw bytes, two per element, little
# endian, exactly as stored. Every bad shape refuses by name.
#
# Standalone usage mirrors k1_gemv:
#   f32_gemv <fixture_dir> [--dtype f32|f16] [--rows N]
#            [--cols N] [--iters N] [--warmup N] [--out PATH]
# Fixture files: weights_f32.bin (f32) or weights_f16.bin
# (f16), plus input_f32.bin. Defaults rows 64, cols 6656.

from std.time import perf_counter_ns
from std.sys import argv, exit
from max.gpu import thread_idx, block_idx, barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from std.memory import stack_allocation

from diag import log_debug
from kernels.dequant import f16_to_f32, note_loaded

comptime THREADS = 256


def refuse(what: String, detail: String) -> None:
    print("REFUSAL f32_gemv_mojo: ", what, ": ", detail, sep="")
    exit(2)


def f32_gemv_kernel(
    w: Pointer[Float32, MutAnyOrigin],
    x: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    cols: Int32,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var base = row * Int(cols)
    var shared = stack_allocation[
        THREADS, Float32, address_space=AddressSpace.SHARED
    ]()
    var acc = Float32(0.0)
    var steps = Int(cols) // THREADS
    for k in range(steps):
        var col = tid + k * THREADS
        acc += w[unsafe_offset=base + col] * x[unsafe_offset=col]
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


def f16_gemv_kernel(
    w: Pointer[UInt8, MutAnyOrigin],
    x: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    cols: Int32,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var base = row * Int(cols) * 2
    var shared = stack_allocation[
        THREADS, Float32, address_space=AddressSpace.SHARED
    ]()
    var acc = Float32(0.0)
    var steps = Int(cols) // THREADS
    for k in range(steps):
        var col = tid + k * THREADS
        var off = base + col * 2
        var bits = UInt16(w[unsafe_offset=off]) | (
            UInt16(w[unsafe_offset=off + 1]) << 8
        )
        acc += f16_to_f32(bits) * x[unsafe_offset=col]
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


def check_shape(
    kind: String, w_len: Int, x_len: Int, rows: Int, cols: Int, elem_bytes: Int
) raises:
    if cols <= 0 or rows <= 0:
        raise Error(
            kind
            + "_bad_shape: rows and cols must be positive, got "
            + String(rows)
            + " x "
            + String(cols)
        )
    if w_len != rows * cols * elem_bytes:
        raise Error(
            kind
            + "_bad_shape: weights have "
            + String(w_len)
            + " storage units, expected "
            + String(rows * cols * elem_bytes)
        )
    if x_len != cols:
        raise Error(
            kind
            + "_bad_shape: input has "
            + String(x_len)
            + " f32, expected "
            + String(cols)
        )


def f32_gemv_cpu(
    w: List[Float32], x: List[Float32], rows: Int, cols: Int
) raises -> List[Float32]:
    """Plain F32 GEMV on the host. Raises a named error on a bad shape."""
    check_shape("f32_gemv_cpu", len(w), len(x), rows, cols, 1)
    var y = List[Float32](capacity=rows)
    y.resize(rows, Float32(0.0))
    for row in range(rows):
        var acc = Float32(0.0)
        for col in range(cols):
            acc += w[row * cols + col] * x[col]
        y[row] = acc
    _ = note_loaded("f32_gemv.cpu", rows, cols)
    return y^


def f16_gemv_cpu(
    w: List[UInt8], x: List[Float32], rows: Int, cols: Int
) raises -> List[Float32]:
    """F16-stored GEMV on the host: exact f16 decode + dot per row.

    Weights are raw file bytes, two per element. Raises a named
    error on a bad shape.
    """
    check_shape("f16_gemv_cpu", len(w), len(x), rows, cols, 2)
    var y = List[Float32](capacity=rows)
    y.resize(rows, Float32(0.0))
    for row in range(rows):
        var acc = Float32(0.0)
        for col in range(cols):
            var off = (row * cols + col) * 2
            var bits = UInt16(w[off]) | (UInt16(w[off + 1]) << 8)
            acc += f16_to_f32(bits) * x[col]
        y[row] = acc
    _ = note_loaded("f16_gemv.cpu", rows, cols)
    return y^


def read_file_exact(path: String, expected: Int) raises -> List[UInt8]:
    with open(path, "r") as f:
        var data = f.read_bytes(expected + 1)
        if len(data) != expected:
            print(
                "REFUSAL f32_gemv_mojo: fixture size mismatch: ",
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
                "f32_gemv <fixture_dir> [--dtype f32|f16] [--rows N]"
                " [--cols N] [--iters N] [--warmup N] [--out PATH]"
            ),
        )
    var fixture_dir = args[1]
    var dtype = arg_value(args, "--dtype", "f32")
    if dtype != "f32" and dtype != "f16":
        refuse("bad dtype", "expected f32 or f16, got " + dtype)
    var rows = Int(arg_value(args, "--rows", "64"))
    var cols = Int(arg_value(args, "--cols", "6656"))
    var iters = Int(arg_value(args, "--iters", "0"))
    var warmup = Int(arg_value(args, "--warmup", "3"))
    var out_path = arg_value(args, "--out", fixture_dir + "/mojo_out_f32.bin")
    if cols % THREADS != 0:
        refuse(
            "bad shape",
            "cols must be a multiple of the 256-thread block, got "
            + String(cols),
        )
    var elem_bytes = 4 if dtype == "f32" else 2
    var weights_bytes = rows * cols * elem_bytes
    var wname = "/weights_f32.bin" if dtype == "f32" else "/weights_f16.bin"
    var w_host = read_file_exact(fixture_dir + wname, weights_bytes)
    var x_host = read_file_exact(fixture_dir + "/input_f32.bin", cols * 4)
    _ = log_debug(
        "f32_gemv.load",
        "dtype "
        + dtype
        + ", weights "
        + String(weights_bytes)
        + " bytes, input "
        + String(cols)
        + " f32",
    )

    var ctx = DeviceContext()
    var x_dev = ctx.enqueue_create_buffer[DType.float32](cols)
    var o_dev = ctx.enqueue_create_buffer[DType.float32](rows)
    var x_src = ctx.enqueue_create_host_buffer[DType.float32](cols)
    for i in range(cols):
        var p = x_host.unsafe_ptr().unsafe_bitcast[Float32]()
        x_src[i] = p[unsafe_offset=i]
    ctx.enqueue_copy(src_buf=x_src, dst_buf=x_dev)

    if dtype == "f32":
        var w_dev = ctx.enqueue_create_buffer[DType.float32](rows * cols)
        var w_src = ctx.enqueue_create_host_buffer[DType.float32](rows * cols)
        var wp = w_host.unsafe_ptr().unsafe_bitcast[Float32]()
        for i in range(rows * cols):
            w_src[i] = wp[unsafe_offset=i]
        ctx.enqueue_copy(src_buf=w_src, dst_buf=w_dev)
        ctx.enqueue_function[f32_gemv_kernel](
            w_dev, x_dev, o_dev, Int32(cols), grid_dim=rows, block_dim=THREADS
        )
        ctx.synchronize()
        if iters > 0:
            for _ in range(warmup):
                ctx.enqueue_function[f32_gemv_kernel](
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
                ctx.enqueue_function[f32_gemv_kernel](
                    w_dev,
                    x_dev,
                    o_dev,
                    Int32(cols),
                    grid_dim=rows,
                    block_dim=THREADS,
                )
            ctx.synchronize()
            var elapsed = Float64(perf_counter_ns() - t0) / 1.0e9
            var gbs = Float64(weights_bytes) * Float64(iters) / elapsed / 1.0e9
            print("MOJO_TIMING iters", iters, "GB/s", gbs)
    else:
        var w_dev = ctx.enqueue_create_buffer[DType.uint8](weights_bytes)
        var w_src = ctx.enqueue_create_host_buffer[DType.uint8](weights_bytes)
        for i in range(weights_bytes):
            w_src[i] = w_host[i]
        ctx.enqueue_copy(src_buf=w_src, dst_buf=w_dev)
        ctx.enqueue_function[f16_gemv_kernel](
            w_dev, x_dev, o_dev, Int32(cols), grid_dim=rows, block_dim=THREADS
        )
        ctx.synchronize()
        if iters > 0:
            for _ in range(warmup):
                ctx.enqueue_function[f16_gemv_kernel](
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
                ctx.enqueue_function[f16_gemv_kernel](
                    w_dev,
                    x_dev,
                    o_dev,
                    Int32(cols),
                    grid_dim=rows,
                    block_dim=THREADS,
                )
            ctx.synchronize()
            var elapsed = Float64(perf_counter_ns() - t0) / 1.0e9
            var gbs = Float64(weights_bytes) * Float64(iters) / elapsed / 1.0e9
            print("MOJO_TIMING iters", iters, "GB/s", gbs)
    _ = log_debug("f32_gemv.run", "correctness launch done")

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
