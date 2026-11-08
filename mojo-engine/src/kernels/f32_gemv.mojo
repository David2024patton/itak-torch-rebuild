# f32_gemv.mojo - the plain-float GEMV kernels (F32 and F16
# weight storage) for the Mojo/MAX engine. Lane A3 (straight
# ports), tuned in lane mojo-d3-gemv (Wave D, 2026-11-07).
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
# TUNED SHAPES (Wave D): the tuned-K1 lessons translated to
# plain floats, with opposite outcomes per dtype, both
# measured in cleared-card window 1 (bench note
# bench/mojo-d3-gemv-ladder-2026-11-07.md). The F16 kernel
# took the full translation and kept it: 2 rows per
# 128-thread workgroup as 8 groups of 16, each group
# striding the row in 256-element segments, one thread
# owning four quads per segment (quads t, t+16, t+32, t+48,
# so consecutive threads read consecutive quads), each quad
# arriving as two u32 words decoded through the bit-trick
# f16 route, activation quads loaded once per segment and
# shared by both rows, K1's segmented shuffle_xor butterfly
# plus a short shared-memory finish: 180.19 -> 717.94 GB/s
# median (3.98x the port). The same shape applied to F32
# measured 415.42 against the port's 432.13 GB/s median,
# slower in every rep - the port was already a clean
# streaming kernel and the vec4 segment shape bought
# nothing - so the F32 tuned iteration was REJECTED and
# f32_gemv_kernel below is the Wave A port, unchanged. The
# CPU twins below are byte-behaved and unchanged.
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
from max.gpu.primitives.warp import shuffle_xor
from std.memory import stack_allocation, bitcast

from diag import log_debug
from kernels.dequant import f16_to_f32, note_loaded

comptime THREADS = 128
comptime F32_THREADS = 256
comptime ROWS_PER_WG = 2
comptime GROUP_THREADS = 16
comptime GROUPS = THREADS // GROUP_THREADS
comptime SEG_ELEMS = 256


def refuse(what: String, detail: String) -> None:
    print("REFUSAL f32_gemv_mojo: ", what, ": ", detail, sep="")
    exit(2)


def f16_to_f32_gpu(bits: UInt32) -> Float32:
    # Bit-trick f16 -> f32 for the GPU path, the same
    # construction k1_gemv.mojo carries (and the shipped wgpu
    # kernels use): normals rebias the exponent into f32 bits
    # directly, subnormals go through frac * 2^-24 (exact),
    # inf/NaN keep their class.
    var sign = (bits >> 15) & 1
    var exp = (bits >> 10) & 31
    var frac = bits & 1023
    if exp == 31:
        return bitcast[DType.float32, 1](
            SIMD[DType.uint32, 1]((sign << 31) | 0x7F800000 | (frac << 13))
        )
    if exp == 0:
        var v = Float32(frac) * Float32(5.9604644775390625e-08)
        if sign == 1:
            return -v
        return v
    return bitcast[DType.float32, 1](
        SIMD[DType.uint32, 1]((sign << 31) | ((exp + 112) << 23) | (frac << 13))
    )


def f16_quad_from_words(lo: UInt32, hi: UInt32) -> SIMD[DType.float32, 4]:
    # One F16 quad (4 halves, little endian across the two
    # words) decoded to f32, element order preserved.
    var out = SIMD[DType.float32, 4]()
    out[0] = f16_to_f32_gpu(lo & 0xFFFF)
    out[1] = f16_to_f32_gpu(lo >> 16)
    out[2] = f16_to_f32_gpu(hi & 0xFFFF)
    out[3] = f16_to_f32_gpu(hi >> 16)
    return out


def f32_gemv_kernel(
    w: Pointer[Float32, MutAnyOrigin],
    x: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    cols: Int32,
):
    # The Wave A straight port, restored after the tuned
    # iteration was measured and rejected (see the header):
    # one row per 256-thread workgroup, scalar loads, a
    # shared-memory tree reduction. On the synthetic timing
    # fixture the port's streaming shape beat the vec4
    # segment shape in every rep, so this kernel keeps it.
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var base = row * Int(cols)
    var shared = stack_allocation[
        F32_THREADS, Float32, address_space=AddressSpace.SHARED
    ]()
    var acc = Float32(0.0)
    var steps = Int(cols) // F32_THREADS
    for k in range(steps):
        var col = tid + k * F32_THREADS
        acc += w[unsafe_offset=base + col] * x[unsafe_offset=col]
    shared[unsafe_offset=tid] = acc
    barrier()
    var stride = F32_THREADS // 2
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
    # The F32 kernel's tuned shape over F16 storage: weights
    # arrive as raw half bytes, each quad as two u32 words
    # decoded by f16_quad_from_words. Quad element e sits at
    # byte 2e, a multiple of 8, so both words are aligned.
    var lane = Int(thread_idx.x)
    var group = lane // GROUP_THREADS
    var t = lane % GROUP_THREADS
    var row0 = Int(block_idx.x) * ROWS_PER_WG
    var w32 = w.unsafe_bitcast[UInt32]()
    var words_per_row = Int(cols) // 2
    var base0 = row0 * words_per_row
    var base1 = (row0 + 1) * words_per_row
    var shared = stack_allocation[
        ROWS_PER_WG * GROUPS, Float32, address_space=AddressSpace.SHARED
    ]()
    var acc0 = Float32(0.0)
    var acc1 = Float32(0.0)
    var segments = Int(cols) // SEG_ELEMS
    for seg in range(group, segments, GROUPS):
        var xb = seg * SEG_ELEMS + t * 4
        var xa = x.unsafe_load[width=4](xb)
        var xbq = x.unsafe_load[width=4](xb + 64)
        var xc = x.unsafe_load[width=4](xb + 128)
        var xd = x.unsafe_load[width=4](xb + 192)
        # Weight word-pair index of a quad: element e is half
        # e, word e // 2; a quad starts at an even half pair.
        var wb = seg * (SEG_ELEMS // 2) + t * 2
        acc0 += (
            f16_quad_from_words(
                w32.unsafe_load[width=1](base0 + wb),
                w32.unsafe_load[width=1](base0 + wb + 1),
            )
            * xa
        ).reduce_add()
        acc0 += (
            f16_quad_from_words(
                w32.unsafe_load[width=1](base0 + wb + 32),
                w32.unsafe_load[width=1](base0 + wb + 33),
            )
            * xbq
        ).reduce_add()
        acc0 += (
            f16_quad_from_words(
                w32.unsafe_load[width=1](base0 + wb + 64),
                w32.unsafe_load[width=1](base0 + wb + 65),
            )
            * xc
        ).reduce_add()
        acc0 += (
            f16_quad_from_words(
                w32.unsafe_load[width=1](base0 + wb + 96),
                w32.unsafe_load[width=1](base0 + wb + 97),
            )
            * xd
        ).reduce_add()
        acc1 += (
            f16_quad_from_words(
                w32.unsafe_load[width=1](base1 + wb),
                w32.unsafe_load[width=1](base1 + wb + 1),
            )
            * xa
        ).reduce_add()
        acc1 += (
            f16_quad_from_words(
                w32.unsafe_load[width=1](base1 + wb + 32),
                w32.unsafe_load[width=1](base1 + wb + 33),
            )
            * xbq
        ).reduce_add()
        acc1 += (
            f16_quad_from_words(
                w32.unsafe_load[width=1](base1 + wb + 64),
                w32.unsafe_load[width=1](base1 + wb + 65),
            )
            * xc
        ).reduce_add()
        acc1 += (
            f16_quad_from_words(
                w32.unsafe_load[width=1](base1 + wb + 96),
                w32.unsafe_load[width=1](base1 + wb + 97),
            )
            * xd
        ).reduce_add()
    acc0 += shuffle_xor(acc0, UInt32(1))
    acc0 += shuffle_xor(acc0, UInt32(2))
    acc0 += shuffle_xor(acc0, UInt32(4))
    acc0 += shuffle_xor(acc0, UInt32(8))
    acc1 += shuffle_xor(acc1, UInt32(1))
    acc1 += shuffle_xor(acc1, UInt32(2))
    acc1 += shuffle_xor(acc1, UInt32(4))
    acc1 += shuffle_xor(acc1, UInt32(8))
    if t == 0:
        shared[unsafe_offset=group] = acc0
        shared[unsafe_offset=GROUPS + group] = acc1
    barrier()
    if lane == 0:
        var s0 = Float32(0.0)
        var s1 = Float32(0.0)
        for g in range(GROUPS):
            s0 += shared[unsafe_offset=g]
            s1 += shared[unsafe_offset=GROUPS + g]
        y[unsafe_offset=row0] = s0
        y[unsafe_offset=row0 + 1] = s1


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
    if cols % SEG_ELEMS != 0:
        refuse(
            "bad shape",
            "cols must be a multiple of the 256-element segment, got "
            + String(cols),
        )
    if dtype == "f16" and rows % ROWS_PER_WG != 0:
        refuse(
            "bad shape",
            "rows must be a multiple of the 2-row workgroup tile, got "
            + String(rows),
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
            w_dev,
            x_dev,
            o_dev,
            Int32(cols),
            grid_dim=rows,
            block_dim=F32_THREADS,
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
                    block_dim=F32_THREADS,
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
                    block_dim=F32_THREADS,
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
            w_dev,
            x_dev,
            o_dev,
            Int32(cols),
            grid_dim=rows // ROWS_PER_WG,
            block_dim=THREADS,
        )
        ctx.synchronize()
        if iters > 0:
            for _ in range(warmup):
                ctx.enqueue_function[f16_gemv_kernel](
                    w_dev,
                    x_dev,
                    o_dev,
                    Int32(cols),
                    grid_dim=rows // ROWS_PER_WG,
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
                    grid_dim=rows // ROWS_PER_WG,
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
