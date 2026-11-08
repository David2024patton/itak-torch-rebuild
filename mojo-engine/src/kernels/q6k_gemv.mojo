# q6k_gemv.mojo - the Q6_K GEMV kernel (K2 family) for the
# Mojo/MAX engine. Lane A3 (straight port), tuned in lane
# mojo-d3-gemv (Wave D, 2026-11-07).
#
# Real Glimmer shapes served (census 2026-10-29, see
# crates/muse-glimmer/src/weights.rs and the draft census in
# bench/dflash-e3-2026-10-30.md, same architecture):
#   blk.*.ffn_down.weight  [19968 in, 6656 out] Q6_K
#   blk.*.attn_v.weight    [6656 in, 1024 out]  Q6_K
# The math mirrors crates/gguf/src/dequant.rs dequant_q6_k term
# for term via kernels/dequant.mojo (the CPU twin below is
# byte-behaved and unchanged by the tuning). The kernel reads
# the RAW 210-byte file layout directly, so no repack is
# needed at load.
#
# TUNED SHAPE (Wave D): the shipped wgpu K2 structure
# (crates/wgpu-compute/src/kernels/k2_q6k_gemv.rs, its
# shipped_shape r1sg: one row per 256-thread workgroup, median
# 831.9 GB/s on this card) translated to the raw layout:
# 16 groups of 16 threads, one group per super-block in
# flight, each thread owning the four streams' quads at its
# byte offset within its block half (llama.cpp's
# mul_mat_vec_q6_k assignment, the same one K2's shader
# carries). Per super-block the thread loads its ql words
# once, its qh word once, the block's f16 super-scale once,
# and the four signed sub-block scales it needs as single
# bytes; the 6-bit quants are rebuilt as whole SIMD quads
# (nibble plane plus the qh bit-pair plane, ggml's masks), so
# no per-element dequant recompute remains. Activation quads
# are loaded once per super-block. The reduction is K1's:
# a segmented shuffle_xor butterfly inside each 16-thread
# group plus a short shared-memory finish across groups.
#
# One layout fact drives the load route: a Q6_K super-block
# is 210 bytes, not a multiple of four, so odd blocks start
# two bytes into a u32 (the wgpu K2 upload repacks to a
# 212-byte stride to dodge exactly this; this engine's device
# weights carry the raw file bytes, so the kernel instead
# rebuilds each u32 from its two aligned u16 halves - every
# offset it builds a word at is even). The f16 super-scale is
# read as a single u16 so the last block of a tensor never
# reads past its end.
#
# Every bad input shape gets a named REFUSAL line and exit 2.
# Standalone usage mirrors k1_gemv:
#   q6k_gemv <fixture_dir> [--rows N] [--cols N] [--iters N]
#            [--warmup N] [--out PATH]
# Defaults: rows 6656, cols 19968 (ffn_down), fixture files
# weights_q6k.bin / input_f32.bin in the fixture dir.

from std.time import perf_counter_ns
from std.sys import argv, exit
from max.gpu import thread_idx, block_idx, barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.primitives.warp import shuffle_xor
from std.memory import stack_allocation, bitcast

from diag import log_debug
from kernels.dequant import q6k_element_list, note_loaded

comptime THREADS = 256
comptime GROUP_THREADS = 16
comptime GROUPS = THREADS // GROUP_THREADS
comptime SB_ELEMS = 256
comptime SB_BYTES = 210


def refuse(what: String, detail: String) -> None:
    print("REFUSAL q6k_gemv_mojo: ", what, ": ", detail, sep="")
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


def unpack4_f32(word: UInt32) -> SIMD[DType.float32, 4]:
    # WGSL unpack4xU8: the word's 4 bytes, low byte first, as
    # f32. Identical to k1_gemv.mojo's helper.
    var wv = SIMD[DType.uint32, 4](word)
    var sh = SIMD[DType.uint32, 4](0, 8, 16, 24)
    return ((wv >> sh) & UInt32(0xFF)).cast[DType.float32]()


def q6k_stream_quads(
    ql0: UInt32, ql32: UInt32, qhw: UInt32
) -> Tuple[
    SIMD[DType.float32, 4],
    SIMD[DType.float32, 4],
    SIMD[DType.float32, 4],
    SIMD[DType.float32, 4],
]:
    # The four streams' centered 6-bit quant quads (q - 32 as
    # f32) for one thread's byte column of a Q6_K half-block:
    # stream 0 pairs ql0's low nibbles with qh bits 0..1,
    # stream 1 pairs ql32's low nibbles with qh bits 2..3,
    # stream 2 pairs ql0's high nibbles with qh bits 4..5,
    # stream 3 pairs ql32's high nibbles with qh bits 6..7.
    # The masks are ggml's, exactly as the wgpu K2 shader
    # carries them; the per-element route in
    # kernels/dequant.q6k_element_ptr assembles the same q one
    # element at a time (the VM helper test proves the two
    # routes agree on every element of synthetic blocks).
    var c32 = SIMD[DType.float32, 4](32.0)
    var q0 = unpack4_f32((ql0 & 0x0F0F0F0F) | ((qhw & 0x03030303) << 4)) - c32
    var q1 = unpack4_f32((ql32 & 0x0F0F0F0F) | ((qhw & 0x0C0C0C0C) << 2)) - c32
    var q2 = unpack4_f32(((ql0 >> 4) & 0x0F0F0F0F) | (qhw & 0x30303030)) - c32
    var q3 = (
        unpack4_f32(((ql32 >> 4) & 0x0F0F0F0F) | ((qhw & 0xC0C0C0C0) >> 2))
        - c32
    )
    return (q0, q1, q2, q3)


def load_word(w16: Pointer[UInt16, MutAnyOrigin], byte_off: Int) -> UInt32:
    # One little-endian u32 rebuilt from its two aligned u16
    # halves. Every byte offset the Q6_K kernel builds a word
    # at is even (the block base is a multiple of 210's factor
    # 2 and every in-block word offset is a multiple of 4), so
    # both halves are aligned u16 loads even when the block
    # itself starts two bytes into a u32.
    var idx = byte_off // 2
    return UInt32(w16.unsafe_load[width=1](idx)) | (
        UInt32(w16.unsafe_load[width=1](idx + 1)) << 16
    )


def signed_scale(w: Pointer[UInt8, MutAnyOrigin], off: Int) -> Float32:
    # One signed sub-block scale byte (the block's 16 scales
    # are int8), as f32.
    var b = Int(w[unsafe_offset=off])
    return Float32(b - 256 if b >= 128 else b)


def q6k_gemv_kernel(
    w: Pointer[UInt8, MutAnyOrigin],
    x: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    cols: Int32,
):
    # The tuned shape (see the file header): one workgroup
    # covers one row; inside it, GROUPS groups of
    # GROUP_THREADS threads each own one super-block at a
    # time. Thread t of a group owns 16 elements of that
    # block: the four streams' quads at byte column 4*(t % 8)
    # of its half h = t // 8, exactly the wgpu K2 assignment.
    var row = Int(block_idx.x)
    var lane = Int(thread_idx.x)
    var group = lane // GROUP_THREADS
    var t = lane % GROUP_THREADS
    var half = t // 8
    var v_in = t % 8
    # This thread's byte offsets within a super-block: its
    # ql word (streams 0 and 2; the +32 word serves streams
    # 1 and 3), its qh word, and its first activation quad
    # within the block's 64-quad span (streams sit 8 quads
    # apart). Scale index base: stream s of this half reads
    # scale 8h + is + 2s, where is = v_in // 4 splits the
    # 32-element streams at element 16.
    var ql0_off = 64 * half + 4 * v_in
    var qh_off = 128 + 32 * half + 4 * v_in
    var yv = 32 * half + v_in
    var s_off = 8 * half + v_in // 4
    var blocks_per_row = Int(cols) // SB_ELEMS
    var base = row * blocks_per_row * SB_BYTES
    var w16 = w.unsafe_bitcast[UInt16]()
    var shared = stack_allocation[
        GROUPS, Float32, address_space=AddressSpace.SHARED
    ]()
    var acc = Float32(0.0)
    for blk in range(group, blocks_per_row, GROUPS):
        # Activation quads are per super-block: xa pairs with
        # stream 0, xb with stream 1 (32 elements later), xc
        # with stream 2 (64 later), xd with stream 3 (96
        # later), exactly ggml's y offsets.
        var xb = blk * SB_ELEMS + yv * 4
        var xa = x.unsafe_load[width=4](xb)
        var xbq = x.unsafe_load[width=4](xb + 32)
        var xc = x.unsafe_load[width=4](xb + 64)
        var xd = x.unsafe_load[width=4](xb + 96)
        var sb_off = base + blk * SB_BYTES
        var quads = q6k_stream_quads(
            load_word(w16, sb_off + ql0_off),
            load_word(w16, sb_off + ql0_off + 32),
            load_word(w16, sb_off + qh_off),
        )
        # The super-scale as a single u16 (never a word: the
        # tensor's last block ends at byte 210 of the block).
        var d = f16_to_f32_gpu(
            UInt32(w16.unsafe_load[width=1]((sb_off + 208) // 2))
        )
        var sc_base = sb_off + 192 + s_off
        acc += d * (
            signed_scale(w, sc_base) * (quads[0] * xa).reduce_add()
            + signed_scale(w, sc_base + 2) * (quads[1] * xbq).reduce_add()
            + signed_scale(w, sc_base + 4) * (quads[2] * xc).reduce_add()
            + signed_scale(w, sc_base + 6) * (quads[3] * xd).reduce_add()
        )
    # Segmented butterfly inside each 16-thread group (the xor
    # masks never cross the aligned sixteen), then one
    # shared-memory finish across groups: K1's hybrid
    # reduction, the wgpu K2 shipped shape's own finish.
    acc += shuffle_xor(acc, UInt32(1))
    acc += shuffle_xor(acc, UInt32(2))
    acc += shuffle_xor(acc, UInt32(4))
    acc += shuffle_xor(acc, UInt32(8))
    if t == 0:
        shared[unsafe_offset=group] = acc
    barrier()
    if lane == 0:
        var s = Float32(0.0)
        for g in range(GROUPS):
            s += shared[unsafe_offset=g]
        y[unsafe_offset=row] = s


def q6k_gemv_cpu(
    w: List[UInt8], x: List[Float32], rows: Int, cols: Int
) raises -> List[Float32]:
    """The Q6_K math on the host: dequant + dot per row.

    Identical math to q6k_gemv_kernel, sequential accumulation.
    Raises a named error on a bad shape.
    """
    if cols % SB_ELEMS != 0:
        raise Error(
            "q6k_gemv_cpu_bad_shape: cols must be a multiple of 256"
            " (Q6_K super-block), got "
            + String(cols)
        )
    var row_bytes = (cols // SB_ELEMS) * SB_BYTES
    if len(w) != rows * row_bytes:
        raise Error(
            "q6k_gemv_cpu_bad_shape: weights have "
            + String(len(w))
            + " bytes, expected "
            + String(rows * row_bytes)
        )
    if len(x) != cols:
        raise Error(
            "q6k_gemv_cpu_bad_shape: input has "
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
            acc += q6k_element_list(w, sb_off, col & 255) * x[col]
        y[row] = acc
    _ = note_loaded("q6k_gemv.cpu", rows, cols)
    return y^


def read_file_exact(path: String, expected: Int) raises -> List[UInt8]:
    with open(path, "r") as f:
        var data = f.read_bytes(expected + 1)
        if len(data) != expected:
            print(
                "REFUSAL q6k_gemv_mojo: fixture size mismatch: ",
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
                "q6k_gemv <fixture_dir> [--rows N] [--cols N] [--iters N]"
                " [--warmup N] [--out PATH]"
            ),
        )
    var fixture_dir = args[1]
    var rows = Int(arg_value(args, "--rows", "6656"))
    var cols = Int(arg_value(args, "--cols", "19968"))
    var iters = Int(arg_value(args, "--iters", "0"))
    var warmup = Int(arg_value(args, "--warmup", "3"))
    var out_path = arg_value(args, "--out", fixture_dir + "/mojo_out_f32.bin")
    if cols % SB_ELEMS != 0:
        refuse(
            "bad shape",
            "cols must be a multiple of 256 (Q6_K super-block), got "
            + String(cols),
        )
    var row_bytes = (cols // SB_ELEMS) * SB_BYTES
    var weights_bytes = rows * row_bytes
    _ = log_debug("q6k_gemv.load", "fixture dir " + fixture_dir)
    var w_host = read_file_exact(
        fixture_dir + "/weights_q6k.bin", weights_bytes
    )
    var x_host = read_file_exact(fixture_dir + "/input_f32.bin", cols * 4)
    _ = log_debug(
        "q6k_gemv.load",
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
    _ = log_debug("q6k_gemv.device", "fixture uploaded")

    ctx.enqueue_function[q6k_gemv_kernel](
        w_dev, x_dev, o_dev, Int32(cols), grid_dim=rows, block_dim=THREADS
    )
    ctx.synchronize()
    _ = log_debug("q6k_gemv.run", "correctness launch done")

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
            ctx.enqueue_function[q6k_gemv_kernel](
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
            ctx.enqueue_function[q6k_gemv_kernel](
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
