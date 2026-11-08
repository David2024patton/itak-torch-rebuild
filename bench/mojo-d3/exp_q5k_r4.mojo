# exp_q5k_r4.mojo - EXPERIMENT VARIANT of src/kernels/q5k_gemv.mojo
# with ROWS_PER_WG = 4 (bench/mojo-d3, Wave D lane D3 ladder).
# Not part of the engine tree; built standalone against the
# tuned tree's src. Kept or dropped by its window-2 numbers,
# recorded in bench/mojo-d3-gemv-ladder-2026-11-07.md.

# q5k_gemv.mojo - the Q5_K GEMV kernel (K3 family) for the
# Mojo/MAX engine. Lane A3 (straight port), tuned in lane
# mojo-d3-gemv (Wave D, 2026-11-07).
#
# Real Glimmer shape served (census 2026-10-29, see
# crates/muse-glimmer/src/weights.rs: "output.weight: the
# untied output head (Q5_K in the real file)", and
# crates/wgpu-compute/examples/k3_check.rs):
#   output.weight  [6656 in, 202048 out] Q5_K
# The math mirrors crates/gguf/src/dequant.rs dequant_q5_k term
# for term via kernels/dequant.mojo (176-byte super-block:
# d, dmin, 12 scale bytes, qh[16..48], ql[48..176]). The CPU
# twin below is byte-behaved and unchanged by the tuning.
#
# TUNED SHAPE (Wave D): the shipped wgpu K3 structure
# (crates/wgpu-compute/src/kernels/k3_q5k_gemv.rs, its
# shipped_shape r2sg - K1's own tuned shape, confirmed best
# for the head's 26-blocks-per-row geometry by K3's bench
# matrix): 2 rows per 128-thread workgroup as 8 groups of
# 16, one 16-thread group per Q5_K super-block, packed scale
# extraction from the header words loaded as one vec4,
# activation quads loaded once per super-block and shared by
# both rows, per-sub-block dots in the
# d*sc*dot(q,x) - dmin*m*sum(x) form, and a segmented
# shuffle_xor butterfly plus a short shared-memory finish.
# The one addition over K1 is the fifth-bit assembly: one qh
# word per thread (its four bytes are the high bits of its
# four positions in every sub-block) and, per sub-block, the
# bit plane (qh >> s) & 0x01010101 unpacked to the vec4 of
# bits the quant quads need. A Q5_K block is 176 bytes, a
# whole number of words, so unlike Q6_K every load is a
# direct aligned u32 load, exactly as in K1.
#
# Every bad input shape gets a named REFUSAL line and exit 2.
# Standalone usage mirrors k1_gemv:
#   q5k_gemv <fixture_dir> [--rows N] [--cols N] [--iters N]
#            [--warmup N] [--out PATH]
# Defaults: rows 202048, cols 6656 (output.weight), fixture
# files weights_q5k.bin / input_f32.bin in the fixture dir.

from std.time import perf_counter_ns
from std.sys import argv, exit
from max.gpu import thread_idx, block_idx, barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.primitives.warp import shuffle_xor
from std.memory import stack_allocation, bitcast

from diag import log_debug
from kernels.dequant import q5k_element_list, note_loaded

comptime THREADS = 128
comptime ROWS_PER_WG = 4
comptime GROUP_THREADS = 16
comptime GROUPS = THREADS // GROUP_THREADS
comptime SB_ELEMS = 256
comptime SB_BYTES = 176
comptime SB_WORDS = SB_BYTES // 4


def refuse(what: String, detail: String) -> None:
    print("REFUSAL q5k_gemv_mojo: ", what, ": ", detail, sep="")
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


def q5k_packed_scales(
    w1: UInt32, w2: UInt32, w3: UInt32, half: Int
) -> Tuple[SIMD[DType.float32, 4], SIMD[DType.float32, 4]]:
    # The packed scale extraction, identical to K1's (the 12
    # scale bytes are packed the same way in Q4_K and Q5_K):
    # header words 1..3 yield, for block half `half`, the
    # (scale, scale, min, min) quads of sub-blocks 2h, 2h+1
    # and 2h+4, 2h+5, equal to dequant.scale_min_k4 on those
    # indices (the VM helper test proves the equality).
    var s16 = (w1 >> UInt32(16 * half)) & 0xFFFF
    var m16 = (w2 >> UInt32(16 * half)) & 0xFFFF
    var h16 = (w3 >> UInt32(16 * half)) & 0xFFFF
    var lo = (m16 << 16) | s16
    var sc_lo = unpack4_f32(lo & 0x3F3F3F3F)
    var hi = (((h16 << 12) | h16) & 0x0F0F0F0F) | ((lo & 0xC0C0C0C0) >> 2)
    var sc_hi = unpack4_f32(hi)
    return (sc_lo, sc_hi)


def q5k_qh_bit_quads(
    qhw: UInt32, half: Int
) -> Tuple[
    SIMD[DType.float32, 4],
    SIMD[DType.float32, 4],
    SIMD[DType.float32, 4],
    SIMD[DType.float32, 4],
]:
    # The fifth-bit quads for one thread's four positions:
    # bit s of byte j of the qh word is the fifth bit of the
    # thread's j-th element in sub-block s. Returns the bit
    # quads (each element 0.0 or 1.0) for the four sub-blocks
    # this thread serves, 2h, 2h+1, 2h+4, 2h+5, in that
    # order; the quant quads add 16 * bits to their nibbles.
    # The shifts and mask are the wgpu K3 shader's.
    var bits_a = unpack4_f32((qhw >> UInt32(2 * half)) & 0x01010101)
    var bits_b = unpack4_f32((qhw >> UInt32(2 * half + 1)) & 0x01010101)
    var bits_c = unpack4_f32((qhw >> UInt32(2 * half + 4)) & 0x01010101)
    var bits_d = unpack4_f32((qhw >> UInt32(2 * half + 5)) & 0x01010101)
    return (bits_a, bits_b, bits_c, bits_d)


def q5k_block_dot(
    w32: Pointer[UInt32, MutAnyOrigin],
    bw: Int,
    half: Int,
    qh_off: Int,
    w0_off: Int,
    xa: SIMD[DType.float32, 4],
    xbq: SIMD[DType.float32, 4],
    xc: SIMD[DType.float32, 4],
    xd: SIMD[DType.float32, 4],
    sa: Float32,
    sb: Float32,
    sc_: Float32,
    sd: Float32,
) -> Float32:
    # One row's contribution from one super-block (the tuned
    # kernel's per-row body, factored for the rows=4 tile).
    var hdr = w32.unsafe_load[width=4](bw)
    var d = f16_to_f32_gpu(hdr[0] & 0xFFFF)
    var dmin = f16_to_f32_gpu(hdr[0] >> 16)
    var scales = q5k_packed_scales(hdr[1], hdr[2], hdr[3], half)
    var bits = q5k_qh_bit_quads(w32.unsafe_load[width=1](bw + qh_off), half)
    var w0 = w32.unsafe_load[width=1](bw + w0_off)
    var w1 = w32.unsafe_load[width=1](bw + w0_off + 16)
    var qa = unpack4_f32(w0 & 0x0F0F0F0F) + Float32(16.0) * bits[0]
    var qb = unpack4_f32((w0 >> 4) & 0x0F0F0F0F) + Float32(16.0) * bits[1]
    var qc = unpack4_f32(w1 & 0x0F0F0F0F) + Float32(16.0) * bits[2]
    var qd = unpack4_f32((w1 >> 4) & 0x0F0F0F0F) + Float32(16.0) * bits[3]
    return d * (
        scales[0][0] * (qa * xa).reduce_add()
        + scales[0][1] * (qb * xbq).reduce_add()
        + scales[1][0] * (qc * xc).reduce_add()
        + scales[1][1] * (qd * xd).reduce_add()
    ) - dmin * (
        scales[0][2] * sa
        + scales[0][3] * sb
        + scales[1][2] * sc_
        + scales[1][3] * sd
    )


def q5k_gemv_kernel(
    w: Pointer[UInt8, MutAnyOrigin],
    x: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    cols: Int32,
):
    # EXPERIMENT (bench only): the tuned r2sg shape widened
    # to ROWS_PER_WG rows per workgroup, activation quads
    # loaded once per super-block and shared by all rows.
    var lane = Int(thread_idx.x)
    var group = lane // GROUP_THREADS
    var t = lane % GROUP_THREADS
    var half = t // 8
    var v = t % 8
    var w0_off = 12 + 8 * half + v
    var qh_off = 4 + v
    var yv = 16 * half + v
    var blocks_per_row = Int(cols) // SB_ELEMS
    var row0 = Int(block_idx.x) * ROWS_PER_WG
    var w32 = w.unsafe_bitcast[UInt32]()
    var shared = stack_allocation[
        ROWS_PER_WG * GROUPS, Float32, address_space=AddressSpace.SHARED
    ]()
    var acc0 = Float32(0.0)
    var acc1 = Float32(0.0)
    var acc2 = Float32(0.0)
    var acc3 = Float32(0.0)
    for blk in range(group, blocks_per_row, GROUPS):
        var xb = blk * SB_ELEMS + yv * 4
        var xa = x.unsafe_load[width=4](xb)
        var xbq = x.unsafe_load[width=4](xb + 32)
        var xc = x.unsafe_load[width=4](xb + 128)
        var xd = x.unsafe_load[width=4](xb + 160)
        var sa = xa.reduce_add()
        var sb = xbq.reduce_add()
        var sc_ = xc.reduce_add()
        var sd = xd.reduce_add()
        acc0 += q5k_block_dot(
            w32,
            (row0 * blocks_per_row + blk) * SB_WORDS,
            half,
            qh_off,
            w0_off,
            xa,
            xbq,
            xc,
            xd,
            sa,
            sb,
            sc_,
            sd,
        )
        acc1 += q5k_block_dot(
            w32,
            ((row0 + 1) * blocks_per_row + blk) * SB_WORDS,
            half,
            qh_off,
            w0_off,
            xa,
            xbq,
            xc,
            xd,
            sa,
            sb,
            sc_,
            sd,
        )
        acc2 += q5k_block_dot(
            w32,
            ((row0 + 2) * blocks_per_row + blk) * SB_WORDS,
            half,
            qh_off,
            w0_off,
            xa,
            xbq,
            xc,
            xd,
            sa,
            sb,
            sc_,
            sd,
        )
        acc3 += q5k_block_dot(
            w32,
            ((row0 + 3) * blocks_per_row + blk) * SB_WORDS,
            half,
            qh_off,
            w0_off,
            xa,
            xbq,
            xc,
            xd,
            sa,
            sb,
            sc_,
            sd,
        )
    acc0 += shuffle_xor(acc0, UInt32(1))
    acc0 += shuffle_xor(acc0, UInt32(2))
    acc0 += shuffle_xor(acc0, UInt32(4))
    acc0 += shuffle_xor(acc0, UInt32(8))
    acc1 += shuffle_xor(acc1, UInt32(1))
    acc1 += shuffle_xor(acc1, UInt32(2))
    acc1 += shuffle_xor(acc1, UInt32(4))
    acc1 += shuffle_xor(acc1, UInt32(8))
    acc2 += shuffle_xor(acc2, UInt32(1))
    acc2 += shuffle_xor(acc2, UInt32(2))
    acc2 += shuffle_xor(acc2, UInt32(4))
    acc2 += shuffle_xor(acc2, UInt32(8))
    acc3 += shuffle_xor(acc3, UInt32(1))
    acc3 += shuffle_xor(acc3, UInt32(2))
    acc3 += shuffle_xor(acc3, UInt32(4))
    acc3 += shuffle_xor(acc3, UInt32(8))
    if t == 0:
        shared[unsafe_offset=group] = acc0
        shared[unsafe_offset=GROUPS + group] = acc1
        shared[unsafe_offset=2 * GROUPS + group] = acc2
        shared[unsafe_offset=3 * GROUPS + group] = acc3
    barrier()
    if lane == 0:
        var s0 = Float32(0.0)
        var s1 = Float32(0.0)
        var s2 = Float32(0.0)
        var s3 = Float32(0.0)
        for g in range(GROUPS):
            s0 += shared[unsafe_offset=g]
            s1 += shared[unsafe_offset=GROUPS + g]
            s2 += shared[unsafe_offset=2 * GROUPS + g]
            s3 += shared[unsafe_offset=3 * GROUPS + g]
        y[unsafe_offset=row0] = s0
        y[unsafe_offset=row0 + 1] = s1
        y[unsafe_offset=row0 + 2] = s2
        y[unsafe_offset=row0 + 3] = s3


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
    if rows % ROWS_PER_WG != 0:
        refuse(
            "bad shape",
            "rows must be a multiple of the 4-row workgroup tile, got "
            + String(rows),
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
        w_dev,
        x_dev,
        o_dev,
        Int32(cols),
        grid_dim=rows // ROWS_PER_WG,
        block_dim=THREADS,
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
                grid_dim=rows // ROWS_PER_WG,
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
                grid_dim=rows // ROWS_PER_WG,
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
