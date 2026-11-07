# k1_gemv.mojo - the K1 Q4_K GEMV kernel, founding kernel citizen
# of the Mojo/MAX engine (mojo-engine/src/kernels/).
#
# Carried over from the kernel test lane's bench port
# (bench/mojo/k1_gemv.mojo, bench/mojo-kernel-test-2026-11-07.md)
# and adapted to this engine's seams: debug lines flow through
# src/diag.mojo (MOJO_ENGINE_DEBUG), not an ad hoc per-kernel
# flag. TUNED in lane mojo-a2-k1 (2026-11-07): the straight port
# recorded 39.8 GB/s median against the tuned wgpu K1's 838.4
# (see DESIGN.md, the speed law); the GPU kernel below now
# mirrors the shipped wgpu r2sg shape (see
# crates/wgpu-compute/src/kernels/k1_q4k_gemv.rs, llama family):
# 2 rows per workgroup, 128 threads as 8 groups of 16, one
# 16-thread group per Q4_K super-block, packed scale extraction
# from the 3 header words, activation quads loaded once per
# super-block and shared by both rows, per-sub-block dot
# products in the d*sc*dot(q,x) - dmin*m*sum(x) form, and a
# segmented shuffle butterfly (xor 1/2/4/8, never crossing the
# aligned 16) plus a short shared-memory finish. The ladder of
# measured iterations is recorded in the lane's bench note.
#
# The math is the same Q4_K GEMV as crates/wgpu-compute's K1
# kernel (ggml Q4_K super-block dequant plus a dot product per
# output row). k1_gemv_cpu below runs the identical math on the
# host for fixture tests on machines without a GPU.
#
# Every bad input shape gets a named REFUSAL line and exit 2.
#
# Standalone usage (bench/fixture entry, unchanged from the
# kernel test):
#   k1_gemv <fixture_dir> [--rows N] [--cols N] [--iters N]
#           [--warmup N] [--out PATH]
# Defaults: rows 19968, cols 6656 (the fixture tensor's shape),
# iters 0 (correctness only), warmup 3, out
# <fixture_dir>/mojo_out_f32.bin. With --iters N > 0 the process
# additionally runs a timed loop of N launches after warmup and
# prints a MOJO_TIMING line whose GB/s counts the same streamed
# weight bytes the wgpu k1_bench counts.

from std.time import perf_counter_ns
from std.sys import argv, exit
from max.gpu import thread_idx, block_idx, block_dim, barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.primitives.warp import shuffle_xor
from std.memory import stack_allocation, bitcast

from diag import log_debug

comptime THREADS = 128
comptime ROWS_PER_WG = 2
comptime GROUP_THREADS = 16
comptime GROUPS = THREADS // GROUP_THREADS
comptime SB_ELEMS = 256
comptime SB_BYTES = 144
comptime SB_WORDS = SB_BYTES // 4


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


def f16_to_f32_gpu(bits: UInt32) -> Float32:
    # Bit-trick f16 -> f32 for the GPU path, the same construction
    # the shipped wgpu K1 uses: normals rebias the exponent into
    # f32 bits directly, subnormals go through frac * 2^-24 (exact),
    # inf/NaN keep their class so a bad weight still fails the
    # comparison instead of silently changing the math.
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
    # WGSL unpack4xU8: the word's 4 bytes, low byte first, as f32.
    var wv = SIMD[DType.uint32, 4](word)
    var sh = SIMD[DType.uint32, 4](0, 8, 16, 24)
    return ((wv >> sh) & UInt32(0xFF)).cast[DType.float32]()


def unpack_q_f32(
    word: UInt32,
) -> Tuple[SIMD[DType.float32, 4], SIMD[DType.float32, 4]]:
    # One u32 of Q4_K nibbles: (low nibbles, high nibbles), each
    # byte of the word giving one element of each quad, in the
    # ggml order the wgpu kernel's unpack4xU8 form produces.
    var wv = SIMD[DType.uint32, 4](word)
    var sh = SIMD[DType.uint32, 4](0, 8, 16, 24)
    var bytes_ = (wv >> sh) & UInt32(0xFF)
    var lo = (bytes_ & UInt32(0xF)).cast[DType.float32]()
    var hi = ((bytes_ >> 4) & UInt32(0xF)).cast[DType.float32]()
    return (lo, hi)


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
    # The r2sg shape (see the file header): one workgroup covers
    # ROWS_PER_WG rows; inside it, GROUPS groups of GROUP_THREADS
    # threads each own one super-block at a time. Thread t of a
    # group owns 16 elements of that block: 4-element quads from
    # sub-blocks 2h, 2h+1, 2h+4, 2h+5, where h = t // 8 is the
    # block half and the quad's word sits at qs word
    # 4 + 8h + (t % 8), exactly the wgpu kernel's assignment.
    var lane = Int(thread_idx.x)
    var group = lane // GROUP_THREADS
    var t = lane % GROUP_THREADS
    var half = t // 8
    var w0_off = 4 + 8 * half + (t % 8)
    var yv = 16 * half + (t % 8)
    var blocks_per_row = Int(cols) // SB_ELEMS
    var row0 = Int(block_idx.x) * ROWS_PER_WG
    var w32 = w.unsafe_bitcast[UInt32]()
    var shared = stack_allocation[
        ROWS_PER_WG * GROUPS, Float32, address_space=AddressSpace.SHARED
    ]()
    var acc0 = Float32(0.0)
    var acc1 = Float32(0.0)
    for blk in range(group, blocks_per_row, GROUPS):
        # Activation quads and their sums are per super-block and
        # shared by both rows of the tile: loaded once, used twice.
        var xb = blk * SB_ELEMS + yv * 4
        var xa = x.unsafe_load[width=4](xb)
        var xbq = x.unsafe_load[width=4](xb + 32)
        var xc = x.unsafe_load[width=4](xb + 128)
        var xd = x.unsafe_load[width=4](xb + 160)
        var sa = xa.reduce_add()
        var sb = xbq.reduce_add()
        var sc_ = xc.reduce_add()
        var sd = xd.reduce_add()
        # Row 0 of the tile.
        var bw0 = (row0 * blocks_per_row + blk) * SB_WORDS
        var hdr0 = w32.unsafe_load[width=4](bw0)
        var hw0 = hdr0[0]
        var d0 = f16_to_f32_gpu(hw0 & 0xFFFF)
        var dmin0 = f16_to_f32_gpu(hw0 >> 16)
        var s16_0 = (hdr0[1] >> UInt32(16 * half)) & 0xFFFF
        var m16_0 = (hdr0[2] >> UInt32(16 * half)) & 0xFFFF
        var h16_0 = (hdr0[3] >> UInt32(16 * half)) & 0xFFFF
        var lo0 = (m16_0 << 16) | s16_0
        var sc_lo0 = unpack4_f32(lo0 & 0x3F3F3F3F)
        var hi0 = (((h16_0 << 12) | h16_0) & 0x0F0F0F0F) | (
            (lo0 & 0xC0C0C0C0) >> 2
        )
        var sc_hi0 = unpack4_f32(hi0)
        var q0 = unpack_q_f32(w32.unsafe_load[width=1](bw0 + w0_off))
        var q1 = unpack_q_f32(w32.unsafe_load[width=1](bw0 + w0_off + 16))
        acc0 += d0 * (
            sc_lo0[0] * (q0[0] * xa).reduce_add()
            + sc_lo0[1] * (q0[1] * xbq).reduce_add()
            + sc_hi0[0] * (q1[0] * xc).reduce_add()
            + sc_hi0[1] * (q1[1] * xd).reduce_add()
        ) - dmin0 * (
            sc_lo0[2] * sa + sc_lo0[3] * sb + sc_hi0[2] * sc_ + sc_hi0[3] * sd
        )
        # Row 1 of the tile: the next row of the same tensor,
        # reusing the activation quads above. The host refuses an
        # odd row count by name (see main), so row0 + 1 is always
        # a real row here.
        var bw1 = ((row0 + 1) * blocks_per_row + blk) * SB_WORDS
        var hdr1 = w32.unsafe_load[width=4](bw1)
        var hw1 = hdr1[0]
        var d1 = f16_to_f32_gpu(hw1 & 0xFFFF)
        var dmin1 = f16_to_f32_gpu(hw1 >> 16)
        var s16_1 = (hdr1[1] >> UInt32(16 * half)) & 0xFFFF
        var m16_1 = (hdr1[2] >> UInt32(16 * half)) & 0xFFFF
        var h16_1 = (hdr1[3] >> UInt32(16 * half)) & 0xFFFF
        var lo1 = (m16_1 << 16) | s16_1
        var sc_lo1 = unpack4_f32(lo1 & 0x3F3F3F3F)
        var hi1 = (((h16_1 << 12) | h16_1) & 0x0F0F0F0F) | (
            (lo1 & 0xC0C0C0C0) >> 2
        )
        var sc_hi1 = unpack4_f32(hi1)
        var q2 = unpack_q_f32(w32.unsafe_load[width=1](bw1 + w0_off))
        var q3 = unpack_q_f32(w32.unsafe_load[width=1](bw1 + w0_off + 16))
        acc1 += d1 * (
            sc_lo1[0] * (q2[0] * xa).reduce_add()
            + sc_lo1[1] * (q2[1] * xbq).reduce_add()
            + sc_hi1[0] * (q3[0] * xc).reduce_add()
            + sc_hi1[1] * (q3[1] * xd).reduce_add()
        ) - dmin1 * (
            sc_lo1[2] * sa + sc_lo1[3] * sb + sc_hi1[2] * sc_ + sc_hi1[3] * sd
        )
    # Segmented butterfly inside each 16-thread group: the xor
    # masks stay inside the aligned sixteen, so each group sums
    # its own partials in registers, whatever the hardware warp
    # width is. Then one shared-memory finish across groups.
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


def scale_min_k4_list(
    j: Int, w: List[UInt8], sb_off: Int
) -> Tuple[Float32, Float32]:
    # scale_min_k4 over a host List instead of a device pointer;
    # identical byte math (see gguf::dequant::scale_min_k4).
    if j < 4:
        var sc = Int(w[sb_off + 4 + j]) & 63
        var m = Int(w[sb_off + 8 + j]) & 63
        return (Float32(sc), Float32(m))
    else:
        var hi = Int(w[sb_off + 4 + (j + 4)])
        var sc = (hi & 0xF) | ((Int(w[sb_off + 4 + (j - 4)]) >> 6) << 4)
        var m = (hi >> 4) | ((Int(w[sb_off + 4 + j]) >> 6) << 4)
        return (Float32(sc), Float32(m))


def k1_gemv_cpu(
    w: List[UInt8], x: List[Float32], rows: Int, cols: Int
) raises -> List[Float32]:
    """The K1 math on the host: Q4_K dequant + dot per row.

    Identical math to k1_gemv_kernel, sequential accumulation.
    Exists for fixture tests on GPU-less machines; the GPU kernel
    is the engine path. Raises a named error on a bad shape.
    """
    if cols % SB_ELEMS != 0:
        raise Error(
            "k1_gemv_cpu_bad_shape: cols must be a multiple of 256"
            " (Q4_K super-block), got "
            + String(cols)
        )
    var row_bytes = (cols // SB_ELEMS) * SB_BYTES
    if len(w) != rows * row_bytes:
        raise Error(
            "k1_gemv_cpu_bad_shape: weights have "
            + String(len(w))
            + " bytes, expected "
            + String(rows * row_bytes)
        )
    if len(x) != cols:
        raise Error(
            "k1_gemv_cpu_bad_shape: input has "
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
            var sb = col >> 8
            var within = col & 255
            var sb_off = base + sb * SB_BYTES
            var d_bits = UInt16(w[sb_off]) | (UInt16(w[sb_off + 1]) << 8)
            var dm_bits = UInt16(w[sb_off + 2]) | (UInt16(w[sb_off + 3]) << 8)
            var d = f16_to_f32(d_bits)
            var dmin = f16_to_f32(dm_bits)
            var g = within >> 6
            var p = within & 63
            var sc: Float32
            var m: Float32
            var q: Float32
            if p < 32:
                var sm = scale_min_k4_list(2 * g, w, sb_off)
                sc = sm[0]
                m = sm[1]
                q = Float32(Int(w[sb_off + 16 + g * 32 + p]) & 0xF)
            else:
                var sm = scale_min_k4_list(2 * g + 1, w, sb_off)
                sc = sm[0]
                m = sm[1]
                q = Float32(Int(w[sb_off + 16 + g * 32 + (p - 32)]) >> 4)
            var val = d * sc * q - dmin * m
            acc += val * x[col]
        y[row] = acc
    _ = log_debug(
        "k1_gemv.cpu", "ran " + String(rows) + " rows x " + String(cols)
    )
    return y^


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
    if rows % ROWS_PER_WG != 0:
        refuse(
            "bad shape",
            "rows must be a multiple of the 2-row workgroup tile, got "
            + String(rows),
        )
    var row_bytes = (cols // SB_ELEMS) * SB_BYTES
    var weights_bytes = rows * row_bytes
    _ = log_debug("k1_gemv.load", "fixture dir " + fixture_dir)
    var w_host = read_file_exact(
        fixture_dir + "/weights_q4k.bin", weights_bytes
    )
    var x_host = read_file_exact(fixture_dir + "/input_f32.bin", cols * 4)
    _ = log_debug(
        "k1_gemv.load",
        "weights "
        + String(weights_bytes)
        + " bytes, input "
        + String(cols)
        + " f32",
    )

    var ctx = DeviceContext()
    _ = log_debug("k1_gemv.device", "DeviceContext created")
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
    _ = log_debug("k1_gemv.device", "fixture uploaded")

    ctx.enqueue_function[k1_gemv_kernel](
        w_dev,
        x_dev,
        o_dev,
        Int32(cols),
        grid_dim=rows // ROWS_PER_WG,
        block_dim=THREADS,
    )
    ctx.synchronize()
    _ = log_debug("k1_gemv.run", "correctness launch done")

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
                grid_dim=rows // ROWS_PER_WG,
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
        _ = log_debug("k1_gemv.timing", "loop done")
