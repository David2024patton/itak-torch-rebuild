# gemm.mojo - the batched (matrix-matrix) projection kernels
# for PREFILL, Wave D Lane D1 (mojo-engine/src/kernels/).
#
# Why this module exists: through Wave C, prefill ran the GEMV
# family once per prompt row - one launch per projection per
# position, the weight stream re-read from the card for every
# row. A prefill projection is really one matrix-matrix
# product: many prompt rows against the SAME resident weight
# matrix. This module is that product, per quantized shape the
# GPU layer dispatches for layer projections (the Wave A
# census: Q4_K 365 tensors, Q6_K 52, Q5_K 1 - the head - F32
# norms-as-vectors plus any F32 matrix slot; F16 has zero
# weight tensors in Glimmer and the batch path refuses it by
# name - single rows keep the GEMV path, f16_gemv_kernel).
# Decode is untouched: it stays on the GEMV family.
#
# Layouts (the whole contract, stated once): the activation
# batch X is flat row-major [batch][cols] (one prompt row per
# batch element, contiguous), the weight matrix W is the
# slot's raw storage exactly as the GEMVs consume it
# ([rows][cols], row-major, quantized super-blocks inline),
# and the output Y is flat row-major [batch][rows], so batch
# element b's output row is Y[b*rows .. (b+1)*rows) - the same
# row the GEMV would have produced for X[b*cols ..].
# Dequant is IN-KERNEL and identical to the GEMVs': Q4_K uses
# k1_gemv.mojo's own unpack/scale helpers and the
# d*sc*dot(q,x) - dmin*m*sum(x) form; Q6_K / Q5_K use
# kernels/dequant.mojo's per-element helpers, the same ones
# q6k_gemv / q5k_gemv call; F32 is the plain product.
#
# Structures, in the order the lane evaluated them (the bench
# note bench/mojo-engine-wave-d1-gemm-2026-11-07.md records
# every iteration's numbers; correctness gate FIRST each time):
#   (1) MAX's own kernel library matmul: NO FIT, recorded in
#       the bench note. In MAX 26.7 the only quantized matmul
#       surface is the Python graph op max.graph.ops.qmatmul
#       (Q4_K / Q6_K encodings only, no Q5_K), which lives in
#       the MAX graph/pipeline world: it takes TensorValues in
#       MAX's repacked quantized layout, QUANTIZES the lhs
#       activations to the rhs encoding before the product
#       (not this engine's f32-activation semantics, and not
#       the K1 accumulation form the Wave A bar is stated
#       against), and is not callable from Mojo kernel code -
#       no Mojo-side matmul module is importable in this
#       toolchain (probed: max.kernels.matmul, max.nn,
#       kernels.matmul all fail to resolve). Adopting it would
#       mean rebuilding the forward inside MAX graphs, a
#       different engine, not a prefill path for this one.
#   (2) *_gemm_kernel: the straightforward batched form. One
#       workgroup per (batch element, output row) - the GEMV
#       workgroup body verbatim with batch offsets - launched
#       ONCE for the whole batch. The weight stream is shared
#       across batch elements through the card's caches (the
#       launches are one grid, back to back), never re-uploaded.
#   (3) *_gemm_tiled_kernel: a register-tiled form. One
#       workgroup owns ONE output row for a tile of
#       BATCH_TILE batch elements: each weight element (or
#       Q4_K super-block header + packed quads) is loaded and
#       dequantized ONCE per workgroup and applied to every
#       batch element in the tile, so the weight stream is
#       read once per tile instead of once per row. The host
#       picks (3) at or above a configured batch size and (2)
#       below it (src/config.mojo: MOJO_ENGINE_GEMM_TILED_MIN_BATCH).
#
# The CPU twins below are the parity reference: each one runs
# the corresponding GEMV CPU twin per batch row on that row's
# slice, so a CPU-twin row is bit-identical to the GEMV path's
# row by construction, and the GPU kernels are measured
# against these twins under the Wave A bars (stated per shape
# in tests/server_proof_gemm.mojo).
#
# Named refusals: gemm_bad_shape (non-positive dims, cols not
# a multiple of the shape's kernel step, Q4_K rows not a
# multiple of the K1 workgroup tile, weight/input length
# mismatches). Diag seam: phase `gemm`, via src/diag.mojo.

from max.gpu import thread_idx, block_idx, block_dim, barrier
from max.gpu.memory import AddressSpace
from max.gpu.primitives.warp import shuffle_xor
from std.memory import stack_allocation

from diag import log_debug
from kernels.dequant import (
    q6k_element_ptr,
    q5k_element_ptr,
    f16_to_f32 as dequant_f16_to_f32,
)
from kernels.k1_gemv import (
    k1_gemv_cpu,
    f16_to_f32_gpu,
    unpack4_f32,
    unpack_q_f32,
    THREADS as K1_THREADS,
    ROWS_PER_WG as K1_ROWS_PER_WG,
    GROUP_THREADS as K1_GROUP_THREADS,
    GROUPS as K1_GROUPS,
    SB_ELEMS as K1_SB_ELEMS,
    SB_WORDS as K1_SB_WORDS,
)
from kernels.q6k_gemv import q6k_gemv_cpu, THREADS as Q6_THREADS
from kernels.q5k_gemv import q5k_gemv_cpu, THREADS as Q5_THREADS
from kernels.f32_gemv import f32_gemv_cpu, THREADS as F_THREADS

comptime BATCH_TILE = 4
comptime SB_ELEMS = 256
comptime Q6_SB_BYTES = 210
comptime Q5_SB_BYTES = 176
comptime Q4_SB_BYTES = 144


# ---- pure shape helpers (VM-testable, no device) -------------


def check_gemm_dims(name: String, batch: Int, rows: Int, cols: Int) raises:
    """The dims every GEMM shape shares: all positive."""
    if batch <= 0 or rows <= 0 or cols <= 0:
        raise Error(
            "gemm_bad_shape: "
            + name
            + " got batch/rows/cols "
            + String(batch)
            + "/"
            + String(rows)
            + "/"
            + String(cols)
            + ", all must be positive"
        )


def gemm_plain_grid(rows: Int, batch: Int) raises -> Int:
    """Workgroups for a structure-(2) launch over byte/F32
    shapes: one per (output row, batch element)."""
    if rows <= 0 or batch <= 0:
        raise Error(
            "gemm_bad_shape: plain grid needs positive rows/batch, got "
            + String(rows)
            + "/"
            + String(batch)
        )
    return rows * batch


def gemm_tiled_grid(rows: Int, batch: Int) raises -> Int:
    """Workgroups for a structure-(3) launch: one per
    (output row, batch tile of BATCH_TILE)."""
    if rows <= 0 or batch <= 0:
        raise Error(
            "gemm_bad_shape: tiled grid needs positive rows/batch, got "
            + String(rows)
            + "/"
            + String(batch)
        )
    return rows * ((batch + BATCH_TILE - 1) // BATCH_TILE)


def gemm_k1_grid(rows: Int, batch: Int) raises -> Int:
    """Workgroups for a structure-(2) K1 launch: the K1
    workgroup covers K1_ROWS_PER_WG output rows of ONE batch
    element, so rows must be a multiple of the tile (the GEMV
    host rule, kept)."""
    if batch <= 0 or rows <= 0 or rows % K1_ROWS_PER_WG != 0:
        raise Error(
            "gemm_bad_shape: k1 grid needs rows a multiple of "
            + String(K1_ROWS_PER_WG)
            + " and positive batch, got rows="
            + String(rows)
            + " batch="
            + String(batch)
        )
    return (rows // K1_ROWS_PER_WG) * batch


def gemm_k1_tiled_grid(rows: Int, batch: Int) raises -> Int:
    """Workgroups for a structure-(3) K1 launch: one workgroup
    covers K1_ROWS_PER_WG output rows for a whole batch tile."""
    if batch <= 0 or rows <= 0 or rows % K1_ROWS_PER_WG != 0:
        raise Error(
            "gemm_bad_shape: k1 tiled grid needs rows a multiple of "
            + String(K1_ROWS_PER_WG)
            + " and positive batch, got rows="
            + String(rows)
            + " batch="
            + String(batch)
        )
    return (rows // K1_ROWS_PER_WG) * ((batch + BATCH_TILE - 1) // BATCH_TILE)


def _slice_f32(x: List[Float32], off: Int, n: Int) -> List[Float32]:
    var out = List[Float32](capacity=n)
    out.resize(n, Float32(0.0))
    for i in range(n):
        out[i] = x[off + i]
    return out^


def _place_f32(
    mut y: List[Float32], off: Int, row: List[Float32]
):
    for i in range(len(row)):
        y[off + i] = row[i]


# ---- CPU twins (parity reference; GEMV twin per batch row) ---


def q4k_gemm_cpu(
    w: List[UInt8], x: List[Float32], batch: Int, rows: Int, cols: Int
) raises -> List[Float32]:
    """Q4_K GEMM on the host: k1_gemv_cpu per batch row."""
    check_gemm_dims("q4k_gemm", batch, rows, cols)
    if cols % SB_ELEMS != 0:
        raise Error(
            "gemm_bad_shape: q4k cols must be a multiple of 256, got "
            + String(cols)
        )
    if len(x) != batch * cols:
        raise Error(
            "gemm_bad_shape: q4k input has "
            + String(len(x))
            + " f32, expected batch*cols "
            + String(batch * cols)
        )
    var y = List[Float32](capacity=batch * rows)
    y.resize(batch * rows, Float32(0.0))
    for b in range(batch):
        var row = k1_gemv_cpu(w, _slice_f32(x, b * cols, cols), rows, cols)
        _place_f32(y, b * rows, row)
    _ = log_debug(
        "gemm", "q4k cpu batch=" + String(batch) + " rows=" + String(rows)
    )
    return y^


def q6k_gemm_cpu(
    w: List[UInt8], x: List[Float32], batch: Int, rows: Int, cols: Int
) raises -> List[Float32]:
    """Q6_K GEMM on the host: q6k_gemv_cpu per batch row."""
    check_gemm_dims("q6k_gemm", batch, rows, cols)
    if cols % SB_ELEMS != 0:
        raise Error(
            "gemm_bad_shape: q6k cols must be a multiple of 256, got "
            + String(cols)
        )
    if len(x) != batch * cols:
        raise Error(
            "gemm_bad_shape: q6k input has "
            + String(len(x))
            + " f32, expected batch*cols "
            + String(batch * cols)
        )
    var y = List[Float32](capacity=batch * rows)
    y.resize(batch * rows, Float32(0.0))
    for b in range(batch):
        var row = q6k_gemv_cpu(w, _slice_f32(x, b * cols, cols), rows, cols)
        _place_f32(y, b * rows, row)
    _ = log_debug(
        "gemm", "q6k cpu batch=" + String(batch) + " rows=" + String(rows)
    )
    return y^


def q5k_gemm_cpu(
    w: List[UInt8], x: List[Float32], batch: Int, rows: Int, cols: Int
) raises -> List[Float32]:
    """Q5_K GEMM on the host: q5k_gemv_cpu per batch row."""
    check_gemm_dims("q5k_gemm", batch, rows, cols)
    if cols % SB_ELEMS != 0:
        raise Error(
            "gemm_bad_shape: q5k cols must be a multiple of 256, got "
            + String(cols)
        )
    if len(x) != batch * cols:
        raise Error(
            "gemm_bad_shape: q5k input has "
            + String(len(x))
            + " f32, expected batch*cols "
            + String(batch * cols)
        )
    var y = List[Float32](capacity=batch * rows)
    y.resize(batch * rows, Float32(0.0))
    for b in range(batch):
        var row = q5k_gemv_cpu(w, _slice_f32(x, b * cols, cols), rows, cols)
        _place_f32(y, b * rows, row)
    _ = log_debug(
        "gemm", "q5k cpu batch=" + String(batch) + " rows=" + String(rows)
    )
    return y^


def f32_gemm_cpu(
    w: List[Float32], x: List[Float32], batch: Int, rows: Int, cols: Int
) raises -> List[Float32]:
    """F32 GEMM on the host: f32_gemv_cpu per batch row."""
    check_gemm_dims("f32_gemm", batch, rows, cols)
    if len(w) != rows * cols:
        raise Error(
            "gemm_bad_shape: f32 weights have "
            + String(len(w))
            + " f32, expected rows*cols "
            + String(rows * cols)
        )
    if len(x) != batch * cols:
        raise Error(
            "gemm_bad_shape: f32 input has "
            + String(len(x))
            + " f32, expected batch*cols "
            + String(batch * cols)
        )
    var y = List[Float32](capacity=batch * rows)
    y.resize(batch * rows, Float32(0.0))
    for b in range(batch):
        var row = f32_gemv_cpu(w, _slice_f32(x, b * cols, cols), rows, cols)
        _place_f32(y, b * rows, row)
    _ = log_debug(
        "gemm", "f32 cpu batch=" + String(batch) + " rows=" + String(rows)
    )
    return y^


# ---- structure (2): batched GEMV workgroups, one launch ------
#
# Grid is a flat gemm_plain_grid / gemm_k1_grid count; the
# block index decomposes as idx = row * batch + b for the
# 256-thread shapes (same-row workgroups adjacent, so the
# weight stream shared across the batch sits in adjacent
# launches for the cache), and idx = wg * batch + b for K1.


def q6k_gemm_kernel(
    w: Pointer[UInt8, MutAnyOrigin],
    x: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    cols: Int32,
    batch: Int32,
    rows: Int32,
):
    var idx = Int(block_idx.x)
    var row = idx // Int(batch)
    var b = idx % Int(batch)
    var tid = Int(thread_idx.x)
    var row_bytes = (Int(cols) // SB_ELEMS) * Q6_SB_BYTES
    var base = row * row_bytes
    var xbase = b * Int(cols)
    var shared = stack_allocation[
        Q6_THREADS, Float32, address_space=AddressSpace.SHARED
    ]()
    var acc = Float32(0.0)
    var steps = Int(cols) // Q6_THREADS
    for k in range(steps):
        var col = tid + k * Q6_THREADS
        var sb_off = base + (col >> 8) * Q6_SB_BYTES
        var val = q6k_element_ptr(w, sb_off, col & 255)
        acc += val * x[unsafe_offset=xbase + col]
    shared[unsafe_offset=tid] = acc
    barrier()
    var stride = Q6_THREADS // 2
    while stride > 0:
        if tid < stride:
            shared[unsafe_offset=tid] = (
                shared[unsafe_offset=tid] + shared[unsafe_offset=tid + stride]
            )
        barrier()
        stride //= 2
    if tid == 0:
        y[unsafe_offset=b * Int(rows) + row] = shared[unsafe_offset=0]


def q5k_gemm_kernel(
    w: Pointer[UInt8, MutAnyOrigin],
    x: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    cols: Int32,
    batch: Int32,
    rows: Int32,
):
    var idx = Int(block_idx.x)
    var row = idx // Int(batch)
    var b = idx % Int(batch)
    var tid = Int(thread_idx.x)
    var row_bytes = (Int(cols) // SB_ELEMS) * Q5_SB_BYTES
    var base = row * row_bytes
    var xbase = b * Int(cols)
    var shared = stack_allocation[
        Q5_THREADS, Float32, address_space=AddressSpace.SHARED
    ]()
    var acc = Float32(0.0)
    var steps = Int(cols) // Q5_THREADS
    for k in range(steps):
        var col = tid + k * Q5_THREADS
        var sb_off = base + (col >> 8) * Q5_SB_BYTES
        var val = q5k_element_ptr(w, sb_off, col & 255)
        acc += val * x[unsafe_offset=xbase + col]
    shared[unsafe_offset=tid] = acc
    barrier()
    var stride = Q5_THREADS // 2
    while stride > 0:
        if tid < stride:
            shared[unsafe_offset=tid] = (
                shared[unsafe_offset=tid] + shared[unsafe_offset=tid + stride]
            )
        barrier()
        stride //= 2
    if tid == 0:
        y[unsafe_offset=b * Int(rows) + row] = shared[unsafe_offset=0]


def f32_gemm_kernel(
    w: Pointer[Float32, MutAnyOrigin],
    x: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    cols: Int32,
    batch: Int32,
    rows: Int32,
):
    var idx = Int(block_idx.x)
    var row = idx // Int(batch)
    var b = idx % Int(batch)
    var tid = Int(thread_idx.x)
    var base = row * Int(cols)
    var xbase = b * Int(cols)
    var shared = stack_allocation[
        F_THREADS, Float32, address_space=AddressSpace.SHARED
    ]()
    var acc = Float32(0.0)
    var steps = Int(cols) // F_THREADS
    for k in range(steps):
        var col = tid + k * F_THREADS
        acc += w[unsafe_offset=base + col] * x[unsafe_offset=xbase + col]
    shared[unsafe_offset=tid] = acc
    barrier()
    var stride = F_THREADS // 2
    while stride > 0:
        if tid < stride:
            shared[unsafe_offset=tid] = (
                shared[unsafe_offset=tid] + shared[unsafe_offset=tid + stride]
            )
        barrier()
        stride //= 2
    if tid == 0:
        y[unsafe_offset=b * Int(rows) + row] = shared[unsafe_offset=0]


def k1_gemm_kernel(
    w: Pointer[UInt8, MutAnyOrigin],
    x: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    cols: Int32,
    batch: Int32,
    rows: Int32,
):
    # The K1 r2sg workgroup body (k1_gemv_kernel) verbatim, with
    # the activation stream offset by the batch element and the
    # two output rows written into that element's Y row block.
    var idx = Int(block_idx.x)
    var wg = idx // Int(batch)
    var b = idx % Int(batch)
    var lane = Int(thread_idx.x)
    var group = lane // K1_GROUP_THREADS
    var t = lane % K1_GROUP_THREADS
    var half = t // 8
    var w0_off = 4 + 8 * half + (t % 8)
    var yv = 16 * half + (t % 8)
    var blocks_per_row = Int(cols) // K1_SB_ELEMS
    var row0 = wg * K1_ROWS_PER_WG
    var xbase = b * Int(cols)
    var ybase = b * Int(rows)
    var w32 = w.unsafe_bitcast[UInt32]()
    var shared = stack_allocation[
        K1_ROWS_PER_WG * K1_GROUPS, Float32, address_space=AddressSpace.SHARED
    ]()
    var acc0 = Float32(0.0)
    var acc1 = Float32(0.0)
    for blk in range(group, blocks_per_row, K1_GROUPS):
        var xb = xbase + blk * K1_SB_ELEMS + yv * 4
        var xa = x.unsafe_load[width=4](xb)
        var xbq = x.unsafe_load[width=4](xb + 32)
        var xc = x.unsafe_load[width=4](xb + 128)
        var xd = x.unsafe_load[width=4](xb + 160)
        var sa = xa.reduce_add()
        var sb = xbq.reduce_add()
        var sc_ = xc.reduce_add()
        var sd = xd.reduce_add()
        var bw0 = (row0 * blocks_per_row + blk) * K1_SB_WORDS
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
        var bw1 = ((row0 + 1) * blocks_per_row + blk) * K1_SB_WORDS
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
        shared[unsafe_offset=K1_GROUPS + group] = acc1
    barrier()
    if lane == 0:
        var s0 = Float32(0.0)
        var s1 = Float32(0.0)
        for g in range(K1_GROUPS):
            s0 += shared[unsafe_offset=g]
            s1 += shared[unsafe_offset=K1_GROUPS + g]
        y[unsafe_offset=ybase + row0] = s0
        y[unsafe_offset=ybase + row0 + 1] = s1


# ---- structure (3): register batch-tiled kernels -------------
#
# One workgroup owns ONE output row (K1: the usual row pair)
# for a tile of BATCH_TILE consecutive batch elements. Each
# weight value is loaded and dequantized once per workgroup
# and applied to every element of the tile from registers, so
# the weight stream crosses the memory system once per tile
# instead of once per batch element. Grid is the flat
# gemm_tiled_grid / gemm_k1_tiled_grid count, decomposed as
# idx = row * tiles + tile (K1: wg * tiles + tile). A partial
# final tile is handled by clamping the element index and
# zeroing its contributions (n_in_tile).


def q6k_gemm_tiled_kernel(
    w: Pointer[UInt8, MutAnyOrigin],
    x: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    cols: Int32,
    batch: Int32,
    rows: Int32,
):
    var tiles = (Int(batch) + BATCH_TILE - 1) // BATCH_TILE
    var idx = Int(block_idx.x)
    var row = idx // tiles
    var b0 = (idx % tiles) * BATCH_TILE
    var n_in = BATCH_TILE
    if b0 + n_in > Int(batch):
        n_in = Int(batch) - b0
    var tid = Int(thread_idx.x)
    var row_bytes = (Int(cols) // SB_ELEMS) * Q6_SB_BYTES
    var base = row * row_bytes
    var accs = stack_allocation[BATCH_TILE, Float32]()
    for j in range(BATCH_TILE):
        accs[unsafe_offset=j] = Float32(0.0)
    var steps = Int(cols) // Q6_THREADS
    for k in range(steps):
        var col = tid + k * Q6_THREADS
        var sb_off = base + (col >> 8) * Q6_SB_BYTES
        var val = q6k_element_ptr(w, sb_off, col & 255)
        for j in range(BATCH_TILE):
            if j < n_in:
                accs[unsafe_offset=j] = (
                    accs[unsafe_offset=j]
                    + val * x[unsafe_offset=(b0 + j) * Int(cols) + col]
                )
    var shared = stack_allocation[
        BATCH_TILE * Q6_THREADS, Float32, address_space=AddressSpace.SHARED
    ]()
    for j in range(BATCH_TILE):
        shared[unsafe_offset=j * Q6_THREADS + tid] = accs[unsafe_offset=j]
    barrier()
    var stride = Q6_THREADS // 2
    while stride > 0:
        if tid < stride:
            for j in range(BATCH_TILE):
                shared[unsafe_offset=j * Q6_THREADS + tid] = (
                    shared[unsafe_offset=j * Q6_THREADS + tid]
                    + shared[unsafe_offset=j * Q6_THREADS + tid + stride]
                )
        barrier()
        stride //= 2
    if tid == 0:
        for j in range(BATCH_TILE):
            if j < n_in:
                y[unsafe_offset=(b0 + j) * Int(rows) + row] = shared[
                    unsafe_offset=j * Q6_THREADS
                ]


def q5k_gemm_tiled_kernel(
    w: Pointer[UInt8, MutAnyOrigin],
    x: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    cols: Int32,
    batch: Int32,
    rows: Int32,
):
    var tiles = (Int(batch) + BATCH_TILE - 1) // BATCH_TILE
    var idx = Int(block_idx.x)
    var row = idx // tiles
    var b0 = (idx % tiles) * BATCH_TILE
    var n_in = BATCH_TILE
    if b0 + n_in > Int(batch):
        n_in = Int(batch) - b0
    var tid = Int(thread_idx.x)
    var row_bytes = (Int(cols) // SB_ELEMS) * Q5_SB_BYTES
    var base = row * row_bytes
    var accs = stack_allocation[BATCH_TILE, Float32]()
    for j in range(BATCH_TILE):
        accs[unsafe_offset=j] = Float32(0.0)
    var steps = Int(cols) // Q5_THREADS
    for k in range(steps):
        var col = tid + k * Q5_THREADS
        var sb_off = base + (col >> 8) * Q5_SB_BYTES
        var val = q5k_element_ptr(w, sb_off, col & 255)
        for j in range(BATCH_TILE):
            if j < n_in:
                accs[unsafe_offset=j] = (
                    accs[unsafe_offset=j]
                    + val * x[unsafe_offset=(b0 + j) * Int(cols) + col]
                )
    var shared = stack_allocation[
        BATCH_TILE * Q5_THREADS, Float32, address_space=AddressSpace.SHARED
    ]()
    for j in range(BATCH_TILE):
        shared[unsafe_offset=j * Q5_THREADS + tid] = accs[unsafe_offset=j]
    barrier()
    var stride = Q5_THREADS // 2
    while stride > 0:
        if tid < stride:
            for j in range(BATCH_TILE):
                shared[unsafe_offset=j * Q5_THREADS + tid] = (
                    shared[unsafe_offset=j * Q5_THREADS + tid]
                    + shared[unsafe_offset=j * Q5_THREADS + tid + stride]
                )
        barrier()
        stride //= 2
    if tid == 0:
        for j in range(BATCH_TILE):
            if j < n_in:
                y[unsafe_offset=(b0 + j) * Int(rows) + row] = shared[
                    unsafe_offset=j * Q5_THREADS
                ]


def f32_gemm_tiled_kernel(
    w: Pointer[Float32, MutAnyOrigin],
    x: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    cols: Int32,
    batch: Int32,
    rows: Int32,
):
    var tiles = (Int(batch) + BATCH_TILE - 1) // BATCH_TILE
    var idx = Int(block_idx.x)
    var row = idx // tiles
    var b0 = (idx % tiles) * BATCH_TILE
    var n_in = BATCH_TILE
    if b0 + n_in > Int(batch):
        n_in = Int(batch) - b0
    var tid = Int(thread_idx.x)
    var base = row * Int(cols)
    var accs = stack_allocation[BATCH_TILE, Float32]()
    for j in range(BATCH_TILE):
        accs[unsafe_offset=j] = Float32(0.0)
    var steps = Int(cols) // F_THREADS
    for k in range(steps):
        var col = tid + k * F_THREADS
        var val = w[unsafe_offset=base + col]
        for j in range(BATCH_TILE):
            if j < n_in:
                accs[unsafe_offset=j] = (
                    accs[unsafe_offset=j]
                    + val * x[unsafe_offset=(b0 + j) * Int(cols) + col]
                )
    var shared = stack_allocation[
        BATCH_TILE * F_THREADS, Float32, address_space=AddressSpace.SHARED
    ]()
    for j in range(BATCH_TILE):
        shared[unsafe_offset=j * F_THREADS + tid] = accs[unsafe_offset=j]
    barrier()
    var stride = F_THREADS // 2
    while stride > 0:
        if tid < stride:
            for j in range(BATCH_TILE):
                shared[unsafe_offset=j * F_THREADS + tid] = (
                    shared[unsafe_offset=j * F_THREADS + tid]
                    + shared[unsafe_offset=j * F_THREADS + tid + stride]
                )
        barrier()
        stride //= 2
    if tid == 0:
        for j in range(BATCH_TILE):
            if j < n_in:
                y[unsafe_offset=(b0 + j) * Int(rows) + row] = shared[
                    unsafe_offset=j * F_THREADS
                ]


def k1_gemm_tiled_kernel(
    w: Pointer[UInt8, MutAnyOrigin],
    x: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    cols: Int32,
    batch: Int32,
    rows: Int32,
):
    # The K1 r2sg workgroup over a batch tile: per super-block
    # the two rows' headers and packed quads are loaded and
    # unpacked ONCE (the structure-(2) kernel loads them once
    # per batch element), then applied to each tile element's
    # activation quads. Accumulators: accs[j] for row 0 of the
    # pair, accs[BATCH_TILE + j] for row 1.
    var tiles = (Int(batch) + BATCH_TILE - 1) // BATCH_TILE
    var idx = Int(block_idx.x)
    var wg = idx // tiles
    var b0 = (idx % tiles) * BATCH_TILE
    var n_in = BATCH_TILE
    if b0 + n_in > Int(batch):
        n_in = Int(batch) - b0
    var lane = Int(thread_idx.x)
    var group = lane // K1_GROUP_THREADS
    var t = lane % K1_GROUP_THREADS
    var half = t // 8
    var w0_off = 4 + 8 * half + (t % 8)
    var yv = 16 * half + (t % 8)
    var blocks_per_row = Int(cols) // K1_SB_ELEMS
    var row0 = wg * K1_ROWS_PER_WG
    var w32 = w.unsafe_bitcast[UInt32]()
    var accs = stack_allocation[2 * BATCH_TILE, Float32]()
    for j in range(2 * BATCH_TILE):
        accs[unsafe_offset=j] = Float32(0.0)
    for blk in range(group, blocks_per_row, K1_GROUPS):
        var bw0 = (row0 * blocks_per_row + blk) * K1_SB_WORDS
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
        var bw1 = ((row0 + 1) * blocks_per_row + blk) * K1_SB_WORDS
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
        for j in range(BATCH_TILE):
            if j < n_in:
                var xb = (b0 + j) * Int(cols) + blk * K1_SB_ELEMS + yv * 4
                var xa = x.unsafe_load[width=4](xb)
                var xbq = x.unsafe_load[width=4](xb + 32)
                var xc = x.unsafe_load[width=4](xb + 128)
                var xd = x.unsafe_load[width=4](xb + 160)
                var sa = xa.reduce_add()
                var sb = xbq.reduce_add()
                var sc_ = xc.reduce_add()
                var sd = xd.reduce_add()
                accs[unsafe_offset=j] = (
                    accs[unsafe_offset=j]
                    + d0
                    * (
                        sc_lo0[0] * (q0[0] * xa).reduce_add()
                        + sc_lo0[1] * (q0[1] * xbq).reduce_add()
                        + sc_hi0[0] * (q1[0] * xc).reduce_add()
                        + sc_hi0[1] * (q1[1] * xd).reduce_add()
                    )
                    - dmin0
                    * (
                        sc_lo0[2] * sa
                        + sc_lo0[3] * sb
                        + sc_hi0[2] * sc_
                        + sc_hi0[3] * sd
                    )
                )
                accs[unsafe_offset=BATCH_TILE + j] = (
                    accs[unsafe_offset=BATCH_TILE + j]
                    + d1
                    * (
                        sc_lo1[0] * (q2[0] * xa).reduce_add()
                        + sc_lo1[1] * (q2[1] * xbq).reduce_add()
                        + sc_hi1[0] * (q3[0] * xc).reduce_add()
                        + sc_hi1[1] * (q3[1] * xd).reduce_add()
                    )
                    - dmin1
                    * (
                        sc_lo1[2] * sa
                        + sc_lo1[3] * sb
                        + sc_hi1[2] * sc_
                        + sc_hi1[3] * sd
                    )
                )
    for j in range(2 * BATCH_TILE):
        var a = accs[unsafe_offset=j]
        a += shuffle_xor(a, UInt32(1))
        a += shuffle_xor(a, UInt32(2))
        a += shuffle_xor(a, UInt32(4))
        a += shuffle_xor(a, UInt32(8))
        accs[unsafe_offset=j] = a
    var shared = stack_allocation[
        K1_ROWS_PER_WG * K1_GROUPS * BATCH_TILE,
        Float32,
        address_space=AddressSpace.SHARED,
    ]()
    if t == 0:
        for j in range(BATCH_TILE):
            shared[unsafe_offset=j * K1_GROUPS + group] = accs[unsafe_offset=j]
            shared[unsafe_offset=(BATCH_TILE + j) * K1_GROUPS + group] = accs[
                unsafe_offset=BATCH_TILE + j
            ]
    barrier()
    if lane == 0:
        for j in range(BATCH_TILE):
            if j < n_in:
                var s0 = Float32(0.0)
                var s1 = Float32(0.0)
                for g in range(K1_GROUPS):
                    s0 += shared[unsafe_offset=j * K1_GROUPS + g]
                    s1 += shared[unsafe_offset=(BATCH_TILE + j) * K1_GROUPS + g]
                y[unsafe_offset=(b0 + j) * Int(rows) + row0] = s0
                y[unsafe_offset=(b0 + j) * Int(rows) + row0 + 1] = s1
