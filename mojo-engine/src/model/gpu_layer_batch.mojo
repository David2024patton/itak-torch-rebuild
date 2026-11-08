# gpu_layer_batch.mojo - the batched-prefill GPU layer
# assembly. Wave D Lane D1 built the multi-row twin of
# gpu_layer.mojo's gpu_layer_step over a flat batch of
# prompt rows (GEMM projections, host-chained ops, Wave B
# attention per position). Wave E Lane E1 rewires the
# batch step onto the Wave D resident shape:
#
#   gpu_layer_step_batch_cfg(rt, weights, kv, cpu_model,
#       x_flat, batch, layer, start_position, tiled_min_batch,
#       attn_tile, attn_max_splits, attn_split_span)
#
# x_flat is [batch][hidden] row-major; the return is the
# same shape. Batch element i is processed at absolute
# position start_position + i, exactly as gpu_layer_step
# would process it alone.
#
# RESIDENCY (Wave E): activations stay in device buffers
# across the whole layer, the way gpu_layer_step_resident
# chains the per-row path: the input batch uploads once,
# the residual stream, norm outputs, projection outputs,
# the attention output and the MLP intermediates are
# device buffers passed op to op, the KV appends are
# device-to-device scatters from the resident K/V batch
# buffers, the residual adds are device adds, and the
# output batch downloads once. The only host crossings
# are those two boundary crossings (plus the small RoPE
# positions buffer, exactly as the resident step builds
# it). The Wave D host-chained form is gone from this
# file; the per-row hostchain baseline in gpu_layer.mojo
# is the A/B surface that remains.
#
# ATTENTION (Wave E): the per-position Wave B attention
# launches are replaced by a batched twin of Lane D2's
# tiled split-KV pair (attention_batch_tiled_kernel +
# attention_batch_merge_kernel below): the same shared-
# memory K/V tile staging, one block per (batch element,
# Q head, split), the same online-softmax partials, and
# the merge kernel's exact logsumexp rescaling - no
# approximation. Causality is per batch element: element
# i's blocks partition [0, start_position + i + 1) stored
# rows, which is exactly the row set the per-position
# steps would have seen (the whole batch is appended
# before the layer's attention launch, as in Wave D, and
# each element reads only its causal prefix of the
# visible-order buffers). Split count is chosen once per
# layer from the LAST element's stored count via
# attn_split_count; earlier elements get proportionally
# smaller ranges from the same chooser arithmetic, with
# trailing empty splits merging as no-ops, exactly as
# the split CPU twin treats splits > stored. The KV
# buffers are read through DeviceKvCache.gather_visible,
# like the resident step. The pre-wrap guard is kept:
# a windowed batch whose last position would reach the
# window refuses gpu_kv_window_wrapped, and attention
# refuses by the same name if the visible rows do not
# start at position 0 (the prefix arithmetic above is
# stated for the no-wrap layout, as the Wave D batch
# step's per-position launch was).
#
# Shape/dtype discipline mirrors gpu_layer.mojo: every
# slot is cross-checked against the CPU model's slot
# before launch (gpu_layer_shape_mismatch), bad shapes
# refuse gpu_layer_bad_shape, and dtypes outside the
# GEMM family's set refuse gpu_layer_unsupported_dtype -
# that includes F16: Glimmer carries no F16 weight
# tensors (Wave A census) and the GEMM family builds
# only the shapes prefill dispatches (Q4_K, Q6_K, Q5_K,
# F32); an F16 projection slot in a batch refuses by
# name here while single rows keep the GEMV path.
#
# Diag seam: phase `gpu.layer`, as gpu_layer.mojo.

from std.math import sqrt, exp
from std.memory import stack_allocation

from max.gpu import thread_idx, block_idx, block_dim, barrier
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceBuffer

from diag import log_debug
from gpu.context import GpuRuntime, upload_f32, download_f32, create_zero_f32
from gpu.weights import DeviceWeights, DeviceSlotU8, DeviceSlotF32
from gpu.device_kv import DeviceKvCache
from model.model import (
    GlimmerModel,
    POST_NORM_EPS,
    DT_F32,
    DT_Q4_K,
    DT_Q5_K,
    DT_Q6_K,
    LAYER_SUFFIX_COUNT,
)
from model.gpu_layer import (
    gpu_layer_check_norm,
    rope_launch_shape,
    attn_gate_kernel,
    GATE_THREADS,
)
from kernels.gemm import (
    k1_gemm_kernel,
    k1_gemm_tiled_kernel,
    q6k_gemm_kernel,
    q6k_gemm_tiled_kernel,
    q5k_gemm_kernel,
    q5k_gemm_tiled_kernel,
    f32_gemm_kernel,
    f32_gemm_tiled_kernel,
    gemm_plain_grid,
    gemm_tiled_grid,
    gemm_k1_grid,
    gemm_k1_tiled_grid,
    BATCH_TILE,
)
from kernels.k1_gemv import THREADS as K1_THREADS, ROWS_PER_WG as K1_RPW
from kernels.q6k_gemv import THREADS as Q6_THREADS
from kernels.f32_gemv import THREADS as F_THREADS
from ops.rms_norm import rms_norm_kernel, RMS_THREADS
from ops.rope import rope_kernel
from ops.swiglu import swiglu_kernel
from ops.residual import residual_add_kernel
from attention.attention import (
    ATTN_THREADS,
    ATTN_MAX_HEAD_DIM,
    ATTN_NEG_INF,
    attention_split_cpu,
    attn_split_count,
    attn_validate_tile,
)
from config import (
    DEFAULT_ATTN_MAX_SPLITS,
    DEFAULT_ATTN_SPLIT_SPAN_POSITIONS,
    DEFAULT_ATTN_TILE_POSITIONS,
)

comptime _KV_SCATTER_BLOCK = 256


# ---- pure shape helpers (VM-testable, no device) -------------


def gpu_layer_check_proj_batch(
    name: String, type_id: Int, rows: Int, cols: Int, batch: Int, x_len: Int
) raises:
    """Validate one batched projection launch against its
    slot shape and the GEMM family's launch rules, by name."""
    if batch <= 0 or rows <= 0 or cols <= 0:
        raise Error(
            "gpu_layer_bad_shape: "
            + name
            + " has non-positive batch/shape "
            + String(batch)
            + "/"
            + String(rows)
            + "x"
            + String(cols)
        )
    if x_len != batch * cols:
        raise Error(
            "gpu_layer_bad_shape: "
            + name
            + " expects a flat batch input of "
            + String(batch * cols)
            + " elements, got "
            + String(x_len)
        )
    if (
        type_id != DT_Q4_K
        and type_id != DT_Q6_K
        and type_id != DT_Q5_K
        and type_id != DT_F32
    ):
        raise Error(
            "gpu_layer_unsupported_dtype: "
            + name
            + " has ggml type id "
            + String(type_id)
            + "; the prefill GEMM family covers Q4_K/Q6_K/Q5_K/F32"
        )
    if cols % 256 != 0:
        raise Error(
            "gpu_layer_bad_shape: "
            + name
            + " cols "
            + String(cols)
            + " is not a multiple of the 256-element kernel step"
        )
    if type_id == DT_Q4_K and rows % K1_RPW != 0:
        raise Error(
            "gpu_layer_bad_shape: "
            + name
            + " rows "
            + String(rows)
            + " is not a multiple of the K1 workgroup tile "
            + String(K1_RPW)
        )


def batch_row_slice(x: List[Float32], i: Int, width: Int) -> List[Float32]:
    """Batch element i's row of a flat [batch][width] list."""
    var out = List[Float32](capacity=width)
    out.resize(width, Float32(0.0))
    for j in range(width):
        out[j] = x[i * width + j]
    return out^


def batch_row_place(
    mut x: List[Float32], i: Int, width: Int, row: List[Float32]
):
    for j in range(width):
        x[i * width + j] = row[j]


def batch_attention_stored(start_position: Int, index: Int) raises -> Int:
    """The causal stored-row count for batch element
    `index`: it attends to every position appended up to
    and including its own, start_position + index + 1 in
    the no-wrap layout this module's attention reads."""
    if start_position < 0 or index < 0:
        raise Error(
            "gpu_layer_bad_shape: batch attention stored count needs"
            " start_position >= 0 and index >= 0, got "
            + String(start_position)
            + "/"
            + String(index)
        )
    return start_position + index + 1


def attention_batch_causal_cpu(
    q_rows: List[Float32],
    keys: List[Float32],
    values: List[Float32],
    first_stored: Int,
    num_q_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    splits: Int,
    tile: Int,
) raises -> List[Float32]:
    """CPU twin of the batched tiled attention: per batch
    element, Lane D2's attention_split_cpu over that
    element's causal prefix of the PRE-ROTATED flat stores
    (keys/values hold every batch element's rows in
    visible order; element i reads the first
    batch_attention_stored rows). The VM-side proof
    surface for the causal prefix arithmetic; the GPU
    kernels below mirror the same per-element ranges."""
    var q_width = num_q_heads * head_dim
    var kv_width = num_kv_heads * head_dim
    if q_width <= 0 or kv_width <= 0 or len(q_rows) % q_width != 0:
        raise Error(
            "gpu_layer_bad_shape: batch attention Q rows have "
            + String(len(q_rows))
            + " elements, not a whole number of "
            + String(q_width)
            + "-wide rows"
        )
    var batch = len(q_rows) // q_width
    var total_stored = first_stored + batch - 1
    if batch <= 0 or first_stored <= 0:
        raise Error(
            "gpu_layer_bad_shape: batch attention needs a positive batch"
            " and first_stored, got "
            + String(batch)
            + "/"
            + String(first_stored)
        )
    if len(keys) != total_stored * kv_width or len(values) != len(keys):
        raise Error(
            "gpu_layer_bad_shape: batch attention stores have "
            + String(len(keys))
            + "/"
            + String(len(values))
            + " elements, expected "
            + String(total_stored * kv_width)
            + " each"
        )
    var out = List[Float32](capacity=len(q_rows))
    out.resize(len(q_rows), Float32(0.0))
    for i in range(batch):
        var stored = batch_attention_stored(first_stored - 1, i)
        var q_row = batch_row_slice(q_rows, i, q_width)
        var k_sub = List[Float32](capacity=stored * kv_width)
        k_sub.resize(stored * kv_width, Float32(0.0))
        var v_sub = List[Float32](capacity=stored * kv_width)
        v_sub.resize(stored * kv_width, Float32(0.0))
        for j in range(stored * kv_width):
            k_sub[j] = keys[j]
            v_sub[j] = values[j]
        var row = attention_split_cpu(
            q_row,
            k_sub,
            v_sub,
            stored,
            num_q_heads,
            num_kv_heads,
            head_dim,
            splits,
            tile,
        )
        batch_row_place(out, i, q_width, row)
    return out^


# ---- batched attention kernels (Wave E) ----------------------
#
# attention_batch_tiled_kernel is attention_tiled_kernel
# (attention/attention.mojo, Lane D2) with one added
# dimension: the block index decomposes as
# ((batch element) * q_heads + head) * splits + split, the
# Q row and the partial slot are offset by the batch
# element, and each element's split ranges partition its
# OWN causal stored count (first_stored + element). The
# tile staging, online softmax, and partial layout per
# element are D2's, statement for statement; the merge
# kernel is attention_merge_kernel with the same batch
# offset on its input partials and output row.


def attention_batch_tiled_kernel[
    TILE: Int
](
    q: Pointer[Float32, MutAnyOrigin],
    keys: Pointer[Float32, MutAnyOrigin],
    values: Pointer[Float32, MutAnyOrigin],
    p_out: Pointer[Float32, MutAnyOrigin],
    p_max: Pointer[Float32, MutAnyOrigin],
    p_sum: Pointer[Float32, MutAnyOrigin],
    first_stored: Int32,
    num_kv_heads: Int32,
    num_q_heads: Int32,
    head_dim: Int32,
    group: Int32,
    scale: Float32,
    splits: Int32,
):
    var n_splits = Int(splits)
    var qh = Int(num_q_heads)
    var idx = Int(block_idx.x)
    var split = idx % n_splits
    var head = (idx // n_splits) % qh
    var b = idx // (n_splits * qh)
    var tid = Int(thread_idx.x)
    var hd = Int(head_dim)
    var kv_head = head // Int(group)
    var row = Int(num_kv_heads) * hd
    var nstored = Int(first_stored) + b
    var base_chunk = nstored // n_splits
    var rem = nstored % n_splits
    var start: Int
    var length: Int
    if split < rem:
        start = split * (base_chunk + 1)
        length = base_chunk + 1
    else:
        start = rem * (base_chunk + 1) + (split - rem) * base_chunk
        length = base_chunk
    var end = start + length
    var tile_buf = stack_allocation[
        TILE * ATTN_MAX_HEAD_DIM, Float32, address_space=AddressSpace.SHARED
    ]()
    var scores = stack_allocation[
        TILE, Float32, address_space=AddressSpace.SHARED
    ]()
    var q_sh = stack_allocation[
        ATTN_MAX_HEAD_DIM, Float32, address_space=AddressSpace.SHARED
    ]()
    var red = stack_allocation[
        ATTN_THREADS, Float32, address_space=AddressSpace.SHARED
    ]()
    if tid < hd:
        q_sh[unsafe_offset=tid] = q[unsafe_offset=b * qh * hd + head * hd + tid]
    barrier()
    var d = tid
    var acc = Float32(0.0)
    var run_max = ATTN_NEG_INF
    var run_sum = Float32(0.0)
    var tile_start = start
    while tile_start < end:
        var tcount = TILE
        if end - tile_start < tcount:
            tcount = end - tile_start
        var e = tid
        while e < tcount * hd:
            var t = e // hd
            var dd = e % hd
            tile_buf[unsafe_offset=e] = keys[
                unsafe_offset=(tile_start + t) * row + kv_head * hd + dd
            ]
            e += ATTN_THREADS
        barrier()
        if tid < tcount:
            var s = Float32(0.0)
            for dd2 in range(hd):
                s += (
                    q_sh[unsafe_offset=dd2]
                    * tile_buf[unsafe_offset=tid * hd + dd2]
                )
            scores[unsafe_offset=tid] = s * scale
        barrier()
        if tid == 0:
            var m = run_max
            for t2 in range(tcount):
                if scores[unsafe_offset=t2] > m:
                    m = scores[unsafe_offset=t2]
            red[unsafe_offset=0] = m
        barrier()
        var new_max = red[unsafe_offset=0]
        var alpha = Float32(0.0)
        if run_sum != Float32(0.0):
            alpha = exp(run_max - new_max)
        if tid < tcount:
            scores[unsafe_offset=tid] = exp(scores[unsafe_offset=tid] - new_max)
        barrier()
        e = tid
        while e < tcount * hd:
            var t3 = e // hd
            var dd3 = e % hd
            tile_buf[unsafe_offset=e] = values[
                unsafe_offset=(tile_start + t3) * row + kv_head * hd + dd3
            ]
            e += ATTN_THREADS
        barrier()
        if tid == 0:
            var psum = Float32(0.0)
            for t4 in range(tcount):
                psum += scores[unsafe_offset=t4]
            red[unsafe_offset=1] = psum
        if d < hd:
            var a = acc * alpha
            for t5 in range(tcount):
                a += (
                    scores[unsafe_offset=t5]
                    * tile_buf[unsafe_offset=t5 * hd + d]
                )
            acc = a
        barrier()
        run_sum = run_sum * alpha + red[unsafe_offset=1]
        run_max = new_max
        tile_start += TILE
    var p_base = ((b * n_splits + split) * qh + head) * hd
    if d < hd:
        p_out[unsafe_offset=p_base + d] = acc
    if tid == 0:
        p_max[unsafe_offset=(b * n_splits + split) * qh + head] = run_max
        p_sum[unsafe_offset=(b * n_splits + split) * qh + head] = run_sum


def attention_batch_merge_kernel(
    p_out: Pointer[Float32, MutAnyOrigin],
    p_max: Pointer[Float32, MutAnyOrigin],
    p_sum: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    num_q_heads: Int32,
    head_dim: Int32,
    splits: Int32,
):
    """Batched split-KV merge: one block per (batch
    element, Q head); splits combined in order with the
    exact logsumexp rescaling of attention_merge_kernel,
    including the empty-partial skip and the first-non-
    empty adoption."""
    var qh = Int(num_q_heads)
    var idx = Int(block_idx.x)
    var head = idx % qh
    var b = idx // qh
    var tid = Int(thread_idx.x)
    var hd = Int(head_dim)
    var d = tid
    var acc = Float32(0.0)
    var run_max = ATTN_NEG_INF
    var run_sum = Float32(0.0)
    for s in range(Int(splits)):
        var slot = (b * Int(splits) + s) * qh + head
        var ps = p_sum[unsafe_offset=slot]
        if ps != Float32(0.0):
            var pm = p_max[unsafe_offset=slot]
            var base = slot * hd
            if run_sum == Float32(0.0):
                run_max = pm
                run_sum = ps
                if d < hd:
                    acc = p_out[unsafe_offset=base + d]
            else:
                var new_max = run_max if run_max > pm else pm
                var a1 = exp(run_max - new_max)
                var a2 = exp(pm - new_max)
                if d < hd:
                    acc = acc * a1 + p_out[unsafe_offset=base + d] * a2
                run_sum = run_sum * a1 + ps * a2
                run_max = new_max
    if d < hd:
        y[unsafe_offset=b * qh * hd + head * hd + d] = acc / run_sum


def kv_batch_scatter_kernel(
    dst: Pointer[Float32, MutAnyOrigin],
    src: Pointer[Float32, MutAnyOrigin],
    row_len: Int32,
    dst_offset: Int32,
    src_offset: Int32,
):
    """One row between a resident batch buffer and its
    slot in a resident KV buffer, both at explicit
    offsets: the batched twin of device_kv's
    kv_scatter_kernel (which carries no source offset,
    because the per-row path's K/V buffers hold exactly
    one row). Threads stride the row, same shape."""
    var i = Int(thread_idx.x)
    while i < Int(row_len):
        dst[unsafe_offset=Int(dst_offset) + i] = src[
            unsafe_offset=Int(src_offset) + i
        ]
        i += _KV_SCATTER_BLOCK


# ---- device-resident launch helpers --------------------------
#
# The Wave D batch step's launches, chained on DeviceBuffers:
# no upload, no download, no per-op sync (the context's single
# queue orders the launches; the one sync happens in
# download_f32 at the layer boundary). Launch geometry mirrors
# the Wave D host helpers of this file site for site (GEMM
# grids/block dims, rms_norm grid = row count, gate/SwiGLU
# grids from GATE_THREADS) - the kernels did not change.


def _rms_norm_buf_batch(
    mut rt: GpuRuntime,
    x_buf: DeviceBuffer[DType.float32],
    n: Int,
    w_buf: DeviceBuffer[DType.float32],
    row_len: Int,
    eps: Float32,
) raises -> DeviceBuffer[DType.float32]:
    gpu_layer_check_norm(n, row_len, row_len)
    var rows = n // row_len
    var y_dev = create_zero_f32(rt, n)
    rt.ctx.enqueue_function[rms_norm_kernel](
        x_buf,
        w_buf,
        y_dev,
        Int32(row_len),
        eps,
        grid_dim=rows,
        block_dim=RMS_THREADS,
    )
    return y_dev^


def _gemm_u8_buf_batch(
    mut rt: GpuRuntime,
    view: DeviceSlotU8,
    x_buf: DeviceBuffer[DType.float32],
    batch: Int,
    tiled_min_batch: Int,
) raises -> DeviceBuffer[DType.float32]:
    """One GEMM launch for a byte-stored slot over the whole
    resident batch; tiled structure at or above
    tiled_min_batch. Launch geometry as the Wave D
    _gemm_u8_batch site."""
    gpu_layer_check_proj_batch(
        view.name, view.type_id, view.rows, view.cols, batch, batch * view.cols
    )
    var y_dev = create_zero_f32(rt, batch * view.rows)
    var tiled = batch >= tiled_min_batch
    if view.type_id == DT_Q4_K:
        if tiled:
            rt.ctx.enqueue_function[k1_gemm_tiled_kernel](
                view.buffer,
                x_buf,
                y_dev,
                Int32(view.cols),
                Int32(batch),
                Int32(view.rows),
                grid_dim=gemm_k1_tiled_grid(view.rows, batch),
                block_dim=K1_THREADS,
            )
        else:
            rt.ctx.enqueue_function[k1_gemm_kernel](
                view.buffer,
                x_buf,
                y_dev,
                Int32(view.cols),
                Int32(batch),
                Int32(view.rows),
                grid_dim=gemm_k1_grid(view.rows, batch),
                block_dim=K1_THREADS,
            )
    elif view.type_id == DT_Q6_K:
        if tiled:
            rt.ctx.enqueue_function[q6k_gemm_tiled_kernel](
                view.buffer,
                x_buf,
                y_dev,
                Int32(view.cols),
                Int32(batch),
                Int32(view.rows),
                grid_dim=gemm_tiled_grid(view.rows, batch),
                block_dim=Q6_THREADS,
            )
        else:
            rt.ctx.enqueue_function[q6k_gemm_kernel](
                view.buffer,
                x_buf,
                y_dev,
                Int32(view.cols),
                Int32(batch),
                Int32(view.rows),
                grid_dim=gemm_plain_grid(view.rows, batch),
                block_dim=Q6_THREADS,
            )
    elif view.type_id == DT_Q5_K:
        if tiled:
            rt.ctx.enqueue_function[q5k_gemm_tiled_kernel](
                view.buffer,
                x_buf,
                y_dev,
                Int32(view.cols),
                Int32(batch),
                Int32(view.rows),
                grid_dim=gemm_tiled_grid(view.rows, batch),
                block_dim=Q6_THREADS,
            )
        else:
            rt.ctx.enqueue_function[q5k_gemm_kernel](
                view.buffer,
                x_buf,
                y_dev,
                Int32(view.cols),
                Int32(batch),
                Int32(view.rows),
                grid_dim=gemm_plain_grid(view.rows, batch),
                block_dim=Q6_THREADS,
            )
    else:
        raise Error(
            "gpu_layer_unsupported_dtype: "
            + view.name
            + " reached the byte GEMM path with ggml type id "
            + String(view.type_id)
        )
    _ = log_debug(
        "gpu.layer",
        "gemm resident "
        + view.name
        + " batch="
        + String(batch)
        + " tiled="
        + String(tiled),
    )
    return y_dev^


def _gemm_f32_buf_batch(
    mut rt: GpuRuntime,
    view: DeviceSlotF32,
    x_buf: DeviceBuffer[DType.float32],
    batch: Int,
    tiled_min_batch: Int,
) raises -> DeviceBuffer[DType.float32]:
    """One GEMM launch for an F32 slot's resident buffer
    over the resident batch."""
    gpu_layer_check_proj_batch(
        view.name, view.type_id, view.rows, view.cols, batch, batch * view.cols
    )
    var y_dev = create_zero_f32(rt, batch * view.rows)
    if batch >= tiled_min_batch:
        rt.ctx.enqueue_function[f32_gemm_tiled_kernel](
            view.buffer,
            x_buf,
            y_dev,
            Int32(view.cols),
            Int32(batch),
            Int32(view.rows),
            grid_dim=gemm_tiled_grid(view.rows, batch),
            block_dim=F_THREADS,
        )
    else:
        rt.ctx.enqueue_function[f32_gemm_kernel](
            view.buffer,
            x_buf,
            y_dev,
            Int32(view.cols),
            Int32(batch),
            Int32(view.rows),
            grid_dim=gemm_plain_grid(view.rows, batch),
            block_dim=F_THREADS,
        )
    _ = log_debug("gpu.layer", "gemm f32 resident " + view.name)
    return y_dev^


def _check_view_batch(
    name: String,
    type_id: Int,
    rows: Int,
    cols: Int,
    cpu_type: Int,
    cpu_rows: Int,
    cpu_cols: Int,
) raises:
    if type_id != cpu_type or rows != cpu_rows or cols != cpu_cols:
        raise Error(
            "gpu_layer_shape_mismatch: "
            + name
            + " device slot (type "
            + String(type_id)
            + ", "
            + String(rows)
            + "x"
            + String(cols)
            + ") disagrees with the CPU slot (type "
            + String(cpu_type)
            + ", "
            + String(cpu_rows)
            + "x"
            + String(cpu_cols)
            + ")"
        )


def _norm_buffer_batch(
    weights: DeviceWeights,
    cpu_model: GlimmerModel,
    layer: Int,
    suffix_index: Int,
    row_len: Int,
) raises -> DeviceBuffer[DType.float32]:
    """A per-layer norm's RESIDENT float32 buffer, validated
    as in gpu_layer._norm_buffer."""
    var idx = layer * LAYER_SUFFIX_COUNT + suffix_index
    if layer < 0 or idx < 0 or idx >= len(cpu_model.layer_weights):
        raise Error(
            "gpu_layer_bad_shape: norm slot layer "
            + String(layer)
            + " suffix "
            + String(suffix_index)
            + " outside the CPU model's slots"
        )
    if cpu_model.layer_weights[idx].type_id != DT_F32:
        raise Error(
            "gpu_layer_unsupported_dtype: norm slot "
            + cpu_model.layer_weights[idx].name
            + " has ggml type id "
            + String(cpu_model.layer_weights[idx].type_id)
            + ", expected F32 values"
        )
    var view = weights.slot_f32(layer, suffix_index)
    if view.cols != row_len or view.rows != 1:
        raise Error(
            "gpu_layer_shape_mismatch: norm slot "
            + view.name
            + " device shape "
            + String(view.rows)
            + "x"
            + String(view.cols)
            + " is not a row of "
            + String(row_len)
        )
    return view.buffer.copy()


def _proj_layer_buf_batch(
    mut rt: GpuRuntime,
    weights: DeviceWeights,
    cpu_model: GlimmerModel,
    layer: Int,
    suffix_index: Int,
    x_buf: DeviceBuffer[DType.float32],
    batch: Int,
    tiled_min_batch: Int,
) raises -> DeviceBuffer[DType.float32]:
    """One per-layer projection over the whole resident
    batch against the resident weight buffer, cross-checked
    like _proj_layer_batch."""
    var idx = layer * LAYER_SUFFIX_COUNT + suffix_index
    if layer < 0 or idx < 0 or idx >= len(cpu_model.layer_weights):
        raise Error(
            "gpu_layer_bad_shape: layer "
            + String(layer)
            + " suffix "
            + String(suffix_index)
            + " outside the CPU model's slots"
        )
    var cpu_type = cpu_model.layer_weights[idx].type_id
    var cpu_rows = cpu_model.layer_weights[idx].rows
    var cpu_cols = cpu_model.layer_weights[idx].cols
    var name = cpu_model.layer_weights[idx].name.copy()
    if cpu_type == DT_F32:
        var fview = weights.slot_f32(layer, suffix_index)
        _check_view_batch(
            name,
            fview.type_id,
            fview.rows,
            fview.cols,
            cpu_type,
            cpu_rows,
            cpu_cols,
        )
        return _gemm_f32_buf_batch(rt, fview, x_buf, batch, tiled_min_batch)
    var view = weights.slot_u8(layer, suffix_index)
    _check_view_batch(
        name, view.type_id, view.rows, view.cols, cpu_type, cpu_rows, cpu_cols
    )
    return _gemm_u8_buf_batch(rt, view, x_buf, batch, tiled_min_batch)


def _rope_buf_batch(
    mut rt: GpuRuntime,
    x_buf: DeviceBuffer[DType.float32],
    n: Int,
    batch: Int,
    vectors_per_batch: Int,
    head_dim: Int,
    start_position: Int,
    base: Float32,
) raises -> DeviceBuffer[DType.float32]:
    """rope_kernel over a resident flat [batch][vectors]
    [head_dim] batch; the positions buffer carries each
    element's own absolute position, as the Wave D batch
    step built it on the host path."""
    var vectors = batch * vectors_per_batch
    if n != vectors * head_dim:
        raise Error(
            "gpu_layer_bad_shape: rope batch input has "
            + String(n)
            + " elements, expected "
            + String(vectors * head_dim)
        )
    var shape = rope_launch_shape(vectors * (head_dim // 2))
    var y_dev = create_zero_f32(rt, n)
    var pos_host = rt.ctx.enqueue_create_host_buffer[DType.int32](vectors)
    for v in range(vectors):
        pos_host[v] = Int32(start_position + v // vectors_per_batch)
    var pos_dev = rt.ctx.enqueue_create_buffer[DType.int32](vectors)
    rt.ctx.enqueue_copy(src_buf=pos_host, dst_buf=pos_dev)
    rt.ctx.enqueue_function[rope_kernel](
        x_buf,
        y_dev,
        pos_dev,
        Int32(head_dim),
        base,
        grid_dim=shape[0],
        block_dim=shape[1],
    )
    return y_dev^


def _kv_append_batch_device(
    mut rt: GpuRuntime,
    mut kv: DeviceKvCache,
    layer: Int,
    k_buf: DeviceBuffer[DType.float32],
    v_buf: DeviceBuffer[DType.float32],
    batch: Int,
    start_position: Int,
) raises:
    """Append the whole batch's K/V rows from the RESIDENT
    batch buffers into the layer's KV buffers, device to
    device: per element, the cache's own slot validation
    and commit (DeviceKvCache._append_slot/_append_commit,
    the exact calls append_device makes for one row) with
    kv_batch_scatter_kernel carrying the batch row at its
    source offset. No host crossing, no per-row sync - the
    queue orders the scatters and the host-side commits
    mirror append order."""
    var row = kv.row_len()
    var k_dst = kv.key_buffer(layer)
    var v_dst = kv.value_buffer(layer)
    for i in range(batch):
        var position = start_position + i
        var slot = kv._append_slot(layer, position, row, row)
        var dst_offset = slot * row
        var src_offset = i * row
        rt.ctx.enqueue_function[kv_batch_scatter_kernel](
            k_dst,
            k_buf,
            Int32(row),
            Int32(dst_offset),
            Int32(src_offset),
            grid_dim=1,
            block_dim=_KV_SCATTER_BLOCK,
        )
        rt.ctx.enqueue_function[kv_batch_scatter_kernel](
            v_dst,
            v_buf,
            Int32(row),
            Int32(dst_offset),
            Int32(src_offset),
            grid_dim=1,
            block_dim=_KV_SCATTER_BLOCK,
        )
        kv._append_commit(layer, slot, position)


def _launch_batch_tiled(
    mut rt: GpuRuntime,
    tile: Int,
    q_buf: DeviceBuffer[DType.float32],
    k_buf: DeviceBuffer[DType.float32],
    v_buf: DeviceBuffer[DType.float32],
    p_out: DeviceBuffer[DType.float32],
    p_max: DeviceBuffer[DType.float32],
    p_sum: DeviceBuffer[DType.float32],
    first_stored: Int,
    kv_heads: Int,
    q_heads: Int,
    head_dim: Int,
    scale: Float32,
    splits: Int,
    batch: Int,
) raises:
    """One attention_batch_tiled_kernel launch at the
    compiled instantiation for `tile` (attn_validate_tile
    has already gated the value): grid = batch * q_heads *
    splits blocks, block = ATTN_THREADS, mirroring
    gpu_layer._launch_tiled's geometry for the pair."""
    var grid = batch * q_heads * splits
    if tile == 8:
        rt.ctx.enqueue_function[attention_batch_tiled_kernel[8]](
            q_buf,
            k_buf,
            v_buf,
            p_out,
            p_max,
            p_sum,
            Int32(first_stored),
            Int32(kv_heads),
            Int32(q_heads),
            Int32(head_dim),
            Int32(q_heads // kv_heads),
            scale,
            Int32(splits),
            grid_dim=grid,
            block_dim=ATTN_THREADS,
        )
    elif tile == 16:
        rt.ctx.enqueue_function[attention_batch_tiled_kernel[16]](
            q_buf,
            k_buf,
            v_buf,
            p_out,
            p_max,
            p_sum,
            Int32(first_stored),
            Int32(kv_heads),
            Int32(q_heads),
            Int32(head_dim),
            Int32(q_heads // kv_heads),
            scale,
            Int32(splits),
            grid_dim=grid,
            block_dim=ATTN_THREADS,
        )
    elif tile == 32:
        rt.ctx.enqueue_function[attention_batch_tiled_kernel[32]](
            q_buf,
            k_buf,
            v_buf,
            p_out,
            p_max,
            p_sum,
            Int32(first_stored),
            Int32(kv_heads),
            Int32(q_heads),
            Int32(head_dim),
            Int32(q_heads // kv_heads),
            scale,
            Int32(splits),
            grid_dim=grid,
            block_dim=ATTN_THREADS,
        )
    else:
        rt.ctx.enqueue_function[attention_batch_tiled_kernel[64]](
            q_buf,
            k_buf,
            v_buf,
            p_out,
            p_max,
            p_sum,
            Int32(first_stored),
            Int32(kv_heads),
            Int32(q_heads),
            Int32(head_dim),
            Int32(q_heads // kv_heads),
            scale,
            Int32(splits),
            grid_dim=grid,
            block_dim=ATTN_THREADS,
        )


def _attention_batch_tiled_buf(
    mut rt: GpuRuntime,
    mut kv: DeviceKvCache,
    layer: Int,
    q_buf: DeviceBuffer[DType.float32],
    batch: Int,
    start_position: Int,
    q_heads: Int,
    kv_heads: Int,
    head_dim: Int,
    tile: Int,
    max_splits: Int,
    split_span: Int,
) raises -> DeviceBuffer[DType.float32]:
    """Batched tiled split-KV attention over the layer's
    visible rows, Q resident: one tiled launch and one
    merge launch for the whole batch. The K/V sources are
    DeviceKvCache.gather_visible buffers (the raw buffers
    when their order already is the visible order), and
    element i reads only its causal prefix of them."""
    var stored_total = kv.stored_len(layer)
    if stored_total <= 0:
        raise Error(
            "gpu_layer_bad_shape: attention on layer "
            + String(layer)
            + " with no stored positions"
        )
    var first_vis = kv.first_visible_position(layer)
    if first_vis != 0:
        raise Error(
            "gpu_kv_window_wrapped: layer "
            + String(layer)
            + " first visible position is "
            + String(first_vis)
            + "; the batch attention prefix layout covers the"
            " pre-wrap layout only"
        )
    var stored_max = batch_attention_stored(start_position, batch - 1)
    if stored_max > stored_total:
        raise Error(
            "gpu_layer_bad_shape: batch attention last element needs "
            + String(stored_max)
            + " stored rows on layer "
            + String(layer)
            + ", cache holds "
            + String(stored_total)
        )
    attn_validate_tile(tile)
    var splits = attn_split_count(stored_max, split_span, max_splits)
    var y_dev = create_zero_f32(rt, batch * q_heads * head_dim)
    var k_vis = kv.gather_visible(rt, layer, False)
    var v_vis = kv.gather_visible(rt, layer, True)
    var p_out = rt.ctx.enqueue_create_buffer[DType.float32](
        splits * batch * q_heads * head_dim
    )
    var p_max = rt.ctx.enqueue_create_buffer[DType.float32](
        splits * batch * q_heads
    )
    var p_sum = rt.ctx.enqueue_create_buffer[DType.float32](
        splits * batch * q_heads
    )
    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    _launch_batch_tiled(
        rt,
        tile,
        q_buf,
        k_vis,
        v_vis,
        p_out,
        p_max,
        p_sum,
        start_position + 1,
        kv_heads,
        q_heads,
        head_dim,
        scale,
        splits,
        batch,
    )
    rt.ctx.enqueue_function[attention_batch_merge_kernel](
        p_out,
        p_max,
        p_sum,
        y_dev,
        Int32(q_heads),
        Int32(head_dim),
        Int32(splits),
        grid_dim=batch * q_heads,
        block_dim=ATTN_THREADS,
    )
    _ = log_debug(
        "gpu.layer",
        "attention batch tiled stored_max="
        + String(stored_max)
        + " batch="
        + String(batch)
        + " tile="
        + String(tile)
        + " splits="
        + String(splits),
    )
    return y_dev^


def _attn_gate_buf_batch(
    mut rt: GpuRuntime,
    a_buf: DeviceBuffer[DType.float32],
    g_buf: DeviceBuffer[DType.float32],
    n: Int,
) raises -> DeviceBuffer[DType.float32]:
    var y_dev = create_zero_f32(rt, n)
    var grid = (n + GATE_THREADS - 1) // GATE_THREADS
    rt.ctx.enqueue_function[attn_gate_kernel](
        a_buf,
        g_buf,
        y_dev,
        Int32(n),
        grid_dim=grid,
        block_dim=GATE_THREADS,
    )
    return y_dev^


def _swiglu_buf_batch(
    mut rt: GpuRuntime,
    g_buf: DeviceBuffer[DType.float32],
    u_buf: DeviceBuffer[DType.float32],
    n: Int,
) raises -> DeviceBuffer[DType.float32]:
    var y_dev = create_zero_f32(rt, n)
    var grid = (n + GATE_THREADS - 1) // GATE_THREADS
    rt.ctx.enqueue_function[swiglu_kernel](
        g_buf,
        u_buf,
        y_dev,
        Int32(n),
        grid_dim=grid,
        block_dim=GATE_THREADS,
    )
    return y_dev^


def _residual_add_buf_batch(
    mut rt: GpuRuntime,
    a_buf: DeviceBuffer[DType.float32],
    b_buf: DeviceBuffer[DType.float32],
    n: Int,
) raises -> DeviceBuffer[DType.float32]:
    var y_dev = create_zero_f32(rt, n)
    var grid = (n + GATE_THREADS - 1) // GATE_THREADS
    rt.ctx.enqueue_function[residual_add_kernel](
        a_buf,
        b_buf,
        y_dev,
        Int32(n),
        grid_dim=grid,
        block_dim=GATE_THREADS,
    )
    return y_dev^


def _batch_ffn_width(cpu_model: GlimmerModel, layer: Int) -> Int:
    """The FFN intermediate width: the ffn_gate slot's rows
    (suffix 10), as gpu_layer._ffn_width reads it."""
    var idx = layer * LAYER_SUFFIX_COUNT + 10
    return cpu_model.layer_weights[idx].rows


# ---- public surface ------------------------------------------


def gpu_layer_step_batch_cfg(
    mut rt: GpuRuntime,
    weights: DeviceWeights,
    mut kv: DeviceKvCache,
    cpu_model: GlimmerModel,
    x: List[Float32],
    batch: Int,
    layer: Int,
    start_position: Int,
    tiled_min_batch: Int,
    attn_tile: Int,
    attn_max_splits: Int,
    attn_split_span: Int,
) raises -> List[Float32]:
    """A batch of prompt rows through one layer on device,
    activations RESIDENT across the layer's ops and the
    batched tiled split-KV attention (see the module
    header). Positions are start_position ..
    start_position + batch - 1. Statement for statement
    gpu_layer_step_cfg per element."""
    var cfg = cpu_model.cfg.copy()
    if layer < 0 or layer >= cfg.layers:
        raise Error(
            "gpu_layer_bad_shape: layer "
            + String(layer)
            + " outside 0.."
            + String(cfg.layers)
        )
    if batch <= 0 or len(x) != batch * cfg.hidden:
        raise Error(
            "gpu_layer_bad_shape: layer batch input has "
            + String(len(x))
            + " elements, expected batch*hidden "
            + String(batch * cfg.hidden)
        )
    if start_position < 0:
        raise Error(
            "gpu_layer_bad_shape: start position must be >= 0, got "
            + String(start_position)
        )
    var kind = cfg.kinds[layer].copy()
    var sliding = not kind.is_global
    if sliding and start_position + batch - 1 >= kind.window:
        raise Error(
            "gpu_kv_window_wrapped: layer "
            + String(layer)
            + " batch positions "
            + String(start_position)
            + ".."
            + String(start_position + batch - 1)
            + " would wrap its window "
            + String(kind.window)
            + "; the CPU path is the fallback beyond the window (Wave C limit)"
        )
    # The attention knobs gate BEFORE any KV append, so a
    # bad knob refuses without mutating the cache (the
    # resident step validates inside its attention launch,
    # after its single append; a batch appends batch rows,
    # so the batch step checks up front instead).
    attn_validate_tile(attn_tile)
    _ = attn_split_count(
        batch_attention_stored(start_position, batch - 1),
        attn_split_span,
        attn_max_splits,
    )
    _ = log_debug(
        "gpu.layer",
        "step_batch resident layer="
        + String(layer)
        + " start="
        + String(start_position)
        + " batch="
        + String(batch),
    )
    var qkv_w = cfg.kv_heads * cfg.head_dim
    var q_w = cfg.q_heads * cfg.head_dim
    var ffn_w = _batch_ffn_width(cpu_model, layer)
    var residual = upload_f32(rt, x)
    var h = _rms_norm_buf_batch(
        rt,
        residual,
        batch * cfg.hidden,
        _norm_buffer_batch(weights, cpu_model, layer, 0, cfg.hidden),
        cfg.hidden,
        cfg.rms_eps,
    )
    var q = _proj_layer_buf_batch(
        rt, weights, cpu_model, layer, 1, h, batch, tiled_min_batch
    )
    var k = _proj_layer_buf_batch(
        rt, weights, cpu_model, layer, 2, h, batch, tiled_min_batch
    )
    var v = _proj_layer_buf_batch(
        rt, weights, cpu_model, layer, 3, h, batch, tiled_min_batch
    )
    q = _rms_norm_buf_batch(
        rt,
        q,
        batch * q_w,
        _norm_buffer_batch(weights, cpu_model, layer, 4, cfg.head_dim),
        cfg.head_dim,
        cfg.rms_eps,
    )
    k = _rms_norm_buf_batch(
        rt,
        k,
        batch * qkv_w,
        _norm_buffer_batch(weights, cpu_model, layer, 5, cfg.head_dim),
        cfg.head_dim,
        cfg.rms_eps,
    )
    var q_for_attn = q.copy()
    if sliding:
        k = _rope_buf_batch(
            rt,
            k,
            batch * qkv_w,
            batch,
            cfg.kv_heads,
            cfg.head_dim,
            start_position,
            cfg.rope_base_swa,
        )
        q_for_attn = _rope_buf_batch(
            rt,
            q,
            batch * q_w,
            batch,
            cfg.q_heads,
            cfg.head_dim,
            start_position,
            cfg.rope_base_swa,
        )
    _kv_append_batch_device(rt, kv, layer, k, v, batch, start_position)
    var attn = _attention_batch_tiled_buf(
        rt,
        kv,
        layer,
        q_for_attn,
        batch,
        start_position,
        cfg.q_heads,
        cfg.kv_heads,
        cfg.head_dim,
        attn_tile,
        attn_max_splits,
        attn_split_span,
    )
    var gate = _proj_layer_buf_batch(
        rt, weights, cpu_model, layer, 7, h, batch, tiled_min_batch
    )
    var gated = _attn_gate_buf_batch(rt, attn, gate, batch * q_w)
    var out = _proj_layer_buf_batch(
        rt, weights, cpu_model, layer, 6, gated, batch, tiled_min_batch
    )
    out = _rms_norm_buf_batch(
        rt,
        out,
        batch * cfg.hidden,
        _norm_buffer_batch(weights, cpu_model, layer, 8, cfg.hidden),
        cfg.hidden,
        POST_NORM_EPS,
    )
    var residual2 = _residual_add_buf_batch(
        rt, residual, out, batch * cfg.hidden
    )
    var h2 = _rms_norm_buf_batch(
        rt,
        residual2,
        batch * cfg.hidden,
        _norm_buffer_batch(weights, cpu_model, layer, 9, cfg.hidden),
        cfg.hidden,
        cfg.rms_eps,
    )
    var gate_ffn = _proj_layer_buf_batch(
        rt, weights, cpu_model, layer, 10, h2, batch, tiled_min_batch
    )
    var up_ffn = _proj_layer_buf_batch(
        rt, weights, cpu_model, layer, 11, h2, batch, tiled_min_batch
    )
    var act = _swiglu_buf_batch(rt, gate_ffn, up_ffn, batch * ffn_w)
    var ffn = _proj_layer_buf_batch(
        rt, weights, cpu_model, layer, 12, act, batch, tiled_min_batch
    )
    ffn = _rms_norm_buf_batch(
        rt,
        ffn,
        batch * cfg.hidden,
        _norm_buffer_batch(weights, cpu_model, layer, 13, cfg.hidden),
        cfg.hidden,
        POST_NORM_EPS,
    )
    var result = _residual_add_buf_batch(rt, residual2, ffn, batch * cfg.hidden)
    return download_f32(rt, result, batch * cfg.hidden)


def gpu_layer_step_batch(
    mut rt: GpuRuntime,
    weights: DeviceWeights,
    mut kv: DeviceKvCache,
    cpu_model: GlimmerModel,
    x: List[Float32],
    batch: Int,
    layer: Int,
    start_position: Int,
    tiled_min_batch: Int,
) raises -> List[Float32]:
    """The batch step with the config-default attention
    knobs (the Wave D signature, kept for its existing
    callers); the body lives in gpu_layer_step_batch_cfg."""
    return gpu_layer_step_batch_cfg(
        rt,
        weights,
        kv,
        cpu_model,
        x,
        batch,
        layer,
        start_position,
        tiled_min_batch,
        DEFAULT_ATTN_TILE_POSITIONS,
        DEFAULT_ATTN_MAX_SPLITS,
        DEFAULT_ATTN_SPLIT_SPAN_POSITIONS,
    )
