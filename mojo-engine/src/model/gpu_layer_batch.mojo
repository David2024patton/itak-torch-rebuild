# gpu_layer_batch.mojo - the batched-prefill GPU layer
# assembly, Wave D Lane D1. The multi-row twin of
# gpu_layer.mojo's gpu_layer_step, statement for statement,
# over a flat batch of prompt rows:
#
#   gpu_layer_step_batch(rt, weights, kv, cpu_model, x_flat,
#       batch, layer, start_position, tiled_min_batch)
#
# x_flat is [batch][hidden] row-major; the return is the same
# shape. Batch element i is processed at absolute position
# start_position + i, exactly as gpu_layer_step would process
# it alone: same norms (the rms_norm launch already grids
# over rows, so a flat batch is one launch), same RoPE rule
# (sliding layers rotate K before the KV append and Q before
# attention, at each element's own position), same gate and
# SwiGLU assemblies (elementwise kernels over the flat batch),
# same host residual adds. The ONLY change of shape is the
# projections: where the single step launches the GEMV family
# once per row, the batch step launches the GEMM family
# (src/kernels/gemm.mojo) once per projection for the whole
# batch against the same resident weight buffer. Attention
# stays per position (its causal row count differs per
# element: element i attends to stored = start_position+i+1
# rows, appended for the whole batch before the first
# attention launch of the layer, which is exactly the set of
# rows the per-position steps would have appended by then).
#
# Shape/dtype discipline mirrors gpu_layer.mojo: every slot
# is cross-checked against the CPU model's slot before
# launch (gpu_layer_shape_mismatch), bad shapes refuse
# gpu_layer_bad_shape, and dtypes outside the GEMM family's
# set refuse gpu_layer_unsupported_dtype - that includes F16:
# Glimmer carries no F16 weight tensors (Wave A census) and
# the GEMM family builds only the shapes prefill dispatches
# (Q4_K, Q6_K, Q5_K, F32); an F16 projection slot in a batch
# refuses by name here while single rows keep the GEMV path.
#
# Diag seam: phase `gpu.layer`, as gpu_layer.mojo.

from std.math import sqrt

from diag import log_debug
from max.gpu.host import DeviceBuffer
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
from ops.residual import residual_add_cpu
from attention.attention import attention_kernel, ATTN_THREADS


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


def batch_row_place(mut x: List[Float32], i: Int, width: Int, row: List[Float32]):
    for j in range(width):
        x[i * width + j] = row[j]


# ---- device launch helpers -----------------------------------


def _rms_norm_batch(
    mut rt: GpuRuntime,
    x: List[Float32],
    w_buf: DeviceBuffer[DType.float32],
    row_len: Int,
    eps: Float32,
) raises -> List[Float32]:
    """rms_norm_kernel over a flat batch (grid = total rows);
    the norm weight is a RESIDENT device buffer. Mirrors
    gpu_layer._rms_norm_device, which is one row-set of this."""
    gpu_layer_check_norm(len(x), row_len, row_len)
    var rows = len(x) // row_len
    var x_dev = upload_f32(rt, x)
    var y_dev = create_zero_f32(rt, len(x))
    rt.ctx.enqueue_function[rms_norm_kernel](
        x_dev, w_buf, y_dev, Int32(row_len), eps,
        grid_dim=rows, block_dim=RMS_THREADS,
    )
    rt.ctx.synchronize()
    return download_f32(rt, y_dev, len(x))


def _gemm_u8_batch(
    mut rt: GpuRuntime,
    view: DeviceSlotU8,
    x: List[Float32],
    batch: Int,
    tiled_min_batch: Int,
) raises -> List[Float32]:
    """One GEMM launch for a byte-stored slot over the whole
    batch; tiled structure at or above tiled_min_batch."""
    gpu_layer_check_proj_batch(
        view.name, view.type_id, view.rows, view.cols, batch, len(x)
    )
    var x_dev = upload_f32(rt, x)
    var y_dev = create_zero_f32(rt, batch * view.rows)
    var tiled = batch >= tiled_min_batch
    if view.type_id == DT_Q4_K:
        if tiled:
            rt.ctx.enqueue_function[k1_gemm_tiled_kernel](
                view.buffer, x_dev, y_dev, Int32(view.cols), Int32(batch),
                Int32(view.rows),
                grid_dim=gemm_k1_tiled_grid(view.rows, batch),
                block_dim=K1_THREADS,
            )
        else:
            rt.ctx.enqueue_function[k1_gemm_kernel](
                view.buffer, x_dev, y_dev, Int32(view.cols), Int32(batch),
                Int32(view.rows),
                grid_dim=gemm_k1_grid(view.rows, batch),
                block_dim=K1_THREADS,
            )
    elif view.type_id == DT_Q6_K:
        if tiled:
            rt.ctx.enqueue_function[q6k_gemm_tiled_kernel](
                view.buffer, x_dev, y_dev, Int32(view.cols), Int32(batch),
                Int32(view.rows),
                grid_dim=gemm_tiled_grid(view.rows, batch),
                block_dim=Q6_THREADS,
            )
        else:
            rt.ctx.enqueue_function[q6k_gemm_kernel](
                view.buffer, x_dev, y_dev, Int32(view.cols), Int32(batch),
                Int32(view.rows),
                grid_dim=gemm_plain_grid(view.rows, batch),
                block_dim=Q6_THREADS,
            )
    elif view.type_id == DT_Q5_K:
        if tiled:
            rt.ctx.enqueue_function[q5k_gemm_tiled_kernel](
                view.buffer, x_dev, y_dev, Int32(view.cols), Int32(batch),
                Int32(view.rows),
                grid_dim=gemm_tiled_grid(view.rows, batch),
                block_dim=Q6_THREADS,
            )
        else:
            rt.ctx.enqueue_function[q5k_gemm_kernel](
                view.buffer, x_dev, y_dev, Int32(view.cols), Int32(batch),
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
    rt.ctx.synchronize()
    _ = log_debug(
        "gpu.layer",
        "gemm " + view.name + " batch=" + String(batch)
        + " tiled=" + String(tiled),
    )
    return download_f32(rt, y_dev, batch * view.rows)


def _gemm_f32_batch(
    mut rt: GpuRuntime,
    view: DeviceSlotF32,
    x: List[Float32],
    batch: Int,
    tiled_min_batch: Int,
) raises -> List[Float32]:
    """One GEMM launch for an F32 slot's resident buffer."""
    gpu_layer_check_proj_batch(
        view.name, view.type_id, view.rows, view.cols, batch, len(x)
    )
    var x_dev = upload_f32(rt, x)
    var y_dev = create_zero_f32(rt, batch * view.rows)
    if batch >= tiled_min_batch:
        rt.ctx.enqueue_function[f32_gemm_tiled_kernel](
            view.buffer, x_dev, y_dev, Int32(view.cols), Int32(batch),
            Int32(view.rows),
            grid_dim=gemm_tiled_grid(view.rows, batch),
            block_dim=F_THREADS,
        )
    else:
        rt.ctx.enqueue_function[f32_gemm_kernel](
            view.buffer, x_dev, y_dev, Int32(view.cols), Int32(batch),
            Int32(view.rows),
            grid_dim=gemm_plain_grid(view.rows, batch),
            block_dim=F_THREADS,
        )
    rt.ctx.synchronize()
    _ = log_debug("gpu.layer", "gemm f32 " + view.name)
    return download_f32(rt, y_dev, batch * view.rows)


def _check_view_batch(
    name: String, type_id: Int, rows: Int, cols: Int,
    cpu_type: Int, cpu_rows: Int, cpu_cols: Int,
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


def _proj_layer_batch(
    mut rt: GpuRuntime,
    weights: DeviceWeights,
    cpu_model: GlimmerModel,
    layer: Int,
    suffix_index: Int,
    x: List[Float32],
    batch: Int,
    tiled_min_batch: Int,
) raises -> List[Float32]:
    """One per-layer projection over the whole batch against
    the resident buffer, cross-checked like _proj_layer."""
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
            name, fview.type_id, fview.rows, fview.cols, cpu_type, cpu_rows,
            cpu_cols,
        )
        return _gemm_f32_batch(rt, fview, x, batch, tiled_min_batch)
    var view = weights.slot_u8(layer, suffix_index)
    _check_view_batch(
        name, view.type_id, view.rows, view.cols, cpu_type, cpu_rows, cpu_cols
    )
    return _gemm_u8_batch(rt, view, x, batch, tiled_min_batch)


def _rope_batch(
    mut rt: GpuRuntime,
    x: List[Float32],
    batch: Int,
    vectors_per_batch: Int,
    head_dim: Int,
    start_position: Int,
    base: Float32,
) raises -> List[Float32]:
    """rope_kernel over a flat [batch][vectors][head_dim]
    batch; the positions buffer carries each element's own
    absolute position (start_position + batch index)."""
    var vectors = batch * vectors_per_batch
    if len(x) != vectors * head_dim:
        raise Error(
            "gpu_layer_bad_shape: rope batch input has "
            + String(len(x))
            + " elements, expected "
            + String(vectors * head_dim)
        )
    var shape = rope_launch_shape(vectors * (head_dim // 2))
    var x_dev = upload_f32(rt, x)
    var y_dev = create_zero_f32(rt, len(x))
    var pos_host = rt.ctx.enqueue_create_host_buffer[DType.int32](vectors)
    for v in range(vectors):
        pos_host[v] = Int32(start_position + v // vectors_per_batch)
    var pos_dev = rt.ctx.enqueue_create_buffer[DType.int32](vectors)
    rt.ctx.enqueue_copy(src_buf=pos_host, dst_buf=pos_dev)
    rt.ctx.enqueue_function[rope_kernel](
        x_dev, y_dev, pos_dev, Int32(head_dim), base,
        grid_dim=shape[0], block_dim=shape[1],
    )
    rt.ctx.synchronize()
    return download_f32(rt, y_dev, len(x))


def _attention_at(
    mut rt: GpuRuntime,
    kv: DeviceKvCache,
    q: List[Float32],
    layer: Int,
    q_heads: Int,
    kv_heads: Int,
    head_dim: Int,
    stored: Int,
) raises -> List[Float32]:
    """attention_kernel on the DeviceKv buffers for ONE
    position whose causal row count is `stored` (the batch
    twin of gpu_layer._attention_device, which passes the
    cache's full stored length for its single position)."""
    if stored <= 0:
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
            + "; the direct-buffer attention launch covers the"
            " pre-wrap layout only (Wave C limit)"
        )
    if len(q) != q_heads * head_dim:
        raise Error(
            "gpu_layer_bad_shape: attention Q row has "
            + String(len(q))
            + " elements, expected "
            + String(q_heads * head_dim)
        )
    var q_dev = upload_f32(rt, q)
    var y_dev = create_zero_f32(rt, q_heads * head_dim)
    var k_buf = kv.key_buffer(layer)
    var v_buf = kv.value_buffer(layer)
    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    rt.ctx.enqueue_function[attention_kernel](
        q_dev, k_buf, v_buf, y_dev,
        Int32(stored), Int32(kv_heads), Int32(head_dim),
        Int32(q_heads // kv_heads), scale,
        grid_dim=q_heads, block_dim=ATTN_THREADS,
    )
    rt.ctx.synchronize()
    return download_f32(rt, y_dev, q_heads * head_dim)


def _attn_gate_batch(
    mut rt: GpuRuntime, attn: List[Float32], gate: List[Float32]
) raises -> List[Float32]:
    if len(attn) != len(gate):
        raise Error(
            "gpu_layer_bad_shape: gate assembly rows differ: attn "
            + String(len(attn))
            + " gate "
            + String(len(gate))
        )
    var a_dev = upload_f32(rt, attn)
    var g_dev = upload_f32(rt, gate)
    var y_dev = create_zero_f32(rt, len(attn))
    var grid = (len(attn) + GATE_THREADS - 1) // GATE_THREADS
    rt.ctx.enqueue_function[attn_gate_kernel](
        a_dev, g_dev, y_dev, Int32(len(attn)), grid_dim=grid,
        block_dim=GATE_THREADS,
    )
    rt.ctx.synchronize()
    return download_f32(rt, y_dev, len(attn))


def _swiglu_batch(
    mut rt: GpuRuntime, gate: List[Float32], up: List[Float32]
) raises -> List[Float32]:
    if len(gate) != len(up):
        raise Error(
            "gpu_layer_bad_shape: swiglu rows differ: gate "
            + String(len(gate))
            + " up "
            + String(len(up))
        )
    var g_dev = upload_f32(rt, gate)
    var u_dev = upload_f32(rt, up)
    var y_dev = create_zero_f32(rt, len(gate))
    var grid = (len(gate) + GATE_THREADS - 1) // GATE_THREADS
    rt.ctx.enqueue_function[swiglu_kernel](
        g_dev, u_dev, y_dev, Int32(len(gate)), grid_dim=grid,
        block_dim=GATE_THREADS,
    )
    rt.ctx.synchronize()
    return download_f32(rt, y_dev, len(gate))


# ---- public surface ------------------------------------------


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
    """A batch of prompt rows through one layer on device,
    statement for statement gpu_layer_step per element (see
    the module header). Positions are start_position ..
    start_position + batch - 1."""
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
            + "; the CPU path is the fallback beyond the window"
            " (Wave C limit)"
        )
    _ = log_debug(
        "gpu.layer",
        "step_batch layer="
        + String(layer)
        + " start="
        + String(start_position)
        + " batch="
        + String(batch),
    )
    var residual = x.copy()
    var h = _rms_norm_batch(
        rt, residual,
        _norm_buffer_batch(weights, cpu_model, layer, 0, cfg.hidden),
        cfg.hidden, cfg.rms_eps,
    )
    var q = _proj_layer_batch(rt, weights, cpu_model, layer, 1, h, batch, tiled_min_batch)
    var k = _proj_layer_batch(rt, weights, cpu_model, layer, 2, h, batch, tiled_min_batch)
    var v = _proj_layer_batch(rt, weights, cpu_model, layer, 3, h, batch, tiled_min_batch)
    q = _rms_norm_batch(
        rt, q,
        _norm_buffer_batch(weights, cpu_model, layer, 4, cfg.head_dim),
        cfg.head_dim, cfg.rms_eps,
    )
    k = _rms_norm_batch(
        rt, k,
        _norm_buffer_batch(weights, cpu_model, layer, 5, cfg.head_dim),
        cfg.head_dim, cfg.rms_eps,
    )
    var q_for_attn = q.copy()
    if sliding:
        k = _rope_batch(
            rt, k, batch, cfg.kv_heads, cfg.head_dim, start_position,
            cfg.rope_base_swa,
        )
        q_for_attn = _rope_batch(
            rt, q, batch, cfg.q_heads, cfg.head_dim, start_position,
            cfg.rope_base_swa,
        )
    var k_width = cfg.kv_heads * cfg.head_dim
    for i in range(batch):
        kv.append(
            rt, layer,
            batch_row_slice(k, i, k_width),
            batch_row_slice(v, i, k_width),
            start_position + i,
        )
    var attn_width = cfg.q_heads * cfg.head_dim
    var attn = List[Float32](capacity=batch * attn_width)
    attn.resize(batch * attn_width, Float32(0.0))
    for i in range(batch):
        var row = _attention_at(
            rt, kv, batch_row_slice(q_for_attn, i, attn_width), layer,
            cfg.q_heads, cfg.kv_heads, cfg.head_dim,
            start_position + i + 1,
        )
        batch_row_place(attn, i, attn_width, row)
    var gate = _proj_layer_batch(rt, weights, cpu_model, layer, 7, h, batch, tiled_min_batch)
    var gated = _attn_gate_batch(rt, attn, gate)
    var out = _proj_layer_batch(rt, weights, cpu_model, layer, 6, gated, batch, tiled_min_batch)
    out = _rms_norm_batch(
        rt, out,
        _norm_buffer_batch(weights, cpu_model, layer, 8, cfg.hidden),
        cfg.hidden, POST_NORM_EPS,
    )
    var after_attn = residual_add_cpu(residual, out)
    var residual2 = after_attn.copy()
    var h2 = _rms_norm_batch(
        rt, residual2,
        _norm_buffer_batch(weights, cpu_model, layer, 9, cfg.hidden),
        cfg.hidden, cfg.rms_eps,
    )
    var gate_ffn = _proj_layer_batch(rt, weights, cpu_model, layer, 10, h2, batch, tiled_min_batch)
    var up_ffn = _proj_layer_batch(rt, weights, cpu_model, layer, 11, h2, batch, tiled_min_batch)
    var act = _swiglu_batch(rt, gate_ffn, up_ffn)
    var ffn = _proj_layer_batch(rt, weights, cpu_model, layer, 12, act, batch, tiled_min_batch)
    ffn = _rms_norm_batch(
        rt, ffn,
        _norm_buffer_batch(weights, cpu_model, layer, 13, cfg.hidden),
        cfg.hidden, POST_NORM_EPS,
    )
    return residual_add_cpu(residual2, ffn)
