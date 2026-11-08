# gpu_layer.mojo - the GPU layer assembly, Lane C2 (Wave C),
# activation residency added by Lane D4 (Wave D).
#
# Host-side launch wrappers and the device layer step, per
# WAVE_C_INTERFACES.md. Public surface:
#   gpu_logits(rt, weights, cpu_model, hidden) -> List[Float32]
#   gpu_layer_step(rt, weights, kv, cpu_model, x, layer,
#                  position) -> List[Float32]
#   gpu_layer_step_cfg(same args, attn_tile, attn_max_splits,
#     attn_span) - the device-resident chain with explicit
#     Wave D attention knobs; gpu_layer_step delegates to it
#     with the config defaults, and
#   gpu_layer_step_resident(...) - same signature minus the
#     knobs, the config-default resident form (the D4 proof's
#     entry point, delegating to gpu_layer_step_cfg).
#   gpu_layer_step_hostchain(...) - the original Wave C
#     per-op host-chained step, kept as the A/B baseline the
#     residency proof measures against (and the only path
#     that still refuses past the KV window, because it
#     launches attention on the raw KV buffers).
#   attention_device_resident(rt, kv, q_buf, layer, q_heads,
#     kv_heads, head_dim) -> DeviceBuffer - the resident
#     attention launch over the visible-order KV gather, with
#     the config-default attention knobs;
#   attention_device_resident_cfg(same args, tile, max_splits,
#     span) - the same launch with explicit knobs: the D2
#     tiled split-KV pair (attention_tiled_kernel +
#     attention_merge_kernel) over the gathered buffers.
# (The contract's gpu_logits shorthand omits cpu_model; the
# CPU model is a parameter here because the task fixes the
# final-logit scale/softcap as "from the CPU model's config
# values, not literals", and the CPU model is the holder of
# those values and the cross-check for every slot's shape.
# Lane C3 holds the CPU model for config/fallback anyway.)
#
# Port shape (Wave C contract): weights and KV are resident
# on device; activations are host List[Float32] rows at the
# layer boundary. Wave C's step then crossed per op: each op
# uploaded its input row(s), launched, and downloaded its
# output row, and the residual adds were HOST adds on the
# downloaded rows.
#
# RESIDENCY (Wave D, Lane D4): gpu_layer_step now keeps the
# layer's activations ON DEVICE across the whole layer - the
# residual stream, norm outputs, projection outputs, the
# attention output, and the MLP intermediates are device
# buffers passed op to op, the KV append is device-to-device
# (DeviceKvCache.append_device), and the residual adds are
# device adds (residual_add_kernel). The host sees exactly
# two crossings per layer: the input row upload and the
# output row download. gpu_logits chains the same way
# (upload the hidden row once, download the logits once).
# Remaining host crossings in the resident path, stated
# plainly: (1) the layer boundary itself, one upload + one
# download of the hidden row per layer - the contract's
# boundary shape, and the seam the model-level chain crosses
# 52 times per step; (2) inside attention past a KV wrap,
# the visible-order gather is a device-to-device copy, not a
# host crossing. No other activation leaves the card inside
# RECONCILIATION (Wave D, D2+D4): the resident chain's
# attention launch is D2's TILED SPLIT-KV pair, not the
# Wave B attention_kernel: attention_device_resident_cfg
# gathers the layer's visible rows (DeviceKvCache.
# gather_visible: [stored][kv_heads][head_dim] flat, rows in
# absolute-position order, PRE-ROTATED K - exactly the
# layout D2's kernels consume, the same layout the Wave B
# kernel and D2's standalone proof consume), then launches
# attention_tiled_kernel + attention_merge_kernel over the
# gathered buffers. Tile / max splits / split span are
# parameters threaded from EngineConfig (src/config.mojo);
# the split chooser keeps splits at 1 at or below one span.
# The hostchain A/B baseline below is NOT retargeted: it
# still launches the Wave B attention_kernel exactly as in
# Wave C. Statement for statement the CPU
# GlimmerModel.layer_step / .logits (model.mojo), with every
# projection and op launched on device:
#   rms_norm_kernel (grid = row count, width = row length;
#     per-head Q/K norms launch grid = head count,
#     width = head_dim, exactly as gpu_proof_ops proved the
#     launcher), GEMV dispatch by slot dtype (k1 grid
#     rows//ROWS_PER_WG block THREADS from k1_gemv.mojo;
#     q6k/q5k/f32/f16 grid rows block 256), rope_kernel,
#     attention_kernel on the DeviceKv buffers, swiglu_kernel,
#     final_logit_kernel, and one small new elementwise
#     kernel in this file for the sigmoid gate assembly
#     (attn[i] * sigmoid(gate[i])).
#
# RoPE law (Wave B, kept): sliding layers rotate the K row
# BEFORE the DeviceKv append (write-time rotation; the cache
# holds rotated K) and rotate the Q row before the attention
# launch, because attention_kernel's contract is PRE-ROTATED
# Q and K. Global layers never rotate. The rotation base is
# the CPU model's rope_base_swa config value (500,000 on
# Glimmer, where the file carries no separate SWA key).
#
# Residual adds: Wave C did them as HOST adds
# (residual_add_cpu) on the downloaded rows. Wave D's
# resident step does them ON DEVICE with
# residual_add_kernel - the same plain f32 per-element add,
# so the arithmetic is identical; the hostchain baseline
# below keeps the host adds, which is exactly the A/B the
# residency proof measures. Stated per the contract's
# "say which, and why".
#
# Norm weights are RESIDENT: C1's DeviceWeights serves them
# as float32 buffers (slot_f32 for the per-layer norms,
# output_norm() for the final norm), so no norm is uploaded
# per call. Every device slot's (type_id, rows, cols) is
# cross-checked against the CPU model's slot before launch;
# a disagreement refuses gpu_layer_shape_mismatch.
#
# Captures: the CPU model's captures (embed, named-layer
# in/out) are the isolation tool for divergence work, per
# the contract; this GPU step adds no capture hooks of its
# own.
#
# C1 surface used (gpu package, Lane C1 as built):
#   gpu.context: GpuRuntime (its .ctx DeviceContext is the
#     launch surface), upload_f32 / download_f32 /
#     create_zero_f32.
#   gpu.weights: DeviceWeights.slot_u8 / slot_f32 /
#     output_u8 / output_f32 (views carrying buffer, name,
#     type_id, rows, cols) and .output_norm().
#   gpu.device_kv: DeviceKvCache.append(rt, layer, k, v,
#     position), .key_buffer(layer), .value_buffer(layer),
#     .stored_len(layer), .first_visible_position(layer).
#
# PROFILE timing (Wave E, Lane E2): with MOJO_ENGINE_DEBUG
# on, gpu_layer_step_cfg prints one PROFILE_LAYER line per
# layer call and gpu_logits one PROFILE_LOGITS line, each
# carrying the call's wall time in microseconds (the closing
# download synchronizes the queue, so the interval covers
# every launch of the call). Off by default; when off there
# is no behavior change beyond the seam's own env check.
#
# Named refusals: gpu_layer_bad_shape,
# gpu_layer_unsupported_dtype, gpu_layer_shape_mismatch,
# gpu_kv_window_wrapped (windowed append beyond the window,
# and any attention launch whose first visible position is
# not 0 - the Wave C direct-buffer limit).

from std.math import sqrt, exp
from std.time import perf_counter_ns

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceBuffer

from diag import log_debug, debug_enabled
from gpu.context import (
    GpuRuntime,
    upload_f32,
    download_f32,
    create_zero_f32,
)
from gpu.weights import DeviceWeights, DeviceSlotU8, DeviceSlotF32
from gpu.device_kv import DeviceKvCache
from model.model import (
    GlimmerModel,
    POST_NORM_EPS,
    DT_F32,
    DT_F16,
    DT_Q4_K,
    DT_Q5_K,
    DT_Q6_K,
    LAYER_SUFFIX_COUNT,
)
from kernels.k1_gemv import (
    k1_gemv_kernel,
    THREADS as K1_THREADS,
    ROWS_PER_WG as K1_ROWS_PER_WG,
)
from kernels.q6k_gemv import q6k_gemv_kernel, THREADS as Q6_THREADS
from kernels.q5k_gemv import (
    q5k_gemv_kernel,
    THREADS as Q5_THREADS,
    ROWS_PER_WG as Q5_ROWS_PER_WG,
)
from kernels.f32_gemv import (
    f32_gemv_kernel,
    f16_gemv_kernel,
    THREADS as F_THREADS,
    F32_THREADS,
    ROWS_PER_WG as F_ROWS_PER_WG,
)
from ops.rms_norm import rms_norm_kernel, RMS_THREADS
from ops.rope import rope_kernel
from ops.swiglu import swiglu_kernel
from ops.residual import residual_add_cpu, residual_add_kernel
from ops.final_logit import final_logit_kernel
from attention.attention import (
    ATTN_THREADS,
    attention_kernel,
    attention_merge_kernel,
    attention_tiled_direct_kernel,
    attention_tiled_kernel,
    attn_split_count,
    attn_validate_tile,
)
from config import (
    DEFAULT_ATTN_MAX_SPLITS,
    DEFAULT_ATTN_SPLIT_SPAN_POSITIONS,
    DEFAULT_ATTN_TILE_POSITIONS,
)

comptime GATE_THREADS = 256


# ---- pure shape helpers (VM-testable, no device) -------------


def gpu_layer_check_proj(
    name: String, type_id: Int, rows: Int, cols: Int, x_len: Int
) raises:
    """Validate one projection launch against its slot shape
    and the kernel family's launch rules, by name."""
    if rows <= 0 or cols <= 0:
        raise Error(
            "gpu_layer_bad_shape: "
            + name
            + " has non-positive shape "
            + String(rows)
            + "x"
            + String(cols)
        )
    if x_len != cols:
        raise Error(
            "gpu_layer_bad_shape: "
            + name
            + " expects an input of "
            + String(cols)
            + " elements, got "
            + String(x_len)
        )
    if (
        type_id != DT_Q4_K
        and type_id != DT_Q6_K
        and type_id != DT_Q5_K
        and type_id != DT_F32
        and type_id != DT_F16
    ):
        raise Error(
            "gpu_layer_unsupported_dtype: "
            + name
            + " has ggml type id "
            + String(type_id)
        )
    if cols % 256 != 0:
        raise Error(
            "gpu_layer_bad_shape: "
            + name
            + " cols "
            + String(cols)
            + " is not a multiple of the 256-element kernel step"
        )
    if type_id == DT_Q4_K and rows % K1_ROWS_PER_WG != 0:
        raise Error(
            "gpu_layer_bad_shape: "
            + name
            + " rows "
            + String(rows)
            + " is not a multiple of the K1 workgroup tile "
            + String(K1_ROWS_PER_WG)
        )
    if type_id == DT_Q5_K and rows % Q5_ROWS_PER_WG != 0:
        raise Error(
            "gpu_layer_bad_shape: "
            + name
            + " rows "
            + String(rows)
            + " is not a multiple of the Q5_K workgroup tile "
            + String(Q5_ROWS_PER_WG)
        )
    if type_id == DT_F16 and rows % F_ROWS_PER_WG != 0:
        raise Error(
            "gpu_layer_bad_shape: "
            + name
            + " rows "
            + String(rows)
            + " is not a multiple of the plain-float workgroup tile "
            + String(F_ROWS_PER_WG)
        )


def gpu_layer_check_norm(x_len: Int, w_len: Int, row_len: Int) raises:
    """Validate one rms_norm launch: whole rows, weight width
    equal to the row length."""
    if row_len <= 0 or x_len <= 0 or x_len % row_len != 0:
        raise Error(
            "gpu_layer_bad_shape: norm over "
            + String(x_len)
            + " elements with row length "
            + String(row_len)
            + " is not a whole number of rows"
        )
    if w_len != row_len:
        raise Error(
            "gpu_layer_bad_shape: norm weight has "
            + String(w_len)
            + " elements, row length is "
            + String(row_len)
        )


def rope_launch_shape(pairs: Int) raises -> Tuple[Int, Int]:
    """(grid, block) for rope_kernel, which has NO bounds
    guard: the launch must cover exactly `pairs` threads."""
    if pairs <= 0:
        raise Error(
            "gpu_layer_bad_shape: rope pair count must be positive, got "
            + String(pairs)
        )
    if pairs <= 1024:
        return (1, pairs)
    if pairs % 256 == 0:
        return (pairs // 256, 256)
    if pairs % 128 == 0:
        return (pairs // 128, 128)
    if pairs % 64 == 0:
        return (pairs // 64, 64)
    raise Error(
        "gpu_layer_bad_shape: rope pair count "
        + String(pairs)
        + " cannot be covered exactly by a guarded-free launch"
    )


# ---- the sigmoid gate assembly -------------------------------


def _sigmoid(v: Float32) -> Float32:
    """Branch-stable sigmoid, model.mojo's shape."""
    if v >= Float32(0.0):
        return Float32(1.0) / (Float32(1.0) + exp(-v))
    var e = exp(v)
    return e / (Float32(1.0) + e)


def attn_gate_kernel(
    a: Pointer[Float32, MutAnyOrigin],
    g: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    n: Int32,
):
    """GPU gate assembly: y[i] = a[i] * sigmoid(g[i]).
    One thread per element, bounds-guarded."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        y[unsafe_offset=i] = a[unsafe_offset=i] * _sigmoid(g[unsafe_offset=i])


def attn_gate_cpu(a: List[Float32], g: List[Float32]) raises -> List[Float32]:
    """Host twin of attn_gate_kernel (VM tests + parity)."""
    if len(a) != len(g):
        raise Error(
            "gpu_layer_bad_shape: gate assembly rows differ: attn "
            + String(len(a))
            + " gate "
            + String(len(g))
        )
    var out = List[Float32](capacity=len(a))
    out.resize(len(a), Float32(0.0))
    for i in range(len(a)):
        out[i] = a[i] * _sigmoid(g[i])
    return out^


# ---- device launch helpers -----------------------------------


def _rms_norm_device(
    mut rt: GpuRuntime,
    x: List[Float32],
    w_buf: DeviceBuffer[DType.float32],
    row_len: Int,
    eps: Float32,
) raises -> List[Float32]:
    """rms_norm_kernel over x as rows of row_len (grid = row
    count); the norm weight is a RESIDENT device buffer."""
    gpu_layer_check_norm(len(x), row_len, row_len)
    var rows = len(x) // row_len
    var x_dev = upload_f32(rt, x)
    var y_dev = create_zero_f32(rt, len(x))
    rt.ctx.enqueue_function[rms_norm_kernel](
        x_dev,
        w_buf,
        y_dev,
        Int32(row_len),
        eps,
        grid_dim=rows,
        block_dim=RMS_THREADS,
    )
    rt.ctx.synchronize()
    return download_f32(rt, y_dev, len(x))


def _check_view(
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


def _gemv_u8_device(
    mut rt: GpuRuntime, view: DeviceSlotU8, x: List[Float32]
) raises -> List[Float32]:
    """One GEMV launch for a byte-stored slot (Q4_K / Q6_K /
    Q5_K / F16), dispatched by the view's dtype."""
    gpu_layer_check_proj(view.name, view.type_id, view.rows, view.cols, len(x))
    var x_dev = upload_f32(rt, x)
    var y_dev = create_zero_f32(rt, view.rows)
    if view.type_id == DT_Q4_K:
        rt.ctx.enqueue_function[k1_gemv_kernel](
            view.buffer,
            x_dev,
            y_dev,
            Int32(view.cols),
            grid_dim=view.rows // K1_ROWS_PER_WG,
            block_dim=K1_THREADS,
        )
    elif view.type_id == DT_Q6_K:
        rt.ctx.enqueue_function[q6k_gemv_kernel](
            view.buffer,
            x_dev,
            y_dev,
            Int32(view.cols),
            grid_dim=view.rows,
            block_dim=Q6_THREADS,
        )
    elif view.type_id == DT_Q5_K:
        rt.ctx.enqueue_function[q5k_gemv_kernel](
            view.buffer,
            x_dev,
            y_dev,
            Int32(view.cols),
            grid_dim=view.rows // Q5_ROWS_PER_WG,
            block_dim=Q5_THREADS,
        )
    elif view.type_id == DT_F16:
        rt.ctx.enqueue_function[f16_gemv_kernel](
            view.buffer,
            x_dev,
            y_dev,
            Int32(view.cols),
            grid_dim=view.rows // F_ROWS_PER_WG,
            block_dim=F_THREADS,
        )
    else:
        raise Error(
            "gpu_layer_unsupported_dtype: "
            + view.name
            + " reached the byte GEMV path with ggml type id "
            + String(view.type_id)
        )
    rt.ctx.synchronize()
    _ = log_debug(
        "gpu.layer", "gemv " + view.name + " rows=" + String(view.rows)
    )
    return download_f32(rt, y_dev, view.rows)


def _gemv_f32_device(
    mut rt: GpuRuntime, view: DeviceSlotF32, x: List[Float32]
) raises -> List[Float32]:
    """One GEMV launch for an F32 slot's resident buffer."""
    gpu_layer_check_proj(view.name, view.type_id, view.rows, view.cols, len(x))
    var x_dev = upload_f32(rt, x)
    var y_dev = create_zero_f32(rt, view.rows)
    rt.ctx.enqueue_function[f32_gemv_kernel](
        view.buffer,
        x_dev,
        y_dev,
        Int32(view.cols),
        grid_dim=view.rows,
        block_dim=F32_THREADS,
    )
    rt.ctx.synchronize()
    _ = log_debug("gpu.layer", "gemv f32 " + view.name)
    return download_f32(rt, y_dev, view.rows)


def _proj_layer(
    mut rt: GpuRuntime,
    weights: DeviceWeights,
    cpu_model: GlimmerModel,
    layer: Int,
    suffix_index: Int,
    x: List[Float32],
) raises -> List[Float32]:
    """One per-layer projection against the resident buffer,
    cross-checked against the CPU slot's shape and dtype."""
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
        _check_view(
            name,
            fview.type_id,
            fview.rows,
            fview.cols,
            cpu_type,
            cpu_rows,
            cpu_cols,
        )
        return _gemv_f32_device(rt, fview, x)
    var view = weights.slot_u8(layer, suffix_index)
    _check_view(
        name, view.type_id, view.rows, view.cols, cpu_type, cpu_rows, cpu_cols
    )
    return _gemv_u8_device(rt, view, x)


def _proj_output(
    mut rt: GpuRuntime,
    weights: DeviceWeights,
    cpu_model: GlimmerModel,
    x: List[Float32],
) raises -> List[Float32]:
    """The head (output) projection, same dispatch."""
    var cpu_type = cpu_model.output.type_id
    var cpu_rows = cpu_model.output.rows
    var cpu_cols = cpu_model.output.cols
    if cpu_type == DT_F32:
        var fview = weights.output_f32()
        _check_view(
            "output.weight",
            fview.type_id,
            fview.rows,
            fview.cols,
            cpu_type,
            cpu_rows,
            cpu_cols,
        )
        return _gemv_f32_device(rt, fview, x)
    var view = weights.output_u8()
    _check_view(
        "output.weight",
        view.type_id,
        view.rows,
        view.cols,
        cpu_type,
        cpu_rows,
        cpu_cols,
    )
    return _gemv_u8_device(rt, view, x)


def _norm_buffer(
    weights: DeviceWeights,
    cpu_model: GlimmerModel,
    layer: Int,
    suffix_index: Int,
    row_len: Int,
) raises -> DeviceBuffer[DType.float32]:
    """A per-layer norm's RESIDENT float32 buffer, validated
    against the CPU slot (an F32 slot of exactly row_len
    values)."""
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


def _rope_device(
    mut rt: GpuRuntime,
    x: List[Float32],
    vectors: Int,
    head_dim: Int,
    position: Int,
    base: Float32,
) raises -> List[Float32]:
    """rope_kernel over a flat [vectors][head_dim] row; the
    positions buffer is filled with this step's one absolute
    position, the CPU step's construction."""
    if len(x) != vectors * head_dim:
        raise Error(
            "gpu_layer_bad_shape: rope input has "
            + String(len(x))
            + " elements, expected "
            + String(vectors * head_dim)
        )
    var shape = rope_launch_shape(vectors * (head_dim // 2))
    var x_dev = upload_f32(rt, x)
    var y_dev = create_zero_f32(rt, len(x))
    var pos_host = rt.ctx.enqueue_create_host_buffer[DType.int32](vectors)
    for i in range(vectors):
        pos_host[i] = Int32(position)
    var pos_dev = rt.ctx.enqueue_create_buffer[DType.int32](vectors)
    rt.ctx.enqueue_copy(src_buf=pos_host, dst_buf=pos_dev)
    rt.ctx.enqueue_function[rope_kernel](
        x_dev,
        y_dev,
        pos_dev,
        Int32(head_dim),
        base,
        grid_dim=shape[0],
        block_dim=shape[1],
    )
    rt.ctx.synchronize()
    return download_f32(rt, y_dev, len(x))


def _attention_device(
    mut rt: GpuRuntime,
    kv: DeviceKvCache,
    q: List[Float32],
    layer: Int,
    q_heads: Int,
    kv_heads: Int,
    head_dim: Int,
) raises -> List[Float32]:
    """attention_kernel launched directly on the DeviceKv
    buffers with the PRE-ROTATED Q row (the kernel's
    contract). Direct-buffer launch is exact only while the
    visible rows start at buffer offset 0 (pre-wrap); any
    other first visible position refuses by the contract's
    window name."""
    var stored = kv.stored_len(layer)
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
        q_dev,
        k_buf,
        v_buf,
        y_dev,
        Int32(stored),
        Int32(kv_heads),
        Int32(head_dim),
        Int32(q_heads // kv_heads),
        scale,
        grid_dim=q_heads,
        block_dim=ATTN_THREADS,
    )
    rt.ctx.synchronize()
    return download_f32(rt, y_dev, q_heads * head_dim)


def _attn_gate_device(
    mut rt: GpuRuntime, attn: List[Float32], gate: List[Float32]
) raises -> List[Float32]:
    """attn_gate_kernel over the downloaded rows."""
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
        a_dev,
        g_dev,
        y_dev,
        Int32(len(attn)),
        grid_dim=grid,
        block_dim=GATE_THREADS,
    )
    rt.ctx.synchronize()
    return download_f32(rt, y_dev, len(attn))


def _swiglu_device(
    mut rt: GpuRuntime, gate: List[Float32], up: List[Float32]
) raises -> List[Float32]:
    """swiglu_kernel over the downloaded rows."""
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
        g_dev,
        u_dev,
        y_dev,
        Int32(len(gate)),
        grid_dim=grid,
        block_dim=GATE_THREADS,
    )
    rt.ctx.synchronize()
    return download_f32(rt, y_dev, len(gate))


# ---- device-resident launch helpers (Wave D) -----------------
#
# The same kernels as the host helpers above, chained on
# DeviceBuffers: no upload, no download, no per-op sync (the
# context's single queue orders the launches; the one sync
# happens in download_f32 at the layer boundary). Buffer
# lengths are carried host-side by the caller (n arguments),
# exactly as the host helpers carry List lengths.


def _alloc_f32(
    mut rt: GpuRuntime, n: Int
) raises -> DeviceBuffer[DType.float32]:
    """A raw (UNINITIALIZED) device float32 buffer of n
    elements, for resident-chain outputs that their kernel
    writes IN FULL before any read (Wave E, Lane E2: the
    create_zero_f32 zero-fill launch per op - about 21 per
    layer - was pure overhead on this chain). Full-write
    audit, kernel by kernel: rms_norm_kernel writes every
    row element; the GEMV family (k1/q6k/q5k/f32/f16) writes
    every output row its launch grid covers, and
    gpu_layer_check_proj refuses row counts the tile cannot
    cover exactly; rope_kernel's launch covers exactly the
    input's pairs (rope_launch_shape); attention_merge_kernel
    and the splits==1 direct kernel write every head/channel;
    attn_gate_kernel, swiglu_kernel, residual_add_kernel and
    final_logit_kernel write every element under their
    bounds guard. Any future kernel that ACCUMULATES into
    its output or writes partially must NOT take its output
    from here - it needs create_zero_f32."""
    if n <= 0:
        raise Error(
            "gpu_layer_bad_shape: raw alloc length must be positive, got "
            + String(n)
        )
    return rt.ctx.enqueue_create_buffer[DType.float32](n)


def _rms_norm_buf(
    mut rt: GpuRuntime,
    x_buf: DeviceBuffer[DType.float32],
    n: Int,
    w_buf: DeviceBuffer[DType.float32],
    row_len: Int,
    eps: Float32,
) raises -> DeviceBuffer[DType.float32]:
    gpu_layer_check_norm(n, row_len, row_len)
    var rows = n // row_len
    var y_dev = _alloc_f32(rt, n)
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


def _gemv_u8_buf(
    mut rt: GpuRuntime,
    view: DeviceSlotU8,
    x_buf: DeviceBuffer[DType.float32],
    x_len: Int,
) raises -> DeviceBuffer[DType.float32]:
    gpu_layer_check_proj(view.name, view.type_id, view.rows, view.cols, x_len)
    var y_dev = _alloc_f32(rt, view.rows)
    if view.type_id == DT_Q4_K:
        rt.ctx.enqueue_function[k1_gemv_kernel](
            view.buffer,
            x_buf,
            y_dev,
            Int32(view.cols),
            grid_dim=view.rows // K1_ROWS_PER_WG,
            block_dim=K1_THREADS,
        )
    elif view.type_id == DT_Q6_K:
        rt.ctx.enqueue_function[q6k_gemv_kernel](
            view.buffer,
            x_buf,
            y_dev,
            Int32(view.cols),
            grid_dim=view.rows,
            block_dim=Q6_THREADS,
        )
    elif view.type_id == DT_Q5_K:
        rt.ctx.enqueue_function[q5k_gemv_kernel](
            view.buffer,
            x_buf,
            y_dev,
            Int32(view.cols),
            grid_dim=view.rows // Q5_ROWS_PER_WG,
            block_dim=Q5_THREADS,
        )
    elif view.type_id == DT_F16:
        rt.ctx.enqueue_function[f16_gemv_kernel](
            view.buffer,
            x_buf,
            y_dev,
            Int32(view.cols),
            grid_dim=view.rows // F_ROWS_PER_WG,
            block_dim=F_THREADS,
        )
    else:
        raise Error(
            "gpu_layer_unsupported_dtype: "
            + view.name
            + " reached the byte GEMV path with ggml type id "
            + String(view.type_id)
        )
    _ = log_debug("gpu.layer", "gemv resident " + view.name)
    return y_dev^


def _gemv_f32_buf(
    mut rt: GpuRuntime,
    view: DeviceSlotF32,
    x_buf: DeviceBuffer[DType.float32],
    x_len: Int,
) raises -> DeviceBuffer[DType.float32]:
    gpu_layer_check_proj(view.name, view.type_id, view.rows, view.cols, x_len)
    var y_dev = _alloc_f32(rt, view.rows)
    rt.ctx.enqueue_function[f32_gemv_kernel](
        view.buffer,
        x_buf,
        y_dev,
        Int32(view.cols),
        grid_dim=view.rows,
        block_dim=F32_THREADS,
    )
    return y_dev^


def _proj_layer_buf(
    mut rt: GpuRuntime,
    weights: DeviceWeights,
    cpu_model: GlimmerModel,
    layer: Int,
    suffix_index: Int,
    x_buf: DeviceBuffer[DType.float32],
    x_len: Int,
) raises -> DeviceBuffer[DType.float32]:
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
        _check_view(
            name,
            fview.type_id,
            fview.rows,
            fview.cols,
            cpu_type,
            cpu_rows,
            cpu_cols,
        )
        return _gemv_f32_buf(rt, fview, x_buf, x_len)
    var view = weights.slot_u8(layer, suffix_index)
    _check_view(
        name, view.type_id, view.rows, view.cols, cpu_type, cpu_rows, cpu_cols
    )
    return _gemv_u8_buf(rt, view, x_buf, x_len)


def _proj_output_buf(
    mut rt: GpuRuntime,
    weights: DeviceWeights,
    cpu_model: GlimmerModel,
    x_buf: DeviceBuffer[DType.float32],
    x_len: Int,
) raises -> DeviceBuffer[DType.float32]:
    var cpu_type = cpu_model.output.type_id
    var cpu_rows = cpu_model.output.rows
    var cpu_cols = cpu_model.output.cols
    if cpu_type == DT_F32:
        var fview = weights.output_f32()
        _check_view(
            "output.weight",
            fview.type_id,
            fview.rows,
            fview.cols,
            cpu_type,
            cpu_rows,
            cpu_cols,
        )
        return _gemv_f32_buf(rt, fview, x_buf, x_len)
    var view = weights.output_u8()
    _check_view(
        "output.weight",
        view.type_id,
        view.rows,
        view.cols,
        cpu_type,
        cpu_rows,
        cpu_cols,
    )
    return _gemv_u8_buf(rt, view, x_buf, x_len)


def _rope_buf(
    mut rt: GpuRuntime,
    x_buf: DeviceBuffer[DType.float32],
    n: Int,
    vectors: Int,
    head_dim: Int,
    position: Int,
    base: Float32,
) raises -> DeviceBuffer[DType.float32]:
    if n != vectors * head_dim:
        raise Error(
            "gpu_layer_bad_shape: rope input has "
            + String(n)
            + " elements, expected "
            + String(vectors * head_dim)
        )
    var shape = rope_launch_shape(vectors * (head_dim // 2))
    var y_dev = _alloc_f32(rt, n)
    var pos_host = rt.ctx.enqueue_create_host_buffer[DType.int32](vectors)
    for i in range(vectors):
        pos_host[i] = Int32(position)
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


def _launch_tiled(
    mut rt: GpuRuntime,
    tile: Int,
    q_dev: DeviceBuffer[DType.float32],
    k_buf: DeviceBuffer[DType.float32],
    v_buf: DeviceBuffer[DType.float32],
    p_out: DeviceBuffer[DType.float32],
    p_max: DeviceBuffer[DType.float32],
    p_sum: DeviceBuffer[DType.float32],
    stored: Int,
    kv_heads: Int,
    q_heads: Int,
    head_dim: Int,
    scale: Float32,
    splits: Int,
) raises:
    """One attention_tiled_kernel launch at the compiled
    instantiation for `tile` (attn_validate_tile has already
    gated the value). Ported from Lane D2's gpu_layer launch
    helper, unchanged apart from taking resident buffers."""
    var grid = q_heads * splits
    if tile == 8:
        rt.ctx.enqueue_function[attention_tiled_kernel[8]](
            q_dev,
            k_buf,
            v_buf,
            p_out,
            p_max,
            p_sum,
            Int32(stored),
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
        rt.ctx.enqueue_function[attention_tiled_kernel[16]](
            q_dev,
            k_buf,
            v_buf,
            p_out,
            p_max,
            p_sum,
            Int32(stored),
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
        rt.ctx.enqueue_function[attention_tiled_kernel[32]](
            q_dev,
            k_buf,
            v_buf,
            p_out,
            p_max,
            p_sum,
            Int32(stored),
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
        rt.ctx.enqueue_function[attention_tiled_kernel[64]](
            q_dev,
            k_buf,
            v_buf,
            p_out,
            p_max,
            p_sum,
            Int32(stored),
            Int32(kv_heads),
            Int32(q_heads),
            Int32(head_dim),
            Int32(q_heads // kv_heads),
            scale,
            Int32(splits),
            grid_dim=grid,
            block_dim=ATTN_THREADS,
        )


def _launch_tiled_direct(
    mut rt: GpuRuntime,
    tile: Int,
    q_dev: DeviceBuffer[DType.float32],
    k_buf: DeviceBuffer[DType.float32],
    v_buf: DeviceBuffer[DType.float32],
    y_dev: DeviceBuffer[DType.float32],
    stored: Int,
    kv_heads: Int,
    q_heads: Int,
    head_dim: Int,
    scale: Float32,
) raises:
    """One attention_tiled_direct_kernel launch at the
    compiled instantiation for `tile` (attn_validate_tile
    has already gated the value): the splits == 1 form -
    one block per Q head, direct normalized output, no
    partial buffers and no merge launch."""
    if tile == 8:
        rt.ctx.enqueue_function[attention_tiled_direct_kernel[8]](
            q_dev,
            k_buf,
            v_buf,
            y_dev,
            Int32(stored),
            Int32(kv_heads),
            Int32(q_heads),
            Int32(head_dim),
            Int32(q_heads // kv_heads),
            scale,
            grid_dim=q_heads,
            block_dim=ATTN_THREADS,
        )
    elif tile == 16:
        rt.ctx.enqueue_function[attention_tiled_direct_kernel[16]](
            q_dev,
            k_buf,
            v_buf,
            y_dev,
            Int32(stored),
            Int32(kv_heads),
            Int32(q_heads),
            Int32(head_dim),
            Int32(q_heads // kv_heads),
            scale,
            grid_dim=q_heads,
            block_dim=ATTN_THREADS,
        )
    elif tile == 32:
        rt.ctx.enqueue_function[attention_tiled_direct_kernel[32]](
            q_dev,
            k_buf,
            v_buf,
            y_dev,
            Int32(stored),
            Int32(kv_heads),
            Int32(q_heads),
            Int32(head_dim),
            Int32(q_heads // kv_heads),
            scale,
            grid_dim=q_heads,
            block_dim=ATTN_THREADS,
        )
    else:
        rt.ctx.enqueue_function[attention_tiled_direct_kernel[64]](
            q_dev,
            k_buf,
            v_buf,
            y_dev,
            Int32(stored),
            Int32(kv_heads),
            Int32(q_heads),
            Int32(head_dim),
            Int32(q_heads // kv_heads),
            scale,
            grid_dim=q_heads,
            block_dim=ATTN_THREADS,
        )


def attention_device_resident(
    mut rt: GpuRuntime,
    mut kv: DeviceKvCache,
    q_buf: DeviceBuffer[DType.float32],
    layer: Int,
    q_heads: Int,
    kv_heads: Int,
    head_dim: Int,
) raises -> DeviceBuffer[DType.float32]:
    """The resident attention launch with the config-default
    tile / split knobs; see attention_device_resident_cfg."""
    return attention_device_resident_cfg(
        rt,
        kv,
        q_buf,
        layer,
        q_heads,
        kv_heads,
        head_dim,
        DEFAULT_ATTN_TILE_POSITIONS,
        DEFAULT_ATTN_MAX_SPLITS,
        DEFAULT_ATTN_SPLIT_SPAN_POSITIONS,
    )


def attention_device_resident_cfg(
    mut rt: GpuRuntime,
    mut kv: DeviceKvCache,
    q_buf: DeviceBuffer[DType.float32],
    layer: Int,
    q_heads: Int,
    kv_heads: Int,
    head_dim: Int,
    tile: Int,
    max_splits: Int,
    split_span: Int,
) raises -> DeviceBuffer[DType.float32]:
    """Tiled split-KV attention over the layer's VISIBLE rows
    in absolute-position order, Q resident (Wave D D2+D4
    reconciliation). The K/V sources are
    DeviceKvCache.gather_visible buffers: the raw buffers
    themselves when their slot order already is the visible
    order, a device-side visible-order gather past a wrap.
    LAYOUT AGREEMENT (checked against both lanes): the
    gather output is flat [stored][kv_heads][head_dim] with
    row i the visible row at absolute position
    first_visible + i - exactly the visible-order layout
    attention_tiled_kernel / attention_merge_kernel consume
    (and the layout D2's gpu_proof_attention_tiled proof fed
    them), so the launch needs no adaptation; the kernels'
    PRE-ROTATED Q/K contract is unchanged (rotation stays
    the caller's rope step, at write time for K). Each
    (head, split) block writes an unnormalized partial;
    the merge kernel combines them with exact logsumexp
    rescaling. No per-op sync: the context's single queue
    orders the tiled launch before the merge."""
    var stored = kv.stored_len(layer)
    if stored <= 0:
        raise Error(
            "gpu_layer_bad_shape: attention on layer "
            + String(layer)
            + " with no stored positions"
        )
    if q_heads * head_dim <= 0:
        raise Error("gpu_layer_bad_shape: attention Q row shape is empty")
    attn_validate_tile(tile)
    var splits = attn_split_count(stored, split_span, max_splits)
    var y_dev = _alloc_f32(rt, q_heads * head_dim)
    var k_vis = kv.gather_visible(rt, layer, False)
    var v_vis = kv.gather_visible(rt, layer, True)
    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    if splits == 1:
        # Short regime (Wave E, Lane E2): the direct-output
        # form - one launch, no partial buffers, no merge.
        # Its float path is the splits==1 partials path plus
        # the merge's final divide (see attention.mojo).
        _launch_tiled_direct(
            rt,
            tile,
            q_buf,
            k_vis,
            v_vis,
            y_dev,
            stored,
            kv_heads,
            q_heads,
            head_dim,
            scale,
        )
    else:
        var p_out = rt.ctx.enqueue_create_buffer[DType.float32](
            splits * q_heads * head_dim
        )
        var p_max = rt.ctx.enqueue_create_buffer[DType.float32](
            splits * q_heads
        )
        var p_sum = rt.ctx.enqueue_create_buffer[DType.float32](
            splits * q_heads
        )
        _launch_tiled(
            rt,
            tile,
            q_buf,
            k_vis,
            v_vis,
            p_out,
            p_max,
            p_sum,
            stored,
            kv_heads,
            q_heads,
            head_dim,
            scale,
            splits,
        )
        rt.ctx.enqueue_function[attention_merge_kernel](
            p_out,
            p_max,
            p_sum,
            y_dev,
            Int32(q_heads),
            Int32(head_dim),
            Int32(splits),
            grid_dim=q_heads,
            block_dim=ATTN_THREADS,
        )
    _ = log_debug(
        "gpu.layer",
        "attention resident tiled stored="
        + String(stored)
        + " tile="
        + String(tile)
        + " splits="
        + String(splits),
    )
    return y_dev^


def _attn_gate_buf(
    mut rt: GpuRuntime,
    a_buf: DeviceBuffer[DType.float32],
    g_buf: DeviceBuffer[DType.float32],
    n: Int,
) raises -> DeviceBuffer[DType.float32]:
    var y_dev = _alloc_f32(rt, n)
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


def _swiglu_buf(
    mut rt: GpuRuntime,
    g_buf: DeviceBuffer[DType.float32],
    u_buf: DeviceBuffer[DType.float32],
    n: Int,
) raises -> DeviceBuffer[DType.float32]:
    var y_dev = _alloc_f32(rt, n)
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


def _residual_add_buf(
    mut rt: GpuRuntime,
    a_buf: DeviceBuffer[DType.float32],
    b_buf: DeviceBuffer[DType.float32],
    n: Int,
) raises -> DeviceBuffer[DType.float32]:
    """residual_add_kernel on resident buffers: the same
    plain f32 per-element add residual_add_cpu does on host
    rows, so the resident chain's arithmetic matches the
    hostchain's exactly."""
    var y_dev = _alloc_f32(rt, n)
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


# ---- public surface ------------------------------------------


def gpu_logits(
    mut rt: GpuRuntime,
    weights: DeviceWeights,
    cpu_model: GlimmerModel,
    hidden: List[Float32],
) raises -> List[Float32]:
    """finish_last on device: final rms_norm (grid 1) + head
    GEMV by the output slot's dtype + final_logit_kernel
    (scale then softcap, both from the CPU model's config).
    Same math as GlimmerModel.logits."""
    var prof = debug_enabled()
    var prof_t0 = perf_counter_ns()
    var cfg = cpu_model.cfg.copy()
    if len(hidden) != cfg.hidden:
        raise Error(
            "gpu_layer_bad_shape: logits got hidden row of "
            + String(len(hidden))
            + ", hidden is "
            + String(cfg.hidden)
        )
    var norm_view = weights.output_norm()
    if norm_view.cols != cfg.hidden:
        raise Error(
            "gpu_layer_shape_mismatch: output_norm device width "
            + String(norm_view.cols)
            + " is not hidden "
            + String(cfg.hidden)
        )
    # Resident chain (Wave D): upload the hidden row once,
    # norm -> head GEMV -> final-logit transform on buffers,
    # download the logits row once.
    var hidden_dev = upload_f32(rt, hidden)
    var normed = _rms_norm_buf(
        rt, hidden_dev, cfg.hidden, norm_view.buffer, cfg.hidden, cfg.rms_eps
    )
    var raw = _proj_output_buf(rt, weights, cpu_model, normed, cfg.hidden)
    var raw_len = cpu_model.output.rows
    var y_dev = _alloc_f32(rt, raw_len)
    var grid = (raw_len + GATE_THREADS - 1) // GATE_THREADS
    rt.ctx.enqueue_function[final_logit_kernel](
        raw,
        y_dev,
        Int32(raw_len),
        cfg.logit_scale,
        cfg.logit_softcap,
        grid_dim=grid,
        block_dim=GATE_THREADS,
    )
    _ = log_debug("gpu.layer", "logits rows=" + String(raw_len))
    var logits_out = download_f32(rt, y_dev, raw_len)
    if prof:
        # PROFILE_LOGITS (Wave E lane E2, MOJO_ENGINE_DEBUG
        # gated): wall time of one gpu_logits call. The
        # download above synchronizes the queue, so the
        # interval covers every launch of the head chain.
        print(
            "PROFILE_LOGITS wall_us="
            + String(Int(perf_counter_ns() - prof_t0) // 1000)
        )
    return logits_out^


def gpu_layer_step_hostchain(
    mut rt: GpuRuntime,
    weights: DeviceWeights,
    mut kv: DeviceKvCache,
    cpu_model: GlimmerModel,
    x: List[Float32],
    layer: Int,
    position: Int,
) raises -> List[Float32]:
    """The Wave C step, kept as the residency A/B baseline:
    one token's hidden row through one layer, every op
    uploading its input row and downloading its output row,
    residual adds on the host. Still refuses past the KV
    window (its attention launches on the raw KV buffers).
    Statement for statement the CPU layer_step."""
    var cfg = cpu_model.cfg.copy()
    if layer < 0 or layer >= cfg.layers:
        raise Error(
            "gpu_layer_bad_shape: layer "
            + String(layer)
            + " outside 0.."
            + String(cfg.layers)
        )
    if len(x) != cfg.hidden:
        raise Error(
            "gpu_layer_bad_shape: layer input has "
            + String(len(x))
            + " elements, hidden is "
            + String(cfg.hidden)
        )
    if position < 0:
        raise Error(
            "gpu_layer_bad_shape: position must be >= 0, got "
            + String(position)
        )
    var kind = cfg.kinds[layer].copy()
    var sliding = not kind.is_global
    if sliding and position >= kind.window:
        raise Error(
            "gpu_kv_window_wrapped: layer "
            + String(layer)
            + " position "
            + String(position)
            + " would wrap its window "
            + String(kind.window)
            + "; the CPU path is the fallback beyond the window (Wave C limit)"
        )
    _ = log_debug(
        "gpu.layer",
        "step layer="
        + String(layer)
        + " position="
        + String(position)
        + " sliding="
        + String(sliding),
    )
    var residual = x.copy()
    var h = _rms_norm_device(
        rt,
        residual,
        _norm_buffer(weights, cpu_model, layer, 0, cfg.hidden),
        cfg.hidden,
        cfg.rms_eps,
    )
    var q = _proj_layer(rt, weights, cpu_model, layer, 1, h)
    var k = _proj_layer(rt, weights, cpu_model, layer, 2, h)
    var v = _proj_layer(rt, weights, cpu_model, layer, 3, h)
    q = _rms_norm_device(
        rt,
        q,
        _norm_buffer(weights, cpu_model, layer, 4, cfg.head_dim),
        cfg.head_dim,
        cfg.rms_eps,
    )
    k = _rms_norm_device(
        rt,
        k,
        _norm_buffer(weights, cpu_model, layer, 5, cfg.head_dim),
        cfg.head_dim,
        cfg.rms_eps,
    )
    var q_for_attn = q.copy()
    if sliding:
        # K rotates HERE (write time; the cache holds rotated
        # K, Rust-exact); Q rotates before the attention
        # launch because the kernel takes pre-rotated Q.
        k = _rope_device(
            rt, k, cfg.kv_heads, cfg.head_dim, position, cfg.rope_base_swa
        )
        q_for_attn = _rope_device(
            rt, q, cfg.q_heads, cfg.head_dim, position, cfg.rope_base_swa
        )
    kv.append(rt, layer, k, v, position)
    var attn = _attention_device(
        rt, kv, q_for_attn, layer, cfg.q_heads, cfg.kv_heads, cfg.head_dim
    )
    var gate = _proj_layer(rt, weights, cpu_model, layer, 7, h)
    var gated = _attn_gate_device(rt, attn, gate)
    var out = _proj_layer(rt, weights, cpu_model, layer, 6, gated)
    out = _rms_norm_device(
        rt,
        out,
        _norm_buffer(weights, cpu_model, layer, 8, cfg.hidden),
        cfg.hidden,
        POST_NORM_EPS,
    )
    var after_attn = residual_add_cpu(residual, out)
    var residual2 = after_attn.copy()
    var h2 = _rms_norm_device(
        rt,
        residual2,
        _norm_buffer(weights, cpu_model, layer, 9, cfg.hidden),
        cfg.hidden,
        cfg.rms_eps,
    )
    var gate_ffn = _proj_layer(rt, weights, cpu_model, layer, 10, h2)
    var up_ffn = _proj_layer(rt, weights, cpu_model, layer, 11, h2)
    var act = _swiglu_device(rt, gate_ffn, up_ffn)
    var ffn = _proj_layer(rt, weights, cpu_model, layer, 12, act)
    ffn = _rms_norm_device(
        rt,
        ffn,
        _norm_buffer(weights, cpu_model, layer, 13, cfg.hidden),
        cfg.hidden,
        POST_NORM_EPS,
    )
    return residual_add_cpu(residual2, ffn)


def gpu_layer_step_cfg(
    mut rt: GpuRuntime,
    weights: DeviceWeights,
    mut kv: DeviceKvCache,
    cpu_model: GlimmerModel,
    x: List[Float32],
    layer: Int,
    position: Int,
    attn_tile: Int,
    attn_max_splits: Int,
    attn_split_span: Int,
) raises -> List[Float32]:
    """One token's hidden row through one layer, activations
    RESIDENT on device across the whole layer (Wave D): the
    input row uploads once, every intermediate (residual
    stream, norm outputs, Q/K/V, gated attention, MLP rows)
    stays a device buffer, the KV append is device-to-device,
    and the output row downloads once. Statement for
    statement the CPU layer_step, same kernels in the same
    order as the hostchain step. Windowed layers run PAST
    the window here: the KV ring wraps (B1 semantics) and
    attention reads through the visible-order gather."""
    var prof = debug_enabled()
    var prof_t0 = perf_counter_ns()
    var cfg = cpu_model.cfg.copy()
    if layer < 0 or layer >= cfg.layers:
        raise Error(
            "gpu_layer_bad_shape: layer "
            + String(layer)
            + " outside 0.."
            + String(cfg.layers)
        )
    if len(x) != cfg.hidden:
        raise Error(
            "gpu_layer_bad_shape: layer input has "
            + String(len(x))
            + " elements, hidden is "
            + String(cfg.hidden)
        )
    if position < 0:
        raise Error(
            "gpu_layer_bad_shape: position must be >= 0, got "
            + String(position)
        )
    var kind = cfg.kinds[layer].copy()
    var sliding = not kind.is_global
    _ = log_debug(
        "gpu.layer",
        "step resident layer="
        + String(layer)
        + " position="
        + String(position)
        + " sliding="
        + String(sliding),
    )
    var qkv_w = cfg.kv_heads * cfg.head_dim
    var q_w = cfg.q_heads * cfg.head_dim
    var residual = upload_f32(rt, x)
    var h = _rms_norm_buf(
        rt,
        residual,
        cfg.hidden,
        _norm_buffer(weights, cpu_model, layer, 0, cfg.hidden),
        cfg.hidden,
        cfg.rms_eps,
    )
    var q = _proj_layer_buf(rt, weights, cpu_model, layer, 1, h, cfg.hidden)
    var k = _proj_layer_buf(rt, weights, cpu_model, layer, 2, h, cfg.hidden)
    var v = _proj_layer_buf(rt, weights, cpu_model, layer, 3, h, cfg.hidden)
    q = _rms_norm_buf(
        rt,
        q,
        q_w,
        _norm_buffer(weights, cpu_model, layer, 4, cfg.head_dim),
        cfg.head_dim,
        cfg.rms_eps,
    )
    k = _rms_norm_buf(
        rt,
        k,
        qkv_w,
        _norm_buffer(weights, cpu_model, layer, 5, cfg.head_dim),
        cfg.head_dim,
        cfg.rms_eps,
    )
    if sliding:
        # K rotates at write time (the cache holds rotated K);
        # Q rotates before the attention launch (pre-rotated
        # contract), exactly as the hostchain step.
        k = _rope_buf(
            rt,
            k,
            qkv_w,
            cfg.kv_heads,
            cfg.head_dim,
            position,
            cfg.rope_base_swa,
        )
        q = _rope_buf(
            rt, q, q_w, cfg.q_heads, cfg.head_dim, position, cfg.rope_base_swa
        )
    kv.append_device(rt, layer, k, v, position)
    var attn = attention_device_resident_cfg(
        rt,
        kv,
        q,
        layer,
        cfg.q_heads,
        cfg.kv_heads,
        cfg.head_dim,
        attn_tile,
        attn_max_splits,
        attn_split_span,
    )
    var gate = _proj_layer_buf(rt, weights, cpu_model, layer, 7, h, cfg.hidden)
    var gated = _attn_gate_buf(rt, attn, gate, q_w)
    var out = _proj_layer_buf(rt, weights, cpu_model, layer, 6, gated, q_w)
    out = _rms_norm_buf(
        rt,
        out,
        cfg.hidden,
        _norm_buffer(weights, cpu_model, layer, 8, cfg.hidden),
        cfg.hidden,
        POST_NORM_EPS,
    )
    var residual2 = _residual_add_buf(rt, residual, out, cfg.hidden)
    var h2 = _rms_norm_buf(
        rt,
        residual2,
        cfg.hidden,
        _norm_buffer(weights, cpu_model, layer, 9, cfg.hidden),
        cfg.hidden,
        cfg.rms_eps,
    )
    var gate_ffn = _proj_layer_buf(
        rt, weights, cpu_model, layer, 10, h2, cfg.hidden
    )
    var up_ffn = _proj_layer_buf(
        rt, weights, cpu_model, layer, 11, h2, cfg.hidden
    )
    var act = _swiglu_buf(rt, gate_ffn, up_ffn, _ffn_width(cpu_model, layer))
    var ffn = _proj_layer_buf(
        rt, weights, cpu_model, layer, 12, act, _ffn_width(cpu_model, layer)
    )
    ffn = _rms_norm_buf(
        rt,
        ffn,
        cfg.hidden,
        _norm_buffer(weights, cpu_model, layer, 13, cfg.hidden),
        cfg.hidden,
        POST_NORM_EPS,
    )
    var result = _residual_add_buf(rt, residual2, ffn, cfg.hidden)
    var layer_out = download_f32(rt, result, cfg.hidden)
    if prof:
        # PROFILE_LAYER (Wave E lane E2, MOJO_ENGINE_DEBUG
        # gated): wall time of one resident layer step. The
        # download above synchronizes the queue, so the
        # interval covers every launch of the layer; the one
        # print per layer lands between layers (inside the
        # NEXT layer's interval, a sub-1% bias at ~1 ms
        # layers, stated in the lane note). When the seam is
        # off this is two getenv-class checks and no other
        # behavior change.
        print(
            "PROFILE_LAYER layer="
            + String(layer)
            + " position="
            + String(position)
            + " sliding="
            + String(1 if sliding else 0)
            + " wall_us="
            + String(Int(perf_counter_ns() - prof_t0) // 1000)
        )
    return layer_out^


def gpu_layer_step_resident(
    mut rt: GpuRuntime,
    weights: DeviceWeights,
    mut kv: DeviceKvCache,
    cpu_model: GlimmerModel,
    x: List[Float32],
    layer: Int,
    position: Int,
) raises -> List[Float32]:
    """The resident chain with the config-default attention
    knobs (Lane D4's entry point; the body lives in
    gpu_layer_step_cfg)."""
    return gpu_layer_step_cfg(
        rt,
        weights,
        kv,
        cpu_model,
        x,
        layer,
        position,
        DEFAULT_ATTN_TILE_POSITIONS,
        DEFAULT_ATTN_MAX_SPLITS,
        DEFAULT_ATTN_SPLIT_SPAN_POSITIONS,
    )


def _ffn_width(cpu_model: GlimmerModel, layer: Int) -> Int:
    """The FFN intermediate width: the ffn_gate slot's rows
    (suffix 10), the width gate/up produce and down consumes.
    """
    var idx = layer * LAYER_SUFFIX_COUNT + 10
    return cpu_model.layer_weights[idx].rows


def gpu_layer_step(
    mut rt: GpuRuntime,
    weights: DeviceWeights,
    mut kv: DeviceKvCache,
    cpu_model: GlimmerModel,
    x: List[Float32],
    layer: Int,
    position: Int,
) raises -> List[Float32]:
    """One token's hidden row through one layer on device.
    Wave D: the resident chain (gpu_layer_step_cfg's body,
    with the config-default attention knobs here); the
    Wave C hostchain survives above as the A/B baseline.
    """
    return gpu_layer_step_cfg(
        rt,
        weights,
        kv,
        cpu_model,
        x,
        layer,
        position,
        DEFAULT_ATTN_TILE_POSITIONS,
        DEFAULT_ATTN_MAX_SPLITS,
        DEFAULT_ATTN_SPLIT_SPAN_POSITIONS,
    )
