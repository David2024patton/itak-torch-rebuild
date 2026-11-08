# gpu_embedding.mojo - the device embedding row, Lane C2
# (Wave C). Public surface (WAVE_C_INTERFACES.md):
#   gpu_embed_token(rt, weights, token) -> List[Float32]
#
# One token's embedding row, dequantized ON DEVICE from the
# resident embedding table buffer held by DeviceWeights. The
# CPU twin is GlimmerModel.embed_token; the element math here
# mirrors it term for term:
#   Q4_K: 144-byte super-blocks of 256 elements, value =
#         d * sc * q - dmin * m (model.mojo's _q4k_element,
#         device flavour via kernels/dequant's pointer
#         helpers: read_f16_ptr + scale_min_k4_ptr).
#   Q6_K: q6k_element_ptr.  Q5_K: q5k_element_ptr.
#   F16:  read_f16_ptr gather.
#   F32:  REFUSED BY NAME (gpu_embedding_unsupported_dtype):
#         an F32 table lives in DeviceWeights as a float32
#         buffer, not the uint8 buffer this path launches
#         against, and the Wave A census says this file's
#         F32 tensors are the norms, so no live model hits
#         the refusal. Any other type id refuses the same way.
#
# Dequant is exact arithmetic per element (no accumulation),
# so the parity bar against the CPU twin is element-exact
# class (proof bar: max abs <= 1e-6).
#
# C1 surface used (contract): GpuRuntime + upload_f32 /
# download_f32 from gpu.context, DeviceWeights.embedding_u8()
# returning (buffer, type_id, rows, cols) from gpu.weights.
# Norm-free module: this file never touches src/gpu/ itself
# beyond those imports.
#
# Named refusals: gpu_embedding_bad_token,
# gpu_embedding_bad_shape, gpu_embedding_unsupported_dtype.

from max.gpu import thread_idx, block_idx, block_dim

from diag import log_debug
from gpu.context import GpuRuntime, upload_f32, download_f32
from gpu.weights import DeviceWeights
from kernels.dequant import (
    read_f16_ptr,
    scale_min_k4_ptr,
    q6k_element_ptr,
    q5k_element_ptr,
)

# ggml type ids (gguf/loader.mojo ggml_type_name; same values
# as model.mojo's DT_ constants).
comptime DT_F32 = 0
comptime DT_F16 = 1
comptime DT_Q4_K = 12
comptime DT_Q5_K = 13
comptime DT_Q6_K = 14

comptime EMB_THREADS = 256
comptime SB_ELEMS = 256


def embedding_sb_bytes(type_id: Int) raises -> Int:
    """Super-block byte size for a quantized embedding dtype.
    F16 has no super-blocks (0); anything else refuses."""
    if type_id == DT_Q4_K:
        return 144
    if type_id == DT_Q6_K:
        return 210
    if type_id == DT_Q5_K:
        return 176
    if type_id == DT_F16:
        return 0
    raise Error(
        "gpu_embedding_unsupported_dtype: ggml type id "
        + String(type_id)
        + " has no embedding super-block layout this path serves"
    )


def embedding_row_bytes(type_id: Int, hidden: Int) raises -> Int:
    """Bytes of one embedding row for the table's dtype."""
    if hidden <= 0:
        raise Error(
            "gpu_embedding_bad_shape: hidden must be positive, got "
            + String(hidden)
        )
    if type_id == DT_F16:
        return hidden * 2
    if type_id == DT_F32:
        return hidden * 4
    var sb = embedding_sb_bytes(type_id)
    if hidden % SB_ELEMS != 0:
        raise Error(
            "gpu_embedding_bad_shape: hidden "
            + String(hidden)
            + " is not a whole number of 256-element super-blocks"
        )
    return (hidden // SB_ELEMS) * sb


def embedding_dequant_kernel(
    w: Pointer[UInt8, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    token: Int32,
    hidden: Int32,
    type_id: Int32,
):
    """One thread per output element: dequantize element idx
    of the token's row. Host validates dtype and shape before
    launch (the named refusals live host-side)."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(hidden):
        return
    var t = Int(token)
    if Int(type_id) == DT_F16:
        y[unsafe_offset=idx] = read_f16_ptr(w, (t * Int(hidden) + idx) * 2)
        return
    var within = idx & 255
    var blk = idx >> 8
    if Int(type_id) == DT_Q4_K:
        var sb_off = t * (Int(hidden) // SB_ELEMS) * 144 + blk * 144
        var d = read_f16_ptr(w, sb_off)
        var dmin = read_f16_ptr(w, sb_off + 2)
        var g = within >> 6
        var p = within & 63
        var sc: Float32
        var m: Float32
        var q: Float32
        if p < 32:
            var sm = scale_min_k4_ptr(2 * g, w, sb_off)
            sc = sm[0]
            m = sm[1]
            q = Float32(Int(w[unsafe_offset=sb_off + 16 + g * 32 + p]) & 0xF)
        else:
            var sm2 = scale_min_k4_ptr(2 * g + 1, w, sb_off)
            sc = sm2[0]
            m = sm2[1]
            q = Float32(Int(w[unsafe_offset=sb_off + 16 + g * 32 + (p - 32)]) >> 4)
        y[unsafe_offset=idx] = d * sc * q - dmin * m
        return
    if Int(type_id) == DT_Q6_K:
        var sb6 = t * (Int(hidden) // SB_ELEMS) * 210 + blk * 210
        y[unsafe_offset=idx] = q6k_element_ptr(w, sb6, within)
        return
    # DT_Q5_K (the only remaining validated dtype).
    var sb5 = t * (Int(hidden) // SB_ELEMS) * 176 + blk * 176
    y[unsafe_offset=idx] = q5k_element_ptr(w, sb5, within)


def gpu_embed_token(
    rt: GpuRuntime, weights: DeviceWeights, token: Int
) raises -> List[Float32]:
    """Device-dequantized embedding row for one token."""
    var slot = weights.embedding_u8()
    var buf = slot[0].copy()
    var type_id = slot[1]
    var rows = slot[2]
    var cols = slot[3]
    if token < 0 or token >= rows:
        raise Error(
            "gpu_embedding_bad_token: token "
            + String(token)
            + " outside vocab 0.."
            + String(rows)
        )
    if (
        type_id != DT_Q4_K
        and type_id != DT_Q6_K
        and type_id != DT_Q5_K
        and type_id != DT_F16
    ):
        raise Error(
            "gpu_embedding_unsupported_dtype: token_embd.weight has"
            " ggml type id "
            + String(type_id)
            + "; this path serves Q4_K/Q6_K/Q5_K/F16 byte tables"
            " (F32 refuses here by design, see the module header)"
        )
    _ = embedding_row_bytes(type_id, cols)
    var zeros = List[Float32](capacity=cols)
    zeros.resize(cols, Float32(0.0))
    var out_dev = upload_f32(rt, zeros)
    var grid = (cols + EMB_THREADS - 1) // EMB_THREADS
    rt.ctx.enqueue_function[embedding_dequant_kernel](
        buf,
        out_dev,
        Int32(token),
        Int32(cols),
        Int32(type_id),
        grid_dim=grid,
        block_dim=EMB_THREADS,
    )
    rt.ctx.synchronize()
    _ = log_debug(
        "gpu.embedding", "token " + String(token) + " hidden " + String(cols)
    )
    return download_f32(rt, out_dev, cols)
