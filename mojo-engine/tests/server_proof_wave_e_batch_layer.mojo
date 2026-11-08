# server_proof_wave_e_batch_layer.mojo - Wave E Lane E1
# per-layer proof: the resident batch layer step
# (gpu_layer_step_batch_cfg: GEMM projections on resident
# buffers + batched tiled split-KV attention) against the
# existing per-row resident path (gpu_layer_step_cfg) on
# the real Glimmer weights, at batch rows 6 and 64, layers
# 0 (sliding) and 3 (global).
# NOT a TestSuite file: run explicitly on a GPU host with
#   mojo run -I src tests/server_proof_wave_e_batch_layer.mojo
#
# Construction: batch row i's input is the model's own
# embedding of a deterministic token id + plain rms_norm
# (the rung-2 input construction of
# server_proof_gpu_layer.mojo, per row). The reference arm
# runs the rows through gpu_layer_step_cfg IN ORDER
# (positions 0..batch-1) on one fresh DeviceKvCache - the
# per-row semantics the batch step must reproduce,
# including each row attending to the rows before it. The
# batch arm runs ONE gpu_layer_step_batch_cfg on a second
# fresh cache. Full-flat-row comparison at the D2 bar
# class: cosine >= 0.999999999, rel <= 1e-5
# (rel = max_abs_err / max(1, max|ref|), k1_check style).
# Exits nonzero on any miss.

from std.math import sqrt
from std.sys import exit
from std.time import perf_counter_ns

from config import (
    DEFAULT_ATTN_MAX_SPLITS,
    DEFAULT_ATTN_SPLIT_SPAN_POSITIONS,
    DEFAULT_ATTN_TILE_POSITIONS,
    DEFAULT_GEMM_TILED_MIN_BATCH,
)
from gpu.context import GpuRuntime
from gpu.weights import DeviceWeights
from gpu.device_kv import DeviceKvCache
from model.model import GlimmerModel
from model.gpu_layer import gpu_layer_step_cfg
from model.gpu_layer_batch import (
    gpu_layer_step_batch_cfg,
    batch_row_place,
    batch_row_slice,
)
from ops.rms_norm import rms_norm_plain_cpu

comptime MODEL_PATH = (
    "/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
)
comptime GPU_BUDGET_CAP = 24696061952
comptime PROOF_CONTEXT_LEN = 4096
comptime COS_BAR = 0.999999999
comptime REL_BAR = 1.0e-5


def lcg_next(state: UInt32) -> UInt32:
    return state * 1664525 + 1013904223


def make_tokens(n: Int) -> List[Int]:
    var out = List[Int](capacity=n)
    var state: UInt32 = 42424242
    for _ in range(n):
        state = lcg_next(state)
        out.append(Int((state >> 8) % 200000))
    return out^


def compare_rows(a: List[Float32], b: List[Float32]) -> Tuple[Float64, Float64]:
    """(rel, cosine), k1_check style: rel denominator
    max(1, max|ref|), Float64 cosine. b is the reference."""
    var max_abs = Float64(0.0)
    var scale = Float64(0.0)
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(b)):
        var x = Float64(a[i])
        var y = Float64(b[i])
        var d = x - y
        if d < 0:
            d = -d
        if d > max_abs:
            max_abs = d
        var ay = y
        if ay < 0:
            ay = -ay
        if ay > scale:
            scale = ay
        dot += x * y
        na += x * x
        nb += y * y
    var denom = scale if scale > 1.0 else 1.0
    return (max_abs / denom, dot / (sqrt(na) * sqrt(nb)))


def main() raises:
    var ok = True
    var cpu_model = GlimmerModel.open(MODEL_PATH)
    var rt = GpuRuntime.create(GPU_BUDGET_CAP)
    var weights = DeviceWeights.build(rt, cpu_model)
    print("WAVE_E_BATCH_LAYER_PROOF weights resident")
    for layer in [0, 3]:
        for batch in [6, 64]:
            var tokens = make_tokens(batch)
            var hidden = cpu_model.cfg.hidden
            var x = List[Float32](capacity=batch * hidden)
            x.resize(batch * hidden, Float32(0.0))
            for i in range(batch):
                var row = cpu_model.embed_token(tokens[i])
                row = rms_norm_plain_cpu(row, cpu_model.cfg.rms_eps)
                batch_row_place(x, i, hidden, row)
            # Reference arm: per-row resident steps in
            # order on a fresh cache. (Ledger hygiene as in
            # server_proof_gpu_layer: the previous cache is
            # destroyed but the runtime ledger has no
            # unreserve, so reset before each build.)
            rt.kv_bytes = 0
            var kv_ref = DeviceKvCache.build(
                rt,
                cpu_model.cfg.kinds.copy(),
                cpu_model.cfg.kv_heads,
                cpu_model.cfg.head_dim,
                PROOF_CONTEXT_LEN,
            )
            var t0 = perf_counter_ns()
            var ref_flat = List[Float32](capacity=batch * hidden)
            ref_flat.resize(batch * hidden, Float32(0.0))
            for i in range(batch):
                var out_row = gpu_layer_step_cfg(
                    rt,
                    weights,
                    kv_ref,
                    cpu_model,
                    batch_row_slice(x, i, hidden),
                    layer,
                    i,
                    DEFAULT_ATTN_TILE_POSITIONS,
                    DEFAULT_ATTN_MAX_SPLITS,
                    DEFAULT_ATTN_SPLIT_SPAN_POSITIONS,
                )
                batch_row_place(ref_flat, i, hidden, out_row)
            var ref_ms = Int((perf_counter_ns() - t0) // 1_000_000)
            # Batch arm: one resident batch step, fresh cache.
            rt.kv_bytes = 0
            var kv_batch = DeviceKvCache.build(
                rt,
                cpu_model.cfg.kinds.copy(),
                cpu_model.cfg.kv_heads,
                cpu_model.cfg.head_dim,
                PROOF_CONTEXT_LEN,
            )
            var t1 = perf_counter_ns()
            var got = gpu_layer_step_batch_cfg(
                rt,
                weights,
                kv_batch,
                cpu_model,
                x.copy(),
                batch,
                layer,
                0,
                DEFAULT_GEMM_TILED_MIN_BATCH,
                DEFAULT_ATTN_TILE_POSITIONS,
                DEFAULT_ATTN_MAX_SPLITS,
                DEFAULT_ATTN_SPLIT_SPAN_POSITIONS,
            )
            var batch_ms = Int((perf_counter_ns() - t1) // 1_000_000)
            var cmp = compare_rows(got, ref_flat)
            var lok = cmp[1] >= COS_BAR and cmp[0] <= REL_BAR
            ok = ok and lok
            print(
                "WAVE_E_BATCH_LAYER layer="
                + String(layer)
                + " batch="
                + String(batch)
                + " cosine="
                + String(cmp[1])
                + " rel="
                + String(cmp[0])
                + " batch_ms="
                + String(batch_ms)
                + " perrow_ms="
                + String(ref_ms)
                + " stored_batch="
                + String(kv_batch.stored_len(layer))
                + " stored_ref="
                + String(kv_ref.stored_len(layer))
                + " ok="
                + String(1 if lok else 0)
            )
    if ok:
        print("WAVE_E_BATCH_LAYER_PROOF PASS")
    else:
        print("WAVE_E_BATCH_LAYER_PROOF FAIL")
        exit(1)
