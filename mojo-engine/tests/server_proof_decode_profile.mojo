# server_proof_decode_profile.mojo - Wave E lane E2 decode
# profile. NOT a TestSuite file: run explicitly on a GPU host
# with
#   mojo run -I src tests/server_proof_decode_profile.mojo
#
# Two phases on the real Glimmer file:
#
# Phase WALL: GpuGlimmerModel decode steps (the production
# step path, no instrumentation): prompt stepped at
# positions 0..5, 3 warmup decode steps, then 10 measured
# free-running steps; per-step wall times for model.step
# (all 52 layers) and model.logits printed as medians.
#
# Phase OPS: per-op timings on the resident chain's own
# launch shapes, taken from src/model/gpu_layer.mojo
# line by line (each timed region is enqueue + synchronize,
# the D2/D3 proof pattern; warmup reps first, medians over
# the timed reps). Ops: every GEMV shape of a layer (layer 0
# slots), the head GEMV via gpu_logits whole and via its
# Q5_K launch, rms norms (hidden row, Q-head rows, K-head
# rows), rope (Q and K), the KV append pair (production
# DeviceKvCache.append_device, sync included as shipped),
# resident attention at stored 16 and 128 (production
# attention_device_residency_cfg entry, config-default
# knobs, splits == 1 in both regimes), attn_gate, swiglu,
# residual add, final_logit, create_zero_f32 vs raw
# enqueue_create_buffer at the chain's buffer sizes,
# upload/download of one hidden row, and a bare synchronize.
# One whole resident layer step (production
# gpu_layer_step_resident) per layer kind closes the phase
# as the cross-check the op sum must reconstruct.
#
# Prints PROFILE_ lines only; correctness of the ops timed
# here is gated by the existing proofs (layer proof, teacher
# golden, attention proofs), not re-proven in this harness.

from std.sys import exit
from std.time import perf_counter_ns

from config import DEFAULT_GPU_BUDGET_BYTES
from generate.sampler import greedy_argmax
from gpu.context import GpuRuntime, create_zero_f32, download_f32, upload_f32
from gpu.device_kv import DeviceKvCache
from gpu.weights import DeviceWeights
from kernels.f32_gemv import (
    f16_gemv_kernel,
    f32_gemv_kernel,
    THREADS as F_THREADS,
    F32_THREADS,
    ROWS_PER_WG as F_ROWS_PER_WG,
)
from kernels.k1_gemv import (
    k1_gemv_kernel,
    THREADS as K1_THREADS,
    ROWS_PER_WG as K1_ROWS_PER_WG,
)
from kernels.q5k_gemv import (
    q5k_gemv_kernel,
    THREADS as Q5_THREADS,
    ROWS_PER_WG as Q5_ROWS_PER_WG,
)
from kernels.q6k_gemv import q6k_gemv_kernel, THREADS as Q6_THREADS
from model.gpu_layer import (
    GATE_THREADS,
    attention_device_resident_cfg,
    attn_gate_kernel,
    gpu_layer_step_resident,
    gpu_logits,
    rope_launch_shape,
)
from model.gpu_model import GpuGlimmerModel
from model.model import (
    DT_F16,
    DT_F32,
    DT_Q4_K,
    DT_Q5_K,
    DT_Q6_K,
    GlimmerModel,
)
from ops.final_logit import final_logit_kernel
from ops.residual import residual_add_kernel
from ops.rms_norm import RMS_THREADS, rms_norm_kernel
from ops.rope import rope_kernel
from ops.swiglu import swiglu_kernel
from attention.attention import attn_split_count
from config import (
    DEFAULT_ATTN_MAX_SPLITS,
    DEFAULT_ATTN_SPLIT_SPAN_POSITIONS,
    DEFAULT_ATTN_TILE_POSITIONS,
)

comptime MODEL_PATH = (
    "/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
)


def _median(mut values: List[Int]) -> Int:
    for i in range(1, len(values)):
        var j = i
        while j > 0 and values[j - 1] > values[j]:
            var tmp = values[j - 1]
            values[j - 1] = values[j]
            values[j] = tmp
            j -= 1
    return values[len(values) // 2]


def _pattern(n: Int, seed: Int) -> List[Float32]:
    var out = List[Float32](capacity=n)
    out.resize(n, Float32(0.0))
    for i in range(n):
        out[i] = Float32(((i * 7 + seed * 13) % 13) - 6) * Float32(0.01)
    return out^


def _report(name: String, mut times: List[Int]):
    var med = _median(times)
    print(
        "PROFILE_OP "
        + name
        + " median_us="
        + String(med // 1000)
        + " median_ns="
        + String(med)
        + " reps="
        + String(len(times))
    )


def phase_wall() raises:
    var model = GpuGlimmerModel.open(MODEL_PATH, 4096, DEFAULT_GPU_BUDGET_BYTES)
    var prompt = List[Int]()
    for t in [200000, 954, 7963, 323, 11698, 373]:
        prompt.append(t)
    var hidden = model.step(prompt[0], 0)
    for i in range(1, len(prompt)):
        hidden = model.step(prompt[i], i)
    var pos = len(prompt)
    var step_times = List[Int]()
    var logits_times = List[Int]()
    # 3 warmup steps + 10 measured steps, free-running.
    for it in range(13):
        var logits0 = model.logits(hidden.copy())
        var pred = greedy_argmax(logits0)
        var t0 = perf_counter_ns()
        hidden = model.step(pred, pos)
        var t1 = perf_counter_ns()
        var logits = model.logits(hidden.copy())
        var t2 = perf_counter_ns()
        if it >= 3:
            step_times.append(Int(t1 - t0))
            logits_times.append(Int(t2 - t1))
        pos += 1
    var step_med = _median(step_times)
    var logits_med = _median(logits_times)
    print(
        "PROFILE_WALL step_median_us="
        + String(step_med // 1000)
        + " logits_median_us="
        + String(logits_med // 1000)
        + " step_plus_logits_us="
        + String((step_med + logits_med) // 1000)
        + " tok_s="
        + String(Float64(1_000_000_000) / Float64(step_med + logits_med))
    )


def phase_ops() raises:
    var cpu_model = GlimmerModel.open(MODEL_PATH)
    var cfg = cpu_model.cfg.copy()
    var rt = GpuRuntime.create(DEFAULT_GPU_BUDGET_BYTES)
    var weights = DeviceWeights.build(rt, cpu_model)
    print(
        "PROFILE_CFG hidden="
        + String(cfg.hidden)
        + " ffn="
        + String(cfg.ffn)
        + " layers="
        + String(cfg.layers)
        + " q_heads="
        + String(cfg.q_heads)
        + " kv_heads="
        + String(cfg.kv_heads)
        + " head_dim="
        + String(cfg.head_dim)
        + " vocab="
        + String(cfg.vocab)
    )
    for suffix in range(14):
        var slot = cpu_model.layer_weights[suffix].copy()
        print(
            "PROFILE_SLOT layer0 suffix="
            + String(suffix)
            + " name="
            + slot.name
            + " type="
            + String(slot.type_id)
            + " rows="
            + String(slot.rows)
            + " cols="
            + String(slot.cols)
        )
    print(
        "PROFILE_SLOT output type="
        + String(cpu_model.output.type_id)
        + " rows="
        + String(cpu_model.output.rows)
        + " cols="
        + String(cpu_model.output.cols)
    )

    var hidden = cfg.hidden
    var q_w = cfg.q_heads * cfg.head_dim
    var kv_w = cfg.kv_heads * cfg.head_dim
    var x_hidden = upload_f32(rt, _pattern(hidden, 1))
    var x_qw = upload_f32(rt, _pattern(q_w, 2))
    var x_ffn = upload_f32(rt, _pattern(cfg.ffn, 3))

    # ---- GEMV launches, resident geometry (gpu_layer._gemv_u8_buf /
    # _gemv_f32_buf shapes) at layer 0's real slots.
    # suffix: 1 q, 2 k, 3 v, 6 out, 7 attn gate, 10 ffn gate,
    # 11 ffn up, 12 ffn down.
    var gemv_specs = List[Tuple[Int, Int]]()
    # (suffix, input width selector: 0 hidden, 1 q_w, 2 ffn)
    gemv_specs.append((1, 0))
    gemv_specs.append((2, 0))
    gemv_specs.append((3, 0))
    gemv_specs.append((7, 0))
    gemv_specs.append((6, 1))
    gemv_specs.append((10, 0))
    gemv_specs.append((11, 0))
    gemv_specs.append((12, 2))
    for si in range(len(gemv_specs)):
        var suffix = gemv_specs[si][0]
        var sel = gemv_specs[si][1]
        var view = weights.slot_u8(0, suffix)
        var x_buf = x_hidden.copy()
        if sel == 1:
            x_buf = x_qw.copy()
        elif sel == 2:
            x_buf = x_ffn.copy()
        var reps = 30
        var times = List[Int]()
        for rep in range(reps + 3):
            var t0 = perf_counter_ns()
            var y_dev = create_zero_f32(rt, view.rows)
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
            rt.ctx.synchronize()
            if rep >= 3:
                times.append(Int(perf_counter_ns() - t0))
        _report(
            "gemv_layer0_suffix"
            + String(suffix)
            + "_type"
            + String(view.type_id)
            + "_"
            + String(view.rows)
            + "x"
            + String(view.cols)
            + "_incl_zero_alloc",
            times,
        )

    # Head GEMV alone (output slot, Q5_K): the zero+alloc is
    # outside the timed region here, launch+sync only.
    var oview = weights.output_u8()
    var oy = create_zero_f32(rt, oview.rows)
    var otimes = List[Int]()
    for rep in range(13):
        var t0 = perf_counter_ns()
        rt.ctx.enqueue_function[q5k_gemv_kernel](
            oview.buffer,
            x_hidden,
            oy,
            Int32(oview.cols),
            grid_dim=oview.rows // Q5_ROWS_PER_WG,
            block_dim=Q5_THREADS,
        )
        rt.ctx.synchronize()
        if rep >= 3:
            otimes.append(Int(perf_counter_ns() - t0))
    _report("gemv_head_q5k_202048x6656_launch_only", otimes)

    # gpu_logits whole (norm + head GEMV + final_logit + download).
    var hlist = _pattern(hidden, 4)
    var ltimes = List[Int]()
    for rep in range(8):
        var t0 = perf_counter_ns()
        var lg = gpu_logits(rt, weights, cpu_model, hlist.copy())
        _ = lg
        if rep >= 2:
            ltimes.append(Int(perf_counter_ns() - t0))
    _report("gpu_logits_whole", ltimes)

    # ---- rms_norm shapes: hidden row (grid 1), Q norm
    # (grid q_heads over head_dim rows), K norm (grid kv_heads).
    var norm_w = weights.slot_f32(0, 0).buffer.copy()
    var qnorm_w = weights.slot_f32(0, 4).buffer.copy()
    var knorm_w = weights.slot_f32(0, 5).buffer.copy()
    var nspecs = List[Tuple[String, Int, Int]]()
    nspecs.append(("rms_hidden", hidden, hidden))
    nspecs.append(("rms_qheads", q_w, cfg.head_dim))
    nspecs.append(("rms_kvheads", kv_w, cfg.head_dim))
    for ni in range(len(nspecs)):
        var n = nspecs[ni][1]
        var row_len = nspecs[ni][2]
        var w_buf = norm_w.copy()
        var src = x_hidden.copy()
        if nspecs[ni][0] == "rms_qheads":
            w_buf = qnorm_w.copy()
            src = x_qw.copy()
        elif nspecs[ni][0] == "rms_kvheads":
            w_buf = knorm_w.copy()
            src = upload_f32(rt, _pattern(kv_w, 5))
        var times = List[Int]()
        for rep in range(53):
            var t0 = perf_counter_ns()
            var y_dev = create_zero_f32(rt, n)
            rt.ctx.enqueue_function[rms_norm_kernel](
                src,
                w_buf,
                y_dev,
                Int32(row_len),
                cfg.rms_eps,
                grid_dim=n // row_len,
                block_dim=RMS_THREADS,
            )
            rt.ctx.synchronize()
            if rep >= 3:
                times.append(Int(perf_counter_ns() - t0))
        _report("norm_" + nspecs[ni][0] + "_incl_zero_alloc", times)

    # ---- rope Q (32 vectors) and K (2 vectors), resident
    # shape (zero alloc + pos host buffer + copy + kernel).
    for which in range(2):
        var vectors = cfg.q_heads if which == 0 else cfg.kv_heads
        var n = vectors * cfg.head_dim
        var src = x_qw.copy() if which == 0 else upload_f32(
            rt, _pattern(kv_w, 6)
        )
        var shape = rope_launch_shape(vectors * (cfg.head_dim // 2))
        var times = List[Int]()
        for rep in range(53):
            var t0 = perf_counter_ns()
            var y_dev = create_zero_f32(rt, n)
            var pos_host = rt.ctx.enqueue_create_host_buffer[DType.int32](
                vectors
            )
            for i in range(vectors):
                pos_host[i] = Int32(7)
            var pos_dev = rt.ctx.enqueue_create_buffer[DType.int32](vectors)
            rt.ctx.enqueue_copy(src_buf=pos_host, dst_buf=pos_dev)
            rt.ctx.enqueue_function[rope_kernel](
                src,
                y_dev,
                pos_dev,
                Int32(cfg.head_dim),
                cfg.rope_base_swa,
                grid_dim=shape[0],
                block_dim=shape[1],
            )
            rt.ctx.synchronize()
            if rep >= 3:
                times.append(Int(perf_counter_ns() - t0))
        _report(
            "rope_" + String("q" if which == 0 else "k") + "_incl_zero_pos",
            times,
        )

    # ---- KV append pair (production append_device; its
    # internal synchronize is part of the shipped shape).
    var kv = DeviceKvCache.build(
        rt, cfg.kinds.copy(), cfg.kv_heads, cfg.head_dim, 4096
    )
    var k_row = upload_f32(rt, _pattern(kv_w, 7))
    var v_row = upload_f32(rt, _pattern(kv_w, 8))
    var atimes = List[Int]()
    for rep in range(53):
        var t0 = perf_counter_ns()
        kv.append_device(rt, 0, k_row, v_row, rep)
        if rep >= 3:
            atimes.append(Int(perf_counter_ns() - t0))
    _report("kv_append_device_pair_incl_sync", atimes)

    # ---- Resident attention at stored 16 and 128 (kv above
    # now holds 53 rows in layer 0; rebuild fresh caches at
    # the two stored counts for exactness).
    for stored_target in [16, 128]:
        rt.kv_bytes = 0
        var kv2 = DeviceKvCache.build(
            rt, cfg.kinds.copy(), cfg.kv_heads, cfg.head_dim, 4096
        )
        for p in range(stored_target):
            kv2.append_device(rt, 0, k_row, v_row, p)
        var splits = attn_split_count(
            stored_target,
            DEFAULT_ATTN_SPLIT_SPAN_POSITIONS,
            DEFAULT_ATTN_MAX_SPLITS,
        )
        var times = List[Int]()
        for rep in range(33):
            var t0 = perf_counter_ns()
            var y = attention_device_resident_cfg(
                rt,
                kv2,
                x_qw,
                0,
                cfg.q_heads,
                cfg.kv_heads,
                cfg.head_dim,
                DEFAULT_ATTN_TILE_POSITIONS,
                DEFAULT_ATTN_MAX_SPLITS,
                DEFAULT_ATTN_SPLIT_SPAN_POSITIONS,
            )
            _ = y
            rt.ctx.synchronize()
            if rep >= 3:
                times.append(Int(perf_counter_ns() - t0))
        _report(
            "attention_resident_stored"
            + String(stored_target)
            + "_splits"
            + String(splits),
            times,
        )

    # ---- Small elementwise ops (zero alloc + kernel + sync).
    # attn_gate over q_w, swiglu over ffn, residual over hidden.
    var gtimes = List[Int]()
    for rep in range(53):
        var t0 = perf_counter_ns()
        var y_dev = create_zero_f32(rt, q_w)
        var grid = (q_w + GATE_THREADS - 1) // GATE_THREADS
        rt.ctx.enqueue_function[attn_gate_kernel](
            x_qw, x_qw, y_dev, Int32(q_w), grid_dim=grid, block_dim=GATE_THREADS
        )
        rt.ctx.synchronize()
        if rep >= 3:
            gtimes.append(Int(perf_counter_ns() - t0))
    _report("attn_gate_4096_incl_zero_alloc", gtimes)

    var stimes = List[Int]()
    for rep in range(53):
        var t0 = perf_counter_ns()
        var y_dev = create_zero_f32(rt, cfg.ffn)
        var grid = (cfg.ffn + GATE_THREADS - 1) // GATE_THREADS
        rt.ctx.enqueue_function[swiglu_kernel](
            x_ffn,
            x_ffn,
            y_dev,
            Int32(cfg.ffn),
            grid_dim=grid,
            block_dim=GATE_THREADS,
        )
        rt.ctx.synchronize()
        if rep >= 3:
            stimes.append(Int(perf_counter_ns() - t0))
    _report("swiglu_19968_incl_zero_alloc", stimes)

    var rtimes = List[Int]()
    for rep in range(53):
        var t0 = perf_counter_ns()
        var y_dev = create_zero_f32(rt, hidden)
        var grid = (hidden + GATE_THREADS - 1) // GATE_THREADS
        rt.ctx.enqueue_function[residual_add_kernel](
            x_hidden,
            x_hidden,
            y_dev,
            Int32(hidden),
            grid_dim=grid,
            block_dim=GATE_THREADS,
        )
        rt.ctx.synchronize()
        if rep >= 3:
            rtimes.append(Int(perf_counter_ns() - t0))
    _report("residual_6656_incl_zero_alloc", rtimes)

    # final_logit over the vocab row.
    var ftimes = List[Int]()
    var big = create_zero_f32(rt, oview.rows)
    for rep in range(23):
        var t0 = perf_counter_ns()
        var y_dev = create_zero_f32(rt, oview.rows)
        var grid = (oview.rows + GATE_THREADS - 1) // GATE_THREADS
        rt.ctx.enqueue_function[final_logit_kernel](
            big,
            y_dev,
            Int32(oview.rows),
            cfg.logit_scale,
            cfg.logit_softcap,
            grid_dim=grid,
            block_dim=GATE_THREADS,
        )
        rt.ctx.synchronize()
        if rep >= 3:
            ftimes.append(Int(perf_counter_ns() - t0))
    _report("final_logit_202048_incl_zero_alloc", ftimes)

    # ---- Zero-fill vs raw allocation at chain buffer sizes.
    for n in [4096, 6656, 19968, 202048]:
        var ztimes = List[Int]()
        for rep in range(53):
            var t0 = perf_counter_ns()
            var b = create_zero_f32(rt, n)
            _ = b
            rt.ctx.synchronize()
            if rep >= 3:
                ztimes.append(Int(perf_counter_ns() - t0))
        _report("create_zero_f32_" + String(n), ztimes)
        var atimes2 = List[Int]()
        for rep in range(53):
            var t0 = perf_counter_ns()
            var b = rt.ctx.enqueue_create_buffer[DType.float32](n)
            _ = b
            rt.ctx.synchronize()
            if rep >= 3:
                atimes2.append(Int(perf_counter_ns() - t0))
        _report("raw_alloc_" + String(n), atimes2)

    # ---- Host crossings of one hidden row + bare sync.
    var hrow = _pattern(hidden, 9)
    var utimes = List[Int]()
    for rep in range(53):
        var t0 = perf_counter_ns()
        var b = upload_f32(rt, hrow.copy())
        _ = b
        rt.ctx.synchronize()
        if rep >= 3:
            utimes.append(Int(perf_counter_ns() - t0))
    _report("upload_f32_6656_plus_sync", utimes)
    var dtimes = List[Int]()
    for rep in range(53):
        var t0 = perf_counter_ns()
        var row = download_f32(rt, x_hidden, hidden)
        _ = row
        if rep >= 3:
            dtimes.append(Int(perf_counter_ns() - t0))
    _report("download_f32_6656_incl_sync", dtimes)
    var synctimes = List[Int]()
    for rep in range(53):
        var t0 = perf_counter_ns()
        rt.ctx.synchronize()
        if rep >= 3:
            synctimes.append(Int(perf_counter_ns() - t0))
    _report("bare_synchronize_empty_queue", synctimes)

    # ---- Whole resident layer steps (production entry) on
    # layer 0 (sliding) and layer 3 (global), fresh caches,
    # position advancing per rep as in decode.
    for target in [0, 3]:
        rt.kv_bytes = 0
        var kv3 = DeviceKvCache.build(
            rt, cfg.kinds.copy(), cfg.kv_heads, cfg.head_dim, 4096
        )
        var x_in = _pattern(hidden, 10)
        var times = List[Int]()
        for rep in range(13):
            var t0 = perf_counter_ns()
            var out = gpu_layer_step_resident(
                rt, weights, kv3, cpu_model, x_in.copy(), target, rep
            )
            _ = out
            if rep >= 3:
                times.append(Int(perf_counter_ns() - t0))
        _report("layer_step_resident_layer" + String(target), times)


def main():
    try:
        phase_wall()
        phase_ops()
        print("DECODE_PROFILE DONE")
    except e:
        print("REFUSAL", e)
        exit(2)
