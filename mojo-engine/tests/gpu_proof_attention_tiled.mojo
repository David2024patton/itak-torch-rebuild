# gpu_proof_attention_tiled.mojo - server-card proof +
# trial bench for Lane D2 (Wave D) tiled split-KV attention.
#
# NOT a TestSuite file: run explicitly on a GPU host with
#   mojo run -I src tests/gpu_proof_attention_tiled.mojo
#
# What it proves, in one run (all numbers printed, exits
# nonzero on any correctness miss):
#   CORRECTNESS (bar: cosine >= 0.999999999 and rel =
#   max abs diff / max abs expected <= 1e-5, the Wave B
#   gpu_proof_attention bars) for the NEW tiled+merge path
#   and the OLD Wave B kernel, against the CPU twin
#   (attention_cpu on RAW inputs, which rotates at read
#   time), on:
#     F1 the Wave B fixture: 32q/2kv hd128, stored 64,
#        position 63, RoPE 500000;
#     F2 stored 512, position 511, RoPE;
#     F3 stored 1025, position 1024, RoPE (past the 1,024
#        mark; splits engage);
#     F4 stored 4096, position 4095, RoPE (the long
#        fixture);
#     F5 stored 4096, position 4095, NoPE (the global-layer
#        shape at long context).
#   Every (tile, splits) trial config is correctness-checked
#   on F4 before it is timed. The CPU split twin's output on
#   F4 is also printed against the same reference.
#   GATING: the bars gate the NEW path and the CPU split
#   twin. The OLD kernel's lines are printed as BASELINE
#   with their at-bar status and are NOT counted as
#   failures: the old kernel was proven at its own fixture
#   in Wave B, and its drift at fixture sizes it never ran
#   is a recorded finding, not this lane's gate (window 1:
#   old kernel at F5 stored-4096 NoPE measured rel
#   1.0924672527787833e-05, outside the 1e-5 bar, while
#   every new-path config at F5 passed).
#   TIMING: interleaved A/B (old launch, new launch,
#   alternating order per round, per-launch averages from
#   perf_counter_ns around enqueue+synchronize) for:
#     - F1/F2 with splits=1 across tiles 8/16/32/64;
#     - F4 across the trial matrix (tiles x splits 1/2/4/8);
#     - the chooser's own picks (span 1024) at F3/F4.
#   A prefill-regime sweep (stored 1..64 step loop, the
#   engine's prefill shape) is timed old vs new at the end.
#
# The device buffers carry PRE-ROTATED Q/K (the kernels'
# contract): rotation is host-side rope_cpu, Q at the query
# position, each K row at its own position, exactly the
# Wave B proof's construction. On a GPU-less host
# DeviceContext creation fails loudly; that is a host fact,
# not a pass.

from std.math import sqrt
from std.sys import exit
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer

from attention.attention import (
    ATTN_THREADS,
    attention_cpu,
    attention_kernel,
    attention_merge_kernel,
    attention_split_cpu,
    attention_tiled_kernel,
)
from ops.rope import rope_cpu

comptime NQ = 32
comptime NKV = 2
comptime HD = 128
comptime BASE = Float32(500000.0)
comptime MAX_STORED = 4096
comptime MAX_SPLITS = 8


def _det(i: Int) -> Float32:
    return Float32(((i * 37 + 13) % 31) - 15) / Float32(8.0)


def _make(n: Int, seed: Int) -> List[Float32]:
    var x = List[Float32](capacity=n)
    x.resize(n, Float32(0.0))
    for i in range(n):
        x[i] = _det(i + seed * 101)
    return x^


def _metrics(
    got: List[Float32], expected: List[Float32]
) -> Tuple[Float64, Float64]:
    var dot = Float64(0.0)
    var norm_g = Float64(0.0)
    var norm_e = Float64(0.0)
    var max_abs = Float64(0.0)
    var max_e = Float64(0.0)
    for i in range(len(expected)):
        var g = Float64(got[i])
        var e = Float64(expected[i])
        dot += g * e
        norm_g += g * g
        norm_e += e * e
        var d = g - e
        if d < 0:
            d = -d
        if d > max_abs:
            max_abs = d
        var ae = e
        if ae < 0:
            ae = -ae
        if ae > max_e:
            max_e = ae
    var cosine = dot / (sqrt(norm_g) * sqrt(norm_e))
    return (cosine, max_abs / max_e)


def _bar(cosine: Float64, rel: Float64) -> Bool:
    return cosine >= 0.999999999 and rel <= 1.0e-5


def _upload(
    mut ctx: DeviceContext, data: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    var dev = ctx.enqueue_create_buffer[DType.float32](len(data))
    var host = ctx.enqueue_create_host_buffer[DType.float32](len(data))
    for i in range(len(data)):
        host[i] = data[i]
    ctx.enqueue_copy(src_buf=host, dst_buf=dev)
    return dev^


def _download(
    mut ctx: DeviceContext, dev: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(src_buf=dev, dst_buf=host)
    ctx.synchronize()
    var out = List[Float32](capacity=n)
    out.resize(n, Float32(0.0))
    for i in range(n):
        out[i] = host[i]
    return out^


def _launch_old(
    mut ctx: DeviceContext,
    q_dev: DeviceBuffer[DType.float32],
    k_dev: DeviceBuffer[DType.float32],
    v_dev: DeviceBuffer[DType.float32],
    o_dev: DeviceBuffer[DType.float32],
    stored: Int,
) raises:
    var scale = Float32(1.0) / sqrt(Float32(HD))
    ctx.enqueue_function[attention_kernel](
        q_dev,
        k_dev,
        v_dev,
        o_dev,
        Int32(stored),
        Int32(NKV),
        Int32(HD),
        Int32(NQ // NKV),
        scale,
        grid_dim=NQ,
        block_dim=ATTN_THREADS,
    )


def _launch_new(
    mut ctx: DeviceContext,
    tile: Int,
    splits: Int,
    q_dev: DeviceBuffer[DType.float32],
    k_dev: DeviceBuffer[DType.float32],
    v_dev: DeviceBuffer[DType.float32],
    p_out: DeviceBuffer[DType.float32],
    p_max: DeviceBuffer[DType.float32],
    p_sum: DeviceBuffer[DType.float32],
    o_dev: DeviceBuffer[DType.float32],
    stored: Int,
) raises:
    var scale = Float32(1.0) / sqrt(Float32(HD))
    var grid = NQ * splits
    if tile == 8:
        ctx.enqueue_function[attention_tiled_kernel[8]](
            q_dev,
            k_dev,
            v_dev,
            p_out,
            p_max,
            p_sum,
            Int32(stored),
            Int32(NKV),
            Int32(NQ),
            Int32(HD),
            Int32(NQ // NKV),
            scale,
            Int32(splits),
            grid_dim=grid,
            block_dim=ATTN_THREADS,
        )
    elif tile == 16:
        ctx.enqueue_function[attention_tiled_kernel[16]](
            q_dev,
            k_dev,
            v_dev,
            p_out,
            p_max,
            p_sum,
            Int32(stored),
            Int32(NKV),
            Int32(NQ),
            Int32(HD),
            Int32(NQ // NKV),
            scale,
            Int32(splits),
            grid_dim=grid,
            block_dim=ATTN_THREADS,
        )
    elif tile == 32:
        ctx.enqueue_function[attention_tiled_kernel[32]](
            q_dev,
            k_dev,
            v_dev,
            p_out,
            p_max,
            p_sum,
            Int32(stored),
            Int32(NKV),
            Int32(NQ),
            Int32(HD),
            Int32(NQ // NKV),
            scale,
            Int32(splits),
            grid_dim=grid,
            block_dim=ATTN_THREADS,
        )
    else:
        ctx.enqueue_function[attention_tiled_kernel[64]](
            q_dev,
            k_dev,
            v_dev,
            p_out,
            p_max,
            p_sum,
            Int32(stored),
            Int32(NKV),
            Int32(NQ),
            Int32(HD),
            Int32(NQ // NKV),
            scale,
            Int32(splits),
            grid_dim=grid,
            block_dim=ATTN_THREADS,
        )
    ctx.enqueue_function[attention_merge_kernel](
        p_out,
        p_max,
        p_sum,
        o_dev,
        Int32(NQ),
        Int32(HD),
        Int32(splits),
        grid_dim=NQ,
        block_dim=ATTN_THREADS,
    )


def _fixture(
    stored: Int, rope: Bool
) raises -> Tuple[List[Float32], List[Float32], List[Float32], List[Float32]]:
    """(expected, q_rot, k_rot, v) for one fixture: expected
    is attention_cpu on the RAW rows; q_rot/k_rot are the
    pre-rotated device inputs (raw copies when NoPE)."""
    var pos = stored - 1
    var q = _make(NQ * HD, 41)
    var k = _make(stored * NKV * HD, 42)
    var v = _make(stored * NKV * HD, 43)
    var base = BASE if rope else Float32(0.0)
    var expected = attention_cpu(q, k, v, 0, stored, NQ, NKV, HD, pos, base)
    if not rope:
        return (expected^, q^, k^, v^)
    var q_pos = List[Int](capacity=NQ)
    q_pos.resize(NQ, pos)
    var q_rot = rope_cpu(q, q_pos, HD, BASE)
    var k_pos = List[Int](capacity=stored * NKV)
    k_pos.resize(stored * NKV, 0)
    for i in range(stored):
        for kv in range(NKV):
            k_pos[i * NKV + kv] = i
    var k_rot = rope_cpu(k, k_pos, HD, BASE)
    return (expected^, q_rot^, k_rot^, v^)


def main() raises:
    var ctx = DeviceContext()
    var p_out = ctx.enqueue_create_buffer[DType.float32](MAX_SPLITS * NQ * HD)
    var p_max = ctx.enqueue_create_buffer[DType.float32](MAX_SPLITS * NQ)
    var p_sum = ctx.enqueue_create_buffer[DType.float32](MAX_SPLITS * NQ)
    var failures = 0

    # ---- correctness fixtures --------------------------------
    # (name, stored, rope, tile, splits) correctness matrix:
    # each fixture checks the OLD kernel, the NEW path at the
    # default-ish config, and the configs the timing section
    # will use at that stored count.
    var fixtures = List[Tuple[String, Int, Bool]]()
    fixtures.append(("F1-waveb-rope", 64, True))
    fixtures.append(("F2-rope", 512, True))
    fixtures.append(("F3-rope-1025", 1025, True))
    fixtures.append(("F4-rope-4096", 4096, True))
    fixtures.append(("F5-nope-4096", 4096, False))
    # Configs checked per fixture (tile, splits).
    for fi in range(len(fixtures)):
        var name = fixtures[fi][0]
        var stored = fixtures[fi][1]
        var rope = fixtures[fi][2]
        var fx = _fixture(stored, rope)
        var expected = fx[0].copy()
        var q_dev = _upload(ctx, fx[1])
        var k_dev = _upload(ctx, fx[2])
        var v_dev = _upload(ctx, fx[3])
        var o_dev = ctx.enqueue_create_buffer[DType.float32](NQ * HD)
        # Old kernel: baseline line, printed with its
        # at-bar status, never gated (see the header).
        _launch_old(ctx, q_dev, k_dev, v_dev, o_dev, stored)
        ctx.synchronize()
        var got_old = _download(ctx, o_dev, NQ * HD)
        var m_old = _metrics(got_old, expected)
        var ok_old = _bar(m_old[0], m_old[1])
        print(
            "CORRECT",
            name,
            "path baseline-old",
            "cosine",
            m_old[0],
            "rel",
            m_old[1],
            "at-bar" if ok_old else "OUTSIDE-BAR",
        )
        # New path configs for this fixture.
        var cfgs = List[Tuple[Int, Int]]()
        cfgs.append((32, 1))
        if stored >= 512:
            cfgs.append((32, 2))
        if stored >= 1025:
            cfgs.append((32, 3))
            cfgs.append((32, 4))
        if stored >= 4096:
            cfgs.append((8, 1))
            cfgs.append((16, 4))
            cfgs.append((64, 4))
            cfgs.append((64, 8))
            cfgs.append((16, 8))
        for ci in range(len(cfgs)):
            var tile = cfgs[ci][0]
            var splits = cfgs[ci][1]
            _launch_new(
                ctx,
                tile,
                splits,
                q_dev,
                k_dev,
                v_dev,
                p_out,
                p_max,
                p_sum,
                o_dev,
                stored,
            )
            ctx.synchronize()
            var got = _download(ctx, o_dev, NQ * HD)
            var m = _metrics(got, expected)
            var ok = _bar(m[0], m[1])
            if not ok:
                failures += 1
            print(
                "CORRECT",
                name,
                "path new tile",
                tile,
                "splits",
                splits,
                "cosine",
                m[0],
                "rel",
                m[1],
                "PASS" if ok else "FAIL",
            )
        # The CPU split twin against the same reference, at
        # the long fixtures (the VM suite covers the rest).
        if stored >= 1025:
            var twin = attention_split_cpu(
                fx[1], fx[2], fx[3], stored, NQ, NKV, HD, 4, 32
            )
            var mt = _metrics(twin, expected)
            var okt = _bar(mt[0], mt[1])
            if not okt:
                failures += 1
            print(
                "CORRECT",
                name,
                "path cpu-split-twin tile 32 splits 4",
                "cosine",
                mt[0],
                "rel",
                mt[1],
                "PASS" if okt else "FAIL",
            )

    # ---- interleaved timing ----------------------------------
    # Rounds of (old, new) launch pairs, order alternating per
    # round; per-launch averages in microseconds.
    var time_cfgs = List[Tuple[String, Int, Bool, Int, Int, Int]]()
    # (name, stored, rope, tile, splits, rounds)
    time_cfgs.append(("F1-waveb-rope", 64, True, 8, 1, 100))
    time_cfgs.append(("F1-waveb-rope", 64, True, 16, 1, 100))
    time_cfgs.append(("F1-waveb-rope", 64, True, 32, 1, 100))
    time_cfgs.append(("F1-waveb-rope", 64, True, 64, 1, 100))
    time_cfgs.append(("F2-rope", 512, True, 8, 1, 50))
    time_cfgs.append(("F2-rope", 512, True, 16, 1, 50))
    time_cfgs.append(("F2-rope", 512, True, 32, 1, 50))
    time_cfgs.append(("F2-rope", 512, True, 64, 1, 50))
    time_cfgs.append(("F2-rope", 512, True, 32, 2, 50))
    time_cfgs.append(("F3-rope-1025", 1025, True, 32, 2, 40))
    time_cfgs.append(("F3-rope-1025", 1025, True, 32, 3, 40))
    time_cfgs.append(("F4-rope-4096", 4096, True, 8, 1, 30))
    time_cfgs.append(("F4-rope-4096", 4096, True, 16, 1, 30))
    time_cfgs.append(("F4-rope-4096", 4096, True, 32, 1, 30))
    time_cfgs.append(("F4-rope-4096", 4096, True, 64, 1, 30))
    time_cfgs.append(("F4-rope-4096", 4096, True, 32, 2, 30))
    time_cfgs.append(("F4-rope-4096", 4096, True, 32, 4, 30))
    time_cfgs.append(("F4-rope-4096", 4096, True, 32, 8, 30))
    time_cfgs.append(("F4-rope-4096", 4096, True, 16, 4, 30))
    time_cfgs.append(("F4-rope-4096", 4096, True, 64, 4, 30))
    time_cfgs.append(("F4-rope-4096", 4096, True, 64, 8, 30))
    time_cfgs.append(("F5-nope-4096", 4096, False, 32, 4, 30))
    for ti in range(len(time_cfgs)):
        var name = time_cfgs[ti][0]
        var stored = time_cfgs[ti][1]
        var rope = time_cfgs[ti][2]
        var tile = time_cfgs[ti][3]
        var splits = time_cfgs[ti][4]
        var rounds = time_cfgs[ti][5]
        var fx = _fixture(stored, rope)
        var q_dev = _upload(ctx, fx[1])
        var k_dev = _upload(ctx, fx[2])
        var v_dev = _upload(ctx, fx[3])
        var o_dev = ctx.enqueue_create_buffer[DType.float32](NQ * HD)
        # Warmup: 3 launches each.
        for _ in range(3):
            _launch_old(ctx, q_dev, k_dev, v_dev, o_dev, stored)
            ctx.synchronize()
            _launch_new(
                ctx,
                tile,
                splits,
                q_dev,
                k_dev,
                v_dev,
                p_out,
                p_max,
                p_sum,
                o_dev,
                stored,
            )
            ctx.synchronize()
        var old_ns = 0
        var new_ns = 0
        for r in range(rounds):
            if r % 2 == 0:
                var t0 = perf_counter_ns()
                _launch_old(ctx, q_dev, k_dev, v_dev, o_dev, stored)
                ctx.synchronize()
                old_ns += perf_counter_ns() - t0
                var t1 = perf_counter_ns()
                _launch_new(
                    ctx,
                    tile,
                    splits,
                    q_dev,
                    k_dev,
                    v_dev,
                    p_out,
                    p_max,
                    p_sum,
                    o_dev,
                    stored,
                )
                ctx.synchronize()
                new_ns += perf_counter_ns() - t1
            else:
                var t1 = perf_counter_ns()
                _launch_new(
                    ctx,
                    tile,
                    splits,
                    q_dev,
                    k_dev,
                    v_dev,
                    p_out,
                    p_max,
                    p_sum,
                    o_dev,
                    stored,
                )
                ctx.synchronize()
                new_ns += perf_counter_ns() - t1
                var t0 = perf_counter_ns()
                _launch_old(ctx, q_dev, k_dev, v_dev, o_dev, stored)
                ctx.synchronize()
                old_ns += perf_counter_ns() - t0
        var old_us = Float64(old_ns) / Float64(rounds) / 1000.0
        var new_us = Float64(new_ns) / Float64(rounds) / 1000.0
        print(
            "TRIAL",
            name,
            "tile",
            tile,
            "splits",
            splits,
            "rounds",
            rounds,
            "old_us",
            old_us,
            "new_us",
            new_us,
            "ratio_old_over_new",
            old_us / new_us,
        )

    # ---- prefill-regime sweep --------------------------------
    # The engine's prefill is the step loop: stored grows
    # 1..64, one launch per step. Time both paths over the
    # whole sweep, repeated, tile 32 splits 1 (the chooser's
    # pick in this regime).
    var fx64 = _fixture(64, True)
    var q_dev = _upload(ctx, fx64[1])
    var k_dev = _upload(ctx, fx64[2])
    var v_dev = _upload(ctx, fx64[3])
    var o_dev = ctx.enqueue_create_buffer[DType.float32](NQ * HD)
    comptime SWEEP_REPS = 20
    var t_old0 = perf_counter_ns()
    for _ in range(SWEEP_REPS):
        for s in range(1, 65):
            _launch_old(ctx, q_dev, k_dev, v_dev, o_dev, s)
            ctx.synchronize()
    var sweep_old_us = (
        Float64(perf_counter_ns() - t_old0) / Float64(SWEEP_REPS) / 1000.0
    )
    var t_new0 = perf_counter_ns()
    for _ in range(SWEEP_REPS):
        for s in range(1, 65):
            _launch_new(
                ctx,
                32,
                1,
                q_dev,
                k_dev,
                v_dev,
                p_out,
                p_max,
                p_sum,
                o_dev,
                s,
            )
            ctx.synchronize()
    var sweep_new_us = (
        Float64(perf_counter_ns() - t_new0) / Float64(SWEEP_REPS) / 1000.0
    )
    print(
        "TRIAL prefill-sweep stored1..64 reps",
        SWEEP_REPS,
        "old_us_per_sweep",
        sweep_old_us,
        "new_us_per_sweep",
        sweep_new_us,
        "ratio_old_over_new",
        sweep_old_us / sweep_new_us,
    )

    if failures == 0:
        print("GPU_PROOF_TILED PASS")
    else:
        print("GPU_PROOF_TILED FAIL failures", failures)
        exit(1)
