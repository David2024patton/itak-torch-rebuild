# server_proof_gpu_kv_wrap.mojo - Lane D4 (Wave D) server
# proof for the device KV window wrap. NOT a TestSuite file:
# run explicitly on a GPU host with
#   mojo run -I src tests/server_proof_gpu_kv_wrap.mojo
#
# What it proves, in order:
# 1. SYNTHETIC WINDOW-4 WRAPAROUND REPLAY: positions 0..11
#    (three ring wraps) appended to DeviceKvCache and B1's
#    CPU KvCache fed identical formula rows; after EVERY
#    append, every visible row compares ELEMENT-EXACT
#    (device read-back and the device visible-order gather
#    buffer vs the CPU cache's rows in absolute-position
#    order), stored_len / first_visible_position agree, and
#    an evicted position refuses gpu_kv_position_evicted.
# 2. REAL 2048 WINDOW: positions 0..2059 on the Glimmer KV
#    geometry (2 heads x 128) - reads at 2047/2048/2049 and
#    beyond ELEMENT-EXACT vs the CPU cache, the full visible
#    gather (2048 rows) element-exact, eviction refusals at
#    the boundary, and a global sibling carried along.
# 3. SYNTHETIC GENERATION PAST 2048 (named synthetic): a
#    decode loop on a small synthetic config (window 2048,
#    2 KV heads x 32, 4 Q heads) stepping positions 0..2063
#    through the RESIDENT device path - append_device +
#    attention_device_resident (the gpu_layer_step_resident
#    attention leg) - with NO CPU fallback and NO
#    gpu_kv_window_wrapped anywhere; at sampled positions
#    (including 2047/2048/2049/2063) the device attention
#    output compares against the CPU twin
#    (attention_cpu_rotated_cache over B1's visible rows,
#    ROPE_NONE: the synthetic rows are pre-rotated by
#    construction) at cosine >= 0.999999, max rel <= 1e-4.
#    This is a synthetic-config generation run; the real
#    model at that length is a coordinator-window decision.
#
# Row formula for parts 1-2 (exact small integers in f32,
# B1's suite formula): k element i of position p:
# (p * 131 + i * 17 + 11) % 997; v: (p * 57 + i * 29 + 3) % 991.
# Part 3 uses small-magnitude formulas so the softmax is
# exercised, not saturated.
#
# Exits nonzero on any miss.

from std.math import sqrt
from std.sys import exit
from std.time import perf_counter_ns

from attention.attention import attention_cpu_rotated_cache, ROPE_NONE
from gpu.context import GpuRuntime, download_f32, upload_f32
from gpu.device_kv import DeviceKvCache
from kv.cache import KvCache, LayerKind
from model.gpu_layer import attention_device_resident

comptime GPU_BUDGET_CAP = 24696061952


def _k_expected(position: Int, i: Int) -> Float32:
    return Float32((position * 131 + i * 17 + 11) % 997)


def _v_expected(position: Int, i: Int) -> Float32:
    return Float32((position * 57 + i * 29 + 3) % 991)


def _rows(
    kv_heads: Int, head_dim: Int, position: Int
) -> Tuple[List[Float32], List[Float32]]:
    var row = kv_heads * head_dim
    var k = List[Float32](capacity=row)
    var v = List[Float32](capacity=row)
    for i in range(row):
        k.append(_k_expected(position, i))
        v.append(_v_expected(position, i))
    return (k^, v^)


def _exact(name: String, got: List[Float32], expected: List[Float32]) -> Bool:
    if len(got) != len(expected):
        print("WRAP MISMATCH", name, "len", len(got), "vs", len(expected))
        return False
    var bad = 0
    for i in range(len(got)):
        if got[i] != expected[i]:
            bad += 1
    if bad != 0:
        print("WRAP MISMATCH", name, "elements differ:", bad)
        return False
    print("WRAP EXACT", name, len(got), "elements")
    return True


def _visible_flat(
    cpu: KvCache, layer: Int
) raises -> Tuple[List[Float32], List[Float32]]:
    """The CPU cache's visible rows in absolute-position
    order, flat [stored][row] for K and V."""
    var first = cpu.first_visible_position(layer)
    var stored = cpu.stored_len(layer)
    var row = cpu.row_len()
    var keys = List[Float32](capacity=stored * row)
    var values = List[Float32](capacity=stored * row)
    for vi in range(stored):
        var kr = cpu.key_row(layer, first + vi)
        var vr = cpu.value_row(layer, first + vi)
        for i in range(row):
            keys.append(kr[i])
            values.append(vr[i])
    return (keys^, values^)


def _compare(
    got: List[Float32], expected: List[Float32]
) -> Tuple[Float64, Float64]:
    var dot = Float64(0.0)
    var ng = Float64(0.0)
    var ne = Float64(0.0)
    var max_abs = Float64(0.0)
    var max_e = Float64(0.0)
    for i in range(len(expected)):
        var g = Float64(got[i])
        var e = Float64(expected[i])
        dot += g * e
        ng += g * g
        ne += e * e
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
    return (dot / (sqrt(ng) * sqrt(ne)), max_abs / max_e)


def run() raises -> Bool:
    var ok = True
    var rt = GpuRuntime.create(GPU_BUDGET_CAP)

    # ---- 1. synthetic window-4 wraparound replay ----------
    var pattern_s = List[LayerKind]()
    pattern_s.append(LayerKind.make_windowed(4))
    pattern_s.append(LayerKind.make_global())
    var dev_s = DeviceKvCache.build(rt, pattern_s.copy(), 2, 4, 64)
    var cpu_s = KvCache.new(pattern_s.copy(), 2, 4)
    for pos in range(12):
        var rows = _rows(2, 4, pos)
        dev_s.append(rt, 0, rows[0], rows[1], pos)
        cpu_s.append(0, rows[0], rows[1], pos)
        if dev_s.stored_len(0) != cpu_s.stored_len(0):
            print("WRAP FAIL stored_len disagrees at pos", pos)
            ok = False
        if dev_s.first_visible_position(0) != cpu_s.first_visible_position(0):
            print("WRAP FAIL first_visible disagrees at pos", pos)
            ok = False
        var flat = _visible_flat(cpu_s, 0)
        var gk = download_f32(
            rt, dev_s.gather_visible(rt, 0, False), len(flat[0])
        )
        if not _exact(
            "synthetic gather K after pos " + String(pos), gk, flat[0]
        ):
            ok = False
        var gv = download_f32(
            rt, dev_s.gather_visible(rt, 0, True), len(flat[1])
        )
        if not _exact(
            "synthetic gather V after pos " + String(pos), gv, flat[1]
        ):
            ok = False
        var first = cpu_s.first_visible_position(0)
        for vi in range(cpu_s.stored_len(0)):
            var dk = dev_s.read_key_row(rt, 0, first + vi)
            var ck = cpu_s.key_row(0, first + vi)
            if not _exact("synthetic read K pos " + String(first + vi), dk, ck):
                ok = False
    # Evicted positions refuse by name on device, as on CPU.
    var saw_evicted = False
    try:
        _ = dev_s.read_key_row(rt, 0, 0)
    except e:
        saw_evicted = String(e).find("gpu_kv_position_evicted") != -1
        print("WRAP synthetic evicted refusal:", e)
    if not saw_evicted:
        print("WRAP FAIL evicted read not refused gpu_kv_position_evicted")
        ok = False
    print(
        "WRAP synthetic window-4 replay done: stored",
        dev_s.stored_len(0),
        "first_visible",
        dev_s.first_visible_position(0),
    )

    # ---- 2. real 2048 window, positions 0..2059 ----------
    var pattern_r = List[LayerKind]()
    pattern_r.append(LayerKind.make_windowed(2048))
    pattern_r.append(LayerKind.make_global())
    var dev_r = DeviceKvCache.build(rt, pattern_r.copy(), 2, 128, 4096)
    var cpu_r = KvCache.new(pattern_r.copy(), 2, 128)
    var t0 = perf_counter_ns()
    for pos in range(2060):
        var rows_r = _rows(2, 128, pos)
        dev_r.append(rt, 0, rows_r[0], rows_r[1], pos)
        cpu_r.append(0, rows_r[0], rows_r[1], pos)
        dev_r.append(rt, 1, rows_r[0], rows_r[1], pos)
        cpu_r.append(1, rows_r[0], rows_r[1], pos)
    print(
        "WRAP real-window 2060 appends x2 layers ms",
        (perf_counter_ns() - t0) // 1_000_000,
    )
    print(
        "WRAP real-window stored",
        dev_r.stored_len(0),
        "first_visible",
        dev_r.first_visible_position(0),
        "cpu stored",
        cpu_r.stored_len(0),
        "cpu first_visible",
        cpu_r.first_visible_position(0),
    )
    if dev_r.stored_len(0) != 2048 or dev_r.first_visible_position(0) != 12:
        print("WRAP FAIL real-window stored/first_visible wrong")
        ok = False
    for pos in [12, 100, 2047, 2048, 2049, 2059]:
        var dk = dev_r.read_key_row(rt, 0, pos)
        var ck = cpu_r.key_row(0, pos)
        if not _exact("real-window K pos " + String(pos), dk, ck):
            ok = False
        var dv = dev_r.read_value_row(rt, 0, pos)
        var cv = cpu_r.value_row(0, pos)
        if not _exact("real-window V pos " + String(pos), dv, cv):
            ok = False
    for pos in [0, 2048, 2059]:
        var dg = dev_r.read_key_row(rt, 1, pos)
        var cg = cpu_r.key_row(1, pos)
        if not _exact("real-window global K pos " + String(pos), dg, cg):
            ok = False
    var flat_r = _visible_flat(cpu_r, 0)
    var gk_r = download_f32(
        rt, dev_r.gather_visible(rt, 0, False), len(flat_r[0])
    )
    if not _exact("real-window full gather K (2048 rows)", gk_r, flat_r[0]):
        ok = False
    var gv_r = download_f32(
        rt, dev_r.gather_visible(rt, 0, True), len(flat_r[1])
    )
    if not _exact("real-window full gather V (2048 rows)", gv_r, flat_r[1]):
        ok = False
    var saw_evicted_real = False
    try:
        _ = dev_r.read_key_row(rt, 0, 11)
    except e:
        saw_evicted_real = String(e).find("gpu_kv_position_evicted") != -1
        print("WRAP real-window evicted refusal:", e)
    if not saw_evicted_real:
        print("WRAP FAIL real-window evicted read not refused by name")
        ok = False

    # ---- 3. synthetic generation past 2048 ----------------
    # Small synthetic config: window 2048, kv_heads 2,
    # head_dim 32, q_heads 4. Deterministic small-magnitude
    # rows; the loop is a decode loop (append, then attend)
    # driven entirely on device through the resident path.
    var pattern_g = List[LayerKind]()
    pattern_g.append(LayerKind.make_windowed(2048))
    var dev_g = DeviceKvCache.build(rt, pattern_g.copy(), 2, 32, 4096)
    var cpu_g = KvCache.new(pattern_g.copy(), 2, 32)
    var kv_row = 2 * 32
    var q_row = 4 * 32
    var check_at = List[Int]()
    for p in [0, 1, 100, 1024, 2047, 2048, 2049, 2050, 2060, 2063]:
        check_at.append(p)
    var gen_ok = True
    for pos in range(2064):
        var k = List[Float32](capacity=kv_row)
        var v = List[Float32](capacity=kv_row)
        var q = List[Float32](capacity=q_row)
        for i in range(kv_row):
            k.append(Float32((pos * 131 + i * 17 + 11) % 23) * Float32(0.05))
            v.append(Float32((pos * 57 + i * 29 + 3) % 19) * Float32(0.05))
        for i in range(q_row):
            q.append(Float32((pos * 31 + i * 7 + 5) % 29) * Float32(0.05))
        var k_dev = upload_f32(rt, k)
        var v_dev = upload_f32(rt, v)
        dev_g.append_device(rt, 0, k_dev, v_dev, pos)
        cpu_g.append(0, k, v, pos)
        var want_check = False
        for ci in range(len(check_at)):
            if check_at[ci] == pos:
                want_check = True
        if want_check:
            var q_dev = upload_f32(rt, q)
            var attn_dev = attention_device_resident(
                rt, dev_g, q_dev, 0, 4, 2, 32
            )
            var got = download_f32(rt, attn_dev, q_row)
            var flat_g = _visible_flat(cpu_g, 0)
            var expected = attention_cpu_rotated_cache(
                q,
                flat_g[0],
                flat_g[1],
                cpu_g.first_visible_position(0),
                cpu_g.stored_len(0),
                4,
                2,
                32,
                pos,
                ROPE_NONE,
            )
            var cmp = _compare(got, expected)
            var pok = cmp[0] >= 0.999999 and cmp[1] <= 1.0e-4
            gen_ok = gen_ok and pok
            print(
                "SYNTH_GEN pos="
                + String(pos)
                + " stored="
                + String(dev_g.stored_len(0))
                + " first_visible="
                + String(dev_g.first_visible_position(0))
                + " cosine="
                + String(cmp[0])
                + " max_rel="
                + String(cmp[1])
                + " ok="
                + String(1 if pok else 0)
            )
    ok = ok and gen_ok
    print(
        "SYNTHETIC_GENERATION_PAST_2048 positions 0..2063 on device,"
        " no CPU fallback, no wrap refusal: "
        + ("PASS" if gen_ok else "FAIL")
    )
    return ok


def main():
    var ok = False
    try:
        ok = run()
    except e:
        print("WRAP PROOF REFUSAL", e)
        exit(2)
    if not ok:
        print("WRAP PROOF FAIL")
        exit(1)
    print("WRAP PROOF PASS")
