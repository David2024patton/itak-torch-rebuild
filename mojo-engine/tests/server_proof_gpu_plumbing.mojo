# server_proof_gpu_plumbing.mojo - Lane C1 server proof for
# src/gpu/ (Wave C). NOT a TestSuite file: run explicitly on
# the GPU host (cleared-card window) with
#   mojo run -I src tests/server_proof_gpu_plumbing.mojo
#
# What it proves, in order:
# 1. The real Glimmer file opens through GlimmerModel and the
#    EXACT resident weights total is computed from the loader
#    (cross-checked against a raw byte sum computed here,
#    independently of the gpu.weights helper).
# 2. DeviceWeights builds under the config budget cap; the
#    blk.0.ffn_gate.weight device buffer downloads and hashes
#    (FNV-1a64) byte-exact against the loader's host bytes,
#    with a full byte-by-byte comparison behind the hash; one
#    F32 slot (blk.0.attn_norm.weight) downloads value-exact.
# 3. DeviceKvCache builds at full context (131,072) with the
#    contract total 3,653,238,784 B; the runtime's reserved
#    total (weights + KV + scratch envelope) prints against
#    the cap.
# 4. KV write/read sequences on device vs B1's CPU KvCache fed
#    identical formula rows, element-exact: a small synthetic
#    cache (window 4 + global sibling) and the real 2048
#    window at positions 0..2047 with boundary reads.
# 5. Wave D (Lane D4) update: the Wave C wrap limit is
#    RETIRED. At position 2048 on a 2048 window (and position
#    4 on the window-4 cache) the DEVICE append now ACCEPTS
#    and evicts exactly as B1's CPU cache does, and the
#    wrapped rows read back element-exact against the CPU
#    cache. The full wrap proof (visible-order gather,
#    generation past 2048) is tests/
#    server_proof_gpu_kv_wrap.mojo; this file keeps the
#    boundary legs at parity scale.
#
# Exits nonzero on any failure. Row formula (exact small
# integers in f32, distinct per position, same as B1's suite):
#   k element i of position p: (p * 131 + i * 17 + 11) % 997
#   v element i of position p: (p * 57 + i * 29 + 3) % 991

from std.sys import exit
from std.time import perf_counter_ns

from config import EngineConfig
from gpu.context import (
    GPU_SCRATCH_ENVELOPE_BYTES,
    GpuRuntime,
    download_f32,
    download_u8,
    gpu_budget_total,
)
from gpu.device_kv import DeviceKvCache
from gpu.weights import DeviceWeights, weights_total_bytes
from kv.cache import KvCache, LayerKind
from model.model import GlimmerModel


def _fnv1a64(data: List[UInt8]) -> UInt64:
    var h = UInt64(14695981039346656037)
    for i in range(len(data)):
        h = (h ^ UInt64(data[i])) * UInt64(1099511628211)
    return h


def _hex64(v: UInt64) -> String:
    var digits = "0123456789abcdef"
    var out = String("0x")
    for shift in range(60, -1, -4):
        var nib = Int((v >> UInt64(shift)) & 0xF)
        out += digits[byte = nib : nib + 1]
    return out


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


def _rows_equal(
    name: String, got: List[Float32], expected: List[Float32]
) -> Bool:
    if len(got) != len(expected):
        print("PROOF MISMATCH", name, "len", len(got), "vs", len(expected))
        return False
    var bad = 0
    for i in range(len(got)):
        if got[i] != expected[i]:
            bad += 1
    if bad != 0:
        print("PROOF MISMATCH", name, "elements differ:", bad)
        return False
    print("PROOF EXACT", name, len(got), "elements")
    return True


def run() raises -> Bool:
    var ok = True
    var cfg = EngineConfig.load()
    print("PROOF config model_path", cfg.model_path)
    print("PROOF config context_len", cfg.context_len)
    print("PROOF config gpu_budget_bytes", cfg.gpu_budget_bytes)

    # ---- 1. real model, exact weights total -------------------
    var t0 = perf_counter_ns()
    var model = GlimmerModel.open(cfg.model_path)
    var t_load = (perf_counter_ns() - t0) // 1_000_000
    print("PROOF model loaded ms", t_load, "layers", model.cfg.layers)
    var exact_weights = weights_total_bytes(model)
    print("PROOF weights_exact_bytes", exact_weights)
    print("PROOF weights_planning_bound 16756683904")
    if exact_weights > 16756683904:
        print("PROOF FAIL weights exact total over the planning bound")
        ok = False
    # Independent cross-check: the raw byte sum over every slot
    # plus the final-norm tensor's raw bytes, computed here
    # from the model's own fields (F32 slots are consistent
    # only when values x 4 == bytes, which weights_total_bytes
    # already refuses on; equality here proves the resident
    # total IS the file's tensor byte sum).
    var raw_sum = len(model.embd.bytes) + len(model.output.bytes)
    raw_sum += len(model.final_norm) * 4
    for layer in range(model.cfg.layers):
        for s in range(14):
            raw_sum += len(model.slot(layer, s).bytes)
    print("PROOF weights_raw_byte_sum", raw_sum)
    if raw_sum != exact_weights:
        print("PROOF FAIL raw byte sum disagrees with resident total")
        ok = False

    # ---- 2. device weights + hash proof -----------------------
    var rt = GpuRuntime.create(cfg.gpu_budget_bytes)
    print("PROOF runtime cap", rt.budget_cap_bytes)
    var t1 = perf_counter_ns()
    var weights = DeviceWeights.build(rt, model)
    var t_weights = (perf_counter_ns() - t1) // 1_000_000
    print("PROOF device_weights built ms", t_weights)
    print("PROOF device_weights slots", weights.slot_count())
    print("PROOF device_weights resident_bytes", weights.resident_bytes())
    if weights.resident_bytes() != exact_weights:
        print("PROOF FAIL device resident total != exact total")
        ok = False
    if rt.weights_bytes != exact_weights:
        print("PROOF FAIL runtime weights ledger != exact total")
        ok = False
    var embd_meta = weights.embedding_meta()
    var out_meta = weights.output_meta()
    var norm_meta = weights.final_norm_meta()
    print(
        "PROOF embd",
        embd_meta.name,
        "type",
        embd_meta.type_id,
        "rows",
        embd_meta.rows,
        "cols",
        embd_meta.cols,
    )
    print(
        "PROOF output",
        out_meta.name,
        "type",
        out_meta.type_id,
        "rows",
        out_meta.rows,
        "cols",
        out_meta.cols,
    )
    print(
        "PROOF final_norm",
        norm_meta.name,
        "cols",
        norm_meta.cols,
        "hidden",
        model.cfg.hidden,
    )
    if norm_meta.cols != model.cfg.hidden:
        print("PROOF FAIL final norm width != hidden")
        ok = False

    var host_slot = model.slot(0, 10)
    print("PROOF hash_slot host name", host_slot.name)
    var view = weights.slot_u8(0, 10)
    if view.name != "blk.0.ffn_gate.weight":
        print("PROOF FAIL slot (0,10) is not blk.0.ffn_gate.weight")
        ok = False
    if (
        view.type_id != host_slot.type_id
        or view.rows != host_slot.rows
        or view.cols != host_slot.cols
    ):
        print("PROOF FAIL slot (0,10) meta disagrees with the loader")
        ok = False
    var dev_bytes = download_u8(rt, view.buffer, len(host_slot.bytes))
    var byte_bad = 0
    if len(dev_bytes) != len(host_slot.bytes):
        byte_bad = -1
    else:
        for i in range(len(dev_bytes)):
            if dev_bytes[i] != host_slot.bytes[i]:
                byte_bad += 1
    var fnv_dev = _fnv1a64(dev_bytes)
    var fnv_host = _fnv1a64(host_slot.bytes)
    print("PROOF hash_slot bytes", len(host_slot.bytes))
    print("PROOF hash_slot fnv1a64_device", _hex64(fnv_dev))
    print("PROOF hash_slot fnv1a64_host", _hex64(fnv_host))
    print("PROOF hash_slot byte_mismatches", byte_bad)
    if byte_bad != 0 or fnv_dev != fnv_host:
        print("PROOF FAIL blk.0.ffn_gate.weight device bytes differ")
        ok = False
    else:
        print("PROOF HASH PASS blk.0.ffn_gate.weight byte-exact")

    # F32 route: blk.0.attn_norm.weight (suffix 0) is an F32
    # slot; its device float32 buffer must equal the values.
    var f32_host = model.slot(0, 0)
    var f32_view = weights.slot_f32(0, 0)
    var f32_dev = download_f32(rt, f32_view.buffer, len(f32_host.values))
    if not _rows_equal("blk.0.attn_norm.weight f32", f32_dev, f32_host.values):
        ok = False

    # ---- 3. full-context device KV + budget ledger ------------
    var t2 = perf_counter_ns()
    var kv = DeviceKvCache.build(
        rt,
        model.cfg.kinds.copy(),
        model.cfg.kv_heads,
        model.cfg.head_dim,
        cfg.context_len,
    )
    var t_kv = (perf_counter_ns() - t2) // 1_000_000
    print("PROOF device_kv full built ms", t_kv)
    print("PROOF device_kv total_bytes", kv.resident_bytes())
    if kv.resident_bytes() != 3653238784:
        print("PROOF FAIL full KV total != 3653238784")
        ok = False
    print("PROOF kv capacity layer0 (windowed)", kv.capacity_positions(0))
    print("PROOF kv capacity layer3 (global)", kv.capacity_positions(3))
    if (
        kv.capacity_positions(0) != 2048
        or kv.capacity_positions(3) != cfg.context_len
    ):
        print("PROOF FAIL KV capacities wrong")
        ok = False
    var reserved = rt.total_reserved()
    var expected_reserved = gpu_budget_total(
        exact_weights, 3653238784, GPU_SCRATCH_ENVELOPE_BYTES
    )
    print("PROOF reserved_total", reserved)
    print("PROOF reserved_expected (full kv)", expected_reserved)
    print("PROOF budget_cap", rt.budget_cap_bytes)
    if reserved < expected_reserved or reserved > rt.budget_cap_bytes:
        print("PROOF FAIL reserved total outside [expected, cap]")
        ok = False

    # ---- 4a. synthetic KV: window 4 + global sibling ----------
    var pattern_s = List[LayerKind]()
    pattern_s.append(LayerKind.make_windowed(4))
    pattern_s.append(LayerKind.make_global())
    var dev_s = DeviceKvCache.build(rt, pattern_s.copy(), 2, 4, 64)
    var cpu_s = KvCache.new(pattern_s.copy(), 2, 4)
    for pos in range(4):
        var rows = _rows(2, 4, pos)
        for layer in range(2):
            dev_s.append(rt, layer, rows[0], rows[1], pos)
            cpu_s.append(layer, rows[0], rows[1], pos)
    for layer in range(2):
        for pos in range(4):
            var dk = dev_s.read_key_row(rt, layer, pos)
            var ck = cpu_s.key_row(layer, pos)
            if not _rows_equal(
                "synthetic K layer " + String(layer) + " pos " + String(pos),
                dk,
                ck,
            ):
                ok = False
            var dv = dev_s.read_value_row(rt, layer, pos)
            var cv = cpu_s.value_row(layer, pos)
            if not _rows_equal(
                "synthetic V layer " + String(layer) + " pos " + String(pos),
                dv,
                cv,
            ):
                ok = False
    print(
        "PROOF synthetic stored_len w",
        dev_s.stored_len(0),
        "g",
        dev_s.stored_len(1),
        "first_visible w",
        dev_s.first_visible_position(0),
    )
    # Wave D: the tiny window WRAPS on device, B1-exact.
    # Both caches accept position 4 and evict position 0;
    # every visible row then compares element-exact, and the
    # evicted position refuses on device by its B1 name.
    var rows4 = _rows(2, 4, 4)
    dev_s.append(rt, 0, rows4[0], rows4[1], 4)
    cpu_s.append(0, rows4[0], rows4[1], 4)
    print(
        "PROOF B1_PARITY window-4 position 4 accepted on both: stored",
        dev_s.stored_len(0),
        "first_visible",
        dev_s.first_visible_position(0),
        "cpu stored",
        cpu_s.stored_len(0),
        "cpu first_visible",
        cpu_s.first_visible_position(0),
    )
    if dev_s.stored_len(0) != cpu_s.stored_len(
        0
    ) or dev_s.first_visible_position(0) != cpu_s.first_visible_position(0):
        print("PROOF FAIL window-4 wrap stored/first_visible disagree")
        ok = False
    for pos in range(1, 5):
        var dk_w = dev_s.read_key_row(rt, 0, pos)
        var ck_w = cpu_s.key_row(0, pos)
        if not _rows_equal("wrapped window-4 K pos " + String(pos), dk_w, ck_w):
            ok = False
        var dv_w = dev_s.read_value_row(rt, 0, pos)
        var cv_w = cpu_s.value_row(0, pos)
        if not _rows_equal("wrapped window-4 V pos " + String(pos), dv_w, cv_w):
            ok = False
    var saw_evicted_s = False
    try:
        _ = dev_s.read_key_row(rt, 0, 0)
    except e:
        saw_evicted_s = String(e).find("gpu_kv_position_evicted") != -1
    if not saw_evicted_s:
        print("PROOF FAIL wrapped window-4 evicted read not refused by name")
        ok = False
    # Global sibling continues in order on device, exact.
    for pos in range(4, 6):
        var rows_g = _rows(2, 4, pos)
        dev_s.append(rt, 1, rows_g[0], rows_g[1], pos)
        cpu_s.append(1, rows_g[0], rows_g[1], pos)
        var dk2 = dev_s.read_key_row(rt, 1, pos)
        if not _rows_equal(
            "synthetic global K pos " + String(pos), dk2, rows_g[0]
        ):
            ok = False
    # Named refusals on the device append/read paths.
    var saw_order = False
    var rows99 = _rows(2, 4, 99)
    try:
        dev_s.append(rt, 1, rows99[0], rows99[1], 99)
    except e:
        saw_order = String(e).find("gpu_kv_out_of_order_position") != -1
    if not saw_order:
        print("PROOF FAIL out-of-order append not refused by name")
        ok = False
    var saw_shape = False
    var short = List[Float32](capacity=4)
    short.resize(4, Float32(0.0))
    var rows6 = _rows(2, 4, 6)
    try:
        dev_s.append(rt, 1, short, rows6[1], 6)
    except e:
        saw_shape = String(e).find("gpu_kv_shape_mismatch") != -1
    if not saw_shape:
        print("PROOF FAIL short row not refused gpu_kv_shape_mismatch")
        ok = False
    var saw_unwritten = False
    try:
        _ = dev_s.read_key_row(rt, 1, 60)
    except e:
        saw_unwritten = String(e).find("gpu_kv_position_not_written") != -1
    if not saw_unwritten:
        print("PROOF FAIL unwritten read not refused by name")
        ok = False
    print("PROOF synthetic refusals order/shape/unwritten named")

    # ---- 4b. real 2048 window, positions 0..2047 --------------
    var pattern_r = List[LayerKind]()
    pattern_r.append(LayerKind.make_windowed(2048))
    pattern_r.append(LayerKind.make_global())
    var dev_r = DeviceKvCache.build(rt, pattern_r.copy(), 2, 128, 4096)
    var cpu_r = KvCache.new(pattern_r.copy(), 2, 128)
    var t3 = perf_counter_ns()
    for pos in range(2048):
        var rows_r = _rows(2, 128, pos)
        dev_r.append(rt, 0, rows_r[0], rows_r[1], pos)
        cpu_r.append(0, rows_r[0], rows_r[1], pos)
    var t_appends = (perf_counter_ns() - t3) // 1_000_000
    print("PROOF real-window 2048 device+cpu appends ms", t_appends)
    for pos in range(8):
        var rows_rg = _rows(2, 128, pos)
        dev_r.append(rt, 1, rows_rg[0], rows_rg[1], pos)
        cpu_r.append(1, rows_rg[0], rows_rg[1], pos)
    print(
        "PROOF real-window stored_len",
        dev_r.stored_len(0),
        "first_visible",
        dev_r.first_visible_position(0),
        "next",
        dev_r.next_position(0),
    )
    if dev_r.stored_len(0) != 2048 or dev_r.first_visible_position(0) != 0:
        print("PROOF FAIL real-window stored/first-visible wrong")
        ok = False
    for pos in [0, 1, 777, 2046, 2047]:
        var dk3 = dev_r.read_key_row(rt, 0, pos)
        var ck3 = cpu_r.key_row(0, pos)
        if not _rows_equal("real-window K pos " + String(pos), dk3, ck3):
            ok = False
        var dv3 = dev_r.read_value_row(rt, 0, pos)
        var cv3 = cpu_r.value_row(0, pos)
        if not _rows_equal("real-window V pos " + String(pos), dv3, cv3):
            ok = False
    for pos in [0, 7]:
        var dk4 = dev_r.read_key_row(rt, 1, pos)
        var ck4 = cpu_r.key_row(1, pos)
        if not _rows_equal("real-window global K pos " + String(pos), dk4, ck4):
            ok = False
    # Boundary (Wave D): BOTH caches accept 2048/2049 and
    # evict 0/1; the wrapped device rows read back
    # element-exact against the CPU cache.
    var rows2048 = _rows(2, 128, 2048)
    dev_r.append(rt, 0, rows2048[0], rows2048[1], 2048)
    cpu_r.append(0, rows2048[0], rows2048[1], 2048)
    var rows2049 = _rows(2, 128, 2049)
    dev_r.append(rt, 0, rows2049[0], rows2049[1], 2049)
    cpu_r.append(0, rows2049[0], rows2049[1], 2049)
    print(
        "PROOF B1_PARITY both accepted 2048/2049: device stored",
        dev_r.stored_len(0),
        "first_visible",
        dev_r.first_visible_position(0),
        "cpu stored",
        cpu_r.stored_len(0),
        "cpu first_visible",
        cpu_r.first_visible_position(0),
    )
    if dev_r.stored_len(0) != cpu_r.stored_len(
        0
    ) or dev_r.first_visible_position(0) != cpu_r.first_visible_position(0):
        print("PROOF FAIL real-window wrap stored/first_visible disagree")
        ok = False
    for pos in [2047, 2048, 2049]:
        var dk5 = dev_r.read_key_row(rt, 0, pos)
        var ck5 = cpu_r.key_row(0, pos)
        if not _rows_equal(
            "wrapped real-window K pos " + String(pos), dk5, ck5
        ):
            ok = False
        var dv5 = dev_r.read_value_row(rt, 0, pos)
        var cv5 = cpu_r.value_row(0, pos)
        if not _rows_equal(
            "wrapped real-window V pos " + String(pos), dv5, cv5
        ):
            ok = False

    print("PROOF reserved_total_final", rt.total_reserved())
    return ok


def main():
    var ok = False
    try:
        ok = run()
    except e:
        print("PROOF REFUSAL", e)
        exit(2)
    if not ok:
        print("PROOF GPU_PLUMBING FAIL")
        exit(1)
    print("PROOF GPU_PLUMBING PASS")
