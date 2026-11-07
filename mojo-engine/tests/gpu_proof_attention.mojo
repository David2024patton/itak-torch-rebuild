# gpu_proof_attention.mojo - server-card proof driver for
# Lane B2 attention.
#
# NOT a TestSuite file: run explicitly on a GPU host with
#   mojo run -I src tests/gpu_proof_attention.mojo
# Fixture: one synthetic sliding layer at Glimmer geometry
# (32 Q heads, 2 KV heads, head_dim 128), 64 visible
# positions, query at position 63, RoPE base 500000.
# The device buffers carry PRE-ROTATED Q/K (the kernel's
# contract, mirroring Rust gqa_attention): rotation is done
# host-side with Wave A's rope_cpu, Q at position 63 and each
# K row at its own position. The reference is attention_cpu
# on the RAW inputs (which rotates at read time inside), so
# this proof covers the whole path: raw -> rotated -> device
# kernel vs raw -> CPU twin. Bars: cosine >= 0.999999999 and
# rel (max abs diff / max abs expected) <= 1e-5. Exits
# nonzero on any miss. On a GPU-less host DeviceContext
# creation fails loudly; that is a host fact, not a pass.

from std.math import sqrt
from std.sys import exit
from max.gpu.host import DeviceContext

from attention.attention import attention_cpu, attention_kernel, ATTN_THREADS
from ops.rope import rope_cpu


def _det(i: Int) -> Float32:
    return Float32(((i * 37 + 13) % 31) - 15) / Float32(8.0)


def _make(n: Int, seed: Int) -> List[Float32]:
    var x = List[Float32](capacity=n)
    x.resize(n, Float32(0.0))
    for i in range(n):
        x[i] = _det(i + seed * 101)
    return x^


def main() raises:
    comptime NQ = 32
    comptime NKV = 2
    comptime HD = 128
    comptime STORED = 64
    comptime POS = 63
    comptime BASE = Float32(500000.0)
    var q = _make(NQ * HD, 41)
    var k = _make(STORED * NKV * HD, 42)
    var v = _make(STORED * NKV * HD, 43)

    # Reference: CPU twin on raw inputs.
    var expected = attention_cpu(q, k, v, 0, STORED, NQ, NKV, HD, POS, BASE)

    # Pre-rotate for the device, exactly the Rust write-time
    # step: Q at POS, each K row at its own absolute position.
    var q_pos = List[Int](capacity=NQ)
    q_pos.resize(NQ, POS)
    var q_rot = rope_cpu(q, q_pos, HD, BASE)
    var k_pos = List[Int](capacity=STORED * NKV)
    k_pos.resize(STORED * NKV, 0)
    for i in range(STORED):
        for kv in range(NKV):
            k_pos[i * NKV + kv] = i
    var k_rot = rope_cpu(k, k_pos, HD, BASE)

    var ctx = DeviceContext()
    var q_dev = ctx.enqueue_create_buffer[DType.float32](NQ * HD)
    var k_dev = ctx.enqueue_create_buffer[DType.float32](STORED * NKV * HD)
    var v_dev = ctx.enqueue_create_buffer[DType.float32](STORED * NKV * HD)
    var o_dev = ctx.enqueue_create_buffer[DType.float32](NQ * HD)
    var q_host = ctx.enqueue_create_host_buffer[DType.float32](NQ * HD)
    var k_host = ctx.enqueue_create_host_buffer[DType.float32](
        STORED * NKV * HD
    )
    var v_host = ctx.enqueue_create_host_buffer[DType.float32](
        STORED * NKV * HD
    )
    for i in range(NQ * HD):
        q_host[i] = q_rot[i]
    for i in range(STORED * NKV * HD):
        k_host[i] = k_rot[i]
        v_host[i] = v[i]
    ctx.enqueue_copy(src_buf=q_host, dst_buf=q_dev)
    ctx.enqueue_copy(src_buf=k_host, dst_buf=k_dev)
    ctx.enqueue_copy(src_buf=v_host, dst_buf=v_dev)
    var scale = Float32(1.0) / sqrt(Float32(HD))
    ctx.enqueue_function[attention_kernel](
        q_dev,
        k_dev,
        v_dev,
        o_dev,
        Int32(STORED),
        Int32(NKV),
        Int32(HD),
        Int32(NQ // NKV),
        scale,
        grid_dim=NQ,
        block_dim=ATTN_THREADS,
    )
    ctx.synchronize()
    var o_host = ctx.enqueue_create_host_buffer[DType.float32](NQ * HD)
    ctx.enqueue_copy(src_buf=o_dev, dst_buf=o_host)
    ctx.synchronize()

    var dot = Float64(0.0)
    var norm_g = Float64(0.0)
    var norm_e = Float64(0.0)
    var max_abs = Float64(0.0)
    var max_e = Float64(0.0)
    for i in range(NQ * HD):
        var g = Float64(o_host[i])
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
    var rel = max_abs / max_e
    print(
        "GPU_PROOF attention 32q/2kv hd128 stored64 pos63 rope500k:",
        "cosine",
        cosine,
        "rel",
        rel,
        "max_abs",
        max_abs,
    )
    if cosine >= 0.999999999 and rel <= 1.0e-5:
        print("GPU_PROOF PASS")
    else:
        print("GPU_PROOF FAIL")
        exit(1)
