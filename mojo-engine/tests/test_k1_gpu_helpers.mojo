# test_k1_gpu_helpers.mojo - host tests for the tuned K1 GPU
# kernel's pure helpers (lane mojo-a2-k1).
#
# The tuned kernel (src/kernels/k1_gemv.mojo) replaces the
# straight port's per-element work with three pure tricks, each
# proven here on the host against the original, already-proven
# routes before any card run is believed:
#
# 1. f16_to_f32_gpu (bit-trick decode) must equal f16_to_f32
#    (the exact arithmetic decode the CPU twin uses) on every
#    one of the 65,536 f16 patterns except the inf/NaN class,
#    where the two deliberately differ in payload handling and
#    weight files never carry them.
# 2. The packed scale extraction (3 header words -> the two
#    (scale, scale, min, min) quads per block half, llama.cpp's
#    construction as shipped in the wgpu r2sg kernel) must equal
#    scale_min_k4_list (ggml's byte route) for all 8 sub-blocks
#    of randomized super-block headers.
# 3. unpack_q_f32 / unpack4_f32 byte and nibble order: low byte
#    first, low nibble first, matching ggml's layout order.

from std.testing import TestSuite, assert_true, assert_equal

from kernels.k1_gemv import (
    f16_to_f32,
    f16_to_f32_gpu,
    scale_min_k4_list,
    unpack4_f32,
    unpack_q_f32,
)


def lcg_next(state: UInt32) -> UInt32:
    return state * 1664525 + 1013904223


def test_f16_gpu_matches_exact_decode() raises:
    var checked = 0
    for bits in range(65536):
        var b = UInt16(bits)
        var exp = (bits >> 10) & 0x1F
        if exp == 31:
            continue  # inf/NaN class: see the header note
        var exact = f16_to_f32(b)
        var fast = f16_to_f32_gpu(UInt32(bits))
        assert_equal(
            fast,
            exact,
            "f16 bit-trick disagrees with the exact decode",
        )
        checked += 1
    assert_true(checked > 60000, "f16 sweep must cover the space")


def sb_word(sb: List[UInt8], sb_off: Int, off: Int) -> UInt32:
    # Little-endian u32 over the super-block's bytes.
    return (
        UInt32(sb[sb_off + off])
        | (UInt32(sb[sb_off + off + 1]) << 8)
        | (UInt32(sb[sb_off + off + 2]) << 16)
        | (UInt32(sb[sb_off + off + 3]) << 24)
    )


def packed_scales(
    sb: List[UInt8], sb_off: Int, half: Int
) -> Tuple[SIMD[DType.float32, 4], SIMD[DType.float32, 4]]:
    # The kernel's packed extraction, replicated over host bytes.
    var s16 = (sb_word(sb, sb_off, 4) >> UInt32(16 * half)) & 0xFFFF
    var m16 = (sb_word(sb, sb_off, 8) >> UInt32(16 * half)) & 0xFFFF
    var h16 = (sb_word(sb, sb_off, 12) >> UInt32(16 * half)) & 0xFFFF
    var lo = (m16 << 16) | s16
    var sc_lo = unpack4_f32(lo & 0x3F3F3F3F)
    var hi = (((h16 << 12) | h16) & 0x0F0F0F0F) | ((lo & 0xC0C0C0C0) >> 2)
    var sc_hi = unpack4_f32(hi)
    return (sc_lo, sc_hi)


def test_packed_scales_match_byte_route() raises:
    var state: UInt32 = 12345
    for _trial in range(64):
        var sb = List[UInt8](capacity=144)
        sb.resize(144, UInt8(0))
        for i in range(144):
            state = lcg_next(state)
            sb[i] = UInt8((state >> 16) & 0xFF)
        for half in range(2):
            var packed = packed_scales(sb, 0, half)
            var lo = packed[0]
            var hi = packed[1]
            var j0 = scale_min_k4_list(2 * half, sb, 0)
            var j1 = scale_min_k4_list(2 * half + 1, sb, 0)
            var j4 = scale_min_k4_list(2 * half + 4, sb, 0)
            var j5 = scale_min_k4_list(2 * half + 5, sb, 0)
            assert_equal(lo[0], j0[0], "packed scale A")
            assert_equal(lo[1], j1[0], "packed scale B")
            assert_equal(lo[2], j0[1], "packed min A")
            assert_equal(lo[3], j1[1], "packed min B")
            assert_equal(hi[0], j4[0], "packed scale C")
            assert_equal(hi[1], j5[0], "packed scale D")
            assert_equal(hi[2], j4[1], "packed min C")
            assert_equal(hi[3], j5[1], "packed min D")


def test_unpack_order() raises:
    var bytes_ = unpack4_f32(UInt32(0x03020100))
    assert_equal(bytes_[0], Float32(0.0))
    assert_equal(bytes_[1], Float32(1.0))
    assert_equal(bytes_[2], Float32(2.0))
    assert_equal(bytes_[3], Float32(3.0))
    # Byte j of the word holds element j's low nibble in its low
    # half and element j+32's in its high half (ggml order).
    var q = unpack_q_f32(UInt32(0xA1B2C3D4))
    assert_equal(q[0][0], Float32(0x4))
    assert_equal(q[0][1], Float32(0x3))
    assert_equal(q[0][2], Float32(0x2))
    assert_equal(q[0][3], Float32(0x1))
    assert_equal(q[1][0], Float32(0xD))
    assert_equal(q[1][1], Float32(0xC))
    assert_equal(q[1][2], Float32(0xB))
    assert_equal(q[1][3], Float32(0xA))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
