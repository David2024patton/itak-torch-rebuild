# test_q5k_gpu_helpers.mojo - host tests for the tuned Q5_K
# GPU kernel's pure helpers (lane mojo-d3-gemv, Wave D).
#
# The tuned kernel (src/kernels/q5k_gemv.mojo) replaces the
# straight port's per-element dequant with three pure tricks,
# each proven here on the host against the original,
# already-proven routes before any card run is believed (the
# test_k1_gpu_helpers.mojo pattern):
#
# 1. f16_to_f32_gpu (bit-trick decode) must equal
#    kernels.dequant.f16_to_f32 on every f16 pattern except
#    the inf/NaN class (see test_q6k_gpu_helpers for the
#    standing note).
# 2. q5k_packed_scales (header words 1..3 -> the two
#    (scale, scale, min, min) quads per block half) must
#    equal dequant.scale_min_k4_list for all 8 sub-blocks of
#    randomized super-block headers.
# 3. q5k_qh_bit_quads (the fifth-bit planes) must reproduce,
#    for both block halves, the per-element bit route:
#    element j of quad k is bit s of qh byte j, where s runs
#    over the thread's sub-blocks 2h, 2h+1, 2h+4, 2h+5.

from std.testing import TestSuite, assert_true, assert_equal

from kernels import dequant
from kernels.q5k_gemv import f16_to_f32_gpu, q5k_packed_scales, q5k_qh_bit_quads


def lcg_next(state: UInt32) -> UInt32:
    return state * 1664525 + 1013904223


def test_f16_gpu_matches_exact_decode() raises:
    var checked = 0
    for bits in range(65536):
        var exp = (bits >> 10) & 0x1F
        if exp == 31:
            continue  # inf/NaN class: see the header note
        var exact = dequant.f16_to_f32(UInt16(bits))
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


def test_packed_scales_match_byte_route() raises:
    var state: UInt32 = 54321
    for _trial in range(64):
        var sb = List[UInt8](capacity=176)
        sb.resize(176, UInt8(0))
        for i in range(176):
            state = lcg_next(state)
            sb[i] = UInt8((state >> 16) & 0xFF)
        for half in range(2):
            var packed = q5k_packed_scales(
                sb_word(sb, 0, 4), sb_word(sb, 0, 8), sb_word(sb, 0, 12), half
            )
            var lo = packed[0]
            var hi = packed[1]
            var j0 = dequant.scale_min_k4_list(2 * half, sb, 0)
            var j1 = dequant.scale_min_k4_list(2 * half + 1, sb, 0)
            var j4 = dequant.scale_min_k4_list(2 * half + 4, sb, 0)
            var j5 = dequant.scale_min_k4_list(2 * half + 5, sb, 0)
            assert_equal(lo[0], j0[0], "packed scale A")
            assert_equal(lo[1], j1[0], "packed scale B")
            assert_equal(lo[2], j0[1], "packed min A")
            assert_equal(lo[3], j1[1], "packed min B")
            assert_equal(hi[0], j4[0], "packed scale C")
            assert_equal(hi[1], j5[0], "packed scale D")
            assert_equal(hi[2], j4[1], "packed min C")
            assert_equal(hi[3], j5[1], "packed min D")


def test_qh_bit_quads_match_element_route() raises:
    var state: UInt32 = 777
    for _trial in range(64):
        state = lcg_next(state)
        var qhw = state
        # The word's four bytes, low byte first (position j).
        var b0 = Int(qhw & 0xFF)
        var b1 = Int((qhw >> 8) & 0xFF)
        var b2 = Int((qhw >> 16) & 0xFF)
        var b3 = Int((qhw >> 24) & 0xFF)
        var bytes_ = List[Int](capacity=4)
        bytes_.append(b0)
        bytes_.append(b1)
        bytes_.append(b2)
        bytes_.append(b3)
        for half in range(2):
            var quads = q5k_qh_bit_quads(qhw, half)
            var subs = List[Int](capacity=4)
            subs.append(2 * half)
            subs.append(2 * half + 1)
            subs.append(2 * half + 4)
            subs.append(2 * half + 5)
            for k in range(4):
                var s = subs[k]
                for j in range(4):
                    var expected = Float32((bytes_[j] >> s) & 1)
                    var got = quads[0][j] if k == 0 else (
                        quads[1][j] if k
                        == 1 else (quads[2][j] if k == 2 else quads[3][j])
                    )
                    assert_equal(
                        got,
                        expected,
                        "qh bit quad disagrees with the element route",
                    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
