# test_q6k_gpu_helpers.mojo - host tests for the tuned Q6_K
# GPU kernel's pure helpers (lane mojo-d3-gemv, Wave D).
#
# The tuned kernel (src/kernels/q6k_gemv.mojo) replaces the
# straight port's per-element dequant with three pure tricks,
# each proven here on the host against the original,
# already-proven routes before any card run is believed (the
# test_k1_gpu_helpers.mojo pattern):
#
# 1. f16_to_f32_gpu (bit-trick decode) must equal
#    kernels.dequant.f16_to_f32 (the exact arithmetic decode
#    the CPU twin uses) on every f16 pattern except the
#    inf/NaN class, where the two deliberately differ in
#    payload handling and weight files never carry them.
# 2. q6k_stream_quads (the SIMD quad reconstruction: nibble
#    plane plus qh bit-pair plane, ggml's masks) must
#    reproduce, for every thread column of both block
#    halves, the centered quant (q - 32) that the
#    per-element route kernels.dequant.q6k_element_list
#    dequantizes, element for element.
# 3. load_word (the u32 rebuilt from two aligned u16 halves
#    that lets the kernel read the raw 210-byte layout) must
#    equal the plain little-endian byte assembly at even
#    offsets and at odd super-block starts (byte 210, 420).

from std.testing import TestSuite, assert_true, assert_equal

from kernels import dequant
from kernels.q6k_gemv import f16_to_f32_gpu, load_word, q6k_stream_quads


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


def byte_word(sb: List[UInt8], off: Int) -> UInt32:
    # Plain little-endian u32 over the block's bytes.
    return (
        UInt32(sb[off])
        | (UInt32(sb[off + 1]) << 8)
        | (UInt32(sb[off + 2]) << 16)
        | (UInt32(sb[off + 3]) << 24)
    )


def test_load_word_matches_byte_assembly() raises:
    var sb = List[UInt8](capacity=630)
    sb.resize(630, UInt8(0))
    var state: UInt32 = 999
    for i in range(len(sb)):
        state = lcg_next(state)
        sb[i] = UInt8((state >> 16) & 0xFF)
    var w16 = sb.unsafe_ptr().unsafe_bitcast[UInt16]().unsafe_origin_cast[
        MutAnyOrigin
    ]()
    # In-block word offsets are multiples of 4; block starts
    # are 0, 210, 420 (odd blocks start two bytes into a u32).
    for base_idx in range(3):
        var base = base_idx * 210
        for off in range(0, 208, 4):
            assert_equal(
                load_word(w16, base + off),
                byte_word(sb, base + off),
                "load_word disagrees with the byte assembly",
            )


def signed_scale_list(sb: List[UInt8], idx: Int) -> Float32:
    var b = Int(sb[192 + idx])
    return Float32(b - 256 if b >= 128 else b)


def test_stream_quads_match_element_route() raises:
    # One synthetic super-block: LCG bytes, tamed so the
    # back-division below is exact (d a power of two, every
    # scale a small nonzero integer).
    var sb = List[UInt8](capacity=210)
    sb.resize(210, UInt8(0))
    var state: UInt32 = 4242
    for i in range(len(sb)):
        state = lcg_next(state)
        sb[i] = UInt8((state >> 16) & 0xFF)
    sb[208] = 0x00
    sb[209] = 0x1C  # d = 0.0078125
    var scale_pattern = List[UInt8](capacity=16)
    scale_pattern.append(3)
    scale_pattern.append(251)
    scale_pattern.append(7)
    scale_pattern.append(16)
    scale_pattern.append(200)
    scale_pattern.append(1)
    scale_pattern.append(9)
    scale_pattern.append(254)
    scale_pattern.append(5)
    scale_pattern.append(13)
    scale_pattern.append(240)
    scale_pattern.append(2)
    scale_pattern.append(11)
    scale_pattern.append(6)
    scale_pattern.append(250)
    scale_pattern.append(4)
    for i in range(16):
        sb[192 + i] = scale_pattern[i]
    var d = dequant.f16_to_f32(UInt16(0x1C00))
    for half in range(2):
        for v_in in range(8):
            var ql0 = byte_word(sb, 64 * half + 4 * v_in)
            var ql32 = byte_word(sb, 64 * half + 32 + 4 * v_in)
            var qhw = byte_word(sb, 128 + 32 * half + 4 * v_in)
            var quads = q6k_stream_quads(ql0, ql32, qhw)
            var is_ = v_in // 4
            for s in range(4):
                var sc = signed_scale_list(sb, half * 8 + is_ + 2 * s)
                for j in range(4):
                    var within = half * 128 + s * 32 + 4 * v_in + j
                    var val = dequant.q6k_element_list(sb, 0, within)
                    var expected = val / (d * sc)
                    var got = quads[0][j] if s == 0 else (
                        quads[1][j] if s
                        == 1 else (quads[2][j] if s == 2 else quads[3][j])
                    )
                    assert_equal(
                        got,
                        expected,
                        "stream quad disagrees with the element route",
                    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
