# test_f32_gpu_helpers.mojo - host tests for the tuned
# plain-float GPU kernels' pure helpers (lane mojo-d3-gemv,
# Wave D).
#
# The tuned kernels (src/kernels/f32_gemv.mojo) move weights
# as vec4 loads and decode F16 quads from word pairs; the
# pure pieces are proven here on the host before any card
# run is believed (the test_k1_gpu_helpers.mojo pattern):
#
# 1. f16_to_f32_gpu (bit-trick decode) must equal
#    kernels.dequant.f16_to_f32 on every f16 pattern except
#    the inf/NaN class (standing note: the two deliberately
#    differ in payload handling there and weight files never
#    carry them).
# 2. f16_quad_from_words must decode a quad's four halves,
#    in order, exactly as the exact decode does, over
#    randomized word pairs (normals, subnormals, zero, and
#    signed values all appear in an LCG sweep).

from std.testing import TestSuite, assert_true, assert_equal

from kernels import dequant
from kernels.f32_gemv import f16_to_f32_gpu, f16_quad_from_words


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


def test_f16_quad_from_words_matches_exact_decode() raises:
    var state: UInt32 = 31337
    for _trial in range(256):
        state = lcg_next(state)
        var lo = state
        state = lcg_next(state)
        var hi = state
        # Skip the trial if any half is in the inf/NaN class.
        var halves = List[UInt16](capacity=4)
        halves.append(UInt16(lo & 0xFFFF))
        halves.append(UInt16(lo >> 16))
        halves.append(UInt16(hi & 0xFFFF))
        halves.append(UInt16(hi >> 16))
        var skip = False
        for j in range(4):
            if ((Int(halves[j]) >> 10) & 0x1F) == 31:
                skip = True
        if skip:
            continue
        var quad = f16_quad_from_words(lo, hi)
        for j in range(4):
            var expected = dequant.f16_to_f32(halves[j])
            var got = quad[0] if j == 0 else (
                quad[1] if j == 1 else (quad[2] if j == 2 else quad[3])
            )
            assert_equal(
                got,
                expected,
                "f16 quad decode disagrees with the exact decode",
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
