# test_poison_gates.mojo - negative controls for the lane's
# correctness judges (David's zero-trust decision sheet,
# 2026-11-08, gate 2). A gate that passes a poisoned fixture
# is a broken judge, so before any positive golden check is
# trusted, these tests prove the judges FAIL on poison:
#
#   1. The golden checker (generate.sampler's
#      golden_first_match, the one comparison the CLI's
#      GOLDEN_FIRST8_MATCH line uses) fed an expected
#      sequence whose 8th token is flipped from 1212 to 1213
#      must answer False, in both directions.
#   2. The GGUF judge (serve.factsheet's fact_sheet over the
#      loader) fed a copy of the tiny fixture with one byte
#      of its GGUF magic flipped must refuse, while the
#      untouched fixture parses.
#
# Positive controls sit next to each negative one so a judge
# that fails EVERYTHING also fails this file.

from std.testing import TestSuite, assert_true, assert_false, assert_equal

from generate.sampler import golden_first_match
from serve.factsheet import fact_sheet
from serve.server import error_code_of

comptime TINY_FIXTURE = "tests/fixtures/tiny.gguf"
comptime POISON_FIXTURE = "/tmp/poison_tiny.gguf"


def _golden() -> List[Int]:
    var g = List[Int]()
    for t in [13796, 26, 1573, 373, 290, 2167, 26899, 1212]:
        g.append(t)
    return g^


def test_golden_checker_rejects_flipped_eighth() raises:
    var expected = _golden()
    var got = _golden()
    # Positive control: the true sequence passes.
    assert_true(golden_first_match(got, expected))
    # Poison: the expected sequence with the 8th token
    # flipped 1212 -> 1213 must FAIL.
    var poisoned_expected = _golden()
    poisoned_expected[7] = 1213
    assert_false(golden_first_match(got, poisoned_expected))
    # And the mirror: a generated stream carrying the
    # flipped 8th token against the true expectation fails.
    var poisoned_got = _golden()
    poisoned_got[7] = 1213
    assert_false(golden_first_match(poisoned_got, expected))
    # A short stream never passes either.
    var short = List[Int]()
    short.append(13796)
    assert_false(golden_first_match(short, expected))


def test_corrupted_fixture_rejected() raises:
    # Positive control: the untouched fixture parses.
    var sheet = fact_sheet(TINY_FIXTURE)
    assert_equal(sheet.architecture, "llama")
    # Poison: copy the fixture with byte 0 of the GGUF magic
    # flipped ('G' -> 'X'), written under /tmp.
    var data: List[UInt8]
    with open(TINY_FIXTURE, "r") as f:
        data = f.read_bytes(1 << 20)
    assert_true(len(data) > 4)
    assert_equal(data[0], UInt8(71))  # 'G' of GGUF
    data[0] = UInt8(88)  # 'X'
    with open(POISON_FIXTURE, "w") as f:
        f.write_bytes(data)
    # The judge must refuse the corrupted file, by name.
    try:
        _ = fact_sheet(POISON_FIXTURE)
        assert_true(False)
    except e:
        assert_equal(error_code_of(String(e)), "gguf_bad_magic")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
