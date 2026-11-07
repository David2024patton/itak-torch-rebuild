# test_diag.mojo - the debug seam's self-test.
#
# Covers the seam rule as a fixed matrix (the pure predicate),
# consistency between debug_enabled() and the live environment,
# and the emit contract: on fires (a named line is returned and
# printed), off is silent (empty return, nothing printed). The
# runner (tests/run_tests.sh) executes this suite twice, once
# with MOJO_ENGINE_DEBUG unset and once set to 1, so both modes
# are exercised against the same assertions.

from std.os import getenv
from std.testing import TestSuite, assert_true, assert_false, assert_equal

from diag import debug_enabled, debug_value_enabled, format_line, log_debug


def test_value_matrix() raises:
    assert_false(debug_value_enabled(""))
    assert_false(debug_value_enabled("0"))
    assert_true(debug_value_enabled("1"))
    assert_true(debug_value_enabled("yes"))
    assert_true(debug_value_enabled("anything"))


def test_enabled_matches_environment() raises:
    var expect = debug_value_enabled(getenv("MOJO_ENGINE_DEBUG"))
    assert_equal(debug_enabled(), expect)


def test_emit_contract() raises:
    var line = log_debug("test", "seam probe")
    if debug_enabled():
        assert_equal(line, format_line("test", "seam probe"))
        assert_true(line.startswith("[mojo_engine][test] "))
    else:
        assert_equal(line, "")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
