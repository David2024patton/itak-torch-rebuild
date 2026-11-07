# test_loader.mojo - the GGUF loader seam's contract.
#
# Foundation contract (loader absent) was replaced by Lane A1
# (2026-11-07): the loader exists, reports its state by name,
# and refuses a missing file by name. The full round-trip and
# crafted-bad-file suites live in test_loader_full.mojo.

from std.testing import TestSuite, assert_true, assert_equal

from gguf.loader import load_model, loader_state


def test_loader_state_is_named() raises:
    assert_equal(loader_state(), "implemented")


def test_load_model_refuses_by_name() raises:
    var refused = False
    try:
        load_model("/nonexistent/model.gguf")
    except e:
        refused = String(e).find("gguf_io_error") != -1
    assert_true(refused, "load_model must refuse a missing file by name")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
