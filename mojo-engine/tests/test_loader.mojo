# test_loader.mojo - the GGUF loader seam's contract.
#
# The loader itself is the next lane; what this foundation must
# prove is that its absence is a named refusal, never a stub
# that pretends a model loaded.

from std.testing import TestSuite, assert_true, assert_equal

from gguf.loader import load_model, loader_state


def test_loader_state_is_named() raises:
    assert_equal(loader_state(), "not_implemented")


def test_load_model_refuses_by_name() raises:
    var refused = False
    try:
        load_model("/nonexistent/model.gguf")
    except e:
        refused = String(e).find("gguf_loader_not_implemented") != -1
    assert_true(refused, "load_model must refuse by name")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
