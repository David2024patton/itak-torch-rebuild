#!/usr/bin/env bash
# run_loader_lane.sh - LANE A1 (GGUF loader) test runner.
#
# The coordinator's tests/run_tests.sh is untouched by this lane;
# this runner covers only the lane's own new suite. It runs the
# loader TestSuite with the diag seam off and on.
set -u
cd "$(dirname "$0")/.."

fail() { echo "run_loader_lane: FAIL: $*" >&2; exit 1; }

echo "== loader TestSuite (seam off) =="
env -u MOJO_ENGINE_DEBUG -u MOJO_ENGINE_MODEL_PATH \
    -u MOJO_ENGINE_CONTEXT_LEN \
    mojo run -I src tests/test_loader_full.mojo || fail "loader suite (seam off)"

echo "== loader TestSuite (MOJO_ENGINE_DEBUG=1, seam on) =="
env MOJO_ENGINE_DEBUG=1 \
    mojo run -I src tests/test_loader_full.mojo || fail "loader suite (seam on)"

echo "run_loader_lane: PASS"
