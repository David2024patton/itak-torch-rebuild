#!/usr/bin/env bash
# wave_e_timing.sh - Wave E Lane E1 server-window harness.
#
# Runs, in order, on a staged tree whose STAGED_COMMIT stamp
# matches --expected:
#   1. server_proof_wave_e_batch_layer (batch step vs the
#      per-row resident path, layers 0/3, batch rows 6/64)
#   2. server_proof_wave_e_prefill_route (batch-vs-per-row
#      crossover sweep 6..1022 + the 1,022-token arms)
#   3. server_proof_gpu_generate (teacher-forced golden, 8/8)
#   4. the end-to-end runner on the GPU path (GOLDEN_FIRST8_MATCH)
# Every verdict is extracted from the on-disk logs in the
# foreground by this script (the evidence law: backgrounded
# delivery text is not evidence). Any missing or failing
# verdict exits nonzero with a named failure.
#
# Usage:
#   wave_e_timing.sh --expected <sha> --staged-root <dir> --out-dir <dir>
# The caller owns the window discipline (phase lock, tenant
# stop/restore, VRAM bookends); this script owns staging
# verification, builds, runs, and extraction.
# shellcheck: NOT RUN (not installed on the authoring VM;
# standing note, same as wave_d_timing.sh); bash -n clean.
set -u

fail() { echo "wave_e_timing: FAIL: $*" >&2; exit 1; }
note() { echo "wave_e_timing: $*"; }

EXPECTED=""
STAGED_ROOT=""
OUT_DIR=""
while [ $# -gt 0 ]; do
    case "$1" in
        --expected) [ $# -ge 2 ] || fail "ARG --expected needs a value"; EXPECTED="$2"; shift 2 ;;
        --staged-root) [ $# -ge 2 ] || fail "ARG --staged-root needs a value"; STAGED_ROOT="$2"; shift 2 ;;
        --out-dir) [ $# -ge 2 ] || fail "ARG --out-dir needs a value"; OUT_DIR="$2"; shift 2 ;;
        *) fail "ARG unknown option: $1" ;;
    esac
done
[ -n "$EXPECTED" ] || fail "ARG --expected is required"
[ -n "$STAGED_ROOT" ] || fail "ARG --staged-root is required"
[ -n "$OUT_DIR" ] || fail "ARG --out-dir is required"
[ -f "$STAGED_ROOT/STAGED_COMMIT" ] || fail "STAMP missing STAGED_COMMIT in $STAGED_ROOT"
STAMP=$(cat "$STAGED_ROOT/STAGED_COMMIT")
[ "$STAMP" = "$EXPECTED" ] || fail "STAMP staged $STAMP != expected $EXPECTED"
ENGINE_DIR="$STAGED_ROOT/mojo-engine"
[ -d "$ENGINE_DIR" ] || fail "STAGED tree has no mojo-engine dir"
mkdir -p "$OUT_DIR"
note "staged commit verified: $STAMP"

cd "$ENGINE_DIR" || fail "cannot cd to staged engine dir"
BUILD_LOG="$OUT_DIR/build.log"
{
    echo "== build batch layer proof =="
    mojo build --target-accelerator gfx1100 -I src tests/server_proof_wave_e_batch_layer.mojo -o "$OUT_DIR/proof_batch_layer" || fail "BUILD server_proof_wave_e_batch_layer"
    echo "== build route proof =="
    mojo build --target-accelerator gfx1100 -I src tests/server_proof_wave_e_prefill_route.mojo -o "$OUT_DIR/proof_route" || fail "BUILD server_proof_wave_e_prefill_route"
    echo "== build teacher-forced proof =="
    mojo build --target-accelerator gfx1100 -I src tests/server_proof_gpu_generate.mojo -o "$OUT_DIR/proof_tf" || fail "BUILD server_proof_gpu_generate"
    echo "== build runner =="
    mojo build --target-accelerator gfx1100 -I src src/runner/generate_main.mojo -o "$OUT_DIR/runner" || fail "BUILD runner"
} >"$BUILD_LOG" 2>&1
note "builds done (log: $BUILD_LOG)"

"$OUT_DIR/proof_batch_layer" >"$OUT_DIR/batch_layer.log" 2>&1 || fail "RUN batch layer proof exit nonzero (see $OUT_DIR/batch_layer.log)"
grep -q "WAVE_E_BATCH_LAYER_PROOF PASS" "$OUT_DIR/batch_layer.log" || fail "EXTRACT WAVE_E_BATCH_LAYER_PROOF PASS missing"
note "batch layer proof PASS (log: $OUT_DIR/batch_layer.log)"

"$OUT_DIR/proof_route" >"$OUT_DIR/route.log" 2>&1 || fail "RUN route proof exit nonzero (see $OUT_DIR/route.log)"
grep -q "WAVE_E_ROUTE_VERDICT PASS" "$OUT_DIR/route.log" || fail "EXTRACT WAVE_E_ROUTE_VERDICT PASS missing"
note "route proof PASS (log: $OUT_DIR/route.log)"

"$OUT_DIR/proof_tf" >"$OUT_DIR/teacher_forced.log" 2>&1 || fail "RUN teacher-forced exit nonzero (see $OUT_DIR/teacher_forced.log)"
grep -q "TEACHER_FORCED_GPU 8/8" "$OUT_DIR/teacher_forced.log" || fail "EXTRACT TEACHER_FORCED_GPU 8/8 missing"
note "teacher-forced 8/8 (log: $OUT_DIR/teacher_forced.log)"

MOJO_ENGINE_GPU=1 "$OUT_DIR/runner" >"$OUT_DIR/runner.log" 2>&1 || fail "RUN runner exit nonzero (see $OUT_DIR/runner.log)"
grep -q "MODEL_PATH_KIND gpu" "$OUT_DIR/runner.log" || fail "EXTRACT MODEL_PATH_KIND gpu missing"
grep -q "GOLDEN_FIRST8_MATCH 1" "$OUT_DIR/runner.log" || fail "EXTRACT GOLDEN_FIRST8_MATCH 1 missing"
note "runner GOLDEN_FIRST8_MATCH 1 (log: $OUT_DIR/runner.log)"

echo "WAVE_E_TIMING_SUMMARY"
grep "WAVE_E_BATCH_LAYER " "$OUT_DIR/batch_layer.log" || true
grep "WAVE_E_ROUTE_SUMMARY" "$OUT_DIR/route.log" || fail "EXTRACT WAVE_E_ROUTE_SUMMARY missing"
grep "WAVE_E_CROSSOVER" "$OUT_DIR/route.log" || fail "EXTRACT WAVE_E_CROSSOVER missing"
grep "WAVE_E_ARMS1022" "$OUT_DIR/route.log" || fail "EXTRACT WAVE_E_ARMS1022 missing"
grep "PREFILL_MS" "$OUT_DIR/runner.log" || true
echo "WAVE_E_TIMING_DONE"
