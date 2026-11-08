#!/usr/bin/env bash
# wave_d_timing.sh - Wave D Lane D1 final-window harness.
#
# Runs, in order, on a staged tree whose STAGED_COMMIT stamp
# matches --expected:
#   1. server_proof_gemm    (kernel correctness + s2/s3 timing)
#   2. server_proof_prefill (1,022-token interleaved prefill)
#   3. server_proof_gpu_generate (teacher-forced golden, 8/8)
#   4. the end-to-end runner on the GPU path (GOLDEN_FIRST8_MATCH)
# Every verdict is extracted from the on-disk logs in the
# foreground by this script (the evidence law: backgrounded
# delivery text is not evidence). Any missing or failing
# verdict exits nonzero with a named failure.
#
# Usage:
#   wave_d_timing.sh --expected <sha> --staged-root <dir> --out-dir <dir>
# The caller owns the window discipline (phase lock, tenant
# stop/restore, VRAM bookends); this script owns staging
# verification, builds, runs, and extraction.
# shellcheck: NOT RUN (not installed on the authoring VM;
# standing note, same as wave_c_timing.sh); bash -n clean.
set -u

fail() { echo "wave_d_timing: FAIL: $*" >&2; exit 1; }
note() { echo "wave_d_timing: $*"; }

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
    echo "== build gemm proof =="
    mojo build --target-accelerator gfx1100 -I src tests/server_proof_gemm.mojo -o "$OUT_DIR/proof_gemm" || fail "BUILD server_proof_gemm"
    echo "== build prefill proof =="
    mojo build --target-accelerator gfx1100 -I src tests/server_proof_prefill.mojo -o "$OUT_DIR/proof_prefill" || fail "BUILD server_proof_prefill"
    echo "== build teacher-forced proof =="
    mojo build --target-accelerator gfx1100 -I src tests/server_proof_gpu_generate.mojo -o "$OUT_DIR/proof_tf" || fail "BUILD server_proof_gpu_generate"
    echo "== build runner =="
    mojo build --target-accelerator gfx1100 -I src src/runner/generate_main.mojo -o "$OUT_DIR/runner" || fail "BUILD runner"
} >"$BUILD_LOG" 2>&1
note "builds done (log: $BUILD_LOG)"

"$OUT_DIR/proof_gemm" >"$OUT_DIR/gemm.log" 2>&1 || fail "RUN server_proof_gemm exit nonzero (see $OUT_DIR/gemm.log)"
grep -q "GEMM_CORRECTNESS PASS" "$OUT_DIR/gemm.log" || fail "EXTRACT GEMM_CORRECTNESS PASS missing"
grep -q "GEMM_PROOF PASS" "$OUT_DIR/gemm.log" || fail "EXTRACT GEMM_PROOF PASS missing"
note "gemm proof PASS (log: $OUT_DIR/gemm.log)"

"$OUT_DIR/proof_prefill" >"$OUT_DIR/prefill.log" 2>&1 || fail "RUN server_proof_prefill exit nonzero (see $OUT_DIR/prefill.log)"
grep -q "PREFILL1022_VERDICT PASS" "$OUT_DIR/prefill.log" || fail "EXTRACT PREFILL1022_VERDICT PASS missing"
note "prefill proof PASS (log: $OUT_DIR/prefill.log)"

"$OUT_DIR/proof_tf" >"$OUT_DIR/teacher_forced.log" 2>&1 || fail "RUN teacher-forced exit nonzero (see $OUT_DIR/teacher_forced.log)"
grep -q "TEACHER_FORCED_GPU 8/8" "$OUT_DIR/teacher_forced.log" || fail "EXTRACT TEACHER_FORCED_GPU 8/8 missing"
note "teacher-forced 8/8 (log: $OUT_DIR/teacher_forced.log)"

MOJO_ENGINE_GPU=1 "$OUT_DIR/runner" >"$OUT_DIR/runner.log" 2>&1 || fail "RUN runner exit nonzero (see $OUT_DIR/runner.log)"
grep -q "MODEL_PATH_KIND gpu" "$OUT_DIR/runner.log" || fail "EXTRACT MODEL_PATH_KIND gpu missing"
grep -q "GOLDEN_FIRST8_MATCH 1" "$OUT_DIR/runner.log" || fail "EXTRACT GOLDEN_FIRST8_MATCH 1 missing"
note "runner GOLDEN_FIRST8_MATCH 1 (log: $OUT_DIR/runner.log)"

echo "WAVE_D_TIMING_SUMMARY"
grep "GEMM_TIMING" "$OUT_DIR/gemm.log" || true
grep "PREFILL1022_SUMMARY" "$OUT_DIR/prefill.log" || fail "EXTRACT PREFILL1022_SUMMARY missing"
grep "PREFILL_MS" "$OUT_DIR/runner.log" || true
echo "WAVE_D_TIMING_DONE"
