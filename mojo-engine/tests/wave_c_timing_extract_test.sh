#!/usr/bin/env bash
# wave_c_timing_extract_test.sh - synthetic-log proof for the
# Wave E verdict-line extraction fix in wave_c_timing.sh
# (Lane E3; the wart is recorded in
# bench/mojo-engine-wave-c-2026-11-07.md under Lane C4).
#
# Cases, all against logs built here in a temp dir (no runs,
# the harness's --extract-only mode):
#   1. GOOD: a teacher log carrying BOTH the proof's header
#      lines (GPU_GENERATE_PROOF ...) and its verdict line
#      (TEACHER_FORCED_GPU 8/8 ...) must extract the VERDICT
#      line verbatim, not the header. (Before the fix this
#      case extracted "GPU_GENERATE_PROOF context_len 4096".)
#   2. MISSING: the same log with the verdict line removed
#      must FAIL LOUDLY with TEACHER_FORCED_VERDICT_LINE_MISSING
#      and exit nonzero. (Before the fix it extracted the
#      header line and exited 0.)
#   3. PARTIAL: a verdict line of 7/8 must fail with
#      TEACHER_FORCED_NOT_FULL_PASS.
#   4. PASS_EQ: a TEACHER line carrying pass=8/8 (the other
#      accepted verdict shape) extracts that line.
#
# Exits 0 only if all four cases behave. shellcheck: NOT RUN
# (not installed on the authoring VM; standing note).
set -u

HARNESS="$(cd "$(dirname "$0")" && pwd)/wave_c_timing.sh"
[ -f "$HARNESS" ] || { echo "extract_test: FAIL: harness missing: $HARNESS" >&2; exit 1; }

TMP=$(mktemp -d) || exit 1
trap 'rm -rf "$TMP"' EXIT

fail() { echo "extract_test: FAIL: $*" >&2; exit 1; }

# A stamp the harness accepts: --expected and the file agree.
STAMP_VAL="e3extracttest0001"
printf '%s\n' "$STAMP_VAL" > "$TMP/STAGED_COMMIT"

write_gpu_log() { # FILE
    cat > "$1" <<'EOF'
MODEL_PATH_KIND gpu
PROMPT_IDS [200000, 954, 7963, 323, 11698, 373]
TOKENS [13796, 26, 1573, 373, 290, 2167, 26899, 1212, 6996, 310, 11698, 341, 290, 2167, 26899, 1212]
TOKENS_COUNT 16
GOLDEN_FIRST8_MATCH 1
TEXT_STRICT_OK 1
LOAD_MS 100
GENERATE_MS 2950
PREFILL_MS 684
EOF
}
write_gpu_log "$TMP/gpu1.log"
write_gpu_log "$TMP/gpu2.log"

run_extract() { # TEACHER_LOG OUTFILE -> harness exit code
    bash "$HARNESS" --extract-only --expected "$STAMP_VAL" \
        --stamp-file "$TMP/STAGED_COMMIT" "$1" "$TMP/gpu1.log" "$TMP/gpu2.log" \
        > "$2" 2>&1
    return $?
}

# ---- case 1: header + verdict -> the verdict line ----
cat > "$TMP/teacher_good.log" <<'EOF'
GPU_GENERATE_PROOF model_path /mnt/models/fake.gguf
GPU_GENERATE_PROOF context_len 4096
LOAD_MS 123
TEACHER_FORCED_GPU idx=0 pos=5 pred=13796 expected=13796 ok=1 win_logit=17.34
TEACHER_FORCED_GPU 8/8 total_ms=999
EOF
run_extract "$TMP/teacher_good.log" "$TMP/out1.txt" \
    || fail "case GOOD: harness exited nonzero: $(cat "$TMP/out1.txt")"
grep -q '^TEACHER_FORCED_VERDICT_LINE=TEACHER_FORCED_GPU 8/8 total_ms=999$' "$TMP/out1.txt" \
    || fail "case GOOD: verdict line is not the proof's verdict line: $(grep TEACHER_FORCED_VERDICT_LINE "$TMP/out1.txt" || echo none)"
echo "extract_test: case GOOD ok (verdict line, not the header)"

# ---- case 2: no verdict -> loud named failure ----
grep -v 'TEACHER_FORCED_GPU 8/8' "$TMP/teacher_good.log" > "$TMP/teacher_missing.log"
if run_extract "$TMP/teacher_missing.log" "$TMP/out2.txt"; then
    fail "case MISSING: harness exited 0 on a verdict-less log"
fi
grep -q 'TEACHER_FORCED_VERDICT_LINE_MISSING' "$TMP/out2.txt" \
    || fail "case MISSING: failure is not named TEACHER_FORCED_VERDICT_LINE_MISSING: $(cat "$TMP/out2.txt")"
echo "extract_test: case MISSING ok (loud named failure, nonzero exit)"

# ---- case 3: partial verdict 7/8 -> not-full failure ----
sed 's|TEACHER_FORCED_GPU 8/8|TEACHER_FORCED_GPU 7/8|' \
    "$TMP/teacher_good.log" > "$TMP/teacher_partial.log"
if run_extract "$TMP/teacher_partial.log" "$TMP/out3.txt"; then
    fail "case PARTIAL: harness exited 0 on a 7/8 verdict"
fi
grep -q 'TEACHER_FORCED_NOT_FULL_PASS' "$TMP/out3.txt" \
    || fail "case PARTIAL: failure is not named TEACHER_FORCED_NOT_FULL_PASS: $(cat "$TMP/out3.txt")"
echo "extract_test: case PARTIAL ok (7/8 refused by name)"

# ---- case 4: pass=<n>/<m> shape still extracts ----
cat > "$TMP/teacher_passeq.log" <<'EOF'
GPU_GENERATE_PROOF context_len 4096
TEACHER_FORCED pass=8/8
EOF
run_extract "$TMP/teacher_passeq.log" "$TMP/out4.txt" \
    || fail "case PASS_EQ: harness exited nonzero: $(cat "$TMP/out4.txt")"
grep -q '^TEACHER_FORCED_VERDICT_LINE=TEACHER_FORCED pass=8/8$' "$TMP/out4.txt" \
    || fail "case PASS_EQ: verdict line wrong: $(grep TEACHER_FORCED_VERDICT_LINE "$TMP/out4.txt" || echo none)"
echo "extract_test: case PASS_EQ ok"

echo "extract_test: PASS (4/4 cases)"
