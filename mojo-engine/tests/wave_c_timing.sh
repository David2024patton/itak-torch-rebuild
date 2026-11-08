#!/usr/bin/env bash
# wave_c_timing.sh - Lane C4 measurement harness for the Wave C
# final integration window (Mojo/MAX engine, GPU path).
#
# =====================================================================
# MEASUREMENT PROTOCOL (Lane C4; the coordinator writes the bench
# note from this harness's output - this script prints arithmetic
# only, no verdict language of its own beyond the lines it copies
# verbatim from the proof / runner logs)
# =====================================================================
#
# WHAT RUNS, IN ORDER (sequential, foreground, on the server,
# against the coordinator's staged tree):
#   1. The C3 teacher-forced proof, exactly once:
#        MOJO_ENGINE_GPU=1 mojo run -I src tests/server_proof_gpu_generate.mojo
#   2. The end-to-end runner on the GPU path:
#        MOJO_ENGINE_GPU=1 mojo run -I src src/runner/generate_main.mojo
#      TWO runs when the first completes (exit 0): run 1 is the
#      warmup, run 2 is the timed run. Both runs' numbers are
#      printed; the summary's unprefixed GPU numbers are run 2's
#      when run 2 exists, run 1's otherwise. No cherry-picking:
#      if only one run completes, that is stated (TIMED_RUN=1).
#   3. OPTIONAL fresh CPU control, only with --with-cpu-control:
#      ONE run of the same runner on the same staged tree with
#      MOJO_ENGINE_GPU unset. "Interleaving" in this wave means
#      exactly this: GPU runs and (window permitting) one fresh
#      CPU control from the same staged tree, in one window, so
#      both paths see the same code, model file, and host state.
#
# WHY ONE RUN EACH IS THE HONEST COUNT FOR A CORRECTNESS WAVE:
#   The Wave B CPU decode took ~253.6 s/step (see below), so a
#   single 16-token CPU generate run takes well over an hour and
#   repeated interleaved CPU/GPU rounds would take many hours to
#   add nothing to a correctness question: the golden comparison
#   is exact-match, not a distribution, and one full pass either
#   matches or it does not. GPU generate runs are cheap by
#   comparison, so the GPU path gets the one extra warmup run
#   above (cold-start effects land in run 1, the timed figure is
#   run 2). If the fresh CPU control is skipped, the Wave B
#   recorded CPU numbers below are the control and are labeled
#   recorded-not-concurrent everywhere they appear:
#     CPU_CONTROL_SOURCE=recorded_wave_b_not_concurrent
#   A fresh control is labeled:
#     CPU_CONTROL_SOURCE=fresh_staged_tree
#
# EVIDENCE LAW: every number in the summary is read back from
# the actual log files on disk, in the foreground, after the
# runs finish (the extract_* / field_value functions below take
# log file paths and grep those files; no number is taken from
# a captured variable, a background job's completion notice, or
# this script's own arithmetic over anything but log values and
# the labeled Wave B constants).
#
# WAVE B RECORDED CPU NUMBERS (bench/mojo-engine-wave-b-2026-11-07.md,
# first-light, CPU path, commit 4dc4921's runner):
#   model open (LOAD_MS)   24,425 ms
#   prefill (PREFILL_MS)   1,411,755 ms  (6 prompt positions,
#                          fresh model stepping the prompt ids)
#   generate total         5,216,444 ms  (16 tokens; context for
#                          the decode derivation, not a summary
#                          input on the GPU side)
#   decode                 ~253.6 s/step, ~0.00394 tok/s
#
# FORMULAS (identical definitions on every path, so the numbers
# are reproducible from the logs alone):
#   decode_ms        = GENERATE_MS - PREFILL_MS
#   decode_steps     = 15   (a 16-token run: 6 prefill steps are
#                      inside GENERATE_MS; the runner's own
#                      stated approximation counts 15 decode
#                      steps - see generate_main.mojo's header)
#   decode_s_per_step = decode_ms / 15 / 1000
#   decode_tok_s      = 1000 / (decode_ms / 15)
#                     = 15 * 1000 / decode_ms
#   Ratios printed are plain divisions of the recorded Wave B
#   CPU figure by the GPU figure (CPU_OVER_GPU), and, when a
#   fresh CPU control ran, of the fresh figure by the GPU figure.
#   PREFILL_MS itself is the runner's own printed measurement:
#   a second fresh model stepping the prompt ids (forward_rows),
#   exactly the Wave B construction.
#
# STAGED TREE: the coordinator stages the exact commit with
# `git archive`, writes a STAGED_COMMIT stamp file, and records
# a staged file hash (the Wave B window pattern). This harness
# takes the staged tree as given: it verifies the stamp exists,
# prints it, and refuses to run if the stamp does not match the
# expected commit passed as an argument. It does NOT stage,
# does NOT take the phase lock, and does NOT stop or restore
# tenants or measure VRAM: the coordinator's window wrapper
# owns those (flock /tmp/itak-phase.lock, tenant drain/restore,
# sysfs bookends), exactly as in the Wave B windows.
#
# RUNNER LINES PARSED (src/runner/generate_main.mojo, byte
# formats as shipped in Wave B plus the Wave C addition):
#   MODEL_PATH_KIND <gpu|cpu>     (Wave C addition, Lane C3)
#   PROMPT_IDS [<ids>]            LOAD_MS <ms>     (model open)
#   GENERATE_MS <ms>              PREFILL_MS <ms>
#   TOKENS_COUNT <n>              TOKENS [<ids>]
#   TEXT_STRICT_OK <0|1>          GOLDEN_FIRST8_MATCH <0|1>
# A separator of space, '=', or ':' after the name is accepted.
# TEACHER-FORCED VERDICT LINE: Lane C3's proof prints its own
# verdict lines and exits nonzero on any mismatch (WAVE_C
# contract: teacher-forced positions 0..7, 8 of 8). The exact
# wording is C3's; the extractor accepts, in order: the
# proof's own verdict line, a line anchored at the start
# TEACHER_FORCED_GPU <n>/<m> (optionally followed by more
# fields, e.g. total_ms=...), or a TEACHER line carrying a
# pass=<n>/<m> count. The matched line is printed verbatim
# in the summary.
# Wave E fix (Lane E3; wart recorded in
# bench/mojo-engine-wave-c-2026-11-07.md): the original shape
# list tried a generic GPU_(GENERATE|PROOF) line BEFORE the
# count shapes, so it matched the proof's header line
# (GPU_GENERATE_PROOF context_len 4096) and reported that as
# the verdict - and a log with NO verdict line at all still
# "extracted" the header and passed. The generic proof shape
# is gone: a log whose verdict line is missing now fails
# loudly with TEACHER_FORCED_VERDICT_LINE_MISSING, and the
# extracted line is the proof's verdict line itself. The
# wave_d_timing.sh lineage never had this bug (it greps the
# exact TEACHER_FORCED_GPU 8/8 string); this fixed wave_c
# harness remains the rich-summary harness for Wave E, and
# a wave_e_timing.sh, if the coordinator wants one, should
# be a copy of THIS file, not of the pre-fix shapes.
#
# FAILURE LAW (fail loudly: nonzero exit, named message):
#   STAGED_COMMIT_MISSING        no stamp file found
#   STAGED_COMMIT_MISMATCH       stamp != expected commit arg
#   GOLDEN_FIRST8_MATCH_MISSING  no such line in a generate log
#   GOLDEN_FIRST8_MATCH_NOT_1    line present, value not 1
#   TOKENS_MISSING               no TOKENS line in a generate log
#   (plus named *_MISSING / *_NOT_GPU / exit-code failures for
#   the other required lines and for the proof / runner runs)
#
# USAGE:
#   wave_c_timing.sh EXPECTED_COMMIT [options]        # server run
#   wave_c_timing.sh --extract-only [options] TEACHER_LOG GPU_RUN1_LOG [GPU_RUN2_LOG]
#                                                     # parse existing logs, no runs
# Options:
#   --expected COMMIT     expected commit (or first positional)
#   --staged-root DIR     staged tree root holding STAGED_COMMIT
#                         (default: search cwd, engine dir, repo root)
#   --stamp-file FILE     explicit STAGED_COMMIT file path
#   --out-dir DIR         log output dir (default: ./wave_c_timing_out)
#   --with-cpu-control    also run ONE fresh CPU control (server mode)
#   --cpu-log FILE        extract-only: a fresh CPU control log
#   --extract-only        do not run anything; parse the given logs
# Env:
#   MOJO_BIN              mojo executable (default: mojo). On the
#                         server, run this harness inside the pixi
#                         env (as tests/run_tests.sh is run), or
#                         set MOJO_BIN to the env's mojo.
# shellcheck: NOT RUN on the authoring VM (shellcheck absent
# there, standing note, same as tests/run_tests.sh); bash -n clean.
set -u

# ---- Wave B recorded CPU constants (see header) ----
CPU_WB_MODEL_OPEN_MS=24425
CPU_WB_PREFILL_MS=1411755
CPU_WB_GENERATE_MS=5216444
CPU_WB_DECODE_S_PER_STEP="253.6"
DECODE_STEPS=15

fail() { echo "wave_c_timing: FAIL: $*" >&2; exit 1; }
note() { echo "wave_c_timing: $*"; }

EXPECTED=""
STAGED_ROOT=""
STAMP_FILE=""
OUT_DIR=""
EXTRACT_ONLY=0
WITH_CPU_CONTROL=0
CPU_LOG=""
POSITIONAL=""

while [ $# -gt 0 ]; do
    case "$1" in
        --expected) [ $# -ge 2 ] || fail "ARG --expected needs a value"; EXPECTED="$2"; shift 2 ;;
        --staged-root) [ $# -ge 2 ] || fail "ARG --staged-root needs a value"; STAGED_ROOT="$2"; shift 2 ;;
        --stamp-file) [ $# -ge 2 ] || fail "ARG --stamp-file needs a value"; STAMP_FILE="$2"; shift 2 ;;
        --out-dir) [ $# -ge 2 ] || fail "ARG --out-dir needs a value"; OUT_DIR="$2"; shift 2 ;;
        --cpu-log) [ $# -ge 2 ] || fail "ARG --cpu-log needs a value"; CPU_LOG="$2"; shift 2 ;;
        --with-cpu-control) WITH_CPU_CONTROL=1; shift ;;
        --extract-only) EXTRACT_ONLY=1; shift ;;
        --help|-h)
            sed -n '2,10p' "$0" 2>/dev/null || true
            echo "usage: see the header comment of $0"
            exit 0 ;;
        --) shift; while [ $# -gt 0 ]; do POSITIONAL="$POSITIONAL $1"; shift; done ;;
        -*) fail "ARG unknown option: $1" ;;
        *) POSITIONAL="$POSITIONAL $1"; shift ;;
    esac
done
# shellcheck disable=SC2086 # positional list is word-split on purpose
set -- $POSITIONAL

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

# ---- staged stamp verification ----
find_stamp() {
    if [ -n "$STAMP_FILE" ]; then
        [ -f "$STAMP_FILE" ] && { echo "$STAMP_FILE"; return 0; }
        return 1
    fi
    local roots="$STAGED_ROOT $PWD $ENGINE_DIR $ENGINE_DIR/.."
    local r
    for r in $roots; do
        [ -n "$r" ] || continue
        if [ -f "$r/STAGED_COMMIT" ]; then echo "$r/STAGED_COMMIT"; return 0; fi
    done
    return 1
}

STAMP_PATH_PRINT=""; STAMP_VAL_PRINT=""; STAMP_HASH_PRINT=""
verify_stamp() {
    local stamp_path stamp_val
    stamp_path=$(find_stamp) || fail "STAGED_COMMIT_MISSING: no STAGED_COMMIT stamp found (searched --stamp-file, --staged-root, cwd, engine dir, repo root)"
    stamp_val=$(tr -d ' \t\r\n' < "$stamp_path")
    [ -n "$stamp_val" ] || fail "STAGED_COMMIT_MISSING: stamp file $stamp_path is empty"
    STAMP_PATH_PRINT="$stamp_path"; STAMP_VAL_PRINT="$stamp_val"
    echo "STAGED_COMMIT_FILE=$stamp_path"
    echo "STAGED_COMMIT=$stamp_val"
    if [ -n "$EXPECTED" ]; then
        # Exact match, or one side an unambiguous (>=7 char) prefix
        # of the other (short vs full form of the same commit).
        if [ "$stamp_val" = "$EXPECTED" ]; then
            :
        elif [ "${#EXPECTED}" -ge 7 ] && [ "${stamp_val#"$EXPECTED"}" != "$stamp_val" ]; then
            :
        elif [ "${#stamp_val}" -ge 7 ] && [ "${EXPECTED#"$stamp_val"}" != "$EXPECTED" ]; then
            :
        else
            fail "STAGED_COMMIT_MISMATCH: stamp '$stamp_val' does not match expected '$EXPECTED'"
        fi
        echo "EXPECTED_COMMIT=$EXPECTED"
    else
        echo "EXPECTED_COMMIT=UNSPECIFIED"
    fi
    # Optional staged file hash, printed when the coordinator's
    # staging left one beside the stamp (not required).
    local d h
    d=$(dirname "$stamp_path")
    for h in "$d/STAGED_FILE_HASH" "$d/STAGED_SHA256" "$d/STAGED_FILE.sha256"; do
        if [ -f "$h" ]; then STAMP_HASH_PRINT=$(head -n 1 "$h"); echo "STAGED_FILE_HASH=$STAMP_HASH_PRINT"; break; fi
    done
}

# ---- log field extraction (reads the files, foreground) ----
# field_value FILE NAME -> value after "NAME", "NAME=", or "NAME:"
field_value() {
    local file="$1" name="$2" line
    [ -f "$file" ] || fail "LOG_MISSING: $file"
    line=$(grep -E "^${name}([[:space:]]|=|:)" "$file" | tail -n 1 || true)
    [ -n "$line" ] || return 1
    printf '%s' "$line" | sed -E "s/^${name}[[:space:]=:]+//"
}

require_int() { # VALUE NAME FILE
    case "$1" in
        ''|*[!0-9-]*) fail "$2_MISSING: no numeric $2 line in $3" ;;
    esac
}

# compute_decode GENERATE_MS PREFILL_MS -> "decode_ms s_per_step tok_s"
compute_decode() {
    awk -v g="$1" -v p="$2" -v steps="$DECODE_STEPS" 'BEGIN {
        d = g - p
        if (d <= 0) { print "NONPOSITIVE 0 0"; exit 0 }
        step_ms = d / steps
        printf "%d %.4f %.6f", d, step_ms / 1000.0, 1000.0 / step_ms
    }'
}

ratio() { # NUMERATOR DENOMINATOR -> %.4f or MISSING
    awk -v n="$1" -v d="$2" 'BEGIN {
        if (d == 0) { print "MISSING"; exit 0 }
        printf "%.4f", n / d
    }'
}

ratio_decode() { # CPU_S_PER_STEP GPU_DECODE_MS -> %.4f or MISSING
    # Uses the unrounded GPU decode_ms, not the printed (rounded)
    # s/step, so the ratio is exact arithmetic on the log values.
    awk -v n="$1" -v dms="$2" -v steps="$DECODE_STEPS" 'BEGIN {
        if (dms == 0) { print "MISSING"; exit 0 }
        printf "%.4f", n / (dms / steps / 1000.0)
    }'
}

teacher_verdict_line() { # FILE -> verbatim verdict line
    local file="$1" line
    [ -f "$file" ] || fail "LOG_MISSING: $file"
    # 1. The proof's own verdict line (Wave E fix; see the
    # header): anchored at line start so the per-step
    # TEACHER_FORCED_GPU idx=... lines and the
    # GPU_GENERATE_PROOF header lines can never match.
    line=$(grep -E '^TEACHER_FORCED_GPU [0-9]+/[0-9]+([[:space:]]|$)' "$file" | tail -n 1 || true)
    [ -n "$line" ] && { printf '%s' "$line"; return 0; }
    # 2. A TEACHER line carrying an explicit pass=<n>/<m>.
    line=$(grep -Ei 'teacher' "$file" | grep -E 'pass=[0-9]+/[0-9]+' | tail -n 1 || true)
    [ -n "$line" ] && { printf '%s' "$line"; return 0; }
    return 1
}

# Extract one generate log; $2 = prefix for its per-run lines.
# Sets globals G_MODEL_OPEN_MS G_PREFILL_MS G_GENERATE_MS for the caller.
extract_generate() {
    local file="$1" prefix="$2"
    local kind prompt_ids tokens tokens_count golden strict load_ms gen_ms pf_ms
    kind=$(field_value "$file" MODEL_PATH_KIND) || fail "MODEL_PATH_KIND_MISSING: no MODEL_PATH_KIND line in $file"
    prompt_ids=$(field_value "$file" PROMPT_IDS) || fail "PROMPT_IDS_MISSING: no PROMPT_IDS line in $file"
    tokens=$(field_value "$file" TOKENS) || fail "TOKENS_MISSING: no TOKENS line in $file"
    [ -n "$tokens" ] || fail "TOKENS_MISSING: empty TOKENS line in $file"
    tokens_count=$(field_value "$file" TOKENS_COUNT || true)
    golden=$(field_value "$file" GOLDEN_FIRST8_MATCH) || fail "GOLDEN_FIRST8_MATCH_MISSING: no GOLDEN_FIRST8_MATCH line in $file"
    if [ "$golden" != "1" ]; then
        fail "GOLDEN_FIRST8_MATCH_NOT_1: GOLDEN_FIRST8_MATCH=$golden in $file"
    fi
    strict=$(field_value "$file" TEXT_STRICT_OK) || fail "TEXT_STRICT_OK_MISSING: no TEXT_STRICT_OK line in $file"
    load_ms=$(field_value "$file" LOAD_MS) || fail "LOAD_MS_MISSING: no LOAD_MS line in $file"
    gen_ms=$(field_value "$file" GENERATE_MS) || fail "GENERATE_MS_MISSING: no GENERATE_MS line in $file"
    pf_ms=$(field_value "$file" PREFILL_MS) || fail "PREFILL_MS_MISSING: no PREFILL_MS line in $file"
    require_int "$load_ms" LOAD_MS "$file"
    require_int "$gen_ms" GENERATE_MS "$file"
    require_int "$pf_ms" PREFILL_MS "$file"
    local dec d_ms d_s d_tok
    dec=$(compute_decode "$gen_ms" "$pf_ms")
    d_ms=$(printf '%s' "$dec" | awk '{print $1}')
    d_s=$(printf '%s' "$dec" | awk '{print $2}')
    d_tok=$(printf '%s' "$dec" | awk '{print $3}')
    [ "$d_ms" != "NONPOSITIVE" ] || fail "DECODE_NONPOSITIVE: GENERATE_MS=$gen_ms PREFILL_MS=$pf_ms in $file"
    echo "${prefix}MODEL_PATH_KIND=$kind"
    echo "${prefix}PROMPT_IDS=$prompt_ids"
    echo "${prefix}TOKENS=$tokens"
    [ -n "${tokens_count:-}" ] && echo "${prefix}TOKENS_COUNT=$tokens_count"
    echo "${prefix}GOLDEN_FIRST8_MATCH=$golden"
    echo "${prefix}TEXT_STRICT_OK=$strict"
    echo "${prefix}MODEL_OPEN_MS=$load_ms"
    echo "${prefix}LOAD_MS=$load_ms"
    echo "${prefix}PREFILL_MS=$pf_ms"
    echo "${prefix}GENERATE_TOTAL_MS=$gen_ms"
    echo "${prefix}GENERATE_MS=$gen_ms"
    echo "${prefix}DECODE_MS=$d_ms"
    echo "${prefix}DECODE_S_PER_STEP=$d_s"
    echo "${prefix}DECODE_TOK_S=$d_tok"
    G_MODEL_OPEN_MS="$load_ms"; G_PREFILL_MS="$pf_ms"; G_GENERATE_MS="$gen_ms"
    G_DECODE_S_PER_STEP="$d_s"; G_DECODE_MS="$d_ms"
}

# ---- run mode: execute proof + runner on the staged tree ----
TEACHER_LOG=""; GPU_LOG1=""; GPU_LOG2=""; CPU_RUN_LOG=""
run_server() {
    if [ -z "$EXPECTED" ] && [ $# -ge 1 ]; then EXPECTED="$1"; fi
    [ -n "$EXPECTED" ] || fail "ARG: expected commit argument required (or --expected)"
    local mojo_bin="${MOJO_BIN:-mojo}"
    command -v "$mojo_bin" >/dev/null 2>&1 || fail "MOJO_NOT_FOUND: '$mojo_bin' not on PATH (run inside the pixi env or set MOJO_BIN)"
    [ -f "$ENGINE_DIR/tests/server_proof_gpu_generate.mojo" ] || fail "PROOF_MISSING: tests/server_proof_gpu_generate.mojo not in staged tree at $ENGINE_DIR"
    [ -f "$ENGINE_DIR/src/runner/generate_main.mojo" ] || fail "RUNNER_MISSING: src/runner/generate_main.mojo not in staged tree at $ENGINE_DIR"
    [ -n "$OUT_DIR" ] || OUT_DIR="$PWD/wave_c_timing_out"
    mkdir -p "$OUT_DIR" || fail "OUT_DIR: cannot create $OUT_DIR"
    TEACHER_LOG="$OUT_DIR/teacher_forced_gpu.log"
    GPU_LOG1="$OUT_DIR/generate_gpu_run1.log"
    GPU_LOG2="$OUT_DIR/generate_gpu_run2.log"
    CPU_RUN_LOG="$OUT_DIR/generate_cpu_control.log"
    note "logs: teacher=$TEACHER_LOG gpu1=$GPU_LOG1 gpu2=$GPU_LOG2"
    local rc
    note "running teacher-forced proof (once, MOJO_ENGINE_GPU=1) ..."
    ( cd "$ENGINE_DIR" && MOJO_ENGINE_GPU=1 "$mojo_bin" run -I src tests/server_proof_gpu_generate.mojo ) > "$TEACHER_LOG" 2>&1
    rc=$?
    echo "TEACHER_FORCED_EXIT=$rc"
    [ "$rc" -eq 0 ] || fail "TEACHER_FORCED_PROOF_EXIT_NONZERO: exit $rc, log $TEACHER_LOG"
    note "running GPU generate run 1 (warmup) ..."
    ( cd "$ENGINE_DIR" && MOJO_ENGINE_GPU=1 "$mojo_bin" run -I src src/runner/generate_main.mojo ) > "$GPU_LOG1" 2>&1
    rc=$?
    echo "GPU_RUN1_EXIT=$rc"
    [ "$rc" -eq 0 ] || fail "GPU_GENERATE_RUN1_EXIT_NONZERO: exit $rc, log $GPU_LOG1"
    note "GPU generate run 1 completed; running run 2 (timed) ..."
    ( cd "$ENGINE_DIR" && MOJO_ENGINE_GPU=1 "$mojo_bin" run -I src src/runner/generate_main.mojo ) > "$GPU_LOG2" 2>&1
    rc=$?
    echo "GPU_RUN2_EXIT=$rc"
    [ "$rc" -eq 0 ] || fail "GPU_GENERATE_RUN2_EXIT_NONZERO: exit $rc, log $GPU_LOG2"
    if [ "$WITH_CPU_CONTROL" -eq 1 ]; then
        note "running fresh CPU control (once, MOJO_ENGINE_GPU unset) ..."
        ( cd "$ENGINE_DIR" && env -u MOJO_ENGINE_GPU "$mojo_bin" run -I src src/runner/generate_main.mojo ) > "$CPU_RUN_LOG" 2>&1
        rc=$?
        echo "CPU_CONTROL_EXIT=$rc"
        [ "$rc" -eq 0 ] || fail "CPU_CONTROL_EXIT_NONZERO: exit $rc, log $CPU_RUN_LOG"
        CPU_LOG="$CPU_RUN_LOG"
    fi
}

# ---- main ----
echo "=== WAVE_C_TIMING_BEGIN ==="
if [ "$EXTRACT_ONLY" -eq 1 ]; then
    [ $# -ge 2 ] || fail "ARG: --extract-only needs TEACHER_LOG GPU_RUN1_LOG [GPU_RUN2_LOG]"
    TEACHER_LOG="$1"; GPU_LOG1="$2"; GPU_LOG2="${3:-}"
else
    run_server "$@"
fi

verify_stamp

# Teacher-forced verdict (from the log file, foreground).
TEACHER_VERDICT=$(teacher_verdict_line "$TEACHER_LOG") || fail "TEACHER_FORCED_VERDICT_LINE_MISSING: no verdict line matched in $TEACHER_LOG"
# The verdict's count, in either carried shape (pass=<n>/<m>
# or the verdict line's own bare <n>/<m>), must be full: a
# partial teacher-forced pass is a failing verdict, not a
# summary line (Wave E: the bare shape is checked too - the
# original code only checked pass=, so a 7/8 verdict line
# would have been reported without failing).
_tf_count=$(printf '%s' "$TEACHER_VERDICT" | grep -oE 'pass=[0-9]+/[0-9]+' | tail -n 1 || true)
if [ -z "$_tf_count" ]; then
    _tf_count=$(printf '%s' "$TEACHER_VERDICT" | grep -oE '[0-9]+/[0-9]+' | tail -n 1 || true)
fi
if [ -n "$_tf_count" ]; then
    _tf_n=${_tf_count#pass=}; _tf_num=${_tf_n%/*}; _tf_den=${_tf_n#*/}
    [ "$_tf_num" = "$_tf_den" ] || fail "TEACHER_FORCED_NOT_FULL_PASS: $_tf_count in $TEACHER_LOG"
fi

# GPU runs: run 1 always; run 2 (timed) when present.
TIMED_LABEL=1

print_summary() {
    echo "=== WAVE_C_TIMING_SUMMARY_BEGIN ==="
    echo "STAGED_COMMIT=$STAMP_VAL_PRINT"
    echo "STAGED_COMMIT_FILE=$STAMP_PATH_PRINT"
    echo "EXPECTED_COMMIT=${EXPECTED:-UNSPECIFIED}"
    [ -n "$STAMP_HASH_PRINT" ] && echo "STAGED_FILE_HASH=$STAMP_HASH_PRINT"
    echo "TEACHER_LOG=$TEACHER_LOG"
    echo "GPU_RUN1_LOG=$GPU_LOG1"
    echo "GPU_RUN2_LOG=${GPU_LOG2:-NONE}"
    echo "CPU_CONTROL_LOG=${CPU_LOG:-NONE}"
    echo "TEACHER_FORCED_VERDICT_LINE=$TEACHER_VERDICT"
    # Per-run blocks, each read fresh from its log file.
    extract_generate "$GPU_LOG1" "GPU_RUN1_"
    local timed_log="$GPU_LOG1"
    if [ -n "$GPU_LOG2" ]; then
        extract_generate "$GPU_LOG2" "GPU_RUN2_"
        timed_log="$GPU_LOG2"; TIMED_LABEL=2
    fi
    echo "TIMED_RUN=$TIMED_LABEL"
    # Canonical unprefixed GPU numbers = the timed run, re-read
    # from the timed run's log file.
    extract_generate "$timed_log" "GPU_"
    # Also expose the same timed numbers unprefixed, per the
    # summary contract (MODEL_PATH_KIND, TOKENS, ...).
    extract_generate "$timed_log" ""
    # GPU timed values for ratios (globals set by last extract).
    local g_open="$G_MODEL_OPEN_MS" g_pf="$G_PREFILL_MS" g_dec_s="$G_DECODE_S_PER_STEP" g_dec_ms="$G_DECODE_MS"
    [ "$(field_value "$timed_log" MODEL_PATH_KIND)" = "gpu" ] || fail "MODEL_PATH_KIND_NOT_GPU: timed GPU log $timed_log does not print MODEL_PATH_KIND gpu"
    echo "DECODE_STEPS=$DECODE_STEPS"
    echo "DECODE_FORMULA=decode_ms=GENERATE_MS-PREFILL_MS; decode_s_per_step=decode_ms/15/1000; decode_tok_s=15*1000/decode_ms"
    # CPU comparison: fresh control if present, else recorded.
    if [ -n "$CPU_LOG" ]; then
        echo "CPU_CONTROL_SOURCE=fresh_staged_tree"
        extract_generate "$CPU_LOG" "CPU_FRESH_"
        # G_* now hold the fresh CPU values; g_* hold the GPU timed run.
        local c_open="$G_MODEL_OPEN_MS" c_pf="$G_PREFILL_MS" c_dec_ms="$G_DECODE_MS"
        echo "RATIO_MODEL_OPEN_CPU_FRESH_OVER_GPU=$(ratio "$c_open" "$g_open")"
        echo "RATIO_PREFILL_CPU_FRESH_OVER_GPU=$(ratio "$c_pf" "$g_pf")"
        # s/step ratio over the same step count = decode_ms ratio.
        echo "RATIO_DECODE_CPU_FRESH_OVER_GPU=$(ratio "$c_dec_ms" "$g_dec_ms")"
    else
        echo "CPU_CONTROL_SOURCE=recorded_wave_b_not_concurrent"
    fi
    echo "CPU_WAVE_B_MODEL_OPEN_MS=$CPU_WB_MODEL_OPEN_MS"
    echo "CPU_WAVE_B_PREFILL_MS=$CPU_WB_PREFILL_MS"
    echo "CPU_WAVE_B_GENERATE_MS=$CPU_WB_GENERATE_MS"
    echo "CPU_WAVE_B_DECODE_S_PER_STEP=$CPU_WB_DECODE_S_PER_STEP"
    echo "RATIO_MODEL_OPEN_CPU_WAVE_B_OVER_GPU=$(ratio "$CPU_WB_MODEL_OPEN_MS" "$g_open")"
    echo "RATIO_PREFILL_CPU_WAVE_B_OVER_GPU=$(ratio "$CPU_WB_PREFILL_MS" "$g_pf")"
    echo "RATIO_DECODE_CPU_WAVE_B_OVER_GPU=$(ratio_decode "$CPU_WB_DECODE_S_PER_STEP" "$g_dec_ms")"
    echo "=== WAVE_C_TIMING_SUMMARY_END ==="
}
print_summary
echo "=== WAVE_C_TIMING_END ==="
