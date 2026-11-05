#!/bin/bash
# Prefill profile: per-kernel breakdown of a ~1,022-token
# prefill on the real Glimmer file, resident GPU path, using
# the diag timestamp instrumentation (lane/prefill-profile).
#
# Runs on ubuntu-server inside the phase lock, from the root
# of the staged tree:
#   flock -w 14400 /tmp/itak-phase.lock bash bench/prefill-profile-run.sh
#
# Session contents, in order:
#   build (release: glimmer_generate + prefill_profile)
#   ON run 1: prefill_profile, WGPU_COMPUTE_DIAG=timing,
#             summary CSV + raw records in the out dir
#   ON run 2: same, second CSV (stability repeat)
#   OFF run:  prefill_profile with the diag env unset (the
#             timing-off control through the same harness)
#   A/B:      glimmer_generate --gpu (the shipped monolithic
#             path) once with the diag env unset, once with
#             WGPU_COMPUTE_DIAG=timing; the phase=prefill wall
#             lines price what enabling timing costs the
#             shipped path itself
#
# Prompts are the bench protocol's own (bench/mmq-wide-bench.sh):
# the repeated fox sentence, 102 repetitions for the long
# length; the runner reports the true token count and all
# math uses that, not the target.
#
# Logs and CSVs land in $ROOT/prefill-profile-out/.
set -u
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_BUILD_JOBS=4
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=0

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2
OUT="$ROOT/prefill-profile-out"
mkdir -p "$OUT"
MODEL="${1:-/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf}"
[ -f "$MODEL" ] || { echo "FAIL: model not found: $MODEL" | tee -a "$OUT/summary.txt"; exit 1; }

echo "=== prefill profile $(date -Is) ===" | tee -a "$OUT/summary.txt"
echo "root: $ROOT" | tee -a "$OUT/summary.txt"
echo "model: $MODEL" | tee -a "$OUT/summary.txt"

# Card-clear gate (taint-window protocol, 2026-11-05): the
# tenants David ordered stopped must be DOWN for every
# measurement in this session. VRAM is read from the amdgpu
# sysfs counters (rocm-smi is not installed on this host).
# The cleared state reads about 33 MB used (driver residue);
# 500 MB is the abort threshold, far above residue and far
# below the tenants' ~6.4 GB. The reading is recorded in the
# summary and quoted in the profile note. Checked at session
# start and again immediately before the first measurement;
# a resident tenant aborts the session instead of tainting
# a number.
vram_check() {
  local when="$1" dev total used
  for dev in /sys/class/drm/card[0-9]*/device; do
    [ -f "$dev/mem_info_vram_total" ] || continue
    total="$(cat "$dev/mem_info_vram_total")"
    [ "$total" -gt 20000000000 ] || continue
    used="$(cat "$dev/mem_info_vram_used")"
    echo "vram_check($when): dev=$dev used_bytes=$used total_bytes=$total" | tee -a "$OUT/summary.txt"
    if [ "$used" -gt 500000000 ]; then
      echo "FAIL: card not clear at $when (used $used bytes); tenants resident, refusing to measure" | tee -a "$OUT/summary.txt"
      exit 1
    fi
    return 0
  done
  echo "FAIL: no amdgpu VRAM counters found at $when" | tee -a "$OUT/summary.txt"
  exit 1
}
vram_check "session-start"

cargo build --release -p itak-loader --example glimmer_generate 2>&1 | tail -2
cargo build --release -p itak-loader --example prefill_profile 2>&1 | tail -2
GEN="$ROOT/target/release/examples/glimmer_generate"
PROF="$ROOT/target/release/examples/prefill_profile"
[ -x "$GEN" ] && [ -x "$PROF" ] || { echo "FAIL: example build missing" | tee -a "$OUT/summary.txt"; exit 1; }

sentence="The quick brown fox jumps over the lazy dog. "
prompt_long=""
for _ in $(seq 1 102); do prompt_long="$prompt_long$sentence"; done

prefill_ms() { grep -E '^phase=prefill ' "$1" | tail -1 | sed -E 's/.*ms=([0-9]+).*/\1/'; }

# --- ON runs (diag timing) ---
vram_check "before-first-measurement"
for rep in 1 2; do
  log="$OUT/on-$rep.log"
  WGPU_COMPUTE_DIAG=timing WGPU_COMPUTE_DIAG_OUT="$OUT/prefill-on-$rep.csv" \
    "$PROF" "$MODEL" "$prompt_long" > "$log" 2>&1
  rc=$?
  ms="$(prefill_ms "$log")"
  echo "ON rep $rep: rc=$rc prefill_ms=${ms:-MISSING} (log $log)" | tee -a "$OUT/summary.txt"
  [ "$rc" -eq 0 ] && [ -f "$OUT/prefill-on-$rep.csv" ] || { echo "FAIL: ON rep $rep" | tee -a "$OUT/summary.txt"; exit 1; }
done

# --- OFF run (same harness, diag env unset) ---
log="$OUT/off.log"
"$PROF" "$MODEL" "$prompt_long" > "$log" 2>&1
rc=$?
ms="$(prefill_ms "$log")"
echo "OFF: rc=$rc prefill_ms=${ms:-MISSING} (log $log)" | tee -a "$OUT/summary.txt"
[ "$rc" -eq 0 ] || { echo "FAIL: OFF run" | tee -a "$OUT/summary.txt"; exit 1; }

# --- glimmer_generate A/B on the shipped path ---
log="$OUT/gen-off.log"
"$GEN" "$MODEL" "$prompt_long" 4 --gpu > "$log" 2>&1
rc=$?
ms="$(prefill_ms "$log")"
echo "GEN-OFF: rc=$rc prefill_ms=${ms:-MISSING} (log $log)" | tee -a "$OUT/summary.txt"
[ "$rc" -eq 0 ] || { echo "FAIL: GEN-OFF run" | tee -a "$OUT/summary.txt"; exit 1; }

log="$OUT/gen-on.log"
WGPU_COMPUTE_DIAG=timing "$GEN" "$MODEL" "$prompt_long" 4 --gpu > "$log" 2>&1
rc=$?
ms="$(prefill_ms "$log")"
echo "GEN-ON: rc=$rc prefill_ms=${ms:-MISSING} (log $log)" | tee -a "$OUT/summary.txt"
[ "$rc" -eq 0 ] || { echo "FAIL: GEN-ON run" | tee -a "$OUT/summary.txt"; exit 1; }

echo "PREFILL_PROFILE_DONE" | tee -a "$OUT/summary.txt"
