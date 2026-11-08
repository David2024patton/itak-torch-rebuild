#!/usr/bin/env bash
# run_d3_window.sh - Wave D lane D3 server window: correctness
# gates FIRST, then interleaved port-vs-tuned timing for the
# Q6_K, Q5_K, F32 and F16 GEMV kernels, on the real Glimmer
# file's tensors (fixtures built by extract_fixtures.py).
#
# Runs ON ubuntu-server. Expects:
#   ~/d3-stage/port/    git archive of the Wave A base tree
#                       (mojo-engine/ subtree) + STAGED_COMMIT
#   ~/d3-stage/tuned/   git archive of the lane head tree
#                       (mojo-engine/ subtree) + STAGED_COMMIT
#   ~/d3-stage/tools/   extract_fixtures.py + compare_one.py
#   MODEL env           path to the Glimmer GGUF (default the
#                       k2_check default path)
#
# Server rules honored here (Wave D standing orders): the
# phase lock is held for the whole window; ONLY
# cortex-embed.service is stopped and restored (the cortex
# containers are never stopped); VRAM bookends are printed
# from sysfs before the stop, after the stop, and after the
# restore; port 8080 is checked clear (Glimmer serving stays
# down); every load-bearing number is printed by this script
# from the runs' own stdout in the foreground.
set -u
STAGE="$HOME/d3-stage"
MODEL="${MODEL:-/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf}"
export PIXI_HOME="$HOME/mojo-probe/pixi"
export PATH="$HOME/mojo-probe/pixi/bin:$PATH"
export PIXI_NO_TELEMETRY=1
FIX="$STAGE/fixtures"
BIN="$STAGE/bin"
mkdir -p "$BIN" "$STAGE/logs"

exec 9>/tmp/itak-phase.lock
echo "window: waiting for phase lock"
flock -w 14400 9
echo "window: phase lock held"

vram_used() { cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | head -1; }

echo "window: STAGED port: $(cat "$STAGE/port/STAGED_COMMIT" 2>/dev/null)"
echo "window: STAGED tuned: $(cat "$STAGE/tuned/STAGED_COMMIT" 2>/dev/null)"
sha256sum "$STAGE/port/mojo-engine/src/kernels/q6k_gemv.mojo" \
          "$STAGE/tuned/mojo-engine/src/kernels/q6k_gemv.mojo" \
          "$STAGE/tuned/mojo-engine/src/kernels/q5k_gemv.mojo" \
          "$STAGE/tuned/mojo-engine/src/kernels/f32_gemv.mojo"
echo "window: VRAM before clear: $(vram_used)"
echo "window: port 8080 listeners: $(ss -ltn | grep -c ':8080 ' || true)"
docker ps --format '{{.Names}} {{.Status}}' | grep -E 'cortex' || true

restore() {
  echo "window: restoring cortex-embed.service"
  systemctl --user start cortex-embed.service
  sleep 10
  echo "window: cortex-embed.service active: $(systemctl --user is-active cortex-embed.service)"
  echo "window: VRAM after restore: $(vram_used)"
  docker ps --format '{{.Names}} {{.Status}}' | grep -E 'cortex' || true
}
trap restore EXIT

echo "window: stopping cortex-embed.service (the only tenant this window stops)"
systemctl --user stop cortex-embed.service
sleep 5
echo "window: VRAM after clear: $(vram_used)"
echo "window: port 8080 listeners after clear: $(ss -ltn | grep -c ':8080 ' || true)"

if [ ! -d "$FIX/q6k_time" ]; then
  echo "window: building fixtures from $MODEL"
  python3 "$STAGE/tools/extract_fixtures.py" "$MODEL" "$FIX" 2>&1 | grep -v Warning
fi

build_arm() {
  local arm="$1" tree="$2"
  for k in q6k_gemv q5k_gemv f32_gemv; do
    if [ ! -x "$BIN/${arm}_${k}" ]; then
      echo "window: building ${arm}_${k}"
      ( cd "$tree/mojo-engine" && pixi run mojo build -I src \
          --target-accelerator gfx1100 "src/kernels/${k}.mojo" \
          -o "$BIN/${arm}_${k}" ) 2>&1 | tail -2
    fi
  done
}
build_arm port "$STAGE/port"
build_arm tuned "$STAGE/tuned"

echo "===== CORRECTNESS GATES (tuned arm first, then port control) ====="
for arm in tuned port; do
  echo "--- gate $arm q6k_corr (blk.0.ffn_down rows 0..63 x 19968) ---"
  "$BIN/${arm}_q6k_gemv" "$FIX/q6k_corr" --rows 64 --cols 19968 --iters 0 \
      --out "$FIX/q6k_corr/${arm}_out.bin" | grep -E 'MOJO_OUT|REFUSAL'
  python3 "$STAGE/tools/compare_one.py" "$FIX/q6k_corr" "$FIX/q6k_corr/${arm}_out.bin" "${arm}_q6k_corr"
  echo "--- gate $arm q5k_corr (output.weight rows 0..63 x 6656) ---"
  "$BIN/${arm}_q5k_gemv" "$FIX/q5k_corr" --rows 64 --cols 6656 --iters 0 \
      --out "$FIX/q5k_corr/${arm}_out.bin" | grep -E 'MOJO_OUT|REFUSAL'
  python3 "$STAGE/tools/compare_one.py" "$FIX/q5k_corr" "$FIX/q5k_corr/${arm}_out.bin" "${arm}_q5k_corr"
  echo "--- gate $arm f32_corr (first 64 real F32 [6656] rows) ---"
  "$BIN/${arm}_f32_gemv" "$FIX/f32_corr" --dtype f32 --rows 64 --cols 6656 --iters 0 \
      --out "$FIX/f32_corr/${arm}_out.bin" | grep -E 'MOJO_OUT|REFUSAL'
  python3 "$STAGE/tools/compare_one.py" "$FIX/f32_corr" "$FIX/f32_corr/${arm}_out.bin" "${arm}_f32_corr"
  echo "--- gate $arm f16_corr (F16 encoding of the same rows) ---"
  "$BIN/${arm}_f32_gemv" "$FIX/f16_corr" --dtype f16 --rows 64 --cols 6656 --iters 0 \
      --out "$FIX/f16_corr/${arm}_out.bin" | grep -E 'MOJO_OUT|REFUSAL'
  python3 "$STAGE/tools/compare_one.py" "$FIX/f16_corr" "$FIX/f16_corr/${arm}_out.bin" "${arm}_f16_corr"
done

echo "===== INTERLEAVED TIMING (port arm and tuned arm alternate per rep) ====="
time_rep() {
  # $1 arm, $2 kernel-binary-suffix, $3 fixture, $4 extra args
  local arm="$1" kern="$2" fixture="$3"; shift 3
  "$BIN/${arm}_${kern}" "$FIX/$fixture" "$@" \
      --out "$FIX/$fixture/${arm}_timed_out.bin" | grep -E 'MOJO_TIMING|REFUSAL'
}
for rep in 1 2 3; do
  echo "--- rep $rep q6k_time (full blk.0.ffn_down, 109025280 B/iter, iters 20) ---"
  time_rep port q6k_gemv q6k_time --rows 6656 --cols 19968 --iters 20 --warmup 3
  time_rep tuned q6k_gemv q6k_time --rows 6656 --cols 19968 --iters 20 --warmup 3
  echo "--- rep $rep q5k_time (output.weight rows 0..16383, 74973184 B/iter, iters 20) ---"
  time_rep port q5k_gemv q5k_time --rows 16384 --cols 6656 --iters 20 --warmup 3
  time_rep tuned q5k_gemv q5k_time --rows 16384 --cols 6656 --iters 20 --warmup 3
  echo "--- rep $rep f32_time (SYNTHETIC 6656x6656, 177209344 B/iter, iters 10) ---"
  time_rep port f32_gemv f32_time --dtype f32 --rows 6656 --cols 6656 --iters 10 --warmup 2
  time_rep tuned f32_gemv f32_time --dtype f32 --rows 6656 --cols 6656 --iters 10 --warmup 2
done
for rep in 1 2; do
  echo "--- rep $rep f16_time (SYNTHETIC F16 6656x6656, 88604672 B/iter, iters 5) ---"
  time_rep port f32_gemv f16_time --dtype f16 --rows 6656 --cols 6656 --iters 5 --warmup 2
  time_rep tuned f32_gemv f16_time --dtype f16 --rows 6656 --cols 6656 --iters 5 --warmup 2
done

echo "===== FULL-FIXTURE CORRECTNESS OF THE TIMED RUNS' FIRST LAUNCH ====="
for arm in port tuned; do
  python3 "$STAGE/tools/compare_one.py" "$FIX/q6k_time" "$FIX/q6k_time/${arm}_timed_out.bin" "${arm}_q6k_time"
  python3 "$STAGE/tools/compare_one.py" "$FIX/q5k_time" "$FIX/q5k_time/${arm}_timed_out.bin" "${arm}_q5k_time"
  python3 "$STAGE/tools/compare_one.py" "$FIX/f32_time" "$FIX/f32_time/${arm}_timed_out.bin" "${arm}_f32_time"
  python3 "$STAGE/tools/compare_one.py" "$FIX/f16_time" "$FIX/f16_time/${arm}_timed_out.bin" "${arm}_f16_time"
done
echo "window: done"
