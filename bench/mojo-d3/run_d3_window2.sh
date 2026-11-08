#!/usr/bin/env bash
# run_d3_window2.sh - Wave D lane D3 server window 2:
# iteration-2 measurements. Arms already on the server from
# window 1 (~/d3-stage/bin/port_*, tuned_* = iteration 1) are
# raced against the final lane tree staged fresh into
# ~/d3-stage/tuned2/ (STAGED_COMMIT printed) plus the
# rows=4 Q5_K experiment built from bench/mojo-d3.
#
# Same server rules as window 1: phase lock held, only
# cortex-embed.service stopped/restored, VRAM bookends,
# port 8080 checked clear, correctness gates FIRST.
set -u
STAGE="$HOME/d3-stage"
export PIXI_HOME="$HOME/mojo-probe/pixi"
export PATH="$HOME/mojo-probe/pixi/bin:$PATH"
export PIXI_NO_TELEMETRY=1
FIX="$STAGE/fixtures"
BIN="$STAGE/bin"

exec 9>/tmp/itak-phase.lock
echo "window2: waiting for phase lock"
flock -w 14400 9
echo "window2: phase lock held"

vram_used() { cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | head -1; }

echo "window2: STAGED tuned2: $(cat "$STAGE/tuned2/STAGED_COMMIT" 2>/dev/null)"
sha256sum "$STAGE/tuned2/mojo-engine/src/kernels/q6k_gemv.mojo" \
          "$STAGE/tuned2/mojo-engine/src/kernels/f32_gemv.mojo" \
          "$STAGE/tuned2/bench/mojo-d3/exp_q5k_r4.mojo"
echo "window2: VRAM before clear: $(vram_used)"
echo "window2: port 8080 listeners: $(ss -ltn | grep -c ':8080 ' || true)"

restore() {
  echo "window2: restoring cortex-embed.service"
  systemctl --user start cortex-embed.service
  sleep 10
  echo "window2: cortex-embed.service active: $(systemctl --user is-active cortex-embed.service)"
  echo "window2: VRAM after restore: $(vram_used)"
  docker ps --format '{{.Names}} {{.Status}}' | grep -E 'cortex' || true
}
trap restore EXIT

systemctl --user stop cortex-embed.service
sleep 5
echo "window2: VRAM after clear: $(vram_used)"

echo "window2: building tuned2 arms"
( cd "$STAGE/tuned2/mojo-engine" && pixi run mojo build -I src \
    --target-accelerator gfx1100 src/kernels/q6k_gemv.mojo \
    -o "$BIN/tuned2_q6k_gemv" ) 2>&1 | tail -2
( cd "$STAGE/tuned2/mojo-engine" && pixi run mojo build -I src \
    --target-accelerator gfx1100 src/kernels/f32_gemv.mojo \
    -o "$BIN/tuned2_f32_gemv" ) 2>&1 | tail -2
( cd "$STAGE/tuned2/mojo-engine" && pixi run mojo build -I src \
    --target-accelerator gfx1100 ../bench/mojo-d3/exp_q5k_r4.mojo \
    -o "$BIN/exp_q5k_r4" ) 2>&1 | tail -2

echo "===== CORRECTNESS GATES (window 2 arms) ====="
echo "--- gate tuned2 q6k_corr ---"
"$BIN/tuned2_q6k_gemv" "$FIX/q6k_corr" --rows 64 --cols 19968 --iters 0 \
    --out "$FIX/q6k_corr/tuned2_out.bin" | grep -E 'MOJO_OUT|REFUSAL'
python3 "$STAGE/tools/compare_one.py" "$FIX/q6k_corr" "$FIX/q6k_corr/tuned2_out.bin" "tuned2_q6k_corr"
echo "--- gate exp_q5k_r4 q5k_corr ---"
"$BIN/exp_q5k_r4" "$FIX/q5k_corr" --rows 64 --cols 6656 --iters 0 \
    --out "$FIX/q5k_corr/exp_r4_out.bin" | grep -E 'MOJO_OUT|REFUSAL'
python3 "$STAGE/tools/compare_one.py" "$FIX/q5k_corr" "$FIX/q5k_corr/exp_r4_out.bin" "exp_r4_q5k_corr"
echo "--- gate tuned2 f32_corr (restored port kernel) ---"
"$BIN/tuned2_f32_gemv" "$FIX/f32_corr" --dtype f32 --rows 64 --cols 6656 --iters 0 \
    --out "$FIX/f32_corr/tuned2_out.bin" | grep -E 'MOJO_OUT|REFUSAL'
python3 "$STAGE/tools/compare_one.py" "$FIX/f32_corr" "$FIX/f32_corr/tuned2_out.bin" "tuned2_f32_corr"
echo "--- gate tuned2 f16_corr (kept tuned kernel) ---"
"$BIN/tuned2_f32_gemv" "$FIX/f16_corr" --dtype f16 --rows 64 --cols 6656 --iters 0 \
    --out "$FIX/f16_corr/tuned2_out.bin" | grep -E 'MOJO_OUT|REFUSAL'
python3 "$STAGE/tools/compare_one.py" "$FIX/f16_corr" "$FIX/f16_corr/tuned2_out.bin" "tuned2_f16_corr"

echo "===== INTERLEAVED TIMING (window 2) ====="
for rep in 1 2 3; do
  echo "--- rep $rep q6k_time: iter1 vs iter2 ---"
  "$BIN/tuned_q6k_gemv" "$FIX/q6k_time" --rows 6656 --cols 19968 --iters 20 --warmup 3 \
      --out "$FIX/q6k_time/iter1_timed_out.bin" | grep -E 'MOJO_TIMING|REFUSAL'
  "$BIN/tuned2_q6k_gemv" "$FIX/q6k_time" --rows 6656 --cols 19968 --iters 20 --warmup 3 \
      --out "$FIX/q6k_time/iter2_timed_out.bin" | grep -E 'MOJO_TIMING|REFUSAL'
  echo "--- rep $rep q5k_time: r2 (kept) vs r4 (experiment) ---"
  "$BIN/tuned_q5k_gemv" "$FIX/q5k_time" --rows 16384 --cols 6656 --iters 20 --warmup 3 \
      --out "$FIX/q5k_time/r2_timed_out.bin" | grep -E 'MOJO_TIMING|REFUSAL'
  "$BIN/exp_q5k_r4" "$FIX/q5k_time" --rows 16384 --cols 6656 --iters 20 --warmup 3 \
      --out "$FIX/q5k_time/r4_timed_out.bin" | grep -E 'MOJO_TIMING|REFUSAL'
done
echo "--- f32_time confirmation: port vs tuned2 (restored port kernel) ---"
"$BIN/port_f32_gemv" "$FIX/f32_time" --dtype f32 --rows 6656 --cols 6656 --iters 10 --warmup 2 \
    --out "$FIX/f32_time/port_timed_out.bin" | grep -E 'MOJO_TIMING|REFUSAL'
"$BIN/tuned2_f32_gemv" "$FIX/f32_time" --dtype f32 --rows 6656 --cols 6656 --iters 10 --warmup 2 \
    --out "$FIX/f32_time/tuned2_timed_out.bin" | grep -E 'MOJO_TIMING|REFUSAL'

echo "===== FULL-FIXTURE CORRECTNESS (window 2 timed runs) ====="
python3 "$STAGE/tools/compare_one.py" "$FIX/q6k_time" "$FIX/q6k_time/iter2_timed_out.bin" "iter2_q6k_time"
python3 "$STAGE/tools/compare_one.py" "$FIX/q5k_time" "$FIX/q5k_time/r4_timed_out.bin" "r4_q5k_time"
python3 "$STAGE/tools/compare_one.py" "$FIX/f32_time" "$FIX/f32_time/tuned2_timed_out.bin" "tuned2_f32_time"
echo "window2: done"
