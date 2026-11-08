#!/usr/bin/env bash
# run_d3_window3.sh - Wave D lane D3 server window 3: FINAL
# verification of the lane head tree (staged fresh into
# ~/d3-stage/final/, STAGED_COMMIT printed):
#   1. correctness gates for the final Q6_K / Q5_K (rows=4) /
#      F32 (restored port) / F16 (tuned) kernels on the corr
#      fixtures;
#   2. spot timing reps for final Q6_K and Q5_K against
#      window 1/2's kept binaries;
#   3. the end-to-end GPU golden on the final tree, the
#      Wave C protocol: teacher-forced proof, then the
#      runner (MOJO_ENGINE_GPU=1), whose GOLDEN_FIRST8_MATCH
#      line is the verdict. This exercises the tuned kernels
#      through gpu_layer on the real model (Q6_K ffn_down +
#      attn_v, Q5_K rows=4 head over all 202048 rows, K1,
#      attention) - the integration proof that the tile and
#      launch changes did not break the forward.
#
# Same server rules as windows 1-2: phase lock held, only
# cortex-embed.service stopped/restored, VRAM bookends,
# port 8080 checked clear, gates FIRST.
set -u
STAGE="$HOME/d3-stage"
export PIXI_HOME="$HOME/mojo-probe/pixi"
export PATH="$HOME/mojo-probe/pixi/bin:$PATH"
export PIXI_NO_TELEMETRY=1
FIX="$STAGE/fixtures"
BIN="$STAGE/bin"
FINAL="$STAGE/final"

exec 9>/tmp/itak-phase.lock
echo "window3: waiting for phase lock"
flock -w 14400 9
echo "window3: phase lock held"

vram_used() { cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | head -1; }

echo "window3: STAGED final: $(cat "$FINAL/STAGED_COMMIT" 2>/dev/null)"
sha256sum "$FINAL/mojo-engine/src/kernels/q6k_gemv.mojo" \
          "$FINAL/mojo-engine/src/kernels/q5k_gemv.mojo" \
          "$FINAL/mojo-engine/src/kernels/f32_gemv.mojo" \
          "$FINAL/mojo-engine/src/model/gpu_layer.mojo"
echo "window3: VRAM before clear: $(vram_used)"
echo "window3: port 8080 listeners: $(ss -ltn | grep -c ':8080 ' || true)"

restore() {
  echo "window3: restoring cortex-embed.service"
  systemctl --user start cortex-embed.service
  sleep 10
  echo "window3: cortex-embed.service active: $(systemctl --user is-active cortex-embed.service)"
  echo "window3: VRAM after restore: $(vram_used)"
  docker ps --format '{{.Names}} {{.Status}}' | grep -E 'cortex' || true
}
trap restore EXIT

systemctl --user stop cortex-embed.service
sleep 5
echo "window3: VRAM after clear: $(vram_used)"

echo "window3: building final arms"
for k in q6k_gemv q5k_gemv f32_gemv; do
  ( cd "$FINAL/mojo-engine" && pixi run mojo build -I src \
      --target-accelerator gfx1100 "src/kernels/${k}.mojo" \
      -o "$BIN/final_${k}" ) 2>&1 | tail -2
done

echo "===== CORRECTNESS GATES (final tree) ====="
"$BIN/final_q6k_gemv" "$FIX/q6k_corr" --rows 64 --cols 19968 --iters 0 \
    --out "$FIX/q6k_corr/final_out.bin" | grep -E 'MOJO_OUT|REFUSAL'
python3 "$STAGE/tools/compare_one.py" "$FIX/q6k_corr" "$FIX/q6k_corr/final_out.bin" "final_q6k_corr"
"$BIN/final_q5k_gemv" "$FIX/q5k_corr" --rows 64 --cols 6656 --iters 0 \
    --out "$FIX/q5k_corr/final_out.bin" | grep -E 'MOJO_OUT|REFUSAL'
python3 "$STAGE/tools/compare_one.py" "$FIX/q5k_corr" "$FIX/q5k_corr/final_out.bin" "final_q5k_corr"
"$BIN/final_f32_gemv" "$FIX/f32_corr" --dtype f32 --rows 64 --cols 6656 --iters 0 \
    --out "$FIX/f32_corr/final_out.bin" | grep -E 'MOJO_OUT|REFUSAL'
python3 "$STAGE/tools/compare_one.py" "$FIX/f32_corr" "$FIX/f32_corr/final_out.bin" "final_f32_corr"
"$BIN/final_f32_gemv" "$FIX/f16_corr" --dtype f16 --rows 64 --cols 6656 --iters 0 \
    --out "$FIX/f16_corr/final_out.bin" | grep -E 'MOJO_OUT|REFUSAL'
python3 "$STAGE/tools/compare_one.py" "$FIX/f16_corr" "$FIX/f16_corr/final_out.bin" "final_f16_corr"

echo "===== SPOT TIMING (final vs kept window binaries) ====="
for rep in 1 2; do
  echo "--- rep $rep q6k_time final ---"
  "$BIN/final_q6k_gemv" "$FIX/q6k_time" --rows 6656 --cols 19968 --iters 20 --warmup 3 \
      --out "$FIX/q6k_time/final_timed_out.bin" | grep -E 'MOJO_TIMING|REFUSAL'
  echo "--- rep $rep q5k_time final (rows=4) ---"
  "$BIN/final_q5k_gemv" "$FIX/q5k_time" --rows 16384 --cols 6656 --iters 20 --warmup 3 \
      --out "$FIX/q5k_time/final_timed_out.bin" | grep -E 'MOJO_TIMING|REFUSAL'
done
python3 "$STAGE/tools/compare_one.py" "$FIX/q6k_time" "$FIX/q6k_time/final_timed_out.bin" "final_q6k_time"
python3 "$STAGE/tools/compare_one.py" "$FIX/q5k_time" "$FIX/q5k_time/final_timed_out.bin" "final_q5k_time"

echo "===== GPU GOLDEN ON THE FINAL TREE (Wave C protocol) ====="
( cd "$FINAL/mojo-engine" && MOJO_ENGINE_GPU=1 pixi run mojo run -I src \
    tests/server_proof_gpu_generate.mojo ) 2>&1 | grep -E 'TEACHER|GOLDEN|REFUSAL|PASS|FAIL|MATCH' | head -20
( cd "$FINAL/mojo-engine" && MOJO_ENGINE_GPU=1 pixi run mojo run -I src \
    src/runner/generate_main.mojo ) 2>&1 | grep -E 'GOLDEN|TOKENS|TEXT|PREFILL|DECODE|REFUSAL|MODEL_PATH_KIND' | head -20
echo "window3: done"
