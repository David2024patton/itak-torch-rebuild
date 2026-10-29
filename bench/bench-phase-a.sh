#!/bin/bash
# Phase A: llama-bench speed + VRAM, Vulkan vs HIP/ROCm x Q3/Q4/Q5/Q6
# Run on ubuntu-server after Glimmer downloads finish. Sequential, logs to /tmp/bench-results/
set -u
OUT=/tmp/bench-results
mkdir -p "$OUT"
MODELS_DIR=/home/server/models/muse-glimmer
declare -A MODELS=(
  [Q3_K_XL]="$MODELS_DIR/Muse-Glimmer-30B-UD-Q3_K_XL.gguf"
  [Q4_K_M]="$MODELS_DIR/meta/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
  [Q5_K_M]="$MODELS_DIR/Muse-Glimmer-30B-UD-Q5_K_M.gguf"
  [Q6_K_XL]="$MODELS_DIR/Muse-Glimmer-30B-UD-Q6_K_XL.gguf"
)
declare -A BACKENDS=(
  [vulkan]=/home/server/llama.cpp/build-vulkan/bin/llama-bench
  [rocm]=/home/server/llama.cpp/build-rocm/bin/llama-bench
)
vram_used() { cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | sort -n | tail -1; }
for b in vulkan rocm; do
  for q in Q3_K_XL Q4_K_M Q5_K_M Q6_K_XL; do
    m="${MODELS[$q]}"; bin="${BACKENDS[$b]}"
    [ -f "$m" ] || { echo "SKIP $b $q missing model" >> "$OUT/summary.txt"; continue; }
    log="$OUT/bench_${b}_${q}.log"
    peak="$OUT/vram_${b}_${q}.txt"
    echo "=== $b $q $(date -Is) ===" | tee -a "$OUT/summary.txt"
    ( max=0; while true; do v=$(vram_used); [ -n "$v" ] && [ "$v" -gt "$max" ] && max=$v; echo "$max" > "$peak"; sleep 2; done ) &
    SAMPLER=$!
    "$bin" -m "$m" -ngl 999 -p 512 -n 128 -r 3 > "$log" 2>&1
    echo "BENCH_EXIT:$?" >> "$log"
    kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null
    grep -E "llama-bench|error|failed" "$log" | tail -4 >> "$OUT/summary.txt" || true
    echo "peak_vram_bytes: $(cat "$peak" 2>/dev/null)" >> "$OUT/summary.txt"
  done
done
echo "PHASE_A_DONE" >> "$OUT/summary.txt"
