#!/bin/bash
# Overnight baseline agreement run, server side.
# Invoked under the phase lock from the operator shell:
#   flock -w 14400 /tmp/itak-phase.lock bash ~/overnight-baseline/bench/overnight_baseline_run.sh
# CPU only. Never touches ports 8080/8091/8092. Production stays down.
set -u
cd ~/overnight-baseline || exit 1
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_BUILD_JOBS=4
export MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=0
MODEL=/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
OUT=~/overnight-baseline/results
mkdir -p "$OUT"

echo "== wait for any gpu-wave6 diag run to clear =="
for i in $(seq 1 120); do
  if pgrep -af "gpu-wave6" | grep -E "cargo|glimmer_generate|itak" | grep -v grep > /dev/null; then
    echo "gpu-wave6 activity present (check $i); waiting 60 s"
    sleep 60
  else
    echo "gpu-wave6 clear"
    break
  fi
done

echo "== environment =="
{
  echo "host: $(hostname)  date: $(date -Is)"
  echo "staged commit: $(cat BUILD_COMMIT 2>/dev/null || echo unknown)"
  uname -a
  free -g | head -2
  sha256sum -c MANIFEST.txt 2>/dev/null | grep -cv ': OK$' | sed 's/^/manifest mismatches: /'
} | tee "$OUT/environment.txt"

echo "== build our example (release, CPU reference) =="
cargo build --release -p itak-loader --example glimmer_generate 2>&1 | tail -3

echo "== agreement run =="
python3 bench/overnight_baseline.py \
  --repo ~/overnight-baseline \
  --model "$MODEL" \
  --out-dir "$OUT" 2>&1 | tee "$OUT/driver.log"
echo "driver exit: ${PIPESTATUS[0]}"
