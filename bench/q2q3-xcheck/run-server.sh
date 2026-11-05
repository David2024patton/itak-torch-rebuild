#!/bin/bash
# Server half of the Q2_K/Q3_K dequantizer cross-check.
# Run under the shared phase lock:
#   flock -w 14400 /tmp/itak-phase.lock bash ~/overnight-q2q3/run-server.sh
set -euo pipefail
cd ~/overnight-q2q3
exec >> ~/overnight-q2q3/run.log 2>&1
echo "=== run-server.sh start $(date -Is) ==="

# Wait for any cargo/test processes tied to ~/gpu-wave6 to clear.
for i in $(seq 1 240); do
    if pgrep -af "gpu-wave6" | grep -Eq "cargo|rustc"; then
        echo "gpu-wave6 build still running, waiting ($i)"
        sleep 30
    else
        break
    fi
done

TREE=/home/server/llama.cpp
BIN=""
for d in "$TREE/build-vulkan2/bin" "$TREE/build-hip/bin"; do
    if [ -f "$d/libggml-base.so" ]; then BIN=$d; break; fi
done
if [ -z "$BIN" ]; then echo "no libggml-base.so found"; exit 2; fi
echo "libggml-base from: $BIN"

gcc -O2 -std=c11 -I"$TREE/ggml/include" -I"$TREE/ggml/src" \
    harness.c -o harness -L"$BIN" -lggml-base -lm -Wl,-rpath,"$BIN"

python3 gen_vectors.py

./harness dequant q2k q2k_synth.bin q2k_synth_ref.f32
./harness dequant q3k q3k_synth.bin q3k_synth_ref.f32
./harness dequant q2k q2k_dsweep.bin q2k_dsweep_ref.f32
./harness dequant q3k q3k_dsweep.bin q3k_dsweep_ref.f32
./harness quant q2k quant_in.f32 q2k_quant.bin
./harness quant q3k quant_in.f32 q3k_quant.bin
./harness dequant q2k q2k_quant.bin q2k_quant_ref.f32
./harness dequant q3k q3k_quant.bin q3k_quant_ref.f32

python3 extract_real.py
for t in q2k q3k; do
    if [ -f "${t}_real.bin" ]; then
        ./harness dequant "$t" "${t}_real.bin" "${t}_real_ref.f32"
    fi
done

sha256sum harness.c gen_vectors.py extract_real.py *.bin *.f32 > SHA256SUMS
echo "=== SHA256SUMS ==="
cat SHA256SUMS
echo "=== DONE $(date -Is) ==="
