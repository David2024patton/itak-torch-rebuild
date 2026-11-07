#!/bin/bash
# Interleaved K1 race driver (bench tooling for the Mojo kernel
# test; see bench/mojo-kernel-test-2026-11-07.md). Runs on the
# server: holds the phase lock, clears the card tenants, proves
# VRAM drained, then alternates the wgpu and Mojo arms.
#   K1_BENCH=wgpu|mojo|both   arm selection (default both)
#   REPS=N ITERS=N WARMUP=N   shape of the race (defaults 3/200/20)
set -u
ARM="${K1_BENCH:-both}"
REPS="${REPS:-3}"
ITERS="${ITERS:-200}"
WARMUP="${WARMUP:-20}"
STAGE="$HOME/gpu-mojo-test"
FIX="$STAGE/fixture"
BIN="$STAGE/target/release/examples"
export PATH="$HOME/.cargo/bin:$HOME/mojo-probe/pixi/bin:$PATH"
export PIXI_HOME="$HOME/mojo-probe/pixi"
export PIXI_NO_TELEMETRY=1

exec 9>/tmp/itak-phase.lock
echo "race: waiting for phase lock"
flock -w 14400 9
echo "race: phase lock held"

vram_used() { cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | head -1; }

echo "race: VRAM before clear: $(vram_used)"
echo "race: stopping tenants (cortex-embed.service, cortex-cortex-chat-1, cortex-cortex-embed-1)"
systemctl --user stop cortex-embed.service
docker stop cortex-cortex-chat-1 cortex-cortex-embed-1 > /dev/null
sleep 5
echo "race: VRAM after clear: $(vram_used)"

restore() {
  echo "race: restoring tenants"
  docker start cortex-cortex-chat-1 cortex-cortex-embed-1 > /dev/null
  systemctl --user start cortex-embed.service
  sleep 10
  echo "race: VRAM after restore: $(vram_used)"
  systemctl --user is-active cortex-embed.service
  docker ps --format '{{.Names}} {{.Status}}' | grep -E 'cortex-cortex-(chat|embed)-1'
}
trap restore EXIT

echo "=== fixture dump (wgpu correctness vs CPU reference) ==="
"$BIN/k1_fixture_dump" --out "$FIX"
echo "=== k1_check anchor (official harness) ==="
"$BIN/k1_check"

echo "=== mojo correctness run ==="
( cd ~/mojo-probe/probe && MOJO_KERNEL_DEBUG=1 ./k1_gemv "$FIX" --iters 0 )

echo "=== mojo compile timing (clean rebuild) ==="
( cd ~/mojo-probe/probe && rm -f k1_gemv && /usr/bin/time -f "mojo build wall: %e s" pixi run mojo build k1_gemv.mojo -o k1_gemv 2>&1 | tail -2 )

echo "=== interleaved race: REPS=$REPS ITERS=$ITERS WARMUP=$WARMUP ARM=$ARM ==="
for rep in $(seq 1 "$REPS"); do
  if [ "$ARM" = "both" ] || [ "$ARM" = "wgpu" ]; then
    echo "--- rep $rep wgpu ---"
    "$BIN/k1_bench" --budget-mib 72 --iters "$ITERS" --warmup "$WARMUP" | grep -E "resident set|iters|verdict" 
  fi
  if [ "$ARM" = "both" ] || [ "$ARM" = "mojo" ]; then
    echo "--- rep $rep mojo ---"
    ( cd ~/mojo-probe/probe && ./k1_gemv "$FIX" --iters "$ITERS" --warmup "$WARMUP" --out "$FIX/mojo_out_rep$rep.bin" | grep -E "MOJO_TIMING|MOJO_OUT|REFUSAL" )
  fi
done
echo "race: done"
