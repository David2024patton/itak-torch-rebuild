#!/bin/bash
# Small-batch sweep: build the harness once, then run
# the variant matrix (control, K18, MMV, chunked)
# through the session's own routing ladder and write
# one timestamped CSV per arm under bench/results/.
#
# Each arm is the same binary (examples/mmv_sweep.rs)
# under a different environment; the harness sets no
# routing env itself and stamps the active env into
# every CSV row. The kernel column of each CSV shows
# which kernel form actually served each cell, so an
# arm whose env name does not match a landed kernel
# reads as the control kernels it really ran, never
# as a silent pass.
#
# The chunked arm's env name is the one line to check
# when the chunked-GEMV prototype lands: it shipped as
# WGPU_COMPUTE_CHUNKED_GEMV_MAX_ROWS (corrected at merge;
# the harness lane had guessed WGPU_COMPUTE_CHUNK_MAX_ROWS
# before the arm landed). Change CHUNK_ENV below and
# nothing else if it ever moves again.
#
# Usage: bench/mmv-sweep.sh [extra mmv_sweep args...]
# Extra args forward to every arm (for example
# --rows 1,2,7 for a partial sweep). Verdict runs take
# no extra args: the defaults are the discipline
# (25 timed repetitions per cell, 3 warmup).
set -u
cd "$(dirname "$0")/.."

CHUNK_ENV=WGPU_COMPUTE_CHUNKED_GEMV_MAX_ROWS
EXTRA=("$@")

export PATH="$HOME/.cargo/bin:$PATH"
cargo build --release -p wgpu-compute --example mmv_sweep || exit 1
BIN=target/release/examples/mmv_sweep

TS=$(date +%Y%m%d-%H%M%S)
OUTDIR="bench/results/mmv-sweep-$TS"
mkdir -p "$OUTDIR"
echo "results: $OUTDIR"

# Diag timestamps are the timing method when the
# device grants them; the harness falls back to wall
# clock per repetition and prints which it used (the
# arm logs record it).
export WGPU_COMPUTE_DIAG=timing

run_arm() {
    label=$1; shift
    echo "arm: $label"
    env "$@" "$BIN" --label "$label" --out "$OUTDIR/$label.csv" "${EXTRA[@]}" \
        > "$OUTDIR/$label.log" 2>&1
    tail -2 "$OUTDIR/$label.log"
}

run_arm control \
    -u WGPU_COMPUTE_SMALL_BATCH_MAX_ROWS \
    -u WGPU_COMPUTE_MMV_MAX_ROWS \
    -u WGPU_COMPUTE_WIDE_MIN_ROWS \
    -u "$CHUNK_ENV"
run_arm k18 \
    WGPU_COMPUTE_SMALL_BATCH_MAX_ROWS=16
run_arm mmv \
    WGPU_COMPUTE_MMV_MAX_ROWS=16
run_arm chunked \
    "$CHUNK_ENV=16"

echo "done: $OUTDIR"
ls -la "$OUTDIR"
