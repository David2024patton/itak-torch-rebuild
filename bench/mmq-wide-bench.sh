#!/bin/bash
# Wide-tile MMQ prefill proof: session-level A/B of the
# resident prefill ladder, old MMQ vs wide MMQ, on the real
# Glimmer file.
#
# The A/B switch is the backend's own config, not a rebuild:
# WGPU_COMPUTE_MMQ_WIDE_MIN_ROWS names the chunk row count
# at which the session starts preferring the wide-tile
# double-buffered products (K15/K16) over the standard ones
# (K13/K14). Setting it to 65 (past the 64-row chunk cap)
# takes the wide form out of the ladder entirely: that is
# the "old" position. The default (32) is the "wide"
# position. Decode is untouched in both positions (the
# wide form only exists on the matrix path).
#
# Method, matching bench/gpu-resident-2026-10-30.md's
# session prefill record: repeated-sentence prompts at two
# lengths (about 254 and about 1,024 tokens; the exact
# counts come from the runner's own prompt_tokens= line),
# gross tok/s at the long length, and the marginal rate
# between the two lengths, which strips the fixed cost.
# Each (config, length) runs twice and the faster prefill
# time stands (clock-state noise only ever slows a run).
# The per-tensor half of the proof is the mmq_bench table
# at batch 64, run once at the end (its `mmq wide w<N>`
# rows against the `mmq m<N>` rows).
#
# Run on ubuntu-server with the GPU free:
#   bench/mmq-wide-bench.sh [MODEL]
# MODEL defaults to the same path mmq_bench uses. Logs and
# the summary land in /tmp/bench-results/.
set -u
OUT=/tmp/bench-results
mkdir -p "$OUT"
MODEL="${1:-/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf}"
[ -f "$MODEL" ] || { echo "SKIP: model not found: $MODEL" | tee -a "$OUT/summary.txt"; exit 0; }

export PATH="$HOME/.cargo/bin:$PATH"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2
echo "=== mmq wide bench $(date -Is) ===" | tee -a "$OUT/summary.txt"
echo "model: $MODEL" | tee -a "$OUT/summary.txt"

cargo build --release -p itak-loader --example glimmer_generate 2>&1 | tail -2
cargo build --release -p wgpu-compute --example mmq_bench 2>&1 | tail -2
GEN="$ROOT/target/release/examples/glimmer_generate"
MMQB="$ROOT/target/release/examples/mmq_bench"
[ -x "$GEN" ] || { echo "FAIL: glimmer_generate did not build" | tee -a "$OUT/summary.txt"; exit 1; }

# Repeated-sentence prompts (the gpu-resident note's
# method). "The quick brown fox jumps over the lazy dog."
# tokenizes to about ten tokens; the runner reports the
# true counts and the math below uses those, not these
# targets.
sentence="The quick brown fox jumps over the lazy dog. "
prompt_short=""
for _ in $(seq 1 25); do prompt_short="$prompt_short$sentence"; done
prompt_long=""
for _ in $(seq 1 102); do prompt_long="$prompt_long$sentence"; done

# run_prefill CONFIG MIN_ROWS PROMPT LOG
# Prints "<ms> <tokens>" of the faster of two runs.
run_prefill() {
  local cfg="$1" min_rows="$2" prompt="$3" log="$4"
  local best_ms="" best_tok="" rep ms tok
  for rep in 1 2; do
    WGPU_COMPUTE_MMQ_WIDE_MIN_ROWS="$min_rows" \
      "$GEN" "$MODEL" "$prompt" 4 --gpu >> "$log" 2>&1
    read -r ms tok <<< "$(grep -E '^phase=prefill ' "$log" | tail -1 \
      | sed -E 's/.*ms=([0-9]+) prompt_tokens=([0-9]+).*/\1 \2/')"
    [ -n "${ms:-}" ] || { echo "FAIL $cfg rep $rep: no prefill line in $log" | tee -a "$OUT/summary.txt"; return 1; }
    if [ -z "$best_ms" ] || [ "$ms" -lt "$best_ms" ]; then best_ms="$ms"; best_tok="$tok"; fi
  done
  echo "$best_ms $best_tok"
}

declare -A MS TOK
for cfg in old wide; do
  if [ "$cfg" = old ]; then min_rows=65; else min_rows=32; fi
  for len in short long; do
    if [ "$len" = short ]; then prompt="$prompt_short"; else prompt="$prompt_long"; fi
    log="$OUT/mmq_wide_${cfg}_${len}.log"
    : > "$log"
    read -r MS[$cfg,$len] TOK[$cfg,$len] <<< "$(run_prefill "$cfg" "$min_rows" "$prompt" "$log")" || exit 1
    echo "$cfg $len: ${TOK[$cfg,$len]} tokens in ${MS[$cfg,$len]} ms (log $log)" | tee -a "$OUT/summary.txt"
  done
done

for cfg in old wide; do
  awk -v cfg="$cfg" \
      -v ms_s="${MS[$cfg,short]}" -v tok_s="${TOK[$cfg,short]}" \
      -v ms_l="${MS[$cfg,long]}" -v tok_l="${TOK[$cfg,long]}" '
    BEGIN {
      gross = tok_l / ms_l * 1000
      marg = (tok_l - tok_s) / ((ms_l - ms_s) / 1000)
      printf "%s: gross at %d tokens %.1f tok/s; marginal (%d -> %d tokens) %.1f tok/s\n", \
        cfg, tok_l, gross, tok_s, tok_l, marg
    }' | tee -a "$OUT/summary.txt"
done

echo "--- per-tensor table (mmq_bench, batch 64) ---" | tee -a "$OUT/summary.txt"
"$MMQB" --model "$MODEL" --batch 64 > "$OUT/mmq_wide_pertensor.log" 2>&1
grep -E "^adapter|mmq |gemm |gemv " "$OUT/mmq_wide_pertensor.log" | tee -a "$OUT/summary.txt" || true
echo "MMQ_WIDE_DONE" | tee -a "$OUT/summary.txt"
