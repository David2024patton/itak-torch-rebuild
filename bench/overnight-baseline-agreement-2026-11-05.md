# Overnight baseline agreement run: our CPU stack vs llama.cpp on the real Glimmer file (2026-11-05)

TODO T-9.3 / T-9.4 baseline: a fresh, fully recorded agreement
run between our stack and llama.cpp on the real Glimmer file at
temperature 0, token agreement and probability agreement on a
fixed prompt set, including the open residue from the T-9.3
record (our first-token top-1 probability 0.4035 vs llama.cpp
0.3962 on the golden prompt, first measured 2026-10-29 in
crates/itak-loader/INTEGRATION-RUN-B.md and repeated on the GPU
at 0.403462 in bench/gpu-proof-2026-10-30.md).

## Verdict

**Full token agreement, and the probability residue reproduces
at exactly its recorded size without growing.** All five
prompts: prompt token ids identical between the engines, and
all 16 greedy tokens identical, 80 of 80 positions, no first
divergence anywhere. The golden prompt's first-token top-1
probability is again ours 0.4035 vs llama.cpp 0.3962 (delta
+0.0072), digit for digit the October values. Across the set
the first-token delta is prompt-dependent in sign (+0.0072,
+0.0049, -0.0016, +0.0002, -0.0005), and per-position top-1
deltas over all 80 shared positions stay within +/-0.0291
with mixed signs and no growth with position. That is the
signature of f32 accumulation-order noise between two correct
implementations, not a structural logit-path difference. No
engine code was changed by this task; this is measurement only.

## Setup

- Branch `lane/overnight-baseline` from main
  bda7ff55ea57f4a764afddf9ba2e050c59b86916. The branch adds
  bench harness files only; every engine crate is byte-identical
  to main. Harness as staged and run: commit
  1605ffcfd1a10c65712be36eadbfa41b065d0d1f. (A compare-phase
  fix, aa7c926, landed after the run; see Incidents. It touches
  no engine code and none of the recorded raw outputs.)
- Server: ubuntu-server, kernel 7.0.0-42-generic, 123 GB RAM
  (69 used / 54 available at start). Staging tree
  `~/overnight-baseline`, rsynced from the worktree; staged file
  hashes verified before the run (MANIFEST.txt, 192 files, 0
  mismatches). Run executed under the phase lock
  (`flock -w 14400 /tmp/itak-phase.lock`), after the sibling
  overnight lanes ahead of it in the lock queue finished.
- Model: `/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`
  (16,756,683,904 bytes), coverage 731/731 in our loader.
- Reference: llama.cpp `version: 0.7.0-dev (build 137, commit
  a7af149)`, `~/llama.cpp/build-vulkan2/bin/llama-server`,
  CPU only (`-ngl 0`), `-c 512`, `-t 8`, loopback port 18923,
  started for this run and killed at the end of its phase.
  Production Glimmer stayed down throughout; ports 8080,
  8091, 8092 were never touched.
- Ours: `target/release/examples/glimmer_generate` (the
  cpu-compute reference path of crates/itak-loader), release
  build on the server, `CARGO_BUILD_JOBS=4`,
  `MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=0`.
- Protocol, both sides: temperature 0, 16 generated tokens.
  The llama.cpp side received each prompt as a token-id array
  (llama-tokenize --ids), so both engines consumed
  byte-identical input; `/completion` with `n_predict` 16,
  `n_probs` 5, `ignore_eos` true, seed 42. This build's
  responses carry `logprob` / `top_logprobs`; probabilities
  below are exp(logprob). Our side prints per-step top-5 with
  f64 softmax probabilities over the full vocabulary.

Exact commands (server, under the lock):

```text
flock -w 14400 /tmp/itak-phase.lock bash bench/overnight_baseline_run.sh
# which runs:
cargo build --release -p itak-loader --example glimmer_generate
python3 bench/overnight_baseline.py --repo ~/overnight-baseline \
  --model /mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf \
  --out-dir ~/overnight-baseline/results
# recovery compare (see Incidents):
python3 bench/overnight_baseline_compare.py --out-dir ~/overnight-baseline/results
```

Raw artifacts (per-prompt llama responses, our engine logs,
compare.json, environment capture) live in the staging tree at
`~/overnight-baseline/results/`.

## Prompt set (fixed; all from golden/prompts.jsonl)

| id | prompt (verbatim) | prompt ids |
| --- | --- | ---: |
| factual_capital (the golden prompt) | "The capital of France is" | 6 |
| factual_science | "The chemical symbol for water is" | 7 |
| arithmetic_add | "17 + 25 =" | 6 |
| continuation_classic | "The quick brown fox jumps over the" | 8 |
| code_python | "def fibonacci(n):\n    if n <= 1:\n        return n\n    return" | 18 |

Our tokenizer's prompt ids matched llama-tokenize's on all
five prompts (the comparison records prompt_ids_match = true
for each), so tokenization is not a variable in this run.

## Results

Generated token ids were identical on every prompt; each list
below is both engines' output. First-token top-5 tables are
side by side; the delta row under each prompt is our top-1
probability minus llama.cpp's at each of the 16 positions.

### factual_capital

Generated (both): [13796, 26, 1573, 373, 290, 2167, 26899,
1212, 6996, 310, 11698, 341, 290, 2167, 26899, 1212]
(" Paris. It is the most populous with Paris and the most
populous"). Agreement 16/16, no divergence. Our wall 2,185 s.

First token top-5:

| rank | llama.cpp id | p | ours id | p |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 13796 " Paris" | 0.3962 | 13796 " Paris" | 0.4035 |
| 2 | 262 " a" | 0.0961 | 262 " a" | 0.0946 |
| 3 | 290 " the" | 0.0513 | 290 " the" | 0.0490 |
| 4 | 1085 " one" | 0.0462 | 1085 " one" | 0.0465 |
| 5 | 5711 " known" | 0.0323 | 1437 " also" | 0.0316 |

Top-1 deltas by position: +0.0072 +0.0097 -0.0035 -0.0021
-0.0012 -0.0054 -0.0147 -0.0000 -0.0035 +0.0083 +0.0274 -0.0004
-0.0060 +0.0264 -0.0154 +0.0001 (max abs 0.0274; positions 1 to
8 max abs 0.0147, inside the 0.015 band the October re-run
recorded for those positions).

The fifth-slot swap (ours 1437 " also" 0.0316 vs reference
5711 " known" 0.0323) is the same near-tie swap recorded in
October, at the same values.

### factual_science

Generated (both): [465, 30, 59, 26, 2473, 2406, 290, 220, 30,
310, 465, 30, 59, 3641, 1044, 45] (" H2O. What does the 2 in
H2O represent?\nA"). Agreement 16/16, no divergence. Our wall
2,369 s.

First token top-5: llama.cpp 465 0.7063, 5789 0.0415, 24420
0.0196, 5325 0.0161, 523 0.0132; ours 465 0.7112, 5789 0.0439,
24420 0.0188, 5325 0.0176, 523 0.0125. Identical ids in
identical order.

Top-1 deltas: +0.0049 +0.0099 -0.0003 +0.0044 +0.0069 +0.0199
+0.0017 +0.0291 +0.0002 -0.0095 +0.0029 +0.0004 -0.0005 -0.0059
-0.0002 +0.0107 (max abs 0.0291, the largest single delta of
the run, at position 8).

### arithmetic_add

Generated (both): [220, 3763, 198, 3763, 430, 220, 1087, 319,
220, 963, 198, 3763, 430, 220, 963, 319]. Agreement 16/16, no
divergence. Our wall 2,175 s.

First token top-5: llama.cpp 220 0.3652, 8656 0.2416, 24420
0.0779, 330 0.0713, 126800 0.0362; ours 220 0.3636, 8656
0.2334, 24420 0.0770, 330 0.0743, 126800 0.0379. Identical ids
in identical order. First-token delta is negative here
(-0.0016): the residue is not a fixed positive bias.

Top-1 deltas: -0.0016 +0.0079 -0.0100 -0.0011 +0.0112 +0.0013
+0.0239 +0.0015 -0.0028 +0.0002 +0.0019 -0.0030 -0.0012 +0.0001
+0.0004 +0.0000 (max abs 0.0239).

### continuation_classic

Generated (both): [37463, 10877, 26, 589, 7202, 27756, 92637,
67072, 1278, 290, 37463, 10877, 26, 589, 7202, 27756].
Agreement 16/16, no divergence. Our wall 2,560 s.

First token top-5: llama.cpp 37463 0.9937, 7202 0.0006, 13251
0.0004, 10877 0.0004, 57522 0.0002; ours 37463 0.9939, 7202
0.0005, 13251 0.0004, 10877 0.0004, 57522 0.0002. Identical
ids in identical order; the distribution is near-saturated,
so the first-token delta is +0.0002.

Top-1 deltas: +0.0002 -0.0003 +0.0246 +0.0231 +0.0004 +0.0001
+0.0001 -0.0001 -0.0001 +0.0000 +0.0000 +0.0001 -0.0030 -0.0015
-0.0000 +0.0001 (max abs 0.0246; the two larger entries sit at
positions 3 and 4, where the top-1 probability itself is
small, so the relative noise is larger).

### code_python

Generated (both): [105436, 2736, 430, 220, 29, 21, 584,
105436, 2736, 430, 220, 30, 1333, 1251, 3081, 9437].
Agreement 16/16, no divergence. Our wall 4,654 s.

First token top-5: llama.cpp 105436 0.9945, 330 0.0018, 1756
0.0016, 16483 0.0006, 220 0.0006; ours 105436 0.9940, 330
0.0020, 1756 0.0019, 220 0.0006, 16483 0.0005. Same five ids;
ranks 4 and 5 swap at the 0.0006 level, another near-tie.
First-token delta -0.0005.

Top-1 deltas: -0.0005 +0.0000 -0.0141 -0.0004 +0.0001 +0.0000
-0.0000 +0.0000 +0.0000 -0.0001 +0.0005 +0.0000 -0.0049 +0.0277
-0.0043 -0.0019 (max abs 0.0277).

## The residue, stated plainly (T-9.3)

- On the golden prompt it **reproduces exactly**: 0.4035 vs
  0.3962, the same four-decimal values recorded on 2026-10-29
  (CPU) and 2026-10-30 (GPU, 0.403462 at six decimals). It has
  not grown and not shrunk in a week of engine changes; the
  value is a stable property of the two implementations'
  arithmetic on this prompt.
- It is **prompt-dependent**: first-token deltas across the
  set are +0.0072, +0.0049, -0.0016, +0.0002, -0.0005. The sign
  flips; there is no constant offset a correction could
  subtract.
- It does **not accumulate**: over 16 positions the deltas
  wander within +/-0.0291 with no trend, and greedy token
  agreement is perfect everywhere, including the two prompts
  whose distributions stay near-saturated for long stretches.
- Reading: this is the expected distance between our f32
  reference math (f64 softmax over finished logits) and
  llama.cpp's kernel accumulation order. T-9.3's proof bar
  (logit comparison against the reference on a fixed prompt
  matches within tolerance at temperature 0) is met on the
  golden prompt inside the historically recorded 0.015 band
  for the first 8 positions; the honest wider statement is
  +/-0.03 per-position top-1 probability noise across this
  set, with exact token agreement.

## Incidents and recoveries

- First launch (staged bfcd09f) failed before any measurement:
  the non-interactive server shell has no cargo on PATH
  ("cargo: command not found"). Fixed in the runner (1605ffc)
  by exporting ~/.cargo/bin; re-staged, hashes re-verified,
  relaunched. No data was produced by the failed attempt.
- The run then waited in the phase-lock queue behind the
  sibling overnight-golden and overnight-logitscale lanes, per
  protocol; environment stamp 2026-11-05T04:40:57Z is when this
  lane's script took the lock.
- The staged driver's compare phase crashed at the very end
  with KeyError 'prob': this llama.cpp build's /completion
  returns logprob-shaped probabilities, which the llama phase
  tolerated (it reads only ids) but the compare code did not.
  All per-prompt raw JSONs and engine logs were already saved,
  so nothing was lost. The fix (shape-tolerant prob reading)
  and a standalone recompute tool landed as aa7c926, and
  compare.json was recomputed from the saved outputs with the
  recorded results above. The engine runs themselves were not
  repeated and did not need to be.
- Total our-side wall time for the five prompts: 13,943 s
  (about 3.9 h). The CPU reference path remains minutes-class
  per token by design; this run measures agreement, not speed.

## Final server state

- The throwaway llama-server was terminated at the end of its
  phase; port 18923 confirmed down afterwards, and no
  Glimmer llama-server process remains.
- Production Glimmer was never started. The Qwen3-Embedding
  tenants on 8091/8092 were never touched. No GPU work of any
  kind ran in this task.
- Nothing was pushed or merged; this note and the harness live
  on lane/overnight-baseline only.
