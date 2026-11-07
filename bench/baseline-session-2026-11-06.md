# Baseline session, 2026-11-06 (the consolidated card session)

Date: 2026-11-06 evening into 2026-11-07 (server clock UTC,
session start 2026-11-07T02:50Z). Lead: baseline-session lane.

This session runs the work every speed-wave-2 lane deferred to
"the cleared card session": the fresh three-way baseline
(llama.cpp Vulkan, llama.cpp HIP, this engine) for speed AND
math agreement, then the queued verdicts for band routing,
draft routing, gated/SwiGLU routing, spec-round economics,
K23, staged flash, and the diag-capture checklist. Every leg
measures and records; no default is flipped by this session.
Decision rules are the ones written in each unit's own note.

## Session shape

- Card: AMD Radeon RX 7900 XTX (Navi 31, gfx1100), RADV /
  Mesa 26.0.8, ubuntu-server. Tenants stopped for the session
  under David's standing GPU rule and restored at the end:
  `docker stop cortex-cortex-chat-1 cortex-cortex-embed-1`,
  `systemctl --user stop cortex-embed.service`. VRAM on the
  clear card read exactly 33,271,808 B (driver residue) before
  every measurement leg; the guard reading is recorded per leg
  in the run logs. A transient 16,261,627,904 B reading before
  the first leg-1(c) attempt was the HIP llama-bench process
  still releasing after its leg; every later guard read the
  residue value.
- Locks: every server run under
  `flock -w 14400 /tmp/itak-phase.lock`. One heavy job at a
  time; `CARGO_BUILD_JOBS=4`, `MALLOC_ARENA_MAX=2`,
  `MALLOC_TRIM_THRESHOLD_=0`.
- Staging: `~/gpu-wave14` from lane head d456f48 (session
  driver), re-staged at 26fe63d when the adapter_features
  probe was added; `MANIFEST.txt` = 395 sha256 lines,
  `sha256sum -c` reported 0 mismatches both times. The
  control tree for leg (g) is main a064b5d7 staged separately
  at `~/gpu-wave12` (368 files; predates
  bench/failure-bundle.sh, so its arms run unwrapped and say
  so). Production llama-server (localhost:8080) untouched;
  production Glimmer serving stays down per standing order.
- Model: /mnt/models/Muse-Glimmer-30B/
  Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf, sha256
  4cc57c0f... (recorded value, matches every prior session);
  DFlash file dflash-Muse-Glimmer-30B-Q4_K_M.gguf.
- Every verdict run is wrapped in bench/failure-bundle.sh;
  the wrapper's INDEX.md on the server records each run. One
  exception class: the leg-1(c) driver runs and the CPU
  collection run unwrapped (driver arms bundle through their
  slugs only where the wrapper could carry them; see below).
- Disclosed contention: the leg-1(c) CPU-stack collection
  (nice 10, no GPU) ran concurrently with the GPU legs. It
  needs no card; same-session A/B arms share the condition.
  Recorded here so timing absolutes can be read with it.

## Leg 1, baseline

### 1(a) llama-bench, real file, -ngl 999, -p 512 -n 128 -r 3

Bundles leg1-llamabench-vulkan / leg1-llamabench-hip, both
PASS. Verdict against the recorded 2026-10-28 references:

| build | pp512 (tok/s) | tg128 (tok/s) | recorded | verdict |
|---|---|---|---|---|
| Vulkan (build-vulkan2) | 995.99 ± 2.39 | 44.59 ± 0.02 | 998.43 / 44.31 | CONFIRMED |
| HIP (build-rocm2) | 1150.89 ± 11.08 | 41.18 ± 0.02 | 1151.71 / 41.13 | CONFIRMED |

llama.cpp tree /home/server/llama.cpp at
a7af149a2a7768d4b1f6f02c6ce26a17fba11376, build 137 (both
binaries report it); the HIP build exists and runs, so no
HIP caveat is needed this session.

### 1(b) our engine, prefill and plain decode, current main

- Prefill, prefill_profile harness, 1,022 tokens, diag off:
  phase=prefill ms=6749, prompt_tokens=1022 (151.4 tok/s),
  bundle leg1b-prefill-profile PASS.
- Prefill, shipped path (glimmer_generate --gpu, same
  prompt): phase=prefill ms=6419 (159.3 tok/s),
  session_stats forwards=4 chunks=19 dispatches=11783
  readbacks=4, bundle leg1b-gen-prefill PASS. Against the
  recorded 6,330 ms the shipped path is within 1.4 percent;
  the profile harness reads slower than the shipped path in
  this session, the reverse of the host-gap session's
  relationship, and both numbers are recorded as measured.
- Plain decode, glimmer_generate --gpu, golden prompt, 128
  tokens: 127 steps, mean 34.17 ms per step (29.3 tok/s),
  median 35 ms, min 26, max 36, bundle leg1-plain-decode
  PASS. This reads BELOW the recorded 35.70 tok/s control
  (2026-11-04). The recorded control came from the resident
  session harness; glimmer_generate prints the top-5 table
  per step and pays print and host cost the resident harness
  does not, so the two numbers are different surfaces. The
  resident-surface plain rate is re-read in leg (d) arms
  (spec_generate_gpu both mode prints it); the flash leg's
  step 4 re-measures decode with its switch on. The note
  keeps all three numbers with their harnesses named rather
  than merging them.

### 1(c) cross-backend math agreement

Harness: bench/baseline-session-driver.py (lane commit
d456f48), mirroring bench/overnight-baseline-agreement-2026-
11-05.md alignment exactly: llama-server per stack
(--n-predict 16 --temperature 0 --seed 42 --n-probs 5
--ignore-eos, generated ids from completion_probabilities),
glimmer_generate for ours (top-5 table; position alignment
prefill = position 1 against llama step pos-1; top-1
probability = exp(logprob) of the greedy token). Five golden
prompts, 16 tokens.

GPU-side result (three stacks; CPU stack lands below):

- Token ids: IDENTICAL across llama-Vulkan, llama-HIP, and
  ours-GPU on all 5 prompts, all 16 positions, and identical
  to the 2026-11-05 overnight lists. Golden first-8 check
  [13796, 26, 1573, 373, 290, 2167, 26899, 1212] reproduced
  by all three stacks on factual_capital.
- Max absolute top-1 probability deltas per pair (worst
  prompt): llama-HIP vs llama-Vulkan 0.0231; llama-HIP vs
  ours-GPU 0.0161; llama-Vulkan vs ours-GPU 0.0153. Every
  pair sits inside the recorded +/-0.03 f32
  accumulation-order noise band, and ours-vs-llama deltas are
  smaller than llama's own cross-backend delta. Full matrix:
  agree14/compare-matrix.json on the server, copied to the
  lane with this note's artifacts.
- Verdict against the recorded expectation (tokens identical
  everywhere, deltas near +/-0.03): CONFIRMED for the three
  GPU stacks. No divergence, so no failure bundle was owed
  for divergence.

CPU stack: glimmer_generate without --gpu, same 5 prompts,
running as a background collection on the server (per-prompt
walls in the 2026-11-05 overnight run were 2,175 to 4,654
seconds). Its section completes when the collection lands;
the compare tool re-runs to extend the matrix to four
stacks.

First-attempt failures (invocation errors, all rerun clean
as leg1b): bundle 20261107T030149Z-leg1-agree-llama-vulkan
and 20261107T030149Z-leg1-agree-llama-hip (driver --bin-dir
missing the trailing /bin, llama-server not found), bundle
20261107T030318Z-leg1-prefill-profile (prefill_profile
called without its required prompt argument). No engine
fault in any of the three.
