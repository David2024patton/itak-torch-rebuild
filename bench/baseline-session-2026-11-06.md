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

## Leg 2, queued verdicts

A note on evidence hygiene for this leg: two completion
reports delivered to the session mid-run did not match the
server's live state (they described runs and clock times the
processes on the box contradicted). Every number below was
re-derived from per-arm log files whose existence and
timestamps were verified in the same command that read them,
and process completion was verified by process state, not by
delivered text. Where a number could not be re-derived that
way, it is not used.

A note on bundle verdicts for spec runs: every
spec_generate_gpu bundle in this session records FAIL in
bench/failures/INDEX.md, because the harness prints
"SPEED: FAIL (effective ... < bar 88.6)" and "T-9.6:
CORRECTNESS PASS, SPEED FAIL" on every run and the wrapper
trips on the FAIL marker. The 88.6 tok/s bar is the standing
target line, not a leg verdict; every one of those logs
carries IDENTITY: PASS and GOLDEN: PASS. The FAIL marks are
recorded here once so the index is not misread later.

### (d) Band routing A/B (bench/mmv-band-routing-2026-11-06.md)

Arms: current main, spec_generate_gpu, 64 tokens, both mode,
diag timing, control (MMV unset) and caps 4/8/12/16, depths
1 and 6, no per-cell overrides. IDENTITY and GOLDEN PASS in
all 10 arms; round counts and acceptance identical across
arms at each depth (38 rounds depth 1, 32 depth 6); the
spec stream is byte-identical in every arm.

Effective tok/s (the harness's "spec rates: effective"
line):

| depth | control | cap 4 | cap 8 | cap 12 | cap 16 |
|---|---|---|---|---|---|
| 1 | 17.39 | 19.46 | 19.26 | 19.34 | 21.73 |
| 6 | 10.14 | 10.15 | 15.06 | 15.25 | 16.33 |

Verify medians (ms, per-round spec.verify.end): depth 1,
control 51, every routed cap 41. Depth 6, control 153,
cap 4 153 (equal), caps 8/12/16 82/82/83.

Decision rule, item by item:
- Every routed cap at or above same-session control: PASS
  at both depths. The flat-routing cliff does not recur:
  caps 4/8/12 at depth 1, which flat routing put at
  14.71/15.23/15.20 against a 16.98 control, all beat this
  session's control. Cap 4 at depth 6 is a wash (10.15 vs
  10.14); at that cap almost nothing routes in a depth-6
  verify (rows run to 7, the cap admits only the KV cells
  at rows 2 to 4), which the tally confirms.
- Cap 16 at or above the recorded flat-routing numbers:
  PASS (21.73 vs recorded 19.26 at depth 1; 16.33 vs
  recorded 10.27 at depth 6).
- Verify medians at or under control: PASS.
- Resident dispatch counts identical across arms: NOT
  literally true, and the exception is the gated unit's own
  predicted behavior: where the SwiGLU/gated decomposition
  fires, it adds dispatches (see (f): +26 per depth-1
  verify forward, +104 per depth-6 rows-7 forward, +104
  once in the plain phase's prefill-chunk forward).
  Acceptance, round counts, streams, and every unrouted
  kernel tally are identical; the deltas are fully
  accounted by the decomposition counts in (f).
- Sweep re-run (bench/mmv-sweep.sh, results in
  bench/results/mmv-sweep-20261107-035843/ on the server,
  CSVs copied to the lane): 0 disagreeing cells in all four
  arms, and the mmv arm's kernel column matches the shipped
  band table in 99 of 99 cells, including the GEMM
  threshold at rows 16 (k13/k14 MMQ for Q4_K/Q6_K, k21 for
  Q5_K whose band includes 16) and GEMV in every unrouted
  cell.

Verdict: band routing PASSES its card protocol. The shipped
cells and the global-gate default-off posture stand as the
lane left them; turning the global gate on by default is a
separate decision for the parent, and these numbers say
cap 16 with the shipped table beats control at both depths
on this session's card.

### (e) Draft routing A/B (bench/draft-mmv-routing-2026-11-06.md)

Step 1: the dflash_session device suite EXECUTED on the
card (7 passed, 0 failed; no skip line in the log; the
adapter is the session's 7900 XTX) including
dflash_session_capped_forward_returns_the_full_prefix.
Bundle leg2-dflash-tests PASS.

Arms: 3 reps, medians; prompt "The capital of France is",
both mode, diag timing. All reps IDENTITY and GOLDEN PASS;
streams byte-identical to the control tree's (sha256 prefix
fedbbec6d508 for every stream file in (e) and (g)).

| arm | eff tok/s | draft_ms med | verify_ms med |
|---|---|---|---|
| d1 unset | 17.44 | 42 | 50 |
| d1 cap 16 | 22.10 | 32 | 41 |
| d6 unset | 10.19 | 44 | 152 |
| d6 cap 16 | 16.47 | 34 | 82 |
| d1 cap 16 + Q5K_FFN_ROWS=1 (attribution) | 19.29 | 42 | 41 |

- Exposure at cap 16: k21_q5k_mmv at batch 16 appears
  exactly once per round (38 of 38 rounds at depth 1, 32
  of 32 at depth 6), and no k19/k20 record exists at
  batch 16 in any arm; the unset arms show zero k21.
  PASS, exactly as the protocol's fingerprint reads.
- The protocol also asks for the draft session's
  route-decision lines in the log ("cap 16 excludes batch
  16" style). This build prints no such lines at the
  default log level: the band table resolves silently in
  these runs (the construction-log table line exists in
  the code but did not appear in the spec runs' output).
  Recorded as a wording gap in the evidence, with the
  behavior itself proven by the kernel stamps above and
  the attribution arm below.
- Attribution: closing the Q5_K ffn cell returns draft_ms
  to 42, the unset value exactly, while verify_ms stays
  routed at 41. The head route is the cause of the draft
  gain. PASS.
- draft_ms falls 10 ms at both depths (42 to 32, 44 to
  34), inside the note's predicted 8 to 12 ms band. PASS.

Verdict: draft routing PASSES its card protocol.

### (f) Gated/SwiGLU A/B (bench/gated-swiglu-routing-2026-11-06.md)

Same arms as (d) (control vs cap 16, depths 1 and 6);
fingerprints counted per verify forward by segmenting the
diag dispatch stream on the round markers.

- Depth 6, rows-7 verify forwards (29 of 32 rounds):
  exactly 52 gate_strided + 52 swiglu_strided per forward,
  +104 total, matching the written prediction exactly.
  The remaining rounds (rows 3 and 5) show swiglu-only
  decomposition with zero gate_strided, as the gated
  site's rows 6-to-7 admission rule requires.
- Depth 1 (rows 2 every round): gate_strided exactly 0
  (PASS, the gated site admits only rows 6 to 7), but
  swiglu_strided counts 26.0 per forward against the
  predicted +52, and the net dispatch delta per verify
  forward is +26 (472 vs 446), not +52. The same
  half-count appears in the depth-6 rows-3 and rows-5
  forwards (26.0 swiglu_strided each). PREDICTION MISS,
  recorded plainly: at rows below 6 the decomposition
  engages half the predicted per-forward count; at rows
  6 to 7 it matches exactly. Correctness is unaffected
  (identity, golden, acceptance, and streams all pass and
  match), and the direction of the timing effect is the
  predicted one; the note's per-forward arithmetic for
  small rows is what is off.
- Verify medians at depth 6 move 153 to 82 in these arms.
  That is the combined band+gated effect (the arms share
  one cap, so the gated share is not separable here); it
  exceeds the note's roughly 8 to 9 ms gated-only estimate
  because band routing is firing in the same forwards.
- No-leak: the control arms contain zero gate_strided,
  zero swiglu_strided, and zero MMV records in their
  verify segments. PASS. The protocol's repeat-control
  step is satisfied by the session's two independent
  unset depth-1 runs on this tree ((d)'s band-d1-ctrl
  and (e)'s e-d1-unset): effective 17.39 against 17.44,
  verify medians 51 against 50, identical streams, zero
  routed records in both. The control reproduces.
- The decomposition also fires once in the plain phase,
  in the prefill-chunk forward at rows 6 (+104 dispatches,
  28,480 to 28,584), which is the same mechanism at the
  same admission rows, not a leak into single-row decode:
  single-row forwards show no strided records.

Verdict: gated/SwiGLU routing PASSES on correctness,
exposure shape, and direction, with the small-rows
fingerprint count at half the written prediction. The
parent may want the gated note's prediction corrected;
the code's admission behavior (gate only at rows 6 to 7)
matches its rule everywhere measured.

### (g) Spec-round economics A/B (bench/spec-round-economics-2026-11-06.md)

Control tree: main a064b5d7 staged at ~/gpu-wave12 (it
predates bench/failure-bundle.sh, so its arms ran
unwrapped, outputs tee'd to the same log directory).
Lane arm: current main. Both MMV cap 16, 3 reps, medians,
same invocation as (e).

| arm | eff tok/s | draft_ms med | verify_ms med | capped lines/round |
|---|---|---|---|---|
| d1 control tree | 17.95 | 47 | 43 | 0 |
| d1 lane tree | 22.10 | 32 | 41 | 1 (returned=2) |
| d6 control tree | 11.84 | 48 | 118 | 0 |
| d6 lane tree | 16.47 | 34 | 82 | 1 (returned=7) |

- The capped engagement line fires exactly once per round
  in every lane-tree rep and never in the control tree.
  PASS.
- Sanity pair, MMV unset, depth 1: control tree 16.89,
  lane tree 17.44 effective, streams identical (same
  sha256 as every other arm). Within noise. PASS.
- Attribution caveat the protocol's framing needs: the
  control tree predates ALL of speed wave 2, so the
  tree-to-tree deltas mix the capped readback with the
  draft head route (priced at 10 ms by (e)'s same-tree
  arms), the laned argmax, and band+gated verify routing.
  The draft medians fall 15 ms (depth 1) and 14 ms
  (depth 6) tree to tree, more than the note's 1.5 to
  2.5 ms readback-only estimate; that estimate is neither
  confirmed nor refuted by these arms, because the lane
  tree's unset arm (draft 42) already contains the
  readback and argmax work, and against the control
  tree's unset arm (46) the readback-plus-argmax share
  reads roughly 4 ms at depth 1 on this session's
  timings. The verify medians also move (118 to 82 at
  depth 6), where the protocol expected noise; that is
  the routing wave in the lane tree, not an economics
  effect, and it is recorded here so the protocol's
  step-3 expectation is not misread as a failure of the
  capped readback itself.

Verdict: the capped readback and laned argmax are live,
engaged exactly as designed, and token-identical; the
combined tree-to-tree gain is larger than the unit's
own estimate because the lane tree carries the whole
routing wave with it.

### (h) K23 MMQ repack gates (bench/k23-mmq-repack-2026-11-06.md)

Prompt caveat first: this leg's script built its fox
prompt without the sentence's period, and it tokenized
to 919 tokens, not the recorded 1,022 (the flash leg
uses the corrected construction, verified at 1,022).
Both K23 arms share the 919-token prompt, so every A/B
below is internally valid; absolutes are 919-token
numbers and are not compared against the recorded
1,022-token figures.

- Step 1 correctness: k23_mmq tests PASS on the card
  (bundle leg2-k23-tests); the resident_session suite
  passes with WGPU_COMPUTE_K23_MMQ=1 (bundle
  leg2-k23-resident PASS); the spec run passes IDENTITY
  and GOLDEN with the switch on. No GEMM-shaped call
  exists in a 64-token spec run, so no k23 stamp appears
  there; the stamp proof is the prefill arm below, where
  k23_q4k_mmq dispatches 2,730 times, exactly k13's
  dispatch count in the control arm. The K12 quantize
  counts are identical across arms (same totals, 9,795
  dispatches each).
- Step 2 kernel A/B (prefill_profile, diag timing, 919
  tokens): k23_q4k_mmq totals 7,942.3 ms over 2,730
  dispatches against k13_q4k_mmq's 3,694.3 ms over the
  same 2,730 dispatches in the control arm. K14, flash,
  and every other row are unchanged within noise. The
  prefill phase wall reads 10,209 ms (K23 on) against
  5,923 ms (off).
- Step 3 session wall (glimmer_generate --gpu, same
  prompt, timing off, 3 runs each): off 5,717 / 5,723 /
  5,741 (median 5,723); on 9,949 / 9,956 / 9,956
  (median 9,956). K23 costs roughly 74 percent on the
  prefill wall at this shape.
- Step 0 autopsy and step 4 no-leak: the autopsy reads
  are in the filtered single-kernel dump section below
  (the full-suite dump interleaves under parallel test
  threads); no-leak is covered by every env-unset arm in
  this session reproducing control behavior with zero
  k23 stamps (band controls, prefill off arm).

Verdict against the note's decision rule (flip only on
an autopsy improvement or a material prefill win with
correctness green): K23 is correct and decisively
SLOWER, 2.15x on its own kernel row at identical
dispatch counts. DO NOT FLIP; the switch stays off. The
llama-style staging bet loses on this card and driver,
and the loss is in the kernel itself (its per-dispatch
rates read 8.7 to 16.8 GB/s in the diag stream against
K13's roughly 46 GB/s class), not in host overhead.

### (h) step 0 autopsy, completed (filtered single-family dumps)

The full-suite dump interleaves under parallel test
threads, so the autopsy was re-read from filtered
single-thread runs (the bench/shader-autopsy.sh family
isolation pattern), cross-checked against the in-situ
dump of a live session, where the same blocks reappear
with identical numbers.

| kernel | VGPRs | spilled VGPRs | scratch | LDS |
|---|---|---|---|---|
| K13 (k12_k14 family dump, block 6) | 256 | 0 | 0 | 41,984 |
| K23 (isolated k23 dump, block 3) | 256 | 248 | 25,088 | 22,528 |

K23's staging design did what it promised on LDS
(22,528 against K13's 41,984, and exactly the note's
predicted 22,528 staged bytes), and the register
allocator paid for it: VGPRs still at the 256 cap, 248
VGPR spills and 25,088 bytes of scratch per workgroup
where K13 has none. The spill traffic is the measured
2.15x kernel loss's mechanism. Autopsy improvement on
the LDS axis alone does not change the (h) verdict.

### (i) Flash staged gates (bench/flash-rework-2026-11-06.md)

- Step 0 autopsy: the compile tests do not construct the
  staged pipeline, so the read comes from an in-situ dump
  (spec_generate_gpu with the switch on, private shader
  cache). The staged variants are the five blocks present
  in the session dump and absent from the K4-to-K8
  compile-test dump: VGPRs 60 (rule: at most 128, PASS),
  spilled VGPRs 0, scratch 0 (PASS), LDS 16,384 per
  workgroup as the driver reports it. The WGSL staging
  constant is 21,632; bench/shader-autopsy.sh's own
  header warns that in-situ LDS values differ from the
  WGSL constants by driver packing, and the 16,384 figure
  is exactly one 32x128 f32 staging block. Recorded with
  that caveat.
- Step 1 correctness: the k6_flash_staged exactness test
  passes on the card (bundle leg2-flash-exact); the
  resident_session suite passes with
  WGPU_COMPUTE_FLASH_STAGED=1 (bundle leg2-flash-resident
  PASS); spec IDENTITY and GOLDEN pass at depths 1 and 6
  with the switch on (their bundles carry the standing
  SPEED-bar FAIL artifact described at the top of this
  leg). PASS.
- Step 2 prefill A/B at 1,022 tokens (diag timing):
  k6_attn_flash totals 972.9 ms over 832 dispatches
  (control) against k6_attn_flash_staged 605.9 ms over
  832 (on): the kernel row improves 37.7 percent and
  clears the decision bar (at most 760 ms, 25 percent
  under the recorded 1,013.1) with room. Every other
  kernel row is unchanged within noise (K13 3,970.1 vs
  3,972.9; K14 1,248.0 vs 1,250.2). BUT the prefill phase
  wall moves the wrong way: 6,383 ms off against 6,558 ms
  on. Roughly 540 ms of non-kernel time appears with the
  switch on at this prompt length, against a 367 ms
  kernel saving. The decision rule requires both the row
  bar and a same-session wall improvement: at 1,022
  tokens the wall clause FAILS.
- Step 3 long context (fox prompt x820, prompt_tokens
  8,202, at or above the 8,192 the protocol demands):
  flash row 47,360.9 ms (off) against 22,820.3 ms (on),
  a 51.8 percent kernel improvement; phase wall 93,509
  ms against 66,704 ms, a 28.7 percent wall improvement.
  At long context the staged kernel wins row and wall
  together, decisively.
- Step 4 no-leak: the control arm's CSV contains zero
  k6_attn_flash_staged rows, and single-row decode never
  routes staged (no staged record appears in any
  single-row context in the session). Decode with the
  switch on (glimmer_generate --gpu, 128 tokens) reads a
  mean step of 26.04 ms (38.4 tok/s) against this
  session's plain control reading of 34.21 ms (29.2
  tok/s) from leg 1(b) and 30.01 ms (33.3 tok/s) from the
  (j) tax arm: the glimmer step clock scatters widely
  under this session's parallel CPU collection, so the
  recorded 35.5 to 35.7 band is neither reproduced nor
  refuted on that surface. On the stable surface (the
  resident spec harness's plain decode line) the session
  reads 34.08 to 34.62 tok/s across arms. The staged
  path provably does not touch single-row forwards, so
  no decode effect is attributable to it either way.

Verdict: staged flash is CORRECT everywhere and much
faster where attention dominates (long context), but at
the standard 1,022-token prefill shape the wall clause
of its own decision rule fails despite the kernel win.
DO NOT FLIP on this evidence; the flip question the
parent has to weigh is whether the long-context win
justifies a length-gated default, and where the roughly
540 ms of non-kernel time at 1,022 tokens comes from.
That host-side cost is the one open thread this leg
leaves, and it is named here rather than smoothed over.

### (j) Diag capture checklist (bench/diag-capture-2026-11-06.md)

- Capture run, glimmer_generate: WGPU_COMPUTE_DIAG=
  capture produced cap-a.csv (timing aggregate),
  cap-a.csv.capture.csv (16 forward records), and
  cap-a.csv.manifest.json. The manifest names the
  adapter exactly ("AMD Radeon RX 7900 XTX (RADV
  NAVI31)", driver radv, Mesa 26.0.8-1ubuntu0.3),
  prints the resolved band table verbatim
  (q4k[attn 6-12 ffn 2,4-12 kv 2-3] q5k[...] q6k[...]),
  and the session geometry (52 layers, hidden 6656,
  vocab 202048). Two honest gaps against the note's
  schema claims: the manifest's build.git reads
  "unknown" (the staging tree is a git archive extract
  with no .git, so the build embedded no SHA), and its
  model field is null in both the resident and the
  dflash manifests this session produced.
- Capture run, one spec phase: produced the same three
  sidecars. The surviving capture file is the DFlash
  session's (32 block records, one per round): capture
  files overwrite per session in one process, exactly
  like the timing CSVs, so the target session's records
  (including the spec_round kind) do not survive to
  disk in spec_generate_gpu runs. The checklist item is
  satisfied for the draft session; the spec_round
  record's on-disk survival is a real limitation this
  session found in the capture wiring, recorded for the
  parent.
- Same-build diff: diag_diff on runs A and B (same
  build, same config) prints "manifests: identical"
  and "capture records: identical (16 compared)", exit
  0, bundle leg2-diagdiff-same PASS. Zero divergence,
  as required.
- Cross-config diff: run C at MMV cap 16 against run A.
  diag_diff reports the manifest difference as exactly
  one field (resolved_config.mmv_max_rows: 1 != 16) and
  the first record divergence at seq 0, field hash:
  the prefill forward, the first place routing engages.
  The divergence is localized to the intended change.
  (Its bundle records FAIL because divergence is a
  failure-marker outcome by design; the divergence is
  the expected result.)
- Capture tax: decode 128 tokens, timing-only against
  capture, same session: mean step 30.01 ms against
  30.16 ms, about +0.5 percent. The capture path's cost
  on this surface is negligible; the note's isolated
  1.60 ms per vocab forward does not show up as a
  visible decode tax here.
- PIPELINE_STATISTICS_QUERY readout: the lane's
  adapter_features probe (commit 26fe63d) reports the
  feature OFFERED on this adapter (timestamp_query and
  pipeline_statistics_query both true), and a probe
  device opened with the feature created a
  pipeline-statistics query set successfully (all three
  enumerated adapters answered the same; adapter[0] is
  the 7900 XTX). Per the checklist's terms: a session
  could be built that reads per-pipeline counters
  without disturbing the verdict path; building it is
  future work, not this session's.
- RGP trace: ABSENT this session. bench/gpu-trace-
  2026-11-06.md makes the first traced run conditional
  on a shellcheck pass over bench/gpu-trace.sh and
  bench/failure-bundle.sh on the server, and shellcheck
  is not installed on ubuntu-server. Recorded as the
  checklist itself instructs (file name or absence):
  absence, with the reason.

## Tenant restoration

All GPU legs finished at about 04:20Z. The tenants were
then restored: docker start cortex-cortex-chat-1 and
cortex-cortex-embed-1 (both Up, healthy), systemctl
--user start cortex-embed.service. VRAM settled at
8,932,196,352 B (about 8.9 GB), the normal tenant
footprint, against the clear-card residue of 33,271,808
B during measurement. Production llama-server was never
touched. The leg-1(c) CPU collection needs no card and
was still running at restoration time; its remaining
sections complete below.
