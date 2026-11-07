# Wave20 consolidated card verdict session, 2026-11-07

One session on ubuntu-server (RX 7900 XTX, RADV NAVI31, Mesa
26.0.8-1ubuntu0.3, kernel 7.0.0-42), staged from itak-torch-rebuild
main 515ce23ab22bf4d281d35151305aabf3c60cc992 at ~/gpu-wave20.
Discharges the card gates owed by the build-all wave: split-K MMQ,
coopmat v2 (K24), f16 KV, prefix-KV session wiring, n-gram GPU
runner, the hygiene manifest field, and the calibrator
re-derivation. No default was flipped in this session; every leg
reports against its note's decision rule only.

## Setup

- Staged via git archive of 515ce23. Staged hashes verified equal
  to the commit: session.rs
  54c7c346731a437e94f296722bd98b540b8a45553b113a0e4f53445dd91d97de,
  profile.rs
  db4bc66ce082100b7c5c8df1a0076e70a9258d21d64fe5cb5c83185f0edae8e7,
  ngram.rs
  c7ea494b42acccd9d5a7ebf394c48dc19125208051c32a2c5fd61ba41ea09a94,
  spec_generate_gpu.rs
  3195005926cba17c20a2a80731660dc83480a15e42ddca260d0f561fd3446bb5.
- Model sha256 re-verified:
  4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e.
  Draft file present at the recorded path. Port 8080: nothing
  listening all session (production Glimmer serving is down per
  David's standing order; never touched).
- Tenants stopped for the session (cortex-embed.service,
  cortex-cortex-chat-1, cortex-cortex-embed-1); VRAM drained to
  33,271,808 B driver residue. Restored at closeout (below).
- All runs serialized through flock -w 14400 /tmp/itak-phase.lock
  and wrapped in bench/failure-bundle.sh via logs20/runner.sh,
  except the leg 2 numeric test binary, which was run directly
  (noted as an exception in leg 2). A run whose only FAIL marker
  is the standing 88.6 tok/s SPEED line with IDENTITY and GOLDEN
  passing is a PASS for its leg; stated per leg below.
- STAGING-ONLY, never to be committed: the throwaway example
  crates/wgpu-compute/examples/prefix_realfile_probe.rs (leg 4b)
  and the wgpu-compute dev-dependency additions (itak-loader,
  tokenizer path deps) that let it build in the staging tree.
- Harness lesson re-learned: prefill_profile takes prompt TEXT as
  argv, not a file path. The first leg 2 timing attempt profiled a
  7-token prompt (the path string) and was discarded; all reported
  numbers come from the fixed runs through logs20/pp.sh / pp82.sh.
  The first four leg 1c A/B runs under a broken runner version
  were likewise discarded and re-run.

## Leg 1, split-K MMQ: verdicts per the note's flip criterion

Criterion (bench/splitk-mmq-2026-11-07.md): the default moves only
if the interleaved prefill A/B shows a real win at both lengths
with no regression on the unsplit shapes, identity+golden hold at
auto and =8, and the autopsy shows no new spills.

- (b) tests/splitk_mmq.rs EXECUTED on the card: 3 passed. Drift is
  reassociation-class, as predicted (set scales about 1e4-1.6e5,
  relative at most about 8.5e-7). K14 KV shape 6656 to 256 batch
  4, split vs host reference: 8.594e-2 (splits 2), 8.594e-2 (4),
  9.375e-2 (8); split vs unsplit GPU: 1.016e-1 / 7.031e-2 /
  8.594e-2. K13 same shape: 1.953e-2 vs ref and unsplit (2),
  1.953e-2 / 2.734e-2 (8). Ragged width 1280 factor 4: 3.906e-3.
  Width 768 (empty slices at factor 8): K13 2.930e-3, K14
  1.172e-2.
- (c) Prefill A/B, interleaved pairs, DIAG=timing. Walls (ms):
  1,022 tokens: off 6,446 / 6,408; auto 6,156 / 6,238 (about
  -3.5%); fixed 4: 5,175 / 5,324 (about -18%); fixed 8: 10,180 /
  9,071 (about +50%, WORSE). 8,202 tokens: off 68,476 / 68,571;
  auto 66,492 / 67,648 (about -2%); fixed 4: 59,047 / 58,877
  (about -14%); fixed 8: 128,771 / 129,251 (about +88%, WORSE).
  Kernel lines at 1,022 (logs20/ab1/): fixed 4 splits everything:
  k13_q4k_mmq_split 2,912 dispatches at 3,299.9 / 3,323.9 ms
  (59.1 / 58.7 GB/s vs plain 47.9), k14_q6k_mmq_split 832 at
  724.6 / 730.9 ms (64.0 / 63.4 vs plain 35.6), splitk_reduce
  3,744 at 50.2 ms. At factor 8 the reduction eats the win:
  splitk_reduce 3,744 at 1,563.2 / 1,347.2 ms at 1,022 tokens, and
  at 8,202 the K13 split form itself collapses (56,702.5 /
  56,917.9 ms, 27.5 GB/s) with splitk_reduce 29,952 at 27,538.9 /
  27,655.5 ms (22.8% of GPU time); total GPU 120,924.3 /
  121,259.0 vs off 66,159.5 / 66,304.4. Under auto only the KV
  projections split (the auto rule's factor-8 pick): K14 plain
  416 (6656-wide downs) plus k14_q6k_mmq_split 416 at 30.9 / 31.0
  ms plus splitk_reduce 416 at 1.34 / 1.36 ms at 1,022 tokens;
  total GPU 5,863.3 / 5,913.0 vs off 6,094.2 / 6,080.5. A direct
  unwrapped confirmation run at =4 reproduced the split lines
  (K13 split 3,289.6 ms, wall 5,106 ms), ruling out a wrapper
  artifact.
- (d) Identity + golden: =auto IDENTITY PASS + GOLDEN PASS;
  =8 IDENTITY PASS + GOLDEN PASS (64-token both-mode runs).
  Golden prompts factual_science and factual_history at 16
  tokens: IDENTITY PASS at =auto and =8 for both. A spec-mode
  capture run at =auto also passed with 38 block records; its
  timing-mode summary carries ZERO split stamps, explained by
  GEMM_MIN_ROWS=16: spec verify forwards are 7 rows or fewer and
  never reach the MMQ rung. Split engagement evidence is the
  prefill A/B CSVs.
- (e) Manifest: the =auto capture run's manifest carries
  "mmq_splitk":"auto" in resolved_config
  (logs20/splitk-auto.diag.manifest.json).
- (a) Autopsy: PARTIAL, as scoped. A RADV_DEBUG=shaderstats dump
  of the shader_compile k12_k14 family test yielded 10 stats
  blocks, all with 0 spilled SGPRs, 0 spilled VGPRs, 0 scratch,
  including VGPR 256 / LDS 41,984 (x2) and VGPR 256 / LDS 44,032
  (x2), the exact plain K13/K14 pictures from the 2026-11-04
  autopsy. The split pipelines share the plain staging, so
  identical LDS is expected by construction; the defensible
  claim is: no new spills in the K13/K14 family, split included.
  Per-form attribution by name was not isolated (RADV dump blocks
  are anonymous); if wanted, the remaining method is a dump run
  of the splitk_mmq test binary itself under shaderstats.
- Reading against the criterion: fixed 4 shows a real prefill win
  at BOTH lengths (-18% / -14% wall) with the split forms faster
  than plain on the wide shapes too (no unsplit-shape regression
  in the kernel lines), identity+golden hold, and the family
  autopsy shows no new spills. Auto's win is small (about -2 to
  -3.5%) and its factor-8 choice for the KV shape is implicated
  by the factor-8 collapse. Fixed 8 is disqualified by its own
  numbers. The session flips nothing; the parent judges.

## Leg 2, coopmat v2 / K24: decision rule MET on the numbers

Rule (bench/coopmat-v2-2026-11-07.md): material kernel-line win
over K13 with identity intact, else the selector stays off.

- Numeric gates FIRST (the wave64-mapping proof): tests/
  k24_coopmat.rs executed on the card, 3 passed, 0 failed, no
  skips, 0.86 s. (Exception: this binary was run directly, not
  through failure-bundle.sh.) K24 vs reference at batch 64: max
  abs 3.320e-2 (set scale 4.173e4, rel 7.956e-7); batch 62:
  4.688e-2 (rel 1.158e-6). K24 vs K13: 9.766e-3 (batch 64),
  1.172e-2 (batch 62). K24 vs K23: 1.172e-2 both batches. Full
  square K24 vs K13: 1.172e-2. Band refusals by name pass. All
  inside the MMQ reassociation band. The test's reference gate
  IS the proof of the wave64 accumulator element mapping (element
  e at row (inv/16) + 4e, column inv % 16): a wrong mapping
  scrambles rows/columns and fails by a wide margin. The mapping
  is PROVEN.
- Timing A/B (interleaved pairs, 1,022-token prefill, DIAG=timing,
  BRIDGE=1 on all arms; logs20/ab2/): K24 (k24_q4k_coopmat):
  2,912 dispatches, 1,737.2 / 1,751.9 ms, 112.2 / 111.3 GB/s;
  walls 4,052 / 4,085 ms; total GPU 3,721.3 / 3,748.6 ms. K13
  control (k13_q4k_mmq): 2,912 at 4,021.0 / 4,046.5 ms, 48.5 /
  48.2 GB/s; walls 6,307 / 6,401 ms. K23 v1 (k23_q4k_coopmat):
  2,912 at 16,900.4 / 16,963.0 ms, 11.5 GB/s; walls 19,314 /
  19,505 ms, reproducing the standing v1 verdict (16,951.3 ms)
  almost exactly. K24 is about 2.3x faster than K13 on the Q4_K
  kernel line and about 9.7x faster than K23; prefill wall
  improves about 36% vs the K13 control. K14 line unchanged
  across arms (about 1,277-1,364 ms).
- Identity + golden with COOPMAT=1 + COOPMAT_V2=1 (+BRIDGE=1):
  IDENTITY PASS + GOLDEN PASS; spec stream byte-identical to the
  golden stream; spec stats rounds=38 proposed=37 accepted=25
  (the shipped-default accounting); effective 16.88 tok/s (SPEED
  line only). Manifest carries "coopmat":"true",
  "coopmat_v2":"true", "mmq_splitk":"off".
- Autopsy (the one v1 never got): k24/k23 test binaries under
  RADV_DEBUG=shaderstats, private MESA_SHADER_CACHE_DIR; raw
  dumps logs20/autopsy2/k24.err and k23.err. Attribution by the
  staging-size fingerprint: K24: LDS 33,792, VGPR 132, scratch 0,
  spills 0, exactly the note's five staging arrays, and VGPR 132
  (not the 256 cap) confirms the register-resident fold. K23 v1:
  LDS 37,888, VGPR 72, scratch 0, spills 0. Also present, matching
  prior records: K13/K14 (LDS 41,984/44,032, VGPR 256, clean),
  K23-repack (LDS 22,528, VGPR 256, scratch 25,088, 248 spilled
  VGPRs), wide MMQ K15/K16 (LDS 59,392/62,464 with 1,041-1,213
  spills). Parser caution: a naive regex over the dump produces
  garbage blocks (VGPR above 512, spill counts in the billions)
  from mis-split records; ignore them.
- Verdict against the rule: the win condition IS met (material
  kernel-line win over K13, identity intact, autopsy clean). The
  selector stays off; the flip is the parent's call.

## Leg 3, f16 KV: identity gate PASSED; default stays OFF

- (a) tests/kv_f16.rs EXECUTED on the card: 4 passed. The
  f32-vs-f32 determinism control reads 0.0000e0 at every step;
  uniform-weight arm steps read up to about 7.0930e-5; the session
  fixture's worst f16-vs-f32 logit delta = 1.5376e-4 (the
  measured number; the test's 0.5 ceiling is a tripwire). The
  test printed the full-context KV line: 1,783,808 rows, f32
  3,653,238,784 B, f16 1,826,619,392 B.
- (b) Real-file identity + golden with WGPU_COMPUTE_KV_F16=1:
  depth 1 IDENTITY PASS + GOLDEN PASS; depth 6 IDENTITY PASS +
  GOLDEN PASS (d6 spec stats rounds=32 proposed=180 accepted=31,
  the shipped-default accounting). All three stamps confirmed:
  resident prefill capture shows k9_attn_prep_f16 832 dispatches
  and k6_attn_flash_staged_f16 832 at 650.9 ms; the d6 spec run
  shows k6_attn_flash_f16 3,328, k6_attn_flash_staged_f16 1,716,
  k9_attn_prep_f16 5,044. Resident manifest carries
  "kv_f16":"true". Prefill argmax at 1,022 tokens under f16: id
  7235, logit 12.248739 (f32: 12.233533; same argmax).
- (c) 8,202-token prefill wall A/B, TWO interleaved pairs,
  timing-off: f16 66,613 / 66,319 ms vs off 65,763 / 65,847 ms.
  f16 is about 1% SLOWER at this length on both pairs; there is
  no prefill speed win. OBSERVATION: at 8,202 tokens the f16
  prefill argmax differs from f32 deterministically (both f16
  reps: id 589, logit 9.080180; both off reps: id 7235, logit
  8.854632), while at 1,022 tokens the argmax matches. The
  identity gate (the protocol's short-prompt spec runs) passed;
  the long-context argmax flip is recorded here as a caveat for
  any future flip decision, not as a gate failure.
- (d) Decode A/B (glimmer_generate --gpu, 32 tokens, short
  prompt, three interleaved reps per arm): generated streams
  IDENTICAL between f16 and off in every rep (first 8 =
  [13796, 26, 1573, 373, 290, 2167, 26899, 1212]). Per-step
  medians: f16 34 / 33 / 34 ms; off 26 / 33 / 34 ms. The single
  26 ms off rep is an outlier against five reps at 33-34 ms; no
  consistent decode delta. (This surface reads about 29-30 tok/s
  because it prints a top-5 table per step; keep surfaces
  separate.)
- The f16 case is memory halving (1,826,619,392 B vs
  3,653,238,784 B at full context), not speed. Default stays OFF
  per the brief; the identity gate result is PASS.

## Leg 4, prefix-KV: real-file proof COMPLETE

- (a) tests/prefix_kv_session.rs EXECUTED on the card: the
  file's single test (resident_session_prefix_store_round_trip,
  covering switch-off behavior, cold-vs-restored continuation,
  the one-call flow, and refusal continuation) passed with NO
  skip: worst continuation logit diff 0.0000e0 over 40 steps.
- (b) Real-GGUF round trip via the staging-only probe: cold run
  forwarded a deterministic 4,096-token prefix, captured and
  stored it (phase=capture_store ms=810; store on disk 261 MiB,
  matching the about 260 MiB prediction); a FRESH process then
  restored it (matched=Some(4096), prefix_restores=1,
  prefix_restored_tokens=4096), proving cross-process
  persistence. Both processes generated 64 tokens; the two id
  files are byte-identical (STREAMS_IDENTICAL, cmp-verified).
- (c) Wall saving: cold phase=cold_prefill ms=29,403 for 4,096
  tokens (139.3 tok/s on this probe's slice path) vs restore
  phase=restore ms=722 plus phase=reforward_last ms=96, total
  818 ms. Net saving 28,585 ms (about 28.6 s) per reuse, against
  the about 25.7 s prediction. The prediction assumed the
  159.3 tok/s prefill rate; this probe's cold rate measured
  slower per token, so the measured saving lands above the
  prediction. Write-side cost is the 810 ms capture_store above.
- The scratch store (logs20/prefix-store) was deleted after the
  runs.

## Leg 5, n-gram: mechanism proven on the real file; prose acceptance is near zero, repetitive acceptance is real

Wave18 DFlash record for comparison: depth 1 effective 22.07
tok/s, depth 6 effective 16.80 tok/s, both identity-passing.

- (a) GPU arms, --drafter=ngram, default prompt, both mode, 64
  tokens. Depth 1: IDENTITY PASS + GOLDEN PASS; spec stats
  rounds=60 proposed=6 accepted=3 acceptance_rate=0.500
  tokens_per_round=1.05; effective 34.61 tok/s, all-in 33.70.
  Depth 6: IDENTITY PASS + GOLDEN PASS; rounds=58 proposed=24
  accepted=5 acceptance_rate=0.208 tokens_per_round=1.09;
  effective 31.45 tok/s, all-in 30.74. Honest reading: on this
  prose prompt the proposer almost never fires (6 proposals in
  60 rounds at depth 1), so both rates are the plain-decode
  floor (about 34.5 tok/s), not a speculation win. They read
  above the DFlash record only because DFlash pays its about
  40.6 ms draft forward every round and the n-gram proposer
  pays nothing. Where the generated stream itself turns
  repetitive (the capital prompt loops), proposals rise (24 at
  depth 6) but acceptance stays low (0.208).
- (b) Repetitive-prompt arm, depth 6 (a six-times-repeated
  instruction block, 113 prompt tokens): IDENTITY PASS (GOLDEN
  prints only for the default prompt). Spec stats rounds=20
  proposed=55 accepted=43 acceptance_rate=0.782
  tokens_per_round=3.15; effective 51.12 tok/s (generate phase),
  all-in 33.21 (the short run's prefill and session build
  dominate the all-in figure). This is the honest best case and
  it works: on truly repetitive text the proposer pays off
  (51.12 vs the about 34.5 plain floor and vs DFlash depth-6
  16.80), at zero draft cost.
- (c) CPU-driver acceptance run (spec_generate --drafter=ngram
  on the real file): CUT, stated plainly. The baseline session
  measured CPU walls of 2,256-4,802 s per 16-token prompt; a
  64-token both-mode CPU run is a 5-10 hour job that would hold
  the phase lock and outlive this session. The proposer code
  path is already proven on the real file by the GPU arms above
  (identity token-identical at both depths) and by the fixture
  and recorded-stream tests in the n-gram lanes. The CPU arm
  remains owed if a future session wants it for completeness.
- Framing, per the notes: acceptance collapses on non-repetitive
  text; n-gram complements DFlash by never paying the draft
  forward, it does not replace it.

## Leg 6, hygiene end-to-end: DONE

The DIAG=capture run of leg 1d (spec_generate_gpu, =auto) wrote
the dflash session manifest at logs20/splitk-auto.diag.manifest
.json. Its resolved_config carries, verbatim:
"k21f_head":"true","flash_staged":"true","mmq_splitk":"auto",
"coopmat":"false","coopmat_v2":"false". The flash_staged pair IS
written by a capture-mode run with its resolved value (true at
shipped defaults), closing the defaults-confirmation leg-2
evidence gap on the manifest side. The resident side carries its
stamped pairs by the same code path; leg 3b's resident manifest
independently confirms it (kv_f16 pair included).

## Leg 7, calibrator re-derivation: table NOT changed

examples/calibrate_profile run on the card at default reps (25),
profile written to logs20/calibrated-profile.txt inside the
staging tree (the default profile path was never written; no
default-path profile exists for the server account, verified).
All 99 evidence lines were printed before the derived table.
The batch-1 exclusion fix is visible in the output: every rows=1
line reads verdict=excluded, including rows whose ratio beats
0.98 (for example q6k kv rows=1 ratio=0.641, excluded), so no
derived cell contains batch 1. Ceiling derived: 16 (= shipped).

Derived vs shipped, cell by cell:
- q4k attn: derived 2,6-12 vs shipped 6-12 (row 2 gained)
- q4k ffn: derived 2-16 vs shipped 2,4-12 (row 3 and rows 13-16
  gained)
- q4k kv: derived 2,7-16 vs shipped 2-3 (row 3 lost, rows 4-6
  measured losses but see evidence, rows 7-16 gained)
- q5k attn: derived 2-16 vs shipped 2-16 (MATCH)
- q5k ffn: derived 2-16 vs shipped 3-16 (row 2 gained)
- q5k kv: derived 2-16 vs shipped 16 (rows 2-15 gained, all on
  marginal ratios; see the instability note below)
- q6k attn: derived 2-16 vs shipped 2-12 (rows 13-16 gained)
- q6k ffn: derived 2-12 vs shipped 2-12 (MATCH)
- q6k kv: derived 2-16 vs shipped 12 (rows 2-11 and 13-16
  gained)

Evidence for the five cells that differed in wave19 (ratios are
mmv/control medians; full 99 lines in logs20/leg7-calibrate.log
and as comments in logs20/calibrated-profile.txt):
- q4k ffn: wins at every routable row: rows 2-12 ratios
  0.771/0.624/0.701/0.803/0.735/0.707/0.606/0.634/0.624, row 16
  ratio 0.955 (win, marginal). Wave19 derived 2-12; wave20 adds
  row 16, which wave19 could not measure (the sweep-era arms
  served row 16 with MMQ).
- q4k kv: genuinely mixed. Row 2 win (0.899); rows 3-6 clear
  LOSSES (1.220/1.393/1.665/1.177); rows 7-12 wins
  (0.840/0.795/0.829/0.880); row 16 win (0.581). The derived
  2,7-16 honestly reflects this shape. Wave19 derived 1-2,4-6,16,
  which disagrees with wave20 on rows 4-6 (wave20 measures them
  as losses by wide margins) and on row 3.
- q5k ffn: wins at every routable row, ratios 0.671 (row 2) down
  to 0.509 (row 16). The row-2 gain wave19 found is confirmed
  with a wide margin (0.671).
- q5k kv: wins at every routable row but ALL marginal: ratios
  0.819/0.893/0.950/0.962/0.963/0.964/0.931/0.932/0.869/0.908
  for rows 2,3,4,5,6,7,8,9,12,16. INSTABILITY FLAG: wave19
  derived this cell CLOSED (1) where wave20 derives it fully
  open (2-16). The cell sits within measurement noise of the
  0.98 line at most rows and has now flipped end to end between
  two sessions; no shipped change should rest on this cell
  without sweep-CSV-grade evidence.
- q6k kv: wins at every routable row with wide margins at rows
  8-12 (0.504/0.402/0.363), narrower at rows 2-7 (0.626-0.714)
  and row 16 (0.972, marginal). Wave19 derived 4-12; wave20
  widens both ends.

Standing guidance from wave19 applies unchanged: any shipped
table change should be decided with sweep-CSV-grade evidence
beside it. This run is a decision input only.

## Failure bundles (~/gpu-wave20/bench/failures/)

Every FAIL bundle in this session is a SPEED-line-only artifact
(IDENTITY and GOLDEN pass in its log) unless noted:
- 20261107T092102Z-leg1de-splitk-auto (SPEED only)
- 20261107T092149Z-leg1de-splitk-f8 (SPEED only)
- 20261107T092237Z-leg1d-golden-auto (SPEED only)
- 20261107T092319Z-leg1d-golden-f8 (SPEED only)
- 20261107T092352Z-leg1d-golden-auto (SPEED only)
- 20261107T092428Z-leg1d-golden-f8 (SPEED only)
- 20261107T093443Z-leg2c-k24-a (HARNESS artifact, not a product
  failure: missing logs20/ab2/ dir failed the diag summary write
  and the run profiled a 7-token prompt because a file path was
  passed where prompt text belongs; superseded by the fixed
  re-runs, all PASS)
- 20261107T093847Z-leg2d-k24-identity (SPEED only)
- 20261107T094050Z-leg3b-f16-identity (SPEED only)
- 20261107T094212Z-leg3b-f16-d6 (SPEED only)
- 20261107T105024Z-leg5a-ngram-d1 (SPEED only)
- 20261107T105114Z-leg5a-ngram-d6 (SPEED only)
- 20261107T105155Z-leg5b-ngram-repetitive-d6 (SPEED only)
The two 20261106T2243* demo bundles predate the session (they
ship in the repo archive). All other runs recorded PASS index
lines; their working logs are logs20/leg*.log and the CSVs in
logs20/ab1/ and logs20/ab2/.

## Closeout

Tenants restored and VRAM verified (see the session report to
the parent for the exact settled figure). Production
llama-server untouched: nothing listened on port 8080 at any
point in the session. No reboot. The scratch prefix store was
deleted. This record lives in the staging tree only; the parent
files it.
