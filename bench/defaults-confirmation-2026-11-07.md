# Defaults confirmation session (2026-11-07)

One consolidated card session on ubuntu-server against main
7940bcf70, the first tree where all three shipped defaults are
live at once: the MMV band gate open to cap 16
(WGPU_COMPUTE_MMV_MAX_ROWS default 16), K21F finish-carrying
head routing behind that gate, and staged flash on
(WGPU_COMPUTE_FLASH_STAGED default on). Four legs: K21F
exposure and identity, flash default exposure, coopmat bridge
smoke, and shipped-default speculative rates.

Headline, stated plainly first: **leg 1 FAILS.** With every
env var at its shipped default, the speculative stream is NOT
token-identical to plain greedy. The divergence is
deterministic, reproduces in every default-configuration run
in this session (eight of them, both depths), and is
attributable to the K21F head route by the isolation arms
below. The configuration is faster than the baseline
session's cap-16 arms at both depths, and that speed and the
divergence are the same phenomenon, so the rate gains in
leg 4 cannot be credited until the K21F numerics are fixed
and identity passes again. Legs 2 and 3 pass on their own
criteria. The control arm (the whole MMV gate closed by
WGPU_COMPUTE_MMV_MAX_ROWS=1) passes identity and golden, so
a one-variable rollback exists, at the price of every MMV
route, not just the head's.

## Session discipline

- Staging: `~/gpu-wave16` from a git archive of 7940bcf70.
  MANIFEST.txt carries 438 sha256 lines; `sha256sum -c`
  passes on all 437 real files (the one FAILED line is
  MANIFEST.txt hashing itself, an artifact of generating the
  manifest inside the tree it lists). Key files match the
  commit byte for byte: crates/wgpu-compute/src/session.rs
  sha256 d5132a0886855a4b96b5997b91cbd37b82172d535c51069188
  e0a28d8f1a6628, crates/wgpu-compute/src/compute.rs sha256
  85fb1142109d2c6747e3963f9df4d988ff91af49db03191a3d1fb3c22
  c3979e6. wgpu-compute, dflash, itak-spec, and itak-loader
  all compiled from the staged tree (build log paths prove
  the source root); binaries built 2026-11-07 05:29-05:30Z.
- Model: /mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-
  KQuant-17GB-Q4_K_M.gguf, sha256 verified this session:
  4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f1
  83f60e. Drafter: dflash-Muse-Glimmer-30B-Q4_K_M.gguf.
- Card: tenants stopped before measurement (cortex-embed
  .service, cortex-cortex-chat-1, cortex-cortex-embed-1);
  VRAM read exactly 33,271,808 B used (driver residue) at
  every measurement. All runs serialized through
  `flock -w 14400 /tmp/itak-phase.lock`, one heavy job at a
  time, CARGO_BUILD_JOBS=4. Production llama-server never
  touched; no reboot.
- Every verdict run wrapped in bench/failure-bundle.sh.
  Bundles live on the server under
  `~/gpu-wave16/bench/failures/`; full run logs under
  `~/gpu-wave16/logs16/`. As in prior sessions, spec-run
  bundles carry FAIL marks from the standing 88.6 tok/s
  SPEED bar line; that bar is a target, not a leg verdict.
  Bundles whose only marks are the bar are named as such
  below.
- Result CSVs and stream files copied into the repo at
  bench/results/defaults-confirmation-20261107/.

Common invocation (legs 1, 3A, 3C, 4), from the staged root:

```
./target/release/examples/spec_generate_gpu \
  /mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf \
  /mnt/models/Muse-Glimmer-30B/dflash-Muse-Glimmer-30B-Q4_K_M.gguf \
  64 <depth> "The capital of France is" both <stream-file>
```

Leg 2 and 3B invocation:

```
./target/release/examples/prefill_profile \
  /mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf \
  "<the fox sentence repeated 102 times>"
```

## Leg 1, K21F exposure: FAIL (identity), exposure itself proven

Default arm (no WGPU_COMPUTE_* routing overrides;
WGPU_COMPUTE_DIAG=timing only), depth 1, bundle
20261107T053257Z-leg1-k21f-default:

- Exposure: `k21_q5k_mmv_finish` fires 36 times, every one
  at `rows=202048 cols=6656 quant=Q5_K shape=batch=2
  finish=2`, durations about 1.36 to 1.41 ms at 655 to
  680 GB/s. 36 is exactly the round structure's proposing
  count (spec stats: rounds=37 proposed=36 accepted=26):
  one finish-form head dispatch per proposing round, at the
  depth-1 verify width of 2 rows. The batch-1 head stays on
  K3: 66 `k3_q5k_gemv ... shape=batch=1 finish=2` dispatch
  lines in the same log.
- GOLDEN: PASS (first 8 match the verified continuation
  [13796, 26, 1573, 373, 290, 2167, 26899, 1212]).
- IDENTITY: FAIL. Plain and spec streams agree through
  index 22 and diverge at index 23: plain 589, spec 1573.
  Spec stats differ from every passing run on record
  (rounds=37 proposed=36 accepted=26, acceptance 0.7222)
  where the control arm below and the baseline session's
  arms read rounds=38.

Control arm (WGPU_COMPUTE_MMV_MAX_ROWS=1, diag timing),
bundle 20261107T053451Z-leg1-k21f-control (FAIL marks are
the speed bar only):

- `k21_q5k_mmv_finish` count: 0, as required. All head
  calls on K3 (141 `k3_q5k_gemv` dispatch lines).
- IDENTITY: PASS (64 tokens token-identical to plain GPU
  greedy). GOLDEN: PASS. Rounds=38 proposed=37 accepted=25.

Isolation arm (shipped defaults plus
WGPU_COMPUTE_FLASH_STAGED=0), bundle
20261107T053656Z-leg1-k21f-flashoff:

- IDENTITY: FAIL, with the spec stream byte-identical to
  the default arm's (same tokens, same rounds=37/36/26,
  36 finish stamps). Staged flash is therefore not a
  contributor.

Attribution. The baseline session (bench/baseline-session-
2026-11-06.md, legs (d) and (i)) supplies the other two
cells of the matrix on its own card evidence: MMV at cap 16
with flash off and no K21F passed identity at depths 1 and
6 (38 rounds at depth 1); flash staged on with the MMV gate
closed and no K21F passed identity and golden at depths 1
and 6. Between that tree (2c1f8c7) and this one (7940bcf70),
the only spec-path code delta that can move a verify
argmax is the K21F chain (the coopmat rung is flag-off and
never fired in any leg-1 arm; the two flips are resolver
defaults, exercised equivalently by the baseline arms'
explicit env). The K21F call as a unit (K21 product core
plus the carried finish) produces head logits that flip a
borderline verify argmax at stream position 23, where the
repeated phrase restarts and the top-2 logits are near
tied. Whether the delta sits in the product core or the
finish application is a question for the fix lane; the
fixture-scale pins in tests/k21f_head_routing.rs hold the
block logits to the CPU model at the cross-backend bar
(max-abs under 0.5) and both arms pass that bar, so the
delta is below fixture-bar size and only a real-file
near-tie exposes it. Single-run rates in this leg, for
reference only: default 23.12 tok/s effective, control
17.68.

## Leg 2, flash default exposure: PASS, with one evidence gap named

Arm A (env unset besides diag timing), prefill_profile at
1,022 tokens, wrapper PASS:

- `k6_attn_flash_staged` fires 832 times, once per multi-row
  attention call. The diag summary table (bench/results/
  defaults-confirmation-20261107/leg2-default.csv) reads
  k6_attn_flash_staged 832 dispatches, 605.3 ms; total GPU
  5,912.3 ms over 10,448 dispatches. Wall phase=prefill
  ms=6141.
- prefill_argmax id=7235 logit=12.233533, identical to the
  value recorded by two prior sessions.

Arm B (WGPU_COMPUTE_FLASH_STAGED=0), wrapper PASS:

- Zero staged stamps; the summary table reads k6_attn_flash
  832 dispatches, 970.4 ms; total GPU 6,314.5 ms over the
  same 10,448 dispatches. Wall ms=6531 (single samples; the
  interleaved verdict stands on bench/flash-host-cost-
  2026-11-06.md, these walls are context only).
- prefill_argmax id=7235 logit=12.233533, bit-identical to
  arm A: the two forms agree exactly on this shape's final
  logits.

Evidence gap, stated plainly: the flip note's owed leg
asks for a construction log line reading enabled=true. No
shipped harness can emit that line. The line is produced by
the crate's `dbg_log!` seam (crates/wgpu-compute/src/
diag.rs), which prints only after a host program calls
`diag::set_debug(true)`, and no example, binary, or test in
the tree calls it; the diag manifest's resolved_config
likewise carries no flash field (it lists small_batch_max
_rows, mmv_max_rows, mmv_bands, chunked_gemv_max_rows,
k22_mmq, mmq_wide_min_rows). The owed leg is therefore
discharged by dispatch-stamp evidence instead, which is
stronger than the log line: the staged kernel provably
serves every multi-row call with the env unset and provably
serves none with the env at 0. Closing the gap properly
(wire set_debug behind an env in the examples, or add the
flash resolution to the manifest) is follow-up work for a
later lane; it was not improvised inside this session.

## Leg 3, coopmat bridge smoke: PASS

Bridge env: WGPU_COMPUTE_COOPMAT_BRIDGE (exact name from
crates/wgpu-compute/src/coopmat_bridge.rs, COOPMAT_BRIDGE
_ENV); routing env WGPU_COMPUTE_COOPMAT.

Arm A (BRIDGE=1, routing unset), spec_generate_gpu depth 1,
bundle 20261107T054213Z-leg3-bridge-smoke:

- The backend opens on the bridged device and the full
  stack runs end to end (plain phase, draft session, spec
  phase all complete; GOLDEN PASS).
- `k23_q4k_coopmat` stamps: 0, as required with the routing
  switch unset.
- The emitted stream file is byte-identical to leg 1's
  default arm (md5 0dec5a1dbe5436232dbef35f6610e93f for
  both): the bridge is numerically transparent. Its
  IDENTITY: FAIL line is leg 1's K21F finding reproducing
  under identical defaults, not a bridge effect; the
  identical stream is the proof.

Arm B (BRIDGE=1, COOPMAT=1), prefill_profile at 1,022
tokens, wrapper PASS:

- `k23_q4k_coopmat` fires 2,912 times, exactly the dispatch
  count K13 records for this shape in every prior profile:
  the coopmat rung served every Q4_K MMQ call.
- prefill_argmax id=7235, the control value: output
  identity holds at the top-1 level. The logit reads
  12.426529 against the control's 12.233533; the coopmat
  product is an integer MMQ form and answers to the
  quantization band, not the f32 noise band, and the
  spot-check criterion (argmax agreement) is met. Wall
  ms=19267, consistent with the standing verdict that the
  kernel is about 4x slower than K13 (bench/coopmat-build-
  2026-11-06.md); recorded, not re-benched.

Arm C (BRIDGE=1, COOPMAT=1), spec_generate_gpu depth 1,
bundle 20261107T054338Z-leg3-coopmat-spec:

- GOLDEN PASS; stream byte-identical to leg 1's default
  arm again. The coopmat rung never engages at spec shapes
  (the prompt is 6 tokens, the verify width is 2; MMQ needs
  chunk rows at or past 16), so this arm re-proves the
  bridge's transparency with both switches on, and its
  IDENTITY: FAIL is leg 1's finding once more.

## Leg 4, shipped-default speculative rates

spec_generate_gpu, 64 tokens, both mode, diag timing, all
routing env unset, 3 reps per depth. Effective tok/s (the
harness's "spec rates: effective" line):

| depth | rep 1 | rep 2 | rep 3 | median | baseline cap-16 arms |
|---|---|---|---|---|---|
| 1 | 22.99 | 23.16 | 23.18 | 23.16 | 21.73 to 22.10 |
| 6 | 21.21 | 21.75 | 21.65 | 21.65 | 16.33 to 16.47 |

Every rep at both depths prints IDENTITY: FAIL and GOLDEN:
PASS. Depth 1 reproduces leg 1 exactly (rounds=37
proposed=36 accepted=26 in all reps). Depth 6 diverges at
the same stream position 23 (plain 589, spec 1573); its
finish stamps fire at the live verify widths (batch=7 on
full rounds, 5 and 6 on tail rounds; 25 finish dispatches
in rep 1), and its round structure moves with the flipped
decisions (rounds=26 proposed=147 accepted=37, acceptance
0.252, identical across reps, against the baseline arms'
32 rounds).

Reading, plainly: K21F's head routing moved the rates up at
both depths (about +5 percent at depth 1, about +31 percent
at depth 6 against the baseline session's routed arms) AND
broke token identity at both depths. Both effects flow from
the same changed verify logits, so the honest summary of
the shipped defaults today is: faster, and not the same
stream. These numbers are the first measured rates of the
exact configuration on main; they are not a win claim.

## Restoration

Tenants restored at session end: cortex-cortex-chat-1 and
cortex-cortex-embed-1 started (both Up, healthy),
cortex-embed.service started (active). VRAM settled at
8,932,147,200 B used, the recorded tenant footprint.
Production llama-server untouched throughout; the server
was not rebooted.

## What did not behave as predicted

1. Leg 1's default arm fails IDENTITY. The K21F routing
   lane's gates (fixture logits at the cross-backend bar,
   batch pins) all pass; the real file's near-tie at
   position 23 is below that bar and flips. The card
   protocol existed to catch exactly this, and it did.
2. The flash construction line the flip note owed is
   unemittable by the shipped harnesses (leg 2): the
   dbg_log seam has no caller that enables it.
3. Leg 4's depth-6 rate jump is not separable from the
   identity failure; quoting it without the failure would
   misread the session.

## Bundle index (server: ~/gpu-wave16/bench/failures/)

- 20261107T053257Z-leg1-k21f-default: IDENTITY FAIL +
  speed bar.
- 20261107T053451Z-leg1-k21f-control: speed bar only (the
  run itself passes identity and golden).
- 20261107T053656Z-leg1-k21f-flashoff: IDENTITY FAIL +
  speed bar.
- 20261107T054213Z-leg3-bridge-smoke and
  20261107T054338Z-leg3-coopmat-spec: the leg-1 identity
  failure reproducing under the bridge; streams identical
  to leg 1's default arm.
- 20261107T054507Z/54554Z/54641Z-leg4-d1-rep1..3 and
  20261107T054725Z/54819Z/54910Z-leg4-d6-rep1..3: IDENTITY
  FAIL + speed bar.
- Leg 2 arms and leg 3 arm B: wrapper PASS (index lines
  only in bench/failures/INDEX.md).
