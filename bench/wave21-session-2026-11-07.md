# Wave21 card confirmation session, 2026-11-07

One session on ubuntu-server (RX 7900 XTX, RADV NAVI31, Mesa
26.0.8-1ubuntu0.3, kernel 7.0.0-42), staged from itak-torch-rebuild
main cb579ad4c9c464d34ca7a6af40492b6b1f7cb064 at ~/gpu-wave21.
Closes the flip sequence: proves the NEW SHIPPED DEFAULTS on the
real card (coopmat rung with K24 selected, split-K fixed factor 4),
measures the combined K24 + split-K prefill figure no earlier run
measured, discharges the card-owed f16 composition test from
bench/prefix-f16-guard-2026-11-07.md, and runs bench/gate.sh on the
card for the first time. No code was changed in this session; the
two defects found are recorded here and their fixes are later
lanes.

## Setup

- Remote main verified = cb579ad4 before staging. Staged via git
  archive created locally as /tmp/wave21-stage.tar (8,417,280 B),
  sha256
  a3445aa1468989ca4f7ae7905be3523477a7aee0388183639934b7ee04c8ea20,
  identical on both sides after transfer. Key staged file hashes
  match the commit's blobs: compute.rs
  1f3bc7054be1d54b54aefae71d7da33dfb7febac32876a2c1baa62dfce2c4fa5,
  session.rs
  0f315baf9191cdd93d2a1b03eeecbc96eba5955aa93366d272ddad43a77997b8,
  spec_generate_gpu.rs
  c455e1d79817269379600c9c8c0b6804623a17a0086cba7517b40f3bbe8c3ccd,
  bench/gate.sh
  de9a3661c9b7607dd27f3b694d413c756ccfff591a09f90575b0dec568b62c60.
- Model sha256 re-verified:
  4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e.
  DFlash draft file present beside it, sha256
  b2e808bf656086fe86bd0d0bd990f01d33e377537a07c02d45371517c8b264ef.
- No default-path calibration profile exists for the server
  account (no ~/.config/itak directory), so all default runs are
  profile-free; the runner also scrubbed XDG_CONFIG_HOME and
  WGPU_COMPUTE_PROFILE_PATH.
- Tenants stopped for the session (cortex-embed.service,
  cortex-cortex-chat-1, cortex-cortex-embed-1); VRAM drained to
  exactly 33,271,808 B before measuring. cortex-cortex-1 and
  cortex-cortex-surrealdb-1 were left running; they are not card
  tenants. Port 8080: nothing listening at any point (production
  Glimmer serving is down per David's standing order; production
  llama-server untouched). No reboot.
- All runs serialized through flock -w 14400 /tmp/itak-phase.lock
  and wrapped in bench/failure-bundle.sh via logs21/runner.sh. The
  runner applies an env scrub list (every WGPU_COMPUTE_* routing
  and profile switch, ITAK_KV_PREFIX_*, PREFILL_PROFILE_SLICE,
  XDG_CONFIG_HOME) before any per-arm assignments, and prints an
  inner-env proof inside the wrapped command's own output stream,
  so each bundle log itself proves the literal unset-ness of the
  switches for the default arms.
- Bundle inventory: every spec run created a bundle
  (20261107T170315Z-leg1a-spec-unset,
  20261107T171818Z-leg1d-spec-coopmat0,
  20261107T171902Z-leg1d-spec-v2off,
  20261107T171950Z-leg1d-spec-bridge0,
  20261107T172039Z-leg1a-spec-nodepth). Each bundle's stdout.log
  was checked: IDENTITY PASS and GOLDEN PASS in all five; the
  only FAIL marker in each is the standing 88.6 tok/s SPEED line.
  All five are SPEED-line artifacts, not failures. No prefill run
  created a bundle (no failure markers in their output). The two
  20261106 demo bundles in the tree ship with the repo archive.
- Prompts are the bench protocol's fox sentence ("The quick brown
  fox jumps over the lazy dog. ", 10 tokens): 102 reps = 1,022
  tokens, 820 reps = 8,202 tokens (prompt_tokens as printed by
  the harness).
- Harness caveat, from prefill_profile's own header: in diag
  timing mode each slice boundary pays a drain, so a timing-mode
  wall is an upper bound and the timing-off wall is the wall of
  record. Both leg 3 arms share the timing mode, so the
  comparison is internally consistent; one timing-off data point
  exists (glimmer_generate at defaults, 8,202 tokens:
  phase=prefill 42,592 ms, below the timing-mode 46,291 ms,
  consistent with the caveat).

## Leg 1, coopmat default confirmation: PASS

Protocol: bench/coopmat-default-flip-2026-11-07.md, at literally
unset env for all three coopmat switches (absence proven by the
runner's scrub list plus the inner-env proof in each bundle log).

- (a) Spec run at unset env (split-K also unset),
  spec_generate_gpu both mode, 64 tokens, depth 1 passed
  explicitly: IDENTITY PASS, GOLDEN PASS, accounting
  rounds=38 proposed=37 accepted=25, exactly the shipped-default
  control. SPEED FAIL 18.02 tok/s is the standing bar artifact
  only (the run was under DIAG=capture; capture overhead applies;
  not a speed claim). A second run with the depth argument
  OMITTED also passed IDENTITY + GOLDEN, confirming the runner's
  new default depth of 1 empirically as well as in source
  (const DEFAULT_DEPTH: usize = 1 in spec_generate_gpu.rs).
- (b) Manifest: the capture manifest (logs21/leg1-spec.csv
  .manifest.json; the dflash session's, last writer at the
  shared DIAG_OUT path, the known wave16/19 behavior) carries
  verbatim "coopmat":"true", "coopmat_v2":"true",
  "mmq_splitk":"fixed:4", and "git":"unknown" (the known
  manifest gap; archive staging has no .git). The spec run's own
  kernel census (logs21/leg1-spec.csv) carries ZERO k24 and ZERO
  k23 stamps: at spec shapes (7-token prompt, verify rows below
  the MMQ floor) the GEMV/MMV forms serve (k1_q4k_gemv 1,178,
  k21_q5k_mmv 38, k19_q4k_mmv 6), so the coopmat rung does not
  engage there. That is consistent with the record and is not a
  leg failure; rung engagement at unset env is proven by the
  1,022-token prefill census in (c).
- (c) 1,022-token prefill at unset env (also leg 3 arm A):
  k24_q4k_coopmat count = 2,912 exactly in all three reps, equal
  to wave20's flagged K24 arm. Kernel line 1,718.678 /
  1,729.950 / 1,768.165 ms: reps 1-2 land just under the wave20
  band (1,737-1,752 ms), rep 3 lands about 1% over the band top.
  Recorded as observed: count exact every rep; the line sits at
  the band, a few ms either side across reps, inside
  cross-session noise. k23_q4k_coopmat count = 0 in all default
  arms.
- (d) Rollback arms, each measured on both harnesses (spec run
  plus a 1,022-token prefill):
  - WGPU_COMPUTE_COOPMAT=0: spec IDENTITY PASS + GOLDEN PASS.
    Prefill wall 5,213 ms; zero coopmat stamps;
    k13_q4k_mmq_split 2,912 @ 3,337.849 ms; k14_q6k_mmq_split
    832 @ 733.004; splitk_reduce 3,744 @ 50.310; prefill argmax
    id=7235 @ 12.275854 (the winner agrees with defaults; the
    logit differs by reassociation, expected).
  - WGPU_COMPUTE_COOPMAT_V2=0: spec IDENTITY PASS + GOLDEN PASS.
    Prefill wall 18,804 ms; k23_q4k_coopmat 2,912 @ 16,929.707
    ms (reproduces the standing K23 about-4x-slower verdict);
    k24 count = 0; argmax 7235 @ 12.493924.
  - WGPU_COMPUTE_COOPMAT_BRIDGE=0: spec IDENTITY PASS + GOLDEN
    PASS. Prefill wall 5,232 ms; zero coopmat stamps; census
    identical in shape to the COOPMAT=0 arm (k13 split 2,912 @
    3,344.456; k14 split 832 @ 733.820; reduce 3,744 @ 50.366);
    argmax 7235 @ 12.275854, bit-identical logit to the
    COOPMAT=0 arm. The plan-build refusal is logged by name
    (visible with WGPU_COMPUTE_DEBUG=1, the diag env twin):
    "[wgpu-compute] mmq coopmat q4k requested by env but no
    coopmat pipeline exists on this device (stock device, or a
    bridged device failing the kernel's gates); the standard
    ladder serves it".

## Leg 2, split-K default confirmation: PASS

Discharged through the leg 1 and leg 3 evidence at literally
unset WGPU_COMPUTE_MMQ_SPLITK:

- The leg 1(a) capture manifest carries "mmq_splitk":"fixed:4".
- Split stamps at factor 4 appear in every default prefill
  census on the shapes that reach the standard rung:
  k14_q6k_mmq_split 832 @ 700.451 / 706.577 ms and
  splitk_reduce 832 @ 3.081 / 3.114 ms at 1,022 tokens (the Q4_K
  projections are served by K24 above the rung, as the flip
  note anticipated). With the coopmat rung off (leg 1(d)
  COOPMAT=0 arm), k13_q4k_mmq_split also appears at 2,912 @
  ~3,338-3,344 ms with splitk_reduce 3,744: the full factor-4
  picture in each configuration.
- Spec identity + golden at unset split-K: PASS (the leg 1(a)
  run itself).

## Leg 3, combined prefill: MEASURED at both shapes

Interleaved arms, DIAG=timing, failure-bundle wrapped. Arm A:
shipped defaults (everything unset). Arm B: pre-flip control
(COOPMAT=0 and MMQ_SPLITK=off).

### 1,022 tokens, 3 rounds each, interleaved A B A B A B

- Arm A walls: 3,348 / 3,371 / 3,539 ms, median 3,371 ms.
  Argmax 7235 @ 12.230860 in all reps. K24 line 1,718.678 /
  1,729.950 / 1,768.165 ms @ 2,912; K14 split line 700.451 /
  706.577 ms @ 832; reduce ~3.1 ms @ 832.
- Arm B walls: 6,678 / 6,471 / 6,433 ms, median 6,471 ms.
  Argmax 7235 @ 12.233533 in all reps. K13 line 4,213.549 /
  4,104.827 / 4,089.647 ms @ 2,912.
- Combined wall improvement: about -48% (3,371 vs 6,471 ms).
  Placed against the wave20 single-feature figures WITHOUT
  summing them: K24 alone was -36% of wall and split-K alone
  -18% at this shape. The combined figure is larger than
  either single because the control here loses both wins at
  once, and the two wins only partly overlap: the Q4_K part
  overlaps exactly as the panel predicted (split-K does not
  stack on the coopmat rung), while the Q6_K part does not
  overlap (the K14 line roughly halves, ~1,310 ms plain to
  ~703 ms split, inside the defaults arm). Kernel-line ratio
  K24 vs K13 on medians: about 2.38x (~1,730 vs ~4,105 ms).

### 8,202 tokens, two interleaved rounds

- Arm A walls: 46,291 / 45,975 / 46,047 ms (three reps; round 2
  contributed two), median 46,047 ms. Census: k24_q4k_coopmat
  23,296 @ 14,033.359 / 13,956.950 ms; k14_q6k_mmq_split 6,656
  @ 5,771.867 / 5,728.037 ms; splitk_reduce 6,656 @ ~25 ms.
- Arm B walls: 68,116 / 68,909 ms, median 68,512 ms. Census:
  k13_q4k_mmq 23,296 @ 31,857.392 / 32,235.388 ms;
  k14_q6k_mmq 6,656 @ 10,126.004 / 10,265.550 ms.
- Combined wall improvement: about -33% (46,047 vs 68,512 ms).
- The winner at this shape differs by configuration; see the
  divergence section below.

## The 8,202-token argmax divergence: investigated, verdict below

At 1,022 tokens every configuration measured (defaults, both
rollbacks, control) agrees on winner 7235. At 8,202 tokens the
configurations divide deterministically:

- Defaults (K24 + split-K 4): id=589 @ 9.131593, reproduced
  bit-identically in three prefill reps and in two independent
  full-forward harnesses (prefill_profile's argmax tail and
  glimmer_generate --gpu-argmax, which share no readback code).
- Coopmat off only (split-K still on): id=589 @ 9.004866.
- Split-K off only (coopmat still on): id=589 @ 9.109748.
- Pre-flip control (both off): id=7235 @ 8.854632,
  bit-identical to wave20's f32 off-arm value at this shape.

So EACH flip alone moves the final-position winner from 7235 to
589, at three distinct logit values; the two flips do not need
each other to do it. 589 was also wave20's f16 winner at 8,202
tokens, but at a different logit (9.080180); KV_F16 was
scrubbed/unset in every wave21 arm and the leg 1(a) manifest
carries the resolved f32 config, so this is not the f16 path
engaging.

VERDICT, stated plainly: this is a near-tie break, not a proven
defect, and not a proven non-defect either. The fox prompt is
periodic with a 10-token period, so after 8,202 tokens the
final position is a near-tie between period-continuation
candidates, and any change in accumulation order (K24's integer
coopmat product, split-K's reassociated reduction, each
documented as reassociation-class drift) can break the tie
differently. Both configurations are internally deterministic
to the last bit across reps and across harnesses, and the
control reproduces wave20's value exactly, which rules out
nondeterminism and rules out a staging error. What this session
did NOT do: a per-layer or per-logit comparison at this shape
that would prove the two winners' true logit gap is inside the
reassociation noise band. The identity and golden gates do not
cover this shape (they run the 7-token prompt). The honest
residual: at long context, the shipped defaults' greedy winner
on a periodic prompt differs from the pre-flip winner, cause
attributed to accumulation-order sensitivity on a near-tie,
with a direct logit-gap measurement at 8,202 tokens still owed
if a stronger verdict is wanted. No default was changed on the
strength of this; the flips' evidence at every gated shape
stands.

## Leg 4, f16 composition test: PASS, executed on the card

Exact command from bench/prefix-f16-guard-2026-11-07.md, run
from the staged tree:

    cargo test -p wgpu-compute --test prefix_kv_session \
      resident_session_prefix_seam_refuses_f16_storage

Output (decisive lines verbatim):

    running 1 test
    test resident_session_prefix_seam_refuses_f16_storage ... ok
    test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured;
    1 filtered out; finished in 44.38s

The test RAN (no skip line) and passed: the first real
execution of an f16 session meeting the guard. The guard
refuses capture and install by name under f16 storage, the
refusals leave the session byte-identical, and the store flow
writes nothing. Run again standalone during leg 5 triage
(--exact): ok again, 45.31 s.

## Leg 5, gate.sh on the card: RAN; 1 test failure, root-caused

bench/gate.sh, first adapter-present execution. Exit code 1.
Summary lines verbatim:

    gate: passed 749, failed 1, ignored 1; skipped 0 adapter
    (adapter present: expectation 0), 6 other (named exceptions,
    not gated)
    gate: FAIL: cargo test exited 101 with 1 failed

The Category J criterion is met on its own terms: the wrapper
enumerated the adapter, and adapter skips are ZERO; the device
tests executed on the card instead of skipping. The 6 "other"
skips are env-arm skips by design, each named in the output:
two dflash diag arms needing DIAG_OUT set; three K21F routing
pins needing their env arms (WGPU_COMPUTE_MMV_MAX_ROWS closing
value, WGPU_COMPUTE_K21F_HEAD=0, WGPU_COMPUTE_DIAG=timing); one
further named exception of the same class. None is an adapter
skip and none is a missing-file skip. shellcheck is not
installed on this server; gate.sh's header NOT RUN note stands;
nothing was installed.

The one failure: resident_session_prefix_store_round_trip, in
the same test binary as leg 4. It panicked at
crates/wgpu-compute/tests/prefix_kv_session.rs:401,
"capture and store after the cold prefill", with the f16
guard's own refusal: "the session's KV storage is f16
(WGPU_COMPUTE_KV_F16=1)". Root cause, established by controlled
reruns in this session:

- The f16 switch is resolved per backend construction from the
  process env (compute.rs, WgpuCompute::new). Inside one test
  binary the two tests run on parallel threads sharing one
  process env. The guard test sets WGPU_COMPUTE_KV_F16=1
  (prefix_kv_session.rs:698) and removes it only at its end
  (line 805), minutes later on card time. The round-trip test
  never touches the variable; a backend it constructs inside
  the guard test's f16 window resolves f16 storage, and its
  capture is then refused, correctly, by the guard.
- Serialized reruns: with --exact, each test passes alone on
  the card (guard 45.31 s; round trip passes). Together in one
  binary invocation they fail the same way as in the gate
  (reproduced twice: in gate.sh and in a direct two-test run).
- The guard itself is not implicated: it fired exactly as
  designed, on a session that really was f16. The defect is in
  the test file: process-global env choreography between two
  tests in one binary is racy under parallel test threads.

This is a test-harness defect at cb579ad, invisible on
adapterless machines (both tests skip there) and visible only
when the suite runs on a card with default parallelism. The
fix (serialize the two tests, or move the f16-window test to
its own binary) is a later lane; no code was changed here.
Consequence for the standing record: the workspace suite is
749/750 on the card at cb579ad, and the one failure is this
race, not a product defect.

## Closeout

- Tenants restored: systemctl --user start cortex-embed.service
  (active); docker start cortex-cortex-chat-1
  cortex-cortex-embed-1 (both Up, healthy). VRAM after restore
  and settle: 8,932,163,584 B. Against the standing tenants-up
  figure of 8,932,147,200 B that is +16,384 B (16 KiB), inside
  driver allocation granularity; the session-start drained
  figure matched exactly (33,271,808 B). Port 8080 still clear
  at closeout. Production Glimmer serving was never started.
- Nothing was cut. Every leg ran; leg 5's failure is reported
  above with its root cause and its controlled-rerun proof.
- Still owed after this session (named, not hidden): the
  test-binary env race fix (the leg 5 finding); a direct
  logit-gap measurement at 8,202 tokens if a stronger
  divergence verdict is wanted; the manifest "git":"unknown"
  under archive staging and the resident-manifest overwrite at
  the shared DIAG_OUT path (both pre-existing, re-observed).
