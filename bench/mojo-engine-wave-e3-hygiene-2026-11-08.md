# Mojo engine Wave E, Lane E3: hygiene (budget ledger, harness verdict, GLIMMER fold)

Date: 2026-11-08. Lane branch `lane/mojo-e3-hygiene`, based on
fa7221e, engine version 1.5 (unchanged: no item here is a
release feature). Worktree: `~/workspace/lanes/mojo-e3-hygiene`.
One commit per item, not pushed (the coordinator integrates):

- Item 1 (budget-ledger unreserve): 9baa030
- Item 2 (harness verdict-line extraction): aecb9f7
- Item 3 (GLIMMER_* defaults into EngineConfig): da170d8
- This record: a separate docs commit after item 3.

Baseline on the fa7221e tree (coordinator-verified, and
reproduced on this lane before any edit): full VM suite
(`~/mojo-probe/bin/pixi run ./tests/run_tests.sh` from
mojo-engine/, which includes the seam-on diag rerun, the
config probes, and the engine hello): 156 passed, 0 failed.

## Item 1: budget-ledger unreserve (owed since Wave C)

Before-state, verified on this tree. `GpuRuntime` in
`src/gpu/context.mojo` had `reserve_weights` / `reserve_kv`
(check-then-apply against the cap) and NO release path of
any kind; the Wave C note lists "A budget-ledger
release/unreserve call" under Still owed. The debt was
visible in two proofs, which reset the ledger field
directly because a destroyed cache's reservation otherwise
kept counting against the cap:

- `tests/server_proof_gpu_layer.mojo`: `rt.kv_bytes = 0`,
  with a comment naming the missing unreserve API.
- `tests/server_proof_gpu_residency.mojo`: `rt.kv_bytes = 0`
  twice per round.

What changed:

- `src/gpu/context.mojo`: pure helper
  `budget_release(current, amount, what)` (returns
  current - amount; refuses `gpu_budget_exceeded` by name on
  a negative amount or on releasing more than the bucket
  holds, so the ledger can never go negative), plus
  `GpuRuntime.unreserve_weights` / `unreserve_kv` delegating
  to it one line each, with diag lines in the existing
  `gpu` phase style ("unreserve weights -N total_weights=M").
- `src/gpu/weights.mojo`: `DeviceWeights.release_reservation(rt)`
  returns the exact total `build()` reserved; a
  `reservation_released` flag makes a double release refuse
  by name even if another build reserved in between.
- `src/gpu/device_kv.mojo`: same shape,
  `DeviceKvCache.release_reservation(rt)`.
- `src/model/gpu_model.mojo`:
  `GpuGlimmerModel.release_reservations()` - the
  model/context destroy path: releases KV, then weights,
  on the model's own runtime.
- Both server proofs now call `release_reservation` instead
  of resetting ledger fields; no direct ledger-field writes
  remain outside `context.mojo` itself.

Proof (before/after). The runtime methods need a
DeviceContext, which this device-less VM cannot construct,
so the VM proof drives the ledger state as plain ints
through the SAME pure functions the methods delegate to
(`check_gpu_budget` is the reserve methods' check;
`budget_release` is the unreserve methods' whole
arithmetic) - a clearly argued unit proof, with the
delegation one line each in the source. New tests in
`tests/test_gpu_plumbing.mojo` (an existing, already-wired
suite file):

- `test_budget_release_reserve_cycle`: at cap 1000 /
  scratch 100, reserve weights 800 + KV 50 (total 950);
  a second KV reserve of 100 REFUSES `gpu_budget_exceeded`
  (the before behavior: destroyed structure still counted);
  release both; totals return to the scratch envelope alone;
  the identical reservations then SUCCEED (950 again); a
  partial release frees exactly its amount.
- `test_budget_release_refusal_names`: over-release,
  release from an empty bucket, and a negative amount each
  refuse `gpu_budget_exceeded`; exact and zero releases are
  legal.

Plumbing suite: 9/9 (was 7). The four GPU consumers compile
on the VM with `mojo build --target-accelerator gfx1100`
(the Wave C route): server_proof_gpu_layer,
server_proof_gpu_residency, server_proof_gpu_generate, and
the runner - all exit 0. Untested leg, plainly: the release
path has not EXECUTED on the card (no server window was
expected for this lane, and none was taken); the next
cleared-card window that runs the layer or residency proof
exercises it for real, since both proofs now call it.

Suite after the commit: 158 passed, 0 failed (156 + 2 new).

## Item 2: harness verdict-line extraction (owed since Wave C)

Before-state, reproduced on this tree before editing, with
synthetic logs in /tmp (a teacher log carrying the proof's
header lines, per-step idx lines, and the verdict line;
valid GPU runner logs; a matching staged stamp):

- Good log: the harness exited 0 but reported
  `TEACHER_FORCED_VERDICT_LINE=GPU_GENERATE_PROOF context_len 4096`
  - the header line, exactly the Wave C wart.
- Verdict-less log (verdict line removed): the harness STILL
  exited 0 and reported the same header line. The extraction
  could not fail; that is the real bug behind the wart.

Root cause (as recorded in Wave C): `teacher_verdict_line`
tried a generic `GPU_(GENERATE|PROOF)` shape before the
count shapes, and the header matches that shape.

What changed (`mojo-engine/tests/wave_c_timing.sh`):

- Extraction accepts only the proof's verdict shapes, in
  order: (1) an anchored `^TEACHER_FORCED_GPU <n>/<m>` line
  (the real verdict, e.g. `TEACHER_FORCED_GPU 8/8
  total_ms=...`; anchored so idx= lines and headers can
  never match), (2) a TEACHER line carrying `pass=<n>/<m>`.
  The generic proof shape and the any-line-with-8/8 shape
  are deleted.
- The full-pass gate now also checks the verdict's bare
  `<n>/<m>` count (it previously checked only `pass=`, so a
  7/8 verdict line would have sailed through): a partial
  verdict fails `TEACHER_FORCED_NOT_FULL_PASS`.
- A verdict-less log fails loudly:
  `TEACHER_FORCED_VERDICT_LINE_MISSING`, exit 1.
- Header comment updated with the fix and the lineage note:
  `wave_d_timing.sh` was checked - it greps the exact
  `TEACHER_FORCED_GPU 8/8` string and never had this bug, so
  no change there. For Wave E the coordinator should use
  this fixed wave_c harness (it is the rich-summary one);
  if a `wave_e_timing.sh` is wanted, it should be a copy of
  THIS fixed file, not of the pre-fix shapes.

Proof: new `mojo-engine/tests/wave_c_timing_extract_test.sh`
builds synthetic logs in a temp dir and runs the harness's
`--extract-only` mode over four cases - GOOD (header +
verdict extracts the verdict verbatim), MISSING (fails by
name, nonzero exit), PARTIAL 7/8 (fails
TEACHER_FORCED_NOT_FULL_PASS), PASS_EQ (the `pass=8/8` shape
still extracts). Result: PASS, 4/4 cases. This test file is
NOT wired into `run_tests.sh` (the coordinator wires it if
wanted; `run_tests.sh` itself was not edited).

Suite after the commit: 158 passed, 0 failed, plus the
extract test 4/4 above.

## Item 3: GLIMMER_* defaults into EngineConfig (Wave A open item)

Before-state, verified on this tree by grep over
`mojo-engine/src/` and `tests/` - Wave B/C/D had NOT
partially folded any of it. Exactly 7 constants in 4 ops
files:

| Constant | File | Value | Readers on this tree |
|---|---|---|---|
| GLIMMER_RMS_EPS | src/ops/rms_norm.mojo | Float32 1.0e-5 | tests/test_ops_rms_norm.mojo only |
| GLIMMER_ROPE_BASE | src/ops/rope.mojo | Float32 500000.0 | tests/test_ops_rope.mojo only |
| GLIMMER_HEAD_DIM | src/ops/rope.mojo | 128 | none |
| GLIMMER_LOGIT_SCALE | src/ops/final_logit.mojo | Float32 0.1961161345243454 | tests/test_ops_softmax_logit.mojo only |
| GLIMMER_LOGIT_SOFTCAP | src/ops/final_logit.mojo | Float32 20.0 | tests/test_ops_softmax_logit.mojo only |
| GLIMMER_HIDDEN | src/ops/embedding.mojo | 6656 | none |
| GLIMMER_VOCAB | src/ops/embedding.mojo | 202048 | none |

Key fact about the before-state: the ops never read their
own constants in the math - eps/base/scale/cap/hidden are
function parameters - and production code reads none of the
seven. For a loaded model, the values in effect come from
the GGUF file through `ModelConfig`
(`src/model/model.mojo`: rms_eps, rope_base / rope_base_swa,
logit_scale, logit_softcap, hidden, head_dim, vocab). The
GLIMMER_* constants served only callers/tests naming the
recorded model without a file in hand.

What changed:

- `src/config.mojo`: the defaults live ONLY here now, in a
  clearly delimited "BEGIN/END Wave E Lane E3 block" grouping
  (defaults block, parsers block, struct fields, init params,
  load() section, each marked) so the merge with lane E1's
  config addition stays mechanical:
  `DEFAULT_RMS_EPS`, `DEFAULT_ROPE_BASE`,
  `DEFAULT_LOGIT_SCALE`, `DEFAULT_LOGIT_SOFTCAP`,
  `DEFAULT_HEAD_DIM`, `DEFAULT_HIDDEN`, `DEFAULT_VOCAB`
  (values byte-for-byte the old constants);
  EngineConfig fields `rms_eps`, `rope_base`, `logit_scale`,
  `logit_softcap`, `head_dim`, `hidden`, `vocab`;
  env overrides `MOJO_ENGINE_RMS_EPS`, `MOJO_ENGINE_ROPE_BASE`,
  `MOJO_ENGINE_LOGIT_SCALE`, `MOJO_ENGINE_LOGIT_SOFTCAP`,
  `MOJO_ENGINE_HEAD_DIM`, `MOJO_ENGINE_HIDDEN`,
  `MOJO_ENGINE_VOCAB`, following the existing pattern
  (getenv in load() only, parsers split out and testable).
  Named refusals are `mojo_engine_config` naming the env
  var, in the existing style; float validation mirrors the
  ops' own rules (eps > 0, rope base > 1, softcap >= 0 with
  0 = no cap, logit scale any finite value; NaN and the
  infinities PARSE as floats in Mojo, so `_parse_float`
  refuses them explicitly).
- The 7 constants are deleted from `src/ops/`
  (rms_norm, rope, final_logit, embedding); each module's
  header now points at the config home. `grep GLIMMER_` over
  the Mojo sources finds only historical comments.
- The three consuming tests now import the defaults from
  `config` (`DEFAULT_RMS_EPS`, `DEFAULT_ROPE_BASE`,
  `DEFAULT_LOGIT_SCALE` / `DEFAULT_LOGIT_SOFTCAP`).
- `tests/config_probe.mojo` prints the seven new fields, so
  the process-level probe shows them.

Deliberately NOT done, per the item's own stop rule (this
item is structure, not tuning): loaded-model execution was
NOT rewired to EngineConfig. The GGUF file stays the source
for a loaded model's values, exactly as before, so no
numeric behavior anywhere changes; the precedence is stated
in the config block header. Nothing was left half-folded:
all 7 constants moved.

Proof:

- `tests/test_config.mojo`: +3 tests -
  `test_glimmer_op_defaults_are_the_recorded_ones` (the
  defaults and a clean-env load() against the recorded
  literals), `test_parse_glimmer_op_knobs_accept` (incl.
  `1e-5` scientific form and softcap 0), and
  `test_parse_glimmer_op_knobs_refuse_by_name` (15 bad-value
  cases, each asserting the refusal names both
  `mojo_engine_config` and its env var). Config suite 11/11
  (was 8).
- Ops suites unchanged in tally and green with the config
  imports: rms_norm 4/4, rope 3/3, softmax_logit 4/4,
  embedding 2/2.
- Process level: config probe with
  `MOJO_ENGINE_RMS_EPS=0.001 MOJO_ENGINE_ROPE_BASE=10000
  MOJO_ENGINE_HEAD_DIM=64` prints rms_eps 0.001, rope_base
  10000.0, head_dim 64 (rest at defaults); with
  `MOJO_ENGINE_RMS_EPS=abc` it prints
  `REFUSAL mojo_engine_config: MOJO_ENGINE_RMS_EPS is not a
  number: 'abc'` and exits 2.

Suite after the commit: 161 passed, 0 failed (158 + 3 new).

## Tests added in this lane (for the coordinator)

Already wired (existing suite files, no run_tests.sh edit
needed): test_gpu_plumbing.mojo +2
(test_budget_release_reserve_cycle,
test_budget_release_refusal_names); test_config.mojo +3
(test_glimmer_op_defaults_are_the_recorded_ones,
test_parse_glimmer_op_knobs_accept,
test_parse_glimmer_op_knobs_refuse_by_name).

New file, NOT wired: tests/wave_c_timing_extract_test.sh
(item 2's synthetic-log proof; wire into run_tests.sh or a
harness gate if wanted - it needs only bash and the harness
itself, no mojo).

`tests/run_tests.sh` was not edited.

## Suite tallies

- Baseline fa7221e: 156 passed, 0 failed.
- After item 1 (9baa030): 158 passed, 0 failed.
- After item 2 (aecb9f7): 158 passed, 0 failed; extract
  test 4/4 separately.
- After item 3 (da170d8): 161 passed, 0 failed.

Each tally is the full `run_tests.sh` (clean env plus the
seam-on diag rerun, config probes, engine hello), run via
`~/mojo-probe/bin/pixi run`.

## Left undone, with reasons

- No server window was used or needed (per the lane brief):
  item 1's release path is compile-proven on the VM
  (gfx1100 builds, exit 0) and unit-proven through its pure
  helpers, but has not executed on the card; the layer and
  residency proofs call it now, so the next Wave E window
  runs it for real.
- Nothing else: all three items landed, each as its own
  commit, with the numeric-behavior stop rule honored by
  leaving loaded-model value sourcing exactly where it was
  (the GGUF file).
