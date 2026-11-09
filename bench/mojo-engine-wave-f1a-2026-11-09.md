# Mojo engine Wave F1-A: residency, no-sync KV append,
# scratch pooling, RoPE sharing (integration + card window)

Date: 2026-11-09. Branch `mojo-max-engine`, engine version
1.07. The F1-A lane (lane/mojo-f1a, four commits over base
e08546b0b) was integrated onto the branch head 202eca870
(engine 1.07 + test bench + gap audit note + no-nightly
note) one feature per push, full suite after each landing,
then measured in one combined card window against the
pre-F1-A tree (202eca870) staged side by side. Remote main
verified at b416987a736b0a63c9f7815ec4b4fbc84e4dd2dd
before and after, never moved.

## The four items (Wave E owed list)

1. Step-level activation residency: the decode step's
   hidden row stays on device from the embedding to the
   logits download (before: upload + download + sync at
   every one of the 52 layer boundaries, plus a separate
   upload into the head). step() is now the resident step
   plus the one hidden-row download; generate_device runs
   the resident seam end to end.
2. No-sync KV append: DeviceKvCache.append_device no
   longer ends in rt.ctx.synchronize(); the sync capped
   host lookahead at half a layer.
3. Scratch-arena pooling: a GpuScratch pool on the
   runtime interns device buffers by (dtype, length);
   the resident chain draws its scratch from the pool
   instead of allocating per call.
4. Shared RoPE position buffers: the step's RoPE position
   buffers are built once on the runtime and shared
   across layers (a fixed 2,048-position table; positions
   at or beyond it refuse by name, gpu_rope_position_cap).

## Landings (integrated SHA = pushed SHA, replay-verified)

| feature | lane commit | integrated | suite after |
| --- | --- | --- | --- |
| 1: items 4+3 mechanism (RoPE sharing + scratch pool) | 4f04082 | ddf45ddeb3c244e6a76b1b20f5399f8b4d5e6dfa | 200 passed, 0 failed, 0 skipped |
| 2: item 2 (no-sync KV append) | 633430c | 0038a310459127d2a1e820aa7401f2f3960616c1 | 200 passed, 0 failed, 0 skipped |
| 3: item 3 wiring (resident chain draws from the pool) | 9e09551 | e9b1690c3382f58b4cc0d6c436f4f627893d4546 | 200 passed, 0 failed, 0 skipped |
| 4: item 1 (activation residency) | 05b6139 | 6ba5754ab4a9ce10c3892267bdd029daa926afbb | 200 passed, 0 failed, 0 skipped |

Suite = tests/run_tests.sh on the VM (pixi, clean env,
seam off + seam on + config probes), hello
"mojo-engine 1.07" at every landing. All four cherry-picks
were conflict-free. The cumulative diff e08546b..6ba5754
is 14 files, all inside mojo-engine/src and
mojo-engine/tests, identical in shape to the lane's own
diff (the two docs-only branch commits it crosses,
56e39e2 and 202eca87, touch bench/ and docs/research/
only).

Suite-count note, stated plainly: the totals summed from
the per-file Summary lines read 200 already at feature 1,
where the pre-integration branch read 197. The +3 comes
from the lane's new test files (the scratch-pool suite
among them) arriving with the first landing; features 2-4
changed behavior the existing suites already cover and
added no new test files. No count dropped at any landing.

## Card window

Staging: git archive of 6ba5754 (tree-f1a) and of
202eca870 (tree-pre) into ~/gpu-wave-f1a on ubuntu-server,
STAGED_COMMIT stamps in place, sha256 of
src/model/gpu_model.mojo verified between the VM staging
and the server staging for both trees (929e7e8f... for
F1-A, 08d08a9a... for pre). One phase lock
(/tmp/itak-phase.lock) held across the whole session,
which ran as an initial pass plus two short re-entry
passes under the same discipline (see "measurement
harness iterations" below). cortex-embed.service was
stopped inside each pass and restored after; both Cortex
containers were never touched; production Glimmer never
started; ports 8080 and 18767 clear at every closeout;
no reboot.

### Hardware witness (Trap 1): PASS

Background sampler over /sys/class/drm/card2 +
/sys/class/hwmon/hwmon6 at ~100 ms cadence for every
pass: 5,211 samples total across the three sidecar logs
(witness.log, witness3.log, witness4.log). Peak VRAM
used 23,412,756,480 B (21.8 GiB, above the 15.5 GiB bar)
with Glimmer loaded, in every pass. gpu_busy_percent
took 47-68 distinct values per pass, 0-100, not flat.
Power column: hwmon6/power1_input read -1 for the whole
session on this host (the file yields no reading to this
reader; recorded as a host fact, not a pass input).
Verdict: GPU execution VERIFIED, not
UNVERIFIED_GPU_EXECUTION. (VRAM at window entry, with
the tenant stopped, matched the drained baseline class
in every pass.)

### Correctness gate: PASS, strict

- Teacher-forced on device (server_proof_gpu_generate,
  first pass): TEACHER_FORCED_GPU 8/8 on BOTH trees
  (total_ms 447 F1-A, 433 pre).
- Runner golden, production generate_main on both trees:
  GOLDEN_FIRST8_MATCH 1, TEXT_STRICT_OK 1, tokens
  [13796, 26, 1573, 373, 290, 2167, 26899, 1212, 6996,
  310, 11698, 341, 290, 2167, 26899, 1212] - identical
  on both trees, identical to the Waves B-E record.
- Poison discipline: exercised inside the timed proof
  itself (below) with the production checker
  generate.sampler.golden_first_match: the sequence with
  the 8th token flipped 1212 -> 1213 reads match 0 in
  both directions (flipped expected, flipped got) on
  both trees, while the true sequence reads 1. The same
  checker is the one the suite's test_poison_gates
  exercises (200/200 above).

### Speed: the audit method, both trees, same window

Timed proof (staging-only, never committed;
server_proof_f1a_step_timing_v3): the prompt built
exactly as the runner builds it (BOS prepended by the
caller, then encode("The capital of France is")), a WARM
generate on a throwaway model first (absorbs the
one-time first-use cost the gap audit identified), then
a fresh model timed directly: prefill timed in-process
(F1-A 170 ms, pre 200 ms), two warm steps, then 10
measured loop iterations, step / logits / host argmax
timed apart; loop = step + logits + argmax. The timed
phase's own tokens satisfied the golden checker on both
trees (printed in the logs), so the timings were taken
on the production token stream.

| tree | step median | logits median | argmax median | loop median | tok/s |
| --- | --- | --- | --- | --- | --- |
| pre-F1-A (202eca870) | 30,813 us | 1,454 us | 238 us | 32,427 us | 30.84 |
| F1-A (6ba5754) | 25,920 us | 1,556 us | 199 us | 27,569 us | 36.27 |

Delta: loop -4,858 us per token (-15.0%), decode
30.84 -> 36.27 tok/s (+17.6%) measured in-window. The
whole gain is in the step (-4,893 us): the 52 per-layer
host crossings are gone. Logits is a wash (1,454 vs
1,556 us, inside run-to-run noise), argmax unchanged in
class (~0.2 ms, the audit's figure).

Cross-check, second construction (the lane's updated
server_proof_decode_profile, phase WALL: per-row stepped
prompt, 3 warmup + 10 measured free-running steps,
step and logits medians): pre step+logits 32,505 us
(30.76 tok/s) vs F1-A 28,311 us (35.32 tok/s) - same
ordering, same class of delta. The pre tree's two
constructions (32,427 / 32,505 us) bracket the gap
audit's arm-C full-loop figure for this code line
(32,368-32,523 us, ~30.6-30.9 tok/s): the baseline of
record ~30.6 tok/s stands confirmed, and F1-A moves it
to ~35.3-36.3 tok/s depending on construction.

### The lane's enqueue-only KV append series

From the same decode-profile runs (phase OPS, production
DeviceKvCache.append_device, 50 reps, median): F1-A
"kv_append_device_pair_enqueue" 2,891 ns enqueue-only
(the proof follows with its own sync before reading
timings back; the series measures enqueue cost, not a
round trip) vs pre "kv_append_device_pair_incl_sync"
20,988 ns including the internal synchronize the item
removed. The append's host-side cost per call fell by
~18 us; across 104 appends per token (52 layers x K+V)
that is the mechanism behind a share of the step gain,
with activation residency carrying the rest.

## Measurement harness iterations (recorded, per the
## evidence law)

The headline numbers come from the v3 timed proof. Two
earlier harness shapes failed inside the window and are
recorded so nobody re-derives them:
- v1 measured the RESIDENT loop (logits_device /
  step_device) on the F1-A tree: loop median 25,599 us
  (39.06 tok/s), step enqueue spans ~0 by construction
  (the step's device work is charged to the following
  logits download, the only host crossing). It could not
  run on the pre tree at all: the resident seam IS item
  1, so no same-loop pre measurement exists. Useful as
  an F1-A-only figure (the serve path's loop shape), not
  as the delta.
- v1 and v2 both fed a hand-built continuation prompt
  ([200000, 13796, 26, ...], the teacher-forced walk's
  data prompt) whose greedy continuation is NOT the
  golden stream; their golden checks read 0 on both
  trees while the production runner read 1 on both. The
  prompt construction, not the engine, was wrong: the
  runner prepends the tokenizer's BOS id (200000 is the
  PAD id in this vocab) and encodes the text. v3 builds
  the prompt the runner's way and both trees pass.
- Harness syntax note: in this Mojo, a conditional
  expression bound by `var x = 1 if f(...) else 0` fails
  to parse when f takes list arguments by value; the
  String(1 if ... else 0) argument form (the codebase's
  own idiom) parses. Cost one compile cycle to find.

## What remains unproven (stated plainly)

- Items 1 (activation residency) and 2 (no-sync KV
  append) have no dedicated VM unit tests: there is no
  device on the VM to test them against, and the lane
  added none. Their proof is this window: teacher-forced
  8/8, runner golden exact, the timed proof's own golden
  check, and the measured step delta, all on the
  integrated tree. The VM suite's device-adjacent files
  exercise the host surfaces only.
- The resident loop figure (25,599 us, 39.06 tok/s) is
  measured on the F1-A tree only; the serve path's
  end-to-end rate through the HTTP endpoint was not
  re-measured in this window.
- The step-time lists drift upward inside each 10-step
  run (both trees; e.g. F1-A loop 27,0xx early to 28,8xx
  late). Medians are quoted; the drift's cause (thermal
  or clock behavior over the run) was not isolated.
- Prefill at production shapes (1,022 tokens) was not
  re-measured; F1-A touches the decode step, and the
  short-prompt prefill figures above (170 vs 200 ms)
  are in-process timings of a 7-token prompt, not a
  prefill benchmark.

## Closeout

cortex-embed.service active after every pass, /health
{"status":"ok"} (it serves on port 8091); containers
cortex-cortex-1 and cortex-cortex-surrealdb-1 up
throughout, never touched; ports 8080 and 18767 clear;
production Glimmer never started; no reboot. Staging
~/gpu-wave-f1a left in place with all logs (window,
rerun and rerun2/3 statuses, three witness sidecars,
per-tree teacher / timing v1-v3 / profile / runner
logs, four closeout files).
