# Mojo engine: runner-vs-wall decode gap audit (2026-11-08)

Question (David's zero-trust sheet, 2026-11-08): Wave E
recorded two decode speeds for the same engine on the same
model: a WALL measure of step+logits = 32,491 us (30.78
tok/s) and the runner's own approx of 49,130 us per step
(20.35 tok/s). The 16,639 us difference per token (34% of
the runner step) was unexplained. Was the wall figure
missing a trailing device sync the runner pays (a), was
the runner burning the gap in host-side work (b), or a
split (c)? Neither figure counted as the true baseline
until this audit answered.

Answer: NEITHER (a) NOR (b). The gap is a measurement
artifact inside the runner's approx construction itself,
proven same-binary below (section 3). The wall figure was
never missing its sync (section 2). Per-step host work
inside the decode loop is real but small and already
inside both measures (section 4). Verdict class: the
sheet's (b) named the right side of the comparison (the
runner's number, not the wall's, is the wrong one) but
the wrong mechanism: no per-step host work accounts for
the gap; a one-time cost inside the runner's timed
window does.

## 1. Method and staging

Tree: mojo-max-engine at e08546b0b63888c6e60fb1aff94f605
0657071b3 (engine 1.07), staged by git archive into
~/gpu-gap-audit on ubuntu-server, STAGED_COMMIT stamp in
place, sha256 of gpu_model.mojo, generate.mojo,
gpu_layer.mojo, config.mojo verified identical between
the integration worktree and the server staging before
anything ran. Measurement builds lived ONLY in staging
(tests/server_proof_gap_audit.mojo,
tests/server_proof_gap_runner.mojo) and are not in the
repo. One card window under the phase lock; two runs of
every binary; every number below is a direct foreground
read of the window's log files.

Arms:
- STOCK: the production runner binary
  (src/runner/generate_main.mojo), two runs.
- A1: the stock runner construction inside the audit
  binary: generate() over a fresh GpuGlimmerModel, then
  prefill timed on a second fresh model (forward_rows),
  approx = (generate_ms - prefill_ms)/15, exactly the
  shipped formula.
- A2: the same stock generate() call, but the prefill
  subtracted is prefill() timed on a fresh model of the
  SAME process BEFORE generate() runs (same value
  class: 210/215 ms vs A1's 209/212), generate() itself
  on a third fresh model. A1 and A2 ran back to back in
  the same binary, same window.
- B: the Wave E wall method (E2's phase_wall shape):
  prompt stepped per-row, 3 warmup + 10 measured
  free-running steps, step and logits timed apart.
- C: direct instrumented decode on one model: batched
  prefill() timed (203/202 ms), two generate-mirror warm
  steps, then 10 measured steps of model.step timed
  directly (the step ends in a host-blocking download,
  so its span is the device-complete cost), model.logits
  timed, and the host argmax timed SEPARATELY from the
  logits call.
- GAPRUNNER: the production runner with ONE added step:
  PREFILL_MS_TRUE (a fresh model's forward_rows timed in
  the same process BEFORE generate()) and the approx
  formula re-applied with it. Two runs.

## 2. The wall figure was never missing its sync

model.step() cannot return early: every layer ends in
download_f32 of the hidden row (a host-blocking copy)
and the append inside the layer synchronizes the queue;
the step's wall span therefore includes all device work
by construction. Arm C timed the step directly, inside
a real decode loop: median 30,737 / 30,835 us. Arm B
(the wall method, same window) read step 31,191 /
31,493 us and logits 1,450 / 1,471 us, step+logits
32,641 / 32,965 us = 30.64 / 30.33 tok/s against Wave
E's 32,491 us / 30.78. Arm C's full loop cost per token
(step + logits + the separately timed host argmax):
32,368 / 32,523 us = 30.89 / 30.75 tok/s. Three
independent constructions agree within noise. The wall
figure stands.

## 3. Where the 16,639 us actually is: inside the
##    runner's generate window, not in any step

Same binary, same window, back to back:

| arm | generate_ms | prefill_ms | per step |
| --- | --- | --- | --- |
| A1 run 1 | 957 | 209 | 49.87 ms (20.05 tok/s) |
| A2 run 1 | 706 | 210 | 33.07 ms (30.25 tok/s) |
| A1 run 2 | 967 | 212 | 50.33 ms (19.87 tok/s) |
| A2 run 2 | 705 | 215 | 32.67 ms (30.62 tok/s) |

The ONLY difference between A1 and A2 is whether a full
model build + prefill ran earlier in the same process.
A1's generate() is the FIRST model use in its process;
A2's generate() runs after two. The generate window
itself differs by 251 / 262 ms (16.7 / 17.5 ms per
nominal step), reproducing the Wave E gap (16,639 us)
almost exactly, with zero change to any decode step.

Corroboration from the runner runs: the stock approx in
this window read 45.13 ms/step and 81.33 ms/step in two
consecutive runs of the SAME binary (generate 891 ms and
1,435 ms; the second run's extra ~544 ms is the same
one-time class landing inside the window a second time,
cause not isolated). The GAPRUNNER's two generate
windows read 716 / 710 ms (per-step totals consistent
with arm C), while its stock PREFILL_MS (measured after
generate, on the process's second model) read 215 / 214
and its PREFILL_MS_TRUE (measured before generate, on
the process's FIRST model) read 463 / 446: the first
model use in a process carries a one-time cost of
roughly +230-250 ms wherever it lands, and the approx
formula books it against decode whenever it lands inside
the generate window, which is exactly where it landed
in Wave E's final window (49.13 ms/step) and in both
audit A1 runs (49.87 / 50.33, bracketing Wave E's
figure).

Mechanism, stated at the level the evidence supports:
a one-time first-use cost of ~200-260 ms attaches to a
process's first full model run (first GEMM-class
launches and first allocations at production shapes).
Wave E's lane note had already spotted its signature (a
one-time ~200 ms stall on the first logits after GEMM
prefill) without attributing the runner gap to it; the
teacher-forced totals show the same shape (450 ms
against ~245 ms accounted). Identifying the precise
first-touch item (kernel module load vs allocator arena
growth vs first-touch page cost) was not needed to
attribute the gap and is NOT claimed here.

## 4. The per-step host work that IS real (inside both
##    measures, not the gap)

- Host argmax: greedy_argmax over the 202,048-wide
  logits row, timed on a live row: median 207 / 261 us
  per call. It is INSIDE the runner's generate window
  and inside arm C's loop figure, and OUTSIDE the wall
  method's step+logits figure (E2 timed
  step-then-logits; its loop also computed an argmax
  per iteration, untimed): the wall figure overstates
  the true loop rate by this ~0.2 ms, ~0.7%: 30.78
  becomes ~30.6 honest. (An inline re-scan of the same
  row printed 0 us at ms-floor precision; the
  sampler's own figure is the one quoted. The scan
  itself is ~0.2 ms of the ~1.4-1.5 ms logits span:
  the rest is the head chain, norm + Q5_K GEMV +
  final_logit + the 808 KB logits download, measured
  by E2 at 1,414-1,574 us.)
- Per-token log flushing: ZERO in the measured runs.
  The diag seam was off; log_debug still builds its
  message string and consults the env var per call
  (~55 calls per step across layers, step, and
  sampler), which is inside every figure above and
  bounded by them: arm C's directly timed step agrees
  with arm B, so the whole class is inside the noise
  band between 30,737 and 31,493 us, well under 1 ms.
- Host allocations per step (~27 buffers/layer) and
  the per-layer host boundary: real, already charged
  to BOTH measures (the wall step contains them; E2's
  profile priced them), and they are the Wave F1-A
  targets. They explain why the true rate is ~30.6
  and not higher; they do not explain any part of
  the 16,639 us.

Attribution of the 16,639 us, in one line: ~16,600 us
one-time first-use cost amortized over 15 steps by the
approx formula (section 3); ~200 us of it is the real
per-step host argmax the wall figure omits (section 4)
- the two overlap in sign only; the measured gap is
the first item.

## 5. Golden guard

Every runner run (stock x2, GAPRUNNER x2) printed the
exact golden 16 tokens with GOLDEN_FIRST8_MATCH 1 and
TEXT_STRICT_OK 1: [13796, 26, 1573, 373, 290, 2167,
26899, 1212, ...], identical to the Wave B/C/D/E
record.

The audit proof's own arms fed the tokenizer-exact
prompt ids as literal data ([200000, 13796, 26, 1573,
373, 290]) and produced the sequence [7963, 323, 11698,
341, 290, 2167, 26899, 1212] from all three arms: that
is the golden stream CONTINUED from position 6 (the
prompt fed as data was the golden stream's own first
ids, so greedy decode continues it: 7963 = " of",
323 = " the", 11698 = " city"...). The arms are
token-identical to each other and decode the same
continuation the runner's own stream shows from its
position 8 onward in shifted form (290, 2167, 26899,
1212 recur). Timings were therefore taken on the
production path with the production token stream, and
the guard's purpose - prove no arm measured a
different path - is met by the runner runs' exact
match plus the arms' mutual identity. Recorded
plainly because the audit arms' literal first-8 does
not equal the golden first-8: the input construction
differed (stream-as-prompt), not the engine path.

## 6. Hardware witness (Trap 1)

Background sampler over /sys/class/drm/card2 +
hwmon6 at ~100 ms cadence for the whole window:
3,276 samples. Peak VRAM used 23,412,744,192 B
(21.8 GiB, above the 15.5 GiB bar) while Glimmer was
loaded. gpu_busy_percent took 82 distinct values,
0-100, not flat. Power swung from 3 W idle to a 400 W
peak reading. Verdict:
GPU execution VERIFIED, not UNVERIFIED_GPU_EXECUTION.

## 7. The honest baseline

DECODE BASELINE OF RECORD: ~30.6 tok/s steady state
(short context, greedy, this tree): the wall method's
32.5-33.0 ms step+logits, plus the ~0.21 ms host
argmax per token the wall figure omits. The runner
approx (20.35 tok/s in Wave E; 19.87-22.16 in this
window's stock-class runs, swinging to 12.30 when the
one-time cost lands twice) is NOT a decode measure and
should not be quoted as one: it divides a window
containing a one-time ~200-260 ms first-use cost by
the step count. Wave F measures against ~30.6.

## 8. Closeout and owed follow-ups

Server closeout, direct reads at window end and again
after: cortex-embed.service stopped for the window
(VRAM drained to 33,271,808 B at entry) and restored
active, /health {"status":"ok"}; containers
cortex-cortex-1 and cortex-cortex-surrealdb-1 up
throughout, never touched; ports 8080 and 18767
clear at entry and closeout; production Glimmer never
started; no reboot. Staging ~/gpu-gap-audit left in
place with its logs (window.log, witness.log, the six
run logs, three build logs).

Proposals (NOT in this commit; the note is the only
change):
1. Runner approx fix: warm the chain before timing
   (one throwaway generate or a pre-timed first model
   use in-process), or time decode steps directly in
   the runner the way arm C does. Until then, quote
   the wall-class figure for decode.
2. First-use cost hunt: ~200-260 ms on a process's
   first model run is worth one profile pass of its
   own; it taxes every short benchmark and every cold
   serve start.
3. The host argmax (~0.21 ms/token) and the logits
   download are the honest small change after F1-A's
   structural items; a device-side argmax would
   retire both.
