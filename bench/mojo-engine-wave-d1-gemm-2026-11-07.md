# Mojo engine Wave D, Lane D1: GEMM prefill path

Date: 2026-11-07 (windows ran 2026-11-08 UTC by the server
clock). Branch `lane/mojo-d1-gemm`, cut from mojo-max-engine
2c1802c (verified equal to the remote branch head before the
cut; remote main verified at b416987 and never moved). Work
was NOT pushed; the coordinator integrates. Engine stays 1.4
(version bumps are the coordinator's wave-close act).

## What was built

Prefill no longer runs the GEMV family once per prompt row
for its projections. New module src/kernels/gemm.mojo: one
GEMM launch per projection for the whole prompt batch
against the same resident weight buffer, dequant in-kernel
exactly as the GEMVs do (Q4_K via k1_gemv's own unpack/scale
helpers and accumulation form; Q6_K/Q5_K via
kernels/dequant's per-element helpers; F32 plain). Layouts:
X flat [batch][cols], Y flat [batch][rows]. Shapes built:
Q4_K, Q6_K, Q5_K, F32 - the GPU layer's projection dispatch
census. F16 (zero Glimmer weight tensors) refuses by name in
the batch path; single rows keep the GEMV path.

Wiring: src/model/gpu_layer_batch.mojo (gpu_layer_step_batch,
statement-for-statement gpu_layer_step over flat batches;
projections via GEMM, norms/RoPE/gate/SwiGLU batched by their
existing launches, KV appends and causal attention per
position with stored = start+i+1), GpuGlimmerModel.prefill /
prefill_batch in gpu_model.mojo, and ONE defaulted method on
the ForwardModel trait (generate/generate.mojo): the default
prefill is the Wave B/C step loop verbatim, so the CPU
adapter and the test mock are byte-identical without defining
anything; the GPU model overrides it. generate() now calls
model.prefill for the prompt phase. step() and logits() -
the decode path - are untouched. Runner threads the two new
config knobs (src/config.mojo:
MOJO_ENGINE_PREFILL_GEMM_MIN_BATCH default 2,
MOJO_ENGINE_GEMM_TILED_MIN_BATCH default 8, named refusals;
setting the first above any prompt length disables the GEMM
path, which is the A/B control used below).

## Trial ladder (correctness gate first, every iteration)

(1) MAX's own kernel library matmul: NO FIT. In MAX 26.7 the
only quantized matmul surface is the Python graph op
max.graph.ops.qmatmul (encodings Q4_K and Q6_K only - no
Q5_K; inspected live in the VM pixi env). It lives in the
MAX graph/pipeline world: it takes TensorValues in MAX's
repacked quantized layout, it QUANTIZES the lhs activations
to the rhs encoding before the product (not this engine's
f32-activation semantics and not the K1 accumulation form),
and it is not callable from Mojo kernel code - no Mojo-side
matmul module resolves in this toolchain (probed:
max.kernels.matmul, max.nn, kernels.matmul, max.kernels all
fail). Adopting it means rebuilding the forward inside MAX
graphs: a different engine, not a prefill path for this one.

(2) Straightforward batched GEMM (the GEMV workgroup body
with batch offsets, one launch per projection): CORRECT
everywhere (below) and kept as the below-tile-threshold
structure. Timing at batch 64, real shapes, interleaved
medians of per-pass ms (server proof, window 2): Q4_K
ffn_gate 19968x6656: 6.34 ms vs the per-row GEMV arm's 6.73
(1.06x). Q6_K ffn_down 6656x19968: 131.58 ms vs GEMV 86.45
(0.66x - SLOWER than per-row GEMV; recorded as measured: the
weight stream is still re-read per batch row and the caches
do not save it).

(3) Register batch-tiled GEMM (one workgroup owns one output
row - K1: the row pair - for a tile of 4 batch rows; each
weight element / Q4_K header+quads loaded and dequantized
once per workgroup and applied to all 4 rows): CORRECT
everywhere and the measured winner. Same timing pass: Q4_K
3.34 ms (2.02x vs GEMV arm, 1.90x vs (2)); Q6_K 41.16 ms
(2.10x vs GEMV, 3.20x vs (2)). Effective GB/s on the
weights-read-once basis: Q4_K 22.4, Q6_K 2.65 (the GEMV arms'
raw streamed figures: 710.8 GB/s Q4_K - K1 at its Wave A
class - and 80.7 GB/s Q6_K, the untuned straight port).
Production dispatch as proven: (3) at batch >= 8, (2) below
it. A future tuning note, recorded not actioned: (3) also
won at every smaller batch it was correctness-proven on, so
lowering the tiled threshold is a candidate change for a
later window, not something this lane changed after its
proof window.

## Correctness: per shape, per row count

Fixtures SYNTHETIC (LCG bytes, tamed f16 super-scales, the
Wave A pattern), rows 64, cols 1024 (fixture A) plus Q4_K at
cols 6656 (fixture B, the real accumulation length); batches
1, 6, 64, 256; both structures; metrics are k1_check's
(rel = max_abs_err / max(1, max|ref|), Float64 cosine).
tests/server_proof_gemm.mojo, window 2 log
~/gpu-wave-d1/out2/gemm.log read in the foreground:
80 GEMM_CHECK lines, 0 FAIL.

Gate K (the decisive one: GEMM output vs the GEMV KERNEL's
own output on the same fixture): rel 0.0 and cosine 1.0 for
every shape, structure, and batch in fixtures A and B -
the GEMM kernels are BIT-IDENTICAL to the GEMV kernels per
output row. The GEMM path is exactly as accurate as the
shipped GEMV path, no more, no less.

Gate T (vs the GEMM CPU twins, which are the GEMV CPU twins
per row; k3-class bar rel <= 1e-4, cosine >= 0.999999):
PASS everywhere. The ONLY nonzero gate-T distances in the
whole log are Q4_K fixture B (cols 6656): batch 64 rel
1.4420996624982054e-07 and batch 256 rel
1.2287723982866891e-06, both structures. Every other shape,
fixture, and batch - Q4_K fixture A, Q6_K, Q5_K, F32 - is
rel 0.0 against its CPU twin. See the rejected iteration
below for why fixture B is the interesting case.

VM side: tests/test_gemm_cpu.mojo proves the CPU twins
bit-identical to per-row GEMV twin calls at batches 1/3/6
plus grid arithmetic and named refusals; tests/
test_gpu_prefill_vm.mojo covers the batch layer's shape
validators (including the F16 batch refusal) and row
slice/place. Full tests/run_tests.sh at d7d9bef: PASS,
135 cases across 25 TestSuite files plus the seam-on diag
rerun, config probes, engine hello (0 failed, 0 skipped).

## REJECTED ITERATION, recorded with numbers

Window 1 (staged d7d9bef, log ~/gpu-wave-d1/out/gemm.log)
gated every shape vs the CPU twin at K1's absolute recorded
bar (rel <= 1.267e-07) on the synthetic fixtures and FAILED
on exactly four checks, all Q4_K fixture B (cols 6656):
batch 64 s2 and s3 rel 1.4420996624982054e-07, batch 256
s2 and s3 rel 1.2287723982866891e-06 (cosine 1.0 and
0.9999999999999992). Fixture A (cols 1024) passed at rel
0.0 at every batch, as did Q6_K, Q5_K, and F32; the proof
stopped at GEMM_CORRECTNESS FAIL and no timing section
ran. Verdict after window 2's gate K: the bar, not the
kernels, was wrong for this fixture - K1's 1.267e-07 was
recorded on the REAL fixture tensor whose output scale
(the rel denominator) dwarfs the tamed synthetic fixture's,
and gate K proves the GEMM outputs are bit-identical to
the GEMV kernel's on identical inputs, so the fixture-B
distance is the GEMV family's own kernel-vs-CPU-twin
distance at cols 6656, not a GEMM defect. The proof was
revised to the two-gate form above (commit af77035; the
only diff from d7d9bef is that test file) and window 2
passed all 80 checks with the identical fixture-B figures
under gate T. Whether K1's absolute bar should also be
re-stated for synthetic fixtures generally is David's
call, the same question Wave A left open; this lane did
not move any kernel's bar.

CORRECTION (2026-11-08, after window 2): the first version
of this note carried different window-1 figures (fixture A
1.328e-07/1.490e-07, fixture B up to 2.383e-07, an F32
4.038e-04) sourced from a backgrounded delivery of the
window-1 log. A direct foreground re-read of both on-disk
logs shows those figures never existed in either log; the
figures above are the logs' actual contents, re-read in
the foreground. Same failure class as Wave B's fabricated
deliveries; the rejected-iteration verdict is unchanged
and the fixture-B exceedance is in fact larger (about 9.7x
the bar at batch 256, not about 1.9x).

## Real-model measurement: 1,022-token prefill

tests/server_proof_prefill.mojo, window 2, interleaved arms
rotated across two rounds, fresh model per arm, same staged
tree; the ONLY arm difference is the configured GEMM
minimum batch (2 = NEW, 1,000,000 = Wave-C path). Prompt:
1,022 deterministic LCG token ids (forward measurement, not
a tokenizer test), context 4096.

  arm=new round=0 prefill_ms=74636  tok_s=13.69 argmax=302
  arm=old round=0 prefill_ms=137001 tok_s=7.46  argmax=302
  arm=old round=1 prefill_ms=136467 tok_s=7.49  argmax=302
  arm=new round=1 prefill_ms=74712  tok_s=13.68 argmax=302
  PREFILL1022_SUMMARY new_ms_min=74636 old_ms_min=136467
    speedup=1.83x  VERDICT PASS

Arm-vs-arm last hidden rows: old vs new rel 4.076e-08,
cosine 0.9999999999999537, argmax exact in both rounds
(bar class: cosine >= 0.999999, rel <= 1e-4). Load times
(21-31 s across arms) are excluded from the prefill figures
by construction and vary with host cache warmth.

Where the remaining time goes (analysis, from the numbers):
the kernel-level projection win is 2.0-2.1x, the end-to-end
win is 1.83x, and the gap is the Wave C shape this lane was
not asked to change - activations cross per op as host rows
(now flat batches: the copies are 1,022x wider per launch),
KV appends and causal attention still run per position
(52 x 1,022 attention launches with a sync each). Activation
residency is the next lever, already on the owed list since
Wave C.

## Golden gate (window 2, same staged tree)

Teacher-forced on device (tests/server_proof_gpu_generate
.mojo; its 6-token prompt prefills through the GEMM path,
batch 6 >= min 2): TEACHER_FORCED_GPU 8/8, winning logits
17.343926, 18.877588, 17.273037, 19.247173, 17.747236,
16.751627, 16.038439, 15.78144 - Wave C's recorded values.
End-to-end runner (MOJO_ENGINE_GPU=1): GOLDEN_FIRST8_MATCH 1,
TEXT_STRICT_OK 1, all 16 tokens identical to the Wave C
record [13796, 26, 1573, 373, 290, 2167, 26899, 1212, 6996,
310, 11698, 341, 290, 2167, 26899, 1212]; PREFILL_MS 402
(Wave C: 684) for the 6-token prompt; GENERATE_MS 2769,
decode approx 157.8 ms/step (Wave C: 151.1 - decode is the
untouched GEMV path; the difference is window noise class).

## Server closeout, as observed

Two windows, both under flock /tmp/itak-phase.lock (window 1
queued behind Lane D4's window, as designed), fresh staging
per window (git archive, STAGED_COMMIT stamp, staged
sha256 verified against the worktree for gemm.mojo,
gpu_layer_batch.mojo, and the revised proof file). Port
8080 clear before and after both windows; cortex-cortex-1
and cortex-cortex-surrealdb-1 Up throughout, never stopped;
Glimmer serving stayed down; llama-server (production)
untouched; no reboot by this lane. Both windows closed well
clear of the 2026-11-08 02:00-04:00 ET maintenance window.

Tenant state, plainly: cortex-embed.service was INACTIVE at
this lane's first server check (before any window) and at
both windows' entry; neither window stopped it
(EMBED_STOPPED_BY_WINDOW 0 in both logs), so there was
nothing of this lane's to restore. Its payload - the
embedding llama-server on port 8091 - is running (started
04:16 server time, outside this lane's windows) and answers
{"status":"ok"} on /health; VRAM at window-2 close measured
1,939,746,816 B (that payload), against 2,584,600,576 B with
the service fully resident in earlier waves. Restoring the
systemd unit to `active` from a non-interactive shell is
blocked (polkit: interactive authentication required -
sudo -n and plain systemctl both refused, in the window
scripts and by hand); the coordinator can reconcile the
unit state with an interactive sudo, and the maintenance
reboot will restart enabled units. Flagged here rather than
papered over: the embedding FUNCTION is up; the unit's
systemd state is not what Wave C's closeout recorded.

One evidence note: several backgrounded poll deliveries
during the windows carried figures that do not match the
on-disk logs (the Wave B failure mode). Every number in
this note was re-read in the foreground from
~/gpu-wave-d1/window.log, window2.log, out/gemm.log, and
out2/{gemm,prefill,teacher_forced,runner}.log after the
windows closed.

## Commits (lane/mojo-d1-gemm, not pushed)

79c419c GEMM kernels + CPU twins + VM test wiring
2c0bf42 GEMM server proof (first form)
8328a1c config knobs (GEMM batch thresholds)
29c3776 batched prefill path (batch layer, model prefill,
        trait default, runner threading, VM tests)
5839e08 1022-token prefill proof + wave_d_timing harness
d7d9bef mojo format pass (staged for window 1)
af77035 GEMM proof two-gate revision (staged for window 2)
plus this note.
