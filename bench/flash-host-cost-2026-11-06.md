# Flash host cost hunt: the 540 ms does not exist (2026-11-06)

Lane: lane/flash-host-cost, based on main
2c1f8c7ef3c5c634d504e687d8c4ae4075f580a4. Evidence under
review: bench/baseline-session-2026-11-06.md leg (i),
where the staged flash kernel won its kernel row by
37.7 percent at 1,022 tokens (972.9 to 605.9 ms) yet the
prefill phase wall moved the wrong way (6,383 off vs
6,558 on), implying roughly 540 ms of non-kernel time
that appears with WGPU_COMPUTE_FLASH_STAGED=1 at the
short shape. This lane's job was to find that time,
segment by segment, and either fix it or ship a
measurement-justified length gate.

**Verdict up front: there is no host cost to fix and
no length gate to ship.** Under interleaved,
cache-controlled, repeated measurement on the server
card, the staged form is faster on the wall at every
shape measured, including 1,022 tokens, and the wall
delta matches the GPU kernel delta to within noise at
every shape. The wall-minus-GPU gap segment is the
same for both forms. The 540 ms was an artifact of
the baseline session's evidence shape: one
non-interleaved pair of runs, from which a +175 ms
single-sample wall delta and a 367 ms kernel saving
were combined into a phantom host penalty. Every
structural candidate this lane was asked to eliminate
(extra submits, extra readbacks, per-call prepare
cost, pipeline compilation inside the measured window,
drain behavior) is eliminated below by measurement,
not by argument. No code changes on this lane; the
switch, its semantics, and the shipped default are
untouched. The staged form's own decision rule from
bench/flash-rework-2026-11-06.md (kernel row bar plus
same-session wall improvement at 1,022) now PASSES on
interleaved evidence; whether the default flips is
the parent's call, as it was before.

## Method

Server session staged from this lane's head into
~/gpu-wave15 (394 files, sha256 manifest verified,
0 mismatches). All card runs inside
`flock -w 14400 /tmp/itak-phase.lock`, Cortex tenants
stopped before and restored after every card block,
VRAM guard read before each block (clear card
33,271,808 B used, the driver-residue value), every
run wrapped in bench/failure-bundle.sh (all 15
wrapped runs PASS in the staging tree's
bench/failures/INDEX.md, slugs fh-*; raw logs on the
server in ~/gpu-wave15/logs15/, per-kernel summaries
committed with this note under
bench/results/flash-host-cost-20261107/).

Harness: prefill_profile, fox prompt ("The quick
brown fox jumps over the lazy dog. " x102 = 1,022
tokens; x205 = 2,052; x410 = 4,102; x820 = 8,202),
slices of 64 rows. Arms at 1,022 were interleaved
off, on, on, off with diag timing on, plus a
timing-off pair (the wall-of-record surface), plus a
cold-cache pair in which each form got its own fresh
MESA_SHADER_CACHE_DIR so first-in-process pipeline
compilation is priced and located. Mid and long
shapes ran as off/on pairs on the warm shared cache.

## Segmentation at 1,022 tokens (diag timing on)

Kernel rows reproduce the baseline session almost
exactly: shipped flash 966.4 to 974.0 ms across the
three off arms; staged flash 605.0 to 606.6 ms across
the three on arms (baseline read 972.9 / 605.9).

| arm | cache | prefill wall | GPU total | gap (wall - GPU) |
|---|---|---|---|---|
| off-cold | fresh | 6,454 | 6,252.5 | 201.5 |
| on-cold | fresh | 6,175 | 5,950.5 | 224.5 |
| off-w1 | fresh | 6,500 | 6,297.4 | 202.6 |
| on-w1 | warm | 6,137 | 5,910.1 | 226.9 |
| on-w2 | warm | 6,125 | 5,911.9 | 213.1 |
| off-w2 | warm | 6,539 | 6,316.2 | 222.8 |

The gap segment (host encode, submits, per-slice
drains and readbacks: everything in the wall that is
not a timed kernel) reads 201.5 to 226.9 ms in BOTH
forms, with no arm-locked offset; the largest
cross-form difference between any two arms is inside
the within-form scatter (off arms alone span 201.5
to 222.8). The wall delta (on faster by 279, 363,
and 414 ms in the three pairings) matches the GPU
total delta (302, 387, 404 ms) to within that same
scatter. The kernel saving passes through to the
wall 1:1. There is no segment left for 540 ms to
hide in.

Timing-off pair (wall of record): off 6,342 ms, on
6,025 ms. The staged form wins the 1,022 wall clause
on the shipped surface too, by 317 ms.

Per-segment elimination of the named candidates:

- Submissions and readbacks: session_stats are
  IDENTICAL in every arm pair at every shape
  (1,022: forwards 16, chunks 16, submissions 16,
  aux_submissions 3, readbacks 16, dispatches 10,448;
  same identity at 2,052, 4,102, and 8,202). The
  staged form submits nothing extra, syncs nothing
  extra, reads back nothing extra.
- Per-call prepare cost: the two prepares share one
  body (prepare_flash_form in k6_attention.rs) and
  differ only in pipeline, diag name, and grid; the
  per-slice times show it. Slice 0 reads 395 to 426
  ms in all twelve 1,022 arms regardless of form;
  steady-state slice means are 382 to 385 ms (on)
  against 403 to 408 ms (off), the kernel saving
  visible slice by slice, with no first-slice or
  first-forward spike in any on arm.
- Pipeline compilation placement: a fresh shader
  cache costs about 1,080 ms, and it lands in
  phase=backend_open (1,495 and 1,484 ms cold vs 406
  to 415 ms warm), BEFORE the prefill window, for
  both forms equally (cold off 1,495 = cold on
  1,495). The staged shader adds no measurable
  compile delta there, and slice 0 of the cold on
  arm (414 ms) shows no deferred compile arriving
  inside the window either. Compilation is real host
  cost, but it is construction cost, it is
  form-neutral, and it never enters the phase wall.
- Session construction: phase=session_build ranges
  9,505 to 12,326 ms across all arms with the forms
  fully interleaved in that range (weight upload
  noise); no form offset.
- Drain behavior: the harness drains the diag
  recorder after every slice in both forms; drained
  patterns and the gap segment above already include
  it, identically.

## Mid and long shapes (diag timing on)

| shape | form | wall | GPU total | gap | flash row |
|---|---|---|---|---|---|
| 2,052 | off | 15,227 | 14,833.1 | 393.9 | 4,184.0 |
| 2,052 | on | 13,500 | 13,121.2 | 378.8 | 2,462.9 |
| 4,102 | off | 36,945 | 36,217.7 | 727.3 | 14,825.2 |
| 4,102 | on | 29,885 | 29,162.0 | 723.0 | 7,944.7 |
| 8,202 | off | 93,402 | 91,956.3 | 1,445.7 | 47,300.1 |
| 8,202 | on | 66,546 | 65,182.6 | 1,363.4 | 22,816.0 |

The long-context win does not regress; it
reproduces the baseline session's step 3 almost to
the digit (their wall 93,509 to 66,704, a 28.7
percent improvement; here 93,402 to 66,546, 28.8
percent; flash row 47,360.9 to 22,820.3 there,
47,300.1 to 22,816.0 here). The wall delta tracks
the kernel delta at every shape (2,052: 1,727 wall
vs 1,712 GPU; 4,102: 7,060 vs 7,056; 8,202: 26,856
vs 26,774), and the gap segment stays form-neutral
at every length. A length gate at any threshold up
to 8,202 would delete measured wall wins of 317 ms
(1,022, timing-off), 1,727 ms (2,052), and 7,060 ms
(4,102) to defend against a cost that does not
measure. That is why this lane ships neither a fix
nor a gate: both branches of its orders were
conditioned on a mechanism the measurements refute.

## Correctness re-proof (this lane, on the card)

- fused_kernels flash tests, bundle fh-flash-
  exactness PASS: k6_flash_matches_two_pass and
  k6_flash_staged_matches_shipped_at_glimmer_shapes
  both ok, executed on the 7900 XTX (3.84 s of real
  device time, not a skip).
- resident_session suite with
  WGPU_COMPUTE_FLASH_STAGED=1, bundle
  fh-resident-staged-on PASS: 7 passed, 0 failed,
  executed on the card.
- wgpu-compute lib flash tests with the switch on,
  bundle fh-k6-flash-tests PASS: 16 passed.
- Output agreement across every prefill arm: the
  harness's prefill_argmax is identical between
  forms at every shape, id and logit bit-for-bit
  (1,022: id 7235, logit 12.233533 in all six arms;
  2,052: 7235, 11.904436; 4,102: 29, 10.458983;
  8,202: 7235, 8.854626).
- Identity and golden with the switch on were
  proven in the baseline session on this same code
  (leg (i) step 1: spec IDENTITY and GOLDEN pass at
  depths 1 and 6); nothing in the flash path changed
  since.

## What this does and does not settle

Settled: the 1,022 wall clause of the staged form's
decision rule passes on interleaved evidence; the
540 ms non-kernel penalty is not a property of the
code, the switch, or the shape. Not settled, and not
claimed: why the baseline session's single
on-sample read 6,558 ms. Its own note records a host
under parallel CPU collection with wide scatter on
other surfaces that day, and single samples carry
that scatter whole; this lane's six interleaved
1,022 arms bound the form effect at every segment
instead. If a future session ever reproduces a real
wall regression for the staged form, the
segmentation table above is the checklist to rerun:
gap segment first, then stats identity, then slice 0
against the cold-cache backend_open price.

## Session hygiene

Tenants (cortex-cortex-chat-1, cortex-cortex-embed-1,
cortex-embed.service) were stopped before each card
block and restored after each; final state verified:
both containers Up (healthy), service active, VRAM
settled at 8,932,147,200 B (about 8.9 GB, the normal
tenant footprint) against the 33,271,808 B
clear-card guard during measurement. Production
llama-server on localhost:8080 was never touched.
No code commits on this lane: the fix branch and the
gate branch of the orders both required a mechanism,
and the measurements found none.
