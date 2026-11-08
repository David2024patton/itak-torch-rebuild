# Mojo engine Wave D: the speed wave (wave record)

Date: 2026-11-08. Branch `mojo-max-engine`, engine version
1.5. Base for the wave: Wave C head 2c1802c (engine 1.4,
GPU golden PASS). All four lanes cut from 2c1802c, proved
on their own trees, and integrated onto this branch; D3
and D4 were integrated by the wave coordinator, D2 arrived
as a pre-reconciled port onto the D3+D4 tree
(lane/mojo-d2d4-reconciled), and D1 was merged last by the
landing agent after the coordinator went inactive. The
per-lane records are bench/mojo-engine-wave-d1-gemm-
2026-11-07.md (D1), bench/mojo-engine-wave-d-d2-2026-11-07
.md (D2), bench/mojo-d3-gemv-ladder-2026-11-07.md (D3),
and bench/mojo-engine-wave-d-lane-d4-2026-11-07.md (D4);
this note is the wave-level summary and the integration
record. Remote main verified at b416987 before and after,
never moved.

## Lane outcomes

### D3, GEMV tuning beyond K1 (integrated first)

- Q6_K tuned to the wgpu K2 r1sg shape: 410.9 GB/s, 4.19x
  its Wave A straight port.
- Q5_K tuned to the K3 r2sg shape with a rows=4 tile
  promoted into the engine kernel: 361.9 GB/s, 10.2x port.
- F16 tuned to the segmented vec4 shape: 717.9 GB/s,
  3.98x port.
- F32: the tuned iteration was REJECTED on window-1
  numbers and the port kernel kept (recorded in the lane
  note, not rounded up).
- K1 untouched, inside its existing bar (624.1 GB/s,
  rel 1.267e-07); the bar question stays David's open call.
- Lane-level engine measures on the D3 tree: decode
  9.09 tok/s (Wave C: 6.62), 6-token prefill 563 ms
  (Wave C: 684 ms).

### D4, activation residency + device KV wrap

- Device-buffer chained layer step (gpu_layer_step_
  resident): residual stream, norms, projections,
  attention, and MLP stay on device; hostchain kept as
  the A/B baseline. Warm medians on the C2 harness
  shapes: 1.91x (layer 0) and 1.60x (layer 3) against
  the hostchain.
- Device KV window wrap: B1 ring semantics on device
  with a visible-order gather kernel; the Wave C refusal
  (gpu_kv_window_wrapped) is retired from DeviceKvCache.
  Wrap proof: 83 element-exact lines, the real 2,048
  window replayed past wrap, and synthetic generation
  0..2063 running on device.

### D2, attention occupancy (via the reconciled port)

- attention_tiled_kernel: shared-memory K/V tile staging
  (tile 32 kept), one block per (Q head, split), barriers
  per tile instead of per position; split-KV partials
  merged by attention_merge_kernel with exact logsumexp
  rescaling - no approximation, CPU twins mirror the
  float path tile by tile.
- Card trials (two windows, interleaved old-vs-new):
  stored 1,025 at the chooser's splits 3: 842.3 -> 315.1
  us (2.67x); stored 4,096 splits 8: 3,138.4 -> 486.9 us
  (6.45x); stored 4,096 NoPE splits 4: 3.64x. Correctness
  inside the Wave B bars on every fixture (cosine >=
  0.999999999, rel <= 1e-5; observed rel 4.36e-07 ..
  9.07e-06).
- Short-regime cost stated plainly: at stored <= 512 the
  single-split new path runs 0.91-0.95x of the old kernel
  per launch and the prefill sweep 0.81-0.83x; the merge
  launch is the mechanism, and per full step it is under
  half a millisecond against a ~109-151 ms step.
- Baseline finding: the retained Wave B kernel drifts
  just past its own 1e-5 rel bar at stored 4,096 NoPE
  (rel 1.09e-05, deterministic across both windows), a
  fixture size Wave B never ran. The production path no
  longer uses it; it stays as the A/B baseline, relabeled
  BASELINE in the proof output and not gated.
- Config: MOJO_ENGINE_ATTN_TILE_POSITIONS 32,
  MOJO_ENGINE_ATTN_MAX_SPLITS 8, MOJO_ENGINE_ATTN_SPLIT_
  SPAN_POSITIONS 512, named refusals; knobs thread
  EngineConfig -> runner -> GpuGlimmerModel -> the
  resident layer launch.

### D1, GEMM prefill path (integrated last)

- New src/kernels/gemm.mojo: one GEMM launch per
  projection for the whole prompt batch (Q4_K/Q6_K/Q5_K/
  F32, dequant in-kernel as the GEMVs do; F16 refuses by
  name in the batch path - zero Glimmer tensors). Two
  structures: straightforward batched (kept below the
  tile threshold) and register batch-tiled (batch >= 8;
  measured winner: Q4_K 3.34 ms vs GEMV arm 6.73 at batch
  64, Q6_K 41.16 vs 86.45).
- MAX's own qmatmul: evaluated and recorded NO FIT (MAX
  graph/pipeline world only, quantizes activations, no
  Q5_K, not callable from Mojo kernel code).
- Decisive correctness gate: GEMM output vs the GEMV
  KERNEL's own output on the same fixtures - rel 0.0,
  cosine 1.0, every shape/structure/batch: the GEMM
  kernels are BIT-IDENTICAL to the GEMV kernels per
  output row. Against the CPU twins (k3-class bar) the
  only nonzero distances are Q4_K at cols 6656 (rel
  1.44e-07 at batch 64, 1.23e-06 at batch 256), which is
  the GEMV family's own kernel-vs-twin distance at that
  accumulation length; the window-1 absolute-bar failure
  and the two-gate revision are recorded in the lane
  note, including the correction of figures that had
  come from a backgrounded delivery (the Wave B failure
  class; all lane numbers were re-read in the foreground
  from the on-disk logs).
- Real-model measurement on the lane tree: 1,022-token
  prefill interleaved, fresh model per arm, the only arm
  difference the configured GEMM minimum batch: NEW
  74,636 ms (13.69 tok/s) vs Wave-C path 136,467 ms
  (7.46 tok/s), speedup 1.83x, last-row argmax exact in
  both arms and both rounds.
- Golden on the lane tree: teacher-forced 8/8, runner
  GOLDEN_FIRST8_MATCH 1, all 16 tokens identical to the
  Wave C record, 6-token prefill 402 ms.

## Integration order, as executed

1. D3 (coordinator): lane commits, then integration
   commit 21f4018's sibling 15550f0 on this branch.
   (D4 integrated as 21f4018 immediately before it.)
2. D2: the coordinator's reconciled branch
   (lane/mojo-d2d4-reconciled, 5 commits based on
   15550f0: attention port, config knobs, resident
   attention rewire to the tiled split-KV pair,
   model/runner threading, run_tests.sh wiring) was
   fast-forwarded in whole - it IS the D2 integration
   onto the D3+D4 tree, not a raw lane merge. Full suite
   after: PASS, 147 tests (the D2 lane's split suite,
   10 cases, now wired into run_tests.sh).
3. D1: git merge of lane/mojo-d1-gemm. Five conflicted
   files, every hunk read and resolved by hand:
   - config.mojo (10 hunks): both knob sets kept
     (attention + GEMM); D1's two full-body parsers were
     folded onto config's shared _parse_positive (the
     reconciled file's shape), so the refusal strings are
     byte-identical to D1's originals and the parse code
     is not duplicated; EngineConfig carries all five
     new fields with defaults and env loads.
   - gpu_model.mojo (6 hunks): both field sets kept;
     imports take gpu_layer_step_cfg (the reconciled
     resident launch the step path needs) plus D1's batch
     imports; open() keeps D1's GEMM-threshold parameters
     and open_attn() is retained - the runner (below)
     threads both knob sets.
   - generate_main.mojo (5 hunks): both phase functions
     already carried both knob sets (each lane had merged
     the other's parameters as context); the construction
     sites now build GpuGlimmerModel directly from
     GlimmerModel.open with the full positional argument
     list (attention knobs then GEMM thresholds), and
     run() passes all five config fields.
   - test_config.mojo (2 hunks): both sides' default
     assertions kept (attention defaults + GEMM defaults
     in the same clean-env test).
   - run_tests.sh (1 hunk): the lane file lists were
     concatenated deliberately - they are two complete
     independent insertions into one list (D2's
     test_attention_split_cpu plus D3's q5k/q6k/f32 GPU
     helper suites on one side, D1's test_gemm_cpu and
     test_gpu_prefill_vm on the other); the merged list
     carries all of them.
   One semantic seam, stated plainly: D1's batch layer
   step (gpu_layer_step_batch) launches the Wave B
   attention kernel per position, as it did on its lane;
   the tiled split-KV attention serves the resident step
   path (decode and per-row steps). Wiring the batch
   step onto the tiled attention is future work, not
   something this integration improvised.
   Full suite after the merge: PASS, 156 tests, plus the
   seam-on diag rerun, config probes, and engine hello.
4. Version bump 1.4 -> 1.5 (config.mojo ENGINE_VERSION
   and the run_tests.sh hello gate together), DESIGN.md
   status updated (steps 3 and 5 owed lists marked
   landed, speed-law measured state added).

## Integrated verification (the landing windows)

The integrated tree was verified in cleared-card windows
staged from the exact pushed commits (git archive,
STAGED_COMMIT stamp, staged file hashes verified against
the branch). Three landing defects were found and fixed
as their own commits during this verification; each is
recorded here because each is a real integration seam,
not a lane defect:

1. 44ab4cf - D1's GEMM proof predated D3's kernel
   promotion: its gate-K reference launches used the
   pre-D3 geometries (Q5_K grid = rows instead of
   rows // ROWS_PER_WG=4; F32 block = the shared
   THREADS=128 instead of F32_THREADS=256; the Q5_K
   tiled GEMM launched with Q6's thread count). The
   proof read garbage rows on the integrated tree while
   every GEMM-vs-GEMV check stayed bit-identical. Proof
   launch geometry updated; engine kernels untouched.
2. c907f1f - the D1 merge put __init__'s attention
   parameters between budget_bytes and the GEMM
   thresholds, so GpuGlimmerModel.open()'s positional
   call landed prefill_gemm_min_batch=2 into
   attn_tile_positions; the first decode step refused
   attention_tiled_bad_tile. open() now passes the
   recorded attention defaults explicitly.
3. 1716407 - the D2 reconciliation's resident GEMV
   dispatch (_gemv_u8_buf) launched q5k_gemv_kernel
   with grid_dim=rows, again the pre-D3 geometry; the
   promoted kernel computes 4 rows per workgroup, so
   decode steps read only the first quarter of
   output.weight's rows and produced token 0 after the
   first generated token. Its hostchain twin already
   carried rows // Q5_ROWS_PER_WG. One-line fix.

FINAL TREE (1716407, staged and hash-verified), full
harness (tests/wave_d_timing.sh), one window:

- GEMM proof: GEMM_CORRECTNESS PASS, GEMM_PROOF PASS,
  zero FAIL lines (80 checks).
- Teacher-forced golden: TEACHER_FORCED_GPU 8/8
  (total_ms 454, against 1,215 ms on D1's lane tree -
  the D3 GEMVs and D4 residency at work in the step
  path).
- End-to-end runner, GPU path: GOLDEN_FIRST8_MATCH 1,
  TEXT_STRICT_OK 1, 16 tokens [13796, 26, 1573, 373,
  290, 2167, 26899, 1212, 6996, 310, 11698, 341, 290,
  2167, 26899, 1212], text " Paris. It is the most
  populous city in France and the most populous" -
  identical to the Wave B/C records. 6-token prefill
  422 ms (Wave C: 684 ms; D1 lane: 402 ms). Decode
  51.7 ms/step approx, 19.35 tok/s over the 16-token
  run (D3 lane measure was 9.09 tok/s on its tree; the
  integrated step path is faster still because the
  teacher-forced and runner steps now run the resident
  chain end to end).
- 1,022-token prefill on the integrated tree:
  PREFILL1022_VERDICT PASS, argmax 302 exact in every
  arm and round, cross-arm last-hidden rel 1.74e-07.
  Timings, stated with the arm meanings on THIS tree:
  the GEMM arm (min batch 2) measured 76,579 ms
  (13.35 tok/s) and the per-row arm (min batch
  1,000,000) measured 48,415 ms (21.11 tok/s), so the
  harness's speedup reads 0.63x - INVERTED against
  D1's lane result (74,636 ms vs 136,467 ms, 1.83x).
  The inversion is real and is D2+D4's doing, not a
  GEMM regression: the GEMM arm's absolute time matches
  its lane figure (76.6 s vs 74.6 s), while the per-row
  arm got 2.8x faster than on D1's tree because the
  per-row path now runs D4's resident chain and D2's
  tiled attention instead of the Wave C hostchain. At
  batch 1,022 the batched path's per-position causal
  attention over host rows is the remaining cost (the
  owed item below). Both arms produce the same last
  hidden row inside 1.74e-07; the GEMM path stays the
  default because it is the only arm whose cost falls
  as kernels improve, and its kernel-level win (2.0x at
  batch 64) is unchanged.
- Per-layer proof on the integrated tree (run in the
  first window): GPU_LAYER_PROOF PASS, layer 0 cosine
  0.9999999999985238, layer 3 cosine 0.9999999999983641,
  logits argmax 150015 exact both paths.

Server closeout, as observed: the Glimmer llama-server
(PID 27465 at window entry) was stopped for the first
window and LEFT DOWN per David's standing order; port
8080 verified CLEAR at final closeout. The embedding
llama-server (port 8091, the cortex-embed payload) was
never stopped and answers /health ok; cortex-embed.
service itself was INACTIVE on entry and remains
inactive (its payload runs outside the unit; restoring
the unit was not attempted - non-interactive start is
polkit-blocked, as D1's note recorded). Containers
cortex-cortex-1 and cortex-cortex-surrealdb-1 up
throughout, never touched. VRAM at closeout:
1,939,746,816 B (the embedding payload's footprint).

## What remains owed after Wave D

- Batched prefill still crosses activations per op as
  host rows and runs KV appends + causal attention per
  position: residency INSIDE prefill is the next lever
  (D1's own analysis; the kernel-level win is 2.0-2.1x,
  the end-to-end win 1.83x, the gap is this shape).
- The batch layer step on the tiled split-KV attention
  (see the integration seam above). This is now the
  measured prefill bottleneck: on the integrated tree
  the per-row resident path prefills 1,022 tokens in
  48.4 s against the batched path's 76.6 s, so wiring
  residency + tiled attention INTO the batch step (or
  routing long prefills through the resident step loop
  until then) is the first speed item of the next wave.
- Launch-geometry drift is a named hazard after this
  landing: three of the wave's integration defects were
  the same shape (a caller keeping a kernel's old
  grid/block geometry after another lane changed the
  kernel). Any future kernel promotion must grep every
  enqueue site for that kernel in the same commit.
- Decode's remaining gap to the reference points (Rust
  engine ~34.5 tok/s, llama Vulkan 44.59 tok/s on this
  card).
- A splits==1 direct-output variant of the tiled kernel
  would remove the merge launch in the short regime
  (D2's note; not built - not the bottleneck at recorded
  step times).
- Budget-ledger unreserve (owed since Wave C) and the
  Wave C harness verdict-line extraction shape fix.
- K1's correctness bar (rel 1.267e-07) still binds its
  tuning: faster reorders land at 1.34-1.49e-07. Whether
  the bar moves is David's call, recorded again here,
  not taken by any lane.
- The sampler family (temperature/top-k/top-p) and the
  serving seam remain the next waves, in that order.
