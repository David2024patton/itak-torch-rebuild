# Mojo engine Wave E: batch-step residency, decode launch
overhead, hygiene debts (wave record)

Date: 2026-11-08. Branch `mojo-max-engine`, engine version
1.6. Base for the wave: Wave D head fa7221e (engine 1.5,
golden PASS, plus the repo-record fix pass). Three lanes
cut from fa7221e, proved on their own trees, integrated
by the coordinator one at a time in the order E3, E1, E2;
the golden gate ran on the final integrated tree
(2f1c091). The per-lane records are
bench/mojo-engine-wave-e1-batch-2026-11-08.md (E1),
bench/mojo-engine-wave-e2-decode-2026-11-08.md (E2), and
bench/mojo-engine-wave-e3-hygiene-2026-11-08.md (E3);
this note is the wave-level summary and the integration
record. Remote main verified at b416987 before and
after, never moved.

## Lane outcomes

### E1, residency + tiled attention in the batch step

The headline fix. Wave D closed with the 1,022-token
prefill INVERTED: the GEMM batch arm (76,579 ms) lost to
the per-row arm (48,415 ms) because the batch step still
crossed activations per op as host rows and ran the
Wave B attention kernel per position, while the per-row
path had gained D4 residency and D2 tiled attention.

- gpu_layer_batch.mojo rewritten: gpu_layer_step_batch_cfg
  keeps activations resident on device across the layer
  (one upload, one download, residual adds on device,
  KV appends device-to-device through a scatter kernel
  driven by the cache's own slot logic), and attention
  is a new batched twin of D2's pair
  (attention_batch_tiled_kernel +
  attention_batch_merge_kernel): one tiled + one merge
  launch per layer for the whole batch, exact logsumexp
  merge, each element's splits partition its own causal
  stored count so causality matches the per-position
  steps exactly. Window-wrap refusals kept by name; a
  CPU twin (attention_batch_causal_cpu) carries the VM
  proofs.
- Per-layer proofs on the lane tree AND re-run on the
  integrated tree in the final window, batch vs per-row
  resident, bar cosine >= 0.999999999 / rel <= 1e-5:
  layer 0 @6 rel 2.49e-07, layer 0 @64 rel 4.76e-07,
  layer 3 @6 rel 1.65e-07, layer 3 @64 rel 3.42e-07,
  stored counts agreeing on both arms. PASS all four,
  digit-identical between the lane tree and the final
  tree.
- Routing, measured on the real model (interleaved
  fresh-model arms): batch loses at 6 tokens (286 vs
  218 ms on the lane tree) and wins at every measured
  length from 16 up (16: 1.53x lane / 1.38x final;
  32: 1.60x / 1.45x; 64: 1.46x / 1.34x; 128: 1.26x /
  1.15x; 256: 1.30x / 1.19x; 512: 1.37x / 1.26x;
  1,022: 1.39x / 1.29x). RULE: per-row below 16 tokens,
  batch at/above 16, as the config knob
  MOJO_ENGINE_PREFILL_BATCH_MIN_TOKENS (field
  prefill_batch_min_tokens, default 16), composed with
  PREFILL_GEMM_MIN_BATCH so the Wave D A/B control
  still forces the per-row arm alone.
- 1,022-token arms on the integrated tree (final
  window): batch 34,852 ms (29.3 tok/s) vs per-row
  45,042 ms (22.7 tok/s), speedup 1.29x, argmax exact.
  Against Wave D: batch arm 76,579 -> 34,852 ms
  (2.20x), per-row arm 48,415 -> 45,042 ms. The
  inversion is gone on the integrated tree, not only
  on the lane tree (lane arms were 34,786 / 48,475).
- Flagged discrepancy (recorded, not hidden): the
  production batch step's Q5_K tiled GEMM launch uses
  block_dim Q6_THREADS (256) while the 44ab4cf-corrected
  GEMM proof uses Q5_THREADS (128). E1's dispatch
  mirrors the production site; Glimmer's per-layer
  slots never exercise the Q5 tiled branch, and every
  measured arm passed bit-exactness and argmax. One of
  the two sites is wrong; settling it needs a Q5 tiled
  run at both block dims (owed, below).

### E2, decode: profile first, then launch overhead

- Profile (window 1, real model, enqueue+sync medians):
  baseline walls step 33,815 us, logits 1,414 us,
  step+logits 35,230 us = 28.38 tok/s steady state
  (step-loop method, context 4096). Isolated ops sum
  to about 1,258 us/layer against a 683.5 us whole-
  layer wall: host launch/alloc time overlaps GPU
  work, and the queue drains only at the KV append's
  sync and the layer-end download. Launch census
  about 76 enqueues/layer (about 3,900/step), 21 of
  them zero-fill launches. The profile pointed at
  launch overhead, not GEMV math.
- Pre-fix (the Wave D defect class, fourth instance):
  the resident _buf dispatch launched the promoted
  F16 GEMV at grid rows (kernel computes 2 rows per
  workgroup: writes up to 2x rows out of bounds) and
  the F32 GEMV at block 128 (kernel reduction spans
  256: half the columns summed). Both latent on
  Glimmer (no layer slot routes F16/F32 through that
  dispatch). Found by grepping every f16/f32 GEMV
  enqueue site; fixed to the hostchain geometry.
- Iteration 1, KEPT: resident-chain outputs take raw
  buffers (_alloc_f32) instead of create_zero_f32;
  every consumer kernel writes every element (audit
  in the docstring). Step 33,815 -> 31,327 us
  (-7.4%).
- Iteration 2, KEPT: splits==1 direct-output tiled
  attention (attention_tiled_direct_kernel): the
  tiled kernel's loop with the final write in the
  same launch, no partial buffers, no merge launch.
  CPU twin proven EXACT (0.0) against the split path
  at splits==1 and <= 1e-5 vs Float64; device proof
  PASS at stored 16/128/512 (rel <= 1.37e-07). Step
  31,327 -> 31,035 us (-0.9%).
- Final lane numbers: step+logits 32,491 us = 30.78
  tok/s steady state (same WALL method; -8.2% step
  alone). The runner approx on the lane tree read
  52.87 ms/step (18.92 tok/s) against Wave D's 51.7
  (19.35); the approx carries a measured one-time
  ~200 ms stall on the first logits after GEMM
  prefill, amortized over 15 steps, so at this delta
  it sits inside window noise; the WALL series is the
  sensitive measure and it moved -7.8%. On the final
  integrated tree the runner approx read 49.13 ms/step
  = 20.35 tok/s (see the golden section).
- Rejected/not attempted, recorded in the lane note:
  rope position-buffer sharing (bounded below
  iteration 2's win); GEMV retuning (K1's bar is
  David's open call, untouched; no gate moved).

### E3, the owed hygiene items

- Budget-ledger unreserve (Wave C debt): GpuRuntime
  was reserve-only; two server proofs reset ledger
  fields directly, naming the missing API. Now: pure
  budget_release (over-release refuses by name, the
  ledger cannot go negative), unreserve_weights /
  unreserve_kv, release_reservation on DeviceWeights
  and DeviceKvCache (exact build totals, double
  release refuses by name), and
  GpuGlimmerModel.release_reservations() as the
  destroy path. Proof: reserve-to-cap, refusal,
  release, identical re-reserve, in the plumbing
  suite (+2 tests). The release path executed on the
  card in the final window via the layer proof.
- Harness verdict-line extraction (Wave C debt):
  wave_c_timing.sh extracted the proof's HEADER line
  as the teacher-forced verdict and could not fail on
  a verdict-less log (reproduced on synthetic logs
  pre-fix). It now accepts only the anchored
  TEACHER_FORCED_GPU <n>/<m> line; 7/8 fails as
  NOT_FULL_PASS; a missing verdict fails loudly,
  exit 1. Self-test tests/wave_c_timing_extract_test.sh
  passes 4/4 and is wired into run_tests.sh.
  wave_d_timing.sh never had the bug.
- GLIMMER_* fold (Wave A item): exactly 7 constants
  in 4 ops files, zero production readers (verified
  by grep; Waves B/C/D had folded none). All 7 now
  live only in config.mojo as EngineConfig fields
  (rms_eps, rope_base, logit_scale, logit_softcap,
  head_dim, hidden, vocab) with MOJO_ENGINE_* env
  overrides and named refusals; the 3 consuming tests
  import from config; config_probe prints them.
  Zero numeric behavior change: loaded-model execution
  still sources these from the GGUF; precedence is
  stated in the config block header.

## Integration order, as executed

Baseline on fa7221e verified by the coordinator before
any merge: suite 156 passed / 0 failed.

1. E3, one cherry-pick per item, suite after each:
   item 1 (bf92786) 158, item 2 (8310e1f) 158,
   item 3 (3904ecd) 161, lane record (0412bc8).
   Each pushed separately, SHA MATCH verified.
2. E1: the batch-step commit (06f03a9) and the proofs
   commit applied clean; the routing-knob commit
   conflicted in config.mojo in six hunks (E3's
   delimited blocks vs E1's knob), each resolved as a
   union with prefill_batch_min_tokens LAST in the
   field list, __init__ params, and the load() return
   (the Wave D positional hazard, handled by
   construction); routing default (c292629) and lane
   record followed. Wiring commit (355dca2) added
   test_wave_e_batch_vm.mojo and the E3 extraction
   self-test to run_tests.sh. Suite after: 166.
3. E2: all six commits applied with zero conflicts
   (disjoint files, as designed); wiring commit
   (2f1c091) added test_attention_direct_cpu.mojo.
   Suite after: 168 passed / 0 failed.
   Launch-site re-checks after the merges: E1's new
   constructor argument is last at open, open_attn,
   and both runner construction sites; E2's geometry
   fix IS the audit of the f16/f32 sites; E1's batch
   dispatch sites grepped for the Q5 question above.
4. Golden window on 2f1c091 (below): PASS on every
   leg. Version bump 1.5 -> 1.6 (config.mojo
   ENGINE_VERSION and the run_tests.sh hello gate
   together), DESIGN.md status updated, this record.

## Integrated verification (the final window)

Staged from 2f1c091375655daab4d117268542e083fe91b3bd
by git archive (archive sha256 identical local and
server), STAGED_COMMIT stamp, five staged file hashes
verified against the integration worktree (batch,
layer, model, config, harness). Harness
tests/wave_e_timing.sh, exit 0, one window under the
phase lock:

- Batch layer proof: WAVE_E_BATCH_LAYER_PROOF PASS,
  the four cases digit-identical to E1's lane record
  (rel 2.49e-07 / 4.76e-07 / 1.65e-07 / 3.42e-07).
- Route proof: WAVE_E_ROUTE_VERDICT PASS;
  WAVE_E_CROSSOVER first_batch_wins_n=16;
  WAVE_E_ARMS1022 batch 34,852 ms vs per-row
  45,042 ms, speedup 1.29x.
- Teacher-forced golden: TEACHER_FORCED_GPU 8/8,
  total_ms 483; winning logits identical to the
  Wave C record (17.343926 .. 15.78144); step_ms
  31/30 for the walk's decode steps.
- End-to-end runner, GPU path: GOLDEN_FIRST8_MATCH 1,
  TEXT_STRICT_OK 1, 16 tokens [13796, 26, 1573, 373,
  290, 2167, 26899, 1212, 6996, 310, 11698, 341, 290,
  2167, 26899, 1212], text " Paris. It is the most
  populous city in France and the most populous" -
  identical to the Wave B/C/D records. 6-token
  prefill 217 ms (per-row route; Wave D: 422 ms,
  Wave C: 684 ms). Decode approx 49.13 ms/step,
  20.35 tok/s over the 15 decode steps (Wave D:
  51.7 ms/step, 19.35 tok/s).
- E2 attention-direct device proof:
  GPU_PROOF_ATTENTION_DIRECT PASS; stored 16/128/512
  cosine >= 0.9999999999999989, rel <= 1.37e-07,
  twin diff 0.0; direct vs pair timing ratios
  1.38x / 1.07x / 1.04x.
- GPU layer proof (exercises E3's budget-release
  path on card): GPU_LAYER_PROOF PASS; layer 0
  cosine 0.9999999999985238, layer 3 cosine
  0.9999999999983641 (digit-identical to Wave D),
  logits argmax 150015 exact.

Server closeout, as observed: cortex-embed.service
was ACTIVE at entry (running since 18:31:57 UTC,
payload on 8091 healthy); stopped for the window
(VRAM drained to 33,271,808 B) and restored active
with /health ok; VRAM at closeout 2,584,600,576 B.
Port 8080 CLEAR at entry and closeout (the Glimmer
llama-server was already down; nothing was stopped
for 8080). Containers cortex-cortex-1 and
cortex-cortex-surrealdb-1 up throughout, never
touched. No reboot. Staging ~/gpu-wave-e left in
place with its logs.

## Numbers against the brief's baselines

- Decode: Wave D runner approx 19.35 tok/s ->
  Wave E final tree 20.35 tok/s; steady-state WALL
  measure (E2's method): 28.38 -> 30.78 tok/s lane
  tree. Rust engine plain decode ~34.5 tok/s and
  llama Vulkan 44.59 tok/s remain the scale points;
  the gap is now measured as launch-bound, and the
  next lever is named below.
- 6-token prefill: 422 ms (Wave D) -> 217 ms.
- 1,022-token prefill: batch arm 76,579 ms (Wave D)
  -> 34,852 ms; per-row arm 48,415 -> 45,042 ms;
  the routed default (batch at this length) is the
  34,852 ms figure, 29.3 tok/s against llama
  Vulkan's recorded 995.99 tok/s prefill at 1,022 -
  prefill remains the largest gap to the reference
  engines, and it is kernel-throughput work now, not
  structure: the batch path's remaining costs are
  the GEMM kernels' own rates and per-layer launch
  counts at batch scale.

## What remains owed after Wave E

- Step-level activation residency: the hidden row
  still crosses host<->device between all 52 layers
  in a decode step (upload + download + sync x52).
  Needs gpu_model's step loop and the layer signature
  changed together (E2's ranked item 1).
- A no-sync DeviceKvCache append: append_device's
  mid-layer synchronize caps host lookahead at half
  a layer (E2's item 2). Scratch/allocator pooling
  (about 27 allocs/layer) and rope position-buffer
  sharing follow (items 3-4).
- The Q5_K tiled GEMM block-dim discrepancy (256 in
  the production batch dispatch vs 128 in the GEMM
  proof): settle with a Q5 tiled run at both block
  dims; unreachable by Glimmer shapes today.
- The Wave D leg-5 class item on the paused tree and
  the batch path at very long contexts (the route
  proof's stored-count growth past 2,048 on windowed
  layers rides the device KV wrap proven in D4; a
  prefill past the window wrap on the batch step is
  not yet separately proven).
- K1's correctness bar (rel 1.267e-07) still binds
  its tuning; whether the bar moves is David's call.
- The sampler family (temperature/top-k/top-p) and
  the serving seam remain the next waves, in that
  order (DESIGN.md steps 6/7).
