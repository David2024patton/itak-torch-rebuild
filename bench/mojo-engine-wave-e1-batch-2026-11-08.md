# Mojo engine Wave E, Lane E1: residency + tiled attention in the batch step

Date: 2026-11-08. Branch `lane/mojo-e1-batch`, cut from
fa7221e (engine 1.5). Lane commits are listed at the
bottom; the coordinator integrates and pushes.

## The problem (Wave D record)

On the Wave D integrated tree, the 1,022-token prefill
arms INVERTED: the GEMM batch arm took 76,579 ms against
the per-row arm's 48,415 ms, even though the GEMM kernels
beat the GEMV kernels 2.0-2.1x at batch 64 with
bit-identical outputs. The batch step lost end to end
because (a) it launched the Wave B attention kernel per
position instead of D2's tiled split-KV attention, and
(b) it round-tripped activations through host rows per
op instead of keeping them resident like D4's chain.

## What was built

1. `src/model/gpu_layer_batch.mojo` rewritten around
   `gpu_layer_step_batch_cfg` (the Wave D
   `gpu_layer_step_batch` signature is kept as a wrapper
   with the config-default attention knobs):
   - Residency: the input batch uploads once; the
     residual stream, norm outputs, Q/K/V, the gated
     attention output and the MLP intermediates stay in
     device buffers op to op; the residual adds are
     device adds (`residual_add_kernel`); the output
     batch downloads once. GEMM launches take resident
     buffers with the Wave D launch geometry unchanged,
     site for site.
   - KV appends are device to device: a new
     `kv_batch_scatter_kernel` (the offset twin of
     device_kv's `kv_scatter_kernel`, which carries no
     source offset) scatters each batch row from the
     resident K/V batch buffers into its cache slot,
     validated and committed through the cache's own
     `_append_slot` / `_append_commit` - the same calls
     `append_device` makes for one row.
   - Attention: `attention_batch_tiled_kernel` +
     `attention_batch_merge_kernel`, a batched twin of
     D2's tiled split-KV pair living in this file: same
     shared-memory tile staging, same online-softmax
     partials, same exact logsumexp merge. One block per
     (batch element, Q head, split); each element's
     splits partition its OWN causal stored count
     (start + i + 1), so element i sees exactly the rows
     the per-position steps would have seen, even though
     the whole batch is appended before the layer's
     attention launch (the Wave D append order). Split
     count comes from `attn_split_count` at the LAST
     element's stored count; earlier elements' ranges
     shrink by the same chooser arithmetic and trailing
     empty splits merge as no-ops. K/V are read through
     `DeviceKvCache.gather_visible`, like the resident
     step. ONE tiled launch + ONE merge launch per layer
     for the whole batch, against the Wave D batch step's
     `batch` Wave B launches (each with a host sync).
   - Guards kept: a windowed batch that would reach its
     window refuses `gpu_kv_window_wrapped`; attention
     refuses by the same name when the visible rows do
     not start at position 0. The attention knobs are
     validated BEFORE any KV append, so a bad knob
     refuses without mutating the cache.
   - CPU twin for the causal arithmetic:
     `attention_batch_causal_cpu` (per element, D2's
     `attention_split_cpu` over the causal prefix),
     VM-proven in tests/test_wave_e_batch_vm.mojo.
2. Routing knob (config.mojo): 
   `MOJO_ENGINE_PREFILL_BATCH_MIN_TOKENS`, field
   `EngineConfig.prefill_batch_min_tokens`, parser
   `parse_prefill_batch_min_tokens` on the shared
   `_parse_positive` (refusal strings byte-identical in
   shape to the other knobs). The rule lives in the pure
   helper `config.prefill_uses_batch`: the batched route
   serves a prompt only when its length reaches BOTH
   the GEMM minimum batch (so the Wave D A/B control -
   raising the GEMM minimum above any prompt - still
   forces the per-row arm alone) and the new routing
   minimum. Threaded config -> runner
   (generate_main's GPU phases) -> GpuGlimmerModel ->
   prefill(). The model field and constructor parameter
   were APPENDED at the end of the parameter list, and
   every positional construction site was re-grepped
   (see the hazard check below). Default: 16 tokens,
   from the crossover measurement below (batch loses at
   6, wins at 16 and at every larger measured length).
3. Proofs (new files):
   - tests/test_wave_e_batch_vm.mojo (VM): routing rule,
     knob parser/default, causal stored counts, and the
     batch attention CPU twin against per-element split
     twins (exact) and the single-position math core
     (Wave B bar class). NOT wired into run_tests.sh
     (lane rule); listed for the coordinator below.
   - tests/server_proof_wave_e_batch_layer.mojo (card):
     per-layer batch vs per-row resident at rows 6 and
     64, layers 0 and 3, bar cosine >= 0.999999999,
     rel <= 1e-5 (the D2 class).
   - tests/server_proof_wave_e_prefill_route.mojo
     (card): interleaved batch-vs-per-row arms at
     6/16/32/64/128/256/512/1022 tokens, fresh model per
     arm, cross-arm last-hidden at the Wave C/D class
     (cosine >= 0.999999, rel <= 1e-4, argmax exact);
     prints the crossover and the 1,022 arms.
   - tests/wave_e_timing.sh: the window harness
     (staging stamp, builds, the four runs, verdict
     extraction from the on-disk logs in the foreground).

## Trial record

- T1 (VM test, first run): test_wave_e_batch_vm ran 4/5;
  the failure was a bug in MY test case, not the engine:
  the "short stores" refusal case handed the twin a
  store of exactly the required length (a valid shape),
  so no refusal fired. Case corrected to a genuinely
  short store; 5/5 after.
- T2 (proof parse): server_proof_wave_e_batch_layer
  named a variable `ref`, which this Mojo version
  reserves; `mojo format` refused to parse the file.
  Renamed to ref_flat; the file then formatted and
  compiled clean. No other file used the name.
- T3 (evidence hazard, the Wave B/D failure class): the
  first full-suite run's backgrounded delivery text
  reported a compile failure in
  tests/test_gpu_prefill_vm.mojo that does not exist in
  the on-disk log. Foreground rerun of that file passed
  3/3, and the suite log read in the foreground shows
  156/156 PASS (see Results). Every number in this note
  was read in the foreground from an on-disk log, local
  or server, never from a delivery text.
- T4 (no rejected kernel iteration): the batched tiled
  attention + resident batch chain compiled for
  gfx1100 on the first build and passed its per-layer
  gate on the first card run at both batch sizes and
  both layers; there is no rejected device iteration
  to record. The design choices that were settled
  BEFORE building, from the Wave D record: splits are
  chosen at the last element's stored count (per-element
  split counts would need per-element launches and lose
  the one-launch shape); the Q5_K tiled GEMM block dim
  mirrors the production batch site (hazard section).

## Hazard check (Wave D's three geometry/positional burns)

- No kernel signature or geometry in kernels/, attention/
  or gpu_layer.mojo was changed by this lane. The new
  GEMM launch sites in gpu_layer_batch.mojo copy the
  Wave D batch sites' grids and block dims exactly
  (gemm_plain_grid / gemm_tiled_grid / gemm_k1_grid /
  gemm_k1_tiled_grid; K1_THREADS / Q6_THREADS /
  F_THREADS), and the rms_norm / rope / gate / SwiGLU /
  residual launches copy the resident step's geometry.
- One observed discrepancy, recorded not changed: the
  Q5_K TILED GEMM launch in the Wave D batch step uses
  block_dim=Q6_THREADS (256), while the Wave D landing
  fix (44ab4cf) corrected the GEMM PROOF's Q5 tiled site
  to Q5_THREADS (128). The new resident GEMM dispatch
  mirrors the production batch site (Q6_THREADS), not
  the proof, so this lane introduces no unproven
  geometry; no Glimmer per-layer projection slot is
  Q5_K (the census's one Q5_K tensor is the head, which
  the batch step never dispatches), so the branch is
  not exercised by the model. Flagged for the
  coordinator: one of the two sites is wrong and a
  future lane should settle it with a Q5 tiled
  correctness run at both block dims.
- GpuGlimmerModel.__init__ gained ONE parameter,
  appended LAST (prefill_batch_min_tokens). Every
  construction site was grepped after the change:
  gpu_model.open (passes it through as the last
  positional), gpu_model.open_attn (defaults), and both
  generate_main.mojo constructions (the runner threads
  config.prefill_batch_min_tokens as the last argument
  through _gpu_generate_phase / _gpu_prefill_phase /
  _run_gpu). EngineConfig.__init__ likewise gained the
  field last; its load() return passes it last.

## Results

VM side (lane tree, final commit ad65eef):
- tests/run_tests.sh via the pixi run form: PASS,
  156 tests, 0 failed - exactly the fa7221e baseline
  tally the coordinator recorded. (Two full runs, both
  read from the on-disk logs; see T3.)
- tests/test_wave_e_batch_vm.mojo (not wired, see the
  coordinator list): 5/5 PASS.
- gfx1100 compile checks on the VM: the runner and both
  new server proofs build clean (exit 0).

Card window (2026-11-08 18:16:57-18:32:02 UTC, staged
from 0943791, sha256 of the five key files verified
equal to the lane tree; the final tree ad65eef differs
from the staged tree ONLY in the routing default value
64 -> 16 decided from this window's measurements, its
VM-test assertion, and comments - the batch arm in the
proofs is forced by constructor knobs, not the default,
and the golden prompt is below both defaults, so every
figure below carries to the final tree):

Per-layer batch proof (batch vs per-row resident,
fresh KV per arm; bar cosine >= 0.999999999,
rel <= 1e-5):
- layer 0, batch 6:  cosine 0.999999999999961,
  rel 2.49e-07, batch 12 ms, per-row 16 ms
- layer 0, batch 64: cosine 0.99999999999985,
  rel 4.76e-07, batch 38 ms, per-row 51 ms
- layer 3, batch 6:  cosine 0.9999999999999406,
  rel 1.65e-07, batch 7 ms, per-row 5 ms
- layer 3, batch 64: cosine 0.9999999999997569,
  rel 3.42e-07, batch 37 ms, per-row 44 ms
WAVE_E_BATCH_LAYER_PROOF PASS; stored counts agree
(6/6 and 64/64) on both arms.

Route sweep (interleaved, fresh model per arm, min of
two rounds; cross-arm last-hidden rel <= 3.42e-07 and
cosine >= 0.9999999999998 at every length, argmax exact
everywhere; WAVE_E_ROUTE_VERDICT PASS):
  n=6:    batch 286 ms   rows 218 ms    0.76x
  n=16:   batch 364 ms   rows 557 ms    1.53x
  n=32:   batch 694 ms   rows 1,108 ms  1.60x
  n=64:   batch 1,541 ms rows 2,247 ms  1.46x
  n=128:  batch 3,729 ms rows 4,698 ms  1.26x
  n=256:  batch 7,743 ms rows 10,053 ms 1.30x
  n=512:  batch 16,964 ms rows 23,171 ms 1.37x
  n=1022: batch 34,786 ms rows 48,475 ms 1.39x
CROSSOVER: the batch arm loses at 6 tokens and wins at
16 and at every larger measured length. Routing rule
(as built): per-row below 16 tokens, batch at or above
16 - MOJO_ENGINE_PREFILL_BATCH_MIN_TOKENS default 16.

The 1,022-token arms on this lane tree (the Wave D
inversion, remeasured): batch 34,786 ms (29.38 tok/s)
vs per-row 48,475 ms (21.08 tok/s), argmax 302 exact in
every arm and round, cross-arm rel 8.70e-08. Against
the Wave D integrated record: the batch arm improved
from 76,579 ms to 34,786 ms (2.20x) and the inversion
is gone - the batch arm is now the faster arm by 1.39x,
and its per-row counterpart remeasured at 48,475 ms
against Wave D's 48,415 ms (the same arm, unchanged).

Golden legs in the same window: teacher-forced
TEACHER_FORCED_GPU 8/8 with Wave C's exact winning
logits (17.343926, 18.877588, ...); runner
GOLDEN_FIRST8_MATCH 1, TEXT_STRICT_OK 1, all 16 tokens
identical to the Wave B/C/D record, PREFILL_MS 237 for
the 6-token prompt (per-row arm: 6 < 16, as the new
default routes it), decode approx 54.7 ms/step.

## Server closeout, as observed

One window under flock /tmp/itak-phase.lock (acquired
immediately; no E2 contention observed). At entry:
cortex-embed.service was ACTIVE (changed since Wave
D's closeout, where it was inactive with its payload
outside the unit - the unit and payload are together
now), its 8091 payload answering {"status":"ok"}, VRAM
1,939,746,816 B. The window stopped ONLY
cortex-embed.service (drained VRAM 33,271,808 B, the
Wave C/D drained figure), ran the harness
(HARNESS_EXIT=0, all four legs PASS), then restarted
it: EMBED_POST active, /health ok, VRAM_POST
2,584,600,576 B (the fully-resident embed footprint
D4 recorded). Port 8080 CLEAR before and after
(Glimmer llama-server stayed down). Containers
cortex-cortex-1 and cortex-cortex-surrealdb-1 Up
throughout, never touched. No reboot.

## Commits (lane/mojo-e1-batch, not pushed)

- 90ed772 resident batch layer step with batched tiled
  split-KV attention (gpu_layer_batch.mojo)
- a7bc578 routing knob MOJO_ENGINE_PREFILL_BATCH_MIN_
  TOKENS (config.mojo, gpu_model.mojo, generate_main)
- 0943791 proofs (VM tests, per-layer proof, route
  proof, wave_e_timing.sh) - the staged window commit
- ad65eef routing default 16 from the measured
  crossover (config.mojo, VM test assertion)
- (this note's commit) the lane record

## For the coordinator

- Wire tests/test_wave_e_batch_vm.mojo into
  tests/run_tests.sh (5 tests).
- The routing default and the measured crossover are
  stated in Results; the runner on the integrated tree
  at the default knob should prefill the 6-token golden
  prompt through the PER-ROW arm (6 < default) - the
  teacher-forced and runner golden proofs in the window
  ran with the tree's defaults, so they prove that arm;
  the batch arm's golden-class proof is the route
  proof's cross-arm comparison at every length.
