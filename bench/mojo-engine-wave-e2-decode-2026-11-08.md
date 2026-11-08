# Mojo engine Wave E, Lane E2: decode speed (2026-11-08)

Lane: `lane/mojo-e2-decode`, cut from `fa7221e` (engine
1.5, Wave D integrated tree). Job: profile one decode step
on the real model, attack the largest measured item with
the Wave A/D ladder discipline, close what measurement
honestly supports of the gap to the reference points
(Rust engine plain decode ~34.5 tok/s, llama Vulkan
44.59 tok/s on the same card; Wave D integrated decode
19.35 tok/s by the runner approx, 51.7 ms/step).

All server numbers below are direct foreground reads of
the window log files. Staging was git archives with
STAGED_COMMIT stamps and sha256-verified sources in every
window; the phase lock (`flock -w 14400
/tmp/itak-phase.lock`) was held for every card window.
Server state at window entry, as observed: port 8080
clear (the Glimmer llama-server stayed DOWN per David's
standing order), the cortex-embed payload answering
`{"status":"ok"}` on 8091 with cortex-embed.service
INACTIVE (the Wave D closeout state; the payload's
~1.9 GB footprint left ample room, so nothing was
stopped for these windows and there was nothing to
restore), containers cortex-cortex-1 and
cortex-cortex-surrealdb-1 Up throughout, never touched.

## 1. The profile (window 1, tree fc9f4ae = fa7221e +
##    geometry fix + gated instrumentation + this proof)

Harness: `tests/server_proof_decode_profile.mojo` (new
this lane). Phase WALL times the production
GpuGlimmerModel path with no instrumentation (prompt
stepped at positions 0..5, 3 warmup decode steps, 10
measured free-running steps, context 4096). Phase OPS
times every resident-chain launch shape on layer 0's
real slots, enqueue + synchronize per rep (the D2/D3
pattern), medians over 30-50 reps after warmup. The
in-path cross-check is the MOJO_ENGINE_DEBUG-gated
PROFILE_LAYER / PROFILE_LOGITS wall lines added to
gpu_layer_step_cfg / gpu_logits this lane (off by
default, zero behavior change when off).

### Clean walls (baseline tree)

- Step (52 layers): median 33,815 us. Logits: median
  1,414 us. Step + logits: 35,230 us = 28.38 tok/s
  steady-state engine rate at short context.
- Teacher-forced proof, same window: 8/8, step_ms 33
  (integer print) per step, total_ms 450.
- Runner, same window: GOLDEN_FIRST8_MATCH 1,
  TEXT_STRICT_OK 1, tokens identical to the Wave B/C/D
  record, PREFILL_MS 423, DECODE_STEP_MS_APPROX 55.27
  (18.09 tok/s). Wave D's final tree read 51.7 / 19.35
  in its window. The approx is (generate_ms -
  prefill_ms) / 15; the generate phase carries a
  one-time ~200 ms stall on the first logits after the
  GEMM prefill (visible in the teacher total too:
  450 ms against 7 x 33 ms of steps + 8 x ~1.4 ms of
  logits ~= 245 ms accounted), which the approx
  amortizes over 15 steps. The steady-state per-token
  cost is the WALL figure above; both measures are
  reported for every arm below, the runner approx for
  comparability with the Wave D record.
- PROFILE_LAYER (debug run, same teacher binary):
  steady layers 582-690 us, median ~630 us; sliding
  layers ~615-660 us, global layers ~582-638 us (the
  sliding delta is the rope pair + windowed append
  shape). PROFILE_LOGITS steady 1,358-1,459 us.

### Per-op table (isolated enqueue+sync medians, us)

| op (layer 0 shapes) | us | notes |
| --- | ---: | --- |
| GEMV attn_q Q4_K 4096x6656 | 64.4 | incl zero+alloc |
| GEMV attn_k Q4_K 256x6656 | 33.5 | incl zero+alloc |
| GEMV attn_v Q6_K 256x6656 | 32.9 | incl zero+alloc |
| GEMV attn_gate Q4_K 4096x6656 | 67.5 | incl zero+alloc |
| GEMV attn_output Q4_K 6656x4096 | 59.3 | incl zero+alloc |
| GEMV ffn_gate Q4_K 19968x6656 | 193.9 | incl zero+alloc; see anomaly |
| GEMV ffn_up Q4_K 19968x6656 | 114.2 | incl zero+alloc |
| GEMV ffn_down Q6_K 6656x19968 | 215.0 | incl zero+alloc |
| GEMV head Q5_K 202048x6656 | 1,379.7 | LAUNCH ONLY (925 MB, ~670 GB/s) |
| gpu_logits whole | 1,574.5 | norm+head+final_logit+download |
| rms_norm hidden 6656 | 33.3 | incl zero+alloc |
| rms_norm Q heads (32x128) | 23.8 | incl zero+alloc |
| rms_norm KV heads (2x128) | 23.8 | incl zero+alloc |
| rope Q (32 vec) | 37.6 | incl zero + pos host alloc + copy |
| rope K (2 vec) | 37.5 | incl zero + pos host alloc + copy |
| KV append pair (append_device) | 22.5 | incl its internal synchronize |
| attention resident, stored 16, splits 1 | 38.7 | tiled + merge + 3 partial allocs |
| attention resident, stored 128, splits 1 | 117.3 | same shape, longer scan |
| attn_gate elem 4096 | 22.3 | incl zero+alloc |
| swiglu 19968 | 22.5 | incl zero+alloc |
| residual_add 6656 | 25.2 | incl zero+alloc |
| final_logit 202048 | 27.5 | incl zero+alloc |
| create_zero_f32 (4096/6656/19968/202048) | 20.5 / 20.8 / 21.1 / 23.2 | alloc + zero launch + sync |
| raw enqueue_create_buffer (same sizes) | 16.5 / 16.7 / 16.6 / 19.3 | alloc + sync |
| upload_f32 6656 + sync | 38.0 | per-layer boundary, in |
| download_f32 6656 (incl sync) | 27.4 | per-layer boundary, out |
| bare synchronize, empty queue | 0.35 | sync itself is cheap |
| WHOLE resident layer step, layer 0 | 683.5 | production entry, all-in |
| WHOLE resident layer step, layer 3 | 657.8 | production entry, all-in |

Anomaly, recorded not smoothed: ffn_gate (193.9) and
ffn_up (114.2) are the same dtype and shape; gate was
timed first among the large GEMVs in the proof's order
and shows the lower effective bandwidth. The
reconstruction below leans on the whole-layer
cross-check, not on either figure alone.

### Reconstruction and where the profile pointed

Sum of the isolated per-op medians for one sliding
layer is ~1,258 us, against 683.5 us for the whole
layer measured through the production entry: the
isolated figures overstate every small op, because in
the real chain the host enqueues ahead while the GPU
works - allocator and launch host time overlaps GPU
execution. The queue drains at exactly two points per
layer: append_device's internal synchronize (mid
layer) and the layer-end download. Launch census of
one sliding layer in the baseline tree: 24 compute
kernels (6 norms, 8 GEMVs, 2 rope, 2 KV scatters,
tiled + merge attention, gate, swiglu, 2 residuals)
+ 21 zero-fill kernels + ~27 buffer allocations
(outputs, partials, rope position buffers) + 4 copies
(upload, 2 rope positions, download) ~= 76 enqueues;
per decode step ~= 3,900 enqueues plus the head
chain. GEMV weight streaming is the largest GPU-time
class (the eight layer GEMVs sum to ~781 us isolated;
the head GEMV alone is 1.38 ms launch-only), but the
largest REDUCIBLE item the profile exposes inside
this lane's files is the launch/allocation overhead
riding on every op:

1. 21 zero-fill launches per layer (1,092 per step)
   in front of kernels that write every output
   element - pure overhead, removable inside
   gpu_layer.mojo alone. -> Iteration 1.
2. The merge launch + 3 partial allocations in the
   splits == 1 regime (stored <= 512: all of early
   decode and, at span 512, everything up to 512
   stored), the D2 note's named candidate (a).
   Attention at stored 16 measured 38.7 us through
   the production pair. -> Iteration 2.
3. Structural items OUTSIDE this lane's file
   ownership, recorded for the coordinator: (a) the
   per-layer host boundary - upload + download +
   synchronize x 52 per step - is baked into the
   gpu_layer_step_cfg List-in/List-out signature and
   gpu_model's step loop; a step-level resident chain
   (hidden row stays on device across all 52 layers)
   removes 104 crossings per step; (b)
   DeviceKvCache.append_device's mid-layer
   synchronize caps host lookahead at half a layer -
   a no-sync append variant in device_kv.mojo would
   let the host enqueue the whole step ahead of the
   GPU; (c) buffer allocation itself (~16-19 us
   isolated per enqueue_create_buffer with sync;
   the chain allocates ~27 buffers per layer) - a
   scratch-pool chain needs a stateful design above
   this lane; (d) gpu_model's embedding norm uploads
   a fresh ones row every step.

## 2. Pre-attack fix (commit d9f2fa0): resident F16/F32
##    launch geometry

Found while reading the chain for the profile, the
exact Wave D landing-defect shape: the resident
(_buf) helpers kept pre-D3 geometry for two kernels -
_gemv_u8_buf launched f16_gemv_kernel with
grid_dim=rows (the promoted kernel computes
ROWS_PER_WG=2 rows per workgroup: up to 2x rows
written past the output buffer) and _gemv_f32_buf
launched f32_gemv_kernel with block F_THREADS=128
(the port kernel's reduction spans F32_THREADS=256:
half the columns summed). Both latent on Glimmer
(no layer slot routes F16/F32 through the GEMV; the
fix branch commit 639d2c3 on lane/mojo-d3-regression
named the same pair but is not an ancestor of the
integrated tree). Fixed to the hostchain geometry.
Enqueue-site audit in the same commit: every
f16_gemv_kernel / f32_gemv_kernel site in src+tests
grepped - gpu_layer's _device twins and the kernels'
own standalone mains already correct,
server_proof_gemm corrected at Wave D landing
(44ab4cf), gpu_layer_batch launches the GEMM family,
not these kernels. No behavior change on the Glimmer
path; no gate moved.

## 3. Iterations (window 2: arms A = fc9f4ae baseline,
##    B = 3d28299 iteration 1, C = 85b5cd5 iteration 2)

Window 2's first pass collided with Lane E1's card
window (their proof_route process, started 18:18 UTC,
held the card; every model run in pass 1 OOM-refused
and the direct proof's first timing set was taken
under that contention - it is recorded below as the
contaminated set and not used for any verdict). E1's
window completed cleanly (their route verdict PASS,
embed restored by their closeout: user unit active,
8091 health ok, VRAM 2,584,600,576 B), and window 2
was re-run in full on the clear card. All numbers in
this section are from the re-run, foreground-read.

Interleaved teacher rounds (A/B/C x 3 rounds in one
window, 21 step samples per arm), step_ms medians:
A 33, B 31, C 30. Teacher verdict 8/8 on every arm in
every round.

WALL medians (profile proof; A measured in window 1,
B and C in window 2 - the interleaved teacher series
above is the same-window control for the A arm):

| arm | step us | logits us | step+logits us | tok/s |
| --- | ---: | ---: | ---: | ---: |
| A baseline | 33,815 | 1,414 | 35,230 | 28.38 |
| B iter 1 | 31,327 | 1,458 | 32,785 | 30.50 |
| C iter 1+2 | 31,035 | 1,455 | 32,491 | 30.78 |

Runner approx (the Wave D-comparable measure; B and
C run in window 2, A in window 1): A 55.27 ms /
18.09 tok/s, B 51.8 / 19.31, C 52.87 / 18.92 - B and
C inside each other's noise on this measure, both
better than A's window-1 reading and level with Wave
D's 51.7 / 19.35; the approx's one-time-stall
amortization (section 1) swamps differences of this
size, which is why the WALL series is the lane's
headline. GOLDEN_FIRST8_MATCH 1 and TEXT_STRICT_OK 1
on both B and C, token stream identical to the
record; PREFILL_MS 416 (B) / 417 (C) vs 423 (A).

### Iteration 1 (commit 3d28299): zero-fill removal -
### KEPT

Change: every resident-chain output buffer in
gpu_layer.mojo (_rms_norm_buf, _gemv_u8_buf,
_gemv_f32_buf, _rope_buf, the attention output,
_attn_gate_buf, _swiglu_buf, _residual_add_buf, and
gpu_logits' final row) now comes from _alloc_f32
(raw enqueue_create_buffer) instead of
create_zero_f32. The full-write audit per kernel
family is recorded in _alloc_f32's docstring; the
hostchain A/B baseline helpers were not touched.
Gates before timing: test_gpu_layer_vm 5/5, gfx1100
runner build clean. Server gates (arm B): teacher
8/8 x 3 rounds, runner golden (above), whole-layer
ops: layer 0 683.5 -> 612.7 us, layer 3 657.8 ->
596.5 us (profile proof, arm B vs A across windows).
Effect: step 33,815 -> 31,327 us (-7.4%), the
largest single win of the lane, from removing 1,092
zero-fill launches per step (plus their enqueue host
time) - smaller than the naive 21 x ~4 us x 52
estimate because much of the isolated zero cost was
host time that already overlapped GPU work.

### Iteration 2 (commit e2b94c7 + proof 85b5cd5):
### splits==1 direct attention - KEPT

Change: attention_tiled_direct_kernel[TILE] in
attention.mojo (the tiled kernel's tile loop and
rescale points statement for statement, range = the
whole visible sequence, final write acc / run_sum in
the same launch) plus its CPU twin
attention_direct_cpu; gpu_layer's resident attention
launches the direct form exactly when
attn_split_count returns 1, and keeps the tiled +
merge pair (and its partial allocations) for
splits >= 2. Enqueue-site audit for the changed
kernels in the same commit: gpu_layer is the only
production site for the tiled/merge pair;
gpu_layer_batch keeps the Wave B kernel per the
Wave D seam (E1's file, untouched); the D2 proof's
own sites launch the unchanged kernels.
Gates before timing: VM twin suite
tests/test_attention_direct_cpu 2/2 - direct twin
EXACT (max diff 0.0) vs attention_split_cpu at
splits == 1 on every fixture incl. stored 512 at
Glimmer geometry, and <= 1e-5 vs the Float64
reference; split suite still 10/10; gfx1100 build
clean. Device proof (tests/
gpu_proof_attention_direct.mojo, clean re-run):
stored 16 / 128 / 512 all PASS, cosine
0.9999999999999979 / ...998 / ...989, rel
6.36e-08 / 1.37e-07 / 1.37e-07 vs both CPU
references, twins exact. Isolated launch timing
(pair -> direct): stored 16: 66.9 -> 46.0 us
(1.45x); stored 128: 202.6 -> 189.7 (1.07x);
stored 512: 635.2 -> 609.4 (1.04x). (The
contaminated first pass read 70.0 -> 37.6 (1.86x),
236.8 -> 219.8 (1.08x), 898.0 -> 1003.7 (0.89x):
the 512 regression in that set was contention
noise - the clean re-run has the direct form ahead
at all three stored counts, and the production
wiring agrees, below.) Through the production
resident path (profile proof): attention at stored
16: B 35.7 -> C 31.6 us; stored 128: B 115.4 ->
C 107.6 us. Whole layer 0: B 612.7 -> C 602.6 us;
layer 3: 596.5 -> 601.5 (inside noise). End to end:
step 31,327 -> 31,035 us (-0.9%), teacher median
31 -> 30 ms. Small, real, and exactly the size the
D2 note predicted for the short regime (under half
a millisecond per full step at Wave C step times;
at this lane's step times it is ~0.3 ms).
Final-arm layer proof (arm C):
GPU_LAYER_PROOF PASS, layer 0 cosine
0.9999999999985238 / max rel 2.33e-06, layer 3
cosine 0.9999999999983641 / max rel 1.01e-06,
logits argmax 150015 exact - the Wave D record's
cosines reproduced digit for digit.

### Rejected / not attempted, with reasons

- Rope position-buffer sharing (one host buffer for
  both rope calls per sliding layer): not attempted.
  The isolated rope figure (37.5 us) is sync-inflated;
  the in-chain saving is bounded by one small host
  alloc + one copy enqueue per sliding layer, an
  order below iteration 2's measured win, and the
  lane's remaining window budget went to verifying
  the two kept iterations. Named here as a micro-item
  for a later wave.
- GEMV kernel retuning (K1/Q6/Q5 shapes): not
  attempted. K1's bar is David's open call and did
  not move; the profile shows the head GEMV already
  at ~670 GB/s launch-only at full size and the
  layer GEMVs are the same kernels D3 tuned - the
  reducible fat the profile actually exposed was
  launch overhead, not GEMV math.
- Prefill-path items (batch attention wiring,
  GEMM-vs-resident routing): E1's lane, untouched.

## 4. Suite tally and closeout

VM suite on the final tree (85b5cd5 + this note):
tests/run_tests.sh PASS, 156 tests (the coordinator's
fa7221e baseline tally, matched), plus the seam-on
diag rerun, config probes, and engine hello inside
the same script. Lane-new suites run standalone and
named here for the coordinator's run_tests.sh
wiring: tests/test_attention_direct_cpu.mojo (2/2,
inside the tally above once wired - it is NOT in
run_tests.sh yet, so the 156 does not include it);
server-only proofs (not TestSuite files):
tests/server_proof_decode_profile.mojo,
tests/gpu_proof_attention_direct.mojo.

Server closeout, as observed at window 2's end:
port 8080 CLEAR (Glimmer left down, per standing
order); 8091 listening, embed health
{"status":"ok"}, cortex-embed.service (user unit)
active - stopped and restored by Lane E1's window,
not by this lane (this lane stopped nothing);
containers cortex-cortex-1 and
cortex-cortex-surrealdb-1 Up, never touched; no
reboot. VRAM bookend at E1's restore:
2,584,600,576 B (the D3 windows' figure).

## 5. What remains

- Decode stands at 30.78 tok/s steady-state (WALL,
  step+logits) / ~19 tok/s by the runner approx on
  the final lane tree, against Wave D's 19.35 approx
  and the reference points (Rust ~34.5, llama Vulkan
  44.59). The remaining gap is structural, in the
  order the profile ranks it: (1) step-level
  activation residency - the hidden row still
  crosses to the host and back between all 52 layers
  (upload 38.0 + download 27.4 us isolated per
  boundary, plus a queue-draining synchronize per
  layer); needs the gpu_model step loop and the
  layer signature changed together (coordinator
  integration work, E2's files alone cannot do it);
  (2) a no-sync DeviceKvCache append (device_kv is
  outside E2 ownership): append_device's mid-layer
  synchronize caps host lookahead at half a layer;
  (3) scratch/allocator pooling for the ~27 buffers
  per layer the chain allocates (raw alloc + sync
  measured 16.5-19.3 us isolated); (4) the rope
  position-buffer micro-item and the embedding
  norm's per-step ones-row upload in gpu_model.
- Config knob proposed (coordinator adds, per lane
  rules): none required by this lane's changes -
  the direct attention form is selected by the
  existing split chooser, no new knob.
- K1's correctness bar: untouched, still David's
  open call.
