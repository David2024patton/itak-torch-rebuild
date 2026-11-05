# Prefill profile: where Glimmer prefill time goes, per kernel (2026-11-05)

Lane: lane/prefill-profile, based on main 88cb83a (harness
commit 683a8b2). Measured on ubuntu-server, RX 7900 XTX, card
in its cleared state, real Glimmer Q4_K_M file, resident GPU
path, 1,022-token prompt.

**Verdict in one line:** prefill is an MMQ-efficiency
problem, not a memory problem and not a chain-overhead
problem. The two MMQ kernels take 83.2% of GPU time while
streaming their weights at 43.5 GB/s combined, 4.5% of the
card's 960 GB/s ceiling, and reaching only 17% (Q4_K) and
9% (Q6_K) of FP32 peak. In the same run the lm_head GEMV
streams at 649.5 GB/s, so the memory system is proven
capable; the MMQ kernel shape is the cost. Flash attention
is second at 15.2% and grows quadratically with context.
Everything else in the chain, all non-projection ops
combined, is 1.2%.

## Method

- Tree staged at ~/gpu-wave9 from the lane worktree;
  sha256 manifest over all 334 tracked files verified with
  0 mismatches before the run. The lane delta over main
  88cb83a is exactly two added files (the harness example
  and the run script); no engine code changed.
- Card state (taint-window protocol): VRAM read from the
  amdgpu sysfs counters inside the locked session, twice:
  used_bytes=33,271,808 of 25,753,026,560 at session start
  and again immediately before the first measurement. That
  is the cleared state's driver residue; the stopped
  tenants would read about 6.4 GB. The session would have
  aborted above 500 MB used.
- Prompt: the bench protocol's own long prompt
  (bench/mmq-wide-bench.sh: the fox sentence repeated 102
  times), tokenized by the runner to exactly 1,022 tokens.
- Harness: `prefill_profile` (crates/itak-loader/examples)
  loads the file exactly like `glimmer_generate`, builds
  the resident session, and feeds the prompt in 64-token
  slices (the session's chunk cap), draining the diag
  timestamp recorder after every slice.
- Why a harness at all: the diag recorder is never drained
  outside the smoke collector test (found by the
  cleared-card baseline lane today: two full runs with
  WGPU_COMPUTE_DIAG=timing and DIAG_OUT set emitted zero
  records and no file; `DiagRecorder::drain` has exactly
  one caller, tests/resident_session.rs). A fix lane is
  adding an end-of-run drain to the resident sessions.
  This profile's numbers exist because the harness drains
  per slice itself: both timing runs captured all 10,448
  dispatches with dropped=0. A plain session-path run
  would have produced nothing.
- Runs, one locked session (flock on /tmp/itak-phase.lock,
  acquired immediately), CARGO_BUILD_JOBS=4,
  MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=0:
  two timing runs through the harness, one timing-off run
  through the same harness, and glimmer_generate --gpu
  (the shipped monolithic path) once with the diag env
  unset and once with WGPU_COMPUTE_DIAG=timing.
- Clocks are not locked on this host; treat sub-1%
  differences as noise. The two timing runs agree to 0.5%
  on total GPU time and to 0.6% or better per kernel.

## Wall results

| run | prefill wall (ms) | gross tok/s |
|---|---|---|
| harness, timing ON, rep 1 | 8,689 | 117.6 |
| harness, timing ON, rep 2 | 8,701 | 117.5 |
| harness, timing OFF | 8,116 | 125.9 |
| glimmer_generate, diag unset | 8,169 | 125.1 |
| glimmer_generate, diag timing | 8,137 | 125.6 |

Reading: enabling timestamps on the shipped path costs
nothing measurable at prefill scale (8,137 vs 8,169 ms,
-0.4%, inside noise). The harness's own timing-mode wall
runs about 570 ms (7%) above its off wall; that is the 16
blocking drains (about 36 ms each), a harness artifact
that does not touch the per-kernel GPU durations and does
not exist in the shipped path. The off walls of the
harness and of glimmer_generate agree within noise
(8,116 vs 8,169), so the per-slice forward calls cost
nothing against the monolithic call. One more wall
component: the first slice carries about 1.7 s of one-time
host latency (first-forward lazy work) inside its 2,064 ms
slice wall; it is in every wall figure above and in none
of the kernel figures.

GPU time totals 6,659 ms (rep 1) against the 8,116 ms off
wall: a host/serialization gap of 1,457 ms (18% of wall),
about 91 ms per chunk of submission, readback, and host
record work the GPU waits on. The GPU-bound rate for this
chain as shipped is 153.5 tok/s; the wall rate is 125.9.
(For reference: llama.cpp Vulkan prefill on this card is
998.43 tok/s, about 1.02 ms/token against our 6.52 ms of
GPU time per token.)

## The timing tables (verbatim)

Run 1 (diag_records=10448, dropped=0):

```csv
kernel,count,total_ms,share_pct,weight_bytes,gbps
k13_q4k_mmq,2912,4199.916,63.1,194974580736,46.4
k14_q6k_mmq,832,1341.959,20.2,46373470208,34.6
k6_attn_flash,832,1013.128,15.2,0,0.0
k4_norm_chain,1648,43.436,0.7,87752704,2.0
k12_q8_quant,3328,31.572,0.5,0,0.0
k3_q5k_gemv,16,22.778,0.3,14793146368,649.5
k9_attn_prep,832,5.232,0.1,851968,0.2
k8_gather_norm,16,0.431,0.0,0,0.0
k4_norm_close,16,0.430,0.0,851968,2.0
k4_rmsnorm,16,0.400,0.0,425984,1.1
TOTAL,10448,6659.281,100.0,,
```

Run 2 (diag_records=10448, dropped=0):

```csv
kernel,count,total_ms,share_pct,weight_bytes,gbps
k13_q4k_mmq,2912,4219.970,63.1,194974580736,46.2
k14_q6k_mmq,832,1350.202,20.2,46373470208,34.3
k6_attn_flash,832,1014.124,15.2,0,0.0
k4_norm_chain,1648,43.565,0.7,87752704,2.0
k12_q8_quant,3328,31.689,0.5,0,0.0
k3_q5k_gemv,16,22.791,0.3,14793146368,649.1
k9_attn_prep,832,5.246,0.1,851968,0.2
k4_rmsnorm,16,0.642,0.0,425984,0.7
k8_gather_norm,16,0.454,0.0,0,0.0
k4_norm_close,16,0.429,0.0,851968,2.0
TOTAL,10448,6689.111,100.0,,
```

## Analysis

### Stream-bound or compute-bound, by the numbers

Reference points: DRAM ceiling 960 GB/s; FP32 peak about
61.4 TFLOPS (7900 XTX). FLOPs per weight dispatch are
2 x batch x weight elements, elements from weight bytes at
the quant block sizes (Q4_K 144 B, Q5_K 176 B, Q6_K 210 B
per 256 elements; the lm_head's per-dispatch byte count
reproduces its 202,048 x 6,656 Q5_K tensor exactly under
this math).

| kernel | ms | share | GB/s | % of 960 | TFLOPS | % of FP32 peak | verdict |
|---|---|---|---|---|---|---|---|
| k13_q4k_mmq | 4,199.9 | 63.1% | 46.4 | 4.8% | 10.54 | 17.2% | neither: occupancy/instruction bound |
| k14_q6k_mmq | 1,342.0 | 20.2% | 34.6 | 3.6% | 5.38 | 8.8% | neither: occupancy/instruction bound |
| k6_attn_flash | 1,013.1 | 15.2% | n/a (KV) | n/a | n/a | n/a | KV-stream + math, grows with context |
| k3_q5k_gemv (lm_head) | 22.8 | 0.3% | 649.5 | 67.7% | 1.89 | 3.1% | stream-bound, the control case |
| everything else | 81.5 | 1.2% | n/a | n/a | n/a | n/a | not a target |

The MMQ pair streams 241.3 GB per prefill (15.08 GB per
64-row chunk; with the head's 0.92 GB per chunk that is
the full 16.0 GB weight set, once per chunk, as the
design says). At the DRAM ceiling that traffic is 251 ms
of work; the MMQ kernels take 5,542 ms for it, 22x off
the streaming bound, while sitting at a sixth or less of
the compute bound. That is the signature the shader
autopsy predicted from the compiled form: VGPR 256 (the
architectural cap) pins occupancy at the floor, so the
kernel cannot hide its own dequant-and-multiply latency.
The memory system is exonerated by the control case in
the same run: the Q5_K GEMV head streams at 649.5 GB/s,
the same ~650-676 GB/s class GEMV has always shown on
this card.

### MMQ by tensor shape (batch 64 rows; the 62-row tail chunk is the same picture)

| kernel | in -> out | dispatches | ms | us/dispatch | GB/s |
|---|---|---|---|---|---|
| k13 Q4_K | 6,656 -> 39,936 | 780 | 2,111.2 | 2,707 | 55.2 |
| k13 Q4_K | 19,968 -> 6,656 (FFN down) | 390 | 866.1 | 2,221 | 33.7 |
| k13 Q4_K | 6,656 -> 4,096 | 780 | 360.0 | 462 | 33.2 |
| k13 Q4_K | 6,656 -> 8,448 | 390 | 301.5 | 773 | 40.9 |
| k13 Q4_K | 6,656 -> 8,704 | 390 | 291.4 | 747 | 43.6 |
| k14 Q6_K | 19,968 -> 6,656 (FFN down) | 390 | 1,082.0 | 2,774 | 39.7 |
| k14 Q6_K | 6,656 -> 256 | 390 | 173.9 | 446 | 3.2 |

Two shape-level facts. First, the single largest tensor
class (the 149.5 MB Q4_K projection) alone is 31.7% of
all GPU time; MMQ work is concentrated, so a kernel-shape
fix lands where the time is. Second, the tiny Q6_K
dispatch (1.41 MB, the 256-wide KV-side projection) pays
446 us per dispatch at 3.2 GB/s: that is a fixed
per-dispatch cost being paid 416 times for 186 ms total,
a fusion candidate, not a streaming one.

### Flash attention: second cost, and the one that grows

832 dispatches (one per layer per chunk), 1,013.1 ms.
Sliding-window layers: 624 dispatches, 764.0 ms. Global
layers: 208 dispatches, 249.2 ms. Per-chunk totals grow
from 3.2 ms at chunk 0 (64 stored) to 125.7 ms at chunk
15 (about 1,022 stored); the last four chunks are 44% of
all flash time. At this context flash is 15.2% of GPU
time; because each chunk attends over everything stored,
its share grows roughly linearly in total (quadratic
work) while the MMQ share is flat per token. At 4k tokens
flash would rival the MMQ pair; at the full 131k context
it is the dominant term by construction. Any prefill plan
that only fixes MMQ has a built-in expiry date.

### The non-projection chain is not the problem

norm_chain 43.4 ms + q8_quant 31.6 ms + attn_prep 5.2 ms
+ the finish tails about 1.3 ms = 81.5 ms, 1.2% of GPU
time. The 2026-11-03 chain profile's suspicion that
per-chunk fixed costs in the small ops would matter at
chunk shapes is answered: at 64-row chunks they do not.
(The host-side serialization gap of 1,457 ms is real but
it is submission/readback latency between chunks, not
kernel time in the small ops.)

### Comparison against the small-batch (spec) picture

In the verify band (rows 2..7) the GEMV-class kernels
stream at up to about 676 GB/s (K18 verdict runs), and
this run's head GEMV confirms 649.5 GB/s at rows=1. The
batch-64 MMQ kernels manage 43.5 GB/s aggregate on the
same card in the same hour: the per-byte efficiency gap
between the decode-band kernels and the prefill kernels
is about 15x. Prefill does 64x the arithmetic per weight
byte and still cannot convert it into throughput, which
is what an occupancy floor does. Batch geometry is
therefore not the lever it looks like: raising chunk rows
from 64 to 128 would halve the 241 GB re-streaming, but
at 43.5 GB/s achieved the streaming is not what the
kernels are waiting on; total MMQ instruction work is
rows-invariant. Geometry pays only after the kernel
streams properly.

## Top three prefill costs and what each implies

1. **k13_q4k_mmq, 4,199.9 ms (63.1%).** Kernel shape.
   Register-saturated at VGPR 256 per the autopsy,
   streaming at 4.8% of ceiling and 17% of FP32 peak.
   The fix is inside the kernel: fewer live accumulators,
   a tile geometry that trades LDS/register pressure for
   occupancy, or a different dequant schedule. Nothing
   outside the kernel (batch size, chunking, submission)
   moves this number materially.
2. **k14_q6k_mmq, 1,342.0 ms (20.2%).** Same disease,
   worse ratio (8.8% of peak): Q6_K's heavier dequant
   costs more instructions per element under the same
   occupancy floor. Any k13 remedy must be proven on k14
   too; a Q4_K-only fix leaves a fifth of prefill behind.
   Its tiny 256-wide shape (186 ms at 3.2 GB/s) is a
   separate fusion fix.
3. **k6_attn_flash, 1,013.1 ms (15.2%).** Context
   scaling. Fine at 1k tokens, dominant at long context
   by construction. The implication is sequencing: MMQ
   first (it is 83% now), but the flash work (KV traffic
   layout, tile size over stored keys) is the next
   project, and at David's 131k context target it is
   not optional.

## Ranked attack options, with ceilings

Ceilings are arithmetic on the measured table: GPU total
6,659 ms, wall 8,116 ms, 1,022 tokens. "Ceiling" means
the named component reaches the stated rate and
everything else stays exactly as measured.

1. **MMQ kernel rework for occupancy** (both quants:
   cut VGPR pressure, retile, reschedule dequant). If
   the pair merely streams at the GEMV-proven 650 GB/s,
   its 5,542 ms becomes 371 ms; GPU total 1,488 ms;
   even with today's host gap untouched, wall about
   2,945 ms, about 347 tok/s. At a more conservative
   300 GB/s: GPU 2,266 ms, wall about 3,723 ms, about
   274 tok/s. This is the only option that moves the
   headline by multiples.
2. **Remove the per-chunk host serialization** (submit
   chunk chains without a blocking logits readback per
   chunk; only the final chunk needs its logits back).
   Ceiling: the whole 1,457 ms gap; wall falls to the
   GPU total, 125.9 to 153.5 tok/s with zero kernel
   changes. Composes with every other option.
3. **Coopmat (WMMA) MMQ for prefill only** (the research
   plan's prefill-only coopmat step, dequant to f16
   tiles): the high-ceiling variant of option 1. At
   even 10% of the card's fp16 WMMA rate the MMQ math
   itself is under 600 ms; the binding constraint
   becomes the dequant feed, so the honest ceiling is
   the same 650 GB/s streaming bound as option 1, about
   347 tok/s wall, with more risk and a second code
   path to keep correct. Ranked below the plain rework
   because its ceiling is not higher, only its risk is.
4. **Flash attention rework** (KV layout / tile over
   stored keys): halving flash saves 507 ms now (GPU
   6,152 ms, about 166 GPU-bound tok/s) and is worth
   more every token of context beyond 1k. At 4k tokens
   this is plausibly the #1 lever; measured here only
   at 1,022, so its ranking is context-dependent by
   measurement, not by guess.
5. **Fuse the tiny Q6_K KV-side dispatches** (416
   dispatches, 186 ms at 3.2 GB/s) into their neighbour
   projections or one batched dispatch. Ceiling: the
   full 186 ms, 2.8% of GPU time. Small, safe, and
   independent.
6. **Raise chunk rows 64 -> 128+**: halves weight
   re-streaming (241 GB to 121 GB) and per-chunk fixed
   costs, but the kernels are not stream-bound, so the
   expected gain at today's kernel shape is the
   dispatch-count savings only, low single digits of
   percent. Listed last on arithmetic, and it becomes
   valuable only after option 1 lands.

What llama.cpp's 998.43 tok/s requires, for scale: a
1,024 ms total wall for this prompt. Options 1 + 2 at
their ceilings give about 2,945 ms; closing the rest is
flash (option 4) plus rates above 650 GB/s on the MMQ
stream, which no kernel on this card has demonstrated
yet. Stated plainly so the gap is a work list, not a
mystery.

## Caveats

- One model, one prompt shape, one context length
  (1,022). The flash ranking in particular must be
  re-measured at longer contexts before it is sequenced.
- GPU clocks on this host are not locked; run-to-run
  agreement was 0.5%, so the table's ordering is solid
  and its second decimal is not.
- The harness forces a logits readback per 64-token
  slice; the shipped forward pays the same per-chunk
  readback (its off wall matches the harness's), so the
  host gap is a shipped property, not a harness one.
- Raw per-dispatch records for both timing runs are on
  the server at ~/gpu-wave9/prefill-profile-out/
  (prefill-on-1.csv.records.csv and
  prefill-on-2.csv.records.csv) and were fetched to the
  profile lane's
  scratch for this analysis.
