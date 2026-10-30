# GPU resident forward: the device-resident session, measured (2026-10-30)

Follow-up to bench/gpu-proof-2026-10-30.md. That proof matched the
reference 8/8 tokens on the GPU trait path but decoded at 1.98 tok/s
against llama.cpp's 44.31: the kernels bench at llama.cpp bandwidth
in isolation (813 to 832 GB/s), so the composition (host Vec<f32>
per step, rotated K read back per layer per token, roughly a
thousand blocking submissions per token) was the cost. This note
records the fix and its measurements.

## Window

2026-10-30, roughly 07:25 to 08:05 ET, on ubuntu-server's RX 7900
XTX (card2). Production Glimmer stayed DOWN the whole window on
David's instruction (it stays down until this work is done). The
only other tenants were the two Qwen3-Embedding llama-servers
(pids 1315625, 1466491), never touched. VRAM baseline before the
runs: 111,964,160 bytes. Tree: local workspace copied to
~/resident-run on the server, release profile.

## What was built

`ResidentSession` in crates/wgpu-compute/src/session.rs, beside
the trait path (which stays as the executable reference):

- Weights upload once at session build (K1/K2/K3 weight handles,
  norm vectors dequantized to f32 buffers, embedding table in its
  file dtype). Session build for the real file: 3.1 to 7.0 s
  depending on page-cache warmth.
- The session owns the activation scratch (two ping-pong stream
  buffers), per-layer KV in `DeviceKv` (the storage half factored
  out of the trait path's `GpuKvState`), and the logits buffer.
- A forward chunk is ONE submission: embedding gather (K8
  multi-row variant, token ids read from a device buffer), the
  weightless embedding norm, then per layer the exact
  `GlimmerModel::forward_rows` step order (K4 norms with the
  fused post-norm-plus-residual entry closing each half-layer,
  K1/K2/K3 projections, K5 RoPE and append on sliding layers
  only, K6 attention, K7 gate and SwiGLU), the last-row copy,
  final norm, head, K8 finishing (scale then softcap), and the
  logits copy into the readback staging, all in the same
  command buffer. The dispatches compose into ONE compute pass
  per segment (chain, then head), not one pass per dispatch:
  the per-pass composition measured about 46 ms/token, the
  single-pass composition about 34 (see below). Argmax stays
  on the host: the session returns the finished logits, one
  readback per chunk, and nothing else crosses the bus.
- Submissions per token after the change: **1** (plus a KV
  growth copy submission on the rare chunks where a global
  layer's buffer doubles; 1 such aux submission across the
  whole 64-token run). Dispatches per token: 1,071, now inside
  the single submission. Readbacks per token: 1 (808 KB of
  logits).

## A correctness find: the chunked sliding-window ring

The first cross-path run at a 70-token prompt produced NaN
logits in BOTH GPU paths (the trait path included: it had only
ever been driven at 9 rows or fewer). Root cause: appending a
whole chunk before attending it evicts, from a window-sized
ring, keys that earlier queries in the same chunk still attend
to; a fully-masked query row then reduces to 0/0 in K6. The
fix is structural and lives in the shared K6 layer:

- The windowed ring capacity is now `window + MAX_CHUNK_ROWS - 1`
  for the resident session (the trait state keeps ring == window;
  its step-at-a-time driving never spans a chunk). With the
  headroom, every position any query in a chunk can see survives
  the chunk's appends.
- K6's single `window` parameter conflated the mask distance
  with the slot modulus. The shaders now take both: `window`
  masks (distance), `ring` slots (modulus). The trait path
  passes ring == window, so its behavior is unchanged (the
  cross-backend test still passes on the server).

## Proof: synthetic cross-path (server, release)

tests/resident_session.rs runs one synthetic model (the
cross-backend fixture: hidden 512, FFN 1024, vocab 1024, six
layers in a sliding/global mix, window 4, Q4_K weights, Q5_K
head) through the CPU reference, the GPU trait path, and the
resident session, comparing every token:

- Prompt of 4 plus 5 greedy steps (9 positions through a
  window of 4): top-1 agreement at every step across all three
  drivers; max-abs logit diffs at most 1.1e-6 (bar: 0.5).
  Session counters after this phase: 6 forwards, 6 chunks, 6
  submissions, 6 readbacks, 762 dispatches, 1 aux submission.
- A 70-token prompt (past the 64-row chunk cap, so prefill
  splits 64 + 6) plus one decode step: agreement again,
  diffs at most 1.1e-6. The session's reset() is exercised
  between phases.

## Proof: the real file (server, release)

Stage 1 prompt "The capital of France is"
(ids [200000, 954, 7963, 323, 11698, 373]), `--gpu`:

```text
generated_ids=[13796, 26, 1573, 373, 290, 2167, 26899, 1212]
generated_text=" Paris. It is the most populous"
```

Exact against the verified reference, 8/8. First-token
probability 0.403462, matching the trait-path GPU proof's
0.403462. The 64-token continuation stays coherent (city /
France / population text, same as the reference family's).

## Performance (server, release, medians of 3 discipline)

| Measurement | Value |
| --- | --- |
| Decode steady state, steps 8..64, median per-token | 34 ms = **29.4 tok/s** (two runs, medians identical; mean 33.0 to 33.2 ms) |
| Decode, tight-loop probe (argmax only between forwards) | 31.8 ms median = 31.4 tok/s |
| Prefill, 98-token SUMMARY golden prompt, first use (plan builds included) | 2,202 ms = 44.5 tok/s |
| Prefill, same prompt, warm plans (probe, reps 2 and 3) | 1,415 to 1,428 ms = **69 tok/s** |
| Prefill, 6-token Stage 1 prompt | 1,038 to 1,060 ms (rows=6 plan build included) |
| Peak VRAM during the 64-token decode | 17,533,157,376 bytes (17.53 GB) |
| Submissions per decode token | 1 (1,071 dispatches inside it) |

Time to first token, from the 64-token run's phase lines:
file open and parse 8,451 ms (cold page cache; 4,841 warm),
host weight load 4,629 ms, backend open 34 ms, session build
(weight upload) 3,087 ms, first forward 1,060 ms. First token
at about 17.3 s from process start cold, about 12.6 s warm;
in a persistent server the load terms are paid once at boot,
not per request.

## Where decode lands, honestly

- Trait path (previous composition): 1.98 tok/s. Resident:
  29.4 tok/s, a 14.8x improvement from composition alone.
- llama.cpp solo baseline: 44.31 tok/s. **Not beaten.** The
  byte-math ceiling at the isolated kernels' 818 GB/s and
  14.89 GiB streamed per token is about 51 to 53 tok/s
  (a 19.5 ms floor); the resident forward's median is 34 ms.
- What the remaining ~14.5 ms is made of, by measurement and
  arithmetic: (1) in-chain dispatch cost: 1,071 dispatches at
  an effective ~12 microseconds each of launch and barrier
  time inside one pass (the per-pass era priced the same
  chain's overhead at ~26 microseconds per dispatch: 46 ms
  total vs 34 after pass merging, with identical math);
  (2) host-side chain recording and the ~160 uniform rewrites
  per token, serialized before the submit (~2 ms); (3) the
  logits readback synchronization (~1 ms). The GEMVs
  themselves stream the same bytes the isolated benches
  measured; nothing in the run suggests they slowed down.
- Closing the rest is kernel-count work, not composition:
  llama.cpp runs this model in roughly 150 dispatches per
  token (fused norm-plus-projection epilogues, one attention
  kernel). The follow-on that buys the most: fuse the four
  per-layer K4 norm dispatches and the RoPE/append into their
  neighboring GEMVs, and give K6 a single fused
  scores-plus-reduce for decode rows. Prefill is bounded by a
  different property: the GEMV kernels only partially amortize
  weight reads across batch rows (69 tok/s warm at 98 tokens),
  so a GEMM-class (MMQ-style) prefill kernel is the prefill
  follow-on.
- Note the trait path's own latent limit found above: a
  single multi-row trait forward over windowed layers was
  only ever correct within the first window-span of positions
  (it NaN'd at 70 rows). The ring split does not change the
  trait layout (ring == window there), so trait multi-row
  forwards over windowed layers remain a decode-shaped tool;
  the resident session is the prefill path.

## Gates

- cargo test -p wgpu-compute: green locally (GPU tests skip
  with a printed reason; this VM has no suitable adapter) and
  the resident_session and cross_backend tests pass on the
  server's GPU in release.
- cargo clippy -p wgpu-compute --all-targets: clean.
- Workspace, `cargo test --workspace --exclude dflash --lib
  --tests`: 1,062 passed, 5 failed. The 5 failures are all in
  the sibling itak-engine lane (generate tests in
  crates/itak-engine/src/engine.rs and src/protocol.rs, e.g.
  run_generate_end_to_end_over_real_socket,
  generate_returns_prompt_echo_prefill_ids); they appear only
  under the workspace invocation's feature unification and
  the same package run solo passes 17/17. The dflash crate
  (also a sibling lane, uncommitted at this snapshot) does
  not compile its own tests yet and is excluded for that
  reason. Every crate this task touched or depends on
  (wgpu-compute, muse-glimmer, cpu-compute, attention, gguf,
  tokenizer, itak-loader) is green in its own run.

## Fusion pass (2026-10-30, later): 1,071 -> 445 dispatches per token

The kernel-count work named above, done in measured stages.
Every stage gated the same way: wgpu-compute suites green
locally and on the server GPU, clippy clean, and the
real-file Stage 1 + 64-token run exact (first-token prob
0.403462, generated ids matching the known-good stream
through all 64 tokens). Decode numbers are the bench
method's clean-run means over steps 8..64, 3 runs each;
medians run 1 ms coarser. Baseline for the pass: 1,071
dispatches/token at ~33.0 ms mean (~29-30 tok/s).

| stage | what fused | dispatches/token | ms/token (mean) |
| --- | --- | ---: | ---: |
| A | K9 attention prep: Q/K norms + RoPE + KV append, one dispatch (was K4 x2 + K5 x3 per layer) | 889 | ~32.7 |
| C+D+E | Batched QKV(+gate) GEMV (upload_multi concat), gate sigmoid fused into the output projection's activation (K1 gated form), FFN gate+up pair with SwiGLU fused into the down projection (K1/K2 act forms) | 655 | ~28.4 |
| F | Flash attention for decode rows (scores + online softmax + V reduce in one kernel; chunk plans keep two-pass) | 603 | ~28.7 (within noise at 69-token context; the win grows with stored length) |
| G | K4 norm chains: post-attention norm + FFN norm in one dispatch, and the boundary chain (post-FFN norm + next layer's attention norm), the intermediate row register-resident | 500 | ~25.5 |
| batch3 | Partial QKV batch for the 26 layers whose V is Q6_K: Q, K, gate concatenated (gate is the third segment), V separate; K9 takes a separate V stride | 448 | ~25.1 |
| F6b | Head finishing (scale / tanh softcap) fused into the Q5_K head kernel's final store; bitwise identical on the server GPU | 447 | ~25.2 (within noise) |
| F6a+F6c | Embedding gather + weightless norm in one dispatch (bitwise identical); final norm folded into the last layer's close (the close emits the head input, the last-row copy disappears) | 445 | ~25.2 |

Final decode: **445 dispatches/token at ~25.2 ms/token,
about 39.7 tok/s**, against llama.cpp's 44.31 tok/s
(22.57 ms). The gap did not close; what remains is not
dispatch count in the fusion sense. Per-layer decode is
now 8 sequential kernels (Q4_K-V layers) or 9 (Q6_K-V
layers): batch, prep, flash, gated output, norm chain,
FFN pair, fused down, boundary chain. Every one consumes
the previous one's full output, so no two can merge
without cross-workgroup synchronization the platform does
not offer. The honest decomposition at 445:

- Weight + KV streaming floor: ~19.5 ms (14.89 GiB/token at
  the kernels' measured ~818 GB/s).
- The remaining ~5.7 ms rides on the ~445 kernel
  boundaries (the pass measured a consistent ~12-15 us of
  pipeline drain per dependent GEMV-class boundary, ~2 us
  for tiny kernels), plus ~1 ms readback sync and the
  host's per-token record/sample work inside the step time.

Named next levers, in order: (1) multi-row decode batching
(amortizes boundaries and weight streams across tokens;
out of scope for this pass, and it is the lever that
remains if the target must be beaten), (2) an in-chain
per-kernel timestamp profile to split the ~5.7 ms between
boundary drain and in-chain stream slowdown precisely,
(3) host-side overlap: an on-GPU argmax head so the
per-token readback shrinks from 808 KB of logits to one
index, letting recording of the next chunk start earlier.
Prefill spot-check after the pass: warm 64-row chunks run
at ~78 tok/s or better (258-token prompt, marginal rate),
at or above the 69 tok/s baseline; chunk plans keep the
two-pass attention by design.

## GEMM prefill path and on-device argmax (2026-10-30, later still)

**Prefill: K10/K11 tiled GEMM.** Past 16 rows, chunk plans
no longer replay the GEMV chain per prompt row. K10 (Q4_K)
and K11 (Q6_K) stage each super-block column's dequantized
tile in workgroup memory once and apply it to all 64 chunk
rows from per-thread 4x4 register sub-tiles, carrying K1's
gated and SwiGLU activation forms with the same K1ActGeom
addressing, so the plan's fused projections (gated output,
SwiGLU-fused down) dispatch through the GEMM unchanged.
Weights bind the K1/K2 uploads' one resident buffer per
tensor. Single-row decode keeps the fused GEMV chain,
untouched by construction; a device granting under 16 KiB
of workgroup storage builds no GEMM pipelines and the
session keeps its GEMV chunk plans (the fallback is read
from the backend, never substituted silently).

Per-tensor bench (examples/k10_bench, batch 64, real file,
two healthy windows; ms per pass over the tensor):

| tensor | GEMV replay | t64x256 | t64x128 | t32x128 (shipped) |
| --- | ---: | ---: | ---: | ---: |
| Q6_K ffn_down 19968x6656 (109 MB) | 6.4-6.8 | 5.15 | 3.18 | 2.99 |
| Q4_K ffn_gate 6656x19968 (74.7 MB) | 4.4-4.5 | 3.50 | 2.04 | 1.95 |

Session prefill (glimmer_generate --gpu, repeated-sentence
prompts, two runs each; totals include first-chunk plan
builds):

| prompt | t64x256 | t32x128 (shipped) |
| --- | ---: | ---: |
| 254 tokens | 5,062 / 4,957 ms (50 tok/s) | 3,495 / 3,534 ms (72 tok/s) |
| 1,024 tokens | 17,744 / 17,647 ms (58 tok/s) | 11,773 / 11,810 ms (87 tok/s) |

Marginal rate between the two lengths: 60 tok/s at
t64x256, **93 tok/s at the shipped t32x128**, against the
pre-GEMM warm marginal of ~78 tok/s and llama.cpp's 998.
The GEMM closes only a slice of the gap, and the reason is
measured: per-tensor effective stream is ~36 GB/s against
the ~818 GB/s the GEMV achieves at batch 1, so the kernel
is bound inside the workgroup at ~5.7 TFLOPS effective
(FP32 peak ~61), not at DRAM. Each staged f32 element is
re-read from workgroup memory at roughly one FMA per LDS
byte, staging barriers land every 128 staged columns, and
256-deep tiles measure worse than 128-deep at both widths
(more barrier-serial staging per workgroup, less
occupancy): the narrow tile wins on shape, not capacity.
Decomposition of the 10.75 ms/token marginal: ~72% the
GEMM projections at the rates above, ~28% the
non-projection chunk chain (two-pass K6 attention over the
chunk's queries and the stored context, prep, norms,
gather, dispatch boundaries).

Named next lever, with its evidence: an MMQ in
llama.cpp's actual sense. Quantize each activation block
to int8 on the fly and run the inner product on DP4A
integer dot instructions against the weights' integer
payloads, folding per-block scales back at the store.
That is the design that reaches 998 tok/s on this card;
the f32-tile GEMM here is the correctness-first stepping
stone, and its staging discipline, act forms, and session
wiring carry over. Secondary: a 4x8 per-thread sub-tile
halves staged-tile LDS reads per FMA (the register budget
allows it on RDNA3) for an estimated 1.5-2x on the
projection share. The 28% non-projection share wants its
own timestamp profile before anyone optimizes it.

Gate honesty: the first server run of the GEMM tests
caught a real bug, in the one-shot helpers rather than the
kernels: K10Pipeline::run and K11Pipeline::run encoded
their dispatch into a command encoder they never
submitted, so their readbacks mapped initial zeros (the
act-form paths and the session, which submit, were correct
throughout; no session proof ever touched the broken
helpers). Isolated by driving prepare with manual buffers
on the device, fixed with the missing submit, and the
test's tolerance re-based to the result set's magnitude
scale: the fixture's random f16 headers put outputs at
~1e3, where summation order alone is worth ~1e-2 absolute.

**Argmax tail.** K8 gains a two-stage reduction (128
workgroups of (value, index) partials, one finish
workgroup) under the host sampler's exact greedy rule:
maximum value, lowest index on an exact tie, a total
order, so the staged result equals a sequential scan on
any finite input. ResidentSession::forward_argmax runs
the identical chunk chain with the reduction appended to
the head segment; the per-chunk readback is 8 bytes
(winning index, winning logit) instead of the logits
vector's 808 KB at the real vocabulary, and SessionStats
counts argmax readbacks separately. Real-file proof:
--gpu-argmax's 64-token generated stream is byte-identical
to the logits-mode stream, the prefill winner's logit
matches to all printed digits (17.343929, prob 0.403462),
and the run's stats show 64 argmax readbacks and 0 logits
readbacks. forward is unchanged for sampling and top-k
callers. Synthetic gates: tests/k8_argmax.rs (bit-exact
winning logit against the host scan, engineered ties, tail
and all-negative cases, empty-input refusal) and the
resident argmax-tail test (two-chunk prompt plus a 12-step
greedy generation against forward plus the host pick).
