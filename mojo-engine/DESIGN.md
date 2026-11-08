# Mojo/MAX engine: design

Status: Waves A, B, C, D, and E landed, 2026-11-07/08, branch
`mojo-max-engine`, engine version 1.6. Wave E is CLOSED:
the batch (GEMM) prefill step now runs the resident chain
and the tiled split-KV attention (E1: 1,022-token batch arm
34,852 ms against the per-row arm's 45,042 ms on the final
tree, inversion removed; routing rule per-row below 16
tokens, batch at/above, as a config knob), decode launch
overhead cut (E2: zero-fill launches removed, splits==1
direct-output attention; step+logits wall 35,230 -> 32,491
us steady state), and the Wave A/C hygiene debts paid (E3:
budget-ledger unreserve, harness verdict-line extraction,
GLIMMER_* defaults folded into EngineConfig); golden PASS
on the final tree, record in
`bench/mojo-engine-wave-e-2026-11-08.md`. Wave D (the speed
wave) is CLOSED: all four lanes integrated (D1 GEMM prefill,
D2 attention occupancy, D3 GEMV tuning beyond K1, D4
activation residency + device KV wrap), record in
`bench/mojo-engine-wave-d-2026-11-08.md`. Wave B is CLOSED: the
end-to-end golden passed on the real Glimmer file (first 8
generated tokens exactly [13796, 26, 1573, 373, 290, 2167,
26899, 1212]), rung 3 (teacher-forced) passed 8/8, and rung 2
was counted by an independent Python Float64 spot check.
Wave C is CLOSED: the same generation now runs on the GPU
(device-resident weights and KV, per-position step semantics
unchanged) and the end-to-end golden passed on device -
GOLDEN_FIRST8_MATCH 1 in two consecutive runs, all 16 tokens
identical to the CPU run, teacher-forced 8/8 on device with
winning logits matching the CPU rung-3 record to about
1e-3. First GPU numbers (no verdict beyond the arithmetic):
prefill 684 ms for the 6-token prompt (CPU: 1,411,755 ms),
decode 0.1511 s/step, about 6.62 tok/s (CPU: about 253.6
s/step). The full records are
`bench/mojo-engine-wave-a-2026-11-07.md`,
`bench/mojo-engine-wave-b-2026-11-07.md`, and
`bench/mojo-engine-wave-c-2026-11-07.md`.
This branch starts the engine over on Mojo and MAX, on David's
order ("pause rust and golang.. create a new branch and let's
start over with Max and Mojo"). The Rust and Go trees are paused,
not deleted: they stay on `main` as the proven reference this
engine must match, piece by piece, under the parity law below.

## The stack, top to bottom

1. **Front end:** the engine's own Mojo API endpoint and
   testing dashboard (step 7, decided Mojo-only by David on
   2026-11-08). The Go platform and MAX Serve were both
   considered and rejected on that ruling.
2. **Host engine: Mojo.** Model loading, KV cache management,
   scheduling, sampling, the generation loop. One language for
   host and kernels is the point of the restart.
3. **Kernels: Mojo.** MAX's own kernel library where it fits
   (its tuned matmul, attention, and MoE kernels are Apache 2.0);
   hand-written, hand-TUNED custom kernels where it does not
   (our quantized formats, Q4_K first). Custom does not mean
   naive: see the speed law.
4. **Plumbing: MAX.** The MAX runtime compiles Mojo to the
   target chip and feeds it work. No wgpu, no Vulkan layer, no
   ash in this stack.
5. **Vendor drivers.** AMD's HIP stack on the RX 7900 XTX
   (gfx1100, a listed MAX target, proven in-house 2026-11-07),
   NVIDIA's driver on NVIDIA cards. Every stack ends here; this
   layer is never ours to write.

## Build order, piece by piece

Each step lands as its own lane on this branch, proves parity
(step's law below), and only then counts.

1. **GGUF loader. DONE (Wave A, Lane A1).** Parses the real
   Glimmer file: header, metadata, tensor directory, staged
   load (Mojo 1.2's stdlib has no mmap binding; staged reads
   serve the same bytes). Proven: 731 tensors, metadata spot
   values exact, fixture tensor bytes hash-matched.
2. **Dequant + GEMV kernels, TUNED from the start. DONE
   (Wave A, Lanes A2/A3).** K1 tuned to the r2sg shape:
   624.1 GB/s median, 74.4% of the recorded wgpu median, under
   the within-10% line; the correctness gate (rel 1.267e-07)
   is the binding constraint, gap analysis in the wave note.
   Q6_K, Q5_K, F32/F16 kernels built and GPU-proven at the
   recorded bars (straight ports; tuning them is later work).
   The elementwise ops (norms, RoPE, SwiGLU, residual, softmax,
   final-logit transform, embedding gather) also landed
   (Lane A4), and the tokenizer (Lane A5, 16/16 prompt parity).
3. **Attention. DONE (Wave B, Lane B2), GPU-proven.** Scores,
   masking, the sliding-window and global layer split, a
   flash-style decoding kernel (online softmax, no
   materialized score row). **OCCUPANCY DONE (Wave D, Lane
   D2):** shared-memory K/V tile staging (tile 32) and
   split-KV parallelism (exact logsumexp merge, splits by
   stored count, span 512, cap 8) are the production GPU
   path; at stored 4,096 the tiled split launch measured
   486.9 us against the Wave B kernel's 3,138.4 us
   interleaved (6.45x), correctness inside the Wave B bars
   on every fixture including 4,096-position RoPE and
   NoPE; record in bench/mojo-engine-wave-d-d2-2026-11-07.md.
4. **One full layer, then all layers. DONE (Wave B, Lane
   B3).** The full 52-layer forward runs per position on the
   CPU path; rung 1 (synthetic, both layer kinds, Float64
   reference) passes, rung 2 (real layers 0 and 3) is counted
   by the independent Float64 spot check, and rung 3
   (teacher-forced golden) passed 8/8 on the real file.
5. **Generation loop. DONE (Wave B, Lanes B1 + B4); GOLDEN
   VERDICT PASS.** Prefill and decode over the KV cache; the
   end-to-end run on the real file (runner at 4dc4921)
   produced 16 tokens whose first 8 are exactly the recorded
   golden [13796, 26, 1573, 373, 290, 2167, 26899, 1212],
   decoded text " Paris. It is the most populous city in
   France and the most populous". First-light CPU numbers
   (no verdict): prefill 1,411,755 ms for the 6-token prompt,
   decode about 253.6 s per step (about 0.0039 tok/s);
   speed is a later wave's work.
   **GPU PORT DONE (Wave C, Lanes C1-C4); GOLDEN VERDICT
   PASS ON DEVICE.** The forward now also runs as a
   device-resident GPU model (src/gpu/ plumbing, GPU layer
   assembly, GpuGlimmerModel behind the same ForwardModel
   trait, runner selects it with MOJO_ENGINE_GPU): weights
   and KV resident on the card (20,933,677,056 B reserved of
   the 24,696,061,952 B budget cap at full context), the
   end-to-end golden passed on device twice, and the
   teacher-forced walk passed 8/8 on device. GPU numbers:
   prefill 684 ms, decode 0.1511 s/step (about 6.62 tok/s),
   model open about 31.8 s (CPU model read plus device
   upload). Still owed, recorded in the wave note: a
   device-side visible-order KV gather, attention
   shared-memory tiling / split-KV occupancy work,
   activation residency, and GEMV tuning beyond K1.
   **ALL FOUR LANDED IN WAVE D** (Lanes D4, D2, D4, and D3
   respectively): windowed layers now run past the 2,048
   wrap on device, layer activations stay resident across
   the layer's ops (1.91x/1.60x warm medians on the C2
   harness shapes), and Q6_K/Q5_K/F16 GEMVs are tuned
   (410.9 / 361.9 / 717.9 GB/s). Wave D also added the
   batched GEMM prefill path (Lane D1): one GEMM launch per
   projection for the whole prompt batch, GEMM outputs
   bit-identical to the GEMV kernels per row, 1,022-token
   prefill 1.83x on its lane tree (74,636 ms vs 136,467 ms)
   and 402 ms for the 6-token golden prompt against Wave
   C's 684 ms. Still owed after Wave D: activation
   residency INSIDE prefill (KV appends and causal
   attention still run per position there) - LANDED IN
   WAVE E (Lane E1: the batch step is resident and runs
   the batched tiled split-KV attention; prefill routes
   per-row below 16 tokens and batched at/above, by the
   MOJO_ENGINE_PREFILL_BATCH_MIN_TOKENS knob). The
   temperature/top-k/top-p sampler family remains owed;
   the serving seam (step 7) is decided and built.
6. **Sampler. GREEDY LANDED (Wave B, Lane B4).** First-max
   tie-break, EOS excluded per engine.rs. The
   temperature/top-k/top-p family is still to come.
7. **Serving seam. DECIDED (David, 2026-11-08): Mojo-only.**
   The endpoint lives inside the engine process, written in
   Mojo (`src/serve/` + `src/runner/serve_main.mojo`): an
   OpenAI-shaped POST /v1/chat/completions over the generation
   loop (greedy only in v1, locked by name), /health,
   /v1/models with per-file fact sheets read from each GGUF's
   own metadata and tensor signatures, /stats, /debug/log,
   /openapi.json, a JSON-fed testing dashboard at /, on-disk
   run history at /history with export/import bundles, a
   measured-only /leaderboard (llama.cpp and the paused Rust
   engine appear as marked reference rows), and an MCP-style
   bridge (/mcp/tools, /mcp/call, JSON-RPC subset at /mcp).
   Transport is pure Mojo over POSIX sockets via libc FFI.
   MAX Serve is rejected (Python-based, not Mojo) and socket
   parity with the paused Go platform is rejected (Go) on the
   same Mojo-only ruling. Proof record:
   bench/mojo-engine-api-endpoint-2026-11-08.md.

## The parity law

Every piece proves its answers against the Rust engine's
recorded references before it counts: same fixture, same bars
(cosine and max relative error as recorded per kernel; identity
and golden token checks at the loop level). A piece that cannot
show its numbers is not done, however clean it compiles. The
kernel test set the pattern: the straight Mojo K1 port recorded
cosine 1.000000000, rel 8.941e-08 against the CPU reference on
the real fixture tensor, tighter than the shipped wgpu kernel's
own 1.267e-07.

## The speed law

A straight port is a correctness proof, not a speed result. The
kernel test measured it: in one interleaved cleared-card window,
the tuned wgpu K1 ran a median 838.4 GB/s; the straight Mojo
port of the same math ran 39.8 GB/s, 21x slower, because it
recomputes dequant per element, loads a byte at a time, and has
no subgroup reduction. Tuning is therefore budgeted work in
every kernel lane's plan, never an afterthought and never a
surprise. No kernel is called fast on a port's numbers.

Measured state after Wave D (2026-11-08): K1 624.1 GB/s
(74.4% of the wgpu median; its correctness gate is the
binding constraint and the bar question stays David's open
call), Q6_K 410.9 GB/s, Q5_K 361.9 GB/s, F16 717.9 GB/s
(D3); decode on the D3 lane tree 9.09 tok/s against Wave
C's 6.62; prefill 402 ms for the 6-token prompt and
74,636 ms at 1,022 tokens on the D1 lane tree (1.83x over
the Wave C path interleaved); attention at stored 4,096
runs 6.45x the Wave B kernel (D2). Reference points on the
same card: the paused Rust engine's plain decode about
34.5 tok/s, llama.cpp Vulkan decode 44.59 tok/s - the
remaining decode gap is the next speed target, and its
named lever is prefill/decode activation residency plus
further kernel tuning, in that order of evidence.

Measured state after Wave E (2026-11-08): prefill
residency landed in the batch step; on the final tree
the 1,022-token arms read batch 34,852 ms (29.3 tok/s)
against per-row 45,042 ms (22.7 tok/s) interleaved, and
the 6-token prefill 217 ms (per-row route). Decode:
E2's profile showed the step is launch-bound (about
3,900 enqueues/step; isolated ops sum well above the
whole-layer wall); zero-fill launch removal and the
splits==1 direct-output attention cut the steady-state
step+logits wall from 35,230 to 32,491 us (28.38 ->
30.78 tok/s by that measure); the runner's approximate
decode read 20.35 tok/s on the final tree. The next
named decode lever is step-level activation residency:
the hidden row still crosses host<->device between all
52 layers, and the KV append's mid-layer synchronize
caps host lookahead.

## Licensing constraints (facts, as of MAX 26.7)

- Mojo the language: Apache 2.0 with LLVM Exceptions.
- MAX: the MAX License. Free and royalty-free, including
  commercial use, on any hardware. Conditions that bind a
  public engine: MAX ships inside a product in object code
  form with the MAX License included; modified MAX source may
  not be published except by forking to contribute back to
  Modular's project; MAX may not be used as AI input to
  reimplement MAX; selling MAX as a hosted inference service
  carries the Modular Accelerated branding condition and
  written approval through Modular's brand portal.
- Each MAX release keeps the license terms it shipped with;
  later term changes do not reach back into 26.7.
- Our own code stays ours, under our own terms.

## Modularity

Grid-and-tiles, as everywhere in David's projects: one module
per job under `src/` (config, diag, gguf/, kernels/, and the
modules the build order adds), each independently testable,
each with the diag seam wired and named refusals for bad
inputs from birth. Operational limits are config with defaults
(`src/config.mojo`), never compiled-in constants.
