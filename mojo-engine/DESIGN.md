# Mojo/MAX engine: design

Status: Waves A, B, and C landed, 2026-11-07/08, branch
`mojo-max-engine`, engine version 1.4. Wave B is CLOSED: the
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

1. **Front end (later, optional):** the existing Go platform, or
   MAX Serve. The serving seam is a later decision; the engine
   does not care which one calls it.
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
   materialized score row); shared-memory K/V tile staging
   and split-KV parallelism are still owed.
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
   device-side visible-order KV gather (the GPU path
   refuses windowed appends that would wrap the 2,048 ring;
   contexts past the window stay on the CPU path),
   attention shared-memory tiling / split-KV occupancy work,
   activation residency (activations cross per op in this
   wave's shape), and GEMV tuning beyond K1.
6. **Sampler. GREEDY LANDED (Wave B, Lane B4).** First-max
   tie-break, EOS excluded per engine.rs. The
   temperature/top-k/top-p family is still to come.
7. **Serving seam.** Socket protocol parity with the paused
   platform, or MAX Serve; decided when step 6 lands.

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
