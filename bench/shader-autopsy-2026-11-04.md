# Shader autopsy, first pass: what ACO actually emitted

Step 1 of the streaming-rate research plan (actions 1 and 3):
before any new kernel is written, look at the compiled form of
every kernel family in the resident path. Tooling:
`bench/shader-autopsy.sh` (this lane). No engine code changed.

## Run

- Where: ubuntu-server, tree staged at `~/gpu-wave5` from
  `git archive` of main `bda7ff55ea57f4a764afddf9ba2e050c59b86916`;
  anchor files (session.rs, compute.rs, k18_small_batch.rs,
  spec_generate_gpu.rs) sha256-verified identical to the
  worktree before the run. Raw output:
  `/tmp/shader-autopsy/2026-11-05-bda7ff55/` on the server
  (dated 2026-11-05 UTC, 2026-11-04 EST).
- GPU: AMD Radeon RX 7900 XTX (RADV NAVI31, gfx1100).
- Driver: Mesa 26.0.8-1ubuntu0.3 (mesa-vulkan-drivers),
  Vulkan 1.4.335. Kernel 7.0.0-42. cargo/rustc 1.99.1.
- Dumps: `RADV_DEBUG=shaderstats,asm,cs` (compute stage
  only), private `MESA_SHADER_CACHE_DIR` per arm so every
  shader compiled cold. RADV's dump headers carry no shader
  name, so attribution is by family-isolated compiles
  (wgpu-compute's `shader_compile` tests, one family per
  run, identical dump env) cross-checked against the
  in-situ dump set of the real verdict run (100 stats
  blocks; the same signatures recur there).
- The dump arm of the verdict harness ran with the opt-in
  families ON (`WGPU_COMPUTE_SMALL_BATCH_MAX_ROWS=16`,
  `WGPU_COMPUTE_MMQ_WIDE_MIN_ROWS=32`) so K15/K16/K18
  compile in situ; the perf arms ran at shipped defaults.
  The family-isolation passes were run manually with the
  script's exact flags and are folded into the script as
  phase 3b.

Reported LDS is RADV's allocated figure; it differs from
the WGSL-declared staging constants by driver packing
(deltas from -1,280 to +960 bytes across families), which
is why the isolated compiles, not LDS alone, carry the
attribution. SGPRs read 108 for every shader in every
family: that is ACO's compute allocation, not a
per-kernel signal, and it is not used below.

## The table (family-isolated compiles, Mesa 26.0.8, gfx1100)

| Family | Shaders | VGPR | Spilled VGPR | Scratch (B) | LDS reported (B) | WGSL staging (B) | Code (B) | s_buffer_load per shader |
|---|---|---|---|---|---|---|---|---|
| GEMV K1 (Q4_K) | 9 variants | 48 | 0 | 0 | 1,024-4,096 | partials only | 1,744-5,824 | 2-3 |
| GEMV K2/K3 (Q6_K/Q5_K) | 14 variants | 48-72 | 0 | 0 | 1,024-2,048 | partials only | 1,852-6,316 | 3-8 |
| MMQ K13 (Q4_K, m64) | 1 | 256 | 0 | 0 | 41,984 | 41,216 | 31,824 | 7 |
| MMQ K14 (Q6_K, m64) | 1 | 256 | 0 | 0 | 44,032 | 45,312 | 34,544 | 7 |
| Wide MMQ K15/K16 | 9 | 256 | 1,041-1,213 | 78,080-86,528 | 41,984-62,464 | 41,472-66,048 | 72,576-82,844 | 17 |
| K17 DFlash | 1 main + 3 util | 24 | 0 | 0 | 1,024 | ~512 + scalars | 2,492 | 6 |
| K18 small-batch, tile 32 | 5 | 256 | 0 | 0 | 16,384 | 15,424 / 16,448 | 31,564-34,324 | 7 |
| K18 small-batch, tile 64 | 6 | 256 | 0 | 0 | 26,624 | 25,664 / 26,688 | 31,792-34,512 | 7 |
| GEMM K10/K11 | 9 | 12-60 | 0 | 0 | up to 16,384 in situ | n/a | 220-7,260 | small |

Static instruction features (per big shader): the MMQ and
K18 bodies each carry 1,024 `v_dot4` (the unrolled 4x4
sub-tile dot products) and roughly 600 `ds_load`/`ds_store`
(LDS operand traffic, by design). No shader anywhere uses
a 16-bit or byte global load (`buffer_load_b16`/ubyte
count is 0 in every dump); GEMV's weight stream loads are
`buffer_load_b128` (128-bit) in the unrolled body.

In situ, the real verdict run compiled K18's tile-64
variants only (6 shaders, LDS 26,624, VGPR 256): Glimmer's
tensor widths route tile-64 shapes; tile 32 appears only
in the isolated compile set.

## Verdicts per kernel

- **GEMV (K1/K2/K3): clean.** VGPR 48-72, no spills, no
  scratch, 1-4 KB of LDS (reduction partials only), weight
  loads at full 128-bit width, buffer descriptors fetched
  2-8 times per shader (once per buffer, at setup). The
  compiled form matches the measured behavior: this is
  the fast family and its codegen gives no reason to
  doubt the measurement.
- **MMQ K13/K14: register-saturated, not spilling.**
  VGPR 256 is the architectural per-wave maximum; ACO
  still kept everything in registers (zero spills, zero
  scratch) in 32-35 KB of fully unrolled code. Occupancy
  per CU is at the floor, which costs latency hiding on
  the prefill path, but there is no codegen defect.
- **Wide MMQ K15/K16: convicted.** Every one of the 9
  wide shaders spills: 1,041-1,213 spilled VGPRs and
  78-86 KB of scratch per shader, on top of VGPR 256 and
  42-62 KB of LDS. A kernel that reloads its own register
  spills from memory while trying to stream weights is
  moving far more bytes than the tensor holds. This is
  a compiled-form explanation, from the driver's own
  statistics, for the recorded wide-MMQ behavior (about
  3.5x slower per tensor than K13/K14, one session hang).
  The shipped default (wide out of the ladder) is the
  right call, now with evidence instead of just timings.
- **K17 DFlash: clean.** VGPR 24, no spills, 1 KB LDS,
  2.5 KB code. Nothing in the draft block kernel's
  compiled form is suspect; the draft's cost sits in the
  GEMV-form block forward around it, as measured before.
- **K18 small-batch: no codegen smoking gun; the ceiling
  is structural.** Zero spills, zero scratch, descriptors
  clean (7 fetches), dots and LDS traffic exactly as
  designed. What the stats do show is VGPR 256 at both
  tile sizes: each K18 wave carries the maximum register
  allocation, so resident waves per CU are few, and every
  super-block column pays an LDS stage plus barrier for
  operands the GEMV form reads straight from DRAM. Set
  against the measured flat 70-130 GB/s stream rate, the
  compiled form points at occupancy plus staging latency,
  not at a defect ACO introduced.
- **K10/K11 GEMM and the K4-K8 utility set: clean.**
  Small, unspilled, nothing to chase.

## GEMV vs K18, and the Mesa 15132 question

The research plan named Mesa issue 15132 as a suspect:
on that report, ACO clobbered loop-invariant buffer
descriptors and emitted about 25 percent more buffer
loads (and 184 percent more scalar loads) than the rival
compiler for the same Q4_K GEMV shader. Our dumps do
**not** show that pattern. In every family the buffer
descriptors are fetched a small constant number of times
per shader (GEMV 2-8, MMQ/K18 7, wide 17), true scalar
loads are 4-24 per shader, and there is no per-iteration
descriptor refetch in the unrolled bodies. One invariant
argument reload repeats 16 times in K18's body
(`s_load_b128` at offset 0x30); that is two orders of
magnitude below the 15132 scale and cannot carry the
measured 5-10x stream-rate gap. GEMV's dynamic load
count is at the width minimum by construction (one
128-bit load per 16 bytes of weight stream). For K18
and MMQ the global side is loop-carried staging, so
static counts cannot give a dynamic total; what can be
said is that nothing on the descriptor or scalar side
is inflated. The 15132 mechanism is cleared for this
codebase on this driver. What remains standing, from
the compiled form alone: register-ceiling occupancy in
every staged family, LDS staging latency by design,
and (wide only) the spill storm.

## Blocked: the speculative phase currently device-losts

All three `spec_generate_gpu` runs in this pass (dump
arm and both perf arms, shipped defaults included)
died at the same instruction: spec round 1, inside the
DFlash draft's `forward_block`, with RADV reporting
"The CS has been cancelled because the context is lost.
This context is guilty of a hard recovery", after which
wgpu panicked on the lost device (exit 101). Phase A
completed in every run: plain decode 35.62-37.63 tok/s,
64 chunks, the stream opening with the known golden
tokens [13796, 26, 1573, 373, 290, 2167, 26899, 1212].

This is not a code regression claim: the same harness
at this same SHA passed phase B four times on
2026-11-04 (bench/small-batch-2026-11-04.md, IDENTITY +
GOLDEN in all verdict runs). The changed variable is
the server's tenant mix. Other tenants now hold 6.54 GB
of VRAM (two Qwen3-Embedding servers on 8091, one
Qwen3-4B server on 8092), against about 2.7 GB during
the verdict runs. Phase B's footprint arithmetic:
6.54 GB tenants + 16.76 GB target weights + 1.63 GB
draft weights + KV and activation buffers is about
25.3 GB against the card's 25.75 GB. The working
theory, labeled as theory (dmesg needs sudo, which is
not available non-interactively): allocation pressure
pushes buffers into GTT, the draft block forward
overruns the amdgpu watchdog, and recovery kills the
context. The tenants were not touched; all three
llama-server processes were still alive after each
recovery. Consequence for this pass: no spec tok/s and
no verify-ms could be collected, for any arm.

## cswave32: one data point (phase A only)

`RADV_PERFTEST=cswave32`, shipped defaults, private
shader cache per arm, 64 tokens, depth 6 harness (the
spec phase never ran, above). Plain decode: control
35.70 tok/s, cswave32 35.62 tok/s, identical token
stream, identical dispatch count (28,480). No
measurable movement on the GEMV decode path. This is
a single data point on phase A only, not a verdict on
wave32 for the staged families; that A/B still needs a
phase B that survives, or a per-tensor bench.

## Fixture note

The script's first staged run also executed the
resident_session mixed fixture with K18 enabled (dump
arm env). `resident_session_matches_trait_and_cpu`
failed on exactly one assertion: the dispatch-count
pin at tests/resident_session.rs:398 (318 recorded vs
306 pinned for the shipped ladder). Every token
agreement check ahead of that pin passed, and the
other two resident tests passed. Enabling K18 changes
the graph's dispatch count by design, so the pin
tripping is expected, not a correctness finding. The
script now runs the fixture at shipped defaults.

## Limits of this pass, and the next step

Shader statistics and disassembly are static: they
convict spills, register ceilings, and load widths,
but they cannot time the LDS staging or count dynamic
loads. The dynamic confirmation is step 2 of the plan
(per-dispatch timestamps plus an SQTT capture with
`MESA_VK_TRACE=rgp` and L2 counters on the surviving
kernel), run when phase B can execute on this server
again. Nothing in this pass changes any default:
K18 stays off, wide MMQ stays out of the ladder, and
both stays are now backed by the compiled form, not
only by timings.
