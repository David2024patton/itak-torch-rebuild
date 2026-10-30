# BACKEND: itak-backend scoping (wgpu on Vulkan, RX 7900 XTX)

Status: scoping document for TODO T-4.1 (Backend trait plus CPU SIMD
backend), T-4.2 (wgpu backend on Vulkan), and the hardware detection
proof inside T-9.5. ARCHITECTURE.md Section 3.1 owns the module
contract (`Backend` trait; `detect() -> HardwareReport`; depends on
itak-kernels and itak-diag). This document owns the internals: which
kernels, in which order, with what memory strategy, what happens when
the GPU path is not available, and the rule that would move us from
wgpu to raw Vulkan (ash). DECISIONS 2026-10-26 stands: ROCm and HIP
are parked; Vulkan is the GPU path.

## 1. Ground truth (measured 2026-10-28, bench/ and TODO T-0.2, T-P.2)

| Fact | Value |
| :--- | :--- |
| Card | AMD Radeon RX 7900 XTX (Navi 31, RDNA3), 24 GiB |
| Memory bandwidth, hardware | 960 GB/s |
| llama.cpp Vulkan decode, Glimmer Q4_K_M | 44.31 tok/s = 708 GB/s achieved (73.8 percent of peak) |
| llama.cpp Vulkan decode with DFlash | 88.6 effective tok/s |
| HIP decode, same build | 41.13 tok/s (Vulkan keeps generation; HIP won prefill only) |
| Ceiling at this quant | 60.0 tok/s (14.89 GiB streamed per token) |
| Specialized target (ARCHITECTURE Section 6) | about 51 tok/s (85 percent of peak) |

Decode is a streaming problem: each token reads essentially every
weight byte once. Kernel work is therefore judged in GB/s first and
FLOPs never. 708 GB/s is the number to match; 960 GB/s is the
hardware wall nobody crosses.

## 2. Adapter probe (run 2026-10-29, on the server, wgpu 27.0.1)

A minimal wgpu probe (source in the Appendix) was built on this
workstation, copied to the server, and run there. It enumerates
adapters and prints the selected adapter's name, backend, and limits.
Enumeration only: it dispatched no work and touched no running
service. Full output condensed:

| # | Adapter | Backend | Type | Notes |
| :--- | :--- | :--- | :--- | :--- |
| 0 | AMD Radeon RX 7900 XTX (RADV NAVI31) | Vulkan | DiscreteGpu | vendor 0x1002, device 0x744c, RADV Mesa 26.0.8, Vulkan API 1.4.335 |
| 1 | Intel(R) UHD Graphics 770 (ADL-S GT1) | Vulkan | IntegratedGpu | the host iGPU |
| 2 | llvmpipe (LLVM 21.1.8, 256 bits) | Vulkan | Cpu | software rasterizer, always present with Mesa |

Selected under `PowerPreference::HighPerformance`: adapter 0, the
7900 XTX. Limits that shape the design, as granted by that adapter:

| Limit | 7900 XTX via wgpu | Design consequence |
| :--- | :--- | :--- |
| max_buffer_size | 2,147,483,647 (2 GiB minus 1) | No single weight buffer: Glimmer is 15.59 GiB. One storage buffer per tensor (largest single tensor, the Q5_K output head, is about 0.86 GiB and fits) |
| max_storage_buffer_binding_size | 2,147,483,647 | Binding slices are not a constraint on this stack; any tensor binds whole |
| max_compute_workgroup_storage_size | 65,536 (64 KiB LDS) | The 6656-wide hidden vector stages in LDS as f32 (26,624 B). The 19968-wide FFN intermediate does not (79,872 B f32); it fits as f16 (39,936 B) or the down projection reads it from L2 instead (Section 4) |
| max_compute_workgroups_per_dimension | 65,535 | One workgroup per output row cannot cover the 202,048-row output head in one dispatch dimension; kernels use two-dimensional dispatch or grid-stride loops |
| min/max subgroup size | 32 / 64 | RDNA wave32 is the native reduction unit; subgroup operations are available (Features::SUBGROUP present) |
| Features present | SUBGROUP, TIMESTAMP_QUERY, SHADER_F16 | Subgroup reductions, in-engine timestamp timing for the bench plan, and optional f16 staging all exist on this stack |
| max_bind_groups / dynamic storage buffers per layout | 8 / 8 | Ample for a matvec pipeline (weights, input, output, parameters) |

Two warnings the probe makes concrete:

1. **Selection is a policy, not a default.** This host exposes three
   Vulkan adapters. HighPerformance happened to pick the right one,
   but that is driver behavior, not a contract. `detect()` must rank
   DiscreteGpu on Vulkan first, reject `device_type: Cpu` outright
   (llvmpipe would report success while running slower than our own
   CPU backend, and its binding cap is the 128 MiB Vulkan floor), and
   record the chosen adapter in the HardwareReport and the diag
   channel.
2. **Request the adapter's limits, not wgpu's defaults.** A device
   created with wgpu's default limits caps buffers far below what
   RADV grants. `detect()` requests the adapter limits at device
   creation and verifies the granted limits against the kernel
   requirements; a shortfall is a fallback event (Section 6), never a
   surprise mid-load. Cross-check: sysfs
   `/sys/class/drm/card2/device/mem_info_vram_total` reads
   25,753,026,560 bytes (24,560 MiB) on this host, and the
   HardwareReport carries both the sysfs number (the fix for the old
   nvidia-smi-only detection bug) and the wgpu adapter identity; a
   disagreement between them is itself a diag event.

## 3. Kernel priority for decode

Order is by bytes streamed per token on Glimmer, then by position on
the critical path. All are fused dequantize-and-multiply GEMV kernels:
quantized blocks are expanded in registers and never materialized as
floating-point weights anywhere.

| Priority | Kernel | Glimmer shapes and share | Why this order |
| :--- | :--- | :--- | :--- |
| K1 | Q4_K dequant GEMV | FFN gate and up (6656 to 19968), attention Q and K projections, embedding; 77.3 percent of all bytes | The bulk of the stream. This kernel's GB/s is the backend's GB/s |
| K2 | Q6_K dequant GEMV | FFN down (19968 to 6656) and attn_v, alternating by layer half; 17.1 percent | Second-largest stream, and it consumes the FFN intermediate, so its input strategy (Section 4) is coupled to K1's epilogue |
| K3 | Q5_K dequant GEMV | The untied output head (6656 to 202,048); 5.5 percent of bytes | Small share but runs every token and produces the logits; the old loader could not even read this type. Includes the dispatch-dimension workaround (202,048 rows) |
| K4 | RMSNorm fused with residual add | 6656 wide, every layer, twice | Bytes are trivial; the point is fusion so the hidden vector is not re-read from VRAM between kernels |
| K5 | RoPE plus KV append | Q and K vectors, head dim 128 | Fused into the projection epilogue where the pipeline allows, else standalone; writes K and V into paged (global layers) or ring (sliding layers) slots in the same pass |
| K6 | GQA decode attention | 32 Q heads over 2 KV heads, head dim 128; ring variant (window 2048) and paged split-K variant for global layers | Attention reads about 78 MiB per token for the 39 sliding layers and up to 1.66 GiB per token for the 13 global layers at the full 131K context. Second-order at chat lengths, real at long context; split-K with online softmax and subgroup reductions |
| K7 | SwiGLU elementwise | 19968 wide | An epilogue of the up projection writing the activated intermediate, in f16 when the down projection will stage it in LDS |
| K8 | Embedding gather, logit softcap and scale | One row of 6656; 202,048 logits | Trivial kernels; the softcap (20.0) and scale (0.1961) come from the arch module and fuse into the K3 epilogue |

After decode: prefill GEMM variants of K1 to K3 (same block decoders,
M equal to the prefill chunk), and GPU kernels for the remaining
types in the ARCHITECTURE 5.2 table (Q8_0, Q5_0, Q8_1, Q4_0, Q4_1,
Q2_K, Q3_K, F16). Until a type has a GPU kernel it runs through the
per-op CPU fallback, disclosed at load time (Section 6). Sampling
stays on CPU at first: the logits vector is 0.8 MB, and moving it is
cheaper than a GPU top-k until batch sizes say otherwise.

Every kernel ships with its CPU SIMD reference beside it (T-4.1) and
a conformance test comparing GPU output against that reference on
real tensor data, per the repo rule that debugging and lint gates are
part of every module from birth.

## 4. Workgroup and memory strategy for a bandwidth-bound GEMV on RDNA3

- **Weights stream exactly once; activations are free by comparison.**
  A step's weight traffic is 14.89 GiB against kilobytes of
  activations. Every choice optimizes sequential read throughput:
  consecutive threads read consecutive packed words (vec4 loads of
  the block stream), and no kernel ever writes a dequantized weight.
- **Multiple output rows per workgroup.** A workgroup computes a
  small tile of output rows (4 to 8) for one projection. The
  activation vector is loaded into LDS once and reused across the
  tile, and one pass over the activation serves every row in the
  tile; weight bytes are untouched by this choice. Workgroup size 64
  or 128 (two to four wave32s); the exact tile is a tuning constant
  per kernel, recorded with its bench result, not a hidden default.
- **Dequantize along super-block boundaries.** Work is partitioned so
  a super-block's scale and min values are loaded once per workgroup
  tile and applied to every row's copy of that block in registers.
- **The down projection is the special case.** Its input is the
  19968-wide intermediate, which exceeds LDS in f32. Two candidate
  strategies, decided by the K2 bench: stage the intermediate in LDS
  as f16 (39,936 B, SHADER_F16 is present), or leave it in global
  memory where the 96 MiB Infinity Cache holds it trivially and skip
  staging. Split-K with a partials buffer is the third option if
  neither saturates; WGSL atomics are integer-only, so split-K
  reduction is a second small kernel, never an atomic shortcut.
- **Small-batch, not batch-1-only.** The engine serves multiple
  slots; kernels take M (tokens this step) as a parameter with the
  GEMV as M equals 1. Weights stream once per step regardless of M,
  so slot batching divides weight traffic per token. The CPU
  reference defines correctness for every M the GPU path accepts.
- **Per-tensor buffers, prebuilt bind groups.** Buffers are created
  once at load (Section 5). The per-token path records dispatches
  only: no allocations, no uploads except the step's token rows and
  KV appends, no pipeline creation (all pipelines compile at load,
  which also moves Naga translation cost out of the token loop).
- **Timestamps on.** TIMESTAMP_QUERY is present; kernels carry
  optional timestamp writes so the bench plan and later profiling
  read per-kernel time from inside the engine instead of guessing.

## 5. VRAM residency across the Go and Rust process split

- **The engine process owns the GPU, exclusively.** Only itak-engine
  creates a wgpu instance and device. The Go platform never links
  Vulkan, never opens the device, and never sees a tensor: the seam
  carries requests, token events, and health. A platform restart
  cannot disturb GPU state; an engine crash takes its device with it
  and the platform restarts the engine within its budget
  (ARCHITECTURE Section 4 failure semantics), after which weights
  re-upload from the mmap'd file. Cross-process device sharing and
  external-memory tricks are a non-goal.
- **Residency is all-or-nothing per model.** At load, the backend
  uploads every tensor of the model (plus companions: DFlash 1.51
  GiB, mmproj 1.30 GiB) into per-tensor storage buffers that live
  until unload. For Glimmer: 15.59 GiB of weights, 2.81 GiB of
  companions, about 1.7 GiB of KV at the full 131K context, plus
  workspace; roughly 20.5 GiB against the card's 24 GiB. If the
  budget does not fit (budget is sysfs VRAM total minus a configured
  reserve), the load refuses by name with the byte counts. There is
  no partial residency and no host-memory spill path: a weight that
  streams from system RAM mid-token silently destroys the bandwidth
  the design exists to protect.
- **Accounting is cross-checked.** The coverage report records
  per-tensor device bytes from the backend, and the load verifies
  the total against the sysfs VRAM-used delta. Weights are immutable
  after upload; LoRA adapters, when they arrive, are separate small
  buffers, never edits to resident tensors.
- **Coexistence is a rollout concern, not a backend feature.** The
  production llama-server holds VRAM on this host today. The backend
  reports free VRAM from sysfs before a load and refuses if the
  budget fails; pausing or restarting another process is the bench
  harness's etiquette under David's rules, never something the
  backend does on its own.

## 6. Fallback ladder

Per ARCHITECTURE Section 4: any fallback is an explicit, logged,
queryable state, never a silent one. Every transition below emits an
itak-diag event with the listed code (codes proposed here, to be
registered in itak-diag when the crate lands) and is visible in the
engine health the platform reads over the seam.

| Level | State | Trigger | Diag event |
| :--- | :--- | :--- | :--- |
| L0 | wgpu Vulkan on the discrete AMD adapter | Normal operation | `backend.selected` at startup (adapter name, backend, sysfs VRAM) |
| L1 | wgpu Vulkan on a non-discrete adapter | Only when configuration explicitly allows it | `backend.selected` with the adapter named; never entered by preference drift |
| L2 | Per-op CPU execution for a quant type with no GPU kernel yet | Decided at load from the coverage report, listed per (op, type) in the load result | `backend.op_fallback_cpu` per pair, at load |
| L3 | Device recreation, once | wgpu device-lost callback (driver reset, GPU hang) | `backend.device_lost`, then `backend.device_recreated` or escalation |
| L4 | CPU SIMD backend for everything | No suitable adapter, granted limits below kernel requirements, L3 recreation failed, or configuration disables the GPU | `backend.fallback_cpu` with a reason field |

A software adapter (llvmpipe) counts as no adapter: it lands on L4,
not on a pretend GPU level. The CPU SIMD backend is the correctness
reference for every kernel, is always compiled in, and is a
supported serving mode in its own right (T-4.1's proof: the engine
serves with the GPU backend disabled).

## 7. Measurement plan (against the 708 GB/s reference)

Correctness gates speed at every tier: a tier's numbers count only
with verified text (temperature 0 output matching the reference
engine on the bench prompt set) at the same commit, per ARCHITECTURE
Section 6. All results report effective bandwidth (tok/s multiplied
by 15.99 GB streamed per token at this quant) beside tok/s, at least
three runs, median reported, with card clocks, driver version, and
VRAM state recorded.

| Tier | What runs | Pass mark |
| :--- | :--- | :--- |
| 0 | This adapter probe, rerun as a smoke check on driver or wgpu changes | The 7900 XTX is selected and granted limits match Section 2 |
| 1 | Stream micro-bench: a read-mostly kernel over Q4_K-shaped buffers totaling one model's stream (14.89 GiB across per-tensor buffers) | Establishes the wgpu stack's streaming ceiling. Below 637 GB/s (90 percent of llama.cpp's 708) after workgroup tuning triggers the ash evaluation in Section 8 |
| 2 | Per-kernel benches at exact Glimmer shapes (K1 gate and up, K2 down, K3 head), timestamps from inside the engine | Each kernel's GB/s recorded; the FFN pair and the head set the expectations for Tier 3 |
| 3 | Model level: T-4.2's bench table (CPU backend, wgpu backend, llama.cpp Vulkan from Phase 0), then T-9.5 with DFlash against the 88.6 baseline | Speed ladder steps in order: verified text on wgpu, then 44.31 tok/s parity, then toward 51 |

## 8. Risks, and the rule for switching to raw Vulkan (ash)

**wgpu's limits on this stack, as measured.** The probe retired the
scary version of this risk: buffer and binding caps (2 GiB) clear
our largest tensor, subgroups and f16 and timestamp queries are all
exposed on RADV, and LDS is a full 64 KiB. What remains is real but
narrower: the 65,535 workgroup dispatch cap forces grid discipline
in K3; wgpu caps are per-device defaults that must be raised
deliberately (Section 2); and wgpu adds a translation layer (WGSL
through Naga to SPIR-V, then RADV's compiler) whose codegen we do
not control instruction by instruction. A Naga or driver update can
move performance without any change in our code, which is why Tier 1
is the standing tripwire on dependency and driver bumps.

**Shader authoring cost per quant type.** Each K-quant layout is a
different block decoder: Q4_K packs eight sub-block scale and min
pairs into a 256-weight super-block, Q5_K adds a separate high-bits
plane, Q6_K carries sixteen int8 scales under one f16 super-scale.
The GEMV skeleton (tiling, LDS staging, reduction) is shared; the
decoder is not. Budget half a day to two days per type including
validation against the CPU reference. The decode-critical set is
three types (Q4_K, Q5_K, Q6_K cover Glimmer completely); the rest of
the 5.2 table can arrive one at a time behind the L2 fallback, each
announced at load. Shape specialization multiplies testing, not
shaders: tile sizes are tuning constants, and only the two FFN
shapes and the output head earn dedicated tuning.

**The decision rule for ash.** Evaluate a raw-Vulkan path behind the
same Backend trait if, and only if, at least one trigger fires:

1. Tier 1 lands below 637 GB/s after workgroup tuning (the wgpu
   abstraction is costing more than 10 percent of the bandwidth
   llama.cpp already achieves on this card), or
2. a capability gap blocks a required kernel on RADV through wgpu
   (a limit that cannot be raised to the adapter's real value, a
   missing feature the kernels genuinely need), or
3. a per-kernel gap above 5 percent against hand-tuned expectations
   is traced to Naga codegen and cannot be closed from WGSL.

If a trigger fires, the evaluation is one prototype: K1 in ash at
Glimmer shapes, adopted only if it beats the wgpu kernel by more
than 5 percent at Tier 2 or removes the blocking gap. Kernel callers
sit above the Backend trait either way, so the swap, if it ever
happens, is local to this crate. Until a trigger fires, wgpu stays:
it removes hand-rolled synchronization and descriptor management
from a one-card product, and the measured limits say it is not the
bottleneck standing between us and 708 GB/s.

## BACKEND-RESULTS: K1 first landing (2026-10-29, crate `wgpu-compute`)

The wgpu member of the itak-backend family landed as its own
crate beside `cpu-compute` (the way the CPU member landed for
T-4.1): `crates/wgpu-compute`, implementing `muse_glimmer::Compute`
with device selection per Section 2 and kernel K1 real. K2 to K8
sit in their own modules under the crate's `kernels` map, and
every step that needs one refuses through that module with a
named `StepFailed`; nothing falls back silently inside a step
(Section 6's L2 composition stays a caller decision).

**Correctness.** Synthetic full-row tests compare the GPU result
against the `gguf` crate's dequantizer plus `torch-math` within
reassociation noise, and the `k1_check` example ran the real
file's `blk.0.ffn_gate.weight` ([6656 x 19968], 74,760,192 bytes)
through `proj` on the server: max relative error 2.98e-7, cosine
1.000000000 against the CPU reference. The Q6_K refusal through
K2 and the cached-name geometry refusal are pinned by tests.

**Bench method.** Server, RX 7900 XTX via RADV Mesa 26.0.8, wgpu
27.0.1. The bench (`k1_bench` example) holds real Q4_K tensors
resident and dispatches the GEMV over all of them per iteration:
6 tensors, 167,731,200 bytes streamed per iteration, 2,000 timed
iterations after 50 warmup, median of 3 runs. Short windows
mislead on this card: an early 25-iteration run (a 68 ms window)
read 61.4 GB/s before clocks ramped, so no number below comes
from a short window; the single-tensor probe's timed window is
about 0.2 s at its speed, the resident-set runs about 1.5 to
2.3 s.

| Measurement | GB/s | Note |
| :--- | :--- | :--- |
| Stream probe (this grid and access pattern, no dequant math; `stream_probe` example) | 746.9 | The wgpu stack's streaming level on this card |
| K1 v1: scalar byte loads, one row per workgroup | 148.6 | First version |
| K1 v2: vec4 traffic, one lane per super-block (SHIPPED) | 216.5 | The kernel in the crate |
| K1 v3: Section 4 tile (4 rows per workgroup, activations staged in workgroup memory) | 147.6 | Built, measured, rejected: staging cost more than shared activation loads saved at these shapes |

**Ash verdict (Section 8 rule, computed).** The shipped K1 lands
at 216.5 GB/s, which is 30.6 percent of llama.cpp's 708 GB/s and
below the 637 GB/s trigger: by the rule as written, the ash
evaluation trigger FIRES for K1 after tuning. The evidence
complicates the trigger's premise, and the complication is
recorded rather than smoothed over: the stream probe through
the same wgpu stack, same grid, reached 746.9 GB/s (105 percent
of llama.cpp's number), so the gap is not the abstraction's
streaming cost; it is the dequant instruction mix inside the
kernel (nibble extraction, per-element conversion, and the
factored scale/min sums). Section 8's prescribed prototype (K1
in ash at Glimmer shapes) remains one valid next experiment; a
llama.cpp-style kernel restructure in WGSL (shared-memory
dequant staging, per-thread K slices) is the other, and the
probe says the stack would not be its ceiling. No ash work was
started under this landing; the trigger state is the record.

**Residency notes.** Nothing hit the 2 GiB buffer cap: the
largest K1 tensor is 74.8 MB, and the cap first matters at K3's
0.86 GiB head, which fits. The bench window itself was
constrained: the production llama-server held 25.35 GB of the
card's 25.75 GB VRAM (about 384 MiB free), so the bench ran with
a 160 MiB resident subset (still roughly twice the 96 MiB
Infinity Cache, so the stream is VRAM traffic) instead of the
full 14.89 GiB model stream. A full-residency Tier 1 run needs
the card to itself and is a scheduled measurement, not a bench
beside production. Coexistence is a first-order effect on this
host: later the same evening, with a second llama-server and
engine processes from parallel work on the card, the identical
bench binary read about 17 GB/s. Bench reports on this host
must state their window, as this one does.

## BACKEND-RESULTS: the K1 tuning pass (2026-10-30)

The v2 kernel (one workgroup per row, one lane per super-block)
was restructured in llama.cpp's own direction. What was adopted
from the ggml-vulkan sources on the server
(`~/llama.cpp/ggml/src/ggml-vulkan/`, read before writing):

- `mul_mat_vec_q4_k.comp`: sixteen threads cooperate on one
  super-block, each owning sixteen elements (two `u32` qs
  words) unpacked by byte-unpack into vec4s, with the
  scale/min arithmetic factored against per-sub-block
  activation sums.
- `ggml-vulkan.cpp`: on RDNA3 the row split for K-quants is
  `rm_kq` 2 rows per workgroup (4 is GCN only); the large
  variant runs subgroup size times four threads (128 here)
  with a hybrid subgroup plus shared-memory reduction.
- The tuned kernel is therefore: two rows per workgroup of
  128, sixteen threads per super-block, packed extraction of
  the twelve scale bytes into per-sub-block scale/min in one
  unpack per half, activation quads loaded once per
  super-block and shared by both rows and all sixteen lanes,
  subgroup butterfly reduction with an LDS finish (a plain
  LDS tree variant remains as the no-subgroup fallback,
  selected at device setup from granted features).

One directed idea was evaluated and rejected on paper, with
the reasoning recorded so it is not retried blind: shared
memory DEQUANT STAGING. A GEMV consumes every dequantized
value exactly once, so staging dequantized tiles in workgroup
memory buys a barrier and a round trip for zero reuse; scale
staging loses the same way, because all sixteen lanes extract
their scales in parallel in the same instruction slots one
staging lane would spend serially. ggml's GEMV stages nothing
either; its shared memory carries reduction partials, as does
the tuned kernel's.

### Correctness first (real tensor, before any bench counted)

`k1_check` on the real `blk.0.ffn_gate.weight` [6656 x 19968]
Q4_K against the `gguf` crate's CPU dequant plus matvec
(pass bar: max rel err below 1e-4 and cosine above 0.999999;
f32 reassociation in a different order lands around 1e-6, so
the bar leaves two orders of headroom and still catches any
layout error, which is off by whole dequantized values):

| Shape | Max rel err | Cosine | Verdict |
| :--- | ---: | ---: | :--- |
| v2 (legacy) | 2.980e-7 | 1.000000000 | pass |
| r1, r2, r4 (LDS reduction) | 1.341e-7 | 1.000000000 | pass |
| r2sg, r4sg (subgroup) | 1.267e-7 | 1.000000000 | pass |
| r2f16 | 5.869e-4 | 0.999999621 | FAIL |
| r4sgf16 | 5.870e-4 | 0.999999621 | FAIL |

The first check round caught a real defect in the new shader:
every llama shape read the qs plane one word early (the plane
starts at word 4, after `d`, `dmin` and the twelve scale
bytes); v2 passing while all new shapes failed identically is
what isolated it to the port, not the design.

The f16 result is the tolerance call the pass owed: the f16
shapes hold per-sub-block dots in f16 and combine in f32
(strictly tighter than ggml's f16 pipeline, whose running
temp is f16), and they still land at rel 5.9e-4 on the real
tensor, outside the crate's 1e-4 bar. f16 activations are
therefore NOT taken: the variants remain benchable under
explicit names, they are never selected by `shipped_shape`,
and the bench below shows the rejection cost nothing, because
both f16 shapes also measured SLOWER than their f32 twins on
this card (activation traffic was never the bottleneck once
the dequant mix was fixed).

### The two windows (read this before the numbers)

This host produced two disjoint bench windows in one night,
and the difference is the window, proven three ways: the
first landing's own v2 binary re-read 24.4 GB/s in the bad
window against its 216.5 from the good one; the dequant-free
stream probe read 12.8 GB/s in the bad window against 746.9
in a good one; and per-process GPU engine counters showed no
tenant computing during the bad window's matrix (the same
effect the first landing's residency note recorded at about
17 GB/s under parallel tenants). In the bad window (around
04:29 UTC, VRAM free 98 MiB, two Qwen3-Embedding llama-servers
and production resident) every kernel compressed toward a
~140 GB/s ceiling: probe 12.8, v2 24.4, and the llama shapes
137.2 to 141.0, medians of three at 6,000 iterations on a
93,929,472-byte subset. That matrix is kept in the bench log
and carries exactly one usable signal: inside one window, the
restructure beat v2 by about 5.7 times. No absolute from that
window is used anywhere in the verdict.

The healthy window opened minutes later (around 04:44 UTC)
after the co-tenants released the card (VRAM free 2,824 MiB,
production resident and answering `/health` before and
after). One caution for future bench work on this host:
`pp_dpm_mclk` read 96 MHz in BOTH windows, including during
816 GB/s runs that are physically impossible at that memory
clock, so the sysfs clock readout is unreliable evidence
here; a same-window stream-probe canary is the test that
counts, and a canary under roughly 100 GB/s means the
window, not the kernel, is the story.

### Healthy-window matrix (the verdict numbers)

Established harness discipline: the six largest Q4_K tensors
as the resident set (160 MiB budget, 167,731,200 bytes
streamed per iteration, the same set as the first landing),
6,000 timed iterations plus 200 warmup (longer than the
first landing's 2,000 plus 50), median of three runs, every
run within 0.3 percent of its siblings:

| Shape | Median GB/s | Note |
| :--- | ---: | :--- |
| v2 (legacy) | 342.6 | same kernel as the 216.5 landing; windows differ |
| r1 (1 row, LDS) | 713.3 | the restructure alone already clears 708 |
| r2f16 | 737.6 | tolerance-failed AND slower than r2 |
| r4 (4 rows, LDS) | 770.7 | |
| r2 (2 rows, LDS) | 779.0 | |
| r4sgf16 | 780.1 | tolerance-failed |
| r4sg (4 rows, subgroup) | 794.3 | |
| r2sg, workgroup 64 | 802.9 | spot check |
| r2sg, workgroup 256 | 808.9 | spot check |
| **r2sg, workgroup 128 (SHIPPED)** | **816.5** | shipped-default confirmation runs: 817.4 |

The winner is two rows per workgroup of 128 with the
subgroup reduction, which is also the configuration
ggml-vulkan itself picks for K-quants on RDNA3 (`rm_kq` 2,
large workgroup, hybrid reduction): the independent tuning
converged on the reference's own answer for this silicon.
`shipped_shape` selects r2sg when the device grants
SUBGROUP and r2 (same kernel, LDS reduction) otherwise; the
shipped default passed its own `k1_check` at rel 1.267e-7
and the synthetic GPU suite on the server (3 of 3).

What 816.5 GB/s is, stated plainly: 115 percent of
llama.cpp's measured 708 GB/s, 128 percent of the 637 GB/s
trigger line, and 85 percent of the card's 960 GB/s peak,
which is the efficiency band ARCHITECTURE.md sets for
specialized kernels. One honest scope note: 708 is
llama.cpp's full-model decode rate through a serving loop;
816.5 is a K1 micro-bench over the FFN-dominated resident
set, which is exactly the Tier 2 comparison this document
prescribes (kernels benched at Glimmer shapes against
llama.cpp's number). Within that prescribed comparison the
tuned WGSL kernel beats the reference.

### The ash verdict, restated on tuned numbers

This supersedes the first landing's computed verdict above,
which stood on the untuned v2 number. Section 8's three
triggers, against the tuned K1:

1. Tuned wgpu still under 90 percent of the reference (637
   GB/s): DOES NOT FIRE. 816.5 GB/s is over the line with 28
   percent headroom, and over llama.cpp's own number by 15
   percent.
2. A capability gap blocks a required kernel: DOES NOT
   FIRE. Subgroups, SHADER_F16 and timestamps are all
   exposed and exercised; nothing K1 needed was missing.
3. A gap over 5 percent traced to Naga codegen: DOES NOT
   FIRE. There is no gap; the WGSL kernel beat the
   reference path's measured number.

**Verdict: wgpu stays the only GPU stack. The raw-Vulkan
(ash) path is not needed, no prototype is started, and the
question is closed for K1.** Section 8 stays armed as
written for the kernels that follow: K2 and K3 take the same
per-kernel comparison when they land, and a future kernel
that lands under its line after tuning reopens the ash
question for that kernel on its own numbers.

### Follow-ups from the tuning pass

- TODO T-4.2 stays open, unchanged: the full Tier 1 bench
  table (a complete forward pass) still needs a scheduled
  window with the card to itself.
- Bench-window law, amended from this night's evidence:
  record VRAM free and the co-tenant list with every matrix,
  run a stream-probe canary in the same window, and distrust
  any absolute whose canary is under roughly 100 GB/s. The
  sysfs DPM readouts on this host are not evidence.

## Appendix: the adapter probe

Lives in scratch (/tmp/wgpu-probe), not in this repo: it is a
measurement tool, rerun when the stack changes. Dependencies: wgpu
27.0.1, pollster 0.4. Build on any Linux host, run on the host whose
adapters you want to see.

```rust
//! wgpu adapter probe: enumerate every adapter wgpu can see, print each
//! one's info and key limits, then print the adapter wgpu selects for a
//! HighPerformance request (the one itak-backend would get by default).
//!
//! Read-only toward the host GPU: enumeration opens the Vulkan driver
//! but dispatches no work. Intended to run on the server that hosts the
//! RX 7900 XTX (RADV), built elsewhere and copied over.

async fn run() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    println!("enumerated_adapters: {}", adapters.len());
    for (i, adapter) in adapters.iter().enumerate() {
        let info = adapter.get_info();
        let limits = adapter.limits();
        println!(
            "[{i}] name={:?} backend={:?} device_type={:?} vendor=0x{:04x} device=0x{:04x} driver={:?} driver_info={:?}",
            info.name, info.backend, info.device_type,
            info.vendor, info.device, info.driver, info.driver_info
        );
        println!(
            "    limits: max_buffer_size={} max_storage_buffer_binding_size={} ...",
            limits.max_buffer_size, limits.max_storage_buffer_binding_size
        );
        println!(
            "    features: subgroup={} timestamp_query={} shader_f16={}",
            adapter.features().contains(wgpu::Features::SUBGROUP),
            adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY),
            adapter.features().contains(wgpu::Features::SHADER_F16)
        );
    }

    let selected = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("request_adapter found no adapter");
    let info = selected.get_info();
    println!(
        "selected: name={:?} backend={:?} device_type={:?}",
        info.name, info.backend, info.device_type
    );
    println!("selected_limits: {:#?}", selected.limits());
}

fn main() {
    pollster::block_on(run());
}
```
