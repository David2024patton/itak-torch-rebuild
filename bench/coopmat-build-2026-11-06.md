# Coopmat prefill build: the device bridge and the Q4_K coopmat kernel (2026-11-06)

Lane: lane/coopmat-build, based on main 09b8d4a.
Specification: bench/coopmat-feasibility-2026-11-06.md
(the verdict: wgpu exposes no cooperative matrix support;
the one honest path is the bridge, create the VkDevice
with ash and VK_KHR_cooperative_matrix enabled, hand it
to wgpu via device_from_raw / create_device_from_hal,
feed hand-compiled SPIR-V through the passthrough path).
This note follows that note's server protocol in order
and stops at the first failed gate.

## Step 0: probe (server, 2026-11-06)

Method: vulkaninfo for the extension list, the feature
struct, and the supported stages; a minimal ash probe
(/tmp/coopmat-probe on the server, ash 0.38.0+1.3.281,
the locked version) for the cooperative matrix property
rows, which this vulkaninfo version does not print.

Device: AMD Radeon RX 7900 XTX (RADV NAVI31), Vulkan
API 1.4.335, subgroupSize 64.

- VK_KHR_cooperative_matrix: EXPOSED, extension
  revision 2.
- VkPhysicalDeviceCooperativeMatrixFeaturesKHR:
  cooperativeMatrix = true,
  cooperativeMatrixRobustBufferAccess = true.
- cooperativeMatrixSupportedStages: SHADER_STAGE_
  COMPUTE_BIT (count 1).
- Property rows: 14, all M=N=K=16, all scope SUBGROUP.
  Twelve are int8 forms (UINT8/SINT8 A and B to
  UINT32/SINT32 results, three of them with saturating
  accumulation). The two f16 forms:
  - A FLOAT16, B FLOAT16, C FLOAT16, result FLOAT16
  - A FLOAT16, B FLOAT16, C FLOAT32, result FLOAT32
- The row this build needs (f16 16x16x16, f32
  accumulate, compute) is the second f16 row. Present.

Toolchain: /usr/bin/glslangValidator, /usr/bin/glslc,
/usr/bin/spirv-dis, /usr/bin/spirv-val are all present
as system tools on the server (not only inside the
llama.cpp build tree).

### Spec-update addendum (llama parity lane, same day)

bench/llama-parity-specs-2026-11-06.md pins the exact
shader llama runs for prefill on this card family:
mul_mmq_cm1.comp, the INTEGER coopmat MMQ, not the
generic f16 mul_mm coopmat. Enablement in ggml is
VK_KHR_cooperative_matrix present + the device feature
cooperativeMatrix on (ggml-vulkan.cpp:4087-4093, 4556);
a subgroup-scope s8 x s8 to s32 16x16x16 shape sets
coopmat_int_support (:4740-4748); on RDNA3/4 the Q4_K,
Q5_K, and Q6_K by Q8_1 products run mul_mmq_cm1.comp
(:1877, :2481-2494) with BM 128, BN 128, BK 32 and
TM/TN/TK = 16 (mul_mmq_cm1.comp:89-100).

Consequence for this lane: the required property row
is the s8 x s8 to s32 16x16x16 subgroup row, not the
f16 row. The probe above recorded twelve int8 rows;
among them the required one is present in both
saturating variants:

- A SINT8, B SINT8, C SINT32, result SINT32,
  saturating accumulation 0, scope SUBGROUP
- A SINT8, B SINT8, C SINT32, result SINT32,
  saturating accumulation 1, scope SUBGROUP

The shader does not request saturating accumulation;
either variant satisfies the gate. The bridge's
runtime probe (step 1) checks for this row.

**Step 0 verdict: PASS.** Extension, required matrix
row (the integer row, per the addendum), and compiler
all present. Proceeding to step 1.

## Step 1: the device bridge

Built in crates/wgpu-compute/src/coopmat_bridge.rs,
wired into GpuContext::open in device.rs. Design, as
implemented:

- Two switches, deliberately separate. The bridge
  switch is WGPU_COMPUTE_COOPMAT_BRIDGE (exact value
  1, fail closed, the K22 resolver idiom); the routing
  switch WGPU_COMPUTE_COOPMAT arrives with the kernel
  in step 2. The bridge smoke in step 3 needs a bridged
  device with routing still off, which is why they are
  not one switch.
- Mechanism: the feasibility note costed creating the
  VkDevice by hand with ash and wrapping it through
  device_from_raw. The implementation uses wgpu-hal's
  own end-to-end form of the same mechanism: the hal
  adapter's `open_with_callback` runs hal's own device
  creation (its extension computation, feature
  selection, and queue setup are the stock ones), and
  the CreateDeviceCallback adds
  VK_KHR_cooperative_matrix to the extension list and
  chains VkPhysicalDeviceCooperativeMatrixFeaturesKHR
  with cooperativeMatrix = true into the create info;
  `open_with_callback` ends in device_from_raw
  internally, and Adapter::create_device_from_hal
  wraps the result. The bridged device differs from
  the stock device by exactly the coopmat extension
  and feature; nothing else about device creation is
  reimplemented. The callback runs synchronously
  inside the open call, so the chained feature struct
  (a local) outlives every use of the chain.
- Runtime probes, all before any device is created:
  the extension is enumerated on the chosen physical
  device; the cooperativeMatrix feature bit is read
  through VkPhysicalDeviceFeatures2; the subgroup size
  through VkPhysicalDeviceProperties2; the property
  rows through the KHR query, requiring the s8 x s8 to
  s32 16x16x16 subgroup row above. Any refusal returns
  by name; GpuContext::open logs it on the diag
  channel ("coopmat bridge refused: ...; the stock
  device serves") and opens the stock device, so a
  machine without coopmat serves exactly as before,
  with the refusal on record. GpuContext.coopmat is
  Some only on a bridged device; the kernel gates its
  existence on that, so routing against a stock device
  refuses by name in step 2, never silently.
- When the bridge succeeds, the device also requests
  EXPERIMENTAL_PASSTHROUGH_SHADERS (masked by the
  adapter's granted features) over the stock feature
  set; the passthrough module path is how the coopmat
  SPIR-V enters. With the switch unset, the
  request_device call is the shipped one, unchanged.
- Unsafe: the crate moves from forbid(unsafe_code) to
  deny(unsafe_code) (the GGUF mmap precedent), and all
  unsafe is confined to open_bridged under
  #[allow(unsafe_code)], with the safety contract for
  each call written in that function's docs. New
  direct dependencies ash = "0.38" and wgpu-hal =
  "27": both already in the workspace lockfile as
  transitive dependencies of wgpu 27, so no new
  packages enter the tree.

First compile caught one defect, fixed in the same
unit: VkPhysicalDeviceSubgroupProperties chains into
the properties pNext family, not the features one
(the compiler rejected it in the features2 chain);
the subgroup size is now read through a separate
get_physical_device_properties2 call.

VM gates for this unit (run with the step-2 unit,
which shares the build): recorded under step 2.

Committed as lane unit 25a51c1.

## Step 2: the coopmat shader and kernel

The shader: crates/wgpu-compute/src/kernels/coopmat/
k23_q4k_coopmat.comp, a standalone GLSL port of
llama's mul_mmq_cm1 design specialized to Q4_K x
Q8_1, single batch per call, this backend's buffer
contract (the same five bindings K13 prepares:
weights, K12 qs, K12 ds, output, params).

Transcribed from llama (mul_mmq_cm1.comp and the
Q4_K parts of mul_mmq_cm1_funcs.glsl, read in full
before writing):

- Staging split: weight codes de-interleaved into
  shared rows of QPITCH = BK_STEP * (BK / 4) + 4 = 36
  u32 words (the +4 bank skew), per-sub-block
  (d * scale, -(dmin * min)) pairs in a separate vec2
  array (buf_a_dm's role); activation Q8_1 words in
  the mirrored tile, (d, d * sum) pairs in two float
  arrays (buf_b_d / buf_b_s's role). Nibbles stage
  raw (0..15 as int8 bytes); the min correction
  absorbs the unsigned offset, as llama's Q4_K
  block_a_to_shmem does. This repo's K12 writes the
  activation sum pre-scaled by d, the same convention
  ggml's MMQ quantizer uses, so the pairs move
  unchanged (K13's epilogue is the in-repo proof of
  the convention).
- The accumulate identity, per 32-element sub-block:
  out += dot_i32 * (d_b * sc) * d_a
       + (-(dmin_b * mn)) * (d_a * sum_q),
  llama's cm1_accumulate for Q4_K, and K13's
  vec_dot_q4_K_q8_1 fold with the signs folded into
  the staged pair.
- Tiles BK 32, BK_STEP 4 (a staged chunk is 128 K
  elements), TM = TN = TK = 16.

Adapted, with the reasons stated:

- Workgroup tile 128 output rows x 64 batch rows
  (llama runs 128 x 128): this backend's MMQ band
  caps the batch at 64 (K13's MMQ_MAX_BATCH
  contract), so the batch side halves. Two subgroups
  across each side; each subgroup owns 64 x 32
  outputs (llama's WM 64 / WN 32 warp tile).
- The accumulator epilogue is a deliberate
  simplification. llama extracts the int32
  accumulator's elements per thread, which needs the
  wave-size-dependent element mapping (cm_elem_row)
  and, at wave 64, the magic-bias path. This shader
  coopMatStores each 16x16 int32 tile into a
  per-subgroup shared tile and folds it with scalar
  reads, which is layout-exact under the cooperative
  matrix memory rules and carries no element-mapping
  assumptions. The extraction form is the named
  follow-up if the step-3 profile convicts the fold.
- Dequant header math (f16 d/dmin via
  unpackHalf2x16, get_scale_min_k4) is transcribed
  from this repo's proven K13 WGSL, not from llama's
  GLSL.

Compile (on the server, system glslangValidator,
target vulkan1.3): clean after one catch, recorded
here because it is the kind of thing the next port
will hit: gl_ScopeSubgroup in the coopmat types
requires GL_KHR_memory_scope_semantics, which
llama's header enables and the first draft here
omitted. spirv-val passes. The artifact
(k23_q4k_coopmat.spv, 19,836 bytes) is checked in
with its source, its full disassembly
(k23_q4k_coopmat.spv.dis.txt), and rebuild.sh, which
recompiles, validates, disassembles, and greps for
the ops. Disassembly inspection: OpCapability
CooperativeMatrixKHR present; OpTypeCooperativeMatrixKHR
declared for int32 16x16 accumulator and int8 16x16
A and B, all subgroup scope; OpCooperativeMatrixLoadKHR,
OpCooperativeMatrixStoreKHR, and
OpCooperativeMatrixMulAddKHR all present.

The Rust side: kernels/k23_q4k_coopmat.rs (K23).
Pipeline construction happens only on a bridged
device (backend construction reads GpuContext.coopmat;
on a stock device the pipeline is never attempted,
since a passthrough module against a device without
the extension would be a spec violation). The
constructor additionally refuses by name on a
subgroup size other than 64 (the shader's compiled
geometry) or workgroup storage under the 37,888
bytes the shader stages. The bind group layout is
declared explicitly in Rust rather than reflected,
because the passthrough module's bindings are the
GLSL contract. The passthrough call itself is the
crate's second unsafe site and lives in
coopmat_bridge.rs, under a written contract (the
words are a checked-in, spirv-val-validated artifact,
fed only to bridge-opened devices).

Routing: session.rs's Q4_K ladder gains a first
rung. When WGPU_COMPUTE_COOPMAT=1 (fail-closed
resolver in compute.rs, the K22 idiom, config echoed
at backend construction), the coopmat prepare is
tried first; a prepare refusal, or the switch on
with no pipeline (stock device), logs by name and
the shipped ladder serves. Default off; K13 remains
the default and the fallback. Diag stamp
k23_q4k_coopmat, registered through the same
DispatchMeta path as every kernel.

On-card numeric gates: tests/k23_coopmat.rs, the
K22 test's mirror: K23 against the host reference
running the same quantized math and against K13, at
input width 6656 with batches 64 and 62 (the
prefill chunk and its tail), K13 parity on the full
6656 x 6656 square at batch 64, and the batch-band
refusals by name. The tests skip with the reason
printed on stock devices and on machines without a
suitable adapter, by design.

VM gates (this machine has no GPU; Naga validation
of the WGSL fleet is the standing stand-in, and it
structurally cannot cover K23, the feasibility
note's disclosed gate hole): cargo test -p
wgpu-compute --lib, 106 passed / 0 failed, including
the new unit tests (the bridge resolver, the coopmat
routing resolver, K23's shape contract and grid
math, and a SPIR-V instruction-stream walk requiring
the four coopmat opcodes in the checked-in artifact);
clippy --all-targets clean under the workspace's
deny-warnings lints; cargo fmt --check clean after
one formatting pass over the bridge callback.

Committed as lane unit 6cc13cf.

## Step 3: card gates (server, 2026-11-06)

Session setup: the lane tree at d5ab2ba was staged to
~/gpu-wave12 on the server by git archive and verified
against a 377-file sha256 manifest (all identical).
Tenants were stopped for the session (docker stop
cortex-cortex-chat-1 and cortex-cortex-embed-1,
systemctl --user stop cortex-embed.service) and the
VRAM guard read 33,271,808 bytes (driver residue only)
before every leg. Every heavy run held
flock /tmp/itak-phase.lock. The profile prompt is the
2026-11-06 profile session's prompt, extracted
byte-exact from its log (102 repetitions of the fox
sentence, 1,022 prompt tokens). Raw logs:
~/gpu-wave12/logs12/ and ~/gpu-wave12/profile12/ on
the server.

### Leg 1: bridge smoke. PASS.

The full resident suite on the bridged device with
coopmat routing off (WGPU_COMPUTE_COOPMAT_BRIDGE=1):
7 passed, 0 failed (resident_session.rs, 19.39 s),
including the multichunk readback-deferral and diag
tests by name. Backend construction on the bridged
device builds every pipeline, K23 included, so the
bridge plus passthrough construction path is proven
inside the suite, not only in the kernel's own test.

### Leg 2: numeric correctness vs K13. PASS.

tests/k23_coopmat.rs on the bridged device:

- vs the host reference running the same quantized
  math, width 6656, out 512: batch 64 max abs diff
  3.125e-2 at set scale 4.173e4 (rel 7.5e-7); batch
  62 (the prefill chunk tail) rel 1.1e-6. Both inside
  the crate's MMQ reassociation bound.
- vs K13 on the same inputs: batch 64 rel 1.9e-7,
  batch 62 rel 1.9e-7.
- K13 parity on the full 6656 x 6656 square, batch
  64: rel 1.7e-7.
- Batch-band refusals (0 and 65) return by name.

The shader's staging, de-interleave, scale algebra,
LDS fold, and transposed store are correct against
two independent references on the first card run.

### Leg 3: the 1,022-token prefill A/B. K23 LOSES.

Diag-timing runs through prefill_profile, same
prompt, same binary, only the env differs:

| arm | Q4_K kernel line | wall (timing mode) |
| --- | --- | --- |
| control (env unset) | k13_q4k_mmq: 2,912 dispatches, 4,250.1 ms, 45.9 GB/s | 7,164 ms |
| coopmat (bridge + routing) | k23_q4k_coopmat: 2,912 dispatches, 16,951.3 ms, 11.5 GB/s | 19,851 ms |

Diag-off walls of record: control 6,453 ms, coopmat
19,518 ms. The Q6_K line is unchanged across arms
(1,360.8 vs 1,382.8 ms), so the delta is the Q4_K
product alone: the coopmat kernel is 4.0x slower than
K13 at the kernel line and 3.0x slower at the wall.
Dispatch totals are identical (10,448 in every arm
and mode): routing swaps the kernel, never the plan.

Why, as labeled hypotheses (no ACO autopsy was run
on this module; the verdict does not need one): the
port's two disclosed simplifications are the prime
suspects. The LDS fold runs per sub-block: eight
tile stores, sixteen subgroup barriers, and a fold
in which half the subgroup idles per tile, all
serializing the matrix units' output. And the port
interleaves load, muladd, store, and fold per tile,
where llama preloads a sub-block's fragments into
register arrays before the muladds. The grid is also
half K13's at the wide shapes (52 workgroups at out
6656 against the m64 shape's 104, both LDS-capped at
one workgroup per CU). The named follow-ups, in
order: the element-extraction epilogue in llama's
cm_elem_row form; fragment preload arrays; an ACO
statistics pass (VGPR, spills, scratch) through
pipeline executable properties.

### Leg 4: env-unset no-leak. PASS.

With every coopmat variable unset, this tree
reproduces the shipped marks: prefill wall 6,349 ms
(glimmer_generate) and 6,453 ms (prefill_profile)
against the recorded 6,330; the K13 kernel line
4,250.1 ms / 45.9 GB/s against the recorded
4,199.9 ms / 46.4 GB/s; dispatch totals 10,448 in
both diag modes; generated ids [7235, 19596, 30054,
28], exact. The bridge's off path is byte-identical
in behavior, as designed.

### Session identity under coopmat routing. EXACT.

glimmer_generate --gpu with the bridge and routing
on, the full 1,022-token prompt: prefill 19,401 ms
through K23 for every Q4_K projection, generated ids
[7235, 19596, 30054, 28], the golden sequence. The
kernel is end-to-end correct through the real model,
not only against synthetic fixtures. (One harness
error recorded for honesty: the first
glimmer_generate invocation omitted --gpu and ran
the CPU backend at 100% CPU with the card idle; it
was killed and rerun with the flag. No measurement
was taken from it.)

### Tenants restored.

docker start on both Cortex containers and
systemctl --user start cortex-embed.service: both
containers Up (healthy), the service active, VRAM
settled at 8,932,200,448 bytes, the tenant set the
session found.

## Lane verdict

The bridge works and is the durable result: it is
the only legal door to VK_KHR_cooperative_matrix on
this wgpu stack, it opens with named fail-closed
probes, its off path is provably the shipped path,
and the full resident suite passes on a bridged
device. The K23 kernel is CORRECT, against the host
reference, against K13, and through the real model's
greedy identity. It is also 4x SLOWER than K13 at the
product it exists to accelerate.

Per the decision rule written before the run
(material A/B win required for a default flip), the
recommendation is: WGPU_COMPUTE_COOPMAT stays off by
default, K13 remains the shipped Q4_K MMQ, and the
coopmat path stays in the tree behind its two
switches as the measured starting point for the
follow-up iteration (extraction epilogue and
fragment preload), which now has a correct reference
implementation to optimize against.
