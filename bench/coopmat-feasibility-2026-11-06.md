# Coopmat prefill feasibility: is WMMA reachable from this stack? (2026-11-06)

Lane: lane/coopmat-prefill-spike, based on main a064b5d.
Local VM work only: crate sources read in the local cargo
registry, repo sources read in this worktree, two web checks.
No server GPU time used.

**Verdict in one line:** cooperative matrix is NOT
reachable through anything wgpu exposes: the locked wgpu
27 stack has no coopmat feature, no WGSL syntax, and no
Naga support for it, at any layer. It IS reachable through
one unsafe bridge this codebase could build: create the
Vulkan device ourselves with VK_KHR_cooperative_matrix
enabled, hand it to wgpu through its hal `device_from_raw`
API, and feed hand-compiled SPIR-V through wgpu's
passthrough shader path. The driver side is not the
obstacle: RADV has exposed the extension on RDNA3 since
Mesa 23.3 and llama.cpp runs coopmat matmuls on this card
class in production. **No prototype is built in this
lane**, for one concrete reason stated in full below: the
coopmat SPIR-V blob, the one artifact that makes it a
prototype, cannot be produced or validated by any tool
present on this VM or in the lockfile, so the lane ships
this evidence file, the costed build plan, and the exact
server protocol to judge the build when it is ordered.

## The three layers of the answer

| Layer | Coopmat? | Evidence |
| --- | --- | --- |
| Hardware + driver (gfx1100, RADV, Mesa 26.0.8) | YES | Extension exposed for GFX11+ under ACO since Mesa 23.3; llama.cpp coopmat pipelines run on RDNA3 today |
| Vulkan bindings in the lock (ash 0.38) | YES | Complete KHR_cooperative_matrix bindings, already a transitive dependency |
| wgpu 27 (safe API, WGSL, Naga, hal device creation) | NO | Zero support at every layer; see below |

## Evidence 1: the wgpu stack has no coopmat anywhere

Versions are the repo's own lockfile (Cargo.lock: ash
line 31, naga line 569, wgpu line 1016, wgpu-hal line
1119): wgpu 27.0.1, wgpu-core 27.0.3, wgpu-hal 27.0.4,
wgpu-types 27.0.1, naga 27.0.3, ash 0.38.0+1.3.281.

Method: case-insensitive grep for `cooperative` over the
`src` trees of the extracted registry sources of wgpu,
wgpu-types, wgpu-core, wgpu-hal, and naga. Result: **0
hits in all five crates.** Only ash has hits (149),
because ash generates bindings for every Vulkan
extension regardless of who uses them.

Corroboration from the feature surface itself:
wgpu-types 27.0.1 `src/features.rs` defines the complete
native `Features` set; the shader-relevant entries are
`SHADER_F16` (bit 9 in the WebGPU set), `SUBGROUP`
(1 << 38, features.rs:1070), `SUBGROUP_VERTEX`,
`SUBGROUP_BARRIER`, and
`EXPERIMENTAL_PASSTHROUGH_SHADERS` (1 << 52,
features.rs:1233). There is no cooperative-matrix
feature, and no feature bit is left ambiguous: the list
was read in full. Naga 27.0.3 has subgroup support
(`src/back/spv/subgroup.rs`) but no cooperative matrix
type in its IR, no WGSL syntax for one, and its SPIR-V
frontend cannot ingest the coopmat opcodes (the
`spirv` header crate it shares does define them:
`TypeCooperativeMatrixKHR = 4456`,
`CooperativeMatrixLoadKHR = 4457`,
`CooperativeMatrixMulAddKHR = 4459`, capability
`CooperativeMatrixKHR = 6022`, in
spirv-0.3.0/autogen_spirv.rs:2232 and :3366-3369; the
opcodes exist as constants, with no consumer).

Consequence for this codebase: every shipped kernel
(K1-K22) is WGSL compiled by Naga, and the crate's
shader_compile test gate validates each one on the VM
before any card time is spent. A coopmat kernel cannot
enter through that door at all.

## Evidence 2: the passthrough door exists but does not open by itself

wgpu 27 does ship an experimental shader passthrough:

- `Device::create_shader_module_passthrough`
  (wgpu-27.0.1/src/api/device.rs:189), unsafe, "passes
  data to the backend as-is".
- wgpu-core gates it on the device feature:
  `require_features(EXPERIMENTAL_PASSTHROUGH_SHADERS)`
  (wgpu-core-27.0.3/src/device/resource.rs:2144), then
  wraps the caller's words as `hal::ShaderInput::SpirV`
  for the Vulkan backend.
- wgpu-hal's Vulkan backend feeds those words verbatim
  to VkShaderModule creation:
  `ShaderInput::SpirV(data) =>
  ShaderModule::Raw(self.create_shader_module_impl(data, ...))`
  (wgpu-hal-27.0.4/src/vulkan/device.rs:1963-1965).
- The Vulkan adapter advertises the feature
  unconditionally (wgpu-hal-27.0.4/src/vulkan/
  adapter.rs:567, in the base feature set).

But a passthrough SPIR-V module that declares
`OpCapability CooperativeMatrixKHR` requires, per the
VK_KHR_cooperative_matrix specification, the extension
enabled on the VkDevice and the
`VkPhysicalDeviceCooperativeMatrixFeaturesKHR.
cooperativeMatrix` feature turned on; pipeline creation
against a device without them violates the spec's
valid-usage rules and is undefined behaviour territory,
not a portability bet. And wgpu-hal's own device
creation can never enable it: the device extension list
is computed from the requested wgpu features by
`PhysicalDeviceCapabilities::get_required_extensions`
(wgpu-hal-27.0.4/src/vulkan/adapter.rs:985), consumed by
`required_device_extensions` (adapter.rs:1878), and the
crate contains no reference to the extension at all
(the zero-hit grep above covers this file). There is no
public wgpu knob, flag, or env var that adds a device
extension. **Path A, passthrough on the wgpu-created
device, is therefore not a shortcut; it is a spec
violation. Nobody should ship it.**

## Evidence 3: the bridge wgpu does offer

wgpu-hal's Vulkan adapter exposes the raw-device
integration route, and the safe wgpu crate wraps it:

- `PhysicalDevice::device_from_raw(raw_device:
  ash::Device, ...)` (wgpu-hal-27.0.4/src/vulkan/
  adapter.rs:1932) builds a hal device around a
  VkDevice the caller created. Its companion
  `physical_device_features` doc comment states the
  intended use: "it's fine to add more extensions to
  the list" (adapter.rs:1906).
- `Adapter::create_device_from_hal`
  (wgpu-27.0.1/src/api/adapter.rs:76) turns that hal
  device into a normal (wgpu::Device, wgpu::Queue);
  `Instance::from_hal` (instance.rs:328) and
  `Instance::create_adapter_from_hal` (instance.rs:385)
  cover the instance and adapter ends.

So the caller can create the ash instance and device
with `VK_KHR_cooperative_matrix` in
ppEnabledExtensionNames and
`vk::PhysicalDeviceCooperativeMatrixFeaturesKHR`
(ash-0.38.0: src/vk/definitions.rs:54385) chained into
device creation with cooperative_matrix = true, hand
the device to wgpu, and from then on use the ordinary
wgpu API, including passthrough shader modules whose
coopmat pipelines the driver will now accept. ash's
bindings for the extension are complete, including
`get_physical_device_cooperative_matrix_properties`
(ash-0.38.0: src/extensions/khr/
cooperative_matrix.rs:1-17), which is how the supported
shapes get probed at runtime. ash 0.38 and wgpu-hal 27
are already in this repo's lockfile as wgpu's
dependencies; making them direct dependencies of
wgpu-compute adds no new packages to the lock.

## Evidence 4: the driver and the reference implementation

- RADV merged VK_KHR_cooperative_matrix for RDNA3
  (GFX11) and newer on top of the WMMA instructions in
  Mesa 23.3 (Phoronix, "RADV Vulkan Driver Merges
  Cooperative Matrix Support Using RDNA3 WMMA"). Mesa's
  own extension table conditions it on
  `gfx_level >= GFX11 && !pdev->use_llvm` and the
  feature on the ACO compiler path. The server's
  gfx1100 under Mesa 26.0.8 with ACO satisfies both,
  three Mesa majors past the introduction.
- llama.cpp's Vulkan backend, the same tree whose
  baselines this project measures against (prefill
  998.43 tok/s on this card), selects cooperative
  matrix variants of mul_mm whenever
  `device->coopmat_support` is true, with GLSL shaders
  (`mul_mm_cm1.comp`, gated on
  `VK_KHR_cooperative_matrix` +
  `GGML_VULKAN_COOPMAT_GLSLC_SUPPORT`) that dequantize
  into shared memory and run fp16 coopmat GEMMs. In
  2026 it added an int8 coopmat1 MMQ shader covering
  q4_k and q6_k on RDNA3/RDNA4 (ggml-org/llama.cpp
  PR #27952, shipped in release b11160). The mechanism
  this note evaluates is therefore not speculative: it
  is the incumbent's production prefill path on this
  hardware family.

What remains genuinely unknown until a probe runs on
the server: the exact cooperative matrix property rows
RADV reports for gfx1100 (the RDNA3 WMMA shape is
16x16x16, f16 inputs, f32 accumulate; the int8 rows
llama's MMQ relies on, and the per-scope support, must
be read from
`vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`,
not assumed), and whether the wave32 subgroup pinning
llama applies to its coopmat pipelines (RDNA WMMA is
wave32-native; at wave64 each op issues as two halves)
can be expressed through wgpu pipeline creation or must
ride in the SPIR-V execution modes. Both are step 0 of
the server protocol below, deliberately before any
shader work.

## Why this lane ships evidence, not a prototype

The buildable path needs one artifact this lane cannot
produce: a compiled coopmat SPIR-V module. The facts:

- No GLSL/SPIR-V toolchain exists on this VM:
  `glslangValidator`, `glslc`, `spirv-as`, `spirv-dis`
  are all absent from PATH, and `apt-get install
  --dry-run glslang-tools` answers "Unable to locate
  package". The lockfile contains no shaderc, glslang,
  or spirv-tools crate; adding shaderc-sys means a
  vendored C++ build (ninja is also absent) and a
  workspace-level dependency change, the opposite of a
  minimal spike.
- Hand-assembling SPIR-V words in Rust is possible in
  principle (the header constants above), but the
  result would be unverifiable here by construction:
  Naga cannot parse it (Evidence 1), no validator is
  installed, and this VM enumerates no Vulkan ICD at
  all (`/usr/share/vulkan/icd.d` does not exist), so
  not even wgpu's passthrough acceptance could be
  smoke-tested. Every prior kernel in this project
  (K19-K22 included) was Naga-validated on the VM
  before it consumed serialized card time; a coopmat
  blob would arrive at the card with zero validation
  of its most failure-prone component, and its first
  card session would be spent on load errors instead
  of measurement.
- The device bridge code itself (300-500 lines of
  unsafe, see costing) type-checks but likewise
  executes nothing on this VM: with no adapter, every
  run is the skip path. Writing it blind, unexecuted,
  alongside an uncompilable shader, produces an
  artifact whose honest label is "draft", and this
  project's rule is that claims separate what was
  executed from what was merely accepted.

The settlement the lane was asked for does not depend
on the prototype: reachability is a property of the
APIs and the driver, and both are now pinned to file
and line. The build is a follow-on lane with a known
shape, costed next.

## The build, costed (Path B, the only viable one)

If ordered, in order:

1. **Probe first** (server, minutes): the step-0
   protocol below. If RADV's property rows lack an
   f16 16x16x16 row usable from compute, the project
   stops here for the cost of one probe run.
2. **Device bridge** in crates/wgpu-compute/src/
   device.rs: replace the `adapter.request_device`
   call (device.rs:89) with hal instance init,
   physical-device selection identical to today's,
   ash device creation adding VK_KHR_cooperative_matrix
   plus its feature struct (and every extension/feature
   wgpu-hal requires for the requested feature set,
   obtained from `required_device_extensions` /
   `physical_device_features` so the two never drift),
   then `device_from_raw` +
   `create_device_from_hal`, requesting
   EXPERIMENTAL_PASSTHROUGH_SHADERS. Estimate 300-500
   lines, all unsafe, behind the existing GpuContext
   constructor so no call site changes. New direct
   deps: ash = "0.38", wgpu-hal = "27" (both already in
   the lock). RISK, stated plainly: the bridge becomes
   load-bearing for every kernel, not one; a bridge
   defect breaks the whole backend, and wgpu minor
   upgrades can move the hal integration surface.
   Mitigation: the bridge is used only when an env
   flag asks for it, with the shipped request_device
   path as the default and fallback, until the coopmat
   verdict is in.
3. **The shader**: a GLSL compute shader modeled on
   llama.cpp's mul_mm_cm1 shape (dequantize Q4_K to
   f16 tiles in LDS, coopmat load/mul-add/store,
   16x16x16), specialized for the prefill band (batch
   64, hidden 6656, the K13 shapes), compiled to SPIR-V
   by glslangValidator. The toolchain decision is part
   of the cost: either a build.rs step invoking
   glslangValidator (present on the server inside the
   llama.cpp build tree; a new host dependency for any
   other machine) or a committed .spv with its .comp
   source beside it and the compile command recorded.
   Estimate 400-700 lines of GLSL. Permanent gate hole,
   disclosed: Naga cannot validate this module, so the
   crate's shader_compile test can never cover it; its
   correctness rests on the on-card numeric test alone,
   like the driver itself.
4. **Routing**: a prefill-only arm beside the MMQ
   ladder in the prepare path, env-gated
   (WGPU_COMPUTE_COOPMAT=1), default off, falling back
   to K13/K14 on any refusal, with diag stamps from
   birth. Q6_K follows only if Q4_K pays; the tiny
   256-wide Q6_K KV shapes stay on the fusion fix
   (attack 5), not on coopmat.
5. **Tests**: numeric correctness against the K13 MMQ
   output on identical inputs under the crate's
   existing tolerance rules, resident identity +
   golden with the flag on, prefill A/B through the
   prefill_profile harness, env-unset no-leak check.
   Same shape as the K22 protocol, plus the probe.

Total estimate: 900-1,400 lines across the bridge, the
shader, routing, and tests, one new host toolchain
dependency, one permanent validation-gate hole, and a
second prefill code path to keep correct forever. The
ceiling it buys is the profile note's own arithmetic
(bench/prefill-profile-2026-11-05.md:281-290): the
same 650 GB/s streaming bound as the plain MMQ rework,
about 347 tok/s wall on the 1,022-token profile,
because the binding constraint becomes the dequant
feed, not the multiply. Coopmat does not raise the
ceiling; it changes which wall you hit first, and it
is the only remaining path whose instruction stream
bypasses the VGPR-256 occupancy wall entirely (the
matrix units hold the accumulators, not the register
file). That, and not a higher ceiling, is the honest
case for building it after K22's verdict.

## Rejected paths

- **Path A, passthrough on the stock device.**
  Invalid, not merely risky: the extension is never
  enabled (Evidence 2). Any "it worked on my run"
  result would be a driver accepting a spec violation.
- **Path C, a native Vulkan side channel** (a second
  ash instance/device beside wgpu, behind the Backend
  trait). wgpu's public API offers no external-memory
  import or export for its buffers; `create_buffer_
  from_hal` (wgpu-27.0.1/src/api/device.rs:371) wraps
  only buffers belonging to the same hal device. A
  side device therefore cannot read the resident
  weights or write results the session can consume:
  it would need its own 16.76 GB copy of the model,
  a host round-trip for every chunk's activations and
  outputs, and its own synchronization domain, forking
  the Compute seam end to end. Strictly worse than
  Path B on every axis; dead unless Path B's probe
  fails in a way that leaves raw Vulkan viable, which
  it would not (same driver).
- **Path D, wait for upstream wgpu/Naga support.**
  Nothing in the locked stack points at it: no feature
  bit, no IR type, no frontend support, and WGSL
  itself has no cooperative matrix types. Not a plan.

## Ranked next-best prefill levers (updated)

From bench/prefill-profile-2026-11-05.md's ranked
list, re-stated at today's state (GPU total 6,659 ms,
MMQ pair 5,542 ms of it, on the 1,022-token profile):

1. **MMQ rework for occupancy** (attack 1). Ceiling:
   at the GEMV-proven 650 GB/s the pair's 5,542 ms
   becomes 371 ms, wall about 2,945 ms, about
   347 tok/s; at 300 GB/s, about 274 tok/s. Status:
   the register-organization variant (K22) is spent,
   ACO re-inflated it to the 256 cap; what remains
   inside this attack is llama's staging repack with
   permuted B order (K22's named follow-up) and
   schedule variants not yet tried. Still the only
   lever with a multiple in it that stays inside the
   WGSL/Naga gate.
2. **Coopmat MMQ** (attack 3, this note). Same
   347 tok/s ceiling, higher risk, second code path;
   ranked above the rest now only because attack 1's
   cheapest variant is spent and coopmat is the one
   path that exits the register file entirely.
3. **Host serialization removal** (attack 2). DONE
   and confirmed since the profile: the shipped
   prefill wall is 6,330 ms (161.5 tok/s) against the
   profile's 8,116 ms. Its ceiling is spent; listed
   so the ranking stays honest.
4. **Flash attention rework** (attack 4). Halving
   flash saves 507 ms at 1,022 tokens (about
   166 GPU-bound tok/s) and grows quadratically; at
   the 131k context target this becomes the first
   lever. Sequencing, not size, keeps it here.
5. **Fuse the tiny Q6_K KV-side dispatches**
   (attack 5): 416 dispatches, 186 ms at 3.2 GB/s,
   2.8% of GPU time. Small, safe, independent; the
   natural filler lane while a big one runs.
6. **Raise chunk rows 64 -> 128+** (attack 6): low
   single digits until a kernel streams properly;
   valuable only after 1 or 2 lands.

## Server protocol to judge the build (for the follow-on lane)

Run under the phase lock, card cleared, staging
verified, in this order; stop at the first failure
and report it by name:

0. **Probe.** A minimal ash probe (or vulkaninfo
   --json) on the server: confirm
   VK_KHR_cooperative_matrix in the device extension
   list for the gfx1100 device and dump every
   CooperativeMatrixPropertiesKHR row (M, N, K,
   A/B/C/accumulator types, scope). Confirm an f16
   16x16x16 f32-accumulate row exists for compute.
   Also confirm glslangValidator's location in the
   llama.cpp build tree. No shader work before this
   passes.
1. **Compile + inspect.** glslangValidator -V
   --target-env vulkan1.3 on the .comp source;
   spirv-dis the result and confirm
   OpTypeCooperativeMatrixKHR and
   OpCooperativeMatrixMulAddKHR are present and the
   CooperativeMatrixKHR capability is declared.
2. **Bridge smoke.** The bridged device initializes
   through GpuContext with the flag on; the full
   resident suite passes on it with coopmat routing
   still off (the bridge must be a no-op for every
   existing kernel before it carries a new one).
3. **Correctness.** The coopmat Q4_K product at batch
   64, hidden 6656 against the K13 MMQ reference on
   identical inputs, under the crate's tolerance
   rules; then resident identity + golden with
   routing on.
4. **Prefill A/B.** prefill_profile harness, flag on
   vs off, against K13's recorded 4,199.9 ms /
   46.4 GB/s for the Q4_K MMQ line and the current
   shipped wall of 6,330 ms.
5. **No-leak.** Flag unset: dispatch-for-dispatch
   identical to today's shipped path.

Decision rule, written before the run: coopmat
routing earns a default only if step 0 shows the
shapes, step 3 is clean, and step 4 beats the K13
line materially (not within noise) at equal
correctness. Anything less and it stays an env-gated
experiment, like K18/K19/K22 before it.

## Caveats and claims ledger

- Executed in this lane: the greps and source reads
  above (all local, reproducible from the registry
  paths cited), the lockfile read, the PATH/apt
  probes. Nothing was compiled, and nothing ran on a
  GPU; this VM has no Vulkan device to run on.
- Read, not executed: every wgpu/wgpu-hal/wgpu-core/
  ash behavior claim rests on the cited source lines
  of the exact locked versions.
- Web-sourced: the RADV support facts (Phoronix on
  the Mesa 23.3 merge; Mesa's extension condition)
  and llama.cpp's coopmat usage (ggml-org/llama.cpp
  PR #27952 and the b11160 release notes). Both are
  corroborated by the server's own recorded baseline:
  llama.cpp Vulkan prefill at 998.43 tok/s on this
  card is not achievable through the scalar path
  alone, and its coopmat pipelines are the documented
  difference.
- The one fact this lane could not settle locally is
  RADV's exact property rows on gfx1100; it is step 0
  of the protocol, priced at minutes, and gates
  everything after it.
