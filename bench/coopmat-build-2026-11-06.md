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

**Step 0 verdict: PASS.** Extension, required matrix
row, and compiler all present. Proceeding to step 1.

## Step 1: the device bridge

(pending)

## Step 2: the coopmat shader

(pending)

## Step 3: card gates

(pending)
