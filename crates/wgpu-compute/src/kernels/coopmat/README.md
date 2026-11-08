# Cooperative-matrix SPIR-V artifacts

The `.spv` files in this folder are checked in on purpose, and this note
is how they are produced and verified.

## Why binaries are checked in

The K23 and K24 cooperative-matrix kernels load precompiled SPIR-V
(`include_bytes!` in `crates/wgpu-compute/src/kernels/k23_q4k_coopmat.rs`
and `k24_q4k_coopmat.rs`) because the wgpu/Naga path cannot compile or
validate this module: wgpu 27 has no cooperative-matrix IR. The GLSL
sources (`.comp`) sit beside the artifacts and are the source of truth;
the binaries are build outputs that the repo carries because the runtime
needs them as-is.

## How they are built

- `rebuild.sh` rebuilds the K23 artifact; `rebuild-k24.sh` does the
  same for K24. Both compile with `glslangValidator -V
  --target-env vulkan1.3`, the conservative floor for the KHR
  cooperative-matrix extension on this backend's Vulkan 1.4 devices.

## How they are verified

Each rebuild script:

1. validates the fresh artifact with `spirv-val`,
2. disassembles it with `spirv-dis` into the checked-in
   `.spv.dis.txt` beside the artifact, and
3. refuses to pass unless the disassembly contains the four cooperative
   matrix ops the kernels exist for: `OpTypeCooperativeMatrixKHR`,
   `OpCooperativeMatrixLoadKHR`, `OpCooperativeMatrixMulAddKHR`, and
   `OpCooperativeMatrixStoreKHR`.

On top of that, the Rust wrappers carry unit tests
(`spirv_artifact_carries_coopmat_ops`) that walk the checked-in
artifact's instruction stream and require the same ops, so a stale or
swapped binary fails the workspace test gate without a rebuild.
