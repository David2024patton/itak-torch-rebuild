#!/bin/sh
# Rebuild the checked-in K24 coopmat SPIR-V artifact
# from its GLSL source, and prove the artifact carries
# the cooperative matrix ops it exists for (and does
# NOT carry the one it exists to remove).
#
# Requires glslangValidator, spirv-dis, and spirv-val
# (Vulkan SDK or distribution packages; on
# ubuntu-server all three are system tools in
# /usr/bin). The compile target is vulkan1.3: the
# coopmat extension is a 1.3-era KHR extension and the
# backend's devices are Vulkan 1.4, so 1.3 is the
# conservative floor the module is built against.
#
# The disassembly is written next to the artifact and
# checked in with it: Naga cannot validate this module
# (wgpu 27 has no coopmat IR), so the in-repo
# inspection record is part of the build's evidence.
#
# K24's structural claim is the absence of
# OpCooperativeMatrixStoreKHR: the v2 fold is
# in-register (bench/coopmat-v2-2026-11-07.md), so a
# store op in the artifact means the fold regressed
# to the K23 LDS round-trip and the rebuild fails.
set -eu
cd "$(dirname "$0")"

glslangValidator -V --target-env vulkan1.3 \
    -o k24_q4k_coopmat.spv k24_q4k_coopmat.comp
spirv-val k24_q4k_coopmat.spv
spirv-dis k24_q4k_coopmat.spv > k24_q4k_coopmat.spv.dis.txt

for op in OpTypeCooperativeMatrixKHR OpCooperativeMatrixLoadKHR \
          OpCooperativeMatrixMulAddKHR; do
    if ! grep -q "$op" k24_q4k_coopmat.spv.dis.txt; then
        echo "rebuild: $op missing from the disassembly" >&2
        exit 1
    fi
done
if grep -q "OpCooperativeMatrixStoreKHR" k24_q4k_coopmat.spv.dis.txt; then
    echo "rebuild: OpCooperativeMatrixStoreKHR present; the v2 fold must stay in-register" >&2
    exit 1
fi
echo "rebuild: k24_q4k_coopmat.spv OK (coopmat load/muladd present, no coopmat store)"
