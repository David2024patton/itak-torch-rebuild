# Key sources read 2026-11-04

Verified live (browser.open):
- Mesa envvars https://docs.mesa3d.org/envvars.html - MESA_VK_TRACE=rgp/rmv/rra, MESA_VK_TRACE_PER_SUBMIT for compute-only, RADV_DEBUG=shaders/shaderstats/spirv/asm/nir/ir/cs, llvm, nocache, RADV_PERFTEST=cswave32, RADV_THREAD_TRACE_* (buffer 32MiB, cache counters, instruction timing), RADV_SPM_COUNTERS_CONFIG, RADV_PROFILE_PSTATE=peak, ACO_DEBUG=perfwarn?/validateir/validatera/force-waitcnt/noopt/nosched/liveinfo/perfinfo
- RGP https://gpuopen.com/rgp/ v2.7.1 - wavefront occupancy, instruction timing per ISA latency, barriers, event timing, pipeline stalls, Linux Ubuntu 24.04 Vulkan only, RX7000 supported
- AMD ROCm blog FP16 GEMM RDNA3/4 https://rocm.blogs.amd.com/software-tools-optimization/rdna-fp16-gemm/README.html - WMMA 16x16x16 wave32, steps: larger tile+128-bit copies+double buffering+staged commit, avoid LDS bank conflicts padding/permuting, buffer descriptors, deepen K, reorder blocks cache reuse, interleave prefetch with WMMA, revisit double buffering vs occupancy. CU vs WGP mode.
- strix fork https://github.com/halo-box/strix-llama.cpp - batched mat-vec 3,5,6 cols several times slower than 1,2,4 on RADV gfx1151, split chunking, GGML_VK_MMV_NO_SPLIT=1, GGML_VK_PERF_LOGGER=1 per-op
- Hanzo paper https://papers.hanzo.ai/pdfs/refutation-driven-perf.pdf - N=1, gfx1151, Vulkan prefill 212->1734, Q6K tiled 76% prefill, BM>=128 2.4x slower occupancy collapse, BK=64, 32KB accumulator spill occupancy 2->6 scratch 32768->0, warm microbench dp4a 74->255 GB/s hot 32MB LLC vs 74->97 in-engine cold, coopmat killed then un-refuted when access pattern fixed, packed-LDS, single-buffer rungs, shaderstats gate free
- llama.cpp PR 27952 https://github.com/ggml-org/llama.cpp/pull/27952 Aug29 2026 merged - int8 coopmat1 MMQ RDNA3/4, Q4_K/Q5_K/Q6_K etc, double buffering, wave32, BK_STEP 4, larger workgroups, workgroup scheduling cache proximity, probe coopmat directly instead of shmem, 4-byte loads

Index only (browser.search):
- wgpu timestamp docs.rs / wgpu examples, wgpu-profiler docs.rs
- Mesa issue 15132 https://gitlab.freedesktop.org/mesa/mesa/-/issues/15132 - ACO 15% slower than AMDVLK LLPC on RDNA4 Q4_K GEMV decode, 507 vs 583 GB/s, 25% more buffer_load, 184% more scalar loads, SGPR allocator clobbering buffer descriptors, occupancy identical max
- dhavli vega notes https://github.com/dhavli/llamacpp-vega-gfx900/commit/6576600e548ad7ffbc006b10f03cf03593fa922e - matvec LDS-latency+occupancy bound, s_waitcnt lgkmcnt ~1167 vs vmcnt ~230 5:1, VGPR63=4 waves/SIMD 40%, issue 20848: 64VGPR/4waves/~208GB/s vs 40VGPR/6waves/229GB/s gfx900
- RADV coopmat Mesa 23.3 GFX11+ https://www.phoronix.com/news/RADV-VK_KHR_cooperative_matrix
- AMDVLK discontinued Sep 2025, discussion 416 https://github.com/GPUOpen-Drivers/AMDVLK/discussions/416 (via NixOS/Gentoo/RGP docs referencing)
- CubeCL README https://raw.githubusercontent.com/tracel-ai/cubecl/main/README.md - autotune, comptime, automatic vectorization, tensor core paths, CubeK matmul
- wgpu subgroup v0.20 release, issue 5555
- wgpu SPIR-V passthrough v25 release https://github.com/gfx-rs/wgpu/releases/tag/v25.0.0 - create_shader_module_passthrough replaces create_shader_module_spirv, SHADER_F16 now naga too
- RGA https://gpuopen.com/rga/ v2.15 - offline ISA, VGPR pressure, LDS/scratch, spills, gfx1100 targets
- umr README cgit - waves, rings, --top, --profiler
- llama.cpp vulkan shaders deepwiki summary of mul_mm.comp BM/BN/BK WM/WN TM/TN, SHMEM_STRIDE padding, LOAD_VEC, mul_mat_vec K_PER_ITER 8, subgroup_no_shmem variants, mul_mmq Q8_1 quantize
- ZINC RDNA4 tuning https://github.com/zolotukhin/zinc/blob/HEAD/docs/RDNA4_TUNING.md - small matmul 2.7% BW vs vocab 93.2%, large near optimal, small cannot saturate; note its RADV_PERFTEST=coop_matrix claim NOT in current Mesa envvars list - flag unverified/obsolete
- ROCm Triton optimizing https://rocm.docs.amd.com/en/docs-6.1.1/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html - occupancy = min(VGPR, LDS floor(65536/L)), s_waitcnt lgkm/vmcnt
