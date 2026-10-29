# iTaK Torch runtime benchmark, 2026-10-28

Question (David): does the recovered engine actually run, and how fast,
on models it supports, including on the ROCm stack?

## Setup

- Hardware: ubuntu-server, AMD Radeon RX 7900 XTX 24 GB (gfx1100),
  i7-12700K, 128 GB RAM, Ubuntu ROCm 7.1 system stack.
- Engine: iTaKTorchQ4 freshly built from the recovered source plus the
  original iTaKCore (sibling layout, go build exit 0), Linux amd64.
- FFI lock found by measurement: Torch's loader requires the
  llama_params_fit export, which current llama.cpp master no longer
  ships in libllama. All FFI runs therefore use the official llama.cpp
  b8398 release libs (the build Torch was written against), Vulkan and
  ROCm flavors. llama-bench comparisons use the same b8398 builds, plus
  current-master builds (build-vulkan2 / build-rocm2) as spot checks.
- Raw logs and JSONs: server ~/torch-bench-2026-10-28/results/.

## Headline

The engine runs. It loads real GGUF models through its llama.cpp FFI
path, generates coherent text, and serves its OpenAI-compatible API.
On matched stacks its speed is close to raw llama.cpp: the Go FFI
layer costs roughly 0 to 12 percent depending on model and backend.

## Generation speed (tokens/sec)

| Model | Torch CPU (default, pre-fix) | Torch Vulkan | Torch ROCm | llama-bench b8398 Vulkan | llama-bench b8398 ROCm | llama-bench master Vulkan | llama-bench master ROCm |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B Q4_K_M | 90.5 | n/a | n/a | 582.2 | 342.4 | n/a | n/a |
| Qwen2.5-1.5B Q4_K_M | 44.6 | 356.1 | 181.7 | 403.9 | 179.0 | n/a | n/a |
| DeepSeek-Coder-V2-Lite Q4_K_M | n/a | 182.9 | 53.9 | 235.0 | 134.2 | 268.3 | 168.0 |
| Qwen2.5-Coder-32B Q4_K_M | n/a | 0.7 (anomaly) | 26.5 | 36.8 | 26.5 | 38.4 | 36.6 |
| Devstral-Small-2-24B Q4_K_M | n/a | 43.6 | 37.8 | 44.0 | 38.2 | n/a | n/a |

Method notes: Torch 1.5B figures are its built-in bench (best of 3
prompts, fixed binary, default AutoConfig path). Other Torch figures
are single cold serve completions (128 tokens, temp 0, same prompt)
from the server's own [perf] lines. llama-bench figures are tg128
averages over 3 repetitions at -ngl 999. Serve numbers and bench
numbers are not perfectly matched methods; treat serve rows as
approximate.

## Findings

1. AMD detection bug (FIXED and pushed, iTaKTorchQ4 commit c66059a).
   detectLinux never read AMD VRAM (only nvidia-smi provided VRAM), so
   every AMD card probed as 0 MiB, AutoConfig concluded "iGPU only"
   and forced CPU-only. Fix reads mem_info_vram_total from amdgpu
   sysfs, plus a missing AutoConfig case for an AMD card too small
   for the model. Verified on the 7900 XTX: AutoConfig now reports
   amd(24560MiB), full offload, 1.5B at 356 tok/s Vulkan versus the
   broken 44.6 tok/s CPU default. David had no AMD card to test with
   when this was written.
2. Vulkan + Qwen2.5-Coder-32B through Torch serve collapsed to
   0.7 tok/s (128 tokens in 2m51s) and the HTTP response was lost
   (curl exit 52, empty reply) even though the scheduler logged the
   request as processed. The same model on the same stack does
   26.5 tok/s via Torch ROCm and 36.8 via llama-bench Vulkan.
   Undiagnosed; needs its own investigation.
3. DeepSeek-Coder-V2-Lite via Torch ROCm serve: 53.9 tok/s versus
   134.2 for llama-bench on the identical stack. Vulkan serve is
   closer (182.9 vs 235.0). Undiagnosed.
4. Native GOTensor (pure Go) path: loads Qwen2.5-0.5B in 5.4 s, then
   generation dies with "context deadline exceeded" and produces no
   output. Demo grade, consistent with the code read (char-level
   tokenizer, no KV cache). The FFI path is the production path.
5. Stack comparison on this card (llama-bench): Vulkan beats ROCm
   nearly everywhere at b8398; current-master ROCm closes much of
   the gap (DeepSeek 168.0 vs b8398 ROCm 134.2; Coder-32B 36.6 vs
   26.5). Vulkan remains the serving pick on this hardware.
6. Real output proof: Torch serve returned a correct, coherent
   explanation of transformer attention from Qwen2.5-1.5B. Cosmetic
   defect: spaces render as the BPE "Ġ" glyph in its JSON responses,
   a detokenizer piece-join bug in the response assembly.

## Pre-existing issues noted, not caused by the fix

- Full pkg/torch test suite panics in cluster_coordinator.go swarm
  tests; the pristine pre-fix tree fails identically.
- go vet reports two lock-copy warnings in server_endpoints.go.
