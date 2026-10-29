# AI Stack Rebuild + Glimmer 30B Benchmark Report, 2026-10-26

Server: ubuntu-server, AMD Radeon RX 7900 XTX 24GB (gfx1100), i7-12700K, 14GB RAM.
Raw logs: `bench-results-2026-10-26/` in this folder. Deep research page: artifact `muse-glimmer-30b-inference-research`; raw research: `~/workspace/research_notes/glimmer-7900xtx-inference-platforms-20261027-0034/report.md`.

## Cleanup (mission steps 1-2)
- Inventory saved: `inventory-2026-10-26.md`. Hung HIP llama-server from Oct 22 (stuck "Loading model", 99.8% CPU) killed.
- Deleted: old models (Laguna, Qwen3-8B, Ornith, test), full HF cache (44G), both ComfyUI model copies (~21G x2), CUDA llama.cpp tree (`/opt/llama.cpp/build-gpu`).
- Freed: ~115G total (/ free 98G -> 192G at deletion time; md0 21G back).
- Kept deliberately: Glimmer Q3 + dflash (target model), Immich family photos 6.4G (never delete), Immich ML cache volume, Open WebUI data.

## Builds (step 3)
- Fresh llama.cpp builds from HEAD `5f5acfe`: `build-vulkan` (GGML_VULKAN=ON) and `build-rocm` (GGML_HIP=ON), targets llama-server/llama-bench/llama-cli. Both BUILD_EXIT:0.
- Docker: no NVIDIA runtime remains; no docker restart needed. AMD container passthrough form: `--device /dev/kfd --device /dev/dri --group-add video --group-add render`.

## Downloads (step 4), all verified byte-exact vs HF listings
- Meta Q4_K_M 16,756,683,904 B (`meta/`), Unsloth UD-Q5_K_M 19,187,195,296 B, UD-Q6_K_XL 26,274,032,480 B, mmproj-kquant 1,400,328,928 B. Pre-existing: UD-Q3_K_XL 13,360,983,072 B, dflash-kquant 1,631,205,312 B.
- Torch test set: Qwen2.5-7B-Instruct Q4_K_M (bartowski single file) 4,683,074,240 B; Qwen2.5-1.5B-Instruct Q4_K_M 1,117,320,736 B, in `/home/server/models/torch-test/`.

## Phase A, llama-bench (-ngl 999, pp512 / tg128, 3 reps), Vulkan
| Quant | Model size | pp512 tok/s | tg128 tok/s | Peak VRAM |
|---|---|---|---|---|
| Q3_K_XL | 12.43 GiB | 680.06 ± 38.46 | **51.98 ± 0.21** | 11.9 GiB |
| Q4_K_M | 15.59 GiB | 759.74 ± 57.12 | **44.07 ± 0.12** | 15.4 GiB |
| Q5_K_M | 17.86 GiB | 711.28 ± 30.96 | **39.48 ± 0.10** | 17.4 GiB |
| Q6_K_XL | 24.45 GiB | 671.28 ± 35.85 | 30.79 ± 0.05 | 23.6 GiB |

Published RDNA reference points were ~35-36 tg tok/s; this card beats them at every quant. Q6 ran fully offloaded at bench context but leaves no room for KV at serving context; not a candidate.

## Phase A, ROCm/HIP: BLOCKED (system-level, not a build defect)
Evidence, four angles:
1. HIP llama-bench: device init OK (gfx1100, 24560 MiB), model loaded into VRAM, then 0% GPU busy with CPU spinning for 6+ minutes, zero test rows.
2. HIP llama-cli: identical wedge; ignored SIGTERM from `timeout 45`, required SIGKILL. Same failure shape as the Oct 22 hung HIP llama-server (different build).
3. Toolchain isolation: 20-line HIP hello-world fails to LINK with system hipcc: `undefined symbol: __hipUnregisterFatBinary` (mixed/mismatched ROCm install, /usr/lib/rocm).
4. `/opt/rocm.disabled` is an old partial tree, not a viable alternate toolchain.
rocminfo enumerates gfx1100 and Vulkan is flawless on the same amdgpu driver: GPU healthy, ROCm userland broken. Remediation (David's call, not done): clean ROCm reinstall and likely a GPU reset / reboot window; server must not reboot per standing rule. **Production backend tonight: Vulkan.**

## Phase B, llama-server Vulkan, shared profile (-c 131072, -np 4, --jinja, f16 KV)
Server log confirms: `n_slots = 4, n_ctx_slot = 32768, kv_unified = 'false'`. This build's logs do not print a KV MiB line; no OOM at the shared config. (Gemini's SWA check: slot split verified; exact KV bytes not directly captured.)

Accuracy (temp 0, max_tokens 512; full outputs in phaseb_vulkan_q4.json / q5.json):
- JSON extraction: correct on Q4 and Q5. Factual (Le Guin / Gethen): correct on both.
- Math: at default reasoning, answers truncated mid-solution at the 512 cap on both quants. With `reasoning_strength: low`: complete and CORRECT ($53.55) on both; 347 tokens / 8.7s (Q4) and 304 / 8.5s (Q5) vs 512-capped / ~12.5-14s default.
- Go code and logic prompts at default reasoning returned EMPTY content on both quants (entire budget consumed by reasoning). Confirms the reasoning-default trap on real hardware: the proxy layer must inject `chat_template_kwargs.reasoning_strength = low` by default.
- Q5 marginally better behaved under the cap (summary completed; factual finished in 336 tokens), ~10% slower. Q4_K_M remains the serving pick.

Concurrency (2 simultaneous users, 512 tokens each):
- Q4_K_M: 37.14 tok/s per user, **74.27 aggregate** (solo bench: 44.07).
- Q5_K_M: 33.18 per user, 66.34 aggregate (solo: 39.48).
Beats the research planning estimate (15-30 tok/s/user).

## iTaK Torch evaluation (repo: David2024patton/iTaKTorchQ4; design target per David: low-VRAM cards, small models, fast output on weak hardware, built in the 16GB 4070 Ti SUPER era)
Three stacked blockers, all verified live:
1. **Fresh build impossible from the published repos.** go.mod (both iTaKTorchQ4 and itak-torch) has `replace github.com/David2024patton/iTaKCore => ../Core`; iTaKCore exists neither on GitHub (checked both names) nor on ubuntu-server; `pkg/torch/core_adapter.go` imports it. Presumed present as `..\Core` on David's Windows dev box.
2. **Shipped Linux binary is FFI version-locked.** It routes this GGUF to the llama.cpp FFI path and needs ITAK_TORCH_LIB; against tonight's libllama it fails: `could not load "llama_params_fit": undefined symbol` (API drift vs the Windows-bundled DLLs it shipped with).
3. **AutoConfig cannot see the 7900 XTX on Linux.** Probe reports `gpu[1]=amd(0MiB)`, "No discrete GPU detected", forces CPU-only even when handed Vulkan libs.
Native path facts from the repo: arch registry (model_configs.go) = llama, mistral, gemma, qwen2, deepseek, phi3, starcoder2, olmo; no muse-glimmer. Medusa self-drafting exists (pkg/torch/native/speculative_v2.go, tree verification, claims 2.5-3.5x) but heads are random-init (InitHead) with no train/load path, so acceptance with real models is expected near zero until heads are trained. Glimmer on Torch = write a muse-glimmer arch module (parts exist: sliding_window.go, attention.go, paged_attention.go) + train Medusa heads; the FFI path unblocked by publishing Core and pinning a matched llama.cpp lib version.

## Recommended production config (from research + tonight's measurements)
llama-server **Vulkan** build, Meta Q4_K_M GGUF, `-ngl 99 -c 131072 -np 4 --jinja`, f16 KV, DFlash OFF for shared use (ON, n-max 4, solo profile), mmproj off for the text instance, behind GPUStack or LiteLLM with per-user keys for David + Chris and a default injection of `reasoning_strength: low`. Expected: ~44 tok/s solo, ~37 tok/s each at 2 users.

## Open items / David's calls
- ROCm remediation (clean reinstall + reset/reboot window), only if he wants the HIP A/B completed; published data says HIP could be faster, unproven on Glimmer.
- Publish or locate iTaKCore (`../Core`) to unblock Torch builds.
- Serving layer deployment (GPUStack/LiteLLM + systemd/Dokploy) not yet stood up; bench stack was ad hoc on port 18090 and is stopped. GPU idle at end of run.
