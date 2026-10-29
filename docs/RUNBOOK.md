<!--
INSTRUCTIONS FOR THIS FILE (user and AI: read before editing. Keep this header.)

WHAT THIS FILE IS: the operations book for this project, at docs/RUNBOOK.md.
How to build it, run it, deploy it, verify it, and where everything lives,
so anyone can operate the project without tribal knowledge.

EXACT STRUCTURE:
- "## Build": the real commands, copied from the working build.
- "## Run locally": how to start it and what you should see.
- "## Deploy": where it is hosted, how a deploy happens, how to roll back.
- "## Verify": the checks that prove it works after a change or deploy.
- "## Where things live": hosts, services, dashboards, and WHERE
  credentials are stored (names of vault entries or connectors only).
- "## Gotchas": the traps already learned the hard way, each with its fix.

RULES:
- NEVER a credential value in this file: no passwords, keys, or tokens.
  Pointers to secure storage only.
- Commands are copied from what actually works; if a command changes,
  this file changes in the same commit.
- No secrets, passwords, or tokens in this file, ever.
-->
# RUNBOOK: iTaK Torch Rebuild

## Build

The engine does not exist yet; there is nothing to build in this repo today. What builds today is the reference stack the benchmarks came from, on the home server (llama.cpp at commit 5f5acfe, two build trees):

```bash
# Vulkan build (the production path)
cmake -B build-vulkan -S . -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j --target llama-bench llama-server llama-cli

# HIP build (compiles, but see Gotchas: it wedges at runtime on this machine)
cmake -B build-rocm -S . -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm -j --target llama-bench llama-server llama-cli
```

## Run locally

Reproduce the benchmark record from this repo's bench/ folder. Phase A is raw speed and VRAM per quant; Phase B is serving behaviour, accuracy, and concurrency:

```bash
bench/bench-phase-a.sh    # llama-bench, pp512 and tg128, per quant
bench/bench-phase-b.sh    # llama-server: accuracy prompts, then 2-user load
```

The serving profile the measurements support (solo profile adds the drafter):

```bash
./build-vulkan/bin/llama-server \
  -m <models-drive>/models/muse-glimmer/meta/muse-glimmer-30b-Q4_K_M.gguf \
  -ngl 99 -c 131072 -np 4 --jinja \
  --cache-type-k f16 --cache-type-v f16
# Solo only: -md <dflash gguf> --spec-type draft-dflash --spec-draft-n-max 4, -np 1
```

Expected: about 44 tok/s solo decode on Q4_K_M, about 37 tok/s per user with two users, 4 slots of 32768 tokens in the server log.

## Deploy

Nothing deploys from this repo yet. The reference llama-server stack on the home server is run ad hoc for bench sessions and stopped afterward; there is no standing Glimmer service (that decision is parked, see TODO T-P.1). When the engine exists, this section gets its deploy and rollback steps in the same commit as the deploy path itself.

## Verify

- Benchmarks: compare new runs against docs/research/2026-10-26-glimmer-7900xtx-benchmarks.md. A result more than a few percent off the recorded numbers is a finding, not a rounding error.
- Accuracy: bench/accuracy-prompts.md at temp 0. The JSON extraction and factual prompts must be correct, and the math prompt must complete with reasoning_strength low injected. Empty content means the reasoning default trap is back.
- Repo changes: run the gates (TODO T-1.5) before every push; a secret scan hit stops the push.

## Where things live

- Home server: reached over Tailscale by its SSH config alias (ts-server). It runs Dokploy and live business sites; treat it as production at all times.
- Model weights: the dedicated NVMe models drive in the server (Kingston NV3 1 TB, DECISIONS.md 2026-10-27) once the hardware swap lands; until then the large array, /mnt/alphaflux-data, is the interim home. Never the root disk. As of this writing the Glimmer weights are deleted and not re-downloaded.
- llama.cpp trees on the server: build-vulkan (production path) and build-rocm (compiles, wedges at runtime).
- The original engine: GitHub repos David2024patton/iTaKTorchQ4 (public) and itak-torch (private twin), plus the GOAgent repo, which still hosts the pkg/debug and pkg/eventbus ancestors Phase 1 ports from.
- Credentials: GitHub access goes through the stored custom.github connector credential. No credential value appears in this repo, ever.

## Gotchas

- **ROCm is broken at the system level on the home server.** Symptoms: HIP programs load the model, then spin a CPU core at 0% GPU busy forever and ignore SIGTERM. Root cause layer: a mixed-version ROCm userland install; a 20-line HIP hello-world fails to link with undefined symbol __hipUnregisterFatBinary. Vulkan on the same driver is flawless, so do not chase the card or the kernel. Fix path is a clean single-version reinstall, which likely needs a reset window; the server must not reboot, so this stays parked until David approves a window.
- **Total context is split across slots.** -c 131072 with -np 4 gives each slot 32768 tokens. A poisoned oversized request can leave its slot failing with context-size errors until eviction; do not read that as a model fault.
- **Reasoning models eat the budget by default.** At default reasoning strength, Glimmer spent the entire 512-token cap thinking and returned empty content on code and logic prompts. Inject reasoning_strength low (chat template kwargs) at the proxy by default.
- **Never strip special tokens.** The model's channel protocol rides on its special tokens; a proxy that strips them collapses reasoning, tool calls, and replies into one mangled stream. Always run llama-server with --jinja.
- **Root disk discipline.** Model-sized files on the root NVMe filled the disk on 2026-10-27 and took Dokploy down. Weights go on the models drive, full stop.
