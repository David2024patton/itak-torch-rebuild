<!--
INSTRUCTIONS FOR THIS FILE (user and AI: read before editing. Keep this header.)

WHAT THIS FILE IS: the project's front door on GitHub. A stranger decides
in ten seconds whether this project is real. Everything here must be true
today: badges only when true, status only with proof, quickstart commands
copied from the working build.

EXACT STRUCTURE: centered title in exact brand casing, one-line promise,
true badges, the standard-files link row, What it is, Quickstart, the
module map (the grid), Status, License. Full art direction lives in the
github-repo-standard skill (README standard + README-PATTERNS.md).

RULES:
- Brand casing is exact: AlphaVision, AlphaFlux, iTaK, Mimir,
  Patriot Pest Control in full.
- David's taste: black background with a bright color on top so things
  pop is his preferred look; black and red are his favorite scheme, not
  a law. Each project picks its own palette.
- No secrets, passwords, or tokens in this file, ever.
-->
<div align="center">

# iTaK Torch Rebuild

An inference engine and serving platform rebuilt the way it should have been built the first time: a Rust core where the garbage collector cannot reach, the Go platform kept where Go is strongest, designed from the start for huge models on small hardware.

![License: MIT](https://img.shields.io/badge/license-MIT-brightgreen)
![Languages](https://img.shields.io/badge/languages-Rust%20%2B%20Go-orange)
![Status](https://img.shields.io/badge/status-design%20%2B%20evidence-blue)

</div>

**Plan:** [PLAN.md](PLAN.md) · **Tasks:** [TODO.md](TODO.md) · **Ideas:** [IDEA.md](IDEA.md) · **Build rules:** [docs/RULES.md](docs/RULES.md)

## What it is

iTaK Torch began as a pure-Go inference engine with an unusually wide reach: an OpenAI-compatible server, speculative decoding, a full training stack (SFT, DPO, GRPO, ORPO, LoRA), continuous batching, clustering, model operations, and a native engine experimenting with hybrid state space models. It also had a fatal flaw: it depended on a shared support module that lived in a relative folder, was never published, and is lost. The engine cannot be built today. The audit of exactly what it contained, feature by feature, is in [PLAN.md](PLAN.md) and drives this rebuild.

The rebuild splits the system by what each language is best at:

- **Rust owns the token loop.** Quantized kernels, KV caches, the state space scan, MoE expert streaming, speculative decoding, training compute, GPU backends. No garbage collector, no pause jitter, memory allocated once and pooled.
- **Go owns the platform.** The API surface, scheduling, request caches, cluster coordination, tenancy, model pulls and integrity, observability. Most of the original Go platform code is architecturally sound and gets reattached, not rewritten.
- **Mojo stays the research lab.** The Latent Reasoning Engine and new kernel ideas are prototyped there; winners graduate into the Rust crates.

The design target has not changed since the original: huge model capability on hardware that should not be able to hold it. MoE experts stream from disk the way Colibri does it, state space layers keep constant-size state instead of a growing KV cache, and speculative decoding multiplies what a small card can produce.

## Proof so far

The design is grounded in measurements taken on the target hardware, an AMD Radeon RX 7900 XTX (24 GB, gfx1100), running Meta Muse Glimmer 30B:

- Q4_K_M decodes at 44.07 tok/s solo (llama-bench, Vulkan), 37.14 tok/s per user with two users sharing the card.
- The full 131K context costs about 1.70 GiB of KV cache thanks to sliding window attention.
- ROCm/HIP on that machine is broken at the system level, proven four ways; Vulkan is the production backend. The diagnosis is written up in full because "ROCm is faster" articles keep assuming a clean install.

Full record: [docs/research/2026-10-26-glimmer-7900xtx-benchmarks.md](docs/research/2026-10-26-glimmer-7900xtx-benchmarks.md). The Mamba lineage the SSM design stands on, including a grading of the author's own extrapolated Mamba-3 variant against the real Mamba-3 paper: [docs/research/2026-10-27-mamba-papers-ingest.md](docs/research/2026-10-27-mamba-papers-ingest.md).

## Quickstart

There is no engine binary yet; this repo is at the design and evidence stage (see Status). What can be reproduced today is the benchmark record:

```bash
# On a machine with an RX 7900 XTX and a Vulkan-capable driver:
git clone https://github.com/ggml-org/llama.cpp
cmake -B llama.cpp/build-vulkan -S llama.cpp -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build-vulkan -j --target llama-bench llama-server
# Download a Muse Glimmer 30B GGUF (for example the Meta Q4_K_M build), then:
bench/bench-phase-a.sh   # per-quant speed and VRAM, llama-bench
bench/bench-phase-b.sh   # serving behaviour, accuracy prompts, 2-user concurrency
```

## The grid: module map

| Module | Kind | What it owns |
| :--- | :--- | :--- |
| itak-types | Rust crate (planned) | The shared contracts: request/response types, model info, token usage. Versioned, published, never a relative folder again. |
| itak-diag | Rust crate (planned) | Engine-side diagnostics: structured logs, health, event channel. |
| itak-loader | Rust crate (planned) | GGUF loading today, .itf later. Weights memory-mapped, never copied. |
| itak-kernels | Rust crate (planned) | Quantized matmul and dequant (Q4_K, Q6_K, Q8_0, FP8, AWQ, ternary), RoPE, norms, softmax. |
| itak-kv | Rust crate (planned) | KV caches: paged, ring, INT8, compressed, offloaded; prefix sharing; attention-sink eviction. |
| itak-ssm | Rust crate (planned) | The state space scan: general three-term recurrence, data-dependent rotation on B and C, MIMO rank as config. |
| itak-moe | Rust crate (planned) | MoE routing, expert streaming and offload, expert cache with prefetch. |
| itak-spec | Rust crate (planned) | Speculative decoding: n-gram, Medusa, MTP, rejection sampling; EKF speculative update. |
| itak-backend | Rust crate (planned) | One trait, many backends: CPU SIMD first, wgpu (Vulkan) second, others behind the seam. |
| itak-train | Rust crates (planned) | Training compute: autograd, optimizers, LoRA, SFT/DPO/GRPO/ORPO trainers. |
| Go platform | Go modules (planned) | API handlers (OpenAI and Ollama compatible), scheduler, batching, caches, cluster, tenancy, model operations, observability, Core (types, health, event, debug) as a versioned module. |
| itak-dissect | Tool (planned) | Per-layer weight analysis and re-quantization planning; successor to the original dissector and itak-quant. |

Full map with links traced both ways: [docs/MAP.md](docs/MAP.md).

## Status

Design and evidence phase. Proven: the hardware benchmark record, the ROCm diagnosis, the complete feature inventory of the original engine, and the Mamba research base, all in this repo. Not started: engine code. Next moves are in [STATUS.md](STATUS.md) and [TODO.md](TODO.md).

## License

MIT. See [LICENSE](LICENSE). The original iTaK Torch repos remain the reference implementation this rebuild is graded against.
