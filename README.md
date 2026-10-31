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
![Status](https://img.shields.io/badge/status-built%2C%20under%20review-blue)

</div>

**Plan:** [PLAN.md](PLAN.md) · **Tasks:** [TODO.md](TODO.md) · **Ideas:** [IDEA.md](IDEA.md) · **Build rules:** [docs/RULES.md](docs/RULES.md)

## What it is

iTaK Torch began as a pure-Go inference engine with an unusually wide reach: an OpenAI-compatible server, speculative decoding, a full training stack (SFT, DPO, GRPO, ORPO, LoRA), continuous batching, clustering, model operations, and a native engine experimenting with hybrid state space models. It also had a fatal flaw: it depended on a shared support module that lived in a relative folder, was never published, and is lost. The original engine cannot be built today. The audit of exactly what it contained, feature by feature, is in [PLAN.md](PLAN.md) and drives this rebuild.

The rebuild splits the system by what each language is best at:

- **Rust owns the token loop.** Twelve crates: the GGUF loader, both tokenizer families, the reference math core, attention and KV caches, two architecture modules (dense llama-family and Muse Glimmer), a CPU reference backend, a wgpu/Vulkan backend, the DFlash draft module, the speculative loop, and the engine binary itself.
- **Go owns the platform.** The OpenAI-compatible API surface, API-key auth, scheduling, response cache, engine supervision, and observability, in [platform/](platform/). The platform serves authenticated chat completions by driving the Rust engine over a Unix socket.
- **Mojo stays the research lab.** New kernel ideas are prototyped there; winners graduate into the Rust crates.

The design target has not changed since the original: huge model capability on hardware that should not be able to hold it.

## Proof so far

Measured on the target hardware, an AMD Radeon RX 7900 XTX (24 GB, gfx1100), against the real Muse-Glimmer-30B GGUF:

- The tokenizer matches the reference token for token on Glimmer's vocabulary (16/16 bench prompts, 16/16 decode round-trips; crates/tokenizer/CONFORMANCE.md).
- The loader covers the real file completely: 731 of 731 tensors through the coverage gate, zero missing, zero unexpected.
- The device-resident GPU forward reproduces the golden continuation exactly, 8 of 8 tokens (bench/gpu-resident-2026-10-30.md).
- The speculative verify loop (DFlash draft plus rejection sampling) produces a stream token-identical to plain greedy over 16 tokens on the real files (bench/spec-loop-2026-10-30.md).
- The serving pair passed its live proof run on the real file: sampling parity, two concurrent slots with independent outputs, embeddings geometry, and chat-template prompt ids matching llama.cpp 63 for 63 (bench/t917-proof-2026-10-30.md).
- The reference-stack record the design stands on: llama.cpp decodes Glimmer Q4_K_M at 44.31 tok/s (Vulkan) on this card, and the DFlash depth sweep found its operating point at depth 6, 107.39 tok/s (bench/gpu-proof-2026-10-30.md; full record in docs/research/2026-10-26-glimmer-7900xtx-benchmarks.md).

## Quickstart

Build both halves and run the serving smoke:

```bash
# Rust engine (workspace root)
cargo build --release -p itak-engine

# Go platform
(cd platform && go build -o torchd ./cmd/torchd)

# Start the engine on a GGUF, then the platform against its socket.
./target/release/itak-engine --model /path/to/model.gguf --socket /tmp/itak-engine.sock
TORCH_API_KEY=<your-key> ./platform/torchd -engine-socket /tmp/itak-engine.sock -models glimmer
```

Operating the pair, flags and all, is in [SERVING.md](SERVING.md); the operations book is [docs/RUNBOOK.md](docs/RUNBOOK.md). The gates that must pass before any commit: `cargo test --workspace`, `cargo clippy --workspace --all-targets`, `cargo fmt --all -- --check`, `cargo deny check`, and in platform/ `go test ./...`, `go vet ./...`, `govulncheck ./...`.

## The grid: module map

| Module | Kind | What it owns |
| :--- | :--- | :--- |
| gguf | Rust crate | Strict GGUF v3 loader: header, metadata, tensor directory, dequantization, fail-loud named errors. |
| tokenizer | Rust crate | Byte-level BPE and SentencePiece BPE over the model's own GGUF vocabulary and merges. |
| torch-math | Rust crate | Reference f32 math: RMSNorm, RoPE, gated FFN activations, logit softcap and scale, matmul and matvec. Every constant is a parameter. |
| attention | Rust crate | KV cache (global layers plus sliding-window ring layers) and grouped-query attention. |
| muse-glimmer | Rust crate | The Muse Glimmer architecture: config from GGUF metadata, the measured weight layout, the structured 52-layer forward. |
| arch-llama | Rust crate | Generic dense architecture for the llama and qwen2 families, config and weights from the file's own metadata. |
| itak-loader | Rust crate | The bridge: implements the architecture modules' Metadata and TensorSource traits over the gguf crate, with the tensor coverage gate. |
| cpu-compute | Rust crate | CPU reference Compute backend: real f32 arithmetic via torch-math, KV storage via the attention crate. |
| wgpu-compute | Rust crate | wgpu (Vulkan) Compute backend: kernels K1 through K14, including the device-resident forward session. |
| dflash | Rust crate | The DFlash draft architecture: conditioning fused from named target layer inputs, non-causal block forward. |
| itak-spec | Rust crate | The speculative decoding loop: block drafting, exact-output verification, KV checkpoint and rewind, rejection sampling. |
| itak-engine | Rust binary | Owns one loaded model and serves the NDJSON-over-unix-socket protocol (hello, ping, debug, generate, embed, chat_template) with streamed token events and named errors. |
| Go platform | Go service | API handlers (OpenAI-compatible), API-key auth, scheduler, response cache, engine client and supervision, chat-template rendering, the torchd daemon. |

Full map with links traced both ways: [docs/MAP.md](docs/MAP.md). Still designed and not yet built: the training stack (Phase 7), the .itf format (Phase 8), MoE expert streaming, and the SSM scan crate; they are in PLAN.md and TODO.md, not in the table above.

## Status

The engine and platform exist and serve. The serving path today is the CPU reference backend behind the platform; the GPU-resident forward is proven as a bench, not yet wired into the serving engine, and the speculative loop is proven on CPU, not in the serving path. A double critic panel graded this tree DO NOT SHIP; fix wave Waves 1 and 2 have landed (socket and bounds hardening, the corrected record, toolchain pins, the dependency policy), and Wave 3 plus a cold panel re-run are still ahead. No claim in this repo outruns its proof note. Detail: [STATUS.md](STATUS.md) and [TODO.md](TODO.md).

## License

MIT. See [LICENSE](LICENSE). The original iTaK Torch repos remain the reference implementation this rebuild is graded against.
