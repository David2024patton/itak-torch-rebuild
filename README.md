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
![Languages](https://img.shields.io/badge/languages-Rust%20%2B%20Go%20%2B%20Mojo-orange)
![Status](https://img.shields.io/badge/status-do%20not%20ship%20(panel%202026--11--07)%2C%20fixes%20landing-red)

</div>

**Plan:** [PLAN.md](PLAN.md) · **Tasks:** [TODO.md](TODO.md) · **Ideas:** [IDEA.md](IDEA.md) · **Build rules:** [docs/RULES.md](docs/RULES.md)

About the HTML comment at the top of this file: it is this project's own editing process, the structure and accuracy rules the owner set for whoever maintains the repo, kept verbatim by the owner's standing instruction. It is project process for this repo's editors; it makes no claim about the software and binds nobody outside this repo.

## What it is

iTaK Torch began as a pure-Go inference engine with an unusually wide reach: an OpenAI-compatible server, speculative decoding, a full training stack (SFT, DPO, GRPO, ORPO, LoRA), continuous batching, clustering, model operations, and a native engine experimenting with hybrid state space models. It also had a fatal flaw: it depended on a shared support module that lived in a relative folder, was never published, and is lost. The original engine cannot be built today. The audit of exactly what it contained, feature by feature, is in [PLAN.md](PLAN.md) and drives this rebuild.

The rebuild splits the system by what each language is best at:

- **Rust owns the token loop.** Thirteen crates: the GGUF loader, both tokenizer families, the reference math core, attention and KV caches, the persistent prefix-KV store, two architecture modules (dense llama-family and Muse Glimmer), a CPU reference backend, a wgpu/Vulkan backend, the DFlash draft module, the speculative loop, and the engine binary itself.
- **Go owns the platform.** The OpenAI-compatible API surface, API-key auth, scheduling, response cache, engine supervision, and observability, in [platform/](platform/). The platform serves authenticated chat completions by driving the Rust engine over a Unix socket. Its data handling, stated plainly: the platform is a single-operator service today. It listens on loopback by default (127.0.0.1:8091), and one API key, supplied through the TORCH_API_KEY environment variable, guards every route except /healthz: constant-time comparison, fail-closed when no key is set, repeated failures throttled (platform/internal/auth/). There are no sessions and no per-user keys, so rotating or revoking access means replacing that one key and restarting. The response cache is exact-match and lives in process memory only: entries expire on a configured TTL (default 5 minutes), the store is capped (default 1,024 entries) with a janitor purge, and nothing is written to disk (platform/internal/cache/; flags in platform/cmd/torchd/). There is no tenant separation: every caller holding the key shares the one cache. Observability is counters and stats (cache hits, auth failures), not stored chat content.
- **Mojo stays the research lab.** New kernel ideas are prototyped there; winners graduate into the Rust crates.

The design target has not changed since the original: huge model capability on hardware that should not be able to hold it.

## Proof so far

Measured on the target hardware, an AMD Radeon RX 7900 XTX (24 GB, gfx1100), against the real Muse-Glimmer-30B GGUF:

- The tokenizer matches the reference token for token on the recorded bench suite: 16 of 16 prompts and 16 of 16 decode round-trips on Glimmer's vocabulary (crates/tokenizer/CONFORMANCE.md). The claim covers that suite, not arbitrary text.
- The loader covers the real file completely: 731 of 731 tensors through the coverage gate, zero missing, zero unexpected (proof: the glimmer_inventory coverage run against the real file on the server, 2026-10-29; the run record is in STATUS.md).
- The device-resident GPU forward reproduces the golden continuation exactly on the recorded check, 8 of 8 tokens (bench/gpu-resident-2026-10-30.md).
- The speculative verify loop (DFlash draft plus rejection sampling) produces a stream token-identical to plain greedy over 16 tokens on the real files (bench/spec-loop-2026-10-30.md).
- The serving pair passed its live proof run on the real file: sampling parity, two concurrent slots with independent outputs, embeddings geometry, and chat-template prompt ids matching llama.cpp 63 for 63 (bench/t917-proof-2026-10-30.md).
- The reference-stack baselines the design stands on, both from llama.cpp's own harnesses, not from this engine: llama.cpp decodes Glimmer Q4_K_M at 44.31 tok/s (Vulkan, llama-bench tg128 after pp512) on this card. The DFlash depth sweep on the llama-server /completion harness measured its operating point at depth 6, 107.39 tok/s, with draft acceptance of 1,749 of 3,235 proposed tokens (0.541) at that depth; the full depth table (depths 1 through 16, median of 9 runs per cell, acceptance per depth) is in bench/gpu-proof-2026-10-30.md, and the full record in docs/research/2026-10-26-glimmer-7900xtx-benchmarks.md. The two numbers come from different harnesses and are not compared against each other; DECISIONS.md (2026-10-30) designates the sweep harness as the comparison shape.

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

The engine socket is owner-only by construction, which is what makes the /tmp path safe. Binding pins the socket file to mode 0600, and the engine reads the connecting process's uid from the kernel (SO_PEERCRED) on every connection and refuses any other uid with a named `peer_refused` error, debug and embed operations included (crates/itak-engine/src/main.rs, `bind_socket` and `check_peer`). The platform verifies the same facts before it dials: the path must be a socket owned by its own uid with mode exactly 0600, or it refuses to connect (platform/internal/engine/socket_unix.go).

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
| wgpu-compute | Rust crate | wgpu (Vulkan) Compute backend: kernels K1 through K24 plus the split-K reducer and the cooperative-matrix variants, including the device-resident forward session. |
| dflash | Rust crate | The DFlash draft architecture: conditioning fused from named target layer inputs, non-causal block forward. |
| itak-kv | Rust crate | The persistent tiered prefix-KV store: versioned KV snapshots keyed by token prefix, verified on load, LRU-bounded by bytes, on the attention crate's KV cache. |
| itak-spec | Rust crate | The speculative decoding loop: block drafting, exact-output verification, KV checkpoint and rewind, rejection sampling. |
| itak-engine | Rust binary | Owns one loaded model and serves the NDJSON-over-unix-socket protocol (hello, ping, debug, generate, embed, chat_template) with streamed token events and named errors. Access is owner-only: the socket file is pinned to mode 0600 and every connection's peer uid is checked against the engine's own, any other uid refused as `peer_refused`. |
| Go platform | Go service | API handlers (OpenAI-compatible), a single shared API key (no sessions; rotation or revocation is replace-the-key and restart), scheduler, response cache (in memory only, TTL and entry cap, no tenant separation), engine client and supervision, chat-template rendering, the torchd daemon. |

Full map with links traced both ways: [docs/MAP.md](docs/MAP.md). Still designed and not yet built: the training stack (Phase 7), the .itf format (Phase 8), MoE expert streaming, and the SSM scan crate; they are in PLAN.md and TODO.md, not in the table above.

## Status

The engine and platform exist and serve. The serving path today is the CPU reference backend behind the platform; the GPU-resident forward, the GPU speculative loop, and the prefix-KV store are proven on the card, not yet wired into the serving engine. Two cold critic panels graded this tree DO NOT SHIP (2026-10-31 and 2026-11-07); every seat on both panels verified the numeric core clean, and every blocker sat at the serving boundary or in the written record (records: the panel entries in [STATUS.md](STATUS.md); the 2026-11-07 panel's merge report, CRITIC-ENGINE-MERGE.md). The first panel's three fix waves landed in full. The second panel's safety, socket, platform-answer, proof-tool, and diagnostics fixes landed the same day; its remaining categories (speculative-session desync, sampler and config validation, gate skip accounting, smaller robustness items) are still open in TODO.md. The card confirmation at the shipped defaults (cooperative-matrix MMQ and split-K on) has since run, on 2026-11-07: identity and golden legs passed at literally unset env, the combined prefill wall was measured at both prompt shapes, and the run's gate leg surfaced one test-binary environment race that is carried as a follow-up fix (bench/wave21-session-2026-11-07.md). Separately, the Mojo/MAX engine living on this branch (mojo-engine/) has since reached token-exact greedy parity at 19.35 tok/s decode (bench/mojo-engine-wave-d-2026-11-08.md). No claim in this repo outruns its proof note. Detail: [STATUS.md](STATUS.md) and [TODO.md](TODO.md).

## License

MIT. See [LICENSE](LICENSE). The original iTaK Torch repos remain the reference implementation this rebuild is graded against.
