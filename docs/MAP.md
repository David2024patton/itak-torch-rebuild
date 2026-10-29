<!--
INSTRUCTIONS FOR THIS FILE (user and AI: read before editing. Keep this header.)

WHAT THIS FILE IS: the whole project laid out, at docs/MAP.md: every
module, page, screen, engine, and service, and every link between them
traced BOTH ways (what it points to, and what points back at it). It
exists so anyone can see the entire system at once and think about it
whole. A rendered visual version is generated from this file.

EXACT STRUCTURE:
- "## The grid": a table of every node: name, kind (page, screen,
  module, engine, service, external), what it owns.
- "## Links": a table of every connection: from, to, kind (navigates,
  calls, reads, writes, emits, hosts), and the backlink (the reverse
  relationship, named, or "none").
- "## Diagram": a Mermaid flowchart of the same nodes and links, kept
  in sync with the tables above it.
- "## Flows": the two or three main journeys through the system, in
  words, step by step.

RULES:
- Every node in the tables appears in the diagram and vice versa.
- When a module or link is added or removed, this file changes in the
  same commit.
- Facts only: the map shows what exists, with planned nodes marked
  (planned) so vision and reality never blur.
- No secrets, passwords, or tokens in this file, ever.
-->
# MAP: iTaK Torch Rebuild

Nodes marked (planned) are designed in PLAN.md and not yet built. Unmarked nodes exist today.

## The grid

| Node | Kind | Owns |
| :--- | :--- | :--- |
| Client | external | Sends OpenAI-compatible requests; owns nothing here. |
| Go platform (planned) | service | The front door: API handlers, scheduler and batching, request caches, constrained decoding, cluster and tenancy, model operations, observability. |
| itak-core (planned) | module | The shared Go contracts: types, health, event, debug, registry. Published and versioned. |
| The seam (planned) | module | Control over a local socket; tokens and tensors over shared memory. Engine restart never takes the platform down. |
| Rust engine (planned) | service | The token loop process: owns generation end to end once a request crosses the seam. |
| itak-types (planned) | module | Rust mirror of the shared contracts, with version fields. |
| itak-diag (planned) | module | Engine-side structured logs, health, events. |
| itak-loader (planned) | module | GGUF (and later .itf) parsing, mmap weight access. |
| itak-kernels (planned) | module | Quantized matmul and dequant, RoPE, norms, softmax. |
| itak-kv (planned) | module | KV cache pools: paged, ring, INT8, compressed, offloaded; prefix sharing. |
| itak-ssm (planned) | module | The state space scan: three-term recurrence, rotation on B and C, MIMO rank. |
| itak-moe (planned) | module | MoE routing, expert cache, expert streaming from disk. |
| itak-spec (planned) | module | Speculative decoding: n-gram, Medusa, MTP, rejection sampling. |
| itak-backend (planned) | module | Backend trait: CPU SIMD, wgpu/Vulkan, others behind the seam. |
| itak-train (planned) | module | Training compute: autograd, optimizers, trainers, LoRA. |
| itak-dissect (planned) | tool | Weight statistics, measured heat maps, .itf conversion, quant planning. |
| Mojo lab | external | David's existing Mojo engines and the Latent Engine research. Prototypes graduate into the crates. |
| Original Torch repos | external | iTaKTorchQ4 and itak-torch on GitHub: the reference implementation and feature specification. |
| GOAgent repo | external | Hosts the surviving pkg/debug and pkg/eventbus ancestors that itak-core ports from. |
| llama.cpp | external | The reference implementation to benchmark against. Never the product. |
| Model registries | external | HuggingFace and OCI: where model pulls come from. |
| RX 7900 XTX server | external | The home server and its 24 GB card: the hardware all performance claims are measured on. |

## Links

| From | To | Kind | Backlink |
| :--- | :--- | :--- | :--- |
| Client | Go platform | calls (HTTPS API) | Go platform returns responses and streams tokens to Client |
| Go platform | itak-core | reads (contracts, health, events) | itak-core is imported by Go platform |
| Go platform | The seam | writes (requests, control) | The seam returns tokens and engine health to Go platform |
| The seam | Rust engine | hosts (the connection both processes share) | Rust engine connects back through The seam |
| Rust engine | itak-types | reads (contracts) | itak-types is imported by Rust engine |
| Rust engine | itak-diag | emits (logs, health, events) | itak-diag reports Rust engine state to the platform via The seam |
| Rust engine | itak-loader | calls (load model) | itak-loader hands weights to Rust engine |
| Rust engine | itak-kernels | calls (tensor math) | itak-kernels returns tensors to Rust engine |
| Rust engine | itak-kv | reads and writes (cache state) | itak-kv holds state for Rust engine |
| Rust engine | itak-ssm | calls (SSM layers) | itak-ssm returns layer outputs to Rust engine |
| Rust engine | itak-moe | calls (expert layers) | itak-moe returns layer outputs to Rust engine |
| Rust engine | itak-spec | calls (draft and verify) | itak-spec returns accepted tokens to Rust engine |
| itak-kernels | itak-backend | calls (device execution) | itak-backend executes for itak-kernels |
| itak-backend | RX 7900 XTX server | runs on (Vulkan via wgpu) | RX 7900 XTX server hosts itak-backend |
| itak-train | Rust engine | writes (trained heads, adapters) | Rust engine loads artifacts written by itak-train |
| Go platform | itak-train | calls (training API) | itak-train reports progress to Go platform |
| itak-dissect | itak-loader | writes (.itf files the loader reads) | itak-loader reads what itak-dissect writes |
| Go platform | Model registries | reads (model pulls) | Model registries serve pulls to Go platform |
| Mojo lab | itak-ssm | informs (algorithms graduate) | itak-ssm is where Mojo lab results land |
| Mojo lab | itak-spec | informs (n-gram and drafting algorithms) | itak-spec is where Mojo lab results land |
| Original Torch repos | Go platform | informs (code reattached module by module) | Go platform is the successor of Original Torch repos |
| GOAgent repo | itak-core | informs (debug and event ports) | itak-core descends from GOAgent repo |
| Rust engine | llama.cpp | compared against (benchmarks) | none |
| Go platform | RX 7900 XTX server | hosts on | RX 7900 XTX server hosts Go platform |

## Diagram

```mermaid
flowchart TD
  C[Client] --> P[Go platform - planned]
  P --> CORE[itak-core - planned]
  P --> SEAM[The seam - planned]
  SEAM --> ENG[Rust engine - planned]
  ENG --> TYPES[itak-types - planned]
  ENG --> DIAG[itak-diag - planned]
  ENG --> LOAD[itak-loader - planned]
  ENG --> KERN[itak-kernels - planned]
  ENG --> KV[itak-kv - planned]
  ENG --> SSM[itak-ssm - planned]
  ENG --> MOE[itak-moe - planned]
  ENG --> SPEC[itak-spec - planned]
  KERN --> BACK[itak-backend - planned]
  BACK --> GPU[RX 7900 XTX server]
  P --> TRAIN[itak-train - planned]
  TRAIN --> ENG
  DIS[itak-dissect - planned] --> LOAD
  P --> REG[Model registries]
  LAB[Mojo lab] -.-> SSM
  LAB -.-> SPEC
  ORIG[Original Torch repos] -.-> P
  GOA[GOAgent repo] -.-> CORE
  ENG -.-> LLAMA[llama.cpp]
  P --> GPU
```

## Flows

1. **A chat request.** The client calls the Go platform. The platform authenticates, applies rate limits and token budgets, checks the request caches, and schedules the request. It crosses the seam: control over the socket, then the engine runs the token loop (loader-supplied weights, kernels on the backend, KV or SSM state per layer, MoE experts streamed as needed, speculation drafting and verifying). Tokens return through shared memory and stream back to the client. Every refusal along the way emits through the diag channel with its reason.
2. **A training run.** The client calls the platform's training API. The platform validates and queues the job; itak-train runs it (SFT, LoRA, or a preference trainer; Medusa head training is a job type of its own). Artifacts land as versioned files. The engine picks them up by version: adapters serve, Medusa heads raise the speculative acceptance rate, and the eval harness measures the change.
3. **A model arrives.** The platform pulls from HuggingFace or OCI, verifies integrity, registers the version, and warms it. If the model is larger than VRAM, itak-moe serves it by streaming experts from disk with its pin and LRU cache fed by itak-dissect's measured heat data once the .itf conversion exists.
