# Relational map: inventory

This is the full inventory for the relational diagram of this repo.
Built from the code at branch head b5522f2 (mojo-max-engine) on
2026-11-18. Code beats paper: every LIVE node below was read in the
tree at that head, and every PLANNED or ABSENT node names the doc or
comment that promises it.

Companion files in this folder:

- NODES.md: the compact node table (id, name, kind, state).
- WIRES.md: the connection table (from, to, kind, backlink).
- DIAGRAM.md: Mermaid diagrams of the same nodes and wires (one
  overview plus one diagram per area).

## State legend

- LIVE: code exists in the tree and does the job named.
- LIVE, paused: code exists and works, but the line is paused by
  David's 2026-11-07 order. The Rust workspace and the Go platform
  are the parity scorekeeper now, not the product line. They are
  kept, not deleted.
- PLANNED: a doc in this repo names it and no code exists yet.
- ABSENT: something in the repo speaks of it as if it were there
  (a target state, a comment, a rule), and it is not in the tree.

## External nodes

| ID | Node | Kind | State | Owns | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| EXT-CLIENT | Client | external | LIVE | Sends requests to whichever front door is up; owns nothing here. | n/a |
| EXT-SERVER | RX 7900 XTX server | external | LIVE | The home server and its 24 GB card; all performance claims are measured here. | docs/research/, bench/ |
| EXT-MODELS | Model files | external | LIVE | The GGUF model files on the server array (Glimmer Q4_K_M, its DFlash draft, the staged Qwen3.5 vision file, others). | /mnt/models on the server (not in repo) |
| EXT-REG | Model registries | external | LIVE | HuggingFace and OCI: where model pulls would come from. | PLAN.md S9 |
| EXT-LLAMA | llama.cpp | external | LIVE | The reference implementation both engines are benchmarked against. Never the product. | bench/ |
| EXT-ORIG | Original Torch repos | external | LIVE | iTaKTorchQ4 and itak-torch on GitHub: reference implementation and feature specification for the rebuild. | README.md, SALVAGE.md |
| EXT-GOAGENT | GOAgent repo | external | LIVE | Hosts the surviving pkg/debug and pkg/event ancestors that itak-core ports from. | PLAN.md S4 |

## Rust workspace (paused line)

| ID | Node | Kind | State | Owns | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| RS-ROOT | Cargo workspace root | workspace | LIVE, paused | The 13 crate members, shared package settings, workspace lint gates (warnings and clippy denied). | Cargo.toml |
| RS-ENGINE | itak-engine | binary crate | LIVE, paused | The Rust engine binary: one loaded model on worker threads, generation slots, a bounded queue, and the socket server. Submodules: engine (load and workers), protocol, json, sampler, streamer, diag. | crates/itak-engine/src/ |
| RS-SEAM | The seam | module | LIVE, paused | The NDJSON over unix socket protocol, defined once: six ops (hello, ping, debug, generate, embed, chat_template), streamed token frames, named errors. Socket is owner only (0600) with a peer uid check. | crates/itak-engine/src/protocol.rs, platform/internal/engine/ |
| RS-GGUF | gguf | library crate | LIVE, paused | Strict GGUF v3 loader: header, metadata, tensor directory, mmap file image, f32 dequantization, fail loud named errors. | crates/gguf/src/ |
| RS-TOK | tokenizer | library crate | LIVE, paused | Byte level BPE and SentencePiece BPE over the model's own GGUF vocabulary, merges, and scores: encode, decode, round trip. | crates/tokenizer/src/ |
| RS-MATH | torch-math | library crate | LIVE, paused | Reference f32 math: RMSNorm, RoPE, gated FFN activations, logit softcap and scale, matmul and matvec. Every constant is a parameter. Dependency free by design. | crates/torch-math/src/ |
| RS-ATTN | attention | library crate | LIVE, paused | The KvCache (global layers plus sliding window ring layers, checkpoint and restore) and grouped query attention reference math. | crates/attention/src/ |
| RS-GLIMMER | muse-glimmer | library crate | LIVE, paused | The Muse Glimmer architecture: config from GGUF metadata, the measured weight layout (14 tensors per layer), the structured forward for the 52 layer hybrid (39 sliding window plus 13 global). Compute and KV storage sit behind traits it owns. | crates/muse-glimmer/src/ |
| RS-ARCH | arch-llama | library crate | LIVE, paused | Generic dense architecture for the llama and qwen2 families: pre-norm, RoPE, GQA with KV cache, SwiGLU; config and weights from GGUF metadata. | crates/arch-llama/src/ |
| RS-LOADER | itak-loader | library crate | LIVE, paused | The loader bridge: implements the muse-glimmer and dflash Metadata and TensorSource traits over the gguf crate, and verifies a file's tensor inventory against expected names (the coverage gate). | crates/itak-loader/src/ |
| RS-CPU | cpu-compute | library crate | LIVE, paused | The CPU reference Compute backend: real f32 arithmetic via torch-math, dequantization via gguf, KV storage via attention. This is the backend the Rust serving engine runs. | crates/cpu-compute/src/ |
| RS-WGPU | wgpu-compute | library crate | LIVE, paused | The wgpu (Vulkan) Compute backend: kernels K1 through K24 plus the split-K reducer and cooperative matrix variants, the device resident forward session, and the DFlash session. Proven against the golden as a bench; not wired into the itak-engine binary. | crates/wgpu-compute/src/ |
| RS-DFLASH | dflash | library crate | LIVE, paused | The DFlash draft architecture: config from the draft file's own metadata, conditioning fused from named target layer inputs, non-causal block forward producing per position draft logits. | crates/dflash/src/ |
| RS-KV | itak-kv | library crate | LIVE, paused | The persistent tiered prefix KV store: versioned KV snapshots keyed by token prefix, verified token for token on load, written atomically, LRU bounded by bytes. | crates/itak-kv/src/ |
| RS-SPEC | itak-spec | library crate | LIVE, paused | The speculative decoding loop: DFlash block drafting against the target, exact output verification, KV checkpoint and rewind, rejection sampling, plus an n-gram proposer. Proven token identical on CPU; not in the serving path. | crates/itak-spec/src/ |

## Go platform (paused line)

| ID | Node | Kind | State | Owns | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GO-TORCHD | torchd | binary | LIVE, paused | The platform daemon entry point: flags, env config, signals and graceful shutdown, wiring of engine, supervisor, and server. Default bind 127.0.0.1:8091. | platform/cmd/torchd/main.go |
| GO-SERVER | server | package | LIVE, paused | HTTP routes and validation, SSE streaming, error mapping, generation caps. Routes: GET /healthz, GET /v1/models, POST /v1/chat/completions, POST /v1/embeddings, GET /debug. | platform/internal/server/server.go |
| GO-AUTH | auth | package | LIVE, paused | Single API key middleware: constant time compare, fail closed on empty key, per source failure throttling. | platform/internal/auth/auth.go |
| GO-ENGINE | engine (client) | package | LIVE, paused | The Client interface, named errors, and the unix socket client for the seam (NDJSON frames, 4 MiB frame cap). Also the Unavailable stand-in used when no engine is configured. | platform/internal/engine/ |
| GO-SCHED | scheduler | package | LIVE, paused | Priority lane concurrency gate (Normal and High lanes), per lane queue caps, single grant loop. Note: the server assigns Normal to every request today; the High lane has no caller. | platform/internal/scheduler/scheduler.go |
| GO-CACHE | cache | package | LIVE, paused | Exact match TTL response cache: entry cap, expiry first eviction, janitor purge, injectable clock. In memory only. | platform/internal/cache/cache.go |
| GO-TEMPLATE | chattemplate | package | LIVE, paused | Bounded renderer for a model's own GGUF chat template (a Jinja subset), with render budgets and named refusals for unsupported constructs. | platform/internal/chattemplate/ |
| GO-SUPER | supervisor | package | LIVE, paused | Rust engine process lifecycle: start, health wait, restart budget and backoff, state for /debug. Implements the engine Client interface by delegation to the socket client. | platform/internal/supervisor/supervisor.go |
| GO-DIAG | diag | package | LIVE, paused | Diagnostics registry: request, route, error, auth failure, and engine counters, with a Snapshot read for /debug. | platform/internal/diag/diag.go |

## Mojo engine (the live line)

Engine version 1.07 (mojo-engine/src/config.mojo). One Mojo source,
compiled per target chip. Toolchain pinned: Mojo 1.2.0, MAX 26.7
(mojo-engine/pixi.toml).

| ID | Node | Kind | State | Owns | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| MJ-MAIN | main.mojo entry | module | LIVE | The bare entry point: prints the engine name and version, loads config, reports loader state, refuses cleanly by name on error. Generation and serving live in the runner. | mojo-engine/src/main.mojo |
| MJ-CONFIG | config | module | LIVE | EngineConfig: every limit in one place as env overridable defaults (model path, context length 131072, GPU budget, attention tile and split sizes, prefill thresholds, serve host and port 127.0.0.1:18767, attachment caps). Bad values refused by name. | mojo-engine/src/config.mojo |
| MJ-DIAG | diag seam | module | LIVE | The debug seam: MOJO_ENGINE_DEBUG consulted per call, named phase lines out. Imported by nearly every module in the engine. | mojo-engine/src/diag.mojo |
| MJ-GGUF | gguf loader (Mojo) | module | LIVE | GGUF loader: header, metadata, tensor directory, staged tensor reads. | mojo-engine/src/gguf/loader.mojo |
| MJ-TOK | tokenizer (Mojo) | module | LIVE | Byte level BPE and SentencePiece BPE selected by GGUF metadata: byte map, pre-tokenizer, SPM encoder, encode and decode. | mojo-engine/src/tokenizer/ |
| MJ-KV | kv cache (Mojo) | module | LIVE | CPU KV cache: per layer flat f32 K and V stores, the Glimmer layer pattern (sliding ring versus global), append and visible range reads. | mojo-engine/src/kv/cache.mojo |
| MJ-ATTN | attention (Mojo) | module | LIVE | Grouped query attention: single position CPU twin and GPU kernel, prefill loop, tiled and split-KV GPU path with exact merge. | mojo-engine/src/attention/attention.mojo |
| MJ-OPS | ops | module | LIVE | Row and elementwise ops, CPU and GPU: embedding gather, RMSNorm, RoPE, softmax, SwiGLU, residual add, final logit transform, device embedding. | mojo-engine/src/ops/ |
| MJ-KERNELS | kernels | module | LIVE | GPU kernels, one module per kernel: shared dequant, F32 and F16 GEMV, batched prefill GEMM, K1 Q4_K GEMV, Q5_K GEMV, Q6_K GEMV, each with a CPU twin. | mojo-engine/src/kernels/ |
| MJ-GPU | gpu runtime | module | LIVE | MAX device plumbing: GpuRuntime and the memory budget ledger, device resident weights (one buffer per tensor slot), device KV cache, and the Wave F scratch pool that recycles step buffers. | mojo-engine/src/gpu/ |
| MJ-MODEL | model forward | module | LIVE | The Glimmer transformer forward: CPU forward with ModelConfig from GGUF, and the GPU twin (GpuGlimmerModel, single position layer assembly, batched prefill layer assembly, step level activation residency from Wave F1-A). | mojo-engine/src/model/ |
| MJ-GEN | generate and sampler | module | LIVE | The generation loop behind the ForwardModel trait (prefill, step, logits), the resident device loop from Wave F1-A, and the greedy sampler. Greedy only today; see PL-SAMPLER. | mojo-engine/src/generate/ |
| MJ-RUNNER | runner entry points | module | LIVE | The two end to end entry points: the CLI generation runner over a real model file, and the serve accept loop (dispatch, generation wiring, MCP bridge dispatch, run ingest). | mojo-engine/src/runner/ |
| MJ-SERVE | serve API | module | LIVE | The Mojo only in-process HTTP API: POSIX socket transport through libc, HTTP parse and route, a minimal JSON dialect, pure dispatch, and the static OpenAPI contract at /openapi.json. Routes include /health, /v1/models, /v1/chat/completions, /stats, /debug/log, /history, /leaderboard, and the /mcp bridge. Non-streaming in v1. | mojo-engine/src/serve/ (http, server, socket, json, openapi) |
| MJ-DASH | dashboard page | page | LIVE | The test bench dashboard: one inline HTML and JS page built inside dashboard.mojo (no static files exist), served at GET /. Chat, debug log, model fact sheets, leaderboard, run history, and stats as tiles. It builds itself from the JSON feeds. | mojo-engine/src/serve/dashboard.mojo |
| MJ-FACT | fact sheets | module | LIVE | Per model GGUF fact sheets: family, layer split (sliding versus global, from metadata and tensor signatures), vision proof three ways, served at /v1/models and /v1/models/{id}. | mojo-engine/src/serve/factsheet.mojo |
| MJ-HIST | history and stats | module | LIVE | Run records (hash chained, record_sha256 per record), run history on disk, the leaderboard view, export and ingest of run bundles, and the in-memory stats plus debug ring behind /stats and /debug/log. | mojo-engine/src/serve/history.mojo, mojo-engine/src/serve/stats.mojo |
| MJ-TESTS | Mojo tests and proofs | tests | LIVE | The test bench gate: unit suites under tests/test_*.mojo run by tests/run_tests.sh (200 tests passing at this head), GPU proofs, and server card proof programs. Poison (negative control) gates live here too. | mojo-engine/tests/ |

## Support nodes

| ID | Node | Kind | State | Owns | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| SUP-BENCH | bench notes and gate | records | LIVE | 101 dated bench notes (every speed and correctness claim ties to one), the Rust gate script bench/gate.sh, and timing harnesses. | bench/ |
| SUP-GOLDEN | golden fixtures | records | LIVE | The golden token records and layer diff results both engine lines are checked against, plus the compare tooling. | golden/ |
| SUP-DOCS | planning docs | docs | LIVE | PLAN.md (the phase plan), TODO.md (the task ledger), mojo-engine/DESIGN.md (the engine and test bench design), DECISIONS.md, STATUS.md, and the dated research notes under docs/research/. | PLAN.md, TODO.md, mojo-engine/DESIGN.md, docs/ |

## Planned nodes (named in docs, no code yet)

| ID | Node | Kind | State | Owns | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| PL-F1B | Wave F1-B striping and fusions | module | PLANNED | GEMV Wave32 striping across Q4_K, Q6_K, Q5_K, plus QKV and Gate/Up fusions in the Mojo kernels and layer assembly. | docs/research/2026-11-08-max-nightly-and-reading-digest.md |
| PL-F2 | Wave F2 WMMA prefill | module | PLANNED | Tiled LDS plus WMMA prefill path for the Mojo engine. No WMMA claim stands until the disassembly gate counts the instructions. | docs/research/2026-11-08-max-nightly-and-reading-digest.md |
| PL-SAMPLER | Full sampler family (Mojo) | module | PLANNED | Temperature, top-k, and top-p sampling on top of the greedy sampler. Until it lands, the serve API refuses non-greedy requests by name. | mojo-engine/DESIGN.md steps 5-6 |
| PL-QWEN | Qwen3.5 architecture port | module | PLANNED | The qwen35 family (hybrid DeltaNet plus full attention layers, MTP tensors skipped by name at load) running in the Mojo engine, for the staged vision file and Chris's build. Fact sheets already read the file; generation does not. | mojo-engine/src/serve/factsheet.mojo header, mojo-engine/DESIGN.md step 7 |
| PL-VISION | Vision generation path | module | PLANNED | Image input generation. Today the serve API answers images with the named refusal vision_not_ready; nothing is silently dropped. | mojo-engine/DESIGN.md step 7 |
| PL-STREAM | Streaming chat (Mojo serve) | module | PLANNED | Token streaming for the serve API. The v1 OpenAPI contract states non-streaming. | mojo-engine/src/serve/openapi.mojo header |
| PL-SPEC | Mojo speculative decoding wave | module | PLANNED | Draft based decoding in the Mojo engine. First build stays DFlash (a draft file for Glimmer already sits beside the model); the drafting method decision comes due when the wave is scheduled, after F1-B and F2. | docs/research/2026-11-08-max-nightly-and-reading-digest.md |
| PL-CORE | itak-core (Go contracts) | module | PLANNED | The shared Go contracts: types, health, event, debug, registry. Published and versioned. | PLAN.md S4, TODO T-1.1 to T-1.3 |
| PL-TYPES | itak-types (Rust contracts) | module | PLANNED | The shared Rust contract types, versioned, named as the contracts home in docs/RULES.md. No crate exists yet. | TODO T-1.4, docs/RULES.md |
| PL-SSM | itak-ssm | module | PLANNED | The state space scan: Mamba-2 scan, Mamba-3 three-term recurrence behind a flag (default off), MIMO rank. | PLAN.md S8, TODO T-5.1 to T-5.3 |
| PL-MOE | itak-moe | module | PLANNED | MoE routing, expert cache, expert streaming from disk. | PLAN.md S8, TODO T-5.4 |
| PL-TRAIN | itak-train | module | PLANNED | Training compute: autograd, optimizers, trainers, LoRA, plus SFT and later DPO/GRPO and Medusa head training. | PLAN.md S10, TODO Phase 7 |
| PL-DISSECT | itak-dissect and .itf | tool | PLANNED | Weight statistics, measured heat maps, .itf (ITF1) conversion, re-quantization planning. | PLAN.md S11, TODO T-8.1, T-8.2 |
| PL-KVT | Advanced KV tiers (Rust) | module | PLANNED | KV tiers beyond the built cache: paged, INT8, compressed, offloaded, radix prefix sharing, attention sink eviction. The built attention crate holds the plain and ring forms only. | PLAN.md S2, TODO Phase 5 notes |
| PL-BACKENDS | More backends and multi-GPU | module | PLANNED | Metal and CUDA backends behind the Compute seam when earned, multi-GPU, tensor parallelism, disaggregated prefill and decode. | PLAN.md S7 |
| PL-GOPHASE6 | Go platform Phase 6 expansion | service | PLANNED | The platform growth the plan names: Ollama compatibility, continuous batching and chunked prefill in the scheduler, model operations (pulls, integrity, versioning), clustering and tenancy. None of it is in the Go tree today. | PLAN.md S9, TODO Phase 6 |

## Absent nodes (spoken of as if present, not in the tree)

| ID | Node | Kind | State | Owns | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| AB-CI | CI and commit gate script | service | ABSENT | Named as the target state in docs/RULES.md (secret scanning in the commit gates, TODO T-1.5; language gates and CI, T-1.7). The tree has no .github/workflows and no CI; bench/gate.sh is a local script and says so in its header. | docs/RULES.md, TODO T-1.5, T-1.7 |
| AB-FUZZ | Fuzz targets | tests | ABSENT | Named as the target state in docs/RULES.md: a fuzz target for each bounded parser. No fuzz target exists in the tree today. | docs/RULES.md |
| AB-DB | Platform database | service | ABSENT | No database exists on the Go platform side: no SurrealDB, SQL, or Redis import or call anywhere in platform/. All platform state is in memory (cache, scheduler, diag). | platform/ (searched in full) |
| AB-SHMEM | Shared memory data plane | module | ABSENT | Tensors moving between the Go platform and the Rust engine over shared memory exists only as a comment ("tensors move over shared memory later"). The real data plane is NDJSON over the socket. | platform/internal/engine/unix.go |

## Drift found while building this map (old MAP.md against the code)

1. The old docs/MAP.md had no node for the Mojo engine at all. Its
   only Mojo entry was an external "Mojo lab" node that prototypes
   supposedly graduate from. The Mojo work is in this repo, in
   mojo-engine/, and it is the live line. Fixed by this map and by
   the MAP.md rewrite in the same landing.
2. mojo-engine/README.md carries a "Deliberately not built yet" list
   (GGUF loader, kernels past K1, attention, generation loop,
   sampler, serving front end). Every item on that list is built and
   in the tree. The README is stale; the code and DESIGN.md agree
   with each other, not with the README. Not fixed here (docs-only
   map landing); named so the next README pass can close it.
3. The old MAP.md grid said the Go platform pulls models from the
   registries. The Go tree has no outbound HTTP and no pull code;
   model operations are Phase 6 plan items. This map marks that
   wire PLANNED (see WIRES.md).
4. docs/RULES.md speaks of itak-types (Rust) as the contracts home
   in the present tense. No such crate exists; TODO T-1.4 still
   lists it. This map marks it PLANNED.
5. The old MAP.md said itak-kv is consumed by the wgpu resident
   session, which is right, but the engine binary itself has no
   itak-kv, itak-spec, wgpu-compute, or arch-llama dependency at
   all: the Rust serving path is CPU only. The wires in WIRES.md
   follow the Cargo manifests, not the old prose.
