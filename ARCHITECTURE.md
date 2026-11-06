# ARCHITECTURE: iTaK Torch Rebuild

Written 2026-10-29, on David's order of the same day: fix everything the
audits found, rebuild with Rust where the hot path needs it, take whatever
is needed from the old project, and get Glimmer working 100 percent on our
own stack. This file is the blueprint the build follows. PLAN.md says why
and in what order at the product level; TODO.md carries the work items;
this file fixes the module grid, the seam between the two processes, and
the design rules that close every audited gap by construction rather than
by patch.

## 1. Fixed decisions (settled, not reopened here)

- The split (David, 2026-10-27, DECISIONS.md): Rust owns the hot token
  loop. Go owns platform, API, scheduling, caches, cluster, and training
  orchestration. Two processes, one seam: a Unix domain socket carrying
  NDJSON frames for everything, control, tensors, and token streams
  alike; there is no shared-memory data plane (Section 4).
- Mojo stays the lab. Flutter is the console. Neither is on the serving
  path.
- First-class architecture target number one is muse-glimmer (Glimmer).
  The other families follow through the same arch seam, in the order the
  stages below set. (PLAN S13, TODO Phase 9; this settles the open
  question parked in PLAN S12.)
- David's build law applies to every module from birth: modular grid,
  debugging and linting built in (diagnostics, structured logs, a way to
  pull the bug information out, lint gates green), limits as config with
  defaults and never compiled constants, version fields in every data
  format, interface seams at every change-prone boundary.

## 2. The shape

Two processes on one host (the seam also works across hosts later; the
transport does not care):

| Process | Language | Owns | Never does |
| --- | --- | --- | --- |
| torchd (platform) | Go | HTTP API, auth, scheduling, batching policy, caches, model operations, cluster, training orchestration, observability | Token math, weight math, anything inside the per-token loop |
| itak-engine | Rust | Model loading, tokenization, the forward pass, sampling, KV state, speculative execution, GPU backends | HTTP, tenancy decisions, anything that requires knowing who asked |

The engine is replaceable without touching the platform and the platform
is replaceable without touching the engine. That is the point of the
seam, and it is also the fix for the old tree's core defect: in the old
tree the platform and a demo engine were fused, so the demo's limits
became the product's limits.

## 3. Module grid

### 3.1 Rust engine crates (the hot path)

| Module | Owns (its grid slot) | Interface | Depends on |
| --- | --- | --- | --- |
| itak-types | The shared contracts of the Section 4 protocol: requests, responses, sampling params, errors, every struct carrying a version field; the Go side mirrors them in platform/internal/engine | Pure data; both sides compile against the same shapes | Nothing |
| itak-diag | Structured logging and the diagnostic channel: every refusal, fallback, and verdict in the engine emits here with a stable code | `diag!(code, fields)` macros; a diagnostics dump endpoint on the control socket | itak-types |
| itak-gguf | GGUF parsing and mmap weight access. The full mainline ggml type table (Section 5.2). Produces a load coverage report for every file | `load(path) -> ModelFile + CoverageReport`; any gap is an error value, never a silent default | itak-types, itak-diag |
| itak-loader | Turns a ModelFile into an in-memory model by asking the arch module for its tensor map and verifying every required tensor arrived, readable, right shape | `build(file, arch) -> Model` or a named load error listing exactly what is missing | itak-gguf, itak-arch |
| itak-arch | One submodule per architecture family, keyed on the file's own `general.architecture` string (Section 5.3). Each declares its tensor map, metadata keys, norm layout, activation, RoPE parameters, and KV shape | `ArchModule` trait: `tensor_map()`, `config(file)`, `forward hooks` | itak-types, itak-gguf |
| itak-arch::muse_glimmer | The Glimmer module (Section 6): 52-layer hybrid schedule, SWA rings, softcap, DFlash companion | Same ArchModule trait, no special cases outside it | itak-arch, itak-kv, itak-kernels |
| itak-tokenizer | Real tokenizers: byte-level BPE (GPT-2 style) and SentencePiece, pretokenizers including the llama4 pre-tokenizer, byte-level detokenization | `encode(text) -> ids`, `decode(ids) -> text`, both round-trip tested | itak-types |
| itak-kv | KV state: paged cache for global layers, fixed ring buffers for sliding-window layers, prefix-cache snapshots, versioned save and load that either works or errors | `KvStore` trait per layer kind; snapshot format carries a version field | itak-types, itak-diag |
| itak-kernels | The math: dequant for every type in the table, GEMV and GEMM, RMSNorm, RoPE, softmax, attention (paged and ring). CPU SIMD first; shape-specialized fused kernels for Glimmer land in Stage 4 | Pure functions over tensor views; allocation-free inside the token loop | itak-types |
| itak-backend | Hardware backends behind one trait: CPU SIMD (always available), wgpu on Vulkan (the 7900 XTX path). Hardware detection reads AMD VRAM from sysfs | `Backend` trait; `detect() -> HardwareReport` | itak-kernels, itak-diag |
| itak-forward | The incremental transformer step: one token in, logits out, KV updated in place. RoPE, final norm, per-family epsilon and activation all come from the arch module | `step(session, token) -> logits` | itak-arch, itak-kv, itak-kernels, itak-backend |
| itak-spec | Speculative execution: DFlash block drafting for Glimmer, n-gram prompt lookup, Medusa heads loaded from versioned artifact files | `draft(session) -> candidates`, `verify(...)`; every mode is config-gated and every gate has a behavior test | itak-forward, itak-arch |
| itak-engine (binary) | Session and slot management inside the engine, the IPC server side of the seam, sampling execution, EOS and stop handling | The IPC contract (Section 4) | All of the above |

### 3.2 Go platform packages

| Module | Owns (its grid slot) | Interface | Depends on |
| --- | --- | --- | --- |
| platform (the Go module) | The platform half of the repo as one module, `github.com/David2024patton/itak-torch-rebuild/platform`. The Section 4 contracts are mirrored in internal/engine's client types; structured logging is slog throughout; diagnostics live in internal/diag | Builds platform/cmd/torchd | Nothing |
| platform/internal/engine | The Go side of the seam: the Client interface and the Unix-socket client for the protocol of Section 4 (hello, ping, debug, generate, embed, chat_template), handshake and protocol version check, named errors, bounded frame reads. Builds the sampling block of the Generate message from the request parameters | `Client.Generate(ctx, req) -> stream of token events` | Nothing (the seam itself) |
| platform/internal/server | HTTP surface: OpenAI-compatible chat, completions, embeddings, models; SSE streaming, validation, error mapping, generation caps. Auth middleware in front of every route group; startup diagnostics list every registered route | HTTP; binds 127.0.0.1 by default, LAN exposure only by explicit flag | platform/internal/engine, platform/internal/auth, platform/internal/scheduler, platform/internal/cache |
| platform/internal/auth | The one API-key middleware: constant-time comparison, fails closed, repeated failures throttled per source. The key is configuration (TORCH_API_KEY), never a file or a chat message | Middleware + identity context | Nothing |
| platform/internal/scheduler | The request scheduling gate: bounds how many engine jobs run at once and queues the rest in priority lanes, each lane capped at a configured number of waiters, every lane provably drained in every mode | Gate in the request path | platform/internal/engine |
| platform/internal/cache | Exact-match response cache: a TTL on every entry, an entry cap with expiry-first eviction, a purge janitor, an injectable clock. Cache keys are caller-built and carry the tenant dimension | Lookup and store hooks in the request path | Nothing |
| platform/internal/chattemplate | Renderer for the model's own GGUF chat template: a bounded Jinja subset with render budgets, so a hostile template refuses by name instead of hanging or blowing the stack | `Render(template, messages) -> prompt text` | Nothing |
| platform/internal/supervisor | Engine process lifecycle: start, health wait, restart within a budget, an allowlisted child environment (PATH plus ITAK_ENGINE_* only) | Process management used by platform/cmd/torchd | platform/internal/engine |
| platform/internal/diag | The diagnostics registry behind /debug: request counts per route, error counts, and engine call outcomes reported by every module | Snapshot served at /debug | Nothing |
| platform/cmd/torchd | The platform daemon binary: flags, env, signals, wiring. Every serving limit is a flag with its default recorded | CLI flags | All of the above |

Not yet built, and nothing in the tree answers for them yet: the
feature tier (guardrails, A/B and payload routing, versioning,
tracing), model operations (HF and OCI pulls, SHA-256 integrity,
versioning, warmup, prefault, GGUF export; quantize tools either
produce the file or exit non-zero), cluster coordination (join
requires the shared secret and an allowlisted address; no automatic
LAN discovery by default), and the training stack (an HTTP training
API over a real orchestration handle, and the orchestration itself:
autograd, optimizers, SFT, LoRA, DPO, GRPO, ORPO, QAT,
checkpointing). They keep their stages in Section 7; a route or verb
that cannot work is not registered.

### 3.3 Off the serving path

| Module | Owner | Role |
| --- | --- | --- |
| Mojo lab engines | Mojo | Research: SSM variants, the Latent Engine experiments. Results file into REFERENCES and PLAN; nothing ships from the lab without a plan change |
| Flutter console | Flutter | Operator UI over the platform API |
| itak-dissect (tool) | Rust | Per-layer weight statistics and heat maps, the GGUF to .itf converter (Phase 8) |

## 4. The seam: Go and Rust IPC contract

As built (this section describes the shipped seam; the definition of
record is `crates/itak-engine/src/protocol.rs` on the Rust side and
`platform/internal/engine/unix.go` on the Go side, and SERVING.md
carries the same definition in prose):

Transport: a Unix domain socket. The engine binds the path given by
`--socket` / `ITAK_ENGINE_SOCKET` (default `/tmp/itak-engine.sock`)
and loads its one model at startup from `--model`; there is no
load-model message, and a model that fails to load, including at the
tensor coverage gate, exits the process non-zero, so a partial model
never serves. Everything crosses the socket: there is no
shared-memory data plane, and no tensor or token stream travels
anywhere else.

Framing and versioning:

- One JSON object per line (NDJSON). A frame is a single line. The
  engine's reader caps a line at the configured maximum frame size
  (`--max-frame-bytes` / `ITAK_ENGINE_MAX_FRAME_BYTES`, default
  4,194,304 bytes) and refuses an over-long line as a named
  `bad_frame` error and closes, instead of growing a buffer without
  bound; the Go client caps a frame at the same 4 MiB.
- The first frame on every connection is Hello:
  `{"op":"hello","proto":1,"client":"torch-platform"}`. The engine
  answers with a `hello_ack` frame carrying its protocol version
  (1) and a detail string. A peer speaking another version gets an
  `error` frame with code `protocol_mismatch`, and the Go client
  likewise refuses a hello_ack carrying any other version, so a
  mismatch fails by name, never by hang and never by silent
  downgrade. Any frame arriving before Hello is refused.
- Before the handshake, the engine checks the connecting process's
  uid from the kernel peer credentials (SO_PEERCRED): a different
  uid gets one `peer_refused` error frame and the connection closes
  at once.
- Every operational limit on the seam is config with a default,
  never a compiled constant: frame size (above), the engine's slot
  count (`--slots`, default 1) and bounded job queue (`--max-queued`,
  default 8), the default generation cap (`--default-max-tokens`,
  default 128), the embed batch limits (`--max-embed-inputs`,
  `--max-embed-tokens`), and the platform supervisor's restart
  budget for the engine process (TODO T-9.11).

Message set (one `op` per frame):

| Message | Direction | Carries |
| --- | --- | --- |
| hello / hello_ack | both | Protocol version (1) and the client name; the ack carries the engine's detail string. A version mismatch is the named `protocol_mismatch` error instead |
| ping / pong | Go to Rust, reply | Health probe and its reply |
| debug / debug_info | Go to Rust, reply | The engine's self-report: model identity, tensor coverage, and slot and job counters |
| generate | Go to Rust | Prompt text (the engine owns the tokenizer; the platform sends text, never token ids), model name, max_tokens, stop strings, and the sampling block: temperature, top_p, top_k, seed, presence and frequency penalties |
| token | Rust to Go | One streamed piece of the completion, on the same socket: text and token_id per frame. The final frame carries `done` true and `finish_reason` set to `stop` or `length`, with empty text |
| embed / embeddings | Go to Rust, reply | Input texts; the reply frame itself carries one vector per input, in input order: mean-pooled token embeddings from the model's own embedding table, L2-normalized, as JSON arrays |
| chat_template / chat_template | Go to Rust, reply | The request asks for the model's own chat template; the reply carries the template text from the GGUF metadata plus the bos and eos token texts. A model whose file carries no template gets the named `no_chat_template` error. The platform renders the template; the engine only serves it |
| error | Rust to Go | A named refusal: `code` and `message`. The platform maps bad_request, context_overflow, model_not_found, and protocol_mismatch to a client error; forward_failed, detokenize_failed, engine_died, unknown_op, bad_frame, and internal to a gateway error; and engine_busy (every slot running and the bounded queue full) to 503 |

There is no abort message and no session or state message: the
engine's slots and job queue are internal, and the platform cancels
a generation by closing its connection, which its client does when
the request context ends; the exchange on that connection ends with
it.

Failure semantics:

- Engine crash: the platform marks the engine unhealthy, restarts it
  within the configured budget, and fails in-flight requests with a
  named error. Nothing hangs waiting on a dead process.
- Load failure: the model does not serve. There is no partial model
  state anywhere in the design.
- Any fallback (CPU backend instead of GPU, speculation off) is an
  explicit, logged, queryable state, never a silent one.

## 5. Design rules that close the audited gaps

Each rule names the audit finding it closes (native audit C and M
numbers, platform audit, gates audit, live test F numbers) and the
module that owns the rule. The full evidence lives in the four reports
in ~/workspace/model-teardown/.

### 5.1 Correctness of generation

| Rule | Closes | Owner |
| --- | --- | --- |
| The tokenizer is a real BPE or SentencePiece implementation, loaded from the model's own files, with token-for-token conformance tests against reference outputs and lossless round-trip tests | Native C1, live test F2, the platform tokenizer finding (the Ġ defect: detokenization reverses byte-level encoding, Ġ back to space, byte tokens reassembled) | itak-tokenizer |
| RoPE is applied in the forward step with the arch module's base and scaling; the final norm runs before the LM head; RMSNorm epsilon, activation (SwiGLU, GEGLU, and family variants), norm positions (pre, post, QK-norm), and logit transforms (softcap, scale) all come from the arch module, never from hardcoded defaults | Native C4 | itak-forward, itak-arch |
| Decode is incremental: one token per step against a real KV cache. Full re-forward per token does not exist anywhere in the design | Native M2, live test F1 | itak-forward, itak-kv |
| Sampling parameters, grammar, JSON mode, and stop rules in a request reach the sampler that runs. A parameter the engine cannot honor is refused in the response, never silently ignored | Platform constrained-decoding finding | platform/internal/engine, itak-engine |
| Every config flag has a behavior test proving the flag changes behavior. A flag read only by a log line is a build failure | Native M6 (the UseSpeculative printf), platform dead-feature pattern | All modules; enforced by the gates (5.6) |

### 5.2 The loader tells the truth

| Rule | Closes | Owner |
| --- | --- | --- |
| The ggml type table uses mainline numbering, full stop: BF16 is 30. Fork-only types (the BitNet I2_S family) live in a namespaced extension table keyed by source, never by renumbering mainline IDs | Native M3, live test notes | itak-gguf |
| Readers exist for every type the dissected corpus contains: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, and the K-quant mixes as shipped | Native C3a, live test F5 | itak-gguf, itak-kernels |
| Load produces a coverage report: required tensors per the arch map, found, readable, shape-checked. Any miss fails the load with a named error listing the tensor. Random or placeholder weights do not exist in this codebase; there is no code path that invents a weight | Native C3, live test F3 and F4 | itak-loader |
| Missing or unrecognized metadata fails the load with the key named. Defaults that fabricate a model (the old vocab 32000, hidden 768 fallback) are forbidden; vocab resolves from the arch metadata or the tokenizer's token count, and the source is in the coverage report | Native m4, live test F4 (five of eight test models ran with a fabricated vocab) | itak-loader, itak-arch |
| Shape mismatches in matmul are errors with shapes attached, and quantized payloads are first-class in every kernel path. A kernel never indexes a payload field it has not checked | Live test F3 (the safeMatMul panics) | itak-kernels |

### 5.3 Architecture dispatch is real

| Rule | Closes | Owner |
| --- | --- | --- |
| Routing is by the file's own `general.architecture` string through the arch registry, and the registry is the only way a model loads. An unrecognized string is a named refusal listing what is supported | Native C2, live test F4 | itak-arch |
| Each arch module's tensor map covers what the family's files actually contain: fused QKV (Phi-3), attention biases (Qwen2, StarCoder2), post-attention and post-FFN norms (Gemma 2, OLMo 2), QK-norms (OLMo 2), tied embeddings (Gemma, StarCoder2), up/down-only MLPs (StarCoder2), MLA tensors and expert tensors read under the file's own arch prefix (DeepSeek: expert counts come from `deepseek2.*` keys, never a hardcoded `llama.*` key) | Live test M1 to M4 and F4, native C3c | itak-arch family modules |
| Registry configuration numbers are verified against the published configs at build time by a conformance test (the audit's fetched corrections are the seed data: Mixtral RoPE base 1000000, Phi-3 mini context 4096 with window 2047, LLaMA-1 context 2048, Gemma vocab 256128, DeepSeek V3 MLA fields present and its memory estimate computed from per-layer expert math, not a dense shortcut) | Native M4, M5, m1 to m3, m6 | itak-arch, gates |
| The serve path never routes by file size. Engine selection is by capability: arch support and backend availability, stated in the load result | Native M1 | model operations (not yet built), itak-engine |
| State save and load either persist and restore, verified by a round-trip test, or the verb does not exist. Silent no-op success is forbidden in every module | Native m5 (SaveKVCache no-ops), platform state endpoint finding | itak-kv, platform/internal/server |

### 5.4 Speculation and subsystems are wired or absent

| Rule | Closes | Owner |
| --- | --- | --- |
| Speculative decoding exists in exactly the modes the engine executes: DFlash block drafting (Glimmer), n-gram lookup, Medusa from trained artifact files. The generation loop calls the decoder; a draft model configured but not called is impossible by construction | Native M6, platform spec finding (draft loaded, never run) | itak-spec |
| Sliding-window attention, paged KV, and ring caches are the KV implementation itself, called by the forward step, not library ornaments beside it | Native M7 | itak-kv |
| Medusa heads are files with version fields, produced by the training stack, loaded and verified at load time. Zero-initialized heads are a load state the engine reports, never a silent default | Native M6, PLAN S6 | itak-spec, training stack (not yet built) |
| A CLI or endpoint that cannot do its work exits non-zero or returns an error saying so. Printing success for simulated output is forbidden (the itak-quant lesson) | Platform itak-quant finding | model operations (not yet built), all CLIs |
| Header comments and docs are checked against behavior in review; a module whose docs claim what its code does not do fails the gate | Native M8 | Gates (5.6) |

### 5.5 Security from birth

| Rule | Closes | Owner |
| --- | --- | --- |
| Every route group sits behind auth middleware from the first server commit. API keys are per-user and hashed at rest; keys reach people through secure links, never files or chat | Platform C1 | platform/internal/auth, platform/internal/server |
| The server binds 127.0.0.1 by default. Binding wider requires an explicit flag, and startup diagnostics state the bind and the auth state in one line | Platform C1 (0.0.0.0 under a localhost comment) | platform/internal/server |
| Filesystem paths from requests are confined to configured directories and validated (the old validatePath discipline, applied everywhere, including state save and load) | Platform C2 (arbitrary path state endpoints) | platform/internal/server, itak-kv |
| Cluster join requires the shared secret and an allowlisted peer address; there is no automatic LAN discovery unless a flag turns it on | Platform swarm finding | cluster (not yet built) |
| Tenant identity from auth is part of scheduler decisions and cache keys from the first version, so isolation is a property of the data flow, not a later layer | Platform tenant finding | platform/internal/auth, platform/internal/scheduler, platform/internal/cache |

### 5.6 Gates from birth

The old tree's gates were red or absent (gates audit: test panic masking
about 120 tests, 165 of 167 FFI tests skipping, no lint config anywhere,
an assembly ABI mismatch in a hot kernel, a mandatory undocumented
sibling folder to build at all). The rebuild's rules:

- The repo carries .golangci.yml and clippy configuration from Stage 0,
  plus CI that runs build, vet, lint, and tests on every push. A red
  gate stops the line; nothing commits on red (commit-gates skill).
- Every package carries tests in proportion to its risk, and the
  training stack carries tests from its first commit.
- Tests that skip for a missing environment piece fail CI unless the
  skip is an explicit, listed exception. A suite that is green because
  it did not run is a red suite.
- Hand-written assembly and unsafe kernels carry ABI conformance tests
  (the dotQ4BlockAVX2 lesson: declaration and assembly must agree, and a
  test, not vet alone, proves it).
- Dependencies are published and versioned. The build never requires an
  undocumented sibling checkout (the ../Core lesson); itak-core is a
  versioned module from Stage 0.
- The conformance corpus is the teardown set: the thirteen dissected
  GGUF files from the 2026-10-29 model teardowns, with expected outcomes
  per file recorded in the test data. Loader, tokenizer, and arch tests
  run against real files, not fixtures invented to pass.

## 6. Glimmer as the first-class target (PLAN S13)

The muse_glimmer arch module is specified from the dissection
(~/workspace/glimmer-blueprint/GLIMMER-DISSECTION.md), not from
assumptions:

| Fact from the file | Design consequence |
| --- | --- |
| 52 layers: 39 sliding-window (window 2048), 13 global at indices 3, 7, 11, up to 51 | The KV store is hybrid by construction: 39 fixed rings of 2048 slots, 13 paged global caches. Whole-context KV stays near 1.7 GiB at 131K, so KV work is deliberately minimal; the effort goes to weight streaming |
| Hidden 6656, FFN 19968, FFN is 75 percent of all bytes | The specialized kernels target exactly two shapes first: the FFN GEMV at 6656 by 19968 and the attention projections. Sensitivity-ranked quant handling starts at the FFN |
| 32 Q heads, 2 KV heads, head dim 128 | GQA at 16 to 1 is a first-class kernel case, not a generic path |
| RoPE base 500000 | From the arch module's config, read from file metadata |
| Logit softcap 20.0, logit scale 0.1961, untied Q5_K output head | Logit transform lives in the arch module; the Q5_K reader is on the critical path for this model, which is why 5.2 lists it by name |
| Vocab 202048, GPT-2 BPE with the llama4 pre-tokenizer | The tokenizer conformance set is Glimmer's own prompts first |
| Quant mix: Q4_K 77.3 percent, Q6_K 17.1 (ffn_down and attn_v alternate by layer half), Q5_K 5.5, F32 norms | Dequant kernels are tuned for this mix; the alternating Q6_K pattern is data in the coverage report, not a surprise |
| DFlash draft: own arch (dflash), 5 layers, block size 16, 1.51 GiB | itak-spec loads it as a companion through the same loader and arch seam; the draft depth sweep (parked T-P.1) runs against the new stack and its result is config, not code |
| mmproj vision tower, 1.30 GiB | Vision enters through the platform's modality routes after text is proven; the tower loads through the same loader rules |

The speed ladder, measured on the 7900 XTX against the recorded
baselines (llama.cpp Vulkan 44.31 tok/s plain, 88.6 effective with
DFlash, ceiling 60 plain at this quant):

1. Verified text on CPU (correctness before speed, always).
2. Verified text on the wgpu Vulkan backend.
3. Plain decode at or above 44.31 tok/s, then toward the specialized
   target near 51.
4. DFlash on the new stack at or above the 88.6 baseline, then the
   depth sweep.

A speed number without the verified-text proof at the same commit does
not count, in either direction.

## 7. Staged build order (skeleton first)

Each stage lands green or the next does not start. Stages map to TODO
phases; Phase 9 is the Glimmer fast path that runs through them.

| Stage | Content | TODO | Exit proof |
| --- | --- | --- | --- |
| 0. Skeleton and contracts | Repo layout, itak-core as a published module, Rust workspace with every crate as a compiling skeleton behind its interface, IPC Hello round trip over the Unix domain socket, gates and CI live | Phase 1 | Both processes start, exchange Hello, and shut down clean; gates green on skeletons |
| 1. Truthful loading and tokenization | Full type table, coverage report, fail-loud loader, arch dispatch, real tokenizer; conformance over the thirteen-file corpus | Phase 2 (T-2.1, T-2.3, T-2.7, T-2.8) | Every corpus file loads with a complete coverage report or a named refusal; tokenizer matches reference token for token |
| 2. First verified text (dense, CPU) | Incremental forward with KV for the llama-shaped families (llama, qwen2 with biases), sampling, EOS | Phase 2 (T-2.4, T-2.5) | Generated text matches the reference engine's output on the bench prompt set at temperature 0 |
| 3. Glimmer end to end in the engine | muse_glimmer module, hybrid KV, softcap path, CPU verified text, then the wgpu backend, then DFlash | Phase 9, with Phase 4 backend items | Glimmer generates verified text on our own stack, CPU then GPU, speed ladder steps 1 to 4 recorded |
| 4. Platform reattach | Server, auth, scheduler, caches, sampling and grammar wiring, model ops, cluster; Glimmer served through the seam | Phase 6, Phase 9 serve items | An authenticated API request returns Glimmer text served by itak-engine; the platform test set passes against the new engine |
| 5. Remaining families | gemma2, phi3, mistral, olmo2, starcoder2, deepseek2 (MLA and MoE), each as an arch module with its conformance entries | Phase 2/5 extensions, Phase 9 corpus items | Every corpus file at its recorded expected outcome: generates, or refuses by name for a stated reason |
| 6. Speculation, training, formats | N-gram and Medusa wired to trained artifacts, training stack reattach with tests, .itf, Mojo lab track continues | Phases 3, 7, 8 | Per-phase proofs in TODO.md |

## 8. Salvage manifest (what comes from the old project)

Taken, audited as it lands: the platform's working core concepts and
code (scheduler and slot batching, response and prefix caches, model
pulls and integrity checks, Ollama-compatible surface, GGUF converter),
the FFI speculative decoding logic as the reference for itak-spec
behavior, the trainer surface as the specification for the training
stack (not yet built), the
Mojo engines as lab artifacts, and the design record (the original PLAN,
DECISIONS, and research notes in this repo).

Rebuilt, not ported: the loader, the tokenizer, the forward pass, the
arch registry, and the KV subsystem. The audits showed these are the
parts whose old implementations cannot be repaired into correctness.

Dropped: the file-size routing rule, the placeholder quant CLI's
behavior (rebuilt to produce files or fail), every silent fallback, and
every feature claim with no wired code behind it. A feature enters the
README when its proof exists, not before.

## 9. Non-goals for this blueprint

- No llama.cpp FFI in the new engine. The FFI path stays available in
  the old tree as the reference implementation and the fallback serving
  path until Stage 4 proves the new stack; the version-lock fix for it
  (making the llama_params_fit prep optional) is old-tree maintenance,
  tracked outside this blueprint.
- ROCm and HIP stay parked per DECISIONS (2026-10-26); Vulkan via wgpu
  is the GPU path.
- Multi-GPU, tensor parallelism, and disaggregated serving are designed
  at the seam (the IPC contract does not forbid them) and built only if
  the single-card path proves out (PLAN S7).
