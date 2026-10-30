<!--
INSTRUCTIONS FOR THIS FILE (user and AI: read before editing. Keep this header.)

WHAT THIS FILE IS: the whole plan for this project, step by step. ENDGOAL.md
is the destination; this file is the route; TODO.md is the work list. When
the plan changes, this file changes FIRST, then TODO.md follows.

EXACT STRUCTURE:
- "## S1: What it is" (the product in plain words, who it serves).
- "## S2: Architecture" (the grid: modules, what each owns, the seams).
- Build steps as numbered sections "## S<n>: <step name> (TODO Phase <n>)",
  one section per build step, each naming the shelf repos it uses (from
  ~/workspace/google-org-harvest/USE-IT.md) and the hosting or delivery
  shape where that matters.
- A final "## S<last>: Open questions" section (owner named per question).

RULES:
- Section numbers are stable once published; TODO items point at them with
  (PLAN S<n>). Renumber only in a deliberate pass that updates both files.
- Facts only: what is decided is stated, what is open is an open question.
- Before designing a step, check the shelf: take or retrofit what already
  exists under a usable license instead of rebuilding it.
- No secrets, passwords, or tokens in this file, ever.
-->
# PLAN: iTaK Torch Rebuild

## S1: What it is

iTaK Torch Rebuild is the second build of David Patton's inference engine, correcting the structural failure of the first. The original iTaKTorchQ4 (pure Go) combined an inference engine, a serving platform, a training stack, and a model lab in one tree of about 31,600 lines in its native engine alone, and it cannot be built today: its shared support module (iTaKCore: types, health, event, debug packages) was referenced by a relative path in go.mod, was never published, and no copy survives anywhere reachable. The measured usage surface of Core is small and known, and the original repos survive as the reference implementation.

The rebuild serves the same audience as the original: David's own projects first (a local model service for his tools and his server), and anyone running serious models on consumer hardware. The design target is unchanged from the original: fast output with good quality on small VRAM, huge models on hardware that should not be able to hold them.

Two bodies of evidence in this repo ground the design:

- The hardware record (docs/research/2026-10-26-glimmer-7900xtx-benchmarks.md): Muse Glimmer 30B benchmarked per quant on the RX 7900 XTX, the ROCm system-level diagnosis, and the production llama-server configuration the measurements support.
- The research base (docs/research/2026-10-27-mamba-papers-ingest.md): the Mamba-1, Mamba-2, and Mamba-3 papers, and a grading of David's extrapolated Mamba-3 variant in his Mojo engine against the real paper.

## S2: Architecture

The grid. Two programs, one seam, a research lab beside them.

**The engine (Rust workspace, one crate per module):**

- itak-types: the shared contracts (inference request/response, chat messages, token usage, latency stats, model info and parameters). Versioned data with version fields.
- itak-diag: structured logging, health state, event emission. The debugging law as a shared crate.
- itak-loader: the arch-trait adapter over the gguf crate (GGUF parsing and memory-mapped weight access live in gguf); .itf support lands in S11.
- itak-kernels: quantized tensor math. Q4_K, Q6_K, Q8_0 dequant and matmul first; FP8, AWQ, and ternary (BitNet-style) after. RoPE, RMSNorm, softmax, SiLU.
- itak-kv: KV cache family as preallocated pools: paged, ring, INT8-quantized, compressed, and offloaded tiers; radix prefix sharing; attention-sink eviction; context freezing.
- itak-ssm: the state space module. One scan implementing the general three-term recurrence (alpha, beta, gamma), so Mamba-1 and Mamba-2 are the beta = 0 special case and Mamba-3 is the full form; data-dependent rotation applied to B and C; MIMO rank R as config (R = 1 is SISO); BCNorm and B, C biases in the block.
- itak-moe: MoE routing (including cache-aware routing as an experiment behind a flag), expert cache with pin and LRU tiers, next-layer prefetch, expert streaming from disk.
- itak-spec: speculative decoding: prompt-lookup n-gram drafting with state rollback, Medusa tree drafting with trained versioned heads, MTP decoding, rejection sampling, the EKF speculative update kernel, early-exit and layer-dropped drafting with calibration.
- itak-backend: the backend trait and its implementations: CPU SIMD, wgpu (Vulkan on the 7900 XTX), Metal and CUDA behind the seam when earned.
- itak-train: training compute: autograd, optimizers (AdamW, Adam8bit, Muon, GaLore), checkpointing (selective and activation), quantization-aware training.
- itak-dissect: analysis tooling (separate binary): per-layer weight statistics, heat mapping from measurement rather than position, re-quantization planning, .itf conversion.

**The platform (Go modules, largely reattached from the original tree):**

- Core, rebuilt as a published versioned Go module: pkg/types, pkg/health, pkg/event, pkg/debug, written to the measured call-site surface of the original.
- API handlers: OpenAI-compatible chat, completions, embeddings, models, plus Ollama compatibility, speech, tools, and LoRA management handlers.
- Serving machinery: continuous batching, chunked prefill, scheduler with priority queue and slot manager, token budgets, rate limiting, A/B routing, payload-aware request routing.
- Request-side caches: prefix, prompt, response, and semantic caches; grammar (GBNF), JSON mode and schema enforcement, token healing, guardrails, output watermarking.
- Clustering: cluster coordination and discovery, swarm behaviour, tenant isolation, graceful shutdown.
- Model operations: HuggingFace and OCI pulls, Ollama pulls, integrity verification, model versioning, warmup, page prefaulting, GGUF conversion and export.
- Observability: metrics and metrics export, KV metrics, request tracing, structured logs, diagnostics endpoints.

**The seam:** engine and platform are separate processes. Control messages travel over a local socket; tokens and tensors travel through shared memory so nothing serializes in the hot loop. The engine restarts or upgrades without the platform going down; if the engine dies, the platform stays up and reports it honestly. This replaces the original's FFI coupling, which version-locked the shipped binary to one llama.cpp build.

**The lab (Mojo):** the Latent Reasoning Engine research (token-free latent reasoning, learned patch compression, EGGROLL memory crystallization) and new kernel prototypes live in Mojo, as they do today. Proven ideas graduate into the Rust crates through the plan, never by copy-paste.

**Feature audit (the bar for completeness):** every file-level capability found in the original tree is accounted for in the phases below: the training wing (trainer, SFT/DPO/GRPO/ORPO, reward model, autograd, three extra optimizers, fused backward, checkpointing, tokenizer trainer, train orchestrator), the platform wing (everything listed above, plus CPU topology detection and auto-config, multi-GPU, tensor/pipeline/sequence parallelism, disaggregated prefill/decode serving, CUDA graphs, FlashAttention and FlashDecode, beam search, sliding window attention, JIT compilation, vision engine and VLM input, embedding pooling, model merging, weight compression, quantize-on-load, eval harness and benchmark suite, health monitor, heterogeneous scheduling, data prefetch, memory arenas and tensor pools). A feature is either built and proven in its phase or retired by a DECISIONS.md entry. Nothing evaporates silently; that is how the original's scope was lost track of the first time.

## S3: Evidence base (TODO Phase 0)

Assemble and publish the record the design stands on: the original engine audit and feature inventory, the RX 7900 XTX benchmark record, the ROCm diagnosis, the inference-platform research, and the Mamba papers ingest with the grading of David's variant. Status: done 2026-10-26 and 2026-10-27; the documents live in docs/research/ and bench/ in this repo. Shelf repos used: none; this phase is evidence, not code.

## S4: Foundations: Core and contracts (TODO Phase 1)

Rebuild Core first, because everything imports it and its loss is what killed the original. Write pkg/types, pkg/health, pkg/event, pkg/registry, and pkg/debug to the measured call-site surface (the six importing files in the original Torch are named in the benchmark report). Port pkg/debug and pkg/event from their surviving ancestors in the GOAgent repo (the structured logger and the event bus still live there) rather than inventing them. Publish as a versioned Go module; the Rust itak-types crate mirrors the same contracts with version fields in the data. Proof of done: the original Torch tree, copied beside the rebuilt Core, compiles against it. That single test is the resurrection gate for the whole project.

## S5: Rust engine core (TODO Phase 2)

Skeleton first: workspace, crate boundaries, the backend trait, config loading with every limit as config. Then the gguf crate (GGUF parsing, mmap) with itak-loader as the arch-trait adapter over it, itak-kernels (Q4_K matmul first), a real BPE tokenizer, the KV-cached transformer forward for the Llama and Qwen families, sampling (temperature, top-p, top-k, penalties as config), and the OpenAI-compatible minimal serve path through the seam. Verification is against llama.cpp on the same models (the Qwen2.5 test set from the benchmark work): same prompts, compared outputs, published speed table. Shelf check before building: candle and mistral.rs (MIT/Apache) are the reference Rust implementations to learn from; the kernels here are written from the project's own Mojo and Go reference code, not copied.

## S6: Speculation and advanced inference (TODO Phase 3)

itak-spec in full: n-gram prompt lookup with rollback (working today in the Mojo engine; port the algorithm), rejection sampling (proven in the original's FFI path), Medusa heads done right (a training path in S10 produces versioned head files; zero-initialized heads with no loader, the original's state, is a DECISIONS-level anti-pattern and is not repeated), MTP decoding, PowerInfer-style activation prediction with the hot/cold neuron data coming from itak-dissect measurements, AttnRes depth-attention residuals, the EKF speculative update, early-exit drafting with auto-discovery calibration actually wired on. Constrained decoding lands here too, on the platform side of the seam: GBNF grammars, JSON mode, JSON schema, token healing.

## S7: GPU backend and hardware (TODO Phase 4)

itak-backend's wgpu implementation targeting Vulkan on the RX 7900 XTX first (the proven path on the actual hardware), CPU SIMD as the always-available fallback, hardware detection and auto-config that correctly identifies AMD cards on Linux (the original reported the 7900 XTX as 0 MiB and forced CPU-only; that bug is a named test case). ROCm/HIP enters only through the open question in S12; nothing in this phase waits on it. Multi-GPU, tensor parallelism, and disaggregated prefill/decode serving are designed at the seam in this phase and built if the single-card path proves out.

## S8: SSM hybrid and MoE streaming (TODO Phase 5)

itak-ssm per S2, with the Mamba-3 mode (three-term recurrence, data-dependent rotation, MIMO rank) behind a config flag, off by default, until the state-tracking checks and a perplexity comparison against the Mamba-2 mode pass on real runs; then it earns its default. Hybrid layouts (SSM layers interleaved with attention, ratio as config; the papers point at roughly 5:1) are a first-class architecture family alongside Llama and Qwen, registered through the same arch-handler seam the original's universal engine design called for. itak-moe delivers Colibri-pattern streaming: experts on disk, pin and LRU cache in RAM, prefetch of the next layer's likely experts, sized so a model several times VRAM still serves. The Nemotron hybrid layout from the Mojo engine is the test fixture.

## S9: Go platform reattach (TODO Phase 6)

Bring the platform modules over from the original tree in dependency order, each audited as it lands (the audit_code standard: the original code is reference, not gospel; its pedagogical native forward path does not come over, the Rust engine replaces it). Order: server and handlers, scheduler and batching, request caches, model operations, clustering and tenancy, observability. The training API handlers land here; the training compute they call lands in S10. Every module arrives with its debug seam and lint gates or it does not land.

## S10: Training stack (TODO Phase 7)

itak-train: autograd, tensor pools and arenas, the optimizer set, SFT trainer first (it feeds Medusa head training in S6), then LoRA end to end through the platform's training API, then DPO, GRPO, and ORPO with the reward model, quantization-aware training, and the tokenizer trainer. The original tree is the specification; its trainer files name the intended surface. Proof standard: a small model fine-tunes on a real dataset and the served result measurably changes on the eval harness.

## S11: .itf format, tools, and the Latent Engine research track (TODO Phase 8)

The .itf (iTaK Tensor Format) specification from the original design record, implemented: header with magic ITF1 and a version field, architecture and layer metadata, quantization version, memory tier sizing, latent state dimensions, per-tensor heat levels and SSM state sections. The converter (GGUF to .itf) ships in itak-dissect, and the heat tiers come from measured per-layer statistics, replacing the original dissector's positional heuristic with data. The Latent Reasoning Engine stays a research track in the lab: byte-level encoding, learned patch compression, latent reasoning passes, tool interception from latent activations, EGGROLL crystallization experiments. It graduates into the engine only through a plan change, with a proof-of-concept result attached.

## S12: Open questions

- ROCm remediation on the home server: clean reinstall plus the reset window it likely needs. The server must not reboot is a standing rule, so this waits for a maintenance window David approves. (owner: David)
- Does the trapezoidal beta term in David's Mojo variant carry the paper's decay factor, and were his rotation angles data-dependent? Two line-level checks in nemotron_hybrid.mojo that decide whether his Mamba-3 mode was fully equivalent or two thirds of it. (owner: research)
- DFlash draft depth for Glimmer on the 7900 XTX: the pasted article claims 16 and 75 to 110+ tok/s; our research said draft n-max 4 and we never benchmarked DFlash itself. One bench session settles it, once the weights are re-downloaded to the array. (owner: bench)
- Sampling defaults for Glimmer (the article's temp 1.0, top-p 0.95, top-k 64 versus the values validated in our accuracy runs). Settle by accuracy prompts, not by preference. (owner: bench)
- Glimmer architecture support in itak-ssm and the arch registry: the muse-glimmer family is not in the original's registry. Scope it as its own TODO phase when the dense families are proven. (owner: David)
- License posture for ports: algorithms from the Mojo and Go trees are David's; the Colibri C engine is a separate upstream project. The MoE streaming module implements the pattern in Rust, it does not vendor Colibri code. Confirm before S8 starts. (owner: David)

## S13: Glimmer first-class, and the audit closure (TODO Phase 9)

Settled by David's order of 2026-10-29, the same day the four audit reports landed (native engine, platform, gates, and the live native test: 0 of 8 families generated a token). The S12 open question about scoping muse-glimmer as its own phase is answered: it is Phase 9, and it is the fast path through the earlier phases rather than a tail item. ARCHITECTURE.md is the blueprint for this section: the module grid, the Go and Rust IPC contract, the design rules that close every audited gap by construction (real tokenizer, RoPE and final norm, per-arch tensor maps keyed on the file's own architecture string, complete quant readers on mainline type numbering, fail-loud loading with a coverage report, incremental KV-cached decode, security and gates from birth), and the salvage manifest for the old tree. The milestone this section exists to reach: Muse Glimmer 30B generates verified text served end to end by our own Rust engine and Go platform, at or above the recorded llama.cpp baselines on the 7900 XTX (44.31 tok/s plain Vulkan, 88.6 effective with DFlash). The remaining families reach their recorded conformance outcomes through the same arch seam in this phase.
