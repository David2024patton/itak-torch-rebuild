# SALVAGE: iTaKTorchQ4 + rescued Core, module-by-module verdicts

Written 2026-10-29 for the rebuild ordered the same day: fix everything,
rebuild the hot parts in Rust, reuse whatever is sound, end state is Muse
Glimmer 30B running 100% on our own Go platform + Rust engine stack
(PLAN.md S2: two processes, socket + shared-memory seam).

Evidence base: the three 2026-10-29 audits in
~/workspace/model-teardown/ (audit-platform.md, audit-native-engine.md,
audit-gates.md), cited below as [P], [N], [G] with their finding IDs, plus
caller greps run during this inventory. Verdict key:

- **REUSE**: carries over as is.
- **REUSE-FIX**: carries over with the named fixes.
- **RUST**: rewritten as Rust in the named crate (design/algorithms salvaged, code not).
- **GO-REWRITE**: rewritten in Go (design salvaged, code not).
- **DROP**: not carried over; reason stated. Per PLAN S2, a DROP here is a
  retirement recommendation for a DECISIONS.md entry, not a silent evaporation.

Ground rules inherited from the audits and DECISIONS.md: no relative-path
module replaces ever again; every reused module ships with debugging and
lint from birth (the old tree had no lint at all [G]); nothing counts as
reused until the gates run green on it.

---

## 1. Destination: Rust engine crates

### itak-types

| Source | Verdict | Reason |
| --- | --- | --- |
| native/model_configs.go (13 configs, 8 arch constants) | RUST | The data, not the code: headline dims verified against published configs [N], but the registry is dead code [N-C2]; port the tables as loader arch configs with the known errors corrected (Mixtral RoPE base, Phi-3 context, LLaMA-1 context, Gemma vocab [N-M4, m1-m3]). MemoryEstimateMB is ~150x off for MoE and does not port [N-M5]. |

### itak-loader

| Source | Verdict | Reason |
| --- | --- | --- |
| native/gguf.go (parser + dequantizers) | RUST | Real dequant code for Q4_0..Q6_K/I2_S, but type IDs collide with mainline GGUF (type 30 read as I2_S, is really BF16) and Q3_K/Q5_K/Q8_1 readers are missing [N-C3, N-M3]; rewrite with mainline numbering and full coverage, salvage the format knowledge. |
| native/mmap_loader.go, mmap_unix/windows.go | RUST | mmap weight access is the engine's load path per PLAN S2; small, sound-shaped, rewrite cleanly in Rust. |
| native/safetensors.go | RUST | Needed for HF-native weights beside GGUF; has tests [G]; no audit defect found. |
| native/model_file.go, model_shard.go | RUST | Shard layout knowledge feeds the MoE streaming loader; deferred behind single-file loading. |
| native/bpe_tokenizer.go | RUST | The engine needs a real tokenizer in the token loop; this is the only engine-side BPE implementation in the tree (the native engine itself used a fake char tokenizer [N-C1]). Port the BPE design, verify against the GGUF vocab. |
| native/quantize_on_load.go, dynamic_quant.go | RUST | Quantize-on-load is a loader seam feature in the PLAN feature audit; no platform wiring existed [P group 4]. |
| loader/ (pkg/torch/loader, lib loader) | DROP | It loads the llama.cpp shared library for the FFI build; the Rust engine has no FFI library to load. Its passing tests [G] die with the FFI path. |

### itak-kernels

| Source | Verdict | Reason |
| --- | --- | --- |
| native/dequant.go, quant_dequant.go | RUST | The dequant tables/routines are the measured core of Q4_K/Q6_K decode; rewrite in Rust with the Glimmer byte census as the test fixture. |
| native/matmul.go, simd_kernels.go, dot_amd64.go, dot_q4_amd64.go (+ .s), cmd/asm/gen_q4_amd64.go | RUST | Hot quantized matmul is the first Rust crate per PLAN S5; the Go asm carries an asmdecl ABI mismatch in the Q4 dot product [G], which the rewrite retires. Salvage the blocking/tiling approach, not the asm. |
| native/tensor.go, tensor_pool.go, memory_arena.go | RUST | Tensor/pool/arena foundations for the crates; the GOTensor design is the salvage, the demo engine built on it is not (see DROP). |
| native/rope.go, rope_scaling.go | RUST | Real RoPE math exists here as library code; the old forward never called it [N-C4]. Port and wire this time, with per-family scaling (YaRN etc.) as config. |
| native/attention.go (GQA) | RUST | Reference GQA math salvages as the correctness oracle for the fused kernels; it takes no position info today [N-C4], so the rewrite adds RoPE at this seam. |
| native/flash_attention.go, flash_decode.go | RUST | Orphan files with zero serve-path references [P group 5]; the designs port to itak-kernels/itak-backend behind the backend trait. |
| native/fused_kernels.go, deep_kernel_fusion.go | RUST | Fusion designs (norm+matmul class) port as kernel specs; deep_kernel_fusion is speculative and ports last, behind flags per DECISIONS 2026-10-27. |
| native/ternary.go (+ _tools/bench_ternary.go) | RUST | BitNet-style ternary is an explicit itak-kernels roadmap item (PLAN S2); the Go code is the design reference and its bench harness concept carries. |
| native/logit_processor.go | RUST | Logit processors run inside the token loop, so they belong engine-side; previously native-only and never invoked from the platform [P group 3]. |
| native/token_healing.go | RUST | Same reasoning as logit processors: per-token engine work, orphaned in the old tree. |

### itak-kv

| Source | Verdict | Reason |
| --- | --- | --- |
| native/kv_cache.go (ring), paged_attention.go, radix_attention.go, kv_cache_int8.go, kv_compress.go, kv_offload.go, kv_prefetch.go, hot_cold.go, attention_sink.go, sliding_window.go | RUST | The full KV family PLAN S2 specifies (paged, ring, INT8, compressed, offloaded, radix sharing, sink eviction, sliding window) exists here as complete-looking library code with zero call sites in the old engine [N-M7]. This is the single biggest design salvage in the tree: port the family as one crate, wired into the forward from day one, with Glimmer's hybrid pattern (39 SWA + 13 global layers, from the teardown) as the acceptance case. |

### itak-ssm

| Source | Verdict | Reason |
| --- | --- | --- |
| (no source in the old tree) | RUST | Nothing to salvage: the old tree has no SSM code at all. Built new from the Mamba papers ingest (PLAN S2/S3). Recorded here so the absence is explicit, not discovered later. |

### itak-moe

| Source | Verdict | Reason |
| --- | --- | --- |
| native/moe.go (LoadMoEFromGGUF + routing) | RUST | The loader hook exists but the forward's MoE branch processes only the last position and zero-fills the rest [N-C4], and correct parsing of DeepSeek expert layouts is unverified [N]. Port the tensor-layout knowledge; write routing fresh. |
| native/expert_offload.go, data_prefetch.go | RUST | Expert cache/prefetch/streaming is itak-moe's core job (PLAN S2) and Glimmer-class MoE streaming is a headline goal; these are the design drafts. |
| native/sparse_ffn.go, sparse.go | DROP | No home in the PLAN grid; MoE routing covers the sparsity story. Retire. |

### itak-spec

| Source | Verdict | Reason |
| --- | --- | --- |
| native/speculative_decoder.go (v1) | RUST | Constructed only by its own test with toy engines [N-M6], but it is a working draft/verify design; port as itak-spec's starting point. |
| native/mtp_decoder.go | RUST | Multi-token-prediction decoding is an itak-spec module per PLAN S2; orphan in the old tree. |
| native/beam_search.go | RUST | Beam search is engine-side decoding; orphan file, sound shape, port with the spec crate. |
| spec_decode.go (pkg/torch, FFI speculative) | DROP | The one speculative path that was wired [N-M6] is wired into the llama.cpp FFI engine, which is retired (see llama/ below); it also never actually ran in Torch's serve loop [P]. Its acceptance-tuning lessons are in the bench record, not the code. |
| native/speculative_v2.go (Medusa) | DROP | Known dead: zero references, no test, and no train/load/serialize path exists for its heads, so they can never hold trained weights [N-M6]. The Medusa concept re-enters via PLAN's itak-spec (trained, versioned heads) as new code, not as this file. |
| native/predictor.go | DROP | Early-exit drafting predictor; the concept is already specified in PLAN's itak-spec. The file itself is an orphan draft. |

### itak-backend

| Source | Verdict | Reason |
| --- | --- | --- |
| native/tensor_parallel.go, pipeline_parallel.go, sequence_parallel.go | RUST | Parallelism designs accounted for in the PLAN feature audit; zero serve-path references [P group 5]. Port deferred, behind the backend trait, after single-device decode is proven. |
| native/wgpu.go, wgpu_stub.go | DROP | The Go WebGPU binding attempt (cogentcore) is superseded by Rust wgpu in itak-backend; 1,241 lines tied to the dead engine. |
| native/gpu_cuda.go, gpu_cuda_unix/windows.go, gpu_metal.go | DROP | Go-side GPU probing for the dead engine; backend detection is itak-backend's job in Rust. Platform gpu_detect.go (below) is the salvage for the Go side. |
| native/cuda_graphs.go | DROP | NVIDIA-only capture machinery with no backend to capture in the new stack's first phases (Vulkan-first per DECISIONS 2026-10-26). Retire; revisit if a CUDA backend is ever earned. |
| native/jit_compiler.go | DROP | A JIT layer for an engine that is being rewritten in ahead-of-time Rust; no target. Retire. |
| native/distributed_launch.go, disaggregated_serve.go, heterogeneous_scheduler.go | DROP | Cluster and scheduling live in the Go platform (PLAN S2); engine-side duplicates with zero callers. Retire. |

### itak-train

| Source | Verdict | Reason |
| --- | --- | --- |
| native/autograd.go | RUST | Training compute is Rust per DECISIONS 2026-10-27; autograd is its foundation. Zero tests exist for the whole training stack [G], so port with tests written against it, not trust in it. |
| native/trainer.go, advanced_training.go, training_progress.go | RUST | The trainer core and progress reporting; reachable only via the CLI in the old tree [P group 1]. |
| native/sft_trainer.go, dpo_trainer.go, grpo_trainer.go, orpo_trainer.go, reward_model.go | RUST | The full post-training family; all design salvage, none of it ever exercised by a test [G]. Ports after the trainer core proves out. |
| native/optimizer.go, adam8bit.go, muon_optimizer.go, galore_optimizer.go, lr_scheduler.go | RUST | The optimizer set PLAN S2 names for itak-train (AdamW, Adam8bit, Muon, GaLore). |
| native/quantization_aware_training.go | RUST | QAT is an itak-train module per PLAN S2 and ties directly to the ternary/quant work David cares about. |
| native/fused_backward.go, activation_checkpoint.go, selective_checkpoint.go, gradient_accum.go, mixed_precision.go | RUST | The memory-saving training machinery; one grouped salvage, all untested [G]. |
| native/lora.go | RUST | Train-side LoRA (adapter math) is engine/lab code; the serving-side fake is judged separately below. |
| native/training_data.go, tokenizer_trainer.go | RUST | Dataset handling and tokenizer training for itak-train; deferred with the trainers. |
| native/attn_residual.go, attn_res_finetune.go, reasoning_optimizer.go | DROP | Research experiments with no PLAN home; the Mojo lab is where experiments live (PLAN S2). Retire from the engine tree. |

### itak-dissect

| Source | Verdict | Reason |
| --- | --- | --- |
| native/gguf_writer.go, gguf_export.go | RUST | A real GGUF writer is the conversion half of itak-dissect and the fix for itak-quant's lie (below); the Go writer is the format reference. |
| native/awq_quantize.go, fp8_quantize.go, weight_compress.go, weight_merge.go | RUST | The quantize/merge/compress toolset PLAN assigns to itak-dissect; all were native-only with no platform wiring [P group 4]. |
| native/eval_harness.go | RUST | Eval belongs with analysis tooling; untested in the old tree [G], port with the dissect bench suites as fixtures. |
| native/benchmark.go, benchmark_suite.go | DROP | Superseded by this repo's bench/ record and itak-dissect's own future bench mode; two more orphan suites help no one. Retire. |

### Engine core itself

| Source | Verdict | Reason |
| --- | --- | --- |
| native/engine.go, engine_features.go | DROP | The pedagogical demo engine: fake tokenizer [N-C1], no RoPE/final norm [N-C4], silent random weights on any load failure [N-C3], no KV cache so generation is a full re-forward per token [N-M2]. Nothing in it is load-bearing; the new engine is written against itak-kernels/itak-kv directly. |

---

## 2. Destination: Go platform packages

### Server, API surface, and request path

| Source | Verdict | Reason |
| --- | --- | --- |
| server.go | REUSE-FIX | The wired server core [P], with three mandatory fixes: bind 127.0.0.1 by default (it binds 0.0.0.0 while claiming localhost [P-C1]), auth middleware on every route group using Core's auth package (below), and no auto-started LAN cluster discovery [P]. |
| types.go, models.go, config.go | REUSE | Request/response and config types the whole platform shares; CompletionParams already carries Grammar/ResponseFormat/Temperature/TopP, they were just never read [P]. |
| engine.go (Engine interface) | REUSE-FIX | The interface is the right seam shape; it gains a Rust-engine client implementation over the socket/shm seam instead of the FFI and native implementations. |
| llama_engine.go | GO-REWRITE | The FFI engine implementation retires with llama.cpp, but its platform-facing behaviour (slot integration, prefix cache hooks, embeddings/vision plumbing) is the spec for the seam client. Rewrite as the engine client; do not port the FFI calls. |
| native_adapter.go | DROP | Adapter wrapping the dead native engine [N]; the seam client replaces it. Its SaveKVCache/LoadKVCache are silent no-ops returning nil [N-m5]. |
| llama/ (whole FFI binding package, 32 files) | DROP | PureGo bindings to llama.cpp, version-locked by one removed symbol: the loader preps llama_params_fit and aborts the whole library load when it is gone, which is why Torch is pinned to b8398 [P]. The rebuild's entire point is our own engine; mtmd/vision and grammar bindings' behaviour specs move to the Rust engine. This is the largest single DROP and the most valuable one: it deletes the dependency that froze the project. |
| handlers_ollama.go + ollama_pull.go | REUSE | Ollama-compatible API and pulls are wired and sound [P]; a user-facing surface worth keeping verbatim. |
| handlers_models.go, handlers_ops.go | REUSE-FIX | Wired handlers [P], but handlers_ops state save/load takes a raw path from the request body: confine to a configured state directory with validatePath-style checks and require auth [P-C2]. Also the platform half of the Glimmer context-freeze feature (kv_cache.go state files), worth keeping once confined. |
| handlers_embeddings.go | REUSE-FIX | Embeddings are real and wired [P group 6]; fix the design caveat: it holds the engine mutex for the whole request and clears shared KV state per input [P-m5]. |
| handlers_tools.go, handlers_raw.go | REUSE | Wired, no audit findings against them. |
| handlers_speech.go | DROP | A route whose engine method always errors "in development" [P-m2]. Speech re-enters only when the engine has a real audio path. |
| handlers_lora.go | REUSE-FIX | The handler shape is fine; the manager underneath fakes success (records a map entry, never calls the adapter bindings [P]). Rebuild Load/apply against the Rust engine's adapter support and make failure loud. |
| server_endpoints.go | REUSE-FIX | Stats/models endpoints; fix the two vet lock-copy findings by not copying mutex-bearing structs into JSON encoding [P-m3, G]. Also /v1/tokenize fabricates its count (len/4) [P-m1]: wire it to the real tokenizer or delete the endpoint. |
| chat_template.go | REUSE | 507 lines of template handling in the wired path, no findings. |
| grammar.go, json_mode.go | REUSE-FIX | The GBNF converter and grammar helpers are finished code that generation never read [P group 3]. Fix is wiring: build the sampler per request from CompletionParams and attach grammar when set, engine-side via the seam. |
| vision_engine.go | REUSE-FIX | Vision is genuinely wired via mtmd today [P group 6]; repoint its backend calls from mtmd FFI to the Rust engine's vision path when that lands, keep the handler/engine shape. |

### Serving machinery

| Source | Verdict | Reason |
| --- | --- | --- |
| scheduler.go | REUSE-FIX | Core wired and sound [P]; one real bug: batch mode never drains the high-priority queue, a latent hang for elevated-priority submitters [P]. Drain highQueue in the batch loop. |
| slot_manager.go (BatchEngine) | REUSE | Slot-based batching is wired into the scheduler and works [P]; per PLAN it stays platform-side. |
| continuous_batch.go (pipeline batcher) | REUSE | Honest pipelining (pre-tokenize next request during generation) with a header that states its own limits; complements the slot manager. |
| priority_queue.go | REUSE-FIX | FeatureHub member orphaned by the unregistered hub [P-M1]; the queue itself is what the scheduler fix above consumes. Wire it. |
| token_budget.go | REUSE-FIX | Same orphan story [P-M1]; budgets are a PLAN platform feature, wire through the hub fix. |
| ratelimit.go | REUSE | Wired (opt-in, chat route) [P]; has tests [G]. Make it default-on when auth lands. |
| response_cache.go | REUSE | Default-on, streaming-aware, genuinely wired [P]. One of the cleanest salvages in the tree. |
| prefix_cache.go | REUSE | Wired into the engine with Lookup/Restore/Save and a sane 32-token minimum [P]. |
| kv_cache.go (platform, state save/load) | REUSE-FIX | The context-freeze implementation behind handlers_ops; keep with the path confinement fix [P-C2] and repoint from llama state bindings to engine state via the seam. |
| kv_metrics.go, metrics.go, metrics_export.go | REUSE | Observability is wired; kv_metrics has tests [G]. |
| request_trace.go | REUSE-FIX | The tracer exists; only its routes were dead [P-M1]. Register via the hub fix. |
| diagnostics.go, structured_log.go, logger.go | REUSE | Platform diagnostics/logging; aligns with Core debug (below). |
| graceful_shutdown.go, retry.go | REUSE | No findings; platform plumbing the PLAN grid names. |
| mem_pool.go, prefault*.go | REUSE | Page prefaulting and pooling are PLAN model-ops features; no findings. |
| batch_tuner.go, auto_config.go | REUSE-FIX | Auto-config is wired and tested [G]; one fix: speculative auto-detection is commented out [N-M6], either enable it against the Rust engine or remove the claim. |
| gpu_detect.go, cpu_topology.go, sysinfo.go, sysram.go | REUSE | Hardware detection the platform needs to launch/configure the engine; gpu_detect and cpu_topology have tests [G]. |
| multi_gpu.go | DROP | ApplyMultiGPU has zero callers [P-m6] and its target was llama.cpp model params; GPU placement is the Rust engine's launch config now. Retire the platform copy. |

### FeatureHub tier: keep the members, kill the dead switchboard

| Source | Verdict | Reason |
| --- | --- | --- |
| feature_hub.go, server_features.go | GO-REWRITE | The hub was never constructed and its routes never registered, so guardrails, A/B, semantic cache, versioning, and tracing did not exist in the shipped server [P-M1]. The members below are mostly real; rewrite the hub as plain server construction (no feature-flag cathedral), register the routes, and cover them with the existing features_test.go. |
| ab_routing.go | REUSE-FIX | Real router, orphaned only by the hub [P-M1]; wire it. |
| model_version.go (versioning/rollback) | REUSE-FIX | Same: real code, dead routes [P group 4]. Wire it. |
| native/semantic_cache.go | GO-REWRITE | The hub's semantic cache implementation lives in the dying native package (feature_hub.go:38); port it to the platform as Go. |
| native/guardrails.go | GO-REWRITE | Same situation (feature_hub.go:29): the guardrails implementation is native-package code referenced by platform types. Port to Go platform. |
| native/payload_aware_router.go | GO-REWRITE | Payload-aware routing is a PLAN platform feature; this orphan is its only implementation. Port to Go, wire into the router. |
| native/watermark.go | GO-REWRITE | Output watermarking is platform-side per PLAN S2; the hub header even claimed it without a field [P-m6]. Port and actually attach it. |
| native/tensor_memo.go | DROP | Memoization cache for native-engine tensors; meaningless without that engine, and its only platform trace was a stats route in the dead hub [P-M1]. Retire. |
| native/request_router.go, context_manager.go, token_pipeline.go, token_stream.go, embedding_pool.go, health_monitor.go, sys_metrics.go, warmup_engine.go, prompt_cache.go, continuous_batch.go, continuous_batching.go, dynamic_batch.go, batch_prefill.go, chunked_prefill.go | DROP | Native-package duplicates of platform machinery that already exists and is wired (scheduler, slot manager, prefix cache, metrics, warmup). Duplication across the seam is how the original lost track of itself; one implementation per feature, on its PLAN side of the seam. |

### Cluster and tenancy

| Source | Verdict | Reason |
| --- | --- | --- |
| swarm.go, cluster_coordinator.go, cluster_discovery.go | REUSE-FIX | Clustering is wired [P] with three mandatory fixes: authenticated join (today anyone can register a peer and receive forwarded prompts [P]), the nil-engine panic at cluster_coordinator.go:238 that kills the package test suite [P, G], and discovery off by default. |
| native/tenant_isolator.go | GO-REWRITE | Tenant isolation is claimed but has zero references in platform code [P]; this orphan is the only design draft. Rewrite platform-side, threading tenant identity from auth through scheduler and caches. |

### Model operations

| Source | Verdict | Reason |
| --- | --- | --- |
| hf_pull.go | REUSE | Wired, filename sanitizing verified sound [P]; has tests [G]. |
| oci_pull.go | REUSE | Wired, safe name handling verified [P]; has tests [G]. |
| model_integrity.go | REUSE | SHA256 sidecar check wired at load [P]; small and exactly right. |
| model_registry.go | REUSE-FIX | Wired registry [P]; mandatory fix: it routes any file under 1 GiB to the demo native engine by size alone [N-M1]. Route everything to the Rust engine via the seam; keep name resolution and metadata. |
| model_warmup.go | REUSE | PLAN platform feature; no findings (its caller chain was not traced in the audit, verify on reattach). |
| validate.go | REUSE | Path validation verified sound and applied at engine construction [P]; has tests [G]. The pattern handlers_ops must adopt [P-C2]. |
| converter.go | REUSE | GGUF conversion via CLI is wired [P]; fixed-argument subprocess from CLI-only paths, no injection surface [P]. |
| tokenizer/ (pkg/torch/tokenizer) | REUSE-FIX | The platform tokenizer has one real defect: Decode returns raw GGUF token strings with no byte-level BPE reversal, which is the Ġ-in-JSON bug [P]. Fix the detokenization mapping and add the package's first tests [G: zero test files today]. |

### Training surface (platform side)

| Source | Verdict | Reason |
| --- | --- | --- |
| training_api.go | DROP | Dead by construction: getNativeEngine() unconditionally returns (nil, false), so every HTTP training route fails in every configuration [P-C3]. Training is CLI + itak-train; if an HTTP surface is ever wanted it is written fresh against the seam. |
| cmd/torch/train_orchestrator.go | REUSE-FIX | The one real training entry point (CLI dispatch works [P group 1]); repoint from native trainers to itak-train over the seam when the crate exists. |

### CLI and commands

| Source | Verdict | Reason |
| --- | --- | --- |
| cmd/torch/main.go (+ build_android.go) | GO-REWRITE | The dispatch, pull commands, and CLI surface salvage; engine construction is welded to the FFI and native engines and to the 10-minute context that made native generation time out [N-M2]. Rewrite construction against the seam client, keep the command tree. |
| cmd/itak-quant/main.go | DROP | Prints SUCCESS and "Output saved" while creating no file; its own comment admits the placeholder [P]. The honest replacement is itak-dissect's GGUF writer (above). A CLI that lies about its deliverable does not get fixed, it gets replaced. |
| scripts/build_backends.go | DROP | Builds the llama.cpp backend libraries the rebuild no longer links. |
| _tools/*.go, test_qwen.go | DROP | Scratch harnesses at the repo root; the ternary bench concept is salvaged under itak-kernels. |
| Test suites (43 files, 367 test funcs [G]) | REUSE-FIX | Keep the platform and tokenizer-adjacent suites as the reattach gate. Mandatory fixes: the cluster panic masks ~120 of 157 pkg/torch tests [G], and the llama FFI suite (168 funcs, ~99% skipping without ITAK_TORCH_LIB [G]) retires with the bindings it tests. Add the lint config and CI the tree never had [G]. |

---

## 3. Destination: Core, republished as a versioned module

The rescued original Core (~/workspace/itak-core-rescue/Core) is what the
rebuild's PLAN S4 re-creates; with the original back in hand, most of it
carries instead of being rewritten. It must be published as a versioned
module: the relative ../Core replace is what killed the original build
[P build note, DECISIONS 2026-10-27]. Torch imports Core from exactly six
files (core_adapter.go, core_health.go, handlers_models.go,
handlers_ops.go, server.go, + test), and only for pkg/types today, so the
reattach surface is small and known.

| Core package | Verdict | Reason |
| --- | --- | --- |
| pkg/types | REUSE-FIX | The measured call-site surface (PLAN S4); add the version fields the PLAN contracts require. Mirrored by Rust itak-types. |
| pkg/auth (apikey, hmac, mtls, rbac) | REUSE | The single most valuable Core salvage: it is the missing ingredient for the platform's worst audit finding, no auth on any endpoint [P-C1], and for authenticated cluster join [P]. It existed all along and Torch never wired it. |
| pkg/contract | REUSE-FIX | Inference/media/voice/storage contract types; align field-for-field with itak-types so the seam carries one contract, not two dialects. |
| pkg/health, pkg/event, pkg/debug, pkg/registry | REUSE | PLAN S4 names exactly these as the Core surface; health/heartbeat, the event bus, the debug handler, and the registry are the platform's shared plumbing. |
| pkg/tunnel | REUSE | Relay/direct/broker tunnel for reaching engines behind NATs; not on the Glimmer critical path but sound, self-contained, and the cluster story's transport option. |
| core_adapter.go, core_health.go (in Torch) | REUSE-FIX | The Torch-side adapters to Core; they reattach against the published module path, which is also the resurrection-gate proof PLAN S4 specifies (original tree compiles against rebuilt Core). |

---

## 4. What this adds up to

- The Go platform survives substantially intact: server, scheduler, slots,
  both caches, pulls, integrity, Ollama surface, cluster (after auth), and
  the FeatureHub members (after the hub is rewritten as plain wiring).
  Its defects were missing wiring and missing auth, not missing code [P].
- The native package (129 files, ~31.6k lines) contributes almost no code
  and a great deal of design: the KV family, dequant/matmul approach,
  MoE streaming, the training stack's shape, and the dissect toolset all
  re-enter as Rust crates. Its engine, GPU bindings, and duplicate
  serving machinery are dropped.
- The llama.cpp FFI package is dropped whole. That is the rebuild's
  premise: Glimmer on our own stack means the token loop is ours.
- Core is republished, not rewritten, and its auth package closes the
  audit's worst finding.

First three moves this inventory implies: (1) republish Core and pass the
PLAN S4 resurrection gate; (2) stand up itak-loader + itak-kernels +
itak-kv against the Glimmer GGUF with the teardown census as fixtures,
since Glimmer 100% on our stack is the end goal and its file is the test
case; (3) reattach the platform with the auth, state-path, priority-drain,
and registry-routing fixes, in that order.
