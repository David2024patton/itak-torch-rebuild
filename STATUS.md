<!--
INSTRUCTIONS FOR THIS FILE (user and AI: read before editing. Keep this header.)

WHAT THIS FILE IS: the living right-now snapshot of this project. Anyone
opening the repo cold reads this first and is current in two minutes.
Updated at the end of every work session on the project; a stale STATUS
is worse than none, so when you finish work here, update it in the same
pass.

EXACT STRUCTURE:
- "## As of <YYYY-MM-DD>": the date this snapshot was written.
- "## Proven state": what is done, with its proof named.
- "## In flight": work currently underway and where it stands.
- "## Blocked": each blocker and exactly who or what it waits on.
- "## Next moves": the next three moves, in order, referencing TODO ids.

RULES:
- Claims here carry the same proof standard as TODO checkboxes.
- Keep it short: a snapshot, not a history (history is CHANGELOG.md).
- No secrets, passwords, or tokens in this file, ever.
-->
# STATUS: iTaK Torch Rebuild

## As of 2026-10-29

## Proven state
- The 20-lane wave is merged (2026-10-29): branch integration-2026-10-29 at 0f322f3 carries all 19 lane branches as one tree, pushed to GitHub with remote hashes verified equal to local. Gates on the merged tree: cargo test --workspace 374 passed, 0 failed; core/ Go 48 passed; platform/ Go 40 passed; cargo clippy and cargo fmt clean; tools/gates.sh ALL GATES GREEN with no skipped tests (proof: the merge run on the integration worktree and its gate outputs). The wave fixed three forward defects isolated by the divergence hunt: RoPE pairing corrected to the NEOX split-half convention the file's HF layout requires (it was interleaved, in all 52 layers), the post norms moved onto each sub-layer output before the residual add (they sat after the adds), and the missing scaleless embedding norm added (proof: crates/torch-math/ROPE-AUDIT.md; crates/muse-glimmer/NORM-AUDIT-2026-10-29.md). The record pass closing the wave checked the TODO boxes T-2.1, T-9.1, and T-P.3 against their proofs; T-2.1 carries its fixture caveat in its proof line.
- The original engine is fully audited: feature inventory, the three stacked build blockers, and the measured Core usage surface are on record (proof: PLAN.md S2; TODO Phase 0 items T-0.1 and T-0.6, checked with proofs).
- The hardware record is measured, not quoted: Glimmer 30B per-quant speed and VRAM on the RX 7900 XTX, 2-user concurrency, accuracy runs including the reasoning_strength default trap and its fix (proof: docs/research/2026-10-26-glimmer-7900xtx-benchmarks.md; raw logs in bench/results-2026-10-26/).
- ROCm is fixed (2026-10-28): single Ubuntu ROCm 7.1 stack, HIP proven end to end from hello-world to a full 30B bench. Vulkan remains the serving backend on generation speed; HIP wins prompt processing. The failure had been diagnosed to the userland layer with the card and kernel driver proven healthy by the Vulkan runs (proof: benchmark report, Phase A ROCm section; TODO T-P.2).
- The research base is filed: inference platform research and the Mamba-1, 2, 3 ingest with David's Mamba-3 variant graded against the real paper (proof: docs/research/).
- The architecture is decided: Go platform plus Rust engine core, two processes, socket plus shared-memory seam (proof: DECISIONS.md, 2026-10-27; ARCHITECTURE.md, committed ef07956).
- The Rust tokenizer is proven token-for-token against the reference on Glimmer's real vocabulary: 16/16 bench prompts identical and 16/16 decode round-trips exact against the production llama-server (proof: crates/tokenizer/CONFORMANCE.md; TODO T-2.3 checked). The run caught and fixed a real defect: the pre-tokenizer implemented the llama-3 contraction split, but llama4 routes to the GPT-4o pattern where the contraction is a word suffix.
- The loader seam is wired: the itak-loader crate implements muse-glimmer's Metadata and TensorSource traits over the gguf crate, with a tensor inventory coverage check against expected_tensor_names (proof: 9 integration tests over synthetic GGUF bytes; the glimmer_inventory example run against the real Glimmer file on the server, 2026-10-29: config parsed from the file's own metadata matching every measured value and agreeing with the gguf crate's own ModelConfig, coverage 731 of 731 with zero missing and zero unexpected, output_norm.weight confirmed present as the final norm (F32, [6656]), and a full Weights::load serving all 731 tensors (16,743,567,360 bytes) through the adapter; the running llama-server was not touched).
- First end-to-end forward comparison on the new stack (integration run A, 2026-10-29): the arch-llama forward scored 0 of 15 token-identical against the llama.cpp golden fixtures on both models, with two different causes. On Qwen2.5-1.5B the tokenization is proven exact (prompt ids identical on all 15 prompts) and the divergence is a forward math discrepancy in the attention and residual pathway, with logit swings from 0.3 to over 5 on a prefix probe. On TinyLlama the zero is a tokenizer family gap (the file's SentencePiece BPE is not implemented in the tokenizer crate), so that half says nothing about the forward (proof: golden/RESULTS-integration-a.md; candidates in golden/candidates/).
- First end-to-end generation on the new stack (integration run B, 2026-10-29): Glimmer loaded through itak-loader (coverage 731/731), tokenized by the Rust tokenizer, 8 greedy tokens generated on cpu-compute from the golden factual prompt. The pipeline completes with no refusals, but the numerics diverge from llama.cpp at the first generated token on identical prompt ids: our top-1 is 290 (" the", p 4.4e-5) against llama.cpp's 13796 (" Paris", p 0.396), our top-5 holds 1.6e-4 of the mass against their 0.62, and at decode time our distribution repeats to six decimals across steps despite different fed tokens (proof: crates/itak-loader/INTEGRATION-RUN-B.md; raw run record on the server in ~/itak-run-b/). Localizing the forward error (per-layer activation diff against the reference graph, the norm-convention question, decode-path KV handling at real scale) is the next engine task; nothing was tuned to force a match.
- The Rust workspace is twelve crates, wired together by path dependencies: gguf (strict GGUF v3 parser, fail-loud named errors, full mainline type table with readers for every teardown corpus type), torch-math (reference f32 math core: RMSNorm, RoPE in both pairings, softmax, sampling primitives; every constant a parameter), tokenizer (real byte-level BPE over the model's own vocabulary), attention (KV cache with global and sliding-window ring layers, grouped-query attention), arch-llama (llama and qwen2 family forward), muse-glimmer (the Glimmer architecture module, TODO T-P.3, checked), cpu-compute (the CPU compute backend behind muse-glimmer's Compute trait), itak-loader (the arch-trait adapter over gguf: the Metadata and TensorSource implementations plus the arch registry and dispatch), itak-kernels (Q4_K dequant and matvec kernels), itak-types (the IPC contract types mirroring the Go contracts), itak-backend (backend trait, CPU backend, hardware detection and auto-config), and itak-engine (the binary crate that assembles loader, tokenizer, model, and compute into a runnable engine). Wiring proof: the crates' Cargo.toml path dependencies (itak-engine depends on gguf, itak-loader, muse-glimmer, tokenizer, and cpu-compute; itak-loader depends on gguf, muse-glimmer, and arch-llama) and integration run B below, which executed the wired path end to end.
- The Go platform is landed at scaffold-plus stage: diag registry, API-key auth, TTL response cache, priority scheduler whose lanes provably drain, engine client seam with named errors and a unix socket skeleton, HTTP server and the torchd daemon (proof: platform/ commits a45e5fe through 3dfff80; hygiene pass run: go test 34 passed, 0 failed; golangci-lint config in tree and clean per commit 3dfff80).
- The salvage inventory is filed (proof: SALVAGE.md, module by module against the original tree). The blueprint set is fully committed: ARCHITECTURE.md, PLAN.md, TODO.md (proof: ef07956, f042064). Cargo.lock is committed (proof: 1b67489).

## In flight
- Forward verification of the merged tree on the real Glimmer file has not run yet: the per-layer activation diff against llama.cpp (plan: crates/itak-loader/FORWARD-DIFF-PLAN.md) is server-side work. Until it runs, whether the three wave fixes close the run B divergence is unproven; nothing has been tuned to force a match.
- The main checkout carries a parallel session's uncommitted itak-loader work (src/adapter.rs, examples/glimmer_inventory.rs), untouched by the integration branch; it lands or not on that session's own record.
- Glimmer weights are on the server (2026-10-28): /mnt/models on the 2TB Kioxia NVMe, byte-verified against the HF listings, alongside Qwen3.6-27B, Gemma-4-31B, DiffusionGemma-26B, Nemotron-Cascade-2-30B, North-Mini-Code-1.0, and Devstral-Small-2-24B. A llama-server (Vulkan build, Glimmer Q4_K_M + mmproj, 128K context, 4 slots) runs on the server's localhost:8080 and answers chat completions with the reasoning_strength=low injection working as verified in October.

## Blocked
- The original Torch tree still does not build; that is the condition this project exists to fix, gated on T-1.1 (itak-core written to the measured call sites).
- DiffusionGemma does not load on the current llama.cpp: its architecture string is not yet known upstream (checked at master a7af149). Reference-only; nothing in this repo waits on it except the comparison bench.

## Next moves
1. Run the merged engine on the server: golden-output comparison and the per-layer activation diff against llama.cpp on the real Glimmer file (TODO T-9.4; plan: crates/itak-loader/FORWARD-DIFF-PLAN.md).
2. Regenerate the four pending golden references on the server (golden/PENDING-REFERENCES.md) and run itak-engine end to end on the real Glimmer file.
3. Close the resurrection gate: compile the original Torch tree against the merged core/ Go module (TODO T-1.1; the core packages landed in the wave, the whole-tree compile is the remaining proof).
