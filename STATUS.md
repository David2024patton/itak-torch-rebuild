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
- The original engine is fully audited: feature inventory, the three stacked build blockers, and the measured Core usage surface are on record (proof: PLAN.md S2; TODO Phase 0 items T-0.1 and T-0.6, checked with proofs).
- The hardware record is measured, not quoted: Glimmer 30B per-quant speed and VRAM on the RX 7900 XTX, 2-user concurrency, accuracy runs including the reasoning_strength default trap and its fix (proof: docs/research/2026-10-26-glimmer-7900xtx-benchmarks.md; raw logs in bench/results-2026-10-26/).
- ROCm is fixed (2026-10-28): single Ubuntu ROCm 7.1 stack, HIP proven end to end from hello-world to a full 30B bench. Vulkan remains the serving backend on generation speed; HIP wins prompt processing. The failure had been diagnosed to the userland layer with the card and kernel driver proven healthy by the Vulkan runs (proof: benchmark report, Phase A ROCm section; TODO T-P.2).
- The research base is filed: inference platform research and the Mamba-1, 2, 3 ingest with David's Mamba-3 variant graded against the real paper (proof: docs/research/).
- The architecture is decided: Go platform plus Rust engine core, two processes, socket plus shared-memory seam (proof: DECISIONS.md, 2026-10-27; ARCHITECTURE.md, committed ef07956).
- The Rust tokenizer is proven token-for-token against the reference on Glimmer's real vocabulary: 16/16 bench prompts identical and 16/16 decode round-trips exact against the production llama-server (proof: crates/tokenizer/CONFORMANCE.md; TODO T-2.3 checked). The run caught and fixed a real defect: the pre-tokenizer implemented the llama-3 contraction split, but llama4 routes to the GPT-4o pattern where the contraction is a word suffix.
- The loader seam is wired: the itak-loader crate implements muse-glimmer's Metadata and TensorSource traits over the gguf crate, with a tensor inventory coverage check against expected_tensor_names (proof: 9 integration tests over synthetic GGUF bytes; the glimmer_inventory example run against the real Glimmer file on the server, 2026-10-29: config parsed from the file's own metadata matching every measured value and agreeing with the gguf crate's own ModelConfig, coverage 731 of 731 with zero missing and zero unexpected, output_norm.weight confirmed present as the final norm (F32, [6656]), and a full Weights::load serving all 731 tensors (16,743,567,360 bytes) through the adapter; the running llama-server was not touched).
- First end-to-end generation on the new stack (integration run B, 2026-10-29): Glimmer loaded through itak-loader (coverage 731/731), tokenized by the Rust tokenizer, 8 greedy tokens generated on cpu-compute from the golden factual prompt. The pipeline completes with no refusals, but the numerics diverge from llama.cpp at the first generated token on identical prompt ids: our top-1 is 290 (" the", p 4.4e-5) against llama.cpp's 13796 (" Paris", p 0.396), our top-5 holds 1.6e-4 of the mass against their 0.62, and at decode time our distribution repeats to six decimals across steps despite different fed tokens (proof: crates/itak-loader/INTEGRATION-RUN-B.md; raw run record on the server in ~/itak-run-b/). Localizing the forward error (per-layer activation diff against the reference graph, the norm-convention question, decode-path KV handling at real scale) is the next engine task; nothing was tuned to force a match.
- Wave 1 of the Rust workspace is landed: five crates, all standalone libraries today. gguf (strict GGUF v3 loader, fail-loud named errors), torch-math (reference f32 math core, every constant a parameter), tokenizer (real byte-level BPE over the model's own vocabulary), attention (KV cache with global and sliding-window ring layers, grouped-query attention), muse-glimmer (architecture skeleton, TODO T-P.3). Gates observed in the 2026-10-29 hygiene pass, not copied from builder reports: cargo test for the workspace 120 passed, 0 failed (gguf 27 including 1 doc test, torch-math 39, tokenizer 16 including 1 doc test, attention 15, muse-glimmer 23); cargo clippy clean under the workspace deny-warnings lints.
- The Go platform is landed at scaffold-plus stage: diag registry, API-key auth, TTL response cache, priority scheduler whose lanes provably drain, engine client seam with named errors and a unix socket skeleton, HTTP server and the torchd daemon (proof: platform/ commits a45e5fe through 3dfff80; hygiene pass run: go test 34 passed, 0 failed; golangci-lint config in tree and clean per commit 3dfff80).
- The salvage inventory is filed (proof: SALVAGE.md, module by module against the original tree). The blueprint set is fully committed: ARCHITECTURE.md, PLAN.md, TODO.md (proof: ef07956, f042064). Cargo.lock is committed (proof: 1b67489).
- Glimmer is served end to end on our own stack (2026-10-30): the Rust itak-engine binary (commit e40b8ba) loads the real Glimmer file (coverage 731/731) and serves the NDJSON protocol on a Unix socket; the Go platform's Generate speaks it (commit 110ed17). Live smoke on ubuntu-server: authenticated chat completion returned " Paris" with finish_reason stop, unauthenticated refused 401, no-engine platform refused 503 (proof: bench/serving-smoke-2026-10-30.md). The serving path is the CPU reference forward, about 2,891 s for a 12-token prompt; GPU speed is T-9.5/T-9.10, supervision is T-9.11.

## In flight
- The crates are wired for serving: itak-engine assembles gguf, itak-loader, muse-glimmer, cpu-compute, and tokenizer behind the socket seam, and the Go platform drives it (commits e40b8ba, 110ed17; proof: bench/serving-smoke-2026-10-30.md). What is missing for production serving is speed (GPU backend), engine supervision, and sampling/chat-template parity (TODO T-9.5, T-9.10 to T-9.16).
- Open reconciliation, loader naming: PLAN.md and TODO.md name the loader module itak-loader (TODO T-2.1); the built crate is named gguf. Nothing has been renamed; one name must win before the engine binary wires the crates together.
- Record lag: TODO checkboxes trail the commits that cite them. T-2.3 (tokenizer) landed in 295a3ec and T-P.3 (muse-glimmer) in d4b3914, and both boxes still read open in TODO.md.
- Glimmer weights are on the server (2026-10-28): /mnt/models on the 2TB Kioxia NVMe, byte-verified against the HF listings, alongside Qwen3.6-27B, Gemma-4-31B, DiffusionGemma-26B, Nemotron-Cascade-2-30B, North-Mini-Code-1.0, and Devstral-Small-2-24B. A llama-server (Vulkan build, Glimmer Q4_K_M + mmproj, 128K context, 4 slots) runs on the server's localhost:8080 and answers chat completions with the reasoning_strength=low injection working as verified in October.

## Blocked
- The original Torch tree still does not build; that is the condition this project exists to fix, gated on T-1.1 (itak-core written to the measured call sites).
- DiffusionGemma does not load on the current llama.cpp: its architecture string is not yet known upstream (checked at master a7af149). Reference-only; nothing in this repo waits on it except the comparison bench.

## Next moves
1. Settle the loader name (gguf vs itak-loader, TODO T-2.1) and sweep the TODO checkboxes against the commit record.
2. Write itak-core to the measured call sites and prove the original Torch compiles against it (TODO T-1.1, T-1.2, T-1.3).
3. Define the IPC contract and prove the Hello round trip (TODO T-1.6); the engine binary that assembles the five crates hangs on that seam.
