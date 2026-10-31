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

## As of 2026-10-31

## Proven state
- The original engine is fully audited: feature inventory, the three stacked build blockers, and the measured Core usage surface are on record (proof: PLAN.md S2; TODO Phase 0 items T-0.1 and T-0.6, checked with proofs). The salvage inventory is filed (proof: SALVAGE.md). The blueprint set is committed: ARCHITECTURE.md, PLAN.md, TODO.md.
- The hardware record is measured, not quoted: Glimmer 30B per-quant speed and VRAM on the RX 7900 XTX, 2-user concurrency, accuracy runs including the reasoning_strength default trap and its fix (proof: docs/research/2026-10-26-glimmer-7900xtx-benchmarks.md; raw logs in bench/results-2026-10-26/).
- ROCm is fixed (2026-10-28): single Ubuntu native ROCm 7.1 stack, HIP proven end to end from hello-world to a full 30B bench. llama.cpp works on both Vulkan and HIP on the server; Vulkan is the serving pick on decode speed, HIP wins prompt processing (proof: benchmark report, Phase A ROCm section; TODO T-P.2).
- The architecture is decided: Go platform plus Rust engine core, two processes, NDJSON over a Unix socket (proof: DECISIONS.md, 2026-10-27; ARCHITECTURE.md).
- The Rust tokenizer is proven token-for-token against the reference on Glimmer's real vocabulary: 16/16 bench prompts identical and 16/16 decode round-trips exact against the production llama-server of the time (proof: crates/tokenizer/CONFORMANCE.md; TODO T-2.3 checked). The run caught and fixed a real defect: the pre-tokenizer implemented the llama-3 contraction split, but llama4 routes to the GPT-4o pattern where the contraction is a word suffix.
- The loader covers the real Glimmer file completely: config parsed from the file's own metadata, coverage 731 of 731 tensors with zero missing and zero unexpected, and a full Weights::load serving all 731 tensors (16,743,567,360 bytes) through itak-loader over the gguf crate (proof: the glimmer_inventory example run against the real file on the server, 2026-10-29). The naming question is settled by the split itself: gguf is the file layer, itak-loader is the bridge that implements the architecture traits over it.
- The forward error from integration run B (2026-10-29, crates/itak-loader/INTEGRATION-RUN-B.md) was localized by the per-layer diff work that followed. The CPU reference forward and the GPU trait path both reproduce the golden continuation exactly (proof: bench/gpu-proof-2026-10-30.md, 8/8 tokens). One residue stays on record unexplained: the first-token probability against llama.cpp (0.4035 here, 0.3962 there); it is carried as TODO T-9.3, the probability-level golden.
- The GPU kernel set K1 through K14 benches at the card's stream rate (about 813 to 832 GB/s on the Q4_K, Q6_K, and Q5_K GEMVs), and the device-resident session built on them reproduces the golden 8 of 8 tokens with one submission per token, decoding at 39.7 tok/s against llama.cpp's 44.31 on the same harness (proof: bench/gpu-resident-2026-10-30.md). The remaining gap is pipeline drain across dependent kernel boundaries, named in that note.
- Prefill has a true packed MMQ path (Q8_1 quantizer plus DP4A kernels): marginal prefill throughput 161 tok/s at 1,024 tokens against llama.cpp's 998 on the same card. The gap and the remaining profile work are recorded, not rounded away.
- The DFlash draft module loads the real draft file behind the coverage gate (58 of 58 tensors) and its block forward runs against the real target (proof: bench/dflash-e3-2026-10-30.md). The speculative verify loop built on it produces a stream token-identical to plain greedy over 16 tokens at depth 6 on the real files, CPU backend (proof: bench/spec-loop-2026-10-30.md).
- The DFlash depth sweep on the reference stack (llama.cpp, Section 9 harness, temperature 0) is on record in full (proof: bench/gpu-proof-2026-10-30.md): plain 41.64 tok/s; depth 1 acceptance 0.858; depth 6 acceptance 0.541 at 107.39 tok/s, the operating point by the SPEC 9.4 rule; depth 12 acceptance 0.300; depths 8 and up collapse to or below the plain rate. The sweep's anchors are harness-relative against the 2026-10-28 record (plain 44.31, depth 4 at 88.6), and DECISIONS.md (2026-10-30) records which harness owns which claim.
- Glimmer weights are on the server (2026-10-28): /mnt/models on the 2TB Kioxia NVMe, byte-verified against the HF listings; the Glimmer file set is /mnt/models/Muse-Glimmer-30B/. Production Glimmer serving is DOWN and stays down on the owner's explicit order until the engine work is finished; no llama-server serves Glimmer on the server today. (Two Qwen3-Embedding llama-server tenants run separately on the server and are not this project's serving path.)
- The serving pair works end to end on our own stack (2026-10-30): the Rust itak-engine binary loads the real Glimmer file (coverage 731/731) and serves the NDJSON protocol on a Unix socket; the Go platform serves authenticated chat completions over it (proof: bench/serving-smoke-2026-10-30.md: authenticated completion returned " Paris" with finish_reason stop, unauthenticated refused 401, no-engine platform refused 503). The served path is the CPU reference forward; it is a correctness path, not a speed tier.
- The production gaps from the smoke are closed in code and proven live on the real file (T-9.11 through T-9.17, proof: bench/t917-proof-2026-10-30.md): engine supervision under a restart budget with fault-injection proof, terminal stream error events (never a truncated 200), sampling parity (seeded streams identical over the socket, a penalty visibly moving a pick), multi-slot serving (two concurrent chats, independent outputs), the embed operation (cosines ranking the related pair first, 0.899662 against 0.370221), and GGUF chat-template rendering with engine prompt ids matching llama.cpp 63 for 63. The proof run caught and fixed three real template defects: the renderer could not render the file's own ATEM template (9379364), chat prompts carried a duplicated BOS id (819b1ae), and the template environment lacked strftime_now so the Current date line never rendered (af21f92).
- A double critic panel (two sealed panels of five, Survey protocol, cold start) graded this tree DO NOT SHIP, ten of ten seats (record: the run's MERGE.md adjudication). The numeric core verified clean from every seat; every blocker sat at the serving boundary or in the written record.
- Fix wave Waves 1 and 2 have landed. Wave 1 (merge 19f2cba): engine socket hardening (owner-only socket, hello enforcement, peer uid check, frame bound, JSON depth cap, the max_tokens wrap closed), chat-template render budgets, platform bounds (generation caps and deadline, scheduler queue caps, response cache cap with janitor, supervisor environment allowlist, the supervisor stderr wedge fix, engine client frame bound, stream terminal honesty, auth throttling and key floor, static /healthz), and metadata magnitude bounds in the architecture config and KV sizing. Wave 2 (commits 258911d through 7eacc30): the formatting debt cleared (cargo fmt clean), deny.toml added with cargo deny check passing, toolchains pinned (rust 1.99.1; platform go1.26.10, under which govulncheck reports no vulnerabilities, closing the panel's GO-2026-6809), manifest descriptions corrected to the real six socket ops and K1 through K14, TODO proof texts corrected (T-3.7, T-9.8, T-P.1 closed, the SPEC numbering collision renumbered to T-9.18), and the chattemplate gosec G101 cleared.

## In flight
- The panel fix wave: Waves 1 and 2 are landed and gated. Wave 3 (operability: the engine diag module, the gguf diag emission fix, the speculator guard test, the sampler named refusals, the worker catch_unwind) and the environment-blocked reruns from the panel (the mutation battery, the live socket probes, the wgpu-compute suite on a real adapter) are still ahead, then a cold panel re-run under the Survey protocol plus the inference-engine profile. Until that re-run lands, the honest verdict on this tree is the panel's: DO NOT SHIP.

## Blocked
- The GPU resident port of the speculative loop plus the on-stack DFlash depth and acceptance sweep (TODO T-9.6, T-9.7) waits on the fix wave finishing, so it builds on a tree that has passed its gates.
- The wide-tile double-buffered MMQ and the non-projection prefill profile (161 tok/s toward the 998 reference) wait behind the GPU speculative work.
- Production Glimmer serving waits on the owner's word; it does not restart on any schedule of this project's.
- DiffusionGemma does not load on the current llama.cpp: its architecture string is not yet known upstream (checked at master a7af149). Reference-only; nothing in this repo waits on it except the comparison bench.

## Next moves
1. Land fix wave Wave 3 (operability items from the panel merge), then re-run the panel cold.
2. GPU resident port of the speculative loop and the on-stack DFlash sweep (TODO T-9.6, T-9.7).
3. Wide-tile MMQ and the prefill profile; the training stack (Phase 7) and the .itf format (Phase 8) stay behind those.
