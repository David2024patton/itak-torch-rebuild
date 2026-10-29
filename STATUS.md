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

## As of 2026-10-27

## Proven state
- The original engine is fully audited: feature inventory, the three stacked build blockers, and the measured Core usage surface are on record (proof: PLAN.md S2; TODO Phase 0 items T-0.1 and T-0.6, checked with proofs).
- The hardware record is measured, not quoted: Glimmer 30B per-quant speed and VRAM on the RX 7900 XTX, 2-user concurrency, accuracy runs including the reasoning_strength default trap and its fix (proof: docs/research/2026-10-26-glimmer-7900xtx-benchmarks.md; raw logs in bench/results-2026-10-26/).
- The ROCm failure is diagnosed to the userland layer with four independent checks; the card and kernel driver are proven healthy by the flawless Vulkan runs (proof: benchmark report, Phase A ROCm section).
- The research base is filed: inference platform research and the Mamba-1, 2, 3 ingest with David's Mamba-3 variant graded against the real paper (proof: docs/research/).
- The architecture is decided: Go platform plus Rust engine core, two processes, socket plus shared-memory seam (proof: DECISIONS.md, 2026-10-27).

## In flight
- This repo: standard file set and research documents being published (this pass).
- Nothing in code yet. Phase 1 has not started.

## Blocked
- Glimmer weights are back on the server (2026-10-28): /mnt/models on the 2TB Kioxia NVMe, byte-verified against the HF listings, alongside Qwen3.6-27B, Gemma-4-31B, DiffusionGemma-26B, Nemotron-Cascade-2-30B, North-Mini-Code-1.0, and Devstral-Small-2-24B. All load and generate on the current llama.cpp except DiffusionGemma, whose architecture string is not yet known to upstream llama.cpp (checked at master a7af149). A llama-server (Vulkan build, Glimmer Q4_K_M + mmproj, 128K context, 4 slots) runs on the server's localhost:8080 and answers chat completions with the reasoning_strength=low injection working as verified in October.
- ROCm is FIXED (2026-10-28, was T-P.2): single Ubuntu ROCm 7.1 stack, HIP proven end to end from hello-world to a full 30B bench. Vulkan remains the serving backend on generation speed; HIP wins prompt processing. Detail in TODO T-P.2.
- The original Torch tree still does not build; that is the condition this project exists to fix, gated on T-1.1.

## Next moves
1. Rebuild Core to the measured call sites and prove the original Torch compiles against it (TODO T-1.1, T-1.2, T-1.3)
2. Scaffold the Rust workspace with itak-types mirroring the contracts (TODO T-1.4)
3. Wire this repo's commit gates (TODO T-1.5)
