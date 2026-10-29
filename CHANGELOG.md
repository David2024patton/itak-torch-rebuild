<!--
INSTRUCTIONS FOR THIS FILE (user and AI: read before editing. Keep this header.)

WHAT THIS FILE IS: the human-readable milestone trail of this project:
what shipped, dated, newest first. Git history is the raw record; this
is the curated one a person can read in a minute.

EXACT STRUCTURE:
- "## <YYYY-MM-DD>: <milestone in one line>" entries, newest first.
- Under each: what changed, and the proof or commit it landed in.

RULES:
- An entry is added when something SHIPS (lands on the main branch or
  goes live), not when it is started.
- One entry per milestone, not per commit.
- No secrets, passwords, or tokens in this file, ever.
-->
# CHANGELOG: iTaK Torch Rebuild

## 2026-10-27: Project repo created with the full standard file set and the evidence base
- The rebuild plan (PLAN.md), task list (TODO.md), idea record (IDEA.md), end goal (ENDGOAL.md), decisions, status, runbook, map, and references were written from the project's real state and published together with the research documents and bench scripts under docs/research/ and bench/.
- Proof: this repository, created public on 2026-10-27 on David's order.

## 2026-10-27: Mamba lineage ingested; David's Mamba-3 variant graded against the real paper
- Mamba-1, Mamba-2, and Mamba-3 (arXiv:2603.15569, March 2026) read end to end. David's scan_head_m3 variant in his Mojo engine matches two of the paper's three headline mechanisms (trapezoidal blend, rotation on B and C); MIMO and BCNorm are absent; two line-level details flagged as open questions in PLAN.md S12.
- Proof: docs/research/2026-10-27-mamba-papers-ingest.md.

## 2026-10-26: Glimmer 30B benchmark record and ROCm diagnosis on the RX 7900 XTX
- Fresh Vulkan and HIP llama.cpp builds; four quants benchmarked for speed and VRAM; serving accuracy and 2-user concurrency measured; Q4_K_M picked for serving; HIP failure isolated to the mixed ROCm userland install with a failing hello-world link test; Vulkan established as the production backend.
- Proof: docs/research/2026-10-26-glimmer-7900xtx-benchmarks.md with raw logs in bench/results-2026-10-26/.
