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

## 2026-10-29: Twenty-lane wave merged into one wired engine tree (integration-2026-10-29)
- Nineteen lane branches (forward-diff trace tooling, norm audit, RoPE and attention audit, KV decode sensitivity tests, logits head audit, Go core types, debug and event, health and registry, the IPC contract, Q4_K kernels, reference ops kernels, sampling, loader type table, arch dispatch, engine binary, AMD detection, CPU backend, gates and CI, golden harness audit) merged in dependency order into branch integration-2026-10-29 at 0f322f3, since pushed to GitHub with remote hashes verified equal to local.
- The wave fixed three forward defects: RoPE pairing corrected to the NEOX split-half convention (it was interleaved, in all 52 layers), the post norms moved onto each sub-layer output before the residual add, and the missing scaleless embedding norm added (proof: crates/torch-math/ROPE-AUDIT.md; crates/muse-glimmer/NORM-AUDIT-2026-10-29.md).
- New crates landed and the workspace is wired: itak-kernels, itak-types, itak-engine (the binary that assembles the crates), and itak-backend (backend trait, CPU backend, and sysfs AMD detection unified from two lanes); twelve crates in total, joined by path dependencies. The Go core module (pkg/types, pkg/debug, pkg/event, pkg/health, pkg/registry) and the platform IPC package landed in the same wave.
- Proof: cargo test --workspace 374 passed, 0 failed; core/ Go 48 passed; platform/ Go 40 passed; cargo clippy and cargo fmt clean; tools/gates.sh ALL GATES GREEN.

## 2026-10-29: Tokenizer proven token-for-token on Glimmer's real vocabulary (TODO T-2.3)
- The Rust BPE tokenizer matches the production llama-server token for token on the 16-prompt bench set (16/16 identical, 16/16 decode round-trips exact), using Glimmer's own 202,048-token vocabulary and 439,802 merges extracted from the production GGUF by the new gguf `tokenizer_extract` example.
- The first run failed one case, contractions: the pre-tokenizer had implemented the llama-3 split (contraction as its own piece), while llama.cpp routes `pre = llama4` to the GPT-4o pattern, where the contraction is a suffix of the word piece. `pretokenize.rs` and the crate docs now implement and document the GPT-4o pattern, with regression tests.
- Proof: crates/tokenizer/CONFORMANCE.md; harness in crates/tokenizer/examples/conformance.rs; reference in crates/tokenizer/conformance/.

## 2026-10-27: Project repo created with the full standard file set and the evidence base
- The rebuild plan (PLAN.md), task list (TODO.md), idea record (IDEA.md), end goal (ENDGOAL.md), decisions, status, runbook, map, and references were written from the project's real state and published together with the research documents and bench scripts under docs/research/ and bench/.
- Proof: this repository, created public on 2026-10-27 on David's order.

## 2026-10-27: Mamba lineage ingested; David's Mamba-3 variant graded against the real paper
- Mamba-1, Mamba-2, and Mamba-3 (arXiv:2603.15569, March 2026) read end to end. David's scan_head_m3 variant in his Mojo engine matches two of the paper's three headline mechanisms (trapezoidal blend, rotation on B and C); MIMO and BCNorm are absent; two line-level details flagged as open questions in PLAN.md S12.
- Proof: docs/research/2026-10-27-mamba-papers-ingest.md.

## 2026-10-26: Glimmer 30B benchmark record and ROCm diagnosis on the RX 7900 XTX
- Fresh Vulkan and HIP llama.cpp builds; four quants benchmarked for speed and VRAM; serving accuracy and 2-user concurrency measured; Q4_K_M picked for serving; HIP failure isolated to the mixed ROCm userland install with a failing hello-world link test; Vulkan established as the production backend.
- Proof: docs/research/2026-10-26-glimmer-7900xtx-benchmarks.md with raw logs in bench/results-2026-10-26/.
