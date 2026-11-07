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

## 2026-11-07: Second cold critic panel graded the tree DO NOT SHIP; its first fix categories landed the same day
- The panel: six cold seats over the full file inventory at 333f354, verdict DO NOT SHIP, 83 findings (0 critical, 16 high, 41 medium), the numeric core verified clean from every seat (record: the panel merge report, CRITIC-ENGINE-MERGE.md, 2026-11-07).
- Category A (snapshots): itak-kv snapshot decode refuses crafted geometry by name instead of aborting or accepting it (7a71727), and the prefix-KV seam refuses F16 KV storage by name (ca73795).
- Category D (socket seam): the platform authenticates the engine socket before dialing (owner uid and 0600 mode, mirroring the engine side) and retires desynced control connections after a timeout (07bce8a).
- Category E (platform answers): tool conversations render with their real names and calls (0578b44); cache key distinctness is pinned by field-by-field tests, unseeded sampled generations bypass the response cache, and template fetch failures are named instead of silently flattening the prompt (07fe5aa).
- Category F (proof tools): the golden and Q2/Q3 cross-check tools fail loudly: verdicts gate on the first values, failing or absent sets reach the exit code, and neither verdict tool exits 0 on a divergence (ce555fd).
- Category H (debug law): every flag-based diag seam gained an env twin in the gguf shape, the tokenizer moved onto the runtime flag, each seam gained a capture test, and the RUNBOOK gained the diagnostics switches table (a47dcf8).

## 2026-11-07: Coopmat v2 (K24) and split-K ship on by default
- The coopmat rung, the K24 selector, and the bridge implication are on when unset, each keeping its explicit 0 as the rollback (ecf8f3d; proof: bench/coopmat-default-flip-2026-11-07.md). The wave20 verdict behind it: K24's kernel line 1,737 ms against the standard MMQ's 4,021 ms at 1,022 tokens, prefill wall -36 percent, identity and golden pass, decision rule recorded as met.
- Split-K ships at fixed factor 4 when unset, with off as the rollback (2606489; proof: bench/splitk-default-flip-2026-11-07.md). The wave20 verdict behind it: prefill walls -18 percent at 1,022 tokens and -14 percent at 8,202; fixed factor 8 was disqualified by its own numbers (+50 and +88 percent).

## 2026-11-07: Build-all wave: prefix-KV store, F16 KV, split-K, coopmat v2, n-gram proposer, calibration profiles
- New crate itak-kv: the persistent tiered prefix-KV store foundation (9d6d434) and the ResidentSession seam over it (capture, install, store flow, bulk restore in attention; e94f30c). Wave20 real-file proof: a 4,096-token prefix restored cross-process in about 0.8 s against a 29.4 s cold prefill, 64 generated tokens byte-identical (bench/wave20-session-2026-11-07.md).
- F16 KV storage for the resident session, gated default-off (4a7343e): the wave20 identity gate passed (worst logit delta 1.54e-4) but prefill ran about 1 percent slower at 8,202 tokens, so it ships off as a memory option, with the 8,202-token argmax divergence from f32 recorded as its caveat.
- Split-K routing for the standard MMQ products (40e5c70) and K24 coopmat MMQ v2, a register-resident product loop behind the v2 selector (7df5d58), landed behind their switches and flipped on later the same day (entry above).
- N-gram prompt-lookup proposer behind a DraftSource seam in itak-spec (debdc73) and its GPU runner arm, which never opens the draft file in n-gram mode (d95234b). Opt-in only: acceptance is near zero on prose and real on repetitive text (wave20 leg 5).
- Auto-tune profile foundation: adapter-keyed calibration profiles with env-over-profile-over-shipped precedence, plus the calibrate_profile example (f9612cc). The calibration-derived band table was not adopted in wave19 or wave20; the shipped table stands.

## 2026-11-07: Default flips on card evidence: MMV ceiling 16, staged flash, K21F head route
- The MMV global gate default moved from 1 (off) to 16 (b342cdf), on the band-routing sweep and the baseline session's card protocol passing at that ceiling at both depths.
- Staged flash became the default for multi-row attention (82017ad), on the interleaved flash-host-cost evidence (kernel 605 ms against 966 to 974 ms at 1,022 tokens; wall wins at three lengths).
- The K21F finish-carrying head route: gained its own fail-closed switch (5f3ef6d); the defaults-confirmation run then failed identity with the route armed and pinned the cause to a one-character dequant mask typo in the kernel (0x0f0f0f0f written as 0x0f0f0f), fixed with all four dequant lines pinned verbatim (f7bc24d); the wave18 retest produced bit-identical head output and the default flipped on (d9825eb).

## 2026-11-06: MMV band routing, staged flash, K23 repack and coopmat v1, diag capture
- MMV band routing: one measured-win band cell per (family, shape) pair under the global ceiling (d5a67ea), with the draft session's projections and the gated and SwiGLU projections routed through the ladder (0e5194f, 7145fe1).
- Staged flash attention with 32-wide K/V blocks staged in workgroup memory, behind its switch (2b6afb8); the diag capture level with a run manifest and per-forward data fingerprints (1325b9d); capped draft block readback wired into the speculator (6d260e2); laned argmax with scalar-exact tie semantics (ba23221).
- K23 Q4_K MMQ with llama's staging repack (a1de267) and the Q4_K coopmat kernel behind the device bridge (e143b61, bb9af9f). Both later measured slower than the standard path on the card (the repack about 2.15x, coopmat v1 about 4.0x in the wave20 reproduction) and ship off.

## 2026-11-05: Overnight verification wave; the register-streaming MMV family; prefill host gap closed
- Verification: golden fixtures regenerated with no drift, a Glimmer subset fixture and the top-5 first-token fixture landed, the per-layer forward diff found no diverging layer, and TODO T-9.3 (the first-token probability golden) closed on that evidence (4c329c5, 8fb984f, 3ac2b64, 9ce57f3). Q2_K and Q3_K dequantizers landed in gguf, cross-checked bit-exact against ggml (b3a1c6c).
- Kernels: register-streaming MMV products K19 for Q4_K (40c11c3), K20 for Q6_K (09a03c9), K21 for Q5_K (433077a), and the chunked GEMV small-batch arm (5eeab9b), all behind switches; the K19 card verdict and the dose-response sweep followed the same day (bench/mmv-verdict-2026-11-05.md).
- Infrastructure: the prefill host gap closed (uploads flush at build, chunk readbacks defer; 2682e02) after the per-kernel prefill profile named it (dff14fd); per-dispatch GPU timestamp instrumentation landed behind its switch (11991a6); the GGUF file image is memory-mapped at open (b24b5d9); the K22 register-cache MMQ occupancy prototype landed behind its switch (9d07b13) and was later convicted by its own autopsy (it compiles onto the 256 VGPR cap like the standard kernel; ships off).

## 2026-11-04: Speculative loop proven on the resident GPU target; the depth sweep closes T-9.7
- TODO T-9.7 closed: the on-stack depth sweep over the real files recorded depth 1 as the winner and the SpecConfig default (72a4b9d). The sweep machinery (e87daf1 and companions) and the GPU sweep runner (ef06cc7) landed with it.
- Flash attention for chunk plans, with the scores scratch and budget decay removed (merge d94b852); quantize-once in chunk plans (merge 504d46f, f025d41).
- K18 small-batch matrix kernels for Q4_K, Q5_K, and Q6_K landed behind their switch (merge ef6ca5d); the server verdict the same day: numerically correct at every probed row count, slower than the GEMV forms, stays off (bda7ff5).
- Wide MMQ (K15/K16) verdict recorded: numerically correct but about 3.5x slower per tensor, and the wide ladder hung the real session; the wide form left the default ladder (2797923, dff467d).

## 2026-11-03: GPU speculative loop, GPU-resident DFlash draft, and wide-tile MMQ land
- The speculative loop runs on the resident GPU target through the TargetDriver seam (merge 3e785cf: block logits, layer captures, bounded rewind in the resident session), TODO T-9.6's core landing.
- The GPU-resident DFlash draft forward (DflashSession) landed behind the Speculator seam (merge fd78479), and the T-9.7 depth sweep machinery with them (merge 2ed7389).
- K15/K16 wide-tile double-buffered MMQ for prefill landed behind its threshold (merge 8b77bf6); its verdict arrived the next day (entry above).

## 2026-10-31: Panel fix waves 1 to 3 landed; the T-9.17 live proof completed
- Wave 1 (merge 19f2cba, four lanes): engine socket hardening (owner-only socket, hello enforcement, peer uid check, frame bound, JSON depth cap, the max_tokens wrap closed), chat-template render budgets, platform bounds (generation caps and deadline, scheduler queue caps, response cache cap with janitor, supervisor environment allowlist, the supervisor stderr wedge fix, engine client frame bound, stream terminal honesty, auth throttling and key floor, static /healthz), and metadata magnitude bounds in the architecture config and KV sizing.
- Wave 2 (258911d through 7eacc30): the formatting debt cleared, deny.toml added with cargo deny check passing, toolchains pinned (rust 1.99.1; platform go1.26.10, closing the panel's GO-2026-6809 under govulncheck), manifest descriptions and TODO proof texts corrected, and the chattemplate gosec G101 cleared.
- Wave 3: the engine diag module with every log path and refusal routed through it (b092782, db38d99), gguf diag emission made unconditional and routed through one seam (2fa4cbb, 590b6ce), sampler entry points refusing contract violations by name with the panel's probes as regression tests (94babd2), engine worker panics contained as engine_died outcomes (a928eb2), the platform lint leftovers resolved, and the authored docs rewritten from the tree (bcabf4d, 6791f90, 44f71d5, e4742ae, 2f3b0b4, 50fb531).
- The T-9.17 live proof run on the real Glimmer file completed and its record was filed (e1d791c), closing T-9.13 through T-9.17.

## 2026-10-30: Serving production gaps closed in the platform and engine (TODO T-9.11 to T-9.16)
- Supervision (T-9.11, closed): torchd supervises itak-engine under a configured restart budget; a mid-generation death answers the in-flight request with the named ErrEngineDied error and the next request is served by the restarted engine (platform/internal/supervisor, commit 4f2dd5e, proven against a real child process that dies mid-generation).
- Terminal stream errors (T-9.12, closed): a mid-generation failure reaches the HTTP client by name (502 or a terminal SSE error frame), never as a truncated 200 (commit 6806ae6, fault-injection proven at the socket and HTTP layers).
- Sampling parity (T-9.13), multi-slot engine (T-9.14), embeddings over the seam (T-9.15), and chat templating (T-9.16) landed end to end (commits 4248eb2, 7163ead, 48f56ce, 311fa1b): seed/top-k/penalties in the protocol and sampler, --slots with a bounded queue and the engine_busy refusal, a real embed operation replacing the 501, and prompts rendered from the model's own GGUF chat template by a bounded Jinja-subset renderer. Their named live proofs on the real Glimmer file are filed as T-9.17, blocked on a server CPU window.

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
