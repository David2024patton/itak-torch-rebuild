<!--
INSTRUCTIONS FOR THIS FILE (user and AI: read before editing. Keep this header.)

WHAT THIS FILE IS: every link David posts about this project, and the
answer to it. When David drops a link in any chat (a repo, an article, a
prompt collection, a product), it lands here with what it is and what we
took from it, so the research is never lost and never repeated.

EXACT STRUCTURE: one entry per link, newest first:
## <YYYY-MM-DD HH:MM TZ>: <link>
- **What it is:** <one or two lines, factual>
- **License:** <license or "n/a">
- **Lulu's answer:** <verdict: TAKE, RETROFIT, REFERENCE, or SKIP, with
  the reason in plain words>
- **Went to:** <where the takeaway lives now: a TODO item id, an IDEA
  entry, the shelf file, or "nowhere, verdict only">

RULES:
- Every posted link gets an entry the same day, even when the verdict
  is SKIP; the skip and its reason are the value.
- Facts only: license and contents verified from the source, not from
  the link's reputation.
- No secrets, passwords, or tokens in this file, ever.
-->
# REFERENCES: iTaK Torch Rebuild

## 2026-10-28 02:25 EDT: https://github.com/meta-models/meta-model-cookbook
- **What it is:** Meta's official developer cookbook for the hosted Meta Model API (Muse Spark, muse-spark-1.3, 1,048,576 token context, OpenAI and Anthropic SDK compatible, free preview). Seven sections: API fundamentals, agent patterns, end-to-end use cases, the Muse Code agent harness, Muse Image, Muse Voice Transcribe, and SAM 3. Colab notebooks plus standalone Python files. Verified from the repo itself on the night it was posted: MIT license, created 2026-07-07, 83 commits, 193 stars.
- **License:** MIT
- **Lulu's answer:** REFERENCE, with one TAKE inside. It is an API cookbook, not a fine-tuning cookbook: it contains no Glimmer training recipes, so the fine-tune guide David pasted the same night keeps its unverified status on the claim that Meta ships official Axolotl YAMLs for Glimmer. The TAKE is use case 3.15: run Muse Glimmer locally and require approval before selected questions escalate to hosted Muse Spark. That local-first-with-a-gate pattern fits the rebuild's serving layer exactly. The Muse Code section (append-only audit log with crash-safe resume, deterministic replay in CI, staged approvals for compound shell commands, subagent fanout across isolated worktrees, goal tracking with a judge) is a pattern catalog for the agent layer: read as reference, not code to vendor.
- **Went to:** verdict recorded here; the local-escalation pattern is noted for the serving and agent layer design

## 2026-10-27 22:13 EDT: Article pasted by David, "Running Meta Muse Glimmer 30B on the XFX Speedster MERC 310 AMD Radeon RX 7900 XTX" (no URL on record; text pasted in chat)
- **What it is:** A walkthrough claiming the 7900 XTX is the sweet spot for Glimmer 30B: VRAM budget tables per quant, a HIP/ROCm superiority argument (WMMA), channel-scoped wire protocol notes, reasoning_strength levels, and two llama-server launch profiles (solo with DFlash, multi-slot without).
- **License:** n/a
- **Lulu's answer:** REFERENCE. Its skeleton matches our own measured record (KV size from sliding window attention, the 4-slot split, Q4_K_M as the pick, Q6 not fitting, --jinja mandatory, DFlash off for multi-user). Its two headline performance claims do not hold on David's machine: HIP beats Vulkan assumes a clean ROCm install, and ours is broken at the system level; the 75 to 110+ tok/s DFlash range is unverified, and its draft depth of 16 conflicts with our research value of 4. Its weights table is sloppy (Q4_K_M listed as 19.0 GiB; the real file is 15.59 GiB). Graded claim by claim against the benchmark report the night it arrived.
- **Went to:** TODO T-P.1 (the DFlash draft-depth bench that settles 4 vs 16) and PLAN.md S12 open questions (sampling defaults)

## 2026-10-27 10:42 CDT: Mamba-3 paper, uploaded by David as PDF (arXiv:2603.15569)
- **What it is:** Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu: "Mamba-3: Improved Sequence Modeling using State Space Principles" (16 Mar 2026, preprint). Exponential-trapezoidal discretization, complex-valued state proven equivalent to data-dependent RoPE on B and C, MIMO rank, BCNorm and B, C biases, short conv removed.
- **License:** n/a (preprint)
- **Lulu's answer:** TAKE. It is the design basis for itak-ssm: the general three-term recurrence, rotation as a scan-interface concern, MIMO rank as config, BCNorm and biases in the block spec. Also the grading standard for David's own extrapolated variant, which matched two of its three headline mechanisms in the same March to April 2026 window; priority is not established and is not claimed.
- **Went to:** docs/research/2026-10-27-mamba-papers-ingest.md; PLAN.md S8; TODO T-5.2 and T-5.3

## 2026-10-27 10:42 CDT: Mamba-2 paper, uploaded by David as PDF (arXiv:2405.21060)
- **What it is:** Dao and Gu: "Transformers are SSMs" (SSD framework). SSMs as semiseparable matrices and structured masked attention; A restricted to scalar times identity per head; chunked algorithm mixing the quadratic dual form inside chunks with the recurrence across chunk boundaries.
- **License:** n/a
- **Lulu's answer:** TAKE. The chunked SSD algorithm is the implementation shape for the Mamba-2 mode of itak-ssm and the reason one kernel can serve all three generations.
- **Went to:** docs/research/2026-10-27-mamba-papers-ingest.md; TODO T-5.1

## 2026-10-27 10:42 CDT: Mamba-1 paper, uploaded by David as PDF (arXiv:2312.00752)
- **What it is:** Gu and Dao: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces". Input-dependent delta, B, and C; hardware-aware parallel scan keeping the state in SRAM; the selection mechanism as a generalized RNN gate.
- **License:** n/a
- **Lulu's answer:** TAKE. The foundation layer: selection semantics every later mode inherits, and the baseline the state-tracking checks are graded against.
- **Went to:** docs/research/2026-10-27-mamba-papers-ingest.md; TODO T-5.1

## 2026-10-26 (research pass): Colibri engine (upstream open source, studied from David's K drive working tree)
- **What it is:** A pure C inference engine that streams a 744B MoE model's int4 experts from disk so it runs in about 25 GB of RAM, plus David's working Mojo port of it (about 2,100 lines across 17 files, all six porting phases checked in his task file).
- **License:** upstream license as published in its own repo; David's port is his
- **Lulu's answer:** REFERENCE for the pattern, with a license caution. The expert-streaming pattern is the design basis for itak-moe. Code is not vendored into this project; the pattern is reimplemented in Rust. The license question is an explicit open item in PLAN.md S12, to close before Phase 5 starts.
- **Went to:** PLAN.md S8 and S12; TODO T-5.4
