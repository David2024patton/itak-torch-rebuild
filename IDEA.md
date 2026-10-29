<!--
INSTRUCTIONS FOR THIS FILE (user and AI: read before editing. Keep this header.)

WHAT THIS FILE IS: David's idea capture for this project, as full records.
When David talks, dictates, or drops a thought about this project in any
chat, it lands here. Ideas are never silently dropped.

EXACT STRUCTURE: one entry per idea, newest first:
## <Category> | <YYYY-MM-DD HH:MM TZ>
- **David:** "<his words verbatim where on record; otherwise a faithful
  summary marked (summary)>"
- **Lulu:** <the reply or assessment given at the time; a recorded summary
  marked (summary) when it is not a verbatim quote. Never invent a quote.>
- **Stats:** would take: <one line>; effort: <S/M/L>; value: <one line>;
  depends on: <items or none>
- **Status:** NEW | MOVED to TODO T-<phase>.<n> | KILLED <YYYY-MM-DD>

CATEGORIES: Product, Feature, Design, Tech, Business, Ops, Research,
Personal. Pick the closest; an idea may be re-filed as it matures.

RULES:
- Times are recorded only when the source message carried one; never
  invent a time. Dates always.
- An idea leaves NEW only via MOVED (with the TODO item id) or KILLED
  (David's call, with the date). Nothing is deleted.
- No secrets, passwords, or tokens in this file, ever.
-->
# IDEA: iTaK Torch Rebuild

## Ops | 2026-10-27 22:17 EDT
- **David:** "Yes make it public. Use the GitHub skill to make the to-do etc files"
- **Lulu:** Created the public repo itak-torch-rebuild with the full standard file set, seeded from the real project state, and the research documents published alongside the plan (summary).
- **Stats:** would take: one working session; effort: S; value: the project's record stops living in one local folder, the failure mode that cost us Core; depends on: none
- **Status:** MOVED to TODO T-0.1

## Tech | 2026-10-27 10:26 CDT
- **David:** "if we use my rule at using the different languages to accommodate different features if we did that would you use go and rust together?"
- **Lulu:** Yes. Split the system by what each language is best at: Rust owns the token loop (kernels, KV, the SSM scan, MoE streaming, speculation, training compute), Go owns the platform (API, scheduling, caches, cluster, model operations, observability), two processes joined by a socket for control and shared memory for tokens, Mojo stays the research lab (summary).
- **Stats:** would take: this framing shapes every phase; effort: L; value: it is the architecture of the whole rebuild; depends on: the Rust rebuild question below
- **Status:** MOVED to TODO T-1.4

## Tech | 2026-10-27 (morning, EDT)
- **David:** asked what rebuilding his inference engine in Rust to bypass the garbage collector would look like (summary; his framing was the GC, the detailed wording is paraphrased in the session notes)
- **Lulu:** The honest answer had two parts: Go's GC was not why the original native path was slow (it had no KV cache and a char tokenizer; it was pedagogical), but GC jitter is real for the intended design (tensor churn, KV growth, expert streaming). The real fix is allocate-once architecture plus Rust on the hot path, delivered as a crate map (summary).
- **Stats:** would take: the full rebuild; effort: L; value: steady per-token latency and a maintainable engine; depends on: Core rebuilt first
- **Status:** MOVED to TODO T-2.4

## Research | 2026-10-27 10:42 CDT
- **David:** uploaded the three Mamba papers (Mamba-1, Mamba-2, Mamba-3) for ingest and ordered arXiv folded into the deep research skill as a standing pass (summary)
- **Lulu:** Ingested all three, graded his Mojo Mamba-3 variant against the real paper (two of three headline mechanisms matched; MIMO and BCNorm absent; two line-level details flagged for verification), and updated the deep research skill with the top-10 arXiv pass (summary).
- **Stats:** would take: one session; effort: M; value: the SSM module design now stands on the primary sources, and his variant's exact standing is on record; depends on: none
- **Status:** MOVED to TODO T-0.5

## Tech | 2026-10-26 21:00 EDT
- **David:** stated the original design intent: Torch had no Glimmer support and only small-model tests because he originally had only the 16 GB card; he was building for low cards, fast output with good quality on small VRAM (summary of his words, recorded in memory the same night)
- **Lulu:** Judged the engine against that target from then on, and carried the same target into this rebuild's end goal (summary).
- **Stats:** would take: a standing design rule, not a task; effort: S; value: every design call gets measured against the real target instead of datacenter assumptions; depends on: none
- **Status:** MOVED to TODO T-5.4

## Research | 2026-03 (from the design record)
- **David:** the Latent Reasoning Engine vision document (LATENT_ENGINE_VISION.md, March 2026, marked Research and Prototype): token-free internal reasoning over raw bytes, learned patch compression from bytes to concepts, latent-space reasoning passes, tool interception from latent activations, token decode only at the output; Mamba constant-size state as the memory backbone, with EGGROLL memory crystallization (summary of the document)
- **Lulu:** Recorded in the 2026-10-27 ingest as the project's research north star, and as an explicit research track in this rebuild rather than engine scope, because no code for it exists yet (summary).
- **Stats:** would take: a research program, not a phase; effort: L; value: it is the differentiating bet of the whole iTaK line; depends on: itak-ssm proven first
- **Status:** MOVED to TODO T-8.3

## Tech | 2026-04-16 (from the design record)
- **David:** the .itf (iTaK Tensor Format) specification in the universal-engine plan: ITF1 header with version field, architecture and quantization metadata, memory tiers, latent state dimensions, per-tensor heat levels (HOT, WARM, COLD) and SSM state sections, GGUF-header-compatible (summary of the document)
- **Lulu:** Recorded in the ingest; the format was specified but the converter was never built, so it enters this rebuild as Phase 8 with measured heat data replacing positional labels (summary).
- **Stats:** would take: one format implementation plus converter; effort: M; value: a model format that carries what this engine actually needs (SSM state, heat tiers) instead of borrowing GGUF forever; depends on: itak-dissect statistics
- **Status:** MOVED to TODO T-8.1
