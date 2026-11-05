# Research archive

This folder holds the research the engine's speed work is built on, kept
inside the repo so the evidence travels with the code it justifies.

## Contents

- `streaming-rate-research-2026-11-04.md` — the full report of the
  exhaustive streaming-rate research run of 2026-11-04: why two
  shared-weight small-batch kernels streamed at 70-130 GB/s while the
  per-row GEMV kernels reach up to 676 GB/s on the same RX 7900 XTX, what
  the profiling tooling on RADV/RDNA3 can and cannot show, which
  driver-level levers exist, and the kernel-design techniques the
  register-streaming prototypes (K19/K20/K21) were built from.
  Copied verbatim from the run's `report.md`.
- `streaming-rate-key-sources-2026-11-04.md` — the run's key-sources
  index: every source the report rests on, marked as verified live or
  index-only. Copied verbatim from the run's `notes/key-sources.md`.

## Origin

Deep research run executed 2026-11-04 (run folder
`gpu-streaming-rate-exhaustive-20261104-2133` in the workspace research
notes), ordered when the wide-MMQ and K18 small-batch attempts both
proved slower than the GEMV path they were meant to beat: the question
put to the run was, exhaustively, what can be done to analyze and fix
that streaming-rate collapse. The PDF edition delivered from this same
report lives outside the repo; this archive is the text source of
record.

## How the bench notes cite it

The bench notes in `bench/` for the register-streaming wave
(`mmv-prototype-2026-11-05.md`, `mmv-q6k-2026-11-05.md`,
`mmv-q5k-2026-11-05.md`, and the verdict notes that follow them) cite
this research by its workspace research-notes path. Those citations
point at the same material archived here; when a claim in a bench note
needs checking, the report section and the key-sources index in this
folder are where it resolves.

## Left out on purpose

- The run's `AGENTS.md`: instructions written for the research agent
  that ran the sweep, not evidence. Its mission statement is quoted
  inside the report's framing where it matters.
- Rendered editions (the delivered PDF, any HTML report): this archive
  keeps text sources only, so every claim stays diffable and quotable.
