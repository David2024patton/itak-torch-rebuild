# Fractal Memory Experiment, 2026-10-29

Status: design only, nothing built. Placement: phase seven experiment
track, gated behind the baseline engine milestone (below). Ordered by
David 2026-10-29 ("Do both") alongside the activation dimension probe,
which is this experiment's instrumentation and lands first.

Question this spec answers: can memory in our engine be organized the
way a fractal is organized, the same structure repeated at every
scale, so that long context stops costing memory in proportion to its
length? Two coupled pieces, each its own module, each testable alone.

---

## The idea

### Piece one: the summary pyramid (hierarchical KV)

Full KV storage keeps one key and one value per token per layer,
forever. The pyramid keeps that only for the recent past:

- The newest `block_tokens` tokens (config, default 128) are held as
  ordinary full-resolution KV, exactly as the baseline stores them.
- When a block fills, it is summarized into one summary entry: a
  single key/value pair produced by a learned-free reduction owned by
  this module (first version: attention-pooled mean of the block's K
  and V, behind the module seam so the reduction can be replaced).
- Summaries live on levels. Level 0 holds block summaries. When a
  level accumulates `ratio` summaries (config, default 4), they merge
  into one summary on the next level: a summary of summaries. Every
  level is capped at `level_cap` entries (config, default 32); a
  merge is what makes room.
- Attention reads the live block plus every level's summaries. Old
  history is present at every scale, coarse at the top, fine at the
  bottom.

Storage math, stated exactly: the number of levels needed for context
`N` is `ceil(log_ratio(N / block_tokens)) + 1`, and each level holds
at most `level_cap` summaries, so stored entries grow with the
logarithm of context length while coverage grows with its power. At
the defaults and 131,072 tokens of context: 1,024 leaf blocks, 5 merge
levels above them, at most 192 summaries held, plus the live block.
One summary costs the same bytes as one token of KV in the layers it
serves, so the honest comparison is entry count against token count.

Scope of the first build: the pyramid serves the 13 global layers
only. The 39 sliding-window layers already forget by design and keep
their existing rings untouched.

### Piece two: the fixed-size evolving state

David's standing design (recorded 2026-10-29): memory as a volume
whose state evolves step by step, instead of a growing row of stored
tokens. The state is a fixed-size tensor per layer; every token
updates it, nothing is appended. Memory cost is the state size,
independent of context length, at token ten or token one hundred
thousand.

The fractal element is how the state's decay rates are chosen. The
state is partitioned into components, and their decay timescales are
spread geometrically from `tau_min` (config, default 1 token) to
`tau_max` (config, default the context target): each component
forgets at a rate a constant ratio slower than its neighbour. Fast
components carry fresh detail, slow components carry the deep past,
and no timescale in between is missing. This scale-free spread is kin
to how state-space models already pick timescales (the HiPPO and S4
lineage initializes timescales spread across orders of magnitude, and
Mamba's step size plays the same role per token); the experiment is
whether making that spread the explicit, configured structure of a
standalone memory module earns its place next to attention.

The two pieces answer different failures and are judged separately.
The pyramid compresses attention's memory. The state replaces it.
The combined configuration runs only if each piece passes alone.

## Placement and seams

- Entry gate: the baseline engine is token-exact against llama.cpp
  on the golden suite over the real Glimmer file. Until that is true,
  no memory experiment means anything, because there is no trusted
  baseline to diverge from. This is the Phase 9 milestone path in
  PLAN S13, and this experiment queues behind it on the phase seven
  track.
- The module plugs in behind the existing KV/state seam: the
  position contract in `muse-glimmer::cache` and the KV storage the
  `attention` crate implements behind it. It is an alternative
  storage implementation selected by config, never a fork of the
  forward path. Tokenizer, weights, per-layer forward math, sampler,
  and the golden harness are shared with the baseline unchanged.
- Every knob is config with a default, validated at load, refused by
  name when out of range: `block_tokens`, `ratio`, `level_cap`,
  `tau_min`, `tau_max`. No limit is compiled into the math.
- Diagnostics from birth, through the crate `Diag` channel: level
  occupancy and merge events for the pyramid, the timescale table at
  init for the state, and a named refusal for every invalid config,
  in the same key=value style as the rest of the engine.

## Precondition: cross-layer weight self-similarity

A third idea rides along whenever fractals come up: if the network
were self-similar across depth, layers could share or generate each
other's weights. Before any such variant is designed, measure it on
the real Glimmer file:

- For every same-role tensor (attention q/k/v/output, FFN gate/up/
  down), compute the cosine similarity and relative Frobenius gap
  between layer `i` and layer `i + 1`, and between layer `i` and
  layer `i + 4` (the global-layer period in Glimmer's pattern), over
  dequantized f32 weights.
- Gate: a weight-sharing variant may be designed only if the median
  adjacent-layer cosine is at least 0.9.

Stated plainly: trained weights are only loosely self-similar.
Adjacent layers correlate, but they drift steadily with depth, and
the expectation is that this measurement lands far below the gate
and the weight-sharing variant dies here. That is the gate doing its
job, and the result gets recorded either way. The memory pieces above
do not depend on this measurement; they exploit structure in the
context, not in the weights.

## Experiment design

What gets swapped: only the memory module behind the seam, one piece
at a time (pyramid alone, state alone, then combined). What stays
identical: the binary, the weights file, the tokenizer, the prompts,
the sampler at temperature 0, the golden harness and its scoring.

Metrics, all against the full-KV baseline running in the same build:

| Metric | How measured | Contexts |
| :--- | :--- | :--- |
| Token agreement | Exact top-1 match rate against baseline decode, golden suite, temperature 0 | full suite |
| Memory bytes | Resident KV/state bytes, growing part only | 1k, 8k, 32k, 131k |
| Decode speed | Tokens per second, 7900 XTX, same build, both modules | 8k |
| Collapse signal | Per-stage participation ratio and TwoNN from the dimension probe, variant vs baseline at the same stage | full suite |

Baseline reference points, from the measured Glimmer geometry: the
growing part of full KV is the 13 global layers at 13,312 bytes per
token, roughly 0.11 GiB at 8k context and 1.7 GiB at 131k including
the capped window layers. Recorded llama.cpp decode on this card is
44.31 tok/s (Vulkan, plain) per PLAN S13; the speed comparison that
counts is variant against our own baseline in the same build.

One harness gap to close before the runs: the golden set is
short-prompt today. Add a long-range exact-recall prompt (a fact
stated early, queried late) so the suite can catch the pyramid's
known weakness instead of grading only what it is good at.

Kill criteria, named margins, no judgement calls at review time:

- Agreement: token-exact agreement below 99 percent of decode
  positions on the golden suite shelves the piece.
- Memory: pyramid memory at 32k context not at least 4x below the
  full-KV growing bytes shelves the pyramid.
- Speed: decode more than 10 percent slower than the same-build
  full-KV baseline at 8k shelves the piece.
- Collapse: a variant whose participation ratio at mid or late
  layers falls toward 1 while the baseline's holds is washing out
  history (the lane 04 signature) and shelves the piece regardless
  of its token agreement.

A shelved piece is recorded in DECISIONS.md with its numbers. The
module stays in the tree behind its config flag, off by default, so
the result is reproducible and the idea can be retried when something
upstream changes.

## Instrumentation

The activation dimension probe (`muse-glimmer::dimension`, landed
with this spec) is the collapse detector for these experiments. It
reads a captured stage's activation matrix and reports two numbers:
the participation ratio of the covariance spectrum (how many
directions the stage's signal actually occupies) and a TwoNN
intrinsic-dimension estimate. Lane 04 proved a forward can quietly
wash a token's contribution out across depth while still producing
plausible tokens; token agreement alone can miss that failure while
it is developing. The probe turns it into a per-stage number that
diffs cleanly between baseline and variant.

## Risks and honest unknowns

- Fractal architectures have lost before. FractalNet (2016) built
  whole networks from self-similar blocks; it matched its
  contemporaries and did not beat them, and the field moved to plain
  residual networks. Self-similarity is a prior, not a guarantee,
  and this spec is written so a negative result is cheap and final.
- Exact recall is what summaries cannot do. Full KV can re-read an
  old token verbatim; a summary has already thrown the verbatim
  away. Needle-style retrieval over long context is the expected
  failure, perplexity-style agreement may look fine while it
  happens, and the recall prompt in the suite exists for exactly
  this reason.
- Summary error compounds. A summary of summaries has been reduced
  `ratio` times more than a leaf summary; the top of the pyramid is
  the least trustworthy memory in the system and the hardest to
  debug when a generation goes wrong far from the cause.
- A fixed-size state fails silently. Its capacity does not grow, so
  past some context length the state piece will not error, it will
  just get worse. The agreement and collapse metrics are the only
  tripwires; nothing in the module itself can announce the loss.
- Interaction with Glimmer's own two scales is untested territory.
  The architecture already pairs local windows with periodic global
  layers; whether a logarithmic pyramid over the global layers
  composes with that pattern or fights it is an empirical question
  the experiment exists to answer.
