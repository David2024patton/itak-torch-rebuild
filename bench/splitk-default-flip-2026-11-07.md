# Split-K default flip: fixed factor 4 ships (2026-11-07)

Lane `lane/splitk-default-flip`, from main 333f354.
The MMQ split-K routing config
(`WGPU_COMPUTE_MMQ_SPLITK`) flips its shipped
default from OFF to fixed factor 4. The resolver's
explicit positions, the factor rule (including the
auto rule's arithmetic), the kernels, and the
reduction are unchanged; only the unset position
moved, in the same shape as the MMV ceiling and
K21F head flips (unset resolves to the shipped
default; a present but unrecognized value resolves
to the closed fallback, never to the default).

## Authority

David's standing order, 2026-11-07: "Anything that
makes sense engine go faster should be set to on."
This flip is that order applied to the wave20
split-K verdict.

## The evidence

bench/splitk-mmq-2026-11-07.md shipped the split-K
forms default-OFF with a written flip criterion:
the default moves only if the interleaved prefill
A/B shows a real win at both lengths with no
regression on the unsplit shapes, identity+golden
hold at `auto` and at `=8`, and the autopsy shows
no new spills.

The wave20 consolidated card session
(bench/wave20-session-2026-11-07.md, leg 1, staged
from main 515ce23 on the RX 7900 XTX) ran that
protocol. Reading against the criterion:

- **Prefill A/B, interleaved pairs, DIAG=timing.**
  Walls at 1,022 tokens: off 6,446 / 6,408 ms;
  fixed 4: 5,175 / 5,324 ms (about -18%). Walls at
  8,202 tokens: off 68,476 / 68,571 ms; fixed 4:
  59,047 / 58,877 ms (about -14%). A direct
  unwrapped confirmation run at factor 4
  reproduced the split lines (wall 5,106 ms),
  ruling out a wrapper artifact.
- **No unsplit-shape regression.** At fixed 4 the
  split forms beat the plain kernels on the wide
  shapes too: `k13_q4k_mmq_split` at 59.1 / 58.7
  GB/s vs plain 47.9; `k14_q6k_mmq_split` at 64.0 /
  63.4 GB/s vs plain 35.6 (1,022-token kernel
  lines). The reduction costs 50.2 ms across 3,744
  dispatches at that shape.
- **Correctness legs executed on the card.**
  tests/splitk_mmq.rs: 3 passed. Drift is
  reassociation-class, as the design predicted
  (relative at most about 8.5e-7); the split form
  is not bit-identical to the unsplit product by
  construction and never claimed to be.
- **Identity + golden.** PASS at `=auto` and at
  `=8` (the identity gate runs the flag at its
  admitted extremes; factor 4 sits between them).
  The `=auto` capture manifest carries
  `"mmq_splitk":"auto"` in `resolved_config`.
- **Autopsy: no new spills.** PARTIAL as scoped:
  the K13/K14 family dump showed 0 spilled SGPRs,
  0 spilled VGPRs, 0 scratch across all blocks,
  with the plain LDS figures reproduced exactly.
  Per-form attribution by name was not isolated
  (RADV dump blocks are anonymous); the remaining
  method (a dump run of the splitk_mmq test binary
  itself) is named in the session record.

The criterion is met at fixed factor 4. This lane
is that flip, and nothing else.

## Factor 8 is disqualified; auto is not the default

Both were measured in the same wave20 A/B, and
the record stands so neither is re-proposed as a
default without new evidence:

- **Fixed 8: DISQUALIFIED.** Walls +50% at 1,022
  tokens (10,180 / 9,071 ms) and +88% at 8,202
  (128,771 / 129,251 ms). The mechanism is in the
  kernel lines: at 8,202 the K13 split form itself
  collapses to 27.5 GB/s and `splitk_reduce` alone
  takes 27,538.9 / 27,655.5 ms across 29,952
  dispatches, 22.8% of total GPU time. Factor 8
  remains an admissible EXPLICIT operator value
  (the resolver still accepts `1`..=`8`); the
  disqualification is of 8 as a default, and it
  was already visible in the design note's slice
  math: at 26 blocks, factor 8's slices walk
  4/4/4/4/4/4/2/0, the last slice empty.
- **Auto: marginal, not shipped.** Walls about
  -3.5% at 1,022 and -2% at 8,202. Under auto only
  the KV projections split, and the rule's pick
  for that shape is factor 8, the disqualified
  factor; its small net win survives only because
  the KV shape is a small share of the prefill.
  The auto rule's arithmetic is unchanged by this
  flip; it remains an operator choice for
  shape-targeted runs.

## The change

- `compute::resolve_mmq_splitk` now resolves over
  two named constants: `DEFAULT_MMQ_SPLITK`
  (`MmqSplitK::Fixed(4)`, the shipped default;
  `pub(crate)` because the mode type is) and
  `MMQ_SPLITK_FALLBACK` (`MmqSplitK::Off`, the
  closed fallback). New semantics matrix:

  | value | resolves |
  |---|---|
  | unset | fixed 4 (the shipped default, `DEFAULT_MMQ_SPLITK`) |
  | `off` (trimmed) | OFF (the explicit off; the one-variable rollback) |
  | `auto` (trimmed) | the per-projection auto rule (arithmetic unchanged) |
  | `1`..=`8` (trimmed) | that fixed factor |
  | any other text | OFF, fail-closed (`MMQ_SPLITK_FALLBACK`), logged by name at construction |

  A present-but-malformed value does NOT inherit
  the new default: it resolves to the fallback,
  and `log_mmq_splitk_fallback` names it in the
  construction log ("failing closed to off, NOT
  the shipped default fixed:4"), on the flash /
  K21F fallback pattern. The construction log line
  now reads: `mmq splitk config: mode=<mode> (env
  WGPU_COMPUTE_MMQ_SPLITK: unset is the shipped
  default fixed:4, "off" disables, "auto" is the
  tile-count rule, 1..=8 a fixed factor; rollback
  WGPU_COMPUTE_MMQ_SPLITK=off)`. Resolution still
  happens once at backend construction; the
  `mmq_splitk` accessor and the `mmq_splitk`
  resolved-config pair in both sessions' diag
  manifests are unchanged in shape and now report
  `fixed:4` at unset.
- Routing is untouched: the split still applies
  at the standard rung only, behind
  coopmat/wide/K22/K23, never stacked on the
  prototypes, with the logged fallback to the
  unsplit prepare. `mmq_splitk_factor` is
  unchanged in every arm (Off, Fixed clamp, the
  auto rule's `ceil(64 / tiles)`).
- Doc updates where the old default was stated:
  the env, resolver, accessor docs and the
  construction comment in compute.rs; the
  `mmq_product_q4k` ladder comment in session.rs;
  the split paragraph in the K13 module header.
  bench/splitk-mmq-2026-11-07.md gains an appended
  supersession section (its history stands).

## Pins

- `compute::mmq_splitk_resolution` re-pinned to
  the full new matrix: unset resolves to
  `DEFAULT_MMQ_SPLITK` (pinned equal to
  `MmqSplitK::Fixed(4)`); `off` off; `auto` auto;
  fixed factors `1`, `4`, ` 8 ` admitted as
  before; malformed values (`0`, `9`, `true`,
  `on`, empty, `-2`) pinned to
  `MMQ_SPLITK_FALLBACK` (pinned equal to
  `MmqSplitK::Off`), never to the default.
- `compute::mmq_splitk_value_recognition`
  unchanged and still green: the recognized
  positions are exactly unset, `off`, `auto`, and
  integers in 1..=8; unset being recognized is
  what keeps the default position out of the
  fallback log.
- `compute::mmq_splitk_factor_rule` unchanged and
  still green: the factor rule, the auto rule
  included, was not touched by the flip.
- Script sweep: every `MMQ_SPLITK` hit in the
  workspace was accounted for (compute.rs, the
  K13/K14/splitk_reduce module docs, the
  session.rs ladder comment, the two bench
  notes). NO bench script references the switch:
  no script's control arm relied on unset meaning
  off, so no script needed the explicit
  `WGPU_COMPUTE_MMQ_SPLITK=off` pin the MMV flip
  had to add to bench/mmv-sweep.sh. The wave20
  session's own arms all set the switch
  explicitly per arm (off / auto / fixed values),
  so its record is unaffected by the default
  move.

## Rollback

```
WGPU_COMPUTE_MMQ_SPLITK=off
```

One variable restores the pre-flip unsplit ladder
exactly: the position the design shipped is now
the rollback instead of the default.

## Gates on this VM

- `cargo test -p wgpu-compute`: 318 passed,
  0 failed, 1 ignored (32 suites; includes
  tests/splitk_mmq.rs, 3 tests on the VM skip
  path, and the three split-K resolver/factor
  pins by name: `mmq_splitk_resolution`,
  `mmq_splitk_value_recognition`,
  `mmq_splitk_factor_rule`). One compile fix on
  the way: `DEFAULT_MMQ_SPLITK` is `pub(crate)`,
  not `pub`, because `MmqSplitK` itself is
  `pub(crate)` (the MMV/K21F constants are `pub`
  only because their types are primitive).
- `cargo test -p dflash`: 24 passed.
- `cargo test -p itak-spec`: 70 passed.
- `cargo test -p itak-kv`: 21 passed.
- `cargo test -p attention`: 26 passed.
- `cargo clippy -p wgpu-compute --all-targets
  -- -D warnings`: clean.
- `cargo fmt --check`: clean.
- `cargo check --workspace`: clean.

No server jobs ran in this lane: the card
evidence for the flip is the wave20 session,
already on record.

## Still owed after merge (card confirmation)

One run at literally unset env on the real file,
in the next card session:

1. `spec_generate_gpu` both-mode with
   `WGPU_COMPUTE_MMQ_SPLITK` unset (and no other
   routing env set): IDENTITY PASS and GOLDEN
   PASS, with `k13_q4k_mmq_split`,
   `k14_q6k_mmq_split`, and `splitk_reduce`
   stamps present at factor 4 in the diag census
   and `"mmq_splitk":"fixed:4"` in the capture
   manifest's `resolved_config`.
2. The combined prefill wall at the new shipped
   defaults together (split-K fixed 4 AND the K24
   coopmat default, when that flip lands): the
   two wins overlap on the Q4_K projections K24
   takes over, so the combined figure is NOT the
   sum of the separate -36% (K24) and -18%
   (split-K) walls, and no run has measured it
   yet.
