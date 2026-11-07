# N-gram self-speculation for itak-spec, 2026-11-07

Lane `lane/ngram-spec`, based on main `0ecdc8ff`. Question: can
the speculative loop draft from the sequence itself, with no
draft model and no training, and what does acceptance look like
where that can be measured on this VM?

## Why

`bench/spec-round-economics-2026-11-06.md` measured the DFlash
draft block forward at about 40.6 ms per round, about 47% of a
depth-1 round, and the drafter is a separate model that must be
loaded, conditioned, and kept in sync. Prompt-lookup (n-gram)
drafting proposes tokens from the sequence itself: find the most
recent prior occurrence of the last N tokens and propose the
tokens that followed it. It costs microseconds on CPU, needs no
training and no second model. The loop's verifier
(`SpecSession::speculative_round` over the target driver)
already guarantees token-identical output for ANY proposer, so
the only question is the acceptance rate.

## Design

New module `crates/itak-spec/src/ngram.rs`:

- `NgramProposer` implements the crate's `Speculator` trait and
  carries an inherent `propose_block(pending, n, arm, rng)` with
  `DflashSpeculator::propose_block`'s exact shape, so the session
  dispatches to either source through one form.
- State is one `Vec<u32>`: the committed token history, appended
  by `commit` under the trait's contract (after a commit, the
  proposer's state equals having processed exactly the committed
  sequence). At propose time the searched sequence is
  `history + [pending]`, indexed virtually; a long history is
  never copied per proposal.
- Lookup: for match length L from the configured maximum down
  to the configured minimum, take the sequence's L-token suffix
  and scan occurrence starts from most recent backwards; the
  first hit returns its continuation, capped by the ask and by
  what actually follows the occurrence. Ranking: longest match
  wins; among occurrences of that match, most recent wins
  (recency is the standard prompt-lookup choice and makes the
  scan a first-hit search). No hit at any length returns an
  empty proposal, and the round degrades to a plain step's
  worth of progress, never a wrong token.
- Sampled arm: distributions are one-hot rows at the proposed
  tokens. That is exact, not an approximation: drawing from a
  one-hot yields its token with probability 1, and
  `sampler::correct_sampled` divides the target row by the
  one-hot and rejects into the normalized residual `(p - q)+`,
  so the emitted distribution is the target's. The greedy arm
  carries no distributions, like the DFlash greedy arm.
- Pure CPU: no GPU or model types. The only geometry the
  proposer knows is the target's vocabulary width, which the
  one-hot rows are built at (the same reason `DflashConfig`
  carries `vocab_size`); session creation validates it against
  the target and refuses mismatches by name.

All limits are `NgramConfig` fields with defaults (build law;
nothing compiled in):

| field | default | meaning |
| --- | --- | --- |
| `max_match_len` | 8 | longest suffix tried first |
| `min_match_len` | 2 | shortest suffix allowed to propose |
| `max_proposals` | 16 | per-proposal cap; the proposer's `max_depth` |
| `search_window_tokens` | 65,536 | how far back an occurrence start may sit (0 = whole history) |
| `max_history_tokens` | 1,048,576 | retained committed history, oldest dropped first (0 = unbounded) |

## The seam change

The trait was not the problem; the session was. `SpecSession`
held `Option<DflashSpeculator>` concretely and called
DFlash-only inherent methods, so a second proposer could not be
attached at all. Smallest change that admits one, all inside
`session.rs`:

- A private `DraftSource` enum (`Dflash(DflashSpeculator)` /
  `Ngram(NgramProposer)`) with four delegating methods
  (`max_depth`, `propose_block`, `note_captures`, `commit`). The
  round loop is otherwise untouched.
- `SpecSession::new` keeps its exact signature and behavior; it
  wraps its draft in the enum and delegates to a shared
  `new_inner`. New constructor `SpecSession::new_ngram(driver,
  Option<NgramProposer>, config, diag)` is the n-gram door.
  Validation: n-gram mode requires the proposer (else
  `DraftNotLoaded`), checks vocabulary agreement
  (`DraftTargetMismatch`), depth in `1..=max_proposals`
  (`DepthOutOfRange`, block size reported as max + 1), and
  `depth + 1 <= max_round_tokens` (`RoundBoundTooSmall`). A
  proposer under mode `dflash` refuses with a named
  `SessionState` mismatch. Capture layers stay empty on the
  n-gram arm: there is nothing to condition.
- One refusal changed in a corner: `SpecSession::new` with mode
  `ngram` and NO draft attached previously returned
  `ModeUnavailable` (mode checked before draft presence); it
  now returns `DraftNotLoaded`, which is the accurate refusal.
  With a DFlash draft attached it still returns
  `ModeUnavailable`, and the existing test
  (`creation_refusals_are_named`) passes unchanged.
- `begin()` gains an n-gram arm: plain `forward(prompt)` (no
  capturing forward exists for this source), then
  `proposer.commit(prompt)`, seeding the lookup history with
  exactly the prompt. DFlash's begin path is byte-identical.

Selection follows the crate's config-first style:
`SpecConfig.mode = SpecMode::Ngram` plus the matching
constructor. `SpecMode::parse` was added as `DraftBackend::parse`'s
twin, and the CPU runner `spec_generate` gained a
`--drafter=<dflash|ngram>` named flag in the GPU runner's
`--draft` style: with `ngram` it never opens the draft file and
runs phase B through `new_ngram`, so the same IDENTITY and
GOLDEN verdicts measure this arm on the real file. The GPU
runner is not wired: the proposer is driver-agnostic and
`new_ngram` works over `ResidentTarget` unchanged, so that
wiring is a flag, not a design question (follow-up below).

## Measured fixture acceptance

Three harnesses, three different claims. All greedy numbers are
under a hard identity assertion (stream equals plain greedy).

**1. Fixture target sessions** (synthetic Glimmer target,
vocab 128, random weights; `ngram::tests::fixture_acceptance_measurement`,
96 generated tokens per row):

| fixture | depth | rounds | proposed | accepted | rate |
| --- | --- | --- | --- | --- | --- |
| repetitive prompt | 1 | 85 | 22 | 10 | 0.4545 |
| repetitive prompt | 4 | 82 | 68 | 13 | 0.1912 |
| repetitive prompt | 6 | 81 | 84 | 14 | 0.1667 |
| prose prompt | 1 | 91 | 14 | 4 | 0.2857 |
| prose prompt | 4 | 91 | 46 | 4 | 0.0870 |
| prose prompt | 6 | 91 | 60 | 4 | 0.0667 |

Reading: accepted count is nearly flat in depth (10/13/14
repetitive, 4/4/4 prose) while proposals grow, so on this
stream extra depth mostly buys rejected tokens. Caveat, stated
plainly: the fixture target's greedy stream is a random-weight
model's stream, not text. These numbers prove the mechanism
end-to-end in the loop (and identity), not real-text
acceptance.

**2. Recorded-stream lockstep** (`examples/ngram_acceptance.rs`:
the session's greedy accounting replayed at the proposer level
over word-tokenized text; prompt 24 tokens):

| stream | depth | proposed | accepted | rate | stream tokens/round |
| --- | --- | --- | --- | --- | --- |
| repetitive template (354 tokens) | 6 | 252 | 252 | 1.0000 | 4.231 |
| prose (143 tokens) | 6 | 6 | 0 | 0.0000 | 1.008 |
| repetitive template | 1 | 147 | 147 | 1.0000 | 1.803 |
| prose | 1 | 1 | 0 | 0.0000 | 1.008 |

Word-level tokenization overstates repetition versus the real
model's subword stream (function words repeat as whole tokens),
so read the rates as directional; the gap is the finding. On
repetitive text the proposer is essentially free throughput
(4.23 tokens per verify round at depth 6); on prose it
proposes almost nothing and the session runs at plain speed
(1.008 tokens/round is the bonus-token floor), never worse.

**3. Stub driver** (`driver::tests::stub_driver_carries_ngram_with_greedy_identity`):
the deterministic successor stream is eventually periodic, so
once an orbit pair repeats, 2-gram continuations are exact.
The test asserts identity over 300 tokens with proposed > 0
and accepted > 0 through the full session/driver seam.

## Honest limits

- Acceptance collapses on non-repetitive text. That is the
  nature of the method, not a defect to tune away: on novel
  prose the proposer is silent and the win is zero (never
  negative; the verifier sees to that). This complements
  DFlash, it does not replace it: DFlash proposes from a
  learned model and accepts on text the lookup has never seen;
  n-gram is free exactly where text repeats itself (code,
  logs, templated and structured output, lists, repeated
  instructions, quoted passages).
- Depth economics differ from DFlash: proposals cost
  microseconds, so depth costs only verify width, but the
  fixture rows show acceptance saturating early. Depth stays
  configuration; the real file decides its operating point.
- The lookup is a naive backward scan, linear in the searched
  window per length tried. At session-scale histories that is
  microseconds; a hash index over suffixes is future work if a
  profile ever names the scan.
- The sampled arm materializes one-hot rows at full vocabulary
  width (at the real vocab and depth 16, about 13 MB per round
  of zeros with depth ones). It is the trait contract's shape
  and the DFlash shim pays the same on its cold path; the
  greedy arm carries no distributions.

## Owed: the real-file measurement

Not run by this lane (VM-only, and the CPU runner is
minutes-to-hours class on the real file):

1. CPU target driver: `spec_generate <glimmer.gguf> <any> 64 6
   "The capital of France is" both plain_stream.txt
   --drafter=ngram` for IDENTITY + GOLDEN plus acceptance from
   the printed spec stats, and the same at depth 1; then a
   repetitive prompt (a repeated instruction block) where the
   proposer should shine.
2. GPU resident driver: wire `--drafter=ngram` into
   `spec_generate_gpu` (a flag over `new_ngram`, no design
   work) and measure acceptance AND round wall at depths 1 and
   6 against the DFlash arms' 22.07 / 16.80 tok/s. The
   interesting question is whether free proposals at prose
   acceptance rates beat a 40.6 ms draft forward at DFlash
   rates, and on repetitive output it should not be close.

## Gates (this lane)

- `cargo test -p itak-spec`: 68 passed, 0 failed (19 of them
  new: 13 proposer unit tests, 4 n-gram session tests, 1 stub
  driver test, and the `SpecMode::parse` round-trip). All
  pre-existing DFlash tests pass unchanged, including
  `creation_refusals_are_named`.
- `cargo test -p dflash`: 24 passed, 0 failed.
- `cargo test -p wgpu-compute`: 268 passed, 1 ignored, 0 failed.
- `cargo clippy -p itak-spec --all-targets -- -D warnings`: clean.
- `cargo fmt --check`: clean.
- `cargo check --workspace`: clean.

Files: `crates/itak-spec/src/ngram.rs` (new),
`src/session.rs` (DraftSource seam, `new_ngram`, begin arm),
`src/config.rs` (`SpecMode::parse`, mode doc), `src/lib.rs`
(module, exports, gap doc), `src/error.rs` (refusal docs),
`src/testkit.rs` (`build_ngram_session`), `src/driver.rs`
(stub test), `examples/spec_generate.rs` (`--drafter`),
`examples/ngram_acceptance.rs` (new harness).
