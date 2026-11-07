# Prefix-KV session wiring: the ResidentSession seam, slot remapping, bulk restore (2026-11-07)

Lane `lane/prefix-kv-session-wiring` on top of main `0ecdc8f` plus the
cherry-picked prefix-KV store foundation (foundation code and note,
landed on this branch as the two commits directly under this work).
This note records what was wired, the exact slot-remapping rule, the
refusal inventory, the fixture evidence, and the real-file card proof
still owed. No merge, no push from this lane.

## What was wired

`crates/wgpu-compute` now depends on `itak-kv` (path dependency, no
new external crates; `itak-kv` depends only on `attention`, so there
is no cycle). All new session behavior is gated by the foundation's
master switch `ITAK_KV_PREFIX_STORE` (fail-closed: only the exact
string `1` enables). With the switch off, construction is
byte-identical except one added resolved-config pair in the run
manifest (`prefix_store=false`, recorded next to the other switch
resolutions), no store directory is created, and no code path reaches
capture or install.

New surface on `ResidentSession` (all in
`crates/wgpu-compute/src/session.rs`, next to `reset()`):

- `ResidentSession::prefix_store_open() -> Option<Result<PrefixStore, KvError>>`
  — `None` when the switch is off (the store is never constructed, so
  its directory is never created); `Some` carries the foundation's
  env-resolved store otherwise.
- `capture_prefix_snapshot(&mut self, model_fingerprint) -> Result<KvSnapshot>`
  — reads the device KV back into a snapshot. Per layer it computes
  the visible range (below), reads each slot run out of the layer's
  device K and V buffers with ranged readback, assembles a snapshot
  whose geometry is derived from the session config, and validates it
  before returning. The fingerprint is the caller's (the GGUF sha256
  at the call site; the session does not hash files).
- `install_prefix_snapshot(&mut self, snapshot, expected_fingerprint) -> Result<usize>`
  — validates everything first (see refusals), then `reset()`, then
  uploads each layer's rows into freshly ensured device KV buffers at
  remapped slots, sets `position` to the snapshot token count, and
  returns the matched length. The next `forward` continues from the
  first unmatched token.
- `restore_prefix_from_store(&mut self, store, tokens, fingerprint) -> Result<Option<usize>>`
  — the session-start flow: longest-prefix lookup, geometry and
  fingerprint re-verified at lookup, snapshot re-validated by the
  install validator, install on a hit. A miss, a store read failure,
  or a hit that fails validation is logged with its reason and
  reported as `Ok(None)` so the caller falls back to cold prefill; a
  device failure during install is the one hard error.
- `capture_and_store_prefix(&mut self, store, tokens, fingerprint)`
  — capture after prefill and `PrefixStore::put`; refuses unless the
  session position is exactly `tokens.len()`.
- `forward_with_prefix_store(&mut self, store, tokens, fingerprint) -> Result<Vec<f32>>`
  — the one-call flow: switch off or session not fresh means plain
  `forward`. On a partial hit it forwards only the unmatched suffix.
  On a full hit the snapshot carries no logits, so it installs in
  full, rewinds one token, and re-forwards the last token so the
  returned logits are the restore's own (the same values the cold
  path returns for that token). After a miss it captures and stores
  the completed prefill; store failures are logged, never fatal to
  the forward.
- Reporting: `SessionStats` gains `prefix_restores`,
  `prefix_restored_tokens`, `prefix_captures`; `prefix_match()`
  exposes the installed length for the session; under diag capture,
  install stamps a `prefix_restored_tokens` pair into the run
  manifest's session section (new capture-gated
  `DiagAccumulator::note_session_pair`).

## The slot-remapping rule, precisely

A snapshot stores, per layer, the visible rows in absolute-position
order, tagged with `first_position`:

- global layer: every row, positions `0..token_count`,
  `first_position = 0`;
- windowed layer with attention window `w`: the last
  `min(token_count, w)` rows, positions
  `token_count - rows .. token_count`.

The snapshot geometry maps sliding layers to the config's
`sliding_window` (the attention-visible window), NOT the device ring
capacity; the ring is a device storage detail and is handled only by
the remap below.

The device ring for a windowed layer has capacity
`ring = sliding_window + chunk_cap - 1` and stores position `p` at
slot `p % ring` (the same rule `forward_chunk` writes with). A global
layer stores position `p` at slot `p` (position addressing, geometric
growth). Capture and install both walk the visible range as slot
runs: maximal runs of consecutive slots starting at
`first_position % ring` (windowed) or `first_position` (global). A
windowed range wraps the ring at most once, so it splits into at most
two runs; the snapshot rows stay in absolute-position order, and each
run maps a contiguous slice of rows to a contiguous slot range.
Capture reads each run with ranged device readback (new
`GpuContext::read_f32_at`, byte offset `slot * row_floats * 4`);
install writes each run with one queue `write_buffer` at the same
offset. At the fixture geometry (window 4, chunk cap 64, ring 67), a
70-token prefix stores windowed positions 66..70 at slots 66, 0, 1, 2
— two runs, `{row 0, slot 66, count 1}` and `{row 1, slot 0, count 3}`
— pinned by a unit test, plus a sweep over chunk caps 1/4/32/64 and
positions 0..300 asserting at most two runs, exact row coverage, and
slot order equal to `position % ring`.

## Refusal inventory

Install validates EVERYTHING before `reset()`: any refusal leaves the
session completely untouched (position, KV, stats, match record), a
stronger guarantee than the brief's clean-reset-state wording, and
the fixture test asserts byte-level continuation equality after three
refusals on a live session. All refusals are `GlimmerError::StepFailed`
(step `attention`) naming the cause:

- switch off: capture, install, and capture-and-store refuse naming
  `ITAK_KV_PREFIX_STORE`; `prefix_store_open` returns `None`; the
  restore and one-call flows degrade to cold prefill.
- capture at position 0 (nothing to capture).
- snapshot internally inconsistent (`KvSnapshot::validate`).
- geometry mismatch: snapshot layer kinds, per-layer windows, KV head
  count, or head dim differ from the session config.
- fingerprint mismatch: snapshot fingerprint differs from the
  expected model fingerprint.
- empty prefix: token count 0.
- oversized prefix: token count beyond the config context length
  ("position overflow past context capacity").

Restore-flow fallbacks that return `Ok(None)` instead of erroring
(logged with the reason): store lookup error, lookup miss, empty hit,
hit failing the install validator. Capture-and-store refuses when the
session position is not exactly the prompt length.

## The named follow-up: bulk restore in attention (landed)

`KvCache::restore_bulk(layer, first_position, keys, values)` in
`crates/attention` installs a snapshot layer's rows directly: global
layers extend storage in one copy per side; windowed layers write
each row straight to ring slot `position % window`. No per-position
append replay, no zero rows: the replay it replaces writes the full
host KV (the foundation note's ~160 MB at a 4k-token Glimmer prefix;
seconds of pure memcpy at full context). `KvSnapshot::restore` now
uses it. Semantics are pinned bitwise-equal to append replay,
including continuation appends after restore. It is deliberately
fresh-layer-only: a new refusal `AttentionError::BulkRestoreNotFresh`
fires on a layer that already holds positions, because snapshot
windowed rows fill every ring slot whenever `first_position > 0`, so
bulk state is provably the replay state only from fresh. Other named
refusals: K/V length disagreement, row-multiple violation, too many
rows for the window, a global layer with nonzero `first_position`.

## Session support change: ranged readback

`GpuContext::read_f32_at(buffer, first_float, count)` copies a float
range out of a device buffer (zero count submits nothing);
`read_f32` delegates to it. Capture is the first caller; nothing else
changed behavior.

## Fixture evidence (this VM)

This VM enumerates no GPU adapter, so device-executing tests skip by
the crate's standing pattern; everything host-side ran for real:

- `cargo test -p attention`: 26 passed (7 lib + 11 cache + 8 kernel),
  including the two new bulk-restore tests (bitwise replay equality,
  named refusals).
- `cargo test -p itak-kv`: 21 passed (15 unit + 5 store integration +
  1 restore fixture); the foundation's restore fixture passing against
  the new bulk restore is the bitwise proof the swap is transparent.
- `cargo test -p wgpu-compute`: 282 passed, 0 failed, 1 ignored
  across 29 binaries. Lib 168 + 1 ignored, including 4 new seam unit
  tests (visible-range rule, slot-run remap including the wrap case
  and the cap sweep, config-derived geometry, validator accept plus
  five named refusals). All pre-existing session suites pass
  unchanged (resident_session 7, diag_capture 3, dflash_session 7),
  which is the switch-off no-behavior-change evidence available on
  this VM.
- `cargo test -p dflash`: 24 passed. `cargo test -p itak-spec`: 49
  passed (regression sweep over the crates that consume the session).
- New integration test `crates/wgpu-compute/tests/prefix_kv_session.rs`
  compiles clean and passes via the skip path here; its assertions
  are written for the card (below).
- `cargo clippy -p attention -p itak-kv -p wgpu-compute
  --all-targets -- -D warnings`: clean. `cargo fmt --check`: clean.
  `cargo check --workspace`: clean.

The integration test's card-side assertions, in one sequential test
(the master switch is process env): a 70-token prompt over a 6-layer
fixture with window 4 (window smaller than the prefix, ring 67):

- switch off: `prefix_store_open()` is `None`, the store root is
  never created, capture and install refuse naming the switch, the
  session is untouched;
- switch on: cold prefill, capture+store writes one entry; a lookup
  with a longer query hits at 70; stored windowed layers carry
  `first_position 66` and 4 rows, global layers 70 rows;
- a fresh session restored from the store reports position 70,
  `prefix_match` 70, stats 1 restore / 70 tokens, and its next 40
  teacher-forced steps match the cold session's greedy picks exactly
  with worst logit drift under the resident suite's 0.5 bound (40
  tokens past the restore, above the 32 required);
- the one-call flow: first session misses and stores, second session
  full-hits and its returned logits match the cold prompt logits;
- refusals on a live session at position 3 (foreign geometry, wrong
  fingerprint, 300-token oversized snapshot against a 256 context)
  each refuse by name and leave position and match state untouched,
  and the session's next forward still matches a reference session.

## Still owed: the real-file card proof

None of the device assertions above have executed yet; this VM has
no adapter and no server GPU jobs were run for this lane. The owed
proof, on the 7900 XTX with the real file
(`/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`,
sha256 `4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e`,
the fingerprint the store entries must carry):

1. run `prefix_kv_session` on the card so the fixture round trip,
   refusal, and switch-off assertions execute for real;
2. real-file round trip: a prompt with a prefix of at least 4,096
   tokens, cold session vs store-restored session, greedy stream
   identity for at least 64 tokens past the restore, plus a
   cross-process check (store written by one process, restored by
   another);
3. the wall measurement: cold prefill wall for that prefix vs
   capture + install + suffix wall, against the foundation note's
   prediction of about 25.7 s saved at 4k tokens (prefix bytes
   272,629,760; prefill 159.3 tok/s), reported as measured, not
   extrapolated.

Until those run, the honest status is: the seam is wired, validated,
and host-proven; device behavior is asserted by tests that have only
compiled and skipped so far.
