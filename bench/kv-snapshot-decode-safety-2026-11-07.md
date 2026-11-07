# KV snapshot decode safety: the Category A crafted-file pair, plus the store preamble message (2026-11-07)

Lane `lane/kv-snapshot-decode-safety`, from main
85beb8ab6ae8accf8aaf7cee916e08b9f455d919. Code and tests in
3e146c0cf88638fb4dee1aa3c79b1957da13b7ab; this note is the
second commit. VM gates only on this lane; no server jobs.

Source of the work: the cold critic panel of 2026-11-07
(`CRITIC-ENGINE-MERGE.md`, Category A; probes executed by the
security seat and the failure seat). The snapshot format's own
docs claim decoding is "strict and total" and the crate's
contract (lib.rs) is "named errors instead of panics on
anything read from disk". Three crafted inputs broke that
contract in the decode and store-open paths. Each fix below
got a regression test written first, proven failing on the
unfixed tree, then proven passing after, in both build
profiles where the defect was profile-dependent.

## Finding 1 [HIGH]: decode pre-allocates from a file-controlled layer count

`KvSnapshot::from_bytes` read `layer_count` (a raw u32) from
the buffer and immediately sized two `Vec::with_capacity`
calls with it, before the checksum, geometry, or any other
trust check could refuse the file.

Crafted input (security seat probe, re-run as the regression
test): a 32-byte snapshot header, magic `ITKV`, version 1,
fingerprint length 0, token_count 0, heads 1, dim 1,
layer_count = u32::MAX.

Before: the decode demanded a 68,719,476,720-byte allocation
and the process died with SIGABRT ("memory allocation of
68719476720 bytes failed", signal 6). An allocation abort is
not a catchable error, and the store decodes entries during
lookup, so one planted file in a store root killed the
serving process.

Fix: bound the count before it sizes anything. Every layer
header costs 28 bytes on the wire (new constant
`LAYER_HEADER_LEN`: tag 1 + reserved 3 + window 8 +
first_position 8 + row_count 8), so a count larger than
`reader.remaining() / LAYER_HEADER_LEN` is impossible and is
refused by name with `KvError::InconsistentSnapshot`. This is
the same "declared length larger than the rest of the file
is impossible" argument the GGUF parser already uses for
metadata arrays (crates/gguf/src/file.rs). After the bound,
the two capacities are proportional to the actual buffer
size.

Regression test: `decode_refuses_a_layer_count_that_cannot_fit_the_buffer`
(snapshot.rs tests). Before: test binary aborted (SIGABRT).
After: named refusal, test passes in debug and release.

## Finding 2 [HIGH]: validate() multiplies file-controlled sizes unchecked

`KvSnapshot::validate()` compared stored lengths against
`expected_rows * row` with unchecked arithmetic, and
`KvGeometry::row_floats()` multiplied heads by dim unchecked.

Crafted input (failure seat probe, re-run as the regression
test): a hand-built snapshot with a valid FNV-1a checksum,
fingerprint "x", token_count = 2^32, num_kv_heads = 65536,
head_dim = 65536 (so the row width is 2^32), one global layer
with row_count 0 and zero payload rows.

Before: release builds ACCEPTED the snapshot
(`Ok(KvSnapshot { token_count: 4294967296, ... })`) because
2^32 * 2^32 wraps to 0, exactly the stored length; debug
builds PANICKED at snapshot.rs:188, "attempt to multiply
with overflow". The two profiles disagreed about what is a
valid snapshot, and any consumer trusting a validated
`token_count` acted on a forged count.

Fix: checked products everywhere the width meets stored
data.

- New `KvGeometry::row_floats_checked()`: `checked_mul` on
  heads x dim, refusing with `KvError::InconsistentSnapshot`
  on overflow. The decoder and the validator both take the
  row width from here. The plain `row_floats()` stays for
  geometry built from a live cache, whose fields the cache
  itself already bounded; every external caller of it in the
  workspace is a test fixture over hand-built sane geometry.
- `validate()` computes `expected_rows.checked_mul(row)` and
  refuses by name on overflow instead of comparing against a
  wrapped product.

One correction to the probe's framing, found while writing
the regression test: two u32 wire fields cannot overflow a
64-bit usize product ((2^32-1)^2 < 2^64), so the row-width
half of this hole is reachable through the wire only as the
expected-rows product above. It IS reachable through
`validate()` on an in-memory snapshot, whose geometry fields
are public usizes: a second regression test builds one with
heads = dim = 2^32 directly. On the unfixed tree that test
panicked in debug at the `row_floats()` multiply
(snapshot.rs:66) and returned Ok(()) in release; after the
fix both profiles refuse by name.

Regression tests: `decode_refuses_geometry_whose_row_products_overflow`
(the wire probe) and `validate_refuses_geometry_whose_row_width_overflows`
(the in-memory geometry). Both fail before (panic in debug,
acceptance in release) and pass after, in both profiles.

## Finding 3 [MEDIUM]: the store preamble's own refusal message multiplies unchecked

`parse_preamble` (store.rs) checked `tokens_bytes` against
`(token_count as u64).saturating_mul(4)` but then formatted
its refusal message with the raw `token_count * 4`.
token_count comes straight from file bytes at offset 40.
`rescan` runs `parse_preamble` over every `.kv` file in the
store root at open.

Crafted input (failure seat, re-run as the regression test):
a planted 72-byte file (exactly one preamble) with the entry
magic, version 1, token_count = u64::MAX, tokens_bytes = 0.

Before: debug builds panicked at store.rs:596, "attempt to
multiply with overflow", during store open; release builds
wrapped and printed a wrong number in a message nobody could
act on.

Fix: bind the saturating product once
(`expected_tokens_bytes`) and use it in both the comparison
and the message, so the check and its refusal can never
disagree and neither can overflow.

Regression test: `open_refuses_a_planted_garbage_entry_by_name`
(store.rs tests): plants the file, opens the store, asserts
the open succeeds with the entry refused by name in the diag
log and left unindexed (zero entries). Before: debug panic
at open. After: passes in debug and release.

## Sibling arithmetic audit (snapshot.rs, whole file)

Every other multiplication in snapshot.rs that can touch a
file-controlled value, examined one by one:

- `from_bytes` payload sizing (`row_count * row`): already
  `checked_mul`, refuses by name. Unchanged.
- `Reader::f32s` (`count * 4`): already `checked_mul`.
  Unchanged.
- `LayerSnapshot::rows`: already `checked_div`. Unchanged.
- `KvGeometry::bytes_per_token` and `stored_bytes_for`:
  unchecked u64 products and a u64 sum over geometry fields.
  These are sizing estimates (the store's byte accounting and
  capture's diag line), and a foreign geometry could wrap or
  panic them even though no decode path trusts them for
  correctness. Made saturating throughout
  (`saturating_mul`, `saturating_add`), the same tool
  `parse_preamble` already used: identical values for every
  real geometry (the Glimmer sizing pins in
  `glimmer_sizing_math` still pass exactly), saturation
  instead of a wrap or panic for a foreign one. This is the
  only behavior-adjacent change beyond the three findings,
  and it cannot alter any value a real cache geometry
  produces.
- `capture()`'s `Vec::with_capacity(visible * row)`: both
  factors come from a live `KvCache` built by the attention
  crate, not from file bytes. Not file-controlled; unchanged.
- `Reader::take`'s `at + n`: an addition over a cursor bounded
  by the buffer length plus one u32-derived length; cannot
  overflow on a 64-bit target. Unchanged.
- store.rs neighbours of finding 3: the failure seat already
  checked `verify_candidate`'s capacity and the `tokens_end`
  computation and found them gated by the candidate filter
  (`token_count <= query.len()`); this lane changed only the
  named message line there, per the panel's own fix text.

No other unchecked multiplication on a file-controlled value
remains in snapshot.rs.

## Gates (lane tree, as observed)

- `cargo test -p itak-kv`: 25 passed, 0 failed (21 base + 4
  new), debug AND `--release` (finding 2 is profile-dependent;
  the regressions pass in both).
- `cargo test -p wgpu-compute`: 318 passed, 1 ignored,
  0 failed (the prefix-KV session suite consumes this crate).
- `cargo test -p attention`: 26 passed. `-p dflash`: 24
  passed. `-p itak-spec`: 70 passed.
- `cargo clippy -p itak-kv --all-targets -D warnings`: clean.
  `cargo clippy -p wgpu-compute --all-targets -D warnings`:
  clean.
- `cargo fmt --check`: clean. `cargo check --workspace`:
  clean.

Not covered by this lane, by design: the remaining Category
A items (the f16/prefix composition guard in
wgpu-compute/session.rs, the midway-failed install, the
unbounded candidate read, temp-file collisions, the 100%-hit
recapture) are separate lanes. This lane closes only the
decode-safety pair and the store preamble message.
