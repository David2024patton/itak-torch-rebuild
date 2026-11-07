# Prefix-KV store foundation (2026-11-07)

Lane `lane/prefix-kv-store` on main `0ecdc8f`. VM-only work; all
proof is fixture-level. Two commits: the crate, then this note.

## Premise correction, on evidence

The task brief said itak-kv already has versioned save/load of KV
state. It does not: at the base commit there is no `itak-kv` crate
and no KV save/load anywhere in the tree. What exists is the
`attention` crate's in-memory `KvCache`, plus the repo's own plan
naming itak-kv as the future home of exactly this contract:
`docs/specs/SPEC-itak-spec.md` ("itak-kv ... the separate save/load
contract"), `crates/muse-glimmer/src/cache.rs` ("That storage is
the KV crate's job (TODO T-2.4, `itak-kv` in the workspace plan)"),
and TODO item T-2.4. So this lane creates `crates/itak-kv` per the
plan instead of building on a crate that was not there. It depends
only on `attention`.

## What was built

### Snapshot format v1 (`crates/itak-kv/src/snapshot.rs`)

`KvSnapshot`: the KV state of exactly one sequence prefix, in the
versioned layout the format documents in its module header (magic
`ITKV`, format version, model fingerprint string, token count,
geometry, per-layer headers and rows, trailing FNV-1a 64 checksum
over all preceding bytes). Only visible rows are stored: global
layers store every row; windowed layers store the last `window`
rows, tagged with the absolute position of the first stored row.

- `KvSnapshot::capture(&KvCache, fingerprint)` refuses a cache
  whose layers disagree on position (`IncoherentCache`) instead of
  snapshotting a torn state.
- `KvSnapshot::from_bytes` is strict and total: bad magic, unknown
  version, truncation, checksum mismatch, trailing bytes, and
  header/row disagreement each refuse by name. A snapshot is
  handed out only whole.
- `restore()` rebuilds a fresh `KvCache` and returns it only on
  full success; nothing is ever partially applied.
  `restore_checked(geometry, fingerprint)` adds the caller's
  required geometry and model fingerprint as named refusals.
- Known cost, stated plainly: restoring a windowed layer replays
  zero rows for evicted positions ahead of the first stored row,
  because the host cache's append API is strictly in-order and
  position-checked. The zeros land only in slots the stored rows
  overwrite or in slots already outside the window, so restored
  views are exact (proven bitwise by the fixture below). For a
  4k-token Glimmer prefix that is 2,048 evicted positions per
  windowed layer, roughly 160 MB of host-side zero traffic,
  negligible. For a full-context snapshot it is 129,024 positions
  per windowed layer, seconds-scale; a bulk-restore constructor in
  `attention` is the named follow-up if full-context restores ever
  become a hot path.

### PrefixStore (`crates/itak-kv/src/store.rs`)

One file per entry under a configurable root. Entry layout: a
fixed 72-byte preamble (magic `ITKVPENT`, version, created and
last-used unix stamps, prefix hash, token count, payload length),
then the full token-id list, then the snapshot bytes.

- `put(tokens, snapshot, now)`: refuses an empty prefix, a
  token-count mismatch, and an entry that alone exceeds the byte
  cap, each by name. Writes are atomic: temp file in the same
  directory, sync, rename. Re-putting the same token list replaces
  its entry.
- `lookup(query, expected_geometry, expected_fingerprint, now)`:
  longest-prefix match over the index, longest first. A verified
  hit restamps last-used with an 8-byte in-place write (entries
  can be gigabytes; the file is never rewritten on a hit).
- Eviction: after every put, while the store is over its cap, the
  least recently used entry other than the one just written is
  deleted. Stamps are the preamble's last-used field, not file
  mtimes, so they survive copies and clock-safe ordering is the
  caller's clock (tests drive a fake clock).
- Configuration (`crates/itak-kv/src/config.rs`), all following
  the codebase's env conventions: `ITAK_KV_PREFIX_STORE` is the
  master switch in the fail-closed shape of `WGPU_COMPUTE_K21F_HEAD`
  (trimmed exact `"1"` enables, everything else is off);
  `ITAK_KV_PREFIX_DIR` overrides the root (default
  `$HOME/.cache/itak/prefix-kv`); `ITAK_KV_PREFIX_MAX_BYTES` is the
  byte cap with `DEFAULT_MAX_BYTES` = 8 GiB, where a malformed or
  zero value falls back to the default logged by name, because a
  cap is a bound, not a gate.
- The index is built by scanning the root at `open` (with a public
  `rescan()`); files another process writes afterwards are seen
  on the next open or rescan. That matches the session model: a
  serving process opens the store once at startup.

## Keying and trust model

Key: FNV-1a 64 over the token ids as little-endian u32 words, the
same hash family as the diag capture fingerprints. FNV is not
cryptographic, and the design never asks it to be: the hash is an
index, never a proof. A lookup candidate becomes a hit only after
all of these pass, in order:

1. The stored token ids hash back to the entry's key
   (`PrefixHashMismatch` refusal otherwise). This is the
   hash-collision guard: a forged or corrupted entry whose token
   copy was swapped cannot reproduce its key.
2. The stored ids equal the query's prefix token for token. A
   genuine hash collision (different ids, same hash) lands here
   and is simply not a match; the search continues to shorter
   prefixes. Filenames carry a second independent FNV hash as a
   discriminator, and if two different token lists ever collide
   on both, a counter suffix keeps both entries.
3. The payload decodes whole (its own checksum and consistency
   checks) and its token count agrees with the preamble.
4. When the caller requires them, the snapshot's geometry and
   model fingerprint match (`GeometryMismatch`,
   `FingerprintMismatch`).

Any refusal is named in the diag log and falls through to the
next-longest candidate: a corrupt long entry never blocks a
shorter honest one. The store directory itself is trusted-local,
the same trust level as the model files; a keyed hash is a
possible future hardening, not a correctness need, because
verification is token-exact.

## The session seam: specified, not wired

State of the seam in this lane: the store, the format, the
KV-state-level restore, and the fail-closed enable switch are
built and tested. The `ResidentSession` wiring is NOT built; the
task's own fallback applies, for three concrete reasons found in
the tree:

- The resident session holds no host-side KV copy and has no
  readback or install path today; capture is new code on the
  serving path, not a small patch.
- The slot laws differ. The host `KvCache` windowed ring is
  exactly `window` slots, slot = position % window
  (`crates/attention/src/cache.rs`). The resident session's
  windowed ring capacity is `sliding_window + chunk_cap - 1`
  (`crates/wgpu-compute/src/session.rs:237`), slot = position %
  ring_capacity. A naive buffer copy between the two corrupts
  every windowed layer.
- Chunk plans bind the KV buffers (`plans` in the session), so an
  install must go through the reset path that drops them.

The wiring, precisely, for the follow-up lane:

- New method `ResidentSession::install_prefix_snapshot` in
  `crates/wgpu-compute/src/session.rs` (struct at line 4135, `kv`
  field at 4151, `reset()` at 4348 as the shape to follow):
  1. Refuse unless the snapshot geometry matches the session
     (layer count and kinds from `cfg`, row width
     `num_kv_heads * head_dim`); refuse by name, session untouched.
  2. `self.reset()` (fresh `DeviceKv`, plans dropped, position 0).
  3. Per layer, `DeviceKv::ensure_layer`
     (`crates/wgpu-compute/src/kernels/k6_attention.rs:665`) with
     `appended_total = token_count`, then upload the stored rows
     through the context's buffer-write path into `LayerKv.k` /
     `LayerKv.v` (`k6_attention.rs:602`): global layers take rows
     at slots 0..n in one contiguous write; windowed layers take
     each stored row (absolute position p) at slot
     p % ring_capacity, which is at most two contiguous runs.
  4. `self.position = token_count`; return it. The caller then
     forwards only the unmatched suffix.
- The caller (session factory) consults
  `itak_kv::prefix_store_enabled()` and opens the store from
  `StoreConfig::from_env()` only when the switch is on; with the
  switch off, no store is opened and no lookup happens.
- The capture direction (for `put` after a prefill): per layer,
  read back via `GpuContext::read_f32`
  (`crates/wgpu-compute/src/device.rs:300`), reassemble positions
  in order (global slots 0..position; windowed positions
  max(0, position - window)..position at their ring slots),
  append them into a host `KvCache`, then
  `KvSnapshot::capture`. Fingerprint: the GGUF file's sha256.

## Sizing math for Glimmer

Measured geometry: 52 layers (39 windowed at 2,048, 13 global),
2 KV heads, head dim 128. One row is 256 floats = 1,024 bytes, for
K and again for V: 2,048 bytes per layer-position, 106,496 bytes
per token across all layers before any window eviction. (The
often-quoted 1.83 GB full-context resident figure counts 1,024
bytes per layer-position; the snapshot stores K and V at f32,
2,048 bytes per layer-position.)

- A 4,096-token system-prompt prefix: global layers hold 4,096
  rows, windowed layers hold 2,048, payload 272,629,760 bytes
  (about 260 MiB), plus 16 KiB of token ids and the 72-byte
  preamble. At the measured 159.3 tok/s prefill, reusing it skips
  about 25.7 seconds of prefill per request. Reading 260 MiB back
  from local disk is a tenth of a second class cost.
- Past the window, each extra token costs only the 13 global
  layers: 26,624 bytes per token.
- A full 131,072-token context snapshot: 3,653,238,784 bytes
  (about 3.40 GiB) of payload.
- The default 8 GiB cap therefore holds roughly thirty 4k-token
  prefixes, or two full-context snapshots with room to spare.

All four numbers are pinned by the `glimmer_sizing_math` unit
test against the real Glimmer layer pattern, not just asserted
here.

## Proof in this lane (VM, fixture-level)

- `cargo test -p itak-kv`: 21 passed (15 unit, 5 store
  integration, 1 seam fixture). The fixture runs a small model
  shape (4 layers, mixed windowed/global) two ways, cold and
  store-restored, and requires bitwise-identical attention
  outputs, identical greedy streams, and bitwise-identical
  restored views including window eviction positions.
- Store integration: round-trip, longest-prefix selection among
  nested entries, eviction respecting the cap and LRU order under
  a fake clock, a forged entry (token copy rewritten in place)
  refused while a shorter honest prefix still serves, geometry
  and fingerprint refusals at lookup, torn temp and torn final
  files ignored at open, persistence across reopen.
- Regression suites: attention 60 + 5 doc-tests, wgpu-compute
  277, dflash 24, itak-spec 12, all passing.
- `cargo clippy -p itak-kv --all-targets -- -D warnings` clean,
  `cargo fmt --check` clean, `cargo check --workspace` clean.
- Housekeeping note: the lane's first build hit the VM disk quota
  (many stale lane target dirs); the target dirs of seven merged
  lanes were deleted to free space, and the lane built into the
  merge worktree's shared target dir. No sources outside this lane
  were touched.

## The real-file proof owed

1. The session wiring above, as its own lane, gated by
   `ITAK_KV_PREFIX_STORE` default off.
2. On the card, on the real Glimmer GGUF (sha256
   4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e):
   forward a prefix of at least 4,096 tokens, capture it into the
   store under that fingerprint, restart the session, install the
   snapshot, continue with the same suffix, and require the
   generated stream to match the cold run token for token for at
   least 64 tokens, the same identity shape as the speculative
   harness.
3. The measured wall saving on that run against the 25.7-second
   prediction above, and a cross-process persistence check (entry
   written by one process, hit by another).
