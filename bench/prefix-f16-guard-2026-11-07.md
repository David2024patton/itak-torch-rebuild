# Prefix-KV x f16 guard: the seam refuses f16 KV storage by name (2026-11-07)

Lane `lane/prefix-f16-guard`, from main 85beb8a.
The prefix-KV snapshot seam and the f16 KV storage
switch no longer compose silently. A session whose
KV storage is packed f16 now refuses prefix capture
and prefix install by name, before any byte moves
and before any session state is touched. This is
the cold critic panel's Category A HIGH #3 fix
(bench record: ~/workspace/reports/critic-engine-2026-11-07/,
CRITIC-ENGINE-MERGE.md Category A; seat-1-correctness.md
carries the full evidence).

## The finding

`capture_prefix_snapshot` read each layer's device
KV with `GpuContext::read_f32_at` (4-byte elements
hard-coded) and `install_prefix_snapshot` wrote
f32 little-endian bytes at `(slot * row * 4)`
offsets. A session built with
`WGPU_COMPUTE_KV_F16=1` stores its KV as
`KvStorage::F16`: packed, two values per u32 word,
half the bytes. Neither method looked at the
layer's storage format, and no test combined the
two switches (`tests/kv_f16.rs` never touches the
prefix methods; `tests/prefix_kv_session.rs` never
set the f16 switch).

With both switches on:

- Capture reinterpreted the packed f16 words as
  f32 at double offsets and produced a garbage
  snapshot that still passed the store's FNV
  checksum (the checksum covers the garbage
  faithfully). The store then served that snapshot
  to later sessions, including f32 ones: persistent,
  cross-run KV poisoning.
- Install wrote twice the bytes at f32 offsets
  into f16-sized buffers: a wgpu validation failure
  or silent corruption of the first half of the ring.

The forward path was never wrong: the attention
plan geometry carries `kv_f16` per layer and the
flash forms have f16 twins. Only the prefix seam
ignored the format.

## The fix: refusal, not conversion

The panel prescribed "refuse by name in both
methods when layer storage is F16 (or convert)".
Refusal is the fix, and it is the honest one:

- The snapshot format (itak-kv format v1) carries
  no storage-format field; its rows are f32 by
  definition. A converting capture would have to
  widen f16 to f32 (lossy values presented as an
  exact snapshot, breaking the seam's bitwise
  restore contract) and a converting install would
  narrow f32 to f16 (silently discarding precision
  the snapshot claims to carry). Either direction
  makes the store hold snapshots whose fidelity
  depends on which session wrote them, with no
  field saying so.
- The two features serve different runs: f16 KV is
  a memory-halving opt-in for long contexts; the
  prefix store is a prefill-saving opt-in. Nothing
  measured says a run needs both at once, and the
  combination can be revisited properly if one ever
  does (a format field in snapshot v2, or a
  converting seam with its own identity bar).

The guard is one pure predicate in
`crates/wgpu-compute/src/session.rs`,
`prefix_storage_refusal(step, storage) -> Option<GlimmerError>`:
`None` for `KvStorage::F32`, and for
`KvStorage::F16` the session's existing error type
(`GlimmerError::StepFailed` under the caller's own
step name) with this detail text:

> the session's KV storage is f16 (WGPU_COMPUTE_KV_F16=1): prefix snapshots carry f32 rows and the prefix-KV seam does not convert between formats, so capturing or installing under f16 storage would corrupt the KV in both directions; run with WGPU_COMPUTE_KV_F16 unset or off to use the prefix store

The predicate reads the session's one uniform
`DeviceKv` storage format (`DeviceKv::storage()`);
a `DeviceKv` is built with a single format for
every layer and growth preserves it, so there is
no per-layer case to miss. (`DflashSession` has no
prefix methods; the seam is `ResidentSession`-only.)

Where the guard sits:

- `capture_prefix_snapshot`: after the switch
  check, before the position check and before any
  readback. Step name `prefix_capture`.
- `install_prefix_snapshot`: after the switch
  check, before snapshot validation and therefore
  before the reset that is the method's first
  mutation. Step name `prefix_install`.
- `restore_prefix_from_store`: checks the same
  predicate after its position check and keeps its
  documented fallback contract: it returns `Ok(None)`
  with the refusal named on the diag channel
  ("prefix restore skipped: ...; cold prefill
  serves") instead of letting the install refusal
  surface as a hard error, which that method
  reserves for device failures.
- `capture_and_store_prefix` and
  `forward_with_prefix_store` need no code of their
  own: both reach the seam only through the two
  guarded methods, so the store flow refuses under
  `prefix_capture` (and `forward_with_prefix_store`
  already treats a post-forward capture failure as
  a named diag event, never a failed forward).

Every refusal leaves the session byte-identical:
the checks run before the readback loop, before
validation, and before the install reset, matching
the seam's existing validate-before-mutation
discipline.

## Tests

Device-free (runs everywhere, including this VM):

- `session::tests::prefix_storage_refusal_gates_the_f16_composition`
  (wgpu-compute lib tests): f32 admits the seam
  under all three step names; f16 refuses under
  each, as a `StepFailed` carrying the caller's
  step, with the cause (f16), the selecting switch
  (`WGPU_COMPUTE_KV_F16`), and the remedy (unset or
  off) all in the detail text.

Device (skips on adapterless hosts; the executing
leg is card-owed, see below):

- `resident_session_prefix_seam_refuses_f16_storage`
  in `crates/wgpu-compute/tests/prefix_kv_session.rs`:
  both switches on (prefix store on, f16 backend),
  two f16 sessions forwarded over the same 12-token
  prompt, one guarded and one an untouched twin.
  Asserts: capture refuses by name under
  `prefix_capture` with position and capture counter
  untouched; `capture_and_store_prefix` refuses and
  the store stays empty (no poisoned snapshot is
  ever written); install of a valid fixture
  snapshot refuses by name under `prefix_install`
  with no reset (position, prefix match, and
  restore counters all unchanged); with that
  snapshot placed in the store directly,
  `restore_prefix_from_store` returns `None`
  rather than erroring; and the guarded session's
  next forward equals the twin's logits exactly,
  proving the refusals left the session
  byte-identical. Env choreography follows the
  file's existing discipline (the f16 switch is set
  only around the f16 backend's construction and
  removed on the way out).

## Gates, as observed on this VM

- `cargo test -p wgpu-compute`: 197 lib passed
  (1 ignored), every integration file green,
  including `prefix_kv_session` 2 passed (the new
  test takes the skip path here) and `kv_f16`
  3 passed.
- `cargo test -p itak-kv`: 21 passed.
  `cargo test -p attention`: 26 passed.
  `cargo test -p dflash`: 24 passed.
  `cargo test -p itak-spec`: 70 passed.
- `cargo clippy -p wgpu-compute --all-targets -- -D warnings`:
  clean. `cargo check --workspace`: clean.
  `cargo fmt --check`: clean.

## Card-owed leg

This VM enumerates no GPU adapter, so the device
test above compiled and took its skip path here;
its assertions have not executed against a real
f16 session yet. The owed run, on the RX 7900 XTX
host, from the merged tree:

```
cargo test -p wgpu-compute --test prefix_kv_session resident_session_prefix_seam_refuses_f16_storage
```

Expected: the test runs (no skip line) and passes,
which is the first execution of an f16 session
actually meeting the guard. Until that run lands,
the composition is proven at the predicate level
only, and this note says so plainly.
