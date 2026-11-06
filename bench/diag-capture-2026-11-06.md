# Diag capture level (2026-11-06)

Lane: `lane/diag-capture`, from main `2f8b3ae`. Status: built and
VM-verified (full wgpu-compute suite green, clippy clean). No card
runs in this lane; the card legs are checklist items at the end.

## What capture adds

`WGPU_COMPUTE_DIAG=timing` answers "how long did each dispatch
take". It cannot answer "what did the run compute": a run can be
timed perfectly and still be wrong, and two runs can only be
compared by eyeballing outputs. `WGPU_COMPUTE_DIAG=capture` is the
third mode of the same diag system (timestamps.rs). It implies
timing everywhere: the device requests the timestamp features,
the recorder is built under the same feature gate, sessions drain
and report exactly as timing runs do, and the timing tally and
summary CSV are byte-identical. Any value other than `timing` or
`capture` still fails closed to off, and the off path is
unchanged (the existing off-path pins pass; their guards now skip
when the mode is anything but off, so they stay correct under a
capture environment too).

On top of timing, capture adds three things:

1. **A run manifest**, begun when a resident session is built,
   printed on the diag channel (`[wgpu-compute-diag] manifest:
   {...}`) and written at finish to the sidecar
   `<WGPU_COMPUTE_DIAG_OUT>.manifest.json`. It states: the diag
   mode; the build version (`CARGO_PKG_VERSION`) and the git
   revision when the build exported `ITAK_GIT_HASH`
   (`"unknown"` otherwise, never guessed); the adapter name,
   backend, device type, and the driver and driver-info strings
   the adapter itself reports; every resolved `WGPU_COMPUTE_*`
   configuration value the run actually runs with (small-batch
   cap, MMV ceiling, the full MMV band table in its described
   form, chunked-GEMV cap, the K22 switch, the wide-MMQ floor:
   resolved values, not raw env text); the model file (path,
   size, content fingerprint, method) when the driver noted it
   (`ResidentSession::note_model_file`) or registered it once
   per process (`register_model_file`); and the session shape
   (kind, layers, hidden, vocab, context length, sliding window,
   key/value lengths, KV ring capacity, chunk rows cap).

2. **Data fingerprints** of what each completed forward
   computed, recorded into the session's accumulator and written
   at finish to `<WGPU_COMPUTE_DIAG_OUT>.capture.csv`, one line
   per record on the diag channel as they happen
   (`[wgpu-compute-diag] capture #<seq> ...`). Kinds:
   - `forward`: a plain forward's (or capturing forward's)
     finished logits: fingerprint plus the greedy argmax token.
   - `block`: a block forward's finished logits, all rows (the
     speculative verify forward's shape); no single argmax.
   - `argmax`: a device-argmax forward's winner; the
     fingerprint covers the (id, logit) pair, the only output
     that forward reads back.
   - `round`: one speculative round: the draft's proposed ids
     (fingerprinted) plus the accepted count, recorded through
     `ResidentSession::note_spec_round` (wiring step below).
   The hash runs over the vector the forward already read back.
   Capture adds no GPU work and no readback.

3. **A mechanical diff**: `examples/diag_diff.rs` takes two
   `WGPU_COMPUTE_DIAG_OUT` prefixes, compares the manifests
   field by field, then the capture records position by
   position, and names the first record whose kind, value
   count, fingerprint, argmax, or accepted count differs (or
   that one run lacks). Exit 0 identical, 1 any difference, 2
   usage/IO errors. Verified end to end on fixtures: a one-nibble
   fingerprint change at seq 1 and a driver-string drift were
   each named exactly.

Capture records are session-lifetime, like the timing groups:
they are never windowed or dropped mid-run (a long run's record
list grows by one small record per forward), and the finish
writes the whole list. The accumulator API is mode-gated: every
capture entry point is a no-op (returning `None` where it
returns a record) below capture mode, so timing and off runs
pay one mode check per forward and nothing else.

## Exact formats

**Fingerprint** (`Fnv1a64` in timestamps.rs): word-wise FNV-1a
64 over the byte stream `count_u64le || payload`, where the
payload is each `f32`'s `to_bits` little-endian bytes (or each
`u32`'s little-endian bytes). The stream folds eight bytes at a
time (XOR the little-endian word into the state, multiply by
the FNV prime `0x100000001b3`, offset basis
`0xcbf29ce484222325`); a final partial word folds byte by byte.
The result depends only on the byte sequence, never on feed
chunking (pinned by a unit test). Rendered as 16 lowercase hex
digits. Golden values, computed independently of the
implementation and pinned in the test suite:

- `fingerprint_f32(&[])` = `af63bd4c8601b7df`
- `fingerprint_f32(&[1.0])` = `d9c063aea94e0421`
- `fingerprint_u32(&[201818])` = `c3bd5d17f5820309`

**Model file fingerprint**: method string
`fnv1a64w(size_u64le || whole_file_if_size<=2MiB_else_first_1MiB || last_1MiB)`,
recorded in the manifest next to the value. At most 2 MiB is
read however large the file is; a byte in the unsampled middle
of a large file cannot move the fingerprint, and a unit test
pins exactly that so the method is never mistaken for a
whole-file hash.

**Capture CSV**: header `seq,kind,values,hash,argmax,accepted`,
one row per record in run order, empty fields for data a kind
does not carry.

**Manifest JSON**: one compact document, fixed field order:
`format` (`wgpu-compute-diag-manifest`), `format_version` (1),
`diag_mode`, `build {version, git}`, `adapter {name, backend,
device_type, driver, driver_info}`, `resolved_config` (object of
name to resolved value), `model` (object `{path, size_bytes,
fingerprint, fingerprint_method}` or `null`), `session` (object
of name to value).

## Overhead (measured on this VM, release build)

The fingerprint cost was measured before and after one
optimization, and the first form was rejected on its numbers:

- Byte-wise FNV-1a (first version): 9.56 ms per forward at
  Glimmer's vocab (202,048 f32 values), 138 ms per 16-row
  block. Too slow: the block form alone would have taxed every
  speculative round.
- Word-wise (shipped): **1.60 ms** per single-row forward
  (505 MB/s of f32 data on this VM's CPU), **29.2 ms** per
  16-row block fingerprint, negligible for an argmax record
  (16 bytes hashed). The measurement test is in the suite
  and ignored by default: `cargo test --release
  -p wgpu-compute --lib capture_hash_cost_measurement
  -- --ignored --nocapture`.

Against the card's current timings (decode step about 52 ms at
19.26 tok/s), a 1.6 ms host hash per decode forward is about
3 percent, paid after the readback on the host. A speculative
round's verify block (17 rows, about 31 ms of hashing at the
VM's rate) is the heaviest leg and is material relative to a
round's GPU time; capture is a diagnostic mode, never on for
verdict runs, and the server CPU will differ from this VM's.
The end-to-end capture tax on the card (tok/s with capture on
versus timing only) is **unmeasured**; it is a checklist item
below, not a claim. The model fingerprint reads at most 2 MiB
once per noted file. The manifest is one line at session start
and two small file writes at finish.

## Integration steps left for other lanes (named, not done here)

This lane's scope was timestamps.rs, the accumulator, and the
resident session's hooks. Two wirings remain, both one-call
sized:

1. **DflashSession mirror hooks** (dflash lane): its
   accumulator already comes from `DiagAccumulator::from_env()`
   (dflash_session.rs:1006), so under `DIAG=capture` it is
   capture-mode today and its timing side already works; what
   it lacks is (a) a `begin_manifest` call in its constructor
   with its own session pairs (kind `dflash`, draft depth,
   block size, target shape), (b) one
   `acc.record_forward(KIND_BLOCK, &logits)` at the
   `forward_block` exit (dflash_session.rs:1698), and (c)
   `note_model_file` / `note_spec_round` / capture-record
   passthroughs mirroring ResidentSession's.
2. **Spec round wiring** (spec lane): call
   `ResidentSession::note_spec_round(&draft_ids, accepted)` on
   the target session once per completed round, at the point
   inside the round where the draft's proposed ids and the
   accepted count are both in hand. `RoundOutcome` carries only
   the proposed count, not the ids, so the call cannot be made
   from the outcome after the fact.

## Additions to the next card session's checklist

- Run `glimmer_generate` and one speculative phase B under
  `WGPU_COMPUTE_DIAG=capture` with `WGPU_COMPUTE_DIAG_OUT` set;
  confirm each run leaves `<out>.manifest.json` and
  `<out>.capture.csv` beside the timing CSV, and the manifest
  states the real adapter driver string and the resolved band
  table of that run.
- Diff two same-build, same-config capture runs with
  `diag_diff`: expected result is zero divergence, exit 0. Any
  divergence is a real nondeterminism find, not a tool artifact.
- Diff one cross-config pair (different MMV band routing):
  expected divergence localized to verify forwards once the
  routing change first bites, with all earlier records agreeing.
- Measure the capture tax: decode tok/s and one spec phase
  under capture versus timing-only in the same session, and
  record both numbers in that session's note (this lane's VM
  numbers are host-only and do not substitute).
- Pipeline statistics queries (open evaluation from the
  2026-11-04 visibility research): from the adapter at runtime,
  state whether `wgpu::Features::PIPELINE_STATISTICS_QUERY` is
  offered on gfx1100 under RADV in the same session, and if it
  is, whether a session can be built that reads per-pipeline
  counters without disturbing the verdict path.
- RGP trace (same open evaluation; the trace lane owns the
  tooling): record whether an RGP trace (`MESA_VK_TRACE=rgp`)
  was captured in the same session as the timing/capture runs,
  and if so the trace file's name beside the session's other
  artifacts, so a kernel-level question can be answered from
  the same run that produced the fingerprints.

## Completion: the two wirings above are closed (lane/diag-capture-wiring)

Both items from "Integration steps left for other lanes" are
now wired, in one code commit on top of this note's main.

**1. DflashSession mirror hooks**
(`crates/wgpu-compute/src/dflash_session.rs`):

- The constructor now builds its accumulator as a local and
  calls `begin_manifest` with `DflashSession::capture_manifest`,
  built exactly the resident way (build identity, adapter and
  driver identity, the backend's resolved configuration in
  resolved form). The session section comes from the pure
  `capture_session_pairs` helper: `kind=dflash`, `depth`,
  `block_size`, then the target shape (`hidden`, `vocab`,
  `context_length`, `target_hidden`, `conditioning_width`,
  `kv_capacity`). One naming fact the original item left
  open: a draft session has no speculative depth (that is
  `SpecConfig.depth` in itak-spec, and this session never
  sees it), so the `depth` pair states the draft model's
  layer count, `block_count`.
- The block record sits at the one real exit: both public
  forwards funnel through `run_block`, whose single `Ok`
  return now calls `capture_block(&logits)`
  (`record_forward(KIND_BLOCK, ...)`) just before it. Two
  corrections to the item's description, from the shipped
  code: the cited line (dflash_session.rs:1698) predates the
  capped-forward merge, there is no standalone
  `forward_block` exit to hook; and for
  `forward_block_capped` the fingerprint covers the
  returned prefix rows, the only data that forward reads
  back, which is the capture level's own
  no-extra-readback rule.
- Passthroughs mirroring ResidentSession's, all four:
  `note_model_file`, `note_spec_round`,
  `diag_capture_records`, `diag_manifest_json`.

**2. Spec round wiring** (`crates/itak-spec`): `SpecSession`
is generic over its target driver, so the call could not
name ResidentSession directly. `TargetDriver` gained a
record-only seam, `note_spec_round(&proposed, accepted)`,
defaulting to a no-op (the CPU driver records nothing);
`ResidentTarget` overrides it to forward to
`ResidentSession::note_spec_round`. `speculative_round`
calls it exactly once per completed round, after the
commit and the `spec.round.accept` event, at the point
where `proposal.tokens` and `accepted` are both in hand;
a round that fails anywhere above is never recorded, and
the call feeds nothing back into sampling, acceptance, or
control flow. `RoundOutcome` is unchanged.

**Test evidence** (VM, this lane): wgpu-compute 234 passed
/ 0 failed, itak-spec 49 / 0, dflash 24 / 0; clippy
`-D warnings` and fmt clean over all three crates. New
pins: `capture_session_pairs_state_the_draft_shape` (the
exact session pairs);
`capture_wiring_is_inert_in_timing_mode` (the constructor
and block-exit calls driven against a timing accumulator:
no manifest, no records, no sidecars, the resident pin
restated for the draft);
`capture_wiring_records_the_block_in_capture_mode`;
`stub_driver_receives_one_spec_round_record_per_round`
(itak-spec: one record per completed round, length and
count equal to the round outcome's, accepted prefix equal
to the tokens the round emitted). Two adapter-bound pins
follow the resident tests' opt-in discipline and self-skip
off the card: `dflash_session_diag_timing_records_no_capture`
and `dflash_session_diag_capture_records_the_block` in
`crates/wgpu-compute/tests/dflash_session.rs`, gated on the
diag env exactly like the resident diag tests; they join
the card checklist above as the draft session's executing
proof.
