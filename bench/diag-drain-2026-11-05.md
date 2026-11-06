# Diag drain fix: real runs emit the per-kernel timing report

Date: 2026-11-05. Branch: `lane/diag-drain` (from main
`3b2afbcc3d59d9e583acc1fe78e1067f371c328a`). Crate:
`wgpu-compute`.

## The defect

The per-kernel GPU timestamp instrumentation (David's order,
2026-11-04: debugging all the way through everything) could
only produce output inside one test. `DiagRecorder::drain` —
the only call that resolves the timestamp query window and
emits records — had exactly one caller in the tree: the smoke
collector in `crates/wgpu-compute/tests/resident_session.rs`.
That test was also the only reader of `WGPU_COMPUTE_DIAG_OUT`.

Proof, from the cleared-card baseline lane on the server
(2026-11-05): two full generation runs of 28,480 dispatches
each, with `WGPU_COMPUTE_DIAG=timing` and
`WGPU_COMPUTE_DIAG_OUT` set, emitted **zero**
`[wgpu-compute-diag]` records and no file. The wave-7
acceptance had proven the machinery only where the collector
drains it; no session, example, or engine path ever drained.

## The fix

The sessions own the drain now.

- **`DiagAccumulator`** (new, `crates/wgpu-compute/src/timestamps.rs`):
  the session-side running summary. It drains the recorder,
  folds the resolved records into per-kernel aggregates (the
  same sums `summarize()` computes, kept incrementally, so
  memory is one row per kernel however long the run), and at
  finish prints the one-line tally through the diag channel
  and writes the summary table to the `WGPU_COMPUTE_DIAG_OUT`
  file when one was named at construction. Its rendered table
  is byte-identical to `render_summary` over the raw record
  set; a unit test pins that equivalence across multiple
  folds. Diagnostics never fail the run they observe: a failed
  drain or write is counted and reported on the channel, and
  generation proceeds.
- **`ResidentSession`**: carries an accumulator exactly when
  its backend's context carries a recorder. `forward_chunk`
  calls `drain_diag()` after the chunk's submission has been
  submitted and read back, so every chunk of every public
  forward (`forward`, `forward_argmax`, `forward_capturing`,
  `forward_block_capturing`) is drained by the same single
  call site. `finish_diag_report()` (one last drain, then the
  tally and file) is public for deterministic callers and is
  also called from `Drop`, so every timing run leaves its
  report behind. Accessors `diag_dispatch_count()`,
  `diag_kernel_names()`, and `diag_summary_csv()` expose the
  folded state mid-run.
- **`DflashSession`**: the same treatment, for the spec
  loop's draft side — drain after every submitted chunk of
  `inject` and `fuse`, after `forward_block`, same accessors,
  same finish-at-drop. The two sessions share one recorder
  per backend, so each one's drains also flush the other's
  pending stamps; the records' kernel names keep the families
  distinguishable in one report.

## Why this cadence cannot lose stamps on long runs

The recorder's window holds 2048 dispatches
(`MAX_DISPATCHES`); overflow is the only stamp-loss path.
One chunk is one submission whose dispatch count does not
grow with its row count (every dispatch spans all rows): the
full model's chunk measured 445 dispatches (28,480 over 64
chunks in the baseline run), the fixture's 69–93. Draining
after every chunk therefore keeps pending stamps at one
chunk's worth — under a quarter of the window — for runs of
any length. A single chunk larger than the whole window
would still drop, counted by the recorder and surfaced in
the finish tally; no current plan shape comes close.

Two recorder-design facts the fix relies on, verified in
`timestamps.rs`:

- Overflow never corrupts: when the window fills, `begin`
  returns `None` and that dispatch runs unstamped, counted in
  `dropped`. Stamps already in the window stay valid because
  `drain` converts ticks by each pending stamp's stored slot,
  not by order.
- `dropped` is cumulative for the recorder's lifetime and is
  never reset by a drain. Per-session finish lines therefore
  report the per-backend total; on a shared backend (target +
  draft sessions) both report the same counter.

One boundary kept deliberately: the `Compute` trait path
(one-shot forwards through `GlimmerModel` on the GPU backend)
still does not drain. Its runs are test-driven, one forward
per process in practice, and a caller there can drain through
the context's recorder directly. The resident sessions are
the production generation paths, and they are covered.

## The off path

Unchanged, by construction. With `WGPU_COMPUTE_DIAG` unset
or not `timing`, `GpuContext` creates no recorder, so the
sessions create no accumulator, `drain_diag()` is one
`None` branch after each chunk, no query resources exist, no
drain cost is paid, and the encode path is byte-identical.
The existing pins stay green (`off_mode_context_carries_no_recorder`),
and a new session-level test
(`resident_session_diag_off_by_default`) asserts no report,
no records, and a no-op finish through the session API.

## Tests

- `timestamps.rs` unit tests (adapter-free, run everywhere):
  `accumulator_folds_and_renders_like_the_collector` (fold
  math + byte-identical render across folds) and
  `accumulator_writes_its_report_once` (file lands once, an
  empty run writes nothing, the finish is idempotent — the
  drain-trigger's sink logic, pinned without a device).
- `resident_session_diag_drains_itself` (new, executing):
  timing mode + adapter; a plain session's records are
  already folded when a forward returns — no collector, no
  manual drain, no finish call. This is the exact shape of
  run that emitted zero records before the fix.
- `resident_session_diag_smoke_report` (the collector,
  re-pointed): same env gates and assertions, now read from
  the session's own report, including the file the session
  wrote at finish.
- `resident_session_diag_off_by_default` (new): the off
  path through the session API.

Gate results (this VM, final tree): `cargo test -p
wgpu-compute` all suites green, 0 failed — lib 90 passed
(including the 2 new accumulator unit tests),
`resident_session` 6 passed (the 3 diag tests skip cleanly
here: this VM has no GPU adapter and the timing tests also
require the timing env, the crate's standing skip design),
`diag_timestamps` 1 passed (the off-path pin). `cargo test
-p itak-spec`: 47 passed. `cargo fmt --check`: clean.
`cargo clippy -p wgpu-compute --all-targets -- -D warnings`:
clean. `cargo check --workspace --all-targets`: clean. The
executing legs of the new tests first run on the server, per
the confirmation protocol below.

## Server confirmation protocol

On the cleared card (or any card window), from the staged
tree:

1. One plain generation run with timing on:
   `WGPU_COMPUTE_DIAG=timing WGPU_COMPUTE_DIAG_OUT=/tmp/diag-plain.csv`
   on the loader's `glimmer_generate` example (it drives a
   `ResidentSession`). Expect: `[wgpu-compute-diag]` dispatch
   lines on stderr during the run, a
   `[wgpu-compute-diag] session report:` tally at exit, and a
   populated `/tmp/diag-plain.csv` whose `TOTAL` row counts
   the run's dispatches.
2. One spec phase A leg with timing on (the spec example's
   plain/target leg): same expectations; the CSV's kernel
   rows are the per-kernel table for that leg.

Note: spec phase B (draft + verify sessions together) still
needs a host-RAM window on ubuntu-server — the phase B
harness peak is ~35 GB and the box offered ~33 GB on
2026-11-05 — which is unrelated to this fix. When it runs,
both sessions drain into the shared recorder and each
finishes its own report; expect two tally lines (draft and
target) and the later finisher's file content at the shared
`WGPU_COMPUTE_DIAG_OUT` path.

## Server confirmation, 2026-11-06: CONFIRMED

Protocol above, run from main
`c439e3af45e30275601129568e3e9cb3a89590cb`, staged at
`~/gpu-wave11` (manifest 364 files, sha256-identical),
card cleared (VRAM used 33,271,808 bytes at both
guards), phase lock held. Logs and CSVs in
`~/gpu-wave11/logs11/` on the server.

1. Plain generation, `glimmer_generate` with
   `WGPU_COMPUTE_DIAG=timing` and
   `WGPU_COMPUTE_DIAG_OUT` set: 28,480 per-dispatch
   `[wgpu-compute-diag]` lines on stderr during the
   run, then the finish line
   `[wgpu-compute-diag] session report: 28480
   dispatches across 9 kernels, 0 dropped, report
   written to .../diag-plain.csv`. The CSV holds the
   9 kernel rows plus a TOTAL row counting 28,480
   dispatches (k1_q4k_gemv 11,648 dispatches /
   1,357.2 ms / 574.6 GB/s at the top).
2. Spec phase A leg (`spec_generate_gpu`, plain mode)
   with the same env: identical shape — 28,480
   dispatch lines, the session report tally (28,480
   dispatches, 9 kernels, 0 dropped), and a populated
   CSV with the same kernel rows (k1_q4k_gemv
   1,132.6 ms / 688.6 GB/s).

The defect this fix closed — two full 28,480-dispatch
real runs with timing on emitting zero records and no
file (2026-11-05 cleared-card baseline lane) — does
not reproduce: both real runs here emitted every
record, the tally, and the file, with zero dropped
stamps. The drain fix is confirmed on the server.
