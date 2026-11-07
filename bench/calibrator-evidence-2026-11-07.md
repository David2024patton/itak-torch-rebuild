# Calibrator evidence: the batch-1 artifact and the missing per-row numbers (2026-11-07)

Lane: `lane/calibrator-evidence`, based on main
0ecdc8ff5efc3da0d00b5806c8f2796b65096132. VM-side build
and gates only; no card run in this lane, and the
shipped band table is NOT changed by it (see the last
section).

## The defects, from the wave19 record

The wave19 card session (bench/wave19-session-2026-11-07.md)
ran `calibrate_profile` on the RX 7900 XTX and proved
the auto-tune profile end to end, but its derived table
exposed two defects in the calibrator itself:

1. **Batch-1 artifact rows.** Three derived cells
   (q4k attn, q4k kv, q5k ffn) included batch 1 as an
   MMV win. It cannot be one: the calibration's MMV arm
   pins its band cells to `2-16`, so at batch 1 the
   routing ladder dispatches the plain GEMV form in
   BOTH arms. The session's diag census is unambiguous:
   50 batch-1 GEMV dispatches per shape (25 per arm)
   and zero MMV dispatches at batch 1. The sub-0.98x
   median the old derivation read as a win is
   inter-arm drift between two runs of the same form.
   The old rule asked only about timing and agreement;
   it never asked whether the MMV form dispatched.

   The artifact was cosmetically harmless to routing
   (the band parser clamps derived cells to the
   kernels' 2..=16 band, so a cell text like "1-12"
   routed exactly "2-12"), but it falsified the
   evidence: the derived table claimed a measurement
   that never happened, and any cell-by-cell comparison
   against the shipped table counted a phantom
   difference at row 1.

2. **No per-cell evidence.** The calibrator printed
   only the derived table: nine cell strings, no
   medians, no ratios. A cell that differed from the
   shipped table could not be checked against the
   numbers that produced it without mining the diag
   stream, and the session showed the diag stream does
   not reproduce the derivation (it records dispatches,
   not the two arms' per-row medians). A calibration
   whose cells cannot be checked from its own output
   is weak evidence, and the wave19 comparison had to
   be argued from the census rather than read off the
   run.

## The fix: exclusion by construction

The rule "a row the MMV form cannot dispatch is never
a win" now lives in the shared derivation helpers in
`crates/wgpu-compute/src/profile.rs`, so every caller
of the derivation inherits it, not only the example
that prints the table:

- `mmv_can_dispatch(rows)`: the kernels' own band,
  `MMV_MIN_BATCH..=MMV_MAX_BATCH` (2..=16), imported
  from the K19 module that defines it. This is the
  same band the routing layer enforces
  (`parse_mmv_band` clamps cells to it;
  `MmvBands::routes` floors at its low end), so the
  derivation and the router now read dispatchability
  from one ground truth.
- `mmv_row_win(rows, mmv_ms, control_ms, agrees)`:
  the whole derivation rule in one predicate:
  dispatchable AND both arms agree with the direct
  GEMV form AND `mmv_wins` (the unchanged 0.98x timing
  rule). The calibrator derives its cells and its
  evidence verdicts from this one predicate, so the
  printed evidence and the derived table cannot
  disagree about a row.
- `derive_band_ranges` enforces the same exclusion at
  the grouping step: a win flag at an undispatchable
  row never joins a range, even when a caller hands in
  raw timing verdicts. An excluded row also does not
  bridge two runs (a win at row 1 next to a loss at
  row 2 does not pull a row-3 win into a span starting
  earlier). This is the same rule at the one pinch
  point every derivation passes through, not a second
  rule and not a filter over the printed table.

Unchanged, deliberately: the 0.98x win rule
(`mmv_wins` and `MMV_WIN_RATIO`), the band semantics,
the profile format's key=value lines, the resolution
precedence, the ceiling derivation, and the
calibrator's refusal behavior.

### Pins

New unit tests in profile.rs:

- `mmv_dispatch_band_edges`: 0 and 1 cannot dispatch;
  2 and 16 can; 17 cannot.
- `batch_one_timing_win_never_enters_a_derived_cell`:
  the artifact as a synthetic cell set. Raw timing
  wins at rows 1-4 derive "2-4"; a batch-1 win alone
  derives the closed cell "1"; wins at 1 and 3 with a
  loss at 2 derive "3" (no bridging); and the composed
  predicate refuses the same batch-1 numbers it would
  accept at row 2.
- `evidence_lines_carry_the_row_numbers_and_verdict`
  and `profile_with_evidence_comments_loads_identically`:
  below.

Foundation pins kept green: the shipped-derivation
win-rule pin and the shipped-table derivation shapes
("2,4-12" from the committed sweep's win pattern) are
unchanged. One existing assertion WAS adjusted, and it
is the one the brief allows: the all-verdicts-true
sweep case asserted the derived cell "1-16", but the
sweep's row-1 entry is exactly the artifact position;
with the exclusion, an all-wins sweep derives "2-16".
The other assertions in that test never marked row 1
as a win and did not move.

## The evidence format

`RowEvidence` (family, shape, rows, control median,
MMV median, agreement) renders one line per measured
row, in a stable greppable format:

```
evidence family=q4k shape=attn rows=1 control_ms=1.0000 mmv_ms=0.9000 ratio=0.900 agree=yes verdict=excluded
evidence family=q4k shape=attn rows=6 control_ms=1.0000 mmv_ms=0.8000 ratio=0.800 agree=yes verdict=win
```

- `ratio` is the MMV median over the control median,
  three decimals; `n/a` when the control median is not
  positive (no measurement, per `mmv_wins`).
- `verdict` is `win` (the row enters the derived
  cell), `no-win` (measured and dispatchable, but the
  timing or the agreement failed), or `excluded` (the
  form cannot dispatch at this row; the ratio is
  printed anyway, so the exclusion itself is
  auditable rather than silent).

The calibrator prints all 99 lines (9 cells x 11 rows)
after the two arms and BEFORE the derived table, under
a header naming the predicate, so the table reads as
the conclusion of the evidence. It then writes the
same lines into the profile file as `#` comments via
`write_profile_with_evidence` /
`CalibrationProfile::render_with_evidence`, between
the header comment and the key=value lines. The
format's key=value lines are byte-identical to the
bare render, and the parser already skips comments;
the round-trip pin proves a written evidence-bearing
profile loads back `Loaded` with identical cells,
ceiling, and adapter key, and that the file on disk
still carries the comments.

## What this lane did not do

- **The shipped table is not changed.** The wave19
  derived table also differed from the shipped table
  in five cells on dispatchable rows; those are real
  measurements to weigh, not artifacts to erase. A
  re-derivation with the evidence printed, judged cell
  by cell with the per-row numbers beside it, is a
  future card run's job. This lane only makes that
  future run produce auditable output and artifact-free
  cells.
- No card run: the calibrator's card behavior is
  exercised by the same future run. The refusal paths
  were re-demonstrated here on the adapterless VM
  (below).

## Verification (this VM, no adapter)

- `cargo test -p wgpu-compute -p dflash -p itak-spec`:
  354 passed, 0 failed across 32 test binaries; the
  profile module is 13/13 (9 foundation tests, 4 new).
- `cargo clippy -p wgpu-compute --all-targets -- -D warnings`:
  clean.
- `cargo fmt --check`: clean.
- `cargo check --workspace`: clean.
- Refusal path 1 (reps floor):
  `calibrate_profile --out /tmp/cal-refusal-out.txt --reps 5`
  prints `REFUSED: reps 5 is below the 20-repetition
  verdict floor; ... nothing written`, exits 1, and the
  output file does not exist afterward.
- Refusal path 2 (no adapter): `calibrate_profile
  --out /tmp/cal-refusal-out2.txt` at default reps
  prints `REFUSED: no suitable GPU on this machine to
  calibrate (arm control; enumerated: []); nothing
  written`, exits 1, writes nothing (no file at the
  `--out` path, no default profile directory created).
  Both refusals behave exactly as on the base commit:
  the refusal code precedes the derivation in `run()`
  and was not touched.
