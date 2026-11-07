# Proof tools verdict plumbing: the Category F pair of HIGHs, plus their sibling MEDIUMs (2026-11-07)

Lane `lane/proof-tools-verdicts`, from main
2a2030e18b1089708c831217c38dc8a0e7ff4e4c. Code in the first
commit on this lane; this note is the second commit. All proof
runs executed on this VM against this lane's tree (before runs
on the unfixed tree, after runs on the fixed tree, same fixture
files both times); no server jobs.

Source of the work: the cold critic panel of 2026-11-07
(`CRITIC-ENGINE-MERGE.md`, Category F; probes executed by the
mop-up seat). The harnesses that produce this project's golden
and cross-check verdicts could print "pass" on a failing run:
divergence was invisible to the verdict logic, and exit codes
said success regardless. The code the harnesses verify was
checked directly by the correctness seat and is clean; only the
verdict plumbing was broken. Four findings were in scope; each
fix below was proven lying-then-honest by execution, and every
tool was also run on a genuine match to prove no false
divergence was introduced.

## Finding 1 [HIGH]: compare_traces never gates on the first values

`golden/tools/compare_traces.py`, `judge()`: the first-values
metrics (`first_max_diff`, `first_rel_std`) were computed and
printed but never entered `ok`, although the module's own
reading rule says a stage matches "with the first values
tracking".

Repro (the mop-up seat's, rebuilt as a fixture): reference
`embed` tensor `[1,2,3,4]`; our trace line carries the
statistics of the permutation `[4,3,2,1]` (identical mean,
population variance, and max_abs; first values completely
different).

Before:
```
embed            match                  0.000e+00     0.000e+00      0.000e+00   3.000e+00
first divergent stage: None
EXIT=0
```
Fix: `first_rel <= 1e-2` (the same band as the other criteria,
in units of the stream's own standard deviation, exactly as
the seat prescribed) joins the `ok` conjunction in `judge()`.

After:
```
embed            DIVERGENT              0.000e+00     0.000e+00      0.000e+00   3.000e+00  <-- FIRST DIVERGENT (input stage matched: True)
first divergent stage: embed
EXIT=1
```
Regression case: `golden/tools/test_verdict_tools.py` generates
this fixture pair itself and asserts the permuted tensor prints
DIVERGENT; it also pins the clean pair (match, exit 0) and the
NO REF exit below.

## Finding 2 [HIGH]: compare.sh exits 0 on a failing vector set

`bench/q2q3-xcheck/compare.sh`: `run_set` returned diff.py's
status, but nothing read it; each call overwrote the last, the
script's exit status was the final (absent) set's, and an absent
set returned 0.

Repro (real script, real Rust probe from the lane it hardcodes):
a directory holding one present set, `q2k_synth.bin`, with a
deliberately wrong reference (all zeros), and the other seven
sets absent.

Before:
```
q2k_synth: values=256 exact=62 (24.2188%) zero_sign_only=0 nan_mismatch=0 max_abs_diff=45
q3k_synth.bin: absent, skipped
... (six more absent lines)
EXIT=0
```
Fix: per-set results accumulate into PASSED / FAILED / SKIPPED
lists (a failed probe run now counts as a set failure instead
of diffing a stale dump), a per-set summary prints at the end,
and the script exits 1 when any present set failed. The server
half's discipline is `set -euo pipefail`; the local half
deliberately does NOT adopt `-e`, because it must visit every
set to report the per-set summary the panel required. Manual
accumulation reaches the same verdict without aborting the
remaining sets.

After, same directory:
```
---- compare.sh summary ----
FAIL    q2k_synth.bin
SKIPPED q3k_synth.bin
... (six more SKIPPED lines)
sets: 0 passed, 1 failed, 7 skipped
EXIT=1
```
Judgment call, absent sets: an absent set does NOT by itself
fail the run. It is named in the summary and counted, and a
partial staging copied back for a quick check stays usable
(the skip path is a designed behavior that predates the
finding; the seat's fix shape tracks absents separately from
failures). One exception: if NO set was present at all, the
run exits 1 with `compare.sh: no sets were compared; nothing
was verified` on stderr, because a verification that verified
nothing must not report success. Executed: the all-absent run
prints 8 SKIPPED lines and exits 1.

Clean-pair proof: the same script against a directory whose
`q2k_synth_ref.f32` was generated from the probe's own output
prints `exact=256 (100.0000%)`, `PASS q2k_synth.bin`, and
exits 0 with the seven absent sets counted as skipped.

## Finding 3 [MEDIUM]: cosine against a zero tensor, and verdict tools that cannot exit nonzero

`golden/tools/compare_layers.py` `stats()` defined cosine as
1.0 whenever the product of the norms was 0, so a dead-vs-live
checkpoint printed a perfect cosine; and neither
compare_layers.py nor compare_traces.py could exit nonzero on
any divergence, missing reference, or partial trace.

Unit repro (module imported and called, the seat's shape):
```
before: stats(zeros((2,4)), full((2,4), 0.05)) == (0.05, 0.05, 1.0)
after:  stats(zeros((2,4)), full((2,4), 0.05)) == (0.05, 0.05, 0.0)
        stats(zeros((2,4)), zeros((2,4)))      == (0.0, 0.0, 1.0)
```
Fix: `stats()` compares the norms separately: cosine 0.0 when
exactly one norm is zero, 1.0 only when both are zero. In
`main()`, a both-sides-all-zeros checkpoint line carries the
marker `(both sides all zeros)` so the legitimate 1.0 is not
read as agreement between live activations. Both tools now
return their verdict as the exit status: compare_layers exits
1 when any checkpoint is flagged divergent; compare_traces
exits 1 when a first divergent stage is set or any stage
reported NO REF; both exit 0 only on a full match.

Full-run proof (generated 28-layer fixture pair; llama `embd`
all zeros on both sides, llama `L00_attn_norm` all zeros
against Rust 0.05, everything else identical constants):
```
before: L00_attn_norm  cosine 1.000000, no flag,
        first divergent checkpoint: None, EXIT=0
after:  embd           cosine 1.000000  (both sides all zeros)
        L00_attn_norm  cosine 0.000000  <-- DIVERGENT
        first divergent checkpoint: L00_attn_norm, EXIT=1
```
Clean-pair proof: the same generator with no zeroed stems
prints no flags, `first divergent checkpoint: None`, EXIT=0;
the faithful compare_traces fixture prints `match`, EXIT=0.

## Finding 4 [MEDIUM]: diff.py's pass criterion is value-exact under a bit-exact claim

`bench/q2q3-xcheck/diff.py` counted signed-zero-only bit
differences (`zero_sign`) but excluded them from the verdict,
while the harness framing claims the comparison is "byte for
byte" (harness.c header) and diff.py headlines a "bit-exact
match count".

Repro (the seat's): ref `[+0.0, 1.5, -0.0, 2.0]` vs Rust text
`[-0.0, 1.5, +0.0, 2.0]`:
```
before: sztest: values=4 exact=2 (50.0000%) zero_sign_only=2 nan_mismatch=0 max_abs_diff=0, EXIT=0
after:  (identical printed line), EXIT=1
```
Judgment call: the panel offered either failing on
`zero_sign > 0` or restating the claim as value-exact wherever
it is made; the task brief prefers the stricter,
claim-preserving option unless a consumer depends on
value-exact semantics. None does. The only exit-code consumer
is compare.sh, and every recorded run of this harness
(`bench/overnight-q2k-q3k-dequant-2026-11-05.md`, all eight
sets, 223,744 values) reports sign-of-zero-only = 0 with exact
at 100%, so the stricter verdict changes no recorded result;
it makes the "byte for byte" claim true going forward. Fix:
the return condition gains `and zero_sign == 0`, and the
docstring states the bit-exact verdict explicitly. NaN handling
was already bit-strict (any NaN bit difference lands in
`nan_mismatch` and fails) and is now pinned by a regression
case in `bench/q2q3-xcheck/test_diff.py`, alongside the
signed-zero, identical, and real-mismatch cases.

## Gates

- Before/after executions above, on identical fixtures.
- `python3 golden/tools/test_verdict_tools.py`: 8/8 cases pass.
  (One case was corrected during the lane: identical-tensor
  cosine is 0.9999999999999998 in floating point, so the test
  asserts within 1e-12 rather than exact 1.0.)
- `python3 bench/q2q3-xcheck/test_diff.py`: 4/4 cases pass.
- `python3 -m py_compile` on both edited .py files: clean.
  `bash -n` on compare.sh: clean.
- Clean-pair runs of every tool print match and exit 0 (no
  false divergence): compare_traces faithful fixture,
  compare_layers clean fixture, diff.py identical dumps,
  compare.sh probe-generated reference.
- Shared tree untouched and whole: `cargo check --workspace`
  clean; `go test ./...` from platform/ all packages ok.

## Out of scope, remaining (named by the panel, not touched here)

- cb_dump_glimmer.cpp: non-F32 tensors skipped with only a
  stderr line; no SKIPPED verdict per stage downstream.
- cb_dump.cpp: token ids parse with atoi (malformed input
  silently becomes token 0); dead separator ternary and
  unchecked fwrite/fopen calls (NIT).
- harness.c / the Q2/Q3 cross-check never compares the
  quantizer direction, only dequantization.
- extract_real.py: no-data_offset fallback provenance, and it
  exits 0 even when every model failed to parse (NIT,
  unverified).
- compare_traces quiet shrinkages beyond the exit path fixed
  here (unparsed trace lines dropped silently; layer count
  derived from our own trace), gen_vectors.py stale comment
  and unexercised quantizer regimes (NITs), compare_layers
  hardcoded geometry (NIT), fork_margins negative-margin label
  (NIT).
