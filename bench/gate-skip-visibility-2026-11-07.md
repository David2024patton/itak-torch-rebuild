# Gate skip visibility: the Category J HIGH, and the spec-runner depth default (2026-11-07)

Lane `lane/gate-skip-visibility`, from main
34358e33f6b0fe8f0ef5977675ed10e04228501a. The wrapper, the
runner defaults, and the sweep wording are the first commit on
this lane; this note is the second. All runs executed on this
VM against this lane's tree; no server jobs. The VM enumerates
no Vulkan adapters at all (the wrapper's probe readout, below),
so every GPU test here takes its skip path by design; the card
is the only place the adapter-present branch runs for real, and
that branch is exercised here by the wrapper's documented
override instead.

Source of the work: the cold critic panel of 2026-11-07
(`CRITIC-ENGINE-MERGE.md`, Category J [HIGH] and the defaults
table's row 22 [NIT]).

## Item 1: the green gate hid the skips

The finding: roughly 120 of 758 Rust tests pass vacuously on
any adapterless machine, and the standard gate hides it. A
`--nocapture` run printed 119 "skipping" lines while every
suite reported ok, because the default gate discards a passing
test's stderr. There is no CI in the tree to run them on the
card (TODO T-1.7 would create one and is still open; this lane
does not build CI).

### The panel's observation, reproduced at this lane's base

Failing-before evidence, run on the untouched tree at base
34358e3 before any edit:

- The documented gate, `cargo test --workspace`, reports
  **781 passed, 0 failed, 1 ignored** across 68 suites, and its
  captured output contains **zero** lines with "skipping". The
  green result shows no trace that anything skipped.
- The same suite under `--nocapture` prints **134 lines
  containing "skipping"**: 126 adapter-skip lines and 8 other
  lines (itemized below). The proof surface for every shipped
  GPU behavior was invisible in the gate that claims it.

### The skip-idiom survey

Every skip is announced by the tests themselves on stderr;
there are exactly two adapter-absence idioms and a small set
of configuration/capability prints. Surveyed file by file:

Adapter-absence idioms (the vacuity signal):

- **A. `skipping <name>: no suitable adapter (enumerated:
  [...])`** — printed when `WgpuCompute::new()` refuses with
  `NoSuitableAdapter`, by the per-file `backend_or_skip()`
  helpers in 24 wgpu-compute integration files (chunked_gemv,
  dflash_session, fused_kernels, k10_k11_gemm, k12_k14_mmq,
  k15_k16_mmq_wide, k18_small_batch, k19_mmv, k1_synthetic,
  k20_mmv, k21_finish, k21_mmv, k21f_head_routing, k22_mmq,
  k23_coopmat, k23_mmq, k24_coopmat, k2_k3_synthetic,
  k4_k8_kernels, k8_argmax, prefix_kv_session, proj_bench,
  resident_block, splitk_mmq), by inline variants in
  cross_backend, kv_f16, resident_session, and
  diag_timestamps, by unit tests inside wgpu-compute
  (src/compute.rs, the k21 finish registration test) and
  itak-spec (src/speculator.rs, the gpu arm test), and by
  itak-spec's ngram_gpu_runner integration test.
- **B. `skipping <name>: wgpu enumerates no adapters
  here`** — the shader-compile wording: shader_compile.rs
  (14 print sites) and one site in k4_k8_kernels.rs.

Named exceptions (printed skips that are NOT adapter absence;
they fire by configuration or by device capability, so zero
is not their expectation on any machine):

- k21f_head_routing's env-arm routing pins (3 lines at
  shipped defaults): each arm prints when the
  WGPU_COMPUTE_K21F_HEAD / WGPU_COMPUTE_MMV_MAX_ROWS position
  it needs is not the process position.
- Diag capture arms needing an opt-in environment:
  dflash_session's diag timing and diag capture tests
  (DIAG_OUT / WGPU_COMPUTE_DIAG unset), resident_session's
  diag smoke (WGPU_COMPUTE_DIAG_OUT unset) and diag
  self-drain (WGPU_COMPUTE_DIAG not `timing`), and the
  "this process runs with diag on" off-mode checks.
- Device-capability gates, reachable only when an adapter IS
  present: the k23/k24 "stock device (no coopmat bridge)" and
  "device fails the kernel's gates" prints, "timing requested
  but the device granted ..." arms, and staged-flash /
  parity-leg device refusals. None of these can fire on an
  adapterless machine, because the adapter print fires first.

### Measured counts at base 34358e3 (adapterless VM)

From the `--nocapture` evidence run (781 passed, 0 failed,
1 ignored):

| Class | Lines |
| :--- | ---: |
| Adapter idiom A ("no suitable adapter") | 111 |
| Adapter idiom B ("wgpu enumerates no adapters here") | 15 |
| **Adapter skips, total** | **126** |
| Env-arm exceptions (k21f pins 3, diag arms 4) | 7 |
| Interleaving artifact (one merged line) | 1 |
| **All lines containing "skipping"** | **134** |

Counting rules, as the wrapper implements them: a line counts
as an adapter skip when it contains "skipping" and either
idiom's marker; any other line containing "skipping" counts
as an other skip. Raw marker counts in the same log are 112
lines containing "no suitable adapter" and 15 containing
"wgpu enumerates no adapters here"; the one-line difference
against the table is output interleaving: under `--nocapture`
parallel tests share stderr, and fragments merge. One merged
line in this log reads `skipping test
resident_block_logits_and_captures_match_cpu ... ok` (a skip
fragment fused with a libtest result line), and one adapter
line is fused with its test's `test ... ` prefix. Counts are
therefore exact to within a line or two, run to run: the
wrapper's own run below counted 127 adapter / 7 other against
this survey's 126 / 8, with raw idiom totals identical (112 +
15) in both logs. The gated rule (any adapter skip fails on
an adapter machine) and the beside-the-pass-count reporting
are both robust at that tolerance.

### The wrapper: bench/gate.sh

Per the panel's fix: a gate wrapper that counts skip lines
and fails when they exceed the adapterless expectation on a
machine with an adapter; the adapterless-expected skips
listed as T-1.7's explicit exception set; until CI exists,
gate reports state the skip count beside the pass count.

`bench/gate.sh` runs the first of the RUNBOOK's plain gate
commands — and only that one; clippy, fmt, cargo deny, and
the Go platform suite stay the plain commands beside it —
with two changes of lens:

1. It runs `cargo test --workspace -- --nocapture`, tee'd to
   the terminal and a log, and sums the per-suite
   `test result:` lines into passed / failed / ignored.
2. It counts adapter skips and other skips by the idioms
   above and prints one summary line with the skip counts
   beside the pass count, in both branches.

Adapter detection mirrors the tests' own skip signal. The
tests skip when `WgpuCompute::new()` refuses under
`DetectPolicy::default()` (crates/wgpu-compute/src/
detect.rs): enumerate Vulkan adapters through a wgpu
instance, and qualify only a discrete one (an integrated or
other adapter qualifies only under an explicit non-default
policy; a CPU-type adapter such as llvmpipe never qualifies,
per BACKEND.md Section 6). The wrapper probes with the
crate's own readout example, `adapter_features`, which
enumerates through the same wgpu Vulkan path and prints each
adapter's facts; the machine counts as adapter-present when
any printed adapter line carries `device_type=DiscreteGpu`,
exactly the default policy's qualifying class. A probe that
fails to run fails the gate: the wrapper will not guess its
premise.

Verdicts:

- Tests failed (cargo exit nonzero or any failed count):
  FAIL, either branch.
- Adapter present: any adapter skip FAILS the gate, with the
  first 10 offending lines printed; other skips are reported
  in the summary, not gated (the card sessions own the
  device-capability verdicts).
- Adapterless: PASS, with the summary naming the skips as
  adapterless-expected. If the adapter count drifts from
  EXPECTED_ADAPTER_SKIPS (126, the base measurement above),
  the wrapper prints a drift note naming the enumerated
  expectation and still passes; the script's header carries
  the full enumeration (suites and idioms composing the set)
  as T-1.7's exception list in starting form, and the note
  asks for it to be updated when drift is reviewed.

`GATE_ASSUME_ADAPTER=1|0` overrides the probe so the gate
logic can be exercised on either class of machine; a forced
decision is always named in the wrapper's output. House
style follows bench/failure-bundle.sh (set -u, quoted
expansions, exit 64 for usage). shellcheck is not installed
on this VM (verified 2026-11-07: `command -v shellcheck` is
empty), so like failure-bundle.sh the script carries that
fact in its header instead of a clean bill; `bash -n` runs
clean.

### Demonstration 1: adapterless (the probe path), this VM

The probe's full readout on this machine is one line —
`adapter_features: no Vulkan adapters enumerated` — so the
decision is ABSENT on enumeration alone, before the policy
question even arises. The wrapper's own lines, verbatim:

```text
gate: adapter decision: ABSENT, no qualifying discrete Vulkan adapter (adapter_features probe (/tmp/gate-probe.crkz2Q))
gate: running: cargo test --workspace -- --nocapture (full output: /tmp/gate-test.nC02WY)
gate: note: adapter skip count 127 differs from the enumerated expectation (126 at base 34358e3); when the drift is reviewed, update the enumeration in bench/gate.sh so T-1.7 inherits a current exception list
gate: passed 781, failed 0, ignored 1; skipped 127 adapter (adapterless-expected: no qualifying adapter on this machine), 7 other (named exceptions, not gated)
gate: full output: /tmp/gate-test.nC02WY
gate: PASS
```

Exit code 0. The 127/7 against the survey's 126/8 is the
interleaving tolerance described above (raw idiom totals
112 + 15 identical in both logs); the drift note firing on a
one-line interleave difference is the conservative direction
— it asks for a review, it does not hide one.

### Demonstration 2: forced adapter-present (GATE_ASSUME_ADAPTER=1)

The same tree, the same machine, the decision forced. The
wrapper's lines, verbatim (summary and verdict from stdout,
offender list from stderr, trimmed to the first lines):

```text
gate: adapter decision: PRESENT (GATE_ASSUME_ADAPTER=1 (forced; probe not run))
gate: running: cargo test --workspace -- --nocapture (full output: /tmp/gate-test.aYqwXG)
gate: passed 781, failed 0, ignored 1; skipped 127 adapter (adapter present: expectation 0), 7 other (named exceptions, not gated)
gate: adapter skips observed (first 10):
skipping gpu arm speculator test: no suitable adapter (enumerated: [])
skipping n-gram GPU runner construction test: no suitable adapter (enumerated: [])
skipping k21 finish registration test: no suitable adapter (enumerated: [])
skipping GPU test: no suitable adapter (enumerated: [])
... (6 more of the first 10, same idiom)
gate: FAIL: 127 adapter skips on a machine with a qualifying adapter
```

Exit code 1. The failure mode is real: a tree whose GPU
tests all skipped can no longer report a clean gate on a
machine that could have run them. The 7 other skips (the
named env-arm exceptions) are reported and do not gate.

## Item 2: the spec runners' depth default (defaults row 22)

The panel's NIT: both spec runners defaulted their depth
argument to 6, contradicting the library's measured default
of 1; anyone benchmarking from runner defaults measured the
slower operating point. The library's own record is
unambiguous: SpecConfig's depth default is 1, the winner of
this stack's T-9.7 sweep (bench/spec-sweep-gpu-fixed-2026-
11-04.md: depth 1, acceptance 0.615, 16.86 tok/s, picked by
SPEC 9.4's smallest-depth-within-2% rule); depth 6 is the
llama.cpp operating point.

Changes, additive only (no flag renames; explicit depth
arguments parse and behave exactly as before):

- `crates/itak-spec/examples/spec_generate.rs` and
  `spec_generate_gpu.rs`: the depth argument's absent and
  unparsable fallbacks now use a named `DEFAULT_DEPTH: usize
  = 1` in each file, documented as the T-9.7 winner and
  SpecConfig's default. The GPU runner's bar comment keeps
  its depth-6 reference deliberately: the 88.6 tok/s bar IS
  the recorded llama-server operating point at depth 6, a
  fact about the bar's provenance, not about this runner's
  default.
- `crates/itak-spec/src/sweep.rs`: the candidate-set doc no
  longer calls 6 "the standing default that sweep picked".
  It now states the set is the 2026-10-30 llama.cpp sweep's
  menu kept for cell-for-cell comparability, that this
  stack's own sweep (T-9.7) picked depth 1, which is
  SpecConfig's standing default, and that the 6 in the set
  is the measured llama.cpp operating point. The set itself
  is unchanged: [1, 2, 3, 4, 6, 8, 12, 16].

Pins: the depth default lives in the two examples' `main()`
functions and is not reachable from a library test (examples
are not importable), so it is pinned by the named constant
and its doc, and by the gates below (both examples build
under clippy --all-targets and cargo check). What IS
reachable is pinned in sweep.rs's
`default_config_is_valid_and_covers_the_operating_point`,
updated in place: it now asserts the default candidate set
contains 1 ("the standing default is swept"), and keeps the
6 assertion under its corrected meaning ("the llama.cpp
operating point is swept"). The neighboring rendering test's
"Chosen default depth: **6**" line is fixture output, not a
claim: its fabricated rows make depth 6 the fastest row, so
6 is the correctly rendered winner of that fixture. It is
deliberately untouched, and named here so a future reader
does not mistake it for a third stale default. SpecConfig's
own default (depth 1) was already pinned before this lane
and is unchanged.

## Gates, as observed on this lane's tree

- `bench/gate.sh` (adapterless, probe path): PASS — 781
  passed, 0 failed, 1 ignored; 127 adapter skips
  (adapterless-expected), 7 other. This is the full
  workspace test suite on the final tree.
- `GATE_ASSUME_ADAPTER=1 bench/gate.sh`: FAIL by design —
  same tallies, 127 adapter skips named as the excess,
  exit 1 (demonstration 2).
- `cargo test -p itak-spec`: 70 passed, 0 failed (includes
  the updated sweep pin); the workspace tally above covers
  the rest.
- `cargo clippy --workspace --all-targets -- -D warnings`:
  clean.
- `cargo fmt --check`: clean.
- `cargo check --workspace`: clean.
- `bash -n bench/gate.sh`: clean. shellcheck: not run —
  the tool is absent on this VM (see the wrapper section).
- Platform (untouched by this lane, run for the record):
  `go build ./...`, `go vet ./...` clean; `go test ./...`:
  123 tests, all passing (9 packages ok).

## Deliberately left

- **CI itself is TODO T-1.7 and stays open.** This lane
  builds no workflow, no CI config, and no clippy/golangci
  configuration. The enumeration in bench/gate.sh's header
  is T-1.7's exception set in starting form, measured at
  this base; T-1.7 inherits the strict per-test enforcement.
- The RUNBOOK's Verify section still describes the gates as
  plain commands with no gate scripts. This wrapper is the
  panel-prescribed exception for the one command whose
  output the plain form structurally cannot show; the
  RUNBOOK text is left for the merge owner to reconcile,
  not silently edited here.
- `crates/itak-spec/examples/ngram_acceptance.rs` carries
  the same `map_or(6, ...)` depth fallback. It is an
  n-gram acceptance probe, not one of the two spec
  generation runners the panel's row 22 names, so it is
  outside this lane's scope and unchanged; flagged here so
  the choice is visible rather than missed.
