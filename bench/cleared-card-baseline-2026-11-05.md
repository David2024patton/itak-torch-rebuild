# Cleared-card spec baseline (server proof, 2026-11-05)

Main `e537764f9bd915e0eecf8739a19c75f4dcdb1210` (all overnight
work merged), staged at `~/gpu-wave8` on ubuntu-server from a
`git archive` of that commit; the staged tree was verified
against a sha256 manifest of the local tree before any run
(332 files, 0 mismatches). Builds and runs under the phase
lock (`flock -w 14400 /tmp/itak-phase.lock`),
`CARGO_BUILD_JOBS=4`, `MALLOC_ARENA_MAX=2
MALLOC_TRIM_THRESHOLD_=0` on every model run. Runner:
`crates/itak-spec/examples/spec_generate_gpu.rs` (release
build, BUILD_EXIT=0, 31.76s), real files
`Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf` (16,756,683,904 B)
and `dflash-Muse-Glimmer-30B-Q4_K_M.gguf` (1,631,208,128 B).
Production Glimmer (localhost:8080) stayed down; the stopped
8091/8092 tenants were not touched; no reboot.

## Headline

- **Plain decode: the 35.45 tok/s figure holds on the cleared
  card.** Three phase A passes on the verified-clear card read
  35.68, 35.59, and 35.56 tok/s (details and one invocation
  defect below). Clearing the card did not move plain decode;
  the idle tenants were never its constraint.
- **Spec phase B: NOT measured. It never ran.** Six phase B
  attempts across two windows all died the same way: the host
  OOM killer took the runner during phase B setup, at
  34.6-34.7 GB anon-rss per kill (kernel log lines quoted
  below). The blocker moved: the autopsy pass lost phase B to
  VRAM (device-lost at about 25.3 GB wanted of a 25.75 GB card
  with 6.54 GB held by tenants); tonight the card was clear
  and the wall is host RAM instead. Tonight's server offered
  32-38 GB available of 123 GB (Cortex SurrealDB 31.3 GB RSS,
  the Chris LXD VM 26.5 GB, plus the usual services; swap
  3/3 GB used throughout). On 2026-11-04 the same phase B
  completed repeatedly in a window where the baseline lane
  recorded 54 GB available. The standing spec numbers
  therefore remain the pressured-card ones (depth 1 verdict
  14.60 tok/s, depth 6 verdict 9.21 tok/s, from
  bench/small-batch-2026-11-04.md), and cleared-card spec
  rates, draft ms, verify ms, and acceptance per depth are
  still unmeasured. IDENTITY and GOLDEN for the spec stream
  are likewise unmeasured tonight; the plain side of the
  golden check passed in every correct-prompt run (phase A
  streams begin exactly [13796, 26, 1573, 373, 290, 2167,
  26899, 1212]).
- **Diag capture for a spec round: no table exists to quote,
  and the reason is a wiring gap, named below.** Two runs
  with `WGPU_COMPUTE_DIAG=timing` and
  `WGPU_COMPUTE_DIAG_OUT` set each completed a full phase A
  (28,480 dispatches) inside the process, and the process
  emitted zero `[wgpu-compute-diag]` records and wrote no
  file. In the shipped code the only caller of
  `DiagRecorder::drain` (the call that resolves the query
  window and emits records) is the smoke collector test
  (`crates/wgpu-compute/tests/resident_session.rs:716`), and
  the only reader of `WGPU_COMPUTE_DIAG_OUT` is that same
  test (line 682). No session, example, or engine path ever
  drains the recorder, so a long-running process can stamp
  dispatches all day and surface nothing. The wave7
  acceptance proved the machinery works where a collector
  drains it; the follow-up this note files is a drain cadence
  (or an end-of-run drain) in the resident sessions
  themselves. Whether the recorder was created inside these
  example processes cannot be confirmed from their output,
  which is part of the same gap: the run says nothing about
  its instrumentation either way.

## VRAM timeline (amdgpu sysfs, card2)

Card total 25,753,026,560 B throughout.

| When (UTC) | VRAM used | Note |
|---|---:|---|
| 21:22 | 6,536,732,672 | tenants still resident; gate not yet met |
| ~21:31 | 33,271,808 | cleared; lock free; gate met |
| 21:31-21:37 | 33,271,808 at start and after every run | attempt 1 window, verified clear |
| 21:47-21:52 | about 6.4 GB | TAINT WINDOW (notice from the Cortex chat): cortex-chat and cortex-embed containers briefly restarted for a deploy, stopped again about 16:52 EST |
| 21:56 | 33,271,808 | clear again, verified |
| 22:16 | 6,380,896,256 | a tenant resident again (not mine; left alone) |
| 22:20-22:26 | 6,380,818,432 at run3 start and end | run3 window; tenant resident, idle |
| 22:26 | 33,271,808 | clear again at the next poll |

**Taint accounting, as instructed:** no measurement in this
note was taken inside the 21:47-21:52 UTC taint window
(attempt 1 ran 21:31-21:37, run3 ran 22:20-22:26), so no
rows are discarded on that account. Run3 is disclosed
separately: it ran with about 6.38 GB of tenant VRAM
resident (the tenant came and went with another room's work;
it was not stopped or touched). Run3's phase A rates are
therefore corroborating evidence only; the cleared-card
plain verdict rests on attempt 1.

## Attempt 1: combined verdict runs, verified-clear card

Launched 21:31:34 UTC, script `~/gpu-wave8-run.sh`. VRAM
verified at start: used 33,271,808, free 25,719,754,752
(over the 24 GB gate), and re-read at 33,271,808 after each
run.

**Invocation defect (this lane's error, corrected in
run3):** the runs were invoked as
`spec_generate_gpu $T $D 64 <depth> --draft=gpu`. The
runner's positional parser takes args[5] as the prompt
(`crates/itak-spec/examples/spec_generate_gpu.rs`), so the
flag string became the prompt text: every attempt 1 run
tokenized the literal prompt "--draft=gpu" to ids [200000,
353, 76812, 63500, 4418] and generated that prompt's
continuation. (The `--draft=` value itself is found by a
separate scan and was honored; the draft backend was gpu as
intended.) Consequence: attempt 1's streams are not the
golden prompt's stream and carry no GOLDEN verdict. The
phase A decode rates stand, because decode rate is the
per-token forward cost (28,480 dispatches = 64 chunks x 445)
and does not depend on prompt content.

Phase A results, attempt 1 (all three runs identical in
shape: 64 chunks, 64 submissions, 28,480 dispatches):

| Run | Prefill | Plain decode |
|---|---:|---:|
| depth 6 invocation | 1.935 s | 35.68 tok/s |
| depth 1 invocation | 1.932 s | 35.59 tok/s |
| depth 1 diag invocation | (log trimmed) | 35.56 tok/s |

Phase B results, attempt 1: none. All three processes were
killed during phase B, after the phase B target coverage
line and the `[dflash] session.ready` line, before any spec
round output. Exit 137 each. Kernel log (server journal):

```
Nov 05 21:34:05 kernel: Out of memory: Killed process 1814233 (spec_generate_g) total-vm:44460216kB, anon-rss:34728688kB
Nov 05 21:35:31 kernel: Out of memory: Killed process 1819974 (spec_generate_g) total-vm:44421680kB, anon-rss:34690156kB
Nov 05 21:36:38 kernel: Out of memory: Killed process 1827590 (spec_generate_g) total-vm:44438452kB, anon-rss:34690288kB
```

## Run3: split-phase retry, correct prompt

Launched 22:20:39 UTC, script `~/gpu-wave8-run3.sh`, phases
as separate processes (phase A `plain` writing `ps.txt`,
phase B `spec` reading it), prompt passed positionally as
"The capital of France is" (ids [200000, 954, 7963, 323,
11698, 373], the Run B ids). VRAM at start and end:
6,380,818,432 used (tenant resident, see timeline). RAM
available during the window: 38 GB at start, 33-35 GB after.

- Phase A, plain only: PA_EXIT=0. Prefill 1.746 s, **decode
  35.84 tok/s**, 64 chunks, 28,480 dispatches. Stream begins
  [13796, 26, 1573, 373, 290, 2167, 26899, 1212, ...]: the
  golden first 8, exact.
- Phase B spec, depth 6: killed after `[dflash]
  session.ready`, SD6_EXIT=137, anon-rss 34,664,964 kB
  (22:24:05 UTC). The split did not lower the phase B peak
  enough to matter: phase B's own host footprint is the
  wall, not inter-phase residue.
- Phase B spec, depth 1: killed at the same point,
  SD1_EXIT=137, anon-rss 34,557,580 kB (22:24:48 UTC).
- Diag combined run, depth 1, `WGPU_COMPUTE_DIAG=timing`,
  `WGPU_COMPUTE_DIAG_OUT=~/gpu-wave8/diag-d1.csv`: phase A
  completed in-process (prefill 1.768 s, decode 35.48
  tok/s, stream golden-exact), phase B killed at the same
  point (DG_EXIT=137, anon-rss 34,690,372 kB, 22:25:47
  UTC). Diag yield from the whole process, verbatim from
  the run log: `diag dispatch lines: 0` and
  `no diag-d1.csv file`. See the headline for the wiring
  gap this proves.

No run device-lost at any point tonight. Every death was a
host OOM kill during setup, before the first spec round;
the GPU itself was never the failing resource in run3, and
VRAM returned to its prior level after every kill.

## Comparison against the pressured-card numbers

| Measure | Pressured card (2026-11-03/04) | Cleared card (2026-11-05) |
|---|---|---|
| Plain decode | 35.45 tok/s | 35.56-35.68 tok/s (attempt 1, verified clear); 35.48-35.84 with an idle tenant resident (run3). **35.45 confirmed.** |
| Spec depth 1 verdict | 14.60 tok/s | not measured (phase B host-OOM) |
| Spec depth 6 verdict | 9.21 tok/s | not measured (phase B host-OOM) |
| Spec phase B failure mode | device-lost, VRAM arithmetic (autopsy) | host OOM kill at about 34.7 GB anon-rss, before round 1 |
| Draft ms / verify ms / acceptance | recorded in the 2026-11-04 sweep and verdict notes | not measured |

## What unblocks the spec baseline

A host RAM window with roughly 50 GB or more available, of
the kind the server had in the 2026-11-04 verdict window
and the 2026-11-05 baseline window (54 GB available
recorded), with the card clear at the same time. The runs
themselves are staged and scripted (`~/gpu-wave8`,
`~/gpu-wave8-run3.sh`); in such a window the split-phase
sequence (plain, spec d6, spec d1, diag d1) is about ten
minutes of machine time. The alternative is engine work:
phase B's host footprint (target weights materialized
host-side alongside the draft session) is the thing to
shrink, and it is now measured at about 34.7 GB anon-rss
and still growing when killed, so its true peak is above
that.

Artifacts: server staging `~/gpu-wave8` (`run.log`,
`run3.log`, `d6.out`/`d1.out`/`d1diag.out` attempt 1;
`pa.out`, `sd6.out`, `sd1.out`, `dg.out` run3; `ps.txt`
the golden plain stream).
