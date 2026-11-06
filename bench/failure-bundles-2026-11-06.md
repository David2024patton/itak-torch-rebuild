# Failure bundles: the wrong-data vault (2026-11-06)

## Why

David's standing want: every form of debugging data we can
squeeze out, and the WRONG data kept in full, because wrong
outputs are what fixes are built on. Until now, failures
were preserved only by hand. The standing example: the
first top-5 probability run came back degenerate (every
record's top-5 list held 1 entry, every top-1 probability
exactly 1.0), the pickup lane refused to commit it, and the
file survived only because someone kept it on the server
by hand as glimmer.top5.degenerate.jsonl. A failure that
exists only in a log someone happened to save is a failure
we will eventually lose.

`bench/failure-bundle.sh` makes the preservation
automatic and uniform: one wrapper around any bench or
harness command, and a failed run assembles its own
evidence bundle before anyone has to remember to save it.

## The convention

Every future verdict protocol wraps its runs in
bench/failure-bundle.sh:

```sh
bench/failure-bundle.sh <slug> [outdir] -- <command...>
```

- A run FAILS when the command exits nonzero OR when its
  combined output matches a failure marker. Both shapes
  produce a full bundle.
- A failure without a bundle is an incomplete report.
  When a verdict note says FAIL, the bundle directory
  name goes in the note next to the claim, the same way
  raw logs are cited today.
- The wrapper never changes the run: output is tee'd to
  the caller's terminal live, and the wrapper exits with
  the wrapped command's exit code in all cases. A
  marker-only failure therefore still exits 0; it is
  detected through the bundle and the INDEX, not the
  exit code. Protocols that need a hard stop on wrong
  data check for the bundle.

## Bundle format

Bundles live in the vault, `bench/failures/` (override
with FB_VAULT), one directory per failed run, named
`<UTC timestamp>-<slug>`:

| file | contents |
| --- | --- |
| manifest.txt | slug, verdict with fail reasons, the exact command (shell-quoted, re-runnable), exit code, matched markers with counts, UTC and local start/end, duration, host and uname, repo git SHA, branch, dirty state with the full porcelain listing, the model block (below), the engine environment, free RAM (MemAvailable) at start and end, VRAM used at start and end (amdgpu sysfs; "unavailable" is recorded, never hidden, on hosts without the counters) |
| stdout.log | the command's full stdout, as printed |
| stderr.log | the command's full stderr, as printed |
| env.txt | every resolved WGPU_COMPUTE_* and ITAK_* variable, sorted |
| outputs/ | copy of the caller-named output directory's contents, when one was given |
| REPRO.md | the exact command line and the pass condition it failed: expected (exit 0 and no markers) against observed (exit code, matched markers), plus the state summary |

The model block records path, size, mtime, and sha256.
The hash is streamed (sha256sum) once and cached by
path+size+mtime in the vault's `model-hash-index.tsv`,
so a 16 GB GGUF is not re-hashed on every run. The model
is taken from FB_MODEL when set, else from the first
command argument that is an existing *.gguf file.

Failure markers are fixed strings, case-sensitive,
matched as substrings against the combined output. The
list is the FAILURE_MARKERS variable at the top of the
script, and it decides what counts as wrong data:

```
FAIL  FAILED  mismatch  DIVERGES  REFUSED  degenerate
```

Extend it in review, never silently.

Passing runs keep no bundle and no logs: exactly one line
in `bench/failures/INDEX.md` (UTC, slug, PASS, command),
so the vault holds every failure in full and every pass
as an index line. FAIL runs also get an INDEX line,
pointing at their bundle.

## Demo evidence (this VM, 2026-11-06)

Three trivial runs through the wrapper, no GPU involved:

1. `demo-fail`: prints "top5 mismatch at row 3", exits 3.
   Bundle 20261106T224349Z-demo-fail assembled (reasons:
   exit code 3; marker mismatch x1); wrapper exit 3. The
   run carried WGPU_COMPUTE_MMV_MAX_ROWS=16, and that line
   is what env.txt and the manifest recorded.
2. `demo-marker`: prints a "degenerate" line, exits 0.
   Bundle 20261106T224352Z-demo-marker assembled (reason:
   marker degenerate x1); wrapper exit 0, the wrapped
   code, per the exit-code rule above. This is the
   top-5 shape: clean exit, wrong data, still vaulted.
3. `demo-pass`: clean output, exits 0. No bundle; one
   PASS line in INDEX.md; wrapper exit 0.

The committed demo bundles are skeletons: their stdout
and stderr logs were stripped before commit, with each
removed log's size and sha256 recorded in the bundle's
LOGS-STRIPPED.txt. Real failure bundles always keep
their logs; the strip is a commit-size rule for the
demo only, and it is labeled in the bundle itself.

shellcheck could not be run on this lane: it is not
installed on the build VM (`command -v shellcheck` is
empty). The script says so in its header and follows the
repo bench style (set -u, everything quoted); run
shellcheck on it from any host that has it.

## Retro-bundle candidates

Historical wrong data that predates the wrapper. None of
it is lost yet, but none of it is in the vault either.
At the next server session, wrap or hand-assemble each
of these into a bundle so they sit under the same
convention:

1. **The degenerate top-5 output (2026-11-05).** Every
   top-5 list held exactly 1 entry and every top-1
   probability was exactly 1.0: the generator requested
   post_sampling_probs at temperature 0, collapsing each
   distribution to a one-hot. Refused at verification,
   never committed. Lives on the server at
   ~/overnight-golden/out/glimmer.top5.degenerate.jsonl;
   the full story is in bench/overnight-golden-2026-11-05.md.
2. **The wide-MMQ session hang (2026-11-04/05).** The
   K15/K16 wide MMQ ladder ran about 3.5x slower per
   tensor than K13/K14 and then hung a real resident
   session: RADV reported the context lost with a hard
   recovery, and no prefill line was ever produced, at
   both bench lengths. Recorded in
   bench/mmq-wide-tile-2026-11-03.md; the compiled-form
   conviction (1,041 to 1,213 spilled VGPRs, 78 to 86 KB
   scratch) is in bench/shader-autopsy-2026-11-04.md,
   with raw dumps under /tmp/shader-autopsy/ on the
   server (ephemeral; may already be gone). The hang's
   own session logs were never bundled. That gap is
   exactly what this wrapper closes.
3. **The K18 loss runs (2026-11-04).** Correct output,
   losing speed, kept deliberately as the negative
   result: depth 6, 8.13 tok/s against the 9.19 control
   (verify 178.0 ms against 149.8); depth 1, 8.90
   against 14.22 (verify 125.1 ms against 55.3). Verdict
   and numbers in bench/small-batch-2026-11-04.md; the
   raw run logs sit in the 2026-11-04 wave staging trees
   (~/gpu-wave*) on the server, exact directory to be
   confirmed when the retro-bundle is assembled.
4. **The phase B OOM kills (2026-11-05).** Six phase B
   attempts across two windows, every one killed by the
   host OOM killer during verify-session construction,
   exit 137, at 34.6 to 34.7 GB anon-rss per kill. The
   accounting that explained them is in
   bench/phase-b-footprint-2026-11-05.md and the verdict
   record in bench/mmv-verdict-2026-11-05.md; the run
   logs are preserved on the server under
   ~/gpu-wave10/logs, ~/gpu-wave10/logs2,
   ~/gpu-wave10/logs3, and the ~/gpu-wave8 outputs.

Each retro-bundle gets a manifest and REPRO written from
its bench note, with the note cited as the source, so a
hand-assembled bundle never pretends to be a wrapped
run.
