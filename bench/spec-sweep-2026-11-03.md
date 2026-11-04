# DFlash depth and acceptance sweep on the new stack (TODO T-9.7): SKELETON, measurements pending

Status: the sweep machinery is landed and tested; the numbers are
not yet measured. This note is the skeleton the server run fills in:
the table below is the exact shape the runner prints, and every
`<pending>` cell is replaced from a direct read of the final run log
(the loop proof's incident rule: no figure from an intermediate
monitoring readout).

Runner: `crates/itak-spec/examples/spec_sweep.rs` (release build),
which is the library sweep (`itak_spec::run_sweep`,
`crates/itak-spec/src/sweep.rs`) wired to a CPU session factory. The
runner is backend-agnostic: it drives sessions built by a
caller-supplied factory, so the GPU-resident loop's lane fills the
speed columns later by wiring the same `run_sweep` to a GPU factory
with the same depths, budget, and prompt. Configuration, all with
defaults, nothing compiled in: the candidate set and per-depth token
budget live in `SweepConfig` (defaults: depths 1, 2, 3, 4, 6, 8, 12,
16, the set the 2026-10-30 llama.cpp sweep measured; 128 tokens per
depth; winner band 0.02), and each session's depth is
`SpecConfig::depth` overridden per depth.

Files (same pair as the loop proof, for comparability):

- Target: `/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`
- Draft: `/mnt/models/Muse-Glimmer-30B/dflash-Muse-Glimmer-30B-Q4_K_M.gguf`
- Prompt: "The capital of France is" (the Stage 1 prompt), temperature 0.

## The sweep table (to be filled from the run log, verbatim)

| Depth | Rounds | Proposed | Accepted | Acceptance rate | Tokens/round | Emitted | Wall (s) | Effective tok/s |
|------:|-------:|---------:|---------:|----------------:|-------------:|--------:|---------:|----------------:|
| 1 | <pending> | <pending> | <pending> | <pending> | <pending> | <pending> | <pending> | <pending> |
| 2 | <pending> | <pending> | <pending> | <pending> | <pending> | <pending> | <pending> | <pending> |
| 3 | <pending> | <pending> | <pending> | <pending> | <pending> | <pending> | <pending> | <pending> |
| 4 | <pending> | <pending> | <pending> | <pending> | <pending> | <pending> | <pending> | <pending> |
| 6 | <pending> | <pending> | <pending> | <pending> | <pending> | <pending> | <pending> | <pending> |
| 8 | <pending> | <pending> | <pending> | <pending> | <pending> | <pending> | <pending> | <pending> |
| 12 | <pending> | <pending> | <pending> | <pending> | <pending> | <pending> | <pending> | <pending> |
| 16 | <pending> | <pending> | <pending> | <pending> | <pending> | <pending> | <pending> | <pending> |

Chosen default depth: **<pending>** (acceptance rate <pending>,
effective <pending> tok/s), picked by the recorded rule: the
smallest depth within 2% of the best effective tok/s (SPEC 9.4).

Closing step once this table is filled: a follow-up commit sets
`SpecConfig::default()`'s `depth` in `crates/itak-spec/src/config.rs`
to the chosen depth, citing this note; that commit is what closes
T-9.7 ("the winning depth recorded as config with its default").

## Anchors the filled table is read against

- llama.cpp on the same files (`bench/gpu-proof-2026-10-30.md`),
  2,304 emitted tokens per cell: acceptance 0.858 at depth 1 falling
  to 0.300 at depth 16 (0.541 at depth 6); operating point depth 6 at
  107.39 tok/s against 41.64 plain. Acceptance is a property of the
  draft and the target at temperature 0, so this stack's acceptance
  column should track llama.cpp's closely; a wide gap is a finding,
  not a shrug.
- This stack's CPU loop proof (`bench/spec-loop-2026-10-30.md`):
  depth 6, 16 tokens, 29 proposed, 9 accepted, acceptance 0.3103.
  Small sample (one prompt, 6 rounds): the sweep's depth-6 cell at a
  larger budget supersedes it, and a disagreement between the two is
  read as small-sample noise first, machinery suspicion second.
- CPU speed expectations, stated before the run so nobody has to
  walk them back after: on the CPU reference backend a block-forward
  row costs roughly 5.5x a single-token step, so no depth is expected
  to beat plain generation on speed here. The CPU run's deliverable
  is the acceptance column and a working sweep; the speed claim
  belongs to the GPU fill-in.

## Exact server command sequence (ubuntu-server)

Staging follows the loop proof's pattern: a fresh directory built
from the merged commit, launched from a script file, argv verified
in `/proc` before any output is trusted, allocator arenas capped
(the loop proof's OOM lesson).

```sh
# 0. Stage the tree at the merged commit (parent lands the lane,
#    then this runs against main's new HEAD).
mkdir -p ~/spec-sweep-2026-11-03
rsync -a --delete <checkout-of-main-at-merge> ~/spec-sweep-2026-11-03/repo/
cd ~/spec-sweep-2026-11-03/repo && git rev-parse HEAD   # record it in this note

# 1. Build the runner.
export PATH="$HOME/.cargo/bin:$PATH"
cargo build --release -p itak-spec --example spec_sweep

# 2. Write the launch script (never a hand-quoted ssh command line).
cat > ~/spec-sweep-2026-11-03/run.sh <<'EOS'
#!/bin/sh
set -eu
cd "$HOME/spec-sweep-2026-11-03/repo"
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=0
exec ./target/release/examples/spec_sweep \
  /mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf \
  /mnt/models/Muse-Glimmer-30B/dflash-Muse-Glimmer-30B-Q4_K_M.gguf \
  default 16 "The capital of France is"
EOS
chmod +x ~/spec-sweep-2026-11-03/run.sh

# 3. Launch detached, then verify the argv before trusting anything.
nohup ~/spec-sweep-2026-11-03/run.sh > ~/spec-sweep-2026-11-03/sweep.log 2>&1 &
PID=$!
sleep 5
tr '\0' ' ' < /proc/$PID/cmdline; echo

# 4. When the process exits, the table is the tail of the log:
tail -20 ~/spec-sweep-2026-11-03/sweep.log
```

Budget note: the arguments above sweep the default candidate set at
16 tokens per depth, the loop proof's budget. Depth 6 alone took
10,231 s at that budget, so the full set is an overnight-class run
(estimated 8 to 12 hours across depths); that is the honest price of
CPU acceptance numbers on the real 17 GB files. A shorter first
confirmation pass, if wanted, is the same script with `6 16` in
place of `default 16`: it should land near the loop proof's 0.3103
acceptance at depth 6. The configured 128-token budget is for the
GPU fill-in, where a depth costs seconds, and for any future CPU run
with a weekend to spare.

## GPU fill-in (when the T-9.6 GPU lane lands)

Same table, same rows: the GPU lane's runner (its own example wiring
`run_sweep` to a GPU session factory) runs with the identical
arguments except the binary path and the budget (`default 128`).
The acceptance column is expected to agree with the CPU column
within float nondeterminism; the effective tok/s column is where
the serving claim is made or lost, against the recorded anchors:
44.31 tok/s plain llama.cpp Vulkan, 107.39 tok/s at depth 6
llama.cpp DFlash.
