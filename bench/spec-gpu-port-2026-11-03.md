# Speculative loop on the GPU-resident stack: the T-9.6 port

Date: 2026-11-03. Branch: `lane/spec-gpu` (base 31634aa). Status:
the port is built and locally gated; the T-9.6 verdict itself is
**not yet proven**, because the two proofs that need a real GPU
(the resident block tests at full execution and the server runner
below) have not run. This note records the design, what is proven
where, the one hard blocker on the speed bar, and the exact server
run still owed. T-9.6 stays open until that run lands.

## What was built

Four commits on the lane branch:

1. `eb72e86` wgpu-compute: resident block logits, layer captures,
   and bounded rewind (session.rs).
2. `62d94dd` wgpu-compute: resident block, capture, and rewind
   tests (tests/resident_block.rs). They do not execute on this
   VM: it enumerates no suitable adapter, so they take the skip
   path here (verified with `--nocapture`: "no suitable adapter
   (enumerated: [])"). They execute on a host with a qualifying
   adapter, the server.
3. `3cc9406` itak-spec: the TargetDriver seam, CpuTarget,
   ResidentTarget, and the stub-driver loop tests.
4. The runner and this note (the commit carrying this file).

## Design decisions

**The loop is generic over a target driver seam; nothing else
moved.** `crates/itak-spec/src/driver.rs` names exactly the
operations the loop performs (config, position, forward,
forward_capturing, forward_block_capturing, checkpoint, rewind,
release). `CpuTarget` wraps the original CPU path unchanged
(model, shared compute, position cache; checkpoint/rewind/release
order preserved), and every pre-existing session test passes
unmodified in behavior (33 tests). `ResidentTarget`
(`resident.rs`) drives the existing `ResidentSession`; the draft
side (`Speculator` seam, `DflashSpeculator`) is untouched.

**The verify forward is the existing resident chain, never a
second forward.** Three additions to `ResidentSession`, all on
the one chain:

- *Block tail.* A verify chunk runs a standalone final norm over
  every row of stream A (the chain leaves every row's final
  residual there; the last layer's close writes it for all rows
  and only additionally emits the standard head input for the
  last row), then the head at the chunk's batch. The head keeps
  the standard head's arithmetic family at any batch: a Q5_K
  head takes K3's fused finishing at batch = rows; Q4_K/Q6_K
  heads take their GEMV prepares plus the K8 finish over the
  flattened rows, instead of `prepare_proj`'s matrix forms,
  because the head is a GEMV at the only size the standard path
  runs (one row) and the block tail must not change families as
  the batch grows. Dispatch accounting swaps the standard head
  segment's share of the plan's count for the block segment's.
- *Captures.* A capturing chunk encodes the chain as one compute
  pass per layer range, with an encoder copy of stream A into a
  capture staging at each named layer's entry (every layer
  enters with its residual in stream A; layer 0's input is the
  post-norm stream, exactly what the CPU model's
  `forward_capturing` returns). Copies are not dispatches and
  are not counted, consistent with the existing readback copies.
  Captures merge across chunks in position order, matching the
  model's contract.
- *Rewind.* `rewind_to(position)` restores the position counter;
  entries past it are never read (views derive from position)
  and are overwritten positionally. Soundness bound, derived
  from the ring geometry: windowed slots are `position % ring`
  with ring = window + MAX_CHUNK_ROWS - 1, so a discarded write
  can alias a committed key inside a later query's window only
  when the discarded span exceeds MAX_CHUNK_ROWS (64). The
  method refuses spans over 64 by name (`session` step);
  `reset()` remains the full-rewind path. A speculative round
  discards at most depth + 1 (default 17), well inside the bound.
  On this driver a checkpoint is therefore just the position at
  checkpoint time and release is a no-op; `resident.rs` says why
  that is sound rather than lazy.

**The draft has no GPU path; it stays on CPU behind the same
seam.** `crates/dflash`'s `DflashModel` is its own CPU
implementation and does not consume the `Compute` trait, so
there is no existing GPU draft to wire. Per the tasking, the
port drives it on the best existing backend (CPU) behind the
unchanged `Speculator` seam and names the consequence below
instead of inventing a draft GPU path inside this item.

**Identity on GPU is empirical, not construction-exact.** On CPU
the speculative stream is token-identical by construction (same
code path). On GPU, identity rests on a multi-row chunk agreeing
with sequential one-row forwards at argmax resolution; the
resident cross-path tests bound that drift at about 1.1e-6 on
logits. One configuration rule follows: chunks of 16+ rows take
the MMQ path, whose Q8_1 activation quantization makes logits
approximate by design (measured band 0.36 to 0.42 percent of
result scale). At the default depth 6 a verify block is 7 rows,
so the proof configuration stays on the GEMV chain end to end.
The runner measures identity on the real files; nothing here
claims it by construction.

## The speed blocker, stated plainly

The T-9.6 bar is effective tok/s at or above 88.6. The verify
side can plausibly reach it (the llama.cpp sweep's depth-6
operating point, 107.39 tok/s at acceptance 0.541, implies the
shape works when drafting is cheap). This build's drafting is
not cheap: the CPU draft block forward measured 9.9 s for a
4-row block on the server (`bench/dflash-e3-2026-10-30.md`), and
spec-loop rounds measured draft time at roughly 31 to 35 s per
16-row block. Against verify rounds that cost tens of
milliseconds on the resident GPU, a CPU draft at seconds per
block puts the effective rate orders of magnitude under the
bar. **Expect the server run to print SPEED: FAIL.** The
identity half of T-9.6 is fully provable by the same run. The
named missing component is a GPU path for the DFlash draft
model (its own item; T-9.7's depth/acceptance sweep should run
on this stack after it exists, since the winning depth depends
on the draft's real cost). The runner's `--bar` flag exists so
that future runs can judge against a revised bar without a code
change; the default stays the recorded 88.6.

## What is proven locally (this VM, no qualifying GPU adapter)

Commands run from the lane worktree with
`CARGO_TARGET_DIR=$PWD/target`:

- `cargo test -p itak-spec`: 35 passed, 0 failed. The 33
  pre-existing tests (greedy identity vs plain at depths 1/2/4/6,
  sabotage refusal and latch, creation refusals, call order) pass
  through the refactored seam unchanged. The 2 new stub-driver
  tests prove the loop through the seam without model
  arithmetic: greedy identity between a DFlash session and a
  mode-off session over a deterministic stub driver, and
  per-round rewind targets landing exactly on base + 1 +
  accepted with checkpoint/release pairing (one each per round)
  and cumulative stats equal to the summed round outcomes.
- `cargo test -p wgpu-compute`: whole suite green. The two new
  `resident_block` tests (block logits and captures vs the CPU
  model's block forward at the cross-backend bar; rewind
  soundness and both refusal shapes) **compile and skip** here:
  "no suitable adapter (enumerated: [])". The pre-existing GPU
  tests skip the same way on this VM; nothing GPU-side executed
  locally. This is the honest limit of the local proof.
- `cargo clippy -p wgpu-compute --all-targets` and
  `cargo clippy -p itak-spec --all-targets`: clean (workspace
  lints deny warnings). `cargo fmt` applied per crate.
- The runner builds (`cargo build -p itak-spec --example
  spec_generate_gpu`) and its usage path exits 2 with the usage
  line, mirroring the CPU runner.

## The server run still owed (exactly)

Preconditions: the server GPU free (parent clears it), the lane
branch landed on the server tree, release build.

```sh
export PATH=$HOME/.cargo/bin:$PATH
export CARGO_TARGET_DIR=$PWD/target
cargo build --release -p itak-spec --example spec_generate_gpu
./target/release/examples/spec_generate_gpu \
  /mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf \
  /mnt/models/Muse-Glimmer-30B/dflash-Muse-Glimmer-30B-Q4_K_M.gguf \
  64 6
```

Also owed on the server, where the adapter qualifies:
`cargo test -p wgpu-compute --test resident_block` (both tests
must execute, not skip).

What the run must show to close T-9.6's identity half:
`IDENTITY: PASS` (spec stream token-identical to the plain
resident GPU stream at temperature 0) and `GOLDEN: PASS` (first
8 tokens [13796, 26, 1573, 373, 290, 2167, 26899, 1212]).
The combined line will read "CORRECTNESS PASS, SPEED FAIL"
until a GPU draft path exists; the exit code follows the
correctness contract, the CPU runner's convention, so the two
halves are never confused. If IDENTITY fails, the first
divergence position prints with both streams, and the suspects
in order are: block-tail arithmetic family drift at 7 rows,
capture row misalignment feeding the draft, and ring aliasing
inside the rewind span (the span here is at most 7, far inside
the proven bound, so this should be last).

## Files in this port

- `crates/wgpu-compute/src/session.rs`: block tail buffers and
  plan segment, segmented capturing encode, `forward_capturing`,
  `forward_block_capturing`, `rewind_to`, module docs.
- `crates/wgpu-compute/tests/resident_block.rs`: the block,
  capture, and rewind pins.
- `crates/itak-spec/src/driver.rs`: the `TargetDriver` trait,
  `CpuTarget`, and the stub-driver loop tests.
- `crates/itak-spec/src/resident.rs`: `ResidentTarget`.
- `crates/itak-spec/src/session.rs`: `SpecSession` generic over
  the driver (default `CpuTarget`); driver accessors.
- `crates/itak-spec/src/lib.rs`, `src/testkit.rs`,
  `examples/spec_generate.rs`: module wiring and call sites.
- `crates/itak-spec/Cargo.toml`, `Cargo.lock`: the wgpu-compute
  dependency edge.
- `crates/itak-spec/examples/spec_generate_gpu.rs`: the server
  proof runner.

## Server verdict (2026-11-04, RX 7900 XTX, main 864dffd + the K17 brace fix)

Ran on the real card against the real files, greedy, 64 tokens,
Stage 1 prompt. Plain arm: first 8 tokens
[13796, 26, 1573, 373, 290, 2167, 26899, 1212], 35.66 tok/s decode.

- CPU-draft arm: IDENTITY PASS, GOLDEN PASS. Acceptance 0.172
  through the shared conditioning path.
- GPU-draft arm: IDENTITY PASS, GOLDEN PASS. Effective 4.79 tok/s.
- SPEED: FAIL. 4.79 against the 88.6 bar.

Root cause is one defect, caught independently by the
dflash_session parity gate: the GPU draft's block forward diverges
from the CPU reference (fixture parity fails at block row 0,
argmax 4 vs 248; the fuse stage passes the drift bar and the inject
bookkeeping matches, so the divergence sits in the per-layer chain
or the K17 attention). Acceptance collapses to 0.003 (1 of 351),
and with no accepted drafts the loop pays verify cost for nothing.
Per-round cost with a working draft is fine (62 rounds in about
13.2 s of generate). T-9.6 stays OPEN: correctness is proven on
both arms; the speed half waits on the draft parity fix.

## Draft parity fix and re-verdict (2026-11-04, RX 7900 XTX)

The divergence is found and fixed. Root cause: BlockPlan cached a
single attention-call index, overwritten per layer, so only the
LAST draft layer's attention call received the per-block runtime
values (block start, committed count). Every earlier layer
attended over zero committed positions at block start 0; the
committed KV was written correctly, the earlier layers never
looked at it. Bisection proof: the fixture passes in full at 1
layer and fails at 2. The fix stores every layer's attention index
and rewrites all of them per block (dflash_session.rs, one file).

After the fix: the dflash_session parity gate is 4 of 4 green on
the card, and on the real files the GPU draft's bookkeeping equals
the CPU draft's exactly (32 rounds, 180 proposed, 31 accepted,
acceptance 0.172 in both arms). IDENTITY PASS, GOLDEN PASS.
Effective speed 9.22 tok/s: SPEED STILL FAILS the 88.6 bar. With
a correct draft the blocker is economics, not correctness: at
depth 6 acceptance 0.172 yields 1.97 tokens per round against a
full target verify forward per round, and the loop trails plain
decode (35.45 tok/s the same night). T-9.6 stays OPEN on its
speed half: acceptance economics and verify cost are the work now.

## Round economics (2026-11-04, RX 7900 XTX, main 72a4b9d)

The round is now fully accounted for. Per round at depth 6, from
the loop's own diag: draft 62.1 ms, verify 151.0 ms, residual
about 4 ms (checkpoint, correction walk, rewind, emission).
Submissions, readbacks, captures and rollback are all cleared
with numbers; the 5.66 MB block-logits readback is about 1 ms.

The cost is the projection form. Below GEMM_MIN_ROWS (16),
projections run GEMV: one full weight stream per row. The probe
fits rows x 16 GB at about 815 GB/s exactly (1 row 30 ms, 2 rows
47 ms, 7 rows 140 ms), so a 7-row verify pays seven streams. The
draft's 16-row block has the same disease at smaller scale
(57.3 ms). Lowering the threshold to 2 was tested and killed with
data: 7-row forwards got slower (140 to 253 ms) and the MMQ
dispatch count roughly doubled, because the existing GEMM tiles
are built for 32 and 64 rows. What is missing is a purpose-built
small-batch matrix kernel that streams the weights once for
2 to 16 rows; that is also the prefill chain's missing piece.

One real cut landed: the greedy propose path built vocab-width
one-hot vectors for every proposed token on every round (5.66 MB
of zeros, 2.04 ms of host time) to back a trait view the loop
never reads. The greedy arm now stores no distributions
(speculator.rs). Re-verdict after the cut: depth 6 effective
9.21 tok/s, depth 1 effective 14.60 tok/s, IDENTITY and GOLDEN
PASS on both. SPEED STILL FAILS the 88.6 bar, with the blocker
now precise: about 150 ms verify plus about 58 ms draft per
round, both small-batch projection cost.
