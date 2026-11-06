# Spec round economics: current cost map and the host-side trim, 2026-11-06

Lane `lane/spec-round-economics`, based on main `a064b5d7`
(code-identical to `c439e3af`, the tree the verdict session
staged: the diff between them over `crates/` and `platform/`
is empty). Question: what does one speculative round cost
now, per component, and what is left of the T-9.6 gap
(88.6 tok/s bar; best arm 19.26 tok/s at depth 1) that the
spec/draft/session layer can still take?

## Evidence

All numbers come from the 2026-11-06 cleared-card verdict
session's own records, server `~/gpu-wave11/logs11/`:

- `exposure.log`: spec phase B, depth 6,
  `WGPU_COMPUTE_MMV_MAX_ROWS=16`, `WGPU_COMPUTE_DIAG=timing`.
  Its 46k `[wgpu-compute-diag]` per-dispatch records were
  segmented by the `[diag] spec.round.begin`,
  `spec.propose.end`, and `spec.verify.end` markers into
  propose / verify / tail phases. Cross-checks: the propose
  segment's kernel totals reproduce the draft session's own
  aggregate (`exposure.csv`, the surviving report) to the
  dispatch once the tail-phase inject chain and the initial
  conditioning inject are added back (1,299.8 + 14.8 + ~1.3
  = 1,315.9 ms); the verify segment's dispatch count plus
  the one prefill forward lands within one dispatch of the
  target session's reported 14,717.
- `on-d1.log` / `on-d6.log` / `off-d1.log` / `off-d6.log`:
  the timing-off A/B arms (per-round `draft_ms` /
  `verify_ms` walls, session counters, verdict lines).
- `diag-plain.csv`: the plain `glimmer_generate` run's
  per-kernel aggregate (single-row GEMV streaming rates).

## The cost map

### Depth 6 (exposure run; round wall 166.0 ms = 63 tokens /
32 rounds / 11.86 tok/s effective)

| phase | wall ms | GPU ms | share |
| --- | ---: | ---: | ---: |
| draft (propose) | 47.8 | 40.6 | 28.8% |
| verify | 114.2 | 102.0 | 68.8% |
| tail (walk + commit) | 4.0 (residual) | 0.5 | 2.4% |

Draft GPU, per round (the block is 16 positions at every
depth; see F3):

| kernel | dispatches | ms | share | weights | eff. rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| k3_q5k_gemv (head) | 1 | 17.21 | 42.4% | 924.6 MB | 53.7 GB/s |
| k1_q4k_gemv | 25 | 14.00 | 34.5% | 920.1 MB | 65.7 GB/s |
| k2_q6k_gemv | 10 | 8.77 | 21.6% | 578.5 MB | 66.0 GB/s |
| k17_dflash_attend | 5 | 0.36 | 0.9% | - | - |
| norms / rotate / swiglu / gather | 47 | 0.27 | 0.7% | - | - |

Verify GPU, per round (7 rows):

| kernel | dispatches | ms | share |
| --- | ---: | ---: | ---: |
| k19_q4k_mmv | 100.75 | 36.95 | 36.2% |
| k1_q4k_gemv (unrouted) | 81.25 | 26.60 | 26.1% |
| k2_q6k_gemv (unrouted) | 26.81 | 25.88 | 25.4% |
| k3_q5k_gemv (head, fused finish) | 1 | 7.55 | 7.4% |
| k4_norm_chain | 103 | 2.77 | 2.7% |
| k6_attn_flash | 52 | 1.65 | 1.6% |
| k20_q6k_mmv | 25.19 | 0.33 | 0.3% |
| rest | ~55 | 0.24 | 0.2% |

Timing-off cross-check (`on-d6.log`): round 191.7 ms =
draft 64.0 + verify 123.6 + residual 4.1. The draft wall is
the run-variable part (exposure: 47.8, tight 46-49; on-d6:
mean 64.0, spread 53-71); the kernel totals are the stable
part of the map.

### Depth 1 (the default; `on-d1.log`, timing off)

Round wall 86.1 ms = 63 tokens / 38 rounds / 19.26 tok/s.

| phase | wall ms | share | basis |
| --- | ---: | ---: | --- |
| draft (propose) | 45.7 | 53.1% | wall per round; GPU ≈ 40.6 (F3) |
| verify | 38.8 | 45.1% | wall per round (2 rows) |
| residual | 1.6 | 1.9% | round wall minus the two walls |

Control (`off-d1.log`, MMV unset): 97.6 = draft 48.3 +
verify 47.1 + residual 2.2. No timing-on depth-1 phase-B
run exists, so the depth-1 verify has no per-kernel split;
its structure is the depth-6 table's, at 2 rows.

## Findings

**F1. The round is a small-batch weight-streaming problem
on both sides.** The draft streams 2,423 MB of weights per
round in the GEMV batch form at 54-66 GB/s. Verify's
unrouted GEMV share (the gated projection and the SwiGLU
pair, which do not route through `prepare_proj`) costs
52.5 ms at depth 6, and K19, already routed, is still
verify's largest single item at 36.9 ms. Against all of
that, the same tensors in single-row GEMV form (plain
decode, `diag-plain.csv`) stream at 648.5-868.6 GB/s. The
card's ceiling is 960 GB/s; nothing in the round comes
close except the form that cannot batch.

**F2. The single largest kernel cost at the default depth
is the draft head:** one Q5_K GEMV dispatch, 17.21 ms,
42.4% of draft GPU and about 20% of the whole depth-1
round, streaming 924.6 MB at 53.7 GB/s. Plain decode
streams the same tensor shape at 868.6 GB/s (1.06 ms per
token, `diag-plain.csv`). 

**F3. The draft's GPU cost is depth-independent.**
`forward_block` always runs the full 16-token block
(`speculator.rs`: the block is `[pending, mask x 15]`
regardless of depth), so the depth-6 draft kernel table is
also the depth-1 draft's, and the depth-1 round pays the
same ~40.6 GPU ms to obtain one candidate. Truncating the
block is not available: block attention is non-causal, so
a shorter block changes the earlier rows' distributions
(settled in `propose_block`'s docs and in the fixture's
`causal_attention: false`).

**F4. The big costs belong to the existing kernel lanes;
this lane documents and does not duplicate them.** The
draft's projections (F1/F2) want MMV routing inside
`dflash_session.rs`'s own `prepare_proj`: the K19/K20
kernels exist, and the draft head is a plain Q5_K
projection with no fused finish, so the K21 kernel shape
fits it directly (K21's known gap is only the finish the
target's head needs). Verify's unrouted share wants the
same routing extended to `prepare_gated_proj` and
`prepare_swiglu_proj`, and the target head (7.55 ms at
depth 6) wants the K21 finish variant. Both are named
follow-ups of the MMV/K21 lanes with protocols already
written; the numbers above are their sizing.

**F5. The round's synchronization is already minimal.**
Per round: the draft forward is one submission plus one
blocking map; the verify forward is one submission plus
its maps (logits and captures drain together); the commit
is one inject submission with no map. `fuse()` is not on
the GPU commit path (it is the CPU model's public seam
and a parity-test entry; the inject plan encodes the
fusion prefix on-device). Two effective drains per round
is the floor for this sequential loop.

**F6. Host CPU work is the only in-layer remainder, and
two of its three pieces are now trimmed (below).**
Measured on the VM (indicative magnitudes; the server CPU
is faster): the readback byte conversion in `map_staging`
costs 0.61 ms for a full block and is already
memcpy-class (a "faster conversion" variant was
benchmarked and was slower; that hypothesis is dead). The
scalar `sampler::argmax` costs ~0.26 ms per 202,048-wide
row and runs on 1 + (depth + 1) rows per round (propose
rows plus the correction walk's rows). And the draft's
full-block readback moves 3,232,768 values (12.93 MB) per
round while every consumer slices only rows `1..=depth`
(`propose_block`); rows past the slice are computed
(unavoidable, F3) but were also transferred, converted,
and cached for nothing.

**F7. The bar is not reachable from this layer.** At
depth 1, 88.6 tok/s needs an 18.7 ms round (1.66
tokens/round). Draft + verify GPU alone measure ~75 ms.
T-9.6 stands or falls with the kernel lanes' streaming
rates (F4); the trim below is worth a few percent, taken
because it is free.

## What this lane built: the host-side trim

Two changes, both pure overhead removal with identical
outputs, so neither carries a gate (the shipped behavior
is not an arm of either; there is nothing to arm):

1. **Laned `sampler::argmax`** (`crates/itak-spec/src/sampler.rs`).
   Eight interleaved accumulators break the scalar scan's
   dependency chain; the merge applies the scalar rule
   (strictly greater wins; equal value goes to the lower
   index), which keeps the result bit-identical to the old
   scan on every input, NaNs, infinities, signed zeros,
   and denormals included (a NaN never wins a `>`
   comparison, so no accumulator can hold one). Callers:
   the propose slice, the correction walk, the plain step.
   Pinned by `argmax_matches_the_scalar_reference`: the
   pre-change scan kept verbatim in the tests, compared
   over hand-built adversarial rows and a seeded sweep at
   14 lengths including the real 202,048 width. VM
   micro-benchmark (noisy, direction only): 7-row walk
   4.3 ms scalar vs 1.9-2.9 ms laned.

2. **Capped draft block readback**
   (`crates/wgpu-compute/src/dflash_session.rs`,
   `crates/itak-spec/src/speculator.rs`).
   `DflashSession::forward_block_capped` runs the
   identical block (same plan, same submission) and copies
   / maps only the first `row_cap` logit rows;
   `forward_block` delegates to the same path uncapped and
   is byte-identical to before. The speculator computes
   its slice bound (`depth + 1`, clamped by `max_drafts`)
   before the forward and passes it down on the GPU arm;
   the CPU arm computes in-process and is untouched. A cap
   outside `1..=rows` refuses by name
   (`dflash forward_block` step) before any device work;
   the resolution rule is the pure, unit-tested
   `capped_rows`. Per round the readback falls from
   3,232,768 values (12.93 MB) to (depth + 1) rows: depth 1
   404,096 values (1.62 MB), depth 6 1,414,336 (5.66 MB).
   Instrumentation: the `[dflash] forward_block.capped`
   diag event (rows / returned) fires whenever a cap
   bites, and `DflashSessionStats::logits_values_read`
   counts the values that actually crossed, so the server
   A/B reads engagement off the logs instead of trusting
   the wiring. The device test
   `dflash_session_capped_forward_returns_the_full_prefix`
   pins that a capped result is the full forward's exact
   prefix at every cap, plus both refusals.

Expected effect (estimate; the protocol below replaces it
with measurements): depth 1 about 1.5-2.5 ms per round
(~2-3%), depth 6 about 3-5 ms per round (~2-3%), from the
smaller transfer and conversion, the propose argmax over
a cache-resident prefix, and the laned walk.

## VM gates (this lane)

- `cargo test -p itak-spec`: green, 48 tests, including
  `argmax_matches_the_scalar_reference`.
- `cargo test -p wgpu-compute`: green across every binary
  (193 tests in 23 binaries; lib: 102, including
  `capped_rows_accepts_only_true_prefixes`).
  The device tests, the new capped-forward one included,
  skip with a printed reason on this adapterless VM; their
  execution is step 1 of the server protocol.
- `cargo test -p dflash`: green, 24 tests.
- `cargo clippy -p itak-spec -p wgpu-compute -p dflash
  --all-targets`: clean. `cargo fmt --all -- --check`:
  clean.

## Server A/B protocol (parent runs; card time serialized)

Staging: `git archive` the lane head to `~/gpu-wave12` on
ubuntu-server; manifest-diff against the local tree. The
control is main `a064b5d7` (code-identical to the
`~/gpu-wave11` staging; a fresh staging is equally fine).
Build in each tree:
`cargo build --release -p itak-spec --example spec_generate_gpu`
and `cargo test -p wgpu-compute --test dflash_session --release --no-run`.
Serialize every run through
`flock -w 14400 /tmp/itak-phase.lock`; before each arm,
guard `/sys/class/drm/card2/device/mem_info_vram_used`
at driver residue (~33,271,808 B; stop the leg if above
1 GB). Never reboot; never touch the production
llama-server.

1. Device test on the real adapter, lane tree:
   `cargo test -p wgpu-compute --test dflash_session --release dflash_session_capped_forward_returns_the_full_prefix`
   must execute (not skip) and pass.
2. Arms, both trees, the verdict session's own invocation
   (models `/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`
   and `dflash-Muse-Glimmer-30B-Q4_K_M.gguf`):
   `WGPU_COMPUTE_MMV_MAX_ROWS=16 spec_generate_gpu <target> <draft> 64 <depth> "The capital of France is" both <stream>`
   at depths 1 and 6, three repetitions each, median taken.
3. Gates per run: `IDENTITY: PASS` and `GOLDEN: PASS` in
   the harness output, and the two arms' spec streams
   token-identical to each other (the harness prints the
   stream; the stream files diff clean).
4. Engagement proof, lane arm only: the log carries
   `[dflash] forward_block.capped rows=16 returned=2`
   (depth 1) / `returned=7` (depth 6) once per round, and
   no such line appears in the control arm.
5. Read: median `draft_ms` and effective tok/s per depth
   against this note's estimate (depth 1 draft down
   ~1.5-2.5 ms; effective up ~2-3%). `verify_ms` is the
   control's twin within noise; if it moves, suspect the
   run, not the change (the change touches no verify code).
6. One MMV-unset pair at depth 1 as a sanity arm: same
   gates, same capped lines.
