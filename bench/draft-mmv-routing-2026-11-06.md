# Draft MMV routing: the DFlash session walks the band ladder, 2026-11-06

Lane `lane/draft-mmv-routing`, based on main `d3adec4`.
Question (from bench/spec-round-economics-2026-11-06.md,
F1/F2/F4): the speculative round's draft block forward
costs ~40.6 ms of GPU per round at 54-66 GB/s because
the draft session's projections never routed through
the MMV band policy; the single fattest dispatch is the
draft's Q5_K head, 17.21 ms streaming 924.6 MB at
53.7 GB/s. This lane routes the draft's projections
through the resident ladder. Nothing here changes a
shipped default.

## What was built

`crates/wgpu-compute/src/dflash_session.rs`:

- `prepare_proj` now walks the same ladder the
  resident session walks below its matrix threshold:
  the register-streaming MMV products first (K19 for
  Q4_K, K20 for Q6_K, K21 for Q5_K), the K18
  small-batch products behind them, the chunked GEMV
  form behind those, and the plain K1/K2/K3 GEMV last.
  A rung whose prepare refuses falls through to the
  next rung with the refusal logged by name, the
  resident ladder's own discipline. The matrix forms
  are the one resident rung the draft does not take:
  its row counts are block and segment counts, and
  the MMQ forms' activation quantization is a prefill
  tool (the dflash module docs carry the updated
  composition paragraph).
- The resolution is one pure function,
  `proj_candidates`, over the resolved policy values
  (the backend's `MmvBands` table, the
  `WGPU_COMPUTE_MMV_MAX_ROWS` ceiling, the small-batch
  ceiling, the chunked ceiling). Its expressions are
  the resident predicates' own (`MmvBands::routes`
  with `mmv_shape_of` classification, the batch bounds
  of `session::small_batch_applies` /
  `session::chunked_gemv_applies`, and
  `session::GEMM_MIN_ROWS` itself, now crate-visible)
  and `prepare_proj` walks exactly the list it
  returns. The ceiling-admits-but-cell-excludes case
  logs the same fall-through line the resident ladder
  logs, so an A/B sees the exclusions it is made of.
- The draft head is a plain Q5_K projection with no
  fused finish, so it routes through the Q5_K band
  cell to the existing K21 kernel directly; the
  finish variant the target's head needs is a
  separate lane's work and is not touched here.
- The K18 rung needs the K12 quantize staging: the
  draft `Scratch` gains a `q8_qs`/`q8_ds` pair sized
  at the largest plan rows and the widest projection
  input, the resident scratch's own sizing rule. The
  pair is allocated unconditionally (about 3 MB at
  the Glimmer draft's geometry) and read only when
  the small-batch ceiling opens the rung.
- Gating: the global gate is the same
  `WGPU_COMPUTE_MMV_MAX_ROWS`; the per-cell env
  overrides (`WGPU_COMPUTE_MMV_*_ROWS`) apply to the
  draft unchanged, because the draft resolves through
  the backend's one table. No new env flags.
- The capped readback (`forward_block_capped`,
  6d260e2) is untouched: routing changes plan
  construction only; the same plan encode and
  readback code serve capped and uncapped forwards.

Supporting, behavior-neutral changes:
`compute.rs` gains a `mmv_bands()` accessor (and the
test-only `default_mmv_bands` is crate-visible so the
draft tests pin against the shipped table itself);
`session.rs` makes `GEMM_MIN_ROWS` and `pow2_chunks`
crate-visible. The resident path's code is otherwise
byte-identical.

Disclosure: the draft module docs used to claim
cross-backend parity "by construction" from staying
on the GEMV family. That claim is retired in the
module docs: routed draft projections now carry the
same tolerance-class guarantee the resident routing
carries, and the card's identity/golden gates are
the arbiter, same as for the resident session.

## Which dispatches change route

At shipped defaults, none: every ceiling is 1, the
candidate list is empty at every batch, and the GEMV
form serves exactly as before (pinned by
`draft_ladder_ships_closed_at_default_ceilings`).

With `WGPU_COMPUTE_MMV_MAX_ROWS=16` (the setting the
consolidated session already runs), at the block's
constant 16 rows:

| dispatch | per round | before | after | why |
| --- | ---: | --- | --- | --- |
| Q5_K head | 1 | k3 GEMV, 17.21 ms | **k21 MMV** | Q5_K cells admit 16 in every shape class; the head is FFN class (6656 in, 202048 out) and its cell is 3-16 |
| Q4_K layer projections | 25 | k1 GEMV, 14.00 ms | k1 GEMV (unchanged) | shipped Q4_K cells end at 12; the sweep never measured MMV at 16 (MMQ served both arms) |
| Q6_K layer projections | 10 | k2 GEMV, 8.77 ms | k2 GEMV (unchanged) | shipped Q6_K cells end at 12, same reason |
| inject fc + K/V | per segment | GEMV | GEMV at the usual 64-row segments; MMV/K18/chunked resolve per policy at tail segments of 2..15 rows | segment counts are the batch; 64 is outside every rung |

The Q4_K/Q6_K stay is policy, not a kernel limit:
K19/K20 accept batch 16, and an operator can widen
the cells by env (`WGPU_COMPUTE_MMV_Q4K_ATTN_ROWS`
and friends) if a future sweep measures those cells.
This lane does not widen them on inference.

## Expected recovery, stated honestly

The head is the only block dispatch whose route
changes, so the recovery is the head's alone.

- The sweep's Q5_K FFN ratio at 16 rows is 0.494
  (bench/mmv-band-routing-2026-11-06.md), which puts
  the head at ~8.5 ms, saving ~8.7 ms.
- K21's absolute streaming on the sweep's shapes is
  64-112 GB/s, not the 648-869 GB/s single-row GEMV
  streams at. The head's output is 10x wider than the
  sweep's FFN shape, which amortizes fixed costs
  better; on rate grounds the head lands at ~5-9 ms,
  saving ~8-12 ms of the 17.21.
- The often-quoted bound (the head at plain decode's
  868.6 GB/s = 1.06 ms, saving ~16 ms) is a ceiling
  K21 has never measured at on any shape. Do not
  plan on it.

At the saving's range: draft GPU 40.6 ms falls to
~29-33 ms; the depth-1 round (86.1 ms wall, verify
unchanged) falls to ~74-78 ms, worth roughly
21-22 tok/s against the current 19.26 best arm. The
depth-6 draft wall falls by the same absolute amount
(the draft is depth-independent, F3).

## VM gates (this lane)

- `cargo test -p wgpu-compute`: green across every
  binary (lib 112, including the four new pins:
  `draft_ladder_ships_closed_at_default_ceilings`,
  `draft_head_routes_to_mmv_at_block_batch`,
  `draft_block_batch_leaves_q4k_q6k_on_the_lower_ladder`,
  `draft_ladder_orders_rungs_and_bounds_the_lower_ones`).
  Device tests skip on this adapterless VM as usual;
  their execution is the server protocol's step 1.
- `cargo test -p dflash`: green, 24 tests.
- `cargo test -p itak-spec`: green, 48 tests.
- `cargo clippy -p wgpu-compute -p dflash -p itak-spec
  --all-targets`: clean (two `too_many_arguments`
  allows added on the widened signatures, the resident
  `prepare_proj` precedent). `cargo fmt --all --
  --check`: clean.

## Server protocol (fold into the consolidated card session)

Staging: `git archive` the lane head to
`~/gpu-wave13` on ubuntu-server; manifest-diff
against the local tree. Control: current main.
Serialize every run through
`flock -w 14400 /tmp/itak-phase.lock`; VRAM guard at
driver residue (~33,271,808 B) before each arm;
tenants stopped and restored per the standing rule.
Never reboot; never touch the production llama-server.

1. Device tests on the real adapter, lane tree,
   release: `cargo test -p wgpu-compute --test
   dflash_session --release` must execute (not skip)
   and pass in full, including
   `dflash_session_capped_forward_returns_the_full_prefix`
   (the capped readback over routed plans).
2. Arms, both trees:
   `spec_generate_gpu <target> <draft> 64 <depth>
   "The capital of France is" both <stream>` at
   depths 1 and 6, three repetitions, median taken,
   under (a) MMV unset and (b)
   `WGPU_COMPUTE_MMV_MAX_ROWS=16`, both with
   `WGPU_COMPUTE_DIAG=timing`.
3. Gates per run: `IDENTITY: PASS` and
   `GOLDEN: PASS`; spec streams token-identical
   across arms and trees.
4. Exposure proof, lane arm (b) only: the draft
   session's own diag report shows `k21_q5k_mmv`
   records inside draft forwards, exactly one per
   block forward (one per round), and the run log
   carries the `mmv q4k ... excludes batch 16` /
   `mmv q6k ... excludes batch 16` fall-through
   lines for the layer projections. No `k19_q4k_mmv`
   or `k20_q6k_mmv` records may appear in the draft
   phase at cap 16 under the shipped cells: if they
   do, the band resolution differs from the unit
   pins and the run is suspect, stop and say so.
5. Attribution control, lane tree, depth 1:
   `WGPU_COMPUTE_MMV_MAX_ROWS=16
   WGPU_COMPUTE_MMV_Q5K_FFN_ROWS=1` (the head's cell
   closed): gates as above, and `draft_ms` returns
   to the cost map's values while arm (b) shows the
   saving. That isolates the head as the mover.
6. Read: median `draft_ms` per depth against the
   cost map (draft wall ~45.7 ms at depth 1, GPU
   40.6): expect down 8-12 ms at both depths;
   `verify_ms` unchanged within noise (this lane
   touches no verify code); effective tok/s at
   depth 1 from 19.26 toward ~21-22.
