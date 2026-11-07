# MMV global gate: shipped default flipped from 1 (off) to 16 (2026-11-06)

Lane: lane/mmv-default-flip, based on main 2c1f8c7.
Decision: David, 2026-11-06, on the baseline session's
card verdict. This note records the evidence, the exact
semantic change, and the re-pinned tests. No server GPU
jobs ran for this change; the card evidence is the
baseline session's, cited below, and the merge carries
the standard obligation to re-run the resident and spec
suites on the card against the new default.

## The evidence

bench/baseline-session-2026-11-06.md leg (d) ran the
band routing A/B on the cleared card (spec_generate_gpu,
64 tokens, both mode, diag timing; control = MMV unset,
caps 4/8/12/16, depths 1 and 6, no per-cell overrides):

- Effective tok/s at cap 16: depth 1, 21.73 against the
  17.39 control; depth 6, 16.33 against 10.14. Cap 16
  also beat the recorded flat-routing numbers (19.26 at
  depth 1, 10.27 at depth 6).
- Every routed cap at or above the same-session control
  at both depths. The flat-routing cliff does not recur
  under the band table: caps 4/8/12 at depth 1, which
  flat routing put at 14.71/15.23/15.20 against a 16.98
  control, all beat this session's control.
- Verify medians at or under control at every cap
  (depth 1: 41 ms routed vs 51 control; depth 6: 82/82/83
  at caps 8/12/16 vs 153 control).
- IDENTITY and GOLDEN PASS in all 10 arms; round counts
  and acceptance identical across arms at each depth;
  the spec stream byte-identical in every arm.
- Sweep re-run: 0 disagreeing cells in all four arms,
  and the mmv arm's kernel column matches the shipped
  band table in 99 of 99 cells.

Leg (e) of the same session priced the draft side at
the same ceiling: exposure exactly one k21_q5k_mmv per
round at cap 16, draft medians down 10 ms at both
depths (42 to 32, 44 to 34), attribution arm returning
draft time to exactly the unset value when the Q5_K
FFN cell is closed.

## The exact semantic change

One resolver, one constant: in
crates/wgpu-compute/src/compute.rs,
`DEFAULT_MMV_MAX_ROWS` moves from 1 to 16, so
`WGPU_COMPUTE_MMV_MAX_ROWS` unset now resolves to the
kernels' full band instead of closing the gate.

Everything else is deliberately untouched:

- Any set value wins exactly as before. In particular
  `WGPU_COMPUTE_MMV_MAX_ROWS=1` still closes the gate,
  and intermediate values still clip the band from
  above.
- A present but unparsable value keeps its pre-flip
  fail-closed behavior. It resolved to the default when
  the default was 1; it now resolves to the new
  constant `MMV_MAX_ROWS_FALLBACK` (1), never to the
  shipped default, so a typo in the environment cannot
  widen routing. This split (unset opens, malformed
  closes) is pinned by test, below.
- The band table (`MmvBands` shipped cells) is
  unchanged, as are the per-cell env overrides
  (`WGPU_COMPUTE_MMV_*_ROWS`), which still narrow or
  close individual cells on top of the ceiling.
- The K18 small-batch ceiling and the chunked-GEMV
  ceiling keep their shipped defaults of 1 (off).
  Only the MMV ceiling moves.
- The construction log already printed the resolved
  ceiling and the default from the constant, so it
  agrees with the new default without a wording
  change; the docs around it were updated to match.

## The re-pinned tests, and why each is still a real pin

- `mmv_max_rows_resolution` (compute.rs). Was: unset
  resolves to a default of 1; malformed falls back to
  the default. Now: unset resolves to 16 (asserted
  against the constant and the literal); a set "1"
  still wins (the explicit close); malformed resolves
  to `MMV_MAX_ROWS_FALLBACK` (asserted 1). Still a real
  pin: it fails if the default moves again, if the
  explicit close stops winning, or if the malformed
  case ever falls back to the open default.
- `mmv_routes_ceiling_composition` (compute.rs). Was:
  the shipped ceiling (1) routes nothing across the
  full family x shape x batch grid. Now: the same full
  grid is asserted against the explicit close
  (ceiling 1), and a second block pins the shipped
  default itself: at `DEFAULT_MMV_MAX_ROWS` routing
  equals the band table (Q4_K attn 6 routes, 2 does
  not; the Q4_K FFN row-3 hole holds; Q4_K KV 2 routes,
  12 does not; Q5_K KV 16 routes; Q6_K KV 12 routes;
  Q6_K attn 16 does not). The cap-clipping and
  kernel-band assertions below it are unchanged. Still
  a real pin: it fails if the default moves, if the
  close leaks, or if ceiling composition stops being
  table-then-clip.
- `draft_ladder_ships_closed_at_default_ceilings`
  (dflash_session.rs), rewritten and renamed
  `draft_ladder_ships_open_at_default_ceilings`. The
  old test passed literal ceilings (1, 1, 1) and so
  pinned the close, not the default, despite its name.
  The new test reads the ceilings from the constants
  (`DEFAULT_MMV_MAX_ROWS`, `DEFAULT_SMALL_BATCH_MAX_ROWS`,
  `DEFAULT_CHUNKED_GEMV_MAX_ROWS`, asserted (16, 1, 1))
  and pins the shipped resolution: the draft head at
  the block batch routes to K21 (ProjForm::Mmv)
  through the Q5_K FFN cell; measured in-band admissions
  route (Q4_K attn 7, Q6_K KV 12, Q5_K KV 16); the
  table's exclusions stay exclusions (Q4_K/Q6_K admit
  nothing at batch 16, the Q4_K FFN row-3 hole, Q5_K KV
  at 12, batches 1 and 64 everywhere); and the original
  full-grid assertion survives verbatim against
  (1, 1, 1), so the explicit close still restores the
  pre-flip ladder exactly. Still a real pin: it fails
  if any shipped ceiling moves, if the head stops
  routing at the default, or if the close leaks.
- `proj_bench_routes_to_gemv_at_shipped_defaults_and_matches_it`
  (tests/proj_bench.rs, device-gated). Its fixture was
  a KV-class shape (512 in, 4 out) at batch 2, inside
  the Q4_K KV cell (rows 2-3): at the new default that
  cell routes to K19, so the test's subject (the
  ProjBench handle naming the GEMV form, and routed
  output bit-identical to a direct GEMV) no longer
  holds at that shape. The fixture moved to an
  attention-class shape (512 in, 512 out), whose Q4_K
  cell starts at row 6, so batch 2 is outside every
  open cell and the GEMV form serves at shipped
  defaults for a stated reason. Still a real pin: it
  fails if the Q4_K attention cell ever widens to
  row 2, if the handle misnames the served form, or
  if the routed GEMV's stream diverges from the
  direct one.

One harness correction rides along, in
bench/mmv-sweep.sh: the control arm used to unset
`WGPU_COMPUTE_MMV_MAX_ROWS`, which meant "the shipped
default" when the default was off. Unset now means
the open default, so the control arm pins
`WGPU_COMPUTE_MMV_MAX_ROWS=1` explicitly to keep
measuring the pre-flip ladder, and the k18 and chunked
arms pin it closed as well, so each arm measures only
its own rung. Without this, every future sweep's
control, k18, and chunked arms would silently include
MMV routing and read as wrong data.

Docs updated to agree with the new default:
compute.rs (env, default, and accessor docs; the
construction comment), session.rs (mmv_applies,
prepare_proj, the ActProj variant, routed_rung_admits),
dflash_session.rs (proj_candidates), kernels/mod.rs
and the K19/K20/K21 module docs. The band table's
`describe()` output carries no default and needed no
change; the diag manifest prints the resolved ceiling
from the backend, so it agrees by construction.

## Gates on this lane (VM, no adapter)

- cargo test -p wgpu-compute: 238 passed, 0 failed,
  1 ignored (the pre-existing hash-cost probe),
  25 binaries. Device legs skip on the VM by design.
- cargo test -p dflash: 24 passed, 0 failed.
- cargo test -p itak-spec: 49 passed, 0 failed.
- cargo clippy -p wgpu-compute -p dflash -p itak-spec
  --all-targets -- -D warnings: clean.
- cargo fmt --check: clean.
- cargo check --workspace: clean.

## Rollback

Set `WGPU_COMPUTE_MMV_MAX_ROWS=1`. That restores the
pre-flip ladder exactly, no rebuild needed.
