# Gated and SwiGLU routing: the last two unrouted projections join the ladder (2026-11-06)

The spec-round cost map
(`bench/spec-round-economics-2026-11-06.md`, finding F1)
left one verify cost unrouted: at depth 6, 52.5 ms of
GPU per round is the gated output projection and the
SwiGLU-fused down projection running their fused GEMV
forms (the unrouted `k1_q4k_gemv` row at 26.60 ms and
the unrouted `k2_q6k_gemv` row at 25.88 ms of that
note's verify table). They were unrouted for a
structural reason, not an oversight: `prepare_gated_proj`
and `prepare_swiglu_proj` in
`crates/wgpu-compute/src/session.rs` never pass through
`prepare_proj`, so the band-routed MMV ladder
(`bench/mmv-band-routing-2026-11-06.md`), the K18
small-batch rung, and the chunked GEMV rung never see
them. This lane closes that gap. The DFlash session's
own prepares are a different lane's territory and are
untouched.

## The routing change, per function

Both functions keep their existing matrix-range forms
exactly (batch at or past `GEMM_MIN_ROWS` = 16: the
fused MMQ forms, the fused GEMM forms behind them).
Below that range, each gains a routed branch in front
of its fused GEMV fallback:

- `prepare_gated_proj` (Q4_K only, as before; its
  signature now takes the weights as `&ProjWeights` and
  destructures, because the routed branch needs the
  enum form for `prepare_proj`): when the caller
  supplies a materialization buffer and
  `routed_product` returns a product, the result is
  `ProjCall::ActProj { act, proj }`: K7's strided gate
  (`prepare_gate_strided`) materializes
  `attn * sigmoid(gate)` into the session's existing
  `scratch.attn_gated` rows, and `proj` is a plain
  `prepare_proj` over those rows.
- `prepare_swiglu_proj` (Q4_K and Q6_K arms; the Q5_K
  refusal is unchanged): the same shape, with K7's
  strided SwiGLU (`prepare_swiglu_strided`)
  materializing `silu(gate) * up` from the pair buffer
  into the existing `scratch.ffn_act` rows, then a
  plain `prepare_proj` for the down product.

`routed_product` is the shared gate, and it is
deliberately narrow. It returns a product only when
**all** of these hold: batch in `2..GEMM_MIN_ROWS`;
`routed_rung_admits` says a rung below the matrix
forms serves this (family, shape, batch), meaning the
band-routed MMV cell admits it (`mmv_applies`), or the
K18 band admits it and the device built that family's
K18 pipeline, or the chunked band admits it
(`chunked_gemv_applies`); and `prepare_proj` itself
succeeds. Any failure (out of range, no rung, a ladder
refusal, logged by name) returns `None` and the fused
form serves the projection, exactly as it did before
this lane. Batch 1 is never routed: every rung starts
at 2 rows, and there the fused form is the ladder's
own answer (the plain GEMV product) with the
activation folded in, in one dispatch instead of two.

The gating is the existing config only:
`WGPU_COMPUTE_MMV_MAX_ROWS` plus the band cells (and,
behind the same composition, the K18 and chunked
ceilings). No new env names. At shipped defaults every
rung is closed, `routed_rung_admits` is false at every
batch, and the plans are byte-identical to the
pre-routing code: same dispatches, same kernels, same
order. The two call sites that already materialized
into `attn_gated` / `ffn_act` for the trait composition
prove the scratch buffers' sizing; the routed branches
reuse those same buffers, which no other dispatch in a
fused layer's plan reads.

One accounting note: a routed projection records
`1 + proj.dispatches()` dispatches (the K7 plus the
product's own count). The layer dispatch accounting
(`layer_dispatch_count`) already adds each
projection's extras over one back into the layer
total, and the fusion shape still subtracts the K7
the fusion removed, so a routed layer's count is
exact by construction: fused count, plus the product
rung's extras (a chunked product at batch 7 records
its three window dispatches, and so on).

## The K7 strided form

Routing needs the activation as data, and the fused
forms read it through a geometry K7's flat kernels
cannot express: the gate rides inside the QKV batch
buffer at a row stride and an offset (`K1ActGeom`:
`a_stride_elems`, `b_stride_elems`, `b_off_elems`),
and the FFN pair rides one buffer at twice the width
with the up half offset by the width. K7 therefore
gains two entry points in its existing shader module,
`gate_strided_main` and `swiglu_strided_main`:
`out[r][c] = a[r * a_stride + c] * sigmoid(b[r *
b_stride + b_off + c])` and the silu twin, writing
packed rows. The params struct grows the four stride
fields for every entry; the flat prepares fill them
with the flat-consistent values (width = strides = n,
offset 0), under which the strided map reduces to the
flat map exactly (pinned by a host test). The strided
prepares stamp the existing diag name
`k7_swiglu_gate`, with the geometry in the stamp's
shape field.

## The math-equivalence argument

The decomposition is not a new approximation; it is
the composition the codebase already trusts, reassembled:

1. **The activation values are identical.** The fused
   kernels were built to reproduce K7's math: K1's
   module docs state the gated form applies "K7's
   exact sigmoid over the same gate values" and the
   SwiGLU form is "K7's exact silu" in its quotient
   form. The routed path runs K7 itself (the strided
   entry applies the same branch-stable sigmoid and
   the same `x / (1 + exp(-x))` quotient, term for
   term; the strided index map is pinned against the
   `K1ActGeom` contract by host tests). So the product
   behind the routed form consumes exactly the f32
   activation rows the fused form computes internally.
2. **The elementwise math after the product does not
   exist.** Neither site applies anything after its
   product: the gate and the SwiGLU activation are the
   product's *input* transform, and both are fully
   carried by step 1. There is no finish to keep
   identical because neither fused form has one (the
   finish-carrying site in this model is the head,
   which is not one of these two functions).
3. **The product is the ladder's product.** What
   changes is only which kernel multiplies: the same
   choice `prepare_proj` already makes for every plain
   projection of the same family and shape, under the
   same correctness discipline the sweep verified
   (`agree=1` in all 99 cells of both arms,
   bench/mmv-band-routing-2026-11-06.md). The MMV and
   chunked and GEMV rungs consume the f32 rows
   directly; the K18 rung quantizes them with the
   plain K12 quantize, the same Q8_1 step the plain
   projections already answer for under the crate's
   quantization tolerance. A routed site's output is
   therefore exactly a trait-composition output (K7
   activation, plain product): the resident forward
   already produces bit-comparable results through
   that composition on every non-fusing layer and
   family today.
4. **Batch at or past 16 is untouched**, so the
   fused MMQ/GEMM activation quantize (K12's gated /
   SwiGLU forms) and its numerics are exactly as
   shipped.

What is NOT claimed: bit-identity with the fused
GEMV form in general. The fused GEMV and the MMV /
chunked / K18 products sum in different orders (and
K18 quantizes), so a routed site's output differs
from the fused form's by the same reassociation /
quantization deltas the ladder already introduces
for plain projections. Identity and golden on the
card are the arbiters, as they were for every
routing change before this one; the server protocol
below runs them in both arms.

## Which band cells these shapes land in

Shape classification is `mmv_shape_of`, unchanged:
out <= in/8 is kv, out >= 2*in is ffn, else attn.

| site | weights | in | out | class | cell | verify rows (2..=7) admitted |
|---|---|---|---|---|---|---|
| gated O | Q4_K | 4,096 | 6,656 | attn | Q4_K attn `6-12` | **6, 7 only** |
| FFN down | Q6_K | 19,968 | 6,656 | attn | Q6_K attn `2-12` | **2, 3, 4, 5, 6, 7 (all)** |

The down projection's family is Q6_K in the real
file (bench/dflash-e3-2026-10-30.md records
`ffn_down` Q6_K [19968, 6656]; the loader uploads
per-tensor, so a Q4_K down tensor would take the
Q4_K arm and its cell `6-12` instead: rows 6, 7 only.
The code routes by the family it is handed, per
layer.)

The caps, stated plainly:

- **The gated projection's cell excludes most of the
  verify band.** Rows 2-5 (depths 1-4) stay on the
  fused GEMV form even with the MMV ceiling wide
  open; only rows 6 and 7 (depths 5 and 6) route.
  This is the measured table's answer, not an
  oversight: Q4_K attention-class cells at rows 2-4
  were measured losses (ratios 1.33 / 1.20 / 1.14 at
  the swept geometry), and routing them would rebuild
  the cliff inside the verify.
- **Both geometries are unmeasured in-widths.** The
  sweep measured in = 6,656 shapes only. The gated
  projection (in 4,096) and the down projection
  (in 19,968) inherit the attn cell by
  classification, the same disclosure standard as
  the band-routing note's treatment of the Q, O, and
  down geometries there. If a future sweep measures
  these widths and disagrees, the cells are env
  config and this table is the thing to correct.
- The K18 and chunked rungs, if the session ever
  enables them, admit by their own ceilings
  (2..=ceiling), not by these cells; at their shipped
  defaults they admit nothing.

## Honest expected recovery against the 52.5 ms

At depth 6 (7 rows, the economics note's measured
round), both sites route: the gated site at the
Q4_K attn cell's row-7 proxy ratio 0.889 and the
down site at the Q6_K attn row-7 proxy ratio 0.762
(both measured at in = 6,656; see the in-width
disclosure above). Applied to that note's per-site
rows:

- gated: 26.60 ms x (1 - 0.889) ≈ **3.0 ms** saved
- down: 25.88 ms x (1 - 0.762) ≈ **6.2 ms** saved
- gross ≈ **9.2 ms of the 52.5 ms (about 17%)**,
  before the materialization cost: two extra K7
  dispatches per layer (104 per verify forward),
  each moving at most 7 x 19,968 f32 of activation
  traffic; the economics note's own chain-kernel rows
  put elementwise dispatches in the ~6 µs class, so
  the honest charge is ≈ 0.5-1 ms, for a **net
  expectation of roughly 8-9 ms per depth-6 round**
  (verify GPU 102.0 ms toward ~93 ms; the 166.0 ms
  round toward ~157 ms). This is an estimate from
  proxy ratios, not a measurement; the A/B below is
  the verdict.

At depth 1 (2 rows) the gated site does not route
(cell starts at 6) and the down site does (Q6_K
attn row-2 proxy ratio 0.690). The economics note
has no timing-on depth-1 verify split, so no
millisecond figure is promised at depth 1; the
protocol measures it.

## VM gates (this lane)

- `cargo test -p wgpu-compute`: **203 passed, 0
  failed** across 23 binaries (lib 112, including
  four new K7 tests: strided entry points and index
  expressions pinned in the source, the whole K7
  module validated under Naga, and the strided map's
  flat-reduction and segment-extraction properties
  pinned on the host). GPU legs skip on this VM by
  design, as everywhere in this crate.
- `cargo test -p itak-spec -p dflash` (the session's
  consumers): **72 passed, 0 failed**.
- `cargo clippy -p wgpu-compute --all-targets -- -D
  warnings`: clean.
- `cargo check --workspace --all-targets`: clean.
- `cargo fmt --all -- --check`: clean.
- `dflash_session.rs` untouched, as assigned.

## Server protocol (for the consolidated card session)

Stage from the main this lane merges into,
manifest-verified, cleared card (VRAM guard reads
33,271,808 bytes before every arm), phase lock held,
tenants stopped before and restored after, per the
standing session rules. Build once:

```
cargo build --release -p itak-spec --example spec_generate_gpu
```

Arms: `spec_generate_gpu`, both mode, default
Stage 1 prompt, 64 tokens, GPU draft, at depths 1
and 6: control (all routing env unset) and routed
(`WGPU_COMPUTE_MMV_MAX_ROWS=16`, no
`WGPU_COMPUTE_MMV_*_ROWS` overrides; K18 and chunked
ceilings left unset in the primary arms):

```
[WGPU_COMPUTE_MMV_MAX_ROWS=16] ./target/release/examples/spec_generate_gpu \
  /mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf \
  /mnt/models/Muse-Glimmer-30B/dflash-Muse-Glimmer-30B-Q4_K_M.gguf \
  64 <depth>
```

Pass criteria, per depth:

- IDENTITY PASS and GOLDEN PASS in every arm, and
  the routed arm's token stream identical to the
  control arm's.
- Acceptance counts identical across arms.
- Dispatch counts: identical to control at depth 1
  except the predicted delta (below); at depth 6,
  exactly the predicted delta. Prediction: each
  routed site adds ONE dispatch per layer per verify
  forward (the K7 materialization; the product stays
  one dispatch under MMV). Depth 6: both sites route
  on all 52 layers, so **+104 dispatches** per verify
  forward against control, and the diag kernel tally
  must show `k7_swiglu_gate` records with
  `mode=gate_strided` / `mode=swiglu_strided` shapes
  (52 each per forward) plus `k19_q4k_mmv` /
  `k20_q6k_mmv` counts higher than control by the
  routed sites (52 each per forward). Depth 1: only
  the down site routes, so **+52 dispatches**, 52
  `swiglu_strided` records, **zero** `gate_strided`
  records, and the gated site's `k1_q4k_gemv` count
  unchanged: that asymmetry is the cell table's
  fingerprint, and any deviation from it is a finding
  against this note.
- Timing (diag on, matched arms): `spec.verify.end`
  medians at depth 6 improve toward the ~8-9 ms net
  estimate (verify GPU segment from the diag CSV:
  the unrouted `k1_q4k_gemv` / `k2_q6k_gemv` totals
  fall by roughly the predicted per-site amounts,
  replaced by `k19_q4k_mmv` / `k20_q6k_mmv` time plus
  the K7 rows). Depth 1: no promised figure; report
  the verify median against control either way.
- No-leak check: with the env unset after the routed
  arms, a repeat control run reproduces the first
  control's dispatch counts and stream exactly (the
  shipped-default byte-identity claim, on the card).

## Correction (2026-11-07): the small-rows dispatch prediction

Appended by the hygiene-bundle lane, 2026-11-07. The
sections above are the lane's record as written and are
not edited; this section corrects one prediction in the
server protocol, on the baseline session's measurement
(bench/baseline-session-2026-11-06.md, leg (f)).

**What was predicted.** Depth 1: only the down site
routes, so +52 dispatches per verify forward, 52
`swiglu_strided` records, zero `gate_strided`. Depth 6:
both sites route on all 52 layers, +104 per verify
forward (52 gate + 52 swiglu).

**What was measured.** Depth 6, rows-7 verify forwards:
exactly 52 gate_strided + 52 swiglu_strided per
forward, +104 total. That half of the prediction was
exact. Depth 1 (rows 2 every round): gate_strided
exactly 0, but swiglu_strided counts **26.0 per
forward** against the predicted 52, and the net
dispatch delta is +26 (472 vs 446), not +52. The same
half-count appears in the depth-6 rows-3 and rows-5
forwards (26.0 swiglu_strided each), and the
decomposition's one plain-phase firing, a prefill
chunk at rows 6, added the full +104.

**The corrected rule, from the session code.** The
decomposition fires per layer, and the layer's own
down-tensor family decides admission:
`prepare_swiglu_proj` (crates/wgpu-compute/src/
session.rs) matches on the down weights' family; its
Q4_K arm consults the routed ladder for
`MmvFamily::Q4K` and its Q6_K arm for
`MmvFamily::Q6K`. The down shape (in 19,968, out
6,656) classifies as attn under `mmv_shape_of`, so
the Q6_K arm admits batches 2..=12 and the Q4_K arm
admits batches 6..=12, per the shipped band table.
This note's own cell-table caveat stated the rule
correctly ("a Q4_K down tensor would take the Q4_K
arm and its cell 6-12 instead: rows 6, 7 only. The
code routes by the family it is handed, per layer.");
the protocol prediction then applied the Q6_K row to
all 52 layers anyway.

**Why the prediction went wrong.** The cell table's
census line took the down family from
bench/dflash-e3-2026-10-30.md, which is the DFlash
draft file's census (5 layers, `ffn_down` Q6_K on
every one). The target file's down tensors are mixed:
**26 layers carry Q6_K down and 26 carry Q4_K
down.** The split is pinned arithmetically by the
defaults-confirmation session's prefill summaries
(bench/results/defaults-confirmation-20261107/
leg2-default.csv and leg2-off.csv; identical counts
in both arms): 16 chunk forwards at 64 rows carry
2,912 `k13_q4k_mmq` dispatches, 182 per chunk, which
is the 3-per-layer Q4_K base (attention batch, gated
output, FFN gate+up pair: 156) plus exactly 26 Q4_K
down products; and 832 `k14_q6k_mmq`, 52 per chunk,
which is 26 Q6_K down products plus the 26 Q6_K V
products of the partial-batch layers (the same half
of the file whose V is Q6_K; see the plan builder's
comment in session.rs).

**Corrected per-forward arithmetic** (verify forwards,
batch = depth + 1, one K7 materialization added per
firing layer, the product staying one dispatch):

| batch | gate_strided | swiglu_strided | net delta |
|---|---|---|---|
| 2..=5 | 0 | 26 (Q6_K-down layers only) | +26 |
| 6..=7 | 52 | 52 | +104 |

So: depth 1 adds +26 per verify forward, not +52;
depth-6 full rounds (batch 7) add +104, as predicted;
depth-6 tail rounds (batch 3 and 5) add +26. The
rows-6 prefill chunk's +104 is the same rule at
batch 6, not a separate case. No timing claim in
this note changes: the depth-6 recovery estimate was
computed at rows 7, where the prediction was exact,
and no millisecond figure was promised at depth 1.
Correctness was never in question: identity, golden,
acceptance, and streams passed in both arms of the
baseline session's A/B.
