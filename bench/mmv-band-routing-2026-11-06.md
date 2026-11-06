# MMV band routing: the mid-band cliff removed by routing (2026-11-06)

The 2026-11-06 K19 verdict continuation
(`bench/mmv-verdict-2026-11-05.md`, steps 2-4) proved
the register-streaming family correct everywhere and
fast at the top of its band, and convicted it in the
middle: at depth 1, session dose caps 4, 8, and 12 all
landed BELOW the shipped control (14.71 / 15.23 /
15.20 effective tok/s against control 16.98, verify
medians about 50 ms against 45), while cap 16 landed
above it (19.26, verify about 38.5 ms). The verdict's
decision rule therefore kept the default at 1 (off):
the no-cliff condition was not met. The verdict named
the remedy itself: "Any future flip needs a band floor
informed by the per-batch table (for example, routing
only the top of the band)".

This lane is that remedy, in its general form. The
single global ceiling (`WGPU_COMPUTE_MMV_MAX_ROWS`)
no longer decides routing alone. Under it sits a
routing table keyed by (quant family, shape class):
each family's register-streaming product (K19 for
Q4_K, K20 for Q6_K, K21 for Q5_K) serves a projection
only at batch rows where the 2026-11-06 sweep
measured it beating the GEMV form it displaces.
Every other cell falls through to the ladder exactly
as before (MMV, then K18, then chunked, then GEMV;
the ladder's order and every kernel are unchanged).
This note is the table's audit trail: the measured
cells, the rule that turned them into ranges, and
the config surface that carries them.

## Method

Data: the committed verdict sweep,
`bench/results/mmv-sweep-20261106-212335/`
(control.csv, mmv.csv; the K18 and chunked arms are
not inputs to this table). Every cell agreed with
GEMV under the harness's correctness guard
(`agree=1` in all 99 cells of both arms), so every
number below is a timing verdict on a correct
kernel, never a correctness trade.

A cell is (family, shape, rows): the control arm's
GEMV median against the MMV arm's median at the same
cell, both from the same run, 25 reps, diag
timestamp timing. Verdicts:

- WIN: MMV median < 0.98 x control median.
- TIE: within +/-2% (0.98-1.02). A tie is not a
  measured win: the cell falls through. GEMV is the
  incumbent; routing needs evidence, not parity.
- LOSS: MMV median > 1.02 x control median.

Measured rows are {2, 3, 4, 5, 6, 7, 8, 9, 12, 16}.
Measured cells are authoritative, including where
they break a run (Q4_K FFN row 3 is a measured LOSS
between wins at rows 2 and 4; the table routes
around it, see below). Rows the sweep did not
measure (10, 11, 13, 14, 15) are admitted only
where two measured WINs bracket them inside the
MMV-routed range: rows 10-11 inherit wherever rows
9 and 12 both win, and rows 13-15 inherit only for
Q5_K attention and FFN, where rows 12 and 16 both
win. For Q4_K and Q6_K there is no MMV measurement
at row 16 at all (both arms serve the MMQ forms
there; the matrix rung runs before the MMV rung in
`prepare_proj`), so their cells end at 12, the last
measured win, and rows 13-15 fall through.

## The measured table

Medians in ms from the two CSVs; ratio is MMV over
control. "Routed" is the shipped table's answer for
the cell under a fully open ceiling.


### Q4_K (K19), attn (out 6656)

| rows | GEMV ctl (ms) | MMV (ms) | ratio | verdict | routed |
|---|---|---|---|---|---|
| 2 | 0.0792 | 0.1054 | 1.331 | LOSS | no |
| 3 | 0.1032 | 0.1241 | 1.203 | LOSS | no |
| 4 | 0.1294 | 0.1479 | 1.143 | LOSS | no |
| 5 | 0.1672 | 0.1697 | 1.015 | TIE | no |
| 6 | 0.2080 | 0.1998 | 0.961 | WIN | yes |
| 7 | 0.2534 | 0.2253 | 0.889 | WIN | yes |
| 8 | 0.3015 | 0.2532 | 0.840 | WIN | yes |
| 9 | 0.3393 | 0.2895 | 0.853 | WIN | yes |
| 12 | 0.3979 | 0.3877 | 0.974 | WIN | yes |
| 16 | 0.7068 (k13_q4k_mmq) | 0.7271 (k13_q4k_mmq) | n/a | no MMV cell (MMQ in both arms) | no |

### Q4_K (K19), ffn (out 19968)

| rows | GEMV ctl (ms) | MMV (ms) | ratio | verdict | routed |
|---|---|---|---|---|---|
| 2 | 0.3149 | 0.2274 | 0.722 | WIN | yes |
| 3 | 0.2886 | 0.2956 | 1.024 | LOSS | no |
| 4 | 0.4236 | 0.2838 | 0.670 | WIN | yes |
| 5 | 0.4967 | 0.3302 | 0.665 | WIN | yes |
| 6 | 0.5704 | 0.3784 | 0.663 | WIN | yes |
| 7 | 0.6511 | 0.4093 | 0.629 | WIN | yes |
| 8 | 0.7306 | 0.4464 | 0.611 | WIN | yes |
| 9 | 0.7805 | 0.4893 | 0.627 | WIN | yes |
| 12 | 1.0250 | 0.6167 | 0.602 | WIN | yes |
| 16 | 1.3146 (k13_q4k_mmq) | 1.3136 (k13_q4k_mmq) | n/a | no MMV cell (MMQ in both arms) | no |

### Q4_K (K19), kv (out 256)

| rows | GEMV ctl (ms) | MMV (ms) | ratio | verdict | routed |
|---|---|---|---|---|---|
| 2 | 0.0709 | 0.0609 | 0.859 | WIN | yes |
| 3 | 0.0651 | 0.0606 | 0.931 | WIN | yes |
| 4 | 0.0615 | 0.0605 | 0.984 | TIE | no |
| 5 | 0.0599 | 0.0608 | 1.015 | TIE | no |
| 6 | 0.0599 | 0.0626 | 1.045 | LOSS | no |
| 7 | 0.0610 | 0.0634 | 1.039 | LOSS | no |
| 8 | 0.0625 | 0.0646 | 1.034 | LOSS | no |
| 9 | 0.0635 | 0.0659 | 1.038 | LOSS | no |
| 12 | 0.0683 | 0.0714 | 1.045 | LOSS | no |
| 16 | 0.4411 (k13_q4k_mmq) | 0.4219 (k13_q4k_mmq) | n/a | no MMV cell (MMQ in both arms) | no |

### Q5_K (K21), attn (out 6656)

| rows | GEMV ctl (ms) | MMV (ms) | ratio | verdict | routed |
|---|---|---|---|---|---|
| 2 | 0.1613 | 0.1219 | 0.756 | WIN | yes |
| 3 | 0.2181 | 0.1471 | 0.674 | WIN | yes |
| 4 | 0.1616 | 0.1582 | 0.979 | WIN | yes |
| 5 | 0.2220 | 0.1723 | 0.776 | WIN | yes |
| 6 | 0.2896 | 0.1905 | 0.658 | WIN | yes |
| 7 | 0.3578 | 0.2143 | 0.599 | WIN | yes |
| 8 | 0.3911 | 0.2304 | 0.589 | WIN | yes |
| 9 | 0.4310 | 0.2515 | 0.584 | WIN | yes |
| 12 | 0.5195 | 0.3629 | 0.699 | WIN | yes |
| 16 | 0.6457 | 0.4757 | 0.737 | WIN | yes |

### Q5_K (K21), ffn (out 19968)

| rows | GEMV ctl (ms) | MMV (ms) | ratio | verdict | routed |
|---|---|---|---|---|---|
| 2 | 0.2619 | 0.3397 | 1.297 | LOSS | no |
| 3 | 0.4179 | 0.3836 | 0.918 | WIN | yes |
| 4 | 0.5124 | 0.4251 | 0.830 | WIN | yes |
| 5 | 0.5928 | 0.3615 | 0.610 | WIN | yes |
| 6 | 0.6884 | 0.3745 | 0.544 | WIN | yes |
| 7 | 0.7770 | 0.3999 | 0.515 | WIN | yes |
| 8 | 0.8852 | 0.4458 | 0.504 | WIN | yes |
| 9 | 0.9821 | 0.4909 | 0.500 | WIN | yes |
| 12 | 1.2060 | 0.5831 | 0.483 | WIN | yes |
| 16 | 1.6520 | 0.8160 | 0.494 | WIN | yes |

### Q5_K (K21), kv (out 256)

| rows | GEMV ctl (ms) | MMV (ms) | ratio | verdict | routed |
|---|---|---|---|---|---|
| 2 | 0.0581 | 0.0649 | 1.117 | LOSS | no |
| 3 | 0.0552 | 0.0625 | 1.132 | LOSS | no |
| 4 | 0.0561 | 0.0634 | 1.130 | LOSS | no |
| 5 | 0.0574 | 0.0892 | 1.554 | LOSS | no |
| 6 | 0.0588 | 0.0940 | 1.599 | LOSS | no |
| 7 | 0.0634 | 0.2047 | 3.229 | LOSS | no |
| 8 | 0.0655 | 0.1316 | 2.009 | LOSS | no |
| 9 | 0.0677 | 0.0879 | 1.298 | LOSS | no |
| 12 | 0.0750 | 0.0829 | 1.105 | LOSS | no |
| 16 | 0.0875 | 0.0775 | 0.886 | WIN | yes |

### Q6_K (K20), attn (out 6656)

| rows | GEMV ctl (ms) | MMV (ms) | ratio | verdict | routed |
|---|---|---|---|---|---|
| 2 | 0.1540 | 0.1063 | 0.690 | WIN | yes |
| 3 | 0.2163 | 0.1352 | 0.625 | WIN | yes |
| 4 | 0.1883 | 0.1404 | 0.746 | WIN | yes |
| 5 | 0.2719 | 0.1632 | 0.600 | WIN | yes |
| 6 | 0.3184 | 0.1881 | 0.591 | WIN | yes |
| 7 | 0.3692 | 0.2813 | 0.762 | WIN | yes |
| 8 | 0.4007 | 0.2223 | 0.555 | WIN | yes |
| 9 | 0.4303 | 0.2461 | 0.572 | WIN | yes |
| 12 | 0.5424 | 0.3771 | 0.695 | WIN | yes |
| 16 | 0.8322 (k14_q6k_mmq) | 0.8608 (k14_q6k_mmq) | n/a | no MMV cell (MMQ in both arms) | no |

### Q6_K (K20), ffn (out 19968)

| rows | GEMV ctl (ms) | MMV (ms) | ratio | verdict | routed |
|---|---|---|---|---|---|
| 2 | 0.3169 | 0.2398 | 0.757 | WIN | yes |
| 3 | 0.4476 | 0.2318 | 0.518 | WIN | yes |
| 4 | 0.5812 | 0.2897 | 0.498 | WIN | yes |
| 5 | 0.7002 | 0.3259 | 0.465 | WIN | yes |
| 6 | 0.7968 | 0.3585 | 0.450 | WIN | yes |
| 7 | 0.9194 | 0.4125 | 0.449 | WIN | yes |
| 8 | 1.0134 | 0.4444 | 0.439 | WIN | yes |
| 9 | 1.1383 | 0.4703 | 0.413 | WIN | yes |
| 12 | 1.4959 | 0.5842 | 0.391 | WIN | yes |
| 16 | 1.5379 (k14_q6k_mmq) | 1.5442 (k14_q6k_mmq) | n/a | no MMV cell (MMQ in both arms) | no |

### Q6_K (K20), kv (out 256)

| rows | GEMV ctl (ms) | MMV (ms) | ratio | verdict | routed |
|---|---|---|---|---|---|
| 2 | 0.0479 | 0.0581 | 1.213 | LOSS | no |
| 3 | 0.0502 | 0.0562 | 1.120 | LOSS | no |
| 4 | 0.0502 | 0.0544 | 1.084 | LOSS | no |
| 5 | 0.0518 | 0.0557 | 1.075 | LOSS | no |
| 6 | 0.0535 | 0.0584 | 1.092 | LOSS | no |
| 7 | 0.0570 | 0.0582 | 1.021 | LOSS | no |
| 8 | 0.0595 | 0.0586 | 0.985 | TIE | no |
| 9 | 0.0611 | 0.0615 | 1.007 | TIE | no |
| 12 | 0.0703 | 0.0652 | 0.927 | WIN | yes |
| 16 | 0.4710 (k14_q6k_mmq) | 0.4664 (k14_q6k_mmq) | n/a | no MMV cell (MMQ in both arms) | no |

## The derived routing table

Reading the WIN cells as inclusive batch ranges
(with the bracketing rule for unmeasured rows)
gives the shipped defaults. This is the whole
policy; it lives in `default_mmv_band` in
`crates/wgpu-compute/src/compute.rs`, and the unit
test `mmv_default_bands_match_the_measured_win_map`
pins it batch by batch against this section.

| family | shape | routed batches | env override |
|---|---|---|---|
| Q4_K (K19) | attn | 6-12 | `WGPU_COMPUTE_MMV_Q4K_ATTN_ROWS` |
| Q4_K (K19) | ffn | 2, 4-12 | `WGPU_COMPUTE_MMV_Q4K_FFN_ROWS` |
| Q4_K (K19) | kv | 2-3 | `WGPU_COMPUTE_MMV_Q4K_KV_ROWS` |
| Q5_K (K21) | attn | 2-16 | `WGPU_COMPUTE_MMV_Q5K_ATTN_ROWS` |
| Q5_K (K21) | ffn | 3-16 | `WGPU_COMPUTE_MMV_Q5K_FFN_ROWS` |
| Q5_K (K21) | kv | 16 | `WGPU_COMPUTE_MMV_Q5K_KV_ROWS` |
| Q6_K (K20) | attn | 2-12 | `WGPU_COMPUTE_MMV_Q6K_ATTN_ROWS` |
| Q6_K (K20) | ffn | 2-12 | `WGPU_COMPUTE_MMV_Q6K_FFN_ROWS` |
| Q6_K (K20) | kv | 12 | `WGPU_COMPUTE_MMV_Q6K_KV_ROWS` |

Cell syntax, for the defaults and the overrides
alike: comma-separated inclusive ranges, each a
single row ("16") or a span ("6-12"), whitespace
ignored. Parsed ranges are intersected with the
kernels' own band (2..=16). Any malformed item
(empty, non-numeric, a reversed span) makes the
whole value unparsable, and the cell falls back to
its shipped value: the fail-closed direction every
resolver in compute.rs takes, with the fallback
logged at construction by name
(`log_mmv_band_fallbacks`). Setting a cell to "1"
closes it (a valid cell that routes nothing).
The global ceiling composes on top, unchanged:
a batch routes only when it is inside its cell
AND at or under `WGPU_COMPUTE_MMV_MAX_ROWS`, so
the dose caps of the A/B clip the table from above
(cap 4 admits only the routed cells at rows 2-4,
and so on). The construction log prints the
resolved table on the diag channel (`mmv config:
...; bands q4k[attn 6-12 ffn 2,4-12 kv 2-3]
q5k[attn 2-16 ffn 3-16 kv 16] q6k[attn 2-12 ffn
2-12 kv 12]`), and a batch the ceiling admits but
a cell excludes is logged by name at plan build
(`mmv q4k attn excludes batch 2 ...; the ladder
continues`) wherever a binary enables the crate's
debug flag (`diag::set_debug`), the same seam the
kernel refusals already use.

Two cells deserve a named sentence each. Q5_K
attention row 4 is the thinnest win in the table
(ratio 0.979 against the 0.98 line); it also sits
between wins at rows 3 and 5, so the bracketing
rule would admit its row even as an unmeasured
one. Q5_K KV row 7 is the worst cell in the table
(ratio 3.229); under the old flat routing every
cap that reached row 7 paid it on any Q5_K
KV-shaped projection the ladder saw.

## Shape classification, and what it does not know

The sweep measured three shapes, all at input
width 6656: attn (out 6656), ffn (out 19968), kv
(out 256). A session sees more geometries than
that, so `mmv_shape_of` (compute.rs) classifies by
relative geometry: out <= in/8 is KV, out >= 2*in
is FFN, everything else is attention. The measured
cells sit far from both boundaries (256 is 1/26 of
6656; 19968 is 3x), so every real Glimmer
projection classifies unambiguously:

| projection | in | out | class | measured? |
|---|---|---|---|---|
| attention K, V | 6656 | 256 | kv | yes, the kv cells |
| attention Q | 6656 | 4096 | attn | class only |
| attention O | 4096 | 6656 | attn | class only |
| FFN gate, up | 6656 | 19968 | ffn | yes, the ffn cells |
| FFN down | 19968 | 6656 | attn | class only |
| lm_head (Q5_K) | 6656 | 202048 | ffn | does not route here |

The disclosures, plainly: input-width variation is
unmeasured everywhere (the sweep fixed in=6656),
and the Q, O, and down geometries inherit the
attention cells by classification, not by
measurement. If a future sweep measures those
geometries and disagrees, the cells are env
config and this note's table is the thing to
correct. The lm_head never reaches `prepare_proj`
(it is served by the fused-finish path), so no
Q5_K cell binds it today; the Q5_K cells bind in
the sweep harness (`ProjBench` prepares through
`prepare_proj` itself) and on any plain Q5_K
projection, which is exactly the exposure finding
of `bench/mmv-q5k-2026-11-05.md` restated.

## The session working points under routing

At the two depths the verdict A/B ran, the verify
batch is the whole story (depth 1: batch 2;
depth 6: batch 7):

- Batch 2 routes: Q4_K ffn (0.722), Q4_K kv
  (0.859), Q5_K attn (0.756), Q6_K attn (0.690),
  Q6_K ffn (0.757). It no longer routes Q4_K attn
  (1.331, the depth-1 cliff's largest single
  cell), Q5_K ffn (1.297), Q5_K kv (1.117), or
  Q6_K kv (1.213).
- Batch 7 routes: Q4_K attn (0.889), Q4_K ffn
  (0.629), Q5_K attn (0.599), Q5_K ffn (0.515),
  Q6_K attn (0.762), Q6_K ffn (0.449). It no
  longer routes any family's kv cell (Q4_K 1.039,
  Q5_K 3.229, Q6_K 1.021).

Under the caps the dose sweep convicted, routing
admits only measured wins: cap 4 admits Q4_K ffn
{2,4}, Q4_K kv {2,3}, Q5_K attn {2,3,4}, Q5_K ffn
{3,4}, Q6_K attn/ffn {2,3,4}; cap 8 adds the
winning cells at rows 5-8; cap 12 adds rows 9 and
12 and Q6_K kv {12}; cap 16 adds the Q5_K run to
16. No cap can rebuild the cliff out of these
cells; that is the claim the server A/B below is
sequenced to test.

## What did not change

The ladder order (MMV, K18, chunked, GEMV), every
kernel, the kernels' own band (2..=16, refused at
prepare), and the fall-through contract: a cell
exclusion returns `None` from the same `try_mmv_*`
composition step a prepare refusal uses, and the
next rung serves the projection. The shipped
ceiling default stays 1 (off). The verdict's rule
moved defaults on session evidence, and this lane
has none: it makes the routed configuration
cliff-free per the micro-bench table, and whether
that earns a default flip is the A/B's question,
not this note's.

## VM gates (this lane)

- `cargo test -p wgpu-compute`: 212 passed, 0
  failed across the crate's binaries, including
  the six new policy tests (parse forms, malformed
  rejection, fallback resolution, shape
  classification, the win-map pin, ceiling
  composition). GPU legs skip on this VM by
  design, as everywhere in this crate.
- `cargo clippy -p wgpu-compute --all-targets -- -D warnings`:
  clean.
- `cargo check --workspace --all-targets`: clean.
- `cargo fmt --check`: clean.

## Server A/B protocol (the cliff proof)

Run on the cleared card (VRAM guard reads
33,271,808 bytes before every arm), under the
phase lock, staged from the main this lane merges
into, manifest-verified like the verdict runs.
Build once:

```
cargo build --release -p itak-spec --example spec_generate_gpu -p wgpu-compute --example mmv_sweep
```

Session arms: `spec_generate_gpu`, both mode,
default Stage 1 prompt, 64 tokens, GPU draft, at
depths 1 and 6, for each of: control (env unset)
and routed at `WGPU_COMPUTE_MMV_MAX_ROWS` =
4, 8, 12, 16, with NO `WGPU_COMPUTE_MMV_*_ROWS`
set (the shipped cells are the configuration
under test):

```
[WGPU_COMPUTE_MMV_MAX_ROWS=<cap>] ./target/release/examples/spec_generate_gpu \
  /mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf \
  /mnt/models/Muse-Glimmer-30B/dflash-Muse-Glimmer-30B-Q4_K_M.gguf \
  64 <depth>
```

Pass criteria, per depth:

- IDENTITY PASS and GOLDEN PASS in every arm.
- Acceptance and resident dispatch counts
  identical across arms (the verdict's purity
  condition: differences are timing only).
- Every routed cap's effective tok/s >= the
  same-session control's. Caps 4, 8, 12 are the
  cliff test: the flat routing put them at
  14.71 / 15.23 / 15.20 against a 16.98 control;
  routed, they must not land below control.
- Routed cap 16 >= the flat-routing cap 16 on
  record (19.26 at depth 1, 10.27 at depth 6),
  within session noise: routing removed only
  losing cells from that configuration.
- Verify medians (`spec.verify.end`) at or under
  control at every cap.

Sweep confirmation, same session: re-run
`bench/mmv-sweep.sh`. In the mmv arm's CSV
(`WGPU_COMPUTE_MMV_MAX_ROWS=16`), the `kernel`
column must show k19/k20/k21 exactly in this
note's routed cells and the GEMV kernels exactly
in the unrouted ones, cell for cell against the
table above; routed-cell medians should reproduce
the 2026-11-06 MMV arm within noise, and
unrouted-cell medians the control arm's. Any cell
that disagrees with its table entry is a finding
against this note, and the cell's env override is
the interim fix while the table is corrected.
