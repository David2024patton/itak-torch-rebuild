# Mojo/MAX engine — Wave D lane D3: GEMV tuning beyond K1 (2026-11-07)

Lane: `lane/mojo-d3-gemv`, cut from `mojo-max-engine` head
`2c1802c`. Job, in priority order: tune the Q6_K GEMV
(ffn_down + attn_v), tune the Q5_K GEMV (output.weight,
202,048 rows per token), F32/F16 if time remained. Method =
Lane A2's ladder discipline: correctness gate FIRST on every
iteration against the shape's Wave A bar; interleaved timing
against the Wave A port measured fresh in the same window on
the same fixtures; rejected iterations recorded with their
numbers. K1 was NOT revisited (see the K1 verdict).

All server numbers below are direct foreground reads of the
window logs (`~/d3-stage/logs/window{,2,3}.log` on
ubuntu-server). Staging was git archives with STAGED_COMMIT
files and sha256-verified kernel sources in every window;
the phase lock was held throughout each window; only
`cortex-embed.service` was stopped and restored (verified
active after every window); `cortex-cortex-1` and
`cortex-cortex-surrealdb-1` were never stopped; port 8080
was clear in every window; VRAM bookends were identical in
all three windows: 2,584,600,576 B before clear, 33,271,808 B
drained, 2,584,600,576 B after restore.

## Fixtures and bars (stated before starting, from the Wave A note)

Fixtures are built by `bench/mojo-d3/extract_fixtures.py`
from the real model file
(`/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`),
with independent Float64 numpy references (the extractor's
dequant was cross-checked exactly against a literal
transliteration of `crates/gguf/src/dequant.rs` on random
blocks before any GPU run). The rel metric is
`bench/mojo/compare.py`'s: max_abs_err / max(1, max|ref|).

- Q6_K corr: blk.0.ffn_down rows 0..63 x 19968, k2_check's
  LCG seed-42 input. Bar (k2_check class): rel <= 1e-4,
  cosine >= 0.999999. Wave A port recorded rel 9.710e-08.
  Timing fixture: the full tensor, 6656 rows, 109,025,280 B.
- Q5_K corr: output.weight rows 0..63 x 6656. Bar (k3
  class): rel < 1e-4, cosine > 0.999999. Wave A port
  recorded rel 9.425e-08. Timing fixture: rows 0..16383,
  74,973,184 B.
- F32/F16 corr: the first 64 real F32 [6656] rows in file
  order (norm weights), F16 = the F16 encoding of the same
  rows. Bar (k3 class): rel < 1e-4, cosine > 0.999999. Wave A
  recorded F32 rel 8.426e-07, F16 rel 1.033e-06. Timing
  fixtures are SYNTHETIC 6656x6656 (seeded; no real F32
  tensor of timing size exists - the real F32 rows are
  [6656] vectors): F32 177,209,344 B, F16 88,604,672 B.
  Wave A recorded no GB/s for any of these ports, so each
  window measured the port arm fresh, interleaved.

Fixture validation: in window 1 the freshly measured port
arms reproduced Wave A's recorded figures almost exactly
(Q6_K port rel 9.686e-08 vs recorded 9.710e-08; Q5_K port
9.695e-08 vs 9.425e-08; F32 port 8.414e-07 vs 8.426e-07;
F16 port 1.033e-06 vs 1.033e-06), so the ladders stand on
the same ground Wave A did. Timed-run outputs were also
compared against the full-fixture references for both arms
in every window; all passed their bars.

## Q6_K ladder (blk.0.ffn_down; wgpu K2 shipped r1sg = 831.9 GB/s reference)

Port (window 1, interleaved): 103.11 / 95.23 / 98.16 GB/s,
median **98.16**. Gate rel 9.686e-08.

Iteration 1 — KEPT. The wgpu K2 r1sg structure translated
onto the RAW 210-byte block layout (no repack: device
weights and the CPU twin are untouched): 256 threads as 16
groups of 16, one group per super-block, thread owning the
four streams' quads at byte column 4*(t%8) of half t//8.
The layout fight is real and is the shape's whole
difference from K1: 210 is not a multiple of 4, so odd
blocks start two bytes into a word and no aligned u32 load
can serve them. Words are rebuilt from aligned u16 pairs
(every word offset the kernel builds is even), the f16
super-scale is read as a single u16 (never overreading the
block tail), the 6-bit quant words use ggml's own masks
(ql low/high nibbles, qh bytes as 2-bit planes at shifts 0
and 4), signed sub-block scales are read as single bytes,
and the reduction is K1's segmented shuffle_xor butterfly
plus a shared finish.
Window 1: 410.93 / 397.64 / 412.68, median **410.93 GB/s**
(4.19x the port, +319%; 49.4% of wgpu K2). Gate rel
5.960e-08, cosine 1.000000000000; full timing fixture rel
6.879e-08. Window 2 re-measure of the same binary: median
410.26. Final-tree gate (window 3): rel 5.960e-08; spot
timing 401.12 / 403.26.

Iteration 2 — REJECTED (no win). `load_word` as a single
u16x2 vector load + bitcast (half the load instructions of
iteration 1's two scalar loads + shift/or). Gate identical
(rel 5.960e-08, as expected for a pure load-route change).
Window 2, interleaved against iteration 1: 416.71 / 409.02
/ 406.27, median **409.02** vs iteration 1's 410.26 in the
same window - reps split, no win. Dropped; iteration 1's
form is what ships.

## Q5_K ladder (output.weight; wgpu K3 shipped r2sg = 813.2 GB/s reference)

Port (window 1): 35.23 / 35.40 / 35.52 GB/s, median
**35.40**. Gate rel 9.695e-08. (The port recomputes every
element's scale/min from the header bytes per element; at
44 words per row-block it is the slowest of the four ports
by an order of magnitude.)

Iteration 1 (rows=2) — superseded by iteration 2. The
wgpu K3 r2sg structure, K1's own tuned shape: 2 rows per
128-thread workgroup as 8 groups of 16, header loaded as
one vec4, packed scales factored once per super-block (not
per element), the fifth-bit assembly done per block (one
qh word per thread; per sub-block the (qh >> s) &
0x01010101 plane unpacked to the bits the four quant quads
need), activation quads loaded once per super-block and
shared by both rows, dots in the d*sc*dot(q,x) -
dmin*m*sum(x) form, segmented butterfly + shared finish.
176-byte blocks are word-aligned, so every load is a
direct u32 load exactly as in K1.
Window 1: 337.38 / 332.55 / 337.70, median **337.38 GB/s**
(9.53x the port). Gate rel 2.216e-07, cosine
1.000000000000; full timing fixture rel 2.650e-07.

Iteration 2 (rows=4) — KEPT (promoted). Same per-row math,
tile widened to 4 rows per workgroup: activation quads
loaded once per super-block serve four rows; per-row block
contribution factored as `q5k_block_dot`. Tried as a bench
experiment (`bench/mojo-d3/exp_q5k_r4.mojo`) because the
wider tile had NOT won on K1's ladder (A2 iteration 5) -
on this shape it did: window 2, interleaved against
rows=2, it won every rep: 366.42 / 361.92 / 358.16,
median **361.92 GB/s** vs rows=2's 341.30 in the same
window (+6.0%; 10.22x the port overall; 44.5% of wgpu K3).
Gate rel identical to rows=2 (2.216e-07; the tile changes
only which rows share a workgroup), full timing fixture
rel 2.650e-07. Final-tree gate (window 3): rel 2.216e-07;
spot timing 360.58 / 361.31. gpu_layer follows through the
imported ROWS_PER_WG constant (grid rows//4; odd
row counts refused by name in `gpu_layer_check_proj`);
output.weight's 202,048 rows are a multiple of 4.

## F32 / F16 ladder

Port (window 1): F32 432.13 / 450.54 / 419.92, median
**432.13 GB/s**, gate rel 8.414e-07. F16 180.22 / 180.15,
median **180.19 GB/s**, gate rel 1.033e-06. (The F16 port
decodes every element through the exact scalar f16 route;
the F32 port is a clean streaming kernel and was already
the fastest port of the four shapes.)

F32 tuned iteration — REJECTED. The K1 segment shape (2
rows per 128-thread workgroup, 8 groups of 16 striding the
row in 256-element segments, vec4 loads only, activation
shared by both rows, hybrid reduction): 415.42 / 419.42 /
403.86, median **415.42 GB/s** - slower than the port in
every rep (0.96x). The vec4 segment shape buys nothing
when the port's bottleneck is not instruction count. The
port kernel is restored verbatim (own F32_THREADS=256
constant; gpu_layer F32 launch back to grid=rows; the
2-row-tile refusal now applies to F16 only). Window 2/3
confirmation gates on the restored kernel reproduce the
port's figure exactly (rel 8.414e-07), and the final tree
carries the port.

F16 tuned iteration — KEPT. The same segment shape over
the raw half bytes, each quad arriving as two u32 words
decoded through the K1 bit-trick route (`f16_quad_from_words`,
proven exhaustively against the exact decode on the VM):
717.68 / 718.20, median **717.94 GB/s** (3.98x the port,
+298%). Gate rel 7.944e-07, cosine 1.000000000000; final-tree
gate identical. No real Glimmer slot routes F32/F16 through
the GEMV (norms and the embedding take other paths -
checked in DESIGN.md's slot map and in gpu_layer), so the
2-row F16 tile cannot refuse a real tensor; the standalone
mains carry the named refusal for odd shapes.

## K1 verdict

NOT revisited. The bar stands where Wave A left it (rel
1.267e-07 class on the synthetic fixture); the recorded
faster reorders land at rel 1.34-1.49e-07, over the bar,
and David's question about moving the bar is still open -
it does not move in this wave. This lane's time went to
priorities 1-3, all of which landed; no K1 iteration was
attempted, so there is nothing to prove twice.

## End-to-end confirmation (window 3, final tree a98d1ca)

The Wave C GPU golden protocol on the staged final tree:
teacher-forced proof **8/8** (step_ms 87-89) and the runner
**GOLDEN_FIRST8_MATCH 1**, TEXT_STRICT_OK 1, tokens
[13796, 26, 1573, 373, 290, 2167, 26899, 1212, ...] -
byte-identical generation to Wave C's golden on the port
kernels, through the tuned Q6_K + Q5_K rows=4 + K1 stack.
Runner PREFILL_MS 563, DECODE_STEP_MS_APPROX 110.07
(DECODE_TOK_S_APPROX 9.09) against Wave C's recorded
148.5 ms/step on the same protocol - the tuned GEMVs show
up end to end as roughly a quarter off the decode step.

## VM tally (final tree)

`tests/run_tests.sh`: PASS - 133 tests run, 133 passed,
0 failed, 0 skipped (26 TestSuite files + the seam-on diag
rerun). New suites: test_q6k_gpu_helpers 3/3 (f16 sweep,
load_word vs byte assembly incl. odd block starts, stream
quads vs the per-element dequant route), test_q5k_gpu_helpers
3/3 (f16 sweep, packed scales vs byte route, qh bit quads vs
element route), test_f32_gpu_helpers 2/2 (f16 sweep,
f16_quad_from_words vs exact decode). CPU twins are
byte-unchanged in all three modules (test_q6k_cpu 2/2,
test_q5k_cpu 2/2, test_f32_cpu 4/4 pass against the same
fixtures as Wave A); `dequant.mojo` untouched; gfx1100
compile checks clean for all three standalones and both GPU
server proofs; `mojo format` clean on every touched file.

## Integration notes

- gpu_layer launch geometry changed: Q5_K grid is rows//4
  (the Wave A port launched one workgroup per row; this
  lane's iteration 1 made it rows//2), F16 grid is rows//2,
  F32 unchanged, Q6_K unchanged (256 threads). `gpu_layer_check_proj`
  refuses Q5_K rows not divisible by 4 and F16 rows not
  divisible by 2, by name. Re-run the Wave C GPU golden on
  the integrated tree (window 3 ran it on the lane tree and
  it passed).
- Standing gap, unchanged from K1: the tuned Mojo GEMVs sit
  at 44-49% of the wgpu kernels' recorded throughput
  (Q6_K 410.93 of 831.9; Q5_K 361.92 of 813.2), the same
  Mojo-codegen ceiling class A2 recorded for K1 (624.1 of
  838.4, 74% - K1's layout is friendlier). Q6_K's raw-layout
  u16-pair word rebuild is the visible extra cost the wgpu
  kernel does not pay (it reads a repacked layout); closing
  that gap means a repack decision, which is a weights-
  residency call for a later wave, not a kernel-local one.
