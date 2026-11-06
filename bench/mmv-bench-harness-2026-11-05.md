# Small-batch sweep harness (2026-11-05)

One harness that measures every small-batch kernel
variant head to head through the session's own
routing, so the MMV verdict (and every later
autotune pass) reads from one table instead of four
ad hoc benches.

## What landed

- `crates/wgpu-compute/examples/mmv_sweep.rs`: the
  harness binary.
- `ProjBench` / `ProjBenchCall` in
  `crates/wgpu-compute/src/session.rs` (exported from
  the crate root): a projection uploaded from raw
  parts that prepares through `prepare_proj` itself,
  the session's routing ladder, and reports which
  kernel form the ladder chose (`kernel_name()`),
  whether that form quantizes its activation
  (`quantized()`), and its dispatch count. The
  harness measures the variant a forward would
  dispatch under the same environment, by
  construction: there is no second routing
  implementation to drift.
- `crates/wgpu-compute/tests/proj_bench.rs`: the
  handle's contract (routes to the GEMV form at
  shipped defaults and matches it exactly; an
  uncovered dtype is refused by name).
- `bench/mmv-sweep.sh`: the arm runner (below).

## The cell matrix

Per run: quant family {Q4_K, Q5_K, Q6_K} x shape x
rows {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16} = 99 cells.
Shapes are synthetic tensors at the real Glimmer
geometry, input width 6656 (the model's hidden)
throughout:

| shape | out | what it stands for |
|---|---|---|
| attn | 6656 | the attention-sized projections |
| ffn | 19968 | FFN up |
| kv | 256 | the K and V projections |

The KV width is the model's real one:
`GlimmerConfig::k_width()`, KV heads times key
length, 2 x 128 = 256
(crates/muse-glimmer/src/config.rs, whose real-file
test pins `k_width() == 256`; session.rs sizes its
K/V projection buffers from it). An earlier sketch
of this harness carried 2048 for this row; that is
the sliding window size, not a projection width,
and no tensor in the model has it.

Synthetic weights use the kernel tests' fixture
discipline (rotating f16 scale headers over LCG
payloads, tests/k18_small_batch.rs), one tensor per
(quant, shape), reused across all row counts. Peak
device memory is about 0.8 GB (both the routed
handle and the direct GEMV control hold a copy).

## Measurement method

- Warmup dispatches (3 by default) are excluded;
  every cell then gets 25 timed repetitions by
  default (the verdict floor is 20; a run below it
  prints that it is a smoke, not a verdict run).
  The CSV reports median and min per cell.
- GPU time comes from the diag timestamps system:
  one `DiagStamp` around the whole projection
  composition per repetition, so a composed form
  (K12 quantize plus its product) is timed as the
  session pays for it, whole. Available when the
  context carries a recorder
  (`WGPU_COMPUTE_DIAG=timing` and a device granting
  `TIMESTAMP_QUERY` plus
  `TIMESTAMP_QUERY_INSIDE_PASSES`); the sweep script
  exports the env. Otherwise each repetition is
  bracketed on the wall clock around submit plus
  poll, which includes submit overhead. The run
  prints `timing_method: diag_timestamps|wall_clock`
  on stdout, and the script keeps it in the arm log.
  A cell whose stamp records do not line up with its
  repetition count (dropped or missing records) is
  re-timed on the wall clock, with the substitution
  printed on stderr and in the log.
- `gbps_median` is the tensor's resident bytes over
  the median time: the stream rate of one dispatch
  of the composition. For the GEMV form at R rows
  the hardware re-reads the tensor per row inside
  the dispatch, so effective DRAM traffic is R times
  the CSV's `weight_bytes`; the matrix and
  small-batch forms stream it once. (The same
  convention as bench/small-batch-2026-11-04.md.)

## Correctness guard

Before any timing, each cell's routed output is
compared against the GEMV form's output for the
same inputs, the GEMV prepared directly from the
kernel pipelines (K1/K2/K3) over the same bytes.
Agreement mirrors the crate's kernel tests
(tests/k18_small_batch.rs): a form that quantizes
its activation (the MMQ and small-batch forms)
answers to the 5% quantization band
(`check_quant_band`), an f32 form to the tight
reassociation bound `2e-3 + 5e-5 * scale`
(`check_close`). A cell that disagrees is written
with `agree=0` and named on stderr; it is timed
anyway, because a fast wrong kernel is a finding,
but it can never pass as agreement.

## CSV shape

One row per cell:

```
variant_env,kernel,quant,rows,out_cols,median_ms,min_ms,weight_bytes,gbps_median,agree
```

`variant_env` is the arm label plus the routing env
actually present in the process
(`WGPU_COMPUTE_SMALL_BATCH_MAX_ROWS`,
`WGPU_COMPUTE_MMV_MAX_ROWS`,
`WGPU_COMPUTE_WIDE_MIN_ROWS`,
`WGPU_COMPUTE_CHUNK_ROWS`,
`WGPU_COMPUTE_CHUNK_MAX_ROWS`), or `env:none`.
`kernel` is the form the ladder chose, under its
diag-system name; at 16 rows the control arm itself
routes to the matrix forms (that is the shipped
`GEMM_MIN_ROWS` behavior), and the column says so.

## Running the verdict

On the server, card clear, under the phase lock:

```
flock -w 14400 /tmp/itak-phase.lock bench/mmv-sweep.sh
```

That builds the harness once and runs four arms,
writing `bench/results/mmv-sweep-<timestamp>/`
with `control.csv`, `k18.csv`, `mmv.csv`,
`chunked.csv`, and a `.log` per arm (adapter name,
timing method, per-cell lines, disagreement
warnings). Read the arms side by side per
(quant, shape, rows): `median_ms` is the verdict,
`kernel` proves which form each arm actually ran,
and any `agree=0` cell disqualifies its row.

Partial sweeps forward extra harness args to every
arm, for example `bench/mmv-sweep.sh --rows 2,7`
for the two verify working points. A single arm by
hand is just the binary under that arm's env:

```
WGPU_COMPUTE_DIAG=timing WGPU_COMPUTE_MMV_MAX_ROWS=16 \
  target/release/examples/mmv_sweep --label mmv --out mmv.csv
```

One naming caveat, stated once: the chunked arm's
env in the script is `WGPU_COMPUTE_CHUNK_MAX_ROWS`.
If the chunked-GEMV prototype lands under a
different variable, edit the script's one `CHUNK_ENV`
line; until it lands at all, the chunked arm's CSV
shows the control kernels it really ran, and its
`variant_env` stamp shows an env nothing consumed.

## VM verification of the harness itself

- `mmv_sweep --enumerate` prints the 99-cell matrix
  with no GPU and exits 0.
- This VM has no Vulkan driver at all (the loader
  library exists, but no ICD: no hardware driver and
  no lavapipe), so no wgpu device can be created
  here. Run mode therefore exercises startup, filter
  and env-stamp handling, and exits cleanly through
  the same SKIP path the crate's other benches use
  (`SKIP: no suitable GPU on this machine`, exit 0).
  The device paths (upload, routing, the guard, both
  timing methods, CSV writing) are behind the
  crate's standard adapter skip in
  tests/proj_bench.rs, like every GPU test file in
  the crate on this machine; their first real
  execution is the server verdict run. Nothing in
  this note claims a VM timing number.
- Gates, all green on this VM: cargo test
  -p wgpu-compute (the proj_bench tests included,
  skipping on adapter absence as above), cargo test
  -p itak-spec (47 passed), cargo fmt --check,
  clippy -p wgpu-compute --all-targets with
  `-D warnings`, cargo check --workspace
  --all-targets.
- Correction at merge (2026-11-05): the chunked arm
  landed as `WGPU_COMPUTE_CHUNKED_GEMV_MAX_ROWS`, not
  the `WGPU_COMPUTE_CHUNK_MAX_ROWS` guessed above and
  in the sweep script; `bench/mmv-sweep.sh` now uses
  the landed name.

## Verdict run, 2026-11-06: the head-to-head table

`bench/mmv-sweep.sh` run unchanged from main
`c439e3af45e30275601129568e3e9cb3a89590cb`, staged at
`~/gpu-wave11`, card cleared (guard read 33,271,808
bytes VRAM used), phase lock held. All four arms ran
the full 99-cell matrix on the RX 7900 XTX with diag
timestamp timing, 25 reps per cell. The four CSVs are
committed beside this note at
`bench/results/mmv-sweep-20261106-212335/`
(control.csv, k18.csv, mmv.csv, chunked.csv); the arm
logs stay in the staged tree on the server.

**Correctness: 0 disagreeing cells in every arm.** The
harness's per-cell guard against GEMV (the crate's own
tolerance rules) passed all 99 cells for control, K18,
MMV, and chunked alike. For the chunked arm this is
its on-card correctness leg: every chunked cell
agrees with the whole-batch GEMV result it decomposes.

**Kernel attribution is exactly as designed.** Control
serves GEMV in band and MMQ at rows 16; the K18 arm
serves the k18 small-batch kernels; the MMV arm serves
k19 (Q4_K), k20 (Q6_K), k21 (Q5_K) at rows 2-12, MMQ
at rows 16 for Q4_K/Q6_K, and k21 at rows 16 for Q5_K
(no Q5_K MMQ exists in the ladder); the chunked arm
serves GEMV kernels with the chunk-count timing
signature (rows 4 cheaper than rows 3 on the attention
shape, 0.1350 vs 0.1755 ms, because batch 4 is one
chunk and batch 3 is two). One reporting defect: the
chunked arm's `variant_env` column reads `env:none`
(the harness does not stamp the chunked env name into
its label), so that arm's engagement rests on its
timing signature and kernel column, not its label.

**Win counts (lowest median of the four arms, per
cell): MMV 53, control 32, chunked 11, K18 3.** K18
loses nearly everywhere in the micro-bench too
(0.39-0.72 ms where control reads 0.08-0.54 ms on the
attention shape), consistent with its 2026-11-04
session verdict; it is not a contender.

The MMV picture is mixed by family and shape, and the
mix explains the session dose response recorded in
`bench/mmv-verdict-2026-11-05.md`. Attention shape
(out 6656), median ms, control vs MMV arm:

| rows | Q4_K ctl | Q4_K k19 | Q5_K ctl | Q5_K k21 | Q6_K ctl | Q6_K k20 |
|---|---|---|---|---|---|---|
| 2 | 0.0792 | 0.1054 | 0.1613 | 0.1219 | 0.1540 | 0.1063 |
| 4 | 0.1294 | 0.1479 | 0.1616 | 0.1582 | 0.1883 | 0.1404 |
| 7 | 0.2534 | 0.2253 | 0.3578 | 0.2143 | 0.3692 | 0.2813 |
| 9 | 0.3393 | 0.2895 | 0.4310 | 0.2515 | 0.4303 | 0.2461 |
| 12 | 0.3979 | 0.3877 | 0.5195 | 0.3629 | 0.5424 | 0.3771 |

K20 (Q6_K) and K21 (Q5_K) beat GEMV at nearly every
in-band row; K19 (Q4_K) loses to GEMV at rows 2-6 on
this shape and only draws level near rows 9-12 (it
does better on the FFN shape; see the CSVs). A depth 1
verify batch is 2 rows, deep in K19's losing range,
while a depth 6 verify and the draft-side forwards
reach the rows and families where the MMV arm wins;
that is consistent with the session finding that cap
16 (which opens the whole band, all three families)
nets positive while caps 4-12 net negative. One cell
needed the harness's wall-clock fallback (Q6_K KV
rows 16, diag records did not line up; agree 1), in
all four arms identically.
