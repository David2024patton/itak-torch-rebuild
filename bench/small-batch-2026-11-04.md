# Small-batch products (K18): design, expectations, server proof

## The problem, measured

From bench/spec-gpu-port-2026-11-03.md's round economics
addendum (RX 7900 XTX, main 72a4b9d): below
GEMM_MIN_ROWS (16), the resident session's projections
run GEMV form, one full weight stream per row. The
probe fits rows x 16 GB at about 815 GB/s exactly
(1 row 30 ms, 2 rows 47 ms, 7 rows 140 ms). A depth-6
speculative round pays about 151 ms of verify, most of
it a 7-row forward re-streaming the target seven
times; the DFlash draft's 16-row block forward pays
57.3 ms the same way. Lowering the matrix threshold
to 2 was tested and killed with data: 7-row forwards
got slower (140 to 253 ms) and the dispatch count
roughly doubled, because the GEMM and MMQ tiles are
built for 32 and 64 rows.

## What K18 is

`crates/wgpu-compute/src/kernels/k18_small_batch.rs`:
the MMQ math at a tile sized for the band.

- One weight stream per dispatch: a workgroup owns a
  tile of 32 or 64 output rows and ALL batch rows in
  the pass (2..=16), stages the tile's packed weight
  words once per super-block column, and applies them
  to every row from workgroup memory. Global weight
  traffic per dispatch is the tensor's bytes, once,
  at any row count in the band.
- The math is K13/K14's, term for term: K12's Q8_1
  activations, exact i32 DP4A dots per sub-block, the
  per-sub-block scale fold (Q4_K's expression for
  Q4_K and Q5_K; Q6_K's per-half fold with the -32
  offset). Per-thread sub-tile is the same 4x4; with
  four batch lanes the workgroup is `tile_n` threads.
- Q5_K's stage assembles each element's 5-bit quant
  into its own byte (ql nibble plus bit `j` of
  `qh[l]`), K14's assembly idiom; it is the only new
  stage. Q4_K's and Q6_K's stages are their MMQ
  siblings' text.
- Staging bytes: Q4_K and Q5_K 15,424 at tile 32
  (fits a default-limits 16 KiB grant), 25,664 at
  tile 64; Q6_K 16,448 at tile 32 (64 bytes over the
  default grant, so Q6_K reports absence there and
  keeps GEMV, the K13/K14 contract), 26,688 at 64.
- Refusals by name: batch outside 1..=16, width not a
  whole number of super-blocks, tile widths outside
  {32, 64}, staging over the device's grant.

## Routing, and the shipped default

One wiring place: `prepare_proj` in session.rs, the
plain-projection ladder every layer projection in
the resident chain goes through (QKV batch, the
plain attention-out form, FFN gate/up). Per family
the ladder is now: matrix forms at 16 rows and up
(unchanged), then K18 inside the configured band,
then GEMV. The band is config in compute.rs, wide-MMQ
style: `WGPU_COMPUTE_SMALL_BATCH_MAX_ROWS`, resolved
at backend construction and logged. The shipped
default is 1, which is OFF: no batch qualifies.

The default is deliberate, and it is the wide-MMQ
lesson applied to this lane's own kernel: K15/K16
were numerically correct, entered the ladder, and
then measured 3.5x slower per tensor and hung a real
session. K18's streaming math is sound and its
parity is gated, but nothing about it is measured on
a real card yet, so merged main must not route a
single projection through it until the server proof
below says otherwise. The proof moving the default
is a follow-up commit, not this lane's assumption.

Out of this lane's scope, plainly:

- The gated and SwiGLU projection forms keep their
  own ladders (their quantize is a different K12
  variant); at small rows they still GEMV. The
  verify forward's down projection is one of them,
  so a wired round keeps that tensor's per-row
  streams until a follow-on extends K18's wiring.
- The output head keeps K3's fused finish: the
  block-logits path prepares it directly, not
  through prepare_proj, because the softcap finish
  must not change family with batch. Its 0.92 GB
  stream per row set is not covered here.
- The DFlash draft's block forward composes its own
  GEMV calls inside dflash_session.rs; its 57 ms is
  the same disease in a second patient and is the
  named follow-on after the verify side proves.

## Expectations, labeled

These are arithmetic on recorded measurements, not
results:

- Stream floor: one pass over the target's streamed
  projection bytes (about 16 GB) at the measured
  815 GB/s is about 20 ms, against about 140 ms for
  today's 7-row GEMV set. That is the floor the
  one-stream design aims at.
- Compute cross-check: the recorded MMQ per-tensor
  rate (Q4_K m64 product at 41,133 tok/s on the
  file's largest Q4_K tensor) works out to roughly
  1.4e12 DP4A/s sustained. A 7-row verify set is
  about 2.4e10 DP4A at the model's streamed element
  count, which prices the compute floor in the same
  20 ms class. The two floors agree, which is the
  design's premise: at these row counts neither DRAM
  nor DP4A should dominate if the small tiles hold
  MMQ-class efficiency.
- Landing band, stated as an expectation: the 7-row
  verify projection set lands between 20 and 45 ms
  (from about 140). The risks that could push it out
  of the band are the small workgroups' staging share
  and register pressure in the unrolled inner loop;
  the rows table below exists to measure exactly
  that.
- Round-level honesty: even at the band's low end,
  K18 alone does not close T-9.6's 88.6 bar. The
  draft's 57 ms, the unwired down projection and
  head, and acceptance economics at depth 6
  (1.97 tokens per round at acceptance 0.172) all
  still stand. K18 removes the verify stream
  multiplier, which the round accounting names as
  the largest single term.

## Server proof sequence

On the server (RX 7900 XTX), staged from the lane
tip, one heavy job at a time:

1. Parity gates execute (no skips):
   `cargo test -p wgpu-compute --test k18_small_batch -- --nocapture`
   Four tests: per-family parity at batches
   [1, 2, 3, 7, 15, 16] against the host quantized
   reference at the reassociation bar and against
   the family's GEMV product inside the 5% band
   with the tie-ruled argmax check, plus the deep
   geometry test at width 768 and 230 output rows.
2. Compile leg executes:
   `cargo test -p wgpu-compute --test shader_compile k18_small_batch`
   on the card's full limits (all candidates of all
   three families compile) and the default-limits
   positions (Q4_K/Q5_K build at s32, Q6_K absent).
3. Rows table on the real tensors:
   `cargo run --release -p wgpu-compute --example small_batch_bench`
   Read the gemv vs `k18 s64 + quantize` lines at
   rows 2, 7, and 16 per family, and the mmq line
   at 16. The claim to check: k18 + quantize beats
   gemv at every row from 2 up, by a factor growing
   with rows, and does not lose to mmq at 16 by
   enough to matter (the session routes 16 to mmq
   first anyway).
4. Session verdicts, threshold on vs off:
   `WGPU_COMPUTE_SMALL_BATCH_MAX_ROWS=16` in the
   environment, run itak-spec's `spec_generate_gpu`
   against the real target and draft files at
   depth 6 and depth 1, greedy, Stage 1 prompt.
   Control: the same runs with the variable unset.
   IDENTITY and GOLDEN must PASS in both positions;
   compare effective tok/s against the recorded
   baselines (depth 6: 9.21, depth 1: 14.60).
   The construction log must show
   `small batch config: max_rows=16` in the on
   position and `max_rows=1` in the control.
5. If 3 and 4 hold, the follow-up commit moves
   `DEFAULT_SMALL_BATCH_MAX_ROWS` to 16 (at batch
   16 the matrix branch still owns the row when MMQ
   exists; 16 only extends K18's reach to devices
   without it) and records the measured rows table
   and verdict numbers in this note.
