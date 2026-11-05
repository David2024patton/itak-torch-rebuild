# Chunked GEMV arm: the dispatch-shape control for the small-batch chase (2026-11-05)

Branch `lane/chunked-gemv`, from main `88cb83a`. VM gates only in this
note; the speed verdict is the server head-to-head protocol at the end.

## Why this arm exists

The streaming-rate research (2026-11-04) carries the strix-llama.cpp
RADV finding: through a shared-weight small-batch path, batches 3, 5,
and 6 ran several times slower than 1, 2, and 4, and the fork's fix
was splitting a batch into scaling (power-of-two) column counts. Our
own K18 shared-weight product shows the same band shape problem (it
streams at a fraction of GEMV's rate). Two explanations are live:

1. The shared-weight kernel shape itself is the problem on RADV.
2. The dispatch shape is the problem: one dispatch at an arbitrary
   row count in the 2..15 band schedules badly, whatever the kernel.

The MMV register-streaming prototype (lane/mmv-prototype) tests a new
kernel. This arm tests explanation 2 with no new kernel at all: it
serves the same batch through the shipped GEMV kernels, split into
dispatches whose row counts are all powers of two. If chunked GEMV
recovers the band, the win was dispatch shape and the cheap fix
ships. If it does not, the kernel shape is convicted and MMV (or its
successor) is the road. Either result decides something; that is the
point of a control arm.

## The strategy

In `prepare_proj` (crates/wgpu-compute/src/session.rs), the single
routing point for plain projections, a new rung sits between the K18
small-batch rung and the whole-batch GEMV fallback, for all three
quant families (Q4_K via K1, Q6_K via K2, Q5_K via K3):

- Band: batch in 2..=WGPU_COMPUTE_CHUNKED_GEMV_MAX_ROWS and below
  GEMM_MIN_ROWS (16). The env resolves exactly like the K18
  threshold (compute.rs, `resolve_chunked_gemv_max_rows`); the
  shipped default is 1, which is OFF, so the ladder is byte-for-byte
  the shipped ladder until a proof moves the value.
- Decomposition (`pow2_chunks`, unit-gated): greedy largest
  power-of-two first. 7 becomes rows [4, 2, 1] at starts [0, 4, 6];
  a power-of-two batch stays one dispatch.
- Each chunk is prepared by the kernel's new `prepare_rows`: the
  SAME pipeline and the SAME shader, with bindings 1 (input) and 2
  (output) bound as buffer windows (`wgpu::BufferBinding` with
  offset and size) over rows `start..start+len` of the resident
  buffers, so the kernel sees the window as its whole batch. The
  weights binding stays whole; the params block carries the chunk's
  row count. `prepare_inner` gained an optional window parameter;
  with no window it binds exactly as before (`as_entire_binding`),
  so every existing prepare path is untouched. Shaders are
  byte-identical: no WGSL changed anywhere in this arm.
- Window validation is shared (`resolve_row_window` in
  k1_q4k_gemv.rs, used by all three kernels): an empty window, a
  window past either buffer, or a window offset that is not a
  multiple of the device's `min_storage_buffer_offset_alignment`
  refuses with a named `WgpuError::InvalidTensor`. The session's
  `try_chunked_*` composition catches a refusal at plan time, logs
  it by name through the diag channel, and returns None, so the
  whole-batch GEMV form serves that projection. Composition never
  substitutes inside a dispatch.
- Alignment reachability, stated plainly: a window at row r needs
  r * in_row_bytes and r * out_row_bytes to be multiples of the
  device alignment (256 on RADV). Every Glimmer projection
  geometry clears it at every row: the input widths (6656, 19968)
  and output widths (256 for each KV projection, 4096 for Q, 6656,
  19968) are all multiples of 64 elements, so every f32 row stride
  is a multiple of 1024 bytes and every window offset aligns. A
  hypothetical geometry whose row stride was not a multiple of
  the alignment would refuse at its first misaligned window and
  fall back whole, per projection, by name; no Glimmer projection
  geometry hits that case.
- Scope boundary: `prepare_gated_proj` and `prepare_swiglu_proj`
  are separate routing functions and do NOT route through
  `prepare_proj`; the gated output projection and the SwiGLU pair
  are not chunked by this arm. If the arm wins the head-to-head,
  extending the same rung to those two functions is the follow-up.

## Quantize-once: nothing to reuse, nothing duplicated

The chunked form prepares zero quantize dispatches, so the C3
quantize-once path is neither reused nor violated: there is no
place in this arm where chunking could requantize. The GEMV forms
never consult the HShare and never read the q8 buffers (K1/K2/K3
prepare takes no quantize input); each chunk reads its activation
rows directly from the same resident input buffer the whole-batch
dispatch reads, through a window binding. The quantized forms'
sharing (MMQ riding the layer share, K18 preparing its own) is
untouched, and `ProjCall::dispatches()` counts a chunked call as
exactly its chunk dispatches, no quantize term.

## Dispatch counts per row count, before and after

Per plain projection in the routed band. Before: one GEMV dispatch
at the batch's own row count (or 2 dispatches on the K18 rung when
that env is on: K12 quantize + K18). After, with the chunked env on
and K18 off: one GEMV dispatch per power-of-two chunk.

| rows | before (GEMV) | after (chunked) | decomposition |
|------|---------------|-----------------|---------------|
| 2    | 1             | 1               | [2] |
| 3    | 1             | 2               | [2, 1] |
| 4    | 1             | 1               | [4] |
| 5    | 1             | 2               | [4, 1] |
| 6    | 1             | 2               | [4, 2] |
| 7    | 1             | 3               | [4, 2, 1] |
| 8    | 1             | 1               | [8] |
| 15   | 1             | 4               | [8, 4, 2, 1] |

Routing priority when both small-batch envs are set: K18 first,
then chunked, then whole-batch GEMV, so the head-to-head arms differ
by exactly one env flag each.

## Instrumentation: the brief's claim verified, nothing added

Claim: the diag timestamps system already times each GEMV dispatch
individually, so this arm needs no new instrumentation. VERIFIED in
code: every K1/K2/K3 prepare (including the new `prepare_rows`,
which flows through the same `prepare_inner`) stamps itself through
`ctx.diag_stamp(DispatchMeta { kernel: "k1_q4k_gemv" /
"k2_q6k_gemv" / "k3_q5k_gemv", rows, .. })`, and each windowed call
is its own prepared call with its own stamp, so a chunked batch
appears in the WGPU_COMPUTE_DIAG=timing report as one timed entry
per chunk dispatch. One addition, cosmetic only: the stamp's shape
string for a windowed call carries a ` win@<first_row>` suffix, so
chunks are distinguishable from whole-batch dispatches and from
each other in the report. No timing code was added or changed.

## Correctness gates (VM, 2026-11-05)

This VM enumerates zero Vulkan adapters, so every GPU test in this
crate (the resident session tests, the K18 gates, and the new file
below alike) prints its skip line and passes vacuously here, by the
harness's standing design; the executing proof is the server
protocol's step 0. What the VM gates do prove: full compilation of
lib and tests under clippy -D warnings, the decomposition and
config-resolution unit tests executing, and the whole suite green
with the env OFF (the shipped ladder, byte-identical dispatch
expectations) and the named session suites green with the env ON.

- `cargo test -p wgpu-compute`: PASS. 80 unit tests (including new
  `session::tests::pow2_chunks_decompose_the_verify_band`,
  `session::tests::pow2_chunks_cover_any_batch_contiguously`, and
  `compute::tests::chunked_gemv_max_rows_resolution`) plus all
  integration suites, 0 failed.
- New `crates/wgpu-compute/tests/chunked_gemv.rs`: 4 tests. Per
  family (Q4_K, Q6_K, Q5_K), for batches [2, 3, 4, 5, 6, 7, 15],
  the chunked product (windowed prepares at the pow2 decomposition
  into one output buffer) must equal the whole-batch GEMV product
  BIT-EXACTLY (each output element's reduction is independent of
  its batchmates, so anything less than bit equality convicts the
  windowed binding). Plus the refusal gate: empty window, window
  past either buffer, and the alignment refusal asserted
  conditionally from the device's own reported alignment (the
  fixture's 400-byte output rows make row 1 misaligned exactly
  when the device alignment does not divide 400). On this VM:
  skip-pass, as designed; they execute in the server protocol.
- Env OFF (default): the full suite above is the OFF run; routing
  cannot engage (band requires batch <= 1 resolved threshold), so
  dispatch expectations are the shipped ones by construction.
- Env ON: `WGPU_COMPUTE_CHUNKED_GEMV_MAX_ROWS=15 cargo test -p
  wgpu-compute --test resident_session --test resident_block`:
  PASS (resident_session 4/4, resident_block 2/2; skip-pass on this
  VM, executing in the server protocol's step 0).
- `cargo test -p itak-spec`: PASS, 47/47.
- `cargo fmt --check`: clean. `cargo clippy -p wgpu-compute
  --all-targets -- -D warnings`: clean. `cargo check --workspace`:
  clean.

## Head-to-head verdict protocol vs MMV (server, cleared card)

Same discipline as every proof in this repo: one staging tree
synced to the exact commit under test, hash verified, the phase
lock held, one heavy job at a time.

0. Correctness first, per arm, before any timing: identity check
   (raw completion first 8 ids = [13796, 26, 1573, 373, 290, 2167,
   26899, 1212]) and the golden subset agreement on the real GGUF,
   plus this crate's resident session and chunked_gemv tests
   executed on the card with the arm's env on. An arm that fails
   identity does not get timed.
1. Arms, one env flag differing per run, everything else identical
   (same binary except the flag, same prompts, same seed):
   - Control A (shipped): no env. Plain GEMV in the band.
   - Control B (K18): WGPU_COMPUTE_SMALL_BATCH_MAX_ROWS=15.
   - Arm MMV: WGPU_COMPUTE_MMV_MAX_ROWS=15 (lane/mmv-prototype).
   - Arm CHUNKED: WGPU_COMPUTE_CHUNKED_GEMV_MAX_ROWS=15.
2. Measurements per arm: speculative verdict at depths 1 and 6
   (the T-9.6 shape), plain single-token decode re-check, and one
   depth-1 run under WGPU_COMPUTE_DIAG=timing for the per-kernel
   table (the chunked arm's per-chunk GEMV entries vs MMV's kernel
   entries vs K18's quantize+product pairs).
3. Verdict rules, fixed before the runs: the arm that posts the
   best depth-1 and depth-6 verdicts with identity intact wins the
   band. If CHUNKED ties or beats MMV, dispatch shape was the
   defect and the cheap arm ships (default decision then: raise
   DEFAULT_CHUNKED_GEMV_MAX_ROWS to 15 in a follow-up commit). If
   MMV wins clearly, kernel shape was the defect and MMV ships.
   If neither beats Control A, both arms close and the chase moves
   to the next research item (SQTT of the GEMV kernel in the band).
