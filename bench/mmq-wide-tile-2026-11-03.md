# Wide-tile double-buffered MMQ for prefill, 2026-11-03 (draft)

Status: code landed on the build lane; parity and compile gates run on
the build VM (results below); the server rate proof is scripted and
waiting on a free GPU window. Nothing in this note claims a server
rate until `bench/mmq-wide-bench.sh` has run there.

## The problem this attacks

From `bench/gpu-resident-2026-10-30.md`'s session record on the
RX 7900 XTX: the MMQ build prefills at a marginal 161 tok/s (gross
135.8 tok/s at 1,024 tokens) against llama.cpp Vulkan's 998 tok/s on
the same card. That note names the causes, in order: (1) occupancy:
the m64 MMQ tiles stage 41-45 KB of workgroup memory, so one
workgroup is resident per CU, one super-block deep, with a barrier
per super-block (78 barriers across one ffn_down workgroup), so
latency hiding rests on in-thread ILP alone; (2) the Q6_K fold does
twice the f32 work per dot; (3) the non-projection chunk chain was
28% of marginal time at GEMM-era measurement, which bounds the
session marginal near 330 tok/s even with infinitely fast
projections, until that chain is profiled and attacked separately.
This lane attacks cause (1) only. The note's own named next step
was "the wide-tile (128x128-class) double-buffered MMQ plus a
profile of that chain"; the chain profile remains open work.

## What shipped

Two new kernels behind the existing selection seam:

- `crates/wgpu-compute/src/kernels/k15_q4k_mmq_wide.rs`: K15, the
  Q4_K wide-tile double-buffered MMQ.
- `crates/wgpu-compute/src/kernels/k16_q6k_mmq_wide.rs`: K16, the
  Q6_K sibling.

Both consume exactly what K13/K14 consume (the weights' existing
device layouts, K12's quantized activation buffers) and write the
same output layout, so the session swaps product forms without
moving a byte. The math is K13/K14 term for term: exact i32
DP4A dots, the per-sub-block scale fold in f32, no downcast before
the accumulation completes. Decode (1-row) paths are untouched:
the wide form exists only on the matrix path past
`GEMM_MIN_ROWS`, same as the standard MMQ.

Selection ladder (session.rs, via `mmq_product_q4k` /
`mmq_product_q6k`, shared by the plain, gated, and SwiGLU forms):

1. Wide product (K15/K16) when the chunk batch is at or past the
   backend's wide row threshold.
2. Standard product (K13/K14) behind it.
3. GEMM form (K10/K11), then the GEMV forms, as before.

Every refusal in the ladder is logged by name through the crate's
diag seam; the choice is made at plan-build time, never inside a
dispatch.

The wide row threshold is config, not a compiled constant:
`WGPU_COMPUTE_MMQ_WIDE_MIN_ROWS`, default 32, resolved at backend
construction in compute.rs and logged. Setting it to 65 (past the
64-row chunk cap) removes the wide form from the ladder entirely:
that is the A/B control position the bench script uses, so old vs
new is one process flag, not a rebuild.

## Tile geometry, and why

Tile: `tile_n` output rows x the full 64-row batch, computed by
`tile_n * 2` threads. Each thread owns a 4-output-row x 8-batch-row
sub-tile (32 accumulators). Against K13/K14's 4x4 sub-tiles:

- The batch axis doubling (4 to 8 rows per thread) is what makes
  the wide tile affordable: every staged weight word feeds 8
  activation rows instead of 4, so the weight tile for 128 output
  rows costs the staging of K13's 64, per step.
- Workgroup sizes are 128-256 threads (vs K13's 128/256 at
  m32/m64), so the wider tile does not shrink the thread count
  that hides latency inside the workgroup.

K direction: staging runs in half-super-block steps of 4
sub-blocks (128 elements). A step is the unit the ping-pong
pipeline moves. The half split is not arbitrary: it is Q4_K's own
group boundary (sub-blocks 2h and 2h+1 share a group's raw words)
and Q6_K's own half boundary (sub-blocks 0-3 are h=0, 4-7 are
h=1), so neither family's staging straddles a step.

Double buffering: two parity buffer sets, generated as separate
named arrays (w_qs0/w_qs1, x_qs0/x_qs1, and the scale planes).
The step loop is unrolled by parity (steps processed in pairs) so
every workgroup-memory offset in the inner product is a
compile-time constant: K13's record is explicit that its first
runtime-indexed build ran at the f32 GEMM's rate, and the wide
kernel keeps that addressing discipline. The pipeline form is:
prologue stages step 0, one barrier; then per pair, stage step
s+1 into parity 1 (its global loads issue first), compute step s
from parity 0, one barrier, stage step s+2 into parity 0,
compute step s+1, one barrier. One barrier per step, against
K13's two barriers per super-block (four per two steps), and the
next step's loads are in flight across the current step's
compute. Parity buffers are only overwritten after a barrier
separates the overwrite from the buffer's last reads.

Why not a full-super-block (or deeper) double buffer: it does
not fit. Workgroup storage for Q4_K totals 320*N + 20,992 bytes
at tile width N (per parity buffer: weight words 128 B/row,
weight scale pairs 32 B/row, the skewed activation tile 8,448 B,
the activation scale pairs 2,048 B), so a full-K-depth scheme
at any useful width, or a 128-wide tile with whole-super-block
steps (640 B/row/parity), exceeds the 64 KiB grant outright. The half-step pipeline at w128 stages 61,952
bytes total for K15. For K16 the same geometry costs 5,120 bytes
more across the two parities (its activation record is a vec4,
carrying the two precomputed half sums the Q6_K fold needs, where
Q4_K's is a vec2), so w128 would stage 66,048 bytes: 512 bytes
over the grant. K16 therefore ships w112 (60,928 bytes) on a
64 KiB device. The candidate ladder [128, 112, 96, 64] is shared;
each family ships the widest candidate its own staging fits, and
refuses by name below the narrowest. The recorded staging table:

| shape | K15 (Q4_K) bytes | K16 (Q6_K) bytes |
|-------|------------------|------------------|
| w128  | 61,952           | 66,048 (over)    |
| w112  | 56,832           | 60,928           |
| w96   | 51,712           | 55,808           |
| w64   | 41,472           | 45,568           |

The activation tile keeps K13/K14's bank-conflict fix: row
stride 33 words (32 + 1 skew) per step.

## Parity results (build VM)

Run 2026-11-03 on the build VM, `cargo test -p wgpu-compute`
(debug profile): all 13 test targets pass, 98 tests total,
including the 48 lib unit tests. Of those, the wide kernels'
own in-module tests pass: the staging table above (asserted
byte for byte at all four candidate widths, both families),
the shipped-shape selection against storage grants (16,384 B
grants nothing; 41,472 B grants K15 w64; 45,568 B grants K16
w64; 65,536 B grants K15 w128 and K16 w112), the candidate-set
refusal, and the shader-structure tests (workgroup size,
DP4A presence, both parity buffer sets present, exactly
three `workgroupBarrier()` sites in the generated source).
The backend's wide-threshold config resolution is unit-tested
(default, explicit values, the 65 control position, garbage
input falling back to the default). `cargo fmt --check` and
`cargo clippy -p wgpu-compute --all-targets` are both clean.

What did NOT run here, stated plainly: this VM enumerates no
wgpu adapters at all. The six GPU parity tests in
`crates/wgpu-compute/tests/k15_k16_mmq_wide.rs` (batch sweep
1-64 at every buildable wide shape for both families, the
deep-width and short-width pipeline geometries, the gated
and SwiGLU activation forms, the prepare refusals) therefore
skip under the default selection policy, exactly as the
K13/K14 gates do on this machine, and the shader-compile leg
in `tests/shader_compile.rs` reports the same absence: the
generated WGSL has not yet been through a driver's validator
anywhere. Both gaps close on the server before any rate is
trusted: the parity file runs there under
`cargo test -p wgpu-compute --test k15_k16_mmq_wide` (the
server adapter is policy-eligible), and both bench vehicles
construct the K15/K16 pipelines before timing anything, so a
WGSL defect fails the run loudly at construction instead of
producing a number.

## What the server run must prove

First the correctness gate that could not run on the build VM:
`cargo test -p wgpu-compute --test k15_k16_mmq_wide` on the
server must pass all six parity tests with the measured diffs
and quantization bands printed (the K13/K14 record's bands were
0.36% for Q4_K and 0.42% for Q6_K; the wide kernels run the
same math, so bands in that neighborhood are the expectation,
and anything visibly wider is a defect, not a tuning fact).

Then run `bench/mmq-wide-bench.sh` on ubuntu-server with the
GPU free. It builds both bench vehicles, then runs the
resident session (`glimmer_generate --gpu`) on the real
Glimmer file at two repeated-sentence prompt lengths (about
254 and about 1,024 tokens) in both ladder positions, and
finishes with the per-tensor `mmq_bench` table at batch 64.
The run proves the lane if, in the same window:

1. Per-tensor: the `mmq wide w128` (Q4_K) and `mmq wide w112`
   (Q6_K) rows beat the best standard-MMQ rows (`mmq m64`) on the
   largest tensors of each family, product-only and with the
   quantize composed in.
2. Session: gross prefill tok/s at the long prompt and the
   marginal rate between the two lengths both improve over the
   old-ladder position in the same run. The comparison that
   matters is old vs wide in one window, not either number
   against the 2026-10-30 record (different window, different
   clocks).
3. Canary discipline from the earlier notes still holds: the
   batch-1 GEMV stream rates in the per-tensor table stay in
   their healthy band (~816 GB/s Q4_K, ~832 GB/s Q6_K), so a
   slow window cannot masquerade as a kernel verdict.
4. The generated tokens are unchanged: same prompt, greedy
   session, the wide ladder must produce the old ladder's token
   stream (the parity tests gate the math; this gates the
   wiring).

Expectation management, stated before the run: even a perfect
projection kernel cannot lift the session marginal past roughly
330 tok/s while the non-projection chunk chain keeps its
measured share (cause 3 above). A wide-MMQ win that lands the
marginal in the 200-330 band is the design working as intended;
closing the rest of the gap to llama.cpp's 998 is the chain
profile and whatever it indicts, which is the next lane, not
this one.

## Open edges

- The default wide threshold (32 rows) is a reasoned starting
  value, not a measured one. The bench script's per-tensor table
  is at batch 64; if the server window allows, a batch-32
  mmq_bench run (`--batch 32`) checks the threshold's edge
  directly, and the default moves if the table says so.
- Occupancy after this change: w128/w112 stage 57-62 KB, still
  one workgroup resident per CU. The win, if it comes, comes
  from barrier count, load/compute overlap, and per-thread ILP
  depth (32 dots per sub-block step against 16), not from
  residency. If the server table shows the wide rows at parity
  with m64, the next lever is a narrower activation record
  (staging the Q6_K half sums only, not the vec4) to buy a
  second resident workgroup, and that trade belongs in a
  follow-up note with this run's numbers attached.
