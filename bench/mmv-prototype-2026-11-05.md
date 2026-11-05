# K19 MMV prototype: the register-streaming small-batch product (2026-11-05)

Lane `lane/mmv-prototype`, from main e537764f. New
kernel K19 (`crates/wgpu-compute/src/kernels/
k19_q4k_mmv.rs`), routed in `prepare_proj` behind
`WGPU_COMPUTE_MMV_MAX_ROWS` (default 1 = OFF). No
server GPU work in this lane; the card verdict is
sequenced below for a separate lane.

## Why a third attempt, and what it changes

Two one-stream attempts are measured and convicted:

- Wide MMQ (K15/K16): numerically correct, about
  3.5x slower per tensor than standard MMQ, one real
  session hang. The shader autopsy
  (bench/shader-autopsy-2026-11-04.md) shows why in
  compiled form: 1,041 to 1,213 spilled VGPRs and
  78,080 to 86,528 bytes of scratch on the wide
  shaders.
- K18 (bench/small-batch-2026-11-04.md): correct,
  streams the weights once as designed, but at
  about 70 to 130 GB/s against GEMV's up to
  676 GB/s on the same tensors; a loss in the spec
  verdict at both depths (9.19 to 8.13 at depth 6,
  14.22 to 8.90 at depth 1).

What the two slow kernels share and the fast one
lacks is weight staging through workgroup memory:
both stage weight tiles in LDS (K18: 15,424 to
25,664 bytes per workgroup; the wide MMQ's compiled
LDS runs to 62,464 bytes), and the autopsy cleared
GEMV's compiled form and the compiler with it
(GEMV: VGPR 48 to 72, zero spills, zero scratch,
LDS only 1 to 4 KB of reduction partials). The
streaming-rate research (run folders
`~/workspace/research_notes/what-is-everything-
that-can-be-done-to-a-20261104-1632/` and
`~/workspace/research_notes/gpu-streaming-rate-
exhaustive-20261104-2133/report.md`) reached the
same verdict from the literature side: llama.cpp's
fast small-batch Vulkan path dequantizes weights in
registers and never stages them.

K19 is that shape, built in this crate: the Q4_K
small-batch product (batch 2..=16) as a widened
GEMV, not as a shrunken MMQ.

## What it borrows from mul_mat_vec, with sources

The design was read from the actual sources, not
reconstructed: `ggml/src/ggml-vulkan/vulkan-shaders/
mul_mat_vec.comp` (264 lines) and
`mul_mat_vec_base.glsl` (242 lines) in the server's
llama.cpp tree at `/home/server/llama.cpp`; the same
files, at the same line counts, sit in this VM's
meta-cookbook mirror at `~/workspace/meta-cookbook/
build-site/repos/ggerganov__llama.cpp/`. Term for
term:

- Register accumulators per (column, row):
  `temp[NUM_COLS][NUM_ROWS]` in the source is
  `acc[rows * batch]` here, one f32 accumulator per
  (output row, batch row) pair, every index a
  compile-time constant (see the WGSL note below).
- The batch is a column count inside the workgroup
  (NUM_COLS), specialized per batch count from 2 to
  16. The dose-response that justifies the cap is
  llama.cpp issue #25356, as recorded in the
  research lane digest
  (`~/workspace/research_notes/what-is-everything-
  that-can-be-done-to-a-20261104-1632/lane-C-kernel-
  techniques.md`, line 5): the stock NUM_COLS cap of
  8 cliffs at batch 9 (Mistral-Small-24B 125.0 t/s
  at batch 8 down to 19.7 at batch 9); a cap of 16
  gives 131.5 at batch 9 and 158.5 at batch 16,
  beating the matmul path even at batch 32. PR
  #27652 proposes raising the cap under
  `GGML_VK_MMV_MAX_COLS`.
- Rows per workgroup (NUM_ROWS) is 2, the value
  ggml-vulkan picks for K-quants on RDNA3 and the
  tile K1 already ships.
- Per super-block, weights are dequantized in
  registers and dotted against the activation quads
  (`dequantize4` + `get_dm` in the source). Here the
  dequant and dot text is K1's llama-family WGSL
  verbatim: the sixteen-thread group assignment,
  the packed scale extraction, and the dot math are
  character-for-character K1's, because K1 is itself
  a proven port of `mul_mat_vec_q4_k.comp` and this
  prototype's question is workgroup organization,
  not dequant arithmetic.
- The reduction keeps the source's one shared-memory
  exception: llama's hybrid mode writes per-group
  partials to `tmpsh`; K19's only workgroup array is
  its reduction partials (at most 256 f32 in the
  subgroup form, 128 in the tree form). No weight
  byte and no activation tile is staged anywhere.
- The input is the f32 activation rows themselves,
  as in the source: no K12 quantize dispatch rides
  in front, and the output layout is the GEMV's
  `[batch][out_rows]`, so a plan can swap K19 for K1
  without moving a byte. Its dispatch count in a
  plan is 1, the GEMV's own count.

## How it differs from K18 and the wide MMQ

| | K18 | Wide MMQ (K15/K16) | K19 |
|---|---|---|---|
| Weights in workgroup memory | Yes, staged tiles (15,424 to 25,664 B) | Yes, double-buffered staged tiles (LDS to 62,464 B compiled) | No. Reduction partials only (at most 1 KB) |
| Activations | Q8_1 quantized on device (extra K12 dispatch), DP4A integer dots | Same MMQ math | f32 rows directly, f32 dots: the GEMV's exact math, no extra dispatch |
| Batch in the shader | Runtime loop bound over staged tiles | Runtime, in tile geometry | Generation literal, accumulators fully unrolled |
| Tile | 32/64 output rows x batch | 128-class output tile | 2 output rows x the whole batch (llama's NUM_ROWS / NUM_COLS split) |
| Correctness bar | Parity band vs GEMV (quantization noise) | Parity band vs GEMV | The CPU f32 reference at the reassociation bound, plus direct K1 agreement |

## The env flag and the ladder

`WGPU_COMPUTE_MMV_MAX_ROWS`, resolved at backend
construction like its two predecessors
(`WGPU_COMPUTE_MMQ_WIDE_MIN_ROWS`,
`WGPU_COMPUTE_SMALL_BATCH_MAX_ROWS`), default 1 =
OFF: `prepare_proj` never consults K19, and shipped
behavior is byte-identical (the off position is
covered by the crate's existing dispatch-count pins,
which do not move). With the band open, K19 is tried
in the Q4_K branch after the matrix ladder and
before K18: when both thresholds are enabled, the
register-streaming form takes precedence, because it
is the band's structural challenger and the A/B
should not be confounded. Every refusal inside the
band is logged by name and the ladder continues to
K18, then K1; nothing substitutes inside a dispatch.

Diag instrumentation is in from birth (David's
rule): K19 prepare stamps `k19_q4k_mmv` through
`ctx.diag_stamp` like every other kernel, with rows,
cols, quant `Q4_K`, weight bytes, and a
`batch=<n> grid=<x>x<y>` shape string, so the
timestamp system can time it the moment the server
lane turns `WGPU_COMPUTE_DIAG=timing` on.

## Where WGSL could not follow the GLSL design

Named precisely, because the verdict must be read
against them:

1. No specialization constants. llama selects
   NUM_COLS / NUM_ROWS / BLOCK_SIZE through GLSL
   `constant_id`; WGSL through wgpu in this crate
   has no equivalent in use, so the batch is baked
   into the generated shader as literals (the
   crate's own idiom: K13 and K18 also generate).
   Cost: 15 compiled variants, one per batch count.
   Mitigations in the build: a per-batch pipeline
   cache, the band-floor variant compiled at
   construction as the family's smoke check, and
   `K19Pipeline::precompile` so a session warms the
   counts it will run at plan time.
2. Register residence is a compiler outcome, not a
   WGSL guarantee. A runtime-indexed local array in
   WGSL is free to spill to function memory, which
   would quietly rebuild the staging this kernel
   exists to avoid. The generator therefore unrolls
   the (row, batch) loops at generation time so
   every accumulator index is a compile-time
   constant: the strongest position WGSL offers,
   and the same trick K13/K18's generators use for
   their inner products. Whether ACO actually keeps
   `acc` plus the dequantized quads in VGPRs at
   batch 16 (32 accumulators, 16 quad values, scales
   and headers per thread) can only be proven by
   the server shader autopsy in step 0 of the
   protocol below. If a spill cliff exists, the
   dose-response sweep (step 4) finds where.
3. Activation quads are re-read per output row.
   llama loads the B vector's quads once per
   super-block into registers (`bv0`/`bv1`) and
   shares them across all NUM_ROWS rows. K19
   re-reads them from the input buffer per row, the
   trade K1's port already made: the activations
   are the small side (16 rows x width, hundreds of
   KB against the weight stream's GBs) and ride the
   cache, while the weights, the stream that
   matters, are loaded and dequantized exactly once
   per workgroup. Deliberate: the prototype differs
   from K1 in exactly one structural variable, the
   workgroup organization.
4. The subgroup reduction needs the SUBGROUP
   device feature, with K1's workgroup-tree
   fallback where it is not granted. The server
   card grants it (K1's shipped shape uses it), so
   the verdict runs the subgroup form; the tree
   form is gated in this lane's tests wherever a
   device builds it.

## Gate results on this VM

Run on the lane tip before this note was committed:

- `cargo test -p wgpu-compute`: 151 passed, 0
  failed (lib binary 82, including K19's four unit
  tests and the `mmv_max_rows` resolver test).
- `cargo test -p itak-spec`: 47 passed, 0 failed
  (the off-path contract: with the env unset, the
  spec suite's behavior is unchanged).
- `cargo fmt --check`: clean.
- `cargo clippy --workspace --all-targets`: clean,
  no warnings.
- `cargo check --workspace`: clean.

What this VM could and could not prove, named
plainly: this machine has no GPU adapter at all for
wgpu to enumerate (no Vulkan ICD, not even a
software one), so every GPU-executing test in the
crate skips here by design, and K19's are no
exception: the three correctness tests in
`tests/k19_mmv.rs` (batches 2, 3, 5, 7, 9, 16
against the CPU reference and against K1, the
768 x 101 deep geometry, the band-edge refusals)
printed the no-adapter refusal and passed, and the
K19 leg of `tests/shader_compile.rs` printed its
no-adapter skip and passed. Two shader gates did
execute here:

- Naga validation (unit test
  `every_variant_validates_under_naga`, naga 27 as a
  dev-dependency): all 30 generated variants
  (batches 2..=16 x the tree and subgroup shapes)
  parse and validate under the same front end wgpu
  feeds the driver with. Well-formed and well-typed
  is proven; driver acceptance and register
  behavior are not, and are not claimed.
- A visual read of the generated batch-16 subgroup
  shader: exactly one workgroup array
  (`partials: array<f32, 256>`, the reduction
  finish, 1 KB), the accumulator array sized 32
  with every one of its 96 uses constant-indexed
  (32 dot accumulations, 64 reduction touches),
  K1's thread assignment and dequant text verbatim,
  and the uniform whole-workgroup early-out ahead
  of every barrier.

Real-adapter compilation and GPU execution are
steps 0 and 1 of the server protocol below; until
they run, K19 is a validated-but-unexecuted
prototype, which is exactly what its shipped-off
default says.

## Server verdict protocol (the next lane)

Preconditions: the card cleared per David's
2026-11-05 order (the Qwen3-Embedding pair on port
8091 and the Qwen3-4B on 8092 gone; the spec phase
needs about 25.3 GB of the 25.75 GB card); the
phase lock `flock /tmp/itak-phase.lock` held; one
heavy job at a time; the staging tree synced to the
lane commit and its hash verified before any run;
production Glimmer stays down.

0. Shader autopsy of the compiled variants, before
   any timing: extend the bench/shader-autopsy.sh
   pattern to the K19 pipelines at batches 2, 7, and
   16 and record VGPR, SGPR, spills, scratch, and
   LDS per variant. The claims to check: zero or
   near-zero spills, LDS at the partials size only
   (1 KB subgroup form). A spill cliff at high batch
   is a result, not a surprise: it sets the ceiling
   the sweep in step 4 then maps.
1. Correctness gates execute (no skips) on the
   card:
   `cargo test -p wgpu-compute --test k19_mmv -- --nocapture`
   (batches 2, 3, 5, 7, 9, 16 against the CPU f32
   reference and against K1, both reduction shapes,
   plus the 768 x 101 deep geometry) and the compile
   leg:
   `cargo test -p wgpu-compute --test shader_compile k19_mmv`
   (all 15 batch variants, both shapes).
2. Identity + golden first:
   `WGPU_COMPUTE_MMV_MAX_ROWS=16` in the
   environment, run itak-spec's `spec_generate_gpu`
   against the real target and draft files at depth
   6 and depth 1, greedy, Stage 1 prompt. IDENTITY
   and GOLDEN must PASS in the on position before
   any speed number is read. Control: the same runs
   with the variable unset. The construction log
   must show `mmv config: max_rows=16` in the on
   position and `max_rows=1` in the control.
3. Spec A/B against the recorded baselines (K18
   note): depth 6 control 9.19 tok/s, depth 1
   control 14.22 tok/s (same-day verdict controls
   9.21 / 14.60 in the spec suite). Report effective
   tok/s and acceptance per arm, plus verify ms:
   verify is the term K19 exists to shrink.
4. Dose-response sweep: depths 1 and 6 at
   `WGPU_COMPUTE_MMV_MAX_ROWS` in {4, 8, 12, 16}.
   Two warnings from the research ride on this
   step: llama's dose-response says the win grows
   through 16, but the strix-llama fork found
   non-monotonic per-batch behavior on RADV (their
   split-column batches 3, 5, 6 ran several times
   slower than 1, 2, 4; exhaustive report, line
   136). The sweep exists to catch exactly that on
   this card and driver (Mesa 26.0.8).
5. Plain decode A/B: plain decode is batch 1,
   below the band, so K19 cannot touch it; the
   honest check is that the off and on positions
   produce the recorded plain-decode rate (35.45
   tok/s) unchanged, proving the routing did not
   leak. No MMV prefill claim is made: chunk rows
   at 16 and up belong to the matrix ladder.

Decision rule: `DEFAULT_MMV_MAX_ROWS` moves only
if identity and golden pass in the on position AND
depth 6 beats its control AND depth 1 beats its
control, with the dose-response sweep showing no
cliff inside the band the default would open.
Anything less, the default stays 1 and the sweep
table is recorded here as the finding: a register-
streaming kernel that still loses on RADV would
close the small-batch line of attack and push
T-9.6's speed half toward chunked GEMV, the
research plan's next entry.
