# llama.cpp parity specs: what llama's Vulkan backend actually runs, field by field against ours, 2026-11-06

David's question: can we open the llama.cpp repo, take the specs
out, and be on par with them? Answer: yes for the specs. This note
is the extraction, with receipts. Every llama value below was read
from the real sources and is cited to file and line; every one of
our values was read from this repo's code the same way. Fields are
marked MATCH, DIFFERENT, or ABSENT, with the measured consequence
where our bench record has one.

## Sources and verification

- llama.cpp: the server tree at `/home/server/llama.cpp`, git
  commit `a7af149a2a7768d4b1f6f02c6ce26a17fba11376` (2026-10-28),
  the tree that built `build-vulkan2`, the binary behind every
  recorded llama baseline in this repo (Vulkan prefill 998.43,
  decode 44.31 tok/s).
- The reading was done in the VM mirror at
  `~/workspace/meta-cookbook/build-site/repos/ggerganov__llama.cpp/`.
  Its git HEAD differs (74a4fb00), so the mirror was verified
  against the server tree before anything was cited from it:
  all 187 files under `ggml/src/ggml-vulkan/vulkan-shaders/` and
  `ggml/src/ggml-vulkan/ggml-vulkan.cpp` are sha256-identical
  between the two trees. Citations below name the path inside
  `ggml/src/ggml-vulkan/`; line numbers are that verified content.
- Ours: this repo at main a064b5d, `crates/wgpu-compute/src/`.

## License note

llama.cpp is MIT licensed (LICENSE, "Copyright (c) 2023-2026 The
ggml authors"). Specs, algorithms, constants, tile shapes, and
buffer layouts are facts about how the program works; reimplementing
them in our own WGSL is clean-room work and needs no permission.
Copying llama's GLSL source text into this repo would require
keeping their copyright and license notice with the copied material,
and mostly would not translate anyway: GLSL specialization constants,
`layout(local_size_x_id)`, and subgroup intrinsics have no direct
WGSL equivalents, which is why every kernel here is a re-expression,
not a port. This note extracts facts, not code.

## The host selection logic (llama's routing, for our routing lane)

This is the cross-check table for the band-routing work: the exact
thresholds llama itself uses to pick a kernel per product.

| Decision | llama's rule | Citation |
| --- | --- | --- |
| vec vs matrix path | `mul_mat_vec` when the batch (dst rows of src1, `ne[1]`) is 1, or `<= mul_mat_vec_max_cols` with the batch dims of src1 equal to 1; anything larger goes to `ggml_vk_mul_mat_q_f16` (the mul_mm family) | ggml-vulkan.cpp:7402 |
| The vec column cap | `mul_mat_vec_max_cols = 8` | ggml-vulkan-types.h:362 |
| Single-row transposed trick | One output row with batch > 8 and no fused ops is run through mul_mat_vec swapped (B^T*A) | ggml-vulkan.cpp:7394 |
| Vec workgroup size on AMD | `DMMV_WG_SIZE_SUBGROUP` (one subgroup) is the default; the LARGE (4-subgroup) variant is chosen only for NVIDIA post-Turing and Intel. The heuristic never fires for AMD | ggml-vulkan.cpp:5628-5648 |
| Rows per vec workgroup (NUM_ROWS) | Defaults rm_stdq 1, rm_kq 2; AMD GCN raises to 2 and 4; RDNA3 keeps rm_kq 2 for K-quants in the f32/f16-activation path | ggml-vulkan.cpp:2866-2874 |
| MMVQ rows on RDNA3/4 | In the Q8_1 integer vec path, NUM_ROWS is forced to 4 when NUM_COLS >= 5 ("above four columns, static 4 rows bench faster") | ggml-vulkan.cpp:2880-2885 |
| MMVQ (quantize activations for the vec path) | Used when integer dot product exists and: batch n > 1 always (subject to vendor rules); on AMD, single-column use needs k >= 2048; Q6_K weights are excluded on every vendor except Intel | ggml-vulkan.cpp:6656-6700, gate at 6783 |
| Matrix tile size class | Selector over (m, n): m <= 32 or n <= 32 gives small; m <= 64 or n <= 64 gives medium; otherwise large. On AMD the large class exists only when coopmat is present and the driver is not AMD proprietary | ggml-vulkan.cpp:2852-2859, 4807 |
| Matrix path activation choice | Activations are quantized to Q8_1 (MMQ path) when integer dot product OR coopmat integer support exists and precision allows; with coopmat present and quantized weights, f16 activations are preferred otherwise | ggml-vulkan.cpp:6431-6445 |
| Split-K | When k >= 2048 and the tile grid fills at most half the shader cores, split K by up to 8x (aligned to 256) and reduce with `mul_mat_split_k_reduce.comp` | ggml-vulkan.cpp:5916-5955 |
| Coopmat enablement | Build must have glslc coopmat support; device must enumerate VK_KHR_cooperative_matrix and `GGML_VK_DISABLE_COOPMAT` must be unset; the device feature `cooperativeMatrix` must be on. f16 shapes are then scanned: first subgroup-scope f16 x f16 shape sets coopmat_m/n/k; a 16x16x16 shape with f32 accumulate sets the f32acc flag, f16 accumulate likewise. An s8 x s8 to s32 subgroup shape sets coopmat_int_support and the integer dims. Coopmat2 is VK_NV_cooperative_matrix2, NVIDIA only | ggml-vulkan.cpp:4087-4101, 4556-4557, 4701-4748 |
| Coopmat integer MMQ on RDNA3/4 | `use_cm1_int = coopmat_int_support && (RDNA3 || RDNA4)`; the cm1 MMQ pipelines are created under exactly that condition. On this card, llama's Q4_K/Q5_K/Q6_K x Q8_1 prefill products are the cm1 (coopmat) shaders, not the scalar mul_mmq | ggml-vulkan.cpp:1877, 2481-2494 |

Ours, for contrast (session.rs, compute.rs): the matrix switch is
one constant, `GEMM_MIN_ROWS = 16` (session.rs:215); below it the
shipped default is one whole-batch GEMV dispatch per projection,
with the MMV band (2..=16), K18 band, and chunked band all behind
env flags that default off (compute.rs:107-215). llama's vec band
ends at 8 and its matrix path starts at 9; ours starts at 16 and
the 2..15 band ships as repeated per-row weight streaming.

## Family 1: mul_mat_vec (decode and small batch) vs GEMV K1/K2/K3

llama shader: `vulkan-shaders/mul_mat_vec.comp` +
`mul_mat_vec_base.glsl`. Ours: `kernels/k1_q4k_gemv.rs`,
`k2_q6k_gemv.rs`, `k3_q5k_gemv.rs` (the "llama shapes" family).

| Field | llama | Ours | Mark |
| --- | --- | --- | --- |
| Workgroup | One subgroup (BLOCK_SIZE = subgroup size, spec constant 0); dispatch x covers rows in NUM_ROWS chunks, y is the batch | 128 threads (4 subgroups of 32) | DIFFERENT |
| Rows per workgroup (NUM_ROWS, spec constant 1) | 2 for K-quants on RDNA3 (rm_kq), 1 for plain quants (rm_stdq), 4 in the MMVQ path at NUM_COLS >= 5 | 2 (shipped_shape, k1_q4k_gemv.rs:271-279) | MATCH |
| Batch per thread (NUM_COLS, spec constant 2) | 1..8 pipelines, one per column count; accumulators `temp[NUM_COLS][NUM_ROWS]` live in registers | The GEMV kernels take the batch as a runtime loop over rows in one dispatch; the register-resident batch form is the separate K19/K20/K21 family (Family 2) | DIFFERENT |
| Weight handling | Dequantized in registers per thread (`dequantize4`), never staged; K_PER_ITER 8 elements per thread per iteration for quantized A, vec4 B loads at a stride of QUANT_K/2 (mul_mat_vec.comp:12-16, 34-46) | Same: register dequant, no weight staging, vec4 traffic (LEGACY shape) and buffer_load-style wide loads in the llama shapes | MATCH |
| Reduction | subgroupAdd within the subgroup, then either nothing (one subgroup per workgroup) or a shared partials array `tmpsh[NUM_COLS][NUM_ROWS][BLOCK_SIZE]` (mul_mat_vec_base.glsl:145) | Segmented subgroup butterfly inside each 16-thread segment plus shared partials across the 128-thread workgroup | DIFFERENT |
| Activation precision | f32 or f16 B pipelines per type; f16 halves activation traffic | f32 shipped; f16 shapes exist but are bench-only (measured slower, error above the crate tolerance; k1_q4k_gemv.rs shipped_shape docs) | DIFFERENT (measured, ours is a deliberate refusal) |
| Fusion | Bias add and scale fused into the reduce step via push-constant fusion flags (mul_mat_vec_base.glsl:124-141) | K1 carries gated and SwiGLU activation forms; the head finish lives in K3's finish form | MATCH in kind |
| Measured standing | llama.cpp Vulkan decode 44.31 tok/s on this card (whole-engine figure) | Our GEMV streams at up to 816.5 GB/s in isolation (115% of llama's measured 708 GB/s for the same kernel class); our whole-engine plain decode is 35.5-35.7 tok/s | Kernel MATCH or better; engine gap is elsewhere (chain and launch structure, not this kernel) |

## Family 2: the small-batch band, mul_mat_vec at NUM_COLS 2..8 and MMVQ vs K19/K20/K21 (+ K18, chunked)

llama runs batches 2..8 through the same mul_mat_vec shader with
NUM_COLS set to the batch; past 8 the matrix path takes over.
There is no third band and no per-family threshold split. When
integer dot product is available, the band's activations are
Q8_1 (MMVQ) and the dots are integer DP4A with the Q8_1 sum
correction; Q6_K weights are excluded from MMVQ off Intel
(ggml-vulkan.cpp:6656-6700).

| Field | llama | Ours | Mark |
| --- | --- | --- | --- |
| Band range | 2..8 in mul_mat_vec; 9 and up in mul_mm | MMV kernels accept 2..16 (k19_q4k_mmv.rs:120,127), shipped off; shipped default band behavior is GEMV per-row streaming | DIFFERENT |
| Accumulator residence | Registers, proven shape (our autopsy of the equivalent form: K19 compiles to VGPR 72-96, zero spills) | K19/K20/K21 copied this form exactly; autopsy matches (VGPR 60-96, zero spills, LDS 1,024 B) | MATCH |
| Activation form in band | Q8_1 integer (MMVQ) when the gate passes; f32/f16 otherwise | f32 activations, f32 dots, no quantize dispatch in front (the GEMV's exact math) | DIFFERENT, sign unmeasured: llama moves 1 byte per activation element in band, ours moves 4; our f32 form still won its cap-16 A/B by 13% (bench/mmv-verdict-2026-11-05.md) |
| Rows per workgroup in band | NUM_ROWS 2 (f32 path), 4 (MMVQ path, NUM_COLS >= 5) on RDNA3 | 2 (k19 shipped_shape, k19_q4k_mmv.rs:221-228) | MATCH for the f32 form; the MMVQ 4-row variant is ABSENT |
| Per-family thresholds | None inside the band; the Q6_K MMVQ exclusion is the only per-type rule | None shipped (one global env cap); the routing lane's per-family win map is the in-flight fix | DIFFERENT (ours is the known cliff: caps 4/8/12 lose, 16 wins) |
| Fallback when the band form refuses | None needed; one shader family covers 1..8 | Ladder: MMV, K18, chunked, GEMV (session.rs prepare_proj docs) | DIFFERENT in structure, no measured cost |

## Family 3: mul_mm / mul_mmq (prefill) vs K13/K14, wide K15/K16, K22, GEMM K10/K11

llama shaders: `mul_mm.comp` (f16/f32 activations, dequant to
shared), `mul_mmq.comp` (Q8_1 activations, integer dots),
`mul_mmq_cm1.comp` (coopmat integer, the RDNA3 path). Ours:
`k13_q4k_mmq.rs`, `k14_q6k_mmq.rs`, `k15_q4k_mmq_wide.rs`,
`k16_q6k_mmq_wide.rs`, `k22_q4k_mmq.rs`, with the f32 GEMMs
`k10_q4k_gemm.rs` / `k11_q6k_gemm.rs` as the fallback rung.

| Field | llama | Ours | Mark |
| --- | --- | --- | --- |
| Tile classes | Three (small, medium, large), picked per product by the (m, n) selector above | One shipped tile per kernel (K13/K14 m64; K22 64x64); the session chunk cap is 64 rows (session.rs:144,158), so the batch side never exceeds llama's medium class | DIFFERENT |
| Medium tile (the class a 64-row chunk runs) | BM 64, BN 64, BK 32, BLOCK_SIZE 128, WN 32, WMITER 2, TM 4, TN 2, TK 1, WM and WARP = mm_warp_8 = min(device subgroup size, 64) (ggml-vulkan.cpp:1678-1684, 1798-1803; warptile field order per mul_mmq.comp:74-85) | K13 m64: 256 threads, tile 64 out rows x the whole 64-row batch, each thread a 4x4 sub-tile (k13_q4k_mmq.rs:69-87) | DIFFERENT in organization: llama splits the batch across BN 64 within a 128-thread workgroup of 4 warps with a 32-wide warp tile; ours gives one 256-thread workgroup the full batch with 4x4 thread tiles |
| Large tile (n > 64, AMD with coopmat) | BM 128, BN 128, BK 32, BLOCK_SIZE 256, WN 64, WMITER 2 (1 for K-quants in the integer path, with TM 4, TN 2), WM and WARP = mm_warp_8 / mm_warp_16 as above (ggml-vulkan.cpp:1846-1850) | ABSENT (no tile above 64 rows of batch, no 128-row weight tile) | ABSENT |
| K step | BK 32, BK_STEP 4: 128 K elements staged per pass in mul_mmq and mul_mm (BK 16 for non-quant mul_mm) (mul_mmq.comp:87-97, mul_mm.comp:173-177) | K13 stages one super-block column (256 elements) per pass | DIFFERENT |
| Staging layout, weights | mul_mmq stages per-super-block structs: packed qs words and the (d, dmin) pair side by side per block (mul_mmq_shmem_types.glsl); mul_mmq_cm1 splits them into separate arrays: `buf_a_qs[BM * QPITCH]` with QPITCH = BK_STEP*(BK/4)+4 (the +4 is a bank-conflict skew), plus `buf_a_dm` vec2 pairs (mul_mmq_cm1.comp:103-122) | K13 stages weight words (tile_n x 64 words), activation words with a 65-word row stride (the same skew idea), and scale pairs, interleaved per super-block column (k13_q4k_mmq.rs:120-122); m64 staging 41,216 B (autopsy: LDS 41,984 B compiled) | MATCH in kind (skew present), DIFFERENT in split: cm1 separates qs from scales into their own arrays; K13/K22 do not |
| Staging repack, activations | The B side is REPACKED during staging: B arrives as `block_q8_1_x4_packed128` (four Q8_1 blocks interleaved in one 128-byte struct) and `block_b_to_shmem` de-interleaves it (ib_outer = ib/4, ib_inner = ib%4, qs scattered to `qs[iqs*4 + 0..3]`) so the dot loop reads unit-stride (mul_mmq_funcs.glsl, block_b_to_shmem) | K13 stages activation words in source order with the row skew only; no de-interleave repack. K22's note names this exact omission: the repack was deliberately not copied | ABSENT (named gap, see ranked list) |
| A-side load width into staging | LOAD_VEC_A per type at shader generation: 4 for Q4_K and Q5_K in mul_mm, 2 for Q6_K (mul_mm.comp:43-48 defaults, per-type defines; Q6_K branch in mul_mm_funcs.glsl uses 2 values per index). mul_mmq uses LOAD_VEC_A = 4*QUANT_R_MMQ, LOAD_VEC_B 16 (mul_mmq.comp:105-106) | K14 stages Q6_K from the 212-byte padded block layout (k14_q6k_mmq.rs:42); the width-2 fact is llama's own staging wart in the f16 path, recorded in bench/mmv-q6k-2026-11-05.md; it does not apply to the cm1 path llama actually runs on RDNA3 | Informational: no action (see gap list tail) |
| Register organization | mul_mmq: register cache `cache_a[WMITER * TM]` per thread plus one `cache_b`, dots through DP4A on staged data. K13's compiled form: VGPR 256 (the cap), zero spills (our autopsy) | K13/K14: VGPR 256, zero spills, same saturation. K22 copied mul_mmq's 128-thread organization and braced column scopes and still compiled to VGPR 256 (verdict, bench/k22-mmq-2026-11-05.md) | MATCH on the scalar path, including the failure mode: register organization alone does not move ACO off the cap in either codebase's scalar form |
| Split-K | Present, rule above, with a reduce shader | ABSENT in K13/K14/K22 (one workgroup owns its tile's whole K) | ABSENT |
| Wide variant | None: llama has no wider-than-large tile; width comes from the tile classes and split-K | K15/K16 wide MMQ exists and is convicted: 1,041-1,213 spilled VGPRs, 78-86 KB scratch (bench/shader-autopsy-2026-11-04.md); shipped off by the row threshold | DIFFERENT (ours is a dead end llama never built) |
| Measured standing | llama.cpp Vulkan prefill 998.43 tok/s (whole engine, 1,024-token class prompts) | Our MMQ pair: 83.2% of prefill GPU time at 43.5 GB/s combined (k13 4,199.9 ms at 46.4 GB/s, k14 1,342.0 ms at 34.6 GB/s); our prefill wall 161.5 tok/s after the host-gap fix (bench/prefill-profile-2026-11-05.md, bench/prefill-host-gap-2026-11-05.md) | The gap is this family plus Family 4 |

## Family 4: the coopmat path vs our stack

On RDNA3 with RADV, llama's device probe finds VK_KHR_cooperative_matrix
with subgroup-scope 16x16x16 shapes (f16 in, f32 or f16 accumulate;
s8 in, s32 accumulate). That flips three things on, per the
enablement chain in the host table: the large mul_mm tile class
(AMD rule, ggml-vulkan.cpp:4807), the cm1 integer MMQ shaders for
Q8_1 products (ggml-vulkan.cpp:2481-2494, tile spec in
mul_mmq_cm1.comp:89-100: BLOCK_SIZE 256, BM 128, BN 128, WM 64,
WN 32, TM/TN/TK = the 16x16x16 fragment, WARP 32; those are the
shader-declared defaults, and the host warptile overrides
BLOCK_SIZE and WM per size class, ggml-vulkan.cpp:1807-1813), and the
coopmat1 flash path for multi-row attention (Family 5).

| Field | llama | Ours | Mark |
| --- | --- | --- | --- |
| Cooperative matrix anywhere | Yes: mul_mm, mul_mmq_cm1, flash_attn_cm1 | No occurrence in any crate (grep over crates/: the only hits are the English word in comments and K22's doc mention) | ABSENT |
| Why absent | n/a | Our backend is wgpu/WGSL; WGSL has no cooperative-matrix type and Naga exposes none, so the path needs either a wgpu extension that does not exist in the pinned wgpu or a non-WGSL side channel | Structural, stated plainly: this is the one spec in this note that cannot be taken by re-expression inside the current shader language |
| Prefill consequence | llama's prefill products on this card run on WMMA tensor pipes at 16x16x16 per subgroup | Our prefill products run scalar DP4A at 43.5 GB/s combined (Family 3) | The single largest measured gap (see ranked list) |

## Family 5: flash_attn vs K6 flash

llama shaders: `flash_attn.comp` + `flash_attn_base.glsl`, plus
`flash_attn_cm1.comp` / `flash_attn_cm2.comp` and the two-phase
decode pair. Ours: `k6_attention.rs` FLASH_SHADER (the resident
form), with the scores+reduce pair as the trait-path form.

Spec constants (flash_attn_base.glsl:4-21): WorkGroupSize 128,
Br 1 (query rows per workgroup), Bc 32 (KV block), HSK/HSV
specialized per head dim, D_split 16, row_split 1, SubGroupSize
32, SHMEM_STAGING and LIMIT_OCCUPANCY_SHMEM flags.

Path choice (ggml-vulkan.cpp:1334-1375): coopmat2 if present,
else coopmat1 when `coopmat1_fa_support` (coopmat plus full
subgroup support) and the 16x16x16 shape and shared-memory checks
pass, else scalar. One query row always takes the scalar path
("scalar is faster than coopmat when N==1"). Scalar tuning on
AMD non-GCN (ggml-vulkan.cpp:1200-1245): subgroup size is the
device size for n_rows >= 4; row_split rises to 4 when
hsk > 128 (for n_rows > 8); block_rows 8 at row_split 1 for
n_rows > 8; block_cols 32 when (hsk|hsv) has no factor 8, else
64. Coopmat1 tuning: 16-row by 16-col fragments, block_cols =
16 * num_subgroups (ggml-vulkan.cpp:1275-1304).

| Field | llama | Ours | Mark |
| --- | --- | --- | --- |
| Workgroup shape | 128 threads default (spec constant), one workgroup per (Br query rows, head) | 256 threads, one workgroup per (1 query row, Q head) (k6_attention.rs:232-262) | MATCH in structure (Br 1 class), DIFFERENT in size |
| KV blocking | Bc 32 keys per block, K and V staged in shared memory (SHMEM_STAGING) for the matrix paths | 256 keys per tile, but only the SCORES are staged (a 256-f32 workgroup array); K and V are re-read from global memory per lane in the reduce loop (k6_attention.rs:264-325) | DIFFERENT: no K/V staging at all |
| Softmax | Online, inside the flash kernel, same as ours in kind | Online, inside the flash kernel | MATCH |
| KV precision | K/V read at cache precision (f16 cache standard; quantized cache types dequantized in-shader via flash_attn_dequant.glsl) | KV buffers are f32 end to end | DIFFERENT: llama moves half the KV bytes per key |
| Decode form | n_rows 1 takes the scalar path, plus dedicated two-phase decode shaders with split-K over the KV length (flash_attn_decode_phase_1/2.comp) | Same fused flash kernel at every shape; no KV split | DIFFERENT: no decode split-K |
| Long-context behavior | Blocked matrix form: cost grows linearly per block with shared-memory reuse | Each query lane re-reads V rows from global memory inside the tile loop; measured flash time grows quadratically across the prompt chunks (125.7 ms in the last chunk of the 1,022-token profile) and totals 1,013.1 ms, 15.2% of prefill GPU time | DIFFERENT with measured cost |

## Family 6: quantize_q8_1 vs K12

llama: `quantize_q8_1.comp`. Ours: `k12_q8_quant.rs`. Both exist
for one reason: feed the MMQ integer dots, once per projection
input, shared across the layer's projections in our session
(quantize-once, HShare) and per product input in llama.

| Field | llama | Ours | Mark |
| --- | --- | --- | --- |
| Block | 32 elements per Q8_1 block, scale d = amax/127, plus the block SUM stored beside d as f16 pair `ds = (d, sum * d)` for the DP4A correction (quantize_q8_1.comp:78-107) | 32 elements per block (Q8_BLOCK_ELEMS, k12_q8_quant.rs:64), d and sum carried in the ds stream the MMQ kernels consume | MATCH |
| Workgroup | GROUP_SIZE = subgroup size (spec constant 0, host-set, ggml-vulkan.cpp:3242); 8 threads per block, one vec4 per thread, subgroup clustered max/add reductions | 256 threads per workgroup, one workgroup covers 256 elements (eight blocks, QUANT_CHUNK_ELEMS, k12_q8_quant.rs:69) | DIFFERENT in shape, same math contract |
| Output packing | The shipped pipeline is the x4 variant: four blocks packed into `block_q8_1_x4` structs (ggml-vulkan.cpp:6328), the layout Family 3's staging repack later de-interleaves | Packed qs words and ds pairs in two flat streams (qs_words / ds_vec4s layout, k12_q8_quant.rs) | DIFFERENT: llama's x4 interleave exists to feed its staging repack; ours feeds K13's direct staging |
| Rounding | `round()` in shader (ties to even via GLSL round) | Documented divergence: our reference rounds ties away from zero, the shader builtin may differ on exact ties; bounded at one quant step, inside the MMQ parity band (k12_q8_quant.rs module docs) | MATCH within a disclosed tie rule |
| Measured standing | Part of the MMQ path cost | k12 is 0.5% of prefill GPU time (32.9 ms of 6,659 ms) | Not a gap |

## Family 7: get_rows, rms_norm, rope, softmax vs K8 gather, K4, K5, K9

| Field | llama | Ours | Mark |
| --- | --- | --- | --- |
| Embedding gather | `get_rows.comp`: 512 threads, one element per thread, grid-stride over rows and batches; a separate `get_rows_quant.comp` dequantizes during the gather for quantized embedding tables | K8's gather reads token ids from a device buffer (session.rs:22); the embedding table in the resident session is uploaded dequantized, so the gather is a plain copy | MATCH for our (dequantized) table; the quantized-gather variant is ABSENT and would matter only if the table stayed quantized on device |
| RMS norm | `rms_norm.comp`: one workgroup per row, BLOCK_SIZE a spec constant (default 512) with host variants 128, 256, 512, 1024 chosen so ncols <= 2*BLOCK_SIZE (ggml-vulkan.cpp:3264-3269); a partials two-pass variant for wide rows; FUSION variants run norm+rope, norm+add, and norm+set_rows as one shader, the norm output passing to rope through shared memory (rms_norm.comp:15-40) | K4: one workgroup per row, fixed 256 threads, lanes striding the row (k4_rmsnorm.rs:48); norm, residual add, and the chain forms are separate small kernels; the attention-side fusion llama does at this seam exists in ours one level up: K9 fuses the Q/K norms, RoPE, and KV append into one dispatch per layer (k9_attn_prep.rs module docs) | MATCH in geometry class; fusion placement DIFFERENT (theirs inside the norm shader, ours at the attention prep seam) |
| RoPE | `rope_norm.comp` over `rope_head.glsl`: local size (1, 256), each thread rotates one pair (i0 = 2*global y); also fused into rms_norm (above) and a neox variant | K5: 256-thread workgroups over pairs, plus the KV append in the same module (k5_rope_kv.rs:45) | MATCH |
| Softmax (standalone) | `soft_max.comp`: one workgroup per row, BLOCK_SIZE = subgroup size spec constant, plus a 512-wide variant and a three-pass large-row form (ggml-vulkan.cpp:3503-3505) | No standalone softmax kernel in the resident path; softmax lives inside K6 (online, in the flash and reduce forms) | ABSENT as a standalone, unused by the Glimmer path; not a gap for this engine |

## Ranked gap list

Ordered by expected payoff against our measured numbers. Each
gap names its owner: a lane in flight, or the follow-up that
should be created.

1. **Coopmat integer MMQ (cm1) is ABSENT from our stack; it is
   the path llama actually runs for prefill on this card.**
   Spec: VK_KHR_cooperative_matrix, subgroup scope, 16x16x16
   s8 x s8 to s32 fragments set coopmat_int dims
   (ggml-vulkan.cpp:4740-4748); on RDNA3/4 with that support,
   Q4_K/Q5_K/Q6_K x Q8_1 products are created as cm1 pipelines
   (ggml-vulkan.cpp:1877, 2481-2494) with BM 128, BN 128,
   BK 32 tiles and the 16x16x16 fragment as TM/TN/TK
   (mul_mmq_cm1.comp:89-100). Payoff: our MMQ pair is 83.2% of
   prefill GPU time at 43.5 GB/s; the profile note prices the
   streaming-bound ceiling of any MMQ rework at about 347 tok/s
   wall against today's 161.5, and coopmat is the only variant
   that changes the math pipe, not just the schedule. Blocker
   is structural (WGSL has no coopmat), so the owner is the
   coopmat lane in flight, whose feasibility verdict decides
   whether this is a build or a permanent documented gap.
2. **The MMQ staging repack and split staging layout are ABSENT
   in K13/K14 and were deliberately not copied by K22.**
   Spec: B de-interleave in `block_b_to_shmem`
   (mul_mmq_funcs.glsl) out of the x4-packed Q8_1 layout, and
   cm1's split arrays (qs in `buf_a_qs` at QPITCH with the +4
   skew, scales in `buf_a_dm`) (mul_mmq_cm1.comp:103-122).
   Payoff: this is the remaining untested structural difference
   between K13 and llama's scalar MMQ after K22 proved register
   organization alone compiles to the same 256 VGPR cap. Owner:
   a K23 follow-up lane (K13-form kernel with the repack and
   split staging, nothing else changed), gated by autopsy first
   (spills and VGPR) before any timing run.
3. **Band routing thresholds DIFFER: llama ends the vec band at
   8 and starts the matrix path at 9, with RDNA3-specific
   NUM_ROWS tuning inside the band (4 rows at NUM_COLS >= 5 in
   the integer path); ours ships a single matrix threshold at
   16 and per-row GEMV streaming below it.**
   Spec: ggml-vulkan-types.h:362, ggml-vulkan.cpp:7402,
   2880-2885. Payoff: our dose sweep shows the cost of the
   wrong band shape directly (caps 4/8/12 below control, 16
   above; bench/mmv-verdict-2026-11-05.md), and decode-side
   verify batches live exactly in this band. Owner: the
   band-routing lane in flight; llama's 8-cap and the RDNA3
   4-row rule are the cross-check values its win map must
   reconcile with our per-family sweep data (our K20/K21 win
   rows llama's MMVQ gate would exclude for Q6_K, so the map
   is a cross-check, not a copy).
4. **Split-K is ABSENT in our MMQ.**
   Spec: k >= 2048 and tile grid at most half the shader cores
   gives up to 8 splits, 256-aligned, with
   `mul_mat_split_k_reduce.comp` (ggml-vulkan.cpp:5916-5955).
   Payoff: our profile's worst per-byte dispatches are the
   small ones split-K exists for: the 256-wide Q6_K KV-side
   projections run 416 dispatches at 3.2 GB/s for 186 ms
   (2.8% of prefill GPU time), and at batch 64 with M 256 the
   tile grid is 4 workgroups on a 96-CU card. Owner: the K23
   MMQ follow-up (same lane as gap 2, second step), or the
   dispatch-fusion follow-up the profile note already names;
   split-K and fusion attack the same 186 ms from opposite
   sides and only one should be built first, decided by the
   K23 autopsy results.
5. **Flash attention: no K/V shared-memory staging, f32 KV,
   and no coopmat1 path in K6.**
   Spec: Bc 32 blocks with staged K/V (flash_attn_base.glsl:
   4-21), path selection and AMD scalar tuning
   (ggml-vulkan.cpp:1200-1245, 1334-1375), f16 cache
   precision. Payoff: flash is 1,013.1 ms, 15.2% of prefill
   GPU time at 1,022 tokens and grows quadratically in chunk
   position; the profile note prices halving it at 507 ms now
   and calls it the likely #1 lever at 4k context. Owner: the
   flash rework follow-up (profile attack 4); the f32 to f16
   KV change is a resident-KV format change and must be its
   own gated step, not folded into a kernel rewrite.
6. **MMVQ (Q8_1 activations in the vec band) is ABSENT; our
   band kernels run f32 activations.**
   Spec: ggml-vulkan.cpp:6656-6700 and 6783 (batch > 1 always
   quantizes on AMD when k >= 2048; Q6_K excluded off Intel).
   Payoff: unmeasured on our side, and the sign is genuinely
   open: llama accepts a quantize dispatch and integer dots
   to cut activation traffic 4x in the band, while our f32
   K19 already beat its control by 13% at cap 16. Owner: a
   small prototype inside the band-routing lane's follow-up,
   judged by the same sweep harness; do not build before the
   routing map lands, because the two changes confound.
7. **RMS norm + RoPE fusion placement DIFFERS (theirs inside
   the norm shader, ours at K9 attention prep), and the
   standalone norm chain has no partials path for wide rows.**
   Spec: rms_norm.comp:15-40 fusion variants, block sizes
   128-1024 host-chosen (ggml-vulkan.cpp:3264-3271). Payoff:
   the whole norm chain is 0.7% of prefill GPU time
   (43.5 ms); at decode it is a handful of small dispatches
   per token against a 28 ms token budget. No lane; record
   only, revisit if decode chain overhead becomes the named
   blocker in the spec-round cost map.
8. **llama-side wart, no action: Q6_K gets staging load
   width 2 in mul_mm while Q4_K/Q5_K get 4**
   (mul_mm.comp:43-48, mul_mm_funcs.glsl Q6_K branch). It
   applies to the f16-activation staging path, not to
   mul_mat_vec and not to the cm1 path RDNA3 runs. Recorded
   so nobody imports it as a spec to match; our K14/K20
   already document the same finding
   (bench/mmv-q6k-2026-11-05.md).

## What this settles

The specs are all takeable, and most of them are already taken:
the GEMV family matches llama's shape and beats its measured
streaming rate; the MMV band matches in form; the chain kernels
match in geometry. The parity gap concentrates in exactly two
places: the prefill product (coopmat cm1, the staging repack,
split-K, tile classes above 64) and the band thresholds. Both
already have lanes or named follow-ups above; nothing in this
note requires a new research question, only builds.
