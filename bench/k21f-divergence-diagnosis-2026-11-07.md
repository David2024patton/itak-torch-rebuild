# K21F divergence diagnosis: card measurements at the split (2026-11-07)

One diagnostic card session on ubuntu-server against main
7940bcf70 (the tree that failed identity in the
defaults-confirmation session), staged fresh at
~/gpu-wave17 (git archive; session.rs sha256
d5132a0886855a4b96b5997b91cbd37b82172d535c51069188e0a28d8f1a6628
and compute.rs 85fb1142109d2c6747e3963f9df4d988ff91af49db03191a3d1fb3c22c3979e6
verified against the commit; model sha256
4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e
re-verified). Goal: pin the K21F speculative identity
divergence at stream index 23 (plain 589, spec 1573;
prompt "The capital of France is", spec_generate_gpu
both mode, 64 tokens) with numbers. Raw logs:
~/gpu-wave17/logs17/ on the server. Bundles:
~/gpu-wave17/bench/failures/ (20261107T060222Z-leg1-
spec-default-d1, 20261107T060302Z-leg1-spec-default-d6,
20261107T060348Z-leg1-spec-control-d1, plus the failed
first shape-probe build 20261107T060946Z, fixed and
rerun). Tenants stopped before card work (VRAM
33,271,808 B) and restored after (8,932,147,200 B);
production llama-server never touched; no reboot.

## Read the reconciliation section before citing attribution

This note's measurements stand. Its original attribution
(the finish, not the product core) does NOT survive the
parent reconciliation at the end of this file: the
kernel probe never measured K21F's own raw product, and
the static math audit proved a product-core defect in
K21F's shader text. The numbers below are the pre-fix
baseline the wave18 retest compares against.

## Reproduction (this staging, this session)

- Default, depth 1: rounds=37 proposed=36 accepted=26,
  IDENTITY FAIL, GOLDEN PASS. Streams agree through
  index 22; index 23 plain 589, spec 1573.
- Default, depth 6: rounds=26 proposed=147 accepted=37,
  IDENTITY FAIL, GOLDEN PASS.
- Control (WGPU_COMPUTE_MMV_MAX_ROWS=1), depth 1:
  rounds=38, IDENTITY PASS, GOLDEN PASS.

Identical to the defaults-confirmation session on its
own staging, as expected.

## Method 1: diag capture cannot yield top-2

WGPU_COMPUTE_DIAG=capture runs at depth 1 (default and
control) and depth 6 (default). What survives to disk
in spec_generate_gpu is one `block` record per round:
the target session's verify block forwards,
fingerprinted over every row's finished logits
(values=404096 = 2 x 202,048 at depth 1; 1,414,336 =
7 x 202,048 at depth 6; 37 records at depth 1, 26 at
depth 6, matching the round counts). diag_diff on the
depth-1 pair: manifests differ only in
resolved_config.mmv_max_rows (16 vs 1, the intended
single variable); the first capture divergence is at
seq 0, field hash: the very first verify block's logits
already differ between the arms, consistent with a head
deviation present on every call whose early argmaxes
simply had margins too large to flip. But a block
record carries only the FNV-1a hash: no argmax, no
logits, and the argmax record kind folds its logit into
the hash rather than storing it. Capture can prove the
verify forwards differ from the first round; it cannot
produce a top-2, a margin, or a per-element delta.

## Method 2: in-situ head comparison at the divergence

A throwaway probe (k21f_insitu_probe.rs, server staging
~/gpu-wave17/crates/itak-spec/examples/, never
committed) replays the plain greedy stream through a
ResidentSession with batch-1 forwards, saves the
batch-1 distribution that emits index 23, then rewinds
and runs the two block geometries that could have
produced that distribution inside a round through
forward_block_capturing: geometry A [g22, 1573] (index
23 as a round's first row) and geometry B [g21, g22]
(index 23 as the bonus row). Three arms: shipped
defaults, gate closed, and a head-only isolation
(ceiling 16 with all nine band cells set to "1", so
every projection falls back to GEMV while the head
still routes to K21F; the finish stamp fired).

Under K3 (batch-1 reference, both probe processes agree
to 6e-6): top-1 is 589 at 15.359117, top-2 is 1573 at
14.976315. **The top-2 margin at the divergence is
+0.382801.**

- Control arm: both block rows are BIT-IDENTICAL to the
  batch-1 distribution (max-abs delta 0.000000 across
  all 202,048 logits, both geometries). The all-GEMV
  block forward is exactly the batch-1 math;
  block-vs-single drift is zero in this stack.
- Default arm: both block rows flip. 1573 at 15.268591
  over 589 at 15.186872, margin -0.081719. **Delta at
  589: -0.172245. Delta at 1573: +0.292275.** Combined
  swing 0.464520 against a margin of 0.382801, which is
  the flip. Max-abs delta across the vocab: **2.040238
  at id 157652**; RMS delta 4.397e-1.
- Head-only arm: the same row to ~3e-6 (1573 at
  15.268585, 589 at 15.186869; delta[589] -0.172242,
  delta[1573] +0.292279; max-abs 2.040238 at the same id
  157652). Band-routed projection drift inside the block
  forward contributes at most ~1e-5 to the head row.
  The K21F head call alone, on identical inputs,
  produces the entire in-situ delta.

## Method 3: product vs finish split on the real weights

A second throwaway probe (k21f_kernel_probe.rs, server
staging ~/gpu-wave17/crates/wgpu-compute/examples/)
loads the real head tensor (output.weight, Q5_K,
6656 -> 202,048) and runs, on identical synthetic
inputs at batches 2 and 7 (the two verify widths),
three input distributions (uniform, unit-RMS Gaussian,
Gaussian with outlier dimensions): K3 plain product,
K21 plain product, K3 fused finish, K21F fused finish
(softcap mode, scale 0.19611613, cap 20.0), and an f64
host reference. 27 rows measured. In every row:

- max|prodK21 - prodK3| = 0.000000.
- max|prodK21 - f64 ref| and max|prodK3 - f64 ref| are
  both at most 1.0e-5.
- K3 finish fidelity, max|finK3 - finish64(prodK3)|:
  1.07e-6 to 1.91e-6.
- K21F finish fidelity, max|finK21F - finish64(prodK21)|:
  0.31 to 1.87, and equal to max|finK21F - finK3| row
  for row (batch 2 gauss row 1: 1.741 against 1.741250;
  batch 7 gauss row 5: 1.868 against 1.868393).

Discriminators (two further probes, same staging):

- Shape and mode (k21f_shape_probe, batch 2 gauss): the
  fidelity error is the same in the workgroup-tree and
  subgroup reduction shapes (agreeing to 1e-6) and in
  FINISH_SCALE (row 1: 1.745408) as in FINISH_SOFTCAP
  (1.741250).
- Ratio (k21f_ratio_probe, scale mode): the implied
  ratio got/(scale*prod21) over elements with
  |want| > 0.5 spans min -1.78 to max +4.02; at the five
  largest elements of a row it wanders 0.897 to 1.216.
  The worst elements have near-zero true values (prod
  0.178 becomes -1.710; prod -1.185 becomes +0.936).

The lane read these results as: the deviation is
manufactured after the product, additive and
element-varying, on the order of one super-block
segment's partial sum. See the reconciliation for why
that reading is consistent with, and superseded by, the
product-core defect the audit proved.

## Why the existing gates missed it (lane's reading)

tests/k21_finish.rs gates K21F against the CPU
composition and against K3 at inn=512 and inn=768 with
synthetic Q5_K headers, tolerance about 1e-3.
tests/k21f_head_routing.rs holds block logits to the
cross-backend bar (max-abs under 0.5) at fixture scale;
the in-situ max delta at real geometry is 2.04. (The
audit's reading differs on why the finish tests missed
it: they never executed on a GPU at all before this
wave. Both readings agree on the missing pin: a
real-geometry, on-card gate. The mask-fix lane added
VM-runnable shader-text pins for the proven defect; the
real-geometry gate remains the retest's job.)

## Parent reconciliation (added at filing, 2026-11-07)

Two investigations ran in parallel and appeared to
disagree. They do not.

1. The static math audit (read-only, against the same
   tree) found the defect by reading K21F's own shader
   text: in crates/wgpu-compute/src/kernels/
   k21_q5k_mmv_finish.rs the qc dequant line masked its
   word with 0x0f0f0f (24-bit) instead of 0x0f0f0f0f,
   zeroing the fourth byte, so 16 of 256 elements per
   Q5_K super-block lose their entire low nibble. Both
   sibling kernels (plain K21, K3) carry the correct
   mask. That is a product-core defect, proven in
   source, and it is fixed on main (6ee2d68).
2. The kernel probe above never ran K21F's raw product:
   K21F exposes only the fused call, so its "finish
   fidelity" compared K21F's finished output against
   the finish of PLAIN K21's product. A product error
   inside K21F therefore presents, in that comparison,
   exactly as a finish error. The probe's other
   findings fit the mask defect one for one: the
   corruption is additive and segment-sized (dropped
   nibble terms over one sub-block segment), appears
   identically in the linear scale mode (it is upstream
   of the finish map), in both reduction shapes (it is
   upstream of the reduction too), with a non-constant
   ratio (an additive error is not a scale error), and
   hits hardest where the true value is near zero
   (dropped terms dominate small sums).
3. The lane's inference that the defect "engages only
   at 26 super-blocks per row" is not established and
   is not needed: the mask defect fires at every
   geometry, and the fixture gates that might have
   caught it at 512/768 wide never executed on a GPU
   before this wave (they skip on adapterless
   machines). What the real-geometry runs add is the
   size of the effect at the geometry that matters:
   margin +0.382801 at the split, deltas -0.172245 /
   +0.292275 at the two ids, max-abs 2.040238.
4. Arbitration: the wave18 retest reruns this note's
   kernel and in-situ probes against the fixed tree.
   If fidelity returns to the 1e-6 class and the
   in-situ delta collapses, the single-defect account
   is confirmed end to end. If a residual remains, this
   note's numbers bound it and the hunt reopens with
   them.
