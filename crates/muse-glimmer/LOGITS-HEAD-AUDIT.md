# Logits-head audit: the tail of the forward path

Lane 05 audit, 2026-10-29. Question under audit: integration run B
(`crates/itak-loader/INTEGRATION-RUN-B.md`) produced a nearly flat
output distribution (top-5 mass 1.6e-4 against the reference's 0.62,
wrong argmax at position 1). Is the defect in the tail of the
forward — the final norm, the logits head, the softcap/scale
finishing, or the greedy pick? Every claim below is backed by a
test named beside it, run on this branch.

## Verdict

The tail computes what this repo's own record of the architecture
prescribes, and the audit proves it with fixtures designed so the
old hiding places (weight-1 norms, a negligible epsilon, logits too
small to saturate the softcap) cannot mask a cross-wired tensor or
a wrong constant. The wrong argmax is exonerated from the tail
completely: the finishing map is strictly monotone, so the greedy
pick over the finished logits is the pick over the raw head output,
and run B's wrong pick (290 over 13796) was already wrong in the
raw logits. The softcap is exonerated as the flatness cause (it
compresses the observed top-5 spread by only 5 to 10 percent). One
tail scalar can flatten a distribution, the 0.19611613 logit scale,
and whether the reference applies it to the logits the way we do is
a genuine open question this audit cannot settle from in-repo
evidence; it is stated precisely below, with the experiment that
settles it. Nothing was changed on a guess.

## Findings, per audit question

### 1. Final norm: present, correct tensor, correct epsilon

`GlimmerModel::forward` (crates/muse-glimmer/src/model.rs) takes
the last position's row and applies `rms_norm` with
`weights.final_norm`, 1 row, `cfg.rms_eps`. `Weights::load` binds
`final_norm` to `output_norm.weight` — the tensor the loader
inventory confirmed in the real file (F32, [6656]) — and never
substitutes another tensor for it.

Proof:
- `weights::tests::global_tensors_bind_by_name_and_content` —
  every expected name serves distinct bytes; the embedding,
  final-norm, and output fields must carry their own names' bytes.
  A cross-wire in `Weights::load` fails here.
- `model::tests::tail_uses_output_norm_and_output_head_with_config_scalars`
  — a recording compute probe under a config with eps overridden
  to 0.05: every norm in the pass runs at eps 0.05, and the last
  two steps of the pass are exactly `rms_norm
  name=output_norm.weight rows=1 eps=0.05` then `lm_head
  name=output.weight`.
- `cpu_compute` test `tail_discriminates_tensors_eps_and_softcap_order`
  — the real arithmetic chain at eps 0.5 with three different norm
  weight vectors; see section 4.

The norm formula itself is `torch_math::rms_norm`: `x / sqrt(mean
(x^2) + eps) * w`, f64 accumulation, eps a parameter (its own
tests pin the formula and the eps sensitivity).

### 2. Logits: untied head, correct tensor, correct orientation

The head is `output.weight`, a tensor of its own in the file
(Q5_K in the real model). A missing `output.weight` fails the load
by name (`weights::tests::missing_output_head_is_not_replaced_by_embedding`);
the embedding table is never substituted. `CpuCompute::lm_head`
validates the stored shape [hidden, vocab] against the config
before computing, and the matvec row convention (stored row o is
output o's coefficients) is the same one the projection tests pin
against hand products. The end-to-end test in section 4 uses a
head deliberately different from the embedding table, with
rectangular shape (32 by 128), so both a tied-head substitution
and a transposed read change the numbers and fail.

### 3. Softcap: required, present, correct formula and order

The finishing is `softcap_scale` in model.rs: `scale * cap *
tanh(z / cap)` when the cap is positive, else `scale * z`, with
cap and scale from the file's own metadata (`final_logit_softcapping`
20.0, `logit_scale` 0.19611613, both required keys in
`GlimmerConfig`). Scale after cap, matching the composition the
torch-math logits module documents. The probe test above pins the
order with scripted logits at z = 30 under cap 3.0, scale 0.25:
the forward returns 0.75 * tanh(10); the opposite order would give
2.96, and the test asserts the output is more than 1.0 away from
that number. Section 4's test repeats the discrimination through
the real arithmetic at saturation.

### 4. End-to-end: hand-known logits through the real chain

`cpu-compute/tests/forward_tests.rs::tail_discriminates_tensors_eps_and_softcap_order`:
a one-layer synthetic model, attention and FFN outputs zeroed, so
the expected logits are a closed form computed in the test's own
f64 math. Distinct norm weights (the output norm's are mixed-sign),
eps 0.5, cap 2.0, scale 0.25, and one head row aimed at the
strongest normed channel with weight 30 / n, driving its raw logit
to exactly 30, deep in saturation. All 128 finished logits match
the hand computation to 1e-4, the saturating logit lands on
`scale * cap * tanh(15)` (within 1e-5 of 0.5, and more than 1.0 from
the 2.0 the opposite composition order predicts), and the greedy
argmax over the finished logits is the aimed token, 37.

Mutation proof that these tests have teeth (run during the audit,
then reverted): swapping the softcap/scale composition order in
`softcap_scale` failed 3 tests (the two new tail tests and the
analytic forward test); pointing the final norm at
`blk.0.attn_norm` instead of `output_norm` failed the end-to-end
tail test.

### 5. Temperature-0 argmax: correct; tie-break was unspecified, now fixed

The greedy pick is rank 0 of the runner's `top_k`
(crates/itak-loader/examples/glimmer_generate.rs), a descending
sort of the finished logits. For distinct logits that is exactly
argmax. On an exact tie the old comparator left the pick to
`sort_unstable_by`'s unspecified order; run B had no ties, so this
caused nothing there, but a greedy decoder must be reproducible.
The comparator now breaks ties by ascending token id, so the pick
is the first maximum, the same choice a first-max scan makes.
Pinned by `top_k_tie_breaks_to_the_lowest_id` and
`top_k_greedy_pick_is_the_argmax` in the example's new test module;
`top_k_orders_by_logit_and_computes_softmax` pins the f64 softmax
probabilities against hand values. (A NaN logit would still compare
as "equal" in the sort; logits are finite in every test here, and
producing a NaN is an upstream failure, not a tail behavior.)

## The run B numbers, inverted

Run B printed only finished logits. Inverting our own finishing
(`z = cap * atanh(finished / (scale * cap))`, exact):

- Our raw top-5 at position 1: 6.455, 4.362, 4.130, 3.984, 3.937.
  Raw spread rank 0 to rank 4: 2.518 nats.
- The reference's finished top-5 spread, from its probabilities:
  ln(0.3962 / 0.0323) = 2.507 nats.
- The softcap's share of our raw-to-finished spread shrink
  (2.518 to 0.461) is small: sech^2(z / cap) at the raw top-5 runs
  0.90 to 0.96. Essentially all of the shrink is the 0.196 scale.
- Inverting the bulk: our finished top prob 4.4e-5 puts the
  vocabulary bulk's finished center near -0.96, i.e. raw near
  -5.0. If the raw logits were softmaxed directly with a bulk that
  tight, the top prob would be about 0.28 (a real bulk spread
  lowers this; it stays orders of magnitude above 4.4e-5).

Two readings remain, and the in-repo record cannot choose:

(a) The reference applies the same scale the same way. Then its
raw top-5 spread must be about 13 nats against our 2.5, the head
receives a far better separated hidden state from the reference's
layer stack, and run B's working diagnosis (the defect is
upstream of the head) stands untouched.

(b) The reference applies `logit_scale` somewhere else, or not to
the logits. Then our raw separation (2.518) already matches its
finished separation (2.507) to half a percent, and our finishing
alone manufactures the concentration gap. The near-equality is
suggestive and nothing more: it could be coincidence, since it
compares spreads, not distributions.

The experiment that decides, in order of cost: (1) one run of our
forward on the real file with the scale forced to 1.0 by a harness
override (no change to the model path), compared against run B's
reference table — under (b) the top prob lands near 0.396 and the
top-5 mass near 0.62, under (a) it stays flat; (2) the llama.cpp
graph-source read for this architecture that run B's diagnosis
already calls for, which also settles the norm-convention item
below. Until one of those runs, the scale stays exactly where the
file and this repo's config documentation put it; removing it to
make the numbers prettier would be tuning to force a match.

## Still open, deliberately not decided here

- **Norm convention (run B diagnosis item 1).** Our stack applies
  plain `x * w` RMSNorm everywhere, including the final norm. The
  file's `output_norm.weight` is centered on zero (mean +0.017),
  which is the profile a Gemma-style `(1 + w)` store would have,
  while the block norms center above 1. That question is
  stack-wide (106 norms), not tail-local: the convention, wherever
  it lands, must land in the shared `torch_math::rms_norm` call
  path or as a deliberate per-site variant, decided by the
  per-layer activation diff and the graph source, not by this
  lane. The tests above pin the current documented convention, so
  a future change shows up as deliberate test changes.
- **Logit scale application (above).** Same evidence gate.

## Minor notes

- `torch_math::logit_softcap` / `logit_scale` are exported but no
  crate calls them; the live finishing path is model.rs's
  `softcap_scale`. The two compose to identical math for valid
  inputs. One real difference: the torch-math versions validate
  their scalars and fail loud on NaN; `softcap_scale` does not,
  and a NaN stored in the file's metadata would flow through
  `required_f32` into the logits. Recommended follow-up (not done
  in this lane, which changes no model behavior on a guess):
  reject non-finite cap/scale in `GlimmerConfig::from_metadata`.
- Diagnostics: the new tests sit beside the existing `dbg_log`
  instrumentation; no new logging surface was needed.

## Gates on this branch

- `cargo test --workspace`: 174 passed, 0 failed. New tests:
  muse-glimmer 23 to 25 (binding, tail probe), cpu-compute
  integration 8 to 9 (end-to-end tail), itak-loader example
  glimmer_generate 0 to 3 (top_k).
- `cargo clippy -p muse-glimmer -p cpu-compute -p itak-loader
  --all-targets`: clean under the workspace deny-warnings lints.
  (Run with RUSTUP_TOOLCHAIN=1.99.1-x86_64-unknown-linux-gnu: this
  sandbox's default resolution mixes the stable toolchain's
  clippy-driver with 1.99.1's cargo and the clippy probe fails
  before reading any source, on untouched crates too.)
