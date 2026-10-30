# Norm audit, 2026-10-29: every normalization in the Glimmer forward

Lane 02-norm, branch `lane-02-norm`. Triggered by integration run
B (`crates/itak-loader/INTEGRATION-RUN-B.md`), whose diagnosis item
1 named the norm convention the prime suspect for the first-token
divergence from llama.cpp: this forward applied plain `x * w`
RMSNorm everywhere, with both post norms placed after their
residual adds, and the run's own note said the convention had to be
settled against the reference before any further guessing.

Method: the checkpoint definition (Hugging Face `MuseGlimmer`
config docs, which document the architecture's norm parameters by
name), two independent reimplementations that mirror the canonical
implementation (NVIDIA NeMo AutoModel's module map, the vLLM model
docs), the MLX port summary that cites llama.cpp's
`src/models/muse-glimmer.cpp` as its reference, one derivative
model card describing the checkpoint's block structure, and the
measured contents of the real GGUF from run B itself. The llama.cpp
graph source could not be fetched directly during this audit (the
page-fetch tooling was down); where a verdict rests on inference
rather than a quoted source, the inference is stated as such.

## Findings

### 1. Post-norm placement: DEVIATION, fixed

The architecture's post norms are sandwich norms: each one
normalizes its sub-layer's output BEFORE the residual add. The
checkpoint config describes `post_norm_eps` as the epsilon for "the
post-attention and post-FFN norms (which sit between the sub-layer
output and the residual)", and a derivative model card states it
flatly: "every attention and MLP block output passes through a
post-norm (Gemma-style sandwich norms) before the residual add."

Ours applied both post norms AFTER the adds. Fixed in
`model.rs`: `x = residual + post_attention_norm(attn_out)` and
`x = residual + post_ffw_norm(ffn_out)`. The structure test in
`model.rs` and the two analytic tests in
`cpu-compute/tests/forward_tests.rs` pin the new order; both
analytic tests were re-run against the pre-fix forward and fail
there, pass here.

### 2. Embedding norm: DEVIATION (missing), fixed

The architecture RMS-normalizes the token embeddings,
parameter-free: NeMo's `MuseGlimmerScalelessRMSNorm` is documented
as "used for embedding and Q/K normalization", and the MLX port
summary lists "RMS-normed embeddings (not ×√hidden)" among the
text model's properties. Ours gathered embedding rows with no
normalization. Fixed: `torch_math::rms_norm_scaleless`, a
`Compute::rms_norm_scaleless` step, and a call in `model.rs`
immediately after `embed`, at the file's `rms_eps`.

### 3. Weighted norm form, (1 + w) vs w: CORRECT as-is (plain)

Verdict: the stored GGUF weights are effective gains and the plain
form this engine already used is the right one. Evidence:

- llama.cpp's standing convention is that graphs multiply the
  stored norm weight plainly; checkpoint-side shifts (Gemma's
  `(1 + w)`) are baked into the stored tensor by the converter.
- The file itself proves its converter materializes effective
  multipliers: run B measured `blk.0.attn_q_norm` as the constant
  3.87 in every entry and `attn_k_norm` as constant 1.0. The
  checkpoint definition has no such tensors (the Q/K norms are
  scaleless, with `qk_scale_factor` 3.87 applied to Q afterward);
  the stored constants are exactly the plain-multiply values that
  reproduce the definition. A graph-side `(1 + w)` applied to them
  would compute 4.87 and 2.0, and llama.cpp demonstrably runs this
  file correctly.
- The measured per-layer stats fit the same reading: stored means
  +1.34 / +1.17 / +0.67 are plausible effective gains; read as raw
  offsets they would imply trained multipliers averaging 2.34 /
  2.17 / 1.67 at every one of the 208 block norms.

Pinned executably in `torch-math`: `rms_norm` reference vectors
with a mixed-sign weight (the two readings disagree in sign, not
just size), and the identity `rms_norm(x, [3.87; n]) == 3.87 *
rms_norm_scaleless(x)` that the QK tensors rely on.

Residual uncertainty, stated plainly: this verdict is an inference
from the converter convention plus the in-file constants, not a
quotation of llama.cpp's muse-glimmer graph. The per-layer
activation diff run B prescribes settles it definitively; if that
diff ever shows the block norms are raw offsets, the change is one
call-site per norm (a shifted form in `torch-math`), and the
reference vectors for it are the same ones this audit computed.

### 4. Final output_norm: CORRECT as-is (present, placed, plain)

The final norm is applied to the last position's hidden state
before the head, from the file's own `output_norm.weight`. Its
form is plain: NeMo gives the final norm its own class,
`MuseGlimmerFinalRMSNorm`, "whose checkpoint weight stores the
actual output gain", in explicit contrast to the block norms'
offset storage; under the file convention of finding 3 the stored
tensor (measured mean +0.017, std 3.13) is that gain, applied
as-is. A dedicated forward test installs a mixed-sign final norm
weight and pins the plain reading end to end; it was verified to
fail under a `(1 + w)` final norm and pass under the plain one.

### 5. Epsilon source and value: CORRECT, with one documented nuance

Every norm takes `eps` from the file's
`muse-glimmer.attention.layer_norm_rms_epsilon` (1e-5), parsed in
`config.rs` with no silent default. That is the only epsilon the
GGUF carries (the measured metadata census has no second key), so
it is the only value a GGUF-driven reference can use.

Nuance recorded, not acted on: the checkpoint config also defines
`post_norm_eps` (default 1e-8) for the two post norms. A
GGUF-side engine cannot honor a value the file does not carry,
and the numerical reach of the difference is bounded by roughly
`eps / (2 * mean_square)` relative on post-norm outputs,
negligible at run B's scale of divergence. If a future file
carries a post-norm epsilon key, `config.rs` should learn to read
it; inventing the key name now would violate this repo's
file-first rule.

### 6. Q/K norms: CORRECT as-is

Per-head, after the projections, before RoPE, with the file's
stored tensors; see finding 3 for why the stored constants
reproduce the checkpoint's scaleless-plus-factor definition under
the plain form. Any residual error here would be a pure attention
temperature shift (run B item 4); none was found.

### 7. Flagged, out of this lane's scope: softcap/scale order

Not a norm, recorded here because the audit walked past it: the
checkpoint defines `output_multiplier` as a scale "applied to
logits BEFORE the final tanh softcap", while `softcap_scale` in
`model.rs` applies the scale after the cap
(`scale * cap * tanh(z / cap)`). Both maps are monotone, so this
cannot be the argmax cause (run B item 6), but the distributions
differ. Owner: the forward/sampling lane.

## What changed

- `crates/torch-math`: `rms_norm_scaleless` added; `rms_norm`
  refactored onto a shared reciprocal-RMS core (same errors, same
  values); convention tests added (6 new tests).
- `crates/muse-glimmer`: `Compute::rms_norm_scaleless` added
  (trait, `UnimplementedCompute`, and the model tests' recording
  stub); forward restructured (embedding norm, sandwich post
  norms); module and weights docs corrected; structure test
  updated.
- `crates/cpu-compute`: `rms_norm_scaleless` implemented with a
  unit test; the two analytic forward tests rewritten to the
  settled conventions (with channel-varying post norm weights so
  the placement is load-bearing, after two weaker fixtures were
  caught passing under the old code); one new forward test pinning
  the final norm's plain-gain form.

## Proof

- `cargo test` for the touched crates, and the full workspace:
  all green (counts in the lane report).
- `cargo clippy` for the touched crates under the workspace's
  deny-warnings lints: clean.
- Pin-bite checks, run during the audit: the rewritten analytic
  forward tests FAIL against the pre-fix forward (post norms after
  the adds, no embedding norm) and PASS against the fixed one; the
  final-norm test FAILS under a temporarily shifted `(1 + w)`
  final norm and PASSes under the plain one. Both hacks were
  reverted after the runs.
