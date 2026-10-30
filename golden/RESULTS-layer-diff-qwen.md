# Layer diff: Qwen2.5-1.5B forward divergence localized and fixed

Date: 2026-10-29. Follow-up to integration run A
(`RESULTS-integration-a.md`), which proved Qwen2.5-1.5B tokenization
exact (15/15 prompt ids) but found the arch-llama forward diverging
from llama.cpp: a salient in-context token (the operator 488 " +")
was over-boosted and predicted as its own continuation, with logit
swings of 0.3 to over 5 from the third context token onward.

## Method

Both engines dumped per-layer activations for one prefill of the
arithmetic_add prefix ids [16, 22, 488, 220] ("17 + "), the shortest
context where run A's probe showed disagreement.

- llama.cpp side: `golden/tools/cb_dump.cpp`, a raw llama.h harness
  whose `cb_eval` callback copies the named graph tensors
  (`attn_norm`, `Qcur`, `Kcur`, `Vcur`, `ffn_inp`, `ffn_norm`,
  `ffn_out`, `l_out`, `result_norm`, `result_output`) to .bin files.
  CPU only (n_gpu_layers = 0); the production Glimmer server was
  not touched. Logits were enabled on all four positions so the
  last layer's intermediates stay unsliced.
- arch-llama side: `LlamaModel::forward_traced` (same code path as
  `forward`, sub-step buffers cloned out as they pass) plus the
  `crates/arch-llama/examples/layer_dump.rs` runner.
- Diff: `golden/tools/compare_layers.py` (max-abs and cosine per
  checkpoint, per-position breakdown for `l_out`, and a RoPE
  pairing experiment described below).

One harness trap worth recording: in this llama.cpp version Qcur's
ggml dims are [head_dim, rows, heads] (the rope node views Q
token-major), so its flat dump is [heads][rows][head_dim], while
Kcur is a flat [k_width, rows]. Reading Qcur as [rows][heads][dim]
scrambles it and mimics a broken Q projection (cosine 0.19);
read correctly, the projection is exact. `compare_layers.py`
encodes the correct reading.

## Localization

With the prefix above, before the fix:

| Checkpoint | max_abs | cosine |
| --- | --- | --- |
| embd | 0.000000 | 1.000000 |
| L00_attn_norm | 0.000001 | 1.000000 |
| L00_q | 106.48 | 0.1887 |
| L00_k | 9.90 | 0.9999 |
| L00_v | 0.027 | 0.9999 |
| L00_ffn_inp | 0.351 | 0.9826 |
| L00_l_out | 1.417 | 0.9433 |

The first divergent checkpoint is layer 0's Q, and the split
between Q/K on one side and V on the other is the fingerprint:
embedding, RMSNorm, all three projections, and biases are proven
by `attn_norm` (1e-6) and `v` (the un-rotated projection, at the
quant-kernel noise floor). Only the rotated tensors diverge, so
the defect sits in the rotation step, before attention mixes
anything. Downstream, L01_ffn_out already shows max_abs 2865: the
error amplifies through the stack exactly as run A's logit swings
said it should.

## Root cause

**RoPE pairing.** `torch_math::rope` paired adjacent channels
(2i, 2i+1) (ggml's NORM pairing) while its own doc comment claimed
it was applying the GPT-NeoX pairing. llama.cpp rotates qwen2 (and
llama) with ggml's NEOX pairing: pair i joins channels i and
i + head_dim/2, at the same frequency schedule base^(-2i/head_dim).

Proof, from the layer-0 dumps alone (identical input, proven
above): re-rotating this crate's pre-RoPE Q and K dumps offline
under each pairing and comparing against llama.cpp's Qcur/Kcur:

| Tensor | NEOX (half-split) | interleaved (as built) |
| --- | --- | --- |
| L00 Q | max_abs 0.090, cos 0.999996 | max_abs 82.2, cos 0.948 |
| L00 K | max_abs 0.078, cos 1.000000 | max_abs 9.90, cos 0.9999 |

NEOX lands exactly on the quant noise floor set by V (0.027) and
by position 0, where rotation is the identity in any convention
(0.090). Interleaved is two to three orders of magnitude off.
Ruled out on the way: the fused-vs-split QKV question (this file
carries no `attn_qkv.weight`; both engines read the split tensors),
attention scale (1/sqrt(head_dim) on both sides), QK norms (qwen2
has none), per-layer config (eps 1e-6 and RoPE base 1e6 are read
from the file and reproduce in the meta dump), and the
decode-versus-prefill cache path (run A reproduced the divergence
in pure prefill probes).

Why the symptom looked like "attention over-boosts the last
token": with channels paired wrongly, each pair rotates at the
wrong frequency, so the relative-position structure of Q.K is
scrambled; the surviving signal is dominated by position-local
similarity, and the model repeats the salient operator. Position 0
is unaffected (identity rotation) and short contexts stay near the
reference, which is why run A saw agreement at prefix lengths 1, 2
and 5 but divergence at 3 and 4.

## The fix

`crates/torch-math/src/rope.rs`: `rope` / `rope_partial` now pair
(i, i + rotary_dim/2). The adjacent pairing survives as the
separately named `rope_interleaved`, because it is not wrong in
itself: llama.cpp selects NORM pairing for muse-glimmer (the Run B
follow-up settled that from `src/models/muse-glimmer.cpp`, and
cpu-compute's Glimmer path calls `rope_interleaved` for exactly
that reason). Pairing is a per-architecture convention, and the
two functions now carry the two names instead of one function
carrying the wrong one silently. The rope unit tests were updated
to values that distinguish the pairings (they previously encoded
the same wrong assumption as the code). torch-math: 42 passed.
arch-llama: 23 passed.

Note for the Glimmer line: this fix changes nothing on the
cpu-compute path (it never called `rope`), so run B's divergence
has a different cause; but any future arch module that reaches for
`torch_math::rope` now gets the llama/qwen2 convention by default,
which is the one the name was always supposed to mean.

## Verification after the fix

Same prefix, same harness, arch-llama rebuilt with the fix:

- Layer 0 is at the noise floor at every checkpoint (max_abs
  0.027 to 0.090, cosine >= 0.9993). Logits for the last position:
  top-5 identical in set, order, and value within 0.08
  (llama 21.404 / ours 21.324 for the winner); whole-vocab max_abs
  0.47, cosine 0.9989. The residual is accumulated Q4_K
  dequant-vs-ggml-kernel noise: its worst single element sits on a
  known outlier channel (L01_ffn_out: -2916.20 vs -2903.41, 0.44%
  relative), and it grows smoothly with depth instead of stepping
  at any layer or sub-step.
- Run A's probe cases, re-run as prefills (top-1):
  - "17 +" [16, 22, 488]: ours 220 (" "), llama.cpp 220. Was 488.
  - "17 -" [16, 22, 481]: ours 220 (" "), llama.cpp 220. Was 481.
  - factual_capital fork context (prompt + the 5 shared completion
    tokens): ours 15344 (" Italy"), golden 15344. Was 279 (" the").
- Full golden re-run (15 prompts, 64 tokens, greedy;
  `golden/candidates/qwen2.5-1.5b-q4km.fixed.candidate.jsonl`):
  **9/15 token-identical**, up from 0/15 in run A. The six
  remaining forks are late (first divergence at completion index
  2, 4, 8, 12, 20) or at a measured near-tie: for accented_spanish
  (fork at index 0) llama.cpp's own margin between its winner
  (668, logit 15.289) and our pick (10919, 15.094 in llama.cpp's
  logits) is 0.195 logits, while the cross-engine logit noise on
  that prompt measures up to 0.78 max (the top cluster sits within
  0.23 on both sides). A 0.195 margin inside a ~0.5 noise band is
  a coin flip, not a systematic error; the layer diff above shows
  no second math divergence for it to come from. Full 15/15 over
  64 greedy steps would require reproducing ggml's quantized
  kernels bit-for-bit, which an f32-dequantize reference does not
  do and run A's fixture comparison was never going to show once
  the real bug was gone.

## Files

- Harness: `golden/tools/cb_dump.cpp`,
  `golden/tools/compare_layers.py`,
  `crates/arch-llama/examples/layer_dump.rs` (runner),
  `crates/gguf/examples/qkv_probe.rs` (the fused-vs-split probe
  that cleared the tensor-source question),
  `LlamaModel::forward_traced` in `crates/arch-llama/src/model.rs`.
- Fix: `crates/torch-math/src/rope.rs`.

## Follow-up: pairing is per-architecture (the llama half of the claim above was wrong)

The root-cause section above says llama.cpp rotates "qwen2 (and
llama)" with NEOX. The qwen2 half is proven by the layer diff;
the llama half was asserted, not measured, and it is wrong.
llama.cpp's own table (`llama_model_rope_type` in
src/llama-model.cpp) maps `LLM_ARCH_LLAMA` to
`LLAMA_ROPE_TYPE_NORM` (adjacent), `LLM_ARCH_QWEN2` to
`LLAMA_ROPE_TYPE_NEOX` (half-split), `LLM_ARCH_MUSE_GLIMMER` to
NORM. Because 13643e0 flipped the one shared function
arch-llama calls for both architectures, it fixed qwen2 and
regressed TinyLlama from 6/15 to 0/15 token-identical in the
same edit; the TinyLlama re-run note
(`RESULTS-tinyllama-spm.md`, regression follow-up) caught it.

The structural fix has now landed: `LlamaConfig::rope_pairing`
(`crates/arch-llama/src/config.rs`) is the one place the mapping
lives, selecting `RopePairing::Interleaved` for llama and
`RopePairing::HalfSplit` for qwen2 from the file's own
architecture string, and refusing an unmapped architecture with
the named error `LlamaError::RopePairingUnknown`. `rope_rows`
dispatches on it; the interleaved partial-rotary form it needs
is `torch_math::rope_interleaved_partial`. The doc comment in
`crates/torch-math/src/rope.rs` that carried the wrong "llama
and qwen2" claim has been corrected to match.

### Re-run (this file's model)

Same conditions as the verification above: David's
ubuntu-server, release build, RUSTFLAGS target-cpu=x86-64-v3,
CPU only, runner `crates/arch-llama/examples/golden_run.rs`,
greedy, 64 tokens max. Production Glimmer llama-server
untouched (health check after both re-runs: `{"status":"ok"}`,
still on its original pid). Wall time 971.7 s.

| Model | Token-identical | Text-identical | Divergent | Missing |
| --- | --- | --- | --- | --- |
| Qwen2.5-1.5B Q4_K_M | 9 / 15 | 0 / 15 | 6 | 0 |

**9/15 holds exactly.** Stronger: the per-arch candidate is
token-for-token identical to this note's `.fixed` candidate on
all 15 prompts: same nine token-identical prompts, same six
fork points (factual_history 4, code_python 2, list_colors 8,
accented_french 12, accented_german 20, accented_spanish 0).
The dispatch is a pure no-op on the qwen2 path, as designed: it
routes to the same `rope_partial` with the same arguments, and
the paired TinyLlama re-run proves the other branch restores
that model's pre-regression trajectories exactly. Neither
model is below its best-known state, so the fix ships with
both fixtures guarding their own direction again.

### Files (per-arch fix)

- Candidate: `golden/candidates/qwen2.5-1.5b-q4km.perarch.candidate.jsonl`
- Fix: `crates/arch-llama/src/config.rs`, `crates/arch-llama/src/model.rs`,
  `crates/arch-llama/src/error.rs`, `crates/arch-llama/src/lib.rs`,
  `crates/torch-math/src/rope_interleaved.rs`,
  `crates/torch-math/src/lib.rs`, `crates/torch-math/src/rope.rs`
  (doc correction only)
