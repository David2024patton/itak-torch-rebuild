# Layer diff: Muse-Glimmer-30B forward vs llama.cpp, stage by stage

Date: 2026-11-05. Executes FORWARD-DIFF-PLAN.md
(`crates/itak-loader/FORWARD-DIFF-PLAN.md`), the localization plan
written after Integration Run B: our side of the diff (the
muse-glimmer trace tooling) was built on lane-01-forward-diff but
never merged, and the reference side (a llama.cpp per-layer
activation dump for Glimmer) never existed. Both sides now exist
and have been run against each other. This file is the Glimmer
sibling of `RESULTS-layer-diff-qwen.md` and is filed beside it for
that reason.

## Provenance

- Model: `/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`
  (the production Q4_K_M file), on ubuntu-server.
- Prompt: the golden `factual_capital` prompt, "The capital of
  France is", tokenized as Run B tokenized it (BOS per the file's
  metadata): ids [200000, 954, 7963, 323, 11698, 373]. Both sides
  consumed exactly these ids; our side prints the match, the
  reference side took them as a literal argument.
- Our side: `crates/itak-loader/examples/glimmer_layer_trace.rs`
  over `GlimmerModel::forward_traced` (muse-glimmer `trace`
  module, ported from lane-01 onto current main in this branch;
  stage semantics follow current main's forward: `embed` is the
  residual after the weightless embedding norm, `layer.<i>.mid`
  is the residual after the attention half with the post-attention
  norm applied to the branch output before the residual add,
  `layer.<i>.out` the same for the FFN half). CPU compute backend,
  one prefill, 107 trace lines.
- Reference side: `golden/tools/cb_dump_glimmer.cpp`, a sibling of
  the proven `cb_dump.cpp`, built against the server's llama.cpp
  tree (`/home/server/llama.cpp`, build-vulkan2 libraries), CPU
  only (n_gpu_layers = 0), one prefill via `llama_decode` with the
  `cb_eval` callback copying the named graph tensors to .bin
  files. The production Glimmer server stayed down and untouched;
  nothing bound port 8080; the Qwen3-Embedding tenants were not
  touched.
- Diff: `golden/tools/compare_traces.py` reduces the reference
  bins to the trace's statistics (mean, population variance, max
  abs, first values; whole tensor and last row, computed in f64)
  and applies the plan's reading rules: a stage matches when mean
  and variance agree to roughly 1e-2 relative (means judged in
  units of the stream's own standard deviation, variances
  relatively), last row first, whole tensor second. The first
  stage whose input stage matched but which does not itself match
  is the defect's home.

## Tensor name mapping (read from the graph source)

Read from `/home/server/llama.cpp/src/models/muse-glimmer.cpp`
(201 lines) on 2026-11-05:

- `inp_embd` (build_inp_embd output) dumped as `embd_raw`, info
  only.
- Line 69-70: `inpL = build_norm(inpL, nullptr, nullptr,
  LLM_NORM_RMS, -1)` (weightless) then `cb(inpL, "embd_norm", -1)`:
  the post-embedding-norm residual, dumped as our `embed` stage.
- Line 150-151: `ffn_inp = ggml_add(cur, inpSA)` then
  `cb(ffn_inp, "ffn_inp", il)`: the residual after the attention
  half, our `layer.<i>.mid`. Lines 141-142 show the post-attention
  norm is `ggml_rms_norm` (eps hardcoded 1e-8, line 66) followed
  by `ggml_mul` with the post-norm weight, applied to the branch
  BEFORE the add: the same placement current main implements.
- Line 170-172: after the FFN post norm (lines 166-167, same
  shape), `cur = ggml_add(cur, ffn_inp)`, `cb(cur, "l_out", il)`:
  our `layer.<i>.out`.
- Line 180-181: final `build_norm` with `output_norm`,
  `cb(cur, "result_norm", -1)`: our `final_norm` (last position
  reduced).
- Line 185-195: `build_lora_mm(model.output)`, `ggml_scale` by
  the logit scale, then the softcap (scale by 1/cap, tanh, scale
  by cap), `cb(cur, "result_output", -1)`: our `logits` (last
  position reduced). Scale-then-softcap, the same order current
  main's `softcap_scale` implements.

Two plan hypotheses are settled by this source read alone, before
any numbers: hypothesis 1 (norm convention): line 90's comment
states the pre-attention norm's "(weight+1) folded at conversion
time", so the reference applies plain `x * w` RMSNorm to the
stored weights, as ours does. Hypothesis 4 (RoPE geometry): line
86-87, RoPE runs on the SWA layers only (`use_rope =
hparams.is_swa(il)`), matching our sliding/global split.

## Results

Self-checks first, per the plan: our side printed
`prompt_ids=[200000, 954, 7963, 323, 11698, 373]` with
`run_b_ids_match=true`, coverage 731/731 tensors, and exactly 107
trace lines. The reference side took the same ids as a literal
argument and its own sanity line (top-1 from the public logits
API) agrees with its dump: id 13796 at 17.3194. Run times: our
traced prefill 1,464 s on the CPU backend (24m38s wall for the
whole example); the reference dump 12 s. The server session ran
under the overnight phase lock, queued behind the other lanes.

Every one of the 106 stages from `embed` through `final_norm`
matches under the plan's reading rules. The worst last-row
variance disagreement anywhere in the stack is 9.6e-3 relative
(layer.39.mid), under the 1e-2 bar; mean shifts never exceed
5.8e-4 of a standard deviation. `embed` is essentially exact
(variance relative diff 3.2e-8), which kills hypothesis 0
(embedding handling) outright. Summary rows (last-row statistics;
mean shift in units of the stream's standard deviation):

| stage | mean shift (std) | var rel diff | whole-tensor var rel |
|---|---|---|---|
| embed | 3.8e-10 | 3.2e-08 | 1.7e-08 |
| layer.0.mid | 1.0e-04 | 6.0e-04 | 1.6e-03 |
| layer.0.out | 2.0e-04 | 3.6e-03 | 2.0e-03 |
| layer.26.out | 2.4e-04 | 3.7e-04 | 1.7e-03 |
| layer.51.out | 3.9e-04 | 6.8e-03 | 2.0e-03 |
| final_norm | 2.7e-04 | 7.5e-03 | 7.5e-03 |
| logits | 1.7e-02 | 3.0e-03 | 3.0e-03 |

The full 107-row table is `compare.txt` in the server staging
(`~/overnight-forward-diff/`); the comparison script prints it on
any rerun from the two dumps.

At `logits` the strict judge flags one metric over the bar (mean
shift 1.7e-2 of a standard deviation; the variance agrees to
3.0e-3). The substance underneath the flag: both sides put token
13796 (" Paris") on top, and the same five tokens hold the top
five slots (the 5th slot differs by 0.014 of a logit, a tie for
practical purposes). Per-token logit differences are small and
signed both ways (13796: +0.0245; 262: -0.0088; 290: -0.0404;
1085: +0.0133), so this is neither a uniform shift nor a scale
error. Top-1 probability: ours 0.403462, reference 0.396118.

One observation from our run's diagnostic stream, recorded for
completeness: the loader logged two metadata events
(`muse-glimmer.attention.sliding_window_pattern` type mismatch,
`muse-glimmer.rope.freq_base_swa` missing) and proceeded on its
fallbacks. The stage statistics above prove the values those
fallbacks produced agree with the reference graph at every layer,
so they are not implicated in anything this diff measured.

## Verdict

**No layer diverges. The forward's layer math is llama.cpp's.**
All 106 pre-logits stages match within the plan's tolerance, so
none of the plan's layer-level hypotheses (0 embedding handling,
1 norm convention, 2 post-norm and gate placement, 3 QK-norm,
4 RoPE geometry) is live: each is decided "same as the
reference" by the stage that was assigned to decide it, and
hypotheses 1 and 4 were additionally settled by the graph source
read above before any numbers ran.

The only remaining difference is at the `logits` stage itself,
and it is exactly the known first-token residue, now measured
end to end: same argmax, same top-5 set, top-1 probability
0.403462 (ours) against 0.396118 (llama.cpp), produced by small
token-dependent logit differences of a few hundredths. With
`final_norm` (the lm_head's input) matching, that residue lives
in the finishing path: the lm_head's accumulation over the
6656-wide dot product (summation order and intermediate
precision differ between the two engines) and its interaction
with the scale-and-softcap finish. It is a finishing-math
question, not a layer question, and this diff closes the layer
question for good. Localizing inside the head (for example,
comparing the pre-softcap logits or accumulating the head in
f64) is a separate, smaller follow-up; per this task's scope no
forward code was changed.
