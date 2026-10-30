# Integration run B: Glimmer's first generation on the new stack

Run 2026-10-29. Runner: `examples/glimmer_generate.rs` in this crate.
Raw record on the server in `~/itak-run-b/` (`run.log`, `run.err`,
`ref-completion.json`, `ref-table.txt`).

## Verdict

Divergence at the first generated token. The pipeline works
mechanically end to end on the real 30B file: parse, coverage gate,
tokenizer, forward, KV cache, greedy loop, no refusals, no
placeholder paths. The forward's numerics do not yet match
llama.cpp: on identical prompt ids its argmax is wrong and its
probability mass is unconcentrated, and a second symptom appears at
decode time (the distribution stops tracking the fed token). No
parameter was tuned to force a match; this note is the evidence.

## Setup

- Prompt: `factual_capital`, the shortest factual prompt in
  `golden/prompts.jsonl`: "The capital of France is".
- Prompt ids, from the tokenizer crate (the file's
  `tokenizer.ggml.add_bos_token` is true, so BOS 200000 is
  prepended; the encode itself is the T-2.3-proven one):
  `[200000, 954, 7963, 323, 11698, 373]`.
- Reference: llama.cpp (build-vulkan2) `llama-server`, CPU only
  (`-ngl 0`, 8 threads), port 18299, the same GGUF with no mmproj
  and no draft model. The prompt was posted to `/completion` as a
  token id array, so both engines consumed byte-identical input;
  tokenization cannot explain any difference. Temperature 0,
  `n_predict` 8, `n_probs` 5, `ignore_eos` true. The production
  GPU server on port 8080 was never touched (health-checked before
  and after).
- Ours: the full crate chain (`gguf`, `itak-loader`,
  `tokenizer`, `muse-glimmer` forward on `cpu-compute`), release
  build, run on the server. `cpu-compute` is single-threaded; the
  process sat at 99.9% of one core throughout.

## Reference result (llama.cpp)

Generated ids: `[13796, 26, 1573, 373, 290, 2167, 26899, 1212]`,
text " Paris. It is the most populous". Timings: prompt 795 ms for
6 tokens, decode 314 ms per token.

Top-5 per position (id, token, probability):

| Pos | Sampled | Top-5 |
| --- | --- | --- |
| 1 | 13796 | 13796 " Paris" 0.3962; 262 " a" 0.0961; 290 " the" 0.0513; 1085 " one" 0.0462; 5711 " known" 0.0323 |
| 2 | 26 | 26 "." 0.4721; 24 "," 0.2480; 1574 ".␊␊" 0.1338; 341 " and" 0.0559; 335 ".␊" 0.0239 |
| 3 | 1573 | 1573 " It" 0.2794; 589 " The" 0.1993; 13796 " Paris" 0.1496; 1394 " This" 0.0278; 220 " " 0.0252 |
| 4 | 373 | 373 " is" 0.8585; 947 " has" 0.0574; 1082 "'s" 0.0470; 812 " was" 0.0098; 23374 " lies" 0.0074 |
| 5 | 290 | 290 " the" 0.3078; 262 " a" 0.1228; 1085 " one" 0.1172; 5711 " known" 0.1113; 1437 " also" 0.1089 |
| 6 | 2167 | 2167 " most" 0.3561; 13467 " largest" 0.2767; 21989 " biggest" 0.0324; 5376 " center" 0.0312; 7963 " capital" 0.0311 |
| 7 | 26899 | 26899 " popul" 0.5940; 70785 " populated" 0.2413; 25600 " visited" 0.0551; 7559 " popular" 0.0337; 24093 " famous" 0.0202 |
| 8 | 1212 | 1212 "ous" 0.9996; 11381 "ously" 0.0001; 66950 "ousand" 0.00004; 474 "ou" 0.00004; 5364 "ouse" 0.00003 |

## Our result

Generated ids: `[290, 290, 15, 15, 15, 15, 15, 15]`, decoded text
" the the######".

Position 1 (the one position where both engines saw the identical
sequence) top-5, from the forward's finished logits:

| Rank | Id | Token | Logit | Prob |
| --- | --- | --- | --- | --- |
| 0 | 290 | " the" | 1.223680 | 0.000044 |
| 1 | 341 | " and" | 0.842154 | 0.000030 |
| 2 | 26 | "." | 0.798573 | 0.000029 |
| 3 | 373 | " is" | 0.771220 | 0.000028 |
| 4 | 262 | " a" | 0.762244 | 0.000028 |

The comparison at position 1:

- Argmax differs: ours 290, reference 13796. Greedy decoding makes
  this a hard mismatch, not a sampling artifact.
- Overlap is partial: our rank 0 is the reference's rank 2 (290),
  our rank 4 is its rank 1 runner-up (262); its top token (13796)
  is not in our top-5.
- The internal spread of the top-5 is similar (about 2.5 nats on
  both sides). The divergence is mass concentration: the
  reference's top-5 holds 0.62 of the probability, ours holds
  1.6e-4. In our forward the vocabulary bulk is not suppressed
  relative to the top, which means the final hidden state's
  direction is wrong enough that the head cannot separate the
  answer from the bulk, while still ranking plausible function
  words first. The failure is upstream of the head, in the layer
  stack, not in the finishing math or the sampler.

Positions 2 to 8 are not continuations of the same sequence (each
engine conditions on its own tokens), so they are reported as our
engine's behavior, not as per-position diffs.

## Second symptom: the decode-time freeze

- Step 2 (feeding 290 at position 6) returned a top-5 identical to
  the prefill's to all six printed decimals on every entry.
- Steps 3 to 8 returned one shared top-5 (15 "#" 0.009546; 1898;
  442; 13266; 2694), identical to six decimals across steps that
  fed different tokens (290, then 15) at advancing positions. The
  single variation observed anywhere is 1e-6 in step 5's rank-4
  logit.
- Reading: at decode time the returned distribution is, to about
  1e-6 of its own scale, independent of the token being fed and of
  the position. The new token's contribution is attenuated to
  noise, or the decode path reads a state that stopped tracking
  the input after the first decode step. The synthetic
  incremental-vs-prefill proof in `cpu-compute` (2 layers, 4
  heads, head dim 8, window 16 and 3) does not expose this; the
  real config is 52 layers, 32 Q heads over 2 KV heads, head dim
  128, window 2048.

## Timings and resources (ours)

- Open file: 4,720 ms. Weight load through the coverage gate:
  5,441 ms (731 tensors, 16,743,567,360 bytes served).
- Prefill, 6 tokens, multi-row path: 1,532,787 ms (about 25.5
  minutes).
- Decode steps 2 to 8: 45,174; 45,641; 44,637; 43,906; 43,988;
  42,844; 43,720 ms. About 44 to 46 s per token, roughly 34x
  cheaper per token than the prefill path.
- Peak RSS: 38,190,400 KiB (about 36.4 GiB): the file image, the
  served tensor copies, and transient f32 dequantization buffers.
  There is no resident f32 weight copy; weights dequantize per
  projection call, as the crate documents.
- Prefill cost suspects, in order: the per-projection strided
  transpose in `cpu-compute`'s `proj` for rows > 1, the full
  dequantize of every tensor on every call, then the f64 matmul.
  Reference contrast: llama.cpp on 8 threads did the same prefill
  in 795 ms and decode at 314 ms per token. This backend is a
  correctness reference; the timings quantify how much the
  specialized kernels have to win.

## Diagnosis, ordered

1. **Norm convention.** Measured through our own loader
   (`load_f32`), the file's norm weights have an unusual profile:
   `output_norm.weight` mean +0.017, std 3.13, range [-4.94,
   +4.38] (centered on zero); per-layer norms centered higher
   (`blk.0.attn_norm` +1.34, `blk.51.attn_norm` +1.30,
   `blk.0.ffn_norm` +1.17, `blk.0.post_attention_norm` +0.67);
   `blk.0.attn_q_norm` is a constant vector (all 128 entries
   exactly 3.87) and `blk.0.attn_k_norm` a constant 1.0. Our
   forward applies plain `x * w` RMSNorm everywhere. If
   llama.cpp's muse-glimmer graph uses the Gemma-style `(1 + w)`
   form, every norm in all 52 layers is wrong in ours, which fits
   a near-total loss of signal. The stats alone do not prove the
   convention (block norms center above 1, the final norm at 0),
   so this is the first thing to settle against the llama.cpp
   graph source for this architecture, not a conclusion.
2. **Attention/KV handling at real scale** (the freeze, above).
   Something in the decode path (position handling, head mapping,
   ring state) behaves differently at 32/2 heads and 52 layers
   than at the synthetic scale where the proofs pass.
3. **The two flagged placements** from the architecture module:
   post norms after each residual add, and the attention gate
   reading the normed input. Still unverified against the
   reference graph; either being wrong degrades the output but
   would not obviously un-concentrate the mass on its own.
4. **QK norm application.** Ours norms Q and K per head after the
   projections, before RoPE. With constant-vector norm weights,
   any convention error here is a pure attention temperature
   shift; it cannot by itself produce the observed bulk behavior.
5. **RoPE geometry.** The file's metadata has no
   `rope.dimension_count` key (checked), so full 128-dimension
   rotation matches the file's surface. llama.cpp could still
   hardcode partial rotation for this architecture in source;
   lower suspicion, settled by the same source read as item 1.
6. **Softcap/scale order is excluded as the argmax cause.** The
   finishing map is monotone, so it cannot reorder candidates;
   and the mass-concentration gap is a property of the head's
   input direction, not of the finishing scalars.

## What this run established

- The complete crate chain runs the real 30B model end to end
  with the coverage gate enforced and no refusals: Glimmer's
  first generation on the new stack exists, and it is wrong in a
  precisely measured way.
- The next step that localizes the error: dump our per-layer
  activations for this prompt and diff them against llama.cpp's
  graph intermediates for the same ids, after settling the norm
  convention from the llama.cpp muse-glimmer graph source. One
  layer's first divergence will name the defect; guessing further
  from end-to-end numbers will not.


## Follow-up (2026-10-29 evening): conventions settled from source, forward fixed, Stage 1 re-run

The conventions were settled by reading llama.cpp's own source on
the server: `src/models/muse-glimmer.cpp` (the graph),
`conversion/muse_glimmer.py` (the converter), plus
`llama-graph.cpp` (`build_norm`), `llama-model.cpp`
(`llama_model_rope_type`, n_rot defaults), and `llama-hparams.cpp`
(SWA pattern). No convention below is a guess; each carries its
citation. Commits: a777db6 (muse-glimmer), 1956b27 (torch-math),
082d7fd (cpu-compute). Raw record of the re-run on the server in
`~/itak-run-c/` (`run.log`, `run.err`).

### Verdict of the re-run

The divergence is gone. Our generated ids are
`[13796, 26, 1573, 373, 290, 2167, 26899, 1212]`, exactly the
reference ids, all 8 positions, and the per-position probabilities
track the reference to within 0.015 (f32 reference math vs
llama.cpp kernel accumulation). The decode freeze is gone with
it: every decode step's distribution answers the fed token and
matches the reference step for step.

### Settled conventions

Wrong in our code, fixed:

1. **Post-norm placement and eps.** The reference applies each
   post norm to the sub-block's branch output, then adds the
   untouched residual: `x = x + rms_norm(branch, post_w)` with a
   hardcoded eps of 1e-8 (`muse-glimmer.cpp`: `post_norm_eps =
   1e-8f`, applied at the attention and FFN branches). Ours
   normalized the sum `residual + branch` at the file eps 1e-5,
   renormalizing the whole residual stream 104 times per token.
   Fixed in `muse-glimmer::model` (`POST_NORM_EPS`).
2. **Embedding norm missing.** The graph runs a weightless
   RMSNorm over the embedding output before layer 0
   (`muse-glimmer.cpp`: `build_norm(inpL, nullptr, nullptr, ...)`,
   which is plain `ggml_rms_norm` in `llama-graph.cpp`). Ours fed
   raw embeddings into layer 0. Added as a distinct
   `Compute::rms_norm_plain` step, called right after `embed`.
   There is no embedding scale factor anywhere in the graph.
3. **RoPE on the wrong layers.** RoPE runs on the sliding-window
   layers ONLY; the 13 global layers are NoPE
   (`muse-glimmer.cpp`: "RoPE runs on the SWA layers, NoPE on
   full ones"). Ours rotated every layer. `CpuCompute::attention`
   now rotates only `LayerKind::SlidingWindow`, at the SWA base
   (`rope.freq_base_swa` when the file carries it, else the train
   base, per `llama-model.cpp`; the config now resolves and stores
   it. This file carries no SWA key, so 500000 either way.)
4. **RoPE pairing.** llama.cpp's rope type for this arch is
   `LLAMA_ROPE_TYPE_NORM` (`llama-model.cpp`), ggml's adjacent-pair
   mode, and the converter unpermuted Q/K into that layout
   (`muse_glimmer.py`: `permute` with `qk_order = NORM`). A new
   `torch_math::rope_interleaved` implements that pairing under
   its own name, and the Glimmer backend calls it; the half-split
   NEOX form stays in `torch_math::rope` for the llama/qwen2
   families, whose reference uses it (sibling fix 13643e0, from
   the Qwen2.5 layer diff). The pairings pair different channels
   and are not interchangeable.
5. **Head finishing order.** The graph scales first, then
   softcaps: `ggml_scale` by `logit_scale`, then `build_softcap`
   (`cap * tanh(z / cap)`), `muse-glimmer.cpp` head block. Ours
   capped then scaled. Monotone either way, so this never moved
   argmax (Run B diagnosis item 6 was right); it did move every
   probability. The unit test that pinned the wrong order is
   rewritten with knee values that tell the orders apart
   (z = 20, cap = 20, scale = 0.5: 9.2423 settled, 7.6159 wrong).

Already matching the source, verified and left alone:

- **RMSNorm form is plain `x * w` everywhere in the graph.**
  Run B diagnosis item 1 is settled as NOT a graph-side error:
  the Gemma-style (1 + w) shift was folded into the stored
  weights at conversion for the four per-layer norms only
  (`muse_glimmer.py`: `norm_shift` returns 1.0 for names ending
  `layernorm.weight`, 0.0 otherwise; the final norm was
  deliberately unshifted, which is why `output_norm` centers on
  zero while block norms center near 1). All other norms (pre,
  QK, final, embedding) use the file eps 1e-5 via `build_norm`.
- **Attention gate.** Gate projection reads the same normed
  pre-attention input as Q/K/V; `sigmoid(gate)` multiplies the
  attention output elementwise before the output projection
  (`muse-glimmer.cpp`). Ours already did exactly this.
- **QK norms.** Per-head RMSNorm over head_dim with the stored
  weights, after the projections, before RoPE. The stored
  weights are synthesized by the converter (q_norm = constant
  `qk_scale_factor` = 3.87, matching the measured constant
  vector; k_norm = ones), so plain application is correct.
- **RoPE geometry.** Full head_dim rotation (n_rot = 128; the
  file carries no `rope.dimension_count` and no `rope.freq_base_swa`
  key, both confirmed by a full metadata strings scan; defaults
  in `llama-model.cpp`), base 500000, and the SWA pattern
  (globals at il % 4 == 3) matches our derived pattern.

### Decode freeze: what it was

No discrete cache bug exists. A new intermediate-scale test
(8 layers with globals at 3 and 7, 32 Q heads over 2 KV heads,
head dim 16, hidden 512, the real eps, base, cap, and scale;
`cpu-compute/tests/forward_tests.rs`) measures decode
sensitivity directly: prefill a prompt, then feed token 7 vs
token 42 and compare logits, plus a one-shot equivalence check.
Sensitivity was healthy at that scale (above the 1e-3 assertion
floor; the freeze signature in Run B was about 1e-6) both before
and after the fixes, and the cache position cross-check, the
attention crate's absolute-position masks and views, and the
incremental-vs-one-shot proofs all hold at every scale tested.
The freeze was the convention errors compounding over 52 layers:
post norms renormalizing the residual stream after every
sub-block (each new token's contribution repeatedly rescaled
against the whole stream), a missing embedding norm, and RoPE
rotating the global layers' Q/K with position garbage. Two more
regression tests pin the fixed conventions decisively: a
tiny-branch test (attention output scaled 1e-4, so the branch
RMS sits near sqrt(1e-8)) separates the post-norm eps and
placement readings by more than an order of magnitude, and the
naive-reference attention proof now runs for both layer kinds,
rotating only the sliding one.

Gates: `cargo test --workspace` 196 passed, 0 failed, and
`cargo clippy --workspace --all-targets` clean, both on the
final tree (including `attention_mixing_matches_a_naive_reference`,
re-confirmed green after the sibling RoPE commits landed; it had
failed transiently mid-flight while these edits and the sibling
pairing flip crossed in the shared tree).

### Re-run (Stage 1, same protocol as Run B)

Same prompt, same ids `[200000, 954, 7963, 323, 11698, 373]`,
same server, CPU-only reference runner on our side (prefill
1,545,093 ms, decode about 47 s/token, peak RSS 38,189,888 KiB;
the production GPU server untouched). Per position, reference
top-1 vs ours:

| pos | reference | p (ref) | ours | p (ours) |
|----:|-----------|--------:|------|---------:|
| 1 | 13796 " Paris" | 0.3962 | 13796 " Paris" | 0.4035 |
| 2 | 26 "." | 0.4721 | 26 "." | 0.4818 |
| 3 | 1573 " It" | 0.2794 | 1573 " It" | 0.2759 |
| 4 | 373 " is" | 0.8585 | 373 " is" | 0.8563 |
| 5 | 290 " the" | 0.3078 | 290 " the" | 0.3066 |
| 6 | 2167 " most" | 0.3561 | 2167 " most" | 0.3507 |
| 7 | 26899 " popul" | 0.5940 | 26899 " popul" | 0.5793 |
| 8 | 1212 "ous" | 0.9996 | 1212 "ous" | 0.9996 |

Generated text: " Paris. It is the most populous", identical to
the reference. Top-5 mass at position 1: ours 0.6252, reference
0.6221; in Run B ours held 0.00016 (top-1 p 4.4e-5 on the wrong
token). Top-5 id sets match the reference exactly at positions
2, 3, 4, 5, 7, 8; at positions 1 and 6 the fifth slot is a
near-tie swap (position 1: ours 1437 " also" 0.0316 vs reference
5711 " known" 0.0323; position 6: ours 7257 " country" 0.0315 vs
reference 5376 " center" 0.0312), consistent with f32-vs-kernel
accumulation noise at the boundary, not a structural difference.
