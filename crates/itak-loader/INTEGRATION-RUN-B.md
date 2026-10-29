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
