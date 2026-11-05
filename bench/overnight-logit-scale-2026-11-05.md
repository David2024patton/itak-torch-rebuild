# Overnight experiment: forcing Glimmer's logit scale to 1.0

Run 2026-11-05 (server, CPU forward only). Experiment, not a fix:
shipped defaults are unchanged. Branch `lane/overnight-logitscale`,
hook commit `5dbaaadab8922fd1752f59f9d71440c46bf94a2b` on top of main
`bda7ff55ea57f4a764afddf9ba2e050c59b86916`.

## Question

TODO T-9.3's residue: on the golden prompt our first-token top-1
probability is 0.4035 against llama.cpp's 0.3962 (gap 0.0073), with
identical tokens. The finish in `crates/muse-glimmer` applies
`logit_scale` (file value 0.19611613) FIRST, then softcap
`20.0 * tanh(z / 20.0)`. This experiment forces the scale to 1.0
(5.0990 times the file value), order and cap unchanged, and measures
exactly what moves: could a scale mishandling produce a
probability-only residue of that size, and does forcing 1.0 move our
number toward or away from llama.cpp's 0.3962?

## Method

- Hook: example-only env override `ITAK_EXPERIMENT_LOGIT_SCALE` in
  `crates/itak-loader/examples/glimmer_generate.rs`. When set to a
  positive finite number it replaces `cfg.logit_scale` after
  `load_config`; unset, the file value is used untouched. No library
  code reads it. Unit tests cover unset / set / invalid values.
  Gates: `cargo test -p itak-loader` green, clippy clean, fmt clean.
- Harness: the same example's CPU path (`cpu-compute`), release
  build, real GGUF
  `/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`,
  staged at `~/overnight-logitscale` on the server (example sha256
  verified against the worktree:
  `f59903d01b0007ecc2d460a36de6c8a654bfbc40b2fa1bfd0d6be1aa878321ca`).
  Runs executed under the shared phase lock; raw logs are in
  `bench/overnight-logit-scale-logs/` and on the server in
  `~/overnight-logitscale/results/`.
- Arms per prompt: baseline (env unset, file scale 0.19611613) and
  forced (`ITAK_EXPERIMENT_LOGIT_SCALE=1.0`), softcap 20.0 in both.
  Recorded per prompt: first-token top-5 (id, token, finished logit,
  probability), a stats line (finished range; pre-softcap range
  recovered by inverting the softcap; implied raw lm_head range;
  count of logits saturated to the cap; count of the top 100
  finished logits beyond `0.9 * cap` = 18.0), and 16 greedy tokens.
- Prompts: golden `factual_capital` "The capital of France is"
  (ids [200000, 954, 7963, 323, 11698, 373]), "The capital of Japan
  is" (6 ids), "The chemical symbol for water is"
  (ids [200000, 954, 12332, 9985, 393, 3427, 373]).
- Note on the recovery math: the pre-softcap range is computed as
  `cap * atanh(finished / cap)` over entries that did not saturate.
  For near-cap entries this inversion is ill-conditioned; the forced
  arm's recovered maximum carries visible error (see golden: forced
  recovery says 135.0004 against the baseline arm's raw maximum
  134.7835 for the identical body output). Minima agree to 1e-3.
  Treat recovered near-cap maxima as approximate.

## Results: golden prompt "The capital of France is"

First-token top-5. llama.cpp reference (INTEGRATION-RUN-B):
13796 " Paris" 0.3962; 262 " a" 0.0961; 290 " the" 0.0513;
1085 " one" 0.0462; 5711 " known" 0.0323.

| Rank | Id | Token | Base logit | Base prob | Forced logit | Forced prob |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 13796 | " Paris" | 17.343927 | 0.403462 | 19.999945 | 0.000968 |
| 1 | 262 | " a" | 15.893860 | 0.094634 | 19.999369 | 0.000967 |
| 2 | 290 | " the" | 15.235032 | 0.048969 | 19.998516 | 0.000966 |
| 3 | 1085 | " one" | 15.183412 | 0.046505 | 19.998421 | 0.000966 |
| 4 | 1437 | " also" | 14.798096 | 0.031635 | 19.997526 | 0.000965 |

- The baseline arm reproduces the recorded values exactly
  (top-1 logit 17.343927, prob 0.403462; the first 8 generated ids
  match llama.cpp's recorded continuation).
- Forced arm: identical top-5 ids in identical order, but every
  probability collapses. Top-1 falls 0.403462 to 0.000968 (417x).
  The top-5 probabilities become nearly equal (0.000968 down to
  0.000965, a 0.3 percent spread): the head of the distribution is
  flat. A top-1 probability of 0.000968 is 1/1033, so roughly a
  thousand tokens share the head mass almost equally.
- The same flattening holds at every generated position, e.g.
  step 4 (id 373): baseline top-1 prob 0.856342, forced 0.003043;
  step 2 (id 26): 0.481786 against 0.002340.
- Greedy tokens, 16, BOTH arms, identical:
  [13796, 26, 1573, 373, 290, 2167, 26899, 1212, 6996, 310, 11698,
  341, 290, 2167, 26899, 1212].

Logit ranges and saturation, first-token distribution:

| Measure | Baseline (scale 0.19611613) | Forced (scale 1.0) |
| --- | --- | --- |
| Finished range | [-13.1550, 17.3439] | [-19.9872, 19.9999] |
| Pre-softcap range (recovered) | [-15.7768, 26.4332] | [-80.4453, 135.0004] |
| Raw lm_head range (implied) | [-80.4463, 134.7835] | (same body output) |
| Logits saturated to exactly cap | 0 | 0 |
| Top-100 finished beyond 0.9 * cap (18.0) | 0 of 100 | 100 of 100 |

No logit reaches exactly 20.0 in f32 (tanh approaches the cap
asymptotically); the forced leaders sit within 0.0025 of it.

## Results: "The capital of Japan is"

First-token top-5:

| Rank | Id | Token | Base logit | Base prob | Forced logit | Forced prob |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 40171 | " Tokyo" | 17.584030 | 0.635992 | 19.999966 | 0.001504 |
| 1 | 262 | " a" | 15.160600 | 0.056360 | 19.998377 | 0.001502 |
| 2 | 3721 | " **" | 14.582867 | 0.031627 | 19.996861 | 0.001499 |
| 3 | 290 | " the" | 14.405860 | 0.026497 | 19.996202 | 0.001498 |
| 4 | 1085 | " one" | 13.849966 | 0.015197 | 19.993313 | 0.001494 |

- Top-1 falls 0.635992 to 0.001504 (423x). Identical ids and order.
- Baseline ranges: finished [-13.4323, 17.5840], pre-softcap
  [-16.2735, 27.4448], raw [-82.9791, 139.9415], top-100 beyond
  0.9 * cap: 0. Forced: finished [-19.9900, 20.0000] (the max is
  19.999966 before print rounding), top-100 beyond: 100 of 100.
- The forced arm ran prefill-only (n=1) in the core matrix; its
  greedy trajectory is the baseline's by the monotonicity argument
  below (a full forced top-up run was queued after the core matrix;
  see the run record). Baseline 16 greedy tokens:
  [40171, 26, 1573, 373, 290, 13467, 6996, 310, 290, 3817, 517,
  262, 7001, 323, 1278, 220].

## Results: "The chemical symbol for water is"

First-token top-5 (baseline measured; forced column filled from the
forced prefill run):

| Rank | Id | Token | Base logit | Base prob | Forced logit | Forced prob |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 465 | " H" | 17.642365 | 0.711187 | 19.999969 | 0.002741 |
| 1 | 5789 | " $\\" | 14.856224 | 0.043851 | 19.997684 | 0.002735 |
| 2 | 24420 | " ____" | 14.007825 | 0.018773 | 19.994280 | 0.002725 |
| 3 | 5325 | " \\(\\" | 13.945222 | 0.017633 | 19.993912 | 0.002724 |
| 4 | 523 | " $" | 13.598858 | 0.012471 | 19.991482 | 0.002718 |

- Top-1 falls 0.711187 to 0.002741 (259x). Identical ids and order.
- Baseline ranges: finished [-12.7153, 17.6424], pre-softcap
  [-15.0206, 27.7047], raw [-76.5904, 141.2668], top-100 beyond
  0.9 * cap: 0. Forced: finished [-19.9811, 20.0000] (max 19.999969
  before print rounding), top-100 beyond: 100 of 100.
- The forced arm ran prefill-only (n=1); greedy identity with the
  baseline's 16 tokens (listed in the run record) follows from the
  monotonicity argument and the golden prompt's full empirical
  confirmation.

## Headline summary, all three prompts

| Prompt | Base top-1 prob | Forced top-1 prob | Factor | Top-5 ids/order | Top-100 beyond 0.9 cap (base to forced) |
| --- | --- | --- | --- | --- | --- |
| France | 0.403462 | 0.000968 | 417x down | identical | 0 to 100 |
| Japan | 0.635992 | 0.001504 | 423x down | identical | 0 to 100 |
| Water | 0.711187 | 0.002741 | 259x down | identical | 0 to 100 |

Greedy tokens across 16 steps: identical between arms on the
golden prompt (measured in full), and invariant by construction on
every prompt (next section).

## Analysis: the residue question, answered explicitly

1. **Could a scale error produce a probability-only residue with
   identical tokens? Yes, in principle, and by construction.** Both
   finish steps (multiply by a positive scale, then
   `cap * tanh(z / cap)`) are strictly increasing functions of each
   logit. Any error confined to the finish (wrong scale, wrong cap,
   swapped order, different precision) remaps every probability and
   can never change an argmax or a ranking, at any position, ever.
   The T-9.3 residue's shape (identical tokens, top-1 prob off by
   0.0073) is exactly the fingerprint of a finishing-stage
   difference. It is not the fingerprint of a body-numerics error,
   which would eventually flip some argmax somewhere. This run
   demonstrates the extreme case empirically: a 5.0990x scale error
   left all 16 greedy tokens and every top-5 ranking untouched on
   the golden prompt while moving the top-1 probability by 0.4025.

2. **Does forcing 1.0 move our number toward llama.cpp's 0.3962?
   No. It moves away, catastrophically, and overshoots.** Forced
   top-1 is 0.000968 against llama.cpp's 0.3962: the forced move is
   -0.4025, about 55 times the 0.0073 residue, in the downward
   direction. So the residue is NOT "the scale is effectively 1.0"
   or any error of that size. Whatever separates us from llama.cpp
   at the finish is percent-level at most.

3. **Why the effect reverses sign: the softcap.** In the unsaturated
   regime a larger scale sharpens softmax (top-1 prob rises). But
   Glimmer's raw logits are huge (range about [-80, +135] on the
   golden prompt): at scale 1.0 the leaders land at tanh arguments
   of 5 to 6.7, where tanh is flat, so the cap crushes their
   differences to under 0.0025 of logit and the head mass smears
   across roughly a thousand near-tied tokens. p1(scale) is
   therefore non-monotone: it rises in the mild regime and collapses
   in the saturated regime. A 0.0073 residue in the upward direction
   (ours is HIGHER than llama.cpp's) is consistent with a slightly
   LARGER effective scale in the mild regime (for example a
   precision difference in how the scale multiply is evaluated),
   but the 5.1x endpoint cannot be linearly extrapolated to size
   it; a dedicated small-perturbation sweep (for example scale
   factors 0.95x to 1.05x of the file value) is the instrument for
   that, and this experiment deliberately did not change the ship
   default to find out.

4. **What the scale is actually doing at the file value.** At
   0.19611613 the scale pulls the raw range [-80, +135] down to
   [-15.8, +26.4] before the cap, so the softcap operates mostly in
   its gentle region (only the top few logits pass tanh argument
   1.0) and the distribution keeps a decisive head (top-1 0.40).
   The scale is the thing that makes the softcap a soft cap instead
   of a wall: at 1.0 the cap becomes a wall and the model's
   preferences among its top thousand tokens are effectively
   erased.

## Run record

- Baseline golden: prefill 1,497,849 ms (6 tokens), 16 tokens in
  2,211 s, rc=0. Forced golden: prefill 1,464,484 ms, 2,186 s, rc=0.
- Baseline japan: prefill 1,468,680 ms, 2,191 s, rc=0. Forced japan
  (n=1): prefill 1,447,180 ms, 1,470 s, rc=0.
- Baseline water: prefill 1,656,259 ms (7 tokens), 2,384 s, rc=0.
  Baseline 16 greedy tokens: [465, 30, 59, 26, 2473, 2406, 290,
  220, 30, 310, 465, 30, 59, 3641, 1044, 45]. Forced water (n=1):
  prefill 1,651,736 ms, rc=0.
- Top-up runs (forced japan n=16, forced water n=16) were queued
  after the core matrix in the same session to confirm greedy
  identity empirically on the remaining two prompts; their logs
  land in `~/overnight-logitscale/results/` on the server. The core
  verdict does not depend on them: greedy identity is proven
  mathematically for any monotone finish and was confirmed in full
  on the golden prompt.
- Server session: build 36 s (warm staging cache), runs serialized
  under `/tmp/itak-phase.lock` behind the other overnight lanes;
  the queue wait (about 8 h across three queued sessions) is
  recorded here because it dominated the lane's wall clock.
- Production Glimmer was never started; ports 8080, 8091, 8092 were
  never touched.
