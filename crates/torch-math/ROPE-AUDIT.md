# RoPE / attention audit against llama.cpp (2026-10-29)

Lane 03, Integration Run B follow-up. Question: is the position /
attention math in the new Rust forward a suspect for the first-token
divergence Run B measured (`crates/itak-loader/INTEGRATION-RUN-B.md`)?
Scope: RoPE theta source, dimension pairing, position offset at
decode; attention scale, GQA head mapping, sliding-window vs global
treatment. Each convention is pinned by tests whose expected values
are computed independently of the code under test (hand literals or
an in-test f64 brute force).

## Finding 1 (FIXED): RoPE used the wrong pairing convention

`torch_math::rope` implements ggml's `GGML_ROPE_TYPE_NORMAL`:
interleaved adjacent pairs `(x[2i], x[2i + 1])`. Its doc comment
claimed this was "the GPT-NeoX pairing llama.cpp applies". That is
exactly backwards:

- ggml `GGML_ROPE_TYPE_NEOX` (value 2) pairs split-half:
  `(x[i], x[i + rotary_dim / 2])`, same frequency ladder
  `position * base^(-2i / rotary_dim)`.
- llama.cpp assigns NEOX to the HF-derived families (llama, qwen2,
  gemma, ...), whose GGUF Q/K weights are stored unpermuted; NORM is
  the variant whose files had Q/K permuted at conversion (llama.cpp
  arch PRs state the pairing explicitly, e.g. "use NEOX RoPE type;
  remove Q/K permutation (NEOX RoPE doesn't need it)").
- Muse Glimmer's file is an HF conversion ("Muse Glimmer Hf") and
  llama.cpp's coherent Run B reference output is produced with its
  graph for the arch, which follows that standard recipe.

So the Glimmer forward rotated the wrong channel pairs in all 52
layers: e.g. at head_dim 4, position 1, channel 2 must rotate as
pair 0's second element (to `-sin(1)` against channel 0); the old
code rotated it as pair 1's first element at rate `base^-1/2`.

Fix: `torch_math::rope_neox` / `rope_neox_partial` implement the
split-half convention; `cpu-compute`'s `rope_row` (the Glimmer
backend) now calls `rope_neox`. The interleaved functions stay,
correctly labeled as ggml NORMAL, for any arch whose file is
actually stored that way. The forward test's naive f64 reference
(`rope64` in `crates/cpu-compute/tests/forward_tests.rs`) was
rewritten to the split-half definition, so the end-to-end test pins
the backend against an independent implementation of the same
convention instead of ratifying the old claim.

Consistency note: the pairing error is applied identically to Q and
K, which is why short contexts still produced plausible rankings
(self-attention scores are pairing-invariant when positions are
equal, and near-diagonal scores degrade gracefully); it does not
produce the Run B decode freeze by itself, and it does not displace
Run B's diagnosis item 1 (norm convention). It is one real defect
removed from the suspect list, proven at unit level. The end-to-end
effect is for the next integration run to measure.

## Finding 2 (correct, now pinned): theta source and ladder

- Theta comes from the file: `muse-glimmer.rope.freq_base` is a
  required metadata key in `GlimmerConfig` (missing = named error),
  measured 500000 on the real file, and `cpu-compute` passes
  `cfg.rope_freq_base` through. No hardcoded base anywhere in the
  Glimmer path.
- The ladder is pinned at Glimmer scale in
  `neox_glimmer_scale_frequency_ladder`: head_dim 128, base 500000,
  pair 0 = (ch 0, ch 64) at rate 1, pair 63 = (ch 63, ch 127) at
  rate `500000^(-126/128) = 2.4551408e-6`, checked against literals
  at positions 3 and 1000, with a no-leak check on all other
  channels.
- Dimension count: the file carries no `rope.dimension_count`
  (checked in Run B), so full 128-channel rotation is what the file
  declares and what the forward does. Partial-rotary variants use
  `rotary_dim` as the frequency denominator, matching ggml's
  `n_dims` argument.

## Finding 3 (correct, now pinned): position offset at decode

`cpu-compute` starts each call at `start = cache.position()` and
rotates position `start + i` for both Q and the K being appended;
prefill and decode share the path, and the crate's
incremental-vs-prefill tests already prove they agree, including
across window eviction. The property that makes decode correct,
`<rope(q, n+d), rope(k, n)> == <rope(q, d), rope(k, 0)>`, is now
pinned directly for the NEOX variant in
`neox_dot_product_depends_only_on_relative_position` (checked at
n = 5, d = 4), together with a check that the score actually moves
with absolute position. Conclusion: the RoPE/offset machinery does
not explain the Run B freeze; that symptom stays with Run B's
diagnosis item 2 (attention/KV handling at real scale), not with
position math.

## Finding 4 (correct, already + newly pinned): attention internals

- Scale: `1 / sqrt(head_dim)` by default, a parameter, no
  Glimmer-specific scale key in the file or the config. Matches
  llama.cpp's scaled dot-product default. (Pinned numerically
  before this audit: `scale_for(128) = 0.08838835`.)
- GQA mapping: query head `h` reads KV head `h / (32 / 2)`,
  contiguous groups of 16, llama.cpp's grouping. Pinned
  exhaustively for 32/2 before this audit; this audit adds
  `gqa_grouping_through_softmax_matches_brute_force`, which checks
  all four heads of a 4/2 shape against an independent f64 brute
  force through real softmax weights (the older tests used the
  single-position weight-1 shortcut).
- Sliding window vs global: layer pattern from the file census
  (global at 3, 7, ..., 51; sliding elsewhere) is what the config
  derives, and the mask is `(p - window, p]` (the query sees itself
  and the previous `window - 1` positions), the HF/llama.cpp SWA
  semantics. This audit adds
  `window_edge_membership_matches_brute_force`: window 3, query at
  position 5, visible set exactly {3, 4, 5}, verified against the
  brute force with a marker value that makes an off-by-one leak
  change the output by hundreds.

## Not in this lane (flagged)

`arch-llama` still applies the interleaved `rope_partial` to the
llama/qwen2 families, whose llama.cpp convention is the same NEOX
split-half. Integration Run A's Qwen2.5 divergence (attention /
residual pathway, first fork at token 3, gross RoPE error ruled
out but the step never localized) is consistent with this being a
shared defect. Switching `arch-llama` is a separate lane's call;
the `rope_neox_partial` it needs now exists and is pinned.

## Gates

- `cargo test --workspace`: 180 passed, 0 failed. Per touched
  crate: torch-math 46 (7 new NEOX tests), attention 17 (2 new
  brute-force tests), cpu-compute 16, muse-glimmer 23.
- `cargo clippy --workspace --all-targets`: clean under the
  workspace deny-warnings lints.
