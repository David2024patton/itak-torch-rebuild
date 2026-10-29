# Tokenizer conformance: TODO T-2.3 proof

**Result: 16 of 16 bench prompts token-identical to the reference
tokenizer on Glimmer's real vocabulary; 16 of 16 decode round-trips
exact.** Run 2026-10-29 against David's production Glimmer
llama-server. The first run was 14/16 and caught a real
pre-tokenizer defect, documented below; the fix is part of the same
landing.

## Inputs

- **Vocabulary**: extracted from the production model file
  `/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`
  on the server with the gguf crate's `tokenizer_extract` example
  (the file is 15 GiB and the parser loads it whole, so extraction
  runs where the model lives; the resulting `.tokdata` is 9.7 MB).
  The file declares `tokenizer.ggml.model = "gpt2"` and
  `tokenizer.ggml.pre = "llama4"`; 202,048 tokens, 439,802 merges,
  bos 200000, eos 200001, pad 200018, no unknown token id.
- **Reference tokenizations**: `conformance/make_reference.py`, run
  on the server against the production llama-server's `/tokenize`
  endpoint with `add_special: false` (pure tokenization, no BOS/EOS
  injection, matching `Tokenizer::encode`). The generated reference
  is committed as `conformance/glimmer.tokref`. The server was only
  ever read through its normal API; it was not restarted or
  reconfigured.
- **Comparison**: `cargo run --release -p tokenizer --example
  conformance -- <glimmer.tokdata> conformance/glimmer.tokref`. The
  example adapts `gguf::TokenizerInfo` into `TokenizerData` (the
  gguf crate is a dev-dependency of the harness only; the library
  stays decoupled), encodes every prompt, compares id sequences
  exactly, and decodes the reference ids back to the prompt text.

## The prompt set

Sixteen prompts chosen to stress the known divergence edges:

| # | Stresses | Tokens | Result |
| --- | --- | --- | --- |
| 0 | plain prose, single spaces | 17 | exact |
| 1 | double spaces | 15 | exact |
| 2 | punctuation clusters, quotes, ellipsis, em dash | 18 | exact |
| 3 | a 20-digit run | 16 | exact |
| 4 | decimals, date, time, leading zeros | 27 | exact |
| 5 | contractions, incl. double `you'd've` | 30 | exact (was the failure, below) |
| 6 | accented Latin (French, Spanish) | 27 | exact |
| 7 | CJK sentences (Japanese, Chinese) | 17 | exact |
| 8 | Latin and CJK mixed | 16 | exact |
| 9 | a Rust code snippet | 35 | exact |
| 10 | newline runs, space indent, tab, trailing newline | 16 | exact |
| 11 | leading and trailing spaces | 7 | exact |
| 12 | leading blank lines | 7 | exact |
| 13 | currency, symbols, arrow, emoji | 29 | exact |
| 14 | camelCase, snake_case, dunder identifiers | 18 | exact |
| 15 | leading apostrophe word, `90's`, double contraction | 20 | exact |

## The one mismatch, and the fix

First run: case 5 diverged at token index 1.

```text
prompt:    "I don't know if we'll make it, ..."
reference: [53, 5863, 2609, 563, 31519, 2170, 536, 24, 1108, 28661, ...]
ours:      [53, 2360, 2050, 2609, 563, 675, 8961, 2170, 536, 24, ...]
```

Reference token 5863 is `Ġdon't`, a single token. Ours produced
2360 `Ġdon` + 2050 `'t`, and the same split repeated at every
contraction in the prompt (`we'll`, `they're`, `isn't`, `we'd`,
`she's`). Cause class: **pre-tokenizer split**. This crate had
implemented the llama-3 pattern, in which the contraction is a
standalone first alternative, so `don't` was split before BPE and
the merged token was unreachable.

The reference build's own source settles what the right pattern
is: in `llama-vocab.cpp`, `tokenizer_pre == "llama4"` maps to
`LLAMA_VOCAB_PRE_TYPE_GPT4O`, whose pattern carries the contraction
as an optional suffix of the word alternatives
(`(?:'s|'t|'re|'ve|'m|'ll|'d)?`). There is no standalone
contraction alternative; a contraction that follows no word (the
`'ve` of `you'd've`, the `'s` of `90's`) is produced by the
prefix-char word branch.

Fix, in `src/pretokenize.rs`: the standalone contraction branch is
gone, and every word piece absorbs one trailing contraction
(`word_end`). The crate docs in `src/lib.rs` now state the GPT-4o
pattern instead of the llama-3 one. Regression tests cover
`don't`, `you'd've`, `'tis`, `90's`, `dogs'`, and `DON'T`, and
case 15 of the bench set exercises the same shapes against the
reference. After the fix: 16/16 exact, and all pre-existing crate
tests still pass.

## What this does not prove

- The prompt set contains no literal special-token text (for
  example `<|im_start|>` typed as prose), and the comparison ran
  with `add_special: false`; special-token matching and BOS/EOS
  policy are covered by the crate's unit tests, not by this run.
- The documented scanner deviations in `src/lib.rs` stand:
  `\p{L}` is `char::is_alphabetic` (the pattern's `\p{M}` mark
  class is not modeled separately), and a whitespace run mixing
  newlines and spaces can split at a different point than a true
  regex engine in rare multi-newline cases. No case in this set
  exercised either gap into a divergence.
