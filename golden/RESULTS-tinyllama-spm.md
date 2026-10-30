# TinyLlama re-run with the SentencePiece tokenizer family

Follow-up to `RESULTS-integration-a.md`, whose TinyLlama half was
blocked on a tokenizer family gap: the file's vocabulary is
SentencePiece BPE and the tokenizer crate implemented only GPT-2
byte-level BPE, so every prompt diverged at prompt-token index 1
and the run said nothing about the forward.

The SentencePiece family is now implemented in the tokenizer crate
(`crates/tokenizer/src/spm.rs`, a port of llama.cpp's SPM path with
rules cited to `llama-vocab.cpp`; family selected from the file's
own `tokenizer.ggml.model`, scores plumbed from
`tokenizer.ggml.scores` through the gguf crate and the `.tokdata`
format v2). Tokenizer-level proof is
`crates/tokenizer/tests/spm_golden.rs`: all 15 golden prompt id
sequences reproduced exactly, all 15 fixture completion texts
reproduced by decode. See `crates/tokenizer/CONFORMANCE.md`.

## Generation re-run

Same conditions as Run A: David's ubuntu-server, release build,
RUSTFLAGS target-cpu=x86-64-v3, CPU only, runner
`crates/arch-llama/examples/golden_run.rs`, greedy, 64 tokens max.
Production Glimmer llama-server untouched (verified still on its
original pid 683675 after the run). Wall time 787.8 s for the full
15 prompts. The runner log confirms the family was selected from
the file: `tokenizer: model=llama pre=<absent> vocab=32000`.

Prompt tokenization now matches the fixture on all 15 prompts
(candidate prompt lengths 6, 7, 10, 9, 10, 19, 24, 16, 18, 15, 19,
16, 15, 17, 10; the Run A candidate had ballooned to 11..41 ids).

## Headline

| Model | Token-identical | Text-identical | Divergent | Missing |
| --- | --- | --- | --- | --- |
| TinyLlama-1.1B Q4_K_M | 6 / 15 | 0 / 15 | 9 | 0 |

Token-identical over the full 64-token greedy trajectory:
factual_capital, code_python, list_colors, list_days,
list_counting, accented_german.

First divergent completion token for the rest:

| Prompt | Index | Golden id | Candidate id |
| --- | --- | --- | --- |
| factual_science | 29 | 289 | 564 |
| factual_history | 31 | 4331 | 6686 |
| arithmetic_add | 4 | 29899 | 13 |
| arithmetic_divide | 39 | 29953 | 29908 |
| arithmetic_word | 23 | 16492 | 13 |
| code_go | 0 | 29896 | 1202 |
| accented_french | 34 | 2123 | 29892 |
| accented_spanish | 17 | 4434 | 2594 |
| continuation_classic | 23 | 1576 | 29899 |

The divergences are the forward's, not the tokenizer's: prompts
are exact, completions are coherent continuations that track the
golden trajectory for tens of tokens before forking (the same
signature Run A measured on Qwen2.5, where a probe feeding fixture
ids directly localized it to the forward's attention/residual
pathway). Diagnosing the llama forward from here is the
layer-by-layer activation diffing follow-up named in
`RESULTS-integration-a.md`, not this task.

## Files

- Candidate: `golden/candidates/tinyllama-1.1b-chat-v1.0.Q4_K_M.candidate.jsonl`
  (replaces the Run A candidate, which was generated on byte-piece
  context and was uninformative about the forward)
- Diff: `python3 golden/compare.py golden/fixtures/tinyllama-1.1b-chat-v1.0.Q4_K_M.golden.jsonl golden/candidates/tinyllama-1.1b-chat-v1.0.Q4_K_M.candidate.jsonl --strict`
  exits 1 with the summary above.

## Follow-up: re-run after the RoPE pairing fix (13643e0)

The Qwen layer diff (`RESULTS-layer-diff-qwen.md`) found the
dense forward's RoPE pairing was adjacent where llama.cpp applies
the half-split NEOX pairing for qwen2, and commit 13643e0 changed
the shared `torch_math::rope` / `rope_partial` to NEOX. Qwen went
0/15 to 9/15 on that change. TinyLlama (arch `llama`, the same
arch-llama forward) was re-run under identical conditions to the
SPM run above (server, release, RUSTFLAGS target-cpu=x86-64-v3,
CPU only, 701.3 s; production Glimmer llama-server untouched,
verified healthy on pid 683675 after the run).

### Headline: a regression for the llama arch, 6/15 to 0/15

| Model | Token-identical | Text-identical | Divergent | Missing |
| --- | --- | --- | --- | --- |
| TinyLlama-1.1B Q4_K_M (adjacent pairing, SPM run) | 6 / 15 | 0 / 15 | 9 | 0 |
| TinyLlama-1.1B Q4_K_M (after 13643e0, NEOX) | 0 / 15 | 0 / 15 | 15 | 0 |

Prompt tokenization is still exact on all 15 prompts; the
completions are coherent English that leaves the golden trajectory
immediately (eight forks at index 0). Candidate:
`golden/candidates/tinyllama-1.1b-chat-v1.0.Q4_K_M.fixed.candidate.jsonl`.

### Fork table with llama.cpp's own margins

Margins were measured by feeding each fork's golden prefix
(prompt ids + golden completion ids before the fork) to a
throwaway CPU llama-server (same binary and flags as
`gen_golden.py`) with `n_probs=5` and reading the winner's
logprob gap over the best other token at that position
(`golden/tools/fork_margins.py`). The band column compares
against the cross-engine noise band the Qwen note measured
(up to ~0.78 logits); that band was measured on Qwen2.5-1.5B,
not on this model, so it is a reference scale, not a
TinyLlama-measured bound.

| Prompt | Fork index | Golden id | Candidate id | llama.cpp winner margin (logits) | vs 0.78 band | Candidate in llama.cpp top-5 |
| --- | --- | --- | --- | --- | --- | --- |
| factual_capital | 0 | 3681 | 278 | 0.519 | within | no |
| factual_science | 1 | 29889 | 29906 | 0.586 | within | yes (gap 1.493) |
| factual_history | 0 | 29334 | 263 | 2.041 | exceeds | no |
| arithmetic_add | 1 | 29946 | 29896 | 0.925 | exceeds | no |
| arithmetic_divide | 2 | 29946 | 29900 | 0.453 | within | no |
| arithmetic_word | 1 | 29953 | 29896 | 0.416 | within | no |
| code_python | 0 | 18755 | 302 | 3.137 | exceeds | yes (gap 3.137) |
| code_go | 7 | 13 | 28956 | 5.190 | exceeds | no |
| list_colors | 0 | 301 | 322 | 1.212 | exceeds | no |
| list_days | 1 | 24211 | 278 | 0.466 | within | yes (gap 4.223) |
| list_counting | 4 | 29947 | 29896 | 7.403 | exceeds | yes (gap 7.403) |
| accented_french | 0 | 10814 | 454 | 0.532 | within | no |
| accented_german | 0 | 6209 | 762 | 0.864 | exceeds | no |
| accented_spanish | 0 | 1208 | 274 | 0.602 | within | no |
| continuation_classic | 0 | 17366 | 18786 | 4.177 | exceeds | no |

Eight forks exceed the band outright. At the seven within-band
forks the candidate token still sits outside llama.cpp's top-5
in six cases, and 1.49 to 4.22 logits behind the winner in the
rest: the engine did not pick a near-tie runner-up, it picked
tokens llama.cpp ranks far down. The forks are a systematic
error, not cross-engine noise, which the root cause below
confirms directly.

### Root cause: RoPE pairing is per-architecture; 13643e0 generalized qwen2's

llama.cpp's own source settles the pairing per architecture
(`llama_model_rope_type` in `src/llama-model.cpp`, the build
that generated the fixtures): `LLM_ARCH_LLAMA` returns
`LLAMA_ROPE_TYPE_NORM` (adjacent pairs), `LLM_ARCH_QWEN2`
returns `LLAMA_ROPE_TYPE_NEOX` (half-split), and
`LLM_ARCH_MUSE_GLIMMER` returns NORM. The 13643e0 commit
message asserts NEOX for "qwen2/llama"; the llama half of that
claim is wrong. The commit changed the one shared pairing
function arch-llama calls for both architectures, so it fixed
qwen2 and broke llama in the same edit.

Two cross-checks isolate the cause to the pairing, not to this
re-run's build or staging:

- The layer-diff agent's own post-fix `golden_run` binary,
  run on TinyLlama for the single prompt factual_capital,
  produces the identical wrong trajectory head
  ([278, 4272, 310, 3681, ...], " the city of Paris...") as
  this re-run's candidate.
- A discriminator experiment in the server staging copy only
  (never committed): the arch-llama call site in `rope_rows`
  was temporarily switched to `torch_math::rope_interleaved`
  for the full-rotary case (this file's rope_dim equals its
  head_dim), and factual_capital returned to fully
  token-identical over all 64 tokens, exactly as in the SPM
  run above.

The fix this names, not landed here: arch-llama must select
the pairing from the file's architecture at the call site,
the same structural rule the Glimmer backend already follows
(`rope_interleaved`'s module docs): `llama` takes the
interleaved pairing (a partial-rotary variant is needed for
llama-arch files whose rope_dim is below the head dim),
`qwen2` keeps the NEOX `rope_partial` from 13643e0. Until the
pairing is per-architecture, the two golden fixtures guard
opposite directions and either arch can regress silently
behind the other's fix.

### Files (this follow-up)

- Candidate: `golden/candidates/tinyllama-1.1b-chat-v1.0.Q4_K_M.fixed.candidate.jsonl`
- Margin probe: `golden/tools/fork_margins.py` (one-off
  analysis; margins above are its output, reproducible from
  the fixture and candidate)
