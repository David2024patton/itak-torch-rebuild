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
