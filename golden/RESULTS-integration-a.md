# Integration Run A: arch-llama vs llama.cpp golden fixtures

Date: 2026-10-29. Engine: crates/arch-llama (Rust CPU reference forward)
with crates/tokenizer fed from each model's own GGUF metadata.
Runner: crates/arch-llama/examples/golden_run.rs. Fixtures: golden/
harness, 15 prompts per model, greedy (temperature 0, seed 42 in the
reference), 64 completion tokens max, EOS ends generation.

Run conditions: David's ubuntu-server, release build,
RUSTFLAGS target-cpu=x86-64-v3, CPU only. The production Glimmer
llama-server on the GPU was not touched (verified still running as
its original pid after both runs). Wall times: TinyLlama 754 s,
Qwen2.5-1.5B 1006 s for the full 15 prompts each. No prompt set was
shrunk; both runs are complete.

## Headlines

| Model | Token-identical | Text-identical | Divergent | Missing |
| --- | --- | --- | --- | --- |
| Qwen2.5-1.5B Q4_K_M | 0 / 15 | 0 / 15 | 15 | 0 |
| TinyLlama-1.1B Q4_K_M | 0 / 15 | 0 / 15 | 15 | 0 |

Both --strict runs give the same verdicts. The two zeros have
completely different causes, detailed below. In short: for Qwen the
tokenizer is proven exact and the divergence is a forward math
discrepancy; for TinyLlama the divergence is a tokenizer family gap
that makes the run uninformative about the forward.

## Qwen2.5-1.5B: tokenization exact, forward diverges

Prompt tokenization matches the fixture on all 15 prompts (compare.py
reports no prompt-id notes, and candidate prompt lengths equal the
fixture's per prompt: 5, 6, 9, 7, 8, 17, 17, 10, 12, 8, 17, 14, 15,
13, 7). The tokenizer crate with the qwen2 pre-tokenizer pattern
reproduces llama.cpp's reference tokenization of this vocab exactly,
including single-digit tokens, accented text, and code.

Completions are coherent, on-topic continuations throughout, and
several track the golden trajectory for many steps before forking
(code_python 11 tokens, list_counting 15, factual_capital 5). Every
prompt eventually forks. First divergent completion token per prompt
(pieces from the model's vocab):

| Prompt | Index | Golden id (piece) | Candidate id (piece) |
| --- | --- | --- | --- |
| factual_capital | 5 | 15344 (" Italy") | 279 (" the") |
| factual_science | 3 | 11 (",") | 13 (".") |
| factual_history | 0 | 33121 (" Neil") | 32671 (" ______") |
| arithmetic_add | 0 | 220 (" ") | 488 (" +") |
| arithmetic_divide | 2 | 17 ("2") | 16 ("1") |
| arithmetic_word | 0 | 510 (":\n") | 220 (" ") |
| code_python | 11 | 77 ("n") | 750 ("def") |
| code_go | 2 | 11009 ("\tfmt") | 92 ("}") |
| list_colors | 4 | 323 (" and") | 1257 (" ind") |
| list_days | 0 | 6602 (" Friday") | 323 (" and") |
| list_counting | 15 | 16 ("1") | 11 (",") |
| accented_french | 0 | 7906 (" au") | 7814 (" bon") |
| accented_german | 0 | 1863 (" Z") | 98886 (" Wien") |
| accented_spanish | 0 | 668 (" ab") | 24249 (" casa") |
| continuation_classic | 2 | 624 (".\n") | 13 (".") |

### Logit-level diagnosis (probe evidence)

A throwaway probe (not committed) fed fixture token ids directly to
both engines, so the tokenizer cannot be the cause of what follows.

Prefix sweep on the arithmetic_add prompt "17 + 25 =" (ids
16 22 488 220 17 20 284), top-1 prediction per prefix length:

| Prefix | llama.cpp top-1 | arch-llama top-1 |
| --- | --- | --- |
| "1" | 24 | 24 (agree) |
| "17" | 13 | 13 (agree) |
| "17 +" | 220 (" ") | 488 (" +", the last prompt token) |
| "17 + " | 16 ("1") | 488 (" +") |
| "17 + 2" | 15 | 15 (agree) |
| "17 + 25" | 284 (" =") | 488 (" +") |
| "17 + 25 =" | 220 (" ") | 488 (" +") |

The same pathology reproduces with "17 -" (ids 16 22 481): llama.cpp
predicts " " (220) first and does not rank 481 in its top 5;
arch-llama's top prediction is 481 itself, the operator just seen.

Measured gaps at three contexts:

- factual_history, prompt only: llama.cpp ranks " Neil" first with
  " ______" 0.207 logprob behind. arch-llama ranks " ______" first
  with " Neil" 0.095 behind. Same top-2 pair, order flipped: a
  near-tie flip, swing about 0.3 logits.
- factual_capital after the 5 shared tokens "...Paris. The capital
  of": llama.cpp ranks " Italy" first, " the" 0.90 behind.
  arch-llama ranks " the" first, " Italy" 2.02 behind. Same
  candidate set near the top; swing 1 to 3 logits.
- arithmetic_add, full prompt: llama.cpp ranks " " first by 2.43
  logprob, with " +" outside its top 10 (more than 5.5 behind).
  arch-llama ranks " +" first, with " " outside its top 10 (more
  than 3.8 behind). Not explainable by accumulation noise.

Diagnosis: cause class is a forward math discrepancy in the
attention/residual pathway. The visible signature is that the engine
over-boosts a salient in-context token (the operator 488 " +" or
481 " -") and predicts its repetition where the reference predicts
a continuation, from the third context token onward, with logit
swings between roughly 0.3 and over 5 depending on context. Ruled
out by the evidence: tokenization (prompt ids identical; the probe
bypasses the tokenizer), EOS handling (both engines' stop behavior
verified separately; the Qwen file declares EOS 151645 while golden
records stop on 151643, and the runner's stop set covers both),
bias handling (Qwen2 biases are exercised in the agreeing contexts
too), and any gross RoPE or config error (short factual contexts
land within 0.3 logits of the reference with the same top
candidates). Not fixed in this run: the black-box evidence does not
isolate the discrepancy to one unambiguous line, and guessing at
attention internals would violate the no-silent-fix rule. Follow-up:
layer-by-layer activation diffing against llama.cpp on the
arithmetic_add prefix (first departing layer localizes the step).

## TinyLlama-1.1B: tokenizer family gap, forward untested

Every prompt diverges at prompt-token index 1 (index 0, the BOS,
matches). The file's vocab is SentencePiece-style BPE: word pieces
carry the "▁" marker (450 "▁The", 29871 "▁") and its 61,249 merges
operate over those pieces. The tokenizer crate implements GPT-2
style byte-level BPE only, so its symbols never match this vocab's
merges and encoding falls back to near-byte pieces: for example
golden 450 ("▁The") vs candidate 1576 ("The") on the factual
prompts, golden 29871 ("▁") vs candidate 29896 ("1") on the
arithmetic prompts. Candidate prompt lengths balloon accordingly
(11 to 41 ids vs the fixture's 6 to 18).

Generation therefore ran on a byte-piece context the model was not
trained on; completions diverge from token 0 on 13 of 15 prompts
and the run says nothing about the arch-llama llama forward, for or
against. Diagnosis: tokenization, family not implemented. The fix
is a second tokenizer family (SentencePiece BPE over the file's own
pieces and merges) in the tokenizer crate, after which this fixture
becomes a real forward test. Not attempted here: a new tokenizer
family is a feature, not an unambiguous small fix.

## Runner fidelity notes

The runner mirrors two llama.cpp behaviors that the fixtures
proved: when tokenizer.ggml.add_bos_token is absent from the file
(the TinyLlama case), BOS follows the family default (llama adds
BOS, gpt2 does not); and the stop set is the declared EOS plus the
<|endoftext|> token when present in the vocab, because the Qwen
fixtures stop on 151643 although the file declares 151645.
Candidate records carry the fixture fields (prompt_token_ids,
completion_token_ids, completion_text) and are written
incrementally, one flushed line per finished prompt.

## Files

- Runner: crates/arch-llama/examples/golden_run.rs
- Candidates: golden/candidates/qwen2.5-1.5b-q4km.candidate.jsonl,
  golden/candidates/tinyllama-1.1b-chat-v1.0.Q4_K_M.candidate.jsonl
- Diffs: python3 golden/compare.py <fixture> <candidate> (and
  --strict); both models exit 1 with the summaries above.
