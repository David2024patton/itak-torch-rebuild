# Golden-output harness (TODO T-2.9)

The rebuild's engines do not get to grade their own homework. This
directory holds reference generations produced by llama.cpp (the
trusted path) on a fixed prompt set at temperature 0, stored as test
data. "Verified text" (T-9.4 and friends) means: an engine's output
on these prompts is diffed against these fixtures by `compare.py`,
mechanically, not eyeballed.

Nothing here diffs against the original Go native engine. llama.cpp
is the reference.

## Layout

- `prompts.jsonl` — the prompt set, version 1.1: 19 short raw
  completion prompts (factual, arithmetic, code, list continuation,
  accented French/German/Spanish text, plus four added 2026-10-29
  for the forward divergence hunt: a strong-factual first-token
  probe, a forced long decode, a number-formatting prompt, and a
  contractions prompt). Raw completions, not chat turns: the engine
  forward pass is what is being verified, so no chat template is
  applied anywhere in this harness. (The long-form Glimmer accuracy
  prompts live separately in `bench/accuracy-prompts.md` and are a
  different check.)
- `gen_golden.py` — the generator. Runs llama.cpp on the host that
  has the binaries and the model, writes one fixture per model.
- `fixtures/*.golden.jsonl` — the golden records. Currently bound
  to prompt set v1.0 (15 prompts) by SHA-256 in each fixture's
  meta; the four v1.1 prompts have no reference yet, see
  `PENDING-REFERENCES.md` for why and for the exact capture
  procedure.
- `compare.py` — the differ. Any engine that can emit the record
  shape below can be checked against a fixture.
- `determinism_check.py` — proves the reference itself is stable:
  two temperature-0 runs of each prompt against a throwaway
  llama-server must produce byte-identical token ids. Run it
  before trusting any fresh capture.
- `AUDIT-2026-10-29.md` — the v1.1 audit: what the fixtures do and
  do not contain (notably: no per-step probabilities or top-k
  anywhere; completions are 64-token budget truncations).

## Fixture format

First line is metadata: model file name, SHA-256 and size, the exact
llama.cpp version, the binaries and flags used, and the prompt set's
SHA-256 (so a fixture can never silently drift from its prompts).
Then one record per prompt:

```json
{"id": "factual_capital", "category": "factual",
 "prompt": "The capital of France is",
 "prompt_token_ids": [1, 450, 7483, 310, 3444, 338],
 "completion_token_ids": [3681, 29889, "..."],
 "completion_text": " Paris.\n\n2. B.C. ...",
 "ended_with_eos": false}
```

`completion_token_ids` are the ids llama.cpp actually sampled,
including the terminal EOS id when generation stops on EOS (its text
piece is empty; see the Qwen2.5 `arithmetic_add` record, 30 tokens,
last id 151643). `ended_with_eos` records that stop.

## Why generation runs through llama-server

The sampled token ids are the point of the fixture, and
`llama-completion` / `llama-cli` cannot report them: no flag exists,
and `--verbose` dumps graph internals instead (verified against
llama.cpp build a7af149). Retokenizing the completion text cannot
recover them either, because tokenization is context-dependent:
`llama-tokenize` on a completion text standalone splits its first
token three ways (`[259, 2177, 275]`) where the in-context
tokenization of the same text has a single token (`[3681]`).

So `gen_golden.py` starts a throwaway `llama-server` on CPU
(`-ngl 0`, localhost, port 18923 by default) and calls `/completion`
with `n_probs=1`, which returns per step the sampled token's id, text
piece and raw bytes. Two assertions guard every record: the server's
evaluated prompt-token count must equal the `llama-tokenize --ids`
count, and the kept steps' bytes must concatenate to the response
text exactly. Ids and text in a fixture therefore provably come from
one and the same generation. The production Glimmer server is never
touched: different port, CPU only, small teardown models.

Decoding parameters, also recorded in each fixture's metadata:
temperature 0, seed 42 (recorded for completeness; the greedy sampler
does not use it), 64 predicted tokens, context 512, 8 threads, no
chat template. Fixtures contain no timestamps; regenerating with the
same llama.cpp build reproduces a fixture byte for byte (verified
for TinyLlama; `determinism_check.py` makes the underlying run-to-run
claim checkable on any model).

## Tooling choice

Both scripts are python3, standard library only. The generator's job
is HTTP plus JSON plus byte-exact consistency checks; bash would need
jq gymnastics for the UTF-8 byte validation, and a compiled tool
would drag in a JSON stack to produce test data. The scripts run
unchanged on the VM and on the server.

## Regenerating

On the server (or any host with the llama.cpp build and the models):

```sh
python3 gen_golden.py --model /mnt/models/teardown/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
python3 gen_golden.py --model /path/to/qwen2.5-1.5b-q4km.gguf
```

`--bin-dir` defaults to `/home/server/llama.cpp/build-vulkan2/bin`;
`--out` defaults to `fixtures/<model-file-stem>.golden.jsonl`;
`--port` picks the throwaway server's port.

## Comparing an engine against a fixture

The engine (or an adapter around it) writes a JSONL file with the
same record fields; the metadata line is optional in candidates.

```sh
python3 compare.py fixtures/tinyllama-1.1b-chat-v1.0.Q4_K_M.golden.jsonl candidate.jsonl
```

Per prompt the verdict is `token-identical`, `text-identical` (ids
differ, text matches; the first token divergence is reported), or
`divergent` (first divergence reported in token space and in text
space), plus `missing` for prompts the candidate did not answer.
Exit code 0 means no divergence and nothing missing. `--strict`
additionally fails `text-identical`: verified text at temperature 0
means token-identical.

Self-test results against the TinyLlama fixture (all as expected):
identity run 15/15 token-identical, exit 0; a candidate whose
`factual_capital` ids were replaced by the standalone retokenization
of the same text (66 ids vs 64) reports text-identical with first
divergence at index 0, exit 0, and exit 1 under `--strict`; a
candidate with one flipped token id (index 5) and one flipped text
character reports divergent at exactly those positions, exit 1; a
candidate missing a record reports it missing, exit 1.
