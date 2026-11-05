# Overnight engine proof: itak-engine on the real Glimmer file, 2026-11-05

Fresh end-to-end run of the `itak-engine` binary itself (no Go
platform in the loop) on current main, because the engine has
changed since the 2026-10-30 serving smoke and the T-9.17 proof:
the diag module now routes every log path and refusal, worker
panics are contained as `engine_died`, the hello handshake is
enforced on every connection, and frame sizes are capped. The
engine loaded the real Glimmer GGUF on ubuntu-server and
generated over its own NDJSON-over-Unix-socket protocol, driven
by a raw Python socket client written for this run. Prior art:
bench/serving-smoke-2026-10-30.md, bench/t917-proof-2026-10-30.md.

- Code: commit `bda7ff55ea57f4a764afddf9ba2e050c59b86916`
  (branch `lane/overnight-engine`). The worktree was staged to
  `~/overnight-engine` on the server and all 290 staged files
  were verified against a sha256 manifest taken from the
  worktree before the build (`sha256sum -c`, 290 OK).
- Engine binary: `itak-engine 0.1.0`, protocol 1, release build,
  sha256 `5028b5095abf729486d762c312e4c15faf76bb655e3654d1447b10b553bc0c93`.
  (The build step itself took 4 s against the server's warm
  cargo caches; the staged sources are the hash-verified tree
  above and the binary's behavior below is current-main code:
  handshake enforcement, diag-channel logging, `--log-content`
  prompt id lines.)
- Model: `/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`
  (16,756,683,904 bytes).
- Run shape: one throwaway engine process, socket
  `~/overnight-engine/run/engine.sock`, `--model-name glimmer`,
  `--log-content` (so the engine log carries the prompt ids and
  completion texts recorded below), slots 1, max_queued 8.
  `CARGO_BUILD_JOBS=4`, `MALLOC_ARENA_MAX=2`,
  `MALLOC_TRIM_THRESHOLD_=0`. All server work held the shared
  phase lock (`flock -w 14400 /tmp/itak-phase.lock`); see the
  queue note at the end.
- Safety: the production Glimmer llama-server stayed DOWN
  throughout (never started by this run); the Qwen3-Embedding
  tenants on 8091/8092 were never touched; nothing bound
  localhost:8080. The llama.cpp cross-check at the end ran on
  127.0.0.1:18297 with `-ngl 0`, only after the engine process
  had exited, and was killed immediately after. No leftover
  processes at the end; the engine log contains zero panics.
- Raw artifacts (per-leg transcripts with per-frame arrival
  times, engine log, RSS samples, progress log) live in the
  server staging tree `~/overnight-engine` (`legs/`,
  `engine.log`, `rss.log`, `progress.log`).
- All times UTC, 2026-11-05.

## Load and coverage

Engine started 08:35:26Z; serving 15 s later.

```text
[itak-engine] loading model /mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf (this takes seconds: parse, coverage gate, weights)
[itak-engine] model loaded in 11279 ms: name=glimmer arch=muse-glimmer layers=52 hidden=6656 vocab=202048 context=131072 coverage=731/731
[itak-engine] serving on /home/server/overnight-engine/run/engine.sock (protocol 1)
[itak-engine] concurrency: slots=1 max_queued=8 (submissions past capacity refuse as engine_busy)
```

The `debug` op before any generation: model `glimmer`,
architecture `muse-glimmer`, layers 52, hidden 6656, vocab
202048, context_length 131072, **coverage_expected 731,
coverage_found 731**, has_chat_template true, generations 0.
The same snapshot after all legs: coverage still 731/731,
generations 4, tokens_generated 26 (16 + 4 + 4 + 2, matching
the legs below), uptime_secs 13491.

Memory: RSS sampled every 60 s for the engine's whole life
(225 samples). Footprint right after load 17,539,012 KB;
session peak **21,755,068 KB** (during the seeded legs).

Two loader notices ride the gguf diag channel at load, both
non-fatal (the coverage gate passed and every generation below
matches its external reference exactly, so no behavior diverged;
they are recorded here because a proof run reports what the
diagnostics said):

```text
[gguf][error] event=metadata_type_mismatch key=muse-glimmer.attention.sliding_window_pattern expected=string found=array
[gguf][error] event=missing_metadata key=muse-glimmer.rope.freq_base_swa
```

## Leg A: raw completion, temperature 0, 16 tokens: PASS

Frame: `{"op":"generate","model":"glimmer","prompt":"The capital of France is","temperature":0,"max_tokens":16}`

Engine-side prompt ids: `[200000, 954, 7963, 323, 11698, 373]`,
the stage-1 golden prompt. Generated ids:

```text
[13796, 26, 1573, 373, 290, 2167, 26899, 1212, 6996, 310, 11698, 341, 290, 2167, 26899, 1212]
```

Text: `" Paris. It is the most populous city in France and the most populous"`,
finish_reason `length`. The first 8 ids are exactly the
recorded identity golden `[13796, 26, 1573, 373, 290, 2167,
26899, 1212]`, and the text begins with `" Paris"` as required.
Timings: first token at 1,486.9 s (the 6-token prefill), total
2,205.7 s (engine `elapsed_ms=2205730`); the 15 decode steps
average about 47.9 s each.

## Leg C: seeded sampled request, run twice: PASS

Frame: `{"op":"generate","model":"glimmer","prompt":"The capital of France is","temperature":0.8,"top_p":0.95,"top_k":40,"seed":42,"max_tokens":4}`,
sent twice on separate connections.

- Run 1: ids `[13796, 24, 290, 392]`, text `" Paris, the \""`,
  finish `length`, first token 1,492.9 s, total 1,635.3 s.
- Run 2: ids `[13796, 24, 290, 392]`, text `" Paris, the \""`,
  finish `length`, first token 1,494.5 s, total 1,635.3 s.

The two streams are token-identical, and they also reproduce
the exact stream the T-9.17 proof recorded for this frame in
October (`[13796, 24, 290, 392]`): seeded sampling parity
holds on current main, across the sampler work since.

## Leg D: refusals: PASS

- Unknown op, after a clean hello on the same connection:
  `{"op":"error","code":"unknown_op","message":"unknown op \"bogus_op\""}`.
  A `ping` on the same connection afterwards answered
  `{"op":"pong","detail":"model=glimmer busy=false"}`: the
  refusal is named, and neither the connection nor the engine
  died.
- Malformed frame (the literal line `this is not json`):
  `{"op":"error","code":"bad_frame","message":"invalid literal at byte 0"}`.
  A hello on the same connection afterwards got a clean
  `hello_ack`.

## Leg B: chat-templated request: PASS

The engine served the model's own template over the
`chat_template` op: 9,992 chars (the full ATEM template, same
length the T-9.17 note records), sha256
`cfc67e5f349f37690dfd31ed1f18bc4442a9dd32fe39a648f993cb4eb3cae678`,
bos_token `<|begin_of_text|>`, eos_token `<|end_of_text|>`.

Messages: system `Be terse.`, user `Capital of France? One word.`
The prompt sent is the repo renderer's unit golden for exactly
these messages (`platform/internal/chattemplate`,
`TestGlimmerATEMTemplateWithSystem`):

```text
<|begin_of_text|><|start|>system<|message|>Be terse.\n\nReasoning strength: high.\n\n# Valid recipients: "self", "user".<|eot|><|start|>user<|message|>Capital of France? One word.<|eot|><|start|>assistant
```

Frame: generate with that prompt, temperature 0, max_tokens 2.
Engine-side prompt ids (38):

```text
[200000, 200022, 15651, 200023, 3654, 260, 6201, 1574, 34956,
300, 9762, 38, 2244, 1574, 15, 14757, 73965, 38, 392, 2540,
706, 392, 1556, 4205, 200008, 200022, 1556, 200023, 65021,
323, 11698, 43, 5615, 2918, 26, 200008, 200022, 140680]
```

Completion: ids `[328, 19669]`, text `" to=self"`, finish
`length`: the ATEM recipient-header opening, the same 2-token
completion the T-9.17 two-slot run recorded for this question.
Timings: first token 7,964.2 s (the 38-token prefill, about
210 s per prompt token in the batch forward), total 8,011.4 s.

Cross-check, run after the engine exited, the T-9.17 way: a
throwaway CPU llama-server (`~/llama.cpp/build-vulkan2`,
`-ngl 0`, 127.0.0.1:18297) rendered the same two messages
through `/apply-template` and tokenized the result through
`/tokenize` with `add_special=true`:

- Rendered string: llama.cpp's string equals the sent prompt
  minus its leading `<|begin_of_text|>` (llama adds the BOS at
  tokenize time; ours rides in the text and the engine's
  assembler keeps exactly one). **String match: true.**
- Token ids: llama.cpp's 38 ids equal the engine's 38 logged
  prompt ids, id for id. **Ids match: true.**

## Speed, stated plainly

Same class as October, as expected: this binary's forward is
the CPU reference path. Prefill about 248 s per prompt token
at 6 tokens and about 210 s per token at 38 (one batch forward
per prompt), decode about 48 s per step. The engine remains a
correctness oracle; nothing in this run claims serving speed.

## Queue note

The first lock waiter (launched 04:15Z) sat behind the other
overnight lanes for the full 4 h `flock -w 14400` cap and
expired without ever holding the lock; no server work ran
under it. It was relaunched unchanged at 08:18Z, acquired the
lock at 08:35Z, and the run above is that second waiter's.
Total queue wait about 4 h 20 min across the two waiters.

## Verdicts

- Load and coverage (731/731 at load and in both debug
  snapshots): **PASS.**
- Leg A, raw greedy completion, 16 tokens, golden first 8 ids
  and `" Paris"` opening: **PASS.**
- Leg B, chat-templated request, engine prompt ids equal to
  llama.cpp's templated ids 38 for 38, strings equal modulo
  the one BOS: **PASS.**
- Leg C, seeded sampled request twice, token-identical
  streams, also identical to the T-9.17 record: **PASS.**
- Leg D, unknown op and malformed frame refused by name
  (`unknown_op`, `bad_frame`), engine and connection alive
  after each: **PASS.**
- No defect was exposed by the run, so no code changed. The
  two loader metadata notices at load are recorded above for
  the record; every behavioral reference available tonight
  (identity golden, T-9.17 sampled stream, llama.cpp template
  ids) matched exactly.
