# Structured Output: JSON mode vs JSON Schema vs few-shot, docs + live experiment (2026-10-28)

Question (David): which technique gets better results per model, plain JSON mode, formal
JSON Schema constraining, or few-shot/template prompting in the prompt? His prior: the Qwen
models do better with template structure. Verdict up front: **he is right, and the mechanism
is documented by the vendors themselves, but the biggest measured effect is not JSON shape
at all. It is reasoning budget starvation, and the few-shot example is what fixes it.**

## Part 1: What the documentation says (researched 2026-10-28, primary sources)

- **Qwen (3.8 / 3.6 / 2.5 family):** JSON Object mode is officially documented and
  contractually requires the word "json" somewhere in the prompt or the hosted API errors
  out. Schema mode is officially documented for "selected models only" on the hosted API;
  locally the engine enforces any schema regardless. Community guidance for Qwen 3.8 27B
  specifically recommends an output contract in the prompt + json_object + few-shot
  examples. (docs.qwencloud.com structured output guide; Alibaba Model Studio docs.)
- **DeepSeek:** the official JSON Output guide requires (1) response_format json_object,
  (2) the word "json" in the prompt, (3) **an example of the desired JSON format in the
  prompt**, (4) a max_tokens high enough not to truncate. Few-shot is the documented
  operating procedure for this family, not an optional extra. No native strict schema mode.
  (api-docs.deepseek.com/guides/json_mode)
- **Mistral / Devstral:** both JSON mode and schema ("custom") structured outputs are
  officially documented; Mistral states schema mode "is more reliable and is recommended
  whenever possible." Devstral is non-thinking, so grammars constrain safely from token
  zero. (docs.mistral.ai structured output pages)
- **gpt-oss:** everything runs through the Harmony format; reasoning lives in the
  `analysis` channel, the answer in `final`, and a message header can carry
  `<|constrain|>json` to switch exactly that span into JSON-constrained sampling. It is
  the cleanest design in the set for structured output: the engine constrains the answer
  channel while reasoning runs free. (OpenAI gpt-oss cookbook)
- **Glimmer:** provider listings confirm JSON-schema structured outputs, but the model
  speaks channel-scoped turns with a native ATEM tool-call format (not JSON tool calls),
  and a documented failure mode is max_tokens clipping mid-thought into empty content.
  (OpenRouter model page; vLLM recipe requires the muse_glimmer parsers.)
- **Gemma 4:** no vendor JSON-mode doc for the open weights (hosted Gemini docs do not
  transfer). What Google documents is trained-in function calling with dedicated tool
  tokens and a `<|"|>` string delimiter, plus a `<|think|>` control token. Local
  enforcement is engine grammar only. (ai.google.dev Gemma docs)
- **Nemotron Cascade 2:** NVIDIA trained structured output in via a dedicated RL stage
  (tool calling + structured output); ChatML template, tool calls wrapped in `<tool_call>`
  tags rather than bare JSON. No model-specific JSON-mode doc found.
- **North Mini Code:** weakest documented position. OpenRouter's route states the model
  "does not support response_format"; the model card recommends tool descriptions as JSON
  schema but documents no JSON mode. Engine grammar is the only enforcement path.
- **Engine reality (llama.cpp):** llama-server converts response_format json_schema into
  a GBNF grammar that masks logits token by token. json_object without a schema applies
  only the generic JSON grammar. The guarantee is **syntax and shape, never value
  correctness**, and a truncated generation (max_tokens) is still unparseable. The
  schema converter silently skips unsupported JSON Schema features. For thinking models
  the grammar path is the hard part: llama.cpp's lazy grammars (which wait for a trigger
  before constraining) exist on the tool-calling path; a plain response_format request
  fights the think block, and reasoning + content share one completion budget.
  (llama.cpp grammars README; JSONSchemaBench, arXiv:2501.10868, found constrained
  decoding beat unconstrained generation on task quality for every engine tested.)

## Part 2: Live experiment (ubuntu-server, 2026-10-28)

Setup: llama.cpp a7af149 Vulkan, GPU exclusive, temp 0, 512-token budget, each model's own
chat template. Task: extract a lead from a customer message into 6 fields (name, phone,
service, city, urgency enum, callback_ok). 3 cases: A emergency with full data, B low
urgency with full data, C a pool question with **no phone number** (hallucination probe:
correct phone value is null). 4 techniques x 3 cases x 10 models = 120 runs.
Scores: parse = valid JSON extracted; keys = all 6 keys present; fields = correct values
of 18 (3 cases x 6 fields). Raw runs: `/mnt/models/structured-2026-10-28/results/`.

| Model | Plain | JSON mode | JSON Schema | Few-shot |
|---|---|---|---|---|
| Muse-Glimmer-30B | 1/3 parse, 6/18 | 1/3, 6/18 | 1/3, 5/18 | **3/3, 17/18** |
| Qwen3.8-27B | 2/3, 11/18 | 2/3, 11/18 | 1/3, 4/18 | 2/3, 11/18 |
| Qwen3.6-27B | 0/3, 0/18 | 0/3, 0/18 | 0/3, 0/18 | 1/3, 6/18 |
| Gemma-4-31B-it | 2/3, 11/18 | 2/3, 11/18 | **3/3, 16/18** | 2/3, 11/18 |
| Qwen2.5-Coder-32B | 3/3, 15/18 | 3/3, 15/18 | **3/3, 17/18** | 3/3, 15/18 |
| Nemotron-Cascade-2 | 2/3, 10/18 | 2/3, 10/18 | 2/3, 9/18 | **3/3, 15/18** |
| North-Mini-Code-1.0 | 3/3, 14/18 | 3/3, 14/18 | 3/3, 15/18 | 2/3, 10/18 |
| Devstral-Small-2 | 3/3, 15/18 | 3/3, 15/18 | 3/3, 15/18 | 3/3, 15/18 |
| DeepSeek-Coder-V2-Lite | 3/3, 13/18 | 3/3, 13/18 | **3/3, 16/18** | 3/3, 15/18 |
| gpt-oss-20b | 3/3, 15/18 | 3/3, 15/18 | 3/3, 15/18 | 3/3, 15/18 |

## What the runs actually show

1. **The dominant failure is reasoning eating the budget, not bad JSON.** Qwen3.8's schema
   run on case A: 512 output tokens produced, 2,022 characters of reasoning, **empty
   content**. Qwen3.6 did this on 11 of 12 runs across every technique. Glimmer's plain
   run began emitting perfect JSON and still failed because 2,089 characters of reasoning
   came first inside the same budget. Almost every response that produced content at all
   parsed cleanly. Shape is nearly free; finished answers are the scarce resource.
2. **Few-shot is the single biggest lever for channel/thinking models.** Glimmer went
   from 6/18 (plain) to **17/18, the best cell in the experiment**, and its reasoning
   dropped to zero characters: the worked example switched it into direct-answer mode.
   Nemotron gained the same way (10 to 15). This matches the vendor docs: for Qwen and
   DeepSeek families the in-prompt example is part of the documented mechanism.
3. **JSON Schema is the best lever for capable-but-sloppy non-thinking models.** Gemma
   11 to 16, DeepSeek 13 to 16, Qwen2.5-Coder 15 to 17 (best total with Glimmer-few-shot),
   North 14 to 15. The grammar guarantees the shape; these models supply the values.
4. **JSON mode (json_object) changed nothing anywhere.** Identical scores to plain for
   8 of 10 models. In llama.cpp it adds only the generic JSON grammar, and these models
   already emit JSON-shaped text when asked. It is a checkbox, not an improvement.
5. **Devstral and gpt-oss are technique-proof:** 15/18 under all four techniques. Their
   remaining misses are value judgments, not format. For a set-and-forget extraction
   endpoint they are the safe defaults.
6. **Nobody hallucinated.** In all 120 runs, no model invented a phone number for the
   phoneless case; every parsed answer used null. The extraction risk on this task class
   is not fabrication.
7. **The real accuracy leak is undefined judgment fields.** Across all parsed runs the
   misses concentrate in exactly two fields: callback_ok (28 misses on case C, 20 on
   case A) and urgency on the ambiguous case C (23). Name, phone, service, and city were
   near-perfect everywhere. Models split on whether callback_ok means "has a number"
   (ground truth) or "seems willing," and whether a signal-free message is low or normal
   urgency. That is a prompt-specification problem: define the fields in the prompt
   ("callback_ok: true only when a phone number is present") and most of the remaining
   error in this table should collapse. Stated as a prediction, not a measurement.

## Recommendation for AlphaFlux lead extraction

Stack all three, each doing the job the others cannot:

1. **JSON Schema grammar on** (llama-server response_format json_schema): hard guarantee
   on keys, types, and the urgency enum.
2. **Field definitions in the prompt**, one line each, especially for judgment fields.
3. **One worked example** (few-shot): suppresses reasoning starvation on thinking and
   channel models, teaches value conventions.
4. **Budget discipline:** thinking models get reasoning disabled for extraction, or a
   completion budget of 1024+; 512 starves them before the grammar ever applies.

Model pick on this evidence: Qwen2.5-Coder-32B + schema (17/18) or Glimmer + few-shot
(17/18, and it is the house model) for quality; Devstral or gpt-oss (15/18 under any
technique, fastest operations) when the pipeline cannot babysit configuration. Avoid
Qwen3.6 for structured work until its reasoning is controlled; avoid relying on
json_object mode to fix anything.

## Appendix: model swap cycle, stopwatch (same day, David's question)

Full cycle = server start to ready, one question answered, process killed, VRAM verified
back at the 31 MiB baseline:

| Model | Load to ready | Question | Unload to VRAM-free | Full cycle |
|---|---|---|---|---|
| Qwen2.5-Coder-32B (19.9 GB, largest) | 4.10 s | 0.87 s | 0.36 s | **5.33 s** |
| gpt-oss-20b (12.1 GB) | 2.77 s | 0.52 s | 0.36 s | **3.65 s** |

Swapping models per task on this box costs single-digit seconds; per-task model routing
is practical.
