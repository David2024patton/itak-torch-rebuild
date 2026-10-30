# Pending reference captures (prompt set v1.1)

Four prompts were added to `prompts.jsonl` on 2026-10-29 (prompt set
version 1.1) for the forward divergence hunt. Each carries
`"reference": "pending"` in its prompts.jsonl record. **No reference
output exists for them yet, and none may be written by hand**: golden
records only ever come out of `gen_golden.py` running against
llama.cpp.

Why pending: the captures must run where the llama.cpp build and the
teardown models live (the server). The machine this prompt-set edit
was made on has no llama-server reachable on any port the repo docs
name (checked 2026-10-29: 8080 production, 18299 run B throwaway,
18923 gen_golden default, all silent), no llama.cpp binaries, and no
teardown models.

## The new prompts

| id | Purpose |
| :--- | :--- |
| `factual_planet` | The Run B pattern on the small models: a prompt whose correct first token is a strong factual completion (" Jupiter"), so a first-token argmax miss is unambiguous. Run B hit exactly this on Glimmer via `factual_capital`; the fixture models get their own probe. |
| `long_decode_counting` | A counting continuation that reliably fills the 64-token budget, guaranteeing a decode sequence of at least 32 tokens to hunt decode-time divergence over a long horizon. |
| `numbers_digits` | Number words to digits ("three thousand four hundred twelve" to "3412"): digit tokenization and multi-digit assembly, which the existing arithmetic prompts (digits already in the prompt) do not exercise. |
| `punct_contractions` | Contractions and quotes (can't / he'd / we're inside a quoted clause): the tokenizer's known trap area for apostrophe merges. |

## Exact capture commands

On the server (the host with `/home/server/llama.cpp/build-vulkan2/bin`
and the teardown models), from this directory:

```sh
# 0. Remove the four `"reference": "pending"` fields from
#    prompts.jsonl FIRST. The fixture meta binds the prompts file by
#    SHA-256, so generation must run against the exact file that
#    gets committed. Do not commit anything until step 3 passes.

# 1. Determinism first: prove the reference still reproduces itself
#    on the new prompts before trusting the capture.
python3 determinism_check.py --model /mnt/models/teardown/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --id factual_planet --id long_decode_counting --id numbers_digits --id punct_contractions
python3 determinism_check.py --model /mnt/models/teardown/qwen2.5-1.5b-q4km.gguf --id factual_planet --id long_decode_counting --id numbers_digits --id punct_contractions

# 2. Regenerate both fixtures over the full v1.1 set (19 prompts).
python3 gen_golden.py --model /mnt/models/teardown/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
python3 gen_golden.py --model /mnt/models/teardown/qwen2.5-1.5b-q4km.gguf

# 3. Confirm the original 15 records reproduced unchanged: diff the
#    regenerated fixture's first 15 records against the committed
#    v1.0 fixture (git show HEAD:golden/fixtures/...). Only the meta
#    line (prompt set version + prompts sha) may differ.
```

(The Qwen2.5 model path above follows the TinyLlama teardown layout;
substitute the actual path of `qwen2.5-1.5b-q4km.gguf` on the server
if it differs.)

After capture passes step 3: commit prompts.jsonl (fields removed),
both regenerated fixtures, and the deletion of this file together.
