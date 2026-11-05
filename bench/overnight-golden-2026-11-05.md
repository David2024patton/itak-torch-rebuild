# Overnight golden regeneration and determinism check, 2026-11-05

Lane: `lane/overnight-golden`, base main `bda7ff55ea57f4a764afddf9ba2e050c59b86916`.
Host: ubuntu-server. Reference build: llama.cpp `0.7.0-dev (build 137,
commit a7af149)`, binaries `/home/server/llama.cpp/build-vulkan2/bin`
(the same build that produced the committed fixtures; its version
string is recorded in every fixture's metadata). All server work ran
in staging `~/overnight-golden` under
`flock -w 14400 /tmp/itak-phase.lock`; staged script and prompt
hashes were verified against the worktree before any proof run
(gen_golden.py 617aee4aa717..., gen_first_token_probs.py as fixed
9804bf302aef..., prompts.jsonl 32a103b474db... which also equals
the prompts_sha256 recorded in the committed fixtures).

## Inventory

Committed fixtures at start: TinyLlama 1.1B Chat Q4_K_M and
Qwen2.5-1.5B Q4_K_M (both generated 2026-10 with build a7af149),
plus six candidate files (base / fixed / perarch per model).
Prompt set: golden/prompts.jsonl, 15 prompts, set version 1.0.

Models on the server: /mnt/models/teardown holds TinyLlama (size
668,788,096, matches fixture metadata) but neither Qwen file;
Qwen2.5-1.5B Q4_K_M was found at
/home/server/torch-bench-2026-10-28/models/qwen2.5-1.5b-q4km.gguf.
No Qwen2.5-7B GGUF exists anywhere on the server (teardown,
/mnt/models, torch-bench models all searched). Glimmer Q4_K_M is at
/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
(16,756,683,904 bytes, sha256
4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e).

The four pending references, per this inventory: (a) TinyLlama
fixture regeneration, (b) Qwen2.5-1.5B fixture regeneration,
(c) Qwen2.5-7B fixture (TODO T-2.1's named model), (d) a Glimmer
reference pair: a token-level fixture for a bounded prompt subset
plus the first-token top-5 probability golden TODO T-9.3 names.

## What was regenerated, and attribution

(a) TinyLlama: regenerated with the server's current build.
Result: byte-identical to the committed fixture (cmp clean, local
and server side). No drift, nothing to attribute: same llama.cpp
build (a7af149 in both metadata blocks), same model sha256
(9fecc3b3...), same prompts sha256. The committed fixture was not
stale.

(b) Qwen2.5-1.5B: regenerated from the torch-bench models copy.
Result: byte-identical to the committed fixture (cmp clean). Same
attribution as (a): build a7af149, model sha256 6a1a2eb6...,
prompts sha256 unchanged. Not stale.

(c) Qwen2.5-7B: NOT GENERATED. The model file is not on the server,
so there is nothing to generate from. This is a hard blocker, not a
skip: TODO T-2.1's proof needs "the Qwen2.5-7B test GGUF", and no
file by that name or shape exists in /mnt/models/teardown,
/mnt/models, or /home/server/torch-bench-2026-10-28/models. The
fixture lands the night the GGUF is placed on the server; the
generator command is the standard one in golden/README.md.

(d) Glimmer token-level subset fixture: NEW, committed as
golden/fixtures/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.subset.golden.jsonl.
Deviation from the standard fixture shape, recorded here and
self-documented in the fixture metadata (prompts_file +
prompts_sha256): the prompt set is the 3 factual prompts of set 1.0
(golden/prompts-glimmer-subset.jsonl: factual_capital,
factual_science, factual_history) instead of all 15, because CPU
generation of 64 tokens x 15 prompts on a 30B is not
overnight-feasible. Token count per prompt is the standard 64, not
reduced. Generation ran CPU only (-ngl 0) through the approved
throwaway-server pattern (127.0.0.1:18923, killed after). Sanity
cross-check: factual_capital's first completion ids
[13796, 26, 1573, 373, 290, 2167, 26899, 1212] match the T-9.6
identity record's first 8 tokens exactly.

(d2) Glimmer first-token top-5 probability golden (TODO T-9.3):
NEW generator golden/gen_first_token_probs.py (same throwaway
CPU-server pattern, n_predict=1, n_probs=5, post_sampling_probs),
committed fixture
golden/fixtures/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.top5.jsonl,
all 15 prompts of set 1.0. First attempt failed with
KeyError 'probs': this llama.cpp build reports per-step
alternatives under the key "top_probs", each entry
{"id", "token", "bytes", "prob"} (tools/server/server-task.cpp,
completion_token_output::to_json in the server's own source tree;
confirmed by a live probe whose step keys were
[bytes, id, prob, token, top_probs]). The generator was fixed to
that shape, restaged with hash verification, and rerun.
STATUS AT COMMIT TIME (2026-11-05T05:20Z): the fixed generator is
committed here and hash-verified in staging, and the rerun is
queued on the phase lock (~/overnight-golden/run-top5.sh, flock
-w 14400) behind the other overnight lanes, whose lock sessions
ran long. The fixture file itself had not landed when this note
was committed. Pickup: when ~/overnight-golden/out/glimmer.top5.jsonl
appears on the server, copy it to
golden/fixtures/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.top5.jsonl,
verify its metadata (model sha256 4cc57c0f..., build a7af149,
15 records), and amend this section with the top-1 probabilities.
Everything else in this note is final.

## Determinism check

Leg (i), generator: gen_golden.py run twice for TinyLlama on the
current build; the two outputs are byte-identical (cmp clean).
The README's claim ("regenerating with the same llama.cpp build
reproduces a fixture byte for byte") is PROVEN on build a7af149.

Leg (ii), our engine: crates/arch-llama/examples/golden_run.rs
(built on the server from this lane's staged tree,
cargo build --release -p arch-llama --example golden_run) run
twice against the TinyLlama fixture set; the two candidates are
byte-identical (cmp clean). Bonus cross-check: the fresh candidate
is also byte-identical to the committed
candidates/tinyllama-1.1b-chat-v1.0.Q4_K_M.candidate.jsonl and to
the .perarch candidate, so the current engine reproduces the
recorded candidates exactly.

## Compare verdicts after regeneration

The regenerated (a) and (b) fixtures are byte-identical to the
committed ones, so no verdict could change; recorded anyway, from
golden/compare.py against the regenerated fixtures:

- TinyLlama vs fresh candidate (this run): 6/15 token-identical,
  0 text-identical, 9 divergent, 0 missing. Identical verdict for
  the committed base and perarch candidates; the committed fixed
  candidate scores 0/15 token-identical, 15 divergent (historical
  artifact, unchanged).
- Qwen2.5-1.5B vs committed candidates: base 0/15 token-identical,
  15 divergent; fixed 9/15 token-identical, 6 divergent; perarch
  9/15 token-identical, 6 divergent. All unchanged.

## Exact commands (server, under the phase lock)

    python3 gen_golden.py --model /mnt/models/teardown/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --out ~/overnight-golden/out/tinyllama.regen1.jsonl   (and regen2, same command)
    python3 gen_golden.py --model /home/server/torch-bench-2026-10-28/models/qwen2.5-1.5b-q4km.gguf --out ~/overnight-golden/out/qwen15.regen.jsonl
    python3 gen_golden.py --model /mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf --prompts prompts-glimmer-subset.jsonl --out ~/overnight-golden/out/glimmer-subset.golden.jsonl
    python3 gen_first_token_probs.py --model /mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf --prompts prompts.jsonl --out ~/overnight-golden/out/glimmer.top5.jsonl
    cargo build --release -p arch-llama --example golden_run
    golden_run --model /mnt/models/teardown/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --prompts golden/prompts.jsonl --out ~/overnight-golden/out/tinyllama.cand1.jsonl   (and cand2, same command)

Run window: 2026-11-05T04:14:04Z to 04:40:57Z for the main script;
the top-5 rerun followed after the generator fix. Production
Glimmer was never started; the 8091/8092 embedding tenants were
never touched; every throwaway server bound only 127.0.0.1:18923,
CPU only, and was terminated by its generator script.
