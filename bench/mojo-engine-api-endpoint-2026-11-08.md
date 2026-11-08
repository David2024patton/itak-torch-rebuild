# Mojo engine API endpoint and test bench: build and proof record

Date: 2026-11-08. Branch: lane/mojo-api-endpoint (worktree
~/workspace/lanes/mojo-api-endpoint). Engine: 1.6 -> 1.7.
Card: AMD RX 7900 XTX (gfx1100), MAX 26.7, Mojo 1.2.0.

This note records what was built, what was proven, and what is
known to be missing. Every number below comes from a direct
foreground read of a log file or an HTTP response in the proof
window, not from a background completion report.

## What shipped

A Mojo-only HTTP API endpoint inside the engine process, an
OpenAI-shaped chat route over the existing generation loop, a
JSON-fed test bench dashboard, on-disk run history, model fact
sheets read from the GGUF files themselves, a measured-only
leaderboard with reference rows, run export/import for syncing
benches between machines, and an MCP-style bridge for tool
callers. David's spec amendments of 2026-11-08 are folded in:
source-tagged history, greedy lock, full run telemetry schema,
bundle export/import, metadata-plus-tensor fact sheets, and
the dashboard archive guardrails.

Commits on the lane (base ca9ba36, engine 1.6):

- 524df46 serve layer foundation (pause commit)
- 53d84ea serve wiring and amendments: accept loop, dashboard,
  history and leaderboard, fact sheets, greedy lock
- b43f5f4 pure-Mojo SHA-256 for model fingerprints
- 83ff156 serve startup diag breadcrumbs
- e01115d serve GPU path uses the proven model constructor
- 57fd096 streaming SHA-256: the startup stall fix
- c66da14 dashboard refusal line uses a colon, not an em dash
- 4087520 greedy lock accepts temperature 0 in both JSON forms
- 918c7cc DESIGN step 7 decided: Mojo-only serving seam
- a8b508b fact sheet finds arch-prefixed vision metadata blocks
- ba16748 engine 1.7: the endpoint answers end to end
- b859d12 dashboard readZip closing brace fix
- 4ba7e44 llama reference row cites its primary record

No merge or push to mojo-max-engine; integration stays with
the coordinator discipline. Remote main untouched.

## Transport: how a Mojo program serves HTTP here

Pure Mojo over POSIX sockets. The transport calls libc
(socket, bind, listen, accept, read, write, close) through
std.ffi.external_call. A compiled probe bound, listened, and
answered a real curl request on the VM before the serve layer
was built on it. No Lightbug, no Python, no Go.

DESIGN.md step 7 is marked decided (David, 2026-11-08):
Mojo-only. MAX Serve was rejected because it is Python-based;
socket parity with the paused Go platform was rejected
because it is Go.

## Module map (mojo-engine/src)

- serve/json.mojo: minimal JSON reader/writer (escaping,
  string/int/float/bool/raw fields, string arrays, \u escapes
  with surrogate pairs). Refusals: serve_bad_json,
  serve_missing_field.
- serve/http.mojo: HTTP request parse, the route table, the
  response builder. Routes: /, /health, /stats, /debug/log,
  /openapi.json, /v1/models, /v1/models/<id>,
  /v1/chat/completions, /v1/attachments/text, /history,
  /history/export, /leaderboard, /v1/runs/ingest,
  /mcp/tools, /mcp/call, /mcp.
- serve/stats.mojo: request counters and the debug ring
  buffer (diag-seam-shaped lines, capacity from config).
- serve/server.mojo: health/models/error/chat-response
  builders, parse_chat_request with the greedy lock and the
  vision_not_ready refusal, MCP tool list, attachment
  validation, ingest result builder.
- serve/openapi.mojo: the static OpenAPI 3 contract. A suite
  test keeps the spec and the route table in step.
- serve/socket.mojo: the libc transport.
- serve/factsheet.mojo: GGUF fact sheets (below).
- serve/history.mojo: run records, SHA-256, persistence,
  bundles, leaderboard (below).
- serve/dashboard.mojo: the dashboard page (below).
- runner/serve_main.mojo: the accept loop wiring transport
  and dispatch to the generation loop, the MCP dispatch, and
  the JSON-RPC subset.

Serve config (EngineConfig, env prefix MOJO_ENGINE_SERVE_):
host 127.0.0.1, port 18767 (deliberately not 8080), max body
65536 bytes, chat default/max tokens 64/512, debug ring 200
lines, attachment context cap 32768 bytes, attachment warn
threshold 4096 tokens, history dir serve_history, models dir
/mnt/models, backend target string "gfx1100 / MAX 26.7"
(operator fact from config, never probed or guessed).

## The generation contract (read this before judging speed)

The ForwardModel seam has no reset verb and both models are
stateful (KV append). So every chat request builds the model
fresh, generates, and tears down. That is the honest v1
state contract on the current seam. On the card at context
4096 a chat round trip measured 12.4 to 17.9 seconds of wall
time, of which prefill was about 200 ms and decode ran at
32.7 tok/s (16-token run) inside the generation loop; the
rest is the per-request model build. A resettable session is
future seam work, recorded here so nobody mistakes the build
cost for decode speed.

Prefill and decode are timed separately by _generate_timed,
an exact twin of the generation loop's pick logic living in
serve_main; generate() itself does not surface the split.

## Greedy lock

v1 generation is greedy only; that is all the engine has. The
API echoes sampler {"name":"greedy","temperature":0.0,
"seed":null} in every chat response and health reply. A
request naming a nonzero temperature, top-k, top-p, or any
seed is refused by name (serve_sampler_not_available, HTTP
422) instead of being silently ignored. Per David's
2026-11-08 clarification, an explicit temperature of zero is
accepted in both JSON forms, integer 0 and float 0.0; both
parse to 0.0 through json_float_field. Suite-pinned, and
proven on the wire in the card window: the integer-0 request
passed the sampler check and was refused later, by name, on
the token cap.

## Run records, history, leaderboard

Every run that hits the endpoint is recorded with: timestamp
(UTC seconds, from libc time() through FFI; Mojo 1.2 std.time
is monotonic only), source tag (dashboard | cli | mcp; the
request's source wins, MCP dispatch passes "mcp"), model id
and model fingerprint (SHA-256, below), backend target
string, prompt token count, prompt bucket, prefill ms and
prefill tok/s, TTFT, decode tok/s, tokens generated, peak
VRAM, reserved VRAM, KV cache footprint, sampler settings,
quality verdict when set, attached filenames, and the
imported flag plus origin backend for imported runs.

- Prompt bucket rule: nearest of 128 / 1024 / 4096, ties to
  the smaller bucket, clamped at the ends (576 and below is
  128; 2560 and below is 1024; above is 4096). Suite-pinned
  at the edges.
- TTFT: v1 is non-streaming, so TTFT cannot be measured
  separately from prefill. Records carry ttft_ms null and
  ttft_note "ttft_equals_prefill_non_streaming_v1". Chat
  responses carry the same null.
- Peak VRAM: MAX 26.7 does not expose a peak-allocation
  counter to Mojo. Records carry peak_vram_mib null and
  peak_vram_note "not exposed by MAX 26.7 to Mojo;
  reserved_vram_mib is the engine reservation ledger".
  Never a guessed number. reserved_vram_mib comes from
  GpuRuntime.total_reserved(); kv_cache_mib from
  DeviceKvCache.total_bytes. Card proof at context 4096:
  reserved 16,739 MiB, KV 260 MiB, matching the engine's own
  budget math (272,629,760 bytes of KV is 260 MiB).
- Persistence: JSON lines at <serve_history_dir>/runs.jsonl,
  surviving restart. GET /history serves them with optional
  source and model filters (raw substring match on the model
  id; query values are not percent-decoded in v1).
- Leaderboard: rows grouped per model + backend target,
  never merged across cards; best decode and prefill tok/s,
  run count, last timestamp, concatenated quality verdicts;
  ranked by best decode descending. Reference rows are
  appended after measured rows, kind "reference": llama.cpp
  Vulkan (decode 44.59, 1,022-token prefill 995.99;
  bench/baseline-session-2026-11-06.md) and the paused Rust
  engine (plain decode about 34.5;
  bench/wave20-session-2026-11-07.md). An earlier draft of
  the llama row cited the wave20 note; the figures live in
  the baseline session record, corrected in 4ba7e44.

## Export and import (Chris sync)

GET /history/export returns a bundle:
{"format":"mojo-engine-run-bundle","format_version":1,
"signed":false,"runs":[...]}. POST /v1/runs/ingest validates
every record against the schema; bad records are refused by
name and counted in the response, never silently dropped;
good records land with imported true and origin_backend set
to the record's own backend target. Leaderboard rows for
imported runs note "imported runs included" and rank per
hardware as above.

KNOWN LIMIT: bundles are UNSIGNED in v1. The format carries
format_version so signing can be added later; until then any
caller can POST a forged bundle to an open ingest endpoint.
The endpoint binds 127.0.0.1 by default; treat wider binds
accordingly.

## Fact sheets

fact_sheet(path) reads only the file's own GGUF header data.
The per-layer kind split is derived from BOTH sides:

- metadata: the arch pattern array when present (Glimmer),
  <arch>.full_attention_interval plus <arch>.ssm.* keys
  (qwen35), head_count_kv scalar or per-layer array.
- tensor signatures: per layer, ssm tensors outrank attn
  tensors outrank other; vision tensors counted under v.
  and clip. prefixes; mtp tensors under mtp.

Vision presence: vision tensors or a vision metadata block
in-file (source "in_file"), else a companion mmproj file in
the same directory (source "companion_mmproj", naming
convention mmproj*.gguf), else none. The metadata block
check covers clip.* keys and the arch-prefixed
<arch>.vision.block_count (a8b508b: the Qwen file writes
qwen35.vision.block_count; a clip-only check read false).

Ground truth reproduction, card window (direct reads):

Qwen3.5-9B-abliterated-vision-Q4_K_M.gguf: arch qwen35,
883 tensors, 32 blocks, attention 8, ssm 24, sliding 24,
global 8, full_attention_interval 4, vision tensors 441,
vision source in_file, metadata_block true (after a8b508b),
mtp 15, context 262144, embedding 4096, ffn 12288,
file_type 15, vocab 248320. Every figure matches the
independently verified ground truth.

Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf: arch muse-glimmer,
731 tensors, 52 blocks, attention 52, ssm 0, sliding 39,
global 13 from the pattern metadata, context 131072,
embedding 6656, ffn 19968, KV heads 2, vocab 202048, vision
tensors 0, vision present via companion_mmproj
(mmproj-Muse-Glimmer-30B-Q4_K_M.gguf sits in the same
directory; reported as companion-sourced, not in-file).

Registry note: the models scan reads one directory level.
The served model is always listed; the window pointed the
scan dir at the Qwen family directory so one instance showed
both sheets.

## SHA-256 and the startup stall

Model fingerprints are SHA-256 of the whole file. libcrypto
through FFI failed at AOT link time (undefined SHA256_Update
and SHA256_Final; the MAX link line carries no libcrypto),
so history.mojo carries a pure-Mojo FIPS 180-4 SHA-256,
pinned by the FIPS "abc" vector and a multi-chunk file test
in the suite.

Three proof windows then stalled at serve startup on the
real 16.7 GB file. Root cause, read from the code: the
round-constant table was rebuilt on every one of the 64
rounds of every 64-byte block (about 262 million blocks for
Glimmer), and every file byte was appended individually to a
growing list. Not an infinite loop; catastrophically slow
(46+ minutes at 100% CPU). The docstring at the time claimed
a 17 GB file "hashes in a few seconds"; it does now, per
file size class below. Fixed in 57fd096: the table is built
once per hash, and the reader compresses full blocks
straight out of each 4 MiB chunk, copying only the sub-block
remainder.

Probe on the server (CPU and disk only, direct log reads):
64 MiB of /dev/urandom hashed in 446 ms, digest equal to
sha256sum. The Qwen file (6,594,462,816 bytes) hashed in
46,785 ms to the Hugging Face LFS oid:

    eebf93fe1af74695f4768535489d2af75c862a6bad443fa4b16e1f5a96d04394

The Glimmer file (16,756,683,904 bytes) hashed in
128,357 ms to the recorded sha256:

    4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e

About 130 MB/s. Consequence:
serve startup spends about 128 s hashing the served model
before it listens; window 4's health wait allowed 660 s and
health answered at about 132 s.

## Dashboard

One page at /, six tiles: CHAT, DEBUG LOG, MODELS,
LEADERBOARD, RUN HISTORY, STATS. The page renders from the
JSON endpoints (health, stats, models, leaderboard, history,
debug log) and carries the OpenAPI-shaped calls it makes;
the spec itself is at /openapi.json (16 paths, parses as
JSON, suite-checked against the route table).

- Chat panel: prompt, max tokens, repeat-run (N small),
  reply with token counts and timings, transcript kept in
  page. Sends temperature 0.0 explicitly, source dashboard.
- Attach: images go in the standard image field and meet the
  server's named vision_not_ready refusal until a vision
  path exists. Text documents are extracted IN THE BROWSER:
  .txt/.md read directly; .zip read by a hand-written reader
  in the page (stored entries directly, DEFLATE entries via
  the browser's DecompressionStream("deflate-raw"), local
  header walk with a central-directory fallback). PDF and
  7z are refused by name in-page (serve_attachment_unpack:
  not carried by this page in v1). This is the brief's
  recorded fallback route: no external JS libraries are
  carried. Extracted text is prepended with per-file
  manifest headers (--- FILE: path ---) and the run record
  stores the attached filenames.
- Guardrails (Piece 3): the page estimates tokens as
  ceil(chars/4), labeled an estimate, and refuses to send
  context estimated over the config threshold (default
  4096 tokens) until an explicit override click; the
  warning shows estimate and cap. The server's 32768-byte
  attachment cap stays as the hard backstop with its named
  refusal (serve_context_too_large).
- Debug panel: polls /debug/log every 2 s, auto-scroll
  toggle, pause, clear-view.
- Downloads: transcript and debug log as .txt, .md, or
  .pdf, generated in the browser from the page's own data,
  full text (the PDF writer is a minimal PDF 1.4 generator,
  Courier, wrapped at 95 chars, 48 lines per page). A bundle
  export button downloads /history/export as JSON.

Defect found by proof, fixed: the page's readZip function
was one closing brace short (b859d12), which would have made
a browser drop the whole script block. Caught by running
node --check on the script extracted from the served page;
the fixed extraction passes node --check, and the suite now
pins the readZip ending and whole-page brace balance.
Honest limit: the page was verified over the wire (HTTP 200,
all six panels, script syntax-checked) and its endpoints
were exercised with the page's own request shapes; the page
was not driven inside a real browser in this lane.

## MCP-style bridge

GET /mcp/tools lists six tools (chat, health, stats,
debug_log, history, leaderboard). POST /mcp/call invokes one
tool by name with the same handlers as the HTTP endpoints.
POST /mcp speaks a JSON-RPC subset: initialize (protocol
2024-11-05), tools/list, tools/call with results wrapped as
text content. This is an MCP-style bridge implementing that
subset, not a full MCP server: no notifications, no
resources, no prompts, no sampling.

## Card proof, window 4 (2026-11-08)

Staging: git archive of 918c7cc into ~/gpu-serve-api on the
server, STAGED_COMMIT stamped, staged file hashes verified
against git show. Window ran under flock
/tmp/itak-phase.lock; cortex-embed.service stopped for the
window and restored after (active, /health ok on 8091);
containers cortex-cortex-1 and cortex-cortex-surrealdb-1 up
throughout; port 8080 clear at entry and closeout;
production Glimmer never started; no reboot.

- Build: serve_main gfx1100 AOT, build_rc=0, 0 errors.
- Startup: tokenizer built; fingerprint line logged
  sha256=4cc57c0f...f60e; listening 127.0.0.1:18767
  backend=gpu; health answered after about 132 s.
- GET /health: ok, version 1.6 (pre-bump tree), sampler
  object, backend_target, history_dir as configured.
- GET /v1/models and both detail reads: the fact-sheet
  table above. Unknown id: 404 serve_model_not_found.
- GET /openapi.json: parses, 16 paths. GET /: 200, all six
  panel markers. GET /stats: counters live. GET /debug/log:
  ring serving seam-shaped lines including refusals.
- Golden chat (dashboard shape, quality_verdict
  golden_check): 200, content " Paris. It is the most
  populous city in France and the most populous" (the golden
  continuation), usage 6/16/22, prefill 199 ms, decode
  32.65 tok/s, sampler echoed greedy 0.0.
- Refusals on the wire: temperature 0.5, seed, top_k each
  422 serve_sampler_not_available; image field 422
  vision_not_ready; integer temperature 0 passed the lock
  (refused later on the token cap, by name).
- CLI chat and MCP chat both answered with the golden
  prefix and recorded with sources cli and mcp.
- GET /history: 3 runs, full telemetry (fingerprint,
  backend, bucket 128, prefill ms and tok/s, ttft null plus
  note, peak VRAM null plus note, reserved 16,739 MiB, KV
  260 MiB, sampler, golden_check verdict on the dashboard
  run). Filters: source=dashboard returned 1; model=Qwen
  returned 0.
- GET /leaderboard: the measured Glimmer row plus both
  reference rows, kinds marked.
- Export: bundle format_version 1, signed false, 3 runs.
- Scratch round trip (second instance, CPU mode, port
  18768, scratch history dir): ingest of the exported
  bundle accepted 3, rejected 0; a mixed bundle accepted 1
  and rejected 1 with "serve_ingest_bad_record: record 1
  is missing 'prompt_tokens'"; scratch leaderboard showed
  the imported Glimmer rows and a foreign-hardware Qwen row
  (sm_89 / MAX 26.7) ranked separately, plus the reference
  rows; scratch history showed all 4 records imported with
  origin backends preserved.
- Closeout: both serves killed by exact PID, window script
  exited, lock free, ports 18767/18768/8080 clear, embed
  service active with health ok.

## Suite

test_serve: 21 tests, all passing, covering JSON, routing,
escaping, the greedy lock in both temperature forms, the
image refusal, attachment validation, fact-sheet split math
on synthetic files, run-record round trips, ingest
validation, leaderboard shape, reference rows, the OpenAPI
and route table agreement, dashboard wiring and brace
balance, FIPS SHA-256 vectors, and file hashing across the
chunk boundary. Full suite on the final tree (4ba7e44,
run log read directly): 189 tests run, 189 passed, 0
failed, 0 skipped across 33 test files, run_tests: PASS,
engine hello gate at 1.7.

## Not built / known limits

- No streaming; TTFT is therefore prefill-bound (recorded,
  not hidden).
- No vision path; image input is plumbed and refused by
  name until the Qwen architecture build lands.
- No sampler family (temperature/top-k/top-p sampling);
  the greedy lock refuses them by name.
- Per-request model rebuild (see the generation contract).
- Bundles unsigned (format_version 1 leaves room).
- MCP bridge is a subset, not a full MCP server.
- History load skips unparseable lines silently (they stay
  in the file); ingest is the path that refuses by name.
- Query filter values are not percent-decoded.
- Dashboard zip reader covers stored and DEFLATE entries;
  PDF and 7z extraction are named refusals in-page.
- The Qwen3.5 model itself does not run on this engine yet
  (Gated DeltaNet layers are a separate build); its fact
  sheet is a reader result, not a run.
