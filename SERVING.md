# SERVING: running the pair (itak-engine + torchd)

The rebuild serves as two processes on one host:

- **itak-engine** (Rust, `crates/itak-engine`): owns one loaded
  model and the token loop. Loads the GGUF behind the tensor
  coverage gate (a model that does not load completely never
  serves), tokenizes with the model's own tokenizer, generates on
  the CPU reference backend.
- **torchd** (Go, `platform/cmd/torchd`): the HTTP platform. Auth,
  scheduling, response cache, and the OpenAI-compatible routes. It
  generates nothing itself; every completion crosses the seam to
  the engine.

They meet on a Unix socket with an NDJSON control protocol. The
protocol is defined once in `crates/itak-engine/src/protocol.rs`;
this file is the operator's view of the same seam.

## Quick start

```sh
# 1. Build both halves.
cargo build --release -p itak-engine
(cd platform && go build -o torchd ./cmd/torchd)

# 2. Start the engine (loads the model; ready when the log says
#    "serving on ..."). The model name requests must use defaults
#    to the file's metadata name; override with --model-name.
./target/release/itak-engine \
  --model /path/to/model.gguf \
  --socket /tmp/itak-engine.sock \
  --model-name glimmer

# 3. Start the platform against the socket.
TORCH_API_KEY=<your-key> ./platform/torchd \
  -engine-socket /tmp/itak-engine.sock \
  -models glimmer            # binds 127.0.0.1:8091

# 4. Ask.
curl -X POST http://127.0.0.1:8091/v1/chat/completions \
  -H "Authorization: Bearer <your-key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"glimmer",
       "messages":[{"role":"user","content":"Capital of France? One word."}],
       "max_tokens":4,"stop":["\n"]}'
```

The daemon refuses to start without `TORCH_API_KEY` unless
`TORCH_ALLOW_NO_AUTH=1` is set deliberately. Without
`-engine-socket`, the platform still runs and engine routes answer
503 `engine_unavailable`; without the key, protected routes answer
401. Both refusals are by name, never a fabricated success.

## Engine configuration

Flags, each with an environment fallback; nothing is compiled in.

| Flag | Env | Default | Meaning |
| --- | --- | --- | --- |
| `--model` | `ITAK_ENGINE_MODEL` | (required) | GGUF file to load |
| `--socket` | `ITAK_ENGINE_SOCKET` | `/tmp/itak-engine.sock` | Unix socket to bind |
| `--model-name` | `ITAK_ENGINE_MODEL_NAME` | file metadata name, else file stem | Name requests must address |
| `--default-max-tokens` | `ITAK_ENGINE_DEFAULT_MAX_TOKENS` | 128 | Generation cap when a request omits `max_tokens` |
| `--slots` | `ITAK_ENGINE_SLOTS` | 1 | Concurrent generations, each with its own compute and KV state |
| `--max-queued` | `ITAK_ENGINE_MAX_QUEUED` | 8 | Jobs that may wait for a slot; submissions past capacity refuse as `engine_busy` |
| `--max-frame-bytes` | `ITAK_ENGINE_MAX_FRAME_BYTES` | 4194304 | Largest single NDJSON frame accepted; larger or unterminated frames refuse as `bad_frame` |
| `--max-embed-inputs` | `ITAK_ENGINE_MAX_EMBED_INPUTS` | 256 | Largest embed batch in one request |
| `--max-embed-tokens` | `ITAK_ENGINE_MAX_EMBED_TOKENS` | 8192 | Largest token count in one embed input |
| `--log-content` | `ITAK_ENGINE_LOG_CONTENT` | off | Also log prompt ids and generated text; by default job logs carry counts, finish reason, and timings only |

The platform adds its own bounds on top: a per-request `max_tokens`
cap (`-max-tokens`, default 8192, refused by name before the engine
is touched), a per-generation deadline (`-generation-timeout`,
default 1h), scheduler queue caps (`-max-waiters-per-lane`, default
128), and HTTP limits (`-max-body-bytes`, `-read-header-timeout`,
`-idle-timeout`) that are all flags with the listed defaults.

## Platform supervision (torchd)

Instead of starting the engine by hand, torchd can own it:

```sh
TORCH_API_KEY=<your-key> ./platform/torchd \
  -engine-bin ./target/release/itak-engine \
  -engine-model /path/to/model.gguf \
  -engine-slots 1 -engine-max-queued 8 \
  -engine-max-restarts 5 -engine-restart-window 5m
```

The supervisor starts the engine, waits for it to load and serve,
and restarts it when it dies, within the restart budget (restarts
allowed inside the window; a model that can no longer load
exhausts the budget and stays down, visibly, instead of crash
looping). A generation caught by a death answers with the named
`engine_died` error; the next request is served by the restarted
engine. The lifecycle (pid, restarts, exhaustion, last exit) is in
the platform's `/debug` under `engine_debug.supervisor`. Without
`-engine-bin`, `-engine-socket` dials an engine started separately,
as in the quick start.

Startup loads the model once (tens of seconds for a 30B file:
parse, coverage gate, weights, tokenizer) and logs the result: name,
architecture, layers, context length, and the coverage counts. A
load failure exits non-zero with the named error.

## The seam, operator's view

- One JSON object per line on the socket. Frame kinds: `hello` /
  `hello_ack` (protocol 1), `ping` / `pong`, `debug` /
  `debug_info`, `generate` answered by streamed `token` frames and a
  terminal `done` frame, or an `error` frame with a named code;
  `embed` answered by an `embeddings` frame; `chat_template`
  answered by the model's template frame or the named
  `no_chat_template` error.
- The socket is owner-only (0600, set after bind), the engine
  never deletes a non-socket file found at the socket path, and a
  connection must open with `hello` (protocol version required)
  from a peer running under the same uid; anything else is refused
  by name and the connection closes. Frames are bounded on both
  ends (engine `--max-frame-bytes`, platform client cap, both
  4 MiB class), and the engine's JSON parser refuses nesting past
  a depth cap instead of overflowing the stack.
- The platform dials a fresh connection per generation, so a
  minutes-long generation never blocks health or debug traffic on
  the control connection. The engine runs a configured number of
  generation slots (the CPU backend serves one sequence per slot
  instance); further jobs queue in a bounded queue, and
  submissions past its capacity refuse as `engine_busy`.
- Error codes and their HTTP mapping: `bad_request`,
  `context_overflow`, `model_not_found`, and `protocol_mismatch`
  map to 400; `forward_failed`, `detokenize_failed`, and
  `engine_died` map to 502; `engine_busy` maps to 503. A failure
  after streaming has begun reaches the client as a named terminal
  error (an SSE error frame when streaming, the honest status when
  not), never as a truncated 200.
- `GET /debug` on the platform includes the engine's self-report
  as `engine_debug`: model identity and path, coverage expected and
  found, generation, token, and embedding counters, the slot state
  (slots, active, queued, max_queued), whether the model carries a
  chat template, last job's prompt and cache position, uptime; and
  `chat_template` says where prompts get their format (config,
  engine, or none/role flattening).

## Honest state of the serving path

Everything in this section is scoped to what the T-9.17 live proof
run actually demonstrated on the real Glimmer file
(bench/t917-proof-2026-10-30.md), plus the earlier smoke
(bench/serving-smoke-2026-10-30.md). Anything past those receipts
is named as unbuilt, not implied.

- Speed: the engine's backend is the CPU reference implementation.
  On the 30B Glimmer file it prefills at roughly 4.3 minutes per
  prompt token and decodes at roughly 45 seconds per token. It
  exists to be correct (it matches llama.cpp token for token at
  temperature 0), not to be fast; the GPU backend and the
  shape-specialized kernels are TODO T-9.5 and T-9.10, and until
  they land this path is a correctness proof, not a serving tier.
- The GPU forward is not in this path. The device-resident session
  in wgpu-compute reproduces the golden at 39.7 tok/s as a bench
  (bench/gpu-resident-2026-10-30.md), but the itak-engine binary
  does not link wgpu-compute, and no GPU serving is claimed until
  that wiring lands and is proven over the socket.
- Speculative decoding is not in this path either. The itak-spec
  verify loop is proven token-identical to plain greedy on CPU
  (bench/spec-loop-2026-10-30.md); the serving engine does not
  link itak-spec, and served generations are plain generations.
- Sampling: temperature, top-p, top-k, a request seed, and the
  presence/frequency penalties are honored by the engine's sampler
  (seeded sampled runs are reproducible across the seam, proven in
  the T-9.17 run: identical seeded frames twice, the greedy
  control matching the golden, and a presence penalty visibly
  moving a pick); stop strings are honored with hold-back, so a
  stop string is never half-emitted. Grammar and JSON mode are
  TODO T-3.5.
- Chat format: chat requests are rendered with the model's own
  GGUF chat template (a bounded Jinja subset; constructs outside
  it refuse by name rather than rendering wrong). Only a model
  whose file carries no template gets the plain `role: content`
  flattening, and /debug says so. On the Glimmer file the
  rendered prompt's token ids match llama.cpp's templated ids
  exactly (T-9.17, bench/t917-proof-2026-10-30.md).
- Supervision and slots: with `-engine-bin`, torchd restarts a
  dead engine within its budget; in-flight generations end with
  the named `engine_died` error, and an engine-reported
  mid-generation failure arrives as a named terminal error event.
  Without supervision, a dead engine stays dead until an operator
  acts. Multi-slot serving is proven at two concurrent slots with
  independent outputs (T-9.17); larger slot counts are
  configuration, not a measured claim.
- Embeddings are mean-pooled token embeddings from the model's own
  embedding table (the forward's lookup), L2-normalized: an honest
  bag-of-embeddings, not hidden-state pooling. The T-9.17 run
  verified the geometry on the real file (related pair cosine
  0.899662 against 0.370221 and 0.389486 for unrelated pairs).
- One model per engine process, a configured number of generation
  slots, token events as JSON text on the socket. The
  shared-memory token rings of ARCHITECTURE.md Section 4 are the
  later data plane, not this one.

The 2026-10-30 smoke transcript (build, engine start, refusal
checks, the end-to-end completion, and the engine-alone reference
run on the same prompt) is filed at
`bench/serving-smoke-2026-10-30.md`.
