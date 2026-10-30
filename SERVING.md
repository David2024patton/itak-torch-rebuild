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

Startup loads the model once (tens of seconds for a 30B file:
parse, coverage gate, weights, tokenizer) and logs the result: name,
architecture, layers, context length, and the coverage counts. A
load failure exits non-zero with the named error.

## The seam, operator's view

- One JSON object per line on the socket. Frame kinds: `hello` /
  `hello_ack` (protocol 1), `ping` / `pong`, `debug` /
  `debug_info`, `generate` answered by streamed `token` frames and a
  terminal `done` frame, or an `error` frame with a named code.
- The platform dials a fresh connection per generation, so a
  minutes-long generation never blocks health or debug traffic on
  the control connection. The engine runs one generation at a time
  (the CPU backend serves one sequence); further generate requests
  queue at the engine.
- Error codes and their HTTP mapping: `bad_request`,
  `context_overflow`, `model_not_found`, and `protocol_mismatch`
  map to 400; `forward_failed`, `detokenize_failed`, and
  `engine_died` map to 502. Embeddings have no protocol operation
  yet and answer 501 `not_implemented`.
- `GET /debug` on the platform includes the engine's self-report
  as `engine_debug`: model identity and path, coverage expected and
  found, generation and token counters, last job's prompt and cache
  position, busy state, uptime.

## Honest state of the serving path

- Speed: the engine's backend is the CPU reference implementation.
  On the 30B Glimmer file it prefills at roughly 4.3 minutes per
  prompt token and decodes at roughly 45 seconds per token. It
  exists to be correct (it matches llama.cpp token for token at
  temperature 0), not to be fast; the GPU backend and the
  shape-specialized kernels are TODO T-9.5 and T-9.10, and until
  they land this path is a correctness proof, not a serving tier.
- Sampling: temperature and top-p are honored by the engine's
  sampler; stop strings are honored with hold-back, so a stop
  string is never half-emitted. There is no seed in the protocol
  (sampled runs are not reproducible across the seam yet), no
  top-k, and no penalties; grammar and JSON mode are TODO T-3.5.
- The platform flattens chat messages into a plain `role: content`
  transcript with no chat template. That is the platform's current
  request shape, and the engine treats it as a raw completion.
- Supervision: if the engine process dies, in-flight generations
  end as closed streams and new ones fail the dial; the platform
  does not yet restart the engine (TODO list in TODO.md, Phase 9).
  A mid-generation engine error likewise closes the stream without
  a done event rather than arriving as a named terminal error.
- One model per engine process, one generation at a time, token
  events as JSON text on the socket. The shared-memory token rings
  of ARCHITECTURE.md Section 4 are the later data plane, not this
  one.

The 2026-10-30 smoke transcript (build, engine start, refusal
checks, the end-to-end completion, and the engine-alone reference
run on the same prompt) is filed at
`bench/serving-smoke-2026-10-30.md`.
