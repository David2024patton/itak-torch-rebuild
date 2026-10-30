# Serving smoke transcript, 2026-10-30 (TODO T-9.8)

End to end proof that the Go platform and the Rust engine serve a real
Glimmer chat completion through our own stack: HTTP in, Unix socket NDJSON
to the engine, CPU forward pass on the real 17 GB model file, streamed
tokens back out as an OpenAI-shaped response.

- Host: ubuntu-server (scratch dir `~/itak-serve/`, sources rsynced from
  this repo; nothing installed system-wide)
- Model: `/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`
- Code: commit `e40b8ba` (Rust `itak-engine` binary crate) and commit
  `110ed17` (platform `Generate` over the socket)
- Production Glimmer llama-server (Vulkan, 127.0.0.1:8080, pid 683675)
  was never touched; `/health` returned `{"status":"ok"}` before and
  after the smoke (last verified 2026-10-30T05:18Z).
- All times UTC.

## Build

```text
# on ubuntu-server, in ~/itak-serve/
cargo build --release            # produces target/release/itak-engine
                                 # and target/release/examples/glimmer_generate
# on the build VM
CGO_ENABLED=0 go build -o torchd ./platform/cmd/torchd   # scp'd to ~/itak-serve/torchd
```

## Engine start (2026-10-30T03:38Z)

```text
./target/release/itak-engine \
  --model /mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf \
  --socket ~/itak-serve/run/engine.sock --model-name glimmer
```

Log: model loaded in 50,486 ms, tensor coverage 731/731
(`itak-loader` `require_complete()`), tokenizer ready, listening on the
socket. RSS about 17.6 GB.

## Platform start

```text
TORCH_API_KEY=test-key ./torchd -engine-socket ~/itak-serve/run/engine.sock \
  -models glimmer -addr 127.0.0.1:19091     # with engine
./torchd -models glimmer -addr 127.0.0.1:19092   # no engine (refusal checks)
```

(Ports 19091/19092 because 8091 on this host is already held by an
unrelated llama-server instance; loopback only either way. `test-key` is
a throwaway smoke credential, not a real key.)

## Fast checks (2026-10-30T03:41Z, all passed)

- `GET /healthz` on :19091: engine `ok: true`, detail
  "socket handshake + ping ok"
- `GET /v1/models` with the key on :19091: lists `glimmer`
- `GET /debug` on :19091: `engine_debug` present, model `glimmer`,
  coverage_expected = coverage_found = 731, context_length 131072,
  generations 0
- Chat with no key: refused 401 (verbatim re-capture below)
- Chat with key on the no-engine instance: refused 503 (below)

Verbatim refusal outputs (re-captured 2026-10-30T05:20:09Z against the
no-engine instance on :19092, same code path):

```text
$ curl -s http://127.0.0.1:19092/healthz
{"engine":{"detail":"no engine configured","ok":false},"status":"ok"}

$ curl -s -H "Authorization: Bearer test-key" http://127.0.0.1:19092/v1/models
{"data":[{"id":"glimmer","object":"model","owned_by":"itak-torch"}],"object":"list"}

$ curl -s -X POST http://127.0.0.1:19092/v1/chat/completions \
    -d '{"model":"glimmer","messages":[{"role":"user","content":"Capital of France? One word."}],"max_tokens":4}'
{"error":{"code":"unauthorized","message":"missing or invalid API key"}}
http=401

$ curl -s -X POST http://127.0.0.1:19092/v1/chat/completions \
    -H "Authorization: Bearer test-key" \
    -d '{"model":"glimmer","messages":[{"role":"user","content":"Capital of France? One word."}],"max_tokens":4}'
{"error":{"code":"engine_unavailable","message":"engine: unavailable"}}
http=503
```

## Why the smoke prompt is a question, not the plain statement

The platform flattens chat messages to plain text as
`user: <content>\nassistant:`. Probes against a throwaway CPU
llama-server (port 18299, greedy, production untouched) showed what the
plain statement does with that shape:

- content `The capital of France is` flattens to 11 prompt ids
  [200000, 1556, 38, 589, 7963, 323, 11698, 373, 198, 140680, 38] and
  llama.cpp's greedy continuation RESTATES the sentence
  (" The capital of France is Paris.\nParis") instead of continuing it.
  Our engine would do the same; it is the same model. A smoke that
  "begins with Paris" cannot be built on that prompt shape honestly.
- content `Capital of France? One word.` flattens to 12 prompt ids
  [200000, 1556, 38, 20555, 323, 11698, 43, 5615, 2918, 335, 140680, 38]
  and llama.cpp greedy gives ids [13796, 198, 1556, 38], text
  " Paris\nuser:". First generated token is exactly " Paris".

So the smoke asks the question directly, with `max_tokens: 4` and
stop `["\n"]`, which makes the expected completion exactly " Paris"
with finish_reason "stop".

## The smoke generation

Request sent 2026-10-30T03:42:50Z:

```text
$ curl -s -X POST http://127.0.0.1:19091/v1/chat/completions \
    -H "Authorization: Bearer test-key" -H "Content-Type: application/json" \
    -d '{"model":"glimmer",
         "messages":[{"role":"user","content":"Capital of France? One word."}],
         "max_tokens":4,"temperature":0,"stop":["\n"]}'
```

Engine log:

```text
[itak-engine] generate start: prompt_tokens=12 max_tokens=4 temperature=0
[itak-engine] prompt ids: [200000, 1556, 38, 20555, 323, 11698, 43, 5615, 2918, 335, 140680, 38]
[muse-glimmer] forward.begin: tokens=12 position=0 layers=52
[muse-glimmer] forward.end: logits=202048 position=12
[itak-engine] generate done: finish_reason=stop generated=2 elapsed_ms=2891315 text=" Paris"
```

Response received 2026-10-30T04:31:01Z (curl exit 0):

```json
{"choices":[{"finish_reason":"stop","index":0,"message":{"content":" Paris","role":"assistant"}}],"created":1793334661,"id":"chatcmpl-4d52a65e1b9eac2a","model":"glimmer","object":"chat.completion"}
http=200 time_total=2891.277553s
```

Reading it plainly: the engine generated 2 tokens. Token 13796
(" Paris") was emitted; token 198 ("\n") matched the stop string, was
held back by the streamer's stop hold-back, and ended the generation
with finish_reason "stop". The completion is exactly " Paris", the
same first token llama.cpp produced for the same flattened prompt.

## Reproduction against the engine alone (glimmer_generate)

The same flattened prompt run through the engine's own example binary
(`glimmer_generate`, greedy, 4 tokens), bypassing the platform entirely.
T-9.8's proof asks that temperature 0 through the platform reproduce
this output.

- Attempt 1 (started 2026-10-30T03:36:44Z): KILLED, exit 137 at
  03:44:08Z, while the platform smoke generation was also running.
  Two 30B CPU jobs on one host pinned swap at 3/3 GB; the kill is
  consistent with the OOM killer (no dmesg access to confirm).
  Lesson recorded: never run two 30B CPU jobs concurrently on this
  host. Its logged prompt ids already matched the smoke's 12 ids.
- Attempt 2 (2026-10-30T05:19:02Z to 06:07:11Z, exit 0), run alone:

  ```text
  prompt_ids=[200000, 1556, 38, 20555, 323, 11698, 43, 5615, 2918, 335, 140680, 38]
  coverage: expected=731 found=731 missing=0
  phase=prefill ms=2752226 prompt_tokens=12
  prefill_top rank=0 id=13796 tok=" Paris" logit=12.998796 prob=0.658840
  gen step=1 id=13796 tok=" Paris" source=prefill
  gen step=2 id=198 tok="\n" ms=41851
  gen step=3 id=1556 tok="user" ms=41892
  gen step=4 id=38 tok=":" ms=42448
  generated_ids=[13796, 198, 1556, 38]
  generated_text=" Paris\nuser:"
  peak_rss_kb=38192312
  ```

  The engine alone produces exactly llama.cpp's greedy ids
  [13796, 198, 1556, 38] for this prompt, and the platform smoke's
  two generated tokens (13796 emitted, 198 stop-triggered) are the
  first two of them. Temperature 0 through the platform reproduces
  the engine-alone output, as T-9.8 requires. Three-way agreement:
  llama.cpp probe, engine alone, platform end to end.

## Incidents and honest edges

- The engine process exited at an unknown time after the smoke
  completed cleanly at 04:31:01Z. No panic or error in its log; the
  platform's health check afterwards reported the socket as a broken
  pipe (verbatim, 2026-10-30T05:20:09Z):

  ```text
  $ curl -s http://127.0.0.1:19091/healthz
  {"engine":{"detail":"engine: handshake failed: write frame: write unix @->/home/server/itak-serve/run/engine.sock: write: broken pipe","ok":false},"status":"ok"}
  ```

  Cause unconfirmed (OOM is the leading hypothesis; swap stayed pinned
  at 3/3 GB). Nothing restarted it, because nothing supervises it yet.
  That is TODO T-9.11, and this smoke is its Exhibit A.

- Speed, stated plainly: this request took 2,891 s wall (one 12-row
  prefill plus two single-token forwards on the CPU reference path).
  llama.cpp on CPU answered the same probe prompt in seconds; the
  production Vulkan build does 998 tok/s prefill and 44.31 tok/s
  decode on this model. The CPU engine is a correctness oracle. GPU
  serving is T-9.5/T-9.10 territory and is not claimed here.
