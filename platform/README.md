# Torch Platform (Go)

The Go half of the iTaK Torch rebuild. Go owns the platform: API,
scheduling, caches, engine supervision, and (later) cluster and
training orchestration. The Rust engine owns the hot token loop. The
two meet at one seam, `internal/engine.Client`, over a Unix socket
with an NDJSON control protocol (six ops: hello, ping, debug,
generate, embed, chat_template). The shared-memory data plane in
ARCHITECTURE.md is a later design, not this seam.

This platform is built clean from the old platform's audit lessons:
interfaces and wiring are real, the engine sits behind the client
interface, and nothing fakes success. An unconfigured engine answers
with a named `engine_unavailable` error. Generation and embeddings
over the socket are real and proven live on the real model file
(repo bench/t917-proof-2026-10-30.md). Serving the pair end to end
is documented in the repo's SERVING.md.

## Module map (the grid)

| Module | Job |
| --- | --- |
| `cmd/torchd` | Daemon entrypoint: flags, env, signals, wiring |
| `internal/server` | HTTP routes, validation, SSE streaming, error mapping, generation caps |
| `internal/auth` | The one API-key middleware (constant-time, fails closed, repeated failures throttled per source) |
| `internal/engine` | Client interface, named errors, Unix-socket client (handshake, ping, debug self-report, streamed generate, embed, chat template) with bounded frame reads |
| `internal/scheduler` | Priority-lane concurrency gate; lanes are server-side only, queue-capped, and provably drain |
| `internal/cache` | Exact-match response cache with TTL, an entry cap with expiry-first eviction, a purge janitor, and an injectable clock |
| `internal/chattemplate` | Renderer for the model's own GGUF chat template (a bounded Jinja subset), with render budgets so a hostile template refuses by name instead of hanging or blowing the stack |
| `internal/supervisor` | Engine process lifecycle: start, health wait, restart within a budget, an allowlisted child environment (PATH plus ITAK_ENGINE_* only) |
| `internal/diag` | Diagnostics registry behind `/debug` (debugging from birth) |

## Run

```sh
export TORCH_API_KEY='your-key-here'
go run ./cmd/torchd                      # 127.0.0.1:8091, no engine
go run ./cmd/torchd -engine-socket /run/torch/engine.sock -models glimmer
```

With `-engine-socket`, start the Rust engine first (see the repo's
SERVING.md), or let torchd supervise it with `-engine-bin` and
`-engine-model`. The daemon logs whether the engine connected, and
`/healthz` reports a coarse state either way.

The daemon refuses to start without `TORCH_API_KEY` (minimum 16
characters) unless `TORCH_ALLOW_NO_AUTH=1` is set deliberately. The
bind is loopback by default; widening it is an explicit `-addr`
decision. Every serving limit (generation cap, deadline, body size,
timeouts, queue and cache caps) is a flag with the default recorded
in `torchd -h`.

## Routes

| Route | Auth | Notes |
| --- | --- | --- |
| `GET /healthz` | no | Coarse state only: `ok`, `unavailable`, or `down`. Engine error detail stays behind the authenticated `/debug` |
| `GET /v1/models` | yes | Configured model list |
| `POST /v1/chat/completions` | yes | Validated; scheduler-gated; cached when not streaming; SSE with `stream: true`. A silent mid-stream engine death ends with a named error frame, never a clean `[DONE]` |
| `POST /v1/embeddings` | yes | String or string-array input; mean-pooled token embeddings from the engine, L2-normalized |
| `GET /debug` | yes | Diag counters, scheduler depths, cache stats, engine self-report when the client offers one |

## Gates

```sh
go build ./...
go vet ./...
go test ./...
govulncheck ./...
golangci-lint run ./...   # config: .golangci.yml, must run clean
```

Every module ships with unit tests and structured `slog` diagnostics;
the scheduler drain tests and cache TTL tests encode the old platform's
two nastiest lessons as regressions that cannot come back silently.
