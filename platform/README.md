# Torch Platform (Go)

The Go half of the iTaK Torch rebuild. Go owns the platform: API,
scheduling, caches, and (later) cluster and training orchestration. The
Rust engine owns the hot token loop. The two meet at one seam,
`internal/engine.Client`, over a Unix socket with a JSON control
protocol; tensors will move over shared memory when the Rust side lands.

This is a skeleton built clean from the old platform's audit lessons:
interfaces and wiring are real, the engine sits behind the client
interface, and nothing fakes success. An unconfigured engine answers
with a named `engine_unavailable` error; an unimplemented engine
operation answers `not_implemented`.

## Module map (the grid)

| Module | Job |
| --- | --- |
| `cmd/torchd` | Daemon entrypoint: flags, env, signals, wiring |
| `internal/server` | HTTP routes, validation, SSE streaming, error mapping |
| `internal/auth` | The one API-key middleware (constant-time, fails closed) |
| `internal/engine` | Client interface, named errors, Unix-socket skeleton |
| `internal/scheduler` | Priority-lane concurrency gate; both lanes provably drain |
| `internal/cache` | Exact-match response cache with TTL and injectable clock |
| `internal/diag` | Diagnostics registry behind `/debug` (debugging from birth) |

## Run

```sh
export TORCH_API_KEY='your-key-here'
go run ./cmd/torchd                      # 127.0.0.1:8091, no engine
go run ./cmd/torchd -engine-socket /run/torch/engine.sock -models glimmer
```

The daemon refuses to start without `TORCH_API_KEY` unless
`TORCH_ALLOW_NO_AUTH=1` is set deliberately. The bind is loopback by
default; widening it is an explicit `-addr` decision.

## Routes

| Route | Auth | Notes |
| --- | --- | --- |
| `GET /healthz` | no | Platform liveness + engine health detail |
| `GET /v1/models` | yes | Configured model list |
| `POST /v1/chat/completions` | yes | Validated; scheduler-gated; cached when not streaming; SSE with `stream: true`. Priority via `X-Torch-Priority: high` |
| `POST /v1/embeddings` | yes | String or string-array input |
| `GET /debug` | yes | Diag counters, scheduler depths, cache stats |

## Gates

```sh
go build ./...
go vet ./...
go test ./...
golangci-lint run ./...   # config: .golangci.yml, must run clean
```

Every module ships with unit tests and structured `slog` diagnostics;
the scheduler drain tests and cache TTL tests encode the old platform's
two nastiest lessons as regressions that cannot come back silently.
