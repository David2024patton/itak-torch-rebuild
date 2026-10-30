# PLAN S4 Resurrection Gate: Original Torch Tree vs Merged core/ Module, 2026-10-29

Branch `lane-resurrection-check` at base `cc165c3` (tip of `integration-2026-10-29`).
Gate question (PLAN S4): does the ORIGINAL Torch tree compile against the merged
`core/` module (pkg/types, pkg/debug, pkg/event, pkg/health, pkg/registry, built by
lanes 06, 07, 08)? Lanes 06 and 08 proved their own call sites in isolation; this run
is the whole-tree compile they left as the integration step.

## Verdict: the gate FAILS as stated, and the failure is fully localized

The original tree does not compile against the merged core. Every one of the 33
compile errors sits in a single package (`pkg/torch`), in three files
(`server.go`, `handlers_models.go`, `handlers_ops.go`), and every error belongs to
exactly two of the five Core packages: **pkg/event and pkg/debug**. Those are the
two packages with the known shape conflict (merged core carries the GOAgent-ancestor
shapes per PLAN/TODO; the original tree was written against the rescued Core's later
divergent shapes per SALVAGE.md). The conflict is therefore confirmed by whole-tree
compilation, not just suspected from reading.

The other three Core packages are now proven in whole-tree context, extending the
lanes' isolated proofs:

| Core package | Verdict | Evidence |
| :--- | :--- | :--- |
| pkg/types | PASS | `core_adapter.go` and `core_adapter_test.go` type-check against it inside the full package check; zero errors name any types symbol |
| pkg/health | PASS | `core_health.go` (Checker implementation) and `server.go` (StartHeartbeat, HeartbeatConfig, ServiceEndpoint wiring) type-check; zero errors |
| pkg/registry | PASS | `server.go` registry client wiring type-checks; zero errors |
| pkg/event | FAIL | 13 errors: the Bus/Type/New/Emit surface does not exist in the merged package |
| pkg/debug | FAIL | 20 errors: the per-instance Logger and HTTP mux surface does not exist in the merged package |

## Setup

Nothing in either source tree was edited. All reconciliation happened in scratch
copies:

- Worktree: `git -C ~/workspace/itak-torch-rebuild worktree add ~/workspace/itak-resurrection -b lane-resurrection-check integration-2026-10-29`. Main tree and integration worktree untouched.
- `~/workspace/itak-resurrection/scratch/torch`: copy of the original tree `~/workspace/itaktorch-ingest/iTaKTorchQ4` (rsync, excluding `.git`, `releases/`, `*.exe~`, `torch_linux_amd64`, `*.dll`; no `go:embed` exists in the tree, so excluding build outputs cannot affect compilation). Source files byte-identical to the original.
- `~/workspace/itak-resurrection/scratch/core`: copy of the integration branch's `core/` module.
- Import path reconciliation, scratch copies only. The original imports `github.com/David2024patton/iTaKCore/pkg/...` and its go.mod already carries `replace github.com/David2024patton/iTaKCore => ../Core`; the merged module is named `github.com/David2024patton/itak-torch-rebuild/core`. Three edits:
  1. `scratch/core/go.mod`: module line changed to `github.com/David2024patton/iTaKCore`.
  2. `scratch/core` Go files: the module's internal imports (`core/pkg/event` imports `core/pkg/debug`; `core/pkg/health` imports `core/pkg/registry`) rewritten from the `itak-torch-rebuild/core/pkg/` prefix to `iTaKCore/pkg/` (3 lines in `bus.go`, `heartbeat.go`, `heartbeat_test.go`).
  3. `scratch/torch/go.mod`: the existing replace target changed from `../Core` to `../core` (the scratch sibling). The original tree's import lines and code are otherwise untouched.
- The scratch tree is self-ignored (`scratch/.gitignore` contains `*`) and is not part of the commit carrying this document. It remains on disk for rerun reproducibility.

## Commands run

Go 1.26.9 (`~/goroot/go/bin`), `TMPDIR` set to a short path under the worktree
(the first attempt died on the sandbox's 512 MB `/tmp` tmpfs with
`no space left on device` in the build cache; environment artifact, rerun clean).

```text
cd scratch/torch
go build ./...                  # default error cap
go build -gcflags=-e ./...      # full error inventory
go vet ./...
go build <each of the 11 packages individually>
go build -o $TMPDIR/scripts-bin ./scripts   # see per-package note on scripts

cd scratch/core
go build ./...
go vet ./...
go test ./...
```

All module dependencies resolved from the local module cache; zero downloads.
No package in the Linux build graph needed the webgpu module source (only its
`.mod` is cached, and nothing failed for want of it).

## Results

### Merged core standalone (scratch copy, renamed module)

```text
go build ./...   OK
go vet ./...     OK
go test ./...    ok pkg/debug, ok pkg/event, ok pkg/health, ok pkg/registry, ok pkg/types (all 5 pass)
```

### Original tree, per-package verdicts (11 packages)

| Package | Verdict | Why |
| :--- | :--- | :--- |
| `pkg/torch` | FAIL, direct | The 33 errors below; the only package with errors of its own |
| `cmd/torch` (the server binary) | FAIL, transitive | Imports `pkg/torch`; blocked only by it |
| module root package | FAIL, transitive | Imports `pkg/torch`; blocked only by it |
| `pkg/torch/llama` | PASS | |
| `pkg/torch/loader` | PASS | |
| `pkg/torch/native` | PASS | |
| `pkg/torch/tokenizer` | PASS | |
| `pkg/torch/utils` | PASS | |
| `cmd/asm` | PASS | |
| `cmd/itak-quant` | PASS | |
| `scripts` | PASS | Compiles with `go build -o` (a plain `go build ./scripts` trips on the output binary name colliding with the directory, a tooling artifact, not a compile error) |

`go vet ./...` fails with the same `pkg/torch` error set and reports no other
package. Vet type-checks test files as well; no error names
`core_adapter_test.go` or any types symbol.

### Verbatim first failure (default `go build ./...`)

```text
# github.com/David2024patton/iTaKTorchQ4/pkg/torch
pkg/torch/server.go:35:25: undefined: event.Bus
pkg/torch/server.go:155:30: undefined: event.Bus
pkg/torch/server.go:405:38: undefined: event.Type
pkg/torch/handlers_models.go:62:16: s.debugLogger.Info undefined (type *"github.com/David2024patton/iTaKCore/pkg/debug".Logger has no field or method Info)
pkg/torch/handlers_models.go:67:17: s.debugLogger.Error undefined (type *"github.com/David2024patton/iTaKCore/pkg/debug".Logger has no field or method Error)
pkg/torch/handlers_models.go:73:20: undefined: event.TypeModelLoaded
pkg/torch/handlers_models.go:83:16: s.debugLogger.Info undefined (type *"github.com/David2024patton/iTaKCore/pkg/debug".Logger has no field or method Info)
pkg/torch/handlers_models.go:166:16: s.debugLogger.Info undefined (type *"github.com/David2024patton/iTaKCore/pkg/debug".Logger has no field or method Info)
pkg/torch/handlers_models.go:171:17: s.debugLogger.Error undefined (type *"github.com/David2024patton/iTaKCore/pkg/debug".Logger has no field or method Error)
pkg/torch/handlers_models.go:177:20: undefined: event.TypeModelLoaded
pkg/torch/handlers_models.go:177:20: too many errors
```

### Full error inventory (`go build -gcflags=-e ./...`: 33 errors, all in pkg/torch)

By missing or mismatched symbol:

- pkg/event (13): `event.Bus` x2 (server.go:35, 155), `event.Type` x1 (server.go:405),
  `event.TypeModuleStarted` (server.go:320), `event.TypeModuleStopped` (server.go:395),
  `event.TypeModelLoaded` x3 (handlers_models.go:73, 177, 327),
  `event.TypeModelUnloaded` (handlers_models.go:233),
  `event.TypeInferenceError` (server.go:574), `event.TypeInferenceDone` (server.go:602),
  `event.New` wrong arity x2 (server.go:407, handlers_ops.go:64).
  Not separately reported because the `event.Bus` type error masks them:
  `s.eventBus.Emit(...)` call sites (server.go:407, handlers_ops.go:64);
  the merged package has no `Emit`.
- pkg/debug (20): `coreDebug.NewLogger` (server.go:180), `coreDebug.Enabled`
  (server.go:258), `coreDebug.NewMux` (server.go:259), `coreDebug.MuxConfig`
  (server.go:259), `(*debug.Logger).Info` missing x13 (handlers_models.go x9,
  handlers_ops.go x3, server.go:264 x1), `(*debug.Logger).Error` missing x3
  (handlers_models.go:67, 171, 322).

Zero errors reference pkg/types, pkg/health, or pkg/registry. Because the Go
compiler type-checks the whole `pkg/torch` package (all six Core-importing files)
and `-e` lifts the error cap, this inventory is complete for the package: the
absence of types/health/registry errors is a positive proof those call sites
type-check against the merged module, not an untested gap.

## The two shape conflicts, side by side

### pkg/event

| Original tree expects (rescued Core shape) | Merged core provides (GOAgent ancestor) |
| :--- | :--- |
| `type Bus`, held as `*event.Bus`, injected via `WithEventBus(bus *event.Bus)` | `type EventBus`, constructed by `New()` |
| `func (b *Bus) Emit(e Event)` fire-and-forget | `func (b *EventBus) Publish(e Event)`, plus `Subscribe(bufSize, topics...) (int, <-chan Event)` channel fan-out |
| `event.New(typ, source string, data) Event` constructor | `NewEvent` / `AgentEvent` / `ToolEvent` constructors over an `Event` struct literal |
| `type Type` (string kind) with constants `TypeModuleStarted`, `TypeModuleStopped`, `TypeModelLoaded`, `TypeModelUnloaded`, `TypeInferenceDone`, `TypeInferenceError` | `Event.Topic string` with `Topic*` constants whose values are the GOAgent set (`user.input`, `agent.start`, `system.ready`, ...); no model or inference topics at all |

### pkg/debug

| Original tree expects (rescued Core shape) | Merged core provides (GOAgent ancestor) |
| :--- | :--- |
| `coreDebug.NewLogger("torch", coreDebug.LevelInfo)` returns a per-module `*Logger` | No `NewLogger`; one package-global logger configured by `SetLevel` / `SetOutput` / `SetColor` / `EnableDebug` |
| Methods `logger.Info(component, format, args...)` and `logger.Error(component, format, args...)` (16 call sites) | Package-level functions `Info(component, format, args...)` / `Error(...)` with the same argument shapes, but no methods on the exported `Logger` type |
| HTTP debug surface: `coreDebug.Enabled()`, `coreDebug.NewMux(coreDebug.MuxConfig{...})` mounted at `/debug/` (server.go:258-263) | No mux surface at all; the package's HTTP-shaped piece is the `StepLogger` trace recorder, a different feature |

What does line up: `debug.Level` and `debug.LevelInfo` exist in the merged package
with the expected meaning (not flagged), and the Info/Error argument shapes match
the package-level functions exactly. The debug break is about object shape
(instance methods plus constructor plus mux), not about log call signatures.

## What remains between here and a full pass

1. **The DECISIONS ruling on the debug/event shape** (David's call, still open).
   This gate is the evidence for it. Two ways to green, neither taken here because
   the brief for this gate was prove or disprove, not fix:
   - Rule for the rescued Core shape: rework `core/pkg/debug` (add `NewLogger`
     with per-instance Info/Error methods and the `Enabled`/`NewMux`/`MuxConfig`
     HTTP surface) and `core/pkg/event` (add the `Bus`/`Type`/`New`/`Emit` surface
     and the model/inference topic constants), keeping or retiring the GOAgent
     shapes beside them.
   - Rule for the GOAgent shape: rewrite the three Torch call-site files
     (`server.go`, `handlers_models.go`, `handlers_ops.go`) onto `EventBus.Publish`
     with topic strings and the package-level debug functions, and re-home or drop
     the `/debug/` mux.
2. **Rerun this gate** after either fix: same scratch method, `go build ./...`
   and `go vet ./...` to zero errors. Expected scope is contained: only
   `pkg/torch` and its two dependents can change verdict.
3. **The runtime half.** This gate proves compilation only. A full resurrection
   pass also needs the resurrected server booted and the Core adapter paths
   exercised (health heartbeat against a registry, event emission on model load,
   inference done/error), which no lane or gate has proven yet.
4. Minor: `core_adapter_test.go` is the only Core-touching test in the original
   tree; the tree's remaining tests were outside this gate's compile-and-vet scope.
