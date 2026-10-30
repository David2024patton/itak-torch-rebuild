# Decision brief: the pkg/debug and pkg/event shape conflict (2026-10-29)

Status: awaiting David's call. This brief changes no code and no decision
record. It lays out the two candidate surfaces, the counted evidence, the
migration cost each way, a recommendation, and a draft DECISIONS.md entry
David can approve as written.

## The question

The rebuilt itak-core module needs one canonical shape for each of
pkg/debug and pkg/event. Two shapes exist, and they are not compatible:

- **Shape G (GOAgent port).** PLAN S4 and TODO T-1.2 direct that pkg/debug
  and pkg/event be ported from their surviving ancestors in the GOAgent
  repo. Lane 07 did exactly that; the merged integration branch carries
  core/pkg/debug and core/pkg/event in this shape, with 21 passing test
  functions written for it.
- **Shape R (rescued Core).** The original iTaK Core was recovered on
  2026-10-28, after PLAN was written. SALVAGE.md, written 2026-10-29 with
  the recovered tree and the three audits in hand, marks the original
  pkg/debug and pkg/event **REUSE**: they are a later, divergent evolution
  of the same two packages.

Both cannot be the surface the original Torch tree compiles against
unchanged, and PLAN S4 names that compile as the resurrection gate for the
whole project. Nobody made an error here: lane 07 satisfied T-1.2 as
written, and SALVAGE judged code PLAN's authors did not yet have. The
recovery changed the facts; the record now has to pick one shape.

Evidence base for this brief: PLAN.md S4 and TODO T-1.2 in this repo;
SALVAGE.md section 3; the merged core/pkg/debug and core/pkg/event on the
integration branch; the rescued originals in
~/workspace/itak-core-rescue/Core/pkg/debug and pkg/event; and caller
greps over the original Torch tree
(~/workspace/itaktorch-ingest/iTaKTorchQ4) and the rescued Core tree.

## The two surfaces, side by side

### pkg/debug

| Aspect | Shape G (merged, GOAgent port) | Shape R (rescued Core) |
| --- | --- | --- |
| Logger shape | One global singleton; package-level functions | Per-module instance: `debug.NewLogger("torch", debug.LevelInfo)` returns a `*Logger` |
| Call shape | `debug.Info(component, format, args...)` | `logger.Info(component, format, args...)` (same argument shape, on the instance) |
| Levels | `Level` int: Error 0, Warn 1, Info 2, Debug 3; `ParseLevel` | Identical numbering and `ParseLevel` |
| Configuration | `SetLevel`, `SetOutput`, `SetColor`, `EnableDebug` on the global | `SetLevel`, `SetOutput` methods per instance |
| Retention | None: lines go to stderr and are gone | Every entry, including level-filtered ones, is kept in a 1000-entry ring buffer (`LogEntry`, `RingBuffer.Recent(n, levelFilter)`) |
| HTTP diagnostics | None | `NewMux(MuxConfig{Logger, StartTime, Health, Events, Config})` serving /debug/snapshot, /debug/logs, /debug/events, /debug/health, /debug/config, and /debug/level (GET and POST), gated by `Enabled()` on the ITAK_DEBUG env var; 404 for everything when off |
| Extras | `StepLogger` workflow trace recorder (JSONL files, `Record`, `RecordTimed`, `LoadTrace`); `JSON(component, label, payload string)` | `TimeTrace` deferred timing; `JSON(component, label string, v interface{})` marshals the value itself; line prefix carries module and component (`[torch/server]`) |
| Env behavior | None | ITAK_DEBUG forces LevelDebug at construction; ITAK_DEBUG_COLOR=false disables color |

### pkg/event

| Aspect | Shape G (merged, GOAgent port) | Shape R (rescued Core) |
| --- | --- | --- |
| Bus type and constructor | `EventBus`, made by `event.New()` | `Bus`, made by `event.NewBus(historySize int)` |
| Publish | `Publish(Event)` | `Emit(Event)` |
| Subscribe | `Subscribe(bufSize, topics...) (int, <-chan Event)`, then read the channel; `Unsubscribe(id)` closes it | `On(Type, Handler) func()` and `OnAll(Handler) func()`; the returned function unsubscribes |
| Delivery | Channel fan-out, non-blocking: a full subscriber buffer drops the event for that subscriber, with a warning log | Synchronous handler callbacks in registration order; a panicking handler is recovered and dispatch continues |
| Event struct | `Topic string`, `Agent`, `Tool`, `Data map[string]interface{}`, `Message`, `Timestamp`; constructors `NewEvent`, `AgentEvent`, `ToolEvent` | `Type Type`, `Source string`, `Data interface{}`, `Timestamp`; constructor `New(typ, source, data)` |
| Topic vocabulary | Agent-workflow topics: `agent.start`, `agent.tool_call`, `memory.fact_saved`, `browser.navigate`, `user.input`, `system.ready`, and similar | Module and inference lifecycle: `module.started`, `model.loaded`, `model.unloaded`, `inference.start`, `inference.done`, `inference.error`, `task.created`, `auth.failure`, and similar |
| History | None | Bounded history ring kept by the bus; `History()` feeds late subscribers and the debug mux's /debug/events |
| Coupling | Imports core/pkg/debug for its own logging | Standalone; the debug mux consumes it through a small `EventProvider` interface, no import cycle |

## Which surface the original call sites actually use, counted

Greps over the original Torch tree (pkg/torch), the only measured consumer
of Core's debug and event packages:

### pkg/debug: Shape R, exclusively

| Use | Count | Where |
| --- | --- | --- |
| Files importing Core pkg/debug | 1 | server.go (aliased `coreDebug`) |
| Distinct Shape R symbols referenced | 6 | `Logger`, `NewLogger`, `LevelInfo`, `Enabled`, `NewMux`, `MuxConfig` |
| Logger method calls on the instance (`s.debugLogger.*`) | 16 | Info 13, Error 3; handlers_models.go 12, handlers_ops.go 3, server.go 1 |
| Shape G call sites (package-level `debug.Info(...)` and friends) | 0 | nowhere in the tree |

### pkg/event: Shape R, exclusively

| Use | Count | Where |
| --- | --- | --- |
| Files importing Core pkg/event | 3 | server.go, handlers_models.go, handlers_ops.go |
| `*event.Bus` as a field and option type | 2 | server.go (`eventBus` field, `WithEventBus`) |
| Direct `Emit(...)` call sites | 2 | the `emitEvent` helper body (server.go), handlers_ops.go |
| `event.New(...)` constructions | 2 | same two sites |
| `emitEvent(...)` helper invocations | 8 | server.go 4, handlers_models.go 4 |
| Named `event.Type*` constants referenced | 7 distinct, 8 references | TypeModelLoaded at 3 sites; TypeModuleStarted, TypeModuleStopped, TypeInferenceDone, TypeInferenceError, TypeModelUnloaded at 1 each |
| Subscribers (`On` / `OnAll`) inside the Torch tree | 0 | the bus is injected through `WithEventBus`; Torch publishes, other modules subscribe |
| `event.NewBus(...)` constructions inside the Torch tree | 0 | construction belongs to whichever process wires Core together |
| Shape G call sites (`Publish`, `Subscribe`, `EventBus`) | 0 | nowhere in the tree |

Two further counts frame the choice:

- Inside the rescued Core tree itself, no other package imports pkg/debug
  or pkg/event either. These packages were Core's offered surface to the
  ecosystem modules, and Torch is the consumer on record. Every measured
  call site in either tree, 16 logger calls and the full event wiring
  above, is written against Shape R.
- Shape G's topic vocabulary contains none of the seven constants Torch
  publishes. Adopting Shape G means inventing the module-lifecycle
  vocabulary over again inside the ported package; the vocabulary is not
  a detail the port happens to lack, it is the platform's event contract.

One wart worth recording either way: handlers_ops.go passes the raw
string `"cache.cleared"` where an `event.Type` belongs, instead of a named
constant. Under Shape R the fix is one added constant
(`TypeCacheCleared`); it is a REUSE-FIX line, not a shape question.

## Behavior differences that matter

1. **Delivery semantics.** Shape G is fire-and-forget fan-out: delivery
   is asynchronous, unordered across subscribers, and lossy under load by
   design (drop-on-full). Shape R is synchronous dispatch in registration
   order with per-handler panic isolation: nothing is dropped, but a slow
   handler stalls its emitter. The original platform was written and
   debugged against the synchronous contract; its emitters publish from
   request paths (inference.done, inference.error, model.loaded) where
   silent loss is the worse failure.
2. **History and late joiners.** Shape R's bus remembers a bounded
   history, which is what the debug mux serves at /debug/events and what
   a late-starting module catches up from. Shape G has no history at all.
3. **Payload typing.** Shape R carries `Data interface{}`: typed payloads
   that handlers type-assert (the package's own example asserts a
   types.ModelInfo). Shape G forces every payload into
   `map[string]interface{}`, so the platform's event contract becomes
   stringly-typed at exactly the seam PLAN S2 says must stay explicit.
4. **Diagnostics retention.** Shape R's logger keeps the last 1000
   entries in memory regardless of the print level, and serves them, plus
   health, events, redacted config, and a runtime level switch, over the
   gated /debug mux. That is the project's standing law in concrete form:
   every module ships with a way to pull the bug information out. Shape
   G's logger writes to stderr and retains nothing; adopting it deletes
   the /debug surface the platform already mounts, and the capability
   would have to be rebuilt later to meet the same law.
5. **Identity and isolation.** Shape R loggers are per-module instances:
   separate levels, outputs, and buffers per module in one process, with
   the module name in every line. Shape G is one process-wide singleton;
   every module shares one level and one output.
6. **What Shape G has that Shape R does not.** The StepLogger JSONL
   workflow trace recorder. It is a GOAgent agent-workflow artifact; no
   Torch or Core call site uses it. If agent tracing is ever wanted, it
   belongs on the platform/agent side, not inside Core's logger.

## Migration cost each way

### Option A: keep Shape G, adapt the platform at reattach

- Core side: no change. Lane 07's packages and their 21 test functions
  stand.
- Platform side (PLAN S9 reattach): surgery in server.go (the
  `debugLogger` field, its NewLogger construction, the `Enabled()` gate,
  and the `NewMux` mount all go or are rewritten; the /debug endpoints
  are lost), a changed `WithEventBus` signature, a rewritten `emitEvent`
  helper, typed payloads re-shaped into maps at 8 helper invocations plus
  1 direct Emit, and a new topic-constant set added to the ported package
  for the 7 lifecycle events. Three files touched, roughly a dozen call
  sites rewritten, plus a permanent capability deletion (ring buffer,
  event history, debug mux) that a later phase must re-buy to satisfy the
  diagnostics law.
- The resurrection gate then passes for these two packages only through
  an adapter layer, which is the relative-fudge pattern the project
  already buried once.

### Option B: adopt Shape R, replace the two ported packages

- Core side: core/pkg/debug and core/pkg/event are replaced by the
  rescued originals (4 files: debug.go, handler.go, bus.go, types.go),
  carried in under the repo's ground rule from SALVAGE.md itself: nothing
  counts as reused until the gates run green on it. The rescued packages
  arrive with no tests, so the real cost is test authorship: lane 07's
  suites port across where the behavior matches (level filtering,
  component tagging, publish/subscribe semantics), and new suites cover
  what only Shape R has (ring buffer ordering and level filter, mux
  gating and endpoints, handler panic isolation, history bounds). The
  StepLogger is dropped with the ported debug package, recorded, not
  silently lost.
- Platform side: zero call-site changes. server.go, handlers_models.go,
  and handlers_ops.go compile as written; only the module path changes,
  which changes under either option. The resurrection gate passes for
  these two packages with no adapter.
- Record side: one DECISIONS.md entry (drafted below), plus the PLAN S4
  and TODO T-1.2 wording amended in the same pass, per this repo's rule
  that a decision and the plan change it causes land together.

### A facade is not a third way

A compatibility shim (Shape R names over Shape G internals, or the
reverse) leaves two vocabularies, the wrong delivery semantics under the
familiar names, and a translation layer every future reader must
mentally subtract. It costs most of Option A's surgery and buys none of
Option B's honesty. Not recommended.

## Recommendation

Adopt Shape R: the rescued Core's pkg/debug and pkg/event become the
canonical itak-core surface, replacing lane 07's GOAgent ports, with the
test suites written to the repo standard before the packages count as
landed, and PLAN S4 / TODO T-1.2 amended in the same pass.

Reasoning, in order of weight:

1. **The gate.** PLAN S4's own proof is that the original Torch tree
   compiles against the rebuilt Core. Every measured call site (16 logger
   calls, 8 event helper invocations, the debug mux mount, the
   `WithEventBus` option) is written against Shape R, and none is written
   against Shape G, in either tree. Option B is the only option under
   which these two packages pass the gate as the gate was stated.
2. **The diagnostics law.** Shape R is the one carrying the ring buffer
   and the gated HTTP debug mux, the mechanism this project requires of
   every module. Option A deletes a shipped, wired capability and defers
   re-buying it.
3. **The documents' own timeline.** PLAN S4's port-from-GOAgent
   instruction was written while Core was lost; it says to port from
   "their surviving ancestors" because the originals were not in hand.
   SALVAGE.md was written after the recovery, with the audits, and judged
   the recovered packages REUSE. When a later, evidence-based inventory
   contradicts an earlier assumption made in the dark, the later document
   should win, and the decision record should say so plainly.
4. **Cost asymmetry.** Option B's cost is bounded test authorship in two
   small packages (about 900 lines of source between them). Option A's
   cost is surgery in the server core at reattach, a re-shaped event
   contract across the platform, and the loss of the debug surface.

## Draft DECISIONS.md entry (ready to paste, newest first)

```markdown
## 2026-10-29: Core pkg/debug and pkg/event take the rescued original Core's shape
- **Decided:** The canonical itak-core surface for pkg/debug and pkg/event is the rescued original Core's shape: per-module `NewLogger` with ring buffer and the gated /debug HTTP mux, and the handler-callback `Bus` with `On`/`Emit` and bounded history. Lane 07's GOAgent ports of the two packages are replaced by the rescued originals, which land only with full test suites per the repo gates. PLAN S4 and TODO T-1.2 are amended in the same pass. The GOAgent StepLogger trace recorder is retired with the port; agent-workflow tracing, if wanted, is platform-side work, not Core logger surface.
- **Why:** PLAN S4's resurrection gate is that the original Torch tree compiles against the rebuilt Core, and every measured call site uses the rescued shape: 16 logger method calls, 8 emitEvent invocations, the NewMux mount, and the WithEventBus option across server.go, handlers_models.go, and handlers_ops.go, with zero call sites anywhere for the GOAgent surface. The rescued logger also carries the ring buffer and debug mux the project's diagnostics law requires of every module. SALVAGE.md, written after the original Core was recovered on 2026-10-28 with the audits in hand, marks both packages REUSE; the PLAN instruction to port from GOAgent ancestors was written while Core was still lost.
- **Supersedes:** the port-from-GOAgent direction for pkg/debug and pkg/event in PLAN S4 and TODO T-1.2 (the rest of S4, including the pkg/types, pkg/health, and pkg/registry call-site shapes, stands unchanged)
- **By:** David
```
