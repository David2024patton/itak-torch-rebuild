# CONTRACT: the Go/Rust IPC seam (ARCHITECTURE.md Section 4, TODO T-1.6)

This crate is the contract in code. The Go mirror lives in
`platform/internal/ipc` (the documented `pkg/ipc` slot, under
`internal/` until the itak-core module of T-1.1 exists to publish
it). Both sides compile against the shapes described here; if this
note and the code disagree, the code wins and the note is the bug.

## Transport

- Control: a Unix domain socket (Linux) or named pipe (Windows).
  The socket never carries tensors.
- Data: shared-memory segments, referenced by `ShmRef` handles with
  explicit ownership; the receiver releases a segment with
  `ReleaseShm`. One token ring per active generation (below).

## Frame layout

Every frame is a fixed 16-byte header plus a payload. All header
integers are little-endian.

| Offset | Size | Field |
| :--- | :--- | :--- |
| 0 | 2 | Protocol version (u16) |
| 2 | 2 | Message type (u16, codes below) |
| 4 | 8 | Request id (u64); replies and refusals echo it |
| 12 | 4 | Payload length (u32) |

Payloads are the contract shapes serialized in the one agreed
encoding: JSON, UTF-8 (Go: `encoding/json`; Rust: the std-only
codec in `src/json.rs`). Optional fields are omitted from the JSON
object when absent, never null, on both sides. Integers keep their
exact JSON integer form; only fractional fields (temperatures,
logprobs, timings) travel as JSON fractions.

## Version rule

The protocol version is a u16: major in the high byte, minor in the
low byte. The current version is `0x0100` (1.0) on both sides
(`PROTOCOL_VERSION`).

- The first exchange on a connection is `Hello` / `HelloAck`: each
  side states its protocol version and capability list; the ack
  also carries the engine's hardware report summary.
- Majors must match. Minor differences are compatible and never
  refuse.
- A major mismatch is a named refusal, code
  `protocol_version_mismatch`, on both sides: the refusing side
  sends a `Refused` frame carrying that code and its own protocol
  version, returns/returns-as-error `ProtocolVersionMismatch`
  (Rust) / `ErrProtocolVersionMismatch` (Go), and the exchange
  ends. Never a hang, never a silent downgrade.

Stable refusal codes live in `src/diag.rs` (`codes`) on the Rust
side and as `Code*` constants on the Go side. They are part of the
contract and never change spelling.

## Message set (codes never renumber)

| Code | Message | Direction | Reply |
| :--- | :--- | :--- | :--- |
| 1 | Hello | both | HelloAck or Refused |
| 2 | HelloAck | engine to platform | |
| 3 | Refused | both | |
| 4 | Ack | engine to platform | |
| 10 | LoadModel | platform to engine | LoadResult |
| 11 | LoadResult | engine to platform | |
| 12 | UnloadModel | platform to engine | Ack |
| 20 | CreateSession | platform to engine | SessionCreated |
| 21 | SessionCreated | engine to platform | |
| 22 | DestroySession | platform to engine | Ack |
| 30 | Tokenize | platform to engine | TokenizeResult |
| 31 | TokenizeResult | engine to platform | |
| 32 | Detokenize | platform to engine | DetokenizeResult |
| 33 | DetokenizeResult | engine to platform | |
| 40 | Generate | platform to engine | TokenEvent stream |
| 41 | TokenEvent | engine to platform | ring wakeup |
| 42 | Abort | platform to engine | Ack |
| 50 | Embed | platform to engine | EmbedResult |
| 51 | EmbedResult | engine to platform | |
| 52 | ReleaseShm | platform to engine | Ack |
| 60 | SaveState | platform to engine | StateResult |
| 61 | LoadState | platform to engine | StateResult |
| 62 | StateResult | engine to platform | |
| 70 | Health | platform to engine | Diagnostics |
| 71 | Diagnostics | engine to platform | |
| 80 | Shutdown | platform to engine | Ack |

`LoadResult` is a tagged union (`status`: `loaded` with handle,
model info, and the coverage report; `refused` with the named load
error). `StateResult` is the same shape (`done` / `refused`).

The mirrored contracts from `pkg/types` (TODO T-1.1's documented
surface: `InferenceRequest`, `InferenceResponse`, `ChatMessage`,
`Role`, `TokenUsage`, `LatencyStats`, `ModelInfo`, `ModelParams`)
live in `src/types.rs`. Every mirrored struct carries a `version`
field stamped with `SHAPE_VERSION`; field-level parity locks
against the T-1.1 module when it lands. The Go `pkg/ipc` package
deliberately does not duplicate them; it carries only IPC-facing
types (plus the `ModelInfo` copy inside `LoadResult`, which is on
the wire).

## Limits (config with defaults, never compiled constants)

`SeamLimits` (Rust) / reader arguments (Go), defaults identical:

| Limit | Default | Meaning |
| :--- | :--- | :--- |
| max frame length | 1 MiB | A header announcing more is the named `frame_too_large` refusal, raised before the payload is read |
| ring capacity | 1024 events | Per-generation token ring size |
| hello timeout | 5000 ms | Callers apply it as the socket timeout around the handshake |

## The token ring: what is real, what is stubbed

Real today: the ring contract as the `TokenRing` trait plus
`MemRing`, an in-process bounded ring with the contract's exact
semantics (FIFO, capacity from config, a full ring returns the
named `ring_full` refusal so the producer applies backpressure;
tokens are never dropped or overwritten).

Stubbed behind that trait: the cross-process mmap shared-memory
ring itself. It lands with the engine and platform wiring that own
the segments, behind the same trait, without the contract
changing. The T-1.6 round-trip test runs the real socket Hello,
a Tokenize exchange, and token events through a `MemRing` shared
by both sides, then the wakeup frame on the socket.

## Proof (as of this landing)

- Rust: `cargo test -p itak-types` — 9 integration tests: Hello
  round trip over a real Unix socket with the ring; version refusal
  by the server, by the client, and named on the wire; minor
  difference accepted; header layout byte-for-byte; oversize
  refusal; ring backpressure; all 26 message types round-tripping
  (27 messages, LoadResult in both shapes). Clippy clean under the
  workspace deny-all lints.
- Go: `go test ./internal/ipc/...` — header layout and round trip,
  the named version refusal, JSON field-name parity with the Rust
  encoder, a Hello exchange over a real Unix socket, oversize
  refusal. `golangci-lint run ./...` clean for the platform module.
