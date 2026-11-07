# Socket seam fixes: engine authentication and stale-frame recovery (2026-11-07)

Lane `lane/socket-seam-auth-recovery`, from main 85beb8a.
Two HIGH findings from the cold critic panel
(`~/workspace/reports/critic-engine-2026-11-07/CRITIC-ENGINE-MERGE.md`,
Category D), both on the Go platform's Unix-socket seam to the Rust
engine. The Failure seat's verdict for this path was NO-SHIP on its
own; the Security seat's was DO NOT SHIP until these two are fixed.

## Finding 1 (HIGH, Security seat): the dial never authenticated the engine

`platform/internal/engine/unix.go` (dialRaw/handshake): the platform
checked only that something at the socket path answered `hello_ack`
with proto 1. The protocol shapes are public, and the default path
is the predictable `/tmp/itak-engine.sock`, so a squatter bound
there during a restart window received all prompt traffic and could
forge completions. The Rust engine authenticates its side
(`bind_socket` pins the file to 0600; `check_peer` refuses a peer
whose SO_PEERCRED uid differs); the platform authenticated nothing.

### Fix

Before any dial (the control connection in `DialUnix` and
Generate's per-call connection share `dialRaw`, so no path reaches
the peer unchecked), `checkSocketTrust` stats the socket file and
refuses by name with the new `ErrUntrustedSocket` unless all three
hold: the file is a socket, its owner uid equals this process's
uid, and its permissions are exactly 0600. This mirrors the engine
side's two facts: the mode it pins, and uid equality, which is the
same predicate `check_peer` enforces from the other direction.

Which layers were built, and why: the panel offered the stat check
or SO_PEERCRED via `SyscallConn` + `GetsockoptUcred`, and preferred
both if the ucred read is clean in this codebase's dependency set.
It is not: `platform/go.mod` is standard-library only, and
`GetsockoptUcred` lives in `golang.org/x/sys/unix`, so the peer-cred
read would add the platform's first external dependency for one
syscall. The stat check alone is the panel's stated minimum and is
built. The residual stat-to-dial race is narrower than it looks: a
replacement socket file at the path can only be created by a
same-uid process (or root), because binding requires unlinking or
creating the directory entry, and the engine's path lives where
only the owner can do that in supported deployments; a same-uid
impostor is inside the trust boundary the engine itself accepts
(its peer check passes any same-uid process). The check lives in
`socket_unix.go` (`//go:build unix`), with the pure predicate
`socketTrustOK` split out for unit tests; `socket_other.go` keeps
non-unix builds verifying the arms they can read (socket type and
mode).

Supervisor interaction: `dialUntil` retries ANY dial error until
its deadline and then fails the launch by name, so a trust refusal
surfaces as an error and a restart-budget entry, never as a hang
and never as "engine is serving". The refusal is deliberately NOT
wrapped in `ErrUnavailable`: the retry behavior is identical, but
the supervisor's log line names the trust failure instead of
reading as a slow engine start. `cmd/torchd`'s one-shot `dialEngine`
treats it the same way (starts with the engine marked unavailable
and logs the named error).

One behavior consequence, stated plainly: the platform now refuses
any engine socket not pinned to 0600. The real engine has always
pinned its socket, so supported deployments are unaffected; test
fakes that stand in for the engine now pin theirs too (see Tests).

## Finding 2 (HIGH, Failure seat, probe-proven): stale frame after a timeout

`exchangeAnyLocked` read replies from the shared buffered reader
with no recovery: when an exchange timed out, the engine's late
reply stayed buffered, and the next exchange consumed it as its
own. The seat's executed probe: after a 100 ms-timeout `DebugInfo`
against an engine answering at 400 ms, the first `Health` ate the
stale `debug_info` frame and reported the healthy engine down
(`got op "debug_info", want "pong"`). Worse, two consecutive Embed
calls share the reply op, so a timed-out Embed's late vectors could
be returned to the next Embed caller as its result, silently.

### Fix

The control connection is now retired on any read failure that can
leave its frame stream desynchronized:

- a read timeout (the reply may still be in flight);
- a partial frame consumed before the failure (its remainder is
  in flight);
- an oversized frame (its unread tail is still in the stream);
- a failed write (the peer's state for that frame is unknowable).

`readFrame` now reports whether the failed read consumed a partial
frame, and `exchangeAnyLocked` poisons the client in the cases
above: the connection closes, the client is marked closed, the
failing call keeps its existing named error (`ErrTimeout`,
`ErrFrameTooLarge`, `ErrHandshake` as classified today), and every
later call refuses with `ErrClosed`. A clean refusal before any
byte is consumed (the engine hung up between requests) does NOT
poison: nothing can be stale, and the next exchange fails or
succeeds on its own. Generate's per-call connections are untouched
beyond the shared `readFrame` signature: a generate stream owns
its connection end to end, so no other caller can consume its
frames.

Supervisor interaction: the supervisor's serving path returns
whatever the live client returns, so a poisoned client surfaces
`ErrClosed` by name (Health reports `client closed`, the same
honest state as after an explicit Close) instead of wrong answers.
The supervisor drops its client when the incarnation dies and
redials on the next launch, which is the existing recovery path;
the regression tests pin that a fresh dial on the same socket
serves normally and returns each request its own reply. What does
not exist, before or after this lane: supervisor detection of a
poisoned-but-process-alive client between incarnation deaths (the
engine process is healthy in the timeout scenario, so no death
fires a redial). That is recorded below as an adjacent gap, not
silently claimed as covered.

## Evidence: fails before, passes after

The panel's probe was reproduced first, in its exact shape (fake
engine answers `debug` after 400 ms; caller context 100 ms; then
`Health`), as a scratch test against the unfixed tree:

- Before: `DebugInfo` returns `ErrTimeout`; the next `Health`
  returns `ok=false, "engine: handshake failed: got op
  \"debug_info\", want \"pong\""` — the healthy engine reported
  down by the stale frame. BUG REPRODUCED, matching the seat's
  probe output.
- After: the same probe's `Health` returns `ok=false, "client
  closed"` — the client was retired by the timed-out exchange and
  the stale frame was never consumed. (The scratch probe was
  deleted after serving as the before/after witness; its shape
  lives on as `TestTimeoutPoisonsControlConnection`.)
- Fails-before for the new dial tests cannot be a compile-clean
  run on the unfixed tree (they name the new `ErrUntrustedSocket`),
  so the before-state evidence for Finding 1 is the seat's code
  read plus the behavioral fact the probe relies on: on the
  unfixed tree the squatter-shaped listener (no 0600 pin) dials
  and handshakes successfully — every pre-existing fake-engine
  test did exactly that, which is why the fakes had to learn the
  pin (below).

## Tests

New, in `platform/internal/engine/unix_seam_test.go` and
`socket_unix_test.go`:

- `TestDialRefusesSocketWithLoosePermissions`: a protocol-correct
  listener whose socket file is not pinned to 0600 refuses with
  `ErrUntrustedSocket`, and not as `ErrUnavailable`.
- `TestDialAcceptsOwnerOnlySocket`: the 0600 happy path dials,
  handshakes, and reports healthy.
- `TestDialMissingSocketIsUnavailableNotUntrusted`: the two dial
  failures stay distinct (startup race vs trust refusal).
- `TestDialRefusesNonSocketFile`: a regular file at 0600 refuses.
- `TestSocketTrustOK`: the predicate arm by arm (the uid arm is
  pinned here because a test process cannot fabricate a socket
  file owned by another uid), and `TestCheckSocketTrustArms` runs
  the stat wrapper under the owner's uid (pass) and a foreign uid
  (refuse).
- `TestTimeoutPoisonsControlConnection`: the probe as a
  regression — timeout, then Health reports `client closed`,
  then exchanges refuse with `ErrClosed`.
- `TestEmbedAfterTimeoutNeverReturnsStaleVectors`: a timed-out
  Embed's late reply can never answer the next Embed; a fresh
  dial returns each request its own vectors.

Adjusted, with reasons:

- The fake engines in `engine_test.go` (`fakeEngine`,
  `capturingEngine`, `noTemplateEngine`), `unix_bounds_test.go`
  (`scriptedEngine`), and `supervisor_test.go`'s fake engine
  process now chmod their sockets to 0600 after binding, modeling
  the real engine; without it the new check refuses them, which
  is the check working as designed.
- `TestOversizedReplyFrameRefusedByName`: its secondary surface
  asserted Health's detail would name the frame cap after an
  oversized reply. Under the stale-frame rule the oversized
  frame's tail retires the client, so Health now reports
  `client closed`; the primary assertion (DebugInfo refuses with
  `ErrFrameTooLarge`) is unchanged, and the test now also pins
  that the tail is never consumed as a reply.

## Gates (lane tree, as observed)

- `go build ./...`: clean.
- `go vet ./...`: clean.
- `gofmt -l` on the platform tree: empty.
- `go test ./...`: all 9 packages ok; 107 tests pass (base 99,
  +8 new in this lane).
- `go test -race ./internal/engine/ ./internal/supervisor/`:
  clean.
- `golangci-lint run ./...` (the README's gate, config
  `.golangci.yml`): 0 issues. One gosec G703 taint note on the
  supervisor test fake's new chmod was resolved with a scoped
  `//nolint:gosec` carrying the same justification as the
  adjacent `os.Remove` (test-controlled path from the test's own
  env).
- `govulncheck ./...`: no vulnerabilities found.
- Cross-builds of the touched package (windows, darwin, freebsd):
  all build, so the non-unix fallback path compiles.
- `cargo check --workspace`: clean (the tree is shared; no Rust
  file changed in this lane).

## Adjacent, deliberately NOT fixed in this lane

- Supervisor redial of a poisoned-but-alive client (above): the
  client now fails honestly by name instead of serving stale
  answers, which closes the panel's wrong-answer finding; wiring
  the supervisor to replace a client that reports `ErrClosed`
  while its process lives is a supervisor change, separate lane.
- The panel's preferred default-path hardening (a 0700 socket
  directory instead of bare `/tmp`): an engine/operator default
  change, not required to close the authentication hole once the
  file itself is verified.
- Category D's MEDIUMs (stderr pump wedge, Stop/launch race,
  engine connection cap, plaintext HTTP warning) are separate
  findings with their own lanes.
