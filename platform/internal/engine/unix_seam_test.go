package engine_test

// Regression tests for the critic panel's two socket-seam HIGHs
// (bench/socket-seam-fixes-2026-11-07.md):
//
//  1. The dial never authenticated the engine end of the socket: a
//     squatter on the path answered the public handshake and
//     received prompt traffic. The dial now verifies the socket
//     file first (socketTrustEngine below plays the squatter: a
//     listener whose socket file is NOT pinned to 0600, exactly
//     what net.Listen leaves behind, versus the real engine, which
//     chmods its socket to 0600 after binding).
//  2. A timed-out exchange left its late reply buffered, and the
//     next exchange consumed the stale frame: the first Health
//     after a timeout reported a healthy engine down, and a
//     timed-out Embed's late vectors could be returned to the next
//     Embed caller. The client now retires the connection on any
//     desynchronizing read failure, and these tests pin that the
//     call after the failure cannot consume the stale reply.

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/David2024patton/itak-torch-rebuild/platform/internal/engine"
)

// socketTrustEngine serves the protocol on a fresh socket,
// optionally pinned to 0600 the way the real engine pins its socket
// after binding (bind_socket in crates/itak-engine/src/main.rs). A
// scripted handler may delay an op's reply to stage a timeout.
func socketTrustEngine(t *testing.T, ownerOnly bool, delay map[string]time.Duration) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "e.sock")
	ln, err := net.Listen("unix", path)
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	t.Cleanup(func() { _ = ln.Close() })
	if ownerOnly {
		if err := os.Chmod(path, 0o600); err != nil {
			t.Fatalf("chmod socket: %v", err)
		}
	}
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go serveTrustConn(conn, delay)
		}
	}()
	return path
}

func serveTrustConn(conn net.Conn, delay map[string]time.Duration) {
	defer func() { _ = conn.Close() }()
	r := bufio.NewReader(conn)
	write := func(m map[string]any) bool {
		out, _ := json.Marshal(m)
		_, err := conn.Write(append(out, '\n'))
		return err == nil
	}
	for {
		line, err := r.ReadBytes('\n')
		if err != nil {
			return
		}
		var f struct {
			Op    string   `json:"op"`
			Input []string `json:"input"`
		}
		if json.Unmarshal(line, &f) != nil {
			return
		}
		if d := delay[f.Op]; d > 0 {
			time.Sleep(d)
		}
		switch f.Op {
		case "hello":
			if !write(map[string]any{"op": "hello_ack", "proto": 1}) {
				return
			}
		case "ping":
			if !write(map[string]any{"op": "pong"}) {
				return
			}
		case "debug":
			if !write(map[string]any{"op": "debug_info", "model": "fake-1b", "generations": 3}) {
				return
			}
		case "embed":
			vecs := make([][]float32, 0, len(f.Input))
			for _, in := range f.Input {
				// The vector encodes the input, so a stale reply is
				// distinguishable from this request's own reply.
				vecs = append(vecs, []float32{float32(len(in))})
			}
			if !write(map[string]any{"op": "embeddings", "model": "fake-1b", "embeddings": vecs}) {
				return
			}
		default:
			if !write(map[string]any{"op": "error", "code": "unknown_op", "message": "unknown op"}) {
				return
			}
		}
	}
}

func dialCtx() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), 5*time.Second)
}

// TestDialRefusesSocketWithLoosePermissions is the mode arm of the
// trust check, executable end to end: a listener that never pinned
// its socket to 0600 (a squatter, or any non-engine program) is
// refused by name before the handshake, even though it answers the
// protocol correctly.
func TestDialRefusesSocketWithLoosePermissions(t *testing.T) {
	path := socketTrustEngine(t, false, nil)
	ctx, cancel := dialCtx()
	defer cancel()
	_, err := engine.DialUnix(ctx, path)
	if !errors.Is(err, engine.ErrUntrustedSocket) {
		t.Fatalf("DialUnix err = %v, want ErrUntrustedSocket", err)
	}
	if errors.Is(err, engine.ErrUnavailable) {
		t.Fatalf("DialUnix err = %v, must not read as a missing engine", err)
	}
}

// TestDialAcceptsOwnerOnlySocket is the happy path: the socket
// pinned to 0600 by its owner dials, handshakes, and serves.
func TestDialAcceptsOwnerOnlySocket(t *testing.T) {
	path := socketTrustEngine(t, true, nil)
	ctx, cancel := dialCtx()
	defer cancel()
	c, err := engine.DialUnix(ctx, path)
	if err != nil {
		t.Fatalf("DialUnix: %v", err)
	}
	defer func() { _ = c.Close() }()
	h, err := c.Health(context.Background())
	if err != nil || !h.OK {
		t.Fatalf("Health = %+v, %v; want OK=true", h, err)
	}
}

// TestDialMissingSocketIsUnavailableNotUntrusted keeps the two dial
// failures distinct: nothing at the path is the supervisor's
// startup race (retry), a file that fails verification is a trust
// refusal.
func TestDialMissingSocketIsUnavailableNotUntrusted(t *testing.T) {
	ctx, cancel := dialCtx()
	defer cancel()
	_, err := engine.DialUnix(ctx, filepath.Join(t.TempDir(), "absent.sock"))
	if !errors.Is(err, engine.ErrUnavailable) {
		t.Fatalf("DialUnix err = %v, want ErrUnavailable", err)
	}
	if errors.Is(err, engine.ErrUntrustedSocket) {
		t.Fatalf("DialUnix err = %v, must not be a trust refusal", err)
	}
}

// TestDialRefusesNonSocketFile: a regular file at the socket path
// is not an engine, whatever its owner and mode.
func TestDialRefusesNonSocketFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "not-a-sock")
	if err := os.WriteFile(path, []byte("x"), 0o600); err != nil {
		t.Fatalf("write file: %v", err)
	}
	ctx, cancel := dialCtx()
	defer cancel()
	_, err := engine.DialUnix(ctx, path)
	if !errors.Is(err, engine.ErrUntrustedSocket) {
		t.Fatalf("DialUnix err = %v, want ErrUntrustedSocket", err)
	}
}

// TestTimeoutPoisonsControlConnection is the panel's seat-3 probe
// as a regression test: the engine answers debug 400 ms late, the
// caller gives up at 100 ms, and the late debug_info frame lands in
// the client's reader. Before the fix, the next Health consumed
// that stale frame and reported the healthy engine down. Now the
// timed-out exchange retires the client: Health reports the client
// closed (the named, honest state), never the stale frame's lie.
func TestTimeoutPoisonsControlConnection(t *testing.T) {
	path := socketTrustEngine(t, true, map[string]time.Duration{"debug": 400 * time.Millisecond})
	ctx, cancel := dialCtx()
	defer cancel()
	c, err := engine.DialUnix(ctx, path)
	if err != nil {
		t.Fatalf("DialUnix: %v", err)
	}
	defer func() { _ = c.Close() }()

	short, cancelShort := context.WithTimeout(context.Background(), 100*time.Millisecond)
	_, err = c.DebugInfo(short)
	cancelShort()
	if !errors.Is(err, engine.ErrTimeout) {
		t.Fatalf("DebugInfo err = %v, want ErrTimeout", err)
	}

	h, herr := c.Health(context.Background())
	if herr != nil {
		t.Fatalf("Health err = %v, want nil (Health reports state, not errors)", herr)
	}
	if h.OK {
		t.Fatalf("Health = %+v, want not-OK: the client must be retired, not answering from a desynced stream", h)
	}
	if h.Detail != "client closed" {
		t.Fatalf("Health detail = %q, want %q (the stale debug_info frame must never be consumed)", h.Detail, "client closed")
	}

	// Exchanges on the retired client refuse by name.
	_, err = c.DebugInfo(context.Background())
	if !errors.Is(err, engine.ErrClosed) {
		t.Fatalf("DebugInfo after poison err = %v, want ErrClosed", err)
	}
}

// TestEmbedAfterTimeoutNeverReturnsStaleVectors covers the worse
// half of the panel's finding: two consecutive Embed calls share
// the reply op, so before the fix a timed-out Embed's late vectors
// were silently returned to the NEXT Embed caller as its result.
// Now the second call refuses with ErrClosed, and a fresh dial
// returns each request its OWN vectors.
func TestEmbedAfterTimeoutNeverReturnsStaleVectors(t *testing.T) {
	path := socketTrustEngine(t, true, map[string]time.Duration{"embed": 400 * time.Millisecond})
	ctx, cancel := dialCtx()
	defer cancel()
	c, err := engine.DialUnix(ctx, path)
	if err != nil {
		t.Fatalf("DialUnix: %v", err)
	}
	defer func() { _ = c.Close() }()

	short, cancelShort := context.WithTimeout(context.Background(), 100*time.Millisecond)
	_, err = c.Embed(short, engine.EmbedRequest{Model: "fake-1b", Input: []string{"aaaa"}})
	cancelShort()
	if !errors.Is(err, engine.ErrTimeout) {
		t.Fatalf("Embed err = %v, want ErrTimeout", err)
	}

	_, err = c.Embed(context.Background(), engine.EmbedRequest{Model: "fake-1b", Input: []string{"bb"}})
	if !errors.Is(err, engine.ErrClosed) {
		t.Fatalf("Embed after timeout err = %v, want ErrClosed (the late reply of the first request must never answer the second)", err)
	}

	// The supervisor's recovery shape: a fresh client on the same
	// socket serves normally, and each request gets its own reply.
	c2, err := engine.DialUnix(ctx, path)
	if err != nil {
		t.Fatalf("redial: %v", err)
	}
	defer func() { _ = c2.Close() }()
	resp, err := c2.Embed(context.Background(), engine.EmbedRequest{Model: "fake-1b", Input: []string{"bb"}})
	if err != nil {
		t.Fatalf("Embed on fresh client: %v", err)
	}
	if len(resp.Vectors) != 1 || len(resp.Vectors[0]) != 1 || resp.Vectors[0][0] != 2 {
		t.Fatalf("Embed vectors = %v, want [[2]] (this request's own reply)", resp.Vectors)
	}
	h, err := c2.Health(context.Background())
	if err != nil || !h.OK {
		t.Fatalf("Health on fresh client = %+v, %v; want OK=true", h, err)
	}
}
