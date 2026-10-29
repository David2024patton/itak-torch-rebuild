package engine_test

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"net"
	"path/filepath"
	"testing"
	"time"

	"github.com/David2024patton/itak-torch-rebuild/platform/internal/engine"
)

func TestUnavailableRefusesEverything(t *testing.T) {
	var c engine.Client = engine.Unavailable{}
	ctx := context.Background()

	if _, err := c.Generate(ctx, engine.GenerateRequest{}); !errors.Is(err, engine.ErrUnavailable) {
		t.Fatalf("Generate err = %v, want ErrUnavailable", err)
	}
	if _, err := c.Embed(ctx, engine.EmbedRequest{}); !errors.Is(err, engine.ErrUnavailable) {
		t.Fatalf("Embed err = %v, want ErrUnavailable", err)
	}
	h, err := c.Health(ctx)
	if err != nil || h.OK {
		t.Fatalf("Health = %+v, %v; want OK=false, nil error", h, err)
	}
}

// fakeEngine serves the socket protocol with configurable replies.
func fakeEngine(t *testing.T, ackProto int, ackOp string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "engine.sock")
	ln, err := net.Listen("unix", path)
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	t.Cleanup(func() { _ = ln.Close() })
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go serveConn(conn, ackProto, ackOp)
		}
	}()
	return path
}

func serveConn(conn net.Conn, ackProto int, ackOp string) {
	defer func() { _ = conn.Close() }()
	r := bufio.NewReader(conn)
	for {
		line, err := r.ReadBytes('\n')
		if err != nil {
			return
		}
		var f struct {
			Op string `json:"op"`
		}
		if err := json.Unmarshal(line, &f); err != nil {
			return
		}
		var reply map[string]any
		switch f.Op {
		case "hello":
			reply = map[string]any{"op": ackOp, "proto": ackProto}
		case "ping":
			reply = map[string]any{"op": "pong"}
		default:
			reply = map[string]any{"op": "error", "message": "unknown op"}
		}
		out, _ := json.Marshal(reply)
		if _, err := conn.Write(append(out, '\n')); err != nil {
			return
		}
	}
}

func dial(t *testing.T, path string) *engine.UnixClient {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	c, err := engine.DialUnix(ctx, path)
	if err != nil {
		t.Fatalf("DialUnix: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return c
}

func TestUnixHandshakeAndHealth(t *testing.T) {
	c := dial(t, fakeEngine(t, 1, "hello_ack"))
	h, err := c.Health(context.Background())
	if err != nil || !h.OK {
		t.Fatalf("Health = %+v, %v; want OK=true", h, err)
	}
}

func TestUnixGenerateIsNamedNotImplemented(t *testing.T) {
	c := dial(t, fakeEngine(t, 1, "hello_ack"))
	if _, err := c.Generate(context.Background(), engine.GenerateRequest{}); !errors.Is(err, engine.ErrNotImplemented) {
		t.Fatalf("Generate err = %v, want ErrNotImplemented", err)
	}
	if _, err := c.Embed(context.Background(), engine.EmbedRequest{}); !errors.Is(err, engine.ErrNotImplemented) {
		t.Fatalf("Embed err = %v, want ErrNotImplemented", err)
	}
}

func TestUnixHandshakeRejectsWrongProtocol(t *testing.T) {
	path := fakeEngine(t, 99, "hello_ack")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := engine.DialUnix(ctx, path); !errors.Is(err, engine.ErrHandshake) {
		t.Fatalf("DialUnix err = %v, want ErrHandshake", err)
	}
}

func TestUnixHandshakeRejectsWrongOp(t *testing.T) {
	path := fakeEngine(t, 1, "nope")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := engine.DialUnix(ctx, path); !errors.Is(err, engine.ErrHandshake) {
		t.Fatalf("DialUnix err = %v, want ErrHandshake", err)
	}
}

func TestUnixDialMissingSocketIsUnavailable(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	missing := filepath.Join(t.TempDir(), "absent.sock")
	if _, err := engine.DialUnix(ctx, missing); !errors.Is(err, engine.ErrUnavailable) {
		t.Fatalf("DialUnix err = %v, want ErrUnavailable", err)
	}
}

func TestUnixUseAfterClose(t *testing.T) {
	c := dial(t, fakeEngine(t, 1, "hello_ack"))
	if err := c.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := c.Health(context.Background()); err != nil {
		t.Fatalf("Health after close should report, not error: %v", err)
	}
	// Second Close is a no-op, not an error.
	if err := c.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}
