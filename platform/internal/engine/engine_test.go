package engine_test

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"net"
	"path/filepath"
	"strings"
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
	write := func(reply map[string]any) bool {
		out, _ := json.Marshal(reply)
		_, err := conn.Write(append(out, '\n'))
		return err == nil
	}
	for {
		line, err := r.ReadBytes('\n')
		if err != nil {
			return
		}
		var f struct {
			Op     string `json:"op"`
			Prompt string `json:"prompt"`
		}
		if err := json.Unmarshal(line, &f); err != nil {
			return
		}
		switch f.Op {
		case "hello":
			if !write(map[string]any{"op": ackOp, "proto": ackProto}) {
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
		case "generate":
			if !serveGenerate(write, f.Prompt) {
				return
			}
		default:
			if !write(map[string]any{"op": "error", "message": "unknown op"}) {
				return
			}
		}
	}
}

// serveGenerate scripts the generate stream off the prompt text, so
// one fake covers the success path and both error placements.
func serveGenerate(write func(map[string]any) bool, prompt string) bool {
	switch {
	case strings.Contains(prompt, "err-overflow"):
		return write(map[string]any{
			"op": "error", "code": "context_overflow",
			"message": "prompt of 9 tokens plus max_tokens 4 exceeds the context length 8",
		})
	case strings.Contains(prompt, "err-mid"):
		if !write(map[string]any{"op": "token", "text": " Paris", "token_id": 13796}) {
			return false
		}
		return write(map[string]any{
			"op": "error", "code": "forward_failed", "message": "decode step 2: boom",
		})
	default:
		if !write(map[string]any{"op": "token", "text": " Paris", "token_id": 13796}) {
			return false
		}
		if !write(map[string]any{"op": "token", "text": ". It", "token_id": 26}) {
			return false
		}
		return write(map[string]any{
			"op": "token", "text": "", "token_id": 0, "done": true, "finish_reason": "stop",
		})
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

func TestUnixGenerateStreamsTokens(t *testing.T) {
	c := dial(t, fakeEngine(t, 1, "hello_ack"))
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	ch, err := c.Generate(ctx, engine.GenerateRequest{Model: "fake-1b", Prompt: "The capital of France is"})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	var events []engine.TokenEvent
	for ev := range ch {
		events = append(events, ev)
	}
	if len(events) != 3 {
		t.Fatalf("got %d events, want 3: %+v", len(events), events)
	}
	if events[0].Text != " Paris" || events[0].TokenID != 13796 || events[0].Done {
		t.Fatalf("event 0 = %+v, want the Paris token event", events[0])
	}
	if events[1].Text != ". It" || events[1].TokenID != 26 {
		t.Fatalf("event 1 = %+v", events[1])
	}
	if !events[2].Done || events[2].FinishReason != "stop" {
		t.Fatalf("event 2 = %+v, want done with finish_reason stop", events[2])
	}
}

func TestUnixGenerateStartErrorIsBadRequest(t *testing.T) {
	c := dial(t, fakeEngine(t, 1, "hello_ack"))
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := c.Generate(ctx, engine.GenerateRequest{Prompt: "err-overflow please"})
	if !errors.Is(err, engine.ErrBadRequest) {
		t.Fatalf("Generate err = %v, want ErrBadRequest", err)
	}
}

func TestUnixGenerateMidStreamErrorClosesWithoutDone(t *testing.T) {
	c := dial(t, fakeEngine(t, 1, "hello_ack"))
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	ch, err := c.Generate(ctx, engine.GenerateRequest{Prompt: "err-mid please"})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	var events []engine.TokenEvent
	for ev := range ch {
		events = append(events, ev)
	}
	if len(events) != 1 || events[0].Text != " Paris" || events[0].Done {
		t.Fatalf("events = %+v, want one token event and a close with no done", events)
	}
}

func TestUnixEmbedStillNotImplemented(t *testing.T) {
	c := dial(t, fakeEngine(t, 1, "hello_ack"))
	if _, err := c.Embed(context.Background(), engine.EmbedRequest{}); !errors.Is(err, engine.ErrNotImplemented) {
		t.Fatalf("Embed err = %v, want ErrNotImplemented", err)
	}
}

func TestUnixDebugInfo(t *testing.T) {
	c := dial(t, fakeEngine(t, 1, "hello_ack"))
	info, err := c.DebugInfo(context.Background())
	if err != nil {
		t.Fatalf("DebugInfo: %v", err)
	}
	if info["model"] != "fake-1b" {
		t.Fatalf("debug info = %v, want model fake-1b", info)
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
