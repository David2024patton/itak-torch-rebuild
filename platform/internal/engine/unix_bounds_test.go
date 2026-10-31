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

// scriptedEngine serves the socket protocol on a short path with a
// per-op behavior scripted by mode:
//
//	"huge"     ping and generate answer with a frame past the
//	           protocol cap (generate sends one good token first)
//	"slowping" ping sleeps past the client's exchange deadline
//
// hello always acks correctly, so the connections establish; the
// defect under test is what happens after.
func scriptedEngine(t *testing.T, mode string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "e.sock")
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
			go serveScripted(conn, mode)
		}
	}()
	return path
}

func serveScripted(conn net.Conn, mode string) {
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
			Op string `json:"op"`
		}
		if json.Unmarshal(line, &f) != nil {
			return
		}
		switch f.Op {
		case "hello":
			if !write(map[string]any{"op": "hello_ack", "proto": 1}) {
				return
			}
		case "ping":
			switch mode {
			case "huge":
				if _, err := conn.Write([]byte(strings.Repeat("x", (4<<20)+1024) + "\n")); err != nil {
					return
				}
			case "slowping":
				time.Sleep(3 * time.Second)
				if !write(map[string]any{"op": "pong"}) {
					return
				}
			default:
				if !write(map[string]any{"op": "pong"}) {
					return
				}
			}
		case "debug":
			switch mode {
			case "huge":
				if _, err := conn.Write([]byte(strings.Repeat("x", (4<<20)+1024) + "\n")); err != nil {
					return
				}
			case "slowping":
				time.Sleep(3 * time.Second)
				if !write(map[string]any{"op": "debug_info", "model": "fake-1b"}) {
					return
				}
			default:
				if !write(map[string]any{"op": "debug_info", "model": "fake-1b"}) {
					return
				}
			}
		case "generate":
			if !write(map[string]any{"op": "token", "text": " Paris", "token_id": 13796}) {
				return
			}
			if mode == "huge" {
				if _, err := conn.Write([]byte(strings.Repeat("y", (4<<20)+1024) + "\n")); err != nil {
					return
				}
			}
			return
		default:
			if !write(map[string]any{"op": "error", "code": "unknown_op", "message": "unknown op"}) {
				return
			}
		}
	}
}

func dialScripted(t *testing.T, mode string) *engine.UnixClient {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	c, err := engine.DialUnix(ctx, scriptedEngine(t, mode))
	if err != nil {
		t.Fatalf("DialUnix(%s): %v", mode, err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return c
}

// TestOversizedReplyFrameRefusedByName is the Wave 1 client bound:
// a peer streaming past the 4 MiB frame cap is refused with
// ErrFrameTooLarge, not buffered into memory. (DebugInfo returns
// exchange errors directly; Health folds them into its Detail,
// asserted as the secondary surface.)
func TestOversizedReplyFrameRefusedByName(t *testing.T) {
	c := dialScripted(t, "huge")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_, err := c.DebugInfo(ctx)
	if !errors.Is(err, engine.ErrFrameTooLarge) {
		t.Fatalf("DebugInfo err = %v, want ErrFrameTooLarge", err)
	}
	if errors.Is(err, engine.ErrHandshake) {
		t.Fatalf("DebugInfo err = %v, must not be mislabeled a handshake failure", err)
	}
	h, herr := c.Health(ctx)
	if herr != nil || h.OK || !strings.Contains(h.Detail, "frame exceeds") {
		t.Fatalf("Health = %+v, %v; want not-OK with the frame-cap name in the detail", h, herr)
	}
}

// TestOversizedGenerateFrameRefusedByName covers the streaming read
// path: one good token, then a frame past the cap, and the terminal
// event carries ErrFrameTooLarge by name.
func TestOversizedGenerateFrameRefusedByName(t *testing.T) {
	c := dialScripted(t, "huge")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	ch, err := c.Generate(ctx, engine.GenerateRequest{Model: "fake-1b", Prompt: "hi"})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	var events []engine.TokenEvent
	for ev := range ch {
		events = append(events, ev)
	}
	if len(events) != 2 {
		t.Fatalf("events = %+v, want token + terminal error", events)
	}
	if !errors.Is(events[1].Err, engine.ErrFrameTooLarge) {
		t.Fatalf("terminal err = %v, want ErrFrameTooLarge", events[1].Err)
	}
}

// TestSlowEngineReportsTimeoutNotHandshake is the honesty fix: the
// control connection's exchange deadline expiring on a slow but
// healthy engine is ErrTimeout. It must not wear ErrHandshake,
// which names a different fault.
func TestSlowEngineReportsTimeoutNotHandshake(t *testing.T) {
	c := dialScripted(t, "slowping")
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	_, err := c.DebugInfo(ctx)
	if !errors.Is(err, engine.ErrTimeout) {
		t.Fatalf("DebugInfo err = %v, want ErrTimeout", err)
	}
	if errors.Is(err, engine.ErrHandshake) {
		t.Fatalf("DebugInfo err = %v, must not be mislabeled a handshake failure", err)
	}
}
