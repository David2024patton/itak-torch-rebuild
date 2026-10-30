package supervisor_test

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/David2024patton/itak-torch-rebuild/platform/internal/engine"
	"github.com/David2024patton/itak-torch-rebuild/platform/internal/supervisor"
)

// The fake engine is this test binary re-executed with an env flag:
// the classic os/exec helper-process pattern. It speaks just enough
// of the socket protocol to serve generations, and a generate whose
// prompt contains "die" kills the process mid-stream after one
// token, which is the T-9.11 fault: engine death during generation.
func TestHelperProcess(_ *testing.T) {
	if os.Getenv("ITAK_FAKE_ENGINE") != "1" {
		return
	}
	runFakeEngine(os.Getenv("ITAK_FAKE_SOCKET"))
	os.Exit(0)
}

func runFakeEngine(socketPath string) {
	// Like the real engine: a restarted process removes the stale
	// socket file its previous incarnation left behind.
	_ = os.Remove(socketPath) //nolint:gosec // test-controlled socket path from the test's own env
	ln, err := net.Listen("unix", socketPath)
	if err != nil {
		os.Exit(2)
	}
	for {
		conn, err := ln.Accept()
		if err != nil {
			return
		}
		go serveFakeConn(conn)
	}
}

func serveFakeConn(conn net.Conn) {
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
			Op     string `json:"op"`
			Prompt string `json:"prompt"`
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
			if !write(map[string]any{"op": "pong", "detail": "fake"}) {
				return
			}
		case "debug":
			if !write(map[string]any{"op": "debug_info", "model": "fake-1b"}) {
				return
			}
		case "generate":
			if strings.Contains(f.Prompt, "die") {
				_ = write(map[string]any{"op": "token", "text": " Paris", "token_id": 13796})
				// Death mid-generation, no terminal frame. The
				// deferred conn close not running is the point:
				// a dead process runs no defers.
				os.Exit(3) //nolint:gocritic // simulated process death
			}
			if !write(map[string]any{"op": "token", "text": " Paris", "token_id": 13796}) {
				return
			}
			if !write(map[string]any{"op": "token", "text": "", "token_id": 0, "done": true, "finish_reason": "stop"}) {
				return
			}
		default:
			if !write(map[string]any{"op": "error", "code": "unknown_op", "message": "unknown op"}) {
				return
			}
		}
	}
}

func startSupervisor(t *testing.T, maxRestarts int) *supervisor.Supervisor {
	t.Helper()
	socket := filepath.Join(t.TempDir(), "engine.sock")
	sup := supervisor.New(supervisor.Config{
		Command:     os.Args[0],
		Args:        []string{"-test.run=TestHelperProcess"},
		SocketPath:  socket,
		MaxRestarts: maxRestarts,
		Window:      time.Minute,
		MinBackoff:  20 * time.Millisecond,
		MaxBackoff:  100 * time.Millisecond,
		DialTimeout: 10 * time.Second,
	})
	// The child learns its socket and role from the environment.
	t.Setenv("ITAK_FAKE_ENGINE", "1")
	t.Setenv("ITAK_FAKE_SOCKET", socket)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := sup.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(sup.Stop)
	return sup
}

func generate(t *testing.T, sup *supervisor.Supervisor, prompt string) []engine.TokenEvent {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	ch, err := sup.Generate(ctx, engine.GenerateRequest{Model: "fake-1b", Prompt: prompt})
	if err != nil {
		t.Fatalf("Generate(%q): %v", prompt, err)
	}
	var events []engine.TokenEvent
	for ev := range ch {
		events = append(events, ev)
	}
	return events
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func healthy(sup *supervisor.Supervisor) bool {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	h, err := sup.Health(ctx)
	return err == nil && h.OK
}

// The T-9.11 proof shape, against a real child process: a generation
// runs; the engine dies mid-generation and the in-flight request
// answers with the named ErrEngineDied terminal event; the next
// request is served by the restarted engine.
func TestSupervisorRestartsEngineAfterMidGenerationDeath(t *testing.T) {
	sup := startSupervisor(t, 3)

	events := generate(t, sup, "The capital of France is")
	if len(events) != 2 || !events[1].Done {
		t.Fatalf("first generation events = %+v, want token + done", events)
	}

	events = generate(t, sup, "die mid-stream")
	if len(events) != 2 {
		t.Fatalf("dying generation events = %+v, want token + terminal error", events)
	}
	if !errors.Is(events[1].Err, engine.ErrEngineDied) {
		t.Fatalf("dying generation terminal err = %v, want ErrEngineDied", events[1].Err)
	}

	waitFor(t, "engine restart", func() bool {
		return sup.State().Restarts == 1 && healthy(sup)
	})

	events = generate(t, sup, "The capital of France is")
	if len(events) != 2 || !events[1].Done {
		t.Fatalf("post-restart generation events = %+v, want token + done", events)
	}
	st := sup.State()
	if !st.Running || st.Restarts != 1 || st.Exhausted {
		t.Fatalf("state = %+v, want running with 1 restart, not exhausted", st)
	}
}

// A budget of one restart means the second death is final: the
// supervisor reports exhaustion and refuses by name instead of
// crash-looping the engine forever.
func TestSupervisorRestartBudgetExhaustion(t *testing.T) {
	sup := startSupervisor(t, 1)

	generate(t, sup, "die once")
	waitFor(t, "first restart", func() bool {
		return sup.State().Restarts == 1 && healthy(sup)
	})

	generate(t, sup, "die twice")
	waitFor(t, "budget exhaustion", func() bool {
		return sup.State().Exhausted
	})

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if _, err := sup.Generate(ctx, engine.GenerateRequest{Prompt: "hello"}); !errors.Is(err, engine.ErrUnavailable) {
		t.Fatalf("Generate after exhaustion err = %v, want ErrUnavailable", err)
	}
	h, err := sup.Health(ctx)
	if err != nil || h.OK || !strings.Contains(h.Detail, "restart budget exhausted") {
		t.Fatalf("Health after exhaustion = %+v, %v; want not-OK naming the budget", h, err)
	}
}
