package supervisor_test

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
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
// The role and socket arrive as ITAK_ENGINE_* variables, which are
// exactly what the supervisor's child-environment allowlist passes.
func TestHelperProcess(_ *testing.T) {
	if os.Getenv("ITAK_ENGINE_FAKE") != "1" {
		return
	}
	runFakeEngine(os.Getenv("ITAK_ENGINE_FAKESOCKET"))
	os.Exit(0)
}

func runFakeEngine(socketPath string) {
	// Wave 1 probes, scripted by the environment: report which
	// variables survived the supervisor's allowlist, and burst a
	// single stderr line past the old 64 KiB scanner default
	// followed by a marker line.
	if reportDir := os.Getenv("ITAK_ENGINE_ENVREPORT"); reportDir != "" {
		_, keyPresent := os.LookupEnv("TORCH_API_KEY")
		_, markPresent := os.LookupEnv("ITAK_ENGINE_TESTMARK")
		_, pathPresent := os.LookupEnv("PATH")
		content := "TORCH_API_KEY=" + presentWord(keyPresent) +
			"\nITAK_ENGINE_TESTMARK=" + presentWord(markPresent) +
			"\nPATH=" + presentWord(pathPresent) + "\n"
		// The environment names only the directory (the test's own
		// TempDir); the file name is fixed here, so this process can
		// only ever write the one report file inside it.
		_ = os.WriteFile(filepath.Join(reportDir, "env-report.txt"), []byte(content), 0o600) //nolint:gosec // test-only: directory is the parent test's own TempDir, file name fixed above
	}
	if os.Getenv("ITAK_ENGINE_STDERRBURST") == "1" {
		fmt.Fprintf(os.Stderr, "%s\n", strings.Repeat("x", 200_000))
		fmt.Fprintln(os.Stderr, "stderr-after-big-line")
	}
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

func presentWord(present bool) string {
	if present {
		return "present"
	}
	return "absent"
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
	// The child learns its socket and role from the environment,
	// under the ITAK_ENGINE_* names the allowlist passes.
	t.Setenv("ITAK_ENGINE_FAKE", "1")
	t.Setenv("ITAK_ENGINE_FAKESOCKET", socket)
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

// TestSupervisorChildEnvIsAllowlisted is the Wave 1 environment fix:
// the engine child receives PATH and ITAK_ENGINE_* variables only.
// TORCH_API_KEY, set in the supervisor's own environment here, must
// not reach the child.
func TestSupervisorChildEnvIsAllowlisted(t *testing.T) {
	reportDir := t.TempDir()
	report := filepath.Join(reportDir, "env-report.txt")
	t.Setenv("TORCH_API_KEY", "must-not-reach-the-engine")
	t.Setenv("ITAK_ENGINE_TESTMARK", "1")
	t.Setenv("ITAK_ENGINE_ENVREPORT", reportDir)
	startSupervisor(t, 1)

	waitFor(t, "child env report", func() bool {
		_, err := os.Stat(report)
		return err == nil
	})
	data, err := os.ReadFile(report) //nolint:gosec // test-only: path is this test's own TempDir joined with the fixed report file name
	if err != nil {
		t.Fatalf("read env report: %v", err)
	}
	for _, want := range []string{"TORCH_API_KEY=absent", "ITAK_ENGINE_TESTMARK=present", "PATH=present"} {
		if !strings.Contains(string(data), want) {
			t.Fatalf("env report missing %q:\n%s", want, data)
		}
	}
}

// logCapture is a slog handler that records messages and the "line"
// attribute of forwarded engine output.
type logCapture struct {
	mu    sync.Mutex
	lines []string
}

func (c *logCapture) Enabled(context.Context, slog.Level) bool { return true }

func (c *logCapture) Handle(_ context.Context, r slog.Record) error {
	text := r.Message
	r.Attrs(func(a slog.Attr) bool {
		if a.Key == "line" {
			text += "|" + a.Value.String()
		}
		return true
	})
	c.mu.Lock()
	c.lines = append(c.lines, text)
	c.mu.Unlock()
	return nil
}

func (c *logCapture) WithAttrs([]slog.Attr) slog.Handler { return c }
func (c *logCapture) WithGroup(string) slog.Handler      { return c }

func (c *logCapture) contains(substr string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, l := range c.lines {
		if strings.Contains(l, substr) {
			return true
		}
	}
	return false
}

// TestSupervisorStderrPumpSurvivesLongLines is the H-2 regression:
// the fake engine writes one 200 KB stderr line (past the bufio
// default that used to kill the pump), then a marker line. Both the
// long line and the marker must reach the platform log; with the
// old scanner the marker never arrived and the pipe wedged.
func TestSupervisorStderrPumpSurvivesLongLines(t *testing.T) {
	t.Setenv("ITAK_ENGINE_STDERRBURST", "1")
	capture := &logCapture{}
	socket := filepath.Join(t.TempDir(), "engine.sock")
	sup := supervisor.New(supervisor.Config{
		Command:     os.Args[0],
		Args:        []string{"-test.run=TestHelperProcess"},
		SocketPath:  socket,
		MaxRestarts: 1,
		Window:      time.Minute,
		DialTimeout: 10 * time.Second,
		Logger:      slog.New(capture),
	})
	t.Setenv("ITAK_ENGINE_FAKE", "1")
	t.Setenv("ITAK_ENGINE_FAKESOCKET", socket)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := sup.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(sup.Stop)

	waitFor(t, "the long stderr line to be forwarded", func() bool {
		return capture.contains(strings.Repeat("x", 1000))
	})
	waitFor(t, "the marker line after the long line", func() bool {
		return capture.contains("stderr-after-big-line")
	})
	if capture.contains("forwarding ended abnormally") {
		t.Fatal("stderr pump reported an abnormal end on a line under the configured cap")
	}
}
