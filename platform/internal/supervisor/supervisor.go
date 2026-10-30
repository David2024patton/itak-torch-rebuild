// Package supervisor keeps the Rust engine process alive.
//
// The engine is a separate process by design (the Go/Rust seam in
// ARCHITECTURE.md), which makes its death a platform problem: before
// this package existed, a dead engine meant in-flight generations
// ended as silently closed streams and every later request failed
// its dial until an operator restarted the process by hand (the
// 2026-10-30 serving smoke recorded exactly that death). The
// supervisor owns the lifecycle instead:
//
//   - It starts the engine binary with the operator's arguments and
//     dials the socket once the engine has loaded its model and is
//     serving (a 30B load takes seconds; the dial polls).
//   - A monitor watches the process. An unexpected exit closes the
//     stale client, is counted against a configured restart budget
//     (a maximum number of restarts inside a window, so a model
//     that can no longer load fails visibly instead of crash
//     looping forever), and is restarted after a bounded backoff.
//   - In-flight generations are not rescued, they are named: the
//     engine client's terminal error event reports ErrEngineDied
//     (TODO T-9.12), and the NEXT request is served by the
//     restarted engine. That pair is the T-9.11 proof shape.
//   - State() exposes the lifecycle (pid, restarts, budget,
//     exhaustion, last exit) for the platform's /debug view, so the
//     running system can explain its engine's health.
//
// The supervisor itself implements engine.Client by delegating to
// the current socket client, so the server neither knows nor cares
// that the process underneath it was replaced.
package supervisor

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os/exec"
	"sync"
	"time"

	"github.com/David2024patton/itak-torch-rebuild/platform/internal/engine"
)

// Config configures one supervised engine process. Command and
// SocketPath are required; the rest take the defaults New applies.
type Config struct {
	// Command is the engine binary path (itak-engine).
	Command string
	// Args are the engine's arguments. They must point the engine
	// at SocketPath (--socket), or the dial will never succeed.
	Args []string
	// SocketPath is the engine's Unix socket.
	SocketPath string
	// MaxRestarts bounds restarts inside Window. 0 means no
	// restarts: the first death is final (and reported exhausted).
	MaxRestarts int
	// Window is the restart-budget window. Default 5 minutes.
	Window time.Duration
	// MinBackoff is the first restart delay; it doubles per
	// consecutive restart up to MaxBackoff. Defaults 250ms / 5s.
	MinBackoff time.Duration
	// MaxBackoff caps the restart delay. Default 5 seconds.
	MaxBackoff time.Duration
	// DialTimeout bounds how long one start waits for the engine
	// to load and serve before the start counts as failed.
	// Default 120 seconds (a 30B model load is slow).
	DialTimeout time.Duration
	// Logger receives lifecycle diagnostics; nil uses slog.Default().
	Logger *slog.Logger
}

// State is the supervisor's self-report for /debug.
type State struct {
	Running   bool   `json:"running"`
	PID       int    `json:"pid,omitempty"`
	Restarts  int    `json:"restarts"`
	Exhausted bool   `json:"exhausted"`
	LastExit  string `json:"last_exit,omitempty"`
	// StartedAt is the current incarnation's start time, RFC 3339.
	StartedAt string `json:"started_at,omitempty"`
}

// incarnation is one run of the engine process.
type incarnation struct {
	cmd  *exec.Cmd
	done chan struct{} // closed when the process has been waited on
	err  error         // the Wait result, valid after done closes
}

// Supervisor is a restart-budgeted owner of one engine process. It
// implements engine.Client by delegating to the live socket client.
type Supervisor struct {
	cfg Config
	log *slog.Logger

	mu         sync.RWMutex
	client     *engine.UnixClient
	inc        *incarnation
	restarts   []time.Time // restart instants inside the budget window
	exhausted  bool
	lastExit   string
	startedAt  time.Time
	stopping   bool
	stopOnce   sync.Once
	stopCh     chan struct{} // closed by Stop
	monitorDon chan struct{}
}

// New applies config defaults. It does not start the process; call
// Start.
func New(cfg Config) *Supervisor {
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}
	if cfg.Window <= 0 {
		cfg.Window = 5 * time.Minute
	}
	if cfg.MinBackoff <= 0 {
		cfg.MinBackoff = 250 * time.Millisecond
	}
	if cfg.MaxBackoff < cfg.MinBackoff {
		cfg.MaxBackoff = 5 * time.Second
	}
	if cfg.DialTimeout <= 0 {
		cfg.DialTimeout = 120 * time.Second
	}
	if cfg.MaxRestarts < 0 {
		cfg.MaxRestarts = 0
	}
	return &Supervisor{
		cfg:        cfg,
		log:        cfg.Logger,
		stopCh:     make(chan struct{}),
		monitorDon: make(chan struct{}),
	}
}

// Start launches the engine and waits for it to serve. An initial
// start failure is returned (the caller decides whether to run
// without an engine); restarts after a successful start are the
// monitor's job and are reported through State and the log, not
// returned.
func (s *Supervisor) Start(ctx context.Context) error {
	if s.cfg.Command == "" || s.cfg.SocketPath == "" {
		return errors.New("supervisor: Command and SocketPath are required")
	}
	if err := s.launch(ctx); err != nil {
		return err
	}
	// The monitor deliberately outlives the Start call's context:
	// its lifetime is the supervisor's, ended by Stop (stopCh).
	go s.monitor() //nolint:gosec // lifecycle bound by Stop, not the start ctx
	return nil
}

// launch starts one incarnation and dials it. On any failure the
// half-started process is killed and waited on before returning.
func (s *Supervisor) launch(ctx context.Context) error {
	cmd := exec.Command(s.cfg.Command, s.cfg.Args...) //nolint:gosec // operator-configured engine binary and args, by design
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("supervisor: engine stderr pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("supervisor: start engine %q: %w", s.cfg.Command, err)
	}
	inc := &incarnation{cmd: cmd, done: make(chan struct{})}
	go func() {
		// The engine's stderr is its diagnostic channel; forward it
		// into the platform log until the process ends.
		sc := bufio.NewScanner(stderr)
		for sc.Scan() {
			s.log.Info("engine output", "line", sc.Text())
		}
	}()
	go func() {
		inc.err = cmd.Wait()
		close(inc.done)
	}()

	s.mu.Lock()
	s.inc = inc
	s.startedAt = time.Now()
	s.mu.Unlock()
	s.log.Info("engine started", "pid", cmd.Process.Pid, "socket", s.cfg.SocketPath)

	client, err := s.dialUntil(ctx, inc)
	if err != nil {
		s.kill(inc)
		return err
	}
	s.mu.Lock()
	if s.client != nil {
		_ = s.client.Close()
	}
	s.client = client
	s.mu.Unlock()
	s.log.Info("engine serving", "pid", cmd.Process.Pid, "socket", s.cfg.SocketPath)
	return nil
}

// dialUntil polls the socket until the engine serves or the
// incarnation dies or the dial budget runs out. Loading a model
// takes seconds, so a refused dial is retried, not fatal.
func (s *Supervisor) dialUntil(ctx context.Context, inc *incarnation) (*engine.UnixClient, error) {
	deadline := time.Now().Add(s.cfg.DialTimeout)
	for {
		select {
		case <-inc.done:
			return nil, fmt.Errorf("supervisor: engine exited during startup: %w", waitErr(inc))
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		dialCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
		client, err := engine.DialUnix(dialCtx, s.cfg.SocketPath)
		cancel()
		if err == nil {
			return client, nil
		}
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("supervisor: engine not serving within %s: %w", s.cfg.DialTimeout, err)
		}
		select {
		case <-inc.done:
			return nil, fmt.Errorf("supervisor: engine exited during startup: %w", waitErr(inc))
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(150 * time.Millisecond):
		}
	}
}

func waitErr(inc *incarnation) error {
	if inc.err != nil {
		return inc.err
	}
	return errors.New("exit status 0")
}

// kill terminates an incarnation and waits for it to be reaped. A
// synthesized incarnation (the failed-launch record) has no process;
// its done channel is already closed.
func (s *Supervisor) kill(inc *incarnation) {
	if inc.cmd != nil && inc.cmd.Process != nil {
		_ = inc.cmd.Process.Kill()
	}
	<-inc.done
}

// monitor is the restart loop: one iteration per incarnation death.
func (s *Supervisor) monitor() {
	defer close(s.monitorDon)
	for {
		s.mu.RLock()
		inc := s.inc
		s.mu.RUnlock()
		if inc == nil {
			return
		}
		<-inc.done

		s.mu.Lock()
		if s.stopping {
			s.mu.Unlock()
			return
		}
		// The client of a dead process is stale the moment the
		// process exits; drop it before anything new is dialed.
		if s.client != nil {
			_ = s.client.Close()
			s.client = nil
		}
		s.lastExit = waitErr(inc).Error()
		now := time.Now()
		cutoff := now.Add(-s.cfg.Window)
		kept := s.restarts[:0]
		for _, t := range s.restarts {
			if t.After(cutoff) {
				kept = append(kept, t)
			}
		}
		s.restarts = kept
		if len(s.restarts) >= s.cfg.MaxRestarts {
			s.exhausted = true
			s.mu.Unlock()
			s.log.Error("engine died and the restart budget is exhausted; engine stays down",
				"last_exit", s.lastExit, "restarts", len(s.restarts), "window", s.cfg.Window)
			return
		}
		backoff := s.cfg.MinBackoff << len(s.restarts)
		if backoff > s.cfg.MaxBackoff || backoff <= 0 {
			backoff = s.cfg.MaxBackoff
		}
		s.restarts = append(s.restarts, now)
		restartNo := len(s.restarts)
		s.mu.Unlock()

		s.log.Warn("engine died; restarting within budget",
			"last_exit", s.lastExit, "restart", restartNo, "budget", s.cfg.MaxRestarts, "backoff", backoff)
		select {
		case <-time.After(backoff):
		case <-s.stopCh:
			return
		}
		if err := s.launch(context.Background()); err != nil {
			s.log.Error("engine restart failed to launch", "err", err)
			// A failed launch is another death for budget
			// purposes: synthesize an incarnation record so the
			// loop counts it instead of spinning.
			s.mu.Lock()
			s.lastExit = err.Error()
			s.inc = &incarnation{done: closedChan(), err: err}
			s.mu.Unlock()
		}
	}
}

var (
	closedOnce sync.Once
	closedCh   = make(chan struct{})
)

func closedChan() chan struct{} {
	closedOnce.Do(func() { close(closedCh) })
	return closedCh
}

// Stop shuts the supervisor down deliberately: no restart follows,
// the live client closes, and the engine process is killed. It is
// safe to call more than once.
func (s *Supervisor) Stop() {
	s.stopOnce.Do(func() {
		s.mu.Lock()
		s.stopping = true
		close(s.stopCh)
		client := s.client
		s.client = nil
		inc := s.inc
		s.mu.Unlock()
		if client != nil {
			_ = client.Close()
		}
		if inc != nil {
			s.kill(inc)
		}
	})
}

// Close is Stop under the name io.Closer users expect.
func (s *Supervisor) Close() error {
	s.Stop()
	return nil
}

// State snapshots the lifecycle for /debug.
func (s *Supervisor) State() State {
	s.mu.RLock()
	defer s.mu.RUnlock()
	st := State{
		Restarts:  len(s.restarts),
		Exhausted: s.exhausted,
		LastExit:  s.lastExit,
	}
	if s.client != nil && s.inc != nil && s.inc.cmd != nil && s.inc.cmd.Process != nil {
		st.Running = true
		st.PID = s.inc.cmd.Process.Pid
		st.StartedAt = s.startedAt.Format(time.RFC3339)
	}
	return st
}

func (s *Supervisor) current() *engine.UnixClient {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.client
}

// Generate delegates to the live engine client. While the engine is
// down (restarting or budget-exhausted) it refuses with
// engine.ErrUnavailable; a generation the death catches mid-stream
// ends with the client's ErrEngineDied terminal event.
func (s *Supervisor) Generate(ctx context.Context, req engine.GenerateRequest) (<-chan engine.TokenEvent, error) {
	c := s.current()
	if c == nil {
		return nil, fmt.Errorf("%w: engine down (supervisor state: %+v)", engine.ErrUnavailable, s.State())
	}
	return c.Generate(ctx, req)
}

// Embed delegates to the live engine client.
func (s *Supervisor) Embed(ctx context.Context, req engine.EmbedRequest) (engine.EmbedResponse, error) {
	c := s.current()
	if c == nil {
		return engine.EmbedResponse{}, fmt.Errorf("%w: engine down (supervisor state: %+v)", engine.ErrUnavailable, s.State())
	}
	return c.Embed(ctx, req)
}

// ChatTemplate delegates to the live engine client, so the
// platform's template probe works the same supervised or not.
func (s *Supervisor) ChatTemplate(ctx context.Context) (engine.ChatTemplate, error) {
	c := s.current()
	if c == nil {
		return engine.ChatTemplate{}, fmt.Errorf("%w: engine down (supervisor state: %+v)", engine.ErrUnavailable, s.State())
	}
	return c.ChatTemplate(ctx)
}

// Health reports the engine's health through the supervisor's own
// state when no client is live, so /healthz can say why it is down.
func (s *Supervisor) Health(ctx context.Context) (engine.HealthStatus, error) {
	c := s.current()
	if c == nil {
		st := s.State()
		detail := "engine down"
		if st.Exhausted {
			detail = fmt.Sprintf("engine down: restart budget exhausted, last exit: %s", st.LastExit)
		} else if st.LastExit != "" {
			detail = fmt.Sprintf("engine down: restarting after exit: %s", st.LastExit)
		}
		return engine.HealthStatus{OK: false, Detail: detail}, nil
	}
	return c.Health(ctx)
}

// DebugInfo merges the engine's self-report with the supervisor's
// lifecycle state, under the "supervisor" key.
func (s *Supervisor) DebugInfo(ctx context.Context) (map[string]any, error) {
	out := map[string]any{"supervisor": s.State()}
	c := s.current()
	if c == nil {
		return out, nil
	}
	info, err := c.DebugInfo(ctx)
	if err != nil {
		out["engine_debug_error"] = err.Error()
		return out, nil
	}
	for k, v := range info {
		out[k] = v
	}
	return out, nil
}

// Compile-time proof the supervisor is an engine.Client.
var _ engine.Client = (*Supervisor)(nil)
