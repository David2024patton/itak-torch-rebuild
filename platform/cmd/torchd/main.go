// Command torchd is the Torch platform daemon: the Go half of the
// rebuild. It serves the API, schedules engine work, and caches
// responses; the Rust engine process does the token math behind the
// engine.Client seam.
//
// Configuration is flags plus two environment variables:
//
//	TORCH_API_KEY        required API key (no default, never hardcoded);
//	                     at least 16 characters or startup refuses
//	TORCH_ALLOW_NO_AUTH  set to 1 to run without a key (loudly logged)
//
// The bind defaults to 127.0.0.1. Exposing the platform beyond localhost
// is an explicit -addr decision by the operator, not a default, which is
// the opposite of how the old platform shipped.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/David2024patton/itak-torch-rebuild/platform/internal/engine"
	"github.com/David2024patton/itak-torch-rebuild/platform/internal/server"
	"github.com/David2024patton/itak-torch-rebuild/platform/internal/supervisor"
)

func main() { os.Exit(run()) }

func run() int {
	addr := flag.String("addr", "127.0.0.1:8091", "listen address (default loopback only)")
	socket := flag.String("engine-socket", "", "path to the Rust engine unix socket; empty = no engine configured")
	engineBin := flag.String("engine-bin", "", "path to the itak-engine binary to supervise; set with -engine-model. Supervision restarts the engine within the restart budget when it dies (TODO T-9.11)")
	engineModel := flag.String("engine-model", "", "GGUF file the supervised engine loads (--model)")
	engineName := flag.String("engine-name", "", "model name the supervised engine serves (--model-name); empty = the file's own name")
	engineMaxRestarts := flag.Int("engine-max-restarts", 5, "supervised engine: restarts allowed inside -engine-restart-window before the engine stays down")
	engineRestartWindow := flag.Duration("engine-restart-window", 5*time.Minute, "supervised engine: restart budget window")
	engineSlots := flag.Int("engine-slots", 1, "supervised engine: concurrent generations (--slots)")
	engineMaxQueued := flag.Int("engine-max-queued", 8, "supervised engine: generations that may wait for a slot before refusing as engine_busy (--max-queued)")
	models := flag.String("models", "glimmer", "comma-separated model ids advertised on /v1/models")
	maxConcurrent := flag.Int("max-concurrent", 4, "maximum simultaneous engine jobs")
	cacheTTL := flag.Duration("cache-ttl", 5*time.Minute, "response cache TTL; 0 disables caching")
	maxTokens := flag.Int("max-tokens", 8192, "maximum max_tokens one chat request may ask for; requests past the cap are refused by name")
	generationTimeout := flag.Duration("generation-timeout", time.Hour, "per-generation wall-clock deadline; a generation that runs past it ends with a named failure")
	maxBodyBytes := flag.Int64("max-body-bytes", 1<<20, "maximum request body size in bytes")
	readHeaderTimeout := flag.Duration("read-header-timeout", 10*time.Second, "how long a client may take to send request headers")
	idleTimeout := flag.Duration("idle-timeout", 90*time.Second, "how long an idle keep-alive connection is held open")
	maxWaiters := flag.Int("max-waiters-per-lane", 128, "maximum requests queued in each scheduler lane; past the cap a request is refused as engine_busy")
	cacheMaxEntries := flag.Int("cache-max-entries", 1024, "maximum response cache entries; past the cap the soonest to expire are evicted")
	cachePurgeInterval := flag.Duration("cache-purge-interval", time.Minute, "how often the response cache janitor purges expired entries")
	flag.Parse()

	log := newLogger()
	slog.SetDefault(log)

	apiKey := os.Getenv("TORCH_API_KEY")
	if problem := apiKeyProblem(apiKey); problem != "" {
		log.Error(problem)
		return 1
	}
	if apiKey == "" {
		log.Warn("TORCH_ALLOW_NO_AUTH=1: running WITHOUT authentication; every route is open")
	}

	var eng engine.Client
	var closeEngine func()
	if *engineBin != "" {
		eng, closeEngine = superviseEngine(*engineBin, *engineModel, *engineName, *socket,
			*engineMaxRestarts, *engineRestartWindow, *engineSlots, *engineMaxQueued, log)
	} else {
		eng, closeEngine = dialEngine(*socket, log)
	}
	if closeEngine != nil {
		defer closeEngine()
	}

	srv := server.New(server.Config{
		Engine:             eng,
		APIKey:             apiKey,
		Models:             splitModels(*models),
		CacheTTL:           *cacheTTL,
		MaxConcurrent:      *maxConcurrent,
		MaxTokensCap:       *maxTokens,
		GenerationTimeout:  *generationTimeout,
		MaxBodyBytes:       *maxBodyBytes,
		ReadHeaderTimeout:  *readHeaderTimeout,
		IdleTimeout:        *idleTimeout,
		MaxWaitersPerLane:  *maxWaiters,
		CacheMaxEntries:    *cacheMaxEntries,
		CachePurgeInterval: *cachePurgeInterval,
		Logger:             log,
	})
	httpSrv := srv.HTTPServer(*addr)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		log.Info("shutdown signal received")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := httpSrv.Shutdown(shutdownCtx); err != nil {
			log.Error("graceful shutdown failed", "err", err)
		}
		srv.Close()
	}()

	log.Info("torchd listening", "addr", *addr, "engine_socket", *socket)
	if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Error("server exited", "err", err)
		return 1
	}
	log.Info("torchd stopped")
	return 0
}

// dialEngine connects to the Rust engine socket when configured. A
// failed dial is not fatal and is not hidden: the platform starts with
// the engine marked unavailable, /healthz shows it, and engine routes
// answer with the named engine_unavailable error.
func dialEngine(socketPath string, log *slog.Logger) (engine.Client, func()) {
	if socketPath == "" {
		log.Warn("no engine socket configured; engine routes will return engine_unavailable")
		return engine.Unavailable{}, nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	client, err := engine.DialUnix(ctx, socketPath)
	if err != nil {
		log.Error("engine socket dial failed; starting with engine unavailable", "socket", socketPath, "err", err)
		return engine.Unavailable{}, nil
	}
	log.Info("engine connected", "socket", socketPath)
	return client, func() { _ = client.Close() }
}

// superviseEngine starts itak-engine under the supervisor: torchd
// owns the process, restarts it within the configured budget when
// it dies, and serves the next request from the restarted engine
// (TODO T-9.11). A failed initial start is not fatal and is not
// hidden, matching dialEngine's behavior: the platform runs with
// the engine marked unavailable and /healthz says so.
func superviseEngine(bin, model, name, socketPath string, maxRestarts int, window time.Duration, slots, maxQueued int, log *slog.Logger) (engine.Client, func()) {
	if model == "" {
		log.Error("engine-bin set without engine-model; starting with engine unavailable")
		return engine.Unavailable{}, nil
	}
	if socketPath == "" {
		socketPath = "/tmp/itak-engine.sock"
	}
	args := []string{
		"--model", model, "--socket", socketPath,
		"--slots", fmt.Sprint(slots), "--max-queued", fmt.Sprint(maxQueued),
	}
	if name != "" {
		args = append(args, "--model-name", name)
	}
	sup := supervisor.New(supervisor.Config{
		Command:     bin,
		Args:        args,
		SocketPath:  socketPath,
		MaxRestarts: maxRestarts,
		Window:      window,
		Logger:      log,
	})
	ctx, cancel := context.WithTimeout(context.Background(), 130*time.Second)
	defer cancel()
	if err := sup.Start(ctx); err != nil {
		log.Error("supervised engine failed to start; starting with engine unavailable", "err", err)
		return engine.Unavailable{}, nil
	}
	log.Info("engine supervised", "bin", bin, "model", model, "socket", socketPath,
		"max_restarts", maxRestarts, "restart_window", window)
	return sup, sup.Stop
}

// minAPIKeyLen is the shortest TORCH_API_KEY torchd will run with.
// The key is the only credential on the platform's front door; a
// short one is a guessing target, so a short configured key is a
// named startup refusal, not a silent weakness.
const minAPIKeyLen = 16

// apiKeyProblem returns the startup refusal for an unusable API key
// configuration, or "" when the configuration may run: a key of at
// least minAPIKeyLen characters, or no key at all with the loud
// TORCH_ALLOW_NO_AUTH=1 opt-out.
func apiKeyProblem(apiKey string) string {
	if apiKey == "" {
		if os.Getenv("TORCH_ALLOW_NO_AUTH") != "1" {
			return "TORCH_API_KEY is not set; refusing to start an unauthenticated server (set TORCH_ALLOW_NO_AUTH=1 to override deliberately)"
		}
		return ""
	}
	if len(apiKey) < minAPIKeyLen {
		return fmt.Sprintf("TORCH_API_KEY is too short: %d characters, minimum %d; refusing to start with a guessable key", len(apiKey), minAPIKeyLen)
	}
	return ""
}

func splitModels(csv string) []string {
	var out []string
	for _, m := range strings.Split(csv, ",") {
		if m = strings.TrimSpace(m); m != "" {
			out = append(out, m)
		}
	}
	return out
}

func newLogger() *slog.Logger {
	level := slog.LevelInfo
	switch strings.ToLower(os.Getenv("TORCH_LOG_LEVEL")) {
	case "debug":
		level = slog.LevelDebug
	case "warn":
		level = slog.LevelWarn
	case "error":
		level = slog.LevelError
	}
	return slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: level}))
}
