// Command torchd is the Torch platform daemon: the Go half of the
// rebuild. It serves the API, schedules engine work, and caches
// responses; the Rust engine process does the token math behind the
// engine.Client seam.
//
// Configuration is flags plus two environment variables:
//
//	TORCH_API_KEY        required API key (no default, never hardcoded)
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
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/David2024patton/itak-torch-rebuild/platform/internal/engine"
	"github.com/David2024patton/itak-torch-rebuild/platform/internal/server"
)

func main() { os.Exit(run()) }

func run() int {
	addr := flag.String("addr", "127.0.0.1:8091", "listen address (default loopback only)")
	socket := flag.String("engine-socket", "", "path to the Rust engine unix socket; empty = no engine configured")
	models := flag.String("models", "glimmer", "comma-separated model ids advertised on /v1/models")
	maxConcurrent := flag.Int("max-concurrent", 4, "maximum simultaneous engine jobs")
	cacheTTL := flag.Duration("cache-ttl", 5*time.Minute, "response cache TTL; 0 disables caching")
	flag.Parse()

	log := newLogger()
	slog.SetDefault(log)

	apiKey := os.Getenv("TORCH_API_KEY")
	if apiKey == "" {
		if os.Getenv("TORCH_ALLOW_NO_AUTH") != "1" {
			log.Error("TORCH_API_KEY is not set; refusing to start an unauthenticated server (set TORCH_ALLOW_NO_AUTH=1 to override deliberately)")
			return 1
		}
		log.Warn("TORCH_ALLOW_NO_AUTH=1: running WITHOUT authentication; every route is open")
	}

	eng, closeEngine := dialEngine(*socket, log)
	if closeEngine != nil {
		defer closeEngine()
	}

	srv := server.New(server.Config{
		Engine:        eng,
		APIKey:        apiKey,
		Models:        splitModels(*models),
		CacheTTL:      *cacheTTL,
		MaxConcurrent: *maxConcurrent,
		Logger:        log,
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
