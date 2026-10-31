// Package auth provides the platform's single API-key middleware.
//
// The old platform shipped with no authentication on any route and bound
// 0.0.0.0. This middleware is the correction, applied in exactly one
// place in front of every route group:
//
//   - The expected key is supplied by the caller (read from the
//     environment in main); it is never hardcoded or defaulted here.
//   - Comparison is constant-time.
//   - An empty configured key fails closed: every request is rejected.
//   - Repeated failures from one source are throttled: past a small
//     threshold, each failure is answered after a growing delay, so
//     online guessing costs time instead of running at line rate.
//   - /healthz is the only exempt path, so liveness probes work while
//     everything else, including /debug, sits behind the key.
package auth

import (
	"crypto/subtle"
	"encoding/json"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"
)

// FailureRecorder is the slice of diag the middleware needs; kept as an
// interface so auth does not import the diagnostics package.
type FailureRecorder interface {
	RecordAuthFailure()
}

// Failure throttle tuning. The first throttleThreshold failures
// from a source are answered at once (a fat-fingered operator is
// not punished); each failure past the threshold waits one more
// throttleStep before its 401, up to throttleMaxDelay. A success
// clears the source's count.
const (
	throttleThreshold = 3
	throttleStep      = 200 * time.Millisecond
	throttleMaxDelay  = 2 * time.Second
	// throttleIdle drops a source's count after this long without
	// a failure; entries are swept lazily when the map grows.
	throttleIdle = 10 * time.Minute
)

// Middleware returns a handler wrapper enforcing the API key.
type Middleware struct {
	key      string
	exempt   map[string]bool
	recorder FailureRecorder

	mu       sync.Mutex
	failures map[string]*failState
}

type failState struct {
	count int
	last  time.Time
}

// New builds the middleware. exempt lists exact paths reachable without a
// key (the server passes only "/healthz").
func New(key string, recorder FailureRecorder, exempt ...string) *Middleware {
	set := make(map[string]bool, len(exempt))
	for _, p := range exempt {
		set[p] = true
	}
	return &Middleware{key: key, exempt: set, recorder: recorder, failures: make(map[string]*failState)}
}

// Wrap enforces the key on next unless the path is exempt.
func (m *Middleware) Wrap(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if m.exempt[r.URL.Path] {
			next.ServeHTTP(w, r)
			return
		}
		if !m.authorized(r) {
			if m.recorder != nil {
				m.recorder.RecordAuthFailure()
			}
			m.throttle(r)
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("WWW-Authenticate", "Bearer")
			w.WriteHeader(http.StatusUnauthorized)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"error": map[string]any{
					"code":    "unauthorized",
					"message": "missing or invalid API key",
				},
			})
			return
		}
		m.clearFailures(r)
		next.ServeHTTP(w, r)
	})
}

// throttle records one failure for the request's source and sleeps
// the penalty it has earned. The sleep happens before the 401 is
// written, so a guessing loop pays it on every attempt.
func (m *Middleware) throttle(r *http.Request) {
	src := sourceOf(r)
	now := time.Now()
	m.mu.Lock()
	st, ok := m.failures[src]
	if !ok || now.Sub(st.last) > throttleIdle {
		st = &failState{}
		m.failures[src] = st
	}
	st.count++
	st.last = now
	over := st.count - throttleThreshold
	if len(m.failures) > 1024 {
		for k, v := range m.failures {
			if now.Sub(v.last) > throttleIdle {
				delete(m.failures, k)
			}
		}
	}
	m.mu.Unlock()
	if over <= 0 {
		return
	}
	delay := time.Duration(over) * throttleStep
	if delay > throttleMaxDelay {
		delay = throttleMaxDelay
	}
	time.Sleep(delay)
}

// clearFailures resets a source's failure count after a success.
func (m *Middleware) clearFailures(r *http.Request) {
	m.mu.Lock()
	delete(m.failures, sourceOf(r))
	m.mu.Unlock()
}

// sourceOf names the request's origin for throttling: the peer IP,
// without its ephemeral port.
func sourceOf(r *http.Request) string {
	if host, _, err := net.SplitHostPort(r.RemoteAddr); err == nil {
		return host
	}
	return r.RemoteAddr
}

// authorized extracts the presented key from the Authorization bearer
// header or X-API-Key and compares in constant time. An empty configured
// key rejects everything (fail closed).
func (m *Middleware) authorized(r *http.Request) bool {
	if m.key == "" {
		return false
	}
	presented := ""
	if h := r.Header.Get("Authorization"); strings.HasPrefix(h, "Bearer ") {
		presented = strings.TrimPrefix(h, "Bearer ")
	} else if h := r.Header.Get("X-API-Key"); h != "" {
		presented = h
	}
	if presented == "" {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(presented), []byte(m.key)) == 1
}
