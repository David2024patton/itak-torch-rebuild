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
//   - /healthz is the only exempt path, so liveness probes work while
//     everything else, including /debug, sits behind the key.
package auth

import (
	"crypto/subtle"
	"encoding/json"
	"net/http"
	"strings"
)

// FailureRecorder is the slice of diag the middleware needs; kept as an
// interface so auth does not import the diagnostics package.
type FailureRecorder interface {
	RecordAuthFailure()
}

// Middleware returns a handler wrapper enforcing the API key.
type Middleware struct {
	key      string
	exempt   map[string]bool
	recorder FailureRecorder
}

// New builds the middleware. exempt lists exact paths reachable without a
// key (the server passes only "/healthz").
func New(key string, recorder FailureRecorder, exempt ...string) *Middleware {
	set := make(map[string]bool, len(exempt))
	for _, p := range exempt {
		set[p] = true
	}
	return &Middleware{key: key, exempt: set, recorder: recorder}
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
		next.ServeHTTP(w, r)
	})
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
