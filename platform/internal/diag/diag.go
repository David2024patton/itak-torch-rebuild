// Package diag is the platform's built-in diagnostics registry.
//
// Every module reports into a Registry: request counts per route, error
// counts, engine call outcomes. The /debug endpoint serves a Snapshot of
// the registry, so a running torchd can always answer "what have you done
// and what went wrong" without log archaeology. Counters are atomic and
// the registry is safe for concurrent use.
package diag

import (
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// Registry collects platform diagnostics from birth: it is constructed
// with the process and never nil in a wired server.
type Registry struct {
	startedAt time.Time

	requests  atomic.Int64
	errors    atomic.Int64
	authFails atomic.Int64

	engineCalls  atomic.Int64
	engineErrors atomic.Int64

	mu     sync.Mutex
	routes map[string]*atomic.Int64
}

// New returns an empty registry stamped with the current time.
func New() *Registry {
	return &Registry{
		startedAt: time.Now(),
		routes:    make(map[string]*atomic.Int64),
	}
}

// RecordRequest counts one served request for a route pattern.
func (r *Registry) RecordRequest(route string) {
	r.requests.Add(1)
	r.mu.Lock()
	c, ok := r.routes[route]
	if !ok {
		c = &atomic.Int64{}
		r.routes[route] = c
	}
	r.mu.Unlock()
	c.Add(1)
}

// RecordError counts one error response.
func (r *Registry) RecordError() { r.errors.Add(1) }

// RecordAuthFailure counts one rejected API key check.
func (r *Registry) RecordAuthFailure() { r.authFails.Add(1) }

// RecordEngineCall counts one call into the engine client.
func (r *Registry) RecordEngineCall() { r.engineCalls.Add(1) }

// RecordEngineError counts one failed engine call.
func (r *Registry) RecordEngineError() { r.engineErrors.Add(1) }

// Snapshot is a point-in-time copy of the registry for /debug.
type Snapshot struct {
	UptimeSeconds int64            `json:"uptime_seconds"`
	Requests      int64            `json:"requests"`
	Errors        int64            `json:"errors"`
	AuthFailures  int64            `json:"auth_failures"`
	EngineCalls   int64            `json:"engine_calls"`
	EngineErrors  int64            `json:"engine_errors"`
	Routes        map[string]int64 `json:"routes"`
}

// Snapshot copies the current counters out of the registry.
func (r *Registry) Snapshot() Snapshot {
	routes := make(map[string]int64)
	r.mu.Lock()
	keys := make([]string, 0, len(r.routes))
	for k := range r.routes {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		routes[k] = r.routes[k].Load()
	}
	r.mu.Unlock()
	return Snapshot{
		UptimeSeconds: int64(time.Since(r.startedAt).Seconds()),
		Requests:      r.requests.Load(),
		Errors:        r.errors.Load(),
		AuthFailures:  r.authFails.Load(),
		EngineCalls:   r.engineCalls.Load(),
		EngineErrors:  r.engineErrors.Load(),
		Routes:        routes,
	}
}
