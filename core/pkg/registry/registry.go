package registry

import (
	"fmt"
	"sync"
	"time"
)

// ServiceRegistry is a thread-safe, in-process service registry.
// Each iTaK module registers itself at startup, and other modules
// discover services by querying the registry.
//
// For single-host deployments (all modules on one machine), this
// in-memory registry is sufficient. For multi-host deployments,
// use Client against a Server wrapping a registry on the broker.
type ServiceRegistry struct {
	mu       sync.RWMutex
	services map[string][]ServiceEndpoint // module name -> endpoints
	ttl      time.Duration                // how long before a stale endpoint is evicted
}

// New creates a ServiceRegistry with the given TTL for stale endpoints.
// The TTL is configuration supplied by the caller, never a compiled-in
// limit. If ttl is 0, endpoints never expire (manual removal only).
func New(ttl time.Duration) *ServiceRegistry {
	return &ServiceRegistry{
		services: make(map[string][]ServiceEndpoint),
		ttl:      ttl,
	}
}

// Register adds or updates an endpoint for a module.
// If the module+host+port combination already exists, it updates the record.
func (r *ServiceRegistry) Register(ep ServiceEndpoint) {
	r.mu.Lock()
	defer r.mu.Unlock()

	ep.LastSeen = time.Now()
	ep.Healthy = true

	endpoints := r.services[ep.Module]
	for i, existing := range endpoints {
		if existing.Host == ep.Host && existing.Port == ep.Port {
			endpoints[i] = ep
			r.services[ep.Module] = endpoints
			return
		}
	}
	r.services[ep.Module] = append(endpoints, ep)
}

// Deregister removes a specific endpoint.
func (r *ServiceRegistry) Deregister(module, host string, port int) {
	r.mu.Lock()
	defer r.mu.Unlock()

	endpoints := r.services[module]
	for i, ep := range endpoints {
		if ep.Host == host && ep.Port == port {
			r.services[module] = append(endpoints[:i], endpoints[i+1:]...)
			return
		}
	}
}

// Discover returns all healthy endpoints for a given module.
// Stale endpoints (past TTL) are filtered out and evicted.
func (r *ServiceRegistry) Discover(module string) []ServiceEndpoint {
	r.mu.Lock()
	defer r.mu.Unlock()

	endpoints := r.services[module]
	if len(endpoints) == 0 {
		return nil
	}

	now := time.Now()
	healthy := make([]ServiceEndpoint, 0, len(endpoints))
	kept := make([]ServiceEndpoint, 0, len(endpoints))

	for _, ep := range endpoints {
		// Evict stale endpoints if TTL is configured
		if r.ttl > 0 && now.Sub(ep.LastSeen) > r.ttl {
			continue
		}
		kept = append(kept, ep)
		if ep.Healthy {
			healthy = append(healthy, ep)
		}
	}

	r.services[module] = kept
	return healthy
}

// DiscoverOne returns a single healthy endpoint for a module.
// Returns an error if no healthy endpoints are available.
func (r *ServiceRegistry) DiscoverOne(module string) (ServiceEndpoint, error) {
	endpoints := r.Discover(module)
	if len(endpoints) == 0 {
		return ServiceEndpoint{}, fmt.Errorf("no healthy endpoints for module %q", module)
	}
	return endpoints[0], nil
}

// All returns every registered endpoint across all modules, including unhealthy ones.
func (r *ServiceRegistry) All() map[string][]ServiceEndpoint {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make(map[string][]ServiceEndpoint, len(r.services))
	for module, eps := range r.services {
		copied := make([]ServiceEndpoint, len(eps))
		copy(copied, eps)
		result[module] = copied
	}
	return result
}

// Heartbeat updates the LastSeen timestamp for an endpoint, keeping it alive.
func (r *ServiceRegistry) Heartbeat(module, host string, port int) {
	r.mu.Lock()
	defer r.mu.Unlock()

	for i, ep := range r.services[module] {
		if ep.Host == host && ep.Port == port {
			r.services[module][i].LastSeen = time.Now()
			r.services[module][i].Healthy = true
			return
		}
	}
}

// MarkUnhealthy flags an endpoint as unhealthy without removing it.
func (r *ServiceRegistry) MarkUnhealthy(module, host string, port int) {
	r.mu.Lock()
	defer r.mu.Unlock()

	for i, ep := range r.services[module] {
		if ep.Host == host && ep.Port == port {
			r.services[module][i].Healthy = false
			return
		}
	}
}
