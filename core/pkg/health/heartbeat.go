package health

import (
	"context"
	"log"
	"time"

	"github.com/David2024patton/itak-torch-rebuild/core/pkg/registry"
)

// DefaultHeartbeatInterval is the fallback heartbeat cadence used when
// HeartbeatConfig leaves Interval unset. It is a default the caller can
// override through configuration, not a compiled-in limit: interval()
// honors any positive Interval the caller supplies.
const DefaultHeartbeatInterval = 10 * time.Second

// HeartbeatConfig controls how often a module announces itself.
type HeartbeatConfig struct {
	// Interval is how often heartbeats are sent.
	// Zero (or negative) selects DefaultHeartbeatInterval.
	Interval time.Duration

	// Registry is the local service registry to heartbeat into.
	// Use this for single-host deployments where all modules share memory.
	Registry *registry.ServiceRegistry

	// RegistryClient is a remote registry client for multi-host deployments.
	// If set, heartbeats are sent over HTTP to the broker's registry API
	// instead of updating the local in-memory registry.
	// Only one of Registry or RegistryClient should be set.
	RegistryClient *registry.Client

	// Endpoint is the module's own endpoint information.
	Endpoint registry.ServiceEndpoint

	// Checker provides the health status to include with each heartbeat.
	// When nil, heartbeats report liveness only: the endpoint stays
	// registered and healthy without a status evaluation.
	Checker Checker
}

// interval resolves the effective heartbeat cadence: the configured value
// when one is set, the package default otherwise.
func (cfg HeartbeatConfig) interval() time.Duration {
	if cfg.Interval > 0 {
		return cfg.Interval
	}
	return DefaultHeartbeatInterval
}

// StartHeartbeat begins periodic registration and health reporting.
// It blocks until the context is cancelled. Run it in a goroutine.
//
// Works with both local registries (single-host) and remote registries
// (multi-host via tunnel broker). Set either Registry or RegistryClient.
//
//	go health.StartHeartbeat(ctx, cfg)
func StartHeartbeat(ctx context.Context, cfg HeartbeatConfig) {
	// Initial registration
	if cfg.RegistryClient != nil {
		if err := cfg.RegistryClient.Register(cfg.Endpoint); err != nil {
			log.Printf("[heartbeat] remote register failed: %v", err)
		}
	} else if cfg.Registry != nil {
		cfg.Registry.Register(cfg.Endpoint)
	}

	ticker := time.NewTicker(cfg.interval())
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			if cfg.RegistryClient != nil {
				cfg.RegistryClient.Deregister(cfg.Endpoint.Module, cfg.Endpoint.Host, cfg.Endpoint.Port)
			} else if cfg.Registry != nil {
				cfg.Registry.Deregister(cfg.Endpoint.Module, cfg.Endpoint.Host, cfg.Endpoint.Port)
			}
			return
		case <-ticker.C:
			if cfg.Checker == nil {
				// Liveness only: keep the endpoint's heartbeat fresh.
				if cfg.RegistryClient != nil {
					if err := cfg.RegistryClient.Heartbeat(cfg.Endpoint.Module, cfg.Endpoint.Host, cfg.Endpoint.Port); err != nil {
						log.Printf("[heartbeat] remote heartbeat failed: %v", err)
					}
				} else if cfg.Registry != nil {
					cfg.Registry.Heartbeat(cfg.Endpoint.Module, cfg.Endpoint.Host, cfg.Endpoint.Port)
				}
				continue
			}

			report := cfg.Checker.HealthCheck()

			if report.Status == StatusHealthy || report.Status == StatusDegraded {
				if cfg.RegistryClient != nil {
					if err := cfg.RegistryClient.Heartbeat(cfg.Endpoint.Module, cfg.Endpoint.Host, cfg.Endpoint.Port); err != nil {
						log.Printf("[heartbeat] remote heartbeat failed: %v", err)
					}
				} else if cfg.Registry != nil {
					cfg.Registry.Heartbeat(cfg.Endpoint.Module, cfg.Endpoint.Host, cfg.Endpoint.Port)
				}
			} else {
				if cfg.Registry != nil {
					cfg.Registry.MarkUnhealthy(cfg.Endpoint.Module, cfg.Endpoint.Host, cfg.Endpoint.Port)
				}
				log.Printf("[heartbeat] %s is %s", cfg.Endpoint.Module, report.Status)
			}
		}
	}
}
