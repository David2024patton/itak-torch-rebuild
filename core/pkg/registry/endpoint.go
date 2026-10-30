package registry

import (
	"fmt"
	"time"
)

// Protocol identifies how a service endpoint communicates.
type Protocol string

const (
	ProtocolGRPC      Protocol = "grpc"      // Internal high-perf calls
	ProtocolREST      Protocol = "rest"      // External / Dashboard calls
	ProtocolWebSocket Protocol = "websocket" // Live streaming (Dashboard, Browser)
	ProtocolSSE       Protocol = "sse"       // Server-sent events (logs, monitoring)
)

// ServiceEndpoint describes a single running service instance.
type ServiceEndpoint struct {
	// Module is the iTaK module name (e.g., "torch", "gateway", "shield").
	Module string `json:"module"`

	// Host is the network address (e.g., "127.0.0.1", "skynet.local").
	Host string `json:"host"`

	// Port is the listening port.
	Port int `json:"port"`

	// Protocol is the communication protocol.
	Protocol Protocol `json:"protocol"`

	// Healthy indicates whether the last health check passed.
	Healthy bool `json:"healthy"`

	// LastSeen is the timestamp of the last successful heartbeat.
	LastSeen time.Time `json:"last_seen"`

	// PublicHost is the external/public IP address, learned from the tunnel
	// broker when the agent connects. Used for direct peer-to-peer connections.
	// Empty for local-only endpoints.
	PublicHost string `json:"public_host,omitempty"`

	// TunnelID identifies the tunnel agent this endpoint is reachable through.
	// Empty means the endpoint is directly addressable (no tunnel needed).
	TunnelID string `json:"tunnel_id,omitempty"`

	// Metadata holds arbitrary key-value pairs (e.g., "backend": "vulkan").
	Metadata map[string]string `json:"metadata,omitempty"`
}

// Address returns the host:port string for this endpoint.
func (e ServiceEndpoint) Address() string {
	return fmt.Sprintf("%s:%d", e.Host, e.Port)
}
