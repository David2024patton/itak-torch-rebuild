// Package engine defines the seam between the Go platform and the Rust
// hot loop.
//
// The platform never generates a token itself. It calls a Client; the
// implementations are Unavailable (wired when no engine is configured)
// and UnixClient, which speaks the socket protocol to the Rust engine
// process: handshake, ping, the debug self-report, and streamed
// generation. Embed still refuses by name on both, because the
// protocol has no embedding operation yet. Every failure mode is a named error so handlers can map
// it to an honest HTTP status instead of inventing a success response,
// the failure pattern the old platform kept falling into (fake LoRA
// loads, an itak-quant that printed SUCCESS and wrote nothing).
package engine

import (
	"context"
	"errors"
)

// Named engine errors. Handlers switch on these with errors.Is.
var (
	// ErrUnavailable means no engine is configured or reachable.
	ErrUnavailable = errors.New("engine: unavailable")
	// ErrNotImplemented means the client exists but the operation is not
	// built yet. It is a deliberate, named refusal, never a fake success.
	ErrNotImplemented = errors.New("engine: operation not implemented")
	// ErrHandshake means the socket client connected but the protocol
	// handshake failed (wrong peer, wrong protocol version).
	ErrHandshake = errors.New("engine: handshake failed")
	// ErrClosed means the client was used after Close.
	ErrClosed = errors.New("engine: client closed")
	// ErrBadRequest means the engine rejected the request as invalid.
	ErrBadRequest = errors.New("engine: bad request")
)

// GenerateRequest is one completion job handed to the engine.
type GenerateRequest struct {
	Model       string   `json:"model"`
	Prompt      string   `json:"prompt"`
	MaxTokens   int      `json:"max_tokens,omitempty"`
	Temperature float64  `json:"temperature,omitempty"`
	TopP        float64  `json:"top_p,omitempty"`
	Stop        []string `json:"stop,omitempty"`
}

// TokenEvent is one streamed piece of a generation. Done marks the final
// event; FinishReason is set on it ("stop", "length", ...).
type TokenEvent struct {
	Text         string `json:"text"`
	TokenID      int32  `json:"token_id"`
	Done         bool   `json:"done"`
	FinishReason string `json:"finish_reason,omitempty"`
}

// EmbedRequest asks for vectors for one or more inputs.
type EmbedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

// EmbedResponse carries one vector per input, in order.
type EmbedResponse struct {
	Model   string      `json:"model"`
	Vectors [][]float32 `json:"embeddings"`
}

// HealthStatus is the engine's answer to a health probe.
type HealthStatus struct {
	OK     bool   `json:"ok"`
	Detail string `json:"detail"`
}

// Client is the platform's only view of the token engine.
//
// Generate returns a channel of token events. A non-nil error return
// means generation never started; errors after that are the caller's
// context ending or the channel closing early, and implementations close
// the channel exactly once in all paths.
type Client interface {
	Generate(ctx context.Context, req GenerateRequest) (<-chan TokenEvent, error)
	Embed(ctx context.Context, req EmbedRequest) (EmbedResponse, error)
	Health(ctx context.Context) (HealthStatus, error)
}

// Unavailable is the Client wired when no engine is configured. Every
// method refuses with ErrUnavailable so routes surface a named
// engine-unavailable error instead of a fabricated answer.
type Unavailable struct{}

// Generate implements Client.
func (Unavailable) Generate(context.Context, GenerateRequest) (<-chan TokenEvent, error) {
	return nil, ErrUnavailable
}

// Embed implements Client.
func (Unavailable) Embed(context.Context, EmbedRequest) (EmbedResponse, error) {
	return EmbedResponse{}, ErrUnavailable
}

// Health implements Client: it reports not-OK rather than erroring, so
// /healthz can show platform-up / engine-down distinctly.
func (Unavailable) Health(context.Context) (HealthStatus, error) {
	return HealthStatus{OK: false, Detail: "no engine configured"}, nil
}
