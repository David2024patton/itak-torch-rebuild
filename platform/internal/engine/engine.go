// Package engine defines the seam between the Go platform and the Rust
// hot loop.
//
// The platform never generates a token itself. It calls a Client; the
// implementations are Unavailable (wired when no engine is configured)
// and UnixClient, which speaks the socket protocol to the Rust engine
// process: handshake, ping, the debug self-report, streamed
// generation, and the embed operation. Every failure mode is a named error so handlers can map
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
	// ErrEngineDied means the engine process or its worker died,
	// before the generation started or mid-stream. Supervision
	// (TODO T-9.11) is what brings it back; this error is how the
	// death is named to the caller, never a silently closed stream.
	ErrEngineDied = errors.New("engine: engine died")
	// ErrGenerationFailed means the engine refused or failed a
	// generation it had already started (a forward or detokenize
	// failure at a decode step). It arrives as the terminal
	// TokenEvent's Err, not as Generate's return value.
	ErrGenerationFailed = errors.New("engine: generation failed")
	// ErrEngineBusy means the engine refused the job before it
	// started: every serving slot is running and the bounded job
	// queue is full (TODO T-9.14). The refusal is explicit and the
	// caller may retry; nothing was queued silently.
	ErrEngineBusy = errors.New("engine: busy")
	// ErrNoChatTemplate means the model file carries no chat
	// template (TODO T-9.16). The platform falls back to role
	// flattening only with this named outcome in hand, never by
	// silently assuming a format.
	ErrNoChatTemplate = errors.New("engine: no chat template")
	// ErrTimeout means the engine did not answer within the
	// exchange deadline. A slow but healthy engine reports as a
	// timeout, never as a handshake failure: the handshake is a
	// different event and mislabeling it sends operators hunting
	// for a protocol fault that does not exist.
	ErrTimeout = errors.New("engine: timed out waiting for the engine")
	// ErrFrameTooLarge means a peer sent a frame past the protocol
	// size cap without terminating it. The read refuses by name
	// instead of buffering without bound.
	ErrFrameTooLarge = errors.New("engine: frame exceeds the maximum size")
)

// GenerateRequest is one completion job handed to the engine. The
// sampling parity fields (TODO T-9.13) are pointers so an explicit
// zero (a seed of 0, a penalty of 0) is distinguishable from an
// absent field all the way to the wire.
type GenerateRequest struct {
	Model       string   `json:"model"`
	Prompt      string   `json:"prompt"`
	MaxTokens   int      `json:"max_tokens,omitempty"`
	Temperature float64  `json:"temperature,omitempty"`
	TopP        float64  `json:"top_p,omitempty"`
	Stop        []string `json:"stop,omitempty"`
	// Seed makes a sampled run reproducible; nil lets the engine
	// seed from its clock.
	Seed *uint64 `json:"seed,omitempty"`
	// TopK caps the sampler's candidate set; nil or 0 disables it.
	TopK *int `json:"top_k,omitempty"`
	// PresencePenalty and FrequencyPenalty are the OpenAI-form
	// repetition penalties; nil or 0 disables each.
	PresencePenalty  *float64 `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64 `json:"frequency_penalty,omitempty"`
}

// TokenEvent is one streamed piece of a generation. Done marks the final
// event; FinishReason is set on it ("stop", "length", ...).
//
// Err is set on exactly one kind of event: the terminal error event.
// When a generation fails after it has started (the engine reports a
// named failure, or the engine dies mid-stream), the client delivers
// one last event with Err non-nil and Done false, then closes the
// channel. A stream that ends without either a Done event or an Err
// event is a contract violation by the client implementation, and
// handlers treat it as an engine death rather than a short success.
type TokenEvent struct {
	Text         string `json:"text"`
	TokenID      int32  `json:"token_id"`
	Done         bool   `json:"done"`
	FinishReason string `json:"finish_reason,omitempty"`
	// Err is the terminal failure, set only on the final event of a
	// failed generation. It is never serialized; the wire form is
	// the protocol's error frame.
	Err error `json:"-"`
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

// ChatTemplate is the model's own chat template as the engine
// serves it: the template text from the GGUF metadata and the
// special-token texts its variables (bos_token, eos_token) refer
// to. The platform renders it; the engine only reports it.
type ChatTemplate struct {
	Template string `json:"template"`
	BOSToken string `json:"bos_token"`
	EOSToken string `json:"eos_token"`
}

// Client is the platform's only view of the token engine.
//
// Generate returns a channel of token events. A non-nil error return
// means generation never started. A failure after that is delivered
// as one terminal TokenEvent with Err set (see TokenEvent), and
// implementations close the channel exactly once in all paths.
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
