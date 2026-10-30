package engine

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"sync"
	"time"
)

// Protocol frames for the platform-to-engine socket. The wire format
// is newline-delimited JSON; tensors move over shared memory later,
// control stays here where it is easy to debug. The protocol is
// defined once, in the Rust engine crate (crates/itak-engine/src/
// protocol.rs); this file implements the same shapes and nothing
// else.
const protoVersion = 1

type frame struct {
	Op      string `json:"op"`
	Proto   int    `json:"proto,omitempty"`
	Client  string `json:"client,omitempty"`
	Detail  string `json:"detail,omitempty"`
	Message string `json:"message,omitempty"`
}

// genFrame is the generate request the client sends and the streamed
// frames the engine answers with: token events (op "token"), one
// terminal token frame with Done set, or a named refusal (op
// "error" with a Code the client maps onto its error vocabulary).
type genFrame struct {
	Op               string   `json:"op"`
	Model            string   `json:"model,omitempty"`
	Prompt           string   `json:"prompt,omitempty"`
	MaxTokens        int      `json:"max_tokens,omitempty"`
	Temperature      float64  `json:"temperature,omitempty"`
	TopP             float64  `json:"top_p,omitempty"`
	Stop             []string `json:"stop,omitempty"`
	Seed             *uint64  `json:"seed,omitempty"`
	TopK             *int     `json:"top_k,omitempty"`
	PresencePenalty  *float64 `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64 `json:"frequency_penalty,omitempty"`
	Text             string   `json:"text,omitempty"`
	TokenID          int32    `json:"token_id,omitempty"`
	Done             bool     `json:"done,omitempty"`
	FinishReason     string   `json:"finish_reason,omitempty"`
	Code             string   `json:"code,omitempty"`
	Message          string   `json:"message,omitempty"`
}

// UnixClient is the Client speaking to the Rust engine process over
// a Unix socket.
//
// What is real: dialing and the protocol handshake, the ping round
// trip behind Health, the debug self-report, and Generate, which
// streams token events from the engine. Generate runs on a fresh
// connection per call (handshake included), so a generation that
// runs for minutes never blocks Health or DebugInfo on the control
// connection; the engine runs a configured number of slots and
// queues a bounded number of jobs behind them, refusing the rest
// with ErrEngineBusy. Embed runs the protocol's embed operation on
// the control connection.
type UnixClient struct {
	conn       net.Conn
	r          *bufio.Reader
	socketPath string
	mu         sync.Mutex // serializes request/response frames on the control conn
	closed     bool
}

// DialUnix connects to the engine socket and performs the handshake.
// A peer that does not answer hello_ack for our protocol version is
// rejected with ErrHandshake; a missing socket wraps ErrUnavailable.
func DialUnix(ctx context.Context, socketPath string) (*UnixClient, error) {
	conn, err := dialRaw(ctx, socketPath)
	if err != nil {
		return nil, err
	}
	c := &UnixClient{conn: conn, r: bufio.NewReader(conn), socketPath: socketPath}
	if err := c.handshake(ctx); err != nil {
		_ = conn.Close()
		return nil, err
	}
	return c, nil
}

func dialRaw(ctx context.Context, socketPath string) (net.Conn, error) {
	var d net.Dialer
	conn, err := d.DialContext(ctx, "unix", socketPath)
	if err != nil {
		return nil, fmt.Errorf("%w: dial %s: %w", ErrUnavailable, socketPath, err)
	}
	return conn, nil
}

func (c *UnixClient) handshake(ctx context.Context) error {
	return c.roundTripLocked(ctx,
		frame{Op: "hello", Proto: protoVersion, Client: "torch-platform"},
		"hello_ack")
}

// exchangeLocked sends one frame on the control connection and
// returns the raw reply line. Callers hold no lock; the method takes
// it.
func (c *UnixClient) exchangeLocked(ctx context.Context, send frame) ([]byte, error) {
	return c.exchangeAnyLocked(ctx, send)
}

// exchangeAnyLocked is exchangeLocked for the other frame shapes
// (generate is not one of them: it runs on its own connection).
func (c *UnixClient) exchangeAnyLocked(ctx context.Context, send any) ([]byte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil, ErrClosed
	}
	deadline, ok := ctx.Deadline()
	if !ok {
		deadline = time.Now().Add(5 * time.Second)
	}
	if err := c.conn.SetDeadline(deadline); err != nil {
		return nil, fmt.Errorf("engine: set deadline: %w", err)
	}
	defer func() { _ = c.conn.SetDeadline(time.Time{}) }()

	line, err := json.Marshal(send)
	if err != nil {
		return nil, fmt.Errorf("engine: marshal frame: %w", err)
	}
	if _, err := c.conn.Write(append(line, '\n')); err != nil {
		return nil, fmt.Errorf("%w: write frame: %w", ErrHandshake, err)
	}
	raw, err := c.r.ReadBytes('\n')
	if err != nil {
		return nil, fmt.Errorf("%w: read reply: %w", ErrHandshake, err)
	}
	return raw, nil
}

// roundTripLocked sends one frame and validates the op of the reply.
// Callers hold no lock; the method takes it.
func (c *UnixClient) roundTripLocked(ctx context.Context, send frame, wantOp string) error {
	raw, err := c.exchangeLocked(ctx, send)
	if err != nil {
		return err
	}
	var reply frame
	if err := json.Unmarshal(raw, &reply); err != nil {
		return fmt.Errorf("%w: reply is not JSON: %w", ErrHandshake, err)
	}
	if reply.Op != wantOp {
		return fmt.Errorf("%w: got op %q, want %q", ErrHandshake, reply.Op, wantOp)
	}
	if send.Op == "hello" && reply.Proto != protoVersion {
		return fmt.Errorf("%w: protocol %d, want %d", ErrHandshake, reply.Proto, protoVersion)
	}
	return nil
}

// Health pings the engine over the socket.
func (c *UnixClient) Health(ctx context.Context) (HealthStatus, error) {
	if err := c.roundTripLocked(ctx, frame{Op: "ping"}, "pong"); err != nil {
		if errors.Is(err, ErrClosed) {
			return HealthStatus{OK: false, Detail: "client closed"}, nil
		}
		return HealthStatus{OK: false, Detail: err.Error()}, nil
	}
	return HealthStatus{OK: true, Detail: "socket handshake + ping ok"}, nil
}

// DebugInfo returns the engine's self-report: the loaded model, its
// coverage-gate result, and its serving counters, as the engine
// reported them. The platform's /debug route includes it when the
// wired client offers it.
func (c *UnixClient) DebugInfo(ctx context.Context) (map[string]any, error) {
	raw, err := c.exchangeLocked(ctx, frame{Op: "debug"})
	if err != nil {
		return nil, err
	}
	var info map[string]any
	if err := json.Unmarshal(raw, &info); err != nil {
		return nil, fmt.Errorf("%w: debug reply is not JSON: %w", ErrHandshake, err)
	}
	if op, _ := info["op"].(string); op != "debug_info" {
		return nil, fmt.Errorf("%w: got op %q, want %q", ErrHandshake, op, "debug_info")
	}
	return info, nil
}

// Generate runs one completion on the engine and streams its token
// events. The call blocks until the engine produces its first frame:
// a refusal that arrives before any token (a bad prompt, a context
// overflow, a model mismatch) is returned as the call's error, mapped
// onto the named vocabulary, instead of surfacing as an empty stream.
// An engine failure after streaming has begun (a named error frame,
// or the connection dying with the engine process) is delivered as
// one terminal TokenEvent with Err set, so a mid-generation failure
// reaches the caller by name instead of ending as a silently closed
// stream (TODO T-9.12).
func (c *UnixClient) Generate(ctx context.Context, req GenerateRequest) (<-chan TokenEvent, error) {
	c.mu.Lock()
	closed := c.closed
	c.mu.Unlock()
	if closed {
		return nil, ErrClosed
	}
	conn, err := dialRaw(ctx, c.socketPath)
	if err != nil {
		return nil, err
	}
	gc := &UnixClient{conn: conn, r: bufio.NewReader(conn), socketPath: c.socketPath}
	if err := gc.handshake(ctx); err != nil {
		_ = conn.Close()
		return nil, err
	}
	wire := genFrame{
		Op:               "generate",
		Model:            req.Model,
		Prompt:           req.Prompt,
		MaxTokens:        req.MaxTokens,
		Temperature:      req.Temperature,
		TopP:             req.TopP,
		Stop:             req.Stop,
		Seed:             req.Seed,
		TopK:             req.TopK,
		PresencePenalty:  req.PresencePenalty,
		FrequencyPenalty: req.FrequencyPenalty,
	}
	line, err := json.Marshal(wire)
	if err != nil {
		_ = conn.Close()
		return nil, fmt.Errorf("engine: marshal generate frame: %w", err)
	}
	if _, err := conn.Write(append(line, '\n')); err != nil {
		_ = conn.Close()
		return nil, fmt.Errorf("%w: write generate frame: %w", ErrUnavailable, err)
	}

	out := make(chan TokenEvent, 16)
	started := make(chan error, 1)
	go gc.streamGenerate(ctx, out, started)
	select {
	case err := <-started:
		if err != nil {
			return nil, err
		}
		return out, nil
	case <-ctx.Done():
		_ = conn.Close()
		return nil, ctx.Err()
	}
}

// streamGenerate is the read loop behind Generate. It owns the
// per-generation connection, closes the event channel exactly once,
// and reports how the stream began on started (nil once the first
// token frame has been accepted, the refusal otherwise), exactly
// once.
func (c *UnixClient) streamGenerate(ctx context.Context, out chan<- TokenEvent, started chan<- error) {
	defer close(out)
	defer func() { _ = c.conn.Close() }()
	// A caller that goes away mid-generation unblocks the read.
	go func() {
		<-ctx.Done()
		_ = c.conn.Close()
	}()

	signaled := false
	signal := func(err error) {
		if !signaled {
			signaled = true
			started <- err
		}
	}
	// fail delivers a post-start failure as the terminal event: the
	// caller sees the named error, never a bare channel close.
	fail := func(err error) {
		select {
		case out <- TokenEvent{Err: err}:
		case <-ctx.Done():
		}
	}
	for {
		raw, err := c.r.ReadBytes('\n')
		if err != nil {
			died := fmt.Errorf("%w: generate stream closed early: %w", ErrEngineDied, err)
			switch {
			case ctx.Err() != nil:
				signal(ctx.Err())
			case signaled:
				// The connection died mid-generation: the engine
				// process is gone or the socket broke under it.
				fail(died)
			default:
				signal(died)
			}
			return
		}
		var f genFrame
		if err := json.Unmarshal(raw, &f); err != nil {
			err := fmt.Errorf("%w: generate reply is not JSON: %w", ErrHandshake, err)
			if signaled {
				fail(err)
			} else {
				signal(err)
			}
			return
		}
		switch f.Op {
		case "token":
			signal(nil)
			ev := TokenEvent{
				Text:         f.Text,
				TokenID:      f.TokenID,
				Done:         f.Done,
				FinishReason: f.FinishReason,
			}
			select {
			case out <- ev:
			case <-ctx.Done():
				return
			}
			if f.Done {
				return
			}
		case "error":
			// Before the first token, the refusal is Generate's
			// error return. After it, the refusal is the terminal
			// error event (TODO T-9.12).
			if signaled {
				fail(mapEngineError(f))
			} else {
				signal(mapEngineError(f))
			}
			return
		default:
			err := fmt.Errorf("%w: unexpected op %q in generate stream", ErrHandshake, f.Op)
			if signaled {
				fail(err)
			} else {
				signal(err)
			}
			return
		}
	}
}

func mapEngineError(f genFrame) error {
	msg := f.Message
	if msg == "" {
		msg = f.Code
	}
	switch f.Code {
	case "bad_request", "context_overflow", "model_not_found", "protocol_mismatch":
		return fmt.Errorf("%w: %s", ErrBadRequest, msg)
	case "not_implemented":
		return fmt.Errorf("%w: %s", ErrNotImplemented, msg)
	case "engine_died":
		return fmt.Errorf("%w: %s", ErrEngineDied, msg)
	case "engine_busy":
		return fmt.Errorf("%w: %s", ErrEngineBusy, msg)
	case "forward_failed", "detokenize_failed":
		return fmt.Errorf("%w: %s: %s", ErrGenerationFailed, f.Code, msg)
	default:
		return fmt.Errorf("engine: %s: %s", f.Code, msg)
	}
}

// embedFrame is the embed request and reply shape: the request
// carries the inputs, the reply carries one vector per input in
// order, or an error frame with a named code (TODO T-9.15).
type embedFrame struct {
	Op         string      `json:"op"`
	Model      string      `json:"model,omitempty"`
	Input      []string    `json:"input,omitempty"`
	Embeddings [][]float32 `json:"embeddings,omitempty"`
	Code       string      `json:"code,omitempty"`
	Message    string      `json:"message,omitempty"`
}

// Embed runs the protocol's embed operation on the control
// connection: the engine tokenizes each input and answers with
// mean-pooled, L2-normalized vectors from the model's own embedding
// table. An engine error frame maps onto the named vocabulary like
// a generate refusal does.
func (c *UnixClient) Embed(ctx context.Context, req EmbedRequest) (EmbedResponse, error) {
	raw, err := c.exchangeAnyLocked(ctx, embedFrame{Op: "embed", Model: req.Model, Input: req.Input})
	if err != nil {
		return EmbedResponse{}, err
	}
	var reply embedFrame
	if err := json.Unmarshal(raw, &reply); err != nil {
		return EmbedResponse{}, fmt.Errorf("%w: embed reply is not JSON: %w", ErrHandshake, err)
	}
	switch reply.Op {
	case "embeddings":
		model := reply.Model
		if model == "" {
			model = req.Model
		}
		return EmbedResponse{Model: model, Vectors: reply.Embeddings}, nil
	case "error":
		return EmbedResponse{}, mapEngineError(genFrame{Code: reply.Code, Message: reply.Message})
	default:
		return EmbedResponse{}, fmt.Errorf("%w: got op %q, want %q", ErrHandshake, reply.Op, "embeddings")
	}
}

// Close shuts the control connection down. Later calls refuse with
// ErrClosed; generations already streaming on their own connections
// are governed by their request contexts.
func (c *UnixClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true
	return c.conn.Close()
}
