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

// Protocol frames for the platform-to-engine socket. The wire format is
// newline-delimited JSON; tensors move over shared memory later, control
// stays here where it is easy to debug.
const protoVersion = 1

type frame struct {
	Op      string `json:"op"`
	Proto   int    `json:"proto,omitempty"`
	Client  string `json:"client,omitempty"`
	Detail  string `json:"detail,omitempty"`
	Message string `json:"message,omitempty"`
}

// UnixClient is the skeleton Client speaking to the Rust engine process
// over a Unix socket.
//
// What is real today: dialing and the protocol handshake, and a ping
// round trip behind Health. Generate and Embed refuse with
// ErrNotImplemented until the Rust side of the protocol lands; that is
// the honest skeleton state, stated in the error, not papered over.
type UnixClient struct {
	conn   net.Conn
	r      *bufio.Reader
	mu     sync.Mutex // serializes request/response frames on the conn
	closed bool
}

// DialUnix connects to the engine socket and performs the handshake.
// A peer that does not answer hello_ack for our protocol version is
// rejected with ErrHandshake; a missing socket wraps ErrUnavailable.
func DialUnix(ctx context.Context, socketPath string) (*UnixClient, error) {
	var d net.Dialer
	conn, err := d.DialContext(ctx, "unix", socketPath)
	if err != nil {
		return nil, fmt.Errorf("%w: dial %s: %w", ErrUnavailable, socketPath, err)
	}
	c := &UnixClient{conn: conn, r: bufio.NewReader(conn)}
	if err := c.handshake(ctx); err != nil {
		_ = conn.Close()
		return nil, err
	}
	return c, nil
}

func (c *UnixClient) handshake(ctx context.Context) error {
	if err := c.roundTripLocked(ctx,
		frame{Op: "hello", Proto: protoVersion, Client: "torch-platform"},
		"hello_ack"); err != nil {
		return err
	}
	return nil
}

// roundTripLocked sends one frame and validates the op of the reply.
// Callers hold no lock; the method takes it.
func (c *UnixClient) roundTripLocked(ctx context.Context, send frame, wantOp string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return ErrClosed
	}
	deadline, ok := ctx.Deadline()
	if !ok {
		deadline = time.Now().Add(5 * time.Second)
	}
	if err := c.conn.SetDeadline(deadline); err != nil {
		return fmt.Errorf("engine: set deadline: %w", err)
	}
	defer func() { _ = c.conn.SetDeadline(time.Time{}) }()

	line, err := json.Marshal(send)
	if err != nil {
		return fmt.Errorf("engine: marshal frame: %w", err)
	}
	if _, err := c.conn.Write(append(line, '\n')); err != nil {
		return fmt.Errorf("%w: write frame: %w", ErrHandshake, err)
	}
	raw, err := c.r.ReadBytes('\n')
	if err != nil {
		return fmt.Errorf("%w: read reply: %w", ErrHandshake, err)
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

// Generate is not built yet: the Rust token loop does not speak the
// socket protocol. Named refusal, never a fabricated stream.
func (c *UnixClient) Generate(context.Context, GenerateRequest) (<-chan TokenEvent, error) {
	return nil, fmt.Errorf("%w: generate over socket", ErrNotImplemented)
}

// Embed is not built yet, same as Generate.
func (c *UnixClient) Embed(context.Context, EmbedRequest) (EmbedResponse, error) {
	return EmbedResponse{}, fmt.Errorf("%w: embed over socket", ErrNotImplemented)
}

// Close shuts the socket down. Later calls refuse with ErrClosed.
func (c *UnixClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true
	return c.conn.Close()
}
