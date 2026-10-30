package engine_test

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"net"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/David2024patton/itak-torch-rebuild/platform/internal/engine"
)

func TestUnavailableRefusesEverything(t *testing.T) {
	var c engine.Client = engine.Unavailable{}
	ctx := context.Background()

	if _, err := c.Generate(ctx, engine.GenerateRequest{}); !errors.Is(err, engine.ErrUnavailable) {
		t.Fatalf("Generate err = %v, want ErrUnavailable", err)
	}
	if _, err := c.Embed(ctx, engine.EmbedRequest{}); !errors.Is(err, engine.ErrUnavailable) {
		t.Fatalf("Embed err = %v, want ErrUnavailable", err)
	}
	h, err := c.Health(ctx)
	if err != nil || h.OK {
		t.Fatalf("Health = %+v, %v; want OK=false, nil error", h, err)
	}
}

// fakeEngine serves the socket protocol with configurable replies.
func fakeEngine(t *testing.T, ackProto int, ackOp string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "engine.sock")
	ln, err := net.Listen("unix", path)
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	t.Cleanup(func() { _ = ln.Close() })
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go serveConn(conn, ackProto, ackOp)
		}
	}()
	return path
}

func serveConn(conn net.Conn, ackProto int, ackOp string) {
	defer func() { _ = conn.Close() }()
	r := bufio.NewReader(conn)
	write := func(reply map[string]any) bool {
		out, _ := json.Marshal(reply)
		_, err := conn.Write(append(out, '\n'))
		return err == nil
	}
	for {
		line, err := r.ReadBytes('\n')
		if err != nil {
			return
		}
		var f struct {
			Op     string   `json:"op"`
			Prompt string   `json:"prompt"`
			Input  []string `json:"input"`
		}
		if err := json.Unmarshal(line, &f); err != nil {
			return
		}
		switch f.Op {
		case "hello":
			if !write(map[string]any{"op": ackOp, "proto": ackProto}) {
				return
			}
		case "ping":
			if !write(map[string]any{"op": "pong"}) {
				return
			}
		case "embed":
			if !serveEmbed(write, f.Input) {
				return
			}
		case "debug":
			if !write(map[string]any{"op": "debug_info", "model": "fake-1b", "generations": 3}) {
				return
			}
		case "chat_template":
			if !write(map[string]any{
				"op": "chat_template", "template": "{% for message in messages %}X{% endfor %}",
				"bos_token": "<s>", "eos_token": "</s>",
			}) {
				return
			}
		case "generate":
			if !serveGenerate(write, f.Prompt) {
				return
			}
		default:
			if !write(map[string]any{"op": "error", "message": "unknown op"}) {
				return
			}
		}
	}
}

// serveEmbed answers an embed request with one fixed vector per
// input; an input of "refuse" scripts the named refusal an engine
// without the operation (or with a full queue) would send.
func serveEmbed(write func(map[string]any) bool, inputs []string) bool {
	vecs := make([][]float32, 0, len(inputs))
	for _, in := range inputs {
		if in == "refuse" {
			return write(map[string]any{
				"op": "error", "code": "not_implemented", "message": "embed is not built",
			})
		}
		vecs = append(vecs, []float32{0.1, 0.2, 0.3})
	}
	return write(map[string]any{"op": "embeddings", "model": "fake-1b", "embeddings": vecs})
}

// serveGenerate scripts the generate stream off the prompt text, so
// one fake covers the success path and both error placements.
func serveGenerate(write func(map[string]any) bool, prompt string) bool {
	switch {
	case strings.Contains(prompt, "err-overflow"):
		return write(map[string]any{
			"op": "error", "code": "context_overflow",
			"message": "prompt of 9 tokens plus max_tokens 4 exceeds the context length 8",
		})
	case strings.Contains(prompt, "err-busy"):
		return write(map[string]any{
			"op": "error", "code": "engine_busy",
			"message": "engine busy: 1 slot(s) running and the queue is full (max_queued 0)",
		})
	case strings.Contains(prompt, "err-mid"):
		if !write(map[string]any{"op": "token", "text": " Paris", "token_id": 13796}) {
			return false
		}
		return write(map[string]any{
			"op": "error", "code": "forward_failed", "message": "decode step 2: boom",
		})
	case strings.Contains(prompt, "die-close"):
		// One token, then the connection drops with no frame at
		// all: the engine process dying mid-generation.
		if !write(map[string]any{"op": "token", "text": " Paris", "token_id": 13796}) {
			return false
		}
		return false
	default:
		if !write(map[string]any{"op": "token", "text": " Paris", "token_id": 13796}) {
			return false
		}
		if !write(map[string]any{"op": "token", "text": ". It", "token_id": 26}) {
			return false
		}
		return write(map[string]any{
			"op": "token", "text": "", "token_id": 0, "done": true, "finish_reason": "stop",
		})
	}
}

func dial(t *testing.T, path string) *engine.UnixClient {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	c, err := engine.DialUnix(ctx, path)
	if err != nil {
		t.Fatalf("DialUnix: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return c
}

func TestUnixHandshakeAndHealth(t *testing.T) {
	c := dial(t, fakeEngine(t, 1, "hello_ack"))
	h, err := c.Health(context.Background())
	if err != nil || !h.OK {
		t.Fatalf("Health = %+v, %v; want OK=true", h, err)
	}
}

func TestUnixGenerateStreamsTokens(t *testing.T) {
	c := dial(t, fakeEngine(t, 1, "hello_ack"))
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	ch, err := c.Generate(ctx, engine.GenerateRequest{Model: "fake-1b", Prompt: "The capital of France is"})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	var events []engine.TokenEvent
	for ev := range ch {
		events = append(events, ev)
	}
	if len(events) != 3 {
		t.Fatalf("got %d events, want 3: %+v", len(events), events)
	}
	if events[0].Text != " Paris" || events[0].TokenID != 13796 || events[0].Done {
		t.Fatalf("event 0 = %+v, want the Paris token event", events[0])
	}
	if events[1].Text != ". It" || events[1].TokenID != 26 {
		t.Fatalf("event 1 = %+v", events[1])
	}
	if !events[2].Done || events[2].FinishReason != "stop" {
		t.Fatalf("event 2 = %+v, want done with finish_reason stop", events[2])
	}
}

func TestUnixGenerateBusyIsEngineBusy(t *testing.T) {
	c := dial(t, fakeEngine(t, 1, "hello_ack"))
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := c.Generate(ctx, engine.GenerateRequest{Prompt: "err-busy please"})
	if !errors.Is(err, engine.ErrEngineBusy) {
		t.Fatalf("Generate err = %v, want ErrEngineBusy", err)
	}
}

func TestUnixGenerateStartErrorIsBadRequest(t *testing.T) {
	c := dial(t, fakeEngine(t, 1, "hello_ack"))
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := c.Generate(ctx, engine.GenerateRequest{Prompt: "err-overflow please"})
	if !errors.Is(err, engine.ErrBadRequest) {
		t.Fatalf("Generate err = %v, want ErrBadRequest", err)
	}
}

func TestUnixGenerateMidStreamErrorIsTerminalEvent(t *testing.T) {
	c := dial(t, fakeEngine(t, 1, "hello_ack"))
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	ch, err := c.Generate(ctx, engine.GenerateRequest{Prompt: "err-mid please"})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	var events []engine.TokenEvent
	for ev := range ch {
		events = append(events, ev)
	}
	if len(events) != 2 {
		t.Fatalf("events = %+v, want one token event and one terminal error event", events)
	}
	if events[0].Text != " Paris" || events[0].Done || events[0].Err != nil {
		t.Fatalf("event 0 = %+v, want the Paris token event", events[0])
	}
	last := events[1]
	if last.Done || !errors.Is(last.Err, engine.ErrGenerationFailed) {
		t.Fatalf("terminal event = %+v (err %v), want ErrGenerationFailed and no done", last, last.Err)
	}
	if !strings.Contains(last.Err.Error(), "decode step 2") {
		t.Fatalf("terminal error = %v, want the engine's message carried", last.Err)
	}
}

func TestUnixGenerateConnectionDeathIsEngineDiedEvent(t *testing.T) {
	c := dial(t, fakeEngine(t, 1, "hello_ack"))
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	ch, err := c.Generate(ctx, engine.GenerateRequest{Prompt: "die-close please"})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	var events []engine.TokenEvent
	for ev := range ch {
		events = append(events, ev)
	}
	if len(events) != 2 {
		t.Fatalf("events = %+v, want one token event and one terminal error event", events)
	}
	if !errors.Is(events[1].Err, engine.ErrEngineDied) {
		t.Fatalf("terminal err = %v, want ErrEngineDied", events[1].Err)
	}
}

func TestUnixEmbedReturnsVectors(t *testing.T) {
	c := dial(t, fakeEngine(t, 1, "hello_ack"))
	resp, err := c.Embed(context.Background(), engine.EmbedRequest{
		Model: "fake-1b", Input: []string{"cats", "dogs"},
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if resp.Model != "fake-1b" || len(resp.Vectors) != 2 {
		t.Fatalf("Embed resp = %+v, want model fake-1b and 2 vectors", resp)
	}
	if len(resp.Vectors[0]) != 3 || resp.Vectors[0][0] != 0.1 {
		t.Fatalf("vector 0 = %v, want the fake's [0.1 0.2 0.3]", resp.Vectors[0])
	}
}

func TestUnixEmbedRefusalMapsByName(t *testing.T) {
	c := dial(t, fakeEngine(t, 1, "hello_ack"))
	_, err := c.Embed(context.Background(), engine.EmbedRequest{Input: []string{"refuse"}})
	if !errors.Is(err, engine.ErrNotImplemented) {
		t.Fatalf("Embed err = %v, want ErrNotImplemented", err)
	}
}

// capturingEngine is a fake that records the raw generate frame it
// receives, so a test can assert exactly what crossed the wire.
func capturingEngine(t *testing.T) (string, func() map[string]any) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "engine.sock")
	ln, err := net.Listen("unix", path)
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	t.Cleanup(func() { _ = ln.Close() })
	var mu sync.Mutex
	var lastGen map[string]any
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go func(conn net.Conn) {
				defer func() { _ = conn.Close() }()
				r := bufio.NewReader(conn)
				for {
					line, err := r.ReadBytes('\n')
					if err != nil {
						return
					}
					var f map[string]any
					if json.Unmarshal(line, &f) != nil {
						return
					}
					var reply map[string]any
					switch f["op"] {
					case "hello":
						reply = map[string]any{"op": "hello_ack", "proto": 1}
					case "generate":
						mu.Lock()
						lastGen = f
						mu.Unlock()
						serveGenerate(func(m map[string]any) bool {
							out, _ := json.Marshal(m)
							_, err := conn.Write(append(out, '\n'))
							return err == nil
						}, "")
						return
					default:
						reply = map[string]any{"op": "pong"}
					}
					out, _ := json.Marshal(reply)
					if _, err := conn.Write(append(out, '\n')); err != nil {
						return
					}
				}
			}(conn)
		}
	}()
	return path, func() map[string]any {
		mu.Lock()
		defer mu.Unlock()
		return lastGen
	}
}

func TestUnixGenerateSamplingFieldsCrossTheWire(t *testing.T) {
	path, lastGen := capturingEngine(t)
	c := dial(t, path)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	seed, topK := uint64(42), 40
	pres, freq := 0.5, -0.25
	ch, err := c.Generate(ctx, engine.GenerateRequest{
		Prompt: "x", Seed: &seed, TopK: &topK,
		PresencePenalty: &pres, FrequencyPenalty: &freq,
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	for range ch {
	}
	frame := lastGen()
	if frame == nil {
		t.Fatal("fake engine captured no generate frame")
	}
	if frame["seed"] != float64(42) || frame["top_k"] != float64(40) ||
		frame["presence_penalty"] != 0.5 || frame["frequency_penalty"] != -0.25 {
		t.Fatalf("generate frame sampling fields = %v, want seed 42, top_k 40, penalties 0.5/-0.25", frame)
	}

	// Absent fields stay absent: an unset seed must not become a
	// wire 0 the engine would mistake for an explicit seed.
	ch, err = c.Generate(ctx, engine.GenerateRequest{Prompt: "x"})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	for range ch {
	}
	frame = lastGen()
	for _, key := range []string{"seed", "top_k", "presence_penalty", "frequency_penalty"} {
		if _, present := frame[key]; present {
			t.Fatalf("unset %s leaked onto the wire: %v", key, frame)
		}
	}
}

func TestUnixChatTemplate(t *testing.T) {
	c := dial(t, fakeEngine(t, 1, "hello_ack"))
	tpl, err := c.ChatTemplate(context.Background())
	if err != nil {
		t.Fatalf("ChatTemplate: %v", err)
	}
	if tpl.BOSToken != "<s>" || tpl.EOSToken != "</s>" || !strings.Contains(tpl.Template, "for message in messages") {
		t.Fatalf("template = %+v, want the fake's template and special tokens", tpl)
	}
}

// noTemplateEngine serves hello and answers the chat_template op
// the way an engine whose model file carries no template does.
func noTemplateEngine(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "engine.sock")
	ln, err := net.Listen("unix", path)
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	t.Cleanup(func() { _ = ln.Close() })
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go func(conn net.Conn) {
				defer func() { _ = conn.Close() }()
				r := bufio.NewReader(conn)
				for {
					line, err := r.ReadBytes('\n')
					if err != nil {
						return
					}
					var f struct {
						Op string `json:"op"`
					}
					if json.Unmarshal(line, &f) != nil {
						return
					}
					var reply map[string]any
					switch f.Op {
					case "hello":
						reply = map[string]any{"op": "hello_ack", "proto": 1}
					case "chat_template":
						reply = map[string]any{
							"op": "error", "code": "no_chat_template",
							"message": "the model file carries no tokenizer.chat_template",
						}
					default:
						reply = map[string]any{"op": "pong"}
					}
					out, _ := json.Marshal(reply)
					if _, err := conn.Write(append(out, '\n')); err != nil {
						return
					}
				}
			}(conn)
		}
	}()
	return path
}

func TestUnixChatTemplateAbsentIsNamed(t *testing.T) {
	c := dial(t, noTemplateEngine(t))
	_, err := c.ChatTemplate(context.Background())
	if !errors.Is(err, engine.ErrNoChatTemplate) {
		t.Fatalf("ChatTemplate err = %v, want ErrNoChatTemplate", err)
	}
}

func TestUnixDebugInfo(t *testing.T) {
	c := dial(t, fakeEngine(t, 1, "hello_ack"))
	info, err := c.DebugInfo(context.Background())
	if err != nil {
		t.Fatalf("DebugInfo: %v", err)
	}
	if info["model"] != "fake-1b" {
		t.Fatalf("debug info = %v, want model fake-1b", info)
	}
}

func TestUnixHandshakeRejectsWrongProtocol(t *testing.T) {
	path := fakeEngine(t, 99, "hello_ack")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := engine.DialUnix(ctx, path); !errors.Is(err, engine.ErrHandshake) {
		t.Fatalf("DialUnix err = %v, want ErrHandshake", err)
	}
}

func TestUnixHandshakeRejectsWrongOp(t *testing.T) {
	path := fakeEngine(t, 1, "nope")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := engine.DialUnix(ctx, path); !errors.Is(err, engine.ErrHandshake) {
		t.Fatalf("DialUnix err = %v, want ErrHandshake", err)
	}
}

func TestUnixDialMissingSocketIsUnavailable(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	missing := filepath.Join(t.TempDir(), "absent.sock")
	if _, err := engine.DialUnix(ctx, missing); !errors.Is(err, engine.ErrUnavailable) {
		t.Fatalf("DialUnix err = %v, want ErrUnavailable", err)
	}
}

func TestUnixUseAfterClose(t *testing.T) {
	c := dial(t, fakeEngine(t, 1, "hello_ack"))
	if err := c.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := c.Health(context.Background()); err != nil {
		t.Fatalf("Health after close should report, not error: %v", err)
	}
	// Second Close is a no-op, not an error.
	if err := c.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}
