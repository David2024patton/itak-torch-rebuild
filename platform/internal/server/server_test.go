package server_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/David2024patton/itak-torch-rebuild/platform/internal/engine"
	"github.com/David2024patton/itak-torch-rebuild/platform/internal/server"
)

const testKey = "test-key"

// fakeEngine is a deterministic stand-in at the platform's engine seam.
type fakeEngine struct {
	generateCalls atomic.Int64
	lastRequest   atomic.Value // engine.GenerateRequest
}

func (f *fakeEngine) Generate(_ context.Context, req engine.GenerateRequest) (<-chan engine.TokenEvent, error) {
	f.generateCalls.Add(1)
	f.lastRequest.Store(req)
	ch := make(chan engine.TokenEvent, 3)
	ch <- engine.TokenEvent{Text: "Hello", TokenID: 1}
	ch <- engine.TokenEvent{Text: " world", TokenID: 2}
	ch <- engine.TokenEvent{Text: "", Done: true, FinishReason: "stop"}
	close(ch)
	return ch, nil
}

func (f *fakeEngine) Embed(_ context.Context, req engine.EmbedRequest) (engine.EmbedResponse, error) {
	vecs := make([][]float32, len(req.Input))
	for i := range vecs {
		vecs[i] = []float32{0.1, 0.2, 0.3}
	}
	return engine.EmbedResponse{Model: req.Model, Vectors: vecs}, nil
}

func (f *fakeEngine) Health(context.Context) (engine.HealthStatus, error) {
	return engine.HealthStatus{OK: true, Detail: "fake engine"}, nil
}

// notImplEngine refuses Generate the way the socket skeleton does.
type notImplEngine struct{ *fakeEngine }

func (notImplEngine) Generate(context.Context, engine.GenerateRequest) (<-chan engine.TokenEvent, error) {
	return nil, engine.ErrNotImplemented
}

// midFailEngine streams one token, then fails the generation the way
// the socket client now reports a mid-stream engine failure: a
// terminal event carrying the named error (TODO T-9.12).
type midFailEngine struct{ *fakeEngine }

func (midFailEngine) Generate(context.Context, engine.GenerateRequest) (<-chan engine.TokenEvent, error) {
	ch := make(chan engine.TokenEvent, 2)
	ch <- engine.TokenEvent{Text: "Hello", TokenID: 1}
	ch <- engine.TokenEvent{Err: engine.ErrGenerationFailed}
	close(ch)
	return ch, nil
}

func newServer(t *testing.T, eng engine.Client) *httptest.Server {
	t.Helper()
	srv := server.New(server.Config{
		Engine:   eng,
		APIKey:   testKey,
		Models:   []string{"glimmer"},
		CacheTTL: time.Minute,
	})
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(func() {
		ts.Close()
		srv.Close()
	})
	return ts
}

func do(t *testing.T, ts *httptest.Server, method, path, body, key string) (int, http.Header, []byte) {
	t.Helper()
	var rdr io.Reader
	if body != "" {
		rdr = strings.NewReader(body)
	}
	req, err := http.NewRequest(method, ts.URL+path, rdr)
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	resp, err := ts.Client().Do(req)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	return resp.StatusCode, resp.Header, b
}

func errorCode(t *testing.T, body []byte) string {
	t.Helper()
	var v struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &v); err != nil {
		t.Fatalf("error body is not JSON: %v (%s)", err, body)
	}
	return v.Error.Code
}

func TestHealthzOpenWithoutKey(t *testing.T) {
	ts := newServer(t, &fakeEngine{})
	code, _, body := do(t, ts, http.MethodGet, "/healthz", "", "")
	if code != http.StatusOK {
		t.Fatalf("healthz status = %d, want 200 (%s)", code, body)
	}
	if !strings.Contains(string(body), `"ok":true`) {
		t.Fatalf("healthz body missing engine ok: %s", body)
	}
}

func TestRoutesRequireKey(t *testing.T) {
	ts := newServer(t, &fakeEngine{})
	for _, path := range []string{"/v1/models", "/debug"} {
		code, _, _ := do(t, ts, http.MethodGet, path, "", "")
		if code != http.StatusUnauthorized {
			t.Fatalf("GET %s without key = %d, want 401", path, code)
		}
	}
	code, _, body := do(t, ts, http.MethodGet, "/v1/models", "", testKey)
	if code != http.StatusOK || !strings.Contains(string(body), "glimmer") {
		t.Fatalf("models with key = %d, %s", code, body)
	}
}

const chatBody = `{"model":"glimmer","messages":[{"role":"user","content":"Say hello"}]}`

func TestChatCompletionAndCache(t *testing.T) {
	fake := &fakeEngine{}
	ts := newServer(t, fake)

	code, hdr, body := do(t, ts, http.MethodPost, "/v1/chat/completions", chatBody, testKey)
	if code != http.StatusOK {
		t.Fatalf("chat status = %d (%s)", code, body)
	}
	if hdr.Get("X-Torch-Cache") != "miss" {
		t.Fatalf("first request X-Torch-Cache = %q, want miss", hdr.Get("X-Torch-Cache"))
	}
	var v struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(body, &v); err != nil || len(v.Choices) != 1 {
		t.Fatalf("chat body = %s, err = %v", body, err)
	}
	if got := v.Choices[0].Message.Content; got != "Hello world" {
		t.Fatalf("content = %q, want %q", got, "Hello world")
	}

	code, hdr, _ = do(t, ts, http.MethodPost, "/v1/chat/completions", chatBody, testKey)
	if code != http.StatusOK || hdr.Get("X-Torch-Cache") != "hit" {
		t.Fatalf("second request = %d, X-Torch-Cache = %q; want 200/hit", code, hdr.Get("X-Torch-Cache"))
	}
	if n := fake.generateCalls.Load(); n != 1 {
		t.Fatalf("engine Generate called %d times, want 1 (second served from cache)", n)
	}
}

func TestChatStreaming(t *testing.T) {
	ts := newServer(t, &fakeEngine{})
	streamBody := `{"model":"glimmer","stream":true,"messages":[{"role":"user","content":"hi"}]}`
	code, hdr, body := do(t, ts, http.MethodPost, "/v1/chat/completions", streamBody, testKey)
	if code != http.StatusOK {
		t.Fatalf("stream status = %d (%s)", code, body)
	}
	if ct := hdr.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("Content-Type = %q, want text/event-stream", ct)
	}
	s := string(body)
	if !strings.Contains(s, "Hello") || !strings.Contains(s, "data: [DONE]") {
		t.Fatalf("stream body missing chunks or DONE: %s", s)
	}
}

func TestChatMidStreamFailureNonStreamingIsNamedError(t *testing.T) {
	ts := newServer(t, midFailEngine{&fakeEngine{}})
	code, _, body := do(t, ts, http.MethodPost, "/v1/chat/completions", chatBody, testKey)
	if code != http.StatusBadGateway || errorCode(t, body) != "generation_failed" {
		t.Fatalf("status = %d, body = %s; want 502 generation_failed, not a truncated 200", code, body)
	}
}

func TestChatMidStreamFailureStreamingIsErrorFrame(t *testing.T) {
	ts := newServer(t, midFailEngine{&fakeEngine{}})
	streamBody := `{"model":"glimmer","stream":true,"messages":[{"role":"user","content":"hi"}]}`
	code, _, body := do(t, ts, http.MethodPost, "/v1/chat/completions", streamBody, testKey)
	if code != http.StatusOK {
		t.Fatalf("stream status = %d (%s)", code, body)
	}
	s := string(body)
	if !strings.Contains(s, `"code":"generation_failed"`) {
		t.Fatalf("stream body missing the named error frame: %s", s)
	}
	if strings.Contains(s, "data: [DONE]") {
		t.Fatalf("failed stream must not end with [DONE]: %s", s)
	}
	if !strings.Contains(s, "Hello") {
		t.Fatalf("stream body missing the token sent before the failure: %s", s)
	}
}

func TestChatSamplingFieldsReachTheEngine(t *testing.T) {
	fake := &fakeEngine{}
	ts := newServer(t, fake)
	body := `{"model":"glimmer","seed":42,"top_k":40,"presence_penalty":0.5,"frequency_penalty":-0.25,` +
		`"messages":[{"role":"user","content":"hi"}]}`
	code, _, respBody := do(t, ts, http.MethodPost, "/v1/chat/completions", body, testKey)
	if code != http.StatusOK {
		t.Fatalf("status = %d (%s)", code, respBody)
	}
	got, ok := fake.lastRequest.Load().(engine.GenerateRequest)
	if !ok {
		t.Fatal("engine saw no generate request")
	}
	if got.Seed == nil || *got.Seed != 42 || got.TopK == nil || *got.TopK != 40 ||
		got.PresencePenalty == nil || *got.PresencePenalty != 0.5 ||
		got.FrequencyPenalty == nil || *got.FrequencyPenalty != -0.25 {
		t.Fatalf("engine request = %+v, want seed 42, top_k 40, penalties 0.5/-0.25", got)
	}
}

const chatMLTemplate = `{% for message in messages %}<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}`

func newServerWithConfig(t *testing.T, eng engine.Client, mutate func(*server.Config)) *httptest.Server {
	t.Helper()
	cfg := server.Config{
		Engine:   eng,
		APIKey:   testKey,
		Models:   []string{"glimmer"},
		CacheTTL: time.Minute,
	}
	if mutate != nil {
		mutate(&cfg)
	}
	srv := server.New(cfg)
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(func() {
		ts.Close()
		srv.Close()
	})
	return ts
}

func lastPrompt(t *testing.T, fake *fakeEngine) string {
	t.Helper()
	got, ok := fake.lastRequest.Load().(engine.GenerateRequest)
	if !ok {
		t.Fatal("engine saw no generate request")
	}
	return got.Prompt
}

func TestChatUsesConfiguredTemplate(t *testing.T) {
	fake := &fakeEngine{}
	ts := newServerWithConfig(t, fake, func(c *server.Config) {
		c.ChatTemplate = chatMLTemplate
	})
	code, _, body := do(t, ts, http.MethodPost, "/v1/chat/completions", chatBody, testKey)
	if code != http.StatusOK {
		t.Fatalf("chat status = %d (%s)", code, body)
	}
	want := "<|im_start|>user\nSay hello<|im_end|>\n<|im_start|>assistant\n"
	if got := lastPrompt(t, fake); got != want {
		t.Fatalf("engine prompt = %q, want the templated %q", got, want)
	}
}

// templatedEngine is a fake engine that also serves a chat template
// through the optional client capability.
type templatedEngine struct{ *fakeEngine }

func (templatedEngine) ChatTemplate(context.Context) (engine.ChatTemplate, error) {
	return engine.ChatTemplate{Template: chatMLTemplate}, nil
}

func TestChatUsesEngineTemplate(t *testing.T) {
	fake := &fakeEngine{}
	ts := newServerWithConfig(t, templatedEngine{fake}, nil)
	code, _, body := do(t, ts, http.MethodPost, "/v1/chat/completions", chatBody, testKey)
	if code != http.StatusOK {
		t.Fatalf("chat status = %d (%s)", code, body)
	}
	want := "<|im_start|>user\nSay hello<|im_end|>\n<|im_start|>assistant\n"
	if got := lastPrompt(t, fake); got != want {
		t.Fatalf("engine prompt = %q, want the engine-templated %q", got, want)
	}

	// /debug reports where the template came from.
	_, _, dbg := do(t, ts, http.MethodGet, "/debug", "", testKey)
	if !strings.Contains(string(dbg), `"chat_template":"engine"`) {
		t.Fatalf("debug missing chat_template source: %s", dbg)
	}
}

// noTemplateEngine answers the template probe with the named
// no-template outcome.
type noTemplateEngine struct{ *fakeEngine }

func (noTemplateEngine) ChatTemplate(context.Context) (engine.ChatTemplate, error) {
	return engine.ChatTemplate{}, engine.ErrNoChatTemplate
}

func TestChatWithoutTemplateFallsBackToFlattening(t *testing.T) {
	fake := &fakeEngine{}
	ts := newServerWithConfig(t, noTemplateEngine{fake}, nil)
	code, _, body := do(t, ts, http.MethodPost, "/v1/chat/completions", chatBody, testKey)
	if code != http.StatusOK {
		t.Fatalf("chat status = %d (%s)", code, body)
	}
	if got, want := lastPrompt(t, fake), "user: Say hello\nassistant:"; got != want {
		t.Fatalf("engine prompt = %q, want the flattened %q", got, want)
	}
	_, _, dbg := do(t, ts, http.MethodGet, "/debug", "", testKey)
	if !strings.Contains(string(dbg), `"chat_template":"none (role flattening)"`) {
		t.Fatalf("debug missing the flattening note: %s", dbg)
	}
}

func TestChatTemplateRenderFailureIsNamed(t *testing.T) {
	fake := &fakeEngine{}
	ts := newServerWithConfig(t, fake, func(c *server.Config) {
		c.ChatTemplate = `{% set x = 1 %}hi`
	})
	code, _, body := do(t, ts, http.MethodPost, "/v1/chat/completions", chatBody, testKey)
	if code != http.StatusBadGateway || errorCode(t, body) != "chat_template_error" {
		t.Fatalf("status = %d, body = %s; want 502 chat_template_error", code, body)
	}
}

func TestChatValidation(t *testing.T) {
	ts := newServer(t, &fakeEngine{})
	cases := map[string]string{
		"no model":      `{"messages":[{"role":"user","content":"x"}]}`,
		"no messages":   `{"model":"glimmer"}`,
		"empty role":    `{"model":"glimmer","messages":[{"role":"","content":"x"}]}`,
		"bad temp":      `{"model":"glimmer","temperature":9,"messages":[{"role":"user","content":"x"}]}`,
		"bad top_k":     `{"model":"glimmer","top_k":-1,"messages":[{"role":"user","content":"x"}]}`,
		"bad presence":  `{"model":"glimmer","presence_penalty":3,"messages":[{"role":"user","content":"x"}]}`,
		"bad frequency": `{"model":"glimmer","frequency_penalty":-3,"messages":[{"role":"user","content":"x"}]}`,
		"not json":      `{oops`,
	}
	for name, body := range cases {
		code, _, respBody := do(t, ts, http.MethodPost, "/v1/chat/completions", body, testKey)
		if code != http.StatusBadRequest || errorCode(t, respBody) != "invalid_request" {
			t.Fatalf("%s: status = %d, body = %s; want 400 invalid_request", name, code, respBody)
		}
	}
}

func TestChatWithoutEngineIsNamedUnavailable(t *testing.T) {
	ts := newServer(t, nil) // no engine configured
	code, _, body := do(t, ts, http.MethodPost, "/v1/chat/completions", chatBody, testKey)
	if code != http.StatusServiceUnavailable || errorCode(t, body) != "engine_unavailable" {
		t.Fatalf("status = %d, body = %s; want 503 engine_unavailable", code, body)
	}
}

// busyEngine refuses the way a full engine queue does (T-9.14).
type busyEngine struct{ *fakeEngine }

func (busyEngine) Generate(context.Context, engine.GenerateRequest) (<-chan engine.TokenEvent, error) {
	return nil, engine.ErrEngineBusy
}

func TestChatEngineBusyMapping(t *testing.T) {
	ts := newServer(t, busyEngine{&fakeEngine{}})
	code, _, body := do(t, ts, http.MethodPost, "/v1/chat/completions", chatBody, testKey)
	if code != http.StatusServiceUnavailable || errorCode(t, body) != "engine_busy" {
		t.Fatalf("status = %d, body = %s; want 503 engine_busy", code, body)
	}
}

func TestChatNotImplementedMapping(t *testing.T) {
	ts := newServer(t, notImplEngine{&fakeEngine{}})
	code, _, body := do(t, ts, http.MethodPost, "/v1/chat/completions", chatBody, testKey)
	if code != http.StatusNotImplemented || errorCode(t, body) != "not_implemented" {
		t.Fatalf("status = %d, body = %s; want 501 not_implemented", code, body)
	}
}

func TestEmbeddings(t *testing.T) {
	ts := newServer(t, &fakeEngine{})
	code, _, body := do(t, ts, http.MethodPost, "/v1/embeddings",
		`{"model":"glimmer","input":["a","b"]}`, testKey)
	if code != http.StatusOK {
		t.Fatalf("embeddings status = %d (%s)", code, body)
	}
	var v struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &v); err != nil || len(v.Data) != 2 {
		t.Fatalf("embeddings body = %s, err = %v", body, err)
	}

	code, _, body = do(t, ts, http.MethodPost, "/v1/embeddings", `{"model":"glimmer"}`, testKey)
	if code != http.StatusBadRequest {
		t.Fatalf("missing input status = %d (%s), want 400", code, body)
	}
}

func TestDebugReportsModules(t *testing.T) {
	ts := newServer(t, &fakeEngine{})
	// Generate some activity first.
	_, _, _ = do(t, ts, http.MethodPost, "/v1/chat/completions", chatBody, testKey)

	code, _, body := do(t, ts, http.MethodGet, "/debug", "", testKey)
	if code != http.StatusOK {
		t.Fatalf("debug status = %d (%s)", code, body)
	}
	var v map[string]any
	if err := json.Unmarshal(body, &v); err != nil {
		t.Fatalf("debug body not JSON: %v", err)
	}
	for _, key := range []string{"diag", "scheduler", "cache", "engine", "models"} {
		if _, ok := v[key]; !ok {
			t.Fatalf("debug summary missing %q: %s", key, body)
		}
	}
}
