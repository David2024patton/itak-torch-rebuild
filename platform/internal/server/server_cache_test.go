package server_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/David2024patton/itak-torch-rebuild/platform/internal/engine"
)

// --- cache key distinctness (critic panel Category E, HIGH) -----------

func cacheEntryCount(t *testing.T, ts *httptest.Server) int {
	t.Helper()
	_, _, body := do(t, ts, http.MethodGet, "/debug", "", testKey)
	var v struct {
		Cache struct {
			Entries int `json:"entries"`
		} `json:"cache"`
	}
	if err := json.Unmarshal(body, &v); err != nil {
		t.Fatalf("debug body is not JSON: %v (%s)", err, body)
	}
	return v.Cache.Entries
}

// TestChatCacheDistinctRequestsGetDistinctEntries posts requests
// that differ from a base request in exactly one field each and
// requires a cache miss, one more engine call, and one more cache
// entry for every one of them. The panel's mutation — a cache key
// hashing only the model name — turns the first same-model variant
// into a hit and fails this test; before it existed, that mutation
// survived the full suite. Every variant here stays cacheable
// (temperature absent, zero, or seeded), so a miss can only mean
// the key told the requests apart.
func TestChatCacheDistinctRequestsGetDistinctEntries(t *testing.T) {
	fake := &fakeEngine{}
	ts := newServer(t, fake)

	base := `{"model":"glimmer","messages":[{"role":"user","content":"Say hello"}]}`
	variants := map[string]string{ //nolint:gosec // request bodies, not credentials: gosec G101 pattern-matches the "max_tokens" field name inside them
		"message content":    `{"model":"glimmer","messages":[{"role":"user","content":"Say goodbye"}]}`,
		"message role":       `{"model":"glimmer","messages":[{"role":"system","content":"Say hello"}]}`,
		"message appended":   `{"model":"glimmer","messages":[{"role":"user","content":"Say hello"},{"role":"user","content":"again"}]}`,
		"max_tokens":         `{"model":"glimmer","max_tokens":8,"messages":[{"role":"user","content":"Say hello"}]}`,
		"temperature zero":   `{"model":"glimmer","temperature":0,"messages":[{"role":"user","content":"Say hello"}]}`,
		"seed":               `{"model":"glimmer","seed":7,"messages":[{"role":"user","content":"Say hello"}]}`,
		"seeded temperature": `{"model":"glimmer","temperature":0.7,"seed":7,"messages":[{"role":"user","content":"Say hello"}]}`,
		"top_p":              `{"model":"glimmer","top_p":0.5,"messages":[{"role":"user","content":"Say hello"}]}`,
		"top_k":              `{"model":"glimmer","top_k":40,"messages":[{"role":"user","content":"Say hello"}]}`,
		"stop":               `{"model":"glimmer","stop":["END"],"messages":[{"role":"user","content":"Say hello"}]}`,
		"model":              `{"model":"other-model","messages":[{"role":"user","content":"Say hello"}]}`,
	}

	post := func(body string) http.Header {
		t.Helper()
		code, hdr, respBody := do(t, ts, http.MethodPost, "/v1/chat/completions", body, testKey)
		if code != http.StatusOK {
			t.Fatalf("chat status = %d (%s)", code, respBody)
		}
		return hdr
	}

	if hdr := post(base); hdr.Get("X-Torch-Cache") != "miss" {
		t.Fatalf("base request X-Torch-Cache = %q, want miss", hdr.Get("X-Torch-Cache"))
	}
	wantCalls, wantEntries := int64(1), 1
	for name, body := range variants {
		if hdr := post(body); hdr.Get("X-Torch-Cache") != "miss" {
			t.Fatalf("%s: X-Torch-Cache = %q, want miss (a hit means its key collides with an earlier request)", name, hdr.Get("X-Torch-Cache"))
		}
		wantCalls++
		wantEntries++
		if n := fake.generateCalls.Load(); n != wantCalls {
			t.Fatalf("%s: engine Generate called %d times, want %d (a repeat means a shared cache entry)", name, n, wantCalls)
		}
		if n := cacheEntryCount(t, ts); n != wantEntries {
			t.Fatalf("%s: cache holds %d entries, want %d distinct entries", name, n, wantEntries)
		}
	}

	// And the base request still hits its own entry afterwards.
	if hdr := post(base); hdr.Get("X-Torch-Cache") != "hit" {
		t.Fatalf("base request repeated: X-Torch-Cache = %q, want hit", hdr.Get("X-Torch-Cache"))
	}
	if n := fake.generateCalls.Load(); n != wantCalls {
		t.Fatalf("engine Generate called %d times after the base repeat, want %d", n, wantCalls)
	}
}

// --- sampled generations bypass the cache (Category E, MEDIUM) --------

// sampledEngine returns a different completion on every call, the
// way a sampling engine does, so a replayed answer is visible in
// the response body itself, not just in call counts.
type sampledEngine struct {
	calls atomic.Int64
}

func (e *sampledEngine) Generate(_ context.Context, _ engine.GenerateRequest) (<-chan engine.TokenEvent, error) {
	n := e.calls.Add(1)
	ch := make(chan engine.TokenEvent, 2)
	ch <- engine.TokenEvent{Text: fmt.Sprintf("take %d", n), TokenID: 1}
	ch <- engine.TokenEvent{Done: true, FinishReason: "stop"}
	close(ch)
	return ch, nil
}

func (e *sampledEngine) Embed(_ context.Context, req engine.EmbedRequest) (engine.EmbedResponse, error) {
	vecs := make([][]float32, len(req.Input))
	for i := range vecs {
		vecs[i] = []float32{0.1, 0.2, 0.3}
	}
	return engine.EmbedResponse{Model: req.Model, Vectors: vecs}, nil
}

func (e *sampledEngine) Health(context.Context) (engine.HealthStatus, error) {
	return engine.HealthStatus{OK: true, Detail: "sampled engine"}, nil
}

func completionContent(t *testing.T, body []byte) string {
	t.Helper()
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
	return v.Choices[0].Message.Content
}

// TestChatUnseededSampledRequestsBypassCache: an unseeded
// temperature > 0 request asks for variation. Serving the second
// identical caller the first caller's sample byte-identical (the
// old behavior) freezes exactly what the temperature was set to
// vary, so these requests must skip the cache on both sides: no
// read, no write, the engine called every time.
func TestChatUnseededSampledRequestsBypassCache(t *testing.T) {
	eng := &sampledEngine{}
	ts := newServer(t, eng)
	body := `{"model":"glimmer","temperature":0.7,"messages":[{"role":"user","content":"roll the dice"}]}`

	contents := map[string]bool{}
	for i := 1; i <= 2; i++ {
		code, hdr, respBody := do(t, ts, http.MethodPost, "/v1/chat/completions", body, testKey)
		if code != http.StatusOK {
			t.Fatalf("request %d status = %d (%s)", i, code, respBody)
		}
		if got := hdr.Get("X-Torch-Cache"); got != "skip" {
			t.Fatalf("request %d X-Torch-Cache = %q, want skip (unseeded sampled requests never touch the cache)", i, got)
		}
		contents[completionContent(t, respBody)] = true
	}
	if len(contents) != 2 {
		t.Fatalf("two unseeded sampled requests produced %d distinct completions, want 2 (the second was a replay): %v", len(contents), contents)
	}
	if n := eng.calls.Load(); n != 2 {
		t.Fatalf("engine Generate called %d times, want 2", n)
	}
	if n := cacheEntryCount(t, ts); n != 0 {
		t.Fatalf("cache holds %d entries after unseeded sampled requests, want 0 (nothing was written)", n)
	}
}

// TestChatSeededSampledRequestsStillCache: a seed pins the sample,
// so a seeded temperature > 0 request is deterministic and keeps
// the cache — the bypass above must not widen to it.
func TestChatSeededSampledRequestsStillCache(t *testing.T) {
	eng := &sampledEngine{}
	ts := newServer(t, eng)
	body := `{"model":"glimmer","temperature":0.7,"seed":42,"messages":[{"role":"user","content":"roll the dice"}]}`

	_, hdr, first := do(t, ts, http.MethodPost, "/v1/chat/completions", body, testKey)
	if got := hdr.Get("X-Torch-Cache"); got != "miss" {
		t.Fatalf("first request X-Torch-Cache = %q, want miss", got)
	}
	_, hdr, second := do(t, ts, http.MethodPost, "/v1/chat/completions", body, testKey)
	if got := hdr.Get("X-Torch-Cache"); got != "hit" {
		t.Fatalf("second request X-Torch-Cache = %q, want hit (a seeded request is deterministic)", got)
	}
	if c1, c2 := completionContent(t, first), completionContent(t, second); c1 != c2 {
		t.Fatalf("seeded repeat returned %q after %q; a cache hit must replay the stored body", c2, c1)
	}
	if n := eng.calls.Load(); n != 1 {
		t.Fatalf("engine Generate called %d times, want 1", n)
	}
}

// TestChatZeroTemperatureRequestsStillCache: temperature 0 is
// greedy decoding, deterministic with or without a seed, so it
// keeps the cache as well.
func TestChatZeroTemperatureRequestsStillCache(t *testing.T) {
	eng := &sampledEngine{}
	ts := newServer(t, eng)
	body := `{"model":"glimmer","temperature":0,"messages":[{"role":"user","content":"be exact"}]}`

	_, hdr, _ := do(t, ts, http.MethodPost, "/v1/chat/completions", body, testKey)
	if got := hdr.Get("X-Torch-Cache"); got != "miss" {
		t.Fatalf("first request X-Torch-Cache = %q, want miss", got)
	}
	_, hdr, _ = do(t, ts, http.MethodPost, "/v1/chat/completions", body, testKey)
	if got := hdr.Get("X-Torch-Cache"); got != "hit" {
		t.Fatalf("second request X-Torch-Cache = %q, want hit", got)
	}
	if n := eng.calls.Load(); n != 1 {
		t.Fatalf("engine Generate called %d times, want 1", n)
	}
}

// --- template fetch failures (Category E, MEDIUM) ---------------------

// flakyTemplateEngine fails the template fetch with a transient
// error (not the named no-template outcome) until told to stop,
// then serves the ChatML template.
type flakyTemplateEngine struct {
	*fakeEngine
	fail atomic.Bool
}

func (e *flakyTemplateEngine) ChatTemplate(context.Context) (engine.ChatTemplate, error) {
	if e.fail.Load() {
		return engine.ChatTemplate{}, errors.New("template store unreachable")
	}
	return engine.ChatTemplate{Template: chatMLTemplate}, nil
}

// TestChatTemplateFetchFailureIsNamedNotFlattened: a transient
// template-fetch error used to return the role-flattened prompt
// with a nil error, and the flattened generation was then cached
// as if correct. It must instead take the handler's named 502
// path, generate nothing, cache nothing — and the next request
// must retry the fetch and succeed.
func TestChatTemplateFetchFailureIsNamedNotFlattened(t *testing.T) {
	fake := &fakeEngine{}
	eng := &flakyTemplateEngine{fakeEngine: fake}
	eng.fail.Store(true)
	ts := newServerWithConfig(t, eng, nil)

	code, _, body := do(t, ts, http.MethodPost, "/v1/chat/completions", chatBody, testKey)
	if code != http.StatusBadGateway || errorCode(t, body) != "chat_template_error" {
		t.Fatalf("status = %d, body = %s; want 502 chat_template_error, never a flattened 200", code, body)
	}
	if n := fake.generateCalls.Load(); n != 0 {
		t.Fatalf("engine Generate called %d times on a failed template fetch, want 0", n)
	}
	if n := cacheEntryCount(t, ts); n != 0 {
		t.Fatalf("cache holds %d entries after a failed template fetch, want 0", n)
	}

	eng.fail.Store(false)
	code, _, body = do(t, ts, http.MethodPost, "/v1/chat/completions", chatBody, testKey)
	if code != http.StatusOK {
		t.Fatalf("retry status = %d (%s), want 200 once the fetch recovers", code, body)
	}
	want := "<|im_start|>user\nSay hello<|im_end|>\n<|im_start|>assistant\n"
	if got := lastPrompt(t, fake); got != want {
		t.Fatalf("engine prompt after recovery = %q, want the templated %q", got, want)
	}
	// The recovered generation is the one that gets cached.
	_, hdr, _ := do(t, ts, http.MethodPost, "/v1/chat/completions", chatBody, testKey)
	if got := hdr.Get("X-Torch-Cache"); got != "hit" {
		t.Fatalf("repeat after recovery X-Torch-Cache = %q, want hit", got)
	}
	if n := fake.generateCalls.Load(); n != 1 {
		t.Fatalf("engine Generate called %d times, want 1", n)
	}
}

// blockingTemplateEngine parks inside the template fetch until
// released, so a test can hold one fetch open and watch what a
// concurrent request does.
type blockingTemplateEngine struct {
	*fakeEngine
	entered chan struct{}
	release chan struct{}
}

func (e *blockingTemplateEngine) ChatTemplate(ctx context.Context) (engine.ChatTemplate, error) {
	e.entered <- struct{}{}
	select {
	case <-e.release:
		return engine.ChatTemplate{Template: chatMLTemplate}, nil
	case <-ctx.Done():
		return engine.ChatTemplate{}, ctx.Err()
	}
}

// TestChatTemplateFetchDoesNotSerializeRequests: the template
// mutex used to be held across the network fetch, so one slow
// fetch serialized every concurrent chat request behind it. With
// the first fetch parked, a second request must still reach its
// own fetch.
func TestChatTemplateFetchDoesNotSerializeRequests(t *testing.T) {
	fake := &fakeEngine{}
	eng := &blockingTemplateEngine{
		fakeEngine: fake,
		entered:    make(chan struct{}, 2),
		release:    make(chan struct{}),
	}
	ts := newServerWithConfig(t, eng, nil)
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(eng.release) }) }
	defer release()

	type result struct {
		code int
		err  error
	}
	results := make(chan result, 2)
	post := func() {
		req, err := http.NewRequest(http.MethodPost, ts.URL+"/v1/chat/completions", strings.NewReader(chatBody))
		if err != nil {
			results <- result{0, err}
			return
		}
		req.Header.Set("Authorization", "Bearer "+testKey)
		req.Header.Set("Content-Type", "application/json")
		resp, err := ts.Client().Do(req)
		if err != nil {
			results <- result{0, err}
			return
		}
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
		results <- result{resp.StatusCode, nil}
	}

	go post()
	<-eng.entered // the first request's fetch is in flight and parked
	go post()
	select {
	case <-eng.entered:
		// The second request reached its own fetch while the
		// first was still parked: the mutex is not held across it.
	case <-time.After(3 * time.Second):
		t.Fatal("second chat request never reached the template fetch while the first was parked: the template mutex is held across the network fetch")
	}
	release()
	for i := 0; i < 2; i++ {
		res := <-results
		if res.err != nil || res.code != http.StatusOK {
			t.Fatalf("concurrent request = %d, err = %v; want 200", res.code, res.err)
		}
	}
}
