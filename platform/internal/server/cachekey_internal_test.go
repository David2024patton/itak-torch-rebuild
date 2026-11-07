package server

import "testing"

// Direct pins on chatCacheKey, in-package because the key function
// is unexported. The critic panel's mutation (hash only the model
// name) survived the whole HTTP-level suite when the only cache
// test repeated one identical request; these tests exist so that
// any field dropping out of the key — a JSON-tag refactor is the
// realistic route — goes red here even if a handler-level test
// happens not to exercise that field.

func f64p(v float64) *float64 { return &v }
func u64p(v uint64) *uint64   { return &v }
func intp(v int) *int         { return &v }
func boolp(v bool) *bool      { return &v }

// baseKeyRequest builds a fresh request with every key input set,
// so each variant below has a field to move. Every call builds new
// slices and maps, so a variant may mutate what it is handed.
func baseKeyRequest() *chatRequest {
	return &chatRequest{
		Model: "glimmer",
		Messages: []chatMessage{
			{Role: "user", Content: "Say hello"},
			{Role: "assistant", ToolCalls: []chatToolCall{{
				ID:   "c1",
				Type: "function",
				Function: toolFunction{
					Name:      "get_weather",
					Arguments: map[string]any{"city": "Paris"},
				},
			}}},
			{Role: "tool", Content: "sunny", Name: "get_weather", ToolCallID: "c1"},
		},
		Tools: []chatTool{{
			Type: "function",
			Function: toolFunction{
				Name:        "get_weather",
				Description: "current weather",
				Parameters:  map[string]any{"type": "object"},
			},
		}},
		MaxTokens:        16,
		Temperature:      f64p(0),
		TopP:             f64p(0.9),
		Stop:             []string{"END"},
		Seed:             u64p(42),
		TopK:             intp(40),
		PresencePenalty:  f64p(0.5),
		FrequencyPenalty: f64p(-0.25),
	}
}

// TestChatCacheKeyFieldByFieldDistinct moves exactly one key input
// at a time and requires a different key every time — and requires
// all the moved keys to differ from each other, not just from the
// base, so two fields cannot silently alias onto one hash input.
func TestChatCacheKeyFieldByFieldDistinct(t *testing.T) {
	variants := map[string]func(*chatRequest){
		"model":                 func(r *chatRequest) { r.Model = "other-model" },
		"message content":       func(r *chatRequest) { r.Messages[0].Content = "Say goodbye" },
		"message role":          func(r *chatRequest) { r.Messages[0].Role = "system" },
		"message role sequence": func(r *chatRequest) { r.Messages[0], r.Messages[1] = r.Messages[1], r.Messages[0] },
		"message appended":      func(r *chatRequest) { r.Messages = append(r.Messages, chatMessage{Role: "user", Content: "again"}) },
		"message name":          func(r *chatRequest) { r.Messages[2].Name = "get_forecast" },
		"message tool_call_id":  func(r *chatRequest) { r.Messages[2].ToolCallID = "c2" },
		"message reasoning":     func(r *chatRequest) { r.Messages[0].ReasoningContent = "thinking" },
		"message recipient":     func(r *chatRequest) { r.Messages[0].Recipient = "functions.get_weather" },
		"message end_turn":      func(r *chatRequest) { r.Messages[1].EndTurn = boolp(false) },
		"tool call id":          func(r *chatRequest) { r.Messages[1].ToolCalls[0].ID = "c9" },
		"tool call type":        func(r *chatRequest) { r.Messages[1].ToolCalls[0].Type = "other" },
		"tool call name":        func(r *chatRequest) { r.Messages[1].ToolCalls[0].Function.Name = "get_forecast" },
		"tool call arguments":   func(r *chatRequest) { r.Messages[1].ToolCalls[0].Function.Arguments = map[string]any{"city": "Lyon"} },
		"tool type":             func(r *chatRequest) { r.Tools[0].Type = "other" },
		"tool name":             func(r *chatRequest) { r.Tools[0].Function.Name = "get_forecast" },
		"tool description":      func(r *chatRequest) { r.Tools[0].Function.Description = "forecast" },
		"tool parameters":       func(r *chatRequest) { r.Tools[0].Function.Parameters = map[string]any{"type": "array"} },
		"stream":                func(r *chatRequest) { r.Stream = true },
		"max_tokens":            func(r *chatRequest) { r.MaxTokens = 17 },
		"temperature value":     func(r *chatRequest) { r.Temperature = f64p(0.7) },
		"temperature absent":    func(r *chatRequest) { r.Temperature = nil },
		"top_p":                 func(r *chatRequest) { r.TopP = f64p(0.5) },
		"stop":                  func(r *chatRequest) { r.Stop = []string{"STOP"} },
		"seed value":            func(r *chatRequest) { r.Seed = u64p(43) },
		"seed absent":           func(r *chatRequest) { r.Seed = nil },
		"top_k":                 func(r *chatRequest) { r.TopK = intp(41) },
		"presence_penalty":      func(r *chatRequest) { r.PresencePenalty = f64p(0.6) },
		"frequency_penalty":     func(r *chatRequest) { r.FrequencyPenalty = f64p(-0.5) },
	}
	// The absent-vs-zero distinction rides on the "temperature
	// absent" arm: a present 0 marshals as 0, an absent field as
	// null, and the two must not share an entry.

	base := chatCacheKey(baseKeyRequest())
	seen := map[string]string{base: "base request"}
	for name, mutate := range variants {
		req := baseKeyRequest()
		mutate(req)
		key := chatCacheKey(req)
		if key == base {
			t.Errorf("%s: key equals the base key; the field is not covered by chatCacheKey", name)
		}
		if other, dup := seen[key]; dup {
			t.Errorf("%s: key collides with %s; two distinct requests share a cache entry", name, other)
		}
		seen[key] = name
	}
}

// TestChatCacheKeyStableForIdenticalRequests is the other half of
// the contract the HTTP cache test leans on: the same request
// marshals to the same key every time (map fields included — Go's
// json sorts map keys, and this pins that the key inherits it).
func TestChatCacheKeyStableForIdenticalRequests(t *testing.T) {
	a := chatCacheKey(baseKeyRequest())
	b := chatCacheKey(baseKeyRequest())
	if a != b {
		t.Fatalf("identical requests keyed differently: %s vs %s", a, b)
	}
}
