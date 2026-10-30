package types

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// roundTrip marshals v to JSON, unmarshals it into a new T, and fails
// unless the result is deeply equal to v. It also fails unless the
// encoded form carries the current schema version.
func roundTrip[T any](t *testing.T, v T) {
	t.Helper()

	data, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal %T: %v", v, err)
	}
	if !strings.Contains(string(data), `"version":1`) {
		t.Errorf("marshaled %T lacks the version field: %s", v, data)
	}

	var got T
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal %T: %v", v, err)
	}
	if !reflect.DeepEqual(got, v) {
		t.Errorf("round trip mismatch:\n got: %+v\nwant: %+v", got, v)
	}
}

func TestVersionConstant(t *testing.T) {
	if Version != 1 {
		t.Errorf("Version = %d, want 1", Version)
	}
}

func TestRoleValues(t *testing.T) {
	cases := map[Role]string{
		RoleSystem:    "system",
		RoleUser:      "user",
		RoleAssistant: "assistant",
		RoleTool:      "tool",
	}
	for role, want := range cases {
		if string(role) != want {
			t.Errorf("Role = %q, want %q", string(role), want)
		}
		// The original adapter converts Torch's string roles with
		// types.Role(m.Role); the conversion must round back.
		if Role(want) != role {
			t.Errorf("Role(%q) = %q, want %q", want, Role(want), role)
		}
	}
}

func TestChatMessageJSONRoundTrip(t *testing.T) {
	msg := ChatMessage{
		Version: Version,
		Role:    RoleAssistant,
		Content: "Calling the tool now.",
		Name:    "torch",
		ToolCalls: []ToolCall{{
			Version: Version,
			ID:      "call_1",
			Type:    "function",
			Function: FunctionCall{
				Version:   Version,
				Name:      "get_weather",
				Arguments: `{"city":"Louisville"}`,
			},
		}},
		ToolCallID: "call_0",
	}
	roundTrip(t, msg)
}

func TestInferenceRequestJSONRoundTrip(t *testing.T) {
	// Constructed in the exact shape of the original call site
	// (pkg/torch/core_adapter_test.go): the model is an untyped string
	// constant assigned to the ModelID field.
	req := InferenceRequest{
		Version: Version,
		Model:   "qwen3-8b",
		Messages: []ChatMessage{
			{Version: Version, Role: RoleUser, Content: "Hello"},
		},
		Temperature: 0.7,
		MaxTokens:   100,
		Stream:      true,
		Tools: []ToolDef{{
			Version: Version,
			Type:    "function",
			Function: ToolDefDetails{
				Version:     Version,
				Name:        "get_weather",
				Description: "Returns the weather for a city.",
				Parameters:  map[string]interface{}{"type": "object"},
			},
		}},
	}
	roundTrip(t, req)
}

func TestInferenceResponseJSONRoundTrip(t *testing.T) {
	// Built the way pkg/torch/core_adapter.go builds it: a literal for
	// the message, then Usage and Latency assigned as fields.
	resp := InferenceResponse{
		Version: Version,
		Model:   "qwen3-8b",
		Message: ChatMessage{
			Version: Version,
			Role:    RoleAssistant,
			Content: "Hello from the engine.",
		},
		Done: true,
	}
	resp.Usage = TokenUsage{
		Version:          Version,
		PromptTokens:     12,
		CompletionTokens: 7,
		TotalTokens:      19,
	}
	resp.Latency = LatencyStats{
		Version:      Version,
		PromptEvalMs: 31,
		GenerationMs: 88,
		TotalMs:      119,
		TokensPerSec: 59.5,
	}
	roundTrip(t, resp)
}

func TestLatencyStatsJSONRoundTrip(t *testing.T) {
	stats := LatencyStats{
		Version:      Version,
		PromptEvalMs: 10.5,
		GenerationMs: 20.25,
		TotalMs:      30.75,
		TokensPerSec: 42,
	}
	roundTrip(t, stats)
}

func TestToolCallJSONRoundTrip(t *testing.T) {
	call := ToolCall{
		Version: Version,
		ID:      "call_9",
		Type:    "function",
		Function: FunctionCall{
			Version:   Version,
			Name:      "lookup",
			Arguments: `{}`,
		},
	}
	roundTrip(t, call)
}
