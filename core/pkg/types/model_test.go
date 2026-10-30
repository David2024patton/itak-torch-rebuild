package types

import (
	"encoding/json"
	"testing"
)

func TestModelBackendValues(t *testing.T) {
	cases := map[ModelBackend]string{
		BackendCPU:    "cpu",
		BackendCUDA:   "cuda",
		BackendVulkan: "vulkan",
		BackendMetal:  "metal",
		BackendCloud:  "cloud",
	}
	for backend, want := range cases {
		if string(backend) != want {
			t.Errorf("ModelBackend = %q, want %q", string(backend), want)
		}
		// The original adapter converts its backend string with
		// types.ModelBackend(a.backend); the conversion must hold.
		if ModelBackend(want) != backend {
			t.Errorf("ModelBackend(%q) = %q, want %q", want, ModelBackend(want), backend)
		}
	}
}

func TestModelParamsJSONRoundTrip(t *testing.T) {
	params := ModelParams{
		Version:     Version,
		ID:          "ollama/qwen3:8b",
		Backend:     BackendVulkan,
		ContextSize: 32768,
		GPULayers:   -1,
		Temperature: 0.7,
		MaxTokens:   512,
	}
	roundTrip(t, params)
}

func TestModelParamsEmptyLiteral(t *testing.T) {
	// The original call site passes a zero ModelParams
	// (pkg/torch/core_adapter_test.go LoadModel calls). A zero value
	// carries version 0, marking data from before versioning landed.
	var params ModelParams
	if params != (ModelParams{}) {
		t.Errorf("zero ModelParams = %+v, want zero value", params)
	}

	data, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal zero ModelParams: %v", err)
	}
	var got ModelParams
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal zero ModelParams: %v", err)
	}
	if got != params {
		t.Errorf("zero ModelParams round trip = %+v, want %+v", got, params)
	}
}

func TestTokenUsageJSONRoundTrip(t *testing.T) {
	usage := TokenUsage{
		Version:          Version,
		PromptTokens:     100,
		CompletionTokens: 250,
		TotalTokens:      350,
	}
	roundTrip(t, usage)
}

func TestModelInfoJSONRoundTrip(t *testing.T) {
	// Populated the way pkg/torch/core_adapter.go's torchModelToCore
	// builds it: ID and Loaded in the literal, the detected fields
	// assigned afterwards.
	info := ModelInfo{
		Version: Version,
		ID:      ModelID("qwen3-8b-instruct-q4_k_m.gguf"),
		Loaded:  false,
	}
	info.Params = 8_000_000_000
	info.Quantized = "Q4_K_M"
	info.Family = "qwen"
	info.Backend = BackendVulkan
	info.ContextSize = 32768
	info.SizeBytes = 4_919_000_000
	roundTrip(t, info)
}

func TestModelInfoSingleModelLiteral(t *testing.T) {
	// The single-model literal from the original adapter's ListModels.
	info := ModelInfo{
		Version: Version,
		ID:      ModelID("muse-glimmer-30b"),
		Backend: ModelBackend("vulkan"),
		Loaded:  true,
	}
	roundTrip(t, info)
}
