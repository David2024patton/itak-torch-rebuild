package types

// ModelID uniquely identifies a model across the ecosystem.
// Format: "provider/name:tag" (e.g., "ollama/qwen3:8b", "openai/gpt-4o").
type ModelID string

// ModelBackend identifies which compute backend to use.
type ModelBackend string

const (
	// BackendCPU runs inference on the CPU.
	BackendCPU ModelBackend = "cpu"

	// BackendCUDA runs inference on an NVIDIA GPU via CUDA.
	BackendCUDA ModelBackend = "cuda"

	// BackendVulkan runs inference on the GPU via Vulkan.
	BackendVulkan ModelBackend = "vulkan"

	// BackendMetal runs inference on Apple silicon via Metal.
	BackendMetal ModelBackend = "metal"

	// BackendCloud routes inference via Gateway to a cloud provider.
	BackendCloud ModelBackend = "cloud"
)

// ModelParams contains configuration for loading and running a model.
type ModelParams struct {
	// Version is the schema version of this data format.
	Version int `json:"version" yaml:"version"`

	// ID is the model identifier (e.g., "ollama/qwen3:8b").
	ID ModelID `json:"id" yaml:"id"`

	// Backend selects the compute backend (cpu, cuda, vulkan, etc.).
	Backend ModelBackend `json:"backend,omitempty" yaml:"backend,omitempty"`

	// ContextSize is the maximum number of tokens in the context window.
	ContextSize int `json:"context_size,omitempty" yaml:"context_size,omitempty"`

	// GPULayers is the number of layers to offload to GPU (-1 = all).
	GPULayers int `json:"gpu_layers,omitempty" yaml:"gpu_layers,omitempty"`

	// Temperature controls randomness in generation (0.0 = deterministic).
	Temperature float32 `json:"temperature,omitempty" yaml:"temperature,omitempty"`

	// MaxTokens caps the number of tokens to generate.
	MaxTokens int `json:"max_tokens,omitempty" yaml:"max_tokens,omitempty"`
}

// TokenUsage tracks token consumption for a single request.
// Every module that calls an LLM (Agent, Gateway, Torch) must
// return this alongside the response for cost tracking.
type TokenUsage struct {
	Version          int `json:"version"`
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ModelInfo describes a loaded model's capabilities.
type ModelInfo struct {
	Version     int          `json:"version"`
	ID          ModelID      `json:"id"`
	Backend     ModelBackend `json:"backend"`
	ContextSize int          `json:"context_size"`
	Params      int64        `json:"params"`     // Parameter count (e.g., 8_000_000_000)
	SizeBytes   int64        `json:"size_bytes"` // On-disk size
	Quantized   string       `json:"quantized"`  // Quantization level (e.g., "Q4_K_M", "FP16")
	Family      string       `json:"family"`     // Model family (e.g., "qwen3", "llama")
	Loaded      bool         `json:"loaded"`     // Whether currently in memory
}
