package ipc

import (
	"encoding/json"
	"fmt"
)

// MsgType is a message type code from the frame header. The codes
// are part of the contract and never renumber; they are grouped by
// family: 1-9 handshake and control, 10-19 model lifecycle, 20-29
// sessions, 30-39 tokenizer, 40-49 generation, 50-59 embeddings and
// shared memory, 60-69 state, 70-79 health, 80 shutdown.
type MsgType uint16

// The Section 4 message set.
const (
	// MsgHello is the first exchange, client side.
	MsgHello MsgType = 1
	// MsgHelloAck is the first exchange, server side.
	MsgHelloAck MsgType = 2
	// MsgRefused carries a named refusal, either direction.
	MsgRefused MsgType = 3
	// MsgAck is the generic confirmation for operations with no
	// result payload.
	MsgAck MsgType = 4
	// MsgLoadModel loads a model file.
	MsgLoadModel MsgType = 10
	// MsgLoadResult carries the load outcome.
	MsgLoadResult MsgType = 11
	// MsgUnloadModel releases a loaded model.
	MsgUnloadModel MsgType = 12
	// MsgCreateSession opens a generation session.
	MsgCreateSession MsgType = 20
	// MsgSessionCreated carries the new session handle.
	MsgSessionCreated MsgType = 21
	// MsgDestroySession closes a session.
	MsgDestroySession MsgType = 22
	// MsgTokenize sends text to the engine's tokenizer.
	MsgTokenize MsgType = 30
	// MsgTokenizeResult carries token ids.
	MsgTokenizeResult MsgType = 31
	// MsgDetokenize sends ids to the engine's tokenizer.
	MsgDetokenize MsgType = 32
	// MsgDetokenizeResult carries decoded text.
	MsgDetokenizeResult MsgType = 33
	// MsgGenerate runs generation on a session.
	MsgGenerate MsgType = 40
	// MsgTokenEvent is the ring wakeup frame for one streamed token.
	MsgTokenEvent MsgType = 41
	// MsgAbort stops a generation at the next token boundary.
	MsgAbort MsgType = 42
	// MsgEmbed embeds texts.
	MsgEmbed MsgType = 50
	// MsgEmbedResult carries the shared-memory vector handle.
	MsgEmbedResult MsgType = 51
	// MsgReleaseShm releases a shared-memory segment to its owner.
	MsgReleaseShm MsgType = 52
	// MsgSaveState persists session state.
	MsgSaveState MsgType = 60
	// MsgLoadState restores session state.
	MsgLoadState MsgType = 61
	// MsgStateResult carries a state operation outcome.
	MsgStateResult MsgType = 62
	// MsgHealth is the liveness check.
	MsgHealth MsgType = 70
	// MsgDiagnostics carries the diagnostic dump.
	MsgDiagnostics MsgType = 71
	// MsgShutdown asks the engine to drain and exit.
	MsgShutdown MsgType = 80
)

// Valid reports whether the code is a message type this build knows.
func (t MsgType) Valid() bool {
	switch t {
	case MsgHello, MsgHelloAck, MsgRefused, MsgAck,
		MsgLoadModel, MsgLoadResult, MsgUnloadModel,
		MsgCreateSession, MsgSessionCreated, MsgDestroySession,
		MsgTokenize, MsgTokenizeResult, MsgDetokenize, MsgDetokenizeResult,
		MsgGenerate, MsgTokenEvent, MsgAbort,
		MsgEmbed, MsgEmbedResult, MsgReleaseShm,
		MsgSaveState, MsgLoadState, MsgStateResult,
		MsgHealth, MsgDiagnostics, MsgShutdown:
		return true
	default:
		return false
	}
}

// MarshalPayload encodes a message payload in the seam's one agreed
// encoding (JSON), exactly as the Rust side encodes it.
func MarshalPayload(v any) ([]byte, error) {
	payload, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrMalformedPayload, err)
	}
	return payload, nil
}

// UnmarshalPayload decodes a frame payload into a message struct.
func UnmarshalPayload(payload []byte, v any) error {
	if err := json.Unmarshal(payload, v); err != nil {
		return fmt.Errorf("%w: %w", ErrMalformedPayload, err)
	}
	return nil
}

// Hello is the first frame on the socket.
type Hello struct {
	// ProtocolVersion is the sender's protocol version.
	ProtocolVersion uint16 `json:"protocol_version"`
	// PeerName is a human-readable peer name, for diagnostics.
	PeerName string `json:"peer_name"`
	// Capabilities lists capability strings the sender supports.
	Capabilities []string `json:"capabilities"`
}

// HardwareSummary is the engine's hardware report summary, carried
// by HelloAck.
type HardwareSummary struct {
	// Backend in use ("cpu", "vulkan", ...).
	Backend string `json:"backend"`
	// DeviceName as the backend reports it.
	DeviceName string `json:"device_name"`
	// VRAMBytes is usable device memory (0 for the CPU backend).
	VRAMBytes uint64 `json:"vram_bytes"`
}

// HelloAck is the server's half of the first exchange.
type HelloAck struct {
	// ProtocolVersion is the server's protocol version.
	ProtocolVersion uint16 `json:"protocol_version"`
	// PeerName is a human-readable peer name.
	PeerName string `json:"peer_name"`
	// Capabilities lists capability strings the server supports.
	Capabilities []string `json:"capabilities"`
	// Hardware is the engine's hardware report summary.
	Hardware HardwareSummary `json:"hardware"`
}

// Refusal is a named refusal. Code is one of the stable diagnostic
// code strings shared by both sides.
type Refusal struct {
	// Code is the stable diagnostic code naming the refusal.
	Code string `json:"code"`
	// Message is human-readable detail.
	Message string `json:"message"`
	// ProtocolVersion is the refusing side's protocol version, set
	// on version refusals; nil otherwise.
	ProtocolVersion *uint16 `json:"protocol_version,omitempty"`
}

// Ack is the generic confirmation for operations with no result
// payload (UnloadModel, DestroySession, Abort, Shutdown,
// ReleaseShm).
type Ack struct {
	// OK reports whether the operation completed.
	OK bool `json:"ok"`
	// Detail is optional human-readable detail.
	Detail string `json:"detail"`
}

// LoadModel loads a model file into the engine.
type LoadModel struct {
	// Path is confined to configured model directories.
	Path string `json:"path"`
	// ExpectedArchitecture is the architecture the platform expects
	// the file to declare; nil means no expectation.
	ExpectedArchitecture *string `json:"expected_architecture,omitempty"`
}

// CoverageReport is the loader's coverage report for one file.
type CoverageReport struct {
	// TensorsExpected counts tensors the arch module required.
	TensorsExpected uint64 `json:"tensors_expected"`
	// TensorsLoaded counts tensors found readable and well-shaped.
	TensorsLoaded uint64 `json:"tensors_loaded"`
	// Missing names required tensors absent from the file.
	Missing []string `json:"missing"`
	// Unreadable names tensors present but unreadable or misshapen.
	Unreadable []string `json:"unreadable"`
}

// LoadError is a named load failure: the model does not serve.
type LoadError struct {
	// Code is the stable diagnostic code.
	Code string `json:"code"`
	// Message is human-readable detail.
	Message string `json:"message"`
	// Missing names the tensors the failure is about, when it is a
	// coverage failure.
	Missing []string `json:"missing"`
}

// ModelInfo carries the static facts about a loaded model that the
// IPC LoadResult returns. Its canonical contract home is pkg/types
// (TODO T-1.1); the IPC-facing copy here exists because LoadResult
// carries it on the wire.
type ModelInfo struct {
	// Version is the contract shape version.
	Version uint32 `json:"version"`
	// ID is the model identifier as the platform names it.
	ID string `json:"id"`
	// Architecture is the file's general.architecture string.
	Architecture string `json:"architecture"`
	// ContextSize is the loaded context in tokens.
	ContextSize uint64 `json:"context_size"`
	// ParameterCount comes from the file metadata.
	ParameterCount uint64 `json:"parameter_count"`
	// Quantization is the quantization label, e.g. "Q4_K_M".
	Quantization string `json:"quantization"`
}

// LoadResult is the outcome of a LoadModel. Status is "loaded" or
// "refused"; the matching half of the fields is populated.
type LoadResult struct {
	// Status is "loaded" or "refused".
	Status string `json:"status"`
	// ModelHandle is set when Status is "loaded".
	ModelHandle *uint64 `json:"model_handle,omitempty"`
	// Model is set when Status is "loaded".
	Model *ModelInfo `json:"model,omitempty"`
	// Coverage is set when Status is "loaded".
	Coverage *CoverageReport `json:"coverage,omitempty"`
	// Error is set when Status is "refused".
	Error *LoadError `json:"error,omitempty"`
}

// UnloadModel releases a loaded model.
type UnloadModel struct {
	// ModelHandle is the handle from LoadResult.
	ModelHandle uint64 `json:"model_handle"`
}

// KvPolicy is a session's KV policy: "paged", "ring", or "hybrid".
type KvPolicy string

// The KV policies of the seam.
const (
	// KvPaged is the paged cache for global layers.
	KvPaged KvPolicy = "paged"
	// KvRing is fixed ring buffers for sliding-window layers.
	KvRing KvPolicy = "ring"
	// KvHybrid is the hybrid schedule: paged global, ring window.
	KvHybrid KvPolicy = "hybrid"
)

// CreateSession opens a generation session on a loaded model.
type CreateSession struct {
	// ModelHandle is the model to serve the session.
	ModelHandle uint64 `json:"model_handle"`
	// ContextSize is the session context in tokens.
	ContextSize uint64 `json:"context_size"`
	// KVPolicy is the session's KV policy.
	KVPolicy KvPolicy `json:"kv_policy"`
}

// SessionCreated is the outcome of a CreateSession.
type SessionCreated struct {
	// SessionHandle is the engine-side session handle.
	SessionHandle uint64 `json:"session_handle"`
}

// DestroySession closes a session and releases its KV state.
type DestroySession struct {
	// SessionHandle is the handle from SessionCreated.
	SessionHandle uint64 `json:"session_handle"`
}

// Tokenize sends text to the engine's tokenizer; the platform never
// guesses token ids (ARCHITECTURE.md Section 4).
type Tokenize struct {
	// SessionHandle optionally binds the request to a session's
	// model tokenizer.
	SessionHandle *uint64 `json:"session_handle,omitempty"`
	// Text is the text to encode.
	Text string `json:"text"`
}

// TokenizeResult carries tokenizer output.
type TokenizeResult struct {
	// TokenIDs are the encoded ids.
	TokenIDs []uint32 `json:"token_ids"`
}

// Detokenize sends ids to the engine's tokenizer.
type Detokenize struct {
	// TokenIDs are the ids to decode.
	TokenIDs []uint32 `json:"token_ids"`
}

// DetokenizeResult carries decoded text.
type DetokenizeResult struct {
	// Text is the decoded output.
	Text string `json:"text"`
}

// ShmRef references bytes in a shared-memory segment: the only way
// bulk data crosses the seam. Ownership is explicit; the receiver
// releases the segment with ReleaseShm when done.
type ShmRef struct {
	// Segment is the segment name agreed at setup.
	Segment string `json:"segment"`
	// Offset is the byte offset of the data inside the segment.
	Offset uint64 `json:"offset"`
	// Length is the byte length of the data.
	Length uint64 `json:"length"`
}

// SamplingParams is the sampling block of a Generate message, built
// by pkg/sampling from the request's ModelParams.
type SamplingParams struct {
	// Temperature is the sampling temperature; 0 is greedy.
	Temperature float64 `json:"temperature"`
	// TopP is the nucleus sampling cutoff.
	TopP float64 `json:"top_p"`
	// TopK is the top-k cutoff; 0 disables.
	TopK uint32 `json:"top_k"`
	// MaxTokens caps generation; 0 means the request sets no cap.
	MaxTokens uint32 `json:"max_tokens"`
	// RepeatPenalty is the repetition penalty; 1.0 disables.
	RepeatPenalty float64 `json:"repeat_penalty"`
	// Seed is an optional deterministic seed.
	Seed *uint64 `json:"seed,omitempty"`
}

// GrammarSpec is the grammar block of a Generate message. Kind is
// "gbnf", "json", or "json_schema".
type GrammarSpec struct {
	// Kind is which grammar language Source is written in.
	Kind string `json:"kind"`
	// Source is the grammar text or JSON schema document.
	Source string `json:"source"`
}

// StopRules are the stop rules of a Generate message.
type StopRules struct {
	// Sequences stop generation when produced.
	Sequences []string `json:"sequences"`
	// TokenIDs stop generation when produced.
	TokenIDs []uint32 `json:"token_ids"`
}

// Generate runs generation on a session. The prompt arrives as
// token ids or as a shared-memory reference, never as raw tensors
// on the socket.
type Generate struct {
	// SessionHandle is the session to generate on.
	SessionHandle uint64 `json:"session_handle"`
	// Sampling is the sampling block.
	Sampling SamplingParams `json:"sampling"`
	// Stop holds the stop rules.
	Stop StopRules `json:"stop"`
	// TokenIDs are the prompt ids, inline for short prompts.
	TokenIDs []uint32 `json:"token_ids,omitempty"`
	// PromptRef is the prompt ids by shared-memory reference.
	PromptRef *ShmRef `json:"prompt_ref,omitempty"`
	// Grammar is an optional grammar constraint.
	Grammar *GrammarSpec `json:"grammar,omitempty"`
}

// TokenEvent is one generated token. Events flow through the
// per-generation shared-memory ring; the socket frame is the wakeup
// and carries the ring sequence it announces.
type TokenEvent struct {
	// TokenID is the generated token id.
	TokenID uint32 `json:"token_id"`
	// RingSeq is the monotonic sequence inside the generation's ring.
	RingSeq uint64 `json:"ring_seq"`
	// Finished is true on the final event of the generation.
	Finished bool `json:"finished"`
	// Logprob is the token's log probability, when reported.
	Logprob *float32 `json:"logprob,omitempty"`
	// FinishReason is set on the final event.
	FinishReason *string `json:"finish_reason,omitempty"`
}

// Abort stops the generation named by TargetRequestID at the next
// token boundary.
type Abort struct {
	// TargetRequestID is the request id of the Generate to stop.
	TargetRequestID uint64 `json:"target_request_id"`
}

// Embed embeds a batch of texts.
type Embed struct {
	// Texts are the texts to embed.
	Texts []string `json:"texts"`
}

// EmbedResult carries the shared-memory handle to the vectors:
// Count rows of Dimension f32 values.
type EmbedResult struct {
	// Vectors is the handle to the vector block.
	Vectors ShmRef `json:"vectors"`
	// Dimension is the components per vector.
	Dimension uint64 `json:"dimension"`
	// Count is the number of vectors.
	Count uint64 `json:"count"`
}

// ReleaseShm releases a shared-memory segment back to its owner.
type ReleaseShm struct {
	// Segment is the segment name from the ShmRef being released.
	Segment string `json:"segment"`
}

// SaveState persists a session's state. Path is confined to the
// configured state directory and validated on the platform side
// before this message is sent.
type SaveState struct {
	// SessionHandle is the session whose state to save.
	SessionHandle uint64 `json:"session_handle"`
	// Path is the destination path inside the state directory.
	Path string `json:"path"`
}

// LoadState restores a session's state from a previous SaveState.
type LoadState struct {
	// SessionHandle is the session to restore into.
	SessionHandle uint64 `json:"session_handle"`
	// Path is the source path inside the state directory.
	Path string `json:"path"`
}

// StateResult is the outcome of a SaveState or LoadState. Status is
// "done" or "refused".
type StateResult struct {
	// Status is "done" or "refused".
	Status string `json:"status"`
	// Path is set when Status is "done".
	Path *string `json:"path,omitempty"`
	// Error is set when Status is "refused".
	Error *LoadError `json:"error,omitempty"`
}

// Health is the liveness check; its payload is empty.
type Health struct{}

// DiagEntry is one line of the diagnostic dump.
type DiagEntry struct {
	// Code is the stable diagnostic code.
	Code string `json:"code"`
	// Message describes the latest occurrence.
	Message string `json:"message"`
	// Count is occurrences since engine start.
	Count uint64 `json:"count"`
}

// Diagnostics replies to Health: liveness plus the dump of what the
// engine refused, fell back on, or decided, with codes.
type Diagnostics struct {
	// Healthy reports whether the engine can serve.
	Healthy bool `json:"healthy"`
	// EngineVersion is the engine build version string.
	EngineVersion string `json:"engine_version"`
	// Entries are the diagnostic entries.
	Entries []DiagEntry `json:"entries"`
}

// Shutdown asks the engine to drain in-flight work and exit.
type Shutdown struct {
	// DrainTimeoutMs bounds the drain before the engine exits anyway.
	DrainTimeoutMs uint64 `json:"drain_timeout_ms"`
}
