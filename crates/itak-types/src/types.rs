//! The shared contracts mirrored from the Go platform's `pkg/types`
//! (TODO T-1.1 documents the surface: `InferenceRequest`,
//! `InferenceResponse`, `ChatMessage`, `Role`, `TokenUsage`,
//! `LatencyStats`, `ModelInfo`, `ModelParams`). This module mirrors
//! that documented surface shape for shape; field-level parity locks
//! against the T-1.1 module when it lands.
//!
//! Per the module grid (ARCHITECTURE.md Section 3.1), every struct
//! here carries a `version` field: the shape version of the contract
//! itself, independent of the wire protocol version. Decoders in this
//! crate accept any shape version (forward compatibility); producers
//! always stamp [`SHAPE_VERSION`].

use crate::error::Error;
use crate::json::{
    field, field_array, field_bool, field_f64, field_opt_str, field_str, field_string_vec,
    field_u32, field_u64, string_vec, Value, Wire,
};

/// The contract shape version stamped into every mirrored struct.
pub const SHAPE_VERSION: u32 = 1;

/// The role of a chat message participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// System prompt.
    System,
    /// End user.
    User,
    /// The model.
    Assistant,
    /// A tool result.
    Tool,
}

impl Role {
    /// The wire string for this role.
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }

    /// Parse the wire string, naming the failure on unknown roles.
    pub fn from_wire(text: &str) -> Result<Role, Error> {
        match text {
            "system" => Ok(Role::System),
            "user" => Ok(Role::User),
            "assistant" => Ok(Role::Assistant),
            "tool" => Ok(Role::Tool),
            other => Err(Error::MalformedPayload(format!("unknown role '{other}'"))),
        }
    }
}

impl Wire for Role {
    fn encode(&self) -> Value {
        Value::Str(self.as_str().to_string())
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        value
            .as_str()
            .ok_or_else(|| Error::MalformedPayload("role is not a string".to_string()))
            .and_then(Role::from_wire)
    }
}

/// One chat message.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatMessage {
    /// Contract shape version.
    pub version: u32,
    /// Who sent the message.
    pub role: Role,
    /// Message text.
    pub content: String,
    /// Optional sender name (tool calls, named participants).
    pub name: Option<String>,
}

impl Default for ChatMessage {
    fn default() -> Self {
        ChatMessage {
            version: SHAPE_VERSION,
            role: Role::User,
            content: String::new(),
            name: None,
        }
    }
}

impl Wire for ChatMessage {
    fn encode(&self) -> Value {
        let mut pairs = vec![
            ("version", Value::UInt(u64::from(self.version))),
            ("role", self.role.encode()),
            ("content", Value::Str(self.content.clone())),
        ];
        if let Some(name) = &self.name {
            pairs.push(("name", Value::Str(name.clone())));
        }
        Value::object(pairs)
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(ChatMessage {
            version: field_u32(value, "version")?,
            role: Role::decode(field(value, "role")?)?,
            content: field_str(value, "content")?,
            name: field_opt_str(value, "name")?,
        })
    }
}

/// Token accounting for one completed request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenUsage {
    /// Contract shape version.
    pub version: u32,
    /// Tokens in the prompt.
    pub prompt_tokens: u64,
    /// Tokens generated.
    pub completion_tokens: u64,
    /// Prompt plus completion.
    pub total_tokens: u64,
}

impl Default for TokenUsage {
    fn default() -> Self {
        TokenUsage {
            version: SHAPE_VERSION,
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        }
    }
}

impl Wire for TokenUsage {
    fn encode(&self) -> Value {
        Value::object(vec![
            ("version", Value::UInt(u64::from(self.version))),
            ("prompt_tokens", Value::UInt(self.prompt_tokens)),
            ("completion_tokens", Value::UInt(self.completion_tokens)),
            ("total_tokens", Value::UInt(self.total_tokens)),
        ])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(TokenUsage {
            version: field_u32(value, "version")?,
            prompt_tokens: field_u64(value, "prompt_tokens")?,
            completion_tokens: field_u64(value, "completion_tokens")?,
            total_tokens: field_u64(value, "total_tokens")?,
        })
    }
}

/// Measured latency for one completed request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatencyStats {
    /// Contract shape version.
    pub version: u32,
    /// Milliseconds from request acceptance to the first token.
    pub time_to_first_token_ms: f64,
    /// Milliseconds from request acceptance to the final token.
    pub total_time_ms: f64,
    /// Completion tokens per second over the generation span.
    pub tokens_per_second: f64,
}

impl Default for LatencyStats {
    fn default() -> Self {
        LatencyStats {
            version: SHAPE_VERSION,
            time_to_first_token_ms: 0.0,
            total_time_ms: 0.0,
            tokens_per_second: 0.0,
        }
    }
}

impl Wire for LatencyStats {
    fn encode(&self) -> Value {
        Value::object(vec![
            ("version", Value::UInt(u64::from(self.version))),
            (
                "time_to_first_token_ms",
                Value::Float(self.time_to_first_token_ms),
            ),
            ("total_time_ms", Value::Float(self.total_time_ms)),
            ("tokens_per_second", Value::Float(self.tokens_per_second)),
        ])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(LatencyStats {
            version: field_u32(value, "version")?,
            time_to_first_token_ms: field_f64(value, "time_to_first_token_ms")?,
            total_time_ms: field_f64(value, "total_time_ms")?,
            tokens_per_second: field_f64(value, "tokens_per_second")?,
        })
    }
}

/// Static facts about a loaded model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    /// Contract shape version.
    pub version: u32,
    /// Model identifier as the platform names it.
    pub id: String,
    /// The file's own `general.architecture` string.
    pub architecture: String,
    /// Maximum context the model was loaded with, in tokens.
    pub context_size: u64,
    /// Parameter count from the file metadata.
    pub parameter_count: u64,
    /// Quantization label (for example `Q4_K_M`).
    pub quantization: String,
}

impl Wire for ModelInfo {
    fn encode(&self) -> Value {
        Value::object(vec![
            ("version", Value::UInt(u64::from(self.version))),
            ("id", Value::Str(self.id.clone())),
            ("architecture", Value::Str(self.architecture.clone())),
            ("context_size", Value::UInt(self.context_size)),
            ("parameter_count", Value::UInt(self.parameter_count)),
            ("quantization", Value::Str(self.quantization.clone())),
        ])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(ModelInfo {
            version: field_u32(value, "version")?,
            id: field_str(value, "id")?,
            architecture: field_str(value, "architecture")?,
            context_size: field_u64(value, "context_size")?,
            parameter_count: field_u64(value, "parameter_count")?,
            quantization: field_str(value, "quantization")?,
        })
    }
}

/// Sampling and generation parameters as the platform's `pkg/types`
/// carries them on a request. The seam's `Generate` message carries
/// the derived sampling block (see `message::SamplingParams`), built
/// from these by `pkg/sampling`.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelParams {
    /// Contract shape version.
    pub version: u32,
    /// Sampling temperature; 0 is greedy.
    pub temperature: f64,
    /// Nucleus sampling cutoff.
    pub top_p: f64,
    /// Top-k cutoff; 0 disables.
    pub top_k: u32,
    /// Maximum tokens to generate; 0 means the request sets no cap.
    pub max_tokens: u32,
    /// Repetition penalty multiplier; 1.0 disables.
    pub repeat_penalty: f64,
    /// Optional deterministic seed.
    pub seed: Option<u64>,
    /// Stop sequences.
    pub stop: Vec<String>,
}

impl Default for ModelParams {
    fn default() -> Self {
        ModelParams {
            version: SHAPE_VERSION,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            max_tokens: 0,
            repeat_penalty: 1.0,
            seed: None,
            stop: Vec::new(),
        }
    }
}

impl Wire for ModelParams {
    fn encode(&self) -> Value {
        let mut pairs = vec![
            ("version", Value::UInt(u64::from(self.version))),
            ("temperature", Value::Float(self.temperature)),
            ("top_p", Value::Float(self.top_p)),
            ("top_k", Value::UInt(u64::from(self.top_k))),
            ("max_tokens", Value::UInt(u64::from(self.max_tokens))),
            ("repeat_penalty", Value::Float(self.repeat_penalty)),
            ("stop", string_vec(&self.stop)),
        ];
        if let Some(seed) = self.seed {
            pairs.push(("seed", Value::UInt(seed)));
        }
        Value::object(pairs)
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(ModelParams {
            version: field_u32(value, "version")?,
            temperature: field_f64(value, "temperature")?,
            top_p: field_f64(value, "top_p")?,
            top_k: field_u32(value, "top_k")?,
            max_tokens: field_u32(value, "max_tokens")?,
            repeat_penalty: field_f64(value, "repeat_penalty")?,
            seed: crate::json::field_opt_u64(value, "seed")?,
            stop: field_string_vec(value, "stop")?,
        })
    }
}

/// One inference request as the platform's contracts carry it.
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceRequest {
    /// Contract shape version.
    pub version: u32,
    /// Model identifier.
    pub model: String,
    /// The conversation so far.
    pub messages: Vec<ChatMessage>,
    /// Generation parameters.
    pub params: ModelParams,
    /// Whether the caller wants streamed token events.
    pub stream: bool,
}

impl Wire for InferenceRequest {
    fn encode(&self) -> Value {
        Value::object(vec![
            ("version", Value::UInt(u64::from(self.version))),
            ("model", Value::Str(self.model.clone())),
            (
                "messages",
                Value::Array(self.messages.iter().map(Wire::encode).collect()),
            ),
            ("params", self.params.encode()),
            ("stream", Value::Bool(self.stream)),
        ])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        let messages = field_array(value, "messages")?
            .iter()
            .map(ChatMessage::decode)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(InferenceRequest {
            version: field_u32(value, "version")?,
            model: field_str(value, "model")?,
            messages,
            params: ModelParams::decode(field(value, "params")?)?,
            stream: field_bool(value, "stream")?,
        })
    }
}

/// One completed inference response.
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceResponse {
    /// Contract shape version.
    pub version: u32,
    /// Model identifier that served the request.
    pub model: String,
    /// The assistant's reply message.
    pub message: ChatMessage,
    /// Why generation stopped (`stop`, `length`, ...).
    pub finish_reason: String,
    /// Token accounting.
    pub usage: TokenUsage,
    /// Measured latency.
    pub latency: LatencyStats,
}

impl Wire for InferenceResponse {
    fn encode(&self) -> Value {
        Value::object(vec![
            ("version", Value::UInt(u64::from(self.version))),
            ("model", Value::Str(self.model.clone())),
            ("message", self.message.encode()),
            ("finish_reason", Value::Str(self.finish_reason.clone())),
            ("usage", self.usage.encode()),
            ("latency", self.latency.encode()),
        ])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(InferenceResponse {
            version: field_u32(value, "version")?,
            model: field_str(value, "model")?,
            message: ChatMessage::decode(field(value, "message")?)?,
            finish_reason: field_str(value, "finish_reason")?,
            usage: TokenUsage::decode(field(value, "usage")?)?,
            latency: LatencyStats::decode(field(value, "latency")?)?,
        })
    }
}
