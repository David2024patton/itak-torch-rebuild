//! The Section 4 control-plane message set as types: every message in
//! the ARCHITECTURE.md table, plus the reply companions the table
//! implies (`SessionCreated`, `TokenizeResult`, `DetokenizeResult`,
//! `EmbedResult`, `StateResult`), a generic `Ack` for the
//! confirm-only operations, a `Refused` frame that carries named
//! refusals (the version mismatch above all), and `ReleaseShm` for
//! the explicit shared-memory ownership release Section 4 requires.
//!
//! Message type codes are part of the contract and never renumber.
//! Codes are grouped by family: 1-9 handshake and control, 10-19
//! model lifecycle, 20-29 sessions, 30-39 tokenizer, 40-49
//! generation, 50-59 embeddings and shared memory, 60-69 state,
//! 70-79 health, 80 shutdown.

use crate::error::Error;
use crate::json::{
    field, field_bool, field_opt, field_opt_f32, field_opt_str, field_opt_u64, field_str,
    field_string_vec, field_u16, field_u32, field_u32_vec, field_u64, string_vec, u32_vec, Value,
    Wire,
};
use crate::types::ModelInfo;

/// Direction a message travels, per the Section 4 table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Go platform to Rust engine.
    GoToRust,
    /// Rust engine to Go platform.
    RustToGo,
    /// Either side may send it.
    Both,
}

/// A message type code from the frame header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    /// First exchange, client side: protocol version, capabilities.
    Hello,
    /// First exchange, server side: version, capabilities, hardware.
    HelloAck,
    /// A named refusal, either direction. Never a hang.
    Refused,
    /// Generic confirmation for operations with no result payload.
    Ack,
    /// Load a model file.
    LoadModel,
    /// Load outcome: coverage report or the named load error.
    LoadResult,
    /// Release a loaded model.
    UnloadModel,
    /// Open a generation session on a loaded model.
    CreateSession,
    /// Session open outcome: the session handle.
    SessionCreated,
    /// Close a session.
    DestroySession,
    /// Text to ids; the engine owns the tokenizer.
    Tokenize,
    /// Tokenizer output.
    TokenizeResult,
    /// Ids to text.
    Detokenize,
    /// Detokenized output.
    DetokenizeResult,
    /// Run generation on a session.
    Generate,
    /// One streamed token, via the shared-memory ring with a socket
    /// wakeup (the wakeup frame carries the ring sequence).
    TokenEvent,
    /// Stop a generation at the next token boundary.
    Abort,
    /// Embed texts; vectors return as a shared-memory handle.
    Embed,
    /// Embedding outcome: the shared-memory handle.
    EmbedResult,
    /// Release a shared-memory segment back to its owner.
    ReleaseShm,
    /// Persist session state to the configured state directory.
    SaveState,
    /// Restore session state from the configured state directory.
    LoadState,
    /// State operation outcome.
    StateResult,
    /// Liveness check.
    Health,
    /// The diagnostic dump: refusals, fallbacks, decisions, with codes.
    Diagnostics,
    /// Graceful drain, then exit.
    Shutdown,
}

impl MessageType {
    /// The wire code for this message type.
    pub fn code(self) -> u16 {
        match self {
            MessageType::Hello => 1,
            MessageType::HelloAck => 2,
            MessageType::Refused => 3,
            MessageType::Ack => 4,
            MessageType::LoadModel => 10,
            MessageType::LoadResult => 11,
            MessageType::UnloadModel => 12,
            MessageType::CreateSession => 20,
            MessageType::SessionCreated => 21,
            MessageType::DestroySession => 22,
            MessageType::Tokenize => 30,
            MessageType::TokenizeResult => 31,
            MessageType::Detokenize => 32,
            MessageType::DetokenizeResult => 33,
            MessageType::Generate => 40,
            MessageType::TokenEvent => 41,
            MessageType::Abort => 42,
            MessageType::Embed => 50,
            MessageType::EmbedResult => 51,
            MessageType::ReleaseShm => 52,
            MessageType::SaveState => 60,
            MessageType::LoadState => 61,
            MessageType::StateResult => 62,
            MessageType::Health => 70,
            MessageType::Diagnostics => 71,
            MessageType::Shutdown => 80,
        }
    }

    /// Resolve a wire code, naming the failure on unknown codes.
    pub fn from_code(code: u16) -> Result<MessageType, Error> {
        let ty = match code {
            1 => MessageType::Hello,
            2 => MessageType::HelloAck,
            3 => MessageType::Refused,
            4 => MessageType::Ack,
            10 => MessageType::LoadModel,
            11 => MessageType::LoadResult,
            12 => MessageType::UnloadModel,
            20 => MessageType::CreateSession,
            21 => MessageType::SessionCreated,
            22 => MessageType::DestroySession,
            30 => MessageType::Tokenize,
            31 => MessageType::TokenizeResult,
            32 => MessageType::Detokenize,
            33 => MessageType::DetokenizeResult,
            40 => MessageType::Generate,
            41 => MessageType::TokenEvent,
            42 => MessageType::Abort,
            50 => MessageType::Embed,
            51 => MessageType::EmbedResult,
            52 => MessageType::ReleaseShm,
            60 => MessageType::SaveState,
            61 => MessageType::LoadState,
            62 => MessageType::StateResult,
            70 => MessageType::Health,
            71 => MessageType::Diagnostics,
            80 => MessageType::Shutdown,
            other => return Err(Error::UnknownMessageType(other)),
        };
        Ok(ty)
    }

    /// The direction this message travels, per the Section 4 table.
    pub fn direction(self) -> Direction {
        match self {
            MessageType::Hello | MessageType::HelloAck | MessageType::Refused => Direction::Both,
            MessageType::LoadResult
            | MessageType::SessionCreated
            | MessageType::TokenizeResult
            | MessageType::DetokenizeResult
            | MessageType::TokenEvent
            | MessageType::EmbedResult
            | MessageType::StateResult
            | MessageType::Diagnostics
            | MessageType::Ack => Direction::RustToGo,
            _ => Direction::GoToRust,
        }
    }
}

/// Hello, the first frame on the socket (either side may speak
/// first by arrangement; today the platform dials and speaks).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hello {
    /// The sender's protocol version (major in the high byte).
    pub protocol_version: u16,
    /// Human-readable peer name, for diagnostics (`platform`, `engine`).
    pub peer_name: String,
    /// Capability strings the sender supports.
    pub capabilities: Vec<String>,
}

impl Wire for Hello {
    fn encode(&self) -> Value {
        Value::object(vec![
            (
                "protocol_version",
                Value::UInt(u64::from(self.protocol_version)),
            ),
            ("peer_name", Value::Str(self.peer_name.clone())),
            ("capabilities", string_vec(&self.capabilities)),
        ])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(Hello {
            protocol_version: field_u16(value, "protocol_version")?,
            peer_name: field_str(value, "peer_name")?,
            capabilities: field_string_vec(value, "capabilities")?,
        })
    }
}

/// A summary of the engine's hardware, carried by `HelloAck`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareSummary {
    /// Backend in use (`cpu`, `vulkan`, ...).
    pub backend: String,
    /// Device name as the backend reports it.
    pub device_name: String,
    /// Usable device memory in bytes (0 for the CPU backend).
    pub vram_bytes: u64,
}

impl Wire for HardwareSummary {
    fn encode(&self) -> Value {
        Value::object(vec![
            ("backend", Value::Str(self.backend.clone())),
            ("device_name", Value::Str(self.device_name.clone())),
            ("vram_bytes", Value::UInt(self.vram_bytes)),
        ])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(HardwareSummary {
            backend: field_str(value, "backend")?,
            device_name: field_str(value, "device_name")?,
            vram_bytes: field_u64(value, "vram_bytes")?,
        })
    }
}

/// HelloAck: the server's half of the first exchange.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HelloAck {
    /// The server's protocol version.
    pub protocol_version: u16,
    /// Human-readable peer name.
    pub peer_name: String,
    /// Capability strings the server supports.
    pub capabilities: Vec<String>,
    /// The engine's hardware report summary.
    pub hardware: HardwareSummary,
}

impl Wire for HelloAck {
    fn encode(&self) -> Value {
        Value::object(vec![
            (
                "protocol_version",
                Value::UInt(u64::from(self.protocol_version)),
            ),
            ("peer_name", Value::Str(self.peer_name.clone())),
            ("capabilities", string_vec(&self.capabilities)),
            ("hardware", self.hardware.encode()),
        ])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(HelloAck {
            protocol_version: field_u16(value, "protocol_version")?,
            peer_name: field_str(value, "peer_name")?,
            capabilities: field_string_vec(value, "capabilities")?,
            hardware: HardwareSummary::decode(field(value, "hardware")?)?,
        })
    }
}

/// A named refusal. The `code` is one of the stable strings in
/// [`crate::codes`]; the refusal is how the seam says no, on the wire
/// and in the diagnostic channel alike.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    /// Stable diagnostic code naming the refusal.
    pub code: String,
    /// Human-readable detail.
    pub message: String,
    /// The refusing side's protocol version, set on version refusals
    /// so the peer can name both versions in its own error.
    pub protocol_version: Option<u16>,
}

impl Wire for Refusal {
    fn encode(&self) -> Value {
        let mut pairs = vec![
            ("code", Value::Str(self.code.clone())),
            ("message", Value::Str(self.message.clone())),
        ];
        if let Some(v) = self.protocol_version {
            pairs.push(("protocol_version", Value::UInt(u64::from(v))));
        }
        Value::object(pairs)
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        let protocol_version = match field_opt_u64(value, "protocol_version")? {
            Some(v) => Some(u16::try_from(v).map_err(|_| {
                Error::MalformedPayload("refusal protocol_version does not fit u16".into())
            })?),
            None => None,
        };
        Ok(Refusal {
            code: field_str(value, "code")?,
            message: field_str(value, "message")?,
            protocol_version,
        })
    }
}

/// Generic confirmation for operations with no result payload
/// (UnloadModel, DestroySession, Abort, Shutdown, ReleaseShm).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ack {
    /// Whether the operation completed.
    pub ok: bool,
    /// Optional human-readable detail; empty when there is none.
    pub detail: String,
}

impl Wire for Ack {
    fn encode(&self) -> Value {
        Value::object(vec![
            ("ok", Value::Bool(self.ok)),
            ("detail", Value::Str(self.detail.clone())),
        ])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(Ack {
            ok: field_bool(value, "ok")?,
            detail: field_str(value, "detail")?,
        })
    }
}

/// Load a model file into the engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadModel {
    /// Path to the model file, confined to configured model dirs.
    pub path: String,
    /// The architecture the platform expects the file to declare;
    /// the engine refuses by name when the file disagrees.
    pub expected_architecture: Option<String>,
}

impl Wire for LoadModel {
    fn encode(&self) -> Value {
        let mut pairs = vec![("path", Value::Str(self.path.clone()))];
        if let Some(arch) = &self.expected_architecture {
            pairs.push(("expected_architecture", Value::Str(arch.clone())));
        }
        Value::object(pairs)
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(LoadModel {
            path: field_str(value, "path")?,
            expected_architecture: field_opt_str(value, "expected_architecture")?,
        })
    }
}

/// The loader's coverage report for one file: what the arch module
/// required, what arrived, and exactly what did not (Section 5.2:
/// any gap is an error value, never a silent default).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoverageReport {
    /// Tensors the arch module required.
    pub tensors_expected: u64,
    /// Tensors found readable and correctly shaped.
    pub tensors_loaded: u64,
    /// Required tensors absent from the file, by name.
    pub missing: Vec<String>,
    /// Tensors present but unreadable or misshapen, by name.
    pub unreadable: Vec<String>,
}

impl Wire for CoverageReport {
    fn encode(&self) -> Value {
        Value::object(vec![
            ("tensors_expected", Value::UInt(self.tensors_expected)),
            ("tensors_loaded", Value::UInt(self.tensors_loaded)),
            ("missing", string_vec(&self.missing)),
            ("unreadable", string_vec(&self.unreadable)),
        ])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(CoverageReport {
            tensors_expected: field_u64(value, "tensors_expected")?,
            tensors_loaded: field_u64(value, "tensors_loaded")?,
            missing: field_string_vec(value, "missing")?,
            unreadable: field_string_vec(value, "unreadable")?,
        })
    }
}

/// A named load failure: the model does not serve, and this says why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadError {
    /// Stable diagnostic code.
    pub code: String,
    /// Human-readable detail.
    pub message: String,
    /// Tensors named by the failure, when it is a coverage failure.
    pub missing: Vec<String>,
}

impl Wire for LoadError {
    fn encode(&self) -> Value {
        Value::object(vec![
            ("code", Value::Str(self.code.clone())),
            ("message", Value::Str(self.message.clone())),
            ("missing", string_vec(&self.missing)),
        ])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(LoadError {
            code: field_str(value, "code")?,
            message: field_str(value, "message")?,
            missing: field_string_vec(value, "missing")?,
        })
    }
}

/// The outcome of a `LoadModel`.
#[derive(Debug, Clone, PartialEq)]
pub enum LoadResult {
    /// The model loaded and serves under this handle.
    Loaded {
        /// Engine-side model handle.
        model_handle: u64,
        /// Static facts about the loaded model.
        model: ModelInfo,
        /// The load coverage report.
        coverage: CoverageReport,
    },
    /// The load was refused; the model does not serve.
    Refused {
        /// The named load error.
        error: LoadError,
    },
}

impl Wire for LoadResult {
    fn encode(&self) -> Value {
        match self {
            LoadResult::Loaded {
                model_handle,
                model,
                coverage,
            } => Value::object(vec![
                ("status", Value::Str("loaded".to_string())),
                ("model_handle", Value::UInt(*model_handle)),
                ("model", model.encode()),
                ("coverage", coverage.encode()),
            ]),
            LoadResult::Refused { error } => Value::object(vec![
                ("status", Value::Str("refused".to_string())),
                ("error", error.encode()),
            ]),
        }
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        match field_str(value, "status")?.as_str() {
            "loaded" => Ok(LoadResult::Loaded {
                model_handle: field_u64(value, "model_handle")?,
                model: ModelInfo::decode(field(value, "model")?)?,
                coverage: CoverageReport::decode(field(value, "coverage")?)?,
            }),
            "refused" => Ok(LoadResult::Refused {
                error: LoadError::decode(field(value, "error")?)?,
            }),
            other => Err(Error::MalformedPayload(format!(
                "unknown load status '{other}'"
            ))),
        }
    }
}

/// Release a loaded model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnloadModel {
    /// The model handle from `LoadResult`.
    pub model_handle: u64,
}

impl Wire for UnloadModel {
    fn encode(&self) -> Value {
        Value::object(vec![("model_handle", Value::UInt(self.model_handle))])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(UnloadModel {
            model_handle: field_u64(value, "model_handle")?,
        })
    }
}

/// The KV policy a session runs under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvPolicy {
    /// Paged cache for global layers.
    Paged,
    /// Fixed ring buffers for sliding-window layers.
    Ring,
    /// The hybrid schedule: paged global, ring sliding-window.
    Hybrid,
}

impl KvPolicy {
    /// The wire string for this policy.
    pub fn as_str(&self) -> &'static str {
        match self {
            KvPolicy::Paged => "paged",
            KvPolicy::Ring => "ring",
            KvPolicy::Hybrid => "hybrid",
        }
    }

    /// Parse the wire string, naming the failure on unknown policies.
    pub fn from_wire(text: &str) -> Result<KvPolicy, Error> {
        match text {
            "paged" => Ok(KvPolicy::Paged),
            "ring" => Ok(KvPolicy::Ring),
            "hybrid" => Ok(KvPolicy::Hybrid),
            other => Err(Error::MalformedPayload(format!(
                "unknown kv policy '{other}'"
            ))),
        }
    }
}

impl Wire for KvPolicy {
    fn encode(&self) -> Value {
        Value::Str(self.as_str().to_string())
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        value
            .as_str()
            .ok_or_else(|| Error::MalformedPayload("kv policy is not a string".to_string()))
            .and_then(KvPolicy::from_wire)
    }
}

/// Open a generation session on a loaded model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CreateSession {
    /// The model to serve the session.
    pub model_handle: u64,
    /// Context size in tokens for this session.
    pub context_size: u64,
    /// KV policy for this session.
    pub kv_policy: KvPolicy,
}

impl Wire for CreateSession {
    fn encode(&self) -> Value {
        Value::object(vec![
            ("model_handle", Value::UInt(self.model_handle)),
            ("context_size", Value::UInt(self.context_size)),
            ("kv_policy", self.kv_policy.encode()),
        ])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(CreateSession {
            model_handle: field_u64(value, "model_handle")?,
            context_size: field_u64(value, "context_size")?,
            kv_policy: KvPolicy::decode(field(value, "kv_policy")?)?,
        })
    }
}

/// The outcome of a `CreateSession`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionCreated {
    /// Engine-side session handle.
    pub session_handle: u64,
}

impl Wire for SessionCreated {
    fn encode(&self) -> Value {
        Value::object(vec![("session_handle", Value::UInt(self.session_handle))])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(SessionCreated {
            session_handle: field_u64(value, "session_handle")?,
        })
    }
}

/// Close a session and release its KV state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DestroySession {
    /// The session handle from `SessionCreated`.
    pub session_handle: u64,
}

impl Wire for DestroySession {
    fn encode(&self) -> Value {
        Value::object(vec![("session_handle", Value::UInt(self.session_handle))])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(DestroySession {
            session_handle: field_u64(value, "session_handle")?,
        })
    }
}

/// Tokenize text. The engine owns the tokenizer; the platform never
/// guesses (Section 4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tokenize {
    /// Session whose model's tokenizer to use, when session-bound.
    pub session_handle: Option<u64>,
    /// The text to encode.
    pub text: String,
}

impl Wire for Tokenize {
    fn encode(&self) -> Value {
        let mut pairs = vec![("text", Value::Str(self.text.clone()))];
        if let Some(h) = self.session_handle {
            pairs.push(("session_handle", Value::UInt(h)));
        }
        Value::object(pairs)
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(Tokenize {
            session_handle: field_opt_u64(value, "session_handle")?,
            text: field_str(value, "text")?,
        })
    }
}

/// Tokenizer output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizeResult {
    /// The token ids.
    pub token_ids: Vec<u32>,
}

impl Wire for TokenizeResult {
    fn encode(&self) -> Value {
        Value::object(vec![("token_ids", u32_vec(&self.token_ids))])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(TokenizeResult {
            token_ids: field_u32_vec(value, "token_ids")?,
        })
    }
}

/// Detokenize ids back to text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Detokenize {
    /// The token ids to decode.
    pub token_ids: Vec<u32>,
}

impl Wire for Detokenize {
    fn encode(&self) -> Value {
        Value::object(vec![("token_ids", u32_vec(&self.token_ids))])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(Detokenize {
            token_ids: field_u32_vec(value, "token_ids")?,
        })
    }
}

/// Detokenized output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetokenizeResult {
    /// The decoded text.
    pub text: String,
}

impl Wire for DetokenizeResult {
    fn encode(&self) -> Value {
        Value::object(vec![("text", Value::Str(self.text.clone()))])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(DetokenizeResult {
            text: field_str(value, "text")?,
        })
    }
}

/// A reference to bytes in a shared-memory segment: the only way
/// bulk data crosses the seam. Ownership is explicit; the receiver
/// releases the segment with `ReleaseShm` when done.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShmRef {
    /// Segment name as both sides agreed it at setup.
    pub segment: String,
    /// Byte offset of the data inside the segment.
    pub offset: u64,
    /// Byte length of the data.
    pub length: u64,
}

impl Wire for ShmRef {
    fn encode(&self) -> Value {
        Value::object(vec![
            ("segment", Value::Str(self.segment.clone())),
            ("offset", Value::UInt(self.offset)),
            ("length", Value::UInt(self.length)),
        ])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(ShmRef {
            segment: field_str(value, "segment")?,
            offset: field_u64(value, "offset")?,
            length: field_u64(value, "length")?,
        })
    }
}

/// The sampling block of a `Generate` message, built by
/// `pkg/sampling` from the request's `ModelParams`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
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
}

impl Wire for SamplingParams {
    fn encode(&self) -> Value {
        let mut pairs = vec![
            ("temperature", Value::Float(self.temperature)),
            ("top_p", Value::Float(self.top_p)),
            ("top_k", Value::UInt(u64::from(self.top_k))),
            ("max_tokens", Value::UInt(u64::from(self.max_tokens))),
            ("repeat_penalty", Value::Float(self.repeat_penalty)),
        ];
        if let Some(seed) = self.seed {
            pairs.push(("seed", Value::UInt(seed)));
        }
        Value::object(pairs)
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(SamplingParams {
            temperature: crate::json::field_f64(value, "temperature")?,
            top_p: crate::json::field_f64(value, "top_p")?,
            top_k: field_u32(value, "top_k")?,
            max_tokens: field_u32(value, "max_tokens")?,
            repeat_penalty: crate::json::field_f64(value, "repeat_penalty")?,
            seed: field_opt_u64(value, "seed")?,
        })
    }
}

/// The kind of grammar constraint on a `Generate`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrammarKind {
    /// A GBNF grammar.
    Gbnf,
    /// Free-form JSON mode.
    Json,
    /// JSON conforming to a schema.
    JsonSchema,
}

impl GrammarKind {
    /// The wire string for this kind.
    pub fn as_str(&self) -> &'static str {
        match self {
            GrammarKind::Gbnf => "gbnf",
            GrammarKind::Json => "json",
            GrammarKind::JsonSchema => "json_schema",
        }
    }

    /// Parse the wire string, naming the failure on unknown kinds.
    pub fn from_wire(text: &str) -> Result<GrammarKind, Error> {
        match text {
            "gbnf" => Ok(GrammarKind::Gbnf),
            "json" => Ok(GrammarKind::Json),
            "json_schema" => Ok(GrammarKind::JsonSchema),
            other => Err(Error::MalformedPayload(format!(
                "unknown grammar kind '{other}'"
            ))),
        }
    }
}

impl Wire for GrammarKind {
    fn encode(&self) -> Value {
        Value::Str(self.as_str().to_string())
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        value
            .as_str()
            .ok_or_else(|| Error::MalformedPayload("grammar kind is not a string".to_string()))
            .and_then(GrammarKind::from_wire)
    }
}

/// The grammar block of a `Generate` message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrammarSpec {
    /// Which grammar language `source` is written in.
    pub kind: GrammarKind,
    /// The grammar text or the JSON schema document.
    pub source: String,
}

impl Wire for GrammarSpec {
    fn encode(&self) -> Value {
        Value::object(vec![
            ("kind", self.kind.encode()),
            ("source", Value::Str(self.source.clone())),
        ])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(GrammarSpec {
            kind: GrammarKind::decode(field(value, "kind")?)?,
            source: field_str(value, "source")?,
        })
    }
}

/// Stop rules for a `Generate`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StopRules {
    /// Stop when any of these sequences is produced.
    pub sequences: Vec<String>,
    /// Stop when any of these token ids is produced.
    pub token_ids: Vec<u32>,
}

impl Wire for StopRules {
    fn encode(&self) -> Value {
        Value::object(vec![
            ("sequences", string_vec(&self.sequences)),
            ("token_ids", u32_vec(&self.token_ids)),
        ])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(StopRules {
            sequences: field_string_vec(value, "sequences")?,
            token_ids: field_u32_vec(value, "token_ids")?,
        })
    }
}

/// Run generation on a session. The prompt arrives as token ids or
/// as a shared-memory reference, never as raw tensors on the socket.
#[derive(Debug, Clone, PartialEq)]
pub struct Generate {
    /// The session to generate on.
    pub session_handle: u64,
    /// Prompt token ids, inline for short prompts.
    pub token_ids: Option<Vec<u32>>,
    /// Prompt token ids by shared-memory reference, for long prompts.
    pub prompt_ref: Option<ShmRef>,
    /// The sampling block.
    pub sampling: SamplingParams,
    /// Optional grammar constraint.
    pub grammar: Option<GrammarSpec>,
    /// Stop rules.
    pub stop: StopRules,
}

impl Wire for Generate {
    fn encode(&self) -> Value {
        let mut pairs = vec![
            ("session_handle", Value::UInt(self.session_handle)),
            ("sampling", self.sampling.encode()),
            ("stop", self.stop.encode()),
        ];
        if let Some(ids) = &self.token_ids {
            pairs.push(("token_ids", u32_vec(ids)));
        }
        if let Some(r) = &self.prompt_ref {
            pairs.push(("prompt_ref", r.encode()));
        }
        if let Some(g) = &self.grammar {
            pairs.push(("grammar", g.encode()));
        }
        Value::object(pairs)
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        let token_ids = match value.get("token_ids") {
            None | Some(Value::Null) => None,
            Some(_) => Some(field_u32_vec(value, "token_ids")?),
        };
        Ok(Generate {
            session_handle: field_u64(value, "session_handle")?,
            token_ids,
            prompt_ref: field_opt::<ShmRef>(value, "prompt_ref")?,
            sampling: SamplingParams::decode(field(value, "sampling")?)?,
            grammar: field_opt::<GrammarSpec>(value, "grammar")?,
            stop: StopRules::decode(field(value, "stop")?)?,
        })
    }
}

/// One generated token. Events flow engine to platform through the
/// per-generation shared-memory ring; the socket `TokenEvent` frame
/// is the wakeup and carries the ring sequence it announces.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenEvent {
    /// The generated token id.
    pub token_id: u32,
    /// Log probability of the token, when the engine reports it.
    pub logprob: Option<f32>,
    /// Monotonic sequence number inside the generation's ring.
    pub ring_seq: u64,
    /// True on the final event of the generation.
    pub finished: bool,
    /// Why generation stopped, set on the final event.
    pub finish_reason: Option<String>,
}

impl Wire for TokenEvent {
    fn encode(&self) -> Value {
        let mut pairs = vec![
            ("token_id", Value::UInt(u64::from(self.token_id))),
            ("ring_seq", Value::UInt(self.ring_seq)),
            ("finished", Value::Bool(self.finished)),
        ];
        if let Some(lp) = self.logprob {
            pairs.push(("logprob", Value::Float(f64::from(lp))));
        }
        if let Some(reason) = &self.finish_reason {
            pairs.push(("finish_reason", Value::Str(reason.clone())));
        }
        Value::object(pairs)
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(TokenEvent {
            token_id: field_u32(value, "token_id")?,
            logprob: field_opt_f32(value, "logprob")?,
            ring_seq: field_u64(value, "ring_seq")?,
            finished: field_bool(value, "finished")?,
            finish_reason: field_opt_str(value, "finish_reason")?,
        })
    }
}

/// Stop the generation named by `target_request_id` at the next
/// token boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Abort {
    /// The request id of the `Generate` frame to stop.
    pub target_request_id: u64,
}

impl Wire for Abort {
    fn encode(&self) -> Value {
        Value::object(vec![(
            "target_request_id",
            Value::UInt(self.target_request_id),
        )])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(Abort {
            target_request_id: field_u64(value, "target_request_id")?,
        })
    }
}

/// Embed a batch of texts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Embed {
    /// The texts to embed.
    pub texts: Vec<String>,
}

impl Wire for Embed {
    fn encode(&self) -> Value {
        Value::object(vec![("texts", string_vec(&self.texts))])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(Embed {
            texts: field_string_vec(value, "texts")?,
        })
    }
}

/// Embedding outcome: the vectors live in shared memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedResult {
    /// Handle to the vector block: `count` rows of `dimension` f32.
    pub vectors: ShmRef,
    /// Components per vector.
    pub dimension: u64,
    /// Number of vectors.
    pub count: u64,
}

impl Wire for EmbedResult {
    fn encode(&self) -> Value {
        Value::object(vec![
            ("vectors", self.vectors.encode()),
            ("dimension", Value::UInt(self.dimension)),
            ("count", Value::UInt(self.count)),
        ])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(EmbedResult {
            vectors: ShmRef::decode(field(value, "vectors")?)?,
            dimension: field_u64(value, "dimension")?,
            count: field_u64(value, "count")?,
        })
    }
}

/// Release a shared-memory segment back to its owner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseShm {
    /// The segment name from the `ShmRef` being released.
    pub segment: String,
}

impl Wire for ReleaseShm {
    fn encode(&self) -> Value {
        Value::object(vec![("segment", Value::Str(self.segment.clone()))])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(ReleaseShm {
            segment: field_str(value, "segment")?,
        })
    }
}

/// Persist a session's state. `path` is confined to the configured
/// state directory and validated on the platform side before this
/// message is sent (Section 4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SaveState {
    /// The session whose state to save.
    pub session_handle: u64,
    /// Destination path inside the state directory.
    pub path: String,
}

impl Wire for SaveState {
    fn encode(&self) -> Value {
        Value::object(vec![
            ("session_handle", Value::UInt(self.session_handle)),
            ("path", Value::Str(self.path.clone())),
        ])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(SaveState {
            session_handle: field_u64(value, "session_handle")?,
            path: field_str(value, "path")?,
        })
    }
}

/// Restore a session's state from a previous `SaveState`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadState {
    /// The session to restore into.
    pub session_handle: u64,
    /// Source path inside the state directory.
    pub path: String,
}

impl Wire for LoadState {
    fn encode(&self) -> Value {
        Value::object(vec![
            ("session_handle", Value::UInt(self.session_handle)),
            ("path", Value::Str(self.path.clone())),
        ])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(LoadState {
            session_handle: field_u64(value, "session_handle")?,
            path: field_str(value, "path")?,
        })
    }
}

/// The outcome of a `SaveState` or `LoadState`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateResult {
    /// The operation completed against this path.
    Done {
        /// The path saved to or loaded from.
        path: String,
    },
    /// The operation was refused; state is unchanged.
    Refused {
        /// The named error.
        error: LoadError,
    },
}

impl Wire for StateResult {
    fn encode(&self) -> Value {
        match self {
            StateResult::Done { path } => Value::object(vec![
                ("status", Value::Str("done".to_string())),
                ("path", Value::Str(path.clone())),
            ]),
            StateResult::Refused { error } => Value::object(vec![
                ("status", Value::Str("refused".to_string())),
                ("error", error.encode()),
            ]),
        }
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        match field_str(value, "status")?.as_str() {
            "done" => Ok(StateResult::Done {
                path: field_str(value, "path")?,
            }),
            "refused" => Ok(StateResult::Refused {
                error: LoadError::decode(field(value, "error")?)?,
            }),
            other => Err(Error::MalformedPayload(format!(
                "unknown state status '{other}'"
            ))),
        }
    }
}

/// Liveness check. Empty payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Health {}

impl Wire for Health {
    fn encode(&self) -> Value {
        Value::object(vec![])
    }

    fn decode(_value: &Value) -> Result<Self, Error> {
        Ok(Health {})
    }
}

/// One line of the diagnostic dump: a stable code, what happened,
/// and how many times it happened since engine start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagEntry {
    /// Stable diagnostic code.
    pub code: String,
    /// Human-readable description of the latest occurrence.
    pub message: String,
    /// Occurrences since engine start.
    pub count: u64,
}

impl Wire for DiagEntry {
    fn encode(&self) -> Value {
        Value::object(vec![
            ("code", Value::Str(self.code.clone())),
            ("message", Value::Str(self.message.clone())),
            ("count", Value::UInt(self.count)),
        ])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(DiagEntry {
            code: field_str(value, "code")?,
            message: field_str(value, "message")?,
            count: field_u64(value, "count")?,
        })
    }
}

/// The reply to `Health`: liveness plus the diagnostic dump, what
/// the engine refused, fell back on, or decided, with codes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostics {
    /// Whether the engine considers itself able to serve.
    pub healthy: bool,
    /// Engine build version string.
    pub engine_version: String,
    /// The diagnostic entries.
    pub entries: Vec<DiagEntry>,
}

impl Wire for Diagnostics {
    fn encode(&self) -> Value {
        Value::object(vec![
            ("healthy", Value::Bool(self.healthy)),
            ("engine_version", Value::Str(self.engine_version.clone())),
            (
                "entries",
                Value::Array(self.entries.iter().map(Wire::encode).collect()),
            ),
        ])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        let entries = crate::json::field_array(value, "entries")?
            .iter()
            .map(DiagEntry::decode)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Diagnostics {
            healthy: field_bool(value, "healthy")?,
            engine_version: field_str(value, "engine_version")?,
            entries,
        })
    }
}

/// Ask the engine to drain in-flight work and exit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shutdown {
    /// How long the engine may take to drain before exiting anyway.
    pub drain_timeout_ms: u64,
}

impl Wire for Shutdown {
    fn encode(&self) -> Value {
        Value::object(vec![(
            "drain_timeout_ms",
            Value::UInt(self.drain_timeout_ms),
        )])
    }

    fn decode(value: &Value) -> Result<Self, Error> {
        Ok(Shutdown {
            drain_timeout_ms: field_u64(value, "drain_timeout_ms")?,
        })
    }
}

/// A decoded control-plane message: the Section 4 set as one type,
/// so the seam's two sides can match exhaustively.
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    /// See [`Hello`].
    Hello(Hello),
    /// See [`HelloAck`].
    HelloAck(HelloAck),
    /// See [`Refusal`].
    Refused(Refusal),
    /// See [`Ack`].
    Ack(Ack),
    /// See [`LoadModel`].
    LoadModel(LoadModel),
    /// See [`LoadResult`].
    LoadResult(LoadResult),
    /// See [`UnloadModel`].
    UnloadModel(UnloadModel),
    /// See [`CreateSession`].
    CreateSession(CreateSession),
    /// See [`SessionCreated`].
    SessionCreated(SessionCreated),
    /// See [`DestroySession`].
    DestroySession(DestroySession),
    /// See [`Tokenize`].
    Tokenize(Tokenize),
    /// See [`TokenizeResult`].
    TokenizeResult(TokenizeResult),
    /// See [`Detokenize`].
    Detokenize(Detokenize),
    /// See [`DetokenizeResult`].
    DetokenizeResult(DetokenizeResult),
    /// See [`Generate`].
    Generate(Generate),
    /// See [`TokenEvent`].
    TokenEvent(TokenEvent),
    /// See [`Abort`].
    Abort(Abort),
    /// See [`Embed`].
    Embed(Embed),
    /// See [`EmbedResult`].
    EmbedResult(EmbedResult),
    /// See [`ReleaseShm`].
    ReleaseShm(ReleaseShm),
    /// See [`SaveState`].
    SaveState(SaveState),
    /// See [`LoadState`].
    LoadState(LoadState),
    /// See [`StateResult`].
    StateResult(StateResult),
    /// See [`Health`].
    Health(Health),
    /// See [`Diagnostics`].
    Diagnostics(Diagnostics),
    /// See [`Shutdown`].
    Shutdown(Shutdown),
}

impl Message {
    /// The header message type for this message.
    pub fn message_type(&self) -> MessageType {
        match self {
            Message::Hello(_) => MessageType::Hello,
            Message::HelloAck(_) => MessageType::HelloAck,
            Message::Refused(_) => MessageType::Refused,
            Message::Ack(_) => MessageType::Ack,
            Message::LoadModel(_) => MessageType::LoadModel,
            Message::LoadResult(_) => MessageType::LoadResult,
            Message::UnloadModel(_) => MessageType::UnloadModel,
            Message::CreateSession(_) => MessageType::CreateSession,
            Message::SessionCreated(_) => MessageType::SessionCreated,
            Message::DestroySession(_) => MessageType::DestroySession,
            Message::Tokenize(_) => MessageType::Tokenize,
            Message::TokenizeResult(_) => MessageType::TokenizeResult,
            Message::Detokenize(_) => MessageType::Detokenize,
            Message::DetokenizeResult(_) => MessageType::DetokenizeResult,
            Message::Generate(_) => MessageType::Generate,
            Message::TokenEvent(_) => MessageType::TokenEvent,
            Message::Abort(_) => MessageType::Abort,
            Message::Embed(_) => MessageType::Embed,
            Message::EmbedResult(_) => MessageType::EmbedResult,
            Message::ReleaseShm(_) => MessageType::ReleaseShm,
            Message::SaveState(_) => MessageType::SaveState,
            Message::LoadState(_) => MessageType::LoadState,
            Message::StateResult(_) => MessageType::StateResult,
            Message::Health(_) => MessageType::Health,
            Message::Diagnostics(_) => MessageType::Diagnostics,
            Message::Shutdown(_) => MessageType::Shutdown,
        }
    }

    /// Encode the payload as the frame body bytes (JSON, UTF-8).
    pub fn encode_payload(&self) -> Vec<u8> {
        let value = match self {
            Message::Hello(m) => m.encode(),
            Message::HelloAck(m) => m.encode(),
            Message::Refused(m) => m.encode(),
            Message::Ack(m) => m.encode(),
            Message::LoadModel(m) => m.encode(),
            Message::LoadResult(m) => m.encode(),
            Message::UnloadModel(m) => m.encode(),
            Message::CreateSession(m) => m.encode(),
            Message::SessionCreated(m) => m.encode(),
            Message::DestroySession(m) => m.encode(),
            Message::Tokenize(m) => m.encode(),
            Message::TokenizeResult(m) => m.encode(),
            Message::Detokenize(m) => m.encode(),
            Message::DetokenizeResult(m) => m.encode(),
            Message::Generate(m) => m.encode(),
            Message::TokenEvent(m) => m.encode(),
            Message::Abort(m) => m.encode(),
            Message::Embed(m) => m.encode(),
            Message::EmbedResult(m) => m.encode(),
            Message::ReleaseShm(m) => m.encode(),
            Message::SaveState(m) => m.encode(),
            Message::LoadState(m) => m.encode(),
            Message::StateResult(m) => m.encode(),
            Message::Health(m) => m.encode(),
            Message::Diagnostics(m) => m.encode(),
            Message::Shutdown(m) => m.encode(),
        };
        value.to_json_string().into_bytes()
    }

    /// Decode a frame body as the message its header type declares.
    pub fn decode_payload(message_type: MessageType, payload: &[u8]) -> Result<Message, Error> {
        let text = std::str::from_utf8(payload)
            .map_err(|_| Error::MalformedPayload("payload is not valid UTF-8".to_string()))?;
        let value = Value::parse(text)?;
        let message = match message_type {
            MessageType::Hello => Message::Hello(Hello::decode(&value)?),
            MessageType::HelloAck => Message::HelloAck(HelloAck::decode(&value)?),
            MessageType::Refused => Message::Refused(Refusal::decode(&value)?),
            MessageType::Ack => Message::Ack(Ack::decode(&value)?),
            MessageType::LoadModel => Message::LoadModel(LoadModel::decode(&value)?),
            MessageType::LoadResult => Message::LoadResult(LoadResult::decode(&value)?),
            MessageType::UnloadModel => Message::UnloadModel(UnloadModel::decode(&value)?),
            MessageType::CreateSession => Message::CreateSession(CreateSession::decode(&value)?),
            MessageType::SessionCreated => Message::SessionCreated(SessionCreated::decode(&value)?),
            MessageType::DestroySession => Message::DestroySession(DestroySession::decode(&value)?),
            MessageType::Tokenize => Message::Tokenize(Tokenize::decode(&value)?),
            MessageType::TokenizeResult => Message::TokenizeResult(TokenizeResult::decode(&value)?),
            MessageType::Detokenize => Message::Detokenize(Detokenize::decode(&value)?),
            MessageType::DetokenizeResult => {
                Message::DetokenizeResult(DetokenizeResult::decode(&value)?)
            }
            MessageType::Generate => Message::Generate(Generate::decode(&value)?),
            MessageType::TokenEvent => Message::TokenEvent(TokenEvent::decode(&value)?),
            MessageType::Abort => Message::Abort(Abort::decode(&value)?),
            MessageType::Embed => Message::Embed(Embed::decode(&value)?),
            MessageType::EmbedResult => Message::EmbedResult(EmbedResult::decode(&value)?),
            MessageType::ReleaseShm => Message::ReleaseShm(ReleaseShm::decode(&value)?),
            MessageType::SaveState => Message::SaveState(SaveState::decode(&value)?),
            MessageType::LoadState => Message::LoadState(LoadState::decode(&value)?),
            MessageType::StateResult => Message::StateResult(StateResult::decode(&value)?),
            MessageType::Health => Message::Health(Health::decode(&value)?),
            MessageType::Diagnostics => Message::Diagnostics(Diagnostics::decode(&value)?),
            MessageType::Shutdown => Message::Shutdown(Shutdown::decode(&value)?),
        };
        Ok(message)
    }
}
