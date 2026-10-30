//! The platform-to-engine socket protocol, defined once, here.
//!
//! Transport: a Unix domain socket, one JSON object per line
//! (NDJSON). The Go platform's client is
//! `platform/internal/engine/unix.go`; SERVING.md carries the same
//! definition in prose. Both sides implement this file's shapes and
//! nothing else.
//!
//! Frames from the platform:
//!
//! - `{"op":"hello","proto":1,"client":"torch-platform"}`: first
//!   frame on every connection.
//! - `{"op":"ping"}`: health probe.
//! - `{"op":"debug"}`: self-report request.
//! - `{"op":"chat_template"}`: ask for the model's own chat
//!   template. The reply is a `chat_template` frame carrying the
//!   template text from the GGUF metadata plus the `bos_token` and
//!   `eos_token` texts the template's variables refer to; a model
//!   whose file carries no template gets an `error` frame with
//!   code `no_chat_template` (TODO T-9.16). The platform renders
//!   the template; the engine only serves it, so the prompt the
//!   model sees is built from the file's own format string.
//! - `{"op":"embed","model":"...","input":["...","..."]}`: embed
//!   one or more texts. The reply is one `embeddings` frame (below)
//!   or an `error` frame. The vectors are mean-pooled token
//!   embeddings from the model's own embedding table (the same
//!   lookup the forward pass starts from), L2-normalized, in input
//!   order (TODO T-9.15).
//! - `{"op":"generate","model":"...","prompt":"...","max_tokens":N,
//!   "temperature":F,"top_p":F,"stop":["..."]}`: one generation.
//!   `max_tokens` absent or 0 means the engine's configured default;
//!   `temperature` absent or 0 is greedy; `top_p` absent, 0, or 1
//!   disables nucleus truncation. Optional sampling parity fields
//!   (TODO T-9.13): `seed` (integer; a seeded sampled run is
//!   reproducible, absent means the engine seeds from the clock),
//!   `top_k` (integer; absent or 0 disables top-k truncation), and
//!   `presence_penalty` / `frequency_penalty` (numbers; absent or 0
//!   disables the penalty).
//!
//! Frames from the engine:
//!
//! - `{"op":"hello_ack","proto":1,"detail":"..."}`: handshake reply.
//!   A peer speaking another protocol version gets an `error` frame
//!   with code `protocol_mismatch` instead, so its handshake fails by
//!   name rather than by hang.
//! - `{"op":"pong","detail":"..."}`.
//! - `{"op":"debug_info", ...}`: model identity, coverage, and
//!   counters; see `debug_info` below for the field set.
//! - `{"op":"token","text":"...","token_id":N,"done":false}`: one
//!   streamed piece of the completion. A generation ends with a
//!   final token frame with `done` true and `finish_reason` set to
//!   `stop` (EOS or a stop string) or `length` (max_tokens reached);
//!   its text is empty and the completion text is the concatenation
//!   of the earlier frames' text.
//! - `{"op":"error","code":"...","message":"..."}`: a named refusal.
//!   Codes the platform maps to a client error (HTTP 400):
//!   `bad_request`, `context_overflow`, `model_not_found`,
//!   `protocol_mismatch`. Codes it maps to a gateway error (502):
//!   `forward_failed`, `detokenize_failed`, `engine_died`,
//!   `unknown_op`, `bad_frame`, `internal`. And `engine_busy` maps
//!   to 503: every slot is running and the bounded job queue is
//!   full (TODO T-9.14).

use crate::json::{obj, Value};

/// The protocol version both sides speak. The Go client refuses a
/// hello_ack carrying any other value.
pub const PROTO_VERSION: f64 = 1.0;

/// A parsed generate request.
#[derive(Debug)]
pub struct GenerateRequest {
    /// Model name the platform asked for; empty means no preference.
    pub model: String,
    /// The flattened prompt text.
    pub prompt: String,
    /// Requested cap on generated tokens; None for the default.
    pub max_tokens: Option<usize>,
    /// Sampling temperature; 0 is greedy.
    pub temperature: f64,
    /// Nucleus mass cap; values outside (0, 1) disable truncation.
    pub top_p: f64,
    /// Stop strings.
    pub stop: Vec<String>,
    /// Request seed for reproducible sampled runs; None means the
    /// engine seeds from the clock.
    pub seed: Option<u64>,
    /// Top-k candidate cap; None or 0 disables truncation.
    pub top_k: Option<usize>,
    /// Presence penalty (OpenAI form); 0 disables.
    pub presence_penalty: f64,
    /// Frequency penalty (OpenAI form); 0 disables.
    pub frequency_penalty: f64,
}

/// Parse a generate frame. A missing or mistyped prompt is the only
/// hard failure; the numeric fields fall back to their documented
/// defaults so an older platform frame still generates.
pub fn parse_generate(frame: &Value) -> Result<GenerateRequest, String> {
    let prompt = frame
        .get("prompt")
        .and_then(Value::as_str)
        .ok_or_else(|| "generate frame has no string prompt".to_string())?
        .to_string();
    let model = frame
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let max_tokens = frame.get("max_tokens").and_then(Value::as_usize);
    let temperature = frame
        .get("temperature")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let top_p = frame.get("top_p").and_then(Value::as_f64).unwrap_or(0.0);
    let stop = frame
        .get("stop")
        .and_then(Value::as_arr)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let seed = frame.get("seed").and_then(Value::as_usize).map(|n| n as u64);
    let top_k = frame.get("top_k").and_then(Value::as_usize);
    let presence_penalty = frame
        .get("presence_penalty")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let frequency_penalty = frame
        .get("frequency_penalty")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    Ok(GenerateRequest {
        model,
        prompt,
        max_tokens,
        temperature,
        top_p,
        stop,
        seed,
        top_k,
        presence_penalty,
        frequency_penalty,
    })
}

/// A parsed embed request: the model name (empty means no
/// preference) and the input texts, in order.
pub fn parse_embed(frame: &Value) -> Result<(String, Vec<String>), String> {
    let model = frame
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let items = frame
        .get("input")
        .and_then(Value::as_arr)
        .ok_or_else(|| "embed frame has no input array".to_string())?;
    if items.is_empty() {
        return Err("embed frame has an empty input array".to_string());
    }
    let mut inputs = Vec::with_capacity(items.len());
    for item in items {
        match item.as_str() {
            Some(s) if !s.is_empty() => inputs.push(s.to_string()),
            _ => return Err("embed input entries must be non-empty strings".to_string()),
        }
    }
    Ok((model, inputs))
}

/// The embeddings reply: one vector per input, in input order.
pub fn embeddings_frame(model: &str, vectors: &[Vec<f32>]) -> Value {
    obj(vec![
        ("op", Value::Str("embeddings".into())),
        ("model", Value::Str(model.into())),
        (
            "embeddings",
            Value::Arr(
                vectors
                    .iter()
                    .map(|v| {
                        Value::Arr(v.iter().map(|&x| Value::Num(f64::from(x))).collect())
                    })
                    .collect(),
            ),
        ),
    ])
}

/// The chat template reply: the template text and the special
/// token texts its variables refer to.
pub fn chat_template_frame(template: &str, bos_token: &str, eos_token: &str) -> Value {
    obj(vec![
        ("op", Value::Str("chat_template".into())),
        ("template", Value::Str(template.into())),
        ("bos_token", Value::Str(bos_token.into())),
        ("eos_token", Value::Str(eos_token.into())),
    ])
}

/// The handshake reply.
pub fn hello_ack(detail: &str) -> Value {
    obj(vec![
        ("op", Value::Str("hello_ack".into())),
        ("proto", Value::Num(PROTO_VERSION)),
        ("detail", Value::Str(detail.into())),
    ])
}

/// The health reply.
pub fn pong(detail: &str) -> Value {
    obj(vec![
        ("op", Value::Str("pong".into())),
        ("detail", Value::Str(detail.into())),
    ])
}

/// One streamed token event, or the terminal event when `done`.
pub fn token_event(text: &str, token_id: u32, done: bool, finish_reason: Option<&str>) -> Value {
    let mut pairs = vec![
        ("op", Value::Str("token".into())),
        ("text", Value::Str(text.into())),
        ("token_id", Value::Num(f64::from(token_id))),
        ("done", Value::Bool(done)),
    ];
    if let Some(reason) = finish_reason {
        pairs.push(("finish_reason", Value::Str(reason.into())));
    }
    obj(pairs)
}

/// A named refusal.
pub fn error_frame(code: &str, message: &str) -> Value {
    obj(vec![
        ("op", Value::Str("error".into())),
        ("code", Value::Str(code.into())),
        ("message", Value::Str(message.into())),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse;

    #[test]
    fn generate_frame_parses_with_defaults() {
        let frame = parse(r#"{"op":"generate","prompt":"user: hi\nassistant:"}"#).expect("frame");
        let req = parse_generate(&frame).expect("request");
        assert_eq!(req.prompt, "user: hi\nassistant:");
        assert_eq!(req.model, "");
        assert_eq!(req.max_tokens, None);
        assert_eq!(req.temperature, 0.0);
        assert!(req.stop.is_empty());
    }

    #[test]
    fn generate_frame_full_shape() {
        let frame = parse(
            r#"{"op":"generate","model":"glimmer","prompt":"x","max_tokens":4,"temperature":0.7,"top_p":0.9,"stop":["\n","END"],"seed":42,"top_k":40,"presence_penalty":0.5,"frequency_penalty":-0.25}"#,
        )
        .expect("frame");
        let req = parse_generate(&frame).expect("request");
        assert_eq!(req.model, "glimmer");
        assert_eq!(req.max_tokens, Some(4));
        assert_eq!(req.temperature, 0.7);
        assert_eq!(req.top_p, 0.9);
        assert_eq!(req.stop, vec!["\n".to_string(), "END".to_string()]);
        assert_eq!(req.seed, Some(42));
        assert_eq!(req.top_k, Some(40));
        assert_eq!(req.presence_penalty, 0.5);
        assert_eq!(req.frequency_penalty, -0.25);
    }

    #[test]
    fn generate_frame_sampling_fields_default_off() {
        let frame = parse(r#"{"op":"generate","prompt":"x"}"#).expect("frame");
        let req = parse_generate(&frame).expect("request");
        assert_eq!(req.seed, None);
        assert_eq!(req.top_k, None);
        assert_eq!(req.presence_penalty, 0.0);
        assert_eq!(req.frequency_penalty, 0.0);
    }

    #[test]
    fn generate_without_prompt_is_a_named_error() {
        let frame = parse(r#"{"op":"generate"}"#).expect("frame");
        assert!(parse_generate(&frame).is_err());
    }

    #[test]
    fn embed_frame_parses_and_validates() {
        let frame = parse(r#"{"op":"embed","model":"glimmer","input":["a","b c"]}"#).expect("frame");
        let (model, inputs) = parse_embed(&frame).expect("request");
        assert_eq!(model, "glimmer");
        assert_eq!(inputs, vec!["a".to_string(), "b c".to_string()]);

        for bad in [
            r#"{"op":"embed"}"#,
            r#"{"op":"embed","input":[]}"#,
            r#"{"op":"embed","input":[""]}"#,
            r#"{"op":"embed","input":[3]}"#,
        ] {
            let frame = parse(bad).expect("frame");
            assert!(parse_embed(&frame).is_err(), "expected error for {bad}");
        }
    }

    #[test]
    fn embeddings_frame_shape() {
        let frame = embeddings_frame("glimmer", &[vec![0.5, -0.25], vec![1.0, 2.0]]);
        let text = crate::json::stringify(&frame);
        let back = parse(&text).expect("own output parses");
        assert_eq!(
            back.get("op").and_then(Value::as_str),
            Some("embeddings")
        );
        let vecs = back.get("embeddings").and_then(Value::as_arr).expect("vecs");
        assert_eq!(vecs.len(), 2);
    }
}
