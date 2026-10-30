//! The one agreed payload encoding of the seam: JSON, UTF-8
//! (ARCHITECTURE.md Section 4: "serialized in one agreed encoding used
//! by both sides"). The Go side uses `encoding/json`; this module is
//! the std-only Rust counterpart, sized to the contract's own shapes:
//! objects, arrays, strings, booleans, null, and numbers. Integers keep
//! their exact value (`Int`/`UInt`); only genuinely fractional fields
//! (temperatures, logprobs, timings) travel as `Float`.
//!
//! Numbers on the wire follow JSON rules. A whole-valued float encodes
//! as an integer literal (`1` rather than `1.0`); readers accept both
//! forms for float fields, so the two sides agree either way.

use std::fmt::Write as _;

use crate::error::Error;

/// A parsed JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// `null`
    Null,
    /// `true` / `false`
    Bool(bool),
    /// A signed integer literal.
    Int(i64),
    /// An unsigned integer literal that does not fit in `i64`.
    UInt(u64),
    /// A fractional or exponent literal.
    Float(f64),
    /// A string.
    Str(String),
    /// An array of values.
    Array(Vec<Value>),
    /// An object: ordered key/value pairs.
    Object(Vec<(String, Value)>),
}

impl Value {
    /// Build an object from key/value pairs.
    pub fn object(pairs: Vec<(&str, Value)>) -> Value {
        Value::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    /// Look up a key in an object value.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The string contents, if this value is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// The boolean, if this value is a boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// The integer value, if this value is a whole number that fits.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::UInt(u) => Some(*u),
            Value::Int(i) => u64::try_from(*i).ok(),
            _ => None,
        }
    }

    /// The integer value as `u32`, if it fits.
    pub fn as_u32(&self) -> Option<u32> {
        self.as_u64().and_then(|v| u32::try_from(v).ok())
    }

    /// The numeric value as `f64`, accepting integer literals too.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Int(i) => Some(*i as f64),
            Value::UInt(u) => Some(*u as f64),
            _ => None,
        }
    }

    /// The numeric value as `f32`, accepting integer literals too.
    pub fn as_f32(&self) -> Option<f32> {
        self.as_f64().map(|f| f as f32)
    }

    /// The elements, if this value is an array.
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(items) => Some(items),
            _ => None,
        }
    }

    /// Serialize to a JSON string. Non-finite floats (NaN, infinities)
    /// have no JSON form and encode as `null`; the contract's float
    /// fields never carry them in practice.
    pub fn to_json_string(&self) -> String {
        let mut out = String::new();
        self.write_to(&mut out);
        out
    }

    fn write_to(&self, out: &mut String) {
        match self {
            Value::Null => out.push_str("null"),
            Value::Bool(true) => out.push_str("true"),
            Value::Bool(false) => out.push_str("false"),
            Value::Int(i) => {
                let _ = write!(out, "{i}");
            }
            Value::UInt(u) => {
                let _ = write!(out, "{u}");
            }
            Value::Float(f) => {
                if f.is_finite() {
                    let _ = write!(out, "{f}");
                } else {
                    out.push_str("null");
                }
            }
            Value::Str(s) => write_json_string(out, s),
            Value::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write_to(out);
                }
                out.push(']');
            }
            Value::Object(pairs) => {
                out.push('{');
                for (i, (k, v)) in pairs.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_json_string(out, k);
                    out.push(':');
                    v.write_to(out);
                }
                out.push('}');
            }
        }
    }

    /// Parse a JSON document. Trailing content after the value is an
    /// error: a frame payload is exactly one value.
    pub fn parse(text: &str) -> Result<Value, Error> {
        let mut parser = Parser {
            bytes: text.as_bytes(),
            pos: 0,
        };
        let value = parser.parse_value()?;
        parser.skip_ws();
        if parser.pos != parser.bytes.len() {
            return Err(Error::MalformedPayload(format!(
                "trailing bytes after JSON value at offset {}",
                parser.pos
            )));
        }
        Ok(value)
    }
}

fn write_json_string(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn err(&self, what: &str) -> Error {
        Error::MalformedPayload(format!("JSON parse error at offset {}: {what}", self.pos))
    }

    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len()
            && matches!(self.bytes[self.pos], b' ' | b'\t' | b'\n' | b'\r')
        {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn parse_value(&mut self) -> Result<Value, Error> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => Ok(Value::Str(self.parse_string()?)),
            Some(b't') => self.parse_lit("true", Value::Bool(true)),
            Some(b'f') => self.parse_lit("false", Value::Bool(false)),
            Some(b'n') => self.parse_lit("null", Value::Null),
            Some(b'-' | b'0'..=b'9') => self.parse_number(),
            _ => Err(self.err("unexpected character")),
        }
    }

    fn parse_lit(&mut self, lit: &str, value: Value) -> Result<Value, Error> {
        if self.bytes[self.pos..].starts_with(lit.as_bytes()) {
            self.pos += lit.len();
            Ok(value)
        } else {
            Err(self.err("invalid literal"))
        }
    }

    fn parse_object(&mut self) -> Result<Value, Error> {
        self.pos += 1; // '{'
        let mut pairs = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Value::Object(pairs));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(self.err("object key must be a string"));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(self.err("expected ':' in object"));
            }
            self.pos += 1;
            let value = self.parse_value()?;
            pairs.push((key, value));
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Value::Object(pairs));
                }
                _ => return Err(self.err("expected ',' or '}' in object")),
            }
        }
    }

    fn parse_array(&mut self) -> Result<Value, Error> {
        self.pos += 1; // '['
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Value::Array(items));
        }
        loop {
            let value = self.parse_value()?;
            items.push(value);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Value::Array(items));
                }
                _ => return Err(self.err("expected ',' or ']' in array")),
            }
        }
    }

    fn parse_string(&mut self) -> Result<String, Error> {
        self.pos += 1; // opening quote
        let mut out = String::new();
        loop {
            let byte = self.peek().ok_or_else(|| self.err("unterminated string"))?;
            self.pos += 1;
            match byte {
                b'"' => return Ok(out),
                b'\\' => {
                    let esc = self.peek().ok_or_else(|| self.err("unterminated escape"))?;
                    self.pos += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{08}'),
                        b'f' => out.push('\u{0c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.parse_hex4()?;
                            if (0xd800..0xdc00).contains(&hi) {
                                // High surrogate: a low surrogate must follow.
                                if self.peek() == Some(b'\\') {
                                    self.pos += 1;
                                    if self.peek() == Some(b'u') {
                                        self.pos += 1;
                                        let lo = self.parse_hex4()?;
                                        let combined = 0x1_0000u32
                                            + (((hi as u32) - 0xd800) << 10)
                                            + ((lo as u32).saturating_sub(0xdc00));
                                        out.push(char::from_u32(combined).unwrap_or('\u{fffd}'));
                                    } else {
                                        return Err(self.err("expected low surrogate escape"));
                                    }
                                } else {
                                    return Err(self.err("unpaired high surrogate"));
                                }
                            } else {
                                out.push(char::from_u32(hi as u32).unwrap_or('\u{fffd}'));
                            }
                        }
                        _ => return Err(self.err("invalid escape")),
                    }
                }
                _ => {
                    // Consume the full UTF-8 sequence for this byte.
                    let start = self.pos - 1;
                    let width = utf8_width(byte);
                    let end = start + width;
                    if end > self.bytes.len() {
                        return Err(self.err("truncated UTF-8 in string"));
                    }
                    match std::str::from_utf8(&self.bytes[start..end]) {
                        Ok(s) => out.push_str(s),
                        Err(_) => return Err(self.err("invalid UTF-8 in string")),
                    }
                    self.pos = end;
                }
            }
        }
    }

    fn parse_hex4(&mut self) -> Result<u16, Error> {
        if self.pos + 4 > self.bytes.len() {
            return Err(self.err("truncated \\u escape"));
        }
        let text = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4])
            .map_err(|_| self.err("invalid \\u escape"))?;
        let value = u16::from_str_radix(text, 16).map_err(|_| self.err("invalid \\u escape"))?;
        self.pos += 4;
        Ok(value)
    }

    fn parse_number(&mut self) -> Result<Value, Error> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        let mut fractional = false;
        while let Some(b) = self.peek() {
            match b {
                b'0'..=b'9' => self.pos += 1,
                b'.' | b'e' | b'E' | b'+' | b'-' => {
                    fractional = true;
                    self.pos += 1;
                }
                _ => break,
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| self.err("invalid number"))?;
        if !fractional {
            if let Ok(i) = text.parse::<i64>() {
                return Ok(Value::Int(i));
            }
            if let Ok(u) = text.parse::<u64>() {
                return Ok(Value::UInt(u));
            }
        }
        text.parse::<f64>()
            .map(Value::Float)
            .map_err(|_| self.err("invalid number"))
    }
}

fn utf8_width(first_byte: u8) -> usize {
    if first_byte < 0x80 {
        1
    } else if first_byte >> 5 == 0b110 {
        2
    } else if first_byte >> 4 == 0b1110 {
        3
    } else {
        4
    }
}

/// A contract shape that encodes to and decodes from the payload
/// encoding. Implemented by every payload struct in this crate.
pub trait Wire: Sized {
    /// Encode as a JSON value (an object for every contract shape).
    fn encode(&self) -> Value;
    /// Decode from a JSON value, naming the failure on bad input.
    fn decode(value: &Value) -> Result<Self, Error>;
}

/// Fetch a required field from an object value.
pub fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value, Error> {
    value
        .get(name)
        .ok_or_else(|| Error::MalformedPayload(format!("missing field '{name}'")))
}

/// Fetch a required string field.
pub fn field_str(value: &Value, name: &str) -> Result<String, Error> {
    field(value, name)?
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| Error::MalformedPayload(format!("field '{name}' is not a string")))
}

/// Fetch an optional string field: absent or null decodes as `None`.
pub fn field_opt_str(value: &Value, name: &str) -> Result<Option<String>, Error> {
    match value.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_str()
            .map(|s| Some(s.to_string()))
            .ok_or_else(|| Error::MalformedPayload(format!("field '{name}' is not a string"))),
    }
}

/// Fetch a required `u64` field.
pub fn field_u64(value: &Value, name: &str) -> Result<u64, Error> {
    field(value, name)?.as_u64().ok_or_else(|| {
        Error::MalformedPayload(format!("field '{name}' is not an unsigned integer"))
    })
}

/// Fetch an optional `u64` field: absent or null decodes as `None`.
pub fn field_opt_u64(value: &Value, name: &str) -> Result<Option<u64>, Error> {
    match value.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v.as_u64().map(Some).ok_or_else(|| {
            Error::MalformedPayload(format!("field '{name}' is not an unsigned integer"))
        }),
    }
}

/// Fetch a required `u32` field.
pub fn field_u32(value: &Value, name: &str) -> Result<u32, Error> {
    field(value, name)?
        .as_u32()
        .ok_or_else(|| Error::MalformedPayload(format!("field '{name}' is not a u32")))
}

/// Fetch a required `u16` field.
pub fn field_u16(value: &Value, name: &str) -> Result<u16, Error> {
    let v = field_u64(value, name)?;
    u16::try_from(v)
        .map_err(|_| Error::MalformedPayload(format!("field '{name}' does not fit in u16")))
}

/// Fetch a required `f64` field.
pub fn field_f64(value: &Value, name: &str) -> Result<f64, Error> {
    field(value, name)?
        .as_f64()
        .ok_or_else(|| Error::MalformedPayload(format!("field '{name}' is not a number")))
}

/// Fetch a required `f32` field.
pub fn field_f32(value: &Value, name: &str) -> Result<f32, Error> {
    field(value, name)?
        .as_f32()
        .ok_or_else(|| Error::MalformedPayload(format!("field '{name}' is not a number")))
}

/// Fetch an optional `f32` field: absent or null decodes as `None`.
pub fn field_opt_f32(value: &Value, name: &str) -> Result<Option<f32>, Error> {
    match value.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_f32()
            .map(Some)
            .ok_or_else(|| Error::MalformedPayload(format!("field '{name}' is not a number"))),
    }
}

/// Fetch a required boolean field.
pub fn field_bool(value: &Value, name: &str) -> Result<bool, Error> {
    field(value, name)?
        .as_bool()
        .ok_or_else(|| Error::MalformedPayload(format!("field '{name}' is not a boolean")))
}

/// Fetch a required array field.
pub fn field_array<'a>(value: &'a Value, name: &str) -> Result<&'a [Value], Error> {
    field(value, name)?
        .as_array()
        .ok_or_else(|| Error::MalformedPayload(format!("field '{name}' is not an array")))
}

/// Fetch an optional nested object field decoded via [`Wire`]:
/// absent or null decodes as `None`.
pub fn field_opt<T: Wire>(value: &Value, name: &str) -> Result<Option<T>, Error> {
    match value.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => T::decode(v).map(Some),
    }
}

/// Decode a required `Vec<String>` field.
pub fn field_string_vec(value: &Value, name: &str) -> Result<Vec<String>, Error> {
    field_array(value, name)?
        .iter()
        .map(|item| {
            item.as_str().map(str::to_string).ok_or_else(|| {
                Error::MalformedPayload(format!("field '{name}' holds a non-string"))
            })
        })
        .collect()
}

/// Decode a required `Vec<u32>` field.
pub fn field_u32_vec(value: &Value, name: &str) -> Result<Vec<u32>, Error> {
    field_array(value, name)?
        .iter()
        .map(|item| {
            item.as_u32()
                .ok_or_else(|| Error::MalformedPayload(format!("field '{name}' holds a non-u32")))
        })
        .collect()
}

/// Encode a string slice as a JSON array value.
pub fn string_vec(items: &[String]) -> Value {
    Value::Array(items.iter().map(|s| Value::Str(s.clone())).collect())
}

/// Encode a `u32` slice as a JSON array value.
pub fn u32_vec(items: &[u32]) -> Value {
    Value::Array(items.iter().map(|i| Value::UInt(u64::from(*i))).collect())
}
