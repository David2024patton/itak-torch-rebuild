//! Minimal JSON for the socket protocol.
//!
//! The workspace carries no external dependencies, and the NDJSON
//! protocol between the platform and this engine needs only objects,
//! arrays, strings, numbers, booleans, and null. This module is a
//! small recursive-descent parser and a serializer for exactly that,
//! with the string escapes (including `\uXXXX` surrogate pairs) the
//! tokenizer's token text can contain. Anything outside the grammar
//! is a named parse error with the byte offset, never a panic.

/// A parsed JSON value. Objects keep insertion order.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// JSON null.
    Null,
    /// JSON true or false.
    Bool(bool),
    /// A JSON number, held as f64 (token ids and counts in this
    /// protocol are far below 2^53, so the representation is exact).
    Num(f64),
    /// A JSON string, fully unescaped.
    Str(String),
    /// A JSON array.
    Arr(Vec<Value>),
    /// A JSON object as ordered key/value pairs.
    Obj(Vec<(String, Value)>),
}

impl Value {
    /// Look up `key` on an object value; None for any other shape.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
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

    /// The numeric value, if this value is a number.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Num(n) => Some(*n),
            _ => None,
        }
    }

    /// The numeric value as usize, if it is a non-negative integer in
    /// range. Fractional or negative numbers return None.
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Value::Num(n) if *n >= 0.0 && n.fract() == 0.0 && *n <= usize::MAX as f64 => {
                Some(*n as usize)
            }
            _ => None,
        }
    }

    /// The elements, if this value is an array.
    pub fn as_arr(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(items) => Some(items),
            _ => None,
        }
    }
}

/// Build an object value from key/value pairs.
pub fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Obj(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

/// Serialize a value as compact JSON. Strings escape the quote, the
/// backslash, and control characters; all other characters pass
/// through as UTF-8, which is valid JSON.
pub fn stringify(value: &Value) -> String {
    let mut out = String::new();
    write_value(value, &mut out);
    out
}

fn write_value(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Num(n) => write_num(*n, out),
        Value::Str(s) => write_str(s, out),
        Value::Arr(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(item, out);
            }
            out.push(']');
        }
        Value::Obj(pairs) => {
            out.push('{');
            for (i, (k, v)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_str(k, out);
                out.push(':');
                write_value(v, out);
            }
            out.push('}');
        }
    }
}

fn write_num(n: f64, out: &mut String) {
    if n.is_nan() || n.is_infinite() {
        // Not representable in JSON; the protocol never produces
        // these, and null is the least wrong rendering.
        out.push_str("null");
    } else if n.fract() == 0.0 && n.abs() <= 9_007_199_254_740_992.0 {
        out.push_str(&(n as i64).to_string());
    } else {
        out.push_str(&n.to_string());
    }
}

fn write_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Parse one complete JSON document. Trailing non-whitespace is an
/// error, so a truncated or doubled frame cannot slip through.
pub fn parse(input: &str) -> Result<Value, String> {
    let mut p = Parser {
        bytes: input.as_bytes(),
        pos: 0,
    };
    p.skip_ws();
    let value = p.parse_value(0)?;
    p.skip_ws();
    if p.pos != p.bytes.len() {
        return Err(format!("trailing data at byte {}", p.pos));
    }
    Ok(value)
}

/// The deepest nesting the parser accepts. Protocol frames nest a
/// handful of levels, so 128 is generous headroom; past it the
/// input is not a frame but an attack on the recursive parser's
/// call stack (20,000 nested arrays overflowed the stack and
/// aborted the whole engine process, adjudication CB), and it is
/// refused with a named parse error instead.
const MAX_DEPTH: usize = 128;

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn skip_ws(&mut self) {
        while let Some(b) = self.bytes.get(self.pos) {
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Result<u8, String> {
        self.bytes
            .get(self.pos)
            .copied()
            .ok_or_else(|| "unexpected end of input".to_string())
    }

    fn parse_value(&mut self, depth: usize) -> Result<Value, String> {
        if depth > MAX_DEPTH {
            return Err(format!(
                "nesting deeper than {MAX_DEPTH} levels at byte {}",
                self.pos
            ));
        }
        match self.peek()? {
            b'{' => self.parse_obj(depth),
            b'[' => self.parse_arr(depth),
            b'"' => Ok(Value::Str(self.parse_string()?)),
            b't' => self.parse_lit("true", Value::Bool(true)),
            b'f' => self.parse_lit("false", Value::Bool(false)),
            b'n' => self.parse_lit("null", Value::Null),
            b'-' | b'0'..=b'9' => self.parse_num(),
            other => Err(format!(
                "unexpected character {:?} at byte {}",
                other as char, self.pos
            )),
        }
    }

    fn parse_lit(&mut self, lit: &str, value: Value) -> Result<Value, String> {
        if self.bytes[self.pos..].starts_with(lit.as_bytes()) {
            self.pos += lit.len();
            Ok(value)
        } else {
            Err(format!("invalid literal at byte {}", self.pos))
        }
    }

    fn parse_num(&mut self) -> Result<Value, String> {
        let start = self.pos;
        if self.peek()? == b'-' {
            self.pos += 1;
        }
        while let Some(b) = self.bytes.get(self.pos) {
            if matches!(b, b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-') {
                self.pos += 1;
            } else {
                break;
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| format!("invalid number at byte {start}"))?;
        text.parse::<f64>()
            .map(Value::Num)
            .map_err(|_| format!("invalid number {text:?} at byte {start}"))
    }

    fn parse_string(&mut self) -> Result<String, String> {
        // Caller has peeked the opening quote.
        self.pos += 1;
        let mut out = String::new();
        loop {
            let b = self.peek()?;
            match b {
                b'"' => {
                    self.pos += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.pos += 1;
                    let esc = self.peek()?;
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
                                // High surrogate: a low surrogate must
                                // follow as a second \u escape.
                                if self.bytes.get(self.pos) == Some(&b'\\')
                                    && self.bytes.get(self.pos + 1) == Some(&b'u')
                                {
                                    self.pos += 2;
                                    let lo = self.parse_hex4()?;
                                    if !(0xdc00..0xe000).contains(&lo) {
                                        return Err(format!(
                                            "invalid low surrogate at byte {}",
                                            self.pos
                                        ));
                                    }
                                    let c = 0x1_0000
                                        + (((hi - 0xd800) as u32) << 10)
                                        + (lo - 0xdc00) as u32;
                                    out.push(char::from_u32(c).ok_or("invalid surrogate pair")?);
                                } else {
                                    return Err(format!("unpaired surrogate at byte {}", self.pos));
                                }
                            } else if (0xdc00..0xe000).contains(&hi) {
                                return Err(format!("unpaired low surrogate at byte {}", self.pos));
                            } else {
                                out.push(
                                    char::from_u32(hi as u32).ok_or("invalid unicode escape")?,
                                );
                            }
                        }
                        other => {
                            return Err(format!(
                                "invalid escape {:?} at byte {}",
                                other as char, self.pos
                            ))
                        }
                    }
                }
                0x00..=0x1f => {
                    return Err(format!(
                        "raw control character in string at byte {}",
                        self.pos
                    ))
                }
                _ => {
                    // Copy one UTF-8 encoded character verbatim. The
                    // input is a &str, so the bytes are valid UTF-8
                    // and the lead byte gives the length.
                    let len = utf8_len(b);
                    let end = self.pos + len;
                    let chunk = std::str::from_utf8(&self.bytes[self.pos..end])
                        .map_err(|_| format!("invalid utf-8 at byte {}", self.pos))?;
                    out.push_str(chunk);
                    self.pos = end;
                }
            }
        }
    }

    fn parse_hex4(&mut self) -> Result<u16, String> {
        let mut value: u16 = 0;
        for _ in 0..4 {
            let b = self.peek()?;
            self.pos += 1;
            let digit = match b {
                b'0'..=b'9' => b - b'0',
                b'a'..=b'f' => b - b'a' + 10,
                b'A'..=b'F' => b - b'A' + 10,
                _ => return Err(format!("invalid hex digit at byte {}", self.pos - 1)),
            };
            value = value * 16 + digit as u16;
        }
        Ok(value)
    }

    fn parse_obj(&mut self, depth: usize) -> Result<Value, String> {
        self.pos += 1; // '{'
        let mut pairs = Vec::new();
        self.skip_ws();
        if self.peek()? == b'}' {
            self.pos += 1;
            return Ok(Value::Obj(pairs));
        }
        loop {
            self.skip_ws();
            if self.peek()? != b'"' {
                return Err(format!("object key must be a string at byte {}", self.pos));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek()? != b':' {
                return Err(format!("expected ':' at byte {}", self.pos));
            }
            self.pos += 1;
            self.skip_ws();
            let value = self.parse_value(depth + 1)?;
            pairs.push((key, value));
            self.skip_ws();
            match self.peek()? {
                b',' => self.pos += 1,
                b'}' => {
                    self.pos += 1;
                    return Ok(Value::Obj(pairs));
                }
                _ => return Err(format!("expected ',' or '}}' at byte {}", self.pos)),
            }
        }
    }

    fn parse_arr(&mut self, depth: usize) -> Result<Value, String> {
        self.pos += 1; // '['
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek()? == b']' {
            self.pos += 1;
            return Ok(Value::Arr(items));
        }
        loop {
            self.skip_ws();
            items.push(self.parse_value(depth + 1)?);
            self.skip_ws();
            match self.peek()? {
                b',' => self.pos += 1,
                b']' => {
                    self.pos += 1;
                    return Ok(Value::Arr(items));
                }
                _ => return Err(format!("expected ',' or ']' at byte {}", self.pos)),
            }
        }
    }
}

fn utf8_len(lead: u8) -> usize {
    match lead {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_protocol_shapes() {
        let frame = obj(vec![
            ("op", Value::Str("generate".into())),
            ("max_tokens", Value::Num(4.0)),
            ("temperature", Value::Num(0.5)),
            ("stop", Value::Arr(vec![Value::Str("\n".into())])),
            ("done", Value::Bool(false)),
            ("nothing", Value::Null),
        ]);
        let text = stringify(&frame);
        let back = parse(&text).expect("own output parses");
        assert_eq!(back, frame);
        assert_eq!(back.get("op").and_then(Value::as_str), Some("generate"));
        assert_eq!(back.get("max_tokens").and_then(Value::as_usize), Some(4));
    }

    #[test]
    fn escapes_and_unicode() {
        let v = parse(r#"{"s":"a\"b\\c\ndé😀"}"#).expect("parses");
        assert_eq!(
            v.get("s").and_then(Value::as_str),
            Some("a\"b\\c\nd\u{e9}\u{1f600}")
        );
        // The serializer escapes what JSON requires and leaves the
        // rest as UTF-8; the result parses back to the same string.
        let text = stringify(&v);
        let back = parse(&text).expect("reserializes");
        assert_eq!(back, v);
    }

    #[test]
    fn rejects_garbage_with_offsets() {
        for bad in ["", "{", "{\"a\":}", "{\"a\":1} extra", "[1,]", "\"open"] {
            assert!(parse(bad).is_err(), "expected error for {bad:?}");
        }
    }

    #[test]
    fn numbers_keep_integer_shape() {
        assert_eq!(stringify(&Value::Num(13796.0)), "13796");
        assert_eq!(stringify(&Value::Num(0.5)), "0.5");
        assert_eq!(parse("1e3").expect("parses"), Value::Num(1000.0));
    }

    #[test]
    fn deep_nesting_is_a_named_error_not_a_stack_overflow() {
        // The adjudication's executed attack: 20,000 nested arrays,
        // about 40 KB of input, aborted the engine process with a
        // stack overflow before the depth cap. It must now be an
        // ordinary parse error.
        let attack = format!("{}{}", "[".repeat(20_000), "]".repeat(20_000));
        let err = parse(&attack).expect_err("the attack input must not parse");
        assert!(err.contains("nesting deeper"), "error: {err}");
        // Objects nest through the same counter.
        let obj_attack = format!("{}1{}", "{\"a\":".repeat(500), "}".repeat(500));
        assert!(parse(&obj_attack).is_err());
    }

    #[test]
    fn nesting_within_the_limit_still_parses() {
        let deep = format!("{}1{}", "[".repeat(100), "]".repeat(100));
        assert!(parse(&deep).is_ok(), "100 levels is legitimate depth");
    }
}
