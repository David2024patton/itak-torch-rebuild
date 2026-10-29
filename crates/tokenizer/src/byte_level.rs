//! The GPT-2 byte-to-unicode map at the heart of byte-level BPE.
//!
//! Every byte 0..=255 is assigned a printable Unicode character so that
//! BPE operates on text with no invisible control characters. Bytes in
//! the ranges `!`..=`~`, `¡`..=`¬`, `®`..=`ÿ` map to the character with
//! the same codepoint; the remaining 68 bytes map to U+0100 and upward
//! in byte order. The space byte 0x20 maps to `Ġ` (U+0120), which is
//! why raw GGUF token strings show `Ġ` where a space belongs: decode
//! must run this map in reverse, which is exactly what the original Go
//! tokenizer failed to do.

use std::collections::HashMap;
use std::sync::OnceLock;

fn forward_table() -> &'static [char; 256] {
    static TABLE: OnceLock<[char; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        // Bytes that map to the character with the same codepoint.
        let mut direct: Vec<u32> = Vec::with_capacity(188);
        direct.extend(0x21..=0x7E);
        direct.extend(0xA1..=0xAC);
        direct.extend(0xAE..=0xFF);
        let mut table = ['\0'; 256];
        for &b in &direct {
            table[b as usize] = char::from_u32(b).expect("direct range is valid char space");
        }
        // The remaining bytes, in order, take codepoints 256, 257, ...
        let mut next: u32 = 256;
        for b in 0u32..=255 {
            if table[b as usize] == '\0' {
                table[b as usize] =
                    char::from_u32(next).expect("overflow range is valid char space");
                next += 1;
            }
        }
        table
    })
}

fn inverse_table() -> &'static HashMap<char, u8> {
    static TABLE: OnceLock<HashMap<char, u8>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut map = HashMap::with_capacity(256);
        for (byte, &ch) in forward_table().iter().enumerate() {
            map.insert(ch, byte as u8);
        }
        map
    })
}

/// Returns the printable character assigned to `byte` by the GPT-2
/// byte-level alphabet. Encoding maps every input byte through this
/// before BPE runs.
pub fn byte_to_char(byte: u8) -> char {
    forward_table()[byte as usize]
}

/// Returns the byte assigned to `ch`, or `None` if `ch` is not part of
/// the byte-level alphabet (for example a literal character from a
/// control token's text). Decoding maps token characters through this.
pub fn char_to_byte(ch: char) -> Option<u8> {
    inverse_table().get(&ch).copied()
}

/// Formats the GGUF byte-token name for a byte: `<0x0A>` and friends.
pub(crate) fn byte_token_name(byte: u8) -> String {
    format!("<0x{byte:02X}>")
}

/// Parses a `<0xXX>` byte-token name back to its byte, accepting upper
/// or lower case hex. Returns `None` for any other shape.
pub(crate) fn parse_byte_token(text: &str) -> Option<u8> {
    let bytes = text.as_bytes();
    if bytes.len() != 6
        || bytes[0] != b'<'
        || bytes[1] != b'0'
        || bytes[2] != b'x'
        || bytes[5] != b'>'
    {
        return None;
    }
    let hi = hex_value(bytes[3])?;
    let lo = hex_value(bytes[4])?;
    Some((hi << 4) | lo)
}

fn hex_value(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn space_maps_to_g_dot() {
        assert_eq!(byte_to_char(0x20), 'Ġ');
        assert_eq!(char_to_byte('Ġ'), Some(0x20));
    }

    #[test]
    fn map_is_a_bijection_over_all_bytes() {
        let mut seen = std::collections::HashSet::new();
        for b in 0u8..=255 {
            let ch = byte_to_char(b);
            assert!(seen.insert(ch), "char {ch:?} assigned twice");
            assert_eq!(char_to_byte(ch), Some(b));
        }
        assert_eq!(seen.len(), 256);
    }

    #[test]
    fn printable_ascii_maps_to_itself() {
        assert_eq!(byte_to_char(b'A'), 'A');
        assert_eq!(byte_to_char(b'~'), '~');
        assert_eq!(char_to_byte('z'), Some(b'z'));
    }

    #[test]
    fn byte_token_names_round_trip() {
        for b in [0x00u8, 0x0A, 0x20, 0xC3, 0xFF] {
            assert_eq!(parse_byte_token(&byte_token_name(b)), Some(b));
        }
        assert_eq!(parse_byte_token("<0xff>"), Some(0xFF));
        assert_eq!(parse_byte_token("<0xZZ>"), None);
        assert_eq!(parse_byte_token("hello"), None);
    }
}
