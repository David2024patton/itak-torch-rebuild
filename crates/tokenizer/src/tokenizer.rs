//! The tokenizer itself: construction from [`TokenizerData`], byte-level
//! BPE encoding, and byte-faithful decoding.

use std::collections::HashMap;

use crate::byte_level::{byte_token_name, parse_byte_token};
use crate::data::{TokenType, TokenizerData};
use crate::diag::dbglog;
use crate::error::Error;
use crate::pretokenize::pretokenize;
use crate::Result;

/// A byte-level BPE tokenizer over a fixed vocabulary and merge table.
///
/// Construct with [`Tokenizer::new`]; construction validates the data
/// and fails loudly on anything malformed rather than letting a bad
/// vocabulary produce silent garbage later.
#[derive(Debug)]
pub struct Tokenizer {
    tokens: Vec<String>,
    token_types: Vec<TokenType>,
    /// Whether token types were supplied at all. When they were not,
    /// byte tokens can only be recognized by their `<0xXX>` shape.
    types_supplied: bool,
    vocab: HashMap<String, u32>,
    merge_ranks: HashMap<(String, String), u32>,
    /// Special token (text, id) pairs, longest text first so the
    /// longest match wins at any position.
    specials: Vec<(String, u32)>,
    bos: Option<u32>,
    eos: Option<u32>,
    unk: Option<u32>,
    pad: Option<u32>,
}

impl Tokenizer {
    /// Builds a tokenizer from parsed GGUF tokenizer metadata.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EmptyVocab`], [`Error::DuplicateToken`],
    /// [`Error::TokenTypeLengthMismatch`], [`Error::InvalidMerge`], or
    /// [`Error::SpecialTokenOutOfRange`] when the data is malformed.
    pub fn new(data: TokenizerData) -> Result<Tokenizer> {
        if data.tokens.is_empty() {
            dbglog!("refusing empty vocabulary");
            return Err(Error::EmptyVocab);
        }
        if !data.token_types.is_empty() && data.token_types.len() != data.tokens.len() {
            return Err(Error::TokenTypeLengthMismatch {
                tokens: data.tokens.len(),
                types: data.token_types.len(),
            });
        }
        let types_supplied = !data.token_types.is_empty();
        let token_types: Vec<TokenType> = if types_supplied {
            data.token_types
        } else {
            vec![TokenType::Undefined; data.tokens.len()]
        };

        let mut vocab: HashMap<String, u32> = HashMap::with_capacity(data.tokens.len());
        for (id, token) in data.tokens.iter().enumerate() {
            if vocab.insert(token.clone(), id as u32).is_some() {
                return Err(Error::DuplicateToken(token.clone()));
            }
        }

        let mut merge_ranks: HashMap<(String, String), u32> =
            HashMap::with_capacity(data.merges.len());
        for (index, merge) in data.merges.iter().enumerate() {
            let parsed = merge.split_once(' ').and_then(|(a, b)| {
                if a.is_empty() || b.is_empty() || b.contains(' ') {
                    None
                } else {
                    Some((a.to_string(), b.to_string()))
                }
            });
            let Some(pair) = parsed else {
                return Err(Error::InvalidMerge {
                    index,
                    value: merge.clone(),
                });
            };
            merge_ranks.insert(pair, index as u32);
        }

        let vocab_size = data.tokens.len();
        let check_role = |role: &'static str, id: Option<u32>| -> Result<()> {
            if let Some(id) = id {
                if id as usize >= vocab_size {
                    return Err(Error::SpecialTokenOutOfRange {
                        role,
                        id,
                        vocab_size,
                    });
                }
            }
            Ok(())
        };
        check_role("bos", data.bos_token_id)?;
        check_role("eos", data.eos_token_id)?;
        check_role("unk", data.unknown_token_id)?;
        check_role("padding", data.padding_token_id)?;

        let mut specials: Vec<(String, u32)> = data
            .tokens
            .iter()
            .enumerate()
            .filter(|(id, text)| token_types[*id].is_special() && !text.is_empty())
            .map(|(id, text)| (text.clone(), id as u32))
            .collect();
        specials.sort_by_key(|a| std::cmp::Reverse(a.0.len()));

        dbglog!(
            "loaded vocab={} merges={} specials={}",
            vocab_size,
            data.merges.len(),
            specials.len()
        );
        Ok(Tokenizer {
            tokens: data.tokens,
            token_types,
            types_supplied,
            vocab,
            merge_ranks,
            specials,
            bos: data.bos_token_id,
            eos: data.eos_token_id,
            unk: data.unknown_token_id,
            pad: data.padding_token_id,
        })
    }

    /// The number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// The text of a token, verbatim as stored (byte-mapped).
    pub fn token_text(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    /// The id of a token by its stored text, if it exists.
    pub fn token_id(&self, text: &str) -> Option<u32> {
        self.vocab.get(text).copied()
    }

    /// The beginning-of-sequence token id, by role.
    pub fn bos_id(&self) -> Option<u32> {
        self.bos
    }

    /// The end-of-sequence token id, by role.
    pub fn eos_id(&self) -> Option<u32> {
        self.eos
    }

    /// The unknown token id, by role.
    pub fn unk_id(&self) -> Option<u32> {
        self.unk
    }

    /// The padding token id, by role.
    pub fn pad_id(&self) -> Option<u32> {
        self.pad
    }

    /// Encodes text to token ids.
    ///
    /// Special (control / user-defined) tokens present in the text are
    /// matched literally, longest first, and emitted as their own ids;
    /// the text between them goes through pre-tokenization and BPE.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnencodableSymbol`] when a symbol has no token,
    /// no byte-token fallback, and no unknown token is configured.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            match self.find_special(rest) {
                Some((byte_pos, id, byte_len)) => {
                    self.encode_plain(&rest[..byte_pos], &mut out)?;
                    out.push(id);
                    rest = &rest[byte_pos + byte_len..];
                }
                None => {
                    self.encode_plain(rest, &mut out)?;
                    break;
                }
            }
        }
        Ok(out)
    }

    /// Finds the earliest special token occurrence in `text`, breaking
    /// ties toward the longest match. Returns (byte position, id, byte
    /// length).
    fn find_special(&self, text: &str) -> Option<(usize, u32, usize)> {
        let mut best: Option<(usize, u32, usize)> = None;
        for (special, id) in &self.specials {
            if let Some(pos) = text.find(special.as_str()) {
                let better = match best {
                    None => true,
                    Some((bp, _, bl)) => pos < bp || (pos == bp && special.len() > bl),
                };
                if better {
                    best = Some((pos, *id, special.len()));
                }
            }
        }
        best
    }

    /// Encodes text known to contain no special tokens.
    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) -> Result<()> {
        for piece in pretokenize(text) {
            self.encode_piece(piece, out)?;
        }
        Ok(())
    }

    /// Byte-maps one pre-tokenizer piece and runs BPE over it.
    fn encode_piece(&self, piece: &str, out: &mut Vec<u32>) -> Result<()> {
        let mapped: String = piece.bytes().map(crate::byte_level::byte_to_char).collect();
        let mut symbols: Vec<String> = mapped.chars().map(|c| c.to_string()).collect();

        while symbols.len() > 1 {
            // Find the adjacent pair with the lowest merge rank.
            let mut best: Option<(u32, usize)> = None;
            for i in 0..symbols.len() - 1 {
                let key = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&key) {
                    if best.is_none_or(|(r, _)| rank < r) {
                        best = Some((rank, i));
                    }
                }
            }
            let Some((_, first)) = best else { break };
            let left = symbols[first].clone();
            let right = symbols[first + 1].clone();
            // Merge every occurrence of that pair in this pass.
            let mut merged: Vec<String> = Vec::with_capacity(symbols.len() - 1);
            let mut i = 0;
            while i < symbols.len() {
                if i + 1 < symbols.len() && symbols[i] == left && symbols[i + 1] == right {
                    merged.push(format!("{left}{right}"));
                    i += 2;
                } else {
                    merged.push(std::mem::take(&mut symbols[i]));
                    i += 1;
                }
            }
            symbols = merged;
        }

        for symbol in &symbols {
            out.push(self.symbol_id(symbol)?);
        }
        Ok(())
    }

    /// Resolves one BPE symbol to a token id: vocabulary first, then
    /// the `<0xXX>` byte-token fallback for single characters, then the
    /// unknown token, then a named error.
    fn symbol_id(&self, symbol: &str) -> Result<u32> {
        if let Some(&id) = self.vocab.get(symbol) {
            return Ok(id);
        }
        let mut chars = symbol.chars();
        if let (Some(ch), None) = (chars.next(), chars.next()) {
            if let Some(byte) = crate::byte_level::char_to_byte(ch) {
                if let Some(&id) = self.vocab.get(&byte_token_name(byte)) {
                    dbglog!("byte fallback for symbol {symbol:?} -> byte {byte:#04x}");
                    return Ok(id);
                }
            }
        }
        if let Some(unk) = self.unk {
            dbglog!("unk fallback for symbol {symbol:?}");
            return Ok(unk);
        }
        dbglog!("refusing unencodable symbol {symbol:?}");
        Err(Error::UnencodableSymbol(symbol.to_string()))
    }

    /// Decodes token ids into the raw byte stream they represent.
    ///
    /// Byte-mapped token text runs through the inverse byte map (so
    /// `Ġ` becomes a real space byte), `<0xXX>` byte tokens contribute
    /// their single byte, and control / user-defined tokens contribute
    /// their literal text. A multibyte UTF-8 character split across
    /// several tokens reassembles in this stream.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidTokenId`] for an id outside the
    /// vocabulary, or [`Error::MalformedByteToken`] when a token typed
    /// as a byte token does not have the `<0xXX>` shape.
    pub fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        for &id in ids {
            self.append_token_bytes(id, &mut out)?;
        }
        Ok(out)
    }

    fn append_token_bytes(&self, id: u32, out: &mut Vec<u8>) -> Result<()> {
        let text = self
            .tokens
            .get(id as usize)
            .ok_or(Error::InvalidTokenId(id))?;
        let ty = self.token_types[id as usize];
        match ty {
            TokenType::Byte => {
                let byte = parse_byte_token(text)
                    .ok_or_else(|| Error::MalformedByteToken(text.clone()))?;
                out.push(byte);
            }
            TokenType::Control | TokenType::UserDefined => {
                out.extend_from_slice(text.as_bytes());
            }
            _ => {
                // With no type array, a <0xXX>-shaped token can only be
                // a byte token; the byte-mapped alphabet can never
                // produce that literal shape as ordinary text.
                if !self.types_supplied {
                    if let Some(byte) = parse_byte_token(text) {
                        out.push(byte);
                        return Ok(());
                    }
                }
                for ch in text.chars() {
                    match crate::byte_level::char_to_byte(ch) {
                        Some(byte) => out.push(byte),
                        None => {
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Decodes token ids to a string, requiring the full byte stream
    /// to be valid UTF-8.
    ///
    /// # Errors
    ///
    /// Everything [`Tokenizer::decode_bytes`] can return, plus
    /// [`Error::InvalidUtf8`] when the byte stream is not valid UTF-8
    /// (for example when a caller decodes a partial stream that ends
    /// mid-character; use [`Tokenizer::decode_lossy`] or buffer the
    /// bytes in that case).
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let bytes = self.decode_bytes(ids)?;
        String::from_utf8(bytes).map_err(|e| Error::InvalidUtf8 {
            valid_up_to: e.utf8_error().valid_up_to(),
        })
    }

    /// Decodes token ids to a string, replacing invalid UTF-8
    /// sequences with the replacement character.
    ///
    /// # Errors
    ///
    /// Everything [`Tokenizer::decode_bytes`] can return.
    pub fn decode_lossy(&self, ids: &[u32]) -> Result<String> {
        Ok(String::from_utf8_lossy(&self.decode_bytes(ids)?).into_owned())
    }
}
