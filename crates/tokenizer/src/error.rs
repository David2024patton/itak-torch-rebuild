//! Named error types for the tokenizer. Nothing in this crate panics on
//! malformed input; every refusal is one of these variants.

use std::fmt;

/// Errors returned by tokenizer construction, encoding, and decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The vocabulary passed to [`crate::Tokenizer::new`] was empty.
    EmptyVocab,
    /// The same token string appeared twice in the vocabulary.
    DuplicateToken(String),
    /// `token_types` was non-empty but its length differed from `tokens`.
    TokenTypeLengthMismatch {
        /// Length of the `tokens` vector.
        tokens: usize,
        /// Length of the `token_types` vector.
        types: usize,
    },
    /// A merge entry was not exactly two non-empty token strings
    /// separated by a single space.
    InvalidMerge {
        /// Index of the offending entry in the merges vector.
        index: usize,
        /// The offending entry, verbatim.
        value: String,
    },
    /// A special token id (bos/eos/unk/padding) pointed outside the
    /// vocabulary.
    SpecialTokenOutOfRange {
        /// Which role was being validated: "bos", "eos", "unk", "padding".
        role: &'static str,
        /// The offending id.
        id: u32,
        /// The vocabulary size it was checked against.
        vocab_size: usize,
    },
    /// A symbol produced by BPE had no vocabulary entry, no byte-token
    /// fallback, and no unknown token was configured.
    UnencodableSymbol(String),
    /// `decode` was given a token id outside the vocabulary.
    InvalidTokenId(u32),
    /// A token marked as a byte token did not have the `<0xXX>` shape.
    MalformedByteToken(String),
    /// The decoded byte stream was not valid UTF-8.
    InvalidUtf8 {
        /// Byte offset of the first invalid sequence, as reported by
        /// [`std::str::Utf8Error::valid_up_to`].
        valid_up_to: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::EmptyVocab => write!(f, "tokenizer vocabulary is empty"),
            Error::DuplicateToken(t) => write!(f, "duplicate token in vocabulary: {t:?}"),
            Error::TokenTypeLengthMismatch { tokens, types } => write!(
                f,
                "token_types length {types} does not match tokens length {tokens}"
            ),
            Error::InvalidMerge { index, value } => write!(
                f,
                "merge entry {index} is not two token strings separated by one space: {value:?}"
            ),
            Error::SpecialTokenOutOfRange {
                role,
                id,
                vocab_size,
            } => write!(
                f,
                "{role} token id {id} is outside the vocabulary of size {vocab_size}"
            ),
            Error::UnencodableSymbol(s) => {
                write!(f, "symbol has no token, byte fallback, or unk: {s:?}")
            }
            Error::InvalidTokenId(id) => write!(f, "token id {id} is outside the vocabulary"),
            Error::MalformedByteToken(s) => {
                write!(f, "byte token is not in <0xXX> form: {s:?}")
            }
            Error::InvalidUtf8 { valid_up_to } => {
                write!(
                    f,
                    "decoded bytes are not valid UTF-8 (valid up to byte {valid_up_to})"
                )
            }
        }
    }
}

impl std::error::Error for Error {}
