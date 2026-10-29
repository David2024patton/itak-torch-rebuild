//! Byte-level BPE tokenizer for the iTaK Torch rebuild (TODO T-2.3).
//!
//! This crate exists because of two audited defects in the original Torch,
//! which it is built to never repeat:
//!
//! 1. The original native engine "tokenized" by mapping each character to
//!    `char % vocab_size`. There is no character arithmetic here: text is
//!    encoded by a real byte-level BPE over the model's own vocabulary and
//!    merge table.
//! 2. The original Go platform tokenizer returned raw GGUF token strings
//!    on decode, so a space came back as the literal character `Ġ` and
//!    byte tokens were never reassembled. Here, decode always runs the
//!    byte-level inverse map (`Ġ` becomes a real space) and `<0xXX>` byte
//!    tokens are folded back into the byte stream before UTF-8 decoding,
//!    so a multibyte character split across tokens decodes correctly.
//!
//! # Input
//!
//! The tokenizer is built from already-parsed GGUF metadata, passed in as
//! plain vectors via [`TokenizerData`]: `tokenizer.ggml.tokens`,
//! `tokenizer.ggml.token_type`, `tokenizer.ggml.merges`, and the special
//! token ids (`bos`, `eos`, `unknown`, `padding`). The crate deliberately
//! does not depend on the GGUF parser crate, so the two stay decoupled.
//!
//! # Pre-tokenizer
//!
//! Before BPE, text is split by a hand-written scanner implementing this
//! llama4 / GPT-2 family pattern (the llama.cpp `llama4` pre-tokenizer is
//! the GPT-2 pattern with a 1-to-3 digit number rule):
//!
//! ```text
//! (?i:'s|'t|'re|'ve|'m|'ll|'d)
//! | [^\r\n\p{L}\p{N}]?\p{L}++
//! | \p{N}{1,3}
//! |  ?[^\s\p{L}\p{N}]++[\r\n]*
//! | \s*[\r\n]++
//! | \s++(?!\S)
//! | \s++
//! ```
//!
//! Deviations from a true regex engine, stated plainly:
//!
//! - `\p{L}` is approximated by [`char::is_alphabetic`] (the Unicode
//!   Alphabetic property: letters plus letter numbers and other
//!   alphabetic characters such as some combining marks).
//! - `\p{N}` is [`char::is_numeric`] (Nd, Nl, No), which matches `\p{N}`.
//! - `\s` is [`char::is_whitespace`] (the Unicode White_Space property).
//! - Contractions are matched ASCII case-insensitively; the pattern's
//!   `(?i)` is Unicode-aware in principle.
//! - A whitespace run containing newlines is emitted up to the end of
//!   its last newline, leaving any following spaces to the plain
//!   whitespace rules; a regex engine's backtracking can split such a
//!   run at a slightly different point in rare multi-newline cases.
//!   Splitting is always lossless: the pieces concatenate to the input.
//!
//! # Example
//!
//! ```
//! use tokenizer::{Tokenizer, TokenizerData};
//!
//! // A two-token vocabulary: "a" and " a" (stored byte-mapped as "Ġa").
//! let data = TokenizerData {
//!     tokens: vec!["a".into(), "Ġ".into(), "Ġa".into()],
//!     merges: vec!["Ġ a".into()],
//!     ..TokenizerData::default()
//! };
//! let tok = Tokenizer::new(data).unwrap();
//! let ids = tok.encode("a a").unwrap();
//! assert_eq!(ids, vec![0, 2]);
//! assert_eq!(tok.decode(&ids).unwrap(), "a a");
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod byte_level;
mod data;
mod diag;
mod error;
mod pretokenize;
mod tokenizer;

pub use byte_level::{byte_to_char, char_to_byte};
pub use data::{TokenType, TokenizerData};
pub use error::Error;
pub use pretokenize::pretokenize;
pub use tokenizer::Tokenizer;

/// Convenience alias for results returned by this crate.
pub type Result<T> = std::result::Result<T, Error>;
