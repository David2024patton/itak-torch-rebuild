//! Tokenizers for the iTaK Torch rebuild (TODO T-2.3): byte-level
//! BPE and SentencePiece BPE, selected by the model's own metadata.
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
//! plain vectors via [`TokenizerData`]: `tokenizer.ggml.model`,
//! `tokenizer.ggml.tokens`, `tokenizer.ggml.token_type`,
//! `tokenizer.ggml.merges`, `tokenizer.ggml.scores`, and the special
//! token ids (`bos`, `eos`, `unknown`, `padding`). The crate deliberately
//! does not depend on the GGUF parser crate, so the two stay decoupled.
//!
//! # Families
//!
//! `tokenizer.ggml.model` selects the family. `"gpt2"` (or an absent
//! model key) is the byte-level BPE described below. `"llama"` is
//! SentencePiece BPE, implemented in `src/spm.rs` as a direct port of
//! llama.cpp's SPM path: spaces escape to `▁` (U+2581), one space is
//! prefixed at the start of the text and after every special token,
//! symbols start as single codepoints, and adjacent pairs merge one
//! at a time by the COMBINED token's score from
//! `tokenizer.ggml.scores` (highest first, ties to the leftmost pair;
//! no scores array means every score is 0.0, as in llama.cpp's vocab
//! load). The merges array, present in these files too, does not
//! drive SentencePiece encoding and is not even parsed for that
//! family. Anything that forms no token is re-segmented, down to
//! `<0xXX>` byte tokens for codepoints outside the vocabulary.
//! Decode unescapes `▁` back to a space; because encoding adds the
//! leading space, `decode` of an encoded string carries one leading
//! space the input did not have, which is the reference behavior
//! (llama.cpp strips it only in its whole-sequence detokenizer, one
//! level above per-token pieces). Any other model value is refused
//! by name at construction.
//!
//! # Pre-tokenizer
//!
//! Before BPE, text is split by a hand-written scanner implementing
//! the GPT-4o family pattern. That is the pattern llama.cpp assigns
//! to Glimmer's `tokenizer.ggml.pre = "llama4"` (its
//! `LLAMA_VOCAB_PRE_TYPE_GPT4O`); its word alternatives carry the
//! contraction as an optional suffix:
//!
//! ```text
//! [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'s|'t|'re|'ve|'m|'ll|'d)?
//! | [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'s|'t|'re|'ve|'m|'ll|'d)?
//! | \p{N}{1,3}
//! |  ?[^\s\p{L}\p{N}]+[\r\n/]*
//! | \s*[\r\n]+
//! | \s+(?!\S)
//! | \s+
//! ```
//!
//! The two word alternatives differ only in case structure; both
//! span the same maximal letter run, so the scanner implements them
//! as one letter run plus the optional contraction suffix. There is
//! no standalone contraction alternative: a contraction that does
//! not follow a word (the `'ve` in `you'd've`, the `'s` in `90's`)
//! is picked up by the prefix-char word branch instead. An earlier
//! version of this crate implemented the llama-3 pattern, in which
//! contractions are a standalone first alternative; that split
//! `don't` into `don` + `'t` and failed conformance against the
//! reference (see `CONFORMANCE.md` beside this crate).
//!
//! Deviations from a true regex engine, stated plainly:
//!
//! - `\p{L}` is approximated by [`char::is_alphabetic`] (the Unicode
//!   Alphabetic property: letters plus letter numbers and other
//!   alphabetic characters such as some combining marks).
//! - `\p{N}` is [`char::is_numeric`] (Nd, Nl, No), which matches `\p{N}`.
//! - `\s` is [`char::is_whitespace`] (the Unicode White_Space property).
//! - Contractions are matched ASCII case-insensitively, as in
//!   llama.cpp's implemented form of the pattern (explicit
//!   `[sS]`-style letter classes); the original pattern's `(?i)` is
//!   Unicode-aware in principle.
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
//!
//! # Conformance
//!
//! The token-for-token proof against the reference tokenizer on
//! Glimmer's real vocabulary (TODO T-2.3) lives in `CONFORMANCE.md`
//! beside this crate; the harness is `examples/conformance.rs` and
//! the reference data sits in `conformance/`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod byte_level;
mod data;
mod diag;
mod error;
mod pretokenize;
mod spm;
mod tokenizer;

pub use byte_level::{byte_to_char, char_to_byte};
pub use data::{TokenType, TokenizerData, TokenizerFamily};
pub use error::Error;
pub use pretokenize::pretokenize;
pub use tokenizer::Tokenizer;

/// Convenience alias for results returned by this crate.
pub type Result<T> = std::result::Result<T, Error>;
