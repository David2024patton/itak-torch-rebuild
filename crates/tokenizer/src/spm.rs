//! SentencePiece BPE: the `"llama"` tokenizer family.
//!
//! This is a direct port of llama.cpp's SPM path. The reference is
//! the llama.cpp checkout on David's server (0.7.0-dev, build 137,
//! commit a7af149, the build that produced the golden fixtures);
//! every rule below names its source so a future reader can re-check
//! it against `src/llama-vocab.cpp` instead of trusting this file.
//!
//! The pipeline, top to bottom:
//!
//! 1. `llama_vocab::impl::tokenize`, `LLAMA_VOCAB_TYPE_SPM` case:
//!    for each raw-text fragment, if `add_space_prefix` holds and the
//!    previous fragment was a special token (or this is the start),
//!    one space is prepended; then `llama_escape_whitespace` replaces
//!    every space with `▁` (U+2581). The type defaults for SPM set
//!    `add_space_prefix = true`, `add_bos = true`, `add_eos = false`
//!    (the `type == LLAMA_VOCAB_TYPE_SPM` branch of the vocab load);
//!    a file can override the prefix via `tokenizer.ggml.add_space_prefix`,
//!    which this crate does not plumb yet, so the type default is used.
//!    BOS/EOS injection is the caller's job in this crate, as with the
//!    byte-level family.
//! 2. `llm_tokenizer_spm_session::tokenize`: the escaped text is
//!    split into symbols, one per UTF-8 codepoint, with the length
//!    taken from `unicode_len_utf8` (`src/unicode.cpp`: a lookup on
//!    the lead byte's high nibble, clamped to the remaining bytes).
//!    Adjacent symbol pairs whose concatenation is a vocabulary token
//!    are pushed onto a priority queue keyed by the COMBINED token's
//!    score (`llm_bigram_spm::comparator`: highest score first, ties
//!    to the smallest left index). The queue pops one merge at a
//!    time; stale entries whose symbols have since merged are
//!    skipped by an exact size check, and each merge re-offers its
//!    two new neighbor pairs.
//! 3. Scores come from `tokenizer.ggml.scores` verbatim. When the
//!    file records no scores array, the vocab load in llama.cpp sets
//!    every token's score to 0.0, which reduces the queue to
//!    leftmost-first merging; this crate mirrors that (the caller
//!    passes a zero-filled slice).
//! 4. `resegment`: a final symbol whose text is a token emits it.
//!    One that is not is split back through `rev_merge` (the last
//!    symbol pair recorded for that exact text by `try_add_bigram`),
//!    and a text with no recorded split is emitted byte by byte via
//!    `byte_to_token`: the `<0xXX>` token, falling back to the single
//!    byte as a one-character token. (With merges only ever forming
//!    vocabulary tokens, the `rev_merge` split is a path ordinary
//!    vocabularies do not reach; it is ported anyway, because a
//!    faithful port beats a plausible simplification.)
//!
//! Decode is the `LLAMA_VOCAB_TYPE_SPM` case of `token_to_piece`:
//! normal tokens are unescaped (`▁` becomes a space), byte tokens
//! contribute their byte, control / user-defined / unknown tokens
//! render literally, and tokens with no recorded type contribute
//! nothing.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::byte_level::byte_token_name;
use crate::diag::dbglog;
use crate::error::Error;
use crate::Result;

/// The SentencePiece word-boundary marker, U+2581.
const SPM_SPACE: char = '\u{2581}';

/// `llama_escape_whitespace`: every space becomes `▁`.
pub(crate) fn escape_whitespace(text: &str) -> String {
    text.replace(' ', &SPM_SPACE.to_string())
}

/// `llama_unescape_whitespace`: every `▁` becomes a space.
pub(crate) fn unescape_whitespace(text: &str) -> String {
    text.replace(SPM_SPACE, " ")
}

/// `unicode_len_utf8` from llama.cpp's `src/unicode.cpp`: the length
/// is a lookup on the lead byte's high nibble. Continuation bytes
/// (0x80..=0xBF) report 1, so invalid sequences degrade to single
/// bytes rather than swallowing their neighbors.
fn utf8_len(lead: u8) -> usize {
    const LOOKUP: [usize; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4];
    LOOKUP[(lead >> 4) as usize]
}

/// One symbol: a byte range of the escaped text plus linked-list
/// links, mirroring `llm_symbol`. `n == 0` marks a symbol that has
/// been merged into its left neighbor.
#[derive(Debug, Clone, Copy)]
struct Symbol {
    /// Byte offset of the symbol's first byte in the escaped text.
    start: usize,
    /// Byte length; grows as right neighbors merge into this symbol.
    n: usize,
    /// Index of the previous live symbol, or -1.
    prev: i32,
    /// Index of the next live symbol, or -1.
    next: i32,
}

/// A merge candidate, mirroring `llm_bigram_spm`.
#[derive(Debug, Clone, Copy)]
struct Bigram {
    /// Index of the left symbol.
    left: usize,
    /// Index of the right symbol.
    right: usize,
    /// Score of the combined token.
    score: f32,
    /// Combined byte length when queued; a pop whose symbols no
    /// longer add up to this is stale and skipped.
    size: usize,
}

impl PartialEq for Bigram {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Bigram {}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Bigram {
    /// The heap pops the "greatest" element; llama.cpp's comparator
    /// makes that the highest score, ties broken toward the smallest
    /// left index.
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.left.cmp(&self.left))
    }
}

/// The SPM merge session over one escaped text, mirroring
/// `llm_tokenizer_spm_session`.
struct Session<'a> {
    /// The escaped text the symbols index into.
    text: &'a str,
    /// Token text to id, the crate's shared vocabulary map.
    vocab: &'a HashMap<String, u32>,
    /// Per-token scores, parallel to the vocabulary (zero-filled by
    /// the caller when the file recorded none, as llama.cpp does).
    scores: &'a [f32],
    symbols: Vec<Symbol>,
    work_queue: BinaryHeap<Bigram>,
    /// Combined text to the symbol pair last recorded for it.
    rev_merge: HashMap<String, (usize, usize)>,
}

/// Tokenizes one raw-text segment the SentencePiece way.
///
/// `prefix_space` mirrors llama.cpp's `add_space_prefix &&
/// is_prev_special`: one space is prepended before escaping, so the
/// segment's first word takes its `▁`-initial form.
///
/// # Errors
///
/// Returns [`Error::MissingByteToken`] when re-segmentation reaches
/// a byte the vocabulary carries no byte token for.
pub(crate) fn encode_segment(
    vocab: &HashMap<String, u32>,
    scores: &[f32],
    segment: &str,
    prefix_space: bool,
) -> Result<Vec<u32>> {
    let mut escaped = String::with_capacity(segment.len() + 3);
    if prefix_space {
        escaped.push(SPM_SPACE);
    }
    escaped.push_str(&escape_whitespace(segment));
    let mut session = Session {
        text: &escaped,
        vocab,
        scores,
        symbols: Vec::new(),
        work_queue: BinaryHeap::new(),
        rev_merge: HashMap::new(),
    };
    session.tokenize()
}

impl Session<'_> {
    fn tokenize(&mut self) -> Result<Vec<u32>> {
        // Split into one symbol per UTF-8 codepoint.
        let bytes = self.text.as_bytes();
        let mut index: i32 = 0;
        let mut offs = 0usize;
        while offs < bytes.len() {
            let len = utf8_len(bytes[offs]).min(bytes.len() - offs);
            self.symbols.push(Symbol {
                start: offs,
                n: len,
                prev: index - 1,
                next: if offs + len == bytes.len() {
                    -1
                } else {
                    index + 1
                },
            });
            offs += len;
            index += 1;
        }

        // Seed the work queue with all possible 2-symbol tokens.
        for i in 1..self.symbols.len() {
            self.try_add_bigram(i as i32 - 1, i as i32);
        }

        // Keep substituting the best pair for as long as we can.
        while let Some(bigram) = self.work_queue.pop() {
            let (left, right) = (bigram.left, bigram.right);
            let left_n = self.symbols[left].n;
            let right_n = self.symbols[right].n;
            // If one of the symbols already got merged, skip it.
            if left_n == 0 || right_n == 0 || left_n + right_n != bigram.size {
                continue;
            }
            // Merge the right symbol into the left one.
            let right_next = self.symbols[right].next;
            let left_prev = self.symbols[left].prev;
            self.symbols[left].n += right_n;
            self.symbols[right].n = 0;
            // Remove the right symbol from the chain.
            self.symbols[left].next = right_next;
            if right_next >= 0 {
                self.symbols[right_next as usize].prev = left as i32;
            }
            // Find more substitutions at the two new seams.
            self.try_add_bigram(left_prev, left as i32);
            self.try_add_bigram(left as i32, right_next);
        }

        let mut output = Vec::new();
        let mut i: i32 = if self.symbols.is_empty() { -1 } else { 0 };
        while i != -1 {
            let idx = i as usize;
            self.resegment(idx, &mut output)?;
            i = self.symbols[idx].next;
        }
        Ok(output)
    }

    /// Offers the pair (`left`, `right`) to the work queue when
    /// their concatenation is a vocabulary token.
    fn try_add_bigram(&mut self, left: i32, right: i32) {
        if left == -1 || right == -1 {
            return;
        }
        let (left, right) = (left as usize, right as usize);
        let text = self.pair_text(left, right);
        let Some(&token) = self.vocab.get(&text) else {
            return;
        };
        if token as usize >= self.scores.len() {
            return;
        }
        self.work_queue.push(Bigram {
            left,
            right,
            score: self.scores[token as usize],
            size: text.len(),
        });
        self.rev_merge.insert(text, (left, right));
    }

    /// The concatenated text of two chain-adjacent symbols, from
    /// their current byte ranges.
    fn pair_text(&self, left: usize, right: usize) -> String {
        let l = &self.symbols[left];
        let r = &self.symbols[right];
        let mut text = String::with_capacity(l.n + r.n);
        text.push_str(&self.text[l.start..l.start + l.n]);
        text.push_str(&self.text[r.start..r.start + r.n]);
        text
    }

    /// Emits the token(s) for one final symbol: the token itself
    /// when its text is in the vocabulary, a `rev_merge` split when
    /// one was recorded, and raw bytes otherwise. Iterative, like
    /// the original: merge chains can be deep.
    fn resegment(&self, root: usize, output: &mut Vec<u32>) -> Result<()> {
        let mut todo = vec![root];
        while let Some(idx) = todo.pop() {
            let sym = self.symbols[idx];
            let text = &self.text[sym.start..sym.start + sym.n];
            if let Some(&token) = self.vocab.get(text) {
                output.push(token);
                continue;
            }
            match self.rev_merge.get(text) {
                None => {
                    // Output any symbols that did not form tokens as bytes.
                    for &byte in text.as_bytes() {
                        output.push(self.byte_to_token(byte)?);
                    }
                }
                Some(&(left, right)) => {
                    todo.push(right);
                    todo.push(left);
                }
            }
        }
        Ok(())
    }

    /// `llama_vocab::byte_to_token` for the SPM type: the `<0xXX>`
    /// token, falling back to the byte as a one-character token.
    fn byte_to_token(&self, byte: u8) -> Result<u32> {
        if let Some(&id) = self.vocab.get(&byte_token_name(byte)) {
            return Ok(id);
        }
        if byte.is_ascii() {
            let single = (byte as char).to_string();
            if let Some(&id) = self.vocab.get(&single) {
                return Ok(id);
            }
        }
        dbglog!("refusing byte {byte:#04x}: no byte token in vocabulary");
        Err(Error::MissingByteToken(byte))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a vocabulary map and a parallel scores vector from
    /// (text, score) pairs in id order.
    fn vocab(entries: &[(&str, f32)]) -> (HashMap<String, u32>, Vec<f32>) {
        let map = entries
            .iter()
            .enumerate()
            .map(|(i, &(text, _))| (text.to_string(), i as u32))
            .collect();
        let scores = entries.iter().map(|&(_, score)| score).collect();
        (map, scores)
    }

    #[test]
    fn utf8_len_matches_the_high_nibble_lookup() {
        assert_eq!(utf8_len(0x41), 1); // 'A'
        assert_eq!(utf8_len(0x80), 1); // continuation byte: 1, not an error
        assert_eq!(utf8_len(0xC3), 2);
        assert_eq!(utf8_len(0xE2), 3);
        assert_eq!(utf8_len(0xF0), 4);
        assert_eq!(utf8_len(0xFF), 4);
    }

    #[test]
    fn escape_and_unescape_round_trip() {
        assert_eq!(escape_whitespace("a b  c"), "a\u{2581}b\u{2581}\u{2581}c");
        assert_eq!(unescape_whitespace("a\u{2581}b"), "a b");
    }

    #[test]
    fn highest_score_pair_merges_first_not_the_leftmost() {
        // "ab" scores 1.0, "bc" scores 2.0: llama.cpp merges "bc"
        // first, which consumes the b and blocks "ab". A leftmost
        // or rank-order merge would produce [▁, ab, c] instead.
        let (vocab, scores) = vocab(&[
            ("<unk>", 0.0),
            ("▁", -1.0),
            ("a", -1.0),
            ("b", -1.0),
            ("c", -1.0),
            ("ab", 1.0),
            ("bc", 2.0),
        ]);
        let ids = encode_segment(&vocab, &scores, "abc", true).unwrap();
        assert_eq!(ids, vec![1, 2, 6]); // ▁, a, bc
    }

    #[test]
    fn equal_scores_merge_leftmost_first() {
        // No scores: every token is 0.0 (llama.cpp's no-array
        // fallback), so the queue's tie break rules: the leftmost
        // pair merges first and consumes the shared symbol.
        let entries: Vec<(&str, f32)> = vec![
            ("<unk>", 0.0),
            ("▁", 0.0),
            ("a", 0.0),
            ("b", 0.0),
            ("c", 0.0),
            ("ab", 0.0),
            ("bc", 0.0),
        ];
        let (vocab, _) = vocab(&entries);
        let zeros = vec![0.0; entries.len()];
        let ids = encode_segment(&vocab, &zeros, "abc", true).unwrap();
        assert_eq!(ids, vec![1, 5, 4]); // ▁, ab, c
    }

    #[test]
    fn merges_chain_through_newly_formed_symbols() {
        // "▁a" forms first (score -0.5 beats "ab" at -3.0), then the
        // merged symbol combines with "b" into "▁ab" (-0.1).
        let (vocab, scores) = vocab(&[
            ("<unk>", 0.0),
            ("▁", -4.0),
            ("a", -1.0),
            ("b", -2.0),
            ("▁a", -0.5),
            ("ab", -3.0),
            ("▁ab", -0.1),
        ]);
        let ids = encode_segment(&vocab, &scores, "ab", true).unwrap();
        assert_eq!(ids, vec![6]); // ▁ab
    }

    #[test]
    fn prefix_space_selects_the_word_initial_piece() {
        let (vocab, scores) = vocab(&[("<unk>", 0.0), ("▁", -1.0), ("a", -1.0), ("▁a", -0.5)]);
        let with = encode_segment(&vocab, &scores, "a", true).unwrap();
        let without = encode_segment(&vocab, &scores, "a", false).unwrap();
        assert_eq!(with, vec![3]); // ▁a
        assert_eq!(without, vec![2]); // a
    }

    #[test]
    fn every_space_escapes_not_just_word_starts() {
        let (vocab, scores) = vocab(&[
            ("<unk>", 0.0),
            ("▁", -1.0),
            ("a", -1.0),
            ("b", -1.0),
            ("▁a", -0.5),
            ("▁b", -0.5),
        ]);
        let ids = encode_segment(&vocab, &scores, "a b", true).unwrap();
        assert_eq!(ids, vec![4, 5]); // ▁a, ▁b
    }

    #[test]
    fn unknown_codepoint_falls_back_to_byte_tokens() {
        // "é" (U+00E9, bytes C3 A9) is not in the vocabulary and
        // forms no pair, so re-segmentation emits its two bytes.
        let (vocab, scores) = vocab(&[
            ("<unk>", 0.0),
            ("▁", -1.0),
            ("<0xC3>", 0.0),
            ("<0xA9>", 0.0),
        ]);
        let ids = encode_segment(&vocab, &scores, "\u{e9}", true).unwrap();
        assert_eq!(ids, vec![1, 2, 3]); // ▁, <0xC3>, <0xA9>
    }

    #[test]
    fn missing_byte_token_is_a_named_error() {
        let (vocab, scores) = vocab(&[("<unk>", 0.0), ("▁", -1.0)]);
        let err = encode_segment(&vocab, &scores, "\u{e9}", true).unwrap_err();
        assert_eq!(err, Error::MissingByteToken(0xC3));
    }

    #[test]
    fn empty_segment_produces_no_tokens() {
        let (vocab, scores) = vocab(&[("<unk>", 0.0), ("▁", -1.0)]);
        // With the prefix, an empty segment is exactly the prefix
        // piece; without it, nothing at all.
        assert_eq!(encode_segment(&vocab, &scores, "", true).unwrap(), vec![1]);
        assert_eq!(
            encode_segment(&vocab, &scores, "", false).unwrap(),
            Vec::<u32>::new()
        );
    }

    #[test]
    fn invalid_utf8_bytes_degrade_to_single_byte_symbols() {
        // The escaped text is always valid UTF-8 in practice (it is
        // built from a &str), but the length table itself must treat
        // a continuation byte as length 1, as unicode.cpp does.
        assert_eq!(utf8_len(0xA9), 1);
    }
}
