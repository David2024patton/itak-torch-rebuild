//! Incremental detokenization for streamed token events.
//!
//! Token bytes arrive one token at a time. A token can end in the
//! middle of a multi-byte UTF-8 character, and a stop string can
//! span two tokens, so emission is a state machine, not a per-token
//! decode:
//!
//! - Bytes accumulate until they form valid UTF-8; only complete
//!   characters are decoded (an invalid byte, which the tokenizer
//!   never produces from real vocabularies, is replaced lossily
//!   rather than stalling the stream).
//! - Decoded text accumulates in full; a stop string found anywhere
//!   in it ends the generation with the text truncated at the stop
//!   string's start.
//! - Before emitting, the longest suffix of the un-emitted text that
//!   is a proper prefix of some stop string is held back, so a stop
//!   string is never half-emitted before it completes.
//!
//! [`Streamer::finish`] flushes whatever is held when the generation
//! ends for another reason (EOS, length), so no text is lost.

/// Streaming text assembler for one generation.
pub struct Streamer {
    stops: Vec<String>,
    /// Bytes not yet decodable as UTF-8.
    pending: Vec<u8>,
    /// All decoded text so far.
    text: String,
    /// Bytes of `text` already handed to the caller.
    emitted: usize,
    /// Set once a stop string matched; the generation is over.
    stopped: bool,
}

impl Streamer {
    /// A streamer with the request's stop strings. Empty stop
    /// strings are dropped (an empty stop would match everywhere).
    pub fn new(stops: Vec<String>) -> Self {
        Streamer {
            stops: stops.into_iter().filter(|s| !s.is_empty()).collect(),
            pending: Vec::new(),
            text: String::new(),
            emitted: 0,
            stopped: false,
        }
    }

    /// Whether a stop string has matched.
    pub fn stopped(&self) -> bool {
        self.stopped
    }

    /// The full decoded text so far, truncated at a matched stop
    /// string. This is the completion text the events add up to.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Feed one token's raw bytes. Returns the text chunks to emit
    /// now (usually zero or one). After a stop match, further feeds
    /// are ignored and return nothing.
    pub fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        if self.stopped {
            return Vec::new();
        }
        self.pending.extend_from_slice(bytes);
        self.decode_pending();
        self.check_stop_and_emit()
    }

    /// Flush everything decodable at the end of a generation that
    /// finished for a reason other than a stop string.
    pub fn finish(&mut self) -> Vec<String> {
        if self.stopped {
            return Vec::new();
        }
        if !self.pending.is_empty() {
            // A truncated final character: emit it lossily rather
            // than dropping bytes the model produced.
            let rest = String::from_utf8_lossy(&self.pending).into_owned();
            self.pending.clear();
            self.text.push_str(&rest);
        }
        let tail = self.text[self.emitted..].to_string();
        self.emitted = self.text.len();
        if tail.is_empty() {
            Vec::new()
        } else {
            vec![tail]
        }
    }

    fn decode_pending(&mut self) {
        loop {
            if self.pending.is_empty() {
                return;
            }
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    let s = s.to_string();
                    self.pending.clear();
                    self.text.push_str(&s);
                    return;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    match e.error_len() {
                        None => {
                            // Incomplete trailing character: keep
                            // its bytes pending, emit the valid part.
                            if valid > 0 {
                                let s =
                                    String::from_utf8_lossy(&self.pending[..valid]).into_owned();
                                self.text.push_str(&s);
                                self.pending.drain(..valid);
                            }
                            return;
                        }
                        Some(bad) => {
                            // A byte that can never complete: lossy
                            // replacement for the span, then continue
                            // with the rest.
                            let end = valid + bad;
                            let s = String::from_utf8_lossy(&self.pending[..end]).into_owned();
                            self.text.push_str(&s);
                            self.pending.drain(..end);
                        }
                    }
                }
            }
        }
    }

    fn check_stop_and_emit(&mut self) -> Vec<String> {
        // A stop occurrence always starts at or after `emitted`:
        // hold-back guarantees an earlier start would have been
        // caught before its first byte was emitted.
        let mut stop_at: Option<usize> = None;
        for stop in &self.stops {
            if let Some(idx) = self.text[self.emitted..].find(stop.as_str()) {
                let at = self.emitted + idx;
                stop_at = Some(stop_at.map_or(at, |prev: usize| prev.min(at)));
            }
        }
        if let Some(at) = stop_at {
            let chunk = self.text[self.emitted..at].to_string();
            self.text.truncate(at);
            self.emitted = at;
            self.stopped = true;
            return if chunk.is_empty() {
                Vec::new()
            } else {
                vec![chunk]
            };
        }
        let hold = self.holdback_len();
        let end = self.text.len() - hold;
        if end > self.emitted {
            let chunk = self.text[self.emitted..end].to_string();
            self.emitted = end;
            vec![chunk]
        } else {
            Vec::new()
        }
    }

    /// Length of the longest suffix of the un-emitted text that is a
    /// proper prefix of some stop string.
    fn holdback_len(&self) -> usize {
        let tail = &self.text[self.emitted..];
        let mut best = 0;
        for stop in &self.stops {
            // Proper prefixes only: a full-length match is handled
            // by the stop check, not by holding back.
            let max_k = stop.len().min(tail.len());
            for k in (1..max_k).rev() {
                if stop.is_char_boundary(k)
                    && tail.is_char_boundary(tail.len() - k)
                    && tail.ends_with(&stop[..k])
                {
                    best = best.max(k);
                    break;
                }
            }
            // Also consider k == tail length when the whole tail is
            // a proper prefix of the stop string.
            if tail.len() < stop.len() && stop.starts_with(tail) && !tail.is_empty() {
                best = best.max(tail.len());
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(streamer: &mut Streamer, feeds: &[&[u8]]) -> String {
        let mut out = String::new();
        for feed in feeds {
            for chunk in streamer.push(feed) {
                out.push_str(&chunk);
            }
        }
        for chunk in streamer.finish() {
            out.push_str(&chunk);
        }
        out
    }

    #[test]
    fn plain_ascii_streams_through() {
        let mut s = Streamer::new(vec![]);
        let out = collect(&mut s, &[b" Paris", b". It", b" is"]);
        assert_eq!(out, " Paris. It is");
        assert_eq!(s.text(), " Paris. It is");
    }

    #[test]
    fn multibyte_character_split_across_tokens() {
        // U+00E9 (e-acute) is 0xC3 0xA9: fed one byte per token.
        let mut s = Streamer::new(vec![]);
        assert!(s.push(&[0xc3]).is_empty());
        let out = collect(&mut s, &[&[0xa9]]);
        assert_eq!(out, "\u{e9}");
    }

    #[test]
    fn stop_string_split_across_tokens() {
        let mut s = Streamer::new(vec!["STOP".to_string()]);
        let mut out = String::new();
        for chunk in s.push(b"abc ST") {
            out.push_str(&chunk);
        }
        // "ST" is a stop prefix: only "abc " may be out so far.
        assert_eq!(out, "abc ");
        assert!(!s.stopped());
        for chunk in s.push(b"OP and more") {
            out.push_str(&chunk);
        }
        assert!(s.stopped());
        assert_eq!(out, "abc ");
        assert_eq!(s.text(), "abc ");
    }

    #[test]
    fn stop_at_token_boundary_and_empty_stop_ignored() {
        let mut s = Streamer::new(vec![String::new(), "END".to_string()]);
        let out = collect(&mut s, &[b"hello ", b"END", b"trailing"]);
        assert_eq!(out, "hello ");
        assert!(s.stopped());
    }

    #[test]
    fn held_back_prefix_flushes_on_finish() {
        let mut s = Streamer::new(vec!["STOP".to_string()]);
        let mut out = String::new();
        for chunk in s.push(b"abc ST") {
            out.push_str(&chunk);
        }
        assert_eq!(out, "abc ");
        for chunk in s.finish() {
            out.push_str(&chunk);
        }
        assert_eq!(out, "abc ST");
    }
}
