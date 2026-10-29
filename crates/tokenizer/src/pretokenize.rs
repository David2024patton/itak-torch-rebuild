//! The pre-tokenizer: a hand-written scanner implementing the
//! llama4 / GPT-2 split pattern documented in the crate root. BPE runs
//! per piece, never across piece boundaries.
//!
//! The split is lossless: concatenating the returned pieces reproduces
//! the input exactly. No piece is ever dropped, including trailing
//! whitespace.

/// Splits `text` into the pieces BPE will encode independently.
///
/// Pieces borrow from `text`. The pattern and its deviations are
/// documented in the crate-level documentation.
pub fn pretokenize(text: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let mut pieces = Vec::new();
    let mut i = 0usize;
    while i < n {
        let start = i;
        i = scan_piece(&chars, i);
        let end_byte = if i < n { chars[i].0 } else { text.len() };
        pieces.push(&text[chars[start].0..end_byte]);
    }
    pieces
}

fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}

fn is_number(c: char) -> bool {
    c.is_numeric()
}

/// A character the symbol branch consumes: not whitespace, not a
/// letter, not a number.
fn is_symbol(c: char) -> bool {
    !c.is_whitespace() && !is_letter(c) && !is_number(c)
}

/// If a contraction starts at `i` (chars[i] is `'`), returns its length
/// in chars including the apostrophe. Suffixes, longest first:
/// `re`, `ve`, `ll`, then `s`, `t`, `m`, `d`; ASCII case-insensitive.
fn contraction_len(chars: &[(usize, char)], i: usize) -> Option<usize> {
    let rest: String = chars
        .iter()
        .skip(i + 1)
        .take(2)
        .map(|&(_, c)| c.to_ascii_lowercase())
        .collect();
    for suffix in ["re", "ve", "ll"] {
        if rest == suffix {
            return Some(3);
        }
    }
    for suffix in ["s", "t", "m", "d"] {
        if rest.starts_with(suffix) {
            return Some(2);
        }
    }
    None
}

/// Consumes a run of letters starting at `i`; returns the end index.
fn letter_run(chars: &[(usize, char)], mut i: usize) -> usize {
    while i < chars.len() && is_letter(chars[i].1) {
        i += 1;
    }
    i
}

/// Consumes a symbol run starting at `i`, plus any `\r`/`\n` run that
/// immediately follows it (the pattern's `[\r\n]*` tail); returns the
/// end index.
fn symbol_run(chars: &[(usize, char)], mut i: usize) -> usize {
    while i < chars.len() && is_symbol(chars[i].1) {
        i += 1;
    }
    while i < chars.len() && matches!(chars[i].1, '\r' | '\n') {
        i += 1;
    }
    i
}

/// Scans one piece starting at char index `i` and returns the index
/// just past its end. Mirrors the alternation order of the pattern:
/// contraction, letter branch (with optional one-char prefix), number,
/// symbol (via the whitespace branch's single-space case), newline and
/// whitespace branches.
fn scan_piece(chars: &[(usize, char)], i: usize) -> usize {
    let n = chars.len();
    let c = chars[i].1;

    // 1. Contraction: 's 't 're 've 'm 'll 'd.
    if c == '\'' {
        if let Some(len) = contraction_len(chars, i) {
            return i + len;
        }
    }

    // 2. Letter branch, bare: \p{L}+
    if is_letter(c) {
        return letter_run(chars, i);
    }

    // 3. Letter branch with one prefix char ([^\r\n\p{L}\p{N}]? before
    //    letters). Covers " word" and "-word". Newlines are excluded
    //    from the prefix class by the pattern.
    if !is_number(c) && !matches!(c, '\r' | '\n') && i + 1 < n && is_letter(chars[i + 1].1) {
        return letter_run(chars, i + 1);
    }

    // 4. Number branch: \p{N}{1,3}.
    if is_number(c) {
        let mut j = i;
        while j < n && j - i < 3 && is_number(chars[j].1) {
            j += 1;
        }
        return j;
    }

    // 5. Whitespace branches.
    if c.is_whitespace() {
        return scan_whitespace(chars, i);
    }

    // 6. Symbol branch: [^\s\p{L}\p{N}]+ with an optional newline tail.
    symbol_run(chars, i)
}

/// The whitespace branches: `\s*[\r\n]+`, then `\s+(?!\S)`, then `\s+`.
fn scan_whitespace(chars: &[(usize, char)], i: usize) -> usize {
    let n = chars.len();
    // Maximal whitespace run starting at i: [i, run_end).
    let mut run_end = i;
    while run_end < n && chars[run_end].1.is_whitespace() {
        run_end += 1;
    }

    // Run contains a newline: \s*[\r\n]+ takes the run up to the end
    // of its last newline; spaces after that newline are rescanned.
    let last_newline = (i..run_end)
        .rev()
        .find(|&k| matches!(chars[k].1, '\r' | '\n'));
    if let Some(k) = last_newline {
        return k + 1;
    }

    // Run reaches the end of the text: \s+(?!\S) matches all of it.
    if run_end == n {
        return run_end;
    }

    // Run of 2+ spaces before non-whitespace: \s+(?!\S) backtracks by
    // one, leaving the final space to prefix the next piece.
    if run_end - i >= 2 {
        return run_end - 1;
    }

    // A single whitespace char before non-whitespace. It prefixes a
    // letter or symbol piece (the pattern's leading ` ?`); before a
    // digit it stands alone (the number branch has no prefix).
    let next = chars[run_end].1;
    if is_letter(next) {
        return letter_run(chars, run_end);
    }
    if is_symbol(next) {
        return symbol_run(chars, run_end);
    }
    run_end
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pieces_are_lossless() {
        let samples = [
            "",
            "Hello world",
            "  leading and trailing  ",
            "don't stop believin'",
            "abc1234567def",
            "line one\nline two\r\n\r\nend",
            "tabs\tand\ttabs",
            "a  b   c    d",
            "price: $3.50 (cheap!)",
            "123 4567 89",
            "café 中 text",
            "word- word -word",
            " \n \n ",
        ];
        for s in samples {
            assert_eq!(pretokenize(s).concat(), s, "split of {s:?} is not lossless");
        }
    }

    #[test]
    fn basic_splits_match_the_pattern() {
        assert_eq!(pretokenize("Hello world"), vec!["Hello", " world"]);
        assert_eq!(
            pretokenize("I don't know"),
            vec!["I", " don", "'t", " know"]
        );
        assert_eq!(pretokenize("abc1234567"), vec!["abc", "123", "456", "7"]);
        assert_eq!(pretokenize("a  b"), vec!["a", " ", " b"]);
        assert_eq!(pretokenize("hello "), vec!["hello", " "]);
        assert_eq!(pretokenize("one\ntwo"), vec!["one", "\n", "two"]);
        assert_eq!(pretokenize("123"), vec!["123"]);
        assert_eq!(pretokenize(" 123"), vec![" ", "123"]);
    }
}
