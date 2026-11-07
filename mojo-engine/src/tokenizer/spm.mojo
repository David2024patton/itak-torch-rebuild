# spm.mojo - SentencePiece BPE: the "llama" tokenizer family.
#
# Direct port of crates/tokenizer/src/spm.rs, which is itself a
# direct port of llama.cpp's SPM path (rules cited there to
# src/llama-vocab.cpp in llama.cpp 0.7.0-dev, commit a7af149):
#
# 1. The escaped text prefixes one space when the caller says
#    the previous fragment was special (or this is the start),
#    then every space becomes ▁ (U+2581).
# 2. Symbols start as single codepoints, lengths from the
#    unicode_len_utf8 high-nibble table. Adjacent pairs merge
#    one at a time by the COMBINED token's score from
#    tokenizer.ggml.scores (highest first, ties to the leftmost
#    pair); no scores array means every score is 0.0.
# 3. Re-segmentation: a final symbol whose text is a token emits
#    it; one that is not is split back through the recorded
#    rev_merge pair, and a text with no recorded split is emitted
#    byte by byte via <0xXX> byte tokens.
#
# Port note: the Rust side drives merges from a binary heap of
# queued bigrams. This port rescans the live symbol chain each
# round and merges the single best adjacent pair (highest
# score, ties to the smallest left index), which selects the
# same merge at every step: the candidate set at any moment is
# exactly the adjacent live pairs whose concatenation is a
# vocabulary token, and a pair's score does not change. The
# rev_merge table here records pair TEXTS (not symbol indices)
# at scan time; that table is only consulted for final symbols
# whose text is not a token, which ordinary vocabularies never
# produce (merges only ever form vocabulary tokens), as the
# Rust side notes too.

from std.collections import Dict

from tokenizer.byte_level import (
    append_utf8,
    byte_token_name,
    bytes_of_codepoints,
    string_from_codepoints,
)

comptime SPM_SPACE = 0x2581  # ▁


def escape_whitespace_cps(cps: List[Int]) -> List[Int]:
    """llama_escape_whitespace on codepoints: space -> ▁."""
    var out = List[Int]()
    for cp in cps:
        out.append(SPM_SPACE if cp == 0x20 else cp)
    return out^


def unescape_whitespace(text: String) -> String:
    """llama_unescape_whitespace: every ▁ becomes a space."""
    from tokenizer.byte_level import codepoints_of

    var cps = codepoints_of(text)
    var out = List[Int]()
    for cp in cps:
        out.append(0x20 if cp == SPM_SPACE else cp)
    return string_from_codepoints(out, 0, len(out))


def utf8_len(lead: Int) -> Int:
    """unicode_len_utf8 from llama.cpp's src/unicode.cpp: a
    lookup on the lead byte's high nibble ([1]*12 + [2,2,3,4]).
    Continuation bytes (0x80..=0xBF) report 1."""
    var nibble = (lead >> 4) & 0xF
    if nibble < 12:
        return 1
    if nibble == 12 or nibble == 13:
        return 2
    if nibble == 14:
        return 3
    return 4


def _pair_text(
    text_bytes: List[UInt8],
    sym_start: List[Int],
    sym_n: List[Int],
    left: Int,
    right: Int,
) raises -> String:
    """The concatenated text of two chain-adjacent symbols,
    from their current byte ranges."""
    from tokenizer.byte_level import decode_utf8_strict

    var both = List[UInt8]()
    for k in range(sym_n[left]):
        both.append(text_bytes[sym_start[left] + k])
    for k in range(sym_n[right]):
        both.append(text_bytes[sym_start[right] + k])
    var cps = decode_utf8_strict(both)
    return string_from_codepoints(cps, 0, len(cps))


def _slice_string(bytes: List[UInt8], start: Int, length: Int) raises -> String:
    """The String for bytes[start:start+length], which is a
    symbol slice of valid escaped text (always valid UTF-8)."""
    from tokenizer.byte_level import decode_utf8_strict

    var slice = List[UInt8]()
    for k in range(length):
        slice.append(bytes[start + k])
    var cps = decode_utf8_strict(slice)
    return string_from_codepoints(cps, 0, len(cps))


def _byte_to_token(vocab: Dict[String, Int], byte: Int) raises -> Int:
    """llama_vocab::byte_to_token for the SPM type: the <0xXX>
    token, falling back to the byte as a one-character token
    (ASCII only). Named refusal when neither exists."""
    from diag import log_debug

    var name = byte_token_name(byte)
    if name in vocab:
        return vocab[name]
    if byte < 0x80:
        var single = string_from_codepoints(List[Int]([byte]), 0, 1)
        if single in vocab:
            return vocab[single]
    _ = log_debug("tokenizer", "refusing byte: no byte token in vocabulary")
    raise Error(
        "tokenizer_missing_byte_token: vocabulary has no byte token for byte "
        + String(byte)
    )


def encode_segment(
    vocab: Dict[String, Int],
    scores: List[Float64],
    segment: List[Int],
    prefix_space: Bool,
) raises -> List[Int]:
    """Tokenizes one raw-text segment (as codepoints) the
    SentencePiece way. `prefix_space` mirrors llama.cpp's
    add_space_prefix && is_prev_special: one space is prepended
    before escaping, so the segment's first word takes its
    ▁-initial form."""
    # Escaped codepoints, then their UTF-8 bytes (symbols index
    # into the byte string, as in llama.cpp).
    var escaped = List[Int]()
    if prefix_space:
        escaped.append(SPM_SPACE)
    var seg_escaped = escape_whitespace_cps(segment)
    for cp in seg_escaped:
        escaped.append(cp)
    var text_bytes = bytes_of_codepoints(escaped)

    # Split into one symbol per UTF-8 codepoint.
    var sym_start = List[Int]()
    var sym_n = List[Int]()
    var sym_prev = List[Int]()
    var sym_next = List[Int]()
    var offs = 0
    var index = 0
    while offs < len(text_bytes):
        var ln = utf8_len(Int(text_bytes[offs]))
        var remain = len(text_bytes) - offs
        if ln > remain:
            ln = remain
        sym_start.append(offs)
        sym_n.append(ln)
        sym_prev.append(index - 1)
        sym_next.append(-1 if offs + ln == len(text_bytes) else index + 1)
        offs += ln
        index += 1

    var rev_left = Dict[String, String]()
    var rev_right = Dict[String, String]()

    # Merge loop: rescan the chain, merge the best pair, repeat
    # (see the module header for the equivalence argument).
    while True:
        var best_found = False
        var best_left = -1
        var best_right = -1
        var best_score = Float64(0.0)
        var cur = 0 if len(sym_n) > 0 else -1
        while cur != -1:
            var nxt = sym_next[cur]
            if nxt != -1:
                var text = _pair_text(text_bytes, sym_start, sym_n, cur, nxt)
                if text in vocab:
                    var tok = vocab[text]
                    if tok >= 0 and tok < len(scores):
                        # Offer: record the split for this text,
                        # as try_add_bigram does.
                        var lt = _slice_string(
                            text_bytes, sym_start[cur], sym_n[cur]
                        )
                        var rt = _slice_string(
                            text_bytes, sym_start[nxt], sym_n[nxt]
                        )
                        rev_left[text] = lt
                        rev_right[text] = rt
                        var score = scores[tok]
                        if not best_found or score > best_score:
                            best_found = True
                            best_left = cur
                            best_right = nxt
                            best_score = score
            cur = nxt
        if not best_found:
            break
        # Merge the right symbol into the left one.
        var right_next = sym_next[best_right]
        sym_n[best_left] = sym_n[best_left] + sym_n[best_right]
        sym_n[best_right] = 0
        sym_next[best_left] = right_next
        if right_next >= 0:
            sym_prev[right_next] = best_left

    # Re-segment the final symbols.
    var output = List[Int]()
    var stack = List[String]()
    var i = 0 if len(sym_n) > 0 else -1
    # Walk the chain, pushing symbol texts in reverse so pops
    # come out in chain order through the shared stack below.
    var chain_texts = List[String]()
    while i != -1:
        chain_texts.append(_slice_string(text_bytes, sym_start[i], sym_n[i]))
        i = sym_next[i]
    for ci in range(len(chain_texts)):
        stack.append(chain_texts[len(chain_texts) - 1 - ci])
    while len(stack) > 0:
        var text = stack.pop()
        if text in vocab:
            output.append(vocab[text])
            continue
        if text in rev_left:
            # Push right first so left is processed first.
            stack.append(rev_right[text])
            stack.append(rev_left[text])
            continue
        var tb = List[UInt8]()
        for b in text.as_bytes():
            tb.append(b)
        for b in tb:
            output.append(_byte_to_token(vocab, Int(b)))
    return output^
