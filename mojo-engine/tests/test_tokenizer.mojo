# test_tokenizer.mojo - the tokenizer's synthetic-vocabulary
# suite: byte alphabet, pre-tokenizer, byte-level BPE, and
# SentencePiece, mirroring the Rust crate's tests
# (crates/tokenizer/tests/roundtrip.rs, spm_family.rs, and the
# unit tests in byte_level.rs / pretokenize.rs / spm.rs) case for
# case where the Mojo port carries the same behavior. The
# real-vocabulary proof is tests/tokenizer_parity.mojo against
# the recorded reference (see crates/tokenizer/CONFORMANCE.md).

from std.testing import TestSuite, assert_true, assert_false, assert_equal

from tokenizer.byte_level import (
    byte_to_codepoint,
    byte_token_name,
    codepoint_to_byte,
    parse_byte_token,
)
from tokenizer.pretokenize import pretokenize_text
from tokenizer.tokenizer import (
    FAMILY_BYTE_LEVEL,
    FAMILY_SENTENCEPIECE,
    TT_BYTE,
    TT_CONTROL,
    TT_NORMAL,
    TT_UNKNOWN,
    Tokenizer,
    TokenizerData,
)

comptime TH_ID = 512
comptime THE_ID = 513
comptime G_THE_ID = 514
comptime HE_ID = 515
comptime G_A_ID = 516
comptime UNK_ID = 517
comptime EOS_ID = 518


def synthetic_data() -> TokenizerData:
    """The Rust roundtrip.rs synthetic vocabulary: ids 0..=255
    the byte-mapped character for each byte, 256..=511 the byte
    tokens, then the merge products, <unk>, and the control
    token (bos = eos = its id)."""
    var data = TokenizerData()
    for b in range(256):
        var cp = byte_to_codepoint(b)
        data.tokens.append(String(Codepoint.from_u32(UInt32(cp)).value()))
        data.token_types.append(TT_NORMAL)
    for b in range(256):
        data.tokens.append(byte_token_name(b))
        data.token_types.append(TT_BYTE)
    for merged in ["th", "the", "Ġthe", "he", "Ġa"]:
        data.tokens.append(merged)
        data.token_types.append(TT_NORMAL)
    data.tokens.append("<unk>")
    data.token_types.append(TT_UNKNOWN)
    data.tokens.append("<|end_of_text|>")
    data.token_types.append(TT_CONTROL)
    data.merges = ["t h", "th e", "Ġ the", "h e", "Ġ a"]
    data.bos_token_id = EOS_ID
    data.eos_token_id = EOS_ID
    data.unknown_token_id = UNK_ID
    return data^


def assert_ids(got: List[Int], want: List[Int], what: String) raises:
    assert_equal(
        len(got),
        len(want),
        what + ": token count " + String(len(got)) + " != " + String(len(want)),
    )
    for i in range(len(want)):
        assert_equal(got[i], want[i], what + ": token at index " + String(i))


# --- byte_level (Rust byte_level.rs unit tests) ---


def test_byte_map_bijection() raises:
    var seen = List[Int]()
    for b in range(256):
        var cp = byte_to_codepoint(b)
        assert_false(cp in seen, "codepoint assigned twice")
        seen.append(cp)
        assert_equal(codepoint_to_byte(cp), b)
    assert_equal(len(seen), 256)
    assert_equal(byte_to_codepoint(0x20), 0x120)  # space -> Ġ
    assert_equal(codepoint_to_byte(0x120), 0x20)
    assert_equal(byte_to_codepoint(0x41), 0x41)  # 'A' maps to itself
    assert_equal(codepoint_to_byte(0x7A), 0x7A)


def test_byte_token_names_round_trip() raises:
    for b in [0x00, 0x0A, 0x20, 0xC3, 0xFF]:
        assert_equal(parse_byte_token(byte_token_name(b)), b)
    assert_equal(parse_byte_token("<0xff>"), 0xFF)
    assert_equal(parse_byte_token("<0xZZ>"), -1)
    assert_equal(parse_byte_token("hello"), -1)


# --- pretokenize (Rust pretokenize.rs unit tests) ---


def assert_pieces(text: String, want: List[String]) raises:
    var got = pretokenize_text(text)
    assert_equal(len(got), len(want), "piece count for: " + text)
    for i in range(len(want)):
        assert_equal(got[i], want[i], "piece " + String(i) + " for: " + text)


def test_pretokenize_basic_splits() raises:
    assert_pieces("Hello world", ["Hello", " world"])
    assert_pieces("I don't know", ["I", " don't", " know"])
    assert_pieces("abc1234567", ["abc", "123", "456", "7"])
    assert_pieces("a  b", ["a", " ", " b"])
    assert_pieces("hello ", ["hello", " "])
    assert_pieces("one\ntwo", ["one", "\n", "two"])
    assert_pieces("123", ["123"])
    assert_pieces(" 123", [" ", "123"])


def test_pretokenize_contractions_are_suffixes() raises:
    assert_pieces("you'd've", ["you'd", "'ve"])
    assert_pieces("'tis the season", ["'tis", " the", " season"])
    assert_pieces("the 90's vibe", ["the", " ", "90", "'s", " vibe"])
    assert_pieces("dogs' bones", ["dogs", "'", " bones"])
    assert_pieces("DON'T", ["DON'T"])


def test_pretokenize_lossless() raises:
    var samples = [
        "",
        "Hello world",
        "  leading and trailing  ",
        "don't stop believin'",
        "you'd've thought we'd have learned",
        "'tis the season, the 90's are back",
        "abc1234567def",
        "line one\nline two\r\n\r\nend",
        "tabs\tand\ttabs",
        "a  b   c    d",
        "price: $3.50 (cheap!)",
        "123 4567 89",
        "café 中 text",
        "word- word -word",
        " \n \n ",
    ]
    for s in samples:
        var joined = String()
        for piece in pretokenize_text(s):
            joined += piece
        assert_equal(joined, s, "split is not lossless")


# --- byte-level family (Rust roundtrip.rs) ---


def test_round_trip_is_identity() raises:
    var tok = Tokenizer.build(synthetic_data())
    var samples = [
        "The capital of France is",
        "Hello, world! Price: $3.50 (cheap).",
        "abc 123 4567",
        "spaces  between   words",
        "trailing spaces  ",
        "one\ntwo\r\nthree",
        "don't stop believin'",
        "a+b=c! x_y-z",
        "café au lait",
        "中文 test 漢字",
        "naïve façade",
        "the the the",
    ]
    for s in samples:
        var ids = tok.encode(s)
        var back = tok.decode(ids)
        assert_equal(back, s, "round trip failed")


def test_merges_are_applied_by_rank() raises:
    var tok = Tokenizer.build(synthetic_data())
    assert_ids(tok.encode("the"), [THE_ID], "the")
    assert_ids(tok.encode(" the"), [G_THE_ID], " the")
    assert_ids(tok.encode("th"), [TH_ID], "th")
    assert_equal(tok.decode([G_THE_ID]), " the")


def test_g_dot_becomes_a_real_space() raises:
    var tok = Tokenizer.build(synthetic_data())
    var g_id = tok.token_id("Ġ")
    assert_true(g_id >= 0, "Ġ is in the vocab")
    assert_equal(tok.decode([g_id]), " ")
    var ids = tok.encode("a b")
    assert_true(g_id in ids, "encode of 'a b' should contain Ġ")
    var text = tok.decode(ids)
    assert_equal(text, "a b")
    assert_true(text.find("Ġ") == -1, "decoded text must not contain Ġ")


def test_byte_tokens_reassemble_multibyte() raises:
    var tok = Tokenizer.build(synthetic_data())
    assert_equal(
        tok.decode([tok.token_id("<0xC3>"), tok.token_id("<0xA9>")]),
        "é",
    )
    assert_equal(
        tok.decode(
            [
                tok.token_id("<0xE4>"),
                tok.token_id("<0xB8>"),
                tok.token_id("<0xAD>"),
            ]
        ),
        "中",
    )
    var a = tok.token_id("a")
    assert_equal(
        tok.decode([a, tok.token_id("<0xC3>"), tok.token_id("<0xA9>")]),
        "aé",
    )
    var c3 = tok.token_id("Ã")
    var a9 = tok.token_id("©")
    assert_equal(tok.decode([c3, a9]), "é")


def test_encode_falls_back_to_byte_tokens() raises:
    var data = TokenizerData()
    for b in range(0x80):
        var cp = byte_to_codepoint(b)
        data.tokens.append(String(Codepoint.from_u32(UInt32(cp)).value()))
        data.token_types.append(TT_NORMAL)
    for b in range(256):
        data.tokens.append(byte_token_name(b))
        data.token_types.append(TT_BYTE)
    var tok = Tokenizer.build(data^)
    var ids = tok.encode("é")
    assert_ids(
        ids,
        [tok.token_id("<0xC3>"), tok.token_id("<0xA9>")],
        "byte fallback",
    )
    assert_equal(tok.decode(ids), "é")


def test_special_tokens_match_literally() raises:
    var tok = Tokenizer.build(synthetic_data())
    assert_equal(tok.bos_id(), EOS_ID)
    assert_equal(tok.eos_id(), EOS_ID)
    assert_equal(tok.unk_id(), UNK_ID)
    assert_equal(tok.pad_id(), -1)
    var ids = tok.encode("hi<|end_of_text|>bye")
    var specials = 0
    for id in ids:
        if id == EOS_ID:
            specials += 1
    assert_equal(specials, 1)
    assert_equal(len(ids), 2 + 1 + 3)
    assert_equal(tok.decode(ids), "hi<|end_of_text|>bye")


def test_digits_split_in_runs_of_at_most_three() raises:
    var tok = Tokenizer.build(synthetic_data())
    var ids = tok.encode("1234567")
    assert_equal(len(ids), 7)
    assert_equal(tok.decode(ids), "1234567")


def test_named_refusals() raises:
    var tok = Tokenizer.build(synthetic_data())
    # Unknown token id on decode.
    var saw = 0
    try:
        _ = tok.decode([999999])
        assert_true(False, "decode of an unknown id must refuse")
    except e:
        assert_true(
            String(e).find("tokenizer_invalid_token_id") != -1,
            "refusal must name tokenizer_invalid_token_id",
        )
        saw += 1
    # Decoded bytes that are not valid UTF-8 (a lone <0xC3>).
    try:
        _ = tok.decode([tok.token_id("<0xC3>")])
        assert_true(False, "decode of a lone byte must refuse")
    except e:
        assert_true(
            String(e).find("tokenizer_invalid_utf8") != -1,
            "refusal must name tokenizer_invalid_utf8",
        )
        saw += 1
    # Empty vocabulary.
    try:
        _ = Tokenizer.build(TokenizerData())
        assert_true(False, "empty vocab must refuse")
    except e:
        assert_true(
            String(e).find("tokenizer_empty_vocab") != -1,
            "refusal must name tokenizer_empty_vocab",
        )
        saw += 1
    # Malformed merge.
    var bad_merge = TokenizerData()
    bad_merge.tokens = ["a"]
    bad_merge.merges = ["no space here"]
    try:
        _ = Tokenizer.build(bad_merge^)
        assert_true(False, "bad merge must refuse")
    except e:
        assert_true(
            String(e).find("tokenizer_invalid_merge") != -1,
            "refusal must name tokenizer_invalid_merge",
        )
        saw += 1
    # Special token id out of range.
    var bad_role = TokenizerData()
    bad_role.tokens = ["a"]
    bad_role.bos_token_id = 7
    try:
        _ = Tokenizer.build(bad_role^)
        assert_true(False, "out-of-range bos must refuse")
    except e:
        assert_true(
            String(e).find("tokenizer_special_token_out_of_range") != -1,
            "refusal must name tokenizer_special_token_out_of_range",
        )
        saw += 1
    # Duplicate token.
    var dup = TokenizerData()
    dup.tokens = ["a", "a"]
    try:
        _ = Tokenizer.build(dup^)
        assert_true(False, "duplicate token must refuse")
    except e:
        assert_true(
            String(e).find("tokenizer_duplicate_token") != -1,
            "refusal must name tokenizer_duplicate_token",
        )
        saw += 1
    # Token type length mismatch.
    var bad_types = TokenizerData()
    bad_types.tokens = ["a", "b"]
    bad_types.token_types = [TT_NORMAL]
    try:
        _ = Tokenizer.build(bad_types^)
        assert_true(False, "type length mismatch must refuse")
    except e:
        assert_true(
            String(e).find("tokenizer_token_type_length_mismatch") != -1,
            "refusal must name tokenizer_token_type_length_mismatch",
        )
        saw += 1
    # Scores length mismatch.
    var bad_scores = TokenizerData()
    bad_scores.tokens = ["a", "b"]
    bad_scores.scores = [Float64(0.0)]
    bad_scores.has_scores = True
    try:
        _ = Tokenizer.build(bad_scores^)
        assert_true(False, "scores mismatch must refuse")
    except e:
        assert_true(
            String(e).find("tokenizer_scores_length_mismatch") != -1,
            "refusal must name tokenizer_scores_length_mismatch",
        )
        saw += 1
    # Unencodable symbol: no byte tokens, no unk.
    var tiny = TokenizerData()
    tiny.tokens = ["a"]
    var tiny_tok = Tokenizer.build(tiny^)
    try:
        _ = tiny_tok.encode("b")
        assert_true(False, "unencodable symbol must refuse")
    except e:
        assert_true(
            String(e).find("tokenizer_unencodable_symbol") != -1,
            "refusal must name tokenizer_unencodable_symbol",
        )
        saw += 1
    assert_equal(saw, 9)


def test_malformed_byte_token_refused_on_decode() raises:
    var data = TokenizerData()
    data.tokens = ["hello"]
    data.token_types = [TT_BYTE]
    var tok = Tokenizer.build(data^)
    try:
        _ = tok.decode([0])
        assert_true(False, "malformed byte token must refuse")
    except e:
        assert_true(
            String(e).find("tokenizer_malformed_byte_token") != -1,
            "refusal must name tokenizer_malformed_byte_token",
        )


# --- SentencePiece family (Rust spm_family.rs + spm.rs units) ---


def spm_data() -> TokenizerData:
    """The Rust spm_family.rs vocabulary, verbatim."""
    var data = TokenizerData()
    data.model = "llama"
    data.has_model = True
    data.tokens = [
        "<unk>",
        "<s>",
        "</s>",
        "▁",
        "a",
        "b",
        "▁a",
        "▁ab",
        "ab",
        "<0xC3>",
        "<0xA9>",
    ]
    data.token_types = [
        TT_UNKNOWN,
        TT_CONTROL,
        TT_CONTROL,
        TT_NORMAL,
        TT_NORMAL,
        TT_NORMAL,
        TT_NORMAL,
        TT_NORMAL,
        TT_NORMAL,
        TT_BYTE,
        TT_BYTE,
    ]
    # Present in the file but not parsed for this family: even a
    # malformed entry must not refuse construction.
    data.merges = ["not a merge at all"]
    data.scores = [
        0.0,
        0.0,
        0.0,
        -4.0,
        -1.0,
        -2.0,
        -0.5,
        -0.1,
        -3.0,
        0.0,
        0.0,
    ]
    data.has_scores = True
    data.bos_token_id = 1
    data.eos_token_id = 2
    data.unknown_token_id = 0
    return data^


def test_model_string_selects_the_family() raises:
    var tok = Tokenizer.build(spm_data())
    assert_equal(tok.family_kind(), FAMILY_SENTENCEPIECE)
    var gpt2 = spm_data()
    gpt2.model = "gpt2"
    gpt2.merges = ["▁ a"]
    var tok2 = Tokenizer.build(gpt2^)
    assert_equal(tok2.family_kind(), FAMILY_BYTE_LEVEL)
    var absent = spm_data()
    absent.has_model = False
    absent.merges = ["▁ a"]
    var tok3 = Tokenizer.build(absent^)
    assert_equal(tok3.family_kind(), FAMILY_BYTE_LEVEL)


def test_unknown_model_is_refused_by_name() raises:
    var data = spm_data()
    data.model = "bert"
    try:
        _ = Tokenizer.build(data^)
        assert_true(False, "unknown model must refuse")
    except e:
        assert_true(
            String(e).find("tokenizer_unsupported_model") != -1,
            "refusal must name tokenizer_unsupported_model",
        )


def test_spm_encode_uses_pieces_and_prefix() raises:
    var tok = Tokenizer.build(spm_data())
    assert_ids(tok.encode("ab"), [7], "spm ab")
    assert_ids(tok.encode("é"), [3, 9, 10], "spm é byte fallback")


def test_spm_special_tokens_still_match_literally() raises:
    var tok = Tokenizer.build(spm_data())
    assert_ids(tok.encode("a</s>a"), [6, 2, 6], "spm specials")


def test_spm_decode_unescapes_and_renders_specials() raises:
    var tok = Tokenizer.build(spm_data())
    assert_equal(tok.decode([7]), " ab")
    assert_equal(tok.decode([6, 5]), " ab")


def test_spm_highest_score_merges_first() raises:
    # spm.rs unit: "ab" scores 1.0, "bc" scores 2.0: llama.cpp
    # merges "bc" first, blocking "ab".
    var data = TokenizerData()
    data.model = "llama"
    data.has_model = True
    data.tokens = ["<unk>", "▁", "a", "b", "c", "ab", "bc"]
    data.scores = [0.0, -1.0, -1.0, -1.0, -1.0, 1.0, 2.0]
    data.has_scores = True
    var tok = Tokenizer.build(data^)
    assert_ids(tok.encode("abc"), [1, 2, 6], "highest score first")


def test_spm_equal_scores_merge_leftmost() raises:
    var data = TokenizerData()
    data.model = "llama"
    data.has_model = True
    data.tokens = ["<unk>", "▁", "a", "b", "c", "ab", "bc"]
    data.has_scores = False  # no scores: all zero, leftmost wins
    var tok = Tokenizer.build(data^)
    assert_ids(tok.encode("abc"), [1, 5, 4], "leftmost on ties")


def test_spm_missing_byte_token_is_a_named_error() raises:
    var data = TokenizerData()
    data.model = "llama"
    data.has_model = True
    data.tokens = ["<unk>", "▁"]
    data.scores = [0.0, -1.0]
    data.has_scores = True
    var tok = Tokenizer.build(data^)
    try:
        _ = tok.encode("é")
        assert_true(False, "missing byte token must refuse")
    except e:
        assert_true(
            String(e).find("tokenizer_missing_byte_token") != -1,
            "refusal must name tokenizer_missing_byte_token",
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
