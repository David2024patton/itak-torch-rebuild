# tokenizer.mojo - the tokenizer itself: construction from
# TokenizerData, byte-level BPE encoding, SentencePiece encoding,
# and byte-faithful decoding.
#
# Mirror of crates/tokenizer/src/tokenizer.rs + data.rs +
# error.rs. The semantics contract, from the Rust crate docs:
#
# - Input is already-parsed GGUF metadata as plain values
#   (TokenizerData): tokenizer.ggml.model, .tokens, .token_type,
#   .merges, .scores, and the bos/eos/unknown/padding ids. This
#   module does not parse GGUF itself; gguf_vocab.mojo is the
#   temporary seam that produces TokenizerData from a file.
# - Family selection by model string: absent or "gpt2" is
#   byte-level BPE; "llama" is SentencePiece BPE; anything else
#   is refused by name (tokenizer_unsupported_model).
# - Byte-level: text is split by pretokenize (the GPT-4o /
#   llama4 scanner), each piece byte-mapped and merged by rank
#   (lowest rank first, every occurrence of the best pair per
#   pass). Special (control / user-defined) tokens in the text
#   match literally, longest first, and emit as their own ids.
# - SentencePiece: see spm.mojo. The merges array does not drive
#   that family and is not even parsed for it.
# - Decode always runs the family's inverse map (Ġ becomes a
#   real space; ▁ becomes a space), <0xXX> byte tokens fold back
#   into the byte stream, and specials render literally, so a
#   multibyte character split across tokens reassembles.
# - Ids use Int with -1 for "absent" where Rust uses
#   Option<u32> (roles, token_id lookups). Every refusal is a
#   raised Error whose message starts with a tokenizer_* name,
#   mirroring the Rust Error variants one for one:
#   tokenizer_empty_vocab, tokenizer_duplicate_token,
#   tokenizer_token_type_length_mismatch, tokenizer_invalid_merge,
#   tokenizer_special_token_out_of_range,
#   tokenizer_unsupported_model, tokenizer_scores_length_mismatch,
#   tokenizer_unencodable_symbol, tokenizer_invalid_token_id,
#   tokenizer_malformed_byte_token, tokenizer_invalid_utf8,
#   tokenizer_missing_byte_token (the last three live in
#   byte_level.mojo / spm.mojo).

from std.collections import Dict

from diag import log_debug
from tokenizer.byte_level import (
    byte_to_codepoint,
    byte_token_name,
    bytes_of_codepoints,
    bytes_of_string,
    codepoint_to_byte,
    codepoints_of,
    decode_utf8_strict,
    parse_byte_token,
    string_from_bytes_lossy,
    string_from_bytes_strict,
    string_from_codepoints,
    append_utf8,
)
from tokenizer.pretokenize import pretokenize
from tokenizer.spm import encode_segment as spm_encode_segment
from tokenizer.spm import unescape_whitespace

# Token types, mirroring GGUF tokenizer.ggml.token_type
# (llama.cpp's llama_token_type) as mapped by the Rust
# TokenType::from_gguf.
comptime TT_UNDEFINED = 0
comptime TT_NORMAL = 1
comptime TT_UNKNOWN = 2
comptime TT_CONTROL = 3
comptime TT_USER_DEFINED = 4
comptime TT_UNUSED = 5
comptime TT_BYTE = 6

# Tokenizer families.
comptime FAMILY_BYTE_LEVEL = 0
comptime FAMILY_SENTENCEPIECE = 1


def token_type_from_gguf(value: Int) -> Int:
    """Raw GGUF token-type integer -> TT_* constant.
    Unrecognized values map to TT_UNDEFINED."""
    if 1 <= value and value <= 6:
        return value
    return TT_UNDEFINED


def token_type_is_special(tt: Int) -> Bool:
    """Whether this token is matched literally against the
    input text during encoding (control / user-defined)."""
    return tt == TT_CONTROL or tt == TT_USER_DEFINED


struct TokenizerData(Copyable, Movable):
    """Everything a tokenizer needs, already parsed: the
    GGUF tokenizer.ggml.* metadata as plain values, so this
    module stays decoupled from the GGUF parser (the Rust
    crate's TokenizerData, with Option fields as has_* flags /
    -1 sentinels, plus the file's pre-tokenizer name, which the
    Rust tokdata format carries beside the data)."""

    var model: String
    var has_model: Bool
    var pre: String
    var has_pre: Bool
    var tokens: List[String]
    var token_types: List[Int]
    var merges: List[String]
    var scores: List[Float64]
    var has_scores: Bool
    var bos_token_id: Int
    var eos_token_id: Int
    var unknown_token_id: Int
    var padding_token_id: Int

    def __init__(out self):
        self.model = String()
        self.has_model = False
        self.pre = String()
        self.has_pre = False
        self.tokens = List[String]()
        self.token_types = List[Int]()
        self.merges = List[String]()
        self.scores = List[Float64]()
        self.has_scores = False
        self.bos_token_id = -1
        self.eos_token_id = -1
        self.unknown_token_id = -1
        self.padding_token_id = -1


def _string_from_bytes(
    bytes: List[UInt8], start: Int, end: Int
) raises -> String:
    var slice = List[UInt8]()
    for i in range(start, end):
        slice.append(bytes[i])
    var cps = decode_utf8_strict(slice)
    return string_from_codepoints(cps, 0, len(cps))


def _check_role(role: String, id: Int, vocab_size: Int) raises:
    if id >= 0 and id >= vocab_size:
        raise Error(
            "tokenizer_special_token_out_of_range: "
            + role
            + " token id "
            + String(id)
            + " is outside the vocabulary of size "
            + String(vocab_size)
        )


struct Tokenizer(Movable):
    """A tokenizer over a fixed vocabulary, in one of the two
    families the file's tokenizer.ggml.model selects. Build via
    Tokenizer.build(); construction validates the data and
    refuses loudly on anything malformed."""

    var family: Int
    var tokens: List[String]
    var token_types: List[Int]
    var types_supplied: Bool
    var vocab: Dict[String, Int]
    var merge_ranks: Dict[String, Int]
    var merge_sep: String
    var scores: List[Float64]
    var specials_texts: List[String]
    var specials_ids: List[Int]
    var specials_cps: List[List[Int]]
    var bos: Int
    var eos: Int
    var unk: Int
    var pad: Int

    def __init__(
        out self,
        family: Int,
        var tokens: List[String],
        var token_types: List[Int],
        types_supplied: Bool,
        var vocab: Dict[String, Int],
        var merge_ranks: Dict[String, Int],
        var merge_sep: String,
        var scores: List[Float64],
        var specials_texts: List[String],
        var specials_ids: List[Int],
        var specials_cps: List[List[Int]],
        bos: Int,
        eos: Int,
        unk: Int,
        pad: Int,
    ):
        self.family = family
        self.tokens = tokens^
        self.token_types = token_types^
        self.types_supplied = types_supplied
        self.vocab = vocab^
        self.merge_ranks = merge_ranks^
        self.merge_sep = merge_sep^
        self.scores = scores^
        self.specials_texts = specials_texts^
        self.specials_ids = specials_ids^
        self.specials_cps = specials_cps^
        self.bos = bos
        self.eos = eos
        self.unk = unk
        self.pad = pad

    @staticmethod
    def build(var data: TokenizerData) raises -> Tokenizer:
        """Builds a tokenizer from parsed GGUF tokenizer
        metadata, mirroring Tokenizer::new on the Rust side,
        validation order included."""
        # Family selection.
        var family: Int
        if not data.has_model or data.model == "gpt2":
            family = FAMILY_BYTE_LEVEL
        elif data.model == "llama":
            family = FAMILY_SENTENCEPIECE
        else:
            _ = log_debug(
                "tokenizer",
                "refusing unsupported tokenizer model " + data.model,
            )
            raise Error(
                "tokenizer_unsupported_model: unsupported"
                " tokenizer.ggml.model family: "
                + data.model
            )
        if len(data.tokens) == 0:
            _ = log_debug("tokenizer", "refusing empty vocabulary")
            raise Error("tokenizer_empty_vocab: tokenizer vocabulary is empty")
        # Scores.
        var scores = List[Float64]()
        if data.has_scores:
            if len(data.scores) != len(data.tokens):
                raise Error(
                    "tokenizer_scores_length_mismatch: scores length "
                    + String(len(data.scores))
                    + " does not match tokens length "
                    + String(len(data.tokens))
                )
            for s in data.scores:
                scores.append(s)
        else:
            # llama.cpp's vocab load: no scores array means
            # every token scores 0.0.
            for _ in range(len(data.tokens)):
                scores.append(Float64(0.0))
        # Token types.
        if len(data.token_types) != 0 and len(data.token_types) != len(
            data.tokens
        ):
            raise Error(
                "tokenizer_token_type_length_mismatch: token_types length "
                + String(len(data.token_types))
                + " does not match tokens length "
                + String(len(data.tokens))
            )
        var types_supplied = len(data.token_types) != 0
        var token_types = List[Int]()
        if types_supplied:
            for raw in data.token_types:
                token_types.append(token_type_from_gguf(raw))
        else:
            for _ in range(len(data.tokens)):
                token_types.append(TT_UNDEFINED)
        # Vocabulary map, refusing duplicates.
        var vocab = Dict[String, Int]()
        for idx in range(len(data.tokens)):
            var token = data.tokens[idx]
            if token in vocab:
                raise Error(
                    "tokenizer_duplicate_token: duplicate token in vocabulary: "
                    + token
                )
            vocab[token] = idx
        # Merge ranks drive only the byte-level family; the
        # SentencePiece family does not parse the array at all.
        var sep = string_from_codepoints(List[Int]([1]), 0, 1)
        var merge_ranks = Dict[String, Int]()
        if family == FAMILY_BYTE_LEVEL:
            for index in range(len(data.merges)):
                var merge = data.merges[index]
                var mb = bytes_of_string(merge)
                var split = -1
                for bi in range(len(mb)):
                    if Int(mb[bi]) == 0x20:
                        split = bi
                        break
                var valid = split > 0 and split < len(mb) - 1
                var right_has_space = False
                if valid:
                    for bi in range(split + 1, len(mb)):
                        if Int(mb[bi]) == 0x20:
                            right_has_space = True
                            break
                    valid = not right_has_space
                if not valid:
                    raise Error(
                        "tokenizer_invalid_merge: merge entry "
                        + String(index)
                        + " is not two token strings separated by one space: "
                        + merge
                    )
                var left = _string_from_bytes(mb, 0, split)
                var right = _string_from_bytes(mb, split + 1, len(mb))
                merge_ranks[left + sep + right] = index
        # Special token roles.
        _check_role("bos", data.bos_token_id, len(data.tokens))
        _check_role("eos", data.eos_token_id, len(data.tokens))
        _check_role("unk", data.unknown_token_id, len(data.tokens))
        _check_role("padding", data.padding_token_id, len(data.tokens))
        # Specials, longest text first (stable, as Rust's
        # sort_by_key over byte length).
        var specials_texts = List[String]()
        var specials_ids = List[Int]()
        var specials_cps = List[List[Int]]()
        for idx in range(len(data.tokens)):
            var text = data.tokens[idx]
            if (
                token_type_is_special(token_types[idx])
                and text.byte_length() > 0
            ):
                var pos = len(specials_texts)
                for si in range(len(specials_texts)):
                    if specials_texts[si].byte_length() < text.byte_length():
                        pos = si
                        break
                specials_texts.insert(pos, text)
                specials_ids.insert(pos, idx)
                specials_cps.insert(pos, codepoints_of(text))
        _ = log_debug(
            "tokenizer",
            "loaded family="
            + (
                String("llama") if family
                == FAMILY_SENTENCEPIECE else String("gpt2")
            )
            + " vocab="
            + String(len(data.tokens))
            + " merges="
            + String(len(data.merges))
            + " specials="
            + String(len(specials_texts)),
        )
        var bos_id = data.bos_token_id
        var eos_id = data.eos_token_id
        var unk_id = data.unknown_token_id
        var pad_id = data.padding_token_id
        return Tokenizer(
            family,
            data.tokens.copy(),
            token_types^,
            types_supplied,
            vocab^,
            merge_ranks^,
            sep^,
            scores^,
            specials_texts^,
            specials_ids^,
            specials_cps^,
            bos_id,
            eos_id,
            unk_id,
            pad_id,
        )

    def family_kind(self) -> Int:
        """FAMILY_BYTE_LEVEL or FAMILY_SENTENCEPIECE."""
        return self.family

    def family_name(self) -> String:
        """The GGUF model string for this family."""
        return "llama" if self.family == FAMILY_SENTENCEPIECE else "gpt2"

    def vocab_size(self) -> Int:
        return len(self.tokens)

    def has_token(self, id: Int) -> Bool:
        return 0 <= id and id < len(self.tokens)

    def token_text(self, id: Int) -> String:
        """The text of a token, verbatim as stored
        (byte-mapped). Empty string for an out-of-range id;
        check has_token() when that distinction matters."""
        if not self.has_token(id):
            return String()
        return self.tokens[id]

    def token_id(self, text: String) -> Int:
        """The id of a token by its stored text, or -1."""
        return self.vocab.get(text, -1)

    def bos_id(self) -> Int:
        return self.bos

    def eos_id(self) -> Int:
        return self.eos

    def unk_id(self) -> Int:
        return self.unk

    def pad_id(self) -> Int:
        return self.pad

    # --- encode ---

    def encode(self, text: String) raises -> List[Int]:
        """Encodes text to token ids. Special tokens present in
        the text are matched literally, longest first; the text
        between them goes through pre-tokenization and BPE (or
        the SentencePiece path). No BOS/EOS is added: that is
        the caller's job, as on the Rust side."""
        var cps = codepoints_of(text)
        var out = List[Int]()
        if len(self.specials_texts) == 0:
            self._encode_plain(cps, 0, len(cps), out)
            return out^
        var seg_start = 0
        while seg_start < len(cps):
            var hit_pos = -1
            var hit_id = -1
            var hit_len = 0
            var p = seg_start
            while p < len(cps) and hit_pos < 0:
                for si in range(len(self.specials_cps)):
                    var sc_len = len(self.specials_cps[si])
                    if p + sc_len <= len(cps):
                        var matches = True
                        for k in range(sc_len):
                            if cps[p + k] != self.specials_cps[si][k]:
                                matches = False
                                break
                        if matches:
                            hit_pos = p
                            hit_id = self.specials_ids[si]
                            hit_len = sc_len
                            break
                if hit_pos < 0:
                    p += 1
            if hit_pos < 0:
                self._encode_plain(cps, seg_start, len(cps), out)
                seg_start = len(cps)
            else:
                if hit_pos > seg_start:
                    self._encode_plain(cps, seg_start, hit_pos, out)
                out.append(hit_id)
                seg_start = hit_pos + hit_len
        return out^

    def _encode_plain(
        self, cps: List[Int], start: Int, end: Int, mut out: List[Int]
    ) raises:
        """Encodes a segment known to contain no special
        tokens, by family."""
        if end <= start:
            return
        var segment = List[Int]()
        for i in range(start, end):
            segment.append(cps[i])
        if self.family == FAMILY_SENTENCEPIECE:
            # The space prefix is unconditional here, as on the
            # Rust side: a plain segment only ever occurs at the
            # start or right after a special token, and llama.cpp
            # prefixes in both cases.
            var ids = spm_encode_segment(self.vocab, self.scores, segment, True)
            for id in ids:
                out.append(id)
            return
        var pieces = pretokenize(segment)
        for piece in pieces:
            self._encode_piece(piece, out)

    def _encode_piece(self, piece: List[Int], mut out: List[Int]) raises:
        """Byte-maps one pre-tokenizer piece and runs BPE over
        it (lowest merge rank first; every occurrence of the
        best pair merges in each pass)."""
        var piece_bytes = bytes_of_codepoints(piece)
        var symbols = List[String]()
        for b in piece_bytes:
            var mapped = List[Int]([byte_to_codepoint(Int(b))])
            symbols.append(string_from_codepoints(mapped, 0, 1))
        while len(symbols) > 1:
            var best_rank = -1
            var best_at = -1
            for i in range(len(symbols) - 1):
                var key = symbols[i] + self.merge_sep + symbols[i + 1]
                if key in self.merge_ranks:
                    var rank = self.merge_ranks[key]
                    if best_rank < 0 or rank < best_rank:
                        best_rank = rank
                        best_at = i
            if best_at < 0:
                break
            var left = symbols[best_at]
            var right = symbols[best_at + 1]
            var merged = List[String]()
            var i = 0
            while i < len(symbols):
                if (
                    i + 1 < len(symbols)
                    and symbols[i] == left
                    and symbols[i + 1] == right
                ):
                    merged.append(left + right)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged^
        for symbol in symbols:
            out.append(self._symbol_id(symbol))

    def _symbol_id(self, symbol: String) raises -> Int:
        """Resolves one BPE symbol to a token id: vocabulary
        first, then the <0xXX> byte-token fallback for single
        characters, then the unknown token, then a named
        refusal."""
        if symbol in self.vocab:
            return self.vocab[symbol]
        var cps = codepoints_of(symbol)
        if len(cps) == 1:
            var byte = codepoint_to_byte(cps[0])
            if byte >= 0:
                var name = byte_token_name(byte)
                if name in self.vocab:
                    _ = log_debug(
                        "tokenizer", "byte fallback for symbol " + symbol
                    )
                    return self.vocab[name]
        if self.unk >= 0:
            _ = log_debug("tokenizer", "unk fallback for symbol " + symbol)
            return self.unk
        _ = log_debug("tokenizer", "refusing unencodable symbol " + symbol)
        raise Error(
            "tokenizer_unencodable_symbol: symbol has no token,"
            " byte fallback, or unk: "
            + symbol
        )

    # --- decode ---

    def decode_bytes(self, ids: List[Int]) raises -> List[UInt8]:
        """Decodes token ids into the raw byte stream they
        represent. A multibyte UTF-8 character split across
        several tokens reassembles in this stream."""
        var out = List[UInt8]()
        for id in ids:
            self._append_token_bytes(id, out)
        return out^

    def _append_token_bytes(self, id: Int, mut out: List[UInt8]) raises:
        if not self.has_token(id):
            raise Error(
                "tokenizer_invalid_token_id: token id "
                + String(id)
                + " is outside the vocabulary"
            )
        var text = self.tokens[id]
        var ty = self.token_types[id]
        if self.family == FAMILY_SENTENCEPIECE:
            self._append_token_bytes_spm(text, ty, out)
            return
        if ty == TT_BYTE:
            var byte = parse_byte_token(text)
            if byte < 0:
                raise Error(
                    "tokenizer_malformed_byte_token: byte token is"
                    " not in <0xXX> form: "
                    + text
                )
            out.append(UInt8(byte))
            return
        if ty == TT_CONTROL or ty == TT_USER_DEFINED:
            for b in text.as_bytes():
                out.append(b)
            return
        # With no type array, a <0xXX>-shaped token can only be
        # a byte token.
        if not self.types_supplied:
            var byte = parse_byte_token(text)
            if byte >= 0:
                out.append(UInt8(byte))
                return
        for cp in codepoints_of(text):
            var byte = codepoint_to_byte(cp)
            if byte >= 0:
                out.append(UInt8(byte))
            else:
                append_utf8(cp, out)

    def _append_token_bytes_spm(
        self, text: String, ty: Int, mut out: List[UInt8]
    ) raises:
        """SentencePiece decode, the LLAMA_VOCAB_TYPE_SPM case
        of llama.cpp's token_to_piece (see the Rust side)."""
        if ty == TT_BYTE:
            var byte = parse_byte_token(text)
            if byte < 0:
                raise Error(
                    "tokenizer_malformed_byte_token: byte token is"
                    " not in <0xXX> form: "
                    + text
                )
            out.append(UInt8(byte))
            return
        if ty == TT_CONTROL or ty == TT_USER_DEFINED or ty == TT_UNKNOWN:
            for b in text.as_bytes():
                out.append(b)
            return
        if ty == TT_NORMAL:
            var un = unescape_whitespace(text)
            for b in un.as_bytes():
                out.append(b)
            return
        if ty == TT_UNUSED:
            return
        # TT_UNDEFINED.
        if not self.types_supplied:
            var byte = parse_byte_token(text)
            if byte >= 0:
                out.append(UInt8(byte))
                return
            var un = unescape_whitespace(text)
            for b in un.as_bytes():
                out.append(b)

    def decode(self, ids: List[Int]) raises -> String:
        """Decodes token ids to a string, requiring the full
        byte stream to be valid UTF-8."""
        var bytes = self.decode_bytes(ids)
        return string_from_bytes_strict(bytes)

    def decode_lossy(self, ids: List[Int]) raises -> String:
        """Decodes token ids to a string, replacing invalid
        UTF-8 sequences with the replacement character."""
        var bytes = self.decode_bytes(ids)
        return string_from_bytes_lossy(bytes)
