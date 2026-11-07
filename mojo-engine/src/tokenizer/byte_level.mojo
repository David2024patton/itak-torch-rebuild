# byte_level.mojo - the GPT-2 byte-to-unicode map at the heart of
# byte-level BPE, plus the UTF-8 codec helpers the package shares.
#
# Mirror of crates/tokenizer/src/byte_level.rs. Every byte
# 0..=255 is assigned a printable Unicode codepoint so BPE
# operates on text with no invisible control characters. Bytes in
# the ranges !..=~, ¡..=¬, ®..=ÿ map to the codepoint with the
# same value; the remaining 68 bytes map to U+0100 and upward in
# byte order. The space byte 0x20 maps to Ġ (U+0120), which is
# why raw GGUF token strings show Ġ where a space belongs: decode
# must run this map in reverse.
#
# The UTF-8 helpers work on codepoint lists (List[Int], one entry
# per Unicode scalar) because the rest of the package scans text
# by codepoint, exactly as the Rust side scans by char. Strict
# decoding validates like Rust's String::from_utf8: overlong
# forms, surrogates, and values above U+10FFFF are refused, and
# the refusal names the byte offset of the valid prefix
# (Utf8Error::valid_up_to on the Rust side).


def is_direct_byte(b: Int) -> Bool:
    """True when byte `b` maps to the codepoint with its value."""
    return (
        (0x21 <= b and b <= 0x7E)
        or (0xA1 <= b and b <= 0xAC)
        or (0xAE <= b and b <= 0xFF)
    )


def byte_to_codepoint(b: Int) -> Int:
    """The printable codepoint assigned to byte `b` (0..=255)."""
    if is_direct_byte(b):
        return b
    var next_cp = 256
    for cand in range(256):
        if not is_direct_byte(cand):
            if cand == b:
                return next_cp
            next_cp += 1
    return b  # unreachable for 0..=255


def codepoint_to_byte(cp: Int) -> Int:
    """The byte assigned to `cp`, or -1 when `cp` is not part of
    the byte-level alphabet (for example a literal character from
    a control token's text). Mirrors char_to_byte's None."""
    if cp >= 0 and cp <= 0xFF and is_direct_byte(cp):
        return cp
    if cp >= 256 and cp < 256 + 68:
        var idx = cp - 256
        var seen = 0
        for cand in range(256):
            if not is_direct_byte(cand):
                if seen == idx:
                    return cand
                seen += 1
    return -1


def byte_token_name(b: Int) -> String:
    """The GGUF byte-token name for a byte: `<0x0A>` and friends
    (uppercase hex, as the Rust side formats it)."""
    var hi = (b >> 4) & 0xF
    var lo = b & 0xF
    return "<0x" + _hex_char(hi) + _hex_char(lo) + ">"


def _hex_char(v: Int) -> String:
    var cp = (0x30 + v) if v < 10 else (0x41 + v - 10)
    return String(Codepoint.from_u32(UInt32(cp)).value())


def _hex_value(b: Int) -> Int:
    if 0x30 <= b and b <= 0x39:  # '0'..='9'
        return b - 0x30
    if 0x61 <= b and b <= 0x66:  # 'a'..='f'
        return b - 0x61 + 10
    if 0x41 <= b and b <= 0x46:  # 'A'..='F'
        return b - 0x41 + 10
    return -1


def parse_byte_token(text: String) -> Int:
    """Parse a `<0xXX>` byte-token name back to its byte,
    accepting upper or lower case hex. Returns -1 for any other
    shape (the Rust side's None)."""
    var bytes = List[UInt8]()
    for b in text.as_bytes():
        bytes.append(b)
    if len(bytes) != 6:
        return -1
    if (
        Int(bytes[0]) != 0x3C  # '<'
        or Int(bytes[1]) != 0x30  # '0'
        or Int(bytes[2]) != 0x78  # 'x'
        or Int(bytes[5]) != 0x3E  # '>'
    ):
        return -1
    var hi = _hex_value(Int(bytes[3]))
    var lo = _hex_value(Int(bytes[4]))
    if hi < 0 or lo < 0:
        return -1
    return (hi << 4) | lo


# --- UTF-8 codec over codepoint lists ---


def codepoints_of(text: String) -> List[Int]:
    """The codepoints of `text`, in order."""
    var out = List[Int]()
    for cp in text.codepoints():
        out.append(Int(cp.to_u32()))
    return out^


def string_from_codepoints(cps: List[Int], start: Int, end: Int) -> String:
    """Build the String for cps[start:end]. Callers pass valid
    scalar values from real text or the byte alphabet."""
    var out = String()
    for i in range(start, end):
        var opt = Codepoint.from_u32(UInt32(cps[i]))
        if Bool(opt):
            out += String(opt.value())
    return out^


def append_utf8(cp: Int, mut out: List[UInt8]):
    """Append the UTF-8 encoding of codepoint `cp` to `out`."""
    if cp < 0x80:
        out.append(UInt8(cp))
    elif cp < 0x800:
        out.append(UInt8(0xC0 | (cp >> 6)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    elif cp < 0x10000:
        out.append(UInt8(0xE0 | (cp >> 12)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    else:
        out.append(UInt8(0xF0 | (cp >> 18)))
        out.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))


def utf8_len_of_codepoint(cp: Int) -> Int:
    if cp < 0x80:
        return 1
    if cp < 0x800:
        return 2
    if cp < 0x10000:
        return 3
    return 4


def bytes_of_codepoints(cps: List[Int]) -> List[UInt8]:
    var out = List[UInt8]()
    for cp in cps:
        append_utf8(cp, out)
    return out^


def bytes_of_string(text: String) -> List[UInt8]:
    var out = List[UInt8]()
    for b in text.as_bytes():
        out.append(b)
    return out^


def decode_utf8_strict(bytes: List[UInt8]) raises -> List[Int]:
    """Decode UTF-8 bytes to codepoints, strictly.

    Raises tokenizer_invalid_utf8 naming the valid prefix length
    on the first invalid or truncated sequence, mirroring
    String::from_utf8 + Error::InvalidUtf8 on the Rust side.
    """
    var out = List[Int]()
    var n = len(bytes)
    var i = 0
    while i < n:
        var b0 = Int(bytes[i])
        if b0 < 0x80:
            out.append(b0)
            i += 1
            continue
        var need: Int
        var cp: Int
        var lo2 = 0x80
        var hi2 = 0xBF
        if 0xC2 <= b0 and b0 <= 0xDF:
            need = 2
            cp = b0 & 0x1F
        elif b0 == 0xE0:
            need = 3
            cp = b0 & 0x0F
            lo2 = 0xA0
        elif 0xE1 <= b0 and b0 <= 0xEC:
            need = 3
            cp = b0 & 0x0F
        elif b0 == 0xED:
            need = 3
            cp = b0 & 0x0F
            hi2 = 0x9F
        elif 0xEE <= b0 and b0 <= 0xEF:
            need = 3
            cp = b0 & 0x0F
        elif b0 == 0xF0:
            need = 4
            cp = b0 & 0x07
            lo2 = 0x90
        elif 0xF1 <= b0 and b0 <= 0xF3:
            need = 4
            cp = b0 & 0x07
        elif b0 == 0xF4:
            need = 4
            cp = b0 & 0x07
            hi2 = 0x8F
        else:
            raise Error(
                "tokenizer_invalid_utf8: decoded bytes are not valid"
                " UTF-8 (valid up to byte "
                + String(i)
                + ")"
            )
        if i + need > n:
            raise Error(
                "tokenizer_invalid_utf8: decoded bytes are not valid"
                " UTF-8 (valid up to byte "
                + String(i)
                + ")"
            )
        var b1 = Int(bytes[i + 1])
        if b1 < lo2 or b1 > hi2:
            raise Error(
                "tokenizer_invalid_utf8: decoded bytes are not valid"
                " UTF-8 (valid up to byte "
                + String(i)
                + ")"
            )
        cp = (cp << 6) | (b1 & 0x3F)
        var ok = True
        for k in range(2, need):
            var bk = Int(bytes[i + k])
            if bk < 0x80 or bk > 0xBF:
                ok = False
                break
            cp = (cp << 6) | (bk & 0x3F)
        if not ok:
            raise Error(
                "tokenizer_invalid_utf8: decoded bytes are not valid"
                " UTF-8 (valid up to byte "
                + String(i)
                + ")"
            )
        out.append(cp)
        i += need
    return out^


def string_from_bytes_strict(bytes: List[UInt8]) raises -> String:
    """The String for a byte stream that must be valid UTF-8
    (decode's contract); see decode_utf8_strict."""
    var cps = decode_utf8_strict(bytes)
    return string_from_codepoints(cps, 0, len(cps))


def string_from_bytes_lossy(bytes: List[UInt8]) -> String:
    """The String for a byte stream with invalid sequences
    replaced by U+FFFD, mirroring String::from_utf8_lossy.

    Simplification, stated plainly: an invalid sequence is
    replaced per byte here, where Rust replaces the maximal
    invalid subpart, so a truncated multi-byte sequence at the
    end of the stream can yield more replacement characters
    here than in Rust. decode_lossy is not on the conformance
    path; decode (strict) matches exactly.
    """
    var out = String()
    var n = len(bytes)
    var i = 0
    while i < n:
        var b0 = Int(bytes[i])
        if b0 < 0x80:
            out += String(Codepoint.from_u32(UInt32(b0)).value())
            i += 1
            continue
        var need = 0
        if 0xC2 <= b0 and b0 <= 0xDF:
            need = 2
        elif 0xE0 <= b0 and b0 <= 0xEF:
            need = 3
        elif 0xF0 <= b0 and b0 <= 0xF4:
            need = 4
        if need == 0 or i + need > n:
            out += String(Codepoint.from_u32(UInt32(0xFFFD)).value())
            i += 1
            continue
        var slice = List[UInt8]()
        for k in range(need):
            slice.append(bytes[i + k])
        try:
            var cps = decode_utf8_strict(slice)
            out += string_from_codepoints(cps, 0, len(cps))
            i += need
        except:
            out += String(Codepoint.from_u32(UInt32(0xFFFD)).value())
            i += 1
    return out^
