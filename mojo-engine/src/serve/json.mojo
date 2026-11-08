# json.mojo - the small JSON reader/writer the serve layer
# needs. Mojo 1.2's stdlib has no JSON module, and the endpoint
# is Mojo-only, so this is a deliberate MINIMAL dialect:
#
#   write: json_escape() for embedding a string in JSON.
#   read:  string fields, integer fields, and the "content"
#          strings of a chat messages array, extracted with a
#          real string scanner (escapes honored, \" does not
#          end a string early).
#
# It is NOT a general JSON parser: nested values are not
# materialized, arrays other than messages are not walked, and
# numbers other than plain integers are refused. That is the
# v1 contract, stated here so nobody mistakes it for one.
#
# Named refusals: serve_bad_json (malformed string/number or a
# field of the wrong shape), serve_missing_field (a required
# string field is absent). Byte-oriented throughout: Mojo
# strings are UTF-8, lengths are byte lengths.

from tokenizer.byte_level import string_from_bytes_lossy


def _bytes_to_string(bytes: List[UInt8]) -> String:
    return string_from_bytes_lossy(bytes)


def _append_utf8(cp: Int, mut out: List[UInt8]):
    """Append one codepoint as UTF-8 (BMP + astral)."""
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


def json_escape(text: String) -> String:
    """Escape a string for embedding inside JSON quotes."""
    var out = List[UInt8]()
    for b in text.as_bytes():
        var v = Int(b)
        if v == 34:  # "
            out.append(UInt8(92))
            out.append(UInt8(34))
        elif v == 92:  # backslash
            out.append(UInt8(92))
            out.append(UInt8(92))
        elif v == 10:
            out.append(UInt8(92))
            out.append(UInt8(110))  # n
        elif v == 13:
            out.append(UInt8(92))
            out.append(UInt8(114))  # r
        elif v == 9:
            out.append(UInt8(92))
            out.append(UInt8(116))  # t
        elif v < 32:
            # \u00XX with lowercase hex.
            out.append(UInt8(92))
            out.append(UInt8(117))  # u
            out.append(UInt8(48))  # 0
            out.append(UInt8(48))  # 0
            var hi = (v >> 4) & 0xF
            var lo = v & 0xF
            out.append(UInt8(48 + hi) if hi < 10 else UInt8(87 + hi))
            out.append(UInt8(48 + lo) if lo < 10 else UInt8(87 + lo))
        else:
            out.append(b)
    return _bytes_to_string(out)


def _find_substring(hay: String, needle: String, start: Int) -> Int:
    """First index >= start where needle occurs, or -1."""
    var h = hay.as_bytes()
    var n = needle.as_bytes()
    if len(n) == 0 or len(h) < len(n):
        return -1
    var i = start
    while i + len(n) <= len(h):
        var ok = True
        for j in range(len(n)):
            if h[i + j] != n[j]:
                ok = False
                break
        if ok:
            return i
        i += 1
    return -1


def _skip_ws(text: String, i: Int) -> Int:
    var bytes = text.as_bytes()
    var j = i
    while j < len(bytes) and (
        bytes[j] == 32 or bytes[j] == 9 or bytes[j] == 10 or bytes[j] == 13
    ):
        j += 1
    return j


def _hex_val(b: UInt8) raises -> Int:
    var v = Int(b)
    if 48 <= v and v <= 57:
        return v - 48
    if 97 <= v and v <= 102:
        return v - 87
    if 65 <= v and v <= 70:
        return v - 55
    raise Error("serve_bad_json: bad \\u escape hex digit")


def _parse_unicode_escape(text: String, i: Int) raises -> Tuple[Int, Int]:
    """Parse the 4 hex digits after a \\u at position i (i points
    at the 'u'); returns (codepoint, index past the digits),
    combining a following \\uXXXX low surrogate when present."""
    var bytes = text.as_bytes()
    if i + 4 >= len(bytes):
        raise Error("serve_bad_json: truncated \\u escape")
    var cp = 0
    for k in range(1, 5):
        cp = cp * 16 + _hex_val(bytes[i + k])
    var nxt = i + 5
    if 0xD800 <= cp and cp <= 0xDBFF:
        # High surrogate: try to combine with a following escape.
        if nxt + 5 < len(bytes) and bytes[nxt] == 92 and bytes[nxt + 1] == 117:
            var lo = 0
            for k in range(2, 6):
                lo = lo * 16 + _hex_val(bytes[nxt + k])
            if 0xDC00 <= lo and lo <= 0xDFFF:
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
                nxt += 6
    return (cp, nxt)


def parse_json_string_at(text: String, start: Int) raises -> Tuple[String, Int]:
    """Parse a JSON string whose opening quote is at `start`.

    Returns (value, index just past the closing quote). Refuses
    serve_bad_json on an unterminated string or a bad escape.
    """
    var bytes = text.as_bytes()
    if start >= len(bytes) or bytes[start] != 34:
        raise Error("serve_bad_json: expected a JSON string")
    var out = List[UInt8]()
    var i = start + 1
    while i < len(bytes):
        var b = bytes[i]
        if b == 34:
            return (_bytes_to_string(out), i + 1)
        if b == 92:  # backslash escape
            if i + 1 >= len(bytes):
                break
            var e = bytes[i + 1]
            if e == 34:
                out.append(UInt8(34))
                i += 2
            elif e == 92:
                out.append(UInt8(92))
                i += 2
            elif e == 47:  # \/
                out.append(UInt8(47))
                i += 2
            elif e == 98:  # \b
                out.append(UInt8(8))
                i += 2
            elif e == 102:  # \f
                out.append(UInt8(12))
                i += 2
            elif e == 110:  # \n
                out.append(UInt8(10))
                i += 2
            elif e == 114:  # \r
                out.append(UInt8(13))
                i += 2
            elif e == 116:  # \t
                out.append(UInt8(9))
                i += 2
            elif e == 117:  # \uXXXX
                var parsed = _parse_unicode_escape(text, i + 1)
                _append_utf8(parsed[0], out)
                i = parsed[1]
            else:
                raise Error("serve_bad_json: unknown escape in string")
        else:
            out.append(b)
            i += 1
    raise Error("serve_bad_json: unterminated JSON string")


def _key_value_start(body: String, key: String) raises -> Int:
    """Index of the first byte of the value for `"key":`, or -1
    when the key is absent. A `"key"` occurrence not followed
    by a colon is skipped (it was a value, not a field)."""
    var needle = String('"') + key + '"'
    var from_idx = 0
    while from_idx >= 0:
        var at = _find_substring(body, needle, from_idx)
        if at == -1:
            return -1
        var i = _skip_ws(body, at + needle.byte_length())
        if i < body.byte_length() and body.as_bytes()[i] == 58:  # ':'
            return _skip_ws(body, i + 1)
        from_idx = at + needle.byte_length()
    return -1


def has_json_key(body: String, key: String) raises -> Bool:
    return _key_value_start(body, key) != -1


def json_string_field(body: String, key: String) raises -> String:
    """The string value of a top-level field, by name.

    Refuses serve_missing_field when absent and serve_bad_json
    when the value is not a JSON string.
    """
    var at = _key_value_start(body, key)
    if at == -1:
        raise Error("serve_missing_field: no string field '" + key + "'")
    var parsed = parse_json_string_at(body, at)
    return parsed[0]


def json_int_field(body: String, key: String, default: Int) raises -> Int:
    """The integer value of a field, or `default` when absent.

    A present non-integer (float, string, bool) refuses
    serve_bad_json: silently truncating a caller's max_tokens
    would generate a different amount than they asked for.
    """
    var at = _key_value_start(body, key)
    if at == -1:
        return default
    var bytes = body.as_bytes()
    var i = at
    var neg = False
    if i < len(bytes) and bytes[i] == 45:  # '-'
        neg = True
        i += 1
    var value = 0
    var digits = 0
    while i < len(bytes) and 48 <= Int(bytes[i]) and Int(bytes[i]) <= 57:
        value = value * 10 + (Int(bytes[i]) - 48)
        digits += 1
        i += 1
    if digits == 0:
        raise Error("serve_bad_json: field '" + key + "' is not an integer")
    if i < len(bytes) and (bytes[i] == 46 or bytes[i] == 101 or bytes[i] == 69):
        raise Error("serve_bad_json: field '" + key + "' is not a whole number")
    return -value if neg else value


def json_float_field(body: String, key: String, default: Float64) raises -> Float64:
    """The numeric value of a field (integer or float form),
    or `default` when absent. Used for run-record fields a
    foreign bench may write either way (tok/s figures)."""
    var at = _key_value_start(body, key)
    if at == -1:
        return default
    var bytes = body.as_bytes()
    var i = _skip_ws(body, at)
    var start = i
    if i < len(bytes) and bytes[i] == 45:
        i += 1
    var saw_digit = False
    while i < len(bytes) and (
        (48 <= Int(bytes[i]) and Int(bytes[i]) <= 57)
        or bytes[i] == 46
        or bytes[i] == 101
        or bytes[i] == 69
        or bytes[i] == 43
        or bytes[i] == 45
    ):
        if 48 <= Int(bytes[i]) and Int(bytes[i]) <= 57:
            saw_digit = True
        i += 1
    if not saw_digit:
        raise Error("serve_bad_json: field '" + key + "' is not a number")
    var text = String(body[byte=start:i])
    try:
        return Float64(text)
    except:
        raise Error("serve_bad_json: field '" + key + "' is not a number")


def json_bool_field(body: String, key: String, default: Bool) raises -> Bool:
    var at = _key_value_start(body, key)
    if at == -1:
        return default
    var i = _skip_ws(body, at)
    if body[byte=i : i + 4] == "true":
        return True
    if body[byte=i : i + 5] == "false":
        return False
    raise Error("serve_bad_json: field '" + key + "' is not a boolean")


def json_raw_field(body: String, key: String) raises -> String:
    """The raw JSON text of a field's value ("" when absent):
    how an array/object field (sampler, attachments, rows) is
    lifted out of a record for a dedicated sub-parse."""
    var at = _key_value_start(body, key)
    if at == -1:
        return String("")
    var bytes = body.as_bytes()
    var i = _skip_ws(body, at)
    var start = i
    if i >= len(bytes):
        raise Error("serve_bad_json: field '" + key + "' has no value")
    var b = bytes[i]
    if b == 34:  # string
        var parsed = parse_json_string_at(body, i)
        return String(body[byte=start : parsed[1]])
    if b == 91 or b == 123:  # [ or {
        var open_c = b
        var close_c = UInt8(93) if b == 91 else UInt8(125)
        var depth = 0
        var in_str = False
        var esc = False
        while i < len(bytes):
            var c = bytes[i]
            if in_str:
                if esc:
                    esc = False
                elif c == 92:
                    esc = True
                elif c == 34:
                    in_str = False
            else:
                if c == 34:
                    in_str = True
                elif c == open_c:
                    depth += 1
                elif c == close_c:
                    depth -= 1
                    if depth == 0:
                        return String(body[byte=start : i + 1])
            i += 1
        raise Error("serve_bad_json: field '" + key + "' value is unbalanced")
    # scalar: up to the next comma or closing brace at this level
    while i < len(bytes) and bytes[i] != 44 and bytes[i] != 125:
        i += 1
    return String(body[byte=start:i])


def json_string_array_field(body: String, key: String) raises -> List[String]:
    """A string array field's values ([] when absent)."""
    var out = List[String]()
    var raw = json_raw_field(body, key)
    if raw == "":
        return out^
    var bytes = raw.as_bytes()
    var i = 0
    while i < len(bytes):
        if bytes[i] == 34:
            var parsed = parse_json_string_at(raw, i)
            out.append(parsed[0])
            i = parsed[1]
        else:
            i += 1
    return out^


def split_json_array(raw: String) raises -> List[String]:
    """Split a top-level JSON array text into its element
    texts (objects, arrays, scalars), honoring strings and
    nesting. Refuses serve_bad_json when the text is not a
    single array."""
    var out = List[String]()
    var bytes = raw.as_bytes()
    var i = _skip_ws(raw, 0)
    if i >= len(bytes) or bytes[i] != 91:
        raise Error("serve_bad_json: expected a JSON array")
    i += 1
    i = _skip_ws(raw, i)
    if i < len(bytes) and bytes[i] == 93:
        return out^
    while i < len(bytes):
        var start = i
        var c = bytes[i]
        if c == 34:
            var parsed = parse_json_string_at(raw, i)
            out.append(String(raw[byte=start : parsed[1]]))
            i = parsed[1]
        elif c == 123 or c == 91:
            var stack = List[UInt8]()
            stack.append(c)
            i += 1
            var in_str = False
            var esc = False
            while i < len(bytes) and len(stack) > 0:
                var cc = bytes[i]
                if in_str:
                    if esc:
                        esc = False
                    elif cc == 92:
                        esc = True
                    elif cc == 34:
                        in_str = False
                else:
                    if cc == 34:
                        in_str = True
                    elif cc == 123 or cc == 91:
                        stack.append(cc)
                    elif cc == 125 or cc == 93:
                        var want = UInt8(123) if cc == 125 else UInt8(91)
                        if stack[len(stack) - 1] != want:
                            raise Error(
                                "serve_bad_json: mismatched brackets in array"
                            )
                        _ = stack.pop()
                i += 1
            if len(stack) > 0:
                raise Error("serve_bad_json: unbalanced element in array")
            out.append(String(raw[byte=start:i]))
        else:
            while i < len(bytes) and bytes[i] != 44 and bytes[i] != 93:
                i += 1
            out.append(String(raw[byte=start:i]))
        i = _skip_ws(raw, i)
        if i < len(bytes) and bytes[i] == 44:
            i = _skip_ws(raw, i + 1)
            continue
        if i < len(bytes) and bytes[i] == 93:
            return out^
        raise Error("serve_bad_json: array element not followed by , or ]")
    raise Error("serve_bad_json: unterminated JSON array")


def json_content_strings(body: String) raises -> List[String]:
    """Every `"content": "..."` string value in order.

    This is how a chat `messages` array is read in v1: roles
    are not interpreted (the engine has no chat template; the
    texts are joined by the caller), only the contents carry.
    """
    var out = List[String]()
    var needle = String('"content"')
    var from_idx = 0
    while from_idx >= 0:
        var at = _find_substring(body, needle, from_idx)
        if at == -1:
            return out^
        var i = _skip_ws(body, at + needle.byte_length())
        if i < body.byte_length() and body.as_bytes()[i] == 58:
            var vstart = _skip_ws(body, i + 1)
            var parsed = parse_json_string_at(body, vstart)
            out.append(parsed[0])
            from_idx = parsed[1]
        else:
            from_idx = at + needle.byte_length()
    return out^
