# gguf_vocab.mojo - TEMPORARY SEAM: a minimal GGUF metadata
# read that produces TokenizerData, so the tokenizer lane can
# load the real Glimmer vocabulary in parallel with Lane A1
# (the GGUF loader proper, landing separately on this branch).
#
# *** COORDINATOR INTEGRATION NOTE ***
# This whole file is the temporary seam the tokenizer lane was
# chartered to build. At integration, delete it and produce
# TokenizerData from Lane A1's loader instead (the Rust shape:
# crates/gguf TokenizerInfo -> TokenizerData, as in
# crates/tokenizer/examples/conformance.rs's data_from_info).
# Nothing outside this file knows GGUF exists: tokenizer.mojo
# consumes TokenizerData only. The one interface to preserve is
# load_tokenizer_data(path) -> TokenizerData.
# *** END NOTE ***
#
# What it reads: the GGUF header (magic, version 3, counts) and
# the metadata key-values only, streamed through a small buffer
# so a 16 GB model file costs megabytes of memory, never the
# tensor section. Keys captured: tokenizer.ggml.model, .pre,
# .tokens, .token_type, .merges, .scores, .bos_token_id,
# .eos_token_id, .unknown_token_id, .padding_token_id (the
# crates/gguf TokenizerInfo set plus pre). Every other key's
# value is skipped by type. Named refusals:
# tokenizer_gguf_bad_magic, tokenizer_gguf_unsupported_version,
# tokenizer_gguf_truncated, tokenizer_gguf_bad_value_type,
# tokenizer_gguf_type_mismatch, tokenizer_missing_vocab_metadata
# (the file carries no tokenizer.ggml.tokens).

from diag import log_debug
from tokenizer.byte_level import decode_utf8_strict, string_from_codepoints
from tokenizer.tokenizer import TokenizerData

# GGUF metadata value types (gguf spec; crates/gguf/src/file.rs
# read_metadata_value).
comptime VT_UINT8 = 0
comptime VT_INT8 = 1
comptime VT_UINT16 = 2
comptime VT_INT16 = 3
comptime VT_UINT32 = 4
comptime VT_INT32 = 5
comptime VT_FLOAT32 = 6
comptime VT_BOOL = 7
comptime VT_STRING = 8
comptime VT_ARRAY = 9
comptime VT_UINT64 = 10
comptime VT_INT64 = 11
comptime VT_FLOAT64 = 12

comptime CHUNK = 65536


struct GgufReader(Movable):
    """A buffered streaming reader over a GGUF file: the parse
    only ever holds one chunk plus the value being read."""

    var file: FileHandle
    var buf: List[UInt8]
    var pos: Int
    var eof: Bool

    def __init__(out self, path: String) raises:
        self.file = open(path, "r")
        self.buf = List[UInt8]()
        self.pos = 0
        self.eof = False

    def close(mut self) raises:
        self.file.close()

    def _fill(mut self) raises:
        if self.eof:
            return
        if self.pos == len(self.buf):
            self.buf = List[UInt8]()
            self.pos = 0
        elif self.pos >= CHUNK:
            var rest = List[UInt8]()
            for i in range(self.pos, len(self.buf)):
                rest.append(self.buf[i])
            self.buf = rest^
            self.pos = 0
        var chunk = self.file.read_bytes(CHUNK)
        if len(chunk) == 0:
            self.eof = True
            return
        for b in chunk:
            self.buf.append(b)

    def _need(mut self, n: Int) raises -> Bool:
        """Ensure at least n buffered bytes or hit EOF."""
        while len(self.buf) - self.pos < n and not self.eof:
            self._fill()
        return len(self.buf) - self.pos >= n

    def read_exact(mut self, n: Int) raises -> List[UInt8]:
        if not self._need(n):
            raise Error(
                "tokenizer_gguf_truncated: GGUF metadata ended early (wanted "
                + String(n)
                + " more bytes)"
            )
        var out = List[UInt8]()
        for k in range(n):
            out.append(self.buf[self.pos + k])
        self.pos += n
        return out^

    def skip_exact(mut self, n: Int) raises:
        var remaining = n
        while remaining > 0:
            if not self._need(1):
                raise Error(
                    "tokenizer_gguf_truncated: GGUF metadata ended"
                    " early while skipping a value"
                )
            var avail = len(self.buf) - self.pos
            var take = avail if avail < remaining else remaining
            self.pos += take
            remaining -= take

    def read_u8(mut self) raises -> Int:
        var b = self.read_exact(1)
        return Int(b[0])

    def read_u32(mut self) raises -> Int:
        var b = self.read_exact(4)
        return (
            Int(b[0]) | (Int(b[1]) << 8) | (Int(b[2]) << 16) | (Int(b[3]) << 24)
        )

    def read_u64_int(mut self) raises -> Int:
        var b = self.read_exact(8)
        var v = 0
        for i in range(8):
            v |= Int(b[i]) << (8 * i)
        return v

    def read_string(mut self) raises -> String:
        var n = self.read_u64_int()
        if n < 0 or n > 1_000_000_000:
            raise Error(
                "tokenizer_gguf_truncated: implausible GGUF string length "
                + String(n)
            )
        var bytes = self.read_exact(n)
        var cps = decode_utf8_strict(bytes)
        return string_from_codepoints(cps, 0, len(cps))

    def skip_string(mut self) raises:
        var n = self.read_u64_int()
        if n < 0 or n > 4_000_000_000_000:
            raise Error(
                "tokenizer_gguf_truncated: implausible GGUF string length "
                + String(n)
            )
        self.skip_exact(n)


def _scalar_size(type_id: Int) raises -> Int:
    if type_id == VT_UINT8 or type_id == VT_INT8 or type_id == VT_BOOL:
        return 1
    if type_id == VT_UINT16 or type_id == VT_INT16:
        return 2
    if type_id == VT_UINT32 or type_id == VT_INT32 or type_id == VT_FLOAT32:
        return 4
    if type_id == VT_UINT64 or type_id == VT_INT64 or type_id == VT_FLOAT64:
        return 8
    raise Error(
        "tokenizer_gguf_bad_value_type: not a scalar GGUF value type: "
        + String(type_id)
    )


def _read_scalar_int(mut r: GgufReader, type_id: Int) raises -> Int:
    var size = _scalar_size(type_id)
    var b = r.read_exact(size)
    if type_id == VT_INT8:
        var v = Int(b[0])
        return v - 256 if v >= 128 else v
    if type_id == VT_INT16:
        var v = Int(b[0]) | (Int(b[1]) << 8)
        return v - 65536 if v >= 32768 else v
    if type_id == VT_INT32:
        var v = (
            Int(b[0]) | (Int(b[1]) << 8) | (Int(b[2]) << 16) | (Int(b[3]) << 24)
        )
        return v - 4294967296 if v >= 2147483648 else v
    if type_id == VT_INT64:
        return r_u64_signed(b)
    var v = 0
    for i in range(size):
        v |= Int(b[i]) << (8 * i)
    return v


def r_u64_signed(b: List[UInt8]) -> Int:
    # Two's complement read on a 64-bit Int host: accumulate in
    # wrapping arithmetic by building the value as unsigned
    # then reinterpreting via subtraction when the top bit is
    # set (token ids never come near this; it exists so INT64
    # metadata skips/reads stay honest).
    var v = UInt64(0)
    for i in range(8):
        v |= UInt64(b[i]) << UInt64(8 * i)
    if v >= UInt64(0x8000000000000000):
        return -Int((~v) + 1)
    return Int(v)


def _read_scalar_f64(mut r: GgufReader, type_id: Int) raises -> Float64:
    if type_id == VT_FLOAT32:
        var b = r.read_exact(4)
        var words = List[UInt32](capacity=1)
        var bits = UInt32(0)
        for i in range(4):
            bits |= UInt32(b[i]) << UInt32(8 * i)
        words.append(bits)
        var f = words.unsafe_ptr().unsafe_bitcast[Float32]()[unsafe_offset=0]
        return Float64(f)
    if type_id == VT_FLOAT64:
        var b = r.read_exact(8)
        var words = List[UInt64](capacity=1)
        var bits = UInt64(0)
        for i in range(8):
            bits |= UInt64(b[i]) << UInt64(8 * i)
        words.append(bits)
        var f = words.unsafe_ptr().unsafe_bitcast[Float64]()[unsafe_offset=0]
        return f
    if (
        type_id == VT_UINT8
        or type_id == VT_INT8
        or type_id == VT_UINT16
        or (type_id == VT_INT16)
        or type_id == VT_UINT32
        or type_id == VT_INT32
        or type_id == VT_UINT64
        or (type_id == VT_INT64)
        or type_id == VT_BOOL
    ):
        return Float64(_read_scalar_int(r, type_id))
    raise Error(
        "tokenizer_gguf_bad_value_type: not a numeric GGUF value type: "
        + String(type_id)
    )


def _skip_scalar(mut r: GgufReader, type_id: Int) raises:
    if type_id == VT_STRING:
        r.skip_string()
        return
    var size = _scalar_size(type_id)
    r.skip_exact(size)


def _skip_value(mut r: GgufReader) raises:
    var type_id = r.read_u32()
    if type_id == VT_ARRAY:
        var elem = r.read_u32()
        var count = r.read_u64_int()
        if count < 0:
            raise Error("tokenizer_gguf_truncated: negative array length")
        for _ in range(count):
            _skip_scalar(r, elem)
        return
    _skip_scalar(r, type_id)


def _read_string_value(mut r: GgufReader, key: String) raises -> String:
    var type_id = r.read_u32()
    if type_id != VT_STRING:
        raise Error(
            "tokenizer_gguf_type_mismatch: "
            + key
            + " must be a GGUF string, found value type "
            + String(type_id)
        )
    return r.read_string()


def _read_string_array(mut r: GgufReader, key: String) raises -> List[String]:
    var type_id = r.read_u32()
    if type_id != VT_ARRAY:
        raise Error(
            "tokenizer_gguf_type_mismatch: "
            + key
            + " must be a GGUF array, found value type "
            + String(type_id)
        )
    var elem = r.read_u32()
    if elem != VT_STRING:
        raise Error(
            "tokenizer_gguf_type_mismatch: "
            + key
            + " must be an array of strings, found element type "
            + String(elem)
        )
    var count = r.read_u64_int()
    if count < 0:
        raise Error("tokenizer_gguf_truncated: negative array length")
    var out = List[String]()
    for _ in range(count):
        out.append(r.read_string())
    return out^


def _read_int_array(mut r: GgufReader, key: String) raises -> List[Int]:
    var type_id = r.read_u32()
    if type_id != VT_ARRAY:
        raise Error(
            "tokenizer_gguf_type_mismatch: "
            + key
            + " must be a GGUF array, found value type "
            + String(type_id)
        )
    var elem = r.read_u32()
    var count = r.read_u64_int()
    if count < 0:
        raise Error("tokenizer_gguf_truncated: negative array length")
    var out = List[Int]()
    for _ in range(count):
        out.append(_read_scalar_int(r, elem))
    return out^


def _read_float_array(mut r: GgufReader, key: String) raises -> List[Float64]:
    var type_id = r.read_u32()
    if type_id != VT_ARRAY:
        raise Error(
            "tokenizer_gguf_type_mismatch: "
            + key
            + " must be a GGUF array, found value type "
            + String(type_id)
        )
    var elem = r.read_u32()
    var count = r.read_u64_int()
    if count < 0:
        raise Error("tokenizer_gguf_truncated: negative array length")
    var out = List[Float64]()
    for _ in range(count):
        out.append(_read_scalar_f64(r, elem))
    return out^


def _read_int_value(mut r: GgufReader, key: String) raises -> Int:
    var type_id = r.read_u32()
    if type_id == VT_STRING or type_id == VT_ARRAY:
        raise Error(
            "tokenizer_gguf_type_mismatch: "
            + key
            + " must be a GGUF integer, found value type "
            + String(type_id)
        )
    return _read_scalar_int(r, type_id)


def load_tokenizer_data(path: String) raises -> TokenizerData:
    """TEMPORARY SEAM (see this file's header): read the
    tokenizer metadata from the GGUF file at `path` into
    TokenizerData. Raises the named refusals in the header on
    anything malformed; a file with no tokenizer.ggml.tokens
    refuses with tokenizer_missing_vocab_metadata."""
    var r = GgufReader(path)
    var magic = r.read_exact(4)
    if (
        Int(magic[0]) != 0x47  # 'G'
        or Int(magic[1]) != 0x47
        or Int(magic[2]) != 0x55  # 'U'
        or Int(magic[3]) != 0x46  # 'F'
    ):
        raise Error(
            "tokenizer_gguf_bad_magic: "
            + path
            + " does not start with the GGUF magic"
        )
    var version = r.read_u32()
    if version != 3:
        raise Error(
            "tokenizer_gguf_unsupported_version: GGUF version "
            + String(version)
            + " (the Rust reference loader supports version 3)"
        )
    _ = r.read_u64_int()  # tensor_count: the seam stops at metadata
    var kv_count = r.read_u64_int()
    if kv_count < 0:
        raise Error("tokenizer_gguf_truncated: negative metadata count")
    var data = TokenizerData()
    for _ in range(kv_count):
        var key = r.read_string()
        if key == "tokenizer.ggml.model":
            data.model = _read_string_value(r, key)
            data.has_model = True
        elif key == "tokenizer.ggml.pre":
            data.pre = _read_string_value(r, key)
            data.has_pre = True
        elif key == "tokenizer.ggml.tokens":
            data.tokens = _read_string_array(r, key)
        elif key == "tokenizer.ggml.token_type":
            data.token_types = _read_int_array(r, key)
        elif key == "tokenizer.ggml.merges":
            data.merges = _read_string_array(r, key)
        elif key == "tokenizer.ggml.scores":
            data.scores = _read_float_array(r, key)
            data.has_scores = True
        elif key == "tokenizer.ggml.bos_token_id":
            data.bos_token_id = _read_int_value(r, key)
        elif key == "tokenizer.ggml.eos_token_id":
            data.eos_token_id = _read_int_value(r, key)
        elif key == "tokenizer.ggml.unknown_token_id":
            data.unknown_token_id = _read_int_value(r, key)
        elif key == "tokenizer.ggml.padding_token_id":
            data.padding_token_id = _read_int_value(r, key)
        else:
            _skip_value(r)
    r.close()
    if len(data.tokens) == 0:
        _ = log_debug(
            "tokenizer",
            "refusing: no tokenizer.ggml.tokens in " + path,
        )
        raise Error(
            "tokenizer_missing_vocab_metadata: GGUF file "
            + path
            + " carries no tokenizer.ggml.tokens metadata"
        )
    _ = log_debug(
        "tokenizer",
        "gguf vocab seam loaded tokens="
        + String(len(data.tokens))
        + " merges="
        + String(len(data.merges))
        + " from "
        + path,
    )
    return data^
