# loader.mojo - the GGUF loader (Mojo/MAX engine, DESIGN.md step 1).
#
# A real loader mirroring the paused Rust loader on main
# (crates/gguf: file.rs, metadata.rs, ggml.rs, error.rs) in
# semantics, metadata keys, and error-naming spirit: every failure
# is a typed refusal BY NAME, and no path ever substitutes
# placeholder data for bytes it could not read.
#
# Tensor data access choice: STAGED FILE READ, not mmap.
# Mojo 1.2's standard library has no mmap binding, and MAX's
# host-buffer APIs tie memory to a device context, which a pure
# host file loader must not require. So the directory and
# metadata are parsed through a buffered sequential reader, and
# tensor bytes are served by seek + chunked read on demand
# (read_tensor_chunk / tensor_bytes), reopening the file per
# call so a parsed GgufFile is plain data with no live handle.
# The Rust loader mmaps; the observable semantics (same bytes,
# same refusals) are identical, only the residency differs:
# staged reads cost a syscall per chunk instead of page faults.
#
# Interface note (LOUD, per the foundation contract): the fixed
# seam from the foundation lane is preserved exactly:
# load_model(path) raises -> None still parses-and-validates or
# refuses by name, and loader_state() now returns "implemented"
# instead of "not_implemented" because the loader exists. The
# rich API (parse_gguf + GgufFile getters) is additive.

from diag import log_debug

# GGUF format constants (spec-level, mirrored from the Rust
# loader: GGUF_MAGIC, SUPPORTED_VERSION, DEFAULT_ALIGNMENT).
# These are format facts, not operational limits; operational
# configuration lives in src/config.mojo as everywhere else.
comptime GGUF_VERSION = 3
comptime DEFAULT_ALIGNMENT = 32

# IO staging sizes (implementation detail of the staged-read
# choice above, not operational limits): the header reader's
# refill size and the tensor chunk size for tensor_bytes().
comptime HEADER_BUFFER_BYTES = 262144
comptime STAGED_READ_CHUNK_BYTES = 4194304

# GGUF metadata value type ids (GGUF spec / Rust metadata.rs).
comptime META_U8 = 0
comptime META_I8 = 1
comptime META_U16 = 2
comptime META_I16 = 3
comptime META_U32 = 4
comptime META_I32 = 5
comptime META_F32 = 6
comptime META_BOOL = 7
comptime META_STRING = 8
comptime META_ARRAY = 9
comptime META_U64 = 10
comptime META_I64 = 11
comptime META_F64 = 12


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def f32_from_bits(bits: UInt32) -> Float32:
    var tmp = List[UInt32](capacity=1)
    tmp.append(bits)
    return tmp.unsafe_ptr().unsafe_bitcast[Float32]()[unsafe_offset=0]


def f64_from_bits(bits: UInt64) -> Float64:
    var tmp = List[UInt64](capacity=1)
    tmp.append(bits)
    return tmp.unsafe_ptr().unsafe_bitcast[Float64]()[unsafe_offset=0]


def u64_to_int_checked(v: UInt64, context: String) raises -> Int:
    if v > UInt64(4611686018427387904):
        raise Error(
            "gguf_length_overflow: "
            + context
            + " length "
            + String(v)
            + " cannot fit in memory"
        )
    return Int(v)


def align_up(value: Int, alignment: Int) -> Int:
    return ((value + alignment - 1) // alignment) * alignment


# ---------------------------------------------------------------------------
# ggml tensor types (mainline ids, mirrored from crates/gguf/src/ggml.rs;
# id 30 is BF16, never I2_S: that collision killed the old Go loader)
# ---------------------------------------------------------------------------


def ggml_type_name(type_id: Int) -> String:
    if type_id == 0:
        return "F32"
    if type_id == 1:
        return "F16"
    if type_id == 2:
        return "Q4_0"
    if type_id == 3:
        return "Q4_1"
    if type_id == 6:
        return "Q5_0"
    if type_id == 7:
        return "Q5_1"
    if type_id == 8:
        return "Q8_0"
    if type_id == 9:
        return "Q8_1"
    if type_id == 10:
        return "Q2_K"
    if type_id == 11:
        return "Q3_K"
    if type_id == 12:
        return "Q4_K"
    if type_id == 13:
        return "Q5_K"
    if type_id == 14:
        return "Q6_K"
    if type_id == 15:
        return "Q8_K"
    if type_id == 16:
        return "IQ2_XXS"
    if type_id == 17:
        return "IQ2_XS"
    if type_id == 18:
        return "IQ3_XXS"
    if type_id == 19:
        return "IQ1_S"
    if type_id == 20:
        return "IQ4_NL"
    if type_id == 21:
        return "IQ3_S"
    if type_id == 22:
        return "IQ2_S"
    if type_id == 23:
        return "IQ4_XS"
    if type_id == 24:
        return "I8"
    if type_id == 25:
        return "I16"
    if type_id == 26:
        return "I32"
    if type_id == 27:
        return "I64"
    if type_id == 28:
        return "F64"
    if type_id == 29:
        return "IQ1_M"
    if type_id == 30:
        return "BF16"
    if type_id == 34:
        return "TQ1_0"
    if type_id == 35:
        return "TQ2_0"
    if type_id == 39:
        return "MXFP4"
    return "unknown"


def ggml_supported(type_id: Int) -> Bool:
    """True for the types this loader can size (Rust GgmlType set)."""
    return (
        type_id == 0
        or type_id == 1
        or type_id == 8
        or type_id == 10
        or type_id == 11
        or type_id == 12
        or type_id == 13
        or type_id == 14
        or type_id == 30
    )


def ggml_block_size(type_id: Int) -> Int:
    """Elements per quantization block; 1 for unquantized types."""
    if type_id == 8:
        return 32
    if (
        type_id == 10
        or type_id == 11
        or type_id == 12
        or type_id == 13
        or type_id == 14
    ):
        return 256
    return 1


def ggml_block_bytes(type_id: Int) -> Int:
    """Bytes one block occupies in the file (supported types only)."""
    if type_id == 0:
        return 4
    if type_id == 1 or type_id == 30:
        return 2
    if type_id == 8:
        return 34
    if type_id == 10:
        return 84
    if type_id == 11:
        return 110
    if type_id == 12:
        return 144
    if type_id == 13:
        return 176
    if type_id == 14:
        return 210
    return 0


def ggml_byte_len(type_id: Int, elements: Int) -> Int:
    """On-disk byte length, or -1 when unsupported / not whole blocks."""
    if not ggml_supported(type_id):
        return -1
    var block = ggml_block_size(type_id)
    if elements % block != 0:
        return -1
    return (elements // block) * ggml_block_bytes(type_id)


def dtype_refusal(name: String, type_id: Int) -> String:
    """The named refusal for serving bytes of an unsized dtype.

    Unassigned ids refuse as gguf_unknown_dtype; assigned but
    unsupported ids as gguf_unsupported_tensor_type (Rust spirit:
    UnsupportedTensorType naming the type the file contains).
    """
    var tname = ggml_type_name(type_id)
    if tname == "unknown":
        return (
            "gguf_unknown_dtype: tensor "
            + name
            + " has ggml type id "
            + String(type_id)
            + " which mainline ggml does not assign; refusing to"
            + " guess its byte layout"
        )
    return (
        "gguf_unsupported_tensor_type: tensor "
        + name
        + " is "
        + tname
        + " (ggml type id "
        + String(type_id)
        + "), which this loader does not size or serve"
    )


# ---------------------------------------------------------------------------
# Metadata values (flat: GGUF arrays are homogeneous and cannot nest,
# so array payloads live in one typed list per element kind)
# ---------------------------------------------------------------------------


struct MetadataValue(Copyable, Movable):
    var type_id: Int
    var uint_value: UInt64
    var int_value: Int64
    var float_value: Float64
    var bool_value: Bool
    var string_value: String
    var array_elem_type: Int
    var array_len: Int
    var array_uints: List[UInt64]
    var array_ints: List[Int64]
    var array_floats: List[Float64]
    var array_bools: List[Bool]
    var array_strings: List[String]

    def __init__(out self, type_id: Int):
        self.type_id = type_id
        self.uint_value = UInt64(0)
        self.int_value = Int64(0)
        self.float_value = Float64(0.0)
        self.bool_value = False
        self.string_value = String("")
        self.array_elem_type = -1
        self.array_len = 0
        self.array_uints = List[UInt64]()
        self.array_ints = List[Int64]()
        self.array_floats = List[Float64]()
        self.array_bools = List[Bool]()
        self.array_strings = List[String]()

    def type_name(self) -> String:
        if self.type_id == META_U8:
            return "u8"
        if self.type_id == META_I8:
            return "i8"
        if self.type_id == META_U16:
            return "u16"
        if self.type_id == META_I16:
            return "i16"
        if self.type_id == META_U32:
            return "u32"
        if self.type_id == META_I32:
            return "i32"
        if self.type_id == META_F32:
            return "f32"
        if self.type_id == META_BOOL:
            return "bool"
        if self.type_id == META_STRING:
            return "string"
        if self.type_id == META_ARRAY:
            return "array"
        if self.type_id == META_U64:
            return "u64"
        if self.type_id == META_I64:
            return "i64"
        if self.type_id == META_F64:
            return "f64"
        return "unknown"


struct MetadataEntry(Copyable, Movable):
    var key: String
    var value: MetadataValue

    def __init__(out self, key: String, var value: MetadataValue):
        self.key = key
        self.value = value^


struct TensorInfo(Copyable, Movable):
    var name: String
    var dims: List[Int]
    var type_id: Int
    var rel_offset: Int
    var elements: Int

    def __init__(
        out self,
        name: String,
        var dims: List[Int],
        type_id: Int,
        rel_offset: Int,
        elements: Int,
    ):
        self.name = name
        self.dims = dims^
        self.type_id = type_id
        self.rel_offset = rel_offset
        self.elements = elements

    def type_name(self) -> String:
        return ggml_type_name(self.type_id)

    def byte_len(self) -> Int:
        return ggml_byte_len(self.type_id, self.elements)


# ---------------------------------------------------------------------------
# Buffered sequential reader over the header region
# ---------------------------------------------------------------------------


struct Reader(Movable):
    var handle: FileHandle
    var file_len: Int
    var pos: Int
    var buffer: List[UInt8]
    var buf_start: Int

    def __init__(out self, var handle: FileHandle, file_len: Int):
        self.handle = handle^
        self.file_len = file_len
        self.pos = 0
        self.buffer = List[UInt8]()
        self.buf_start = 0

    def remaining(self) -> Int:
        return self.file_len - self.pos

    def fill(mut self) raises -> Bool:
        """Refill the buffer at the logical position; False at EOF."""
        if self.pos < self.buf_start + len(self.buffer):
            return True
        var chunk = self.handle.read_bytes(HEADER_BUFFER_BYTES)
        if len(chunk) == 0:
            return False
        self.buf_start = self.pos
        self.buffer = chunk^
        return True

    def read_exact(mut self, n: Int, context: String) raises -> List[UInt8]:
        if n < 0 or self.pos + n > self.file_len:
            raise Error(
                "gguf_truncated: file ended while reading "
                + context
                + " at offset "
                + String(self.pos)
                + ": needed "
                + String(n)
                + " bytes, file is "
                + String(self.file_len)
                + " bytes"
            )
        var out = List[UInt8](capacity=n)
        var got = 0
        while got < n:
            if not self.fill():
                raise Error(
                    "gguf_truncated: file ended while reading "
                    + context
                    + " at offset "
                    + String(self.pos)
                    + ": needed "
                    + String(n)
                    + " bytes, file is "
                    + String(self.file_len)
                    + " bytes"
                )
            var avail = self.buf_start + len(self.buffer) - self.pos
            var take = avail
            if take > n - got:
                take = n - got
            var base = self.pos - self.buf_start
            for i in range(take):
                out.append(self.buffer[base + i])
            self.pos += take
            got += take
        return out^

    def read_u8(mut self, context: String) raises -> Int:
        var b = self.read_exact(1, context)
        return Int(b[0])

    def read_i8(mut self, context: String) raises -> Int:
        var v = self.read_u8(context)
        return v - 256 if v >= 128 else v

    def read_u16(mut self, context: String) raises -> Int:
        var b = self.read_exact(2, context)
        return Int(b[0]) | (Int(b[1]) << 8)

    def read_i16(mut self, context: String) raises -> Int:
        var v = self.read_u16(context)
        return v - 65536 if v >= 32768 else v

    def read_u32(mut self, context: String) raises -> UInt32:
        var b = self.read_exact(4, context)
        return (
            UInt32(b[0])
            | (UInt32(b[1]) << 8)
            | (UInt32(b[2]) << 16)
            | (UInt32(b[3]) << 24)
        )

    def read_i32(mut self, context: String) raises -> Int:
        var u = self.read_u32(context)
        var v = Int64(u)
        return Int(v - 4294967296) if v >= 2147483648 else Int(v)

    def read_u64(mut self, context: String) raises -> UInt64:
        var b = self.read_exact(8, context)
        var v = UInt64(0)
        for i in range(8):
            v |= UInt64(b[i]) << UInt64(8 * i)
        return v

    def read_i64(mut self, context: String) raises -> Int64:
        var u = self.read_u64(context)
        var tmp = List[UInt64](capacity=1)
        tmp.append(u)
        return tmp.unsafe_ptr().unsafe_bitcast[Int64]()[unsafe_offset=0]

    def read_f32(mut self, context: String) raises -> Float32:
        return f32_from_bits(self.read_u32(context))

    def read_f64(mut self, context: String) raises -> Float64:
        return f64_from_bits(self.read_u64(context))

    def read_bool(mut self, context: String) raises -> Bool:
        return self.read_u8(context) != 0

    def read_string(mut self, context: String) raises -> String:
        var len_u = self.read_u64("string length")
        var n = u64_to_int_checked(len_u, context)
        var raw = self.read_exact(n, context)
        try:
            return String(from_utf8=raw)
        except:
            raise Error("gguf_invalid_utf8: " + context + " is not valid UTF-8")


def read_metadata_scalar(
    mut reader: Reader, type_id: Int
) raises -> MetadataValue:
    """One non-array metadata value (array elements land here too)."""
    if type_id == META_U8:
        var v = MetadataValue(META_U8)
        v.uint_value = UInt64(reader.read_u8("metadata u8"))
        return v^
    if type_id == META_I8:
        var v = MetadataValue(META_I8)
        v.int_value = Int64(reader.read_i8("metadata i8"))
        return v^
    if type_id == META_U16:
        var v = MetadataValue(META_U16)
        v.uint_value = UInt64(reader.read_u16("metadata u16"))
        return v^
    if type_id == META_I16:
        var v = MetadataValue(META_I16)
        v.int_value = Int64(reader.read_i16("metadata i16"))
        return v^
    if type_id == META_U32:
        var v = MetadataValue(META_U32)
        v.uint_value = UInt64(reader.read_u32("metadata u32"))
        return v^
    if type_id == META_I32:
        var v = MetadataValue(META_I32)
        v.int_value = Int64(reader.read_i32("metadata i32"))
        return v^
    if type_id == META_F32:
        var v = MetadataValue(META_F32)
        v.float_value = Float64(reader.read_f32("metadata f32"))
        return v^
    if type_id == META_BOOL:
        var v = MetadataValue(META_BOOL)
        v.bool_value = reader.read_bool("metadata bool")
        return v^
    if type_id == META_STRING:
        var v = MetadataValue(META_STRING)
        v.string_value = reader.read_string("metadata string")
        return v^
    if type_id == META_U64:
        var v = MetadataValue(META_U64)
        v.uint_value = reader.read_u64("metadata u64")
        return v^
    if type_id == META_I64:
        var v = MetadataValue(META_I64)
        v.int_value = reader.read_i64("metadata i64")
        return v^
    if type_id == META_F64:
        var v = MetadataValue(META_F64)
        v.float_value = reader.read_f64("metadata f64")
        return v^
    raise Error(
        "gguf_unknown_metadata_type: metadata value type id "
        + String(type_id)
        + " is not a GGUF value type"
    )


def read_metadata_value(mut reader: Reader) raises -> MetadataValue:
    var type_id = Int(reader.read_u32("metadata value type"))
    if type_id != META_ARRAY:
        return read_metadata_scalar(reader, type_id)
    var elem_type = Int(reader.read_u32("metadata array element type"))
    var len_u = reader.read_u64("metadata array length")
    # Every element costs at least one byte on disk (Rust rule).
    if len_u > UInt64(reader.remaining()):
        raise Error(
            "gguf_truncated: file ended while reading metadata array at offset "
            + String(reader.pos)
            + ": needed "
            + String(len_u)
            + " bytes, file is "
            + String(reader.file_len)
            + " bytes"
        )
    var n = u64_to_int_checked(len_u, "metadata array")
    var v = MetadataValue(META_ARRAY)
    v.array_elem_type = elem_type
    v.array_len = n
    for _ in range(n):
        var item = read_metadata_scalar(reader, elem_type)
        if (
            elem_type == META_U8
            or elem_type == META_U16
            or elem_type == META_U32
            or elem_type == META_U64
        ):
            v.array_uints.append(item.uint_value)
        elif (
            elem_type == META_I8
            or elem_type == META_I16
            or elem_type == META_I32
            or elem_type == META_I64
        ):
            v.array_ints.append(item.int_value)
        elif elem_type == META_F32 or elem_type == META_F64:
            v.array_floats.append(item.float_value)
        elif elem_type == META_BOOL:
            v.array_bools.append(item.bool_value)
        elif elem_type == META_STRING:
            v.array_strings.append(item.string_value)
        else:
            raise Error(
                "gguf_unknown_metadata_type: metadata array element type id "
                + String(elem_type)
                + " is not a GGUF value type"
            )
    return v^


# ---------------------------------------------------------------------------
# The parsed file
# ---------------------------------------------------------------------------


struct GgufFile(Copyable, Movable):
    var path: String
    var version: Int
    var alignment: Int
    var data_start: Int
    var file_len: Int
    var metadata: List[MetadataEntry]
    var tensors: List[TensorInfo]

    def __init__(
        out self,
        path: String,
        version: Int,
        alignment: Int,
        data_start: Int,
        file_len: Int,
        var metadata: List[MetadataEntry],
        var tensors: List[TensorInfo],
    ):
        self.path = path
        self.version = version
        self.alignment = alignment
        self.data_start = data_start
        self.file_len = file_len
        self.metadata = metadata^
        self.tensors = tensors^

    # -- inventory ------------------------------------------------------

    def tensor_count(self) -> Int:
        return len(self.tensors)

    def metadata_count(self) -> Int:
        return len(self.metadata)

    def tensor_name(self, idx: Int) -> String:
        return self.tensors[idx].copy().name

    # -- metadata lookup --------------------------------------------------

    def find_metadata(self, key: String) -> Int:
        """Entry index for key, or -1 when absent (file order)."""
        for i in range(len(self.metadata)):
            if self.metadata[i].key == key:
                return i
        return -1

    def has_metadata(self, key: String) -> Bool:
        return self.find_metadata(key) != -1

    def require_metadata(self, key: String) raises -> Int:
        var idx = self.find_metadata(key)
        if idx == -1:
            raise Error("gguf_missing_metadata: no metadata key " + key)
        return idx

    def mismatch(self, key: String, expected: String) raises -> Error:
        var idx = self.find_metadata(key)
        if idx == -1:
            return Error("gguf_missing_metadata: no metadata key " + key)
        return Error(
            "gguf_metadata_type_mismatch: key "
            + key
            + " holds "
            + self.metadata[idx].value.type_name()
            + ", not "
            + expected
        )

    def get_u64(self, key: String) raises -> UInt64:
        var idx = self.require_metadata(key)
        var tid = self.metadata[idx].value.type_id
        if (
            tid == META_U8
            or tid == META_U16
            or tid == META_U32
            or tid == META_U64
        ):
            return self.metadata[idx].value.uint_value
        if (
            tid == META_I8
            or tid == META_I16
            or tid == META_I32
            or tid == META_I64
        ):
            if self.metadata[idx].value.int_value >= 0:
                return UInt64(self.metadata[idx].value.int_value)
        var _merr = self.mismatch(key, "u64")
        raise _merr

    def get_u32(self, key: String) raises -> UInt32:
        var v = self.get_u64(key)
        if v > UInt64(4294967295):
            var _merr = self.mismatch(key, "u32")
            raise _merr
        return UInt32(v)

    def get_i64(self, key: String) raises -> Int64:
        var idx = self.require_metadata(key)
        var tid = self.metadata[idx].value.type_id
        if (
            tid == META_I8
            or tid == META_I16
            or tid == META_I32
            or tid == META_I64
        ):
            return self.metadata[idx].value.int_value
        if tid == META_U8 or tid == META_U16 or tid == META_U32:
            return Int64(self.metadata[idx].value.uint_value)
        if tid == META_U64 and self.metadata[idx].value.uint_value <= UInt64(
            9223372036854775807
        ):
            return Int64(self.metadata[idx].value.uint_value)
        var _merr = self.mismatch(key, "i64")
        raise _merr

    def get_f32(self, key: String) raises -> Float32:
        var idx = self.require_metadata(key)
        var tid = self.metadata[idx].value.type_id
        if tid == META_F32 or tid == META_F64:
            return Float32(self.metadata[idx].value.float_value)
        var _merr = self.mismatch(key, "f32")
        raise _merr

    def get_f64(self, key: String) raises -> Float64:
        var idx = self.require_metadata(key)
        var tid = self.metadata[idx].value.type_id
        if tid == META_F32 or tid == META_F64:
            return self.metadata[idx].value.float_value
        var _merr = self.mismatch(key, "f64")
        raise _merr

    def get_bool(self, key: String) raises -> Bool:
        var idx = self.require_metadata(key)
        if self.metadata[idx].value.type_id != META_BOOL:
            var _merr = self.mismatch(key, "bool")
            raise _merr
        return self.metadata[idx].value.bool_value

    def get_string(self, key: String) raises -> String:
        var idx = self.require_metadata(key)
        if self.metadata[idx].value.type_id != META_STRING:
            var _merr = self.mismatch(key, "string")
            raise _merr
        return self.metadata[idx].value.string_value

    def architecture(self) raises -> String:
        return self.get_string("general.architecture")

    # -- metadata arrays ----------------------------------------------------

    def metadata_array_len(self, key: String) raises -> Int:
        var idx = self.require_metadata(key)
        if self.metadata[idx].value.type_id != META_ARRAY:
            var _merr = self.mismatch(key, "array")
            raise _merr
        return self.metadata[idx].value.array_len

    def metadata_array_string(self, key: String, i: Int) raises -> String:
        var idx = self.require_metadata(key)
        if (
            self.metadata[idx].value.type_id != META_ARRAY
            or self.metadata[idx].value.array_elem_type != META_STRING
        ):
            var _merr = self.mismatch(key, "string array")
            raise _merr
        if i < 0 or i >= len(self.metadata[idx].value.array_strings):
            raise Error(
                "gguf_invalid_format: array index "
                + String(i)
                + " out of range for metadata key "
                + key
            )
        return self.metadata[idx].value.array_strings[i]

    def metadata_array_u64(self, key: String, i: Int) raises -> UInt64:
        var idx = self.require_metadata(key)
        if self.metadata[idx].value.type_id != META_ARRAY:
            var _merr = self.mismatch(key, "array")
            raise _merr
        var et = self.metadata[idx].value.array_elem_type
        if et == META_U8 or et == META_U16 or et == META_U32 or et == META_U64:
            if i < 0 or i >= len(self.metadata[idx].value.array_uints):
                raise Error(
                    "gguf_invalid_format: array index out of range for " + key
                )
            return self.metadata[idx].value.array_uints[i]
        if et == META_I8 or et == META_I16 or et == META_I32 or et == META_I64:
            if i < 0 or i >= len(self.metadata[idx].value.array_ints):
                raise Error(
                    "gguf_invalid_format: array index out of range for " + key
                )
            if self.metadata[idx].value.array_ints[i] < 0:
                var _merr = self.mismatch(key, "unsigned array")
                raise _merr
            return UInt64(self.metadata[idx].value.array_ints[i])
        var _merr = self.mismatch(key, "integer array")
        raise _merr

    def metadata_array_i64(self, key: String, i: Int) raises -> Int64:
        var idx = self.require_metadata(key)
        if self.metadata[idx].value.type_id != META_ARRAY:
            var _merr = self.mismatch(key, "array")
            raise _merr
        var et = self.metadata[idx].value.array_elem_type
        if et == META_I8 or et == META_I16 or et == META_I32 or et == META_I64:
            if i < 0 or i >= len(self.metadata[idx].value.array_ints):
                raise Error(
                    "gguf_invalid_format: array index out of range for " + key
                )
            return self.metadata[idx].value.array_ints[i]
        if et == META_U8 or et == META_U16 or et == META_U32 or et == META_U64:
            if i < 0 or i >= len(self.metadata[idx].value.array_uints):
                raise Error(
                    "gguf_invalid_format: array index out of range for " + key
                )
            return Int64(self.metadata[idx].value.array_uints[i])
        var _merr = self.mismatch(key, "integer array")
        raise _merr

    def metadata_array_f64(self, key: String, i: Int) raises -> Float64:
        var idx = self.require_metadata(key)
        var et = self.metadata[idx].value.array_elem_type
        if self.metadata[idx].value.type_id != META_ARRAY or (
            et != META_F32 and et != META_F64
        ):
            var _merr = self.mismatch(key, "float array")
            raise _merr
        if i < 0 or i >= len(self.metadata[idx].value.array_floats):
            raise Error(
                "gguf_invalid_format: array index out of range for " + key
            )
        return self.metadata[idx].value.array_floats[i]

    def metadata_array_bool(self, key: String, i: Int) raises -> Bool:
        var idx = self.require_metadata(key)
        if (
            self.metadata[idx].value.type_id != META_ARRAY
            or self.metadata[idx].value.array_elem_type != META_BOOL
        ):
            var _merr = self.mismatch(key, "bool array")
            raise _merr
        if i < 0 or i >= len(self.metadata[idx].value.array_bools):
            raise Error(
                "gguf_invalid_format: array index out of range for " + key
            )
        return self.metadata[idx].value.array_bools[i]

    # -- tensor directory ---------------------------------------------------

    def find_tensor(self, name: String) -> Int:
        for i in range(len(self.tensors)):
            if self.tensors[i].name == name:
                return i
        return -1

    def has_tensor(self, name: String) -> Bool:
        return self.find_tensor(name) != -1

    def tensor_info(self, name: String) raises -> TensorInfo:
        var idx = self.find_tensor(name)
        if idx == -1:
            raise Error(
                "gguf_tensor_not_found: no tensor named "
                + name
                + " in "
                + self.path
            )
        return self.tensors[idx].copy()

    def tensor_byte_len(self, name: String) raises -> Int:
        var info = self.tensor_info(name)
        var n = info.byte_len()
        if n == -1:
            if ggml_supported(info.type_id):
                raise Error(
                    "gguf_bad_element_count: tensor "
                    + name
                    + " has "
                    + String(info.elements)
                    + " elements, not a whole number of "
                    + String(ggml_block_size(info.type_id))
                    + "-element blocks"
                )
            raise Error(dtype_refusal(name, info.type_id))
        return n

    # -- tensor data (staged file read; see the header comment) --------------

    def read_tensor_chunk(
        self, name: String, chunk_offset: Int, count: Int
    ) raises -> List[UInt8]:
        """Up to `count` raw bytes of a tensor starting at
        `chunk_offset` within the tensor. Short only at the
        tensor's own end; a short read against the file is a
        named truncation refusal."""
        var info = self.tensor_info(name)
        var total = self.tensor_byte_len(name)
        if chunk_offset < 0 or count < 0 or chunk_offset > total:
            raise Error(
                "gguf_invalid_format: tensor chunk offset "
                + String(chunk_offset)
                + " out of range for tensor "
                + name
                + " ("
                + String(total)
                + " bytes)"
            )
        var want = count
        if chunk_offset + want > total:
            want = total - chunk_offset
        if want == 0:
            return List[UInt8]()
        var absolute = self.data_start + info.rel_offset + chunk_offset
        var f: FileHandle
        try:
            f = open(self.path, "r")
        except e:
            raise Error(
                "gguf_io_error: cannot reopen " + self.path + ": " + String(e)
            )
        _ = f.seek(absolute)
        var data = f.read_bytes(want)
        f.close()
        if len(data) != want:
            raise Error(
                "gguf_truncated: file ended while reading tensor data for "
                + name
                + " at offset "
                + String(absolute)
                + ": needed "
                + String(want)
                + " bytes, file is "
                + String(self.file_len)
                + " bytes"
            )
        return data^

    def tensor_bytes(self, name: String) raises -> List[UInt8]:
        """The tensor's full raw stored bytes, staged in chunks."""
        var total = self.tensor_byte_len(name)
        _ = log_debug(
            "gguf",
            "tensor_load name=" + name + " bytes=" + String(total),
        )
        var out = List[UInt8](capacity=total)
        var off = 0
        while off < total:
            var n = STAGED_READ_CHUNK_BYTES
            if off + n > total:
                n = total - off
            var chunk = self.read_tensor_chunk(name, off, n)
            for i in range(len(chunk)):
                out.append(chunk[i])
            off += n
        return out^


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------


def parse_gguf(path: String) raises -> GgufFile:
    """Open and parse the GGUF file at `path`.

    Header, metadata, and the tensor directory are parsed and
    validated once here (element counts against block geometry,
    tensor data ranges against file length), mirroring the Rust
    loader, so a malformed file fails at open, not mid-run.
    """
    _ = log_debug("gguf", "file_open path=" + path)
    var handle: FileHandle
    try:
        handle = open(path, "r")
    except e:
        raise Error("gguf_io_error: cannot open " + path + ": " + String(e))
    var file_len = Int(handle.seek(0, 2))
    _ = handle.seek(0, 0)
    var reader = Reader(handle^, file_len)

    var magic = reader.read_exact(4, "magic")
    if not (
        magic[0] == 0x47
        and magic[1] == 0x47
        and magic[2] == 0x55
        and magic[3] == 0x46
    ):
        raise Error(
            "gguf_bad_magic: expected GGUF, found bytes ["
            + String(Int(magic[0]))
            + ", "
            + String(Int(magic[1]))
            + ", "
            + String(Int(magic[2]))
            + ", "
            + String(Int(magic[3]))
            + "] in "
            + path
        )
    var version = Int(reader.read_u32("version"))
    if version != GGUF_VERSION:
        raise Error(
            "gguf_unsupported_version: GGUF version "
            + String(version)
            + " (this loader reads version 3)"
        )
    var tensor_count = u64_to_int_checked(
        reader.read_u64("tensor_count"), "tensor"
    )
    var metadata_count = u64_to_int_checked(
        reader.read_u64("metadata_kv_count"), "metadata"
    )
    _ = log_debug(
        "gguf",
        "parse_header version="
        + String(version)
        + " tensors="
        + String(tensor_count)
        + " metadata_kv="
        + String(metadata_count),
    )

    var entries = List[MetadataEntry]()
    for _ in range(metadata_count):
        var key = reader.read_string("metadata key")
        var value = read_metadata_value(reader)
        entries.append(MetadataEntry(key, value^))
    _ = log_debug("gguf", "metadata_done kv=" + String(len(entries)))

    var tensors = List[TensorInfo]()
    for _ in range(tensor_count):
        var name = reader.read_string("tensor name")
        var n_dims = Int(reader.read_u32("tensor dimension count"))
        if n_dims < 1 or n_dims > 4:
            raise Error(
                "gguf_invalid_format: tensor "
                + name
                + ": dimension count "
                + String(n_dims)
                + " outside 1..=4"
            )
        var dims = List[Int]()
        var elements = 1
        for _ in range(n_dims):
            var d = u64_to_int_checked(
                reader.read_u64("tensor dimension"), "tensor dimension"
            )
            dims.append(d)
            if d != 0 and elements > 4611686018427387904 // d:
                raise Error(
                    "gguf_length_overflow: tensor element count for " + name
                )
            elements *= d
        var type_id = Int(reader.read_u32("tensor type"))
        var rel_offset = u64_to_int_checked(
            reader.read_u64("tensor offset"), "tensor offset"
        )
        if ggml_supported(type_id):
            if elements % ggml_block_size(type_id) != 0:
                raise Error(
                    "gguf_bad_element_count: tensor "
                    + name
                    + " has "
                    + String(elements)
                    + " elements, not a whole number of "
                    + String(ggml_block_size(type_id))
                    + "-element blocks"
                )
        for i in range(len(tensors)):
            if tensors[i].name == name:
                raise Error(
                    "gguf_invalid_format: duplicate tensor name " + name
                )
        tensors.append(TensorInfo(name, dims^, type_id, rel_offset, elements))
    _ = log_debug("gguf", "tensor_dir_done tensors=" + String(len(tensors)))

    # Alignment: general.alignment when present, else the spec
    # default (Rust DEFAULT_ALIGNMENT), validated the Rust way.
    var alignment = DEFAULT_ALIGNMENT
    for i in range(len(entries)):
        if entries[i].key == "general.alignment":
            var tid = entries[i].value.type_id
            if (
                tid == META_U8
                or tid == META_U16
                or tid == META_U32
                or tid == META_U64
            ):
                alignment = Int(entries[i].value.uint_value)
            elif (
                tid == META_I8
                or tid == META_I16
                or tid == META_I32
                or tid == META_I64
            ) and entries[i].value.int_value >= 0:
                alignment = Int(entries[i].value.int_value)
            else:
                raise Error(
                    "gguf_metadata_type_mismatch: key general.alignment holds "
                    + entries[i].value.type_name()
                    + ", not u32"
                )
            if (
                alignment == 0
                or (alignment & (alignment - 1)) != 0
                or alignment > 1048576
            ):
                raise Error(
                    "gguf_invalid_format: general.alignment "
                    + String(alignment)
                    + " is not a power of two within 1..=1048576"
                )
            break

    var header_len = reader.pos
    var data_start = align_up(header_len, alignment)

    for i in range(len(tensors)):
        var n = tensors[i].byte_len()
        if n != -1:
            var start = data_start + tensors[i].rel_offset
            if start + n > file_len:
                raise Error(
                    "gguf_tensor_out_of_bounds: tensor "
                    + tensors[i].name
                    + " data range ["
                    + String(start)
                    + ", "
                    + String(start + n)
                    + ") falls outside the file ("
                    + String(file_len)
                    + " bytes); the file is truncated or corrupt"
                )

    reader.handle.close()
    _ = log_debug(
        "gguf",
        "parse_done tensors="
        + String(len(tensors))
        + " metadata_kv="
        + String(len(entries))
        + " alignment="
        + String(alignment)
        + " data_start="
        + String(data_start)
        + " file_len="
        + String(file_len),
    )
    return GgufFile(
        path, version, alignment, data_start, file_len, entries^, tensors^
    )


# ---------------------------------------------------------------------------
# The foundation seam (signature and semantics preserved)
# ---------------------------------------------------------------------------


def loader_state() -> String:
    """The loader's implementation state, for hello/smoke output."""
    return "implemented"


def load_model(path: String) raises -> None:
    """Load (parse and fully validate) the GGUF model at `path`.

    Raises a named refusal (gguf_bad_magic,
    gguf_unsupported_version, gguf_truncated, gguf_unknown_dtype
    on byte access, gguf_tensor_not_found on lookup, and the
    rest of the gguf_* family) on any malformed file. Callers
    that need the parsed inventory use parse_gguf().
    """
    _ = log_debug("gguf", "load_model called for " + path)
    var model = parse_gguf(path)
    _ = log_debug(
        "gguf",
        "load_model ok tensors="
        + String(model.tensor_count())
        + " metadata_kv="
        + String(model.metadata_count()),
    )
