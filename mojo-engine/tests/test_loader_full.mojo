# test_loader_full.mojo - the GGUF loader's parity proofs (VM half).
#
# Writes a synthetic GGUF file with known metadata of every GGUF
# value type and known tensors (F32 and Q4_K), round-trips it
# through the loader, and asserts every metadata value and every
# tensor byte. Each named refusal (bad magic, unsupported
# version, truncated header / metadata / tensor data, unknown
# dtype, unsupported dtype, tensor not found, missing metadata,
# unknown metadata type, bad element count) is fired on its own
# crafted bad file. The real-file half lives on the server (see
# tests/loader_server_proof.mojo).

from std.testing import TestSuite, assert_true, assert_equal, assert_false

from gguf.loader import (
    load_model,
    loader_state,
    parse_gguf,
)


# -- synthetic file writer ---------------------------------------------------


def put_u32(mut b: List[UInt8], v: UInt32):
    for i in range(4):
        b.append(UInt8((v >> UInt32(8 * i)) & 0xFF))


def put_u64(mut b: List[UInt8], v: UInt64):
    for i in range(8):
        b.append(UInt8((v >> UInt64(8 * i)) & 0xFF))


def put_i64_bits(mut b: List[UInt8], v: Int64):
    var tmp = List[Int64](capacity=1)
    tmp.append(v)
    var u = tmp.unsafe_ptr().unsafe_bitcast[UInt64]()[unsafe_offset=0]
    put_u64(b, u)


def put_f32(mut b: List[UInt8], v: Float32):
    var tmp = List[Float32](capacity=1)
    tmp.append(v)
    var u = tmp.unsafe_ptr().unsafe_bitcast[UInt32]()[unsafe_offset=0]
    put_u32(b, u)


def put_f64(mut b: List[UInt8], v: Float64):
    var tmp = List[Float64](capacity=1)
    tmp.append(v)
    var u = tmp.unsafe_ptr().unsafe_bitcast[UInt64]()[unsafe_offset=0]
    put_u64(b, u)


def put_str(mut b: List[UInt8], s: String):
    put_u64(b, UInt64(s.byte_length()))
    for byte in s.bytes():
        b.append(byte)


def put_kv_header(mut b: List[UInt8], key: String, type_id: UInt32):
    put_str(b, key)
    put_u32(b, type_id)


def write_file(path: String, data: List[UInt8]) raises:
    with open(path, "w") as f:
        f.write_bytes(data)


def f32_bytes(v: Float32) -> List[UInt8]:
    var out = List[UInt8](capacity=4)
    put_f32(out, v)
    return out^


def build_good_file() -> Tuple[List[UInt8], Int]:
    """The known synthetic file; returns (bytes, data_start)."""
    var b = List[UInt8]()
    # Header.
    b.append(0x47)
    b.append(0x47)
    b.append(0x55)
    b.append(0x46)
    put_u32(b, 3)
    put_u64(b, 3)  # tensor count
    put_u64(b, 17)  # metadata kv count
    # Metadata: every GGUF value type.
    put_kv_header(b, "general.architecture", 8)
    put_str(b, "synth-arch")
    put_kv_header(b, "general.alignment", 4)
    put_u32(b, 32)
    put_kv_header(b, "synth.u8", 0)
    b.append(200)
    put_kv_header(b, "synth.i8", 1)
    b.append(251)  # -5
    put_kv_header(b, "synth.u16", 2)
    b.append(UInt8(60000 & 0xFF))
    b.append(UInt8((60000 >> 8) & 0xFF))
    put_kv_header(b, "synth.i16", 3)
    var i16v = -30000 + 65536
    b.append(UInt8(i16v & 0xFF))
    b.append(UInt8((i16v >> 8) & 0xFF))
    put_kv_header(b, "synth.u32", 4)
    put_u32(b, 4000000000)
    put_kv_header(b, "synth.i32", 5)
    put_u32(b, UInt32(4294967295) - 2000000000 + 1)
    put_kv_header(b, "synth.f32", 6)
    put_f32(b, Float32(1.5))
    put_kv_header(b, "synth.bool", 7)
    b.append(1)
    put_kv_header(b, "synth.u64", 10)
    put_u64(b, 12345678901234567890)
    put_kv_header(b, "synth.i64", 11)
    put_i64_bits(b, -9223372036854775807)
    put_kv_header(b, "synth.f64", 12)
    put_f64(b, Float64(-2.25))
    put_kv_header(b, "synth.arr_u32", 9)
    put_u32(b, 4)
    put_u64(b, 3)
    put_u32(b, 7)
    put_u32(b, 8)
    put_u32(b, 9)
    put_kv_header(b, "synth.arr_str", 9)
    put_u32(b, 8)
    put_u64(b, 2)
    put_str(b, "aa")
    put_str(b, "bb")
    put_kv_header(b, "synth.arr_bool", 9)
    put_u32(b, 7)
    put_u64(b, 2)
    b.append(1)
    b.append(0)
    put_kv_header(b, "synth.arr_f32", 9)
    put_u32(b, 6)
    put_u64(b, 2)
    put_f32(b, Float32(0.5))
    put_f32(b, Float32(-0.5))
    # Tensor directory.
    put_str(b, "w.f32")
    put_u32(b, 1)
    put_u64(b, 4)
    put_u32(b, 0)
    put_u64(b, 0)
    put_str(b, "w.q4k")
    put_u32(b, 1)
    put_u64(b, 256)
    put_u32(b, 12)
    put_u64(b, 16)
    put_str(b, "w.f32_2d")
    put_u32(b, 2)
    put_u64(b, 2)
    put_u64(b, 3)
    put_u32(b, 0)
    put_u64(b, 160)
    # Alignment padding to 32.
    var data_start = ((len(b) + 31) // 32) * 32
    while len(b) < data_start:
        b.append(0)
    # Tensor data: w.f32 [1.0, -2.0, 3.5, 0.25] (16 bytes).
    for v in [Float32(1.0), Float32(-2.0), Float32(3.5), Float32(0.25)]:
        var fb = f32_bytes(v)
        for i in range(4):
            b.append(fb[i])
    # w.q4k: 144 bytes, byte i = (i * 3 + 1) % 256.
    for i in range(144):
        b.append(UInt8((i * 3 + 1) % 256))
    # w.f32_2d: 6 values 10..15 (24 bytes).
    for k in range(6):
        var fb = f32_bytes(Float32(10 + k))
        for i in range(4):
            b.append(fb[i])
    return (b^, data_start)


def refusal_of(path: String) -> String:
    """Parse path and return the refusal text, or "" if it parsed."""
    try:
        _ = parse_gguf(path)
        return String("")
    except e:
        return String(e)


# -- round-trip proofs ---------------------------------------------------------


def test_loader_state_implemented() raises:
    assert_equal(loader_state(), "implemented")


def test_synthetic_round_trip() raises:
    var built = build_good_file()
    var path = "/tmp/test_loader_full_good.gguf"
    write_file(path, built[0].copy())
    var model = parse_gguf(path)
    assert_equal(model.version, 3)
    assert_equal(model.tensor_count(), 3)
    assert_equal(model.metadata_count(), 17)
    assert_equal(model.alignment, 32)
    assert_equal(model.data_start, built[1])
    assert_equal(model.file_len, len(built[0]))
    # Metadata, every type.
    assert_equal(model.get_string("general.architecture"), "synth-arch")
    assert_equal(model.architecture(), "synth-arch")
    assert_equal(Int(model.get_u32("general.alignment")), 32)
    assert_equal(Int(model.get_u64("synth.u8")), 200)
    assert_equal(Int(model.get_i64("synth.i8")), -5)
    assert_equal(Int(model.get_u64("synth.u16")), 60000)
    assert_equal(Int(model.get_i64("synth.i16")), -30000)
    assert_equal(Int(model.get_u64("synth.u32")), 4000000000)
    assert_equal(Int(model.get_i64("synth.i32")), -2000000000)
    assert_true(model.get_f32("synth.f32") == Float32(1.5))
    assert_true(model.get_bool("synth.bool"))
    assert_equal(model.get_u64("synth.u64"), 12345678901234567890)
    assert_equal(model.get_i64("synth.i64"), -9223372036854775807)
    assert_true(model.get_f64("synth.f64") == Float64(-2.25))
    # Arrays.
    assert_equal(model.metadata_array_len("synth.arr_u32"), 3)
    assert_equal(Int(model.metadata_array_u64("synth.arr_u32", 0)), 7)
    assert_equal(Int(model.metadata_array_u64("synth.arr_u32", 2)), 9)
    assert_equal(model.metadata_array_len("synth.arr_str"), 2)
    assert_equal(model.metadata_array_string("synth.arr_str", 1), "bb")
    assert_equal(model.metadata_array_len("synth.arr_bool"), 2)
    assert_true(model.metadata_array_bool("synth.arr_bool", 0))
    assert_false(model.metadata_array_bool("synth.arr_bool", 1))
    assert_equal(model.metadata_array_len("synth.arr_f32"), 2)
    assert_true(model.metadata_array_f64("synth.arr_f32", 1) == Float64(-0.5))
    # Tensor directory.
    var f32info = model.tensor_info("w.f32")
    assert_equal(f32info.type_id, 0)
    assert_equal(f32info.type_name(), "F32")
    assert_equal(f32info.elements, 4)
    assert_equal(len(f32info.dims), 1)
    assert_equal(f32info.dims[0], 4)
    assert_equal(f32info.byte_len(), 16)
    var qinfo = model.tensor_info("w.q4k")
    assert_equal(qinfo.type_id, 12)
    assert_equal(qinfo.type_name(), "Q4_K")
    assert_equal(qinfo.elements, 256)
    assert_equal(qinfo.byte_len(), 144)
    var info2d = model.tensor_info("w.f32_2d")
    assert_equal(len(info2d.dims), 2)
    assert_equal(info2d.dims[0], 2)
    assert_equal(info2d.dims[1], 3)
    assert_equal(info2d.elements, 6)
    assert_true(model.has_tensor("w.q4k"))
    assert_false(model.has_tensor("w.nope"))
    # Tensor bytes, exact.
    var fb = model.tensor_bytes("w.f32")
    assert_equal(len(fb), 16)
    var expect = List[UInt8]()
    for v in [Float32(1.0), Float32(-2.0), Float32(3.5), Float32(0.25)]:
        var one = f32_bytes(v)
        for i in range(4):
            expect.append(one[i])
    for i in range(16):
        assert_equal(Int(fb[i]), Int(expect[i]))
    var qb = model.tensor_bytes("w.q4k")
    assert_equal(len(qb), 144)
    for i in range(144):
        assert_equal(Int(qb[i]), (i * 3 + 1) % 256)
    var b2 = model.tensor_bytes("w.f32_2d")
    assert_equal(len(b2), 24)
    # Chunked access agrees with the whole-tensor read.
    var chunk = model.read_tensor_chunk("w.q4k", 100, 44)
    assert_equal(len(chunk), 44)
    for i in range(44):
        assert_equal(Int(chunk[i]), Int(qb[100 + i]))
    # The fixed seam validates the good file without refusing.
    load_model(path)


# -- named refusals ------------------------------------------------------------


def test_refusal_bad_magic() raises:
    var built = build_good_file()
    var b = built[0].copy()
    b[0] = 0x58
    var path = "/tmp/test_loader_full_badmagic.gguf"
    write_file(path, b)
    assert_true(
        refusal_of(path).find("gguf_bad_magic") != -1,
        "bad magic must refuse as gguf_bad_magic",
    )


def test_refusal_unsupported_version() raises:
    var built = build_good_file()
    var b = built[0].copy()
    b[4] = 2
    var path = "/tmp/test_loader_full_version.gguf"
    write_file(path, b)
    assert_true(
        refusal_of(path).find("gguf_unsupported_version") != -1,
        "version 2 must refuse as gguf_unsupported_version",
    )


def test_refusal_truncated_header() raises:
    var b = List[UInt8]()
    b.append(0x47)
    b.append(0x47)
    b.append(0x55)
    var path = "/tmp/test_loader_full_trunc_header.gguf"
    write_file(path, b)
    assert_true(
        refusal_of(path).find("gguf_truncated") != -1,
        "3-byte file must refuse as gguf_truncated",
    )


def test_refusal_truncated_metadata() raises:
    var built = build_good_file()
    var full = built[0].copy()
    var b = List[UInt8]()
    for i in range(60):
        b.append(full[i])
    var path = "/tmp/test_loader_full_trunc_meta.gguf"
    write_file(path, b)
    assert_true(
        refusal_of(path).find("gguf_truncated") != -1,
        "file cut inside metadata must refuse as gguf_truncated",
    )


def test_refusal_truncated_tensor_data() raises:
    var built = build_good_file()
    var full = built[0].copy()
    var b = List[UInt8]()
    for i in range(len(full) - 10):
        b.append(full[i])
    var path = "/tmp/test_loader_full_trunc_tensor.gguf"
    write_file(path, b)
    assert_true(
        refusal_of(path).find("gguf_tensor_out_of_bounds") != -1,
        "file cut inside tensor data must refuse as gguf_tensor_out_of_bounds",
    )


def build_one_tensor_file(type_id: UInt32, dim: UInt64) -> List[UInt8]:
    var b = List[UInt8]()
    b.append(0x47)
    b.append(0x47)
    b.append(0x55)
    b.append(0x46)
    put_u32(b, 3)
    put_u64(b, 1)
    put_u64(b, 0)
    put_str(b, "w.mystery")
    put_u32(b, 1)
    put_u64(b, dim)
    put_u32(b, type_id)
    put_u64(b, 0)
    var data_start = ((len(b) + 31) // 32) * 32
    while len(b) < data_start:
        b.append(0)
    for _ in range(1024):
        b.append(0xAB)
    return b^


def test_refusal_unknown_dtype() raises:
    var path = "/tmp/test_loader_full_dtype999.gguf"
    write_file(path, build_one_tensor_file(999, 4))
    var model = parse_gguf(path)
    var msg = String("")
    try:
        _ = model.tensor_bytes("w.mystery")
    except e:
        msg = String(e)
    assert_true(
        msg.find("gguf_unknown_dtype") != -1,
        "type id 999 byte access must refuse as gguf_unknown_dtype",
    )


def test_refusal_unsupported_dtype() raises:
    var path = "/tmp/test_loader_full_dtype9.gguf"
    write_file(path, build_one_tensor_file(9, 32))
    var model = parse_gguf(path)
    var msg = String("")
    try:
        _ = model.tensor_bytes("w.mystery")
    except e:
        msg = String(e)
    assert_true(
        msg.find("gguf_unsupported_tensor_type") != -1,
        "Q8_1 byte access must refuse as gguf_unsupported_tensor_type",
    )
    assert_true(msg.find("Q8_1") != -1, "refusal must name the type")


def test_refusal_tensor_not_found() raises:
    var built = build_good_file()
    var path = "/tmp/test_loader_full_notfound.gguf"
    write_file(path, built[0].copy())
    var model = parse_gguf(path)
    var msg = String("")
    try:
        _ = model.tensor_info("no.such.tensor")
    except e:
        msg = String(e)
    assert_true(
        msg.find("gguf_tensor_not_found") != -1,
        "missing tensor lookup must refuse as gguf_tensor_not_found",
    )
    var msg2 = String("")
    try:
        _ = model.tensor_bytes("no.such.tensor")
    except e:
        msg2 = String(e)
    assert_true(msg2.find("gguf_tensor_not_found") != -1)


def test_refusal_missing_metadata() raises:
    var built = build_good_file()
    var path = "/tmp/test_loader_full_missmeta.gguf"
    write_file(path, built[0].copy())
    var model = parse_gguf(path)
    var msg = String("")
    try:
        _ = model.get_u32("no.such.key")
    except e:
        msg = String(e)
    assert_true(
        msg.find("gguf_missing_metadata") != -1,
        "missing key must refuse as gguf_missing_metadata",
    )
    var msg2 = String("")
    try:
        _ = model.get_u32("general.architecture")
    except e:
        msg2 = String(e)
    assert_true(
        msg2.find("gguf_metadata_type_mismatch") != -1,
        "wrong-typed key must refuse as gguf_metadata_type_mismatch",
    )


def test_refusal_unknown_metadata_type() raises:
    var b = List[UInt8]()
    b.append(0x47)
    b.append(0x47)
    b.append(0x55)
    b.append(0x46)
    put_u32(b, 3)
    put_u64(b, 0)
    put_u64(b, 1)
    put_kv_header(b, "synth.bad", 77)
    b.append(0)
    var path = "/tmp/test_loader_full_badmeta.gguf"
    write_file(path, b)
    assert_true(
        refusal_of(path).find("gguf_unknown_metadata_type") != -1,
        "metadata type 77 must refuse as gguf_unknown_metadata_type",
    )


def test_refusal_bad_element_count() raises:
    var path = "/tmp/test_loader_full_badcount.gguf"
    write_file(path, build_one_tensor_file(12, 100))
    assert_true(
        refusal_of(path).find("gguf_bad_element_count") != -1,
        "Q4_K with 100 elements must refuse as gguf_bad_element_count",
    )


def test_refusal_io_missing_file() raises:
    assert_true(
        refusal_of("/tmp/test_loader_full_absent_file.gguf").find(
            "gguf_io_error"
        )
        != -1,
        "absent file must refuse as gguf_io_error",
    )
    var msg = String("")
    try:
        load_model("/tmp/test_loader_full_absent_file.gguf")
    except e:
        msg = String(e)
    assert_true(msg.find("gguf_io_error") != -1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
