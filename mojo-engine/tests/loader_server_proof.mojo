# loader_server_proof.mojo - the GGUF loader's real-file proof.
#
# NOT a TestSuite file: a standalone proof program run on the
# server against the real Glimmer file (CPU / file IO only).
#
#   mojo run -I src tests/loader_server_proof.mojo [--model PATH]
#       [--tensor NAME] [--dump PATH]
#
# Prints PROOF lines: the tensor inventory count, the model
# metadata spot values under their actual GGUF keys (read through
# the loader), and for the fixture tensor a staged-read FNV-1a 64
# hash over its full byte range plus first/last 16 bytes in hex.
# With --dump, the tensor bytes served through the loader are
# also written out so the caller can sha256 them independently.

from std.sys import argv, exit

from config import EngineConfig
from gguf.loader import parse_gguf


def arg_value(args: List[String], flag: String, default: String) -> String:
    for i in range(len(args) - 1):
        if args[i] == flag:
            return args[i + 1]
    return default


def has_flag(args: List[String], flag: String) -> Bool:
    for i in range(len(args)):
        if args[i] == flag:
            return True
    return False


def hex64(v: UInt64) -> String:
    var digits = "0123456789abcdef"
    var out = String("0x")
    for shift in range(60, -1, -4):
        var nib = Int((v >> UInt64(shift)) & 0xF)
        out += digits[byte = nib : nib + 1]
    return out


def hex_bytes(data: List[UInt8], start: Int, count: Int) -> String:
    var digits = "0123456789abcdef"
    var out = String("")
    for i in range(count):
        var b = Int(data[start + i])
        out += digits[byte = b >> 4 : (b >> 4) + 1]
        out += digits[byte = b & 0xF : (b & 0xF) + 1]
    return out


def run() raises:
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    var default_path = EngineConfig.load().model_path
    var path = arg_value(args, "--model", default_path)
    var tensor_name = arg_value(args, "--tensor", "blk.0.ffn_gate.weight")
    var dump_path = arg_value(args, "--dump", "")

    var model = parse_gguf(path)
    print("PROOF file", path)
    print("PROOF file_len", model.file_len)
    print("PROOF version", model.version)
    print("PROOF alignment", model.alignment)
    print("PROOF data_start", model.data_start)
    print("PROOF metadata_kv", model.metadata_count())
    print("PROOF tensors", model.tensor_count())

    var arch = model.architecture()
    print("PROOF key general.architecture =", arch)
    print(
        "PROOF key",
        arch + ".block_count",
        "=",
        Int(model.get_u32(arch + ".block_count")),
    )
    print(
        "PROOF key",
        arch + ".embedding_length",
        "=",
        Int(model.get_u32(arch + ".embedding_length")),
    )
    print(
        "PROOF key",
        arch + ".attention.head_count",
        "=",
        Int(model.get_u32(arch + ".attention.head_count")),
    )
    print(
        "PROOF key",
        arch + ".attention.head_count_kv",
        "=",
        Int(model.get_u32(arch + ".attention.head_count_kv")),
    )
    print(
        "PROOF key",
        arch + ".context_length",
        "=",
        Int(model.get_u32(arch + ".context_length")),
    )
    print(
        "PROOF key tokenizer.ggml.tokens (array length) =",
        model.metadata_array_len("tokenizer.ggml.tokens"),
    )

    var info = model.tensor_info(tensor_name)
    var total = model.tensor_byte_len(tensor_name)
    print("PROOF tensor", tensor_name)
    print("PROOF tensor dims", info.dims[0], "x", info.dims[1])
    print("PROOF tensor type", info.type_name(), "id", info.type_id)
    print("PROOF tensor elements", info.elements)
    print("PROOF tensor rel_offset", info.rel_offset)
    print("PROOF tensor bytes", total)

    # Staged read: FNV-1a 64 over the full range, chunk by chunk
    # through the loader, plus an in-memory copy for the hex
    # probes and the optional dump.
    var fnv = UInt64(14695981039346656037)
    var all = List[UInt8](capacity=total)
    var off = 0
    while off < total:
        var n = 4194304
        if off + n > total:
            n = total - off
        var chunk = model.read_tensor_chunk(tensor_name, off, n)
        for i in range(len(chunk)):
            fnv = (fnv ^ UInt64(chunk[i])) * UInt64(1099511628211)
            all.append(chunk[i])
        off += n
    print("PROOF tensor_fnv1a64", hex64(fnv))
    print("PROOF tensor_first16", hex_bytes(all, 0, 16))
    print("PROOF tensor_last16", hex_bytes(all, total - 16, 16))
    if dump_path != "":
        with open(dump_path, "w") as f:
            f.write_bytes(all)
        print("PROOF dumped", dump_path, len(all))
    print("PROOF DONE")


def main():
    try:
        run()
    except e:
        print("PROOF REFUSAL", e)
        exit(2)
