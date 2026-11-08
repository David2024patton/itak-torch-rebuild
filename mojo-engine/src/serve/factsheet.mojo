# factsheet.mojo - per-model fact sheets read from a GGUF
# file's OWN metadata and tensor directory (Piece 7,
# amendments 2026-11-08). Nothing here runs the model: the
# loader's parse_gguf reads header + metadata + tensor names
# only, so a fact sheet can be served for a file this engine
# cannot execute (the Qwen3.5 vision file is the standing
# example - fact sheet yes, generation no until its
# architecture lands).
#
# The per-layer kind split is derived from BOTH sources the
# amendments require, and both are reported:
#   metadata side: <arch>.full_attention_interval (qwen35: 4,
#     so layers with (i+1) % 4 == 0 are full attention),
#     <arch>.attention.sliding_window_pattern bool array
#     (muse-glimmer: true = sliding), any <arch>.ssm.* keys,
#     per-layer head_count_kv arrays when present;
#   tensor side: layers carrying ssm_* tensors vs attn_q
#     tensors, sliding vs global by the same pattern, and the
#     vision tensor census (names under v. / clip prefixes).
# Vision presence: vision tensors or a vision metadata block
# inside the file, OR a companion mmproj*.gguf in the same
# directory (config pairing by naming convention); the sheet
# says which source proved it ("in_file" / "companion_mmproj"
# / "none").

from std.ffi import external_call
from std.memory import alloc

from gguf.loader import GgufFile, parse_gguf
from serve.json import json_escape

comptime MODEL_FILE_SUFFIX = ".gguf"


struct LayerSplit(Copyable, Movable):
    var attention_layers: Int
    var ssm_layers: Int
    var sliding_layers: Int
    var global_layers: Int
    var other_layers: Int
    var source: String

    def __init__(
        out self,
        attention_layers: Int,
        ssm_layers: Int,
        sliding_layers: Int,
        global_layers: Int,
        other_layers: Int,
        source: String,
    ):
        self.attention_layers = attention_layers
        self.ssm_layers = ssm_layers
        self.sliding_layers = sliding_layers
        self.global_layers = global_layers
        self.other_layers = other_layers
        self.source = source


struct FactSheet(Copyable, Movable):
    var path: String
    var file_name: String
    var file_bytes: Int
    var architecture: String
    var name: String
    var file_type: Int
    var tensor_count: Int
    var block_count: Int
    var context_length: Int
    var embedding_length: Int
    var feed_forward_length: Int
    var head_count: Int
    var head_count_kv: Int
    var head_count_kv_per_layer: String
    var sliding_window: Int
    var full_attention_interval: Int
    var vocab_size: Int
    var split: LayerSplit
    var vision_tensor_count: Int
    var mtp_tensor_count: Int
    var has_vision_block: Bool
    var vision_present: Bool
    var vision_source: String
    var companion_mmproj: String

    def __init__(
        out self,
        path: String,
        file_name: String,
        file_bytes: Int,
        architecture: String,
        name: String,
        file_type: Int,
        tensor_count: Int,
        block_count: Int,
        context_length: Int,
        embedding_length: Int,
        feed_forward_length: Int,
        head_count: Int,
        head_count_kv: Int,
        head_count_kv_per_layer: String,
        sliding_window: Int,
        full_attention_interval: Int,
        vocab_size: Int,
        var split: LayerSplit,
        vision_tensor_count: Int,
        mtp_tensor_count: Int,
        has_vision_block: Bool,
        vision_present: Bool,
        vision_source: String,
        companion_mmproj: String,
    ):
        self.path = path
        self.file_name = file_name
        self.file_bytes = file_bytes
        self.architecture = architecture
        self.name = name
        self.file_type = file_type
        self.tensor_count = tensor_count
        self.block_count = block_count
        self.context_length = context_length
        self.embedding_length = embedding_length
        self.feed_forward_length = feed_forward_length
        self.head_count = head_count
        self.head_count_kv = head_count_kv
        self.head_count_kv_per_layer = head_count_kv_per_layer
        self.sliding_window = sliding_window
        self.full_attention_interval = full_attention_interval
        self.vocab_size = vocab_size
        self.split = split^
        self.vision_tensor_count = vision_tensor_count
        self.mtp_tensor_count = mtp_tensor_count
        self.has_vision_block = has_vision_block
        self.vision_present = vision_present
        self.vision_source = vision_source
        self.companion_mmproj = companion_mmproj


def base_name(path: String) -> String:
    var base = path
    var slash = -1
    var bytes = path.as_bytes()
    for i in range(len(bytes)):
        if bytes[i] == 47:
            slash = i
    if slash != -1:
        base = String(path[byte = slash + 1 : path.byte_length()])
    return base


def dir_name(path: String) -> String:
    var slash = -1
    var bytes = path.as_bytes()
    for i in range(len(bytes)):
        if bytes[i] == 47:
            slash = i
    if slash == -1:
        return String(".")
    if slash == 0:
        return String("/")
    return String(path[byte=0:slash])


def model_id_of(path: String) -> String:
    """The display id used across /models, /history and the
    leaderboard: file base name without the .gguf suffix (the
    same convention as server.model_display_name; duplicated
    here so this module does not import the dispatch layer)."""
    var base = base_name(path)
    if base.endswith(MODEL_FILE_SUFFIX):
        var trimmed = String(base[byte = 0 : base.byte_length() - 5])
        base = trimmed
    return base


def _layer_index_of(name: String) -> Int:
    """The block index in a blk.<n>. / v.blk.<n>. tensor name,
    or -1 when the name carries no block index."""
    var bytes = name.as_bytes()
    var i = 0
    # optional vision prefix
    if name.startswith("v.blk."):
        i = 6
    elif name.startswith("blk."):
        i = 4
    else:
        return -1
    var value = 0
    var digits = 0
    while i < len(bytes) and 48 <= Int(bytes[i]) and Int(bytes[i]) <= 57:
        value = value * 10 + (Int(bytes[i]) - 48)
        digits += 1
        i += 1
    if digits == 0 or i >= len(bytes) or bytes[i] != 46:
        return -1
    return value


def _opt_u32(gf: GgufFile, key: String) -> Int:
    if not gf.has_metadata(key):
        return 0
    try:
        return Int(gf.get_u32(key))
    except:
        return 0


def _opt_i64(gf: GgufFile, key: String) -> Int:
    if not gf.has_metadata(key):
        return 0
    try:
        return Int(gf.get_i64(key))
    except:
        return 0


def _opt_string(gf: GgufFile, key: String) -> String:
    if not gf.has_metadata(key):
        return String("")
    try:
        return gf.get_string(key)
    except:
        return String("")


def _has_ssm_metadata(gf: GgufFile, arch: String) -> Bool:
    var prefix = arch + ".ssm."
    for i in range(gf.metadata_count()):
        if gf.metadata[i].key.startswith(prefix):
            return True
    return False


def _per_layer_kv(gf: GgufFile, arch: String) -> String:
    """Per-layer head_count_kv when the file stores it as an
    array: "uniform N" when every layer agrees, "varies"
    otherwise, "scalar" when only the scalar key exists, ""
    when the file does not say."""
    var key = arch + ".attention.head_count_kv"
    var idx = gf.find_metadata(key)
    if idx == -1:
        return String("")
    # type 9 is GGUF array; the loader exposes array length
    # through metadata_array_len, which refuses non-arrays.
    try:
        var n = gf.metadata_array_len(key)
        if n <= 0:
            return String("")
        var first = Int(gf.metadata_array_u64(key, 0))
        var uniform = True
        for i in range(1, n):
            if Int(gf.metadata_array_u64(key, i)) != first:
                uniform = False
        if uniform:
            return "uniform " + String(first)
        return String("varies")
    except:
        return String("scalar")


def compute_split(gf: GgufFile, arch: String, blocks: Int) -> LayerSplit:
    """The per-layer kind split from tensor signatures, with
    the metadata cross-checks the amendments require. Tensor
    signatures decide the attention/ssm split (they are what
    the file actually stores); the sliding/global split comes
    from the pattern array when present, else the
    full_attention_interval metadata, else it stays 0/0 with
    the source saying so."""
    var attn = List[Bool]()
    var ssm = List[Bool]()
    for _ in range(blocks):
        attn.append(False)
        ssm.append(False)
    for t in range(gf.tensor_count()):
        var name = gf.tensor_name(t)
        var li = _layer_index_of(name)
        if li < 0 or li >= blocks:
            continue
        if name.startswith("v.blk."):
            continue
        if not attn[li] and (
            name.find(".attn_q.") != -1 or name.find(".attn_qkv.") != -1
        ):
            attn[li] = True
        if not ssm[li] and name.find(".ssm_") != -1:
            ssm[li] = True
    var n_attn = 0
    var n_ssm = 0
    var n_other = 0
    for i in range(blocks):
        if ssm[i]:
            n_ssm += 1
        elif attn[i]:
            n_attn += 1
        else:
            n_other += 1

    var sliding = 0
    var glob = 0
    var source = String("tensor_signatures")
    var pattern_key = arch + ".attention.sliding_window_pattern"
    if gf.has_metadata(pattern_key):
        try:
            var n = gf.metadata_array_len(pattern_key)
            if n == blocks:
                for i in range(blocks):
                    if gf.metadata_array_bool(pattern_key, i):
                        sliding += 1
                    else:
                        glob += 1
                source = "metadata_pattern+tensor_signatures"
        except:
            pass
    if sliding == 0 and glob == 0:
        var interval = _opt_u32(gf, arch + ".full_attention_interval")
        if interval > 0:
            for i in range(blocks):
                if (i + 1) % interval == 0:
                    glob += 1
                else:
                    sliding += 1
            source = "metadata_interval+tensor_signatures"
    if _has_ssm_metadata(gf, arch):
        source += "+ssm_metadata"
    return LayerSplit(n_attn, n_ssm, sliding, glob, n_other, source)


def count_prefixed(gf: GgufFile, prefixes: List[String]) -> Int:
    var n = 0
    for t in range(gf.tensor_count()):
        var name = gf.tensor_name(t)
        for p in range(len(prefixes)):
            if name.startswith(prefixes[p]):
                n += 1
                break
    return n


def list_dir_names(dir_path: String) raises -> List[String]:
    """Directory entries via opendir/readdir FFI (pure Mojo,
    no shell). Refuses serve_models_dir by name when the
    directory cannot be opened."""
    var names = List[String]()
    var buf = alloc[UInt8](dir_path.byte_length() + 1)
    var db = dir_path.as_bytes()
    for i in range(len(db)):
        buf[i] = db[i]
    buf[len(db)] = UInt8(0)
    var handle = external_call["opendir", Int](buf)
    if handle == 0:
        buf.free()
        raise Error(
            "serve_models_dir: cannot open models directory " + dir_path
        )
    while True:
        var entry = external_call["readdir", Int](handle)
        if entry == 0:
            break
        # glibc struct dirent (x86_64): d_ino u64 @0, d_off
        # i64 @8, d_reclen u16 @16, d_type u8 @18, d_name
        # char[256] @19.
        var ep = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=entry)
        var name_bytes = List[UInt8]()
        var j = 19
        while j < 19 + 255:
            var c = ep[j]
            if c == 0:
                break
            name_bytes.append(c)
            j += 1
        if len(name_bytes) > 0:
            names.append(_bytes_of(name_bytes))
    _ = external_call["closedir", Int](handle)
    buf.free()
    return names^


def _bytes_of(bytes: List[UInt8]) -> String:
    from tokenizer.byte_level import string_from_bytes_lossy

    return string_from_bytes_lossy(bytes)


def find_companion_mmproj(model_path: String) -> String:
    """The first mmproj*.gguf in the model's directory, or ""
    (naming-convention pairing, recorded on the sheet)."""
    try:
        var names = list_dir_names(dir_name(model_path))
        for i in range(len(names)):
            var n = names[i]
            if n.startswith("mmproj") and n.endswith(MODEL_FILE_SUFFIX):
                return n
    except:
        pass
    return String("")


def fact_sheet(path: String) raises -> FactSheet:
    var gf = parse_gguf(path)
    var arch = _opt_string(gf, "general.architecture")
    if arch == "":
        arch = String("unknown")
    var blocks = _opt_u32(gf, arch + ".block_count")
    var split = compute_split(gf, arch, blocks)

    var vision_prefixes = List[String]()
    vision_prefixes.append("v.")
    vision_prefixes.append("clip.")
    var vision_tensors = count_prefixed(gf, vision_prefixes)
    var mtp_prefixes = List[String]()
    mtp_prefixes.append("mtp.")
    var mtp_tensors = count_prefixed(gf, mtp_prefixes)
    # Vision metadata block: clip.* keys on llama-style vision
    # files, or the arch-prefixed <arch>.vision.block_count the
    # qwen35 family writes (verified on the staged Qwen3.5-9B
    # vision file: qwen35.vision.block_count = 27; a clip-only
    # check reported metadata_block false on that file).
    var has_vision_block = (
        gf.has_metadata("clip.vision.block_count")
        or gf.has_metadata("clip.block_count")
        or gf.has_metadata(arch + ".vision.block_count")
    )

    var companion = String("")
    var vision_present = vision_tensors > 0 or has_vision_block
    var vision_source = String("none")
    if vision_present:
        vision_source = String("in_file")
    else:
        companion = find_companion_mmproj(path)
        if companion != "":
            vision_present = True
            vision_source = String("companion_mmproj")

    var vocab = 0
    if gf.has_metadata("tokenizer.ggml.tokens"):
        try:
            vocab = gf.metadata_array_len("tokenizer.ggml.tokens")
        except:
            vocab = 0

    return FactSheet(
        path,
        base_name(path),
        gf.file_len,
        arch,
        _opt_string(gf, "general.name"),
        _opt_i64(gf, "general.file_type"),
        gf.tensor_count(),
        blocks,
        _opt_u32(gf, arch + ".context_length"),
        _opt_u32(gf, arch + ".embedding_length"),
        _opt_u32(gf, arch + ".feed_forward_length"),
        _opt_u32(gf, arch + ".attention.head_count"),
        _opt_u32(gf, arch + ".attention.head_count_kv"),
        _per_layer_kv(gf, arch),
        _opt_u32(gf, arch + ".attention.sliding_window"),
        _opt_u32(gf, arch + ".full_attention_interval"),
        vocab,
        split^,
        vision_tensors,
        mtp_tensors,
        has_vision_block,
        vision_present,
        vision_source,
        companion,
    )


def fact_sheet_json(sheet: FactSheet, served: Bool) -> String:
    var out = String("{")
    out += '"id":"' + json_escape(model_id_of(sheet.path)) + '"'
    out += ',"path":"' + json_escape(sheet.path) + '"'
    out += ',"file_name":"' + json_escape(sheet.file_name) + '"'
    out += ',"file_bytes":' + String(sheet.file_bytes)
    out += ',"served":' + ("true" if served else "false")
    out += ',"architecture":"' + json_escape(sheet.architecture) + '"'
    out += ',"name":"' + json_escape(sheet.name) + '"'
    out += ',"file_type":' + String(sheet.file_type)
    out += ',"tensor_count":' + String(sheet.tensor_count)
    out += ',"block_count":' + String(sheet.block_count)
    out += ',"context_length":' + String(sheet.context_length)
    out += ',"embedding_length":' + String(sheet.embedding_length)
    out += ',"feed_forward_length":' + String(sheet.feed_forward_length)
    out += ',"head_count":' + String(sheet.head_count)
    out += ',"head_count_kv":' + String(sheet.head_count_kv)
    out += ',"head_count_kv_detail":"' + json_escape(
        sheet.head_count_kv_per_layer
    ) + '"'
    out += ',"sliding_window":' + String(sheet.sliding_window)
    out += ',"full_attention_interval":' + String(sheet.full_attention_interval)
    out += ',"vocab_size":' + String(sheet.vocab_size)
    out += ',"layers":{'
    out += '"total":' + String(sheet.block_count)
    out += ',"attention":' + String(sheet.split.attention_layers)
    out += ',"ssm":' + String(sheet.split.ssm_layers)
    out += ',"sliding":' + String(sheet.split.sliding_layers)
    out += ',"global":' + String(sheet.split.global_layers)
    out += ',"other":' + String(sheet.split.other_layers)
    out += ',"source":"' + json_escape(sheet.split.source) + '"}'
    out += ',"vision":{'
    out += '"present":' + ("true" if sheet.vision_present else "false")
    out += ',"source":"' + json_escape(sheet.vision_source) + '"'
    out += ',"tensor_count":' + String(sheet.vision_tensor_count)
    out += ',"metadata_block":' + ("true" if sheet.has_vision_block else "false")
    out += ',"companion_mmproj":"' + json_escape(sheet.companion_mmproj) + '"}'
    out += ',"mtp_tensor_count":' + String(sheet.mtp_tensor_count)
    out += "}"
    return out


def scan_model_files(dir_path: String) raises -> List[String]:
    """All *.gguf paths directly under a directory, sorted by
    name, one level (no recursion: /mnt/models holds one
    directory per model family in this bench's layout, and
    the served file itself is added by the caller)."""
    var names = list_dir_names(dir_path)
    var files = List[String]()
    for i in range(len(names)):
        if names[i].endswith(MODEL_FILE_SUFFIX):
            files.append(names[i])
    # insertion sort by name (registries are small)
    for i in range(1, len(files)):
        var j = i
        while j > 0 and files[j] < files[j - 1]:
            var tmp = files[j - 1]
            files[j - 1] = files[j]
            files[j] = tmp
            j -= 1
    var base = dir_path
    if not base.endswith("/"):
        base += "/"
    var out = List[String]()
    for i in range(len(files)):
        out.append(base + files[i])
    return out^
