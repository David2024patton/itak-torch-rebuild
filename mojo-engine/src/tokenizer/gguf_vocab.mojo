# gguf_vocab.mojo - the tokenizer's vocabulary source: produce
# TokenizerData from a GGUF file through Lane A1's loader
# (src/gguf/loader.mojo).
#
# History: Lane A5 landed with a temporary self-contained GGUF
# metadata reader in this file so the tokenizer could be built
# in parallel with Lane A1. At Wave A integration the seam was
# swapped, as chartered: this module now parses the file with
# the engine's GGUF loader and maps its metadata getters onto
# TokenizerData (the Rust shape: crates/gguf TokenizerInfo ->
# TokenizerData, as in crates/tokenizer/examples/
# conformance.rs's data_from_info). Nothing outside this file
# knows GGUF exists: tokenizer.mojo consumes TokenizerData
# only. The interface preserved is load_tokenizer_data(path)
# -> TokenizerData.
#
# Keys read: tokenizer.ggml.model, .pre, .tokens, .token_type,
# .merges, .scores, .bos_token_id, .eos_token_id,
# .unknown_token_id, .padding_token_id (the crates/gguf
# TokenizerInfo set plus pre). Malformed files refuse inside
# the loader with its gguf_* names; a file whose tokenizer
# metadata has the wrong shape refuses here with
# tokenizer_gguf_type_mismatch, and a file with no
# tokenizer.ggml.tokens refuses with
# tokenizer_missing_vocab_metadata.

from diag import log_debug
from gguf.loader import (
    META_ARRAY,
    META_F32,
    META_F64,
    META_STRING,
    GgufFile,
    parse_gguf,
)
from tokenizer.tokenizer import TokenizerData


def _string_array(
    file: GgufFile, key: String, required: Bool
) raises -> List[String]:
    var out = List[String]()
    if not file.has_metadata(key):
        if required:
            raise Error(
                "tokenizer_missing_vocab_metadata: GGUF file carries no " + key
            )
        return out^
    var idx = file.require_metadata(key)
    var value = file.metadata[idx].value.copy()
    if value.type_id != META_ARRAY or value.array_elem_type != META_STRING:
        raise Error(
            "tokenizer_gguf_type_mismatch: "
            + key
            + " must be an array of strings"
        )
    for s in value.array_strings:
        out.append(s)
    return out^


def _int_array(file: GgufFile, key: String) raises -> List[Int]:
    var out = List[Int]()
    if not file.has_metadata(key):
        return out^
    var n = file.metadata_array_len(key)
    for i in range(n):
        out.append(Int(file.metadata_array_i64(key, i)))
    return out^


def _float_array(file: GgufFile, key: String) raises -> List[Float64]:
    var out = List[Float64]()
    if not file.has_metadata(key):
        return out^
    var idx = file.require_metadata(key)
    var value = file.metadata[idx].value.copy()
    if value.type_id != META_ARRAY or (
        value.array_elem_type != META_F32 and value.array_elem_type != META_F64
    ):
        raise Error(
            "tokenizer_gguf_type_mismatch: "
            + key
            + " must be an array of floats"
        )
    for v in value.array_floats:
        out.append(v)
    return out^


def _int_value(file: GgufFile, key: String) raises -> Int:
    if not file.has_metadata(key):
        return -1
    return Int(file.get_i64(key))


def load_tokenizer_data(path: String) raises -> TokenizerData:
    """Read the tokenizer metadata from the GGUF file at `path`
    into TokenizerData, through the engine's GGUF loader.
    Refuses by name on anything malformed (the loader's gguf_*
    refusals, or this module's tokenizer_* ones)."""
    var file = parse_gguf(path)
    var data = TokenizerData()
    if file.has_metadata("tokenizer.ggml.model"):
        data.model = file.get_string("tokenizer.ggml.model")
        data.has_model = True
    if file.has_metadata("tokenizer.ggml.pre"):
        data.pre = file.get_string("tokenizer.ggml.pre")
        data.has_pre = True
    data.tokens = _string_array(file, "tokenizer.ggml.tokens", True)
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
    data.token_types = _int_array(file, "tokenizer.ggml.token_type")
    data.merges = _string_array(file, "tokenizer.ggml.merges", False)
    if file.has_metadata("tokenizer.ggml.scores"):
        data.scores = _float_array(file, "tokenizer.ggml.scores")
        data.has_scores = True
    data.bos_token_id = _int_value(file, "tokenizer.ggml.bos_token_id")
    data.eos_token_id = _int_value(file, "tokenizer.ggml.eos_token_id")
    data.unknown_token_id = _int_value(file, "tokenizer.ggml.unknown_token_id")
    data.padding_token_id = _int_value(file, "tokenizer.ggml.padding_token_id")
    _ = log_debug(
        "tokenizer",
        "gguf vocab loaded tokens="
        + String(len(data.tokens))
        + " merges="
        + String(len(data.merges))
        + " from "
        + path,
    )
    return data^
