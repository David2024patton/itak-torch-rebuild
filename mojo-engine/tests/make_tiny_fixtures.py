#!/usr/bin/env python3
"""Generate the tiny tokenizer fixtures under tests/fixtures/.

tiny.gguf: a minimal GGUF v3 file (0 tensors) whose tokenizer
metadata carries the same synthetic vocabulary as the Rust
tokenizer crate's tests/roundtrip.rs: ids 0..=255 are the
byte-mapped character for each byte (id == byte), ids 256..=511
are the <0xXX> byte tokens, then the merge products th/the/
Ġthe/he/Ġa at 512..=516, <unk> at 517, <|end_of_text|> at 518
(bos = eos = 518, unk = 517, no padding), model "gpt2",
pre "llama4", merges ["t h", "th e", "Ġ the", "h e", "Ġ a"].
A few unrelated metadata keys (an architecture string, a u32,
an f32 array, an i32 array) ride along so the GGUF seam's
skip-by-type path is exercised too.

tiny.tokref: the ITAK-TOKREF-1 reference for that vocabulary.
The expected ids are HAND-COMPUTED from the vocabulary design
(see the comments per case), not produced by any tokenizer
implementation, so the Mojo parity program checks against an
independent expectation. (The real-vocabulary reference is
crates/tokenizer/conformance/glimmer.tokref, from llama.cpp.)

notoken.gguf: a GGUF file with no tokenizer keys at all; the
seam must refuse it with tokenizer_missing_vocab_metadata.

Usage: python3 tests/make_tiny_fixtures.py
"""

import struct
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "fixtures"

# GGUF value types.
U32, I32, F32, STRING, ARRAY = 4, 5, 6, 8, 9


def gguf_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_string(key: str, value: str) -> bytes:
    return gguf_string(key) + struct.pack("<I", STRING) + gguf_string(value)


def kv_u32(key: str, value: int) -> bytes:
    return gguf_string(key) + struct.pack("<I", U32) + struct.pack("<I", value)


def kv_i32(key: str, value: int) -> bytes:
    return gguf_string(key) + struct.pack("<I", I32) + struct.pack("<i", value)


def kv_string_array(key: str, values: list) -> bytes:
    out = gguf_string(key) + struct.pack("<I", ARRAY)
    out += struct.pack("<I", STRING) + struct.pack("<Q", len(values))
    for v in values:
        out += gguf_string(v)
    return out


def kv_i32_array(key: str, values: list) -> bytes:
    out = gguf_string(key) + struct.pack("<I", ARRAY)
    out += struct.pack("<I", I32) + struct.pack("<Q", len(values))
    for v in values:
        out += struct.pack("<i", v)
    return out


def kv_f32_array(key: str, values: list) -> bytes:
    out = gguf_string(key) + struct.pack("<I", ARRAY)
    out += struct.pack("<I", F32) + struct.pack("<Q", len(values))
    for v in values:
        out += struct.pack("<f", v)
    return out


def gguf_file(kvs: list) -> bytes:
    out = b"GGUF" + struct.pack("<I", 3)
    out += struct.pack("<Q", 0)  # tensor_count
    out += struct.pack("<Q", len(kvs))
    for kv in kvs:
        out += kv
    return out


def byte_to_char(b: int) -> str:
    # The GPT-2 byte alphabet, same rule as the Rust/Mojo code:
    # direct ranges keep their codepoint, the rest take 256+.
    if 0x21 <= b <= 0x7E or 0xA1 <= b <= 0xAC or 0xAE <= b <= 0xFF:
        return chr(b)
    nxt = 256
    for cand in range(256):
        direct = 0x21 <= cand <= 0x7E or 0xA1 <= cand <= 0xAC or 0xAE <= cand <= 0xFF
        if not direct:
            if cand == b:
                return chr(nxt)
            nxt += 1
    raise AssertionError


def synthetic_vocab():
    tokens, types = [], []
    for b in range(256):
        tokens.append(byte_to_char(b))
        types.append(1)  # Normal
    for b in range(256):
        tokens.append(f"<0x{b:02X}>")
        types.append(6)  # Byte
    for merged in ["th", "the", "Ġthe", "he", "Ġa"]:
        tokens.append(merged)
        types.append(1)  # Normal
    tokens.append("<unk>")
    types.append(2)  # Unknown
    tokens.append("<|end_of_text|>")
    types.append(3)  # Control
    merges = ["t h", "th e", "Ġ the", "h e", "Ġ a"]
    return tokens, types, merges


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tokens, types, merges = synthetic_vocab()
    tiny = gguf_file(
        [
            kv_string("general.architecture", "llama"),
            kv_u32("llama.block_count", 1),
            kv_f32_array("test.floats", [1.5, 2.5]),
            kv_i32_array("test.ints", [1, 2, 3]),
            kv_string("tokenizer.ggml.model", "gpt2"),
            kv_string("tokenizer.ggml.pre", "llama4"),
            kv_string_array("tokenizer.ggml.tokens", tokens),
            kv_i32_array("tokenizer.ggml.token_type", types),
            kv_string_array("tokenizer.ggml.merges", merges),
            kv_u32("tokenizer.ggml.bos_token_id", 518),
            kv_u32("tokenizer.ggml.eos_token_id", 518),
            kv_u32("tokenizer.ggml.unknown_token_id", 517),
        ]
    )
    (OUT / "tiny.gguf").write_bytes(tiny)

    notoken = gguf_file([kv_string("general.architecture", "llama")])
    (OUT / "notoken.gguf").write_bytes(notoken)

    # Hand-computed expectations. Single byte-mapped characters
    # carry id == byte value; th=512, the=513, Ġthe=514, he=515,
    # Ġa=516, <unk>=517, <|end_of_text|>=518.
    cases = [
        # "the": one piece; t+h (rank 0) -> th, th+e (rank 1) -> the.
        ("the", [513]),
        # " the": one piece (space prefix); Ġ,t,h,e -> Ġ,th,e ->
        # Ġ,the -> Ġthe (rank 2).
        (" the", [514]),
        # "hello": h+e (rank 3) -> he; l, l, o stay single.
        ("hello", [515, 108, 108, 111]),
        # "a a": pieces "a" and " a"; Ġ+a (rank 4) -> Ġa.
        ("a a", [97, 516]),
        # digits split in runs of at most three; all single here.
        ("1234567", [49, 50, 51, 52, 53, 54, 55]),
        # café: bytes c a f C3 A9 map to ids 99 97 102 195 169.
        ("café", [99, 97, 102, 195, 169]),
        # The control token matches literally between BPE text.
        ("hi<|end_of_text|>bye", [104, 105, 518, 98, 121, 101]),
        # "don't" is one pre-tokenizer piece; no merges apply.
        ("don't", [100, 111, 110, 39, 116]),
    ]
    out = ["ITAK-TOKREF-1", f"cases {len(cases)}"]
    for prompt, ids in cases:
        raw = prompt.encode("utf-8")
        out.append(str(len(raw)))
        out.append(prompt)
        out.append(f"ids {len(ids)}")
        out.append(" ".join(str(i) for i in ids))
    (OUT / "tiny.tokref").write_text("\n".join(out) + "\n", encoding="utf-8")

    print(
        f"wrote tiny.gguf ({len(tiny)} bytes), notoken.gguf"
        f" ({len(notoken)} bytes), tiny.tokref ({len(cases)} cases)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
