# tokenizer_parity.mojo - the tokenizer conformance harness.
#
# Mirror of crates/tokenizer/examples/conformance.rs: loads a
# vocabulary from a GGUF file (through the lane's temporary
# gguf_vocab seam), encodes every prompt in a .tokref reference,
# compares id sequences exactly, and decodes the reference ids
# back to the prompt text. Exit 0 only when every case is
# token-identical and every decode round-trips; exit 1 on any
# mismatch; exit 2 with a REFUSAL line on a load/parse refusal.
#
# Usage (from the mojo-engine/ directory):
#   mojo run -I src tests/tokenizer_parity.mojo [gguf] [tokref]
# Defaults: the model path from src/config.mojo
# (MOJO_ENGINE_MODEL_PATH over DEFAULT_MODEL_PATH) and the
# committed Glimmer reference at
# ../crates/tokenizer/conformance/glimmer.tokref (llama.cpp
# /tokenize output on the real vocabulary; see
# crates/tokenizer/CONFORMANCE.md). For a VM smoke run against
# the synthetic fixture:
#   mojo run -I src tests/tokenizer_parity.mojo \
#       tests/fixtures/tiny.gguf tests/fixtures/tiny.tokref

from std.sys import argv, exit

from config import EngineConfig
from tokenizer.byte_level import (
    codepoints_of,
    decode_utf8_strict,
    string_from_codepoints,
)
from tokenizer.gguf_vocab import load_tokenizer_data
from tokenizer.tokenizer import Tokenizer


struct RefCase(Copyable, Movable):
    var prompt: String
    var ids: List[Int]

    def __init__(out self, var prompt: String, var ids: List[Int]):
        self.prompt = prompt^
        self.ids = ids^


struct Cursor(Movable):
    var bytes: List[UInt8]
    var pos: Int

    def __init__(out self, var bytes: List[UInt8]):
        self.bytes = bytes^
        self.pos = 0

    def line(mut self) raises -> String:
        var start = self.pos
        while self.pos < len(self.bytes) and Int(self.bytes[self.pos]) != 0x0A:
            self.pos += 1
        var slice = List[UInt8]()
        for i in range(start, self.pos):
            slice.append(self.bytes[i])
        if self.pos < len(self.bytes):
            self.pos += 1  # the newline itself
        # Strip a trailing CR, as the Rust harness's line read
        # trims both.
        if len(slice) > 0 and Int(slice[len(slice) - 1]) == 0x0D:
            _ = slice.pop()
        var cps = decode_utf8_strict(slice)
        return string_from_codepoints(cps, 0, len(cps))

    def framed(mut self) raises -> String:
        var len_text = self.line()
        var n = Int(len_text)
        if self.pos + n > len(self.bytes):
            raise Error("tokenizer_parity: tokref frame overruns the file")
        var slice = List[UInt8]()
        for i in range(n):
            slice.append(self.bytes[self.pos + i])
        self.pos += n
        if self.pos >= len(self.bytes) or Int(self.bytes[self.pos]) != 0x0A:
            raise Error("tokenizer_parity: tokref frame not newline-terminated")
        self.pos += 1
        var cps = decode_utf8_strict(slice)
        return string_from_codepoints(cps, 0, len(cps))


def _field(line: String, key: String) raises -> String:
    var prefix = key + " "
    if not line.startswith(prefix):
        raise Error(
            "tokenizer_parity: expected '" + key + " ...' line, found: " + line
        )
    var cps = codepoints_of(line)
    var rest = List[Int]()
    var skip = len(codepoints_of(prefix))
    for i in range(skip, len(cps)):
        rest.append(cps[i])
    return string_from_codepoints(rest, 0, len(rest))


def load_reference(path: String) raises -> List[RefCase]:
    var raw: List[UInt8]
    with open(path, "r") as f:
        raw = f.read_bytes(1 << 26)
    var cur = Cursor(raw^)
    var header = cur.line()
    if header != "ITAK-TOKREF-1":
        raise Error("tokenizer_parity: bad tokref header: " + header)
    var n_cases = Int(_field(cur.line(), "cases"))
    var cases = List[RefCase]()
    for _ in range(n_cases):
        var prompt = cur.framed()
        var n_ids = Int(_field(cur.line(), "ids"))
        var id_line = cur.line()
        var ids = List[Int]()
        if n_ids > 0:
            var parts = id_line.split(" ")
            for part in parts:
                if part.byte_length() > 0:
                    ids.append(Int(part))
        if len(ids) != n_ids:
            raise Error(
                "tokenizer_parity: expected "
                + String(n_ids)
                + " ids, found "
                + String(len(ids))
            )
        cases.append(RefCase(prompt^, ids^))
    return cases^


def display(prompt: String) -> String:
    """The prompt with control characters escaped, capped, for
    the per-case report line."""
    var out = String()
    var cps = codepoints_of(prompt)
    var shown = 0
    for cp in cps:
        if shown >= 72:
            out += "..."
            break
        if cp == 0x0A:
            out += "\\n"
        elif cp == 0x0D:
            out += "\\r"
        elif cp == 0x09:
            out += "\\t"
        else:
            out += string_from_codepoints(List[Int]([cp]), 0, 1)
        shown += 1
    return out^


def first_divergence(ours: List[Int], reference: List[Int]) -> Int:
    var shared = len(ours) if len(ours) < len(reference) else len(reference)
    for i in range(shared):
        if ours[i] != reference[i]:
            return i
    if len(ours) != len(reference):
        return shared
    return -1


def run() raises -> Int:
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    var gguf_path: String
    var tokref_path: String
    if len(args) > 1:
        gguf_path = args[1]
    else:
        gguf_path = EngineConfig.load().model_path
    if len(args) > 2:
        tokref_path = args[2]
    else:
        tokref_path = "../crates/tokenizer/conformance/glimmer.tokref"

    var data = load_tokenizer_data(gguf_path)
    var model = data.model if data.has_model else String("-")
    var pre = data.pre if data.has_pre else String("-")
    var n_tokens = len(data.tokens)
    var bos = data.bos_token_id
    var eos = data.eos_token_id
    var unk = data.unknown_token_id
    var pad = data.padding_token_id
    var tok = Tokenizer.build(data^)
    print(
        "vocab: gguf "
        + gguf_path
        + " model="
        + model
        + " pre="
        + pre
        + "; "
        + String(n_tokens)
        + " tokens, bos="
        + String(bos)
        + " eos="
        + String(eos)
        + " unk="
        + String(unk)
        + " pad="
        + String(pad)
    )
    var cases = load_reference(tokref_path)
    var exact = 0
    var roundtrip_ok = 0
    for idx in range(len(cases)):
        var prompt = cases[idx].prompt
        var reference = cases[idx].ids.copy()
        var ours = tok.encode(prompt)
        var decoded = tok.decode(reference)
        var ids_match = len(ours) == len(reference)
        if ids_match:
            for i in range(len(ours)):
                if ours[i] != reference[i]:
                    ids_match = False
                    break
        var text_match = decoded == prompt
        if ids_match:
            exact += 1
        if text_match:
            roundtrip_ok += 1
        print(
            "case "
            + String(idx)
            + ": "
            + (String("EXACT") if ids_match else String("DIFF"))
            + " ("
            + String(len(reference))
            + " tokens) decode-roundtrip "
            + (String("ok") if text_match else String("DIFF"))
            + ' -- prompt "'
            + display(prompt)
            + '"'
        )
        if not ids_match:
            var at = first_divergence(ours, reference)
            print("  first divergence at token index " + String(at))
            if at >= 0 and at < len(reference):
                print(
                    "  reference["
                    + String(at)
                    + "] = "
                    + String(reference[at])
                    + ' "'
                    + display(tok.token_text(reference[at]))
                    + '"'
                )
            if at >= 0 and at < len(ours):
                print(
                    "  ours     ["
                    + String(at)
                    + "] = "
                    + String(ours[at])
                    + ' "'
                    + display(tok.token_text(ours[at]))
                    + '"'
                )
            print("  reference ids:", end="")
            for id in reference:
                print(" " + String(id), end="")
            print("")
            print("  ours ids:     ", end="")
            for id in ours:
                print(" " + String(id), end="")
            print("")
        if not text_match:
            print('  decoded reference ids: "' + display(decoded) + '"')
    print(
        "conformance: "
        + String(exact)
        + "/"
        + String(len(cases))
        + " prompts token-identical; "
        + String(roundtrip_ok)
        + "/"
        + String(len(cases))
        + " decode round-trips exact"
    )
    return 0 if exact == len(cases) and roundtrip_ok == len(cases) else 1


def main():
    try:
        exit(run())
    except e:
        print("REFUSAL", e)
        exit(2)
