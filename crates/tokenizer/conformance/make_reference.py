#!/usr/bin/env python3
"""Generate the tokenizer conformance reference from a live llama-server.

This is the bench prompt set for TODO T-2.3: sixteen prompts chosen to
stress the edges where BPE tokenizers diverge (spacing, punctuation
clusters, long digit runs, contractions, accents, CJK, code, newline
runs, boundary whitespace, symbols). Each prompt is sent to the
reference server's /tokenize endpoint with add_special disabled, so the
comparison is pure tokenization with no BOS/EOS injection, matching
`tokenizer::Tokenizer::encode`.

Usage:
    LLAMA_BASE=http://127.0.0.1:8080 python3 make_reference.py [out.tokref]

The output is the `.tokref` format (version 1) read by the tokenizer
crate's `conformance` example:

    ITAK-TOKREF-1
    cases <count>
    <prompt byte length>\n<prompt bytes>\n
    ids <count>
    <space-separated token ids>
    (repeated per case)

Run it against David's production Glimmer server on the server itself;
the committed glimmer.tokref was generated exactly that way.
"""

import json
import os
import sys
import urllib.request

BASE = os.environ.get("LLAMA_BASE", "http://127.0.0.1:8080")

PROMPTS = [
    # Plain prose with ordinary single spacing.
    "The capital of France is Paris, and it is known for the Eiffel Tower.",
    # Double spaces between words.
    "Hello  world, this  sentence has  double spaces in  it.",
    # Punctuation clusters, quotes, ellipsis character.
    "Wait... what?! Really?? She paused\u2026 then said: 'stop' \u2014 twice.",
    # A digit run far past the 1-to-3 digit split rule.
    "The invoice total came to 12345678901234567890 dollars exactly.",
    # Mixed digits: decimals, a date, a time, leading zeros.
    "Version 3.14159 shipped on 2026-10-29 at 13:37:35 with build 007.",
    # Contractions, including the double contraction you'd've.
    "I don't know if we'll make it, but they're saying it isn't over yet; "
    "you'd've thought we'd have learned, and I'm not sure she's ready.",
    # Accented Latin text across several languages.
    "Le caf\u00e9 na\u00eff servait un r\u00e9sum\u00e9 avec la pi\u00f1ata "
    "et le jalape\u00f1o, fa\u00e7ade d\u00e9j\u00e0 vu, se\u00f1or.",
    # CJK sentences (Japanese and Chinese).
    "\u4eba\u5de5\u77e5\u80fd\u306f\u672a\u6765\u3067\u3059\u3002"
    "\u6771\u4eac\u306f\u65e5\u672c\u306e\u9996\u90fd\u3067\u3059\u3002"
    "\u5317\u4eac\u4e5f\u662f\u9996\u90fd\u3002",
    # Latin and CJK mixed in one line.
    "Tokyo \u6771\u4eac is a city, and \u5317\u4eac (Beijing) is another one.",
    # A Rust code snippet: brackets, operators, a format string.
    'fn main() { let xs: Vec<u32> = (0..10).map(|i| i * i).collect(); '
    'println!("{:?}", xs); }',
    # Newline-heavy: blank lines, space indent, a tab, trailing newline.
    "first line\n\n\n    indented by four\n\ttabbed line\nlast line\n",
    # Leading and trailing spaces around short text.
    "   leading spaces and trailing spaces   ",
    # Text that starts with blank lines.
    "\n\n  hello after blank lines\n\n",
    # Currency, symbols, an arrow, an emoji.
    "Total: $5, \u20ac10, \u00a33.50 (\u00b10.5\u00b0C) \u2192 ship it "
    "\U0001f680 today \u00a9 2026.",
    # Identifier shapes: camelCase, snake_case, digits, dunder.
    "getUserById calls max_token_count on the HTTPServer2 instance __privately.",
    # Contraction placement stress: a leading apostrophe word, a
    # digit-adjacent contraction, a double contraction, all in prose.
    "'Tis the season, the 90's are back, and you'd've agreed, hadn't you?",
]


def tokenize(content: str) -> list:
    body = json.dumps({"content": content, "add_special": False}).encode("utf-8")
    req = urllib.request.Request(
        BASE + "/tokenize", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))["tokens"]


def main() -> int:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "glimmer.tokref"
    cases = []
    for i, prompt in enumerate(PROMPTS):
        ids = tokenize(prompt)
        cases.append((prompt, ids))
        print(f"case {i}: {len(ids)} tokens")
    with open(out_path, "wb") as f:
        f.write(b"ITAK-TOKREF-1\n")
        f.write(f"cases {len(cases)}\n".encode("ascii"))
        for prompt, ids in cases:
            raw = prompt.encode("utf-8")
            f.write(f"{len(raw)}\n".encode("ascii"))
            f.write(raw + b"\n")
            f.write(f"ids {len(ids)}\n".encode("ascii"))
            f.write((" ".join(str(t) for t in ids) + "\n").encode("ascii"))
    print(f"wrote {len(cases)} cases -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
