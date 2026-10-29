#!/usr/bin/env python3
"""Diff an engine's generation output against llama.cpp golden fixtures.

Usage:
    compare.py GOLDEN.jsonl CANDIDATE.jsonl [--strict]

Both files are JSONL in the golden fixture shape written by gen_golden.sh:
an optional first line {"meta": {...}} followed by one record per prompt:

    {"id": ..., "prompt": ...,
     "prompt_token_ids": [...],
     "completion_token_ids": [...],
     "completion_text": "..."}

Per prompt the verdict is one of:

    token-identical   completion token ids match the golden ids exactly
    text-identical    ids differ but the completion text matches exactly
    divergent         neither matches; the first divergence is reported
                      in token space (index, golden id vs candidate id)
                      and in text space (character offset)
    missing           the candidate has no record for this prompt

Exit code is 0 when every prompt is token-identical or text-identical,
1 otherwise. With --strict, text-identical also fails: verified text
(TODO T-9.4) means token-identical at temperature 0.

Standard library only, so this runs anywhere python3 does.
"""

import json
import sys


def load_records(path):
    meta = None
    records = {}
    order = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "meta" in obj and "id" not in obj:
                meta = obj["meta"]
                continue
            records[obj["id"]] = obj
            order.append(obj["id"])
    return meta, records, order


def first_token_divergence(golden_ids, cand_ids):
    """Return (index, golden_id_or_None, cand_id_or_None) of first mismatch."""
    n = min(len(golden_ids), len(cand_ids))
    for i in range(n):
        if golden_ids[i] != cand_ids[i]:
            return i, golden_ids[i], cand_ids[i]
    if len(golden_ids) != len(cand_ids):
        g = golden_ids[n] if n < len(golden_ids) else None
        c = cand_ids[n] if n < len(cand_ids) else None
        return n, g, c
    return None


def first_text_divergence(golden_text, cand_text):
    n = min(len(golden_text), len(cand_text))
    for i in range(n):
        if golden_text[i] != cand_text[i]:
            return i
    if len(golden_text) != len(cand_text):
        return n
    return None


def fmt_tok(t):
    return "EOS/end" if t is None else str(t)


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    strict = "--strict" in argv
    if len(args) != 2:
        print(__doc__)
        return 2
    golden_path, cand_path = args
    _, golden, order = load_records(golden_path)
    _, cand, _ = load_records(cand_path)

    counts = {"token-identical": 0, "text-identical": 0, "divergent": 0, "missing": 0}
    for pid in order:
        g = golden[pid]
        c = cand.get(pid)
        if c is None:
            counts["missing"] += 1
            print(f"{pid}: missing (no candidate record)")
            continue

        notes = []
        if c.get("prompt_token_ids") != g["prompt_token_ids"]:
            div = first_token_divergence(g["prompt_token_ids"], c.get("prompt_token_ids", []))
            notes.append(
                f"prompt tokenization differs at index {div[0]} "
                f"(golden {fmt_tok(div[1])} vs candidate {fmt_tok(div[2])})"
            )

        g_ids, c_ids = g["completion_token_ids"], c["completion_token_ids"]
        if g_ids == c_ids:
            verdict = "token-identical"
        elif g["completion_text"] == c["completion_text"]:
            verdict = "text-identical"
            div = first_token_divergence(g_ids, c_ids)
            notes.append(
                f"first token divergence at index {div[0]} "
                f"(golden {fmt_tok(div[1])} vs candidate {fmt_tok(div[2])}), text still identical"
            )
        else:
            verdict = "divergent"
            div = first_token_divergence(g_ids, c_ids)
            if div is not None:
                notes.append(
                    f"first token divergence at index {div[0]} "
                    f"(golden {fmt_tok(div[1])} vs candidate {fmt_tok(div[2])})"
                )
            tdiv = first_text_divergence(g["completion_text"], c["completion_text"])
            if tdiv is not None:
                g_ctx = g["completion_text"][max(0, tdiv - 20):tdiv + 20]
                c_ctx = c["completion_text"][max(0, tdiv - 20):tdiv + 20]
                notes.append(f"first text divergence at char {tdiv}: golden ...{g_ctx!r} vs candidate ...{c_ctx!r}")

        counts[verdict] += 1
        suffix = "; ".join(notes)
        print(f"{pid}: {verdict}" + (f" ({suffix})" if suffix else ""))

    extra = [pid for pid in cand if pid not in golden]
    if extra:
        print(f"warning: candidate has {len(extra)} record(s) not in golden: {', '.join(extra)}")

    total = len(order)
    print(
        f"summary: {counts['token-identical']}/{total} token-identical, "
        f"{counts['text-identical']} text-identical, "
        f"{counts['divergent']} divergent, {counts['missing']} missing"
    )
    ok = counts["divergent"] == 0 and counts["missing"] == 0
    if strict:
        ok = ok and counts["text-identical"] == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
