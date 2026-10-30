#!/usr/bin/env python3
"""Determinism check for the golden-output reference itself.

The golden harness rests on one claim: llama.cpp at temperature 0
reproduces the same token trajectory every run, so a fixture is a
stable reference. README.md records that claim as verified for
TinyLlama regeneration; this script makes it a check anyone can run,
on any fixture model, before trusting a fresh capture (in particular
the pending v1.1 prompts in PENDING-REFERENCES.md).

For each prompt the script runs llama-server /completion twice at
temperature 0 with the exact gen_golden.py decoding parameters and
requires the two sampled token id sequences to be byte-identical
(same ids, same order, same length). Any difference is reported with
its first divergence index and fails the run.

The server is a throwaway instance started exactly like
gen_golden.py's: CPU only (-ngl 0), localhost, production servers
are never touched. Helpers (tokenize / complete / extract_record)
are imported from gen_golden.py so the check measures the same
pipeline that writes the fixtures, not a reimplementation.

Usage:
    determinism_check.py --model /path/to/model.gguf \
                         [--bin-dir /home/server/llama.cpp/build-vulkan2/bin] \
                         [--prompts prompts.jsonl] [--port 18931] \
                         [--id factual_capital] [--id ...]

    determinism_check.py --self-test

--self-test exercises the comparator on synthetic id sequences (no
server needed): identical runs pass, a flipped id and a truncated
run fail at the expected index.

Exit code 0 means every checked prompt reproduced byte-identically.

Standard library only.
"""

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_gen_golden():
    spec = importlib.util.spec_from_file_location("gen_golden", HERE / "gen_golden.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compare_runs(ids_a, ids_b):
    """Return None if the two id sequences are identical, else
    (index, id_from_a_or_None, id_from_b_or_None) of first difference."""
    n = min(len(ids_a), len(ids_b))
    for i in range(n):
        if ids_a[i] != ids_b[i]:
            return i, ids_a[i], ids_b[i]
    if len(ids_a) != len(ids_b):
        a = ids_a[n] if n < len(ids_a) else None
        b = ids_b[n] if n < len(ids_b) else None
        return n, a, b
    return None


def byte_identical(ids_a, ids_b):
    """Byte-identical means the canonical serializations match too,
    not just list equality: same ids, same length, no coercion."""
    return json.dumps(ids_a) == json.dumps(ids_b) and compare_runs(ids_a, ids_b) is None


def self_test():
    cases = [
        ("identical", [1, 2, 3, 4], [1, 2, 3, 4], None),
        ("flipped middle id", [1, 2, 3, 4], [1, 2, 99, 4], (2, 3, 99)),
        ("truncated run", [1, 2, 3, 4], [1, 2], (2, 3, None)),
        ("extended run", [1, 2], [1, 2, 3], (2, None, 3)),
        ("empty vs empty", [], [], None),
        ("empty vs nonempty", [], [7], (0, None, 7)),
    ]
    failed = 0
    for name, a, b, expected in cases:
        got = compare_runs(a, b)
        ok = got == expected and byte_identical(a, b) == (expected is None)
        print(f"self-test {name}: {'ok' if ok else f'FAIL (got {got}, expected {expected})'}")
        failed += 0 if ok else 1
    print(f"self-test: {len(cases) - failed}/{len(cases)} ok")
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--bin-dir", type=Path,
                    default=Path("/home/server/llama.cpp/build-vulkan2/bin"))
    ap.add_argument("--prompts", type=Path, default=HERE / "prompts.jsonl")
    ap.add_argument("--port", type=int, default=18931)
    ap.add_argument("--id", action="append", dest="ids", default=None,
                    help="only check this prompt id (repeatable)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.model is None:
        ap.error("--model is required unless --self-test is given")

    gg = load_gen_golden()
    prompts = gg.load_prompts(args.prompts)
    if args.ids:
        wanted = set(args.ids)
        prompts = [p for p in prompts if p["id"] in wanted]
        missing = wanted - {p["id"] for p in prompts}
        if missing:
            print(f"unknown prompt id(s): {', '.join(sorted(missing))}")
            return 2
    if not prompts:
        print("no prompts selected")
        return 2

    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as log:
        log_path = log.name
    proc = subprocess.Popen(
        [str(args.bin_dir / "llama-server"), "-m", str(args.model),
         "-ngl", "0", "-c", str(gg.CTX), "-t", str(gg.THREADS),
         "--host", "127.0.0.1", "--port", str(args.port)],
        stdout=open(log_path, "ab"), stderr=subprocess.STDOUT,
    )
    failures = 0
    try:
        gg.wait_health(args.port, proc, log_path)
        for p in prompts:
            prompt_ids = gg.tokenize(args.bin_dir, args.model, p["prompt"])
            runs = []
            for _ in range(2):
                resp = gg.complete(args.port, p["prompt"])
                comp_ids, _, _ = gg.extract_record(resp, prompt_ids)
                runs.append(comp_ids)
            div = compare_runs(runs[0], runs[1])
            if div is None and byte_identical(runs[0], runs[1]):
                print(f"{p['id']}: deterministic ({len(runs[0])} tokens, byte-identical)")
            else:
                failures += 1
                print(f"{p['id']}: NON-DETERMINISTIC, first divergence at index "
                      f"{div[0]} (run A {div[1]} vs run B {div[2]}); "
                      f"lengths {len(runs[0])} vs {len(runs[1])}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    print(f"summary: {len(prompts) - failures}/{len(prompts)} prompts deterministic")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
