#!/usr/bin/env python3
"""Regression tests for the golden verdict tools' verdict plumbing.

Pins the critic-panel Category F fixes (2026-11-07):

- compare_traces: the first values gate the match (a permuted tensor
  with identical mean/variance must print DIVERGENT), and the process
  exits 1 on a divergence or a stage with no reference, 0 only on a
  full match.
- compare_layers stats(): cosine is 0.0 when exactly one side is all
  zeros, 1.0 only when both are (or when the tensors truly agree).

Fixtures are generated in a temp dir; nothing is read from the tree
beyond the tools themselves. Usage: test_verdict_tools.py; exits 0
when every case passes.
"""

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FAILURES = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def run_traces(trace_path, ref_dir):
    return subprocess.run(
        [sys.executable, str(HERE / "compare_traces.py"), str(trace_path), str(ref_dir)],
        capture_output=True, text=True)


def trace_line(stage, first):
    f = ",".join(repr(float(v)) for v in first)
    return (f"stage={stage} rows=1 cols=4 n=4 mean=2.5 var=1.25 max_abs=4.0 "
            f"last_mean=2.5 last_var=1.25 last_max_abs=4.0 "
            f"first=[{f}] last_first=[{f}]")


def main():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ref = td / "ref"
        ref.mkdir()
        # Reference embed tensor [1,2,3,4].
        np.array([1, 2, 3, 4], dtype="<f4").tofile(ref / "embed.bin")
        (ref / "manifest.json").write_text(json.dumps({"embed": [4, 1]}))

        # A permutation has identical mean, population variance and
        # max_abs; only the first values differ. It must NOT match.
        perm = td / "permuted.txt"
        perm.write_text(trace_line("embed", [4, 3, 2, 1]) + "\n")
        r = run_traces(perm, ref)
        check("compare_traces: permuted tensor is DIVERGENT",
              "DIVERGENT" in r.stdout and "first divergent stage: embed" in r.stdout,
              f"rc={r.returncode}")
        check("compare_traces: permuted tensor exits 1", r.returncode == 1)

        # The faithful trace matches and exits 0.
        clean = td / "clean.txt"
        clean.write_text(trace_line("embed", [1, 2, 3, 4]) + "\n")
        r = run_traces(clean, ref)
        check("compare_traces: faithful trace matches and exits 0",
              "match" in r.stdout and "first divergent stage: None" in r.stdout
              and r.returncode == 0, f"rc={r.returncode}")

        # A stage with no reference tensor fails the run.
        noref = td / "noref.txt"
        noref.write_text(trace_line("embed", [1, 2, 3, 4]) + "\n"
                         + trace_line("layer.0.out", [1, 2, 3, 4]) + "\n")
        r = run_traces(noref, ref)
        check("compare_traces: NO REF stage exits 1",
              "NO REF" in r.stdout and r.returncode == 1, f"rc={r.returncode}")

    spec = importlib.util.spec_from_file_location("compare_layers", HERE / "compare_layers.py")
    cl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cl)
    zeros = np.zeros((2, 4))
    live = np.full((2, 4), 0.05)
    check("compare_layers stats: dead vs live cosine is 0.0",
          cl.stats(zeros, live)[2] == 0.0, f"cos={cl.stats(zeros, live)[2]}")
    check("compare_layers stats: live vs dead cosine is 0.0",
          cl.stats(live, zeros)[2] == 0.0)
    check("compare_layers stats: both dead cosine is 1.0",
          cl.stats(zeros, zeros)[2] == 1.0)
    check("compare_layers stats: identical live cosine is 1.0",
          abs(cl.stats(live, live)[2] - 1.0) < 1e-12,
          f"cos={cl.stats(live, live)[2]}")

    print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nall cases passed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
