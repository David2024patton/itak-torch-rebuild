#!/usr/bin/env python3
"""Regression tests for diff.py's bit-exact verdict.

Pins the critic-panel Category F fix (2026-11-07): the exit status is
0 only when every value's bits match. A signed-zero-only difference
(bits differ, values compare equal) fails, as does a NaN whose bit
pattern differs; identical dumps pass.

Usage: test_diff.py; exits 0 when every case passes.
"""

import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAILURES = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def run_diff(ref_bytes, rust_lines):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "ref.f32").write_bytes(ref_bytes)
        (td / "rust.txt").write_text("".join(f"{v}\n" for v in rust_lines))
        r = subprocess.run(
            [sys.executable, str(HERE / "diff.py"), str(td / "ref.f32"),
             str(td / "rust.txt"), "test"],
            capture_output=True, text=True)
        return r


def main():
    # Signed zeros: +0.0 vs -0.0 in two positions. Values compare
    # equal, bits differ: the bit-exact verdict must fail.
    r = run_diff(struct.pack("<4f", 0.0, 1.5, -0.0, 2.0),
                 ["-0.0", "1.5", "0.0", "2.0"])
    check("diff.py: signed-zero-only difference exits 1",
          r.returncode == 1 and "zero_sign_only=2" in r.stdout,
          f"rc={r.returncode}")

    # Identical dumps pass.
    r = run_diff(struct.pack("<4f", 1.5, -2.25, 0.0, 3.0),
                 ["1.5", "-2.25", "0.0", "3.0"])
    check("diff.py: identical dumps exit 0",
          r.returncode == 0 and "exact=4 (100.0000%)" in r.stdout,
          f"rc={r.returncode}")

    # A real value mismatch fails.
    r = run_diff(struct.pack("<f", 1.0), ["2.0"])
    check("diff.py: value mismatch exits 1", r.returncode == 1,
          f"rc={r.returncode}")

    # A NaN with a different payload bit pattern fails: the text
    # "nan" packs to the canonical quiet NaN, the reference carries
    # a payload bit.
    r = run_diff(struct.pack("<I", 0x7FC00001), ["nan"])
    check("diff.py: NaN bit difference exits 1",
          r.returncode == 1 and "nan_mismatch=1" in r.stdout,
          f"rc={r.returncode}")

    print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nall cases passed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
