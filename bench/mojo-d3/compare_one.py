#!/usr/bin/env python3
"""Compare one kernel output against a fixture's Float64
reference, with the k1_check metric definitions (the same
definitions bench/mojo/compare.py carries):

  rel    = max_abs_err / max(1.0, scale), scale = max |ref|
  cosine = dot(a, b) / (|a| * |b|), accumulated in float64

Usage:
  compare_one.py FIXTURE_DIR CANDIDATE_BIN [LABEL]

Prints one COMPARE line and exits 0; a length mismatch or an
unreadable file exits 2 (the harness treats that as a failed
gate, never as a pass).
"""

import math
import struct
import sys
from pathlib import Path


def load_f32(path: Path) -> list[float]:
    data = path.read_bytes()
    if len(data) % 4 != 0:
        raise SystemExit(f"{path}: {len(data)} bytes is not a whole f32 count")
    return list(struct.unpack(f"<{len(data) // 4}f", data))


def main() -> int:
    fixture = Path(sys.argv[1])
    cand_path = Path(sys.argv[2])
    label = sys.argv[3] if len(sys.argv) > 3 else cand_path.name
    ref = load_f32(fixture / "ref_f32.bin")
    cand = load_f32(cand_path)
    if len(cand) != len(ref):
        print(f"COMPARE {label}: LENGTH MISMATCH cand {len(cand)} vs ref {len(ref)}")
        return 2
    max_abs = 0.0
    scale = 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(cand, ref):
        max_abs = max(max_abs, abs(a - b))
        scale = max(scale, abs(b))
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    cosine = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    rel = max_abs / max(1.0, scale)
    print(
        f"COMPARE {label}: outputs {len(ref)}: max_abs_err {max_abs:.6f} "
        f"at output scale {scale:.3f} (rel {rel:.3e}), cosine {cosine:.12f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
