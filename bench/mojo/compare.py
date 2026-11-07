#!/usr/bin/env python3
"""Compare K1 fixture outputs with the harness's own metrics.

Bench tooling for the Mojo kernel test (see
bench/mojo-kernel-test-2026-11-07.md). Definitions match
crates/wgpu-compute/examples/k1_check.rs exactly:

  rel    = max_abs_err / max(1.0, scale), scale = max |reference|
  cosine = dot(a, b) / (|a| * |b|), accumulated in float64

Usage:
  compare.py FIXTURE_DIR [MOJO_OUT ...]

With no MOJO_OUT arguments, compares gpu_f32.bin and
mojo_out_f32.bin in FIXTURE_DIR against ref_f32.bin, plus the
direct mojo-vs-gpu pair. Extra arguments are treated as
additional candidate files compared against ref_f32.bin and
gpu_f32.bin.
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


def compare(name_a: str, a: list[float], name_b: str, b: list[float]) -> None:
    if len(a) != len(b):
        raise SystemExit(f"{name_a} has {len(a)} values, {name_b} has {len(b)}")
    max_abs = 0.0
    scale = 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        max_abs = max(max_abs, abs(x - y))
        scale = max(scale, abs(y))
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    cosine = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    rel = max_abs / max(1.0, scale)
    print(
        f"{name_a} vs {name_b}: outputs {len(a)}: "
        f"max_abs_err {max_abs:.6f} at output scale {scale:.3f} "
        f"(rel {rel:.3e}), cosine {cosine:.9f}"
    )


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    fixture = Path(sys.argv[1])
    ref = load_f32(fixture / "ref_f32.bin")
    gpu = load_f32(fixture / "gpu_f32.bin")
    compare("gpu", gpu, "ref", ref)
    candidates = sys.argv[2:] or [str(fixture / "mojo_out_f32.bin")]
    for cand in candidates:
        values = load_f32(Path(cand))
        label = Path(cand).name.removesuffix(".bin")
        compare(label, values, "ref", ref)
        compare(label, values, "gpu", gpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
