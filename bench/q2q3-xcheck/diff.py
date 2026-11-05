#!/usr/bin/env python3
"""Compare ggml reference f32 output against the Rust probe dump.

Usage: diff.py <ref.f32> <rust.txt> [label]

The reference file is little-endian f32 binary. The Rust file is the
gguf probe example's text dump (one shortest-round-trip f32 per line),
which parses back to the exact f32 values.

Reports: value count, bit-exact match count, signed-zero-only mismatch
count (bit differs, value compares equal), NaN mismatch count, max abs
diff, and the worst block (256 values per block) by max abs diff.
"""
import struct
import sys


def main():
    ref_path, rust_path = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else ref_path

    ref_raw = open(ref_path, "rb").read()
    n = len(ref_raw) // 4
    ref = struct.unpack("<%df" % n, ref_raw)
    ref_bits = struct.unpack("<%dI" % n, ref_raw)

    rust_vals = []
    with open(rust_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rust_vals.append(float(line))
    if len(rust_vals) != n:
        print(f"{label}: LENGTH MISMATCH ref={n} rust={len(rust_vals)}")
        return 1
    rust_raw = struct.pack("<%df" % n, *rust_vals)
    rust = struct.unpack("<%df" % n, rust_raw)
    rust_bits = struct.unpack("<%dI" % n, rust_raw)

    exact = 0
    zero_sign = 0
    nan_mismatch = 0
    max_diff = 0.0
    block_max = {}
    for i in range(n):
        a, b = ref[i], rust[i]
        if ref_bits[i] == rust_bits[i]:
            exact += 1
            continue
        if a != a or b != b:  # NaN on either side
            nan_mismatch += 1
            continue
        d = abs(a - b)
        if d == 0.0:
            zero_sign += 1
            continue
        if d > max_diff:
            max_diff = d
        blk = i // 256
        if d > block_max.get(blk, 0.0):
            block_max[blk] = d

    worst = sorted(block_max.items(), key=lambda kv: -kv[1])[:5]
    print(f"{label}: values={n} exact={exact} ({100.0 * exact / n:.4f}%) "
          f"zero_sign_only={zero_sign} nan_mismatch={nan_mismatch} "
          f"max_abs_diff={max_diff:.6g}")
    for blk, d in worst:
        print(f"  worst block {blk}: max_abs_diff={d:.6g}")
    return 0 if (nan_mismatch == 0 and max_diff == 0.0) else 1


if __name__ == "__main__":
    sys.exit(main())
