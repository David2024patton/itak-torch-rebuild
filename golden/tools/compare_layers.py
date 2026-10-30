#!/usr/bin/env python3
"""Layer-by-layer activation diff: llama.cpp cb_eval dump vs
arch-llama forward_traced dump (follow-up to integration run A).

Usage: compare_layers.py <llama_dir> <rust_dir>

Both directories hold <stem>.bin raw little-endian f32 files in
[rows][width] layout plus manifest.json mapping stem -> [width,
rows]. Reports, for every checkpoint in execution order, the
max-abs difference and cosine similarity, a per-position breakdown
for l_out, and a RoPE pairing experiment: it re-rotates the Rust
pre-RoPE Q/K dumps under both pairing conventions (interleaved
adjacent pairs, ggml NORMAL; half-split pairs, ggml NEOX) and
reports which one reproduces llama.cpp's Qcur/Kcur.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np


def load(d, stem, manifest, shape=None):
    """Load a Rust dump tensor: manifest is [width, rows] and the
    file is [rows][width] row-major."""
    dims = manifest[stem]
    a = np.fromfile(Path(d) / f"{stem}.bin", dtype="<f4")
    total = 1
    for v in dims:
        total *= v
    assert a.size == total, f"{stem}: {a.size} != prod({dims})"
    if shape is not None:
        assert int(np.prod(shape)) == a.size, f"{stem}: cannot view as {shape}"
        return a.reshape(shape)
    if len(dims) == 2:
        return a.reshape(dims[1], dims[0])
    return a


def load_llama(d, stem, manifest):
    """Load a llama.cpp dump tensor, returned as [rows][width].

    Most graph tensors are ggml [width, rows]. Qcur is the
    exception in this llama.cpp version: its ggml dims are
    [head_dim, rows, heads] (the rope node views Q token-major),
    so the flat dump is [heads][rows][head_dim] and is transposed
    back here. Verified empirically: only under this reading does
    Qcur at position 0 (where RoPE is the identity) reproduce the
    split attn_q projection of the proven attn_norm input.
    """
    dims = manifest[stem]
    a = np.fromfile(Path(d) / f"{stem}.bin", dtype="<f4")
    total = 1
    for v in dims:
        total *= v
    assert a.size == total, f"{stem}: {a.size} != prod({dims})"
    if stem.endswith("_q") and len(dims) == 3:
        dim, rows, heads = dims
        return a.reshape(heads, rows, dim).transpose(1, 0, 2).reshape(rows, heads * dim)
    if len(dims) == 2:
        return a.reshape(dims[1], dims[0])
    raise ValueError(f"{stem}: unexpected llama dims {dims}")


def stats(a, b):
    d = np.abs(a - b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cos = float(np.dot(a.ravel(), b.ravel()) / denom) if denom > 0 else 1.0
    return float(d.max()), float(d.mean()), cos


def rope(x, positions, base, mode):
    """x: [rows][heads][dim]. Rotate per head vector at its position.
    mode 'neox': pair (i, i + dim/2); mode 'interleaved': pair (2i, 2i+1).
    Frequency for pair i in both: base^(-2i/dim)."""
    rows, heads, dim = x.shape
    out = x.copy()
    half = dim // 2
    freqs = base ** (-2.0 * np.arange(half) / dim)
    for r, p in enumerate(positions):
        ang = p * freqs
        cos = np.cos(ang)[None, :, None]  # [1][half][1]
        sin = np.sin(ang)[None, :, None]
        if mode == "neox":
            x1 = x[r, :, :half]
            x2 = x[r, :, half:]
            out[r, :, :half] = x1 * cos[:, :, 0] - x2 * sin[:, :, 0]
            out[r, :, half:] = x2 * cos[:, :, 0] + x1 * sin[:, :, 0]
        else:
            pairs = x[r].reshape(heads, half, 2)
            x1 = pairs[:, :, 0]
            x2 = pairs[:, :, 1]
            rot = np.empty_like(pairs)
            rot[:, :, 0] = x1 * cos[:, :, 0] - x2 * sin[:, :, 0]
            rot[:, :, 1] = x2 * cos[:, :, 0] + x1 * sin[:, :, 0]
            out[r] = rot.reshape(heads, dim)
    return out


def main():
    llama_dir, rust_dir = sys.argv[1], sys.argv[2]
    lm = json.loads((Path(llama_dir) / "manifest.json").read_text())
    rm = json.loads((Path(rust_dir) / "manifest.json").read_text())
    n_layers = 28
    positions = [0, 1, 2, 3]

    order = ["embd"]
    for il in range(n_layers):
        for s in ("attn_norm", "q", "k", "v", "ffn_inp", "ffn_norm", "ffn_out", "l_out"):
            order.append(f"L{il:02d}_{s}")
    order += ["result_norm", "logits"]

    print(f"{'checkpoint':22s} {'max_abs':>12s} {'mean_abs':>12s} {'cosine':>10s}")
    first_div = None
    for stem in order:
        b = load(rust_dir, stem, rm)
        a = load_llama(llama_dir, stem, lm)
        if stem in ("result_norm", "logits"):
            a = a[-1:]  # llama dump carries all positions; rust only the last
        assert a.shape == b.shape, f"{stem}: llama {a.shape} vs rust {b.shape}"
        mx, mean, cos = stats(a, b)
        flag = ""
        if cos < 0.999 or mx > 0.1:
            flag = "  <-- DIVERGENT"
            if first_div is None:
                first_div = stem
        print(f"{stem:22s} {mx:12.6f} {mean:12.6f} {cos:10.6f}{flag}")
    print(f"\nfirst divergent checkpoint: {first_div}")

    print("\nl_out per-position max_abs:")
    print(f"{'layer':>5s} " + " ".join(f"{'pos'+str(p):>12s}" for p in positions))
    for il in range(n_layers):
        stem = f"L{il:02d}_l_out"
        b = load(rust_dir, stem, rm)
        a = load_llama(llama_dir, stem, lm)
        per = [float(np.abs(a[p] - b[p]).max()) for p in positions]
        print(f"{il:5d} " + " ".join(f"{v:12.6f}" for v in per))

    print("\nRoPE pairing experiment (base 1e6, head_dim 128):")
    for il in (0, 1, 13, 27):
        for kind, heads in (("q", 12), ("k", 2)):
            pre = load(rust_dir, f"L{il:02d}_{kind}_pre_rope", rm).reshape(4, heads, 128)
            ref = load_llama(llama_dir, f"L{il:02d}_{kind}", lm).reshape(4, heads, 128)
            ours = load(rust_dir, f"L{il:02d}_{kind}", rm).reshape(4, heads, 128)
            for mode in ("neox", "interleaved"):
                rot = rope(pre.astype(np.float64), positions, 1e6, mode)
                mx, _, cos = stats(ref, rot.astype(np.float32))
                print(f"  L{il:02d} {kind}: rust-pre + {mode:12s} vs llama {kind}cur: "
                      f"max_abs={mx:.6f} cos={cos:.6f}")
            mx, _, cos = stats(ref, ours)
            print(f"  L{il:02d} {kind}: rust {kind} (as built)   vs llama {kind}cur: "
                  f"max_abs={mx:.6f} cos={cos:.6f}")
            # position 0 only: rotation is identity there in any convention
            mx0, _, _ = stats(ref[0], ours[0])
            print(f"        position 0 alone: max_abs={mx0:.6f}")


if __name__ == "__main__":
    main()
