#!/usr/bin/env python3
"""Build the Wave D lane D3 GEMV fixtures from the real Glimmer
GGUF file (bench tooling for the Q6_K / Q5_K / F32 tuning
ladder; see bench/mojo-d3-gemv-ladder-2026-11-07.md).

For each fixture this writes, into its own directory:
  weights_<kind>.bin   raw tensor bytes (a whole number of
                       leading rows of the named tensor, or
                       the stacked F32 rows / synthetic matrix
                       for the plain-float fixtures)
  input_f32.bin        the k2_check LCG input (seed 42, the
                       64-bit LCG of crates/wgpu-compute's
                       k2_check/k3_check examples)
  ref_f32.bin          an independent Float64 reference:
                       numpy dequant that mirrors
                       crates/gguf/src/dequant.rs loop for
                       loop, dotted in float64

Usage:
  extract_fixtures.py MODEL.gguf OUT_DIR

Fixtures built:
  q6k_corr   blk.0.ffn_down.weight rows 0..63,  cols 19968
  q6k_time   blk.0.ffn_down.weight all 6656 rows, cols 19968
  q5k_corr   output.weight rows 0..63,          cols 6656
  q5k_time   output.weight rows 0..16383,       cols 6656
  f32_corr   first 64 F32 tensors of 6656 elements in file
             order, stacked as rows (cols 6656)
  f16_corr   the F16 encoding of those same rows
  f32_time   SYNTHETIC 6656x6656 F32 matrix (LCG weights;
             named synthetic: no real F32 GEMV tensor is
             large enough to time)
  f16_time   the F16 encoding of that synthetic matrix
"""

import struct
import sys
from pathlib import Path

import numpy as np

TYPE_INFO = {0: (1, 4), 1: (1, 2), 12: (256, 144), 13: (256, 176), 14: (256, 210)}
F32_ID, Q5K_ID, Q6K_ID = 0, 13, 14


class Reader:
    def __init__(self, data: bytes):
        self.d = data
        self.o = 0

    def take(self, n: int) -> bytes:
        b = self.d[self.o : self.o + n]
        if len(b) != n:
            raise SystemExit("truncated GGUF")
        self.o += n
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> str:
        n = self.u64()
        return self.take(n).decode("utf-8")


def read_metadata_value(r: Reader, ty: int):
    if ty == 0:
        return struct.unpack("<B", r.take(1))[0]
    if ty == 1:
        return struct.unpack("<b", r.take(1))[0]
    if ty == 2:
        return struct.unpack("<H", r.take(2))[0]
    if ty == 3:
        return struct.unpack("<h", r.take(2))[0]
    if ty == 4:
        return r.u32()
    if ty == 5:
        return struct.unpack("<i", r.take(4))[0]
    if ty == 6:
        return struct.unpack("<f", r.take(4))[0]
    if ty == 7:
        return r.take(1) != b"\x00"
    if ty == 8:
        return r.string()
    if ty == 9:
        ety = r.u32()
        count = r.u64()
        return [read_metadata_value(r, ety) for _ in range(count)]
    if ty == 10:
        return r.u64()
    if ty == 11:
        return struct.unpack("<q", r.take(8))[0]
    if ty == 12:
        return struct.unpack("<d", r.take(8))[0]
    raise SystemExit(f"unknown metadata type {ty}")


def parse_gguf(path: Path):
    data = path.read_bytes()
    r = Reader(data)
    if r.take(4) != b"GGUF":
        raise SystemExit("bad magic")
    version = r.u32()
    if version != 3:
        raise SystemExit(f"unsupported GGUF version {version}")
    n_tensors = r.u64()
    n_kv = r.u64()
    meta = {}
    for _ in range(n_kv):
        key = r.string()
        ty = r.u32()
        meta[key] = read_metadata_value(r, ty)
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        n_dims = r.u32()
        dims = [r.u64() for _ in range(n_dims)]
        ty = r.u32()
        off = r.u64()
        tensors.append((name, dims, ty, off))
    alignment = int(meta.get("general.alignment", 32))
    data_start = (r.o + alignment - 1) // alignment * alignment
    return data, meta, tensors, data_start


def tensor_bytes(data, tensors, data_start, name):
    for tname, dims, ty, off in tensors:
        if tname == name:
            elems = 1
            for d in dims:
                elems *= d
            block_elems, block_bytes = TYPE_INFO[ty]
            nbytes = elems * block_bytes // block_elems
            start = data_start + off
            return data[start : start + nbytes], dims, ty
    raise SystemExit(f"tensor {name} not found")


def lcg_input(seed: int, count: int) -> np.ndarray:
    state = max(seed, 1)
    out = np.empty(count, dtype=np.float64)
    for i in range(count):
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        bits = state >> 40
        out[i] = bits / float(1 << 24) - 0.5
    return out.astype(np.float32)


def f16_block(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float16).astype(np.float64)


def block_f16(raw: bytes, block_bytes: int, off: int) -> np.ndarray:
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, block_bytes)
    pair = np.ascontiguousarray(blocks[:, off : off + 2])
    return pair.view(np.float16).astype(np.float64).reshape(-1)


def dequant_q6k(raw: bytes) -> np.ndarray:
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 210).astype(np.int64)
    nb = blocks.shape[0]
    ql = blocks[:, :128].reshape(nb, 2, 64)
    qh = blocks[:, 128:192].reshape(nb, 2, 32)
    sc = blocks[:, 192:208].astype(np.int8).astype(np.float64).reshape(nb, 2, 8)
    dall = block_f16(raw, 210, 208)
    vals = np.zeros((nb, 256), dtype=np.float64)
    l_idx = np.arange(32)
    is_idx = l_idx // 16
    for half in range(2):
        ql0 = ql[:, half, :32]
        ql1 = ql[:, half, 32:]
        qh_h = qh[:, half, :]
        q1 = ((ql0 & 0xF) | ((qh_h & 3) << 4)) - 32
        q2 = ((ql1 & 0xF) | (((qh_h >> 2) & 3) << 4)) - 32
        q3 = ((ql0 >> 4) | (((qh_h >> 4) & 3) << 4)) - 32
        q4 = ((ql1 >> 4) | (((qh_h >> 6) & 3) << 4)) - 32
        base = half * 128
        dcol = dall[:, None]
        vals[:, base + l_idx] = dcol * sc[:, half, is_idx] * q1
        vals[:, base + 32 + l_idx] = dcol * sc[:, half, is_idx + 2] * q2
        vals[:, base + 64 + l_idx] = dcol * sc[:, half, is_idx + 4] * q3
        vals[:, base + 96 + l_idx] = dcol * sc[:, half, is_idx + 6] * q4
    return vals


def scale_min_k4(scales: np.ndarray, j: int):
    # scales: (nb, 12) int64 byte values; mirrors gguf::dequant.
    if j < 4:
        return scales[:, j] & 63, scales[:, j + 4] & 63
    hi = scales[:, j + 4]
    sc = (hi & 0xF) | ((scales[:, j - 4] >> 6) << 4)
    m = (hi >> 4) | ((scales[:, j] >> 6) << 4)
    return sc, m


def dequant_q5k(raw: bytes) -> np.ndarray:
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 176).astype(np.int64)
    nb = blocks.shape[0]
    d = block_f16(raw, 176, 0)
    dmin = block_f16(raw, 176, 2)
    scales = blocks[:, 4:16]
    qh = blocks[:, 16:48]
    ql = blocks[:, 48:176].reshape(nb, 4, 32)
    vals = np.zeros((nb, 256), dtype=np.float64)
    for g in range(4):
        sc1, m1 = scale_min_k4(scales, 2 * g)
        sc2, m2 = scale_min_k4(scales, 2 * g + 1)
        u1 = 1 << (2 * g)
        u2 = 1 << (2 * g + 1)
        byte = ql[:, g, :]
        lo = (byte & 0xF) + np.where(qh & u1 != 0, 16, 0)
        hi = (byte >> 4) + np.where(qh & u2 != 0, 16, 0)
        vals[:, g * 64 : g * 64 + 32] = (
            d[:, None] * sc1[:, None] * lo - dmin[:, None] * m1[:, None]
        )
        vals[:, g * 64 + 32 : g * 64 + 64] = (
            d[:, None] * sc2[:, None] * hi - dmin[:, None] * m2[:, None]
        )
    return vals


def write_fixture(out: Path, kind: str, weights: bytes, x: np.ndarray, ref: np.ndarray):
    out.mkdir(parents=True, exist_ok=True)
    (out / f"weights_{kind}.bin").write_bytes(weights)
    x.astype(np.float32).tofile(out / "input_f32.bin")
    ref.astype(np.float32).tofile(out / "ref_f32.bin")
    print(f"fixture {out}: weights {len(weights)} B, rows {len(ref)}, cols {len(x)}")


def rows_slice(raw: bytes, row_bytes: int, rows: int) -> bytes:
    return raw[: row_bytes * rows]


def main() -> int:
    model = Path(sys.argv[1])
    root = Path(sys.argv[2])
    data, meta, tensors, data_start = parse_gguf(model)
    print(f"parsed: {len(tensors)} tensors, data_start {data_start}")

    # Q6_K: blk.0.ffn_down.weight [19968 in, 6656 out]
    raw, dims, ty = tensor_bytes(data, tensors, data_start, "blk.0.ffn_down.weight")
    assert ty == Q6K_ID and dims == [19968, 6656], (ty, dims)
    cols, rows = dims[0], dims[1]
    row_bytes = (cols // 256) * 210
    x = lcg_input(42, cols)
    xf = x.astype(np.float64)
    for name, nrows in (("q6k_corr", 64), ("q6k_time", rows)):
        w = rows_slice(raw, row_bytes, nrows)
        ref = np.empty(nrows, dtype=np.float64)
        for r0 in range(0, nrows, 64):
            chunk = w[r0 * row_bytes : (r0 + 64) * row_bytes]
            vals = dequant_q6k(chunk).reshape(-1, cols)
            ref[r0 : r0 + vals.shape[0]] = vals @ xf
        write_fixture(root / name, "q6k", w, x, ref)

    # Q5_K: output.weight [6656 in, 202048 out]
    raw, dims, ty = tensor_bytes(data, tensors, data_start, "output.weight")
    assert ty == Q5K_ID and dims == [6656, 202048], (ty, dims)
    cols = dims[0]
    row_bytes = (cols // 256) * 176
    x = lcg_input(42, cols)
    xf = x.astype(np.float64)
    for name, nrows in (("q5k_corr", 64), ("q5k_time", 16384)):
        w = rows_slice(raw, row_bytes, nrows)
        ref = np.empty(nrows, dtype=np.float64)
        for r0 in range(0, nrows, 64):
            chunk = w[r0 * row_bytes : (r0 + 64) * row_bytes]
            vals = dequant_q5k(chunk).reshape(-1, cols)
            ref[r0 : r0 + vals.shape[0]] = vals @ xf
        write_fixture(root / name, "q5k", w, x, ref)

    # F32: first 64 F32 tensors with exactly 6656 elements, file order
    picked = []
    for tname, dims, tty, off in tensors:
        if tty == F32_ID:
            elems = 1
            for dd in dims:
                elems *= dd
            if elems == 6656:
                start = data_start + off
                picked.append((tname, data[start : start + 6656 * 4]))
        if len(picked) == 64:
            break
    assert len(picked) == 64, f"only {len(picked)} F32 6656-tensors found"
    print("f32 rows from:", ", ".join(n for n, _ in picked[:4]), "...")
    w_f32 = b"".join(b for _, b in picked)
    vals32 = np.frombuffer(w_f32, dtype=np.float32).astype(np.float64).reshape(64, 6656)
    x = lcg_input(42, 6656)
    write_fixture(root / "f32_corr", "f32", w_f32, x, vals32 @ x.astype(np.float64))
    halves = np.frombuffer(w_f32, dtype=np.float32).astype(np.float16)
    w_f16 = halves.tobytes()
    valsf16 = halves.astype(np.float64).reshape(64, 6656)
    write_fixture(root / "f16_corr", "f16", w_f16, x, valsf16 @ x.astype(np.float64))

    # Synthetic plain-float timing matrices, 6656 x 6656.
    # Weights from numpy PCG64 seed 1234567 in [-2, 2): this
    # fixture is synthetic by construction (see the header),
    # so it does not use the k2_check LCG; its reference is
    # still the float64 dot of the same values.
    rng = np.random.default_rng(1234567)
    total = 6656 * 6656
    w = (rng.random(total, dtype=np.float64) * 4.0 - 2.0).astype(np.float32)
    ref = np.empty(6656, dtype=np.float64)
    xf = x.astype(np.float64)
    for r0 in range(0, 6656, 512):
        chunk = w[r0 * 6656 : (r0 + 512) * 6656].astype(np.float64).reshape(-1, 6656)
        ref[r0 : r0 + chunk.shape[0]] = chunk @ xf
    write_fixture(root / "f32_time", "f32", w.tobytes(), x, ref)
    halves = w.astype(np.float16)
    ref16 = np.empty(6656, dtype=np.float64)
    for r0 in range(0, 6656, 512):
        chunk = (
            halves[r0 * 6656 : (r0 + 512) * 6656].astype(np.float64).reshape(-1, 6656)
        )
        ref16[r0 : r0 + chunk.shape[0]] = chunk @ xf
    write_fixture(root / "f16_time", "f16", halves.tobytes(), x, ref16)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
