#!/usr/bin/env python3
"""Scan every GGUF under /mnt/models for Q2_K (id 10) and Q3_K (id 11)
tensors and extract up to 48 real blocks per type into
q2k_real.bin / q3k_real.bin, with a manifest of where they came from.

Uses the llama.cpp tree's own gguf-py reader. Only the extracted blocks
are copied out of the (multi-GB) model files, via memmap slices.
"""
import glob
import sys

sys.path.insert(0, "/home/server/llama.cpp/gguf-py")
import gguf  # noqa: E402

BLOCK = {10: 84, 11: 110}
NAME = {10: "q2k", 11: "q3k"}
MAX_BLOCKS_PER_TYPE = 48


def tensor_bytes(reader, tensor, want_bytes):
    """Raw bytes from the start of a tensor's data, block aligned."""
    offset = getattr(tensor, "data_offset", None)
    if offset is not None:
        return bytes(reader.data[offset:offset + want_bytes])
    arr = tensor.data
    if arr.dtype.itemsize == want_bytes:
        return arr.reshape(-1)[:1].tobytes()
    return arr.tobytes()[:want_bytes]


def main():
    found = {10: 0, 11: 0}
    outs = {10: bytearray(), 11: bytearray()}
    manifest = []
    for path in sorted(glob.glob("/mnt/models/**/*.gguf", recursive=True)):
        try:
            reader = gguf.GGUFReader(path)
        except Exception as e:  # noqa: BLE001 - report and move on
            manifest.append(f"SKIP {path}: {type(e).__name__}: {e}")
            continue
        for t in reader.tensors:
            tid = int(t.tensor_type)
            if tid in BLOCK and found[tid] < MAX_BLOCKS_PER_TYPE:
                want = (MAX_BLOCKS_PER_TYPE - found[tid]) * BLOCK[tid]
                raw = tensor_bytes(reader, t, want)
                take = len(raw) // BLOCK[tid]
                if take > 0:
                    outs[tid] += raw[:take * BLOCK[tid]]
                    found[tid] += take
                    manifest.append(
                        f"{NAME[tid]}: {path} tensor={t.name} blocks={take}")
    for tid in (10, 11):
        if outs[tid]:
            with open(f"{NAME[tid]}_real.bin", "wb") as f:
                f.write(outs[tid])
        manifest.append(f"TOTAL {NAME[tid]} blocks={found[tid]}")
    with open("real_manifest.txt", "w") as f:
        f.write("\n".join(manifest) + "\n")
    print("\n".join(manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
