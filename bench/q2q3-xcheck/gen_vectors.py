#!/usr/bin/env python3
"""Deterministic test vectors for the Q2_K / Q3_K dequantizer cross-check.

Writes, into the current directory:
  q2k_synth.bin / q3k_synth.bin   hand-built edge-case blocks
  q2k_dsweep.bin / q3k_dsweep.bin blocks sweeping the f16 scale `d`
                                  over every finite exponent pattern
  quant_in.f32                    f32 rows for ggml's reference
                                  quantizers (quantize_row_*_ref)

All f16 scale fields in synthetic blocks are finite (exponent != 0x1F):
NaN/Inf scales would test the f16 conversion table's NaN payload
handling, not the dequantizers, and the two implementations are allowed
to differ there. Everything else (quants, scales, masks) is exercised
over its full byte range.
"""
import math
import random
import struct
import sys

QK = 256


def put_f16(buf, off, bits):
    struct.pack_into("<H", buf, off, bits)


def q2k_block(scales, qs, d, dmin):
    b = bytearray(84)
    b[0:16] = scales
    b[16:80] = qs
    put_f16(b, 80, d)
    put_f16(b, 82, dmin)
    return bytes(b)


def q3k_block(hmask, qs, scales, d):
    b = bytearray(110)
    b[0:32] = hmask
    b[32:96] = qs
    b[96:108] = scales
    put_f16(b, 108, d)
    return bytes(b)


def f16_finite(rng):
    while True:
        b = rng.randrange(0, 65536)
        if ((b >> 10) & 0x1F) != 0x1F:
            return b


def rand_bytes(rng, n):
    return bytes(rng.randrange(0, 256) for _ in range(n))


def main():
    rng = random.Random(20261105)

    # ---------------- Q2_K synthetic edge blocks ----------------
    q2 = []
    q2.append(q2k_block(bytes(16), bytes(64), 0x0000, 0x0000))          # all zero
    q2.append(q2k_block(b"\xff" * 16, b"\xff" * 64, 0x7BFF, 0x7BFF))    # max finite scales, max quants
    q2.append(q2k_block(b"\xff" * 16, b"\xff" * 64, 0x0000, 0x3C00))    # d = 0, dmin = 1
    q2.append(q2k_block(b"\xff" * 16, b"\xff" * 64, 0x3C00, 0x0000))    # d = 1, dmin = 0
    q2.append(q2k_block(bytes((i * 0x11) & 0xFF for i in range(16)),    # nibble ladder scales
                        bytes((i * 37 + 11) & 0xFF for i in range(64)),
                        0x3800, 0x3400))                                # d = 0.5, dmin = 0.25
    q2.append(q2k_block(bytes((i * 17 + 3) & 0xFF for i in range(16)),
                        bytes(0xAA if i % 2 == 0 else 0x55 for i in range(64)),
                        0xC100, 0x3555))                                # d = -2.5, dmin ~ 0.2
    for _ in range(128):                                                # seeded random blocks
        q2.append(q2k_block(rand_bytes(rng, 16), rand_bytes(rng, 64),
                            f16_finite(rng), f16_finite(rng)))
    with open("q2k_synth.bin", "wb") as f:
        f.write(b"".join(q2))

    # ---------------- Q3_K synthetic edge blocks ----------------
    unit_scales = bytes([0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC,
                         0xDE, 0xF0, 0x21, 0x43, 0x65, 0x87])
    q3 = []
    q3.append(q3k_block(bytes(32), bytes(64), bytes(12), 0x0000))       # all zero
    q3.append(q3k_block(b"\xff" * 32, b"\xff" * 64, b"\xff" * 12, 0x7BFF))  # max everything, max finite d
    q3.append(q3k_block(bytes(32), b"\xff" * 64, b"\xff" * 12, 0x3C00))     # mask all clear (negatives), d = 1
    q3.append(q3k_block(b"\xff" * 32, b"\xff" * 64, b"\xff" * 12, 0x3C00))  # mask all set, d = 1
    q3.append(q3k_block(bytes(0xAA if i % 2 == 0 else 0x55 for i in range(32)),
                        bytes((i * 37 + 11) & 0xFF for i in range(64)),
                        unit_scales, 0x3800))                           # alternating mask, unit-test scales, d = 0.5
    q3.append(q3k_block(rand_bytes(rng, 32),
                        bytes((i * 37 + 11) & 0xFF for i in range(64)),
                        bytes(12), 0x2800))                             # scales all zero (factor -32), d = 2^-5
    for _ in range(128):                                                # seeded random blocks
        q3.append(q3k_block(rand_bytes(rng, 32), rand_bytes(rng, 64),
                            rand_bytes(rng, 12), f16_finite(rng)))
    with open("q3k_synth.bin", "wb") as f:
        f.write(b"".join(q3))

    # ---------------- f16 d sweeps ----------------
    # d (and dmin for Q2_K) walk every finite exponent, four mantissa
    # patterns, both signs: 30 * 4 * 2 = 240 blocks per type.
    sweep_bits = []
    for exp in range(0, 31):
        for mant in (0x000, 0x155, 0x2AA, 0x3FF):
            for sign in (0, 1):
                sweep_bits.append((sign << 15) | (exp << 10) | mant)
    scales2 = bytes((i * 0x11) & 0xFF for i in range(16))
    qs2 = bytes((i * 37 + 11) & 0xFF for i in range(64))
    with open("q2k_dsweep.bin", "wb") as f:
        for bits in sweep_bits:
            f.write(q2k_block(scales2, qs2, bits, 0x3C00))
    hmask3 = bytes((i * 91 + 7) & 0xFF for i in range(32))
    qs3 = bytes((i * 53 + 3) & 0xFF for i in range(64))
    scales3 = bytes((i * 29 + 5) & 0xFF for i in range(12))
    with open("q3k_dsweep.bin", "wb") as f:
        for bits in sweep_bits:
            f.write(q3k_block(hmask3, qs3, scales3, bits))

    # ---------------- quantizer inputs ----------------
    rows = []
    r = random.Random(42)
    rows.append([r.uniform(-1, 1) for _ in range(QK)])                  # uniform
    rows.append([(r.uniform(0, 1) + r.uniform(0, 1) + r.uniform(0, 1) - 1.5) * 2
                 for _ in range(QK)])                                   # approx normal
    rows.append([0.0] * QK)                                             # zeros
    rows.append([0.5] * QK)                                             # constant
    rows.append([math.sin(i / 8.0) * 4 for i in range(QK)])             # sine sweep
    rows.append([r.uniform(-100, 100) for _ in range(QK)])              # wide range
    spike = [r.uniform(-1e-4, 1e-4) for _ in range(QK)]
    spike[100] = 50.0
    rows.append(spike)                                                  # tiny + one spike
    with open("quant_in.f32", "wb") as f:
        for row in rows:
            f.write(struct.pack("<%df" % QK, *row))

    print("q2k_synth blocks:", len(q2), " q3k_synth blocks:", len(q3),
          " dsweep blocks per type:", len(sweep_bits),
          " quant rows:", len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
