# Q2_K / Q3_K dequantizers: port and cross-check against ggml (2026-11-05)

Branch `lane/overnight-q2q3` (from main `bda7ff5`). Verdict up front:
**both dequantizers are bit-exact against ggml's reference
implementation on every vector tried: 223,744 values, 0 diffs.**

## What landed

- `crates/gguf/src/dequant.rs`: `dequant_q2_k` and `dequant_q3_k`
  (plus `unpack_scales_q3_k`), ported operation-for-operation from
  ggml, wired into `dequantize`. Unit tests pin the Q3_K scale
  unpacking against a hand-computed vector, the Q2_K plane/chunk/
  sub-block walk, the Q3_K mask sign rule (set bit: value stands;
  clear bit: subtract 4), the second chunk's mask bits 4..8, and the
  scale extremes. Mutation-checked: flipping the Q3_K mask rule fails
  3 tests, swapping the Q2_K min nibble fails 1 test.
- `crates/gguf/src/ggml.rs`: `GgmlType::Q2K` (id 10) and
  `GgmlType::Q3K` (id 11), `from_id`, block size 256, block bytes
  84 / 110. The audit-era pin asserting id 11 must stay unsupported
  was replaced deliberately (comment in the test) with pins asserting
  support; Q8_1 (id 9) remains the pinned unsupported type.
- Supported-set text updated everywhere it is user-visible:
  `error.rs` (the `UnsupportedTensorType` message), `lib.rs`,
  `file.rs` (`load_f32` docs), `ggml.rs` header, and the `probe`
  example gained `q2_k` / `q3_k` arms.
- `crates/gguf/tests/synthetic.rs`: the unsupported-types fixture
  swapped Q3_K for Q4_0 (id 2, still unsupported) with a comment
  recording why, and a new `q2_k_and_q3_k_tensors_load` integration
  test loads both types end to end through `GgufFile`.
- Cross-check tooling committed under `bench/q2q3-xcheck/`
  (`harness.c`, `gen_vectors.py`, `diff.py`, `extract_real.py`,
  `run-server.sh`, `compare.sh`) so the whole check re-runs.

Scope note: this is loader support. The compute crates
(`cpu-compute`, `wgpu-compute`) keep their own `Dtype` sets and still
refuse Q2_K/Q3_K tensors for math; nothing there changed.

## Sources read (primary: the server's llama.cpp tree)

From `/home/server/llama.cpp/ggml/`:

- `src/ggml-quants.c`: `dequantize_row_q2_K`, `dequantize_row_q3_K`
  (the ports), `quantize_row_q2_K_ref`, `quantize_row_q3_K_ref`
  (round-trip vector generation).
- `src/ggml-common.h`: `block_q2_K` (scales[16], qs[64], d, dmin;
  static_assert 2*2 + 256/16 + 256/4 = **84 bytes**) and
  `block_q3_K` (hmask[32], qs[64], scales[12], d; static_assert
  2 + 256/4 + 256/8 + 12 = **110 bytes**). QK_K = 256.
- `src/ggml.c`: type traits for both types (blck_size QK_K,
  type_size sizeof(block)).

Geometry was confirmed three ways: the static_assert arithmetic,
the harness printing ggml's own `sizeof` (84 / 110, QK_K 256), and
`GgmlType::byte_len` tests (84 per 256, 220 per 512, non-multiples
rejected).

Note the block field order differs from Q4_K..Q6_K: in Q2_K the
scales and quants come FIRST and d/dmin last; in Q3_K the order is
hmask, quants, scales, d. The ports follow the structs, not the
Q4_K habit.

## Cross-check method

Harness (`bench/q2q3-xcheck/harness.c`) calls ggml's own
`dequantize_row_q2_K` / `dequantize_row_q3_K` on caller-supplied
blocks and `quantize_row_*_ref` on f32 rows. The Rust side is the
crate's `probe` example (shortest round-trip text, parses back to
the exact f32). `diff.py` compares bit patterns, not tolerances.

Vector sets (generator seeded, deterministic):

- **synth** (134 blocks/type): all-zero block; all-0xFF payload with
  max finite scales; d = 0 and dmin = 0 variants; nibble-ladder
  scales with patterned quants; alternating mask patterns for Q3_K
  including the unit-test scale bytes; 128 seeded random blocks.
  All f16 scale fields are finite (exponent != 0x1F) by design:
  NaN payload handling belongs to the f16 conversion table, not to
  these dequantizers.
- **dsweep** (248 blocks/type): the scale `d` walks every finite
  f16 exponent (0..30), four mantissa patterns, both signs.
- **quant** (7 blocks/type): blocks produced by ggml's own
  `quantize_row_*_ref` from uniform, approx-normal, zero, constant,
  sine, wide-range (+-100), and tiny-with-one-spike f32 rows, then
  dequantized by both sides.
- **real** (48 blocks/type): real model bytes, see below.

### Where the harness ran, stated plainly

The server protocol queues all server work behind
`/tmp/itak-phase.lock`; six overnight lanes were ahead and the
baseline lane held the lock for hours, so the queued server run
(`~/overnight-q2q3/run-server.sh`, linking the tree's built
`libggml-base.so`) had not fired when this note was filed. It is
still queued and will drop its outputs and `SHA256SUMS` in
`~/overnight-q2q3` as an independent re-confirmation.

The results below come from the identical harness built locally
against the checkout at
`~/workspace/meta-cookbook/build-site/repos/ggerganov__llama.cpp/`,
after verifying that checkout's ggml is byte-identical to the
server tree's for everything this check touches (sha256 of the
extracted texts, first 16 hex digits):

| text | sha256 prefix (both trees) |
|---|---|
| `dequantize_row_q2_K` | 5dbd7936514edc14 |
| `dequantize_row_q3_K` | f1ee74eb91b6cdfa |
| `quantize_row_q2_K_ref` | cfcac20bce126975 |
| `quantize_row_q3_K_ref` | 264e4eda1b92b700 |
| `block_q2_K`/`block_q3_K` structs | 54ed0c8a6ef0023e |
| Q2_K/Q3_K type traits | 81be9313f95e7943 |

The port itself was written from the server tree's source, read
directly over SSH before any local file was touched.

## Results

Per type; every value compared over all 256 positions of every
block. "Exact" means bit-identical f32.

| set | blocks | values | exact | max abs diff | sign-of-zero only | NaN mismatch |
|---|---|---|---|---|---|---|
| Q2_K synth | 134 | 34,304 | 34,304 (100%) | 0 | 0 | 0 |
| Q2_K dsweep | 248 | 63,488 | 63,488 (100%) | 0 | 0 | 0 |
| Q2_K quant round-trip | 7 | 1,792 | 1,792 (100%) | 0 | 0 | 0 |
| Q2_K real | 48 | 12,288 | 12,288 (100%) | 0 | 0 | 0 |
| Q3_K synth | 134 | 34,304 | 34,304 (100%) | 0 | 0 | 0 |
| Q3_K dsweep | 248 | 63,488 | 63,488 (100%) | 0 | 0 | 0 |
| Q3_K quant round-trip | 7 | 1,792 | 1,792 (100%) | 0 | 0 | 0 |
| Q3_K real | 48 | 12,288 | 12,288 (100%) | 0 | 0 | 0 |

Total: 874 blocks, **223,744 values, 100% bit-exact** for both
types. The bar was exact-or-f32-rounding; the ports are exact.

## Real bytes: yes, for both types

`/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-UD-Q3_K_XL.gguf`
(Unsloth dynamic quant of Glimmer itself) mixes types: of its 731
tensors, 206 are Q3_K and 42 are Q2_K (313 F32, 112 Q4_K, 38 Q5_K,
20 Q6_K). Extracted 16 blocks each from:

- Q3_K: `blk.0.attn_gate.weight`, `blk.0.attn_q.weight`,
  `blk.0.ffn_down.weight`
- Q2_K: `blk.10.attn_q.weight`, `blk.12.attn_gate.weight`,
  `blk.12.attn_q.weight`

Tensor offsets came from a local parse of the file's directory
(prefix fetched read-only over SSH, blocks cut with `dd`). The
parse was proven globally, not spot-checked: with each type's
block geometry, all 731 tensors tile the data section back to back
with zero mismatches (first data offset 13,113,888; last tensor
ends at 13,360,983,072). Extracted blocks' `d` fields are all
small finite values of weight-scale magnitude, as real blocks
should be. No other file under `/mnt/models` was needed; the
brief's fallback (quantize-ref blocks) also ran, as the `quant`
set above.

## Gates

- `cargo test -p gguf`: pass (19 lib + 20 integration + 1 doc).
- `cargo clippy -p gguf --all-targets -- -D warnings`: clean.
- `cargo fmt --check -p gguf`: clean.
- `cargo check --workspace --all-targets`: pass (the new enum
  variants break no exhaustive match; other crates only construct
  `GgmlType`).

## Verdict

Q2_K: **exact**. Q3_K: **exact**. Support is real, not faked:
the loader now dequantizes both types with ggml-identical output
on synthetic edge cases, a full finite-f16 scale sweep,
ggml-quantized blocks, and real Glimmer UD-Q3_K_XL bytes.
