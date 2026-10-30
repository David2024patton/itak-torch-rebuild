//! Q4_K kernels: dequantization and matmul (matrix times vector).
//!
//! The block layout is the GGUF Q4_K super-block as the `gguf`
//! crate's type definitions pin it ([`gguf::GgmlType::Q4K`]: 256
//! elements, 144 bytes per block):
//!
//! ```text
//! offset  size  field
//! 0       2     d     (f16) super-block scale
//! 2       2     dmin  (f16) super-block minimum scale
//! 4       12    scales: eight 6-bit sub-block scales and eight
//!               6-bit sub-block mins, packed ggml K-quant style
//! 16      128   qs: 4-bit codes, four groups of 32 bytes; in each
//!               group the low nibbles are the first 32-element
//!               sub-block and the high nibbles the second
//! ```
//!
//! Element `i` of sub-block `j` decodes as
//! `d * scale[j] * code - dmin * min[j]`, with `(scale[j], min[j])`
//! unpacked by the same rule ggml's `get_scale_min_k4` uses. The
//! unit tests pin this module against a naive scalar reference
//! written independently from this layout, not against the code
//! here, and cross-check the loader's dequantizer in the `gguf`
//! crate for exact agreement.
//!
//! Allocation behaviour, stated plainly: [`dequantize_q4k`] writes
//! into a caller-provided buffer and allocates nothing.
//! [`matvec_q4k`] allocates exactly two buffers per call (the output
//! vector and one scratch row of `cols` f32, reused for every row);
//! the per-row inner loop allocates nothing.

use gguf::dequant::f16_to_f32;

use crate::config::KernelConfig;
use crate::diag::dbg_log;
use crate::error::{KernelError, Result};

/// Elements per Q4_K super-block: [`gguf::GgmlType::Q4K`]'s block size.
pub const BLOCK_ELEMENTS: usize = 256;

/// Bytes per Q4_K super-block: [`gguf::GgmlType::Q4K`]'s block bytes.
pub const BLOCK_BYTES: usize = 144;

/// Elements covered by a payload of `data.len()` bytes, or the loud
/// refusal every entry point shares: a length that is not a whole
/// number of blocks is malformed, never partially decoded.
fn elements_for(op: &'static str, data: &[u8]) -> Result<usize> {
    if data.len() % BLOCK_BYTES != 0 {
        return Err(KernelError::MalformedBlockLength {
            op,
            actual: data.len(),
            block_bytes: BLOCK_BYTES,
        });
    }
    Ok(data.len() / BLOCK_BYTES * BLOCK_ELEMENTS)
}

/// The config ceiling check, run before any arithmetic.
fn check_limit(op: &'static str, elements: usize, config: &KernelConfig) -> Result<()> {
    if elements > config.max_elements {
        return Err(KernelError::LimitExceeded {
            op,
            limit: config.max_elements,
            actual: elements,
        });
    }
    Ok(())
}

/// Log a refusal through the diagnostic seam, then return it.
fn refuse<T>(err: KernelError) -> Result<T> {
    dbg_log!("refusing: {err}");
    Err(err)
}

/// Unpack sub-block `j`'s (scale, min) pair from the 12-byte scales
/// array. Mirrors ggml's `get_scale_min_k4`: the first four bytes hold
/// the low 6 bits of scales 0..4 and the next four the low 6 bits of
/// mins 0..4; the last four bytes carry the high bits of all eight
/// pairs folded together.
fn scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

fn read_f16(block: &[u8], byte_offset: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([
        block[byte_offset],
        block[byte_offset + 1],
    ]))
}

/// Dequantize one already-validated 144-byte block into an
/// already-validated 256-element output slice.
fn dequant_block(block: &[u8], out: &mut [f32]) {
    let d = read_f16(block, 0);
    let dmin = read_f16(block, 2);
    let scales = &block[4..16];
    let qs = &block[16..144];
    // Four groups of 64 elements; each group holds two 32-element
    // sub-blocks, low nibbles first, then high nibbles.
    for group in 0..4 {
        let (sc1, m1) = scale_min_k4(2 * group, scales);
        let (sc2, m2) = scale_min_k4(2 * group + 1, scales);
        let d1 = d * f32::from(sc1);
        let dm1 = dmin * f32::from(m1);
        let d2 = d * f32::from(sc2);
        let dm2 = dmin * f32::from(m2);
        let out_group = &mut out[group * 64..(group + 1) * 64];
        for l in 0..32 {
            let byte = qs[group * 32 + l];
            out_group[l] = d1 * f32::from(byte & 0xF) - dm1;
            out_group[32 + l] = d2 * f32::from(byte >> 4) - dm2;
        }
    }
}

/// Dequantize a whole slice of validated blocks into `out`, whose
/// length the caller has already matched to the payload.
fn dequant_blocks(data: &[u8], out: &mut [f32]) {
    for (block, out_block) in data
        .chunks_exact(BLOCK_BYTES)
        .zip(out.chunks_exact_mut(BLOCK_ELEMENTS))
    {
        dequant_block(block, out_block);
    }
}

/// Dequantize Q4_K payload bytes into a caller-provided f32 buffer.
///
/// `data` must be a whole number of 144-byte blocks; `out` must hold
/// exactly 256 elements per block. Returns the number of elements
/// written. Allocates nothing.
///
/// # Errors
///
/// - [`KernelError::MalformedBlockLength`] if `data.len()` is not a
///   multiple of [`BLOCK_BYTES`].
/// - [`KernelError::OutputLengthMismatch`] if `out` is not exactly
///   the element count implied by `data`.
/// - [`KernelError::LimitExceeded`] if that element count exceeds
///   `config.max_elements`.
pub fn dequantize_q4k(data: &[u8], out: &mut [f32], config: &KernelConfig) -> Result<usize> {
    let elements = match elements_for("dequantize_q4k", data) {
        Ok(n) => n,
        Err(e) => return refuse(e),
    };
    if let Err(e) = check_limit("dequantize_q4k", elements, config) {
        return refuse(e);
    }
    if out.len() != elements {
        return refuse(KernelError::OutputLengthMismatch {
            op: "dequantize_q4k",
            expected: elements,
            actual: out.len(),
        });
    }
    dbg_log!(
        "dequantize_q4k blocks={} elements={elements} max_elements={}",
        data.len() / BLOCK_BYTES,
        config.max_elements
    );
    dequant_blocks(data, out);
    Ok(elements)
}

/// Dequantize Q4_K payload bytes into a fresh f32 vector. The
/// allocating convenience over [`dequantize_q4k`]; same errors.
pub fn dequantize_q4k_vec(data: &[u8], config: &KernelConfig) -> Result<Vec<f32>> {
    let elements = match elements_for("dequantize_q4k_vec", data) {
        Ok(n) => n,
        Err(e) => return refuse(e),
    };
    if let Err(e) = check_limit("dequantize_q4k_vec", elements, config) {
        return refuse(e);
    }
    let mut out = vec![0.0f32; elements];
    dequant_blocks(data, &mut out);
    Ok(out)
}

/// Q4_K matmul, matrix times vector: `out[i] = sum_j W[i, j] * v[j]`
/// where `W` is a `rows x cols` matrix stored as Q4_K payload bytes
/// (row-major, each row a whole number of super-blocks) and `v` has
/// length `cols`. Each row is dequantized into a scratch buffer and
/// dotted against `v` with f64 accumulation, the house convention
/// for reference-grade dot products.
///
/// Decode is matvec-bound (one token against the whole weight
/// matrix), so this GEMV is the Q4_K matmul the token loop needs;
/// a dense GEMM over Q4_K operands is later work.
///
/// With `cols == 0` the product is the zero vector of length `rows`
/// (an empty sum), provided the payload is empty and `v` is empty.
///
/// # Errors
///
/// - [`KernelError::MalformedBlockLength`] if `weights.len()` is
///   not a multiple of [`BLOCK_BYTES`].
/// - [`KernelError::InvalidConfig`] if `cols` is not a whole number
///   of super-blocks, or if `rows * cols` overflows.
/// - [`KernelError::ShapeMismatch`] if `weights.len()` is not the
///   byte count implied by `rows x cols`.
/// - [`KernelError::LengthMismatch`] if `vector.len() != cols`.
/// - [`KernelError::LimitExceeded`] if `rows * cols` exceeds
///   `config.max_elements`.
pub fn matvec_q4k(
    weights: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    config: &KernelConfig,
) -> Result<Vec<f32>> {
    if weights.len() % BLOCK_BYTES != 0 {
        return refuse(KernelError::MalformedBlockLength {
            op: "matvec_q4k",
            actual: weights.len(),
            block_bytes: BLOCK_BYTES,
        });
    }
    if cols % BLOCK_ELEMENTS != 0 {
        return refuse(KernelError::InvalidConfig {
            op: "matvec_q4k",
            param: "cols",
            reason: "not a whole number of Q4_K super-blocks",
        });
    }
    let elements = match rows.checked_mul(cols) {
        Some(n) => n,
        None => {
            return refuse(KernelError::InvalidConfig {
                op: "matvec_q4k",
                param: "shape",
                reason: "rows * cols overflows usize",
            })
        }
    };
    if let Err(e) = check_limit("matvec_q4k", elements, config) {
        return refuse(e);
    }
    let row_bytes = cols / BLOCK_ELEMENTS * BLOCK_BYTES;
    // Cannot overflow: elements fit, and bytes per element < 1.
    let expected_bytes = rows * row_bytes;
    if weights.len() != expected_bytes {
        return refuse(KernelError::ShapeMismatch {
            op: "matvec_q4k",
            expected: expected_bytes,
            actual: weights.len(),
        });
    }
    if vector.len() != cols {
        return refuse(KernelError::LengthMismatch {
            op: "matvec_q4k",
            expected: cols,
            actual: vector.len(),
        });
    }
    dbg_log!(
        "matvec_q4k rows={rows} cols={cols} max_elements={}",
        config.max_elements
    );
    let mut scratch = vec![0.0f32; cols];
    let mut out = vec![0.0f32; rows];
    if cols > 0 {
        for (i, row) in weights.chunks_exact(row_bytes).enumerate() {
            dequant_blocks(row, &mut scratch);
            out[i] = scratch
                .iter()
                .zip(vector.iter())
                .map(|(w, v)| f64::from(*w) * f64::from(*v))
                .sum::<f64>() as f32;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gguf::GgmlType;

    // ---- Independent naive reference ------------------------------
    //
    // Written directly from the block layout in this module's docs,
    // scalar, in f64, with its own f16 conversion. It shares no code
    // with the kernel, so agreement is evidence, not tautology.

    /// Half-precision conversion by the definition: sign, exponent,
    /// mantissa, evaluated in f64.
    fn naive_f16(bits: u16) -> f64 {
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exp = i32::from((bits >> 10) & 0x1f);
        let frac = f64::from(bits & 0x3ff);
        match exp {
            0 => sign * frac * 2f64.powi(-24),
            31 => {
                if frac == 0.0 {
                    sign * f64::INFINITY
                } else {
                    f64::NAN
                }
            }
            e => sign * (1.0 + frac / 1024.0) * 2f64.powi(e - 15),
        }
    }

    fn naive_scale_min(j: usize, scales: &[u8]) -> (f64, f64) {
        let (sc, m) = if j < 4 {
            (scales[j] & 63, scales[j + 4] & 63)
        } else {
            (
                (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
                (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
            )
        };
        (f64::from(sc), f64::from(m))
    }

    fn naive_dequant_block(block: &[u8]) -> [f64; BLOCK_ELEMENTS] {
        assert_eq!(block.len(), BLOCK_BYTES);
        let d = naive_f16(u16::from_le_bytes([block[0], block[1]]));
        let dmin = naive_f16(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let mut out = [0.0f64; BLOCK_ELEMENTS];
        for group in 0..4 {
            let (sc1, m1) = naive_scale_min(2 * group, scales);
            let (sc2, m2) = naive_scale_min(2 * group + 1, scales);
            for l in 0..32 {
                let byte = block[16 + group * 32 + l];
                out[group * 64 + l] = d * sc1 * f64::from(byte & 0xF) - dmin * m1;
                out[group * 64 + 32 + l] = d * sc2 * f64::from(byte >> 4) - dmin * m2;
            }
        }
        out
    }

    // ---- Fixtures -------------------------------------------------

    /// A hand-crafted block whose outputs are computable by hand:
    /// d = 0.5, dmin = 0.25, sub-block (scale, min) pairs (1, 5),
    /// (2, 6), (3, 7), (4, 8), then (0, 0) for the last four, and
    /// every code byte 0x21 (low nibble 1, high nibble 2).
    fn crafted_block() -> [u8; BLOCK_BYTES] {
        let mut b = [0u8; BLOCK_BYTES];
        b[0..2].copy_from_slice(&0x3800u16.to_le_bytes()); // d = 0.5
        b[2..4].copy_from_slice(&0x3400u16.to_le_bytes()); // dmin = 0.25
        b[4..16].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0]);
        for byte in &mut b[16..BLOCK_BYTES] {
            *byte = 0x21;
        }
        b
    }

    /// The expected dequantization of [`crafted_block`], computed by
    /// hand from the layout: per 32-element sub-block, in order,
    /// 0.5*1*1 - 0.25*5 = -0.75; 0.5*2*2 - 0.25*6 = 0.5;
    /// 0.5*3*1 - 0.25*7 = -0.25; 0.5*4*2 - 0.25*8 = 2.0; then the
    /// four (0, 0) sub-blocks give 0. All exact in f32.
    fn crafted_expected() -> Vec<f64> {
        let mut v = Vec::with_capacity(BLOCK_ELEMENTS);
        for (value, count) in [(-0.75, 32), (0.5, 32), (-0.25, 32), (2.0, 32), (0.0, 128)] {
            v.extend(std::iter::repeat_n(value, count));
        }
        v
    }

    /// A deterministic non-trivial block: every byte from a fixed
    /// arithmetic pattern, with the header overwritten by d = 1.5
    /// and dmin = 0.75 so magnitudes stay checkable.
    fn pattern_block() -> [u8; BLOCK_BYTES] {
        let mut b = [0u8; BLOCK_BYTES];
        for (i, byte) in b.iter_mut().enumerate() {
            *byte = ((i * 37 + 11) % 256) as u8;
        }
        b[0..2].copy_from_slice(&0x3e00u16.to_le_bytes()); // d = 1.5
        b[2..4].copy_from_slice(&0x3a00u16.to_le_bytes()); // dmin = 0.75
        b
    }

    fn assert_vec_close_f64(actual: &[f32], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (f64::from(*a) - e).abs();
            assert!(
                diff <= tol,
                "index {i}: expected {e}, got {a} (diff {diff})"
            );
        }
    }

    // ---- Dequantize ----------------------------------------------

    #[test]
    fn block_geometry_matches_gguf_type_definitions() {
        assert_eq!(BLOCK_ELEMENTS as u64, GgmlType::Q4K.block_size());
        assert_eq!(BLOCK_BYTES as u64, GgmlType::Q4K.block_bytes());
        assert_eq!(GgmlType::Q4K.byte_len(512), Some(288));
    }

    #[test]
    fn single_block_dequant_matches_hand_computed_values() {
        let block = crafted_block();
        let out = dequantize_q4k_vec(&block, &KernelConfig::default()).unwrap();
        assert_vec_close_f64(&out, &crafted_expected(), 0.0);
        // The naive reference agrees with the hand computation too,
        // which pins the reference itself, not just the kernel.
        assert_vec_close_f64(
            &naive_dequant_block(&block)
                .iter()
                .map(|v| *v as f32)
                .collect::<Vec<_>>(),
            &crafted_expected(),
            0.0,
        );
    }

    #[test]
    fn pattern_block_matches_naive_reference() {
        let block = pattern_block();
        let out = dequantize_q4k_vec(&block, &KernelConfig::default()).unwrap();
        assert_vec_close_f64(&out, &naive_dequant_block(&block), 2e-3);
    }

    #[test]
    fn kernel_agrees_exactly_with_loader_dequantizer() {
        // The gguf crate's row dequantizer is the loader's reference;
        // the kernel must produce identical numbers from the same
        // bytes so the two paths can never drift apart.
        let mut payload = Vec::new();
        payload.extend_from_slice(&crafted_block());
        payload.extend_from_slice(&pattern_block());
        let kernel = dequantize_q4k_vec(&payload, &KernelConfig::default()).unwrap();
        let loader = gguf::dequant::dequantize(GgmlType::Q4K, &payload, 2 * BLOCK_ELEMENTS);
        assert_eq!(kernel, loader);
    }

    #[test]
    fn multi_block_row_dequantizes_block_by_block() {
        // Second block: same scales and codes, d and dmin negated,
        // so its outputs are exactly the negation of the first's.
        let mut negated = crafted_block();
        negated[0..2].copy_from_slice(&0xb800u16.to_le_bytes()); // d = -0.5
        negated[2..4].copy_from_slice(&0xb400u16.to_le_bytes()); // dmin = -0.25
        let mut payload = Vec::new();
        payload.extend_from_slice(&crafted_block());
        payload.extend_from_slice(&negated);

        let mut out = vec![0.0f32; 2 * BLOCK_ELEMENTS];
        let written = dequantize_q4k(&payload, &mut out, &KernelConfig::default()).unwrap();
        assert_eq!(written, 2 * BLOCK_ELEMENTS);
        let expected = crafted_expected();
        assert_vec_close_f64(&out[..BLOCK_ELEMENTS], &expected, 0.0);
        let negated_expected: Vec<f64> = expected.iter().map(|v| -v).collect();
        assert_vec_close_f64(&out[BLOCK_ELEMENTS..], &negated_expected, 0.0);
    }

    #[test]
    fn empty_payload_is_an_empty_success() {
        let mut out: Vec<f32> = Vec::new();
        let written = dequantize_q4k(&[], &mut out, &KernelConfig::default()).unwrap();
        assert_eq!(written, 0);
    }

    // ---- Matvec ---------------------------------------------------

    #[test]
    fn matvec_against_ones_sums_the_dequantized_row() {
        // Sum of the crafted row: 32 * (-0.75 + 0.5 - 0.25 + 2.0) = 48.
        let out = matvec_q4k(
            &crafted_block(),
            1,
            BLOCK_ELEMENTS,
            &vec![1.0f32; BLOCK_ELEMENTS],
            &KernelConfig::default(),
        )
        .unwrap();
        assert_eq!(out, vec![48.0]);
    }

    #[test]
    fn matvec_two_rows() {
        let mut negated = crafted_block();
        negated[0..2].copy_from_slice(&0xb800u16.to_le_bytes());
        negated[2..4].copy_from_slice(&0xb400u16.to_le_bytes());
        let mut weights = Vec::new();
        weights.extend_from_slice(&crafted_block());
        weights.extend_from_slice(&negated);
        let out = matvec_q4k(
            &weights,
            2,
            BLOCK_ELEMENTS,
            &vec![1.0f32; BLOCK_ELEMENTS],
            &KernelConfig::default(),
        )
        .unwrap();
        assert_eq!(out, vec![48.0, -48.0]);
    }

    #[test]
    fn matvec_matches_naive_reference_dot() {
        // Pattern row against a non-uniform vector; the expected
        // value is the naive reference row dotted in f64.
        let block = pattern_block();
        let vector: Vec<f32> = (0..BLOCK_ELEMENTS)
            .map(|i| ((i % 7) as f32) - 3.0)
            .collect();
        let reference = naive_dequant_block(&block);
        let expected: f64 = reference
            .iter()
            .zip(vector.iter())
            .map(|(w, v)| w * f64::from(*v))
            .sum();
        let out = matvec_q4k(&block, 1, BLOCK_ELEMENTS, &vector, &KernelConfig::default()).unwrap();
        assert_eq!(out.len(), 1);
        let diff = (f64::from(out[0]) - expected).abs();
        assert!(
            diff <= 2e-3,
            "expected {expected}, got {} (diff {diff})",
            out[0]
        );
    }

    #[test]
    fn matvec_zero_cols_is_zero_vector() {
        let out = matvec_q4k(&[], 3, 0, &[], &KernelConfig::default()).unwrap();
        assert_eq!(out, vec![0.0, 0.0, 0.0]);
    }

    // ---- Loud errors ----------------------------------------------

    #[test]
    fn malformed_block_length_is_a_loud_error() {
        for len in [
            1usize,
            BLOCK_BYTES - 1,
            BLOCK_BYTES + 1,
            2 * BLOCK_BYTES - 1,
        ] {
            let data = vec![0u8; len];
            let mut out = vec![0.0f32; 2 * BLOCK_ELEMENTS];
            assert_eq!(
                dequantize_q4k(&data, &mut out, &KernelConfig::default()),
                Err(KernelError::MalformedBlockLength {
                    op: "dequantize_q4k",
                    actual: len,
                    block_bytes: BLOCK_BYTES,
                }),
                "len {len} must refuse"
            );
            assert_eq!(
                matvec_q4k(
                    &data,
                    1,
                    BLOCK_ELEMENTS,
                    &vec![1.0; BLOCK_ELEMENTS],
                    &KernelConfig::default()
                ),
                Err(KernelError::MalformedBlockLength {
                    op: "matvec_q4k",
                    actual: len,
                    block_bytes: BLOCK_BYTES,
                }),
                "len {len} must refuse in matvec too"
            );
        }
    }

    #[test]
    fn output_length_mismatch_is_an_error() {
        let block = crafted_block();
        let mut short = vec![0.0f32; BLOCK_ELEMENTS - 1];
        assert_eq!(
            dequantize_q4k(&block, &mut short, &KernelConfig::default()),
            Err(KernelError::OutputLengthMismatch {
                op: "dequantize_q4k",
                expected: BLOCK_ELEMENTS,
                actual: BLOCK_ELEMENTS - 1,
            })
        );
    }

    #[test]
    fn matvec_shape_errors_are_loud() {
        let ones = vec![1.0f32; BLOCK_ELEMENTS];
        // cols not a whole number of super-blocks.
        assert!(matches!(
            matvec_q4k(
                &crafted_block(),
                1,
                128,
                &ones[..128],
                &KernelConfig::default()
            ),
            Err(KernelError::InvalidConfig {
                op: "matvec_q4k",
                param: "cols",
                ..
            })
        ));
        // Payload holds two blocks but the shape declares one row.
        let mut two_blocks = Vec::new();
        two_blocks.extend_from_slice(&crafted_block());
        two_blocks.extend_from_slice(&crafted_block());
        assert_eq!(
            matvec_q4k(
                &two_blocks,
                1,
                BLOCK_ELEMENTS,
                &ones,
                &KernelConfig::default()
            ),
            Err(KernelError::ShapeMismatch {
                op: "matvec_q4k",
                expected: BLOCK_BYTES,
                actual: 2 * BLOCK_BYTES,
            })
        );
        // Vector shorter than cols.
        assert_eq!(
            matvec_q4k(
                &crafted_block(),
                1,
                BLOCK_ELEMENTS,
                &ones[..255],
                &KernelConfig::default()
            ),
            Err(KernelError::LengthMismatch {
                op: "matvec_q4k",
                expected: BLOCK_ELEMENTS,
                actual: 255,
            })
        );
    }

    #[test]
    fn configured_limit_refuses_oversized_calls() {
        let tight = KernelConfig::new(BLOCK_ELEMENTS - 1);
        let mut out = vec![0.0f32; BLOCK_ELEMENTS];
        assert_eq!(
            dequantize_q4k(&crafted_block(), &mut out, &tight),
            Err(KernelError::LimitExceeded {
                op: "dequantize_q4k",
                limit: BLOCK_ELEMENTS - 1,
                actual: BLOCK_ELEMENTS,
            })
        );
        assert_eq!(
            matvec_q4k(
                &crafted_block(),
                1,
                BLOCK_ELEMENTS,
                &vec![1.0; BLOCK_ELEMENTS],
                &tight
            ),
            Err(KernelError::LimitExceeded {
                op: "matvec_q4k",
                limit: BLOCK_ELEMENTS - 1,
                actual: BLOCK_ELEMENTS,
            })
        );
    }
}
