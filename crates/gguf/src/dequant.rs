//! Dequantization to f32, following mainline ggml's reference row
//! dequantizers exactly (same block layouts, same scale unpacking), so
//! values produced here match what llama.cpp computes from the same bytes.
//!
//! Every function in this module assumes the caller has already validated
//! that `data.len()` is the exact on-disk length for `elements` elements of
//! the type (see `GgmlType::byte_len`); `dequantize` debug-asserts it.

use crate::ggml::GgmlType;

/// Convert an IEEE 754 half-precision bit pattern to f32.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15);
    let exp = u32::from((bits >> 10) & 0x1f);
    let frac = u32::from(bits & 0x3ff);
    if exp == 0x1f {
        let out = if frac == 0 {
            0x7f80_0000
        } else {
            0x7fc0_0000 | (frac << 13)
        };
        return f32::from_bits((sign << 31) | out);
    }
    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal half: value is frac * 2^-24, exact in f32.
        let v = frac as f32 * f32::from_bits(0x3380_0000);
        return if sign == 1 { -v } else { v };
    }
    let out = ((exp + 127 - 15) << 23) | (frac << 13);
    f32::from_bits((sign << 31) | out)
}

/// Convert a bfloat16 bit pattern to f32 (a pure high-half copy).
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn read_f16(data: &[u8], byte_offset: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([
        data[byte_offset],
        data[byte_offset + 1],
    ]))
}

/// Unpack one sub-block's (scale, min) pair from a K-quant scales array.
/// Mirrors ggml's `get_scale_min_k4` exactly: the 12-byte array packs eight
/// 6-bit scales and eight 6-bit mins with the high bits folded into the
/// last four bytes.
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

fn dequant_q8_0(data: &[u8], out: &mut [f32]) {
    for (b, block) in data.chunks_exact(34).enumerate() {
        let d = read_f16(block, 0);
        let o = &mut out[b * 32..(b + 1) * 32];
        for (i, slot) in o.iter_mut().enumerate() {
            *slot = d * f32::from(block[2 + i] as i8);
        }
    }
}

/// Port of ggml's `dequantize_row_q2_K` (ggml/src/ggml-quants.c) over the
/// `block_q2_K` layout (ggml/src/ggml-common.h): 16 packed scale/min
/// bytes, 64 two-bit quant bytes, then f16 `d` and f16 `dmin` (note the
/// scales and quants come FIRST in this block, unlike Q4_K..Q6_K).
///
/// Each scale byte serves one 16-element sub-block: low nibble is the
/// scale, high nibble is the min. The block walks two chunks of 128
/// elements; per chunk, four 2-bit planes (shifts 0, 2, 4, 6), and each
/// plane covers two sub-blocks read from `qs[l]` and `qs[l + 16]`.
fn dequant_q2_k(data: &[u8], out: &mut [f32]) {
    for (b, block) in data.chunks_exact(84).enumerate() {
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = read_f16(block, 80);
        let dmin = read_f16(block, 82);
        let o = &mut out[b * 256..(b + 1) * 256];
        let mut is = 0usize;
        for chunk in 0..2 {
            let q = &qs[chunk * 32..chunk * 32 + 32];
            for (plane, shift) in [0u32, 2, 4, 6].into_iter().enumerate() {
                for sub in 0..2 {
                    let sc = scales[is];
                    is += 1;
                    let dl = d * f32::from(sc & 0xF);
                    let ml = dmin * f32::from(sc >> 4);
                    for l in 0..16 {
                        let qv = (q[sub * 16 + l] >> shift) & 3;
                        o[chunk * 128 + plane * 32 + sub * 16 + l] = dl * f32::from(qv) - ml;
                    }
                }
            }
        }
    }
}

/// Unpack the 16 six-bit scales of a Q3_K block from its 12 packed bytes.
/// Mirrors the uint32 shuffle at the top of ggml's `dequantize_row_q3_K`
/// (ggml/src/ggml-quants.c) operation for operation: bytes 0..8 supply
/// the low four bits (low nibbles for scales 0..8, high nibbles for
/// scales 8..16) and bytes 8..12 supply the top two bits of every scale.
fn unpack_scales_q3_k(packed: &[u8]) -> [i8; 16] {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let mut aux = [
        u32::from_le_bytes([packed[0], packed[1], packed[2], packed[3]]),
        u32::from_le_bytes([packed[4], packed[5], packed[6], packed[7]]),
        u32::from_le_bytes([packed[8], packed[9], packed[10], packed[11]]),
        0,
    ];
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    let mut scales = [0i8; 16];
    for (i, word) in aux.iter().enumerate() {
        for (j, byte) in word.to_le_bytes().into_iter().enumerate() {
            scales[i * 4 + j] = byte as i8;
        }
    }
    scales
}

/// Port of ggml's `dequantize_row_q3_K` (ggml/src/ggml-quants.c) over the
/// `block_q3_K` layout (ggml/src/ggml-common.h): 32 high-bit mask bytes,
/// 64 two-bit quant bytes, 12 packed scale bytes, then f16 `d`.
///
/// The third bit of each value is NOT stored as a plain bit: a SET mask
/// bit means the 2-bit quant stands as-is (values 0..3), a CLEAR mask bit
/// subtracts 4 (values -4..-1). The mask bit index advances one bit per
/// 32-element plane pair and is never reset, so the second 128-element
/// chunk reads bits 4..8 of the same 32 mask bytes while the quant
/// pointer advances by 32 bytes per chunk.
fn dequant_q3_k(data: &[u8], out: &mut [f32]) {
    for (b, block) in data.chunks_exact(110).enumerate() {
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let scales = unpack_scales_q3_k(&block[96..108]);
        let d = read_f16(block, 108);
        let o = &mut out[b * 256..(b + 1) * 256];
        let mut is = 0usize;
        let mut m: u8 = 1;
        for chunk in 0..2 {
            let q = &qs[chunk * 32..chunk * 32 + 32];
            for (plane, shift) in [0u32, 2, 4, 6].into_iter().enumerate() {
                for sub in 0..2 {
                    let dl = d * f32::from(scales[is] - 32);
                    is += 1;
                    for l in 0..16 {
                        let qv = i32::from((q[sub * 16 + l] >> shift) & 3);
                        let sub4 = if hmask[sub * 16 + l] & m != 0 { 0 } else { 4 };
                        let v = (qv - sub4) as f32;
                        o[chunk * 128 + plane * 32 + sub * 16 + l] = dl * v;
                    }
                }
                m <<= 1;
            }
        }
    }
}

fn dequant_q4_k(data: &[u8], out: &mut [f32]) {
    for (b, block) in data.chunks_exact(144).enumerate() {
        let d = read_f16(block, 0);
        let dmin = read_f16(block, 2);
        let scales = &block[4..16];
        let qs = &block[16..144];
        let o = &mut out[b * 256..(b + 1) * 256];
        // Four groups of 64 elements; each group holds two 32-element
        // sub-blocks, low nibbles first, then high nibbles.
        for group in 0..4 {
            let (sc1, m1) = scale_min_k4(2 * group, scales);
            let (sc2, m2) = scale_min_k4(2 * group + 1, scales);
            let d1 = d * f32::from(sc1);
            let dm1 = dmin * f32::from(m1);
            let d2 = d * f32::from(sc2);
            let dm2 = dmin * f32::from(m2);
            for l in 0..32 {
                let byte = qs[group * 32 + l];
                o[group * 64 + l] = d1 * f32::from(byte & 0xF) - dm1;
                o[group * 64 + 32 + l] = d2 * f32::from(byte >> 4) - dm2;
            }
        }
    }
}

fn dequant_q5_k(data: &[u8], out: &mut [f32]) {
    for (b, block) in data.chunks_exact(176).enumerate() {
        let d = read_f16(block, 0);
        let dmin = read_f16(block, 2);
        let scales = &block[4..16];
        let qh = &block[16..48];
        let ql = &block[48..176];
        let o = &mut out[b * 256..(b + 1) * 256];
        // Same 64-element grouping as Q4_K. The fifth bits live in one
        // shared 32-byte qh array for the whole block: group g reads bit
        // 2g (first sub-block) and bit 2g+1 (second) of qh[l], so the byte
        // index stays l and only the masks advance.
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for group in 0..4 {
            let (sc1, m1) = scale_min_k4(2 * group, scales);
            let (sc2, m2) = scale_min_k4(2 * group + 1, scales);
            let d1 = d * f32::from(sc1);
            let dm1 = dmin * f32::from(m1);
            let d2 = d * f32::from(sc2);
            let dm2 = dmin * f32::from(m2);
            for l in 0..32 {
                let low_byte = ql[group * 32 + l];
                let high_byte = qh[l];
                let lo = (low_byte & 0xF) + if high_byte & u1 != 0 { 16 } else { 0 };
                let hi = (low_byte >> 4) + if high_byte & u2 != 0 { 16 } else { 0 };
                o[group * 64 + l] = d1 * f32::from(lo) - dm1;
                o[group * 64 + 32 + l] = d2 * f32::from(hi) - dm2;
            }
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

fn dequant_q6_k(data: &[u8], out: &mut [f32]) {
    for (b, block) in data.chunks_exact(210).enumerate() {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let sc = &block[192..208];
        let d = read_f16(block, 208);
        let o = &mut out[b * 256..(b + 1) * 256];
        // Two halves of 128 elements. Each half interleaves four 32-element
        // streams (ql low, next-chunk ql low, ql high, next-chunk ql high),
        // and each qh byte carries the top two bits of all four streams for
        // its index: bits 0-1, 2-3, 4-5, 6-7 respectively.
        for half in 0..2 {
            let ql_h = &ql[half * 64..half * 64 + 64];
            let qh_h = &qh[half * 32..half * 32 + 32];
            let sc_h = &sc[half * 8..half * 8 + 8];
            let base = half * 128;
            for l in 0..32 {
                let is = l / 16;
                let q1 = i32::from((ql_h[l] & 0xF) | ((qh_h[l] & 3) << 4)) - 32;
                let q2 = i32::from((ql_h[l + 32] & 0xF) | (((qh_h[l] >> 2) & 3) << 4)) - 32;
                let q3 = i32::from((ql_h[l] >> 4) | (((qh_h[l] >> 4) & 3) << 4)) - 32;
                let q4 = i32::from((ql_h[l + 32] >> 4) | (((qh_h[l] >> 6) & 3) << 4)) - 32;
                o[base + l] = d * f32::from(sc_h[is] as i8) * q1 as f32;
                o[base + 32 + l] = d * f32::from(sc_h[is + 2] as i8) * q2 as f32;
                o[base + 64 + l] = d * f32::from(sc_h[is + 4] as i8) * q3 as f32;
                o[base + 96 + l] = d * f32::from(sc_h[is + 6] as i8) * q4 as f32;
            }
        }
    }
}

/// Dequantize `elements` elements of type `ty` from their exact on-disk
/// bytes into a fresh f32 vector.
pub fn dequantize(ty: GgmlType, data: &[u8], elements: usize) -> Vec<f32> {
    debug_assert_eq!(
        ty.byte_len(elements as u64),
        Some(data.len() as u64),
        "caller must pass the exact byte length for the element count"
    );
    let mut out = vec![0.0f32; elements];
    match ty {
        GgmlType::F32 => {
            for (slot, chunk) in out.iter_mut().zip(data.chunks_exact(4)) {
                *slot = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        GgmlType::F16 => {
            for (slot, chunk) in out.iter_mut().zip(data.chunks_exact(2)) {
                *slot = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        GgmlType::Bf16 => {
            for (slot, chunk) in out.iter_mut().zip(data.chunks_exact(2)) {
                *slot = bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        GgmlType::Q8_0 => dequant_q8_0(data, &mut out),
        GgmlType::Q2K => dequant_q2_k(data, &mut out),
        GgmlType::Q3K => dequant_q3_k(data, &mut out),
        GgmlType::Q4K => dequant_q4_k(data, &mut out),
        GgmlType::Q5K => dequant_q5_k(data, &mut out),
        GgmlType::Q6K => dequant_q6_k(data, &mut out),
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_known_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        // Smallest subnormal half is exactly 2^-24.
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn bf16_known_values() {
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xc020), -2.5);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn scale_min_k4_low_subblocks() {
        let mut scales = [0u8; 12];
        scales[0] = 0b10_000011; // high bits must be masked off for j < 4
        scales[4] = 0b01_000010;
        assert_eq!(scale_min_k4(0, &scales), (3, 2));
    }

    /// Build one Q2_K block from its parts (f16 values as raw bits).
    fn q2_k_block(scales: &[u8; 16], qs: &[u8; 64], d: u16, dmin: u16) -> Vec<u8> {
        let mut block = Vec::with_capacity(84);
        block.extend_from_slice(scales);
        block.extend_from_slice(qs);
        block.extend_from_slice(&d.to_le_bytes());
        block.extend_from_slice(&dmin.to_le_bytes());
        block
    }

    /// Build one Q3_K block from its parts (f16 value as raw bits).
    fn q3_k_block(hmask: &[u8; 32], qs: &[u8; 64], scales: &[u8; 12], d: u16) -> Vec<u8> {
        let mut block = Vec::with_capacity(110);
        block.extend_from_slice(hmask);
        block.extend_from_slice(qs);
        block.extend_from_slice(scales);
        block.extend_from_slice(&d.to_le_bytes());
        block
    }

    #[test]
    fn q2_k_zero_quants_hit_the_mins() {
        // scale byte s: low nibble 1 (scale), high nibble s (min).
        let mut scales = [0u8; 16];
        for (s, byte) in scales.iter_mut().enumerate() {
            *byte = ((s as u8) << 4) | 1;
        }
        // d = 2.0 (0x4000), dmin = 0.25 (0x3400), all quants zero, so
        // every output is -(dmin * min) = -0.25 * s in sub-block s.
        let block = q2_k_block(&scales, &[0u8; 64], 0x4000, 0x3400);
        let out = dequantize(GgmlType::Q2K, &block, 256);
        for s in 0..16 {
            let base = (s / 8) * 128 + ((s % 8) / 2) * 32 + (s % 2) * 16;
            assert_eq!(out[base], -0.25 * s as f32, "sub-block {s} first");
            assert_eq!(out[base + 15], -0.25 * s as f32, "sub-block {s} last");
        }
    }

    #[test]
    fn q2_k_planes_chunks_and_subblocks() {
        // d = 1.0, dmin = 0, every scale byte 0x01 (scale 1, min 0), so
        // outputs equal the raw 2-bit quants. 0xE4 = planes 0,1,2,3.
        let mut qs = [0u8; 64];
        qs[0] = 0xE4; // chunk 0, sub-block 0, l = 0
        qs[16] = 0xE4; // chunk 0, sub-block 1, l = 0
        qs[32] = 0xE4; // chunk 1, sub-block 0, l = 0
        let block = q2_k_block(&[0x01u8; 16], &qs, 0x3C00, 0x0000);
        let out = dequantize(GgmlType::Q2K, &block, 256);
        for base in [0usize, 16, 128] {
            assert_eq!(out[base], 0.0);
            assert_eq!(out[base + 32], 1.0);
            assert_eq!(out[base + 64], 2.0);
            assert_eq!(out[base + 96], 3.0);
        }
        assert_eq!(out[1], 0.0);
        assert_eq!(out[200], 0.0);
        assert_eq!(out[255], 0.0);
    }

    #[test]
    fn q2_k_max_scale_min_and_quant() {
        // scale 15, min 15, quant 3 everywhere: 0.5*15*3 - 0.5*15 = 15.
        let block = q2_k_block(&[0xFFu8; 16], &[0xFFu8; 64], 0x3800, 0x3800);
        let out = dequantize(GgmlType::Q2K, &block, 256);
        assert!(out.iter().all(|&v| v == 15.0));
    }

    #[test]
    fn q2_k_zero_scales_give_zero() {
        // d and dmin both zero: no scale or min can produce a value.
        let block = q2_k_block(&[0xFFu8; 16], &[0xFFu8; 64], 0x0000, 0x0000);
        let out = dequantize(GgmlType::Q2K, &block, 256);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q3_k_scale_unpack_matches_hand_computation() {
        // Expected values worked byte-by-byte from ggml's uint32 shuffle:
        // bytes 0..8 give the low four bits, bytes 8..12 give the top two
        // bits in four successive 2-bit fields per byte.
        let packed = [
            0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x21, 0x43, 0x65, 0x87,
        ];
        let expected: [i8; 16] = [18, 52, 22, 56, 10, 12, 30, 16, 33, 3, 37, 7, 9, 27, 29, 47];
        assert_eq!(unpack_scales_q3_k(&packed), expected);
    }

    #[test]
    fn q3_k_mask_clear_subtracts_four() {
        // All scale bytes zero unpack to scale 0, so dl = d * (0 - 32).
        // d = 2^-5 (0x2800) makes dl exactly -1. Quants are all zero.
        let scales = [0u8; 12];
        // Mask clear everywhere: value is (0 - 4), output 4.0 everywhere.
        let block = q3_k_block(&[0u8; 32], &[0u8; 64], &scales, 0x2800);
        let out = dequantize(GgmlType::Q3K, &block, 256);
        assert!(out.iter().all(|&v| v == 4.0));
        // Mask set everywhere: value is 0, output 0.0 everywhere.
        let block = q3_k_block(&[0xFFu8; 32], &[0u8; 64], &scales, 0x2800);
        let out = dequantize(GgmlType::Q3K, &block, 256);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q3_k_planes_and_second_chunk_mask_bits() {
        // Same dl = -1 setup as above; mask fully set so values are the
        // raw quants negated. 0xE4 in qs[0] and qs[32] exercises planes
        // in chunk 0 and chunk 1 (chunk 1 reads mask bits 4..8).
        let mut qs = [0u8; 64];
        qs[0] = 0xE4;
        qs[32] = 0xE4;
        let block = q3_k_block(&[0xFFu8; 32], &qs, &[0u8; 12], 0x2800);
        let out = dequantize(GgmlType::Q3K, &block, 256);
        for base in [0usize, 128] {
            assert_eq!(out[base], 0.0);
            assert_eq!(out[base + 32], -1.0);
            assert_eq!(out[base + 64], -2.0);
            assert_eq!(out[base + 96], -3.0);
        }
        assert_eq!(out[1], 0.0);
        assert_eq!(out[255], 0.0);
    }

    #[test]
    fn q3_k_scale_extremes() {
        // All nibbles and 2-bit fields maxed: every scale unpacks to 63,
        // factor (63 - 32) = 31, d = 1.0.
        let block = q3_k_block(&[0xFFu8; 32], &[0xFFu8; 64], &[0xFFu8; 12], 0x3C00);
        let out = dequantize(GgmlType::Q3K, &block, 256);
        assert!(out.iter().all(|&v| v == 93.0));
        // Same quants with the mask clear: value (3 - 4) = -1, output -31.
        let block = q3_k_block(&[0u8; 32], &[0xFFu8; 64], &[0xFFu8; 12], 0x3C00);
        let out = dequantize(GgmlType::Q3K, &block, 256);
        assert!(out.iter().all(|&v| v == -31.0));
    }

    #[test]
    fn scale_min_k4_high_subblocks() {
        // j = 4: sc = (scales[8] & 0xF) | ((scales[0] >> 6) << 4)
        //      m  = (scales[8] >> 4)  | ((scales[4] >> 6) << 4)
        let mut scales = [0u8; 12];
        scales[0] = 0b10_000000;
        scales[4] = 0b01_000000;
        scales[8] = 0b0101_1001;
        assert_eq!(scale_min_k4(4, &scales), (0b10_1001, 0b01_0101));
    }
}
