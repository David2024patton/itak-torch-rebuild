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
