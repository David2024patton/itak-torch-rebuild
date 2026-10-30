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

fn dequant_q4_0(data: &[u8], out: &mut [f32]) {
    for (b, block) in data.chunks_exact(18).enumerate() {
        let d = read_f16(block, 0);
        let qs = &block[2..18];
        let o = &mut out[b * 32..(b + 1) * 32];
        // Low nibbles are elements 0..16, high nibbles elements 16..32;
        // both are offset by 8 (symmetric range -8..=7).
        for (j, byte) in qs.iter().enumerate() {
            o[j] = d * (i32::from(byte & 0xF) - 8) as f32;
            o[j + 16] = d * (i32::from(byte >> 4) - 8) as f32;
        }
    }
}

fn dequant_q4_1(data: &[u8], out: &mut [f32]) {
    for (b, block) in data.chunks_exact(20).enumerate() {
        let d = read_f16(block, 0);
        let m = read_f16(block, 2);
        let qs = &block[4..20];
        let o = &mut out[b * 32..(b + 1) * 32];
        for (j, byte) in qs.iter().enumerate() {
            o[j] = d * f32::from(byte & 0xF) + m;
            o[j + 16] = d * f32::from(byte >> 4) + m;
        }
    }
}

/// The fifth bits of a Q5_0/Q5_1 block: one u32 whose bit `j` belongs to
/// element `j` and bit `j + 16` to element `j + 16`.
fn q5_value(byte: u8, nibble_high: bool, qh: u32, bit: u32) -> u32 {
    let low = if nibble_high { byte >> 4 } else { byte & 0xF };
    u32::from(low) | (((qh >> bit) & 1) << 4)
}

fn dequant_q5_0(data: &[u8], out: &mut [f32]) {
    for (b, block) in data.chunks_exact(22).enumerate() {
        let d = read_f16(block, 0);
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let qs = &block[6..22];
        let o = &mut out[b * 32..(b + 1) * 32];
        for (j, byte) in qs.iter().enumerate() {
            let lo = q5_value(*byte, false, qh, j as u32);
            let hi = q5_value(*byte, true, qh, j as u32 + 16);
            o[j] = d * (lo as i32 - 16) as f32;
            o[j + 16] = d * (hi as i32 - 16) as f32;
        }
    }
}

fn dequant_q5_1(data: &[u8], out: &mut [f32]) {
    for (b, block) in data.chunks_exact(24).enumerate() {
        let d = read_f16(block, 0);
        let m = read_f16(block, 2);
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let qs = &block[8..24];
        let o = &mut out[b * 32..(b + 1) * 32];
        for (j, byte) in qs.iter().enumerate() {
            let lo = q5_value(*byte, false, qh, j as u32);
            let hi = q5_value(*byte, true, qh, j as u32 + 16);
            o[j] = d * lo as f32 + m;
            o[j + 16] = d * hi as f32 + m;
        }
    }
}

fn dequant_q8_1(data: &[u8], out: &mut [f32]) {
    for (b, block) in data.chunks_exact(36).enumerate() {
        let d = read_f16(block, 0);
        // block[2..4] is the f16 block sum ggml keeps for dot products;
        // row values are scale * q only.
        let o = &mut out[b * 32..(b + 1) * 32];
        for (i, slot) in o.iter_mut().enumerate() {
            *slot = d * f32::from(block[4 + i] as i8);
        }
    }
}

fn dequant_q2_k(data: &[u8], out: &mut [f32]) {
    for (b, block) in data.chunks_exact(84).enumerate() {
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = read_f16(block, 80);
        let dmin = read_f16(block, 82);
        let o = &mut out[b * 256..(b + 1) * 256];
        // Sixteen sub-blocks of 16 elements; sub-block s has its 4-bit
        // scale in the low nibble of scales[s] and its 4-bit minimum in
        // the high nibble. The 2-bit fields are stream-packed like every
        // K-quant: element e reads byte (e mod 64), bit pair (e / 64).
        for (s, &scale_byte) in scales.iter().enumerate() {
            let sc = f32::from(scale_byte & 0xF);
            let mn = f32::from(scale_byte >> 4);
            for l in 0..16 {
                let e = s * 16 + l;
                let q = f32::from((qs[e % 64] >> (2 * (e / 64))) & 3);
                o[e] = d * sc * q - dmin * mn;
            }
        }
    }
}

fn dequant_q3_k(data: &[u8], out: &mut [f32]) {
    for (b, block) in data.chunks_exact(110).enumerate() {
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let scales = &block[96..108];
        let d = read_f16(block, 108);
        // Eight 6-bit sub-block scales: low 4 bits in scales[0..8], the
        // top 2 bits packed into scales[8..12] (byte j+8 carries scale j
        // in bits 0-1 and scale j+4 in bits 2-3).
        let mut aux = [0u8; 8];
        for j in 0..4 {
            aux[j] = (scales[j] & 0xF) | ((scales[j + 8] & 0x3) << 4);
            aux[j + 4] = (scales[j + 4] & 0xF) | (((scales[j + 8] >> 2) & 0x3) << 4);
        }
        let o = &mut out[b * 256..(b + 1) * 256];
        // Element e: 2-bit field from byte (e mod 64), bit pair (e / 64);
        // the third bit is hmask bit e. The 3-bit value is offset by 4.
        for e in 0..256 {
            let q2 = u32::from((qs[e % 64] >> (2 * (e / 64))) & 3);
            let h = u32::from((hmask[e / 8] >> (e % 8)) & 1);
            let v = (q2 | (h << 2)) as i32 - 4;
            o[e] = d * f32::from(aux[e / 32]) * v as f32;
        }
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
        GgmlType::Q4_0 => dequant_q4_0(data, &mut out),
        GgmlType::Q4_1 => dequant_q4_1(data, &mut out),
        GgmlType::Q5_0 => dequant_q5_0(data, &mut out),
        GgmlType::Q5_1 => dequant_q5_1(data, &mut out),
        GgmlType::Q8_0 => dequant_q8_0(data, &mut out),
        GgmlType::Q8_1 => dequant_q8_1(data, &mut out),
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

    const F16_ONE_BITS: u16 = 0x3c00;
    const F16_TWO_BITS: u16 = 0x4000;
    const F16_HALF_BITS: u16 = 0x3800;

    fn block_bytes<const N: usize>(d: u16, fill: &[(usize, u8)]) -> Vec<u8> {
        let mut block = vec![0u8; N];
        block[0..2].copy_from_slice(&d.to_le_bytes());
        for &(i, v) in fill {
            block[i] = v;
        }
        block
    }

    #[test]
    fn q4_0_known_values() {
        // d = 2.0; qs[0] = 0x21 -> element 0 is (1 - 8) * 2, element 16
        // is (2 - 8) * 2; qs[1] = 0xF0 -> (0 - 8) * 2 and (15 - 8) * 2.
        let data = block_bytes::<18>(F16_TWO_BITS, &[(2, 0x21), (3, 0xF0)]);
        let out = dequantize(GgmlType::Q4_0, &data, 32);
        assert_eq!(out[0], -14.0);
        assert_eq!(out[16], -12.0);
        assert_eq!(out[1], -16.0);
        assert_eq!(out[17], 14.0);
        assert_eq!(out[2], -16.0); // zero nibble is the -8 floor
    }

    #[test]
    fn q4_1_known_values() {
        // d = 0.5, m = 1.0 in bytes 2..4; qs[0] = 0x21.
        let mut block = vec![0u8; 20];
        block[0..2].copy_from_slice(&F16_HALF_BITS.to_le_bytes());
        block[2..4].copy_from_slice(&F16_ONE_BITS.to_le_bytes());
        block[4] = 0x21;
        let out = dequantize(GgmlType::Q4_1, &block, 32);
        assert_eq!(out[0], 1.5); // 0.5 * 1 + 1
        assert_eq!(out[16], 2.0); // 0.5 * 2 + 1
        assert_eq!(out[1], 1.0); // zero nibble is exactly the minimum
    }

    #[test]
    fn q5_0_known_values() {
        // d = 1.0; qh bit 0 set; qs[0] = 0x03. Element 0 is (3 | 16) - 16,
        // element 16 is (0 | 0) - 16.
        let mut block = vec![0u8; 22];
        block[0..2].copy_from_slice(&F16_ONE_BITS.to_le_bytes());
        block[2..6].copy_from_slice(&1u32.to_le_bytes());
        block[6] = 0x03;
        let out = dequantize(GgmlType::Q5_0, &block, 32);
        assert_eq!(out[0], 3.0);
        assert_eq!(out[16], -16.0);
        // qh bit 17 belongs to element 17: set qs[1] high nibble to 1 too.
        block[2..6].copy_from_slice(&(1u32 | (1 << 17)).to_le_bytes());
        block[7] = 0x10;
        let out = dequantize(GgmlType::Q5_0, &block, 32);
        assert_eq!(out[17], 1.0); // (1 | 16) - 16
    }

    #[test]
    fn q5_1_known_values() {
        // d = 1.0, m = 0.5, qh = 0; qs[0] = 0x21.
        let mut block = vec![0u8; 24];
        block[0..2].copy_from_slice(&F16_ONE_BITS.to_le_bytes());
        block[2..4].copy_from_slice(&F16_HALF_BITS.to_le_bytes());
        block[8] = 0x21;
        let out = dequantize(GgmlType::Q5_1, &block, 32);
        assert_eq!(out[0], 1.5);
        assert_eq!(out[16], 2.5);
    }

    #[test]
    fn q8_1_known_values() {
        // d = 0.5; the sum field (bytes 2..4) must not affect values.
        let mut block = vec![0u8; 36];
        block[0..2].copy_from_slice(&F16_HALF_BITS.to_le_bytes());
        block[2..4].copy_from_slice(&0x7c00u16.to_le_bytes()); // +inf "sum"
        block[4] = 4;
        block[5] = (-2i8) as u8;
        let out = dequantize(GgmlType::Q8_1, &block, 32);
        assert_eq!(out[0], 2.0);
        assert_eq!(out[1], -1.0);
        assert_eq!(out[31], 0.0);
    }

    #[test]
    fn q2_k_known_values() {
        // d = 1.0, dmin = 1.0 (both at the tail: scales 0..16, qs 16..80).
        // Sub-block 0: scale 2, min 1 (byte 0x12). Sub-block 4: scale 1,
        // min 3 (byte 0x31). qs[0] has bit pair 0 = 0 and bit pair 1 = 1;
        // qs[1] low pair is 3.
        let mut block = vec![0u8; 84];
        block[0] = 0x12;
        block[4] = 0x31;
        block[16] = 0b0000_0100;
        block[17] = 0b0000_0011;
        block[80..82].copy_from_slice(&F16_ONE_BITS.to_le_bytes());
        block[82..84].copy_from_slice(&F16_ONE_BITS.to_le_bytes());
        let out = dequantize(GgmlType::Q2K, &block, 256);
        assert_eq!(out[0], -1.0); // 1*2*0 - 1*1
        assert_eq!(out[1], 5.0); // 1*2*3 - 1*1
        assert_eq!(out[64], -2.0); // sub-block 4: 1*1*1 - 1*3
        assert_eq!(out[2], -1.0); // zero field still pays the minimum
    }

    #[test]
    fn q3_k_known_values() {
        // d = 1.0. aux[0] = 5 (scales[0] low nibble 5, no high bits);
        // aux[1] = 3 (scales[1] low nibble 3). Element 0: qs pair 1 plus
        // hmask bit 0 set -> value (1 | 4) - 4 = 1, times scale 5.
        // Element 32 (sub-block 1): qs[32] pair 2, hmask clear -> -2 * 3.
        let mut block = vec![0u8; 110];
        block[0] = 0x01; // hmask bit 0
        block[32] = 0b0000_0001; // qs[0]
        block[32 + 32] = 0b0000_0010; // qs[32]
        block[96] = 5; // scales[0]
        block[97] = 3; // scales[1]
        block[108..110].copy_from_slice(&F16_ONE_BITS.to_le_bytes());
        let out = dequantize(GgmlType::Q3K, &block, 256);
        assert_eq!(out[0], 5.0);
        assert_eq!(out[32], -6.0);
        assert_eq!(out[1], -20.0); // zero field: (0 - 4) * 5
    }

    #[test]
    fn q3_k_scale_high_bits_come_from_tail_bytes() {
        // aux[4] takes its low nibble from scales[4] and its top 2 bits
        // from scales[8] bits 2-3: 1 | (2 << 4) = 33. Element 128 is in
        // sub-block 4; give it value (0 | 4) - 4 + ... use hmask + pair 0
        // -> v = 0, wait: choose pair 3 and hmask set -> v = 7, (7-4)=3.
        let mut block = vec![0u8; 110];
        block[100] = 1; // scales[4] low nibble
        block[104] = 0b0000_1000; // scales[8]: bits 2-3 = 2 -> aux[4] high bits
                                  // Element 128: byte qs[128 % 64] = qs[0], pair 128 / 64 = 2.
        block[32] = 0b0011_0000; // pair 2 = 3
        block[16] = 0x01; // hmask bit 128 = byte 16, bit 0
        block[108..110].copy_from_slice(&F16_ONE_BITS.to_le_bytes());
        let out = dequantize(GgmlType::Q3K, &block, 256);
        assert_eq!(out[128], 99.0); // 1.0 * 33 * (7 - 4)
    }
}
