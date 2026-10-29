//! ggml tensor types, using MAINLINE ggml type ids.
//!
//! The old Go loader died on this table: it mapped id 30 to I2_S, but in
//! mainline ggml id 30 is BF16, so every BF16 tensor decoded as 2-bit
//! ternary garbage. The ids below are the mainline `ggml_type` enum values
//! and are pinned by unit tests so the collision can never come back.
//!
//! Only the types this loader can dequantize to f32 are modeled as
//! supported: F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K. Any other id is
//! reported by name in a hard error (see [`type_name`]) instead of being
//! silently skipped.

/// A ggml tensor element type this loader supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    /// 32-bit float, stored raw.
    F32 = 0,
    /// 16-bit IEEE half float.
    F16 = 1,
    /// 8-bit integer blocks of 32 with one f16 scale per block.
    Q8_0 = 8,
    /// K-quant 4-bit, 256-element super-blocks.
    Q4K = 12,
    /// K-quant 5-bit, 256-element super-blocks.
    Q5K = 13,
    /// K-quant 6-bit, 256-element super-blocks.
    Q6K = 14,
    /// Brain float 16. Mainline ggml id 30. NOT I2_S.
    Bf16 = 30,
}

impl GgmlType {
    /// Look up a supported type by its mainline ggml id. Returns `None` for
    /// every other id, including ids that name real ggml types this loader
    /// does not dequantize (Q3_K, Q8_1, and friends).
    pub fn from_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            8 => Some(Self::Q8_0),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            30 => Some(Self::Bf16),
            _ => None,
        }
    }

    /// The mainline ggml type id.
    pub fn id(self) -> u32 {
        self as u32
    }

    /// Elements per quantization block. Unquantized types have block size 1.
    pub fn block_size(self) -> u64 {
        match self {
            Self::F32 | Self::F16 | Self::Bf16 => 1,
            Self::Q8_0 => 32,
            Self::Q4K | Self::Q5K | Self::Q6K => 256,
        }
    }

    /// Bytes one block occupies in the file.
    pub fn block_bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            // f16 scale + 32 x i8
            Self::Q8_0 => 2 + 32,
            // f16 d + f16 dmin + 12 scale bytes + 128 nibble bytes
            Self::Q4K => 2 + 2 + 12 + 128,
            // f16 d + f16 dmin + 12 scale bytes + 32 high-bit bytes + 128 nibble bytes
            Self::Q5K => 2 + 2 + 12 + 32 + 128,
            // 128 low-nibble bytes + 64 high-bit bytes + 16 i8 scales + f16 d
            Self::Q6K => 128 + 64 + 16 + 2,
        }
    }

    /// On-disk byte length of a tensor with `elements` elements, or `None`
    /// when the count is not a whole number of blocks.
    pub fn byte_len(self, elements: u64) -> Option<u64> {
        let block = self.block_size();
        if elements % block != 0 {
            return None;
        }
        (elements / block).checked_mul(self.block_bytes())
    }
}

/// Human-readable name for ANY mainline ggml type id, supported or not, so
/// hard errors can name what the file actually contains. Returns `unknown`
/// for ids mainline ggml has not assigned.
///
/// The trap from the old loader is called out here: I2_S in Microsoft's
/// bitnet.cpp fork is id 36 there, but has no mainline id; mainline id 30
/// is BF16 and is what this function reports.
pub fn type_name(id: u32) -> &'static str {
    match id {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        34 => "TQ1_0",
        35 => "TQ2_0",
        39 => "MXFP4",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mainline_ids_are_pinned() {
        // The exact collision that corrupted BF16 tensors in the old loader.
        assert_eq!(GgmlType::Bf16.id(), 30);
        assert_eq!(type_name(30), "BF16");
        assert_eq!(GgmlType::from_id(30), Some(GgmlType::Bf16));
        // The audit's missing readers must stay hard errors, not support.
        assert_eq!(GgmlType::from_id(11), None); // Q3_K
        assert_eq!(GgmlType::from_id(9), None); // Q8_1
        assert_eq!(type_name(11), "Q3_K");
        assert_eq!(type_name(9), "Q8_1");
    }

    #[test]
    fn block_geometry_matches_ggml() {
        assert_eq!(GgmlType::Q4K.block_bytes(), 144);
        assert_eq!(GgmlType::Q5K.block_bytes(), 176);
        assert_eq!(GgmlType::Q6K.block_bytes(), 210);
        assert_eq!(GgmlType::Q8_0.block_bytes(), 34);
        assert_eq!(GgmlType::Q4K.byte_len(256), Some(144));
        assert_eq!(GgmlType::Q4K.byte_len(255), None);
        assert_eq!(GgmlType::F32.byte_len(7), Some(28));
    }
}
