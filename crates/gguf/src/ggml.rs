//! ggml tensor types, using MAINLINE ggml type ids.
//!
//! The old Go loader died on this table: it mapped id 30 to I2_S, but in
//! mainline ggml id 30 is BF16, so every BF16 tensor decoded as 2-bit
//! ternary garbage. The ids below are the mainline `ggml_type` enum values
//! and are pinned by unit tests so the collision can never come back.
//!
//! Three layers, kept separate on purpose:
//!
//! - [`GgmlType`]: the types this loader can dequantize to f32. That is
//!   exactly the set the teardown corpus contains (ARCHITECTURE.md 5.2):
//!   F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K,
//!   Q5_K, Q6_K.
//! - [`type_info`] / [`MAINLINE_TYPE_TABLE`]: every id mainline ggml has
//!   assigned (plus the removed ones, marked [`TypeStatus::Removed`]), so
//!   errors can name what a file actually contains even when no reader
//!   exists here.
//! - [`fork_type_name`]: fork-only types (the BitNet I2_S family) live in
//!   a table keyed by fork source. Fork ids are the fork's own numbering
//!   and are never merged into, or renumbered against, the mainline
//!   table. A mainline file carrying id 36 contains an *unassigned*
//!   mainline id; I2_S exists only inside the bitnet.cpp namespace.

/// A ggml tensor element type this loader dequantizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    /// 32-bit float, stored raw.
    F32 = 0,
    /// 16-bit IEEE half float.
    F16 = 1,
    /// 4-bit blocks of 32, one f16 scale per block, values offset by 8.
    Q4_0 = 2,
    /// 4-bit blocks of 32, one f16 scale and one f16 minimum per block.
    Q4_1 = 3,
    /// 5-bit blocks of 32, one f16 scale per block, values offset by 16;
    /// the fifth bits are packed into one u32 per block.
    Q5_0 = 6,
    /// 5-bit blocks of 32, f16 scale and minimum, u32 of fifth bits.
    Q5_1 = 7,
    /// 8-bit integer blocks of 32 with one f16 scale per block.
    Q8_0 = 8,
    /// 8-bit integer blocks of 32 with an f16 scale and an f16 block sum
    /// (the sum serves dot products; row values are `scale * q`).
    Q8_1 = 9,
    /// K-quant 2-bit, 256-element super-blocks of sixteen 16-element
    /// sub-blocks, each with a 4-bit scale and 4-bit minimum packed into
    /// one byte.
    Q2K = 10,
    /// K-quant 3-bit, 256-element super-blocks: 2-bit fields plus a
    /// one-bit-per-element high mask, eight 6-bit sub-block scales.
    Q3K = 11,
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
    /// Every type this loader dequantizes: exactly the teardown corpus
    /// set from ARCHITECTURE.md 5.2.
    pub const ALL: [Self; 14] = [
        Self::F32,
        Self::F16,
        Self::Q4_0,
        Self::Q4_1,
        Self::Q5_0,
        Self::Q5_1,
        Self::Q8_0,
        Self::Q8_1,
        Self::Q2K,
        Self::Q3K,
        Self::Q4K,
        Self::Q5K,
        Self::Q6K,
        Self::Bf16,
    ];

    /// Look up a supported type by its mainline ggml id. Returns `None`
    /// for every other id, including ids that name real ggml types this
    /// loader does not dequantize (Q8_K, the IQ family, and friends).
    pub fn from_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
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
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K => 256,
        }
    }

    /// Bytes one block occupies in the file.
    pub fn block_bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            // f16 scale + 16 nibble bytes
            Self::Q4_0 => 2 + 16,
            // f16 scale + f16 min + 16 nibble bytes
            Self::Q4_1 => 2 + 2 + 16,
            // f16 scale + u32 high bits + 16 nibble bytes
            Self::Q5_0 => 2 + 4 + 16,
            // f16 scale + f16 min + u32 high bits + 16 nibble bytes
            Self::Q5_1 => 2 + 2 + 4 + 16,
            // f16 scale + 32 x i8
            Self::Q8_0 => 2 + 32,
            // f16 scale + f16 sum + 32 x i8
            Self::Q8_1 => 2 + 2 + 32,
            // 16 scale/min bytes + 64 packed bytes + f16 d + f16 dmin
            Self::Q2K => 16 + 64 + 2 + 2,
            // 32 high-mask bytes + 64 packed bytes + 12 scale bytes + f16 d
            Self::Q3K => 32 + 64 + 12 + 2,
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

/// Whether a mainline ggml type id has a reader here, is a real ggml
/// type this loader does not dequantize, or was removed from ggml.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeStatus {
    /// This loader dequantizes the type (it is in [`GgmlType`]).
    Reader,
    /// A real, current mainline ggml type with no reader in this loader.
    /// Files using it are refused by name, never decoded as another type.
    Known,
    /// The id was assigned by an older ggml and later removed (Q4_2,
    /// Q4_3, and the Q4_0 repacks 31 to 33). Named so errors stay
    /// truthful about what an old file contains.
    Removed,
}

/// One row of the mainline ggml type table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeInfo {
    /// Mainline ggml type id.
    pub id: u32,
    /// ggml's own name for the type.
    pub name: &'static str,
    /// Reader status in this loader.
    pub status: TypeStatus,
}

/// The full mainline ggml type table: every id ggml has assigned,
/// including the removed ones. Ids 36, 37, and 38 are unassigned in
/// mainline ggml and deliberately absent (they are NOT I2_S; see
/// [`fork_type_name`]).
pub const MAINLINE_TYPE_TABLE: &[TypeInfo] = &[
    TypeInfo {
        id: 0,
        name: "F32",
        status: TypeStatus::Reader,
    },
    TypeInfo {
        id: 1,
        name: "F16",
        status: TypeStatus::Reader,
    },
    TypeInfo {
        id: 2,
        name: "Q4_0",
        status: TypeStatus::Reader,
    },
    TypeInfo {
        id: 3,
        name: "Q4_1",
        status: TypeStatus::Reader,
    },
    TypeInfo {
        id: 4,
        name: "Q4_2",
        status: TypeStatus::Removed,
    },
    TypeInfo {
        id: 5,
        name: "Q4_3",
        status: TypeStatus::Removed,
    },
    TypeInfo {
        id: 6,
        name: "Q5_0",
        status: TypeStatus::Reader,
    },
    TypeInfo {
        id: 7,
        name: "Q5_1",
        status: TypeStatus::Reader,
    },
    TypeInfo {
        id: 8,
        name: "Q8_0",
        status: TypeStatus::Reader,
    },
    TypeInfo {
        id: 9,
        name: "Q8_1",
        status: TypeStatus::Reader,
    },
    TypeInfo {
        id: 10,
        name: "Q2_K",
        status: TypeStatus::Reader,
    },
    TypeInfo {
        id: 11,
        name: "Q3_K",
        status: TypeStatus::Reader,
    },
    TypeInfo {
        id: 12,
        name: "Q4_K",
        status: TypeStatus::Reader,
    },
    TypeInfo {
        id: 13,
        name: "Q5_K",
        status: TypeStatus::Reader,
    },
    TypeInfo {
        id: 14,
        name: "Q6_K",
        status: TypeStatus::Reader,
    },
    TypeInfo {
        id: 15,
        name: "Q8_K",
        status: TypeStatus::Known,
    },
    TypeInfo {
        id: 16,
        name: "IQ2_XXS",
        status: TypeStatus::Known,
    },
    TypeInfo {
        id: 17,
        name: "IQ2_XS",
        status: TypeStatus::Known,
    },
    TypeInfo {
        id: 18,
        name: "IQ3_XXS",
        status: TypeStatus::Known,
    },
    TypeInfo {
        id: 19,
        name: "IQ1_S",
        status: TypeStatus::Known,
    },
    TypeInfo {
        id: 20,
        name: "IQ4_NL",
        status: TypeStatus::Known,
    },
    TypeInfo {
        id: 21,
        name: "IQ3_S",
        status: TypeStatus::Known,
    },
    TypeInfo {
        id: 22,
        name: "IQ2_S",
        status: TypeStatus::Known,
    },
    TypeInfo {
        id: 23,
        name: "IQ4_XS",
        status: TypeStatus::Known,
    },
    TypeInfo {
        id: 24,
        name: "I8",
        status: TypeStatus::Known,
    },
    TypeInfo {
        id: 25,
        name: "I16",
        status: TypeStatus::Known,
    },
    TypeInfo {
        id: 26,
        name: "I32",
        status: TypeStatus::Known,
    },
    TypeInfo {
        id: 27,
        name: "I64",
        status: TypeStatus::Known,
    },
    TypeInfo {
        id: 28,
        name: "F64",
        status: TypeStatus::Known,
    },
    TypeInfo {
        id: 29,
        name: "IQ1_M",
        status: TypeStatus::Known,
    },
    TypeInfo {
        id: 30,
        name: "BF16",
        status: TypeStatus::Reader,
    },
    TypeInfo {
        id: 31,
        name: "Q4_0_4_4",
        status: TypeStatus::Removed,
    },
    TypeInfo {
        id: 32,
        name: "Q4_0_4_8",
        status: TypeStatus::Removed,
    },
    TypeInfo {
        id: 33,
        name: "Q4_0_8_8",
        status: TypeStatus::Removed,
    },
    TypeInfo {
        id: 34,
        name: "TQ1_0",
        status: TypeStatus::Known,
    },
    TypeInfo {
        id: 35,
        name: "TQ2_0",
        status: TypeStatus::Known,
    },
    TypeInfo {
        id: 39,
        name: "MXFP4",
        status: TypeStatus::Known,
    },
];

/// Table row for a mainline ggml type id, or `None` when mainline ggml
/// has not assigned the id.
pub fn type_info(id: u32) -> Option<&'static TypeInfo> {
    MAINLINE_TYPE_TABLE.iter().find(|row| row.id == id)
}

/// Human-readable name for ANY mainline ggml type id, supported or not,
/// so hard errors can name what the file actually contains. Returns
/// `unknown` for ids mainline ggml has not assigned.
pub fn type_name(id: u32) -> &'static str {
    type_info(id).map_or("unknown", |row| row.name)
}

/// A ggml fork whose private type ids this loader can name.
///
/// Fork type ids are the fork's own numbering. They are looked up only
/// through [`fork_type_name`], keyed by fork, and never enter the
/// mainline table: the old loader's fatal defect was reading a fork id
/// (BitNet's I2_S) as if it were a mainline id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForkSource {
    /// Microsoft's bitnet.cpp fork of ggml.
    BitNetCpp,
}

impl ForkSource {
    /// Short label for diagnostics.
    pub fn label(self) -> &'static str {
        match self {
            Self::BitNetCpp => "microsoft/bitnet.cpp",
        }
    }
}

/// Name a fork-only type id inside its fork's namespace.
///
/// In bitnet.cpp, id 36 is I2_S (2-bit ternary). That id means nothing in
/// a mainline file: mainline ggml has not assigned 36, and
/// [`type_name(36)`](type_name) is `unknown`. Returns `None` for ids the
/// fork has not assigned either.
pub fn fork_type_name(source: ForkSource, id: u32) -> Option<&'static str> {
    match source {
        ForkSource::BitNetCpp => match id {
            36 => Some("I2_S"),
            _ => None,
        },
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
        assert_eq!(
            type_info(30),
            Some(&TypeInfo {
                id: 30,
                name: "BF16",
                status: TypeStatus::Reader
            })
        );
    }

    #[test]
    fn corpus_types_all_have_readers() {
        // ARCHITECTURE.md 5.2: the dissected corpus contains exactly these
        // types, and every one of them must have a reader.
        let corpus = [
            (0, "F32"),
            (1, "F16"),
            (30, "BF16"),
            (2, "Q4_0"),
            (3, "Q4_1"),
            (6, "Q5_0"),
            (7, "Q5_1"),
            (8, "Q8_0"),
            (9, "Q8_1"),
            (10, "Q2_K"),
            (11, "Q3_K"),
            (12, "Q4_K"),
            (13, "Q5_K"),
            (14, "Q6_K"),
        ];
        assert_eq!(GgmlType::ALL.len(), corpus.len());
        for (id, name) in corpus {
            assert!(GgmlType::from_id(id).is_some(), "no reader for {name}");
            assert_eq!(type_name(id), name);
            assert_eq!(type_info(id).map(|r| r.status), Some(TypeStatus::Reader));
        }
    }

    #[test]
    fn known_types_without_readers_stay_refused() {
        for (id, name) in [(15, "Q8_K"), (16, "IQ2_XXS"), (28, "F64"), (39, "MXFP4")] {
            assert_eq!(GgmlType::from_id(id), None, "{name} must not decode");
            assert_eq!(type_name(id), name);
            assert_eq!(type_info(id).map(|r| r.status), Some(TypeStatus::Known));
        }
    }

    #[test]
    fn removed_and_unassigned_ids_are_named_honestly() {
        for (id, name) in [
            (4, "Q4_2"),
            (5, "Q4_3"),
            (31, "Q4_0_4_4"),
            (32, "Q4_0_4_8"),
            (33, "Q4_0_8_8"),
        ] {
            assert_eq!(type_name(id), name);
            assert_eq!(type_info(id).map(|r| r.status), Some(TypeStatus::Removed));
            assert_eq!(GgmlType::from_id(id), None);
        }
        // Unassigned in mainline ggml: no name, no reader, no table row.
        for id in [36, 37, 38, 40, 99] {
            assert_eq!(type_name(id), "unknown", "id {id}");
            assert_eq!(type_info(id), None, "id {id}");
            assert_eq!(GgmlType::from_id(id), None, "id {id}");
        }
    }

    #[test]
    fn fork_types_are_namespaced_never_mainline() {
        // I2_S exists only inside the bitnet.cpp namespace, at the fork's
        // own id 36. Mainline id 36 stays unassigned, so the old loader's
        // collision (reading a mainline file's BF16 as I2_S, or an I2_S
        // file as a mainline type) cannot recur through this table.
        assert_eq!(fork_type_name(ForkSource::BitNetCpp, 36), Some("I2_S"));
        assert_eq!(fork_type_name(ForkSource::BitNetCpp, 30), None);
        assert_eq!(type_name(36), "unknown");
        assert_eq!(ForkSource::BitNetCpp.label(), "microsoft/bitnet.cpp");
    }

    #[test]
    fn table_is_complete_and_consistent() {
        // Every row's reader status agrees with GgmlType::from_id, and the
        // table has no duplicate or out-of-order ids.
        let mut prev: Option<u32> = None;
        for row in MAINLINE_TYPE_TABLE {
            if let Some(p) = prev {
                assert!(row.id > p, "table ids must strictly increase");
            }
            prev = Some(row.id);
            assert_eq!(
                GgmlType::from_id(row.id).is_some(),
                row.status == TypeStatus::Reader,
                "id {} ({})",
                row.id,
                row.name
            );
            if row.status == TypeStatus::Reader {
                assert_eq!(GgmlType::from_id(row.id).map(|t| t.id()), Some(row.id));
            }
        }
        // The assigned mainline ids are 0..=35 and 39 (GGML_TYPE_COUNT is
        // 40 in mainline ggml); 36..=38 are the only gaps.
        assert_eq!(MAINLINE_TYPE_TABLE.len(), 37);
        assert_eq!(MAINLINE_TYPE_TABLE.last().map(|r| r.id), Some(39));
    }

    #[test]
    fn block_geometry_matches_ggml() {
        assert_eq!(GgmlType::Q4K.block_bytes(), 144);
        assert_eq!(GgmlType::Q5K.block_bytes(), 176);
        assert_eq!(GgmlType::Q6K.block_bytes(), 210);
        assert_eq!(GgmlType::Q8_0.block_bytes(), 34);
        assert_eq!(GgmlType::Q4_0.block_bytes(), 18);
        assert_eq!(GgmlType::Q4_1.block_bytes(), 20);
        assert_eq!(GgmlType::Q5_0.block_bytes(), 22);
        assert_eq!(GgmlType::Q5_1.block_bytes(), 24);
        assert_eq!(GgmlType::Q8_1.block_bytes(), 36);
        assert_eq!(GgmlType::Q2K.block_bytes(), 84);
        assert_eq!(GgmlType::Q3K.block_bytes(), 110);
        assert_eq!(GgmlType::Q4K.byte_len(256), Some(144));
        assert_eq!(GgmlType::Q4K.byte_len(255), None);
        assert_eq!(GgmlType::Q2K.byte_len(256), Some(84));
        assert_eq!(GgmlType::Q3K.byte_len(512), Some(220));
        assert_eq!(GgmlType::Q4_0.byte_len(32), Some(18));
        assert_eq!(GgmlType::Q4_0.byte_len(31), None);
        assert_eq!(GgmlType::F32.byte_len(7), Some(28));
    }
}
