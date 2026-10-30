//! Error types for the GGUF loader.
//!
//! Design rule (from the 2026-10-29 audit of the old Go loader): every
//! failure is loud and names the thing that failed. There is no error path
//! in this crate that substitutes placeholder data.

use std::fmt;

/// Result alias for loader operations.
pub type Result<T> = std::result::Result<T, GgufError>;

/// Everything that can go wrong while parsing or loading a GGUF file.
#[derive(Debug)]
pub enum GgufError {
    /// Underlying I/O failure (opening or reading the file).
    Io(std::io::Error),
    /// The first four bytes were not the `GGUF` magic.
    BadMagic {
        /// The four bytes actually found.
        found: [u8; 4],
    },
    /// GGUF format version other than 3.
    UnsupportedVersion(u32),
    /// The file ended before a structure was complete.
    Truncated {
        /// What was being read when the file ran out.
        context: &'static str,
        /// Byte offset where the read started.
        offset: u64,
        /// Bytes the read needed.
        needed: u64,
        /// Total file size in bytes.
        file_len: u64,
    },
    /// A metadata key or tensor name was not valid UTF-8.
    InvalidUtf8 {
        /// What the string was (metadata key, tensor name, string value).
        context: &'static str,
    },
    /// A metadata value type id that GGUF does not define.
    UnknownMetadataType(u32),
    /// A string/array length or element count that cannot fit in memory.
    LengthOverflow {
        /// What carried the impossible length.
        context: &'static str,
    },
    /// A tensor was requested by name and is not in the file.
    MissingTensor(String),
    /// A tensor's element count is not a whole number of quantization blocks.
    BadElementCount {
        /// Tensor name.
        tensor: String,
        /// Element count from the tensor directory.
        elements: u64,
        /// Block size of the tensor's type.
        block_size: u64,
    },
    /// A tensor's declared data range falls outside the file.
    TensorOutOfBounds {
        /// Tensor name.
        tensor: String,
        /// Absolute byte offset of the tensor data.
        offset: u64,
        /// Byte length of the tensor data.
        len: u64,
        /// Total file size in bytes.
        file_len: u64,
    },
    /// A tensor uses a type this loader does not dequantize. This is a hard
    /// error by design: the old loader silently substituted random weights
    /// for types it could not read, and that defect class is banned here.
    UnsupportedTensorType {
        /// Tensor name.
        tensor: String,
        /// Mainline ggml type id from the file.
        type_id: u32,
        /// Human-readable type name, or `unknown` for unassigned ids.
        type_name: String,
    },
    /// A structural rule of the GGUF format was violated.
    InvalidFormat {
        /// The structure being parsed.
        context: &'static str,
        /// What was wrong, in plain words.
        detail: String,
    },
    /// A metadata key the caller required is absent.
    MissingMetadata(String),
    /// A metadata key exists but holds a different value type than asked for.
    MetadataTypeMismatch {
        /// Metadata key.
        key: String,
        /// Type the caller asked for.
        expected: &'static str,
        /// Type actually stored in the file.
        found: &'static str,
    },
    /// A coverage check failed: one or more required tensors are missing,
    /// unreadable, or misshapen. Each entry names the tensor and the
    /// reason. The load is refused as a whole; there is no partial load.
    CoverageIncomplete {
        /// One `tensor <name>: <reason>` string per failing tensor.
        failures: Vec<String>,
    },
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::BadMagic { found } => {
                write!(f, "bad magic: expected GGUF, found {found:?}")
            }
            Self::UnsupportedVersion(v) => {
                write!(f, "unsupported GGUF version {v} (this loader reads version 3)")
            }
            Self::Truncated {
                context,
                offset,
                needed,
                file_len,
            } => write!(
                f,
                "file truncated reading {context}: need {needed} bytes at offset {offset}, file is {file_len} bytes"
            ),
            Self::InvalidUtf8 { context } => write!(f, "invalid UTF-8 in {context}"),
            Self::UnknownMetadataType(id) => write!(f, "unknown metadata value type id {id}"),
            Self::LengthOverflow { context } => {
                write!(f, "declared length for {context} overflows addressable memory")
            }
            Self::MissingTensor(name) => write!(f, "tensor not found: {name}"),
            Self::BadElementCount {
                tensor,
                elements,
                block_size,
            } => write!(
                f,
                "tensor {tensor}: element count {elements} is not a multiple of block size {block_size}"
            ),
            Self::TensorOutOfBounds {
                tensor,
                offset,
                len,
                file_len,
            } => write!(
                f,
                "tensor {tensor}: data range [{offset}, {}) exceeds file length {file_len}",
                offset.saturating_add(*len)
            ),
            Self::UnsupportedTensorType {
                tensor,
                type_id,
                type_name,
            } => write!(
                f,
                "tensor {tensor}: unsupported type {type_name} (ggml type id {type_id}); this loader dequantizes F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K only"
            ),
            Self::InvalidFormat { context, detail } => {
                write!(f, "invalid GGUF format in {context}: {detail}")
            }
            Self::MissingMetadata(key) => write!(f, "metadata key not found: {key}"),
            Self::CoverageIncomplete { failures } => write!(
                f,
                "coverage incomplete: {} required tensor(s) failed: {}",
                failures.len(),
                failures.join("; ")
            ),
            Self::MetadataTypeMismatch {
                key,
                expected,
                found,
            } => write!(
                f,
                "metadata key {key}: expected {expected}, found {found}"
            ),
        }
    }
}

impl std::error::Error for GgufError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
