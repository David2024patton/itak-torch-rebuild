//! GGUF metadata: the key-value section between the header and the tensor
//! directory. Values keep the exact type the file stored; the typed getters
//! on [`Metadata`] widen only where the conversion is lossless and report a
//! named mismatch everywhere else.

use crate::error::{GgufError, Result};

/// One metadata value, mirroring the GGUF value-type enum.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    /// Type 0.
    UInt8(u8),
    /// Type 1.
    Int8(i8),
    /// Type 2.
    UInt16(u16),
    /// Type 3.
    Int16(i16),
    /// Type 4.
    UInt32(u32),
    /// Type 5.
    Int32(i32),
    /// Type 6.
    Float32(f32),
    /// Type 7.
    Bool(bool),
    /// Type 8.
    String(String),
    /// Type 9. GGUF arrays are homogeneous; elements share one value type.
    Array(Vec<MetadataValue>),
    /// Type 10.
    UInt64(u64),
    /// Type 11.
    Int64(i64),
    /// Type 12.
    Float64(f64),
}

impl MetadataValue {
    /// Short type name used in mismatch errors.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::UInt8(_) => "u8",
            Self::Int8(_) => "i8",
            Self::UInt16(_) => "u16",
            Self::Int16(_) => "i16",
            Self::UInt32(_) => "u32",
            Self::Int32(_) => "i32",
            Self::Float32(_) => "f32",
            Self::Bool(_) => "bool",
            Self::String(_) => "string",
            Self::Array(_) => "array",
            Self::UInt64(_) => "u64",
            Self::Int64(_) => "i64",
            Self::Float64(_) => "f64",
        }
    }

    /// The GGUF value-type id for this variant.
    pub fn type_id(&self) -> u32 {
        match self {
            Self::UInt8(_) => 0,
            Self::Int8(_) => 1,
            Self::UInt16(_) => 2,
            Self::Int16(_) => 3,
            Self::UInt32(_) => 4,
            Self::Int32(_) => 5,
            Self::Float32(_) => 6,
            Self::Bool(_) => 7,
            Self::String(_) => 8,
            Self::Array(_) => 9,
            Self::UInt64(_) => 10,
            Self::Int64(_) => 11,
            Self::Float64(_) => 12,
        }
    }

    /// Lossless view as u64, for the unsigned and small signed integers.
    fn as_u64(&self) -> Option<u64> {
        match self {
            Self::UInt8(v) => Some(u64::from(*v)),
            Self::UInt16(v) => Some(u64::from(*v)),
            Self::UInt32(v) => Some(u64::from(*v)),
            Self::UInt64(v) => Some(*v),
            Self::Int8(v) => u64::try_from(i64::from(*v)).ok(),
            Self::Int16(v) => u64::try_from(i64::from(*v)).ok(),
            Self::Int32(v) => u64::try_from(i64::from(*v)).ok(),
            Self::Int64(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }

    /// Lossless view as i64.
    fn as_i64(&self) -> Option<i64> {
        match self {
            Self::UInt8(v) => Some(i64::from(*v)),
            Self::UInt16(v) => Some(i64::from(*v)),
            Self::UInt32(v) => Some(i64::from(*v)),
            Self::UInt64(v) => i64::try_from(*v).ok(),
            Self::Int8(v) => Some(i64::from(*v)),
            Self::Int16(v) => Some(i64::from(*v)),
            Self::Int32(v) => Some(i64::from(*v)),
            Self::Int64(v) => Some(*v),
            _ => None,
        }
    }

    /// View as f64: exact for f32/f64, and for integers up to 2^53.
    fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float32(v) => Some(f64::from(*v)),
            Self::Float64(v) => Some(*v),
            _ => self
                .as_i64()
                .map(|v| v as f64)
                .or_else(|| self.as_u64().map(|v| v as f64)),
        }
    }
}

/// The full metadata section of one GGUF file, in file order.
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    entries: Vec<(String, MetadataValue)>,
}

impl Metadata {
    pub(crate) fn from_entries(entries: Vec<(String, MetadataValue)>) -> Self {
        Self { entries }
    }

    /// Number of key-value pairs.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the section is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate keys and values in file order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &MetadataValue)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Raw lookup; `None` when the key is absent.
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Raw lookup that fails loudly when the key is absent.
    pub fn require(&self, key: &str) -> Result<&MetadataValue> {
        self.get(key)
            .ok_or_else(|| GgufError::MissingMetadata(key.to_string()))
    }

    fn mismatch(&self, key: &str, expected: &'static str) -> GgufError {
        match self.get(key) {
            None => GgufError::MissingMetadata(key.to_string()),
            Some(v) => GgufError::MetadataTypeMismatch {
                key: key.to_string(),
                expected,
                found: v.type_name(),
            },
        }
    }

    /// Integer read as u32. Accepts any integer type whose value fits.
    pub fn get_u32(&self, key: &str) -> Result<u32> {
        let v = self.get_u64(key)?;
        u32::try_from(v).map_err(|_| self.mismatch(key, "u32"))
    }

    /// Integer read as u64. Accepts any integer type whose value fits.
    pub fn get_u64(&self, key: &str) -> Result<u64> {
        match self.require(key)?.as_u64() {
            Some(v) => Ok(v),
            None => Err(self.mismatch(key, "u64")),
        }
    }

    /// Integer read as i64. Accepts any integer type whose value fits.
    pub fn get_i64(&self, key: &str) -> Result<i64> {
        match self.require(key)?.as_i64() {
            Some(v) => Ok(v),
            None => Err(self.mismatch(key, "i64")),
        }
    }

    /// Float read as f32. Accepts f32 and f64 storage.
    pub fn get_f32(&self, key: &str) -> Result<f32> {
        match self.require(key)? {
            MetadataValue::Float32(v) => Ok(*v),
            MetadataValue::Float64(v) => Ok(*v as f32),
            _ => Err(self.mismatch(key, "f32")),
        }
    }

    /// Float read as f64. Accepts f32 and f64 storage.
    pub fn get_f64(&self, key: &str) -> Result<f64> {
        match self.require(key)?.as_f64() {
            Some(v) => Ok(v),
            None => Err(self.mismatch(key, "f64")),
        }
    }

    /// Strict bool read.
    pub fn get_bool(&self, key: &str) -> Result<bool> {
        match self.require(key)? {
            MetadataValue::Bool(v) => Ok(*v),
            _ => Err(self.mismatch(key, "bool")),
        }
    }

    /// Strict string read.
    pub fn get_string(&self, key: &str) -> Result<&str> {
        match self.require(key)? {
            MetadataValue::String(v) => Ok(v.as_str()),
            _ => Err(self.mismatch(key, "string")),
        }
    }

    /// Optional string read: `Ok(None)` when absent, error when present
    /// with the wrong type.
    pub fn opt_string(&self, key: &str) -> Result<Option<&str>> {
        match self.get(key) {
            None => Ok(None),
            Some(MetadataValue::String(v)) => Ok(Some(v.as_str())),
            Some(_) => Err(self.mismatch(key, "string")),
        }
    }

    /// Optional u32 read: `Ok(None)` when absent.
    pub fn opt_u32(&self, key: &str) -> Result<Option<u32>> {
        match self.get(key) {
            None => Ok(None),
            Some(_) => self.get_u32(key).map(Some),
        }
    }

    /// Optional f32 read: `Ok(None)` when absent.
    pub fn opt_f32(&self, key: &str) -> Result<Option<f32>> {
        match self.get(key) {
            None => Ok(None),
            Some(_) => self.get_f32(key).map(Some),
        }
    }

    /// Array read: the raw element list.
    pub fn get_array(&self, key: &str) -> Result<&Vec<MetadataValue>> {
        match self.require(key)? {
            MetadataValue::Array(v) => Ok(v),
            _ => Err(self.mismatch(key, "array")),
        }
    }

    /// Array of strings read (tokenizer vocabularies, merges, and friends).
    pub fn get_string_array(&self, key: &str) -> Result<Vec<String>> {
        let arr = self.get_array(key)?;
        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            match item {
                MetadataValue::String(s) => out.push(s.clone()),
                other => {
                    return Err(GgufError::MetadataTypeMismatch {
                        key: key.to_string(),
                        expected: "array of string",
                        found: other.type_name(),
                    })
                }
            }
        }
        Ok(out)
    }

    /// Array of i32 read (token type ids and similar small tables).
    pub fn get_i32_array(&self, key: &str) -> Result<Vec<i32>> {
        let arr = self.get_array(key)?;
        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            match item.as_i64().and_then(|v| i32::try_from(v).ok()) {
                Some(v) => out.push(v),
                None => {
                    return Err(GgufError::MetadataTypeMismatch {
                        key: key.to_string(),
                        expected: "array of i32",
                        found: item.type_name(),
                    })
                }
            }
        }
        Ok(out)
    }

    /// Array of u32 read (for example per-layer head counts in hybrid
    /// architectures).
    pub fn get_u32_array(&self, key: &str) -> Result<Vec<u32>> {
        let arr = self.get_array(key)?;
        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            match item.as_u64().and_then(|v| u32::try_from(v).ok()) {
                Some(v) => out.push(v),
                None => {
                    return Err(GgufError::MetadataTypeMismatch {
                        key: key.to_string(),
                        expected: "array of u32",
                        found: item.type_name(),
                    })
                }
            }
        }
        Ok(out)
    }

    /// Array of f32 read (for example RoPE scaling factor tables).
    pub fn get_f32_array(&self, key: &str) -> Result<Vec<f32>> {
        let arr = self.get_array(key)?;
        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            match item {
                MetadataValue::Float32(v) => out.push(*v),
                MetadataValue::Float64(v) => out.push(*v as f32),
                other => {
                    return Err(GgufError::MetadataTypeMismatch {
                        key: key.to_string(),
                        expected: "array of f32",
                        found: other.type_name(),
                    })
                }
            }
        }
        Ok(out)
    }
}
