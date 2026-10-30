//! The GGUF file parser and tensor loader.
//!
//! Parse order follows the format: magic, version, counts, metadata
//! key-values, tensor directory, alignment padding, tensor data. Validation
//! is done once at parse time for everything that can be checked without
//! dequantizing (element counts against block geometry, tensor data ranges
//! against file length), so a malformed file fails at open, not mid-run.
//!
//! Loading is by name and fails loudly: absent tensor, unsupported type,
//! out-of-bounds data are all hard errors. There is deliberately no API
//! that returns placeholder or partially-filled weights.

use std::collections::HashMap;
use std::path::Path;

use crate::dequant;
use crate::diag;
use crate::error::{GgufError, Result};
use crate::ggml::{type_name, GgmlType};
use crate::metadata::{Metadata, MetadataValue};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const SUPPORTED_VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: u32 = 32;

/// One entry of the tensor directory.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    name: String,
    dims: Vec<u64>,
    type_id: u32,
    rel_offset: u64,
    elements: u64,
}

impl TensorInfo {
    /// Tensor name as stored in the file.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Dimensions as stored in the file. GGUF stores the contiguous
    /// (row) dimension first: `dims()[0]` is the length of one row.
    pub fn dims(&self) -> &[u64] {
        &self.dims
    }

    /// Total element count (product of all dimensions).
    pub fn element_count(&self) -> u64 {
        self.elements
    }

    /// Mainline ggml type id as stored in the file.
    pub fn type_id(&self) -> u32 {
        self.type_id
    }

    /// Human-readable ggml type name (for example `Q4_K`, `BF16`).
    pub fn type_name(&self) -> &'static str {
        type_name(self.type_id)
    }

    /// The type, when this loader can dequantize it.
    pub fn ggml_type(&self) -> Option<GgmlType> {
        GgmlType::from_id(self.type_id)
    }

    /// Offset of the tensor data relative to the start of the tensor data
    /// section, exactly as stored in the directory.
    pub fn relative_offset(&self) -> u64 {
        self.rel_offset
    }

    /// On-disk byte length, when the type is supported and the element
    /// count is a whole number of blocks.
    pub fn byte_len(&self) -> Option<u64> {
        self.ggml_type().and_then(|t| t.byte_len(self.elements))
    }
}

/// A parsed GGUF file, memory-resident.
///
/// The whole file is held in memory; memory-mapped backing is the planned
/// next step for the loader module (PLAN T-2.1) and does not change this
/// API. Everything the engine needs for correctness (typed metadata,
/// per-tensor geometry, checked ranges) is validated here.
#[derive(Debug)]
pub struct GgufFile {
    bytes: Vec<u8>,
    version: u32,
    metadata: Metadata,
    tensors: Vec<TensorInfo>,
    index: HashMap<String, usize>,
    alignment: u32,
    data_start: usize,
}

impl GgufFile {
    /// Read and parse a GGUF file from disk.
    pub fn open(path: &Path) -> Result<Self> {
        diag::debug("file_open", &format!("path={}", path.display()));
        let bytes = std::fs::read(path)?;
        Self::from_bytes(bytes)
    }

    /// Parse a GGUF file from an in-memory byte buffer.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let mut reader = Reader::new(&bytes);

        let magic = reader.read_bytes(4, "magic")?;
        if magic != GGUF_MAGIC {
            let found = [magic[0], magic[1], magic[2], magic[3]];
            diag::error("bad_magic", &format!("found={found:?}"));
            return Err(GgufError::BadMagic { found });
        }
        let version = reader.read_u32("version")?;
        if version != SUPPORTED_VERSION {
            diag::error("unsupported_version", &format!("version={version}"));
            return Err(GgufError::UnsupportedVersion(version));
        }
        let tensor_count = reader.read_u64("tensor_count")?;
        let metadata_count = reader.read_u64("metadata_kv_count")?;
        diag::debug(
            "parse_header",
            &format!("version={version} tensors={tensor_count} metadata_kv={metadata_count}"),
        );

        let mut entries = Vec::new();
        for _ in 0..metadata_count {
            let key = reader.read_string("metadata key")?;
            let value = read_metadata_value(&mut reader)?;
            entries.push((key, value));
        }
        let metadata = Metadata::from_entries(entries);

        let mut tensors = Vec::new();
        let mut index = HashMap::new();
        for _ in 0..tensor_count {
            let name = reader.read_string("tensor name")?;
            let n_dims = reader.read_u32("tensor dimension count")?;
            if !(1..=4).contains(&n_dims) {
                return Err(GgufError::InvalidFormat {
                    context: "tensor directory",
                    detail: format!("tensor {name}: dimension count {n_dims} outside 1..=4"),
                });
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            let mut elements: u64 = 1;
            for _ in 0..n_dims {
                let d = reader.read_u64("tensor dimension")?;
                dims.push(d);
                elements = elements.checked_mul(d).ok_or(GgufError::LengthOverflow {
                    context: "tensor element count",
                })?;
            }
            let type_id = reader.read_u32("tensor type")?;
            let rel_offset = reader.read_u64("tensor offset")?;
            if let Some(ty) = GgmlType::from_id(type_id) {
                if ty.byte_len(elements).is_none() {
                    return Err(GgufError::BadElementCount {
                        tensor: name,
                        elements,
                        block_size: ty.block_size(),
                    });
                }
            }
            diag::debug(
                "tensor_dir",
                &format!(
                    "name={name} type={} dims={dims:?} elements={elements} rel_offset={rel_offset}",
                    type_name(type_id)
                ),
            );
            let slot = tensors.len();
            if index.insert(name.clone(), slot).is_some() {
                diag::error("duplicate_tensor", &format!("name={name}"));
                return Err(GgufError::InvalidFormat {
                    context: "tensor directory",
                    detail: format!("duplicate tensor name {name}"),
                });
            }
            tensors.push(TensorInfo {
                name,
                dims,
                type_id,
                rel_offset,
                elements,
            });
        }

        let alignment = match metadata.opt_u32("general.alignment")? {
            Some(a) => a,
            None => DEFAULT_ALIGNMENT,
        };
        if alignment == 0 || !alignment.is_power_of_two() || alignment > (1 << 20) {
            return Err(GgufError::InvalidFormat {
                context: "header",
                detail: format!(
                    "general.alignment {alignment} is not a power of two within 1..=1048576"
                ),
            });
        }
        let header_len = reader.position() as u64;
        let data_start_u = align_up(header_len, u64::from(alignment));
        let data_start = usize::try_from(data_start_u).map_err(|_| GgufError::LengthOverflow {
            context: "tensor data offset",
        })?;

        // Bounds-check every supported tensor's data range against the file.
        let file_len = bytes.len() as u64;
        for tensor in &tensors {
            if let Some(len) = tensor.byte_len() {
                let offset = data_start_u.checked_add(tensor.rel_offset).ok_or(
                    GgufError::LengthOverflow {
                        context: "tensor data offset",
                    },
                )?;
                let in_bounds = offset.checked_add(len).is_some_and(|end| end <= file_len);
                if !in_bounds {
                    diag::error(
                        "tensor_out_of_bounds",
                        &format!("name={} offset={offset} len={len}", tensor.name),
                    );
                    return Err(GgufError::TensorOutOfBounds {
                        tensor: tensor.name.clone(),
                        offset,
                        len,
                        file_len,
                    });
                }
            }
        }

        diag::debug(
            "parse_done",
            &format!(
                "tensors={} metadata_kv={} alignment={alignment} data_start={data_start} file_len={file_len}",
                tensors.len(),
                metadata.len()
            ),
        );

        Ok(Self {
            bytes,
            version,
            metadata,
            tensors,
            index,
            alignment,
            data_start,
        })
    }

    /// GGUF format version (always 3 for a successfully parsed file).
    pub fn version(&self) -> u32 {
        self.version
    }

    /// The metadata section.
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// The architecture string (`general.architecture`), the file's own
    /// declaration of what it is.
    pub fn architecture(&self) -> Result<&str> {
        self.metadata.get_string("general.architecture")
    }

    /// Alignment used for the tensor data section.
    pub fn alignment(&self) -> u32 {
        self.alignment
    }

    /// Absolute byte offset where the tensor data section begins.
    pub fn data_start(&self) -> usize {
        self.data_start
    }

    /// Total file size in bytes.
    pub fn file_len(&self) -> usize {
        self.bytes.len()
    }

    /// All tensor directory entries, in file order.
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// Tensor names, in file order.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(TensorInfo::name)
    }

    /// Whether a tensor with this name exists.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Directory entry for a tensor, or a hard `MissingTensor` error.
    pub fn tensor_info(&self, name: &str) -> Result<&TensorInfo> {
        match self.index.get(name) {
            Some(&slot) => Ok(&self.tensors[slot]),
            None => {
                diag::error("missing_tensor", &format!("name={name}"));
                Err(GgufError::MissingTensor(name.to_string()))
            }
        }
    }

    /// The raw stored bytes of a tensor.
    ///
    /// # Errors
    /// `MissingTensor` when absent; `UnsupportedTensorType` when the type
    /// is not one this loader sizes and dequantizes (a raw read of an
    /// unsized type would be a guess, so it is refused too).
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let info = self.tensor_info(name)?;
        self.tensor_bytes_for(info)
    }

    fn tensor_bytes_for(&self, info: &TensorInfo) -> Result<&[u8]> {
        let len = match info.byte_len() {
            Some(len) => len,
            None => {
                diag::error(
                    "unsupported_type",
                    &format!("name={} type_id={}", info.name, info.type_id),
                );
                return Err(GgufError::UnsupportedTensorType {
                    tensor: info.name.clone(),
                    type_id: info.type_id,
                    type_name: type_name(info.type_id).to_string(),
                });
            }
        };
        let start = self.data_start + info.rel_offset as usize;
        let end = start + len as usize;
        // Ranges were checked at parse time; this slice can only fail if the
        // file was mutated in memory, which cannot happen through this API.
        Ok(&self.bytes[start..end])
    }

    /// Load a tensor fully dequantized to f32.
    ///
    /// # Errors
    /// `MissingTensor` when absent, `UnsupportedTensorType` when the
    /// stored type is not one of the corpus types this loader dequantizes
    /// (F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K,
    /// Q4_K, Q5_K, Q6_K). No partial or placeholder data is ever returned.
    pub fn load_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self.tensor_info(name)?;
        self.load_f32_for(info)
    }

    /// [`load_f32`](Self::load_f32) from a directory entry directly.
    pub fn load_f32_for(&self, info: &TensorInfo) -> Result<Vec<f32>> {
        let ty = match info.ggml_type() {
            Some(ty) => ty,
            None => {
                diag::error(
                    "unsupported_type",
                    &format!("name={} type_id={}", info.name, info.type_id),
                );
                return Err(GgufError::UnsupportedTensorType {
                    tensor: info.name.clone(),
                    type_id: info.type_id,
                    type_name: type_name(info.type_id).to_string(),
                });
            }
        };
        let raw = self.tensor_bytes_for(info)?;
        let elements = usize::try_from(info.elements).map_err(|_| GgufError::LengthOverflow {
            context: "tensor element count",
        })?;
        diag::debug(
            "tensor_load",
            &format!(
                "name={} type={} elements={elements} bytes={}",
                info.name,
                info.type_name(),
                raw.len()
            ),
        );
        Ok(dequant::dequantize(ty, raw, elements))
    }
}

fn align_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

/// Byte cursor over the file with format-aware truncation errors.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn position(&self) -> usize {
        self.pos
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn read_bytes(&mut self, n: usize, context: &'static str) -> Result<&'a [u8]> {
        if self.remaining() < n {
            return Err(GgufError::Truncated {
                context,
                offset: self.pos as u64,
                needed: n as u64,
                file_len: self.bytes.len() as u64,
            });
        }
        let out = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    fn read_u8(&mut self, context: &'static str) -> Result<u8> {
        Ok(self.read_bytes(1, context)?[0])
    }

    fn read_i8(&mut self, context: &'static str) -> Result<i8> {
        Ok(self.read_u8(context)? as i8)
    }

    fn read_u16(&mut self, context: &'static str) -> Result<u16> {
        let b = self.read_bytes(2, context)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_i16(&mut self, context: &'static str) -> Result<i16> {
        Ok(self.read_u16(context)? as i16)
    }

    fn read_u32(&mut self, context: &'static str) -> Result<u32> {
        let b = self.read_bytes(4, context)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_i32(&mut self, context: &'static str) -> Result<i32> {
        Ok(self.read_u32(context)? as i32)
    }

    fn read_u64(&mut self, context: &'static str) -> Result<u64> {
        let b = self.read_bytes(8, context)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_i64(&mut self, context: &'static str) -> Result<i64> {
        Ok(self.read_u64(context)? as i64)
    }

    fn read_f32(&mut self, context: &'static str) -> Result<f32> {
        Ok(f32::from_bits(self.read_u32(context)?))
    }

    fn read_f64(&mut self, context: &'static str) -> Result<f64> {
        Ok(f64::from_bits(self.read_u64(context)?))
    }

    fn read_bool(&mut self, context: &'static str) -> Result<bool> {
        Ok(self.read_u8(context)? != 0)
    }

    fn read_string(&mut self, context: &'static str) -> Result<String> {
        let len = self.read_u64("string length")?;
        let len = usize::try_from(len).map_err(|_| GgufError::LengthOverflow { context })?;
        let raw = self.read_bytes(len, context)?;
        String::from_utf8(raw.to_vec()).map_err(|_| GgufError::InvalidUtf8 { context })
    }
}

/// Read one metadata value, dispatching on its GGUF value-type id.
fn read_metadata_value(reader: &mut Reader<'_>) -> Result<MetadataValue> {
    let type_id = reader.read_u32("metadata value type")?;
    let value = match type_id {
        0 => MetadataValue::UInt8(reader.read_u8("metadata u8")?),
        1 => MetadataValue::Int8(reader.read_i8("metadata i8")?),
        2 => MetadataValue::UInt16(reader.read_u16("metadata u16")?),
        3 => MetadataValue::Int16(reader.read_i16("metadata i16")?),
        4 => MetadataValue::UInt32(reader.read_u32("metadata u32")?),
        5 => MetadataValue::Int32(reader.read_i32("metadata i32")?),
        6 => MetadataValue::Float32(reader.read_f32("metadata f32")?),
        7 => MetadataValue::Bool(reader.read_bool("metadata bool")?),
        8 => MetadataValue::String(reader.read_string("metadata string")?),
        9 => {
            let elem_type = reader.read_u32("metadata array element type")?;
            let len = reader.read_u64("metadata array length")?;
            // Every element costs at least one byte on disk; a declared
            // length larger than the rest of the file is impossible.
            if len > reader.remaining() as u64 {
                return Err(GgufError::Truncated {
                    context: "metadata array",
                    offset: reader.position() as u64,
                    needed: len,
                    file_len: reader.bytes.len() as u64,
                });
            }
            let mut items = Vec::new();
            for _ in 0..len {
                items.push(read_metadata_scalar(reader, elem_type)?);
            }
            MetadataValue::Array(items)
        }
        10 => MetadataValue::UInt64(reader.read_u64("metadata u64")?),
        11 => MetadataValue::Int64(reader.read_i64("metadata i64")?),
        12 => MetadataValue::Float64(reader.read_f64("metadata f64")?),
        other => return Err(GgufError::UnknownMetadataType(other)),
    };
    Ok(value)
}

/// Read one array element (arrays cannot nest in GGUF).
fn read_metadata_scalar(reader: &mut Reader<'_>, type_id: u32) -> Result<MetadataValue> {
    let value = match type_id {
        0 => MetadataValue::UInt8(reader.read_u8("array u8")?),
        1 => MetadataValue::Int8(reader.read_i8("array i8")?),
        2 => MetadataValue::UInt16(reader.read_u16("array u16")?),
        3 => MetadataValue::Int16(reader.read_i16("array i16")?),
        4 => MetadataValue::UInt32(reader.read_u32("array u32")?),
        5 => MetadataValue::Int32(reader.read_i32("array i32")?),
        6 => MetadataValue::Float32(reader.read_f32("array f32")?),
        7 => MetadataValue::Bool(reader.read_bool("array bool")?),
        8 => MetadataValue::String(reader.read_string("array string")?),
        10 => MetadataValue::UInt64(reader.read_u64("array u64")?),
        11 => MetadataValue::Int64(reader.read_i64("array i64")?),
        12 => MetadataValue::Float64(reader.read_f64("array f64")?),
        other => return Err(GgufError::UnknownMetadataType(other)),
    };
    Ok(value)
}
