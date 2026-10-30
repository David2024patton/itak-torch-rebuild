//! The adapters: muse-glimmer's [`Metadata`] and [`TensorSource`]
//! traits implemented over a parsed [`GgufFile`].
//!
//! The two traits have different error shapes and the adapters respect
//! both. [`Metadata`] returns `Option`: `None` means absent or a
//! genuinely different kind of value, per the trait contract. The
//! `gguf` crate's getters fail loud with named errors; the adapter
//! translates, and when a key is present but unreadable as the
//! requested kind it says so on the diag channel, because that is the
//! fact a debugging session needs and the `Option` has no room for it.
//! Integer coercion (any stored width into `u32`) is the `gguf`
//! getter's own lossless widening, not a second implementation here.
//!
//! [`TensorSource`] returns the arch module's `Result`: absent tensors
//! map to [`GlimmerError::MissingTensor`], and present-but-undeliverable
//! tensors (a ggml type the `gguf` crate cannot size or read) map to
//! [`GlimmerError::TensorUnreadable`] with the loader's reason attached.
//! Neither adapter ever invents a value: no default metadata, no
//! placeholder bytes.

use std::sync::Arc;

use gguf::{GgufError, GgufFile, MetadataValue};
use muse_glimmer::{
    Dtype, GlimmerError, Metadata, Result as GlimmerResult, TensorSource, TensorView,
};

use crate::diag;

/// muse-glimmer's [`Metadata`] trait over one parsed GGUF file.
#[derive(Debug)]
pub struct GgufMetadata<'a> {
    file: &'a GgufFile,
}

impl<'a> GgufMetadata<'a> {
    /// Wrap a parsed file.
    pub fn new(file: &'a GgufFile) -> Self {
        Self { file }
    }

    /// The wrapped file.
    pub fn file(&self) -> &'a GgufFile {
        self.file
    }

    /// A present key whose value cannot be read as the requested kind
    /// is worth a diag line even though the trait answer is `None`.
    fn note_unreadable(&self, key: &str, err: &GgufError) {
        if self.file.metadata().get(key).is_some() {
            diag::debug("metadata_unreadable", &format!("key={key} reason={err}"));
        }
    }
}

impl Metadata for GgufMetadata<'_> {
    fn get_string(&self, key: &str) -> Option<String> {
        match self.file.metadata().get_string(key) {
            Ok(v) => Some(v.to_string()),
            Err(e) => {
                self.note_unreadable(key, &e);
                None
            }
        }
    }

    fn get_u32(&self, key: &str) -> Option<u32> {
        match self.file.metadata().get_u32(key) {
            Ok(v) => Some(v),
            Err(e) => {
                self.note_unreadable(key, &e);
                None
            }
        }
    }

    fn get_f32(&self, key: &str) -> Option<f32> {
        match self.file.metadata().get_f32(key) {
            Ok(v) => Some(v),
            Err(e) => {
                self.note_unreadable(key, &e);
                None
            }
        }
    }

    fn get_array_len(&self, key: &str) -> Option<usize> {
        match self.file.metadata().get(key) {
            Some(MetadataValue::Array(items)) => Some(items.len()),
            Some(_) => {
                diag::debug(
                    "metadata_unreadable",
                    &format!("key={key} reason=stored value is not an array"),
                );
                None
            }
            None => None,
        }
    }
}

/// muse-glimmer's [`TensorSource`] trait over one parsed GGUF file.
///
/// Tensor bytes are copied out of the file image into an [`Arc`]:
/// [`TensorView`] owns its bytes by contract, and the `gguf` crate's
/// borrowed slices cannot outlive the file through that API. The copy
/// is one memcpy per tensor at load time; nothing in the token loop
/// touches this path.
#[derive(Debug)]
pub struct GgufTensorSource<'a> {
    file: &'a GgufFile,
}

impl<'a> GgufTensorSource<'a> {
    /// Wrap a parsed file.
    pub fn new(file: &'a GgufFile) -> Self {
        Self { file }
    }

    /// The wrapped file.
    pub fn file(&self) -> &'a GgufFile {
        self.file
    }
}

impl TensorSource for GgufTensorSource<'_> {
    fn tensor(&self, name: &str) -> GlimmerResult<TensorView> {
        let info = self
            .file
            .tensor_info(name)
            .map_err(|e| map_tensor_error(name, e))?;
        let bytes = self
            .file
            .tensor_bytes(name)
            .map_err(|e| map_tensor_error(name, e))?;
        diag::debug(
            "tensor_served",
            &format!(
                "name={name} type={} elements={} bytes={}",
                info.type_name(),
                info.element_count(),
                bytes.len()
            ),
        );
        Ok(TensorView {
            name: name.to_string(),
            dtype: Dtype::from_ggml_id(info.type_id()),
            shape: info.dims().to_vec(),
            data: Arc::from(bytes),
        })
    }
}

/// Translate a `gguf` refusal into the arch module's vocabulary:
/// absent is `MissingTensor`, everything else is `TensorUnreadable`
/// with the loader's own words kept as the detail.
fn map_tensor_error(name: &str, err: GgufError) -> GlimmerError {
    match err {
        GgufError::MissingTensor(_) => GlimmerError::MissingTensor {
            name: name.to_string(),
        },
        other => GlimmerError::TensorUnreadable {
            name: name.to_string(),
            detail: other.to_string(),
        },
    }
}

/// dflash's [`Metadata`](dflash::Metadata) trait over one parsed
/// GGUF file. Same translation discipline as [`GgufMetadata`]: an
/// unreadable-but-present key earns a diag line, never a fabricated
/// value.
#[derive(Debug)]
pub struct DflashMetadata<'a> {
    file: &'a GgufFile,
}

impl<'a> DflashMetadata<'a> {
    /// Wrap a parsed file.
    pub fn new(file: &'a GgufFile) -> Self {
        Self { file }
    }

    /// The wrapped file.
    pub fn file(&self) -> &'a GgufFile {
        self.file
    }

    fn note_unreadable(&self, key: &str, err: &GgufError) {
        if self.file.metadata().get(key).is_some() {
            diag::debug("metadata_unreadable", &format!("key={key} reason={err}"));
        }
    }
}

impl dflash::Metadata for DflashMetadata<'_> {
    fn get_string(&self, key: &str) -> Option<String> {
        match self.file.metadata().get_string(key) {
            Ok(v) => Some(v.to_string()),
            Err(e) => {
                self.note_unreadable(key, &e);
                None
            }
        }
    }

    fn get_u32(&self, key: &str) -> Option<u32> {
        match self.file.metadata().get_u32(key) {
            Ok(v) => Some(v),
            Err(e) => {
                self.note_unreadable(key, &e);
                None
            }
        }
    }

    fn get_f32(&self, key: &str) -> Option<f32> {
        match self.file.metadata().get_f32(key) {
            Ok(v) => Some(v),
            Err(e) => {
                self.note_unreadable(key, &e);
                None
            }
        }
    }

    fn get_bool(&self, key: &str) -> Option<bool> {
        match self.file.metadata().get_bool(key) {
            Ok(v) => Some(v),
            Err(e) => {
                self.note_unreadable(key, &e);
                None
            }
        }
    }

    fn get_u32_array(&self, key: &str) -> Option<Vec<u32>> {
        match self.file.metadata().get_u32_array(key) {
            Ok(v) => Some(v),
            Err(e) => {
                self.note_unreadable(key, &e);
                None
            }
        }
    }

    fn get_bool_array(&self, key: &str) -> Option<Vec<bool>> {
        match self.file.metadata().get_array(key) {
            Ok(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        MetadataValue::Bool(v) => out.push(*v),
                        other => {
                            diag::debug(
                                "metadata_unreadable",
                                &format!(
                                    "key={key} reason=array element is {}, not bool",
                                    other.type_name()
                                ),
                            );
                            return None;
                        }
                    }
                }
                Some(out)
            }
            Err(e) => {
                self.note_unreadable(key, &e);
                None
            }
        }
    }

    fn get_array_len(&self, key: &str) -> Option<usize> {
        match self.file.metadata().get(key) {
            Some(MetadataValue::Array(items)) => Some(items.len()),
            Some(_) => {
                diag::debug(
                    "metadata_unreadable",
                    &format!("key={key} reason=stored value is not an array"),
                );
                None
            }
            None => None,
        }
    }
}

/// dflash's [`TensorSource`](dflash::TensorSource) trait over one
/// parsed GGUF file. Unlike the muse-glimmer adapter, which serves
/// raw stored views, this one serves tensors dequantized to f32
/// (the dflash module's `TensorData` contract) through the `gguf`
/// crate's own loader, so dtype support is exactly the `gguf`
/// crate's and anything else refuses by name.
#[derive(Debug)]
pub struct DflashTensorSource<'a> {
    file: &'a GgufFile,
}

impl<'a> DflashTensorSource<'a> {
    /// Wrap a parsed file.
    pub fn new(file: &'a GgufFile) -> Self {
        Self { file }
    }

    /// The wrapped file.
    pub fn file(&self) -> &'a GgufFile {
        self.file
    }
}

impl dflash::TensorSource for DflashTensorSource<'_> {
    fn tensor(&self, name: &str) -> dflash::Result<dflash::TensorData> {
        let info = self
            .file
            .tensor_info(name)
            .map_err(|e| map_dflash_tensor_error(name, e))?;
        let data = self
            .file
            .load_f32(name)
            .map_err(|e| map_dflash_tensor_error(name, e))?;
        diag::debug(
            "tensor_served",
            &format!(
                "name={name} type={} elements={}",
                info.type_name(),
                info.element_count()
            ),
        );
        Ok(dflash::TensorData {
            name: name.to_string(),
            shape: info.dims().to_vec(),
            data,
        })
    }

    fn has_tensor(&self, name: &str) -> bool {
        self.file.has_tensor(name)
    }
}

/// Translate a `gguf` refusal into the dflash module's vocabulary.
fn map_dflash_tensor_error(name: &str, err: GgufError) -> dflash::DflashError {
    match err {
        GgufError::MissingTensor(_) => dflash::DflashError::MissingTensor {
            name: name.to_string(),
        },
        other => dflash::DflashError::TensorUnreadable {
            name: name.to_string(),
            detail: other.to_string(),
        },
    }
}
