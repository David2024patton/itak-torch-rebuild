//! Read access to GGUF metadata, abstracted behind a trait.
//!
//! The config parser never touches a GGUF file directly. It reads through
//! [`Metadata`], which the loader crate (TODO T-2.1, `itak-loader` in
//! this workspace) implements over the `gguf` crate's parsed file.
//! Keeping the parser behind this seam means config logic is testable
//! with an in-memory map and the loader can be dropped in without
//! touching this crate.
//!
//! Implementor contract: GGUF stores integers at several widths. The
//! getters coerce any integer width the file used into the requested
//! type, and return `None` only when the key is absent or holds a value
//! of a genuinely different kind (a string where a number was required).

use std::collections::HashMap;

/// Typed read access to a GGUF file's metadata block.
pub trait Metadata {
    /// String value for `key`, or `None` when absent.
    fn get_string(&self, key: &str) -> Option<String>;
    /// Unsigned 32-bit value for `key`, coerced from any integer width
    /// the file used, or `None` when absent.
    fn get_u32(&self, key: &str) -> Option<u32>;
    /// 32-bit float value for `key`, coerced from any float width the
    /// file used, or `None` when absent.
    fn get_f32(&self, key: &str) -> Option<f32>;
    /// Element count of the array stored at `key`, or `None` when absent.
    /// The config reads the vocabulary size from the length of the
    /// tokenizer token array, which is how GGUF carries it.
    fn get_array_len(&self, key: &str) -> Option<usize>;
    /// Elements of the bool array stored at `key`, or `None` when
    /// absent or not a bool array. The sliding-window pattern key is
    /// carried this way: one bool per layer, `true` for a
    /// sliding-window layer, exactly llama.cpp's `is_swa` semantics
    /// (src/llama-model.cpp: `load_swa_pattern` reads the array form
    /// of the key before any other shape).
    fn get_bool_array(&self, key: &str) -> Option<Vec<bool>>;
}

/// In-memory [`Metadata`] for tests and tools. Not a file parser: it
/// holds exactly what the caller put in it, which is what makes missing
/// key tests honest (a key that was not inserted is absent, full stop).
#[derive(Debug, Default)]
pub struct MapMetadata {
    strings: HashMap<String, String>,
    u32s: HashMap<String, u32>,
    f32s: HashMap<String, f32>,
    array_lens: HashMap<String, usize>,
    bool_arrays: HashMap<String, Vec<bool>>,
}

impl MapMetadata {
    /// An empty metadata map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a string value.
    pub fn insert_string(&mut self, key: &str, value: &str) {
        self.strings.insert(key.to_string(), value.to_string());
    }

    /// Insert an unsigned 32-bit value.
    pub fn insert_u32(&mut self, key: &str, value: u32) {
        self.u32s.insert(key.to_string(), value);
    }

    /// Insert a 32-bit float value.
    pub fn insert_f32(&mut self, key: &str, value: f32) {
        self.f32s.insert(key.to_string(), value);
    }

    /// Insert an array length.
    pub fn insert_array_len(&mut self, key: &str, len: usize) {
        self.array_lens.insert(key.to_string(), len);
    }

    /// Insert a bool array.
    pub fn insert_bool_array(&mut self, key: &str, values: Vec<bool>) {
        self.bool_arrays.insert(key.to_string(), values);
    }

    /// Remove a key from every map, for missing-key tests.
    pub fn remove(&mut self, key: &str) {
        self.strings.remove(key);
        self.u32s.remove(key);
        self.f32s.remove(key);
        self.array_lens.remove(key);
        self.bool_arrays.remove(key);
    }
}

impl Metadata for MapMetadata {
    fn get_string(&self, key: &str) -> Option<String> {
        self.strings.get(key).cloned()
    }

    fn get_u32(&self, key: &str) -> Option<u32> {
        self.u32s.get(key).copied()
    }

    fn get_f32(&self, key: &str) -> Option<f32> {
        self.f32s.get(key).copied()
    }

    fn get_array_len(&self, key: &str) -> Option<usize> {
        self.array_lens.get(key).copied()
    }

    fn get_bool_array(&self, key: &str) -> Option<Vec<bool>> {
        self.bool_arrays.get(key).cloned()
    }
}
