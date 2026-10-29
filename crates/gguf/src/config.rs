//! Model configuration read from the file's own metadata.
//!
//! GGUF keys the architecture parameters under the architecture's own
//! name: a file declaring `general.architecture = "qwen2"` stores
//! `qwen2.block_count`, `qwen2.embedding_length`, and so on. This module
//! resolves those keys from the file itself. Nothing here invents values:
//! required fields are hard errors when absent, and the two fields with a
//! genuine format-level convention (KV head count falls back to the query
//! head count; RoPE base may be absent) are explicit `Option`s documenting
//! that convention at the call site.

use crate::error::Result;
use crate::file::GgufFile;

/// The architecture parameters an engine needs before it can build a
/// forward pass, read from one GGUF file.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    /// Architecture string, the file's own declaration (`general.architecture`).
    pub architecture: String,
    /// Transformer block (layer) count, `<arch>.block_count`.
    pub layer_count: u32,
    /// Hidden size, `<arch>.embedding_length`.
    pub hidden_size: u32,
    /// Feed-forward inner size, `<arch>.feed_forward_length`, when present.
    pub ffn_size: Option<u32>,
    /// Query attention heads, `<arch>.attention.head_count`. When a file
    /// stores the count as a per-layer array (hybrid architectures), this
    /// is the maximum and the full array is in `head_count_per_layer`.
    pub head_count: u32,
    /// Per-layer query head counts, when the file stores them as an array.
    pub head_count_per_layer: Option<Vec<u32>>,
    /// Key/value attention heads. GGUF omits `<arch>.attention.head_count_kv`
    /// for full multi-head attention; the fallback to `head_count` is the
    /// format's own convention and is applied here, once, visibly. When a
    /// file stores the count as a per-layer array (hybrid architectures,
    /// where non-attention layers carry 0), this is the maximum, i.e. the
    /// KV head count of the attention layers, and the full array is in
    /// `head_count_kv_per_layer`.
    pub head_count_kv: u32,
    /// Per-layer KV head counts, when the file stores them as an array.
    pub head_count_kv_per_layer: Option<Vec<u32>>,
    /// Context length the model was trained for, `<arch>.context_length`.
    pub context_length: u32,
    /// RoPE frequency base, `<arch>.rope.freq_base` (the spec key; the
    /// `<arch>.attention.rope.freq_base` spelling some converters emit is
    /// accepted as a fallback), when present.
    pub rope_base: Option<f32>,
    /// RMSNorm epsilon, `<arch>.attention.layer_norm_rms_epsilon`, when present.
    pub norm_eps: Option<f32>,
    /// Vocabulary size, from `tokenizer.ggml.tokens` when a tokenizer is
    /// embedded, else `<arch>.vocab_size` when present.
    pub vocab_size: Option<usize>,
}

impl ModelConfig {
    /// Read the configuration from a parsed file's metadata.
    ///
    /// # Errors
    /// Missing or mistyped required keys are hard errors naming the key.
    pub fn from_file(file: &GgufFile) -> Result<Self> {
        let md = file.metadata();
        let arch = file.architecture()?.to_string();
        let key = |suffix: &str| format!("{arch}.{suffix}");

        let layer_count = md.get_u32(&key("block_count"))?;
        let hidden_size = md.get_u32(&key("embedding_length"))?;
        let ffn_size = md.opt_u32(&key("feed_forward_length"))?;
        let (head_count, head_count_per_layer) = read_head_count(md, &key("attention.head_count"))?;
        let head_count = match head_count {
            Some(v) => v,
            None => {
                return Err(crate::error::GgufError::MissingMetadata(key(
                    "attention.head_count",
                )))
            }
        };
        let (kv_scalar, head_count_kv_per_layer) =
            read_head_count(md, &key("attention.head_count_kv"))?;
        let head_count_kv = match kv_scalar {
            Some(v) => v,
            None => head_count,
        };
        let context_length = md.get_u32(&key("context_length"))?;
        let rope_base = match md.opt_f32(&key("rope.freq_base"))? {
            Some(v) => Some(v),
            None => md.opt_f32(&key("attention.rope.freq_base"))?,
        };
        let norm_eps = md.opt_f32(&key("attention.layer_norm_rms_epsilon"))?;
        let vocab_size = match md.get("tokenizer.ggml.tokens") {
            Some(crate::metadata::MetadataValue::Array(items)) => Some(items.len()),
            _ => md.opt_u32(&key("vocab_size"))?.map(|v| v as usize),
        };

        Ok(Self {
            architecture: arch,
            layer_count,
            hidden_size,
            ffn_size,
            head_count,
            head_count_per_layer,
            head_count_kv,
            head_count_kv_per_layer,
            context_length,
            rope_base,
            norm_eps,
            vocab_size,
        })
    }
}

/// Read a head-count key that a file may store either as a scalar or as a
/// per-layer array (hybrid architectures do the latter, with 0 on layers
/// that have no attention). Returns `(scalar, per_layer)`: for array
/// storage the scalar is the array maximum, for scalar storage the array
/// is `None`, and for an absent key both are `None`.
fn read_head_count(
    md: &crate::metadata::Metadata,
    key: &str,
) -> Result<(Option<u32>, Option<Vec<u32>>)> {
    match md.get(key) {
        None => Ok((None, None)),
        Some(crate::metadata::MetadataValue::Array(_)) => {
            let layers = md.get_u32_array(key)?;
            let scalar = layers.iter().max().copied();
            Ok((scalar, Some(layers)))
        }
        Some(_) => Ok((Some(md.get_u32(key)?), None)),
    }
}

/// Tokenizer data embedded in a GGUF file.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenizerInfo {
    /// Token strings by id (`tokenizer.ggml.tokens`).
    pub tokens: Vec<String>,
    /// Token type ids (`tokenizer.ggml.token_type`), when present.
    pub token_types: Option<Vec<i32>>,
    /// Byte-pair merge rules (`tokenizer.ggml.merges`), when present.
    pub merges: Option<Vec<String>>,
    /// Beginning-of-sequence token id (`tokenizer.ggml.bos_token_id`).
    pub bos_token_id: Option<u32>,
    /// End-of-sequence token id (`tokenizer.ggml.eos_token_id`).
    pub eos_token_id: Option<u32>,
    /// Padding token id (`tokenizer.ggml.padding_token_id`).
    pub padding_token_id: Option<u32>,
}

impl TokenizerInfo {
    /// Read tokenizer data from a parsed file's metadata.
    ///
    /// # Errors
    /// `MissingMetadata` when the file embeds no tokenizer tokens;
    /// mistyped arrays are named mismatches.
    pub fn from_file(file: &GgufFile) -> Result<Self> {
        let md = file.metadata();
        let tokens = md.get_string_array("tokenizer.ggml.tokens")?;
        let token_types = match md.get("tokenizer.ggml.token_type") {
            Some(_) => Some(md.get_i32_array("tokenizer.ggml.token_type")?),
            None => None,
        };
        let merges = match md.get("tokenizer.ggml.merges") {
            Some(_) => Some(md.get_string_array("tokenizer.ggml.merges")?),
            None => None,
        };
        Ok(Self {
            tokens,
            token_types,
            merges,
            bos_token_id: md.opt_u32("tokenizer.ggml.bos_token_id")?,
            eos_token_id: md.opt_u32("tokenizer.ggml.eos_token_id")?,
            padding_token_id: md.opt_u32("tokenizer.ggml.padding_token_id")?,
        })
    }
}

impl GgufFile {
    /// Architecture parameters from this file's own metadata.
    pub fn model_config(&self) -> Result<ModelConfig> {
        ModelConfig::from_file(self)
    }

    /// Embedded tokenizer data, when the file carries a tokenizer.
    pub fn tokenizer(&self) -> Result<TokenizerInfo> {
        TokenizerInfo::from_file(self)
    }
}
