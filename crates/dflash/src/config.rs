//! [`DflashConfig`]: the architecture parameters of a DFlash draft
//! model, parsed from GGUF metadata.
//!
//! Every value comes from the file, read through the [`Metadata`]
//! seam under the file's own `dflash` prefixed keys (plus the
//! `general` and `tokenizer` blocks). Nothing is defaulted silently:
//! a missing required key is a [`DflashError::MissingMetadata`]
//! naming the key. The two optional keys carry defined meanings in
//! the reference implementation and are handled explicitly:
//! `dflash.attention.causal` absent means the draft's block
//! attention is non-causal (llama.cpp's graph reads the flag with a
//! false default, src/models/dflash.cpp), and `dflash.hidden_act`
//! absent means silu; any other activation value is refused as an
//! unsupported feature rather than approximated.
//!
//! Measured reference values (the real draft file,
//! `dflash-Muse-Glimmer-30B-Q4_K_M.gguf`, census 2026-10-30 read
//! with this workspace's `gguf` crate): 5 layers, hidden 6656,
//! FFN 19968, 32 Q heads over 8 KV heads, key and value length 128,
//! RoPE base 500000, RMS eps 1e-5, block size 16, target layers
//! [2, 14, 26, 38, 50], sliding window 2048 with an all-true
//! sliding pattern, mask token 201818, vocab 202048, context 131072.

use crate::error::{DflashError, Result};
use crate::metadata::Metadata;

/// The one architecture string this module serves.
pub const ARCHITECTURE: &str = "dflash";

// Metadata keys, spelled exactly as they appear in the real file.
pub(crate) const KEY_ARCHITECTURE: &str = "general.architecture";
pub(crate) const KEY_NAME: &str = "general.name";
pub(crate) const KEY_BLOCK_COUNT: &str = "dflash.block_count";
pub(crate) const KEY_CONTEXT_LENGTH: &str = "dflash.context_length";
pub(crate) const KEY_HIDDEN: &str = "dflash.embedding_length";
pub(crate) const KEY_FFN_HIDDEN: &str = "dflash.feed_forward_length";
pub(crate) const KEY_Q_HEADS: &str = "dflash.attention.head_count";
pub(crate) const KEY_KV_HEADS: &str = "dflash.attention.head_count_kv";
pub(crate) const KEY_KEY_LENGTH: &str = "dflash.attention.key_length";
pub(crate) const KEY_VALUE_LENGTH: &str = "dflash.attention.value_length";
pub(crate) const KEY_ROPE_FREQ_BASE: &str = "dflash.rope.freq_base";
pub(crate) const KEY_RMS_EPS: &str = "dflash.attention.layer_norm_rms_epsilon";
pub(crate) const KEY_BLOCK_SIZE: &str = "dflash.block_size";
pub(crate) const KEY_TARGET_LAYERS: &str = "dflash.target_layers";
pub(crate) const KEY_SLIDING_WINDOW: &str = "dflash.attention.sliding_window";
pub(crate) const KEY_SLIDING_PATTERN: &str = "dflash.attention.sliding_window_pattern";
pub(crate) const KEY_CAUSAL: &str = "dflash.attention.causal";
pub(crate) const KEY_HIDDEN_ACT: &str = "dflash.hidden_act";
pub(crate) const KEY_MASK_TOKEN: &str = "tokenizer.ggml.mask_token_id";
pub(crate) const KEY_VOCAB_TOKENS: &str = "tokenizer.ggml.tokens";

/// Architecture parameters for one DFlash draft model, all read
/// from its GGUF metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct DflashConfig {
    /// Model name from `general.name`, when the file carries one.
    pub name: Option<String>,
    /// Number of draft transformer layers.
    pub block_count: u32,
    /// Maximum context length the draft was built for.
    pub context_length: u32,
    /// Draft hidden size (`embedding_length`).
    pub hidden: u32,
    /// Feed-forward intermediate size.
    pub ffn_hidden: u32,
    /// Query head count.
    pub q_heads: u32,
    /// Key/value head count.
    pub kv_heads: u32,
    /// Per-head key dimension.
    pub key_length: u32,
    /// Per-head value dimension.
    pub value_length: u32,
    /// RoPE frequency base. The draft has a single rope: llama.cpp
    /// uses this base for every layer, sliding or not.
    pub rope_freq_base: f32,
    /// RMSNorm epsilon, from the file.
    pub rms_eps: f32,
    /// Noise block size: the anchor plus at most `block_size - 1`
    /// drafted positions.
    pub block_size: u32,
    /// Target layer indices whose INPUT residual rows condition the
    /// draft, in concatenation order (`dflash.target_layers`;
    /// llama.cpp throws when this key is absent).
    pub target_layers: Vec<u32>,
    /// Sliding-window size in tokens, shared by every sliding layer.
    pub sliding_window: u32,
    /// Per-layer sliding flags, in layer order: `true` means the
    /// layer's attention reads through the sliding window.
    pub sliding_pattern: Vec<bool>,
    /// Whether block attention is causal. Absent key means false
    /// (non-causal block attention), the reference default.
    pub causal_attention: bool,
    /// The mask token id that fills the noise block after the
    /// anchor (`tokenizer.ggml.mask_token_id`).
    pub mask_token_id: u32,
    /// Vocabulary size, from the tokenizer token array length.
    pub vocab_size: u32,
}

impl DflashConfig {
    /// Parse a config from GGUF metadata. Fails loud on the first
    /// missing or impossible value; see the module docs.
    pub fn from_metadata<M: Metadata>(meta: &M) -> Result<Self> {
        let arch = meta
            .get_string(KEY_ARCHITECTURE)
            .ok_or(DflashError::MissingMetadata {
                key: KEY_ARCHITECTURE.to_string(),
            })?;
        if arch != ARCHITECTURE {
            return Err(DflashError::WrongArchitecture { found: arch });
        }

        if let Some(act) = meta.get_string(KEY_HIDDEN_ACT) {
            if act != "silu" {
                return Err(DflashError::UnsupportedFeature {
                    name: KEY_HIDDEN_ACT.to_string(),
                    detail: format!(
                        "hidden activation \"{act}\" is not implemented; this module implements silu (the reference graph's SwiGLU path)"
                    ),
                });
            }
        }

        let block_count = required_u32(meta, KEY_BLOCK_COUNT)?;
        if block_count == 0 {
            return Err(invalid(KEY_BLOCK_COUNT, "block_count is zero"));
        }
        let q_heads = required_u32(meta, KEY_Q_HEADS)?;
        let kv_heads = required_u32(meta, KEY_KV_HEADS)?;
        if q_heads == 0 || kv_heads == 0 {
            return Err(invalid(KEY_KV_HEADS, "head counts must be at least 1"));
        }
        if q_heads % kv_heads != 0 {
            return Err(invalid(
                KEY_KV_HEADS,
                "query head count is not a whole multiple of the KV head count, so grouped-query grouping is impossible",
            ));
        }
        let key_length = required_u32(meta, KEY_KEY_LENGTH)?;
        if key_length < 2 || key_length % 2 != 0 {
            return Err(invalid(
                KEY_KEY_LENGTH,
                "key length must be even and at least 2 for RoPE pairing",
            ));
        }
        let value_length = required_u32(meta, KEY_VALUE_LENGTH)?;
        if value_length == 0 {
            return Err(invalid(KEY_VALUE_LENGTH, "value length is zero"));
        }
        let block_size = required_u32(meta, KEY_BLOCK_SIZE)?;
        if block_size == 0 {
            return Err(invalid(KEY_BLOCK_SIZE, "block_size is zero"));
        }

        let target_layers =
            meta.get_u32_array(KEY_TARGET_LAYERS)
                .ok_or(DflashError::MissingMetadata {
                    key: KEY_TARGET_LAYERS.to_string(),
                })?;
        if target_layers.is_empty() {
            return Err(invalid(
                KEY_TARGET_LAYERS,
                "target layer list is empty; the draft would have no conditioning input",
            ));
        }
        for (i, &layer) in target_layers.iter().enumerate() {
            if target_layers[..i].contains(&layer) {
                return Err(invalid(
                    KEY_TARGET_LAYERS,
                    "target layer list contains a duplicate; each named layer is concatenated once",
                ));
            }
        }

        let sliding_pattern =
            meta.get_bool_array(KEY_SLIDING_PATTERN)
                .ok_or(DflashError::MissingMetadata {
                    key: KEY_SLIDING_PATTERN.to_string(),
                })?;
        if sliding_pattern.len() != block_count as usize {
            return Err(invalid(
                KEY_SLIDING_PATTERN,
                "sliding window pattern length does not equal the block count",
            ));
        }
        let sliding_window = required_u32(meta, KEY_SLIDING_WINDOW)?;
        if sliding_pattern.iter().any(|s| *s) && sliding_window == 0 {
            return Err(invalid(
                KEY_SLIDING_WINDOW,
                "pattern marks sliding layers but the sliding window is zero",
            ));
        }

        let causal_attention = match meta.get_string(KEY_CAUSAL) {
            Some(text) => match text.as_str() {
                "true" => true,
                "false" => false,
                other => {
                    return Err(invalid(
                        KEY_CAUSAL,
                        &format!(
                            "causal flag is the string \"{other}\", expected \"true\" or \"false\""
                        ),
                    ))
                }
            },
            None => meta.get_bool(KEY_CAUSAL).unwrap_or(false),
        };

        let rope_freq_base = required_f32(meta, KEY_ROPE_FREQ_BASE)?;
        if !rope_freq_base.is_finite() || rope_freq_base <= 1.0 {
            return Err(invalid(
                KEY_ROPE_FREQ_BASE,
                "rope frequency base must be finite and greater than 1",
            ));
        }
        let rms_eps = required_f32(meta, KEY_RMS_EPS)?;
        if !rms_eps.is_finite() || rms_eps <= 0.0 {
            return Err(invalid(
                KEY_RMS_EPS,
                "rms epsilon must be finite and greater than 0",
            ));
        }

        let vocab = meta
            .get_array_len(KEY_VOCAB_TOKENS)
            .ok_or(DflashError::MissingMetadata {
                key: KEY_VOCAB_TOKENS.to_string(),
            })?;
        let vocab_size = u32::try_from(vocab)
            .map_err(|_| invalid(KEY_VOCAB_TOKENS, "vocabulary size does not fit in u32"))?;
        if vocab_size == 0 {
            return Err(invalid(KEY_VOCAB_TOKENS, "vocabulary is empty"));
        }
        let mask_token_id = required_u32(meta, KEY_MASK_TOKEN)?;
        if mask_token_id >= vocab_size {
            return Err(invalid(
                KEY_MASK_TOKEN,
                "mask token id is outside the vocabulary",
            ));
        }

        Ok(DflashConfig {
            name: meta.get_string(KEY_NAME),
            block_count,
            context_length: required_u32(meta, KEY_CONTEXT_LENGTH)?,
            hidden: required_u32(meta, KEY_HIDDEN)?,
            ffn_hidden: required_u32(meta, KEY_FFN_HIDDEN)?,
            q_heads,
            kv_heads,
            key_length,
            value_length,
            rope_freq_base,
            rms_eps,
            block_size,
            target_layers,
            sliding_window,
            sliding_pattern,
            causal_attention,
            mask_token_id,
            vocab_size,
        })
    }

    /// Whether draft layer `index` attends through the sliding
    /// window, or `None` when out of range.
    pub fn is_sliding(&self, index: usize) -> Option<bool> {
        self.sliding_pattern.get(index).copied()
    }

    /// Width of the query projection per position.
    pub fn q_width(&self) -> u32 {
        self.q_heads * self.key_length
    }

    /// Width of the key projection per position.
    pub fn k_width(&self) -> u32 {
        self.kv_heads * self.key_length
    }

    /// Width of the value projection per position.
    pub fn v_width(&self) -> u32 {
        self.kv_heads * self.value_length
    }

    /// Width of the attention output per position (the output
    /// projection's input width).
    pub fn attn_out_width(&self) -> u32 {
        self.q_heads * self.value_length
    }

    /// Grouped-query group size: how many Q heads share one KV head.
    pub fn gqa_group(&self) -> u32 {
        self.q_heads / self.kv_heads
    }

    /// The largest number of draft tokens one block can propose:
    /// the block's first position is the anchor, which is not a
    /// draft.
    pub fn max_drafts(&self) -> u32 {
        self.block_size - 1
    }
}

fn required_u32<M: Metadata>(meta: &M, key: &'static str) -> Result<u32> {
    meta.get_u32(key).ok_or(DflashError::MissingMetadata {
        key: key.to_string(),
    })
}

fn required_f32<M: Metadata>(meta: &M, key: &'static str) -> Result<f32> {
    meta.get_f32(key).ok_or(DflashError::MissingMetadata {
        key: key.to_string(),
    })
}

fn invalid(key: &'static str, detail: &str) -> DflashError {
    DflashError::InvalidMetadata {
        key: key.to_string(),
        detail: detail.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{real_draft_metadata, tiny_metadata};

    #[test]
    fn real_file_metadata_parses_to_measured_values() {
        let cfg = DflashConfig::from_metadata(&real_draft_metadata()).expect("real metadata");
        assert_eq!(cfg.name.as_deref(), Some("Hf_Museglimmer"));
        assert_eq!(cfg.block_count, 5);
        assert_eq!(cfg.context_length, 131_072);
        assert_eq!(cfg.hidden, 6656);
        assert_eq!(cfg.ffn_hidden, 19_968);
        assert_eq!(cfg.q_heads, 32);
        assert_eq!(cfg.kv_heads, 8);
        assert_eq!(cfg.gqa_group(), 4);
        assert_eq!(cfg.key_length, 128);
        assert_eq!(cfg.value_length, 128);
        assert_eq!(cfg.q_width(), 4096);
        assert_eq!(cfg.k_width(), 1024);
        assert!((cfg.rope_freq_base - 500_000.0).abs() < 0.01);
        assert!((cfg.rms_eps - 1e-5).abs() < 1e-9);
        assert_eq!(cfg.block_size, 16);
        assert_eq!(cfg.max_drafts(), 15);
        assert_eq!(cfg.target_layers, vec![2, 14, 26, 38, 50]);
        assert_eq!(cfg.sliding_window, 2048);
        assert_eq!(cfg.sliding_pattern, vec![true; 5]);
        assert!(!cfg.causal_attention);
        assert_eq!(cfg.mask_token_id, 201_818);
        assert_eq!(cfg.vocab_size, 202_048);
    }

    #[test]
    fn tiny_synthetic_config_parses() {
        let cfg = DflashConfig::from_metadata(&tiny_metadata()).expect("tiny metadata");
        assert_eq!(cfg.block_count, 2);
        assert_eq!(cfg.target_layers, vec![0, 1]);
        assert_eq!(cfg.sliding_pattern, vec![true, false]);
        assert_eq!(cfg.is_sliding(0), Some(true));
        assert_eq!(cfg.is_sliding(1), Some(false));
        assert_eq!(cfg.is_sliding(2), None);
    }

    #[test]
    fn missing_target_layers_is_a_named_error() {
        // llama.cpp throws on this exact absence; the config must
        // name the key, never invent a conditioning layer.
        let mut meta = real_draft_metadata();
        meta.remove(KEY_TARGET_LAYERS);
        let err = DflashConfig::from_metadata(&meta).expect_err("missing key must fail");
        match err {
            DflashError::MissingMetadata { key } => assert_eq!(key, KEY_TARGET_LAYERS),
            other => panic!("expected MissingMetadata, got {other:?}"),
        }
    }

    #[test]
    fn wrong_architecture_is_refused() {
        let mut meta = tiny_metadata();
        meta.insert_string(KEY_ARCHITECTURE, "muse-glimmer");
        let err = DflashConfig::from_metadata(&meta).expect_err("wrong arch must fail");
        match err {
            DflashError::WrongArchitecture { found } => assert_eq!(found, "muse-glimmer"),
            other => panic!("expected WrongArchitecture, got {other:?}"),
        }
    }

    #[test]
    fn pattern_length_mismatch_is_invalid() {
        let mut meta = tiny_metadata();
        meta.insert_bool_array(KEY_SLIDING_PATTERN, &[true]);
        let err = DflashConfig::from_metadata(&meta).expect_err("short pattern must fail");
        assert!(matches!(err, DflashError::InvalidMetadata { .. }));
    }

    #[test]
    fn duplicate_target_layer_is_invalid() {
        let mut meta = tiny_metadata();
        meta.insert_u32_array(KEY_TARGET_LAYERS, &[1, 1]);
        let err = DflashConfig::from_metadata(&meta).expect_err("duplicate must fail");
        assert!(matches!(err, DflashError::InvalidMetadata { .. }));
    }

    #[test]
    fn causal_flag_reads_string_and_bool_forms() {
        let mut meta = tiny_metadata();
        meta.insert_string(KEY_CAUSAL, "true");
        let cfg = DflashConfig::from_metadata(&meta).expect("string form parses");
        assert!(cfg.causal_attention);
        let mut meta = tiny_metadata();
        meta.insert_bool(KEY_CAUSAL, true);
        let cfg = DflashConfig::from_metadata(&meta).expect("bool form parses");
        assert!(cfg.causal_attention);
        let mut meta = tiny_metadata();
        meta.insert_string(KEY_CAUSAL, "sometimes");
        let err = DflashConfig::from_metadata(&meta).expect_err("junk flag must fail");
        assert!(matches!(err, DflashError::InvalidMetadata { .. }));
    }

    #[test]
    fn non_silu_activation_is_an_unsupported_feature() {
        let mut meta = tiny_metadata();
        meta.insert_string(KEY_HIDDEN_ACT, "gelu");
        let err = DflashConfig::from_metadata(&meta).expect_err("gelu must refuse");
        assert!(matches!(err, DflashError::UnsupportedFeature { .. }));
    }
}
