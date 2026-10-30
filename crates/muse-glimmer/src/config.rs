//! [`GlimmerConfig`]: the architecture parameters of a Muse Glimmer
//! model, parsed from GGUF metadata.
//!
//! Every value comes from the file, read through the [`Metadata`] seam
//! under the file's own `muse-glimmer` prefixed keys (plus the `general`
//! block). Nothing is defaulted silently: a missing required key is a
//! [`GlimmerError::MissingMetadata`] naming the key, because a config
//! with invented numbers is exactly how the original engine ended up
//! running models it had never actually described.
//!
//! Measured reference values (Muse-Glimmer-30B, census 2026-10-29):
//! 52 layers, hidden 6656, FFN 19968, 32 Q heads, 2 KV heads, head dim
//! 128, RoPE base 500000, RMS eps 1e-5, sliding window 2048, logit
//! softcap 20.0, logit scale 0.19611613, vocab 202048, context 131072.

use crate::error::{GlimmerError, Result};
use crate::layer::{derive_layer_pattern, LayerKind, PatternSource, SWA_PATTERN_KEY};
use crate::metadata::Metadata;

/// The one architecture string this module serves.
pub const ARCHITECTURE: &str = "muse-glimmer";

// Metadata keys, spelled exactly as they appear in the real file.
pub(crate) const KEY_ARCHITECTURE: &str = "general.architecture";
pub(crate) const KEY_NAME: &str = "general.name";
pub(crate) const KEY_BLOCK_COUNT: &str = "muse-glimmer.block_count";
pub(crate) const KEY_CONTEXT_LENGTH: &str = "muse-glimmer.context_length";
pub(crate) const KEY_HIDDEN: &str = "muse-glimmer.embedding_length";
pub(crate) const KEY_FFN_HIDDEN: &str = "muse-glimmer.feed_forward_length";
pub(crate) const KEY_Q_HEADS: &str = "muse-glimmer.attention.head_count";
pub(crate) const KEY_KV_HEADS: &str = "muse-glimmer.attention.head_count_kv";
pub(crate) const KEY_KEY_LENGTH: &str = "muse-glimmer.attention.key_length";
pub(crate) const KEY_VALUE_LENGTH: &str = "muse-glimmer.attention.value_length";
pub(crate) const KEY_ROPE_FREQ_BASE: &str = "muse-glimmer.rope.freq_base";
pub(crate) const KEY_ROPE_FREQ_BASE_SWA: &str = "muse-glimmer.rope.freq_base_swa";
pub(crate) const KEY_RMS_EPS: &str = "muse-glimmer.attention.layer_norm_rms_epsilon";
pub(crate) const KEY_SLIDING_WINDOW: &str = "muse-glimmer.attention.sliding_window";
pub(crate) const KEY_LOGIT_SOFTCAP: &str = "muse-glimmer.final_logit_softcapping";
pub(crate) const KEY_LOGIT_SCALE: &str = "muse-glimmer.logit_scale";
pub(crate) const KEY_VOCAB_TOKENS: &str = "tokenizer.ggml.tokens";

/// Architecture parameters for one Muse Glimmer model, all read from
/// its GGUF metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct GlimmerConfig {
    /// Model name from `general.name`, when the file carries one.
    pub name: Option<String>,
    /// Number of transformer layers.
    pub block_count: u32,
    /// Maximum context length the model was built for.
    pub context_length: u32,
    /// Hidden size (`embedding_length`).
    pub hidden: u32,
    /// Feed-forward intermediate size.
    pub ffn_hidden: u32,
    /// Query head count.
    pub q_heads: u32,
    /// Key/value head count (grouped-query attention when below
    /// `q_heads`; Glimmer ships 2, a 16 to 1 grouping).
    pub kv_heads: u32,
    /// Per-head key dimension.
    pub key_length: u32,
    /// Per-head value dimension.
    pub value_length: u32,
    /// RoPE frequency base.
    pub rope_freq_base: f32,
    /// RoPE base for the sliding-window layers, the only layers
    /// llama.cpp rotates for this arch (global layers are NoPE;
    /// src/models/muse-glimmer.cpp: "RoPE runs on the SWA layers,
    /// NoPE on full ones"). llama.cpp resolves it from the optional
    /// `rope.freq_base_swa` key and falls back to the train base
    /// when the file does not carry it (src/llama-model.cpp:
    /// `rope_freq_base_train_swa`). The real file carries no such
    /// key, so here it equals `rope_freq_base`, 500000.
    pub rope_freq_base_swa: f32,
    /// RMSNorm epsilon, from the file (1e-5 on the real model; the
    /// original engine hardcoded 1e-6, one of its silent wrong numbers).
    pub rms_eps: f32,
    /// Sliding-window size in tokens for sliding-window layers.
    pub sliding_window: u32,
    /// Final logit softcap cap, applied as `cap * tanh(z / cap)`
    /// AFTER the logit scale (src/models/muse-glimmer.cpp: scale,
    /// then `build_softcap`).
    pub logit_softcap: f32,
    /// Logit scale applied to the lm_head output BEFORE the
    /// softcap (src/models/muse-glimmer.cpp: `ggml_scale` then
    /// `build_softcap`).
    pub logit_scale: f32,
    /// Vocabulary size, from the tokenizer token array length.
    pub vocab_size: u32,
    /// Per-layer attention kinds, in layer order.
    pub layers: Vec<LayerKind>,
    /// Whether `layers` came from the file or from the measured fallback.
    pub pattern_source: PatternSource,
}

impl GlimmerConfig {
    /// Parse a config from GGUF metadata. Fails loud on the first
    /// missing or impossible value; see the module docs.
    pub fn from_metadata<M: Metadata>(meta: &M) -> Result<Self> {
        let arch = meta
            .get_string(KEY_ARCHITECTURE)
            .ok_or(GlimmerError::MissingMetadata {
                key: KEY_ARCHITECTURE,
            })?;
        if arch != ARCHITECTURE {
            return Err(GlimmerError::WrongArchitecture { found: arch });
        }

        let block_count = required_u32(meta, KEY_BLOCK_COUNT)?;
        let q_heads = required_u32(meta, KEY_Q_HEADS)?;
        let kv_heads = required_u32(meta, KEY_KV_HEADS)?;
        if block_count == 0 {
            return Err(invalid(KEY_BLOCK_COUNT, "block_count is zero"));
        }
        if q_heads == 0 || kv_heads == 0 {
            return Err(invalid(KEY_KV_HEADS, "head counts must be at least 1"));
        }
        if q_heads % kv_heads != 0 {
            return Err(invalid(
                KEY_KV_HEADS,
                "query head count is not a whole multiple of the KV head count, so grouped-query grouping is impossible",
            ));
        }
        let sliding_window = required_u32(meta, KEY_SLIDING_WINDOW)?;
        if sliding_window == 0 {
            return Err(invalid(KEY_SLIDING_WINDOW, "sliding window is zero"));
        }

        let pattern_text = meta.get_string(SWA_PATTERN_KEY);
        let (layers, pattern_source) = derive_layer_pattern(block_count, pattern_text.as_deref())?;

        let vocab = meta
            .get_array_len(KEY_VOCAB_TOKENS)
            .ok_or(GlimmerError::MissingMetadata {
                key: KEY_VOCAB_TOKENS,
            })?;

        let rope_freq_base = required_f32(meta, KEY_ROPE_FREQ_BASE)?;

        Ok(GlimmerConfig {
            name: meta.get_string(KEY_NAME),
            block_count,
            context_length: required_u32(meta, KEY_CONTEXT_LENGTH)?,
            hidden: required_u32(meta, KEY_HIDDEN)?,
            ffn_hidden: required_u32(meta, KEY_FFN_HIDDEN)?,
            q_heads,
            kv_heads,
            key_length: required_u32(meta, KEY_KEY_LENGTH)?,
            value_length: required_u32(meta, KEY_VALUE_LENGTH)?,
            rope_freq_base,
            rope_freq_base_swa: meta
                .get_f32(KEY_ROPE_FREQ_BASE_SWA)
                .unwrap_or(rope_freq_base),
            rms_eps: required_f32(meta, KEY_RMS_EPS)?,
            sliding_window,
            logit_softcap: required_f32(meta, KEY_LOGIT_SOFTCAP)?,
            logit_scale: required_f32(meta, KEY_LOGIT_SCALE)?,
            vocab_size: u32::try_from(vocab)
                .map_err(|_| invalid(KEY_VOCAB_TOKENS, "vocabulary size does not fit in u32"))?,
            layers,
            pattern_source,
        })
    }

    /// The kind of layer `index`, or `None` when out of range.
    pub fn layer_kind(&self, index: usize) -> Option<LayerKind> {
        self.layers.get(index).copied()
    }

    /// Indices of the global-attention layers, in order.
    pub fn global_layer_indices(&self) -> Vec<usize> {
        self.layers
            .iter()
            .enumerate()
            .filter(|(_, k)| **k == LayerKind::Global)
            .map(|(i, _)| i)
            .collect()
    }

    /// Number of global-attention layers.
    pub fn global_layer_count(&self) -> usize {
        self.layers
            .iter()
            .filter(|k| **k == LayerKind::Global)
            .count()
    }

    /// Number of sliding-window layers.
    pub fn sliding_layer_count(&self) -> usize {
        self.block_count as usize - self.global_layer_count()
    }

    /// Width of the query projection per position: Q heads times head dim.
    pub fn q_width(&self) -> u32 {
        self.q_heads * self.key_length
    }

    /// Width of the key projection per position: KV heads times head dim.
    pub fn k_width(&self) -> u32 {
        self.kv_heads * self.key_length
    }

    /// Width of the value projection per position.
    pub fn v_width(&self) -> u32 {
        self.kv_heads * self.value_length
    }

    /// Grouped-query group size: how many Q heads share one KV head.
    pub fn gqa_group(&self) -> u32 {
        self.q_heads / self.kv_heads
    }
}

fn required_u32<M: Metadata>(meta: &M, key: &'static str) -> Result<u32> {
    meta.get_u32(key)
        .ok_or(GlimmerError::MissingMetadata { key })
}

fn required_f32<M: Metadata>(meta: &M, key: &'static str) -> Result<f32> {
    meta.get_f32(key)
        .ok_or(GlimmerError::MissingMetadata { key })
}

fn invalid(key: &'static str, detail: &str) -> GlimmerError {
    GlimmerError::InvalidMetadata {
        key,
        detail: detail.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::MapMetadata;
    use crate::testkit::{real_glimmer_metadata, tiny_metadata};

    #[test]
    fn real_file_metadata_parses_to_measured_values() {
        let meta = real_glimmer_metadata();
        let cfg = GlimmerConfig::from_metadata(&meta).expect("real metadata parses");
        assert_eq!(cfg.name.as_deref(), Some("Muse Glimmer Hf"));
        assert_eq!(cfg.block_count, 52);
        assert_eq!(cfg.context_length, 131072);
        assert_eq!(cfg.hidden, 6656);
        assert_eq!(cfg.ffn_hidden, 19968);
        assert_eq!(cfg.q_heads, 32);
        assert_eq!(cfg.kv_heads, 2);
        assert_eq!(cfg.gqa_group(), 16);
        assert_eq!(cfg.key_length, 128);
        assert_eq!(cfg.value_length, 128);
        assert_eq!(cfg.q_width(), 4096);
        assert_eq!(cfg.k_width(), 256);
        assert!((cfg.rope_freq_base - 500_000.0).abs() < 0.01);
        // The real file carries no rope.freq_base_swa key, so the
        // SWA base falls back to the train base, as in llama.cpp.
        assert!((cfg.rope_freq_base_swa - 500_000.0).abs() < 0.01);
        assert!((cfg.rms_eps - 1e-5).abs() < 1e-9);
        assert_eq!(cfg.sliding_window, 2048);
        assert!((cfg.logit_softcap - 20.0).abs() < f32::EPSILON);
        assert!((cfg.logit_scale - 0.196_116_13).abs() < 1e-6);
        assert_eq!(cfg.vocab_size, 202_048);
        assert_eq!(cfg.pattern_source, PatternSource::MeasuredFallback);
        assert_eq!(
            cfg.global_layer_indices(),
            vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51]
        );
        assert_eq!(cfg.global_layer_count(), 13);
        assert_eq!(cfg.sliding_layer_count(), 39);
        assert_eq!(cfg.layer_kind(3), Some(LayerKind::Global));
        assert_eq!(cfg.layer_kind(4), Some(LayerKind::SlidingWindow));
        assert_eq!(cfg.layer_kind(52), None);
    }

    #[test]
    fn tiny_synthetic_config_parses() {
        let meta = tiny_metadata();
        let cfg = GlimmerConfig::from_metadata(&meta).expect("tiny metadata parses");
        assert_eq!(cfg.block_count, 2);
        assert_eq!(cfg.hidden, 32);
        assert_eq!(cfg.ffn_hidden, 64);
        assert_eq!(cfg.q_heads, 4);
        assert_eq!(cfg.kv_heads, 2);
        assert_eq!(cfg.layers.len(), 2);
        // Two layers contain no every-fourth index, so both slide.
        assert_eq!(cfg.global_layer_count(), 0);
        assert_eq!(cfg.sliding_layer_count(), 2);
        assert_eq!(cfg.vocab_size, 128);
    }

    #[test]
    fn explicit_swa_rope_base_wins_over_the_train_base() {
        let mut meta = tiny_metadata();
        meta.insert_f32(KEY_ROPE_FREQ_BASE_SWA, 1_000.0);
        let cfg = GlimmerConfig::from_metadata(&meta).expect("tiny metadata parses");
        assert!((cfg.rope_freq_base_swa - 1_000.0).abs() < f32::EPSILON);
        assert!((cfg.rope_freq_base - 10_000.0).abs() < f32::EPSILON);
    }

    #[test]
    fn missing_key_is_a_named_error() {
        let mut meta = real_glimmer_metadata();
        meta.remove(KEY_BLOCK_COUNT);
        let err = GlimmerConfig::from_metadata(&meta).expect_err("missing key must fail");
        match err {
            GlimmerError::MissingMetadata { key } => assert_eq!(key, KEY_BLOCK_COUNT),
            other => panic!("expected MissingMetadata, got {other:?}"),
        }
    }

    #[test]
    fn missing_vocab_is_a_named_error() {
        let mut meta = tiny_metadata();
        meta.remove(KEY_VOCAB_TOKENS);
        let err = GlimmerConfig::from_metadata(&meta).expect_err("missing vocab must fail");
        assert!(matches!(err, GlimmerError::MissingMetadata { .. }));
    }

    #[test]
    fn wrong_architecture_is_refused() {
        let mut meta = tiny_metadata();
        meta.insert_string(KEY_ARCHITECTURE, "llama");
        let err = GlimmerConfig::from_metadata(&meta).expect_err("wrong arch must fail");
        match err {
            GlimmerError::WrongArchitecture { found } => assert_eq!(found, "llama"),
            other => panic!("expected WrongArchitecture, got {other:?}"),
        }
    }

    #[test]
    fn impossible_grouping_is_invalid_metadata() {
        let mut meta = tiny_metadata();
        meta.insert_u32(KEY_Q_HEADS, 5);
        meta.insert_u32(KEY_KV_HEADS, 2);
        let err = GlimmerConfig::from_metadata(&meta).expect_err("5 over 2 cannot group");
        assert!(matches!(err, GlimmerError::InvalidMetadata { .. }));
    }

    #[test]
    fn map_metadata_remove_clears_all_kinds() {
        let mut meta = MapMetadata::new();
        meta.insert_u32("k", 7);
        assert_eq!(meta.get_u32("k"), Some(7));
        meta.remove("k");
        assert_eq!(meta.get_u32("k"), None);
    }
}
