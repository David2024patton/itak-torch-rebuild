//! [`LlamaConfig`]: the architecture parameters of a llama-family or
//! qwen2-family model, read from GGUF metadata.
//!
//! Every value comes from the file, under the file's own
//! architecture-prefixed keys (`llama.*` or `qwen2.*`, plus the
//! `general` block). The bulk read goes through the gguf crate's
//! [`gguf::ModelConfig`]; this module then enforces what this dense
//! arch module needs on top of it. Nothing is defaulted silently: a
//! missing required key is a [`LlamaError::MissingMetadata`] naming
//! the key, because a config with invented numbers is exactly how the
//! original engine ended up running models it had never actually
//! described (it fabricated vocab 32000 for five of the eight audit
//! test models).
//!
//! The two fields where absence has a defined meaning are handled
//! explicitly, not defaulted: the KV head count falls back to the
//! query head count inside the gguf crate because that is the GGUF
//! format's own convention for full multi-head attention, and
//! `<arch>.rope.dimension_count` falls back to the head dimension
//! because both families rotate the full head when the key is absent
//! (TinyLlama carries the key and it equals its head dim; Qwen2.5-1.5B
//! omits it and rotates its full 128).
//!
//! Measured reference values (census 2026-10-29): TinyLlama-1.1B is
//! `llama`, 22 layers, hidden 2048, FFN 5632, 32 Q heads over 4 KV
//! heads, head dim 64, RoPE base 10000, RMS eps 1e-5, vocab 32000,
//! context 2048. Qwen2.5-1.5B is `qwen2`, 28 layers, hidden 1536, FFN
//! 8960, 12 Q heads over 2 KV heads, head dim 128, RoPE base
//! 1000000, RMS eps 1e-6, vocab 151936, context 32768.

use gguf::{GgufFile, MetadataValue};

use crate::error::{LlamaError, Result};

/// The architecture strings this module serves.
pub const ARCHITECTURES: [&str; 2] = ["llama", "qwen2"];

/// Which channels RoPE pairs when it rotates a head vector.
///
/// Pairing is a per-architecture convention, not a property of the
/// math. llama.cpp's authority is `llama_model_rope_type` in
/// src/llama-model.cpp (the build that generated this project's
/// golden fixtures): `LLM_ARCH_LLAMA` returns
/// `LLAMA_ROPE_TYPE_NORM` (adjacent pairs) and `LLM_ARCH_QWEN2`
/// returns `LLAMA_ROPE_TYPE_NEOX` (half-split pairs).
/// (`LLM_ARCH_MUSE_GLIMMER` also returns NORM; that architecture
/// is served by the muse-glimmer crate, whose backend calls
/// `torch_math::rope_interleaved` directly.)
///
/// The pairings rotate different channels at the same rates, so a
/// file converted for one is garbage under the other. That is not
/// theory here: commit 13643e0 flipped the one shared pairing
/// function to NEOX to fix Qwen2.5 and regressed TinyLlama (arch
/// llama) from 6/15 to 0/15 token-identical in the same edit,
/// because arch-llama called the shared function for both
/// architectures. This enum and [`LlamaConfig::rope_pairing`] exist
/// so the choice is made once, by name, from the file's own
/// architecture string.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopePairing {
    /// Adjacent pairs `(2i, 2i + 1)`: ggml's NORM pairing, applied
    /// by `torch_math::rope_interleaved_partial`. The llama family.
    Interleaved,
    /// Half-split pairs `(i, i + rotary_dim / 2)`: ggml's NEOX
    /// pairing, applied by `torch_math::rope_partial`. The qwen2
    /// family.
    HalfSplit,
}

// Metadata key suffixes, appended to the file's own architecture
// string. Spelled exactly as the real files carry them.
pub(crate) const SUFFIX_BLOCK_COUNT: &str = "block_count";
pub(crate) const SUFFIX_CONTEXT_LENGTH: &str = "context_length";
pub(crate) const SUFFIX_HIDDEN: &str = "embedding_length";
pub(crate) const SUFFIX_FFN_HIDDEN: &str = "feed_forward_length";
pub(crate) const SUFFIX_Q_HEADS: &str = "attention.head_count";
pub(crate) const SUFFIX_KV_HEADS: &str = "attention.head_count_kv";
pub(crate) const SUFFIX_ROPE_FREQ_BASE: &str = "rope.freq_base";
pub(crate) const SUFFIX_ROPE_DIMENSION_COUNT: &str = "rope.dimension_count";
pub(crate) const SUFFIX_RMS_EPS: &str = "attention.layer_norm_rms_epsilon";
pub(crate) const KEY_NAME: &str = "general.name";
pub(crate) const KEY_VOCAB_TOKENS: &str = "tokenizer.ggml.tokens";

/// Architecture parameters for one llama-family or qwen2-family
/// model, all read from its GGUF metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct LlamaConfig {
    /// The file's own architecture string: `llama` or `qwen2`.
    pub architecture: String,
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
    /// `q_heads`; TinyLlama ships 4 under 32, Qwen2.5-1.5B ships 2
    /// under 12).
    pub kv_heads: u32,
    /// Per-head dimension, derived as `hidden / q_heads` (an identity
    /// for both families, validated to divide exactly).
    pub head_dim: u32,
    /// RoPE frequency base, from the file (10000 on TinyLlama,
    /// 1000000 on Qwen2.5).
    pub rope_freq_base: f32,
    /// How many channels of each head RoPE rotates: the file's
    /// `<arch>.rope.dimension_count` when present, else the full head
    /// dimension.
    pub rope_dim: u32,
    /// RMSNorm epsilon, from the file (1e-5 on llama, 1e-6 on qwen2;
    /// the original engine hardcoded 1e-6 for everything, one of its
    /// silent wrong numbers).
    pub rms_eps: f32,
    /// Vocabulary size, from the tokenizer token array length.
    pub vocab_size: u32,
}

impl LlamaConfig {
    /// Parse a config from a GGUF file's metadata. Fails loud on the
    /// first missing or impossible value; see the module docs.
    pub fn from_file(file: &GgufFile) -> Result<Self> {
        let mc = file.model_config()?;
        let arch = mc.architecture.clone();
        if !ARCHITECTURES.contains(&arch.as_str()) {
            return Err(LlamaError::WrongArchitecture { found: arch });
        }
        let key = |suffix: &str| format!("{arch}.{suffix}");

        // Per-layer head-count arrays belong to hybrid files, whose
        // arch modules are not this one. Refuse by name rather than
        // silently taking the array maximum the gguf crate reports.
        if mc.head_count_per_layer.is_some() || mc.head_count_kv_per_layer.is_some() {
            return Err(invalid(
                &key(SUFFIX_Q_HEADS),
                "head counts are stored per layer; this dense module serves uniform files only",
            ));
        }

        let ffn_hidden = mc.ffn_size.ok_or_else(|| LlamaError::MissingMetadata {
            key: key(SUFFIX_FFN_HIDDEN),
        })?;
        let rope_freq_base = mc.rope_base.ok_or_else(|| LlamaError::MissingMetadata {
            key: key(SUFFIX_ROPE_FREQ_BASE),
        })?;
        let rms_eps = mc.norm_eps.ok_or_else(|| LlamaError::MissingMetadata {
            key: key(SUFFIX_RMS_EPS),
        })?;
        let vocab_size = mc.vocab_size.ok_or_else(|| LlamaError::MissingMetadata {
            key: KEY_VOCAB_TOKENS.to_string(),
        })?;

        if mc.layer_count == 0 {
            return Err(invalid(&key(SUFFIX_BLOCK_COUNT), "block_count is zero"));
        }
        if mc.hidden_size == 0 {
            return Err(invalid(&key(SUFFIX_HIDDEN), "embedding_length is zero"));
        }
        if mc.context_length == 0 {
            return Err(invalid(
                &key(SUFFIX_CONTEXT_LENGTH),
                "context_length is zero",
            ));
        }
        if mc.head_count == 0 || mc.head_count_kv == 0 {
            return Err(invalid(
                &key(SUFFIX_KV_HEADS),
                "head counts must be at least 1",
            ));
        }
        if mc.head_count % mc.head_count_kv != 0 {
            return Err(invalid(
                &key(SUFFIX_KV_HEADS),
                "query head count is not a whole multiple of the KV head count, so grouped-query grouping is impossible",
            ));
        }
        if mc.hidden_size % mc.head_count != 0 {
            return Err(invalid(
                &key(SUFFIX_Q_HEADS),
                "embedding_length is not a whole multiple of the query head count, so the head dimension is not an integer",
            ));
        }
        let head_dim = mc.hidden_size / mc.head_count;
        if head_dim < 2 || head_dim % 2 != 0 {
            return Err(invalid(
                &key(SUFFIX_Q_HEADS),
                "head dimension must be even and at least 2 for RoPE pairing",
            ));
        }
        if !rope_freq_base.is_finite() || rope_freq_base <= 1.0 {
            return Err(invalid(
                &key(SUFFIX_ROPE_FREQ_BASE),
                "rope frequency base must be finite and greater than 1",
            ));
        }
        if !rms_eps.is_finite() || rms_eps <= 0.0 {
            return Err(invalid(
                &key(SUFFIX_RMS_EPS),
                "rms epsilon must be finite and greater than 0",
            ));
        }
        let vocab_size = u32::try_from(vocab_size)
            .map_err(|_| invalid(KEY_VOCAB_TOKENS, "vocabulary size does not fit in u32"))?;
        if vocab_size == 0 {
            return Err(invalid(KEY_VOCAB_TOKENS, "vocabulary is empty"));
        }

        let rope_dim = read_rope_dim(file, &arch, head_dim)?;

        Ok(LlamaConfig {
            architecture: arch,
            name: file.metadata().opt_string(KEY_NAME)?.map(str::to_string),
            block_count: mc.layer_count,
            context_length: mc.context_length,
            hidden: mc.hidden_size,
            ffn_hidden,
            q_heads: mc.head_count,
            kv_heads: mc.head_count_kv,
            head_dim,
            rope_freq_base,
            rope_dim,
            rms_eps,
            vocab_size,
        })
    }

    /// Whether this file's family carries attention biases in its
    /// tensor map. Qwen2's reference graph applies q/k/v biases on
    /// every layer, so for `qwen2` files the bias tensors are
    /// required at load; for `llama` files they are applied when
    /// present and not expected.
    pub fn attention_biases_required(&self) -> bool {
        self.architecture == "qwen2"
    }

    /// The RoPE channel pairing this file's architecture uses, from
    /// llama.cpp's per-architecture table (see [`RopePairing`] for
    /// the source citation and the regression that made this a
    /// named choice). This is the one place the mapping lives; the
    /// forward dispatches on it and never names a pairing function
    /// itself.
    ///
    /// # Errors
    ///
    /// [`LlamaError::RopePairingUnknown`] when the architecture
    /// string is not one this mapping covers. [`from_file`] already
    /// refuses any architecture outside [`ARCHITECTURES`], so this
    /// can only fire for a config assembled by hand; it refuses
    /// rather than guessing, because a guessed pairing runs
    /// silently and produces plausible garbage.
    ///
    /// [`from_file`]: LlamaConfig::from_file
    pub fn rope_pairing(&self) -> Result<RopePairing> {
        match self.architecture.as_str() {
            "llama" => Ok(RopePairing::Interleaved),
            "qwen2" => Ok(RopePairing::HalfSplit),
            other => Err(LlamaError::RopePairingUnknown {
                architecture: other.to_string(),
            }),
        }
    }

    /// Width of the query projection per position: Q heads times head dim.
    pub fn q_width(&self) -> u32 {
        self.q_heads * self.head_dim
    }

    /// Width of the key projection per position: KV heads times head dim.
    pub fn k_width(&self) -> u32 {
        self.kv_heads * self.head_dim
    }

    /// Width of the value projection per position.
    pub fn v_width(&self) -> u32 {
        self.kv_heads * self.head_dim
    }

    /// Grouped-query group size: how many Q heads share one KV head.
    pub fn gqa_group(&self) -> u32 {
        self.q_heads / self.kv_heads
    }
}

/// Read `<arch>.rope.dimension_count` when the file carries it,
/// validated against the head dimension. Absent means the family
/// rotates the full head, which is returned as `head_dim`.
fn read_rope_dim(file: &GgufFile, arch: &str, head_dim: u32) -> Result<u32> {
    let key = format!("{arch}.{SUFFIX_ROPE_DIMENSION_COUNT}");
    let Some(value) = file.metadata().get(&key) else {
        return Ok(head_dim);
    };
    let dim = match value {
        MetadataValue::UInt8(v) => u32::from(*v),
        MetadataValue::UInt16(v) => u32::from(*v),
        MetadataValue::UInt32(v) => *v,
        MetadataValue::UInt64(v) => u32::try_from(*v)
            .map_err(|_| invalid(&key, "rope dimension count does not fit in u32"))?,
        MetadataValue::Int8(v) => {
            u32::try_from(*v).map_err(|_| invalid(&key, "rope dimension count is negative"))?
        }
        MetadataValue::Int16(v) => {
            u32::try_from(*v).map_err(|_| invalid(&key, "rope dimension count is negative"))?
        }
        MetadataValue::Int32(v) => {
            u32::try_from(*v).map_err(|_| invalid(&key, "rope dimension count is negative"))?
        }
        MetadataValue::Int64(v) => {
            u32::try_from(*v).map_err(|_| invalid(&key, "rope dimension count is negative"))?
        }
        other => {
            return Err(LlamaError::Gguf(gguf::GgufError::MetadataTypeMismatch {
                key,
                expected: "u32",
                found: other.type_name(),
            }))
        }
    };
    if dim < 2 || dim % 2 != 0 || dim > head_dim {
        return Err(invalid(
            &format!("{arch}.{SUFFIX_ROPE_DIMENSION_COUNT}"),
            "rope dimension count must be even, at least 2, and no larger than the head dimension",
        ));
    }
    Ok(dim)
}

fn invalid(key: &str, detail: &str) -> LlamaError {
    LlamaError::InvalidMetadata {
        key: key.to_string(),
        detail: detail.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{build_gguf, qwen2_metadata, refs, tiny_config, tinyllama_metadata};
    use gguf::MetadataValue;

    fn parse(metadata: &[(String, MetadataValue)]) -> Result<LlamaConfig> {
        let bytes = build_gguf(&refs(metadata), &[]);
        let file = GgufFile::from_bytes(bytes).expect("fixture parses");
        LlamaConfig::from_file(&file)
    }

    fn without(metadata: &[(String, MetadataValue)], key: &str) -> Vec<(String, MetadataValue)> {
        metadata.iter().filter(|(k, _)| k != key).cloned().collect()
    }

    fn replaced(
        metadata: &[(String, MetadataValue)],
        key: &str,
        value: MetadataValue,
    ) -> Vec<(String, MetadataValue)> {
        metadata
            .iter()
            .map(|(k, v)| {
                if k == key {
                    (k.clone(), value.clone())
                } else {
                    (k.clone(), v.clone())
                }
            })
            .collect()
    }

    #[test]
    fn tinyllama_file_metadata_parses_to_measured_values() {
        let cfg = parse(&tinyllama_metadata()).expect("real metadata parses");
        assert_eq!(cfg.architecture, "llama");
        assert_eq!(
            cfg.name.as_deref(),
            Some("tinyllama_tinyllama-1.1b-chat-v1.0")
        );
        assert_eq!(cfg.block_count, 22);
        assert_eq!(cfg.context_length, 2048);
        assert_eq!(cfg.hidden, 2048);
        assert_eq!(cfg.ffn_hidden, 5632);
        assert_eq!(cfg.q_heads, 32);
        assert_eq!(cfg.kv_heads, 4);
        assert_eq!(cfg.gqa_group(), 8);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.q_width(), 2048);
        assert_eq!(cfg.k_width(), 256);
        assert!((cfg.rope_freq_base - 10_000.0).abs() < 0.01);
        // The file carries rope.dimension_count, equal to its head dim.
        assert_eq!(cfg.rope_dim, 64);
        assert!((cfg.rms_eps - 1e-5).abs() < 1e-9);
        assert_eq!(cfg.vocab_size, 32_000);
        assert!(!cfg.attention_biases_required());
    }

    #[test]
    fn qwen2_file_metadata_parses_to_measured_values() {
        let cfg = parse(&qwen2_metadata()).expect("real metadata parses");
        assert_eq!(cfg.architecture, "qwen2");
        assert_eq!(cfg.name.as_deref(), Some("qwen2.5-1.5b-instruct"));
        assert_eq!(cfg.block_count, 28);
        assert_eq!(cfg.context_length, 32_768);
        assert_eq!(cfg.hidden, 1536);
        assert_eq!(cfg.ffn_hidden, 8960);
        assert_eq!(cfg.q_heads, 12);
        assert_eq!(cfg.kv_heads, 2);
        assert_eq!(cfg.gqa_group(), 6);
        assert_eq!(cfg.head_dim, 128);
        assert!((cfg.rope_freq_base - 1_000_000.0).abs() < 0.01);
        // The file carries no rope.dimension_count: full-head rotary.
        assert_eq!(cfg.rope_dim, 128);
        assert!((cfg.rms_eps - 1e-6).abs() < 1e-10);
        assert_eq!(cfg.vocab_size, 151_936);
        assert!(cfg.attention_biases_required());
    }

    #[test]
    fn llama_config_selects_the_interleaved_pairing() {
        // From the file's own metadata through the real parser: the
        // llama architecture string alone decides the pairing
        // (llama.cpp maps LLM_ARCH_LLAMA to NORM).
        let cfg = parse(&tinyllama_metadata()).expect("real metadata parses");
        assert_eq!(
            cfg.rope_pairing().expect("llama maps"),
            RopePairing::Interleaved
        );
    }

    #[test]
    fn qwen2_config_selects_the_half_split_pairing() {
        let cfg = parse(&qwen2_metadata()).expect("real metadata parses");
        assert_eq!(
            cfg.rope_pairing().expect("qwen2 maps"),
            RopePairing::HalfSplit
        );
    }

    #[test]
    fn unmapped_architecture_is_refused_a_pairing_by_name() {
        // A hand-assembled config for an architecture the mapping
        // does not cover (from_file would refuse it earlier, at the
        // arch check): the pairing lookup must refuse by name, not
        // default to either pairing.
        let cfg = tiny_config("phi3");
        let err = cfg.rope_pairing().expect_err("phi3 has no mapped pairing");
        match err {
            LlamaError::RopePairingUnknown { architecture } => {
                assert_eq!(architecture, "phi3")
            }
            other => panic!("expected RopePairingUnknown, got {other:?}"),
        }
    }

    #[test]
    fn missing_ffn_key_is_a_named_error() {
        let meta = without(&qwen2_metadata(), "qwen2.feed_forward_length");
        let err = parse(&meta).expect_err("missing key must fail");
        match err {
            LlamaError::MissingMetadata { key } => {
                assert_eq!(key, "qwen2.feed_forward_length")
            }
            other => panic!("expected MissingMetadata, got {other:?}"),
        }
    }

    #[test]
    fn missing_rope_base_is_a_named_error() {
        let meta = without(&tinyllama_metadata(), "llama.rope.freq_base");
        let err = parse(&meta).expect_err("missing key must fail");
        match err {
            LlamaError::MissingMetadata { key } => assert_eq!(key, "llama.rope.freq_base"),
            other => panic!("expected MissingMetadata, got {other:?}"),
        }
    }

    #[test]
    fn missing_block_count_names_the_key_through_the_loader() {
        let meta = without(&tinyllama_metadata(), "llama.block_count");
        let err = parse(&meta).expect_err("missing key must fail");
        assert!(
            err.to_string().contains("llama.block_count"),
            "error should name the key, got: {err}"
        );
    }

    #[test]
    fn wrong_architecture_is_refused() {
        // A complete file for another family: every key the generic
        // reader wants exists under the gemma2 prefix, so the refusal
        // must come from this module's arch check, by name.
        let meta = crate::testkit::tiny_metadata("gemma2");
        let err = parse(&meta).expect_err("wrong arch must fail");
        match err {
            LlamaError::WrongArchitecture { found } => assert_eq!(found, "gemma2"),
            other => panic!("expected WrongArchitecture, got {other:?}"),
        }
    }

    #[test]
    fn impossible_grouping_is_invalid_metadata() {
        // 4 query heads over 3 KV heads cannot form GQA groups.
        let meta = replaced(
            &tinyllama_metadata(),
            "llama.attention.head_count_kv",
            MetadataValue::UInt32(3),
        );
        let err = parse(&meta).expect_err("4 over 3 cannot group");
        assert!(matches!(err, LlamaError::InvalidMetadata { .. }));
    }

    #[test]
    fn oversized_rope_dim_is_invalid_metadata() {
        let meta = replaced(
            &tinyllama_metadata(),
            "llama.rope.dimension_count",
            MetadataValue::UInt32(128),
        );
        let err = parse(&meta).expect_err("rope dim past head dim must fail");
        match err {
            LlamaError::InvalidMetadata { key, .. } => {
                assert_eq!(key, "llama.rope.dimension_count")
            }
            other => panic!("expected InvalidMetadata, got {other:?}"),
        }
    }
}
