//! The weight layout of a llama-family or qwen2-family model: every
//! tensor the forward needs, by its exact GGUF name.
//!
//! The per-layer inventory is the one measured in the real files'
//! tensor censuses (2026-10-29), not the shorter list the original
//! engine's Llama-shaped template used. That template never looked
//! up the attention biases, so Qwen2 ran with its biases silently
//! dropped; here the biases are named members of the map. For
//! `qwen2` files they are required (the family's reference graph
//! applies them on every layer, so a qwen2 file without them is not
//! the model it claims to be); for `llama` files they are loaded and
//! applied when present.
//!
//! Loading goes through the [`TensorSource`] trait, which the `gguf`
//! crate's [`GgufFile`] implements below. Every tensor's shape is
//! checked against the config before it is accepted: a tensor of the
//! wrong shape is a named error, not a surprise inside a matmul.
//!
//! The output head: when `output.weight` is present it is the head.
//! When it is absent the model is tied and the head reads the
//! embedding table; that state is recorded on [`Weights`] as
//! `tied_embeddings`, not improvised at call time. (The real
//! Qwen2.5-1.5B GGUF carries an explicit `output.weight` even though
//! the Hugging Face checkpoint ties its embeddings: the converter
//! materialized the head. So this file loads untied, and the tied
//! path serves the files that genuinely omit the head.)

use gguf::GgufFile;

use crate::config::LlamaConfig;
use crate::error::{LlamaError, Result};

/// One tensor handed over by a [`TensorSource`]: its name, its shape
/// in GGUF stored order (innermost dimension first), and its elements
/// dequantized to f32 in stored order.
///
/// The forward's projections consume `data` directly as a row-major
/// matrix whose rows are the output features: GGUF stores a
/// projection of shape `[in, out]` as `out` rows of `in` elements,
/// which is exactly the layout `torch_math::matvec` multiplies.
#[derive(Clone, Debug)]
pub struct TensorData {
    /// Full GGUF tensor name, for example `blk.3.ffn_down.weight`.
    pub name: String,
    /// Dimensions in GGUF stored order (innermost first).
    pub shape: Vec<u64>,
    /// Dequantized elements, in stored order.
    pub data: Vec<f32>,
}

impl TensorData {
    /// Total element count implied by the shape.
    pub fn elements(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// A source of tensors by name: the seam between this module and
/// the loader.
///
/// Implementor contract: return the tensor or fail with
/// [`LlamaError::MissingTensor`]. Never substitute, synthesize, or
/// zero-fill a tensor that was not found; silent placeholders are
/// the defect this rebuild exists to remove.
pub trait TensorSource {
    /// Fetch one tensor by its full GGUF name, dequantized to f32.
    fn tensor(&self, name: &str) -> Result<TensorData>;
    /// Whether a tensor with this name exists in the source. The
    /// load uses it only for the tensors whose presence is a model
    /// property (the untied head, llama-family biases), never to
    /// skip a required tensor quietly.
    fn has_tensor(&self, name: &str) -> bool;
}

impl TensorSource for GgufFile {
    fn tensor(&self, name: &str) -> Result<TensorData> {
        let info = gguf::GgufFile::tensor_info(self, name)?;
        let shape = info.dims().to_vec();
        let data = self.load_f32(name)?;
        Ok(TensorData {
            name: name.to_string(),
            shape,
            data,
        })
    }

    fn has_tensor(&self, name: &str) -> bool {
        gguf::GgufFile::has_tensor(self, name)
    }
}

/// The tensors of one transformer layer, in forward order.
#[derive(Clone, Debug)]
pub struct LayerWeights {
    /// `attn_norm.weight`: pre-attention RMSNorm.
    pub attn_norm: TensorData,
    /// `attn_q.weight`: query projection.
    pub attn_q: TensorData,
    /// `attn_k.weight`: key projection.
    pub attn_k: TensorData,
    /// `attn_v.weight`: value projection.
    pub attn_v: TensorData,
    /// `attn_q.bias`: query bias. Required for qwen2, applied when
    /// present for llama, absent otherwise.
    pub attn_q_bias: Option<TensorData>,
    /// `attn_k.bias`: key bias; same rule as the query bias.
    pub attn_k_bias: Option<TensorData>,
    /// `attn_v.bias`: value bias; same rule as the query bias.
    pub attn_v_bias: Option<TensorData>,
    /// `attn_output.weight`: attention output projection.
    pub attn_output: TensorData,
    /// `ffn_norm.weight`: pre-FFN RMSNorm.
    pub ffn_norm: TensorData,
    /// `ffn_gate.weight`: SwiGLU gate projection.
    pub ffn_gate: TensorData,
    /// `ffn_up.weight`: SwiGLU up projection.
    pub ffn_up: TensorData,
    /// `ffn_down.weight`: FFN down projection (the FFN tensors hold
    /// most of both conformance models' bytes; this is the large one).
    pub ffn_down: TensorData,
}

/// The full weight set of a llama-family or qwen2-family model.
#[derive(Clone, Debug)]
pub struct Weights {
    /// `token_embd.weight`: the embedding table.
    pub embedding: TensorData,
    /// `output_norm.weight`: final RMSNorm before the head.
    pub final_norm: TensorData,
    /// `output.weight`: the untied output head, when the file
    /// carries one.
    pub output: Option<TensorData>,
    /// Whether the head is tied to the embedding table (recorded
    /// from the absence of `output.weight` at load).
    pub tied_embeddings: bool,
    /// Per-layer weights, in layer order.
    pub layers: Vec<LayerWeights>,
}

/// Per-layer tensor suffixes that [`Weights::load`] always requires,
/// in load order.
const LAYER_SUFFIXES: [&str; 9] = [
    "attn_norm.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_norm.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
];

/// Per-layer bias suffixes, required for qwen2 and probed for llama.
const BIAS_SUFFIXES: [&str; 3] = ["attn_q.bias", "attn_k.bias", "attn_v.bias"];

/// The name of the output head tensor.
pub const OUTPUT_HEAD_NAME: &str = "output.weight";

/// Every tensor name [`Weights::load`] will always request for
/// `cfg`, in load order. `output.weight` is not in the list: its
/// presence is probed, and its absence is the tied state, not a
/// missing tensor. Exposed so the loader's coverage report can be
/// checked against this module's expectations mechanically.
pub fn expected_tensor_names(cfg: &LlamaConfig) -> Vec<String> {
    let mut names = Vec::with_capacity(
        2 + cfg.block_count as usize * (LAYER_SUFFIXES.len() + BIAS_SUFFIXES.len()),
    );
    names.push("token_embd.weight".to_string());
    names.push("output_norm.weight".to_string());
    for layer in 0..cfg.block_count {
        for suffix in LAYER_SUFFIXES {
            names.push(format!("blk.{layer}.{suffix}"));
        }
        if cfg.attention_biases_required() {
            for suffix in BIAS_SUFFIXES {
                names.push(format!("blk.{layer}.{suffix}"));
            }
        }
    }
    names
}

impl Weights {
    /// Load the full weight set for `cfg` from `source`. The first
    /// missing or misshaped tensor aborts the load with a named
    /// error; a partially loaded model is never returned.
    pub fn load<S: TensorSource>(source: &S, cfg: &LlamaConfig) -> Result<Weights> {
        let hidden = u64::from(cfg.hidden);
        let vocab = u64::from(cfg.vocab_size);
        let ffn = u64::from(cfg.ffn_hidden);
        let q_w = u64::from(cfg.q_width());
        let k_w = u64::from(cfg.k_width());
        let v_w = u64::from(cfg.v_width());

        let embedding = checked(source, "token_embd.weight", &[hidden, vocab])?;
        let final_norm = checked(source, "output_norm.weight", &[hidden])?;
        let output = if source.has_tensor(OUTPUT_HEAD_NAME) {
            Some(checked(source, OUTPUT_HEAD_NAME, &[hidden, vocab])?)
        } else {
            None
        };
        let tied_embeddings = output.is_none();

        let mut layers = Vec::with_capacity(cfg.block_count as usize);
        for index in 0..cfg.block_count {
            let get_checked = |suffix: &str, shape: &[u64]| {
                checked(source, &format!("blk.{index}.{suffix}"), shape)
            };
            let bias = |suffix: &str, width: u64| -> Result<Option<TensorData>> {
                let name = format!("blk.{index}.{suffix}");
                // Required for qwen2 (where a missing bias then
                // fails inside `checked` by name); for llama, loaded
                // when the file carries the tensor.
                if cfg.attention_biases_required() || source.has_tensor(&name) {
                    Ok(Some(checked(source, &name, &[width])?))
                } else {
                    Ok(None)
                }
            };
            layers.push(LayerWeights {
                attn_norm: get_checked(LAYER_SUFFIXES[0], &[hidden])?,
                attn_q: get_checked(LAYER_SUFFIXES[1], &[hidden, q_w])?,
                attn_k: get_checked(LAYER_SUFFIXES[2], &[hidden, k_w])?,
                attn_v: get_checked(LAYER_SUFFIXES[3], &[hidden, v_w])?,
                attn_q_bias: bias(BIAS_SUFFIXES[0], q_w)?,
                attn_k_bias: bias(BIAS_SUFFIXES[1], k_w)?,
                attn_v_bias: bias(BIAS_SUFFIXES[2], v_w)?,
                attn_output: get_checked(LAYER_SUFFIXES[4], &[q_w, hidden])?,
                ffn_norm: get_checked(LAYER_SUFFIXES[5], &[hidden])?,
                ffn_gate: get_checked(LAYER_SUFFIXES[6], &[hidden, ffn])?,
                ffn_up: get_checked(LAYER_SUFFIXES[7], &[hidden, ffn])?,
                ffn_down: get_checked(LAYER_SUFFIXES[8], &[ffn, hidden])?,
            });
        }
        Ok(Weights {
            embedding,
            final_norm,
            output,
            tied_embeddings,
            layers,
        })
    }

    /// The tensor the LM head multiplies: the untied head when the
    /// file carried one, the embedding table when tied.
    pub fn head(&self) -> &TensorData {
        match &self.output {
            Some(head) => head,
            None => &self.embedding,
        }
    }
}

/// Fetch one tensor and verify it against the shape the config
/// requires for it: presence, dimensions, and element count all
/// checked before the tensor is accepted.
fn checked<S: TensorSource>(source: &S, name: &str, expected: &[u64]) -> Result<TensorData> {
    let tensor = source.tensor(name)?;
    if tensor.shape != expected {
        return Err(LlamaError::ShapeMismatch {
            name: name.to_string(),
            expected: expected.to_vec(),
            actual: tensor.shape.clone(),
        });
    }
    let want = tensor.elements();
    if tensor.data.len() as u64 != want {
        return Err(LlamaError::TensorDataLength {
            name: name.to_string(),
            expected: want,
            actual: tensor.data.len(),
        });
    }
    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{tiny_config, BiasMode, FakeSource};

    #[test]
    fn llama_full_source_loads_untied() {
        let cfg = tiny_config("llama");
        let source = FakeSource::full(&cfg, false, BiasMode::None);
        let weights = Weights::load(&source, &cfg).expect("full source loads");
        assert_eq!(weights.layers.len(), 2);
        assert!(!weights.tied_embeddings);
        assert!(weights.output.is_some());
        assert!(weights.layers[0].attn_q_bias.is_none());
    }

    #[test]
    fn qwen2_full_source_loads_with_biases() {
        let cfg = tiny_config("qwen2");
        let source = FakeSource::full(&cfg, false, BiasMode::NonZero);
        let weights = Weights::load(&source, &cfg).expect("full source loads");
        assert!(weights.layers[1].attn_q_bias.is_some());
        assert!(weights.layers[1].attn_k_bias.is_some());
        assert!(weights.layers[1].attn_v_bias.is_some());
    }

    #[test]
    fn missing_output_head_means_tied_not_missing() {
        let cfg = tiny_config("qwen2");
        let source = FakeSource::full(&cfg, true, BiasMode::NonZero);
        let weights = Weights::load(&source, &cfg).expect("tied source loads");
        assert!(weights.tied_embeddings);
        assert!(weights.output.is_none());
        // The head falls back to the embedding table itself.
        assert_eq!(weights.head().name, "token_embd.weight");
    }

    #[test]
    fn missing_ffn_down_is_named() {
        let cfg = tiny_config("llama");
        let mut source = FakeSource::full(&cfg, false, BiasMode::None);
        source.remove("blk.1.ffn_down.weight");
        let err = Weights::load(&source, &cfg).expect_err("missing tensor must fail");
        match err {
            LlamaError::MissingTensor { name } => {
                assert_eq!(name, "blk.1.ffn_down.weight")
            }
            other => panic!("expected MissingTensor, got {other:?}"),
        }
    }

    #[test]
    fn qwen2_missing_bias_is_named() {
        let cfg = tiny_config("qwen2");
        let mut source = FakeSource::full(&cfg, false, BiasMode::NonZero);
        source.remove("blk.0.attn_q.bias");
        let err = Weights::load(&source, &cfg).expect_err("qwen2 without biases must fail");
        match err {
            LlamaError::MissingTensor { name } => assert_eq!(name, "blk.0.attn_q.bias"),
            other => panic!("expected MissingTensor, got {other:?}"),
        }
    }

    #[test]
    fn llama_biases_present_are_loaded() {
        let cfg = tiny_config("llama");
        let source = FakeSource::full(&cfg, false, BiasMode::NonZero);
        let weights = Weights::load(&source, &cfg).expect("loads");
        assert!(weights.layers[0].attn_q_bias.is_some());
    }

    #[test]
    fn wrong_shape_is_named() {
        let cfg = tiny_config("llama");
        let mut source = FakeSource::full(&cfg, false, BiasMode::None);
        source.set(TensorData {
            name: "blk.0.attn_q.weight".to_string(),
            shape: vec![16, 16],
            data: vec![0.0; 256],
        });
        let err = Weights::load(&source, &cfg).expect_err("wrong shape must fail");
        match err {
            LlamaError::ShapeMismatch {
                name,
                expected,
                actual,
            } => {
                assert_eq!(name, "blk.0.attn_q.weight");
                assert_eq!(expected, vec![32, 32]);
                assert_eq!(actual, vec![16, 16]);
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn expected_names_match_the_load_map() {
        let llama = tiny_config("llama");
        let names = expected_tensor_names(&llama);
        assert_eq!(names.len(), 2 + 2 * 9);
        assert!(!names.iter().any(|n| n.ends_with(".bias")));
        let qwen2 = tiny_config("qwen2");
        let names = expected_tensor_names(&qwen2);
        assert_eq!(names.len(), 2 + 2 * 12);
        assert!(names.contains(&"blk.1.attn_v.bias".to_string()));
    }
}
