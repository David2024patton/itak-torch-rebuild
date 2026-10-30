//! The weight layout of a DFlash draft model: every tensor the two
//! passes need, by its exact GGUF name.
//!
//! The inventory is the one measured in the real file's tensor
//! census (2026-10-30): three globals plus eleven tensors per layer,
//! 58 in total. Two structural facts of this arch live here, not in
//! the forward:
//!
//! - The draft carries NO embedding table and NO output head. In
//!   llama.cpp it borrows the target model's through the target
//!   context when its own are absent. [`Weights::load`] therefore
//!   loads `token_embd.weight` and `output.weight` when the file
//!   carries them and records their absence as the borrowed state;
//!   the forward then requires caller-supplied target tensors
//!   ([`crate::TargetTensors`]) and refuses by name without them.
//! - The variant family (DFlash2 selector, DSpark Markov/confidence
//!   heads, a `d2t` vocabulary map) is gated on tensor presence in
//!   llama.cpp's loader. A file carrying any of those tensors is a
//!   different model than this module implements, so the load
//!   refuses it by name instead of silently running the plain graph
//!   over it.
//!
//! Loading goes through the [`TensorSource`] trait, implemented by
//! the loader crate over a parsed GGUF file. Every tensor's shape
//! is checked against the config before it is accepted: a tensor of
//! the wrong shape is a named error, not a surprise inside a matmul.

use crate::config::DflashConfig;
use crate::error::{DflashError, Result};

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
/// [`DflashError::MissingTensor`]. Never substitute, synthesize, or
/// zero-fill a tensor that was not found; silent placeholders are
/// the defect this rebuild exists to remove.
pub trait TensorSource {
    /// Fetch one tensor by its full GGUF name, dequantized to f32.
    fn tensor(&self, name: &str) -> Result<TensorData>;
    /// Whether a tensor with this name exists in the source. The
    /// load uses it only for the tensors whose presence is a model
    /// property (the draft's own embedding/head, the optional post
    /// norms, and the variant gates), never to skip a required
    /// tensor quietly.
    fn has_tensor(&self, name: &str) -> bool;
}

/// The tensors of one draft layer, in forward order.
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
    /// `attn_q_norm.weight`: per-head query RMSNorm.
    pub attn_q_norm: TensorData,
    /// `attn_k_norm.weight`: per-head key RMSNorm.
    pub attn_k_norm: TensorData,
    /// `attn_output.weight`: attention output projection.
    pub attn_output: TensorData,
    /// `post_attention_norm.weight`: optional post-attention
    /// RMSNorm, applied to the attention branch before its residual
    /// add when the file carries it. Absent in the real file.
    pub post_attention_norm: Option<TensorData>,
    /// `ffn_norm.weight`: pre-FFN RMSNorm.
    pub ffn_norm: TensorData,
    /// `ffn_gate.weight`: SwiGLU gate projection.
    pub ffn_gate: TensorData,
    /// `ffn_up.weight`: SwiGLU up projection.
    pub ffn_up: TensorData,
    /// `ffn_down.weight`: FFN down projection.
    pub ffn_down: TensorData,
    /// `post_ffw_norm.weight`: optional post-FFN RMSNorm, same rule
    /// as the post-attention norm. Absent in the real file.
    pub post_ffw_norm: Option<TensorData>,
}

/// The full weight set of a DFlash draft model.
#[derive(Clone, Debug)]
pub struct Weights {
    /// `fc.weight`: the conditioning fusion projection, from the
    /// concatenated target layer inputs to the draft hidden size.
    pub fc: TensorData,
    /// `enc.output_norm.weight`: RMSNorm over the fused
    /// conditioning rows.
    pub enc_output_norm: TensorData,
    /// `output_norm.weight`: final RMSNorm before the head.
    pub output_norm: TensorData,
    /// `token_embd.weight` when the draft file carries its own
    /// embedding table; `None` means the target's table is borrowed.
    pub embedding: Option<TensorData>,
    /// `output.weight` when the draft file carries its own head;
    /// `None` means the target's head is borrowed.
    pub output: Option<TensorData>,
    /// The target model's hidden size, derived at load from the
    /// `fc` input width divided by the target layer count. The
    /// conditioning rows the injection pass accepts are
    /// `target_layers.len() * target_hidden` wide.
    pub target_hidden: u32,
    /// Per-layer weights, in layer order.
    pub layers: Vec<LayerWeights>,
}

/// Per-layer tensor suffixes [`Weights::load`] always requires, in
/// load order.
const LAYER_SUFFIXES: [&str; 11] = [
    "attn_norm.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "attn_output.weight",
    "ffn_norm.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
];

/// Per-layer optional post-norm suffixes, loaded when present.
const POST_NORM_SUFFIXES: [&str; 2] = ["post_attention_norm.weight", "post_ffw_norm.weight"];

/// Tensor names whose presence marks a DFlash variant this module
/// does not implement (the gates llama.cpp's own loader checks).
const VARIANT_GATES: [(&str, &str); 3] = [
    (
        "d2t",
        "a d2t tensor means the draft predicts a reduced vocabulary that maps to target ids; this module implements the plain full-vocabulary draft",
    ),
    (
        "markov_w1.weight",
        "a markov_w1 tensor marks a DSpark draft (Markov + confidence heads); this module implements plain DFlash",
    ),
    (
        "selector_hidden.weight",
        "a selector_hidden tensor marks a DFlash2 draft (conv + selector lattice); this module implements plain DFlash",
    ),
];

/// Every tensor name [`Weights::load`] will always request for
/// `cfg`, in load order. The draft's own `token_embd.weight` and
/// `output.weight` are not in the list: their presence is probed,
/// and their absence is the borrowed state, not a missing tensor.
/// Exposed so the loader's coverage report can be checked against
/// this module's expectations mechanically.
pub fn expected_tensor_names(cfg: &DflashConfig) -> Vec<String> {
    let mut names = Vec::with_capacity(3 + cfg.block_count as usize * LAYER_SUFFIXES.len());
    names.push("fc.weight".to_string());
    names.push("enc.output_norm.weight".to_string());
    names.push("output_norm.weight".to_string());
    for layer in 0..cfg.block_count {
        for suffix in LAYER_SUFFIXES {
            names.push(format!("blk.{layer}.{suffix}"));
        }
    }
    names
}

/// Tensor names [`Weights::load`] loads when the file carries them:
/// the draft's own embedding table and head, and the per-layer post
/// norms. Reported separately by the loader's coverage check so a
/// file's optional inventory is visible, never silently absorbed.
pub fn optional_tensor_names(cfg: &DflashConfig) -> Vec<String> {
    let mut names = vec!["token_embd.weight".to_string(), "output.weight".to_string()];
    for layer in 0..cfg.block_count {
        for suffix in POST_NORM_SUFFIXES {
            names.push(format!("blk.{layer}.{suffix}"));
        }
    }
    names
}

impl Weights {
    /// Load the full weight set for `cfg` from `source`. The first
    /// missing or misshaped tensor aborts the load with a named
    /// error; a partially loaded model is never returned.
    pub fn load<S: TensorSource>(source: &S, cfg: &DflashConfig) -> Result<Weights> {
        for (gate, detail) in VARIANT_GATES {
            if source.has_tensor(gate) {
                return Err(DflashError::UnsupportedFeature {
                    name: gate.to_string(),
                    detail: detail.to_string(),
                });
            }
        }

        let hidden = u64::from(cfg.hidden);
        let ffn = u64::from(cfg.ffn_hidden);
        let vocab = u64::from(cfg.vocab_size);
        let q_w = u64::from(cfg.q_width());
        let k_w = u64::from(cfg.k_width());
        let v_w = u64::from(cfg.v_width());
        let out_w = u64::from(cfg.attn_out_width());

        // The fc input width is the one shape the config cannot
        // fully pin down (it depends on the TARGET's hidden size),
        // so it is validated structurally here: it must be a whole
        // multiple of the target layer count, and its quotient is
        // the target hidden size every conditioning row must match.
        let fc = source.tensor("fc.weight")?;
        if fc.shape.len() != 2 || fc.shape[1] != hidden {
            return Err(DflashError::ShapeMismatch {
                name: fc.name.clone(),
                expected: vec![0, hidden],
                actual: fc.shape.clone(),
            });
        }
        let cond_in = fc.shape[0];
        let n_target = cfg.target_layers.len() as u64;
        if cond_in == 0 || cond_in % n_target != 0 {
            return Err(DflashError::InvalidMetadata {
                key: "dflash.target_layers".to_string(),
                detail: format!(
                    "fc.weight input width {cond_in} is not a whole multiple of the {} target layers, so the concatenated conditioning rows cannot be its input",
                    cfg.target_layers.len()
                ),
            });
        }
        check_len(&fc)?;
        let target_hidden =
            u32::try_from(cond_in / n_target).map_err(|_| DflashError::InvalidMetadata {
                key: "dflash.target_layers".to_string(),
                detail: "target hidden size derived from fc.weight does not fit in u32".to_string(),
            })?;

        let enc_output_norm = checked(source, "enc.output_norm.weight", &[hidden])?;
        let output_norm = checked(source, "output_norm.weight", &[hidden])?;
        let embedding = if source.has_tensor("token_embd.weight") {
            Some(checked(source, "token_embd.weight", &[hidden, vocab])?)
        } else {
            None
        };
        let output = if source.has_tensor("output.weight") {
            Some(checked(source, "output.weight", &[hidden, vocab])?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(cfg.block_count as usize);
        for index in 0..cfg.block_count {
            let get = |suffix: &str, shape: &[u64]| {
                checked(source, &format!("blk.{index}.{suffix}"), shape)
            };
            let post = |suffix: &str| -> Result<Option<TensorData>> {
                let name = format!("blk.{index}.{suffix}");
                if source.has_tensor(&name) {
                    Ok(Some(checked(source, &name, &[hidden])?))
                } else {
                    Ok(None)
                }
            };
            layers.push(LayerWeights {
                attn_norm: get(LAYER_SUFFIXES[0], &[hidden])?,
                attn_q: get(LAYER_SUFFIXES[1], &[hidden, q_w])?,
                attn_k: get(LAYER_SUFFIXES[2], &[hidden, k_w])?,
                attn_v: get(LAYER_SUFFIXES[3], &[hidden, v_w])?,
                attn_q_norm: get(LAYER_SUFFIXES[4], &[u64::from(cfg.key_length)])?,
                attn_k_norm: get(LAYER_SUFFIXES[5], &[u64::from(cfg.key_length)])?,
                attn_output: get(LAYER_SUFFIXES[6], &[out_w, hidden])?,
                post_attention_norm: post(POST_NORM_SUFFIXES[0])?,
                ffn_norm: get(LAYER_SUFFIXES[7], &[hidden])?,
                ffn_gate: get(LAYER_SUFFIXES[8], &[hidden, ffn])?,
                ffn_up: get(LAYER_SUFFIXES[9], &[hidden, ffn])?,
                ffn_down: get(LAYER_SUFFIXES[10], &[ffn, hidden])?,
                post_ffw_norm: post(POST_NORM_SUFFIXES[1])?,
            });
        }
        Ok(Weights {
            fc,
            enc_output_norm,
            output_norm,
            embedding,
            output,
            target_hidden,
            layers,
        })
    }

    /// The row width conditioning features must have: target layer
    /// count times the target hidden size, as loaded from `fc`.
    pub fn conditioning_width(&self) -> usize {
        self.fc.shape[0] as usize
    }
}

/// Fetch one tensor and verify it against the shape the config
/// requires for it: presence, dimensions, and element count all
/// checked before the tensor is accepted.
fn checked<S: TensorSource>(source: &S, name: &str, expected: &[u64]) -> Result<TensorData> {
    let tensor = source.tensor(name)?;
    if tensor.shape != expected {
        return Err(DflashError::ShapeMismatch {
            name: name.to_string(),
            expected: expected.to_vec(),
            actual: tensor.shape.clone(),
        });
    }
    check_len(&tensor)?;
    Ok(tensor)
}

/// A tensor's data must hold exactly the elements its shape claims.
fn check_len(tensor: &TensorData) -> Result<()> {
    let want = tensor.elements();
    if tensor.data.len() as u64 != want {
        return Err(DflashError::TensorDataLength {
            name: tensor.name.clone(),
            expected: want,
            actual: tensor.data.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{tiny_config, FakeSource};

    #[test]
    fn full_source_loads_borrowed() {
        let cfg = tiny_config();
        let source = FakeSource::full(&cfg, false);
        let weights = Weights::load(&source, &cfg).expect("full source loads");
        assert_eq!(weights.layers.len(), 2);
        assert!(weights.embedding.is_none(), "draft borrows the embedding");
        assert!(weights.output.is_none(), "draft borrows the head");
        // fc input width 16 over 2 target layers: target hidden 8.
        assert_eq!(weights.target_hidden, 8);
        assert_eq!(weights.conditioning_width(), 16);
        assert!(weights.layers[0].post_attention_norm.is_none());
    }

    #[test]
    fn own_embedding_and_head_load_when_present() {
        let cfg = tiny_config();
        let source = FakeSource::full(&cfg, true);
        let weights = Weights::load(&source, &cfg).expect("full source loads");
        assert!(weights.embedding.is_some());
        assert!(weights.output.is_some());
    }

    #[test]
    fn missing_ffn_down_is_named() {
        let cfg = tiny_config();
        let mut source = FakeSource::full(&cfg, false);
        source.remove("blk.1.ffn_down.weight");
        let err = Weights::load(&source, &cfg).expect_err("missing tensor must fail");
        match err {
            DflashError::MissingTensor { name } => assert_eq!(name, "blk.1.ffn_down.weight"),
            other => panic!("expected MissingTensor, got {other:?}"),
        }
    }

    #[test]
    fn variant_gate_refuses_by_name() {
        let cfg = tiny_config();
        let mut source = FakeSource::full(&cfg, false);
        source.insert(crate::testkit::tensor(
            "selector_hidden.weight",
            &[4, 4],
            0.0,
        ));
        let err = Weights::load(&source, &cfg).expect_err("DFlash2 file must refuse");
        match err {
            DflashError::UnsupportedFeature { name, .. } => {
                assert_eq!(name, "selector_hidden.weight")
            }
            other => panic!("expected UnsupportedFeature, got {other:?}"),
        }
    }

    #[test]
    fn fc_width_not_a_multiple_of_target_layers_is_invalid() {
        let cfg = tiny_config();
        let mut source = FakeSource::full(&cfg, false);
        source.insert(crate::testkit::tensor("fc.weight", &[15, 8], 0.0));
        let err = Weights::load(&source, &cfg).expect_err("15 over 2 layers cannot fuse");
        assert!(matches!(err, DflashError::InvalidMetadata { .. }));
    }

    #[test]
    fn wrong_shape_is_named() {
        let cfg = tiny_config();
        let mut source = FakeSource::full(&cfg, false);
        // The tiny config's attn_q is [hidden 8, q_width 8]; a
        // [4, 8] tensor cannot be it.
        source.insert(crate::testkit::tensor("blk.0.attn_q.weight", &[4, 8], 0.0));
        let err = Weights::load(&source, &cfg).expect_err("wrong shape must fail");
        match err {
            DflashError::ShapeMismatch {
                name,
                expected,
                actual,
            } => {
                assert_eq!(name, "blk.0.attn_q.weight");
                assert_eq!(expected, vec![8, 8]);
                assert_eq!(actual, vec![4, 8]);
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn expected_names_match_the_real_census_count() {
        let cfg = tiny_config();
        let names = expected_tensor_names(&cfg);
        assert_eq!(names.len(), 3 + 2 * 11);
        assert_eq!(names[0], "fc.weight");
        let real = crate::testkit::real_config();
        assert_eq!(expected_tensor_names(&real).len(), 58);
    }
}
