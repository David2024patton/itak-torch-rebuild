//! The weight layout of a Muse Glimmer model: every tensor the forward
//! needs, by its exact GGUF name.
//!
//! The per-layer inventory is the one measured in the real file's tensor
//! census (2026-10-29), not the shorter list a Llama-shaped template
//! would guess. Beyond the familiar q/k/v/o projections, gate/up/down,
//! and two norms, the real layers also carry Q and K norms (applied per
//! head after the projections), an attention gate projection, and post
//! norms on both sub-layer outputs. A template that never looks those
//! names up silently runs without them; that is the original engine's
//! defect, designed out here: [`Weights::load`] requires every name, and
//! a missing one is a [`GlimmerError::MissingTensor`] naming it. The two
//! post norms are sandwich norms: they normalize their sub-layer's
//! output before the residual add (norm audit, 2026-10-29).
//!
//! Loading goes through the [`TensorSource`] trait. The loader crate
//! (TODO T-2.1, `itak-loader` in this workspace) implements it over the
//! `gguf` crate's parsed file; this crate never parses files.
//!
//! The output head is untied: `output.weight` is a tensor of its own in
//! the file. If it is absent, load fails. Substituting the embedding
//! table for a missing head is a silent behavior change and is never
//! done here.

use std::sync::Arc;

use crate::config::GlimmerConfig;
use crate::error::Result;

/// Element type of a tensor, as declared by its GGUF type id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    /// ggml type 0.
    F32,
    /// ggml type 1.
    F16,
    /// ggml type 2.
    Q4_0,
    /// ggml type 8.
    Q8_0,
    /// ggml type 12.
    Q4K,
    /// ggml type 13.
    Q5K,
    /// ggml type 14.
    Q6K,
    /// ggml type 30 (mainline assignment; the original Go loader read
    /// this id as a 2-bit type and corrupted every BF16 tensor, which is
    /// why the mapping lives in exactly one place).
    Bf16,
    /// A type id this module does not name. Carried, not hidden: the
    /// compute layer decides whether it can work with the tensor, and
    /// refuses by name when it cannot.
    Unknown(u32),
}

impl Dtype {
    /// Map a mainline ggml type id to a [`Dtype`].
    pub fn from_ggml_id(id: u32) -> Dtype {
        match id {
            0 => Dtype::F32,
            1 => Dtype::F16,
            2 => Dtype::Q4_0,
            8 => Dtype::Q8_0,
            12 => Dtype::Q4K,
            13 => Dtype::Q5K,
            14 => Dtype::Q6K,
            30 => Dtype::Bf16,
            other => Dtype::Unknown(other),
        }
    }
}

/// One tensor handed over by a [`TensorSource`]: its name, type, shape
/// in GGUF stored order, and its raw bytes.
///
/// The bytes sit behind an [`Arc`] because weight sets are large (the
/// real model is 15.59 GiB) and views get cloned as they move between
/// the loader, this module, and the compute layer. Dequantization is the
/// compute layer's job; this type is deliberately just the measured
/// facts about the tensor.
#[derive(Clone, Debug)]
pub struct TensorView {
    /// Full GGUF tensor name, for example `blk.3.ffn_down.weight`.
    pub name: String,
    /// Element type.
    pub dtype: Dtype,
    /// Dimensions in GGUF stored order (innermost first).
    pub shape: Vec<u64>,
    /// Raw tensor bytes, exactly as stored in the file.
    pub data: Arc<[u8]>,
}

impl TensorView {
    /// Total element count implied by the shape.
    pub fn elements(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// A source of tensors by name: the seam between this module and the
/// loader.
///
/// Implementor contract: return the tensor or fail loud. Absent names
/// fail with [`GlimmerError::MissingTensor`]; names that are present but
/// cannot be delivered as stored fail with
/// [`GlimmerError::TensorUnreadable`]. Never substitute, synthesize, or
/// zero-fill a tensor that was not found; silent placeholders are the
/// defect this rebuild exists to remove.
pub trait TensorSource {
    /// Fetch one tensor by its full GGUF name.
    fn tensor(&self, name: &str) -> Result<TensorView>;
}

/// The tensors of one transformer layer, in forward order.
#[derive(Clone, Debug)]
pub struct LayerWeights {
    /// `attn_norm.weight`: pre-attention RMSNorm.
    pub attn_norm: TensorView,
    /// `attn_q.weight`: query projection.
    pub attn_q: TensorView,
    /// `attn_k.weight`: key projection.
    pub attn_k: TensorView,
    /// `attn_v.weight`: value projection.
    pub attn_v: TensorView,
    /// `attn_q_norm.weight`: per-head RMSNorm on the query projection.
    pub attn_q_norm: TensorView,
    /// `attn_k_norm.weight`: per-head RMSNorm on the key projection.
    pub attn_k_norm: TensorView,
    /// `attn_output.weight`: attention output projection.
    pub attn_output: TensorView,
    /// `attn_gate.weight`: projection producing the gate that scales the
    /// attention output (gated attention, present in the real file).
    pub attn_gate: TensorView,
    /// `post_attention_norm.weight`: sandwich RMSNorm on the
    /// attention output, before the residual add.
    pub post_attention_norm: TensorView,
    /// `ffn_norm.weight`: pre-FFN RMSNorm.
    pub ffn_norm: TensorView,
    /// `ffn_gate.weight`: SwiGLU gate projection.
    pub ffn_gate: TensorView,
    /// `ffn_up.weight`: SwiGLU up projection.
    pub ffn_up: TensorView,
    /// `ffn_down.weight`: FFN down projection (75 percent of the model's
    /// bytes live in the FFN tensors; this is the large one).
    pub ffn_down: TensorView,
    /// `post_ffw_norm.weight`: sandwich RMSNorm on the FFN output,
    /// before the residual add.
    pub post_ffw_norm: TensorView,
}

/// The full weight set of a Muse Glimmer model.
#[derive(Clone, Debug)]
pub struct Weights {
    /// `token_embd.weight`: the embedding table.
    pub embedding: TensorView,
    /// `output_norm.weight`: final RMSNorm before the head.
    pub final_norm: TensorView,
    /// `output.weight`: the untied output head (Q5_K in the real file).
    pub output: TensorView,
    /// Per-layer weights, in layer order.
    pub layers: Vec<LayerWeights>,
}

/// Per-layer tensor suffixes in forward order, exactly as measured in
/// the real file's census.
const LAYER_SUFFIXES: [&str; 14] = [
    "attn_norm.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "attn_output.weight",
    "attn_gate.weight",
    "post_attention_norm.weight",
    "ffn_norm.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
    "post_ffw_norm.weight",
];

/// Every tensor name [`Weights::load`] will request for `cfg`, in load
/// order. Exposed so the loader's tensor inventory (TODO T-2.1) can be
/// checked against this module's expectations mechanically, instead of
/// by reading two codebases side by side. Verified against the real
/// file on 2026-10-29 through the `itak-loader` inventory check: all
/// 731 expected names present, zero unexpected, and
/// `output_norm.weight` confirmed as the file's final-norm tensor
/// (F32, shape [6656]), not merely a naming-convention assumption.
pub fn expected_tensor_names(cfg: &GlimmerConfig) -> Vec<String> {
    let mut names = Vec::with_capacity(3 + cfg.block_count as usize * LAYER_SUFFIXES.len());
    names.push("token_embd.weight".to_string());
    names.push("output_norm.weight".to_string());
    names.push("output.weight".to_string());
    for layer in 0..cfg.block_count {
        for suffix in LAYER_SUFFIXES {
            names.push(format!("blk.{layer}.{suffix}"));
        }
    }
    names
}

impl Weights {
    /// Load the full weight set for `cfg` from `source`. The first
    /// missing tensor aborts the load with a named error; a partially
    /// loaded model is never returned.
    pub fn load<S: TensorSource>(source: &S, cfg: &GlimmerConfig) -> Result<Weights> {
        let embedding = source.tensor("token_embd.weight")?;
        let final_norm = source.tensor("output_norm.weight")?;
        let output = source.tensor("output.weight")?;
        let mut layers = Vec::with_capacity(cfg.block_count as usize);
        for index in 0..cfg.block_count {
            let get = |suffix: &str| source.tensor(&format!("blk.{index}.{suffix}"));
            layers.push(LayerWeights {
                attn_norm: get(LAYER_SUFFIXES[0])?,
                attn_q: get(LAYER_SUFFIXES[1])?,
                attn_k: get(LAYER_SUFFIXES[2])?,
                attn_v: get(LAYER_SUFFIXES[3])?,
                attn_q_norm: get(LAYER_SUFFIXES[4])?,
                attn_k_norm: get(LAYER_SUFFIXES[5])?,
                attn_output: get(LAYER_SUFFIXES[6])?,
                attn_gate: get(LAYER_SUFFIXES[7])?,
                post_attention_norm: get(LAYER_SUFFIXES[8])?,
                ffn_norm: get(LAYER_SUFFIXES[9])?,
                ffn_gate: get(LAYER_SUFFIXES[10])?,
                ffn_up: get(LAYER_SUFFIXES[11])?,
                ffn_down: get(LAYER_SUFFIXES[12])?,
                post_ffw_norm: get(LAYER_SUFFIXES[13])?,
            });
        }
        Ok(Weights {
            embedding,
            final_norm,
            output,
            layers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GlimmerConfig;
    use crate::error::GlimmerError;
    use crate::testkit::tiny_metadata;
    use std::collections::HashMap;

    /// A tensor source backed by a name set: every expected name is
    /// present unless the test removed it.
    struct FakeSource {
        present: HashMap<String, ()>,
    }

    impl FakeSource {
        fn full(cfg: &GlimmerConfig) -> Self {
            let present = expected_tensor_names(cfg)
                .into_iter()
                .map(|n| (n, ()))
                .collect();
            FakeSource { present }
        }
    }

    impl TensorSource for FakeSource {
        fn tensor(&self, name: &str) -> Result<TensorView> {
            if self.present.contains_key(name) {
                Ok(TensorView {
                    name: name.to_string(),
                    dtype: Dtype::F32,
                    shape: vec![1],
                    data: Arc::from(vec![0u8; 4].into_boxed_slice()),
                })
            } else {
                Err(GlimmerError::MissingTensor {
                    name: name.to_string(),
                })
            }
        }
    }

    fn tiny_cfg() -> GlimmerConfig {
        GlimmerConfig::from_metadata(&tiny_metadata()).expect("tiny config")
    }

    #[test]
    fn expected_names_cover_globals_and_every_layer() {
        let cfg = tiny_cfg();
        let names = expected_tensor_names(&cfg);
        assert_eq!(names.len(), 3 + 2 * 14);
        assert_eq!(names[0], "token_embd.weight");
        assert!(names.contains(&"blk.1.post_ffw_norm.weight".to_string()));
        assert!(names.contains(&"blk.0.attn_gate.weight".to_string()));
        assert!(names.contains(&"blk.0.attn_q_norm.weight".to_string()));
    }

    #[test]
    fn full_source_loads() {
        let cfg = tiny_cfg();
        let source = FakeSource::full(&cfg);
        let weights = Weights::load(&source, &cfg).expect("full source loads");
        assert_eq!(weights.layers.len(), 2);
        assert_eq!(weights.output.name, "output.weight");
    }

    #[test]
    fn missing_ffn_down_is_named() {
        let cfg = tiny_cfg();
        let mut source = FakeSource::full(&cfg);
        source.present.remove("blk.1.ffn_down.weight");
        let err = Weights::load(&source, &cfg).expect_err("missing tensor must fail");
        match err {
            GlimmerError::MissingTensor { name } => {
                assert_eq!(name, "blk.1.ffn_down.weight")
            }
            other => panic!("expected MissingTensor, got {other:?}"),
        }
    }

    #[test]
    fn missing_output_head_is_not_replaced_by_embedding() {
        let cfg = tiny_cfg();
        let mut source = FakeSource::full(&cfg);
        source.present.remove("output.weight");
        let err = Weights::load(&source, &cfg).expect_err("missing head must fail");
        assert!(matches!(err, GlimmerError::MissingTensor { .. }));
    }

    #[test]
    fn global_tensors_bind_by_name_and_content() {
        // Each expected name serves DISTINCT bytes (its index in the
        // expected-names list, as an f32). The three global fields
        // must carry the bytes of their own names: if load ever
        // cross-wired two of them (the final norm reading the head's
        // bytes, say), the forward would still run and produce wrong
        // numbers with no error anywhere. This test makes that swap
        // fail here instead.
        struct IndexedSource {
            values: HashMap<String, f32>,
        }
        impl TensorSource for IndexedSource {
            fn tensor(&self, name: &str) -> Result<TensorView> {
                match self.values.get(name) {
                    Some(v) => Ok(TensorView {
                        name: name.to_string(),
                        dtype: Dtype::F32,
                        shape: vec![1],
                        data: Arc::from(v.to_le_bytes().as_slice()),
                    }),
                    None => Err(GlimmerError::MissingTensor {
                        name: name.to_string(),
                    }),
                }
            }
        }
        let cfg = tiny_cfg();
        let values: HashMap<String, f32> = expected_tensor_names(&cfg)
            .into_iter()
            .enumerate()
            .map(|(i, n)| (n, i as f32))
            .collect();
        let source = IndexedSource { values };
        let weights = Weights::load(&source, &cfg).expect("indexed source loads");
        let served = |view: &TensorView| {
            f32::from_le_bytes(
                view.data.iter().copied().collect::<Vec<u8>>()[..4]
                    .try_into()
                    .unwrap(),
            )
        };
        let expect = |name: &str| source.values[name];
        assert_eq!(weights.embedding.name, "token_embd.weight");
        assert_eq!(served(&weights.embedding), expect("token_embd.weight"));
        assert_eq!(weights.final_norm.name, "output_norm.weight");
        assert_eq!(served(&weights.final_norm), expect("output_norm.weight"));
        assert_eq!(weights.output.name, "output.weight");
        assert_eq!(served(&weights.output), expect("output.weight"));
        assert_eq!(weights.layers[0].attn_norm.name, "blk.0.attn_norm.weight");
        assert_eq!(
            served(&weights.layers[0].attn_norm),
            expect("blk.0.attn_norm.weight")
        );
        assert_eq!(
            weights.layers[1].post_ffw_norm.name,
            "blk.1.post_ffw_norm.weight"
        );
    }

    #[test]
    fn dtype_ids_match_mainline_ggml() {
        assert_eq!(Dtype::from_ggml_id(0), Dtype::F32);
        assert_eq!(Dtype::from_ggml_id(12), Dtype::Q4K);
        assert_eq!(Dtype::from_ggml_id(13), Dtype::Q5K);
        assert_eq!(Dtype::from_ggml_id(14), Dtype::Q6K);
        assert_eq!(Dtype::from_ggml_id(30), Dtype::Bf16);
        assert_eq!(Dtype::from_ggml_id(99), Dtype::Unknown(99));
    }

    #[test]
    fn tensor_view_counts_elements() {
        let view = TensorView {
            name: "x".to_string(),
            dtype: Dtype::Q4K,
            shape: vec![6656, 202_048],
            data: Arc::from(Vec::new().into_boxed_slice()),
        };
        assert_eq!(view.elements(), 6656 * 202_048);
    }
}
