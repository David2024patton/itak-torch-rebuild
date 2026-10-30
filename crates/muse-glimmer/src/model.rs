//! [`GlimmerModel`]: config plus weights plus compute, and the forward
//! pass that ties them together.
//!
//! The forward is structured exactly as the architecture runs: a
//! parameter-free RMSNorm on the token embeddings, then per layer:
//! pre-attention norm, Q/K/V projections, per-head Q and K norms,
//! attention of the layer's kind, attention gate, output projection,
//! post-attention norm, residual add, pre-FFN norm, SwiGLU FFN,
//! post-FFN norm, residual add. Then the final norm, the untied
//! head, and the logit softcap and scale, which are real arithmetic
//! done here (they are a handful of scalar operations on the logit
//! vector, not kernel work).
//!
//! The norm placements and weight convention were settled by the
//! norm audit of 2026-10-29 (`NORM-AUDIT-2026-10-29.md` in this
//! crate): the post norms are sandwich norms, applied to each
//! sub-layer's output BEFORE its residual add (the checkpoint's own
//! config describes them as sitting between the sub-layer output and
//! the residual), and every stored norm weight is applied plainly,
//! as the effective gain the GGUF converter stored. One placement
//! remains marked for verification against the reference graph: the
//! attention gate projection reads the same normed input as the
//! Q/K/V projections.
//!
//! Input validation (non-empty, token range) is done here, for real,
//! before any compute step runs. Everything else delegates to
//! [`Compute`]; with [`UnimplementedCompute`] installed, the first step
//! refuses by name, which is the correct skeleton behavior.

use std::sync::Arc;

use crate::cache::Cache;
use crate::compute::{AttentionRequest, Compute};
use crate::config::GlimmerConfig;
use crate::diag::{Diag, NullDiag};
use crate::error::{GlimmerError, Result};
use crate::weights::Weights;

/// A Muse Glimmer model ready (structurally) to run.
pub struct GlimmerModel {
    config: GlimmerConfig,
    weights: Weights,
    compute: Box<dyn Compute>,
    diag: Arc<dyn Diag>,
}

impl GlimmerModel {
    /// Assemble a model from its parts, with a silent diagnostic sink.
    pub fn new(config: GlimmerConfig, weights: Weights, compute: Box<dyn Compute>) -> Self {
        GlimmerModel {
            config,
            weights,
            compute,
            diag: Arc::new(NullDiag),
        }
    }

    /// Assemble a model with a diagnostic sink attached. Every refusal
    /// and every completed pass emits through it.
    pub fn with_diag(
        config: GlimmerConfig,
        weights: Weights,
        compute: Box<dyn Compute>,
        diag: Arc<dyn Diag>,
    ) -> Self {
        GlimmerModel {
            config,
            weights,
            compute,
            diag,
        }
    }

    /// The model's configuration.
    pub fn config(&self) -> &GlimmerConfig {
        &self.config
    }

    /// Run one forward pass over `tokens` and return the next-token
    /// logits for the final position, softcapped and scaled per the
    /// file's own `final_logit_softcapping` and `logit_scale`.
    ///
    /// On success the cache advances by `tokens.len()`; on any refusal
    /// the cache is untouched and the error names the step or input
    /// that failed.
    pub fn forward(&self, tokens: &[u32], cache: &mut dyn Cache) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(GlimmerError::EmptyInput);
        }
        for &token in tokens {
            if token >= self.config.vocab_size {
                return Err(GlimmerError::TokenOutOfRange {
                    token,
                    vocab_size: self.config.vocab_size,
                });
            }
        }
        self.diag.event(
            "forward.begin",
            &format!(
                "tokens={} position={} layers={}",
                tokens.len(),
                cache.position(),
                self.config.block_count
            ),
        );

        let cfg = &self.config;
        let rows = tokens.len();
        let c = &*self.compute;

        // The architecture RMS-normalizes the token embeddings
        // (parameter-free: no weight tensor exists for this step).
        let embedded = c.embed(&self.weights.embedding, tokens, cfg)?;
        let mut x = c.rms_norm_scaleless(&embedded, rows, cfg.rms_eps)?;

        for (index, (kind, w)) in cfg
            .layers
            .iter()
            .zip(self.weights.layers.iter())
            .enumerate()
        {
            let residual = x;

            let h = c.rms_norm(&residual, &w.attn_norm, rows, cfg.rms_eps)?;
            let q = c.proj(&h, &w.attn_q, rows)?;
            let k = c.proj(&h, &w.attn_k, rows)?;
            let v = c.proj(&h, &w.attn_v, rows)?;
            // Q/K norms: the checkpoint definition is scaleless with
            // a scale factor on Q (3.87); the file materializes that
            // as constant weight tensors (attn_q_norm all 3.87,
            // attn_k_norm all 1.0, measured in integration run B),
            // so the plain weighted norm below IS the definition,
            // with no special case. See the norm audit.
            let q = c.rms_norm(&q, &w.attn_q_norm, rows * cfg.q_heads as usize, cfg.rms_eps)?;
            let k = c.rms_norm(
                &k,
                &w.attn_k_norm,
                rows * cfg.kv_heads as usize,
                cfg.rms_eps,
            )?;
            let attn = c.attention(
                &AttentionRequest {
                    q: &q,
                    k: &k,
                    v: &v,
                    layer: index,
                    kind: *kind,
                    cfg,
                },
                cache,
            )?;
            let gate = c.proj(&h, &w.attn_gate, rows)?;
            let attn = c.attn_gate(&attn, &gate)?;
            let out = c.proj(&attn, &w.attn_output, rows)?;
            // Sandwich post norm: on the attention output, BEFORE
            // the residual add (norm audit, 2026-10-29).
            let out = c.rms_norm(&out, &w.post_attention_norm, rows, cfg.rms_eps)?;
            let after_attn = c.add(&residual, &out)?;

            let h = c.rms_norm(&after_attn, &w.ffn_norm, rows, cfg.rms_eps)?;
            let ffn = c.swiglu_ffn(&h, w, rows, cfg)?;
            // Sandwich post norm: on the FFN output, BEFORE the
            // residual add.
            let ffn = c.rms_norm(&ffn, &w.post_ffw_norm, rows, cfg.rms_eps)?;
            x = c.add(&after_attn, &ffn)?;
        }

        let hidden = cfg.hidden as usize;
        let last = &x[(rows - 1) * hidden..rows * hidden];
        let normed = c.rms_norm(last, &self.weights.final_norm, 1, cfg.rms_eps)?;
        let mut logits = c.lm_head(&normed, &self.weights.output, cfg)?;
        softcap_scale(&mut logits, cfg.logit_softcap, cfg.logit_scale);

        cache.advance(tokens.len());
        self.diag.event(
            "forward.end",
            &format!("logits={} position={}", logits.len(), cache.position()),
        );
        Ok(logits)
    }
}

/// The file's logit finishing, applied in place: softcap
/// `cap * tanh(z / cap)` when the cap is positive, then multiply by the
/// logit scale. Plain scalar math on the logit vector, implemented here
/// rather than delegated, and unit tested below.
pub fn softcap_scale(logits: &mut [f32], cap: f32, scale: f32) {
    if cap > 0.0 {
        for z in logits.iter_mut() {
            *z = scale * cap * (*z / cap).tanh();
        }
    } else {
        for z in logits.iter_mut() {
            *z *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::PositionCache;
    use crate::compute::UnimplementedCompute;
    use crate::metadata::Metadata;
    use crate::testkit::tiny_metadata;
    use crate::weights::{Dtype, TensorSource, TensorView};
    use std::cell::RefCell;
    use std::collections::HashMap;

    /// Source that hands out a dummy tensor for any expected name.
    struct AnySource {
        names: HashMap<String, ()>,
    }

    impl TensorSource for AnySource {
        fn tensor(&self, name: &str) -> Result<TensorView> {
            if self.names.contains_key(name) {
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

    fn tiny_model(compute: Box<dyn Compute>) -> (GlimmerModel, GlimmerConfig) {
        let meta = tiny_metadata();
        let cfg = GlimmerConfig::from_metadata(&meta).expect("tiny config");
        let names = crate::weights::expected_tensor_names(&cfg)
            .into_iter()
            .map(|n| (n, ()))
            .collect();
        let weights = Weights::load(&AnySource { names }, &cfg).expect("tiny weights");
        (GlimmerModel::new(cfg.clone(), weights, compute), cfg)
    }

    /// A compute stub that records its call order and returns zeros of
    /// plausible length. It verifies the forward's STRUCTURE: which
    /// steps run, in what order, for a two-layer model. It asserts
    /// nothing about numerics, which are the kernel crate's to prove.
    /// The call log is shared with the test through an `Rc` so the test
    /// can read it after the stub moves into the model.
    struct RecordingCompute {
        calls: std::rc::Rc<RefCell<Vec<&'static str>>>,
    }

    impl RecordingCompute {
        fn record(&self, step: &'static str) {
            self.calls.borrow_mut().push(step);
        }
    }

    impl Compute for RecordingCompute {
        fn embed(
            &self,
            _table: &TensorView,
            tokens: &[u32],
            cfg: &GlimmerConfig,
        ) -> Result<Vec<f32>> {
            self.record("embed");
            Ok(vec![0.0; tokens.len() * cfg.hidden as usize])
        }

        fn rms_norm(
            &self,
            x: &[f32],
            _weight: &TensorView,
            _rows: usize,
            _eps: f32,
        ) -> Result<Vec<f32>> {
            self.record("rms_norm");
            Ok(vec![0.0; x.len()])
        }

        fn rms_norm_scaleless(&self, x: &[f32], _rows: usize, _eps: f32) -> Result<Vec<f32>> {
            self.record("rms_norm_scaleless");
            Ok(vec![0.0; x.len()])
        }

        fn proj(&self, x: &[f32], _weight: &TensorView, _rows: usize) -> Result<Vec<f32>> {
            self.record("proj");
            Ok(vec![0.0; x.len()])
        }

        fn attention(
            &self,
            req: &AttentionRequest<'_>,
            _cache: &mut dyn Cache,
        ) -> Result<Vec<f32>> {
            self.record("attention");
            Ok(vec![0.0; req.q.len()])
        }

        fn attn_gate(&self, attn_out: &[f32], _gate: &[f32]) -> Result<Vec<f32>> {
            self.record("attn_gate");
            Ok(attn_out.to_vec())
        }

        fn add(&self, a: &[f32], _b: &[f32]) -> Result<Vec<f32>> {
            self.record("add");
            Ok(a.to_vec())
        }

        fn swiglu_ffn(
            &self,
            x: &[f32],
            _w: &crate::weights::LayerWeights,
            _rows: usize,
            _cfg: &GlimmerConfig,
        ) -> Result<Vec<f32>> {
            self.record("swiglu_ffn");
            Ok(vec![0.0; x.len()])
        }

        fn lm_head(&self, _x: &[f32], _head: &TensorView, cfg: &GlimmerConfig) -> Result<Vec<f32>> {
            self.record("lm_head");
            Ok(vec![0.0; cfg.vocab_size as usize])
        }
    }

    #[test]
    fn unimplemented_compute_refuses_at_embed_by_name() {
        let (model, _cfg) = tiny_model(Box::new(UnimplementedCompute));
        let mut cache = PositionCache::new();
        let err = model
            .forward(&[1, 2, 3], &mut cache)
            .expect_err("skeleton compute must refuse");
        assert_eq!(
            err.to_string(),
            "forward step is not implemented yet: embed"
        );
        assert!(matches!(err, GlimmerError::Unimplemented { step: "embed" }));
        // A refused pass must not advance the cache.
        assert_eq!(cache.position(), 0);
    }

    #[test]
    fn forward_validates_input_before_any_compute() {
        let (model, cfg) = tiny_model(Box::new(UnimplementedCompute));
        let mut cache = PositionCache::new();
        assert!(matches!(
            model.forward(&[], &mut cache),
            Err(GlimmerError::EmptyInput)
        ));
        assert!(matches!(
            model.forward(&[cfg.vocab_size], &mut cache),
            Err(GlimmerError::TokenOutOfRange { .. })
        ));
        let meta_vocab = tiny_metadata().get_array_len("tokenizer.ggml.tokens");
        assert_eq!(meta_vocab, Some(128));
    }

    #[test]
    fn forward_walks_the_full_layer_structure_in_order() {
        let calls = std::rc::Rc::new(RefCell::new(Vec::new()));
        let compute = Box::new(RecordingCompute {
            calls: calls.clone(),
        });
        let (model, _cfg) = tiny_model(compute);
        let mut cache = PositionCache::new();
        let logits = model
            .forward(&[1, 2], &mut cache)
            .expect("recording compute completes the walk");
        assert_eq!(logits.len(), 128);
        assert!(logits.iter().all(|z| *z == 0.0));
        assert_eq!(cache.position(), 2);

        let one_layer: [&str; 16] = [
            "rms_norm", // attn pre-norm
            "proj",     // q
            "proj",     // k
            "proj",     // v
            "rms_norm", // q norm
            "rms_norm", // k norm
            "attention",
            "proj", // gate
            "attn_gate",
            "proj",     // output
            "rms_norm", // post-attention norm (sandwich: before the add)
            "add",
            "rms_norm", // ffn pre-norm
            "swiglu_ffn",
            "rms_norm", // post-ffn norm (sandwich: before the add)
            "add",
        ];
        let mut expected: Vec<&str> = vec!["embed", "rms_norm_scaleless"];
        expected.extend(one_layer);
        expected.extend(one_layer);
        expected.extend(["rms_norm", "lm_head"]);
        assert_eq!(calls.borrow().as_slice(), expected.as_slice());
    }

    #[test]
    fn softcap_scale_matches_the_formula() {
        let mut logits = vec![0.0, 20.0, -20.0, 1000.0];
        softcap_scale(&mut logits, 20.0, 0.5);
        assert_eq!(logits[0], 0.0);
        // 0.5 * 20 * tanh(1) = 7.6159416
        assert!((logits[1] - 7.615_941_6).abs() < 1e-5);
        assert!((logits[2] + 7.615_941_6).abs() < 1e-5);
        // Saturates at scale * cap = 10.
        assert!((logits[3] - 10.0).abs() < 1e-4);
    }

    #[test]
    fn softcap_scale_with_no_cap_just_scales() {
        let mut logits = vec![2.0, -4.0];
        softcap_scale(&mut logits, 0.0, 0.25);
        assert_eq!(logits, vec![0.5, -1.0]);
    }
}
