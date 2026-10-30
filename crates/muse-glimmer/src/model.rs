//! [`GlimmerModel`]: config plus weights plus compute, and the forward
//! pass that ties them together.
//!
//! The forward follows the reference graph in llama.cpp
//! (src/models/muse-glimmer.cpp); its conventions were settled from
//! that source in the Run B follow-up (see
//! crates/itak-loader/INTEGRATION-RUN-B.md):
//!
//! - Embedding rows pass a WEIGHTLESS RMSNorm (eps = the file's)
//!   before layer 0; there is no embedding scale factor.
//! - Per layer: pre-attention norm, Q/K/V projections, per-head Q
//!   and K norms, attention of the layer's kind, and the sigmoid
//!   attention gate, whose projection reads the same normed input
//!   as Q/K/V and which multiplies the attention output before the
//!   output projection.
//! - Each sub-block's output passes its POST norm (eps hardcoded
//!   1e-8 in the reference graph) BEFORE the residual add:
//!   `x = x + post_norm(branch)`. The residual stream itself is
//!   never renormalized.
//! - Every stored-norm multiplication is plain `x * w`. The
//!   Gemma-style (1 + w) shift was folded into the four per-layer
//!   norm weights at conversion time (conversion/muse_glimmer.py:
//!   `norm_shift`); the final norm was deliberately left unshifted,
//!   which is why its stored weights center on zero.
//! - RoPE runs on the sliding-window layers only, at the SWA base,
//!   in ggml's NORM (interleaved, adjacent-pair) mode; the global
//!   layers are NoPE.
//! - The head is: final norm, untied lm_head, multiply by
//!   `logit_scale`, then softcap `cap * tanh(z / cap)`.
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

/// Epsilon for the two post norms (post-attention, post-FFN). The
/// reference graph hardcodes 1e-8 for these, independent of the
/// file's RMS epsilon used by every other norm
/// (src/models/muse-glimmer.cpp: `post_norm_eps`).
pub(crate) const POST_NORM_EPS: f32 = 1e-8;

/// The residual stream entering one layer during a forward pass,
/// captured by the captured-states forward variants
/// ([`GlimmerModel::forward_capturing`] and
/// [`GlimmerModel::forward_block_capturing`]).
///
/// `rows` is row-major `[tokens][hidden]`: the exact values the
/// layer's pre-attention norm read during this pass, which for
/// layer 0 is the embedding output after its weightless RMSNorm and
/// for every later layer is the previous layer's output. This is
/// the tensor llama.cpp's target graph exposes as `t_layer_inp`
/// (src/models/muse-glimmer.cpp) and the conditioning input the
/// DFlash draft consumes at the layers its own metadata names
/// (SPEC-itak-spec Q1, settled 2026-10-30).
#[derive(Clone, Debug, PartialEq)]
pub struct LayerInputCapture {
    /// The layer whose input this is.
    pub layer: usize,
    /// Row-major `[tokens][hidden]` residual rows entering `layer`.
    pub rows: Vec<f32>,
}

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

    /// The input validation both forward entry points share, run
    /// for real before any compute step: non-empty, every token in
    /// vocabulary.
    fn validate_tokens(&self, tokens: &[u32]) -> Result<()> {
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
        Ok(())
    }

    /// Run one forward pass over `tokens` and return the next-token
    /// logits for the final position, softcapped and scaled per the
    /// file's own `final_logit_softcapping` and `logit_scale`.
    ///
    /// On success the cache advances by `tokens.len()`; on any refusal
    /// the cache is untouched and the error names the step or input
    /// that failed.
    pub fn forward(&self, tokens: &[u32], cache: &mut dyn Cache) -> Result<Vec<f32>> {
        self.validate_tokens(tokens)?;
        self.diag.event(
            "forward.begin",
            &format!(
                "tokens={} position={} layers={}",
                tokens.len(),
                cache.position(),
                self.config.block_count
            ),
        );

        let (x, _) = self.forward_rows(tokens, cache, &[])?;
        let logits = self.finish_last(&x, tokens.len())?;

        cache.advance(tokens.len());
        self.diag.event(
            "forward.end",
            &format!("logits={} position={}", logits.len(), cache.position()),
        );
        Ok(logits)
    }

    /// [`GlimmerModel::forward`] plus captured layer inputs: runs the
    /// identical pass and additionally returns, for each layer index
    /// in `capture_layers`, the residual rows that entered that
    /// layer ([`LayerInputCapture`], in ascending layer order).
    ///
    /// Capture is additive by construction: the layer loop's math,
    /// step order, and cache contract are `forward`'s, and the only
    /// extra work is cloning the residual at the named layers. The
    /// speculation stack's DFlash proposer consumes these rows
    /// (SPEC-itak-spec Q1); nothing in the verified forward path
    /// depends on them.
    ///
    /// A capture index at or past the layer count is refused with
    /// [`GlimmerError::CaptureLayerOutOfRange`] before any compute
    /// runs, and (like every refusal) leaves the cache untouched.
    pub fn forward_capturing(
        &self,
        tokens: &[u32],
        cache: &mut dyn Cache,
        capture_layers: &[usize],
    ) -> Result<(Vec<f32>, Vec<LayerInputCapture>)> {
        self.validate_tokens(tokens)?;
        self.validate_capture(capture_layers)?;
        self.diag.event(
            "forward_capturing.begin",
            &format!(
                "tokens={} position={} capture={capture_layers:?}",
                tokens.len(),
                cache.position()
            ),
        );

        let (x, captures) = self.forward_rows(tokens, cache, capture_layers)?;
        let logits = self.finish_last(&x, tokens.len())?;

        cache.advance(tokens.len());
        self.diag.event(
            "forward_capturing.end",
            &format!(
                "logits={} captured={} position={}",
                logits.len(),
                captures.len(),
                cache.position()
            ),
        );
        Ok((logits, captures))
    }

    /// Block forward with per-position logits (SPEC-itak-spec
    /// Section 3, dependency E1): run `tokens` through the same layer
    /// loop [`GlimmerModel::forward`] runs, and return the finished
    /// logits for EVERY position in the block, row-major
    /// `[block][vocab]`, each row softcapped and scaled exactly as
    /// `forward` finishes its single row. Row `i` is the distribution
    /// for the token following the prefix extended by
    /// `tokens[..=i]`, which is what a speculative verify round
    /// consumes.
    ///
    /// This is not a second forward: `forward` is this method's
    /// block-of-one case read at the last row (plus the positions the
    /// block already consumed). The cache contract is `forward`'s:
    /// on success the cache advances by `tokens.len()`; on any
    /// refusal the cache is untouched and the error names the step
    /// or input that failed.
    pub fn forward_block(&self, tokens: &[u32], cache: &mut dyn Cache) -> Result<Vec<f32>> {
        self.validate_tokens(tokens)?;
        self.diag.event(
            "forward_block.begin",
            &format!(
                "tokens={} position={} layers={}",
                tokens.len(),
                cache.position(),
                self.config.block_count
            ),
        );

        let (x, _) = self.forward_rows(tokens, cache, &[])?;
        let out = self.finish_block(&x, tokens.len())?;

        cache.advance(tokens.len());
        self.diag.event(
            "forward_block.end",
            &format!(
                "rows={} logits={} position={}",
                tokens.len(),
                out.len(),
                cache.position()
            ),
        );
        Ok(out)
    }

    /// [`GlimmerModel::forward_block`] plus captured layer inputs:
    /// the block variant of [`GlimmerModel::forward_capturing`], for
    /// a verify round that also feeds a DFlash proposer. Same
    /// additive-capture contract and same refusal rules.
    pub fn forward_block_capturing(
        &self,
        tokens: &[u32],
        cache: &mut dyn Cache,
        capture_layers: &[usize],
    ) -> Result<(Vec<f32>, Vec<LayerInputCapture>)> {
        self.validate_tokens(tokens)?;
        self.validate_capture(capture_layers)?;
        self.diag.event(
            "forward_block_capturing.begin",
            &format!(
                "tokens={} position={} capture={capture_layers:?}",
                tokens.len(),
                cache.position()
            ),
        );

        let (x, captures) = self.forward_rows(tokens, cache, capture_layers)?;
        let out = self.finish_block(&x, tokens.len())?;

        cache.advance(tokens.len());
        self.diag.event(
            "forward_block_capturing.end",
            &format!(
                "rows={} logits={} captured={} position={}",
                tokens.len(),
                out.len(),
                captures.len(),
                cache.position()
            ),
        );
        Ok((out, captures))
    }

    /// Every capture index must name a real layer. Checked before
    /// any compute runs so a bad request cannot consume cache
    /// positions or backend state.
    fn validate_capture(&self, capture_layers: &[usize]) -> Result<()> {
        for &layer in capture_layers {
            if layer >= self.config.block_count as usize {
                return Err(GlimmerError::CaptureLayerOutOfRange {
                    layer,
                    block_count: self.config.block_count,
                });
            }
        }
        Ok(())
    }

    /// The head application `forward` performs: final norm and
    /// lm_head over the LAST row of the final hidden states, then
    /// the file's softcap and scale.
    fn finish_last(&self, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let c = &*self.compute;
        let hidden = cfg.hidden as usize;
        let last = &x[(rows - 1) * hidden..rows * hidden];
        let normed = c.rms_norm(last, &self.weights.final_norm, 1, cfg.rms_eps)?;
        let mut logits = c.lm_head(&normed, &self.weights.output, cfg)?;
        softcap_scale(&mut logits, cfg.logit_softcap, cfg.logit_scale);
        Ok(logits)
    }

    /// The head application `forward_block` performs: final norm,
    /// lm_head, softcap and scale for EVERY row, row-major
    /// `[rows][vocab]`.
    fn finish_block(&self, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let c = &*self.compute;
        let hidden = cfg.hidden as usize;
        let vocab = cfg.vocab_size as usize;
        let mut out = Vec::with_capacity(rows * vocab);
        for r in 0..rows {
            let row = &x[r * hidden..(r + 1) * hidden];
            let normed = c.rms_norm(row, &self.weights.final_norm, 1, cfg.rms_eps)?;
            let mut logits = c.lm_head(&normed, &self.weights.output, cfg)?;
            softcap_scale(&mut logits, cfg.logit_softcap, cfg.logit_scale);
            out.extend_from_slice(&logits);
        }
        Ok(out)
    }

    /// The shared body of all forward entry points: embedding,
    /// the weightless embedding norm, and the per-layer loop,
    /// returning the final hidden states `[tokens.len(), hidden]`
    /// plus any requested layer-input captures.
    /// Reads and appends KV through `cache` (via the compute
    /// backend's attention step) but does not advance the position
    /// contract: the entry points advance it once their head
    /// application has succeeded, so a refused pass leaves the cache
    /// untouched.
    ///
    /// Capture is a clone of the residual at the top of the named
    /// layers' iterations, taken before that layer's pre-attention
    /// norm reads it. It performs no compute calls, so a capturing
    /// pass makes exactly the step sequence a plain pass makes.
    fn forward_rows(
        &self,
        tokens: &[u32],
        cache: &mut dyn Cache,
        capture_layers: &[usize],
    ) -> Result<(Vec<f32>, Vec<LayerInputCapture>)> {
        let cfg = &self.config;
        let rows = tokens.len();
        let c = &*self.compute;
        let mut captures: Vec<LayerInputCapture> = Vec::new();

        let mut x = c.embed(&self.weights.embedding, tokens, cfg)?;
        // The reference graph normalizes the embedding output with a
        // weightless RMSNorm before the first layer
        // (src/models/muse-glimmer.cpp: build_norm with no weight).
        x = c.rms_norm_plain(&x, rows, cfg.rms_eps)?;

        for (index, (kind, w)) in cfg
            .layers
            .iter()
            .zip(self.weights.layers.iter())
            .enumerate()
        {
            if capture_layers.contains(&index) {
                captures.push(LayerInputCapture {
                    layer: index,
                    rows: x.clone(),
                });
            }
            let residual = x;

            let h = c.rms_norm(&residual, &w.attn_norm, rows, cfg.rms_eps)?;
            let q = c.proj(&h, &w.attn_q, rows)?;
            let k = c.proj(&h, &w.attn_k, rows)?;
            let v = c.proj(&h, &w.attn_v, rows)?;
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
            // Post norm on the branch output, THEN the residual add:
            // the reference graph normalizes the sub-block output
            // (hardcoded eps 1e-8) and adds the untouched residual.
            let out = c.rms_norm(&out, &w.post_attention_norm, rows, POST_NORM_EPS)?;
            let after_attn = c.add(&residual, &out)?;

            let residual = after_attn;
            let h = c.rms_norm(&residual, &w.ffn_norm, rows, cfg.rms_eps)?;
            let ffn = c.swiglu_ffn(&h, w, rows, cfg)?;
            let ffn = c.rms_norm(&ffn, &w.post_ffw_norm, rows, POST_NORM_EPS)?;
            x = c.add(&residual, &ffn)?;
        }

        Ok((x, captures))
    }
}

/// The file's logit finishing, applied in place: multiply by the
/// logit scale FIRST, then softcap `cap * tanh(z / cap)` when the cap
/// is positive. The reference graph applies them in exactly this
/// order (src/models/muse-glimmer.cpp: `ggml_scale`, then
/// `build_softcap`). The order is invisible to argmax (both maps are
/// monotone) but changes every probability. Plain scalar math on the
/// logit vector, implemented here rather than delegated, and unit
/// tested below.
pub fn softcap_scale(logits: &mut [f32], cap: f32, scale: f32) {
    for z in logits.iter_mut() {
        *z *= scale;
        if cap > 0.0 {
            *z = cap * (*z / cap).tanh();
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

        fn rms_norm_plain(&self, x: &[f32], _rows: usize, _eps: f32) -> Result<Vec<f32>> {
            self.record("rms_norm_plain");
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
            "rms_norm", // post-attention norm (branch, before the add)
            "add",
            "rms_norm", // ffn pre-norm
            "swiglu_ffn",
            "rms_norm", // post-ffn norm (branch, before the add)
            "add",
        ];
        let mut expected: Vec<&str> = vec!["embed", "rms_norm_plain"];
        expected.extend(one_layer);
        expected.extend(one_layer);
        expected.extend(["rms_norm", "lm_head"]);
        assert_eq!(calls.borrow().as_slice(), expected.as_slice());
    }

    #[test]
    fn forward_block_heads_every_row_through_the_same_loop() {
        let calls = std::rc::Rc::new(RefCell::new(Vec::new()));
        let compute = Box::new(RecordingCompute {
            calls: calls.clone(),
        });
        let (model, _cfg) = tiny_model(compute);
        let mut cache = PositionCache::new();
        let logits = model
            .forward_block(&[1, 2], &mut cache)
            .expect("recording compute completes the block walk");
        // [block][vocab] row-major: 2 rows of 128.
        assert_eq!(logits.len(), 2 * 128);
        assert_eq!(cache.position(), 2);

        // The layer walk is forward's, unchanged; what the block
        // adds is one (final norm, lm_head) pair per row, after it.
        let recorded = calls.borrow();
        let walk = &recorded[..recorded.len() - 4];
        let one_layer: [&str; 16] = [
            "rms_norm",
            "proj",
            "proj",
            "proj",
            "rms_norm",
            "rms_norm",
            "attention",
            "proj",
            "attn_gate",
            "proj",
            "rms_norm",
            "add",
            "rms_norm",
            "swiglu_ffn",
            "rms_norm",
            "add",
        ];
        let mut expected_walk: Vec<&str> = vec!["embed", "rms_norm_plain"];
        expected_walk.extend(one_layer);
        expected_walk.extend(one_layer);
        assert_eq!(walk, expected_walk.as_slice());
        assert_eq!(
            &recorded[recorded.len() - 4..],
            ["rms_norm", "lm_head", "rms_norm", "lm_head"].as_slice()
        );
    }

    #[test]
    fn forward_block_of_one_returns_forward_shape() {
        let calls = std::rc::Rc::new(RefCell::new(Vec::new()));
        let compute = Box::new(RecordingCompute {
            calls: calls.clone(),
        });
        let (model, _cfg) = tiny_model(compute);
        let mut cache = PositionCache::new();
        let block = model
            .forward_block(&[7], &mut cache)
            .expect("block of one runs");
        assert_eq!(block.len(), 128);
        assert_eq!(cache.position(), 1);
    }

    #[test]
    fn forward_block_refuses_like_forward() {
        let (model, cfg) = tiny_model(Box::new(UnimplementedCompute));
        let mut cache = PositionCache::new();
        assert!(matches!(
            model.forward_block(&[], &mut cache),
            Err(GlimmerError::EmptyInput)
        ));
        assert!(matches!(
            model.forward_block(&[cfg.vocab_size], &mut cache),
            Err(GlimmerError::TokenOutOfRange { .. })
        ));
        let err = model
            .forward_block(&[1, 2, 3], &mut cache)
            .expect_err("skeleton compute must refuse");
        assert!(matches!(err, GlimmerError::Unimplemented { step: "embed" }));
        assert_eq!(cache.position(), 0);
    }

    #[test]
    fn capturing_forward_makes_the_identical_step_sequence() {
        // Plain forward on one recording stub...
        let plain_calls = std::rc::Rc::new(RefCell::new(Vec::new()));
        let (model, _cfg) = tiny_model(Box::new(RecordingCompute {
            calls: plain_calls.clone(),
        }));
        let mut cache = PositionCache::new();
        let plain_logits = model.forward(&[1, 2], &mut cache).expect("plain forward");
        assert_eq!(cache.position(), 2);

        // ...and the capturing variant on a second one. Capture may
        // clone residuals; it may not add, drop, or reorder a single
        // compute step, and its logits and cache advance must match.
        let cap_calls = std::rc::Rc::new(RefCell::new(Vec::new()));
        let (model, cfg) = tiny_model(Box::new(RecordingCompute {
            calls: cap_calls.clone(),
        }));
        let mut cache = PositionCache::new();
        let (logits, captures) = model
            .forward_capturing(&[1, 2], &mut cache, &[1, 0])
            .expect("capturing forward");
        assert_eq!(logits, plain_logits);
        assert_eq!(cache.position(), 2);
        assert_eq!(
            cap_calls.borrow().as_slice(),
            plain_calls.borrow().as_slice()
        );
        // Captures come back in ascending layer order regardless of
        // request order, one per requested layer, each holding the
        // full [tokens][hidden] residual block.
        assert_eq!(captures.len(), 2);
        assert_eq!(captures[0].layer, 0);
        assert_eq!(captures[1].layer, 1);
        for cap in &captures {
            assert_eq!(cap.rows.len(), 2 * cfg.hidden as usize);
        }
    }

    #[test]
    fn capturing_block_forward_captures_and_heads_every_row() {
        let calls = std::rc::Rc::new(RefCell::new(Vec::new()));
        let (model, _cfg) = tiny_model(Box::new(RecordingCompute {
            calls: calls.clone(),
        }));
        let mut cache = PositionCache::new();
        let (logits, captures) = model
            .forward_block_capturing(&[1, 2], &mut cache, &[0])
            .expect("capturing block forward");
        assert_eq!(logits.len(), 2 * 128);
        assert_eq!(captures.len(), 1);
        assert_eq!(captures[0].layer, 0);
        assert_eq!(cache.position(), 2);
    }

    #[test]
    fn capture_out_of_range_refuses_before_any_compute() {
        let calls = std::rc::Rc::new(RefCell::new(Vec::new()));
        let (model, _cfg) = tiny_model(Box::new(RecordingCompute {
            calls: calls.clone(),
        }));
        let mut cache = PositionCache::new();
        // The tiny model has 2 layers; layer 2 does not exist.
        let err = model
            .forward_capturing(&[1], &mut cache, &[2])
            .expect_err("layer 2 of 2 must refuse");
        assert!(matches!(
            err,
            GlimmerError::CaptureLayerOutOfRange {
                layer: 2,
                block_count: 2
            }
        ));
        assert!(calls.borrow().is_empty(), "no compute step may run");
        assert_eq!(cache.position(), 0);
    }

    #[test]
    fn softcap_scale_matches_the_formula() {
        let mut logits = vec![0.0, 20.0, -20.0, 1000.0];
        softcap_scale(&mut logits, 20.0, 0.5);
        assert_eq!(logits[0], 0.0);
        // Scale first, then cap: 20 * tanh(0.5 * 20 / 20)
        // = 20 * tanh(0.5) = 9.2423431. The opposite order would
        // give 0.5 * 20 * tanh(1) = 7.6159416, so this value tells
        // the two conventions apart.
        assert!((logits[1] - 9.242_343).abs() < 1e-5);
        assert!((logits[2] + 9.242_343).abs() < 1e-5);
        // Saturates at the cap itself, 20, because the scale is
        // applied before the cap, not after it.
        assert!((logits[3] - 20.0).abs() < 1e-4);
    }

    #[test]
    fn softcap_scale_with_no_cap_just_scales() {
        let mut logits = vec![2.0, -4.0];
        softcap_scale(&mut logits, 0.0, 0.25);
        assert_eq!(logits, vec![0.5, -1.0]);
    }
}
