//! [`LlamaModel`]: config plus weights plus diag, and the forward
//! pass that ties them together.
//!
//! The forward is the architecture as both families run it, per
//! layer: embedding lookup, pre-attention RMSNorm, Q/K/V projections
//! (with the family's biases when loaded), RoPE on Q and K at each
//! position, the new K/V appended to the layer's cache, grouped-query
//! attention over everything the cache holds, output projection,
//! residual add, pre-FFN RMSNorm, SwiGLU FFN, residual add. Then the
//! final norm and the head (untied, or the embedding table when
//! tied). Arithmetic delegates to `torch-math`; attention and KV
//! storage are the `attention` crate's, so the cache is the only way
//! attention can see keys and values, by construction.
//!
//! Prefill and incremental decode are one path: [`forward`] takes
//! any number of tokens and appends them to the cache at the cache's
//! current position, [`step`] is the one-token case. The KV proof
//! test below runs a prompt both ways and requires the logits to
//! agree, which is the property the original engine's design (full
//! re-forward per token against a cache nothing called) could not
//! even state.
//!
//! Validation happens before any cache mutation: token range and
//! the context bound are checked up front, and every tensor shape
//! was checked at load, so a refused pass leaves the cache exactly
//! as it found it.
//!
//! [`forward`]: LlamaModel::forward
//! [`step`]: LlamaModel::step

use std::sync::Arc;

use attention::{gqa_attention, AttentionParams, KvCache, LayerKind};

use crate::config::LlamaConfig;
use crate::diag::{Diag, NullDiag};
use crate::error::{LlamaError, Result};
use crate::weights::{TensorData, Weights};

/// A llama-family or qwen2-family model ready to run.
pub struct LlamaModel {
    config: LlamaConfig,
    weights: Weights,
    diag: Arc<dyn Diag>,
}

impl LlamaModel {
    /// Assemble a model from its parts, with a silent diagnostic sink.
    pub fn new(config: LlamaConfig, weights: Weights) -> Self {
        LlamaModel {
            config,
            weights,
            diag: Arc::new(NullDiag),
        }
    }

    /// Assemble a model with a diagnostic sink attached. Every pass
    /// emits its begin and end through it.
    pub fn with_diag(config: LlamaConfig, weights: Weights, diag: Arc<dyn Diag>) -> Self {
        LlamaModel {
            config,
            weights,
            diag,
        }
    }

    /// The model's configuration.
    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }

    /// The model's weights.
    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    /// A fresh KV cache for this model: one global store per layer
    /// (both conformance families attend over the full history;
    /// neither file declares a sliding window).
    pub fn new_cache(&self) -> Result<KvCache> {
        let pattern = vec![LayerKind::Global; self.config.block_count as usize];
        Ok(KvCache::new(
            &pattern,
            self.config.kv_heads as usize,
            self.config.head_dim as usize,
        )?)
    }

    /// Run `tokens` through the model starting at the cache's
    /// current position, append their K/V to the cache, and return
    /// the next-token logits for the final position.
    ///
    /// On any refusal the cache is untouched (all validation happens
    /// before the first append) and the error names what failed.
    pub fn forward(&self, tokens: &[u32], cache: &mut KvCache) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(LlamaError::EmptyInput);
        }
        let cfg = &self.config;
        for &token in tokens {
            if token >= cfg.vocab_size {
                return Err(LlamaError::TokenOutOfRange {
                    token,
                    vocab_size: cfg.vocab_size,
                });
            }
        }
        let base = cache.next_position(0)?;
        for layer in 1..cfg.block_count as usize {
            let at = cache.next_position(layer)?;
            if at != base {
                return Err(LlamaError::CacheDesync {
                    layer,
                    expected: base,
                    actual: at,
                });
            }
        }
        let total = base + tokens.len();
        if total > cfg.context_length as usize {
            return Err(LlamaError::SequenceTooLong {
                total,
                context_length: cfg.context_length,
            });
        }
        self.diag.event(
            "forward.begin",
            &format!(
                "tokens={} position={} layers={}",
                tokens.len(),
                base,
                cfg.block_count
            ),
        );

        let rows = tokens.len();
        let hidden = cfg.hidden as usize;
        let mut x = Vec::with_capacity(rows * hidden);
        for &token in tokens {
            let row = token as usize * hidden;
            x.extend_from_slice(&self.weights.embedding.data[row..row + hidden]);
        }

        let positions: Vec<usize> = (base..base + rows).collect();
        for (index, w) in self.weights.layers.iter().enumerate() {
            let h = rms_rows(&x, hidden, &w.attn_norm.data, cfg.rms_eps)?;
            let mut q = linear(
                &h,
                hidden,
                &w.attn_q,
                cfg.q_width() as usize,
                w.attn_q_bias.as_ref(),
            )?;
            let mut k = linear(
                &h,
                hidden,
                &w.attn_k,
                cfg.k_width() as usize,
                w.attn_k_bias.as_ref(),
            )?;
            let v = linear(
                &h,
                hidden,
                &w.attn_v,
                cfg.v_width() as usize,
                w.attn_v_bias.as_ref(),
            )?;
            rope_rows(&mut q, cfg.q_heads as usize, cfg, &positions)?;
            rope_rows(&mut k, cfg.kv_heads as usize, cfg, &positions)?;
            let k_w = cfg.k_width() as usize;
            let v_w = cfg.v_width() as usize;
            for (r, &position) in positions.iter().enumerate() {
                cache.append(
                    index,
                    &k[r * k_w..(r + 1) * k_w],
                    &v[r * v_w..(r + 1) * v_w],
                    position,
                )?;
            }
            let view = cache.view(index)?;
            let attn = gqa_attention(
                &q,
                rows,
                cfg.q_heads as usize,
                &positions,
                &view,
                &AttentionParams::new(),
            )?;
            let out = linear(&attn, cfg.q_width() as usize, &w.attn_output, hidden, None)?;
            add_in_place(&mut x, &out);

            let h = rms_rows(&x, hidden, &w.ffn_norm.data, cfg.rms_eps)?;
            let gate = linear(&h, hidden, &w.ffn_gate, cfg.ffn_hidden as usize, None)?;
            let up = linear(&h, hidden, &w.ffn_up, cfg.ffn_hidden as usize, None)?;
            let act = torch_math::swiglu(&gate, &up)?;
            let down = linear(&act, cfg.ffn_hidden as usize, &w.ffn_down, hidden, None)?;
            add_in_place(&mut x, &down);
        }

        let last = &x[(rows - 1) * hidden..rows * hidden];
        let normed = torch_math::rms_norm(last, &self.weights.final_norm.data, cfg.rms_eps)?;
        let head = self.weights.head();
        let logits = torch_math::matvec(&head.data, cfg.vocab_size as usize, hidden, &normed)?;

        self.diag.event(
            "forward.end",
            &format!("logits={} position={}", logits.len(), total),
        );
        Ok(logits)
    }

    /// Advance the model by one token: the incremental decode step.
    /// Shares the cache path with [`forward`](Self::forward) exactly;
    /// it exists so call sites read as what they are.
    pub fn step(&self, token: u32, cache: &mut KvCache) -> Result<Vec<f32>> {
        self.forward(&[token], cache)
    }
}

/// RMSNorm applied row by row over a `[rows, width]` buffer.
fn rms_rows(x: &[f32], width: usize, weight: &[f32], eps: f32) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(width) {
        out.extend_from_slice(&torch_math::rms_norm(row, weight, eps)?);
    }
    Ok(out)
}

/// One linear projection over every row of a `[rows, in_w]` buffer:
/// `out = x * W^T (+ bias)`, with `weight` in GGUF stored layout
/// (`out_w` rows of `in_w`). Per-row matvec, deliberately: this is
/// the reference path the kernels are later verified against.
fn linear(
    x: &[f32],
    in_w: usize,
    weight: &TensorData,
    out_w: usize,
    bias: Option<&TensorData>,
) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(x.len() / in_w * out_w);
    for row in x.chunks_exact(in_w) {
        let mut projected = torch_math::matvec(&weight.data, out_w, in_w, row)?;
        if let Some(bias) = bias {
            for (o, b) in projected.iter_mut().zip(bias.data.iter()) {
                *o += b;
            }
        }
        out.extend_from_slice(&projected);
    }
    Ok(out)
}

/// RoPE applied in place to a `[rows, heads * head_dim]` buffer,
/// one position per row.
fn rope_rows(x: &mut [f32], heads: usize, cfg: &LlamaConfig, positions: &[usize]) -> Result<()> {
    let head_dim = cfg.head_dim as usize;
    let width = heads * head_dim;
    for (r, &position) in positions.iter().enumerate() {
        for h in 0..heads {
            let start = (r * heads + h) * head_dim;
            let rotated = torch_math::rope_partial(
                &x[start..start + head_dim],
                position,
                cfg.rope_freq_base,
                head_dim,
                cfg.rope_dim as usize,
            )?;
            x[start..start + head_dim].copy_from_slice(&rotated);
        }
    }
    debug_assert_eq!(x.len(), positions.len() * width);
    Ok(())
}

/// Residual add. Lengths are equal by construction (both operands
/// come from the same row shape), so this cannot fail.
fn add_in_place(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x += y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{build_gguf, refs, tiny_config, tiny_metadata, BiasMode, FakeSource};
    use gguf::GgufFile;

    fn tiny_model(arch: &str, tied: bool, biases: BiasMode) -> LlamaModel {
        let cfg = tiny_config(arch);
        let source = FakeSource::full(&cfg, tied, biases);
        let weights = Weights::load(&source, &cfg).expect("tiny weights load");
        LlamaModel::new(cfg, weights)
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn tiny_synthetic_model_runs_end_to_end() {
        let model = tiny_model("llama", false, BiasMode::None);
        let mut cache = model.new_cache().expect("cache");
        let logits = model
            .forward(&[3, 1, 4, 1], &mut cache)
            .expect("forward runs");
        assert_eq!(logits.len(), 64);
        assert!(logits.iter().all(|z| z.is_finite()));
        // The cache advanced by the prompt length on every layer.
        for stats in cache.stats() {
            assert_eq!(stats.appended, 4);
            assert_eq!(stats.stored, 4);
        }
    }

    #[test]
    fn kv_proof_prefill_and_incremental_decode_agree() {
        let model = tiny_model("qwen2", false, BiasMode::NonZero);
        let prompt = [5u32, 9, 2, 7, 3];

        let mut prefill_cache = model.new_cache().expect("cache");
        let prefill_logits = model
            .forward(&prompt, &mut prefill_cache)
            .expect("prefill runs");

        let mut decode_cache = model.new_cache().expect("cache");
        let mut decode_logits = Vec::new();
        for &token in &prompt {
            decode_logits = model.step(token, &mut decode_cache).expect("step runs");
        }

        let mut chunked_cache = model.new_cache().expect("cache");
        model
            .forward(&prompt[..2], &mut chunked_cache)
            .expect("chunk 1 runs");
        let chunked_logits = model
            .forward(&prompt[2..], &mut chunked_cache)
            .expect("chunk 2 runs");

        // Same math, three ways of feeding it: the logits for the
        // final position must agree to float noise. This is the KV
        // cache proof the old engine could not pass, having no
        // working cache on any path.
        assert!(
            max_abs_diff(&prefill_logits, &decode_logits) <= 1e-4,
            "prefill vs incremental diff {}",
            max_abs_diff(&prefill_logits, &decode_logits)
        );
        assert!(
            max_abs_diff(&prefill_logits, &chunked_logits) <= 1e-4,
            "prefill vs chunked diff {}",
            max_abs_diff(&prefill_logits, &chunked_logits)
        );
        assert_eq!(prefill_cache.next_position(0).expect("pos"), 5);
        assert_eq!(decode_cache.next_position(0).expect("pos"), 5);
    }

    #[test]
    fn tied_head_matches_an_explicit_copy_of_the_embedding() {
        let cfg = tiny_config("llama");
        let tied_source = FakeSource::full(&cfg, true, BiasMode::None);
        let tied_weights = Weights::load(&tied_source, &cfg).expect("tied loads");
        assert!(tied_weights.tied_embeddings);
        let tied_model = LlamaModel::new(cfg.clone(), tied_weights);

        // Same weights, but with output.weight present and holding
        // exactly the embedding table: the tied model's head.
        let mut explicit_source = FakeSource::full(&cfg, true, BiasMode::None);
        let embedding = explicit_source
            .get("token_embd.weight")
            .expect("embedding present")
            .clone();
        explicit_source.set(TensorData {
            name: "output.weight".to_string(),
            shape: embedding.shape.clone(),
            data: embedding.data.clone(),
        });
        let explicit_weights = Weights::load(&explicit_source, &cfg).expect("explicit loads");
        assert!(!explicit_weights.tied_embeddings);
        let explicit_model = LlamaModel::new(cfg, explicit_weights);

        let mut c1 = tied_model.new_cache().expect("cache");
        let mut c2 = explicit_model.new_cache().expect("cache");
        let a = tied_model.forward(&[1, 2, 3], &mut c1).expect("tied runs");
        let b = explicit_model
            .forward(&[1, 2, 3], &mut c2)
            .expect("explicit runs");
        assert!(
            max_abs_diff(&a, &b) <= 1e-5,
            "tied vs explicit-copy diff {}",
            max_abs_diff(&a, &b)
        );
    }

    #[test]
    fn attention_biases_change_the_output() {
        let plain = tiny_model("llama", false, BiasMode::None);
        let biased = tiny_model("llama", false, BiasMode::NonZero);
        let zeroed = tiny_model("llama", false, BiasMode::Zero);
        let prompt = [2u32, 6, 1];

        let mut c = plain.new_cache().expect("cache");
        let plain_logits = plain.forward(&prompt, &mut c).expect("plain runs");
        let mut c = biased.new_cache().expect("cache");
        let biased_logits = biased.forward(&prompt, &mut c).expect("biased runs");
        let mut c = zeroed.new_cache().expect("cache");
        let zeroed_logits = zeroed.forward(&prompt, &mut c).expect("zeroed runs");

        // Identical weights otherwise: the only difference between
        // the runs is the bias values, so a real difference here is
        // the biases being applied, and zero biases reproducing the
        // bias-free result proves presence alone does nothing.
        assert!(
            max_abs_diff(&plain_logits, &biased_logits) > 1e-3,
            "biases had no effect: diff {}",
            max_abs_diff(&plain_logits, &biased_logits)
        );
        assert!(
            max_abs_diff(&plain_logits, &zeroed_logits) <= 1e-6,
            "zero biases should be inert: diff {}",
            max_abs_diff(&plain_logits, &zeroed_logits)
        );
    }

    #[test]
    fn forward_refuses_bad_input_without_touching_the_cache() {
        let model = tiny_model("llama", false, BiasMode::None);
        let mut cache = model.new_cache().expect("cache");
        assert!(matches!(
            model.forward(&[], &mut cache),
            Err(LlamaError::EmptyInput)
        ));
        assert!(matches!(
            model.forward(&[64], &mut cache),
            Err(LlamaError::TokenOutOfRange { .. })
        ));
        let too_long = vec![1u32; 129];
        assert!(matches!(
            model.forward(&too_long, &mut cache),
            Err(LlamaError::SequenceTooLong { .. })
        ));
        assert_eq!(cache.next_position(0).expect("pos"), 0);
    }

    #[test]
    fn desynced_cache_is_refused_by_name() {
        let model = tiny_model("llama", false, BiasMode::None);
        let mut cache = model.new_cache().expect("cache");
        // Advance layer 0 only: a cache no honest forward produced.
        let row = vec![0.0f32; 2 * 8];
        cache.append(0, &row, &row, 0).expect("manual append");
        let err = model
            .forward(&[1], &mut cache)
            .expect_err("desync must fail");
        assert!(matches!(err, LlamaError::CacheDesync { layer: 1, .. }));
    }

    #[test]
    fn end_to_end_through_the_real_gguf_loader() {
        // The whole path on the real loader: synthetic GGUF bytes,
        // parsed by the gguf crate, config from the file, weights
        // through its TensorSource implementation, forward on top.
        let cfg = tiny_config("llama");
        let source = FakeSource::full(&cfg, false, BiasMode::None);
        let metadata = tiny_metadata("llama");
        let bytes = build_gguf(&refs(&metadata), &source.into_test_tensors());
        let file = GgufFile::from_bytes(bytes).expect("synthetic file parses");
        let parsed = LlamaConfig::from_file(&file).expect("config parses");
        assert_eq!(parsed, cfg);
        let weights = Weights::load(&file, &parsed).expect("weights load from file");
        let model = LlamaModel::new(parsed, weights);
        let mut cache = model.new_cache().expect("cache");
        let logits = model
            .forward(&[10, 20, 30], &mut cache)
            .expect("forward runs on file-backed weights");
        assert_eq!(logits.len(), 64);
        assert!(logits.iter().all(|z| z.is_finite()));
    }
}
