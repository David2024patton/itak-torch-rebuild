//! [`DflashModel`]: config plus weights, and the draft's two passes.
//!
//! The passes follow the reference graph in llama.cpp
//! (src/models/dflash.cpp) and the speculative driver that feeds it
//! (common/speculative.cpp, `common_speculative_impl_draft_dflash`);
//! the wiring was settled from that source in the E3 research pass
//! and is recorded in SPEC-itak-spec Q1:
//!
//! - **Injection** ([`DflashModel::inject`]): the target's
//!   layer-input rows for the layers `dflash.target_layers` names,
//!   concatenated per position, are fused by the `fc` projection
//!   and an RMSNorm (`enc.output_norm.weight`). Per draft layer the
//!   fused rows are projected to K and V; K gets the per-head
//!   RMSNorm and NEOX RoPE at the target position, and the pair is
//!   stored in the draft's own KV cache at that position. Q is not
//!   computed and the FFN does not run in this pass.
//! - **Drafting** ([`DflashModel::forward_block`]): a noise block
//!   (anchor token, then mask tokens) is embedded with the TARGET
//!   model's embedding table (this file carries none of its own)
//!   and run through the draft layers. Attention reads the injected
//!   committed K/V plus the block's own K/V, non-causally unless
//!   the file's `dflash.attention.causal` says otherwise, with the
//!   sliding window as a mask on sliding layers. Per layer:
//!   pre-norm, Q/K/V projections, per-head Q/K RMSNorms, RoPE on
//!   Q and K, attention, output projection, optional post norm,
//!   residual add; then pre-norm, SwiGLU FFN, optional post norm,
//!   residual add. The final RMSNorm and the TARGET model's output
//!   head produce per-position logits for the whole block. DFlash
//!   applies no logit scale or softcap of its own.
//!
//! RoPE pairing is NEOX (half-split): `llama_model_rope_type` maps
//! `LLM_ARCH_DFLASH` to NEOX unless the file carries M-RoPE sections
//! or DeepSeek-V4 metadata; a file with either is a different draft
//! and its tensors would trip the variant gates at load.
//!
//! The drafting pass does not write the block's K/V into the cache.
//! Speculative state built while proposing is discarded by design
//! (SPEC-itak-spec Section 4 step 7); committed positions reach the
//! cache only through `inject`, fed by the target's next forward.
//!
//! All arithmetic is the CPU reference through `torch-math`. This
//! module is the correctness oracle for the draft; GPU kernels are
//! later work behind this same interface (E3 scope).

use std::sync::Arc;

use crate::cache::{DflashCache, KvRow};
use crate::config::DflashConfig;
use crate::diag::{Diag, NullDiag};
use crate::error::{DflashError, Result};
use crate::weights::{TensorData, Weights};

/// The target-model tensors a draft borrows when its own file
/// carries none: the embedding table and the output head, as loaded
/// from the TARGET file (`token_embd.weight` and `output.weight`,
/// dequantized, in GGUF stored order). Either may be `None` when
/// the draft file carries its own; when both the file and this
/// struct lack a tensor, the forward refuses with
/// [`DflashError::MissingTargetTensors`].
#[derive(Clone, Copy, Debug, Default)]
pub struct TargetTensors<'a> {
    /// The target's embedding table, shape `[hidden, vocab]`.
    pub embedding: Option<&'a TensorData>,
    /// The target's output head, shape `[hidden, vocab]`.
    pub output: Option<&'a TensorData>,
}

/// A DFlash draft model ready to inject conditioning and draft
/// blocks.
pub struct DflashModel {
    config: DflashConfig,
    weights: Weights,
    diag: Arc<dyn Diag>,
}

impl DflashModel {
    /// Assemble a model from its parts, with a silent diagnostic
    /// sink.
    pub fn new(config: DflashConfig, weights: Weights) -> Self {
        DflashModel {
            config,
            weights,
            diag: Arc::new(NullDiag),
        }
    }

    /// Assemble a model with a diagnostic sink attached. Every
    /// refusal and every completed pass emits through it.
    pub fn with_diag(config: DflashConfig, weights: Weights, diag: Arc<dyn Diag>) -> Self {
        DflashModel {
            config,
            weights,
            diag,
        }
    }

    /// The model's configuration.
    pub fn config(&self) -> &DflashConfig {
        &self.config
    }

    /// The model's weights.
    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    /// The noise block the driver drafts from: the anchor (the last
    /// committed token) at position 0, then the file's mask token
    /// for the remaining `block_size - 1` positions.
    pub fn noise_block(&self, anchor: u32) -> Vec<u32> {
        let mut block = vec![self.config.mask_token_id; self.config.block_size as usize];
        block[0] = anchor;
        block
    }

    /// Fuse conditioning features and write the resulting K/V into
    /// the draft cache at `positions`.
    ///
    /// `features` is row-major `[positions.len()][conditioning
    /// width]`: per position, the target's layer-input rows for
    /// every layer in `dflash.target_layers`, concatenated in that
    /// order (the layout the target-side captured-states forward
    /// produces: one [`muse_glimmer::LayerInputCapture`] per named
    /// layer, rows aligned by position). Positions must be strictly
    /// increasing; re-injecting a held position overwrites its K/V,
    /// matching the reference cache when the target reprocesses a
    /// position after a rewind.
    pub fn inject(
        &self,
        cache: &mut DflashCache,
        features: &[f32],
        positions: &[usize],
    ) -> Result<()> {
        if positions.is_empty() {
            return Err(DflashError::EmptyInput);
        }
        for pair in positions.windows(2) {
            if pair[1] <= pair[0] {
                return Err(DflashError::PositionConflict {
                    position: pair[1],
                    detail: "injection positions must be strictly increasing within one call"
                        .to_string(),
                });
            }
        }
        let total = positions[positions.len() - 1] + 1;
        if total > self.config.context_length as usize {
            return Err(DflashError::SequenceTooLong {
                total,
                context_length: self.config.context_length,
            });
        }
        let width = self.weights.conditioning_width();
        if features.len() != positions.len() * width {
            let actual = reported_width(features.len(), positions.len());
            return Err(DflashError::ConditioningWidth {
                expected: width,
                actual,
            });
        }
        self.diag.event(
            "inject.begin",
            &format!(
                "rows={} positions={}..{} width={}",
                positions.len(),
                positions[0],
                positions[positions.len() - 1],
                width
            ),
        );

        let fused = self.fuse(features, positions.len())?;

        let cfg = &self.config;
        let rows = positions.len();
        for (layer, w) in self.weights.layers.iter().enumerate() {
            let k_all = proj_rows(&w.attn_k, &fused, rows)?;
            let v_all = proj_rows(&w.attn_v, &fused, rows)?;
            let k_w = cfg.k_width() as usize;
            let v_w = cfg.v_width() as usize;
            for (r, &position) in positions.iter().enumerate() {
                let k_row = per_head_norm(
                    &k_all[r * k_w..(r + 1) * k_w],
                    cfg.kv_heads as usize,
                    cfg.key_length as usize,
                    &w.attn_k_norm.data,
                    cfg.rms_eps,
                )?;
                let k_row = rope_row(
                    &k_row,
                    cfg.kv_heads as usize,
                    cfg.key_length as usize,
                    position,
                    cfg.rope_freq_base,
                )?;
                cache.insert(
                    layer,
                    position,
                    KvRow {
                        k: k_row,
                        v: v_all[r * v_w..(r + 1) * v_w].to_vec(),
                    },
                );
            }
        }
        self.diag
            .event("inject.end", &format!("cached_positions={}", cache.len()));
        Ok(())
    }

    /// The conditioning fusion on its own: `fc` projection followed
    /// by the encoder output RMSNorm. This is the standalone encoder
    /// graph of the reference implementation, exposed because the
    /// fused rows are a meaningful seam (the verifier-side tooling
    /// and the closed-form proofs both read the model at it).
    pub fn fuse(&self, features: &[f32], rows: usize) -> Result<Vec<f32>> {
        let width = self.weights.conditioning_width();
        if features.len() != rows * width {
            return Err(DflashError::ConditioningWidth {
                expected: width,
                actual: reported_width(features.len(), rows),
            });
        }
        let projected = proj_rows(&self.weights.fc, features, rows)?;
        rms_rows(
            &projected,
            rows,
            self.config.hidden as usize,
            &self.weights.enc_output_norm.data,
            self.config.rms_eps,
        )
    }

    /// Run one noise block through the draft and return per-position
    /// logits, row-major `[tokens.len()][vocab]`. Row `i` is the
    /// draft's distribution for the token following the block
    /// position `start_position + i`; position 0 is the anchor and
    /// is not itself a draft (the proposer reads rows
    /// `1..=max_drafts`).
    ///
    /// The block attends to the cache's injected K/V and to itself
    /// (non-causally, unless the config says causal), and leaves the
    /// cache untouched. `tokens.len()` must not exceed the file's
    /// block size.
    pub fn forward_block(
        &self,
        cache: &DflashCache,
        tokens: &[u32],
        start_position: usize,
        target: &TargetTensors<'_>,
    ) -> Result<Vec<f32>> {
        let cfg = &self.config;
        if tokens.is_empty() {
            return Err(DflashError::EmptyInput);
        }
        if tokens.len() > cfg.block_size as usize {
            return Err(DflashError::BlockTooLong {
                len: tokens.len(),
                block_size: cfg.block_size,
            });
        }
        for &token in tokens {
            if token >= cfg.vocab_size {
                return Err(DflashError::TokenOutOfRange {
                    token,
                    vocab_size: cfg.vocab_size,
                });
            }
        }
        let total = start_position + tokens.len();
        if total > cfg.context_length as usize {
            return Err(DflashError::SequenceTooLong {
                total,
                context_length: cfg.context_length,
            });
        }
        let embedding = self.weights.embedding.as_ref().or(target.embedding).ok_or(
            DflashError::MissingTargetTensors {
                which: "token_embd.weight",
            },
        )?;
        let output = self.weights.output.as_ref().or(target.output).ok_or(
            DflashError::MissingTargetTensors {
                which: "output.weight",
            },
        )?;
        let hidden = cfg.hidden as usize;
        let vocab = cfg.vocab_size as usize;
        for table in [embedding, output] {
            if table.shape != [u64::from(cfg.hidden), u64::from(cfg.vocab_size)] {
                return Err(DflashError::ShapeMismatch {
                    name: table.name.clone(),
                    expected: vec![u64::from(cfg.hidden), u64::from(cfg.vocab_size)],
                    actual: table.shape.clone(),
                });
            }
        }
        self.diag.event(
            "forward_block.begin",
            &format!(
                "tokens={} start={} cached={}",
                tokens.len(),
                start_position,
                cache.len()
            ),
        );

        let rows = tokens.len();
        let positions: Vec<usize> = (0..rows).map(|i| start_position + i).collect();
        let mut x = Vec::with_capacity(rows * hidden);
        for &token in tokens {
            let row = &embedding.data[token as usize * hidden..(token as usize + 1) * hidden];
            x.extend_from_slice(row);
        }

        for (layer, w) in self.weights.layers.iter().enumerate() {
            x = self.layer_attention(cache, layer, w, &x, &positions)?;
            x = self.layer_ffn(layer, w, &x)?;
        }

        let normed = rms_rows(
            &x,
            rows,
            hidden,
            &self.weights.output_norm.data,
            cfg.rms_eps,
        )?;
        let logits = proj_rows(output, &normed, rows)?;
        debug_assert_eq!(logits.len(), rows * vocab);
        self.diag
            .event("forward_block.end", &format!("logits={}", logits.len()));
        Ok(logits)
    }

    /// One layer's attention sub-block: pre-norm, Q/K/V, per-head
    /// norms, RoPE, cache-and-block attention, output projection,
    /// optional post norm, residual add.
    fn layer_attention(
        &self,
        cache: &DflashCache,
        layer: usize,
        w: &crate::weights::LayerWeights,
        x: &[f32],
        positions: &[usize],
    ) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let rows = positions.len();
        let hidden = cfg.hidden as usize;
        let h = rms_rows(x, rows, hidden, &w.attn_norm.data, cfg.rms_eps)?;
        let q = proj_rows(&w.attn_q, &h, rows)?;
        let k = proj_rows(&w.attn_k, &h, rows)?;
        let v = proj_rows(&w.attn_v, &h, rows)?;

        let q_w = cfg.q_width() as usize;
        let k_w = cfg.k_width() as usize;
        let v_w = cfg.v_width() as usize;
        let mut q_rows: Vec<Vec<f32>> = Vec::with_capacity(rows);
        let mut k_rows: Vec<Vec<f32>> = Vec::with_capacity(rows);
        let mut v_rows: Vec<Vec<f32>> = Vec::with_capacity(rows);
        for r in 0..rows {
            let qn = per_head_norm(
                &q[r * q_w..(r + 1) * q_w],
                cfg.q_heads as usize,
                cfg.key_length as usize,
                &w.attn_q_norm.data,
                cfg.rms_eps,
            )?;
            q_rows.push(rope_row(
                &qn,
                cfg.q_heads as usize,
                cfg.key_length as usize,
                positions[r],
                cfg.rope_freq_base,
            )?);
            let kn = per_head_norm(
                &k[r * k_w..(r + 1) * k_w],
                cfg.kv_heads as usize,
                cfg.key_length as usize,
                &w.attn_k_norm.data,
                cfg.rms_eps,
            )?;
            k_rows.push(rope_row(
                &kn,
                cfg.kv_heads as usize,
                cfg.key_length as usize,
                positions[r],
                cfg.rope_freq_base,
            )?);
            v_rows.push(v[r * v_w..(r + 1) * v_w].to_vec());
        }

        let sliding = cfg.is_sliding(layer) == Some(true);
        let window = cfg.sliding_window as i64;
        let group = cfg.gqa_group() as usize;
        let key_len = cfg.key_length as usize;
        let value_len = cfg.value_length as usize;
        let scale = 1.0 / (key_len as f32).sqrt();

        // The visible key set per query: the injected committed
        // rows plus the block's own rows, filtered by the causal
        // flag and the sliding window, both stated over absolute
        // positions exactly as the reference mask states them.
        let cached: Vec<(usize, &KvRow)> = cache.iter_layer(layer).collect();
        let mut attn = vec![0.0f32; rows * cfg.attn_out_width() as usize];
        for (r, &p) in positions.iter().enumerate() {
            let p = p as i64;
            let mut visible: Vec<(usize, &[f32], &[f32])> = Vec::new();
            for (kp, row) in &cached {
                let kp = *kp as i64;
                if cfg.causal_attention && kp > p {
                    continue;
                }
                if sliding && kp <= p - window {
                    continue;
                }
                visible.push((kp as usize, &row.k, &row.v));
            }
            for (m, &mp) in positions.iter().enumerate() {
                let mp = mp as i64;
                if cfg.causal_attention && mp > p {
                    continue;
                }
                if sliding && mp <= p - window {
                    continue;
                }
                visible.push((mp as usize, &k_rows[m], &v_rows[m]));
            }
            // The block's own row at p is always visible (window and
            // causal tests both admit kp == p), so `visible` is
            // never empty here.
            for head in 0..cfg.q_heads as usize {
                let kv_head = head / group;
                let q_head = &q_rows[r][head * key_len..(head + 1) * key_len];
                let mut scores = Vec::with_capacity(visible.len());
                let mut max = f32::NEG_INFINITY;
                for (_, k_row, _) in &visible {
                    let k_head = &k_row[kv_head * key_len..(kv_head + 1) * key_len];
                    let s = dot(q_head, k_head) * scale;
                    if s > max {
                        max = s;
                    }
                    scores.push(s);
                }
                let mut sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let out_base = r * cfg.attn_out_width() as usize + head * value_len;
                for (i, (_, _, v_row)) in visible.iter().enumerate() {
                    let weight = scores[i] / sum;
                    let v_head = &v_row[kv_head * value_len..(kv_head + 1) * value_len];
                    for d in 0..value_len {
                        attn[out_base + d] += weight * v_head[d];
                    }
                }
            }
        }

        let out = proj_rows(&w.attn_output, &attn, rows)?;
        let out = match &w.post_attention_norm {
            Some(norm) => rms_rows(&out, rows, hidden, &norm.data, cfg.rms_eps)?,
            None => out,
        };
        add_rows(x, &out)
    }

    /// One layer's FFN sub-block: pre-norm, SwiGLU, optional post
    /// norm, residual add.
    fn layer_ffn(
        &self,
        _layer: usize,
        w: &crate::weights::LayerWeights,
        x: &[f32],
    ) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let hidden = cfg.hidden as usize;
        let rows = x.len() / hidden;
        let h = rms_rows(x, rows, hidden, &w.ffn_norm.data, cfg.rms_eps)?;
        let gate = proj_rows(&w.ffn_gate, &h, rows)?;
        let up = proj_rows(&w.ffn_up, &h, rows)?;
        let act = torch_math::swiglu(&gate, &up)?;
        let out = proj_rows(&w.ffn_down, &act, rows)?;
        let out = match &w.post_ffw_norm {
            Some(norm) => rms_rows(&out, rows, hidden, &norm.data, cfg.rms_eps)?,
            None => out,
        };
        add_rows(x, &out)
    }
}

/// The per-row width to report in a width refusal: `total / rows`
/// when a row count exists, else the raw total. Diagnostic only;
/// the refusal itself is decided by the exact length comparison at
/// the call site.
fn reported_width(total: usize, rows: usize) -> usize {
    total.checked_div(rows).unwrap_or(total)
}

/// Multiply every row of `x` (`rows` rows) by the projection
/// `weight`, whose stored shape is `[in, out]`.
fn proj_rows(weight: &TensorData, x: &[f32], rows: usize) -> Result<Vec<f32>> {
    if weight.shape.len() != 2 {
        return Err(DflashError::ShapeMismatch {
            name: weight.name.clone(),
            expected: vec![0, 0],
            actual: weight.shape.clone(),
        });
    }
    let (inn, out) = (weight.shape[0] as usize, weight.shape[1] as usize);
    if x.len() != rows * inn {
        return Err(DflashError::ConditioningWidth {
            expected: inn,
            actual: reported_width(x.len(), rows),
        });
    }
    let mut result = Vec::with_capacity(rows * out);
    for r in 0..rows {
        let row = torch_math::matvec(&weight.data, out, inn, &x[r * inn..(r + 1) * inn])?;
        result.extend_from_slice(&row);
    }
    Ok(result)
}

/// RMSNorm over each row of `width` elements.
fn rms_rows(x: &[f32], rows: usize, width: usize, weight: &[f32], eps: f32) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(x.len());
    for r in 0..rows {
        let row = torch_math::rms_norm(&x[r * width..(r + 1) * width], weight, eps)?;
        out.extend_from_slice(&row);
    }
    Ok(out)
}

/// RMSNorm applied independently to each head slice of one row.
fn per_head_norm(
    row: &[f32],
    heads: usize,
    head_dim: usize,
    weight: &[f32],
    eps: f32,
) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(row.len());
    for h in 0..heads {
        let head = torch_math::rms_norm(&row[h * head_dim..(h + 1) * head_dim], weight, eps)?;
        out.extend_from_slice(&head);
    }
    Ok(out)
}

/// NEOX RoPE applied independently to each head slice of one row.
fn rope_row(
    row: &[f32],
    heads: usize,
    head_dim: usize,
    position: usize,
    base: f32,
) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(row.len());
    for h in 0..heads {
        let head = torch_math::rope(
            &row[h * head_dim..(h + 1) * head_dim],
            position,
            base,
            head_dim,
        )?;
        out.extend_from_slice(&head);
    }
    Ok(out)
}

/// Elementwise residual add, refusing mismatched operands by name
/// through the math error channel.
fn add_rows(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(DflashError::Math(torch_math::MathError::LengthMismatch {
            op: "dflash residual add",
            expected: a.len(),
            actual: b.len(),
        }));
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{tensor, tiny_config, FakeSource};
    use crate::weights::{TensorSource, Weights};

    // ---- f64 helpers: the tests' own math, written from the
    // formulas, sharing no code with the model. ----

    fn rms64(v: &[f64], eps: f64) -> Vec<f64> {
        let mean_sq = v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        v.iter().map(|x| x * inv).collect()
    }

    fn assert_rows_close(actual: &[f32], expected: &[f64], tol: f64, what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (f64::from(*a) - e).abs();
            assert!(
                diff <= tol,
                "{what}[{i}]: expected {e}, got {a} (diff {diff})"
            );
        }
    }

    /// Deterministic pseudo-random values, identical on every run.
    struct Lcg(u64);

    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let v = ((self.0 >> 40) & 0xFF_FFFF) as f32 / (1u64 << 24) as f32;
            (v - 0.5) * 0.6
        }
    }

    // ---- The analytic model: Q and K projections are zero, so
    // attention is a uniform mean over visible V rows, and every
    // other step is a selector or an identity. The whole graph then
    // has a closed form the test writes out by hand. ----

    fn analytic_model() -> (DflashModel, DflashConfig) {
        let cfg = tiny_config();
        let mut source = FakeSource::full(&cfg, false);
        let mut fc = vec![0.0f32; 16 * 8];
        for o in 0..8 {
            fc[o * 16 + o] = 1.0; // fused channel o reads feature channel o
        }
        source.insert(TensorData {
            name: "fc.weight".to_string(),
            shape: vec![16, 8],
            data: fc,
        });
        source.insert(tensor("enc.output_norm.weight", &[8], 1.0));
        source.insert(tensor("output_norm.weight", &[8], 1.0));
        for layer in 0..2 {
            let p = format!("blk.{layer}");
            source.insert(tensor(&format!("{p}.attn_norm.weight"), &[8], 1.0));
            source.insert(tensor(&format!("{p}.ffn_norm.weight"), &[8], 1.0));
            source.insert(tensor(&format!("{p}.attn_q.weight"), &[8, 8], 0.0));
            source.insert(tensor(&format!("{p}.attn_k.weight"), &[8, 4], 0.0));
            // attn_v selector: v channel o reads hidden channel o.
            let mut v = vec![0.0f32; 8 * 4];
            for o in 0..4 {
                v[o * 8 + o] = 1.0;
            }
            source.insert(TensorData {
                name: format!("{p}.attn_v.weight"),
                shape: vec![8, 4],
                data: v,
            });
            source.insert(tensor(&format!("{p}.attn_q_norm.weight"), &[4], 1.0));
            source.insert(tensor(&format!("{p}.attn_k_norm.weight"), &[4], 1.0));
            let mut id = vec![0.0f32; 64];
            for i in 0..8 {
                id[i * 8 + i] = 1.0;
            }
            source.insert(TensorData {
                name: format!("{p}.attn_output.weight"),
                shape: vec![8, 8],
                data: id,
            });
            source.insert(tensor(&format!("{p}.ffn_gate.weight"), &[8, 16], 0.0));
            source.insert(tensor(&format!("{p}.ffn_up.weight"), &[8, 16], 0.0));
            source.insert(tensor(&format!("{p}.ffn_down.weight"), &[16, 8], 0.0));
        }
        let weights = Weights::load(&source, &cfg).expect("analytic weights load");
        (DflashModel::new(cfg.clone(), weights), cfg)
    }

    fn analytic_target() -> (TensorData, TensorData) {
        // token t: value t+1 at channel t % 8.
        let mut emb = vec![0.0f32; 16 * 8];
        for t in 0..16 {
            emb[t * 8 + (t % 8)] = (t + 1) as f32;
        }
        // head row j reads channel j % 8.
        let mut head = vec![0.0f32; 16 * 8];
        for j in 0..16 {
            head[j * 8 + (j % 8)] = 1.0;
        }
        (
            TensorData {
                name: "token_embd.weight".to_string(),
                shape: vec![8, 16],
                data: emb,
            },
            TensorData {
                name: "output.weight".to_string(),
                shape: vec![8, 16],
                data: head,
            },
        )
    }

    #[test]
    fn fuse_matches_the_closed_form() {
        let (model, cfg) = analytic_model();
        // fc reads the first 8 of 16 channels; the second chunk is
        // junk that must not leak into the fused rows.
        let mut features = vec![0.0f32; 2 * 16];
        features[0..8].copy_from_slice(&[1.0, 2.0, 0.0, -1.0, 0.5, 0.0, 0.0, 3.0]);
        features[8..16].fill(9.0);
        features[16..24].copy_from_slice(&[0.0, 1.0, 1.0, 0.0, -2.0, 0.25, 0.0, 0.0]);
        features[24..32].fill(-7.0);
        let fused = model.fuse(&features, 2).expect("fuse runs");
        let eps = f64::from(cfg.rms_eps);
        let ea = rms64(&[1.0, 2.0, 0.0, -1.0, 0.5, 0.0, 0.0, 3.0], eps);
        let eb = rms64(&[0.0, 1.0, 1.0, 0.0, -2.0, 0.25, 0.0, 0.0], eps);
        assert_rows_close(&fused[0..8], &ea, 1e-5, "fused row 0");
        assert_rows_close(&fused[8..16], &eb, 1e-5, "fused row 1");
    }

    #[test]
    fn block_forward_matches_the_closed_form() {
        let (model, cfg) = analytic_model();
        let (emb, head) = analytic_target();
        let target = TargetTensors {
            embedding: Some(&emb),
            output: Some(&head),
        };
        let eps = f64::from(cfg.rms_eps);

        // Condition positions 0 and 1.
        let a: [f64; 8] = [1.0, 2.0, 0.0, -1.0, 0.5, 0.0, 0.0, 3.0];
        let b: [f64; 8] = [0.0, 1.0, 1.0, 0.0, -2.0, 0.25, 0.0, 0.0];
        let mut features = vec![0.0f32; 2 * 16];
        features[0..8].copy_from_slice(&a.map(|v| v as f32));
        features[16..24].copy_from_slice(&b.map(|v| v as f32));
        let fa = rms64(&a, eps);
        let fb = rms64(&b, eps);
        let mut cache = DflashCache::new(&cfg);
        model
            .inject(&mut cache, &features, &[0, 1])
            .expect("inject");
        assert_eq!(cache.positions(), vec![0, 1]);

        // Noise block: anchor 1, masks 15, at positions 2..5.
        let logits = model
            .forward_block(&cache, &[1, 15, 15], 2, &target)
            .expect("block forward");
        assert_eq!(logits.len(), 3 * 16);

        // Closed form. Q = K = 0, so every query attends uniformly
        // over the visible keys: both layers see the 2 cached rows
        // plus all 3 block rows (layer 0's window is 2048, layer 1
        // is global, attention is non-causal).
        let emb_rows: [[f64; 8]; 3] = [
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.0],
        ];
        let mean4 = |rows: &[[f64; 4]]| -> [f64; 4] {
            let mut m = [0.0; 4];
            for row in rows {
                for c in 0..4 {
                    m[c] += row[c] / rows.len() as f64;
                }
            }
            m
        };
        let cached_v: [[f64; 4]; 2] = [[fa[0], fa[1], fa[2], fa[3]], [fb[0], fb[1], fb[2], fb[3]]];
        // Layer 0.
        let mut x: Vec<[f64; 8]> = Vec::new();
        let mut block_v: Vec<[f64; 4]> = Vec::new();
        for row in &emb_rows {
            let h = rms64(row, eps);
            block_v.push([h[0], h[1], h[2], h[3]]);
        }
        let m0 = mean4(&[cached_v[0], cached_v[1], block_v[0], block_v[1], block_v[2]]);
        for row in &emb_rows {
            let mut next = *row;
            for c in 0..4 {
                next[c] += m0[c];
                next[c + 4] += m0[c];
            }
            x.push(next);
        }
        // Layer 1 (its cached V rows are the same fused selectors).
        let mut block_v1: Vec<[f64; 4]> = Vec::new();
        for row in &x {
            let h = rms64(row, eps);
            block_v1.push([h[0], h[1], h[2], h[3]]);
        }
        let m1 = mean4(&[
            cached_v[0],
            cached_v[1],
            block_v1[0],
            block_v1[1],
            block_v1[2],
        ]);
        for (j, row) in x.iter().enumerate() {
            let mut next = *row;
            for c in 0..4 {
                next[c] += m1[c];
                next[c + 4] += m1[c];
            }
            let normed = rms64(&next, eps);
            let expected: Vec<f64> = (0..16).map(|c| normed[c % 8]).collect();
            assert_rows_close(
                &logits[j * 16..(j + 1) * 16],
                &expected,
                1e-4,
                "block logits",
            );
        }
        // The drafting pass must not have written anything anywhere.
        assert_eq!(cache.positions(), vec![0, 1]);
    }

    #[test]
    fn noise_block_is_anchor_then_masks() {
        let (model, cfg) = analytic_model();
        assert_eq!(model.noise_block(7), vec![7, 15, 15, 15]);
        assert_eq!(cfg.max_drafts(), 3);
    }

    #[test]
    fn forward_without_target_tensors_refuses_by_name() {
        let (model, cfg) = analytic_model();
        let cache = DflashCache::new(&cfg);
        let err = model
            .forward_block(&cache, &[1, 15], 0, &TargetTensors::default())
            .expect_err("borrowed tensors absent must refuse");
        assert!(matches!(
            err,
            DflashError::MissingTargetTensors {
                which: "token_embd.weight"
            }
        ));
    }

    #[test]
    fn forward_refusals_are_named() {
        let (model, cfg) = analytic_model();
        let (emb, head) = analytic_target();
        let target = TargetTensors {
            embedding: Some(&emb),
            output: Some(&head),
        };
        let cache = DflashCache::new(&cfg);
        // Five tokens against a block size of four.
        let err = model
            .forward_block(&cache, &[1, 15, 15, 15, 15], 0, &target)
            .expect_err("oversized block must refuse");
        assert!(matches!(err, DflashError::BlockTooLong { .. }));
        // Token outside the vocabulary.
        let err = model
            .forward_block(&cache, &[16], 0, &target)
            .expect_err("token 16 of 16 must refuse");
        assert!(matches!(err, DflashError::TokenOutOfRange { .. }));
        // Empty block.
        let err = model
            .forward_block(&cache, &[], 0, &target)
            .expect_err("empty block must refuse");
        assert!(matches!(err, DflashError::EmptyInput));
    }

    #[test]
    fn inject_refusals_are_named() {
        let (model, cfg) = analytic_model();
        let mut cache = DflashCache::new(&cfg);
        // Rows of 8 against a conditioning width of 16.
        let err = model
            .inject(&mut cache, &[0.0; 16], &[0, 1])
            .expect_err("half-width rows must refuse");
        assert!(matches!(
            err,
            DflashError::ConditioningWidth {
                expected: 16,
                actual: 8
            }
        ));
        // Non-increasing positions.
        let err = model
            .inject(&mut cache, &[0.0; 32], &[1, 1])
            .expect_err("duplicate positions in one call must refuse");
        assert!(matches!(err, DflashError::PositionConflict { .. }));
        assert!(cache.is_empty(), "refused injections store nothing");
    }

    // ---- The naive-reference model: every weight pseudo-random,
    // both layers sliding with a window of 3, and a test-local f64
    // implementation of the documented graph. Agreement between the
    // two is the arithmetic proof for the general case (real Q/K
    // scores, RoPE at nonzero positions, window masking). ----

    fn random_model(causal: bool) -> (DflashModel, DflashConfig, TensorData, TensorData) {
        let mut cfg = tiny_config();
        cfg.sliding_window = 3;
        cfg.sliding_pattern = vec![true, true];
        cfg.causal_attention = causal;
        let mut rng = Lcg(0x5eed);
        let mut source = FakeSource::full(&cfg, false);
        let names = crate::weights::expected_tensor_names(&cfg);
        for name in names {
            let t = source.tensor(&name).expect("fixture tensor");
            let is_norm = name.contains("norm");
            let data: Vec<f32> = t
                .data
                .iter()
                .map(|_| {
                    if is_norm {
                        1.0 + rng.next_f32() * 0.3
                    } else {
                        rng.next_f32()
                    }
                })
                .collect();
            source.insert(TensorData {
                name: t.name,
                shape: t.shape,
                data,
            });
        }
        let weights = Weights::load(&source, &cfg).expect("random weights load");
        let mut emb_data = Vec::new();
        for _ in 0..16 * 8 {
            emb_data.push(rng.next_f32());
        }
        let mut head_data = Vec::new();
        for _ in 0..16 * 8 {
            head_data.push(rng.next_f32());
        }
        let emb = TensorData {
            name: "token_embd.weight".to_string(),
            shape: vec![8, 16],
            data: emb_data,
        };
        let head = TensorData {
            name: "output.weight".to_string(),
            shape: vec![8, 16],
            data: head_data,
        };
        (DflashModel::new(cfg.clone(), weights), cfg, emb, head)
    }

    /// Inputs for [`reference_logits`], bundled so the signature
    /// stays readable.
    struct RefInput<'a> {
        features: &'a [f32],
        inject_positions: &'a [usize],
        tokens: &'a [u32],
        start: usize,
        emb: &'a TensorData,
        head: &'a TensorData,
    }

    /// One layer's reference cache rows: `(position, k, v)`.
    type RefCache = Vec<Vec<(usize, Vec<f64>, Vec<f64>)>>;

    /// The test's own f64 graph, written from the module docs.
    fn reference_logits(model: &DflashModel, cfg: &DflashConfig, input: &RefInput) -> Vec<f64> {
        let (features, inject_positions, tokens, start, emb, head) = (
            input.features,
            input.inject_positions,
            input.tokens,
            input.start,
            input.emb,
            input.head,
        );
        let w = model.weights();
        let eps = f64::from(cfg.rms_eps);
        let hidden = cfg.hidden as usize;
        let to64 = |v: &[f32]| v.iter().map(|x| f64::from(*x)).collect::<Vec<f64>>();
        let matvec = |weight: &TensorData, x: &[f64]| -> Vec<f64> {
            let (inn, out) = (weight.shape[0] as usize, weight.shape[1] as usize);
            (0..out)
                .map(|o| {
                    (0..inn)
                        .map(|i| f64::from(weight.data[o * inn + i]) * x[i])
                        .sum()
                })
                .collect()
        };
        let rms_w = |v: &[f64], weight: &[f32]| -> Vec<f64> {
            let mean_sq = v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64;
            let inv = 1.0 / (mean_sq + eps).sqrt();
            v.iter()
                .zip(weight.iter())
                .map(|(x, wi)| x * inv * f64::from(*wi))
                .collect()
        };
        let rope = |v: &[f64], pos: usize| -> Vec<f64> {
            let dim = v.len();
            let half = dim / 2;
            let mut out = v.to_vec();
            for i in 0..half {
                let angle =
                    pos as f64 * f64::from(cfg.rope_freq_base).powf(-((2 * i) as f64) / dim as f64);
                let (s, c) = angle.sin_cos();
                out[i] = v[i] * c - v[i + half] * s;
                out[i + half] = v[i] * s + v[i + half] * c;
            }
            out
        };
        let per_head = |row: &[f64], heads: usize, dim: usize, f: &dyn Fn(&[f64]) -> Vec<f64>| {
            let mut out = Vec::new();
            for h in 0..heads {
                out.extend(f(&row[h * dim..(h + 1) * dim]));
            }
            out
        };

        // Injection.
        let width = w.conditioning_width();
        let mut cache: RefCache = vec![Vec::new(); cfg.block_count as usize];
        for (r, &pos) in inject_positions.iter().enumerate() {
            let feat = to64(&features[r * width..(r + 1) * width]);
            let fused = rms_w(&matvec(&w.fc, &feat), &w.enc_output_norm.data);
            for (layer, lw) in w.layers.iter().enumerate() {
                let k = matvec(&lw.attn_k, &fused);
                let k = per_head(&k, cfg.kv_heads as usize, cfg.key_length as usize, &|h| {
                    rope(&rms_w(h, &lw.attn_k_norm.data), pos)
                });
                let v = matvec(&lw.attn_v, &fused);
                cache[layer].push((pos, k, v));
            }
        }

        // Block forward.
        let rows = tokens.len();
        let positions: Vec<usize> = (0..rows).map(|i| start + i).collect();
        let mut x: Vec<Vec<f64>> = tokens
            .iter()
            .map(|&t| to64(&emb.data[t as usize * hidden..(t as usize + 1) * hidden]))
            .collect();
        for (layer, lw) in w.layers.iter().enumerate() {
            let sliding = cfg.sliding_pattern[layer];
            let mut next: Vec<Vec<f64>> = Vec::new();
            // Block K/V for this layer first (attention reads them).
            let mut bk: Vec<Vec<f64>> = Vec::new();
            let mut bv: Vec<Vec<f64>> = Vec::new();
            let mut bq: Vec<Vec<f64>> = Vec::new();
            for r in 0..rows {
                let h = rms_w(&x[r], &lw.attn_norm.data);
                let q = per_head(
                    &matvec(&lw.attn_q, &h),
                    cfg.q_heads as usize,
                    cfg.key_length as usize,
                    &|hh| rope(&rms_w(hh, &lw.attn_q_norm.data), positions[r]),
                );
                let k = per_head(
                    &matvec(&lw.attn_k, &h),
                    cfg.kv_heads as usize,
                    cfg.key_length as usize,
                    &|hh| rope(&rms_w(hh, &lw.attn_k_norm.data), positions[r]),
                );
                bq.push(q);
                bk.push(k);
                bv.push(matvec(&lw.attn_v, &h));
            }
            for r in 0..rows {
                let p = positions[r] as i64;
                let mut keys: Vec<(i64, &Vec<f64>, &Vec<f64>)> = Vec::new();
                for (pos, k, v) in &cache[layer] {
                    keys.push((*pos as i64, k, v));
                }
                for m in 0..rows {
                    keys.push((positions[m] as i64, &bk[m], &bv[m]));
                }
                let visible: Vec<&(i64, &Vec<f64>, &Vec<f64>)> = keys
                    .iter()
                    .filter(|(kp, _, _)| {
                        if cfg.causal_attention && *kp > p {
                            return false;
                        }
                        if sliding && *kp <= p - i64::from(cfg.sliding_window) {
                            return false;
                        }
                        true
                    })
                    .collect();
                let mut attn = vec![0.0f64; cfg.attn_out_width() as usize];
                for head_i in 0..cfg.q_heads as usize {
                    let kvh = head_i / cfg.gqa_group() as usize;
                    let kl = cfg.key_length as usize;
                    let vl = cfg.value_length as usize;
                    let q = &bq[r][head_i * kl..(head_i + 1) * kl];
                    let scale = 1.0 / (kl as f64).sqrt();
                    let scores: Vec<f64> = visible
                        .iter()
                        .map(|(_, k, _)| {
                            q.iter()
                                .zip(k[kvh * kl..(kvh + 1) * kl].iter())
                                .map(|(a, b)| a * b)
                                .sum::<f64>()
                                * scale
                        })
                        .collect();
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                    let sum: f64 = exps.iter().sum();
                    for (i, (_, _, v)) in visible.iter().enumerate() {
                        let weight = exps[i] / sum;
                        for d in 0..vl {
                            attn[head_i * vl + d] += weight * v[kvh * vl + d];
                        }
                    }
                }
                let out = matvec(&lw.attn_output, &attn);
                let out = match &lw.post_attention_norm {
                    Some(n) => rms_w(&out, &n.data),
                    None => out,
                };
                let mut row: Vec<f64> = x[r].iter().zip(out.iter()).map(|(a, b)| a + b).collect();
                let h = rms_w(&row, &lw.ffn_norm.data);
                let gate = matvec(&lw.ffn_gate, &h);
                let up = matvec(&lw.ffn_up, &h);
                let act: Vec<f64> = gate
                    .iter()
                    .zip(up.iter())
                    .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                    .collect();
                let fout = matvec(&lw.ffn_down, &act);
                let fout = match &lw.post_ffw_norm {
                    Some(n) => rms_w(&fout, &n.data),
                    None => fout,
                };
                for (a, b) in row.iter_mut().zip(fout.iter()) {
                    *a += b;
                }
                next.push(row);
            }
            x = next;
        }
        let mut logits = Vec::new();
        for row in &x {
            logits.extend(matvec(head, &rms_w(row, &w.output_norm.data)));
        }
        logits
    }

    #[test]
    fn block_forward_matches_naive_reference_noncausal_and_causal() {
        for causal in [false, true] {
            let (model, cfg, emb, head) = random_model(causal);
            let target = TargetTensors {
                embedding: Some(&emb),
                output: Some(&head),
            };
            // Features: distinct pseudo-random rows of width 16.
            let mut rng = Lcg(0xfeed);
            let features: Vec<f32> = (0..3 * 16).map(|_| rng.next_f32()).collect();
            let mut cache = DflashCache::new(&cfg);
            model
                .inject(&mut cache, &features, &[0, 1, 2])
                .expect("inject");
            let tokens = [3u32, 15, 2];
            let logits = model
                .forward_block(&cache, &tokens, 3, &target)
                .expect("block forward");
            let expected = reference_logits(
                &model,
                &cfg,
                &RefInput {
                    features: &features,
                    inject_positions: &[0, 1, 2],
                    tokens: &tokens,
                    start: 3,
                    emb: &emb,
                    head: &head,
                },
            );
            assert_rows_close(&logits, &expected, 2e-3, "reference logits");
        }
    }

    #[test]
    fn chunked_injection_equals_whole_injection() {
        let (model, cfg, emb, head) = random_model(false);
        let target = TargetTensors {
            embedding: Some(&emb),
            output: Some(&head),
        };
        let mut rng = Lcg(0xbeef);
        let features: Vec<f32> = (0..3 * 16).map(|_| rng.next_f32()).collect();

        let mut whole = DflashCache::new(&cfg);
        model
            .inject(&mut whole, &features, &[0, 1, 2])
            .expect("whole");
        let mut chunked = DflashCache::new(&cfg);
        model
            .inject(&mut chunked, &features[0..32], &[0, 1])
            .expect("chunk 1");
        model
            .inject(&mut chunked, &features[32..48], &[2])
            .expect("chunk 2");

        let a = model
            .forward_block(&whole, &[3, 15], 3, &target)
            .expect("a");
        let b = model
            .forward_block(&chunked, &[3, 15], 3, &target)
            .expect("b");
        assert_eq!(a, b, "chunked and whole injection must agree exactly");

        // Overwrite: poison position 1, then re-inject the true
        // features there; the state must equal the clean one.
        let mut overwritten = DflashCache::new(&cfg);
        model
            .inject(&mut overwritten, &features, &[0, 1, 2])
            .expect("clean");
        let poison: Vec<f32> = vec![3.0; 16];
        model
            .inject(&mut overwritten, &poison, &[1])
            .expect("poison");
        let mid = model
            .forward_block(&overwritten, &[3, 15], 3, &target)
            .expect("mid");
        assert_ne!(a, mid, "poisoned state must change the output");
        model
            .inject(&mut overwritten, &features[16..32], &[1])
            .expect("restore");
        let c = model
            .forward_block(&overwritten, &[3, 15], 3, &target)
            .expect("c");
        assert_eq!(a, c, "re-injection must restore the exact state");
    }
}
