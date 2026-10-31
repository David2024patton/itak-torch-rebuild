//! Speculators: the drafting side of the loop (SPEC-itak-spec Section
//! 6.2), plus the shared compute handle a session needs to reach the
//! target's KV checkpoint machinery.
//!
//! The [`Speculator`] trait is the seam every drafting mode plugs into.
//! Its shape is fixed; the concrete DFlash implementation adds one
//! inherent method, [`DflashSpeculator::propose_block`], which returns the
//! owned per-position distributions the sampled correction walk needs
//! (the trait's borrowed view carries greedy material only, matching the
//! spec's contract).

use std::rc::Rc;

use cpu_compute::{Checkpoint, CpuCompute};
use dflash::{DflashCache, DflashModel, TargetTensors, TensorData};
use muse_glimmer::{
    AttentionRequest, Cache, Compute, GlimmerConfig, GlimmerError, LayerInputCapture, LayerWeights,
    Result as GlimmerResult, TensorView,
};

use crate::error::{Result, SpecError};
use crate::sampler;

/// A compute handle shared between a [`muse_glimmer::GlimmerModel`] and its session.
///
/// The target model owns its compute privately, but the loop must call
/// `cache_checkpoint` / `cache_rewind` / `cache_release` on the same
/// [`CpuCompute`] the model forwards through. This wrapper is the bridge:
/// build one `Rc<CpuCompute>`, hand a `SharedCompute` clone to the model
/// as its `Box<dyn Compute>`, and keep the `Rc` for the session.
pub struct SharedCompute(pub Rc<CpuCompute>);

impl Compute for SharedCompute {
    fn embed(
        &self,
        table: &TensorView,
        tokens: &[u32],
        cfg: &GlimmerConfig,
    ) -> GlimmerResult<Vec<f32>> {
        self.0.embed(table, tokens, cfg)
    }
    fn rms_norm(
        &self,
        x: &[f32],
        weight: &TensorView,
        rows: usize,
        eps: f32,
    ) -> GlimmerResult<Vec<f32>> {
        self.0.rms_norm(x, weight, rows, eps)
    }
    fn rms_norm_plain(&self, x: &[f32], rows: usize, eps: f32) -> GlimmerResult<Vec<f32>> {
        self.0.rms_norm_plain(x, rows, eps)
    }
    fn proj(&self, x: &[f32], weight: &TensorView, rows: usize) -> GlimmerResult<Vec<f32>> {
        self.0.proj(x, weight, rows)
    }
    fn attention(&self, req: &AttentionRequest, cache: &mut dyn Cache) -> GlimmerResult<Vec<f32>> {
        self.0.attention(req, cache)
    }
    fn attn_gate(&self, attn_out: &[f32], gate: &[f32]) -> GlimmerResult<Vec<f32>> {
        self.0.attn_gate(attn_out, gate)
    }
    fn add(&self, a: &[f32], b: &[f32]) -> GlimmerResult<Vec<f32>> {
        self.0.add(a, b)
    }
    fn swiglu_ffn(
        &self,
        x: &[f32],
        w: &LayerWeights,
        rows: usize,
        cfg: &GlimmerConfig,
    ) -> GlimmerResult<Vec<f32>> {
        self.0.swiglu_ffn(x, w, rows, cfg)
    }
    fn lm_head(
        &self,
        x: &[f32],
        head: &TensorView,
        cfg: &GlimmerConfig,
    ) -> GlimmerResult<Vec<f32>> {
        self.0.lm_head(x, head, cfg)
    }
}

/// Re-exported so session code names the checkpoint type through one path.
pub type TargetCheckpoint = Checkpoint;

/// One proposal distribution, borrowed from the speculator's buffers.
#[derive(Debug, Clone, Copy)]
pub struct DraftDist<'a> {
    /// Probabilities over the vocabulary for one draft position.
    pub probs: &'a [f32],
}

/// A block of draft tokens with their distributions, borrowed from the
/// speculator's last proposal.
#[derive(Debug)]
pub struct BlockProposal<'a> {
    /// The proposed tokens, in order (`t[1..=n]` in spec indexing).
    pub tokens: &'a [u32],
    /// One distribution per proposed token, aligned with `tokens`. The
    /// vector itself is owned because the borrowed rows live in the
    /// speculator's probability buffer; the spec fixes this shape (tokens
    /// plus per-token distributions), not the container spelling.
    pub dists: Vec<DraftDist<'a>>,
}

/// The drafting seam (SPEC-itak-spec Section 6.2). Implementations own
/// their draft model and cache; the session owns the target and the round
/// structure.
pub trait Speculator {
    /// The drafting mode this speculator implements.
    fn mode(&self) -> crate::SpecMode;

    /// Maximum draft tokens one block can propose.
    fn max_depth(&self) -> usize;

    /// Propose up to `n` tokens after `pending`, the last emitted token.
    /// The result borrows the speculator's internal buffers and is valid
    /// until the next `propose` or `commit` call.
    fn propose(&mut self, pending: u32, n: usize) -> Result<BlockProposal<'_>>;

    /// Tell the draft that exactly these tokens were committed, in order,
    /// so its state equals having processed exactly the committed sequence
    /// (Section 4, step 7).
    fn commit(&mut self, tokens: &[u32]) -> Result<()>;

    /// Hand the draft the target layer-input rows a forward just captured,
    /// tagged by absolute position. Modes that condition on target hidden
    /// states (DFlash) stage them for the next `commit`; modes that do not
    /// ignore them. The default implementation ignores them.
    fn note_captures(
        &mut self,
        _base_position: usize,
        _captures: &[LayerInputCapture],
    ) -> Result<()> {
        Ok(())
    }
}

/// Which arm of the correction walk a proposal is materialized for. The
/// greedy arm reads only argmaxes; the sampled arm needs the full draft
/// distributions at a temperature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DraftArm {
    /// Greedy: proposal tokens are the draft rows' argmaxes.
    Greedy,
    /// Sampled: proposal tokens are drawn from the draft distributions at
    /// this temperature, and the distributions themselves are returned.
    Sampled {
        /// Draft temperature (already resolved: follow-target applied).
        temperature: f32,
    },
}

/// An owned block proposal: tokens plus, for the sampled arm, the full
/// per-position draft distributions the walk divides by.
#[derive(Debug)]
pub struct Proposal {
    /// The proposed tokens, in order.
    pub tokens: Vec<u32>,
    /// Per-position draft distributions, aligned with `tokens`. Empty on
    /// the greedy arm, where the walk never reads distributions.
    pub dists: Vec<Vec<f32>>,
}

/// The DFlash speculator: a loaded draft model, its private cache, the
/// target tensors it borrows (embedding table and LM head), and the
/// staging area for captured conditioning rows.
///
/// Draft sync contract (Section 4, step 7, and Section 10 Q1/Q2 as
/// settled): the draft's cache holds exactly the committed positions
/// `0..len`. Captures from a target forward are staged by
/// [`DflashSpeculator::note_captures`] keyed to their absolute positions;
/// [`Speculator::commit`] injects the staged rows for exactly the newly
/// committed positions and drops the rest of the stage. The draft cache is
/// never checkpointed by the target's machinery; this staging is its whole
/// rollback story, and it is exact because a position's conditioning row
/// depends only on the committed prefix that produced it.
pub struct DflashSpeculator {
    model: DflashModel,
    cache: DflashCache,
    embedding: TensorData,
    output: TensorData,
    /// Staged conditioning rows: `(absolute position, fused row)` pairs in
    /// increasing position order, plus the position the stage starts at.
    staged: Vec<(usize, Vec<f32>)>,
    staged_from: Option<usize>,
    /// Last proposal buffers, backing the trait's borrowed view.
    last_tokens: Vec<u32>,
    last_probs: Vec<Vec<f32>>,
}

impl DflashSpeculator {
    /// Assemble a speculator from a loaded draft model and the target
    /// tensors it borrows. The caller (session creation) validates the
    /// draft against the target; this constructor only takes ownership.
    pub fn new(model: DflashModel, embedding: TensorData, output: TensorData) -> Self {
        let cache = DflashCache::new(model.config());
        DflashSpeculator {
            model,
            cache,
            embedding,
            output,
            staged: Vec::new(),
            staged_from: None,
            last_tokens: Vec::new(),
            last_probs: Vec::new(),
        }
    }

    /// The draft model this speculator drives.
    pub fn model(&self) -> &DflashModel {
        &self.model
    }

    /// How many committed positions the draft's cache currently holds.
    /// At every round boundary this equals the target's cache position.
    pub fn committed_len(&self) -> usize {
        self.cache.len()
    }

    /// Run one full noise block after `pending` and return up to `n`
    /// proposals. The block always runs at the draft's full block size:
    /// block attention is non-causal, so a truncated block would change
    /// the earlier rows' distributions; the session slices the result.
    ///
    /// Row semantics (settled by the E3 run): block row `k` is the draft
    /// distribution for the draft token at block position `k + 1`, with
    /// the anchor (`pending`) at block position 0.
    pub fn propose_block(
        &mut self,
        pending: u32,
        n: usize,
        arm: DraftArm,
        rng: &mut dyn FnMut() -> f64,
    ) -> Result<Proposal> {
        let block = self.model.noise_block(pending);
        let start = self.cache.len();
        let borrowed = TargetTensors {
            embedding: Some(&self.embedding),
            output: Some(&self.output),
        };
        let logits = self
            .model
            .forward_block(&self.cache, &block, start, &borrowed)
            .map_err(|e| SpecError::DraftRefused {
                source: Box::new(e),
            })?;
        let vocab = self.model.config().vocab_size as usize;
        let want = n.min(self.model.config().max_drafts() as usize);
        let mut tokens = Vec::with_capacity(want);
        let mut dists: Vec<Vec<f32>> = Vec::new();
        for k in 0..want {
            // Forward row k + 1 is the distribution for draft token
            // k + 1: row 0 belongs to the anchor's own position and is
            // not itself a draft (the model's forward_block contract).
            let row = &logits[(k + 1) * vocab..(k + 2) * vocab];
            match arm {
                DraftArm::Greedy => tokens.push(sampler::argmax(row)),
                DraftArm::Sampled { temperature } => {
                    let dist = sampler::distribution(row, temperature, 1.0);
                    tokens.push(sampler::sample_from(&dist, rng()));
                    dists.push(dist);
                }
            }
        }
        self.last_probs = match arm {
            DraftArm::Greedy => tokens
                .iter()
                .map(|&t| {
                    let mut one_hot = vec![0.0f32; vocab];
                    one_hot[t as usize] = 1.0;
                    one_hot
                })
                .collect(),
            DraftArm::Sampled { .. } => dists.clone(),
        };
        self.last_tokens.clone_from(&tokens);
        Ok(Proposal { tokens, dists })
    }
}

impl Speculator for DflashSpeculator {
    fn mode(&self) -> crate::SpecMode {
        crate::SpecMode::Dflash
    }

    fn max_depth(&self) -> usize {
        self.model.config().max_drafts() as usize
    }

    fn propose(&mut self, pending: u32, n: usize) -> Result<BlockProposal<'_>> {
        let mut no_draws = || 0.0f64;
        let proposal = self.propose_block(pending, n, DraftArm::Greedy, &mut no_draws)?;
        let _ = proposal;
        let dists = self
            .last_probs
            .iter()
            .map(|row| DraftDist { probs: row })
            .collect();
        Ok(BlockProposal {
            tokens: &self.last_tokens,
            dists,
        })
    }

    fn commit(&mut self, tokens: &[u32]) -> Result<()> {
        let from = self.cache.len();
        if tokens.is_empty() {
            self.staged.clear();
            self.staged_from = None;
            return Ok(());
        }
        let staged_from = self.staged_from.ok_or_else(|| SpecError::SessionState {
            detail: format!(
                "draft commit of {} tokens at position {from} has no staged captures",
                tokens.len()
            ),
        })?;
        if staged_from != from || self.staged.len() < tokens.len() {
            return Err(SpecError::SessionState {
                detail: format!(
                    "draft commit at position {from} needs {} staged rows from {from}, \
                     stage holds {} rows from {staged_from}",
                    tokens.len(),
                    self.staged.len()
                ),
            });
        }
        let width = self.model.weights().conditioning_width();
        let mut features = Vec::with_capacity(tokens.len() * width);
        let mut positions = Vec::with_capacity(tokens.len());
        for i in 0..tokens.len() {
            let (position, row) = &self.staged[i];
            if *position != from + i || row.len() != width {
                return Err(SpecError::SessionState {
                    detail: format!(
                        "staged capture {i} is position {position} width {}, expected position {} width {width}",
                        row.len(),
                        from + i
                    ),
                });
            }
            features.extend_from_slice(row);
            positions.push(*position);
        }
        // The stage is consumed either way: a failed injection leaves the
        // draft desynced, which the session surfaces as a refusal rather
        // than retrying against stale rows.
        self.staged.clear();
        self.staged_from = None;
        self.model
            .inject(&mut self.cache, &features, &positions)
            .map_err(|e| SpecError::DraftRefused {
                source: Box::new(e),
            })?;
        Ok(())
    }

    fn note_captures(
        &mut self,
        base_position: usize,
        captures: &[LayerInputCapture],
    ) -> Result<()> {
        if self.staged_from.is_some() {
            return Err(SpecError::SessionState {
                detail: "captures staged twice without a commit between them".to_string(),
            });
        }
        if captures.is_empty() {
            return Ok(());
        }
        // Concatenate each position's per-layer rows in target_layers
        // order: exactly the flat conditioning layout inject consumes.
        let target_hidden = self.model.weights().target_hidden as usize;
        let width = self.model.weights().conditioning_width();
        let rows = captures[0].rows.len() / target_hidden;
        let mut staged = Vec::with_capacity(rows);
        for i in 0..rows {
            let mut row = Vec::with_capacity(width);
            for capture in captures {
                row.extend_from_slice(&capture.rows[i * target_hidden..(i + 1) * target_hidden]);
            }
            staged.push((base_position + i, row));
        }
        self.staged = staged;
        self.staged_from = Some(base_position);
        Ok(())
    }
}

/// Wrap a target-side refusal from the session's cache operations.
pub(crate) fn cache_refused(op: &'static str, source: GlimmerError) -> SpecError {
    SpecError::CacheRefused {
        op,
        source: Box::new(source),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit;

    fn captures(positions: usize, layers: &[usize]) -> Vec<LayerInputCapture> {
        let target_hidden = testkit::TARGET_HIDDEN;
        layers
            .iter()
            .map(|&layer| LayerInputCapture {
                layer,
                rows: (0..positions * target_hidden)
                    .map(|i| (i % 7) as f32 * 0.01)
                    .collect(),
            })
            .collect()
    }

    /// The draft sync contract: commit consumes exactly the staged rows
    /// for the committed positions; the draft's cache then holds exactly
    /// the committed prefix, and stale or missing stages refuse by name.
    #[test]
    fn commit_consumes_exactly_the_staged_prefix() {
        let compute = std::rc::Rc::new(CpuCompute::new());
        let fixture = testkit::build_target(Box::new(SharedCompute(compute)));
        let mut spec = testkit::build_speculator(&fixture, 5);

        // Commit with nothing staged refuses.
        assert!(matches!(
            spec.commit(&[1, 2]),
            Err(SpecError::SessionState { .. })
        ));

        // Stage 4 positions, commit 3: the draft holds exactly 3.
        spec.note_captures(0, &captures(4, &[1, 3])).expect("stage");
        spec.commit(&[10, 11, 12]).expect("commit prefix");
        assert_eq!(spec.committed_len(), 3);

        // A stage that does not start at the committed length refuses.
        spec.note_captures(9, &captures(2, &[1, 3])).expect("stage");
        assert!(matches!(
            spec.commit(&[13]),
            Err(SpecError::SessionState { .. })
        ));
        assert_eq!(spec.committed_len(), 3, "a refused commit changes nothing");
    }

    /// One agreement probe between the trait's borrowed view and the
    /// owned block proposal: for the same speculator state, pending
    /// token, and depth, the view's tokens must equal the block
    /// proposal's tokens one for one, and each of the view's
    /// distributions must be the one-hot row at exactly its token.
    fn assert_proposals_agree(spec: &mut DflashSpeculator, pending: u32, n: usize) {
        let (view_tokens, view_argmax): (Vec<u32>, Vec<u32>) = {
            let view = spec.propose(pending, n).expect("trait propose");
            let argmaxes = view
                .dists
                .iter()
                .map(|dist| {
                    let mass: f32 = dist.probs.iter().sum();
                    assert!(
                        (mass - 1.0).abs() < 1e-6,
                        "greedy dists are one-hot, mass {mass}"
                    );
                    sampler::argmax(dist.probs)
                })
                .collect();
            (view.tokens.to_vec(), argmaxes)
        };
        assert_eq!(
            view_argmax, view_tokens,
            "view dists must be one-hot at the view tokens (pending {pending}, n {n})"
        );
        let mut no_draws = || 0.0f64;
        let block = spec
            .propose_block(pending, n, DraftArm::Greedy, &mut no_draws)
            .expect("block propose");
        assert_eq!(
            view_tokens, block.tokens,
            "trait view and block proposal diverge (pending {pending}, n {n})"
        );
    }

    /// The trait's borrowed proposal view agrees with the owned block
    /// proposal the session consumes: the same tokens, compared
    /// directly. The discarding shape of the trait view is accepted
    /// design (the panel merge ruled it so); this test is the guard
    /// that keeps the two proposal paths from drifting apart
    /// silently, which a lengths-and-mass check cannot do.
    #[test]
    fn trait_proposal_matches_block_proposal() {
        let compute = std::rc::Rc::new(CpuCompute::new());
        let fixture = testkit::build_target(Box::new(SharedCompute(compute)));

        // A fresh speculator, nothing committed yet.
        let mut fresh = testkit::build_speculator(&fixture, 5);
        assert_proposals_agree(&mut fresh, 0, 2);
        assert_proposals_agree(&mut fresh, 127, 3);

        // A speculator with a committed prefix, probed below, at,
        // and past its maximum depth.
        let mut spec = testkit::build_speculator(&fixture, 5);
        spec.note_captures(0, &captures(3, &[1, 3])).expect("stage");
        spec.commit(&[4, 5, 6]).expect("commit");
        assert_proposals_agree(&mut spec, 6, 1);
        assert_proposals_agree(&mut spec, 6, 3);
        assert_proposals_agree(&mut spec, 6, 4);
        assert_proposals_agree(&mut spec, 9, 2);
        assert_proposals_agree(&mut spec, 6, 5);

        assert_eq!(spec.max_depth(), 4);
        assert_eq!(spec.mode(), crate::SpecMode::Dflash);
    }
}
