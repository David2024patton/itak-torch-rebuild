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
use dflash::{DflashCache, DflashConfig, DflashModel, TargetTensors, TensorData};
use muse_glimmer::{
    AttentionRequest, Cache, Compute, GlimmerConfig, GlimmerError, LayerInputCapture, LayerWeights,
    Result as GlimmerResult, TensorView,
};
use wgpu_compute::DflashSession;

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

/// The draft's execution arm: the same semantics on two
/// device paths. Every piece of loop bookkeeping (capture
/// staging, commit consumption, proposal slicing) lives on
/// [`DflashSpeculator`] itself and is shared; the arm only
/// answers the two passes (inject, forward block) and the
/// committed length.
/// The CPU arm's payload, boxed inside the enum so the
/// two arms' sizes stay comparable.
struct CpuDraft {
    /// The loaded draft model.
    model: DflashModel,
    /// The draft's private KV cache.
    cache: DflashCache,
    /// The borrowed target embedding table.
    embedding: TensorData,
    /// The borrowed target output head.
    output: TensorData,
}

enum DraftBackendImpl {
    /// The CPU reference implementation, with its private
    /// cache and the borrowed target tensors as f32 data.
    Cpu(Box<CpuDraft>),
    /// The device-resident GPU session. The session borrows
    /// its backend at `'static`: the runner leaks the
    /// backend for the process lifetime, the same
    /// discipline it already applies to the target's
    /// resident session. Boxed, like the CPU payload, so
    /// the arms' sizes stay comparable.
    Gpu {
        /// The resident draft session.
        session: Box<DflashSession<'static>>,
    },
}

/// The DFlash speculator: a draft execution arm, the draft
/// config and conditioning geometry both arms agree on,
/// and the staging area for captured conditioning rows.
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
    backend: DraftBackendImpl,
    /// The draft config (identical for both arms; the CPU
    /// arm's model and the GPU arm's session were both
    /// built from it).
    config: DflashConfig,
    /// Conditioning row width, from the loaded `fc`.
    conditioning_width: usize,
    /// Target hidden size, derived from `fc` at load.
    target_hidden: usize,
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
    /// tensors it borrows (the CPU arm). The caller (session creation)
    /// validates the draft against the target; this constructor only
    /// takes ownership.
    pub fn new(model: DflashModel, embedding: TensorData, output: TensorData) -> Self {
        let config = model.config().clone();
        let conditioning_width = model.weights().conditioning_width();
        let target_hidden = model.weights().target_hidden as usize;
        let cache = DflashCache::new(model.config());
        DflashSpeculator {
            backend: DraftBackendImpl::Cpu(Box::new(CpuDraft {
                model,
                cache,
                embedding,
                output,
            })),
            config,
            conditioning_width,
            target_hidden,
            staged: Vec::new(),
            staged_from: None,
            last_tokens: Vec::new(),
            last_probs: Vec::new(),
        }
    }

    /// Assemble a speculator over a device-resident draft
    /// session (the GPU arm, selected by
    /// [`crate::DraftBackend::Gpu`]). The session was built
    /// from the same draft config and weights the CPU arm
    /// would load; every loop-visible behavior (staging,
    /// commit, proposal slicing, refusals) is the shared
    /// code above the arm.
    pub fn gpu(session: DflashSession<'static>) -> Self {
        let config = session.config().clone();
        let conditioning_width = session.conditioning_width();
        let target_hidden = session.target_hidden() as usize;
        DflashSpeculator {
            backend: DraftBackendImpl::Gpu {
                session: Box::new(session),
            },
            config,
            conditioning_width,
            target_hidden,
            staged: Vec::new(),
            staged_from: None,
            last_tokens: Vec::new(),
            last_probs: Vec::new(),
        }
    }

    /// The draft config this speculator serves (either arm).
    pub fn config(&self) -> &DflashConfig {
        &self.config
    }

    /// The draft model of the CPU arm. The GPU arm has no
    /// `DflashModel` to hand back; session creation reads
    /// [`DflashSpeculator::config`] instead, which both
    /// arms answer.
    pub fn model(&self) -> Option<&DflashModel> {
        match &self.backend {
            DraftBackendImpl::Cpu(cpu) => Some(&cpu.model),
            DraftBackendImpl::Gpu { .. } => None,
        }
    }

    /// Run the noise block for `pending` on the active
    /// arm and return its first `row_cap` rows of logits.
    ///
    /// The cap is a host-transfer hint, not a block shape:
    /// the block always runs whole (its attention is
    /// non-causal), and the caller slices only rows
    /// `1..=row_cap - 1` from the result, so rows past the
    /// cap are never consumed. The GPU arm therefore reads
    /// back just the prefix
    /// ([`wgpu_compute::DflashSession::forward_block_capped`]);
    /// the CPU arm computes in-process, has no transfer to
    /// trim, and returns the full block as before.
    fn forward_block_logits(&mut self, pending: u32, row_cap: usize) -> Result<Vec<f32>> {
        let block = {
            let mut block = vec![self.config.mask_token_id; self.config.block_size as usize];
            block[0] = pending;
            block
        };
        let start = self.committed_len();
        match &mut self.backend {
            DraftBackendImpl::Cpu(cpu) => {
                let borrowed = TargetTensors {
                    embedding: Some(&cpu.embedding),
                    output: Some(&cpu.output),
                };
                cpu.model
                    .forward_block(&cpu.cache, &block, start, &borrowed)
                    .map_err(|e| SpecError::DraftRefused {
                        source: Box::new(e),
                    })
            }
            DraftBackendImpl::Gpu { session } => session
                .forward_block_capped(&block, start, row_cap.min(block.len()))
                .map_err(|e| SpecError::DraftRefused {
                    source: Box::new(e),
                }),
        }
    }

    /// Inject fused conditioning rows on the active arm.
    fn inject(&mut self, features: &[f32], positions: &[usize]) -> Result<()> {
        match &mut self.backend {
            DraftBackendImpl::Cpu(cpu) => cpu
                .model
                .inject(&mut cpu.cache, features, positions)
                .map_err(|e| SpecError::DraftRefused {
                    source: Box::new(e),
                }),
            DraftBackendImpl::Gpu { session } => {
                session
                    .inject(features, positions)
                    .map_err(|e| SpecError::DraftRefused {
                        source: Box::new(e),
                    })
            }
        }
    }

    /// How many committed positions the draft's cache currently holds.
    /// At every round boundary this equals the target's cache position.
    pub fn committed_len(&self) -> usize {
        match &self.backend {
            DraftBackendImpl::Cpu(cpu) => cpu.cache.len(),
            DraftBackendImpl::Gpu { session } => session.committed_len(),
        }
    }

    /// Run one full noise block after `pending` and return up to `n`
    /// proposals. The block always runs at the draft's full block size:
    /// block attention is non-causal, so a truncated block would change
    /// the earlier rows' distributions; the session slices the result.
    /// The slice bound is known before the forward, so it goes in as
    /// the readback cap: rows past it never cross to the host on the
    /// GPU arm.
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
        let want = n.min(self.config.max_drafts() as usize);
        // Rows 0..=want are the only rows sliced below: row 0
        // is the anchor's own, rows 1..=want back the drafts.
        let logits = self.forward_block_logits(pending, want + 1)?;
        let vocab = self.config.vocab_size as usize;
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
                    let dist = sampler::distribution(row, temperature, 1.0)?;
                    tokens.push(sampler::sample_from(&dist, rng()));
                    dists.push(dist);
                }
            }
        }
        // The sampled arm's distributions back the trait's
        // borrowed view, so they are kept. The greedy arm's
        // one-hot view is NOT materialized here: at the real
        // vocab it is megabytes of zeros per round on the
        // loop's hot path (measured at about 2 ms of a
        // 217 ms round), and the one consumer, the trait
        // shim below, rebuilds it from the tokens on demand.
        self.last_probs = match arm {
            DraftArm::Greedy => Vec::new(),
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
        self.config.max_drafts() as usize
    }

    fn propose(&mut self, pending: u32, n: usize) -> Result<BlockProposal<'_>> {
        let mut no_draws = || 0.0f64;
        let proposal = self.propose_block(pending, n, DraftArm::Greedy, &mut no_draws)?;
        let _ = proposal;
        if self.last_probs.is_empty() {
            // Greedy proposals carry no stored distributions
            // (see propose_block); the trait's contract is a
            // one-hot view of the proposed tokens, built here
            // on the shim's cold path.
            let vocab = self.config.vocab_size as usize;
            self.last_probs = self
                .last_tokens
                .iter()
                .map(|&t| {
                    let mut one_hot = vec![0.0f32; vocab];
                    one_hot[t as usize] = 1.0;
                    one_hot
                })
                .collect();
        }
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
        let from = self.committed_len();
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
        let width = self.conditioning_width;
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
        self.inject(&features, &positions)?;
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
        let target_hidden = self.target_hidden;
        let width = self.conditioning_width;
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

    /// The GPU arm's shared bookkeeping: a speculator built
    /// over a resident draft session stages, commits, and
    /// proposes through exactly the code the CPU arm uses,
    /// and its committed length tracks the GPU session's.
    /// Adapter-guarded like every GPU test in the workspace:
    /// skips with a printed reason where no adapter
    /// qualifies. The fixture is the smallest draft the
    /// GEMV kernels accept (every projection input a whole
    /// number of Q4_K blocks).
    #[test]
    fn gpu_arm_shares_the_speculator_bookkeeping() {
        use muse_glimmer::{Dtype, GlimmerError, TensorSource as GlimmerSource, TensorView};
        use std::collections::HashMap;
        use std::sync::Arc as StdArc;
        use wgpu_compute::{
            DflashRawWeights, DflashSession, DflashSessionConfig, WgpuCompute, WgpuError,
        };

        let backend = match WgpuCompute::new() {
            Ok(b) => b,
            Err(WgpuError::NoSuitableAdapter { seen, .. }) => {
                eprintln!(
                    "skipping gpu arm speculator test: no suitable adapter (enumerated: [{}])",
                    seen.join("; ")
                );
                return;
            }
            Err(e) => panic!("backend construction failed unexpectedly: {e}"),
        };
        let backend: &'static WgpuCompute = Box::leak(Box::new(backend));

        // Deterministic small-magnitude Q4_K / F32 tensors,
        // the parity fixture's by-construction style.
        fn q4k(name: &str, inn: usize, out: usize, seed: u8) -> TensorView {
            let blocks = inn * out / 256;
            let mut data = Vec::with_capacity(blocks * 144);
            for b in 0..blocks {
                data.extend_from_slice(
                    &wgpu_compute::kernels::k1_q4k_gemv::f32_to_f16_bits(0.005).to_le_bytes(),
                );
                data.extend_from_slice(
                    &wgpu_compute::kernels::k1_q4k_gemv::f32_to_f16_bits(0.002).to_le_bytes(),
                );
                for i in 0..12 {
                    data.push(seed.wrapping_add(i as u8) & 0x07);
                }
                for i in 0..128 {
                    data.push(seed.wrapping_add((b + i) as u8));
                }
            }
            TensorView {
                name: name.to_string(),
                dtype: Dtype::Q4K,
                shape: vec![inn as u64, out as u64],
                data: StdArc::from(data.into_boxed_slice()),
            }
        }
        fn f32t(name: &str, n: usize, center: f32) -> TensorView {
            let data: Vec<u8> = (0..n)
                .flat_map(|i| (center + (i % 5) as f32 * 0.01).to_le_bytes())
                .collect();
            TensorView {
                name: name.to_string(),
                dtype: Dtype::F32,
                shape: vec![n as u64],
                data: StdArc::from(data.into_boxed_slice()),
            }
        }
        struct MapSource {
            tensors: HashMap<String, TensorView>,
        }
        impl GlimmerSource for MapSource {
            fn tensor(&self, name: &str) -> muse_glimmer::Result<TensorView> {
                self.tensors
                    .get(name)
                    .cloned()
                    .ok_or_else(|| GlimmerError::MissingTensor {
                        name: name.to_string(),
                    })
            }
        }

        let cfg = dflash::DflashConfig {
            name: None,
            block_count: 2,
            context_length: 64,
            hidden: 256,
            ffn_hidden: 256,
            q_heads: 2,
            kv_heads: 1,
            key_length: 128,
            value_length: 128,
            rope_freq_base: 10_000.0,
            rms_eps: 1e-5,
            block_size: 2,
            target_layers: vec![0, 1],
            sliding_window: 2048,
            sliding_pattern: vec![false, false],
            causal_attention: false,
            mask_token_id: 255,
            vocab_size: 256,
        };
        let mut tensors: HashMap<String, TensorView> = HashMap::new();
        let mut put = |v: TensorView| {
            tensors.insert(v.name.clone(), v);
        };
        put(q4k("fc.weight", 512, 256, 1));
        put(f32t("enc.output_norm.weight", 256, 1.0));
        put(f32t("output_norm.weight", 256, 1.0));
        for layer in 0..2 {
            let p = |s: &str| format!("blk.{layer}.{s}");
            put(f32t(&p("attn_norm.weight"), 256, 1.0));
            put(q4k(&p("attn_q.weight"), 256, 256, 2));
            put(q4k(&p("attn_k.weight"), 256, 128, 3));
            put(q4k(&p("attn_v.weight"), 256, 128, 4));
            put(f32t(&p("attn_q_norm.weight"), 128, 1.0));
            put(f32t(&p("attn_k_norm.weight"), 128, 1.0));
            put(q4k(&p("attn_output.weight"), 256, 256, 5));
            put(f32t(&p("ffn_norm.weight"), 256, 1.0));
            put(q4k(&p("ffn_gate.weight"), 256, 256, 6));
            put(q4k(&p("ffn_up.weight"), 256, 256, 7));
            put(q4k(&p("ffn_down.weight"), 256, 256, 8));
        }
        let raw = DflashRawWeights::load(&MapSource { tensors }, &cfg).expect("raw loads");
        let embedding = q4k("token_embd.weight", 256, 256, 9);
        let output = q4k("output.weight", 256, 256, 10);
        let session = DflashSession::new(
            backend,
            &cfg,
            &raw,
            Some(&embedding),
            Some(&output),
            DflashSessionConfig { kv_capacity: 32 },
        )
        .expect("draft session builds");
        let mut spec = DflashSpeculator::gpu(session);
        assert_eq!(spec.config().block_size, 2);
        assert!(spec.model().is_none(), "the GPU arm has no CPU model");
        assert_eq!(spec.committed_len(), 0);

        // Commit with nothing staged refuses, arm-independent.
        assert!(matches!(
            spec.commit(&[1]),
            Err(SpecError::SessionState { .. })
        ));

        // Stage three positions of two-layer captures and
        // commit them: the GPU session's held set advances
        // through the shared staging code.
        let caps: Vec<LayerInputCapture> = [0usize, 1]
            .iter()
            .map(|&layer| LayerInputCapture {
                layer,
                rows: (0..3 * 256).map(|i| (i % 7) as f32 * 0.01).collect(),
            })
            .collect();
        spec.note_captures(0, &caps).expect("stage");
        spec.commit(&[10, 11, 12]).expect("gpu commit");
        assert_eq!(spec.committed_len(), 3);

        // A proposal runs the GPU forward and returns
        // in-vocabulary tokens at the requested depth.
        let mut no_draws = || 0.0f64;
        let proposal = spec
            .propose_block(12, 1, DraftArm::Greedy, &mut no_draws)
            .expect("gpu propose");
        assert_eq!(proposal.tokens.len(), 1);
        assert!(proposal.tokens[0] < cfg.vocab_size);
        assert_eq!(spec.committed_len(), 3, "proposing commits nothing");
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
