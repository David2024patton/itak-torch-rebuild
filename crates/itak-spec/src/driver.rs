//! The target driver seam: the one surface the speculative loop
//! drives its target through (TODO T-9.6).
//!
//! The loop's bookkeeping (checkpoint, verify block, rewind to the
//! committed count) was written against the CPU target path. This
//! trait names exactly the operations the loop performs, no more,
//! so another target backend can sit under the identical loop:
//! [`CpuTarget`] wraps the original CPU path (model, shared
//! compute, position cache) unchanged, and `ResidentTarget` (in
//! `resident`) drives the GPU-resident session of `wgpu-compute`.
//! The draft side is untouched by this seam: drafting stays behind
//! the `Speculator` trait in `speculator`, on whichever backend the
//! draft model itself runs.
//!
//! The contract every driver keeps, because the loop's proof
//! obligation rests on it:
//!
//! - `forward` advances the position by `tokens.len()` and returns
//!   the finished logits of the last token.
//! - `forward_block_capturing` advances by `tokens.len()` and
//!   returns every row's finished logits, row-major
//!   `[tokens][vocab]`, row `i` the distribution for the token
//!   following the prefix extended by `tokens[..=i]`, plus the
//!   named layers' input rows in position order.
//! - `checkpoint` snapshots the KV state before a verify block;
//!   `rewind` restores it to an earlier committed position;
//!   `release` drops the snapshot. A driver whose KV addressing
//!   makes rewind exact by construction (position-addressed
//!   slots, views derived from the position) may implement
//!   checkpoint and release as bookkeeping only, and must say so
//!   in its own docs.

use std::rc::Rc;

use cpu_compute::{Checkpoint, CpuCompute};
use muse_glimmer::{
    Cache, GlimmerConfig, GlimmerModel, LayerInputCapture, PositionCache, Result as GlimmerResult,
};

/// The target operations the speculative loop performs. See the
/// module docs for the contract each method keeps.
pub trait TargetDriver {
    /// Opaque snapshot of the driver's KV state, taken before a
    /// verify block so a refused or partially accepted round can
    /// restore it.
    type Checkpoint;

    /// The target model's configuration (the session validates
    /// the draft against it at creation).
    fn config(&self) -> &GlimmerConfig;

    /// The committed position: tokens forwarded and not rewound.
    fn position(&self) -> usize;

    /// Forward `tokens`, returning the last token's finished
    /// logits.
    fn forward(&mut self, tokens: &[u32]) -> GlimmerResult<Vec<f32>>;

    /// Forward `tokens`, returning the last token's finished
    /// logits plus the named layers' input rows.
    fn forward_capturing(
        &mut self,
        tokens: &[u32],
        capture_layers: &[usize],
    ) -> GlimmerResult<(Vec<f32>, Vec<LayerInputCapture>)>;

    /// Forward `tokens` as one block, returning every row's
    /// finished logits plus the named layers' input rows.
    fn forward_block_capturing(
        &mut self,
        tokens: &[u32],
        capture_layers: &[usize],
    ) -> GlimmerResult<(Vec<f32>, Vec<LayerInputCapture>)>;

    /// Snapshot the KV state ahead of a verify block of at most
    /// `rows` tokens.
    fn checkpoint(&mut self, rows: usize) -> GlimmerResult<Self::Checkpoint>;

    /// Restore the KV state to the committed position `to`,
    /// under the snapshot `checkpoint` took.
    fn rewind(&mut self, checkpoint: &Self::Checkpoint, to: usize) -> GlimmerResult<()>;

    /// Drop the snapshot `checkpoint` took (the round committed
    /// or was rolled back; the snapshot is spent either way).
    fn release(&mut self, checkpoint: &Self::Checkpoint) -> GlimmerResult<()>;

    /// Record one completed speculative round: the draft's
    /// proposed token ids and how many of them the target
    /// accepted. The loop calls this exactly once per
    /// completed round, at the point where both are in hand
    /// (the round's outcome carries counts only, never the
    /// ids). Record only: an implementation must not let
    /// this change sampling, acceptance, or control flow,
    /// and drivers with nothing to record (the default, the
    /// CPU driver included) do nothing. The resident driver
    /// forwards it to the session's capture diag, where it
    /// lands only at the capture level.
    fn note_spec_round(&mut self, _proposed: &[u32], _accepted: usize) {}
}

/// The CPU target driver: the loop's original target path,
/// unchanged. The model forwards through the shared compute (see
/// [`crate::SharedCompute`]); checkpoints, rewind, and release go
/// to that same compute, and the position cache rewinds between
/// the compute rewind and the release, exactly the order the
/// loop always used.
pub struct CpuTarget {
    model: GlimmerModel,
    compute: Rc<CpuCompute>,
    cache: PositionCache,
}

impl CpuTarget {
    /// Wrap a CPU target model and the compute it forwards
    /// through. The compute must be the same [`CpuCompute`] the
    /// model was built with (through [`crate::SharedCompute`]):
    /// the checkpoint machinery lives on it.
    pub fn new(model: GlimmerModel, compute: Rc<CpuCompute>) -> Self {
        CpuTarget {
            model,
            compute,
            cache: PositionCache::new(),
        }
    }

    /// The wrapped model.
    pub fn model(&self) -> &GlimmerModel {
        &self.model
    }

    /// The shared compute the model forwards through.
    pub fn compute(&self) -> &Rc<CpuCompute> {
        &self.compute
    }

    /// The position cache paired with the compute's KV state.
    pub fn cache(&self) -> &PositionCache {
        &self.cache
    }
}

impl TargetDriver for CpuTarget {
    type Checkpoint = Checkpoint;

    fn config(&self) -> &GlimmerConfig {
        self.model.config()
    }

    fn position(&self) -> usize {
        self.cache.position()
    }

    fn forward(&mut self, tokens: &[u32]) -> GlimmerResult<Vec<f32>> {
        self.model.forward(tokens, &mut self.cache)
    }

    fn forward_capturing(
        &mut self,
        tokens: &[u32],
        capture_layers: &[usize],
    ) -> GlimmerResult<(Vec<f32>, Vec<LayerInputCapture>)> {
        self.model
            .forward_capturing(tokens, &mut self.cache, capture_layers)
    }

    fn forward_block_capturing(
        &mut self,
        tokens: &[u32],
        capture_layers: &[usize],
    ) -> GlimmerResult<(Vec<f32>, Vec<LayerInputCapture>)> {
        self.model
            .forward_block_capturing(tokens, &mut self.cache, capture_layers)
    }

    fn checkpoint(&mut self, rows: usize) -> GlimmerResult<Checkpoint> {
        self.compute.cache_checkpoint(rows)
    }

    fn rewind(&mut self, checkpoint: &Checkpoint, to: usize) -> GlimmerResult<()> {
        self.compute.cache_rewind(checkpoint, to)?;
        self.cache.rewind_to(to);
        Ok(())
    }

    fn release(&mut self, checkpoint: &Checkpoint) -> GlimmerResult<()> {
        self.compute.cache_release(checkpoint)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;
    use std::sync::Arc;

    use muse_glimmer::{GlimmerConfig, GlimmerError, LayerInputCapture};

    use super::TargetDriver;
    use crate::config::SpecConfig;
    use crate::diag::NullDiag;
    use crate::session::SpecSession;
    use crate::speculator::SharedCompute;
    use crate::testkit;

    /// A deterministic stub target for loop-bookkeeping tests:
    /// each token's finished logits depend only on that token
    /// (a bigram-style rule), so a block forward agrees with
    /// sequential forwards by construction and greedy identity
    /// between a speculative and a plain session over the stub
    /// is a property of the LOOP, not of any model arithmetic.
    /// Its KV is the vector of forwarded tokens; a rewind
    /// truncates it. Checkpoint, rewind, and release calls are
    /// recorded for the accounting assertions.
    /// The stub's spec-round record log: one (proposed ids,
    /// accepted count) entry per completed round, shared
    /// with the owning test.
    type SpecRoundNotes = Rc<RefCell<Vec<(Vec<u32>, usize)>>>;

    struct StubTarget {
        config: GlimmerConfig,
        kv: Vec<u32>,
        checkpoints: Rc<RefCell<Vec<usize>>>,
        rewinds: Rc<RefCell<Vec<usize>>>,
        releases: Rc<Cell<usize>>,
        spec_rounds: SpecRoundNotes,
    }

    /// A stub checkpoint: the KV length it saw, plus the row
    /// budget it was taken for. A round's rewind target lands
    /// anywhere in `[base, base + rows]`: the committed count
    /// after a partial acceptance sits inside the forwarded
    /// block, past the checkpoint's own length (the CPU
    /// checkpoint budgets the same span).
    struct StubCheckpoint {
        base: usize,
        rows: usize,
    }

    /// The stub's recorded seam calls, shared with the test
    /// that owns the stub.
    struct StubRecorders {
        checkpoints: Rc<RefCell<Vec<usize>>>,
        rewinds: Rc<RefCell<Vec<usize>>>,
        releases: Rc<Cell<usize>>,
        spec_rounds: SpecRoundNotes,
    }

    impl StubTarget {
        fn new() -> (StubTarget, StubRecorders) {
            let recorders = StubRecorders {
                checkpoints: Rc::new(RefCell::new(Vec::new())),
                rewinds: Rc::new(RefCell::new(Vec::new())),
                releases: Rc::new(Cell::new(0)),
                spec_rounds: Rc::new(RefCell::new(Vec::new())),
            };
            (
                StubTarget {
                    config: testkit::target_config(),
                    kv: Vec::new(),
                    checkpoints: recorders.checkpoints.clone(),
                    rewinds: recorders.rewinds.clone(),
                    releases: recorders.releases.clone(),
                    spec_rounds: recorders.spec_rounds.clone(),
                },
                recorders,
            )
        }

        /// The stub's finished logits for one token: a one-hot
        /// row at a deterministic successor, so the greedy pick
        /// is a pure function of the token.
        fn row(&self, token: u32) -> Vec<f32> {
            let vocab = self.config.vocab_size as usize;
            let mut row = vec![0.0f32; vocab];
            row[(token as usize * 31 + 17) % vocab] = 1.0;
            row
        }

        /// Deterministic capture rows for `tokens` at `layer`:
        /// one row of hidden width per token.
        fn capture_rows(&self, tokens: &[u32], layer: usize) -> Vec<f32> {
            let hidden = self.config.hidden as usize;
            let mut rows = Vec::with_capacity(tokens.len() * hidden);
            for &token in tokens {
                let value = token as f32 * 0.001 + layer as f32;
                rows.extend(std::iter::repeat_n(value, hidden));
            }
            rows
        }

        fn captures(&self, tokens: &[u32], layers: &[usize]) -> Vec<LayerInputCapture> {
            layers
                .iter()
                .map(|&layer| LayerInputCapture {
                    layer,
                    rows: self.capture_rows(tokens, layer),
                })
                .collect()
        }
    }

    impl TargetDriver for StubTarget {
        type Checkpoint = StubCheckpoint;

        fn config(&self) -> &GlimmerConfig {
            &self.config
        }

        fn position(&self) -> usize {
            self.kv.len()
        }

        fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, GlimmerError> {
            self.kv.extend_from_slice(tokens);
            Ok(self.row(*tokens.last().expect("forward needs a token")))
        }

        fn forward_capturing(
            &mut self,
            tokens: &[u32],
            capture_layers: &[usize],
        ) -> Result<(Vec<f32>, Vec<LayerInputCapture>), GlimmerError> {
            let captures = self.captures(tokens, capture_layers);
            self.kv.extend_from_slice(tokens);
            Ok((
                self.row(*tokens.last().expect("forward needs a token")),
                captures,
            ))
        }

        fn forward_block_capturing(
            &mut self,
            tokens: &[u32],
            capture_layers: &[usize],
        ) -> Result<(Vec<f32>, Vec<LayerInputCapture>), GlimmerError> {
            let captures = self.captures(tokens, capture_layers);
            let mut logits = Vec::with_capacity(tokens.len() * self.config.vocab_size as usize);
            for &token in tokens {
                logits.extend_from_slice(&self.row(token));
            }
            self.kv.extend_from_slice(tokens);
            Ok((logits, captures))
        }

        fn checkpoint(&mut self, rows: usize) -> Result<StubCheckpoint, GlimmerError> {
            self.checkpoints.borrow_mut().push(rows);
            Ok(StubCheckpoint {
                base: self.kv.len(),
                rows,
            })
        }

        fn rewind(&mut self, checkpoint: &StubCheckpoint, to: usize) -> Result<(), GlimmerError> {
            assert!(
                (checkpoint.base..=checkpoint.base + checkpoint.rows).contains(&to),
                "rewind target {to} outside the checkpoint span [{}, {}]",
                checkpoint.base,
                checkpoint.base + checkpoint.rows
            );
            self.rewinds.borrow_mut().push(to);
            self.kv.truncate(to);
            Ok(())
        }

        fn release(&mut self, _checkpoint: &StubCheckpoint) -> Result<(), GlimmerError> {
            self.releases.set(self.releases.get() + 1);
            Ok(())
        }

        fn note_spec_round(&mut self, proposed: &[u32], accepted: usize) {
            self.spec_rounds
                .borrow_mut()
                .push((proposed.to_vec(), accepted));
        }
    }

    fn fixture_draft(block_size: u32) -> crate::speculator::DflashSpeculator {
        let compute = Rc::new(cpu_compute::CpuCompute::new());
        let fixture = testkit::build_target(Box::new(SharedCompute(compute)));
        testkit::build_speculator(&fixture, block_size)
    }

    const PROMPT: &[u32] = &[3, 17, 42];

    /// The driver seam carries the full loop: a DFlash session
    /// over the stub driver emits exactly the stream a mode-off
    /// session over a fresh stub emits (greedy identity through
    /// the seam, proven without any model arithmetic), and the
    /// draft really proposed.
    #[test]
    fn stub_driver_carries_the_loop_with_greedy_identity() {
        let config = SpecConfig {
            depth: 2,
            ..SpecConfig::default()
        };
        let (stub, _) = StubTarget::new();
        let mut spec = SpecSession::new(
            stub,
            Some(fixture_draft(5)),
            config.clone(),
            Arc::new(NullDiag),
        )
        .expect("spec session over stub builds");
        spec.begin(PROMPT).expect("begin");
        let stream = spec.generate(16, &[]).expect("spec generate");

        let (plain_stub, _) = StubTarget::new();
        let mut plain = SpecSession::new(
            plain_stub,
            None,
            crate::config::SpecConfig {
                mode: crate::config::SpecMode::Off,
                ..config
            },
            Arc::new(NullDiag),
        )
        .expect("plain session over stub builds");
        plain.begin(PROMPT).expect("begin");
        let plain_stream = plain.generate(16, &[]).expect("plain generate");

        assert_eq!(
            stream, plain_stream,
            "greedy identity through the driver seam"
        );
        assert_eq!(stream.len(), 16);
        let stats = spec.stats();
        assert!(stats.proposed > 0, "the draft proposed through the seam");
        assert!(stats.accepted <= stats.proposed);
        assert_eq!(spec.cache_position(), PROMPT.len() + 15);
    }

    /// Rewind correctness and acceptance accounting, observed at
    /// the seam: after every round the stub's KV length equals
    /// the session's committed position and the round outcome's
    /// cache position; every recorded rewind target equals its
    /// round's base + 1 + accepted (recovered from the outcome
    /// and the running position); checkpoints and releases pair
    /// up one for one; and the cumulative stats equal the sum of
    /// the round outcomes.
    #[test]
    fn stub_driver_rewinds_land_on_the_committed_count() {
        let config = SpecConfig {
            depth: 3,
            ..SpecConfig::default()
        };
        let (stub, rec) = StubTarget::new();
        let mut session =
            SpecSession::new(stub, Some(fixture_draft(5)), config, Arc::new(NullDiag))
                .expect("spec session over stub builds");
        session.begin(PROMPT).expect("begin");

        let mut base = PROMPT.len();
        let mut sum_proposed = 0usize;
        let mut sum_accepted = 0usize;
        let mut rounds = 0usize;
        while session.cache_position() < PROMPT.len() + 20 && rounds < 40 {
            let outcome = session.round().expect("round");
            rounds += 1;
            sum_proposed += outcome.proposed;
            sum_accepted += outcome.accepted;
            // The stub's own KV is inside the driver; the session's
            // committed position and the outcome must agree, and
            // the last rewind of the round must have landed on it.
            assert_eq!(outcome.cache_position, base + 1 + outcome.accepted);
            assert_eq!(session.cache_position(), outcome.cache_position);
            let log = rec.rewinds.borrow();
            assert_eq!(
                log.last().copied(),
                Some(outcome.cache_position),
                "round {rounds}: rewind landed on the committed count"
            );
            drop(log);
            base = outcome.cache_position;
            if outcome.stopped {
                break;
            }
        }
        assert!(rounds > 0);
        let stats = session.stats();
        assert_eq!(stats.rounds, rounds);
        assert_eq!(stats.proposed, sum_proposed, "proposed accounting");
        assert_eq!(stats.accepted, sum_accepted, "accepted accounting");
        assert_eq!(
            rec.checkpoints.borrow().len(),
            rounds,
            "one checkpoint per speculative round"
        );
        assert!(
            rec.checkpoints.borrow().iter().all(|&rows| rows >= 2),
            "verify blocks always carry at least the anchor and one draft"
        );
        assert_eq!(
            rec.releases.get(),
            rounds,
            "every checkpoint is released exactly once"
        );
        assert_eq!(rec.rewinds.borrow().len(), rounds, "one rewind per round");
    }

    /// The capture wiring at the seam: the loop hands the
    /// driver exactly one spec-round record per completed
    /// round, carrying that round's REAL proposed ids and
    /// accepted count. The record's length is the round's
    /// proposed count, its count is the round's accepted
    /// count, and its accepted prefix is exactly the tokens
    /// the round emitted from the proposal (the session's
    /// default config names no stop tokens, so emission is
    /// never truncated). This is the call the resident
    /// driver forwards to its session's capture
    /// accumulator; here the stub records it so the wiring
    /// itself is pinned without a device.
    #[test]
    fn stub_driver_receives_one_spec_round_record_per_round() {
        let config = SpecConfig {
            depth: 3,
            ..SpecConfig::default()
        };
        let (stub, rec) = StubTarget::new();
        let mut session =
            SpecSession::new(stub, Some(fixture_draft(5)), config, Arc::new(NullDiag))
                .expect("spec session over stub builds");
        session.begin(PROMPT).expect("begin");

        let mut outcomes = Vec::new();
        while session.cache_position() < PROMPT.len() + 12 && outcomes.len() < 20 {
            let outcome = session.round().expect("round");
            let stopped = outcome.stopped;
            outcomes.push(outcome);
            if stopped {
                break;
            }
        }
        assert!(!outcomes.is_empty(), "at least one round ran");
        assert!(
            outcomes.iter().all(|o| o.proposed > 0),
            "every round here is speculative: {outcomes:?}"
        );
        let notes = rec.spec_rounds.borrow();
        assert_eq!(
            notes.len(),
            outcomes.len(),
            "one record per completed round"
        );
        for (round, (note, outcome)) in notes.iter().zip(outcomes.iter()).enumerate() {
            assert_eq!(
                note.0.len(),
                outcome.proposed,
                "round {round}: the record carries the round's proposed ids"
            );
            assert_eq!(
                note.1, outcome.accepted,
                "round {round}: the record carries the round's accepted count"
            );
            assert_eq!(
                &note.0[..outcome.accepted],
                &outcome.emitted[..outcome.accepted],
                "round {round}: the recorded ids are the round's real proposal, \
                 whose accepted prefix is what the round emitted"
            );
        }
    }
}
