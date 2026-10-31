//! The speculative session: one target, one draft, and the round
//! structure of SPEC-itak-spec Section 4.
//!
//! The loop invariant: the emitted stream is always `committed + pending`,
//! where `committed` is exactly the target cache's position and `pending`
//! is the last emitted token, not yet forwarded. A round proposes drafts
//! after `pending`, verifies `[pending, drafts...]` in one block forward
//! under a KV checkpoint, runs the correction walk from
//! [`crate::sampler`], rewinds the cache to the committed count, and tells
//! the draft exactly the newly committed tokens. At temperature 0 the
//! emitted stream is token-identical to non-speculative greedy generation;
//! that identity is the crate's standing proof obligation and is tested on
//! synthetic configs and on the real Glimmer files.

use std::collections::HashSet;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

use cpu_compute::CpuCompute;
use muse_glimmer::{Cache, GlimmerModel, PositionCache};

use crate::config::{OnVerifyError, SpecConfig, SpecMode};
use crate::diag::Diag;
use crate::error::{Result, SpecError};
use crate::rng::Rng;
use crate::sampler;
use crate::speculator::{cache_refused, DflashSpeculator, DraftArm, Speculator};

/// Cumulative session statistics, pullable at any time (the
/// `spec_stats()` readout of SPEC-itak-spec Section 6.4).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpecStats {
    /// Rounds run, including plain (non-speculative) rounds.
    pub rounds: usize,
    /// Draft tokens proposed across all rounds, including a latched
    /// round's discarded proposal (it was proposed, then the verify
    /// refused; the count stays honest).
    pub proposed: usize,
    /// Draft tokens accepted across all rounds.
    pub accepted: usize,
    /// Rounds whose correction walk rejected at least one draft token.
    pub rejections: usize,
    /// Rounds in which at least one speculatively forwarded position was
    /// discarded: partial acceptance, zero acceptance, or a verify
    /// refusal rollback.
    pub rollbacks: usize,
    /// Tokens emitted by rounds (the initial pending token from `begin`
    /// is not counted; it was emitted by the prefill).
    pub tokens_emitted: usize,
    /// Whether speculation has latched off after a verify refusal.
    pub fallback_latched: bool,
}

impl SpecStats {
    /// Accepted over proposed, in `[0, 1]`; zero when nothing was proposed.
    pub fn acceptance_rate(&self) -> f64 {
        if self.proposed == 0 {
            0.0
        } else {
            self.accepted as f64 / self.proposed as f64
        }
    }

    /// Round-emitted tokens per round; zero before the first round.
    pub fn tokens_per_round(&self) -> f64 {
        if self.rounds == 0 {
            0.0
        } else {
            self.tokens_emitted as f64 / self.rounds as f64
        }
    }
}

/// What one round did (SPEC-itak-spec Section 6.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoundOutcome {
    /// Tokens this round emitted (stop-truncated when `stopped`).
    pub emitted: Vec<u32>,
    /// Draft tokens proposed this round.
    pub proposed: usize,
    /// Draft tokens accepted this round.
    pub accepted: usize,
    /// 1-based index of the first rejected draft token, if any.
    pub first_rejection: Option<usize>,
    /// Target cache position after the round's commit.
    pub cache_position: usize,
    /// Whether a stop token ended the session in this round.
    pub stopped: bool,
}

/// A speculative decoding session over one target model.
pub struct SpecSession {
    target: GlimmerModel,
    compute: Rc<CpuCompute>,
    cache: PositionCache,
    speculator: Option<DflashSpeculator>,
    capture_layers: Vec<usize>,
    config: SpecConfig,
    diag: Arc<dyn Diag>,
    rng: Rng,
    pending: u32,
    begun: bool,
    ended: bool,
    poisoned: Option<String>,
    latched_off: bool,
    stats: SpecStats,
}

impl SpecSession {
    /// Create a session, validating everything up front (SPEC-itak-spec
    /// Section 6.1): mode availability, draft presence and agreement with
    /// the target, depth against the draft block and the round bound, and
    /// the grammar conflict. Any refusal emits `spec.refused` and returns
    /// the named error; a session that exists is a session whose rounds
    /// can run.
    ///
    /// `compute` must be the same [`CpuCompute`] the target model forwards
    /// through (see [`crate::SharedCompute`]): the loop drives its
    /// checkpoint machinery directly.
    pub fn new(
        target: GlimmerModel,
        compute: Rc<CpuCompute>,
        draft: Option<DflashSpeculator>,
        config: SpecConfig,
        diag: Arc<dyn Diag>,
    ) -> Result<Self> {
        let refuse = |err: SpecError| -> SpecError {
            diag.event("spec.refused", &format!("error={err}"));
            err
        };
        if config.grammar_active {
            return Err(refuse(SpecError::GrammarConflict));
        }
        match config.mode {
            SpecMode::Ngram | SpecMode::Medusa => {
                return Err(refuse(SpecError::ModeUnavailable {
                    mode: config.mode.name(),
                }));
            }
            SpecMode::Off | SpecMode::Dflash => {}
        }
        let mut capture_layers: Vec<usize> = Vec::new();
        if config.mode == SpecMode::Dflash {
            let spec = draft
                .as_ref()
                .ok_or_else(|| refuse(SpecError::DraftNotLoaded))?;
            let dcfg = spec.model().config();
            let tcfg = target.config();
            if dcfg.vocab_size != tcfg.vocab_size {
                return Err(refuse(SpecError::DraftTargetMismatch {
                    field: "vocab_size",
                    expected: tcfg.vocab_size as usize,
                    actual: dcfg.vocab_size as usize,
                }));
            }
            if dcfg.target_layers.is_empty() {
                return Err(refuse(SpecError::DraftTargetMismatch {
                    field: "target_layers",
                    expected: 1,
                    actual: 0,
                }));
            }
            for &layer in &dcfg.target_layers {
                if layer as usize >= tcfg.block_count as usize {
                    return Err(refuse(SpecError::DraftTargetMismatch {
                        field: "target_layer",
                        expected: tcfg.block_count as usize,
                        actual: layer as usize,
                    }));
                }
                capture_layers.push(layer as usize);
            }
            if config.depth == 0 || config.depth > spec.max_depth() {
                return Err(refuse(SpecError::DepthOutOfRange {
                    n_max: config.depth,
                    block_size: dcfg.block_size as usize,
                }));
            }
            if config.depth + 1 > config.max_round_tokens {
                return Err(refuse(SpecError::RoundBoundTooSmall {
                    depth: config.depth,
                    max_round_tokens: config.max_round_tokens,
                }));
            }
        }
        let rng = Rng::new(config.seed);
        let session = SpecSession {
            target,
            compute,
            cache: PositionCache::new(),
            speculator: draft,
            capture_layers,
            config,
            diag,
            rng,
            pending: 0,
            begun: false,
            ended: false,
            poisoned: None,
            latched_off: false,
            stats: SpecStats::default(),
        };
        if session.config.mode == SpecMode::Off {
            session.diag.event(
                "spec.disabled",
                "mode=off plain generation, no draft attached",
            );
            session
                .diag
                .event("spec.session.begin", "mode=off depth=0 draft=none");
        } else {
            session.diag.event(
                "spec.session.begin",
                &format!(
                    "mode={} depth={} max_round_tokens={} temperature={}",
                    session.config.mode.name(),
                    session.config.depth,
                    session.config.max_round_tokens,
                    session.config.temperature
                ),
            );
        }
        Ok(session)
    }

    /// The configuration this session was created with.
    pub fn config(&self) -> &SpecConfig {
        &self.config
    }

    /// Cumulative statistics so far.
    pub fn stats(&self) -> SpecStats {
        self.stats
    }

    /// The `spec_stats()` readout: the same statistics [`Self::stats`]
    /// returns, named as the spec names it for the engine to expose later.
    pub fn spec_stats(&self) -> SpecStats {
        self.stats
    }

    /// Whether speculation has latched off after a verify refusal.
    pub fn is_latched(&self) -> bool {
        self.latched_off
    }

    /// The last emitted token, not yet forwarded (valid after `begin`).
    pub fn pending(&self) -> u32 {
        self.pending
    }

    /// The target cache's committed position.
    pub fn cache_position(&self) -> usize {
        self.cache.position()
    }

    /// Prefill the prompt and draw the first token. Returns the pending
    /// token: the first element of the generated stream. The prompt's
    /// captured layer inputs are injected into the draft here, so the
    /// draft starts the first round conditioned on exactly the prompt.
    pub fn begin(&mut self, prompt: &[u32]) -> Result<u32> {
        if self.begun {
            return Err(SpecError::SessionState {
                detail: "begin called twice".to_string(),
            });
        }
        let logits = match self.speculator.as_mut() {
            Some(spec) if self.config.mode == SpecMode::Dflash => {
                let (logits, captures) = self
                    .target
                    .forward_capturing(prompt, &mut self.cache, &self.capture_layers)
                    .map_err(|e| SpecError::TargetRefused {
                        source: Box::new(e),
                    })?;
                spec.note_captures(0, &captures)?;
                spec.commit(prompt)?;
                logits
            }
            _ => self.target.forward(prompt, &mut self.cache).map_err(|e| {
                SpecError::TargetRefused {
                    source: Box::new(e),
                }
            })?,
        };
        self.pending = self.draw(&logits);
        self.begun = true;
        Ok(self.pending)
    }

    /// Run one round against the configured stop set.
    pub fn round(&mut self) -> Result<RoundOutcome> {
        let stops: HashSet<u32> = self.config.eos_tokens.iter().copied().collect();
        self.round_internal(None, &stops)
    }

    /// Generate up to `max_new_tokens` tokens, stopping early at any stop
    /// token (the config's `eos_tokens` plus `extra_stops`). The returned
    /// stream starts with the pending token from `begin`; its length is
    /// exactly `max_new_tokens` unless a stop token ended the session.
    /// Round sizes are budget-capped so the stream never overshoots.
    pub fn generate(&mut self, max_new_tokens: usize, extra_stops: &[u32]) -> Result<Vec<u32>> {
        if !self.begun {
            return Err(SpecError::SessionState {
                detail: "generate called before begin".to_string(),
            });
        }
        if max_new_tokens == 0 {
            return Ok(Vec::new());
        }
        let mut stops: HashSet<u32> = self.config.eos_tokens.iter().copied().collect();
        stops.extend(extra_stops.iter().copied());
        let mut stream = vec![self.pending];
        if stops.contains(&self.pending) {
            self.ended = true;
            return Ok(stream);
        }
        while stream.len() < max_new_tokens && !self.ended {
            let cap = max_new_tokens - stream.len() - 1;
            let outcome = self.round_internal(Some(cap), &stops)?;
            stream.extend_from_slice(&outcome.emitted);
        }
        Ok(stream)
    }

    /// End the session, emitting `spec.session.end`, and return the final
    /// statistics.
    pub fn finish(self) -> SpecStats {
        self.diag.event(
            "spec.session.end",
            &format!(
                "rounds={} proposed={} accepted={} acceptance_rate={:.4} tokens_emitted={} fallback_latched={}",
                self.stats.rounds,
                self.stats.proposed,
                self.stats.accepted,
                self.stats.acceptance_rate(),
                self.stats.tokens_emitted,
                self.stats.fallback_latched
            ),
        );
        self.stats
    }

    /// Draw one token from a finished target logit row under the session's
    /// sampling configuration.
    fn draw(&mut self, logits: &[f32]) -> u32 {
        if self.config.temperature <= 0.0 {
            sampler::argmax(logits)
        } else {
            let dist = sampler::distribution(logits, self.config.temperature, self.config.top_p);
            sampler::sample_from(&dist, self.rng.next_f64())
        }
    }

    fn round_internal(&mut self, cap: Option<usize>, stops: &HashSet<u32>) -> Result<RoundOutcome> {
        if !self.begun {
            return Err(SpecError::SessionState {
                detail: "round called before begin".to_string(),
            });
        }
        if self.ended {
            return Err(SpecError::SessionState {
                detail: "round called after a stop token ended the session".to_string(),
            });
        }
        if let Some(detail) = &self.poisoned {
            return Err(SpecError::SessionState {
                detail: format!("session is poisoned by an earlier failure: {detail}"),
            });
        }
        self.stats.rounds += 1;
        let round_no = self.stats.rounds;
        let base = self.cache.position();
        self.diag.event(
            "spec.round.begin",
            &format!("round={round_no} base={base} pending={}", self.pending),
        );
        if self.config.mode == SpecMode::Off || self.latched_off || self.speculator.is_none() {
            return self.plain_step(round_no, base, 0, stops);
        }
        self.speculative_round(round_no, base, cap, stops)
    }

    /// One plain step: forward the pending token, draw the next. Used for
    /// mode-off sessions and as the latched fallback's round completion.
    /// The draft is never touched here: mode-off sessions have no live
    /// draft, and a latched session has abandoned its draft for good.
    fn plain_step(
        &mut self,
        round_no: usize,
        base: usize,
        proposed_count: usize,
        stops: &HashSet<u32>,
    ) -> Result<RoundOutcome> {
        let pending = self.pending;
        let logits = self
            .target
            .forward(&[pending], &mut self.cache)
            .map_err(|e| SpecError::TargetRefused {
                source: Box::new(e),
            })?;
        let token = self.draw(&logits);
        self.pending = token;
        let emitted = vec![token];
        let mut stopped = false;
        if stops.contains(&token) {
            stopped = true;
            self.ended = true;
        }
        self.stats.proposed += proposed_count;
        self.stats.tokens_emitted += emitted.len();
        self.diag.event(
            "spec.cache.commit",
            &format!("round={round_no} position={}", base + 1),
        );
        self.diag.event(
            "spec.round.accept",
            &format!(
                "round={round_no} proposed={proposed_count} accepted=0 first_rejection=none emitted={} plain=true",
                emitted.len()
            ),
        );
        Ok(RoundOutcome {
            emitted,
            proposed: proposed_count,
            accepted: 0,
            first_rejection: None,
            cache_position: base + 1,
            stopped,
        })
    }

    /// Roll the target cache back to `to` and release the checkpoint.
    /// Any failure here poisons the session: cache state is no longer
    /// provably the committed prefix.
    fn rewind_and_release(
        &mut self,
        checkpoint: &crate::speculator::TargetCheckpoint,
        to: usize,
    ) -> Result<()> {
        if let Err(e) = self.compute.cache_rewind(checkpoint, to) {
            let err = cache_refused("rewind", e);
            self.poisoned = Some(err.to_string());
            return Err(err);
        }
        self.cache.rewind_to(to);
        if let Err(e) = self.compute.cache_release(checkpoint) {
            let err = cache_refused("release", e);
            self.poisoned = Some(err.to_string());
            return Err(err);
        }
        Ok(())
    }

    fn speculative_round(
        &mut self,
        round_no: usize,
        base: usize,
        cap: Option<usize>,
        stops: &HashSet<u32>,
    ) -> Result<RoundOutcome> {
        let pending = self.pending;
        let greedy = self.config.temperature <= 0.0;
        let mut n = self.config.depth.min(
            self.speculator
                .as_ref()
                .expect("speculative round requires a draft")
                .max_depth(),
        );
        if let Some(cap) = cap {
            n = n.min(cap);
        }

        // Step 2: propose. The draft conditions on everything committed so
        // far; its cache already holds exactly that prefix.
        let arm = if greedy {
            DraftArm::Greedy
        } else {
            DraftArm::Sampled {
                temperature: self
                    .config
                    .draft_temperature
                    .unwrap_or(self.config.temperature),
            }
        };
        let t0 = Instant::now();
        let proposal = {
            // Disjoint field borrows: the draft handle and the generator
            // are separate fields, so the draw closure can borrow the
            // generator while the speculator is borrowed too.
            let spec = self.speculator.as_mut().expect("checked above");
            let mut draw = || self.rng.next_f64();
            spec.propose_block(pending, n, arm, &mut draw)?
        };
        let draft_ms = t0.elapsed().as_millis();
        let n = proposal.tokens.len();
        self.diag.event(
            "spec.propose.end",
            &format!("round={round_no} proposed={n} draft_ms={draft_ms}"),
        );

        // Step 3: checkpoint, then one block forward over
        // [pending, drafts...] with the conditioning layers captured.
        let checkpoint = self
            .compute
            .cache_checkpoint(n + 1)
            .map_err(|e| cache_refused("checkpoint", e))?;
        let mut verify_tokens = Vec::with_capacity(n + 1);
        verify_tokens.push(pending);
        verify_tokens.extend_from_slice(&proposal.tokens);
        let t0 = Instant::now();
        let verified = self.target.forward_block_capturing(
            &verify_tokens,
            &mut self.cache,
            &self.capture_layers,
        );
        let verify_ms = t0.elapsed().as_millis();
        let (logits, captures) = match verified {
            Ok(pair) => pair,
            Err(e) => {
                // Step 8: roll back before anything else observes the
                // half-forwarded block, then apply the error policy.
                self.rewind_and_release(&checkpoint, base)?;
                self.stats.rollbacks += 1;
                self.diag.event(
                    "spec.cache.rewind",
                    &format!("round={round_no} from={base} to={base} reason=verify_refused"),
                );
                return match self.config.on_verify_error {
                    OnVerifyError::Refuse => {
                        let err = SpecError::VerifyRefused {
                            source: Box::new(e),
                        };
                        self.diag
                            .event("spec.refused", &format!("round={round_no} error={err}"));
                        Err(err)
                    }
                    OnVerifyError::LatchOff => {
                        self.latched_off = true;
                        self.stats.fallback_latched = true;
                        self.diag.event(
                            "spec.fallback.latched",
                            &format!("round={round_no} reason={e}"),
                        );
                        self.plain_step(round_no, base, n, stops)
                    }
                };
            }
        };
        self.diag.event(
            "spec.verify.end",
            &format!("round={round_no} verify_ms={verify_ms}"),
        );

        // Stage the fresh captures for the commit. If staging refuses,
        // the round cannot commit honestly: roll the target back too.
        if let Err(e) = self
            .speculator
            .as_mut()
            .expect("checked above")
            .note_captures(base, &captures)
        {
            self.rewind_and_release(&checkpoint, base)?;
            return Err(e);
        }

        // Step 4: the correction walk over finished distributions.
        let vocab = self.target.config().vocab_size as usize;
        let walk = if greedy {
            let p_argmax: Vec<u32> = logits.chunks_exact(vocab).map(sampler::argmax).collect();
            sampler::correct_greedy(&p_argmax, &proposal.tokens)
        } else {
            let p_rows: Vec<Vec<f32>> = logits
                .chunks_exact(vocab)
                .map(|row| sampler::distribution(row, self.config.temperature, self.config.top_p))
                .collect();
            let p_refs: Vec<&[f32]> = p_rows.iter().map(Vec::as_slice).collect();
            let q_refs: Vec<&[f32]> = proposal.dists.iter().map(Vec::as_slice).collect();
            let mut rng = || self.rng.next_f64();
            sampler::correct_sampled(&p_refs, &q_refs, &proposal.tokens, &mut rng)
        };
        let accepted = walk.accepted;
        let keep = base + 1 + accepted;

        // Step 5: commit. Rewind both position views to the committed
        // count, release the checkpoint, then tell the draft exactly the
        // newly committed tokens.
        self.rewind_and_release(&checkpoint, keep)?;
        self.diag.event(
            "spec.cache.rewind",
            &format!("round={round_no} from={} to={keep}", base + n + 1),
        );
        self.diag.event(
            "spec.cache.commit",
            &format!("round={round_no} position={keep}"),
        );
        if self.cache.position() != keep {
            let err = SpecError::CacheOutOfSync {
                expected: keep,
                actual: self.cache.position(),
            };
            self.poisoned = Some(err.to_string());
            self.diag
                .event("spec.refused", &format!("round={round_no} error={err}"));
            return Err(err);
        }
        if keep < base + n + 1 {
            self.stats.rollbacks += 1;
        }
        let committed = &verify_tokens[..1 + accepted];
        if let Err(e) = self
            .speculator
            .as_mut()
            .expect("checked above")
            .commit(committed)
        {
            self.poisoned = Some(e.to_string());
            return Err(e);
        }

        // Emission, with stop truncation (Section 4, step 6).
        let mut emitted: Vec<u32> = Vec::with_capacity(accepted + 1);
        emitted.extend_from_slice(&proposal.tokens[..accepted]);
        emitted.push(walk.token);
        let mut stopped = false;
        for (i, token) in emitted.iter().enumerate() {
            if stops.contains(token) {
                emitted.truncate(i + 1);
                stopped = true;
                break;
            }
        }
        if stopped {
            self.ended = true;
        }
        self.pending = walk.token;
        self.stats.proposed += n;
        self.stats.accepted += accepted;
        if walk.first_rejection.is_some() {
            self.stats.rejections += 1;
        }
        self.stats.tokens_emitted += emitted.len();
        self.diag.event(
            "spec.round.accept",
            &format!(
                "round={round_no} proposed={n} accepted={accepted} first_rejection={} emitted={}",
                walk.first_rejection
                    .map(|j| j.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                emitted.len()
            ),
        );
        Ok(RoundOutcome {
            emitted,
            proposed: n,
            accepted,
            first_rejection: walk.first_rejection,
            cache_position: keep,
            stopped,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;
    use std::sync::Arc;

    use cpu_compute::CpuCompute;

    use super::*;
    use crate::config::SpecMode;
    use crate::diag::NullDiag;
    use crate::speculator::SharedCompute;
    use crate::testkit;

    const PROMPT_A: &[u32] = &[3, 17, 42];
    const PROMPT_B: &[u32] = &[100, 5, 5, 9, 71];
    const PROMPT_C: &[u32] = &[7];

    fn greedy_config(depth: usize) -> SpecConfig {
        SpecConfig {
            depth,
            ..SpecConfig::default()
        }
    }

    /// THE proof obligation (SPEC-itak-spec Sections 4 and 9, TODO
    /// T-3.6): at temperature 0 the speculative stream equals the plain
    /// greedy stream token for token, at several depths and prompts.
    #[test]
    fn greedy_speculation_matches_plain_generation() {
        for (block_size, depths) in [(5u32, &[1usize, 2, 4][..]), (8, &[6][..])] {
            for &depth in depths {
                for prompt in [PROMPT_A, PROMPT_B, PROMPT_C] {
                    let mut session = testkit::build_session(greedy_config(depth), block_size);
                    let pending = session.begin(prompt).expect("begin");
                    let stream = session.generate(24, &[]).expect("generate");
                    let plain = testkit::plain_greedy(prompt, 24);
                    assert_eq!(pending, plain[0], "pending is the first plain token");
                    assert_eq!(
                        stream, plain,
                        "depth {depth} prompt {prompt:?}: speculative stream diverged"
                    );
                    let stats = session.spec_stats();
                    assert_eq!(stats, session.stats());
                    assert!(stats.proposed > 0, "draft must propose");
                    assert!(stats.accepted <= stats.proposed);
                    assert_eq!(stats.tokens_emitted, 23);
                    assert!((0.0..=1.0).contains(&stats.acceptance_rate()));
                    assert_eq!(session.cache_position(), prompt.len() + 23);
                }
            }
        }
    }

    /// The depth knob observably changes behavior: same output, more
    /// proposals per round at the larger depth.
    #[test]
    fn depth_knob_changes_proposal_volume_not_output() {
        let run = |depth: usize| {
            let mut session = testkit::build_session(greedy_config(depth), 5);
            session.begin(PROMPT_A).expect("begin");
            let stream = session.generate(24, &[]).expect("generate");
            (stream, session.stats())
        };
        let (stream_1, stats_1) = run(1);
        let (stream_4, stats_4) = run(4);
        assert_eq!(stream_1, stream_4);
        assert!(
            stats_4.proposed > stats_1.proposed,
            "depth 4 proposed {} vs depth 1 proposed {}",
            stats_4.proposed,
            stats_1.proposed
        );
    }

    /// A stop token truncates the stream at itself, inclusive, and ends
    /// the session (Section 4, step 6).
    #[test]
    fn stop_token_truncates_the_stream() {
        let plain = testkit::plain_greedy(PROMPT_A, 24);
        let mut config = greedy_config(2);
        config.eos_tokens = vec![plain[2]];
        let mut session = testkit::build_session(config, 5);
        session.begin(PROMPT_A).expect("begin");
        let stream = session.generate(24, &[]).expect("generate");
        assert_eq!(stream, plain[..3].to_vec());
        assert!(matches!(
            session.round(),
            Err(SpecError::SessionState { .. })
        ));
    }

    /// Temperature sampling runs the full correction walk and is
    /// reproducible under a fixed seed (the seed knob's behavior test).
    #[test]
    fn sampled_generation_is_reproducible_per_seed() {
        let run = |seed: u64| {
            let config = SpecConfig {
                depth: 3,
                temperature: 0.7,
                seed,
                ..SpecConfig::default()
            };
            let mut session = testkit::build_session(config, 5);
            session.begin(PROMPT_B).expect("begin");
            let stream = session.generate(24, &[]).expect("generate");
            (stream, session.stats())
        };
        let (stream_a, stats_a) = run(1234);
        let (stream_b, _) = run(1234);
        assert_eq!(stream_a, stream_b, "same seed, same stream");
        assert_eq!(stream_a.len(), 24);
        assert!(stream_a.iter().all(|&t| (t as usize) < testkit::VOCAB));
        assert!(stats_a.proposed > 0);
        let (stream_c, _) = run(99);
        assert_ne!(stream_a, stream_c, "different seed, different stream");
    }

    /// With the refuse policy, a one-shot verify failure surfaces as
    /// VerifyRefused with cache and pending untouched, and the session
    /// then completes with the identical stream.
    #[test]
    fn verify_refusal_surfaces_and_the_session_recovers() {
        let config = greedy_config(2);
        let (mut session, armed) = testkit::build_sabotage_session(config);
        let pending = session.begin(PROMPT_A).expect("begin");
        armed.set(true);
        let err = session.round().expect_err("round must refuse");
        assert!(matches!(err, SpecError::VerifyRefused { .. }), "got {err}");
        assert_eq!(session.cache_position(), PROMPT_A.len());
        assert_eq!(session.pending(), pending);
        assert_eq!(session.stats().rollbacks, 1);
        let stream = session.generate(16, &[]).expect("generate after refusal");
        assert_eq!(stream, testkit::plain_greedy(PROMPT_A, 16));
    }

    /// With the latch policy, the same failure latches speculation off
    /// (visibly, in the stats) and the round still emits the correct
    /// plain token; the rest of the stream stays identical.
    #[test]
    fn verify_refusal_latches_off_when_configured() {
        let mut config = greedy_config(2);
        config.on_verify_error = OnVerifyError::LatchOff;
        let (mut session, armed) = testkit::build_sabotage_session(config);
        session.begin(PROMPT_A).expect("begin");
        armed.set(true);
        let outcome = session.round().expect("latched round completes plainly");
        let plain = testkit::plain_greedy(PROMPT_A, 16);
        assert!(session.is_latched());
        assert!(session.stats().fallback_latched);
        assert_eq!(outcome.emitted, vec![plain[1]]);
        assert_eq!(outcome.proposed, 2, "the discarded proposal is counted");
        let rest = session.generate(15, &[]).expect("generate after latch");
        assert_eq!(rest, plain[1..16].to_vec());
        assert_eq!(session.stats().proposed, 2, "no proposals after the latch");
    }

    /// Mode off generates plainly through the same session accounting.
    #[test]
    fn mode_off_generates_plainly() {
        let compute = Rc::new(CpuCompute::new());
        let fixture = testkit::build_target(Box::new(SharedCompute(compute.clone())));
        let config = SpecConfig {
            mode: SpecMode::Off,
            ..SpecConfig::default()
        };
        let mut session =
            SpecSession::new(fixture.model, compute, None, config, Arc::new(NullDiag))
                .expect("mode-off session builds");
        session.begin(PROMPT_A).expect("begin");
        let stream = session.generate(16, &[]).expect("generate");
        assert_eq!(stream, testkit::plain_greedy(PROMPT_A, 16));
        assert_eq!(session.stats().proposed, 0);
    }

    fn try_new(config: SpecConfig, draft: Option<DflashSpeculator>) -> Result<SpecSession> {
        let compute = Rc::new(CpuCompute::new());
        let fixture = testkit::build_target(Box::new(SharedCompute(compute.clone())));
        SpecSession::new(fixture.model, compute, draft, config, Arc::new(NullDiag))
    }

    fn fixture_draft(block_size: u32) -> DflashSpeculator {
        let compute = Rc::new(CpuCompute::new());
        let fixture = testkit::build_target(Box::new(SharedCompute(compute)));
        testkit::build_speculator(&fixture, block_size)
    }

    /// Creation validates everything up front and refuses by name
    /// (Section 6.1): one test per refusal in the table this crate can
    /// produce.
    #[test]
    fn creation_refusals_are_named() {
        // DraftNotLoaded.
        let err = try_new(SpecConfig::default(), None)
            .err()
            .expect("no draft");
        assert!(matches!(err, SpecError::DraftNotLoaded), "got {err}");
        // ModeUnavailable.
        let config = SpecConfig {
            mode: SpecMode::Ngram,
            ..SpecConfig::default()
        };
        let err = try_new(config, Some(fixture_draft(5)))
            .err()
            .expect("ngram");
        assert!(
            matches!(err, SpecError::ModeUnavailable { .. }),
            "got {err}"
        );
        // GrammarConflict.
        let config = SpecConfig {
            grammar_active: true,
            ..SpecConfig::default()
        };
        let err = try_new(config, Some(fixture_draft(5)))
            .err()
            .expect("grammar");
        assert!(matches!(err, SpecError::GrammarConflict), "got {err}");
        // DepthOutOfRange: zero, and beyond the block's draft count.
        let config = SpecConfig {
            depth: 0,
            ..SpecConfig::default()
        };
        let err = try_new(config, Some(fixture_draft(5)))
            .err()
            .expect("depth 0");
        assert!(
            matches!(err, SpecError::DepthOutOfRange { .. }),
            "got {err}"
        );
        // Block 5 proposes at most 4, so depth 5 is out of range.
        let config = SpecConfig {
            depth: 5,
            ..SpecConfig::default()
        };
        let err = try_new(config, Some(fixture_draft(5)))
            .err()
            .expect("depth 5 of block 5");
        assert!(
            matches!(err, SpecError::DepthOutOfRange { .. }),
            "got {err}"
        );
        // RoundBoundTooSmall: depth 4 needs a bound of at least 5.
        let config = SpecConfig {
            depth: 4,
            max_round_tokens: 4,
            ..SpecConfig::default()
        };
        let err = try_new(config, Some(fixture_draft(5)))
            .err()
            .expect("bound");
        assert!(
            matches!(err, SpecError::RoundBoundTooSmall { .. }),
            "got {err}"
        );
    }

    /// Call-order discipline: rounds need a begin, and begin runs once.
    #[test]
    fn call_order_is_enforced() {
        let mut session = testkit::build_session(greedy_config(2), 5);
        assert!(matches!(
            session.round(),
            Err(SpecError::SessionState { .. })
        ));
        session.begin(PROMPT_A).expect("begin");
        assert!(matches!(
            session.begin(PROMPT_A),
            Err(SpecError::SessionState { .. })
        ));
    }
}
