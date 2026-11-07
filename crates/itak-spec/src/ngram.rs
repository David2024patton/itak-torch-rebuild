//! N-gram prompt-lookup drafting (SPEC-itak-spec Section 6.2, the
//! `ngram` mode): the no-training self-drafting proposer.
//!
//! The proposer holds no model and runs no forward. It keeps the
//! committed token history and, at propose time, looks up the
//! sequence's own suffix: find the most recent prior occurrence of
//! the last `L` tokens (the longest `L` that occurs at all, down to
//! a configured floor) and propose the tokens that followed that
//! occurrence. Repetitive text (code, logs, templated prose, lists)
//! repeats itself, so the continuation is often exactly what the
//! target is about to emit; the session's verifier accepts the
//! prefix it agrees with and corrects the rest, so a wrong guess
//! costs a round, never a token. On non-repetitive text the lookup
//! finds little or nothing and the proposer stays out of the way.
//!
//! Cost: pure CPU, microseconds at session-scale histories. The
//! search is a backward scan from the most recent end (the first
//! hit is the answer, per the ranking rule below), bounded by a
//! configured window so a no-match round cannot scan an unbounded
//! history. No GPU or model types appear here; the only geometry
//! the proposer knows is the target's vocabulary width, which its
//! one-hot proposal distributions are built at (the same reason
//! `DflashConfig` carries `vocab_size`).
//!
//! Ranking, when several occurrences of a suffix exist: the longest
//! match wins; among occurrences of that match, the most recent
//! wins. Recency is the standard prompt-lookup choice: later text
//! is the better predictor of what the sequence does next, and it
//! makes the scan a first-hit search instead of a census.
//!
//! The sampled arm's distributions are one-hot rows at the proposed
//! tokens. That is not an approximation: a one-hot is a genuine
//! distribution, drawing from it yields the proposed token with
//! probability 1, and the rejection-sampling walk in
//! [`crate::sampler::correct_sampled`] divides the target row by it
//! exactly (accept with probability `p(token)`, reject into the
//! normalized residual `(p - q)+`). The greedy arm carries no
//! distributions, like the DFlash greedy arm.
//!
//! Every operational limit is configuration with a default
//! ([`NgramConfig`]), per the build law; nothing is compiled in.

use crate::error::{Result, SpecError};
use crate::speculator::{BlockProposal, DraftArm, DraftDist, Proposal, Speculator};
use crate::SpecMode;

/// Configuration for [`NgramProposer`]. All limits are configuration
/// with defaults, never compiled constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramConfig {
    /// The longest suffix tried first. The lookup walks down from
    /// here to `min_match_len`; the first length with any prior
    /// occurrence wins.
    pub max_match_len: usize,
    /// The shortest suffix allowed to propose. Shorter matches are
    /// too weak a signal (a 1-token suffix matches almost anywhere)
    /// and are refused by construction, not by ranking.
    pub min_match_len: usize,
    /// Maximum tokens one proposal can carry; the proposer's
    /// `max_depth`, and the bound session depth is validated
    /// against.
    pub max_proposals: usize,
    /// How far back from the sequence end an occurrence's start may
    /// sit and still count, in tokens. Bounds the no-match scan;
    /// `0` searches the whole history.
    pub search_window_tokens: usize,
    /// Committed history retained, in tokens; older tokens are
    /// dropped from the front as commits arrive. `0` retains
    /// everything. Dropped history can no longer be matched
    /// against, which is the honest cost of the bound.
    pub max_history_tokens: usize,
}

impl Default for NgramConfig {
    fn default() -> Self {
        NgramConfig {
            max_match_len: 8,
            min_match_len: 2,
            max_proposals: 16,
            search_window_tokens: 65_536,
            max_history_tokens: 1_048_576,
        }
    }
}

impl NgramConfig {
    /// Refuse a configuration the lookup cannot honor, by name.
    fn validate(&self) -> Result<()> {
        let refuse = |detail: String| SpecError::SessionState { detail };
        if self.max_proposals == 0 {
            return Err(refuse(
                "n-gram config: max_proposals must be at least 1".to_string(),
            ));
        }
        if self.min_match_len == 0 {
            return Err(refuse(
                "n-gram config: min_match_len must be at least 1".to_string(),
            ));
        }
        if self.max_match_len < self.min_match_len {
            return Err(refuse(format!(
                "n-gram config: max_match_len {} is below min_match_len {}",
                self.max_match_len, self.min_match_len
            )));
        }
        Ok(())
    }
}

/// The n-gram prompt-lookup proposer. Implements the crate's
/// [`Speculator`] seam plus an inherent `propose_block` mirroring
/// `DflashSpeculator::propose_block`, so the session dispatches to
/// either draft source through one shape.
#[derive(Debug)]
pub struct NgramProposer {
    config: NgramConfig,
    /// The target vocabulary width the one-hot distributions are
    /// built at, validated against the target at session creation.
    vocab_size: usize,
    /// Committed tokens in order: exactly the sequence the session
    /// has committed, which at propose time is everything except
    /// `pending` (the trait's commit contract).
    history: Vec<u32>,
    /// Backing store for the trait's borrowed view.
    last_tokens: Vec<u32>,
    /// One-hot rows backing the trait's borrowed view.
    last_probs: Vec<Vec<f32>>,
}

impl NgramProposer {
    /// Build a proposer over `config` for a target with
    /// `vocab_size` tokens. Refuses an invalid configuration by
    /// name.
    pub fn new(config: NgramConfig, vocab_size: usize) -> Result<Self> {
        config.validate()?;
        if vocab_size == 0 {
            return Err(SpecError::SessionState {
                detail: "n-gram proposer: vocab_size must be at least 1".to_string(),
            });
        }
        Ok(NgramProposer {
            config,
            vocab_size,
            history: Vec::new(),
            last_tokens: Vec::new(),
            last_probs: Vec::new(),
        })
    }

    /// The configuration this proposer runs under.
    pub fn config(&self) -> &NgramConfig {
        &self.config
    }

    /// The vocabulary width the proposal distributions are built at.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// How many committed tokens the history currently holds. At
    /// every round boundary this equals the target's cache position
    /// minus the one pending token.
    pub fn committed_len(&self) -> usize {
        self.history.len()
    }

    /// The lookup, shared by both proposal entry points. `pending`
    /// is the last emitted token, not yet committed: the searched
    /// sequence is `history + [pending]`, indexed virtually so a
    /// long history is never copied per proposal.
    fn lookup(&self, pending: u32, n: usize) -> Vec<u32> {
        let want = n.min(self.config.max_proposals);
        if want == 0 {
            return Vec::new();
        }
        let seq_len = self.history.len() + 1;
        let at = |i: usize| -> u32 {
            if i < self.history.len() {
                self.history[i]
            } else {
                pending
            }
        };
        // A match needs at least one continuation token after the
        // occurrence, so the suffix itself can span at most
        // seq_len - 1 tokens.
        let longest = self.config.max_match_len.min(seq_len.saturating_sub(1));
        if longest < self.config.min_match_len {
            return Vec::new();
        }
        for len in (self.config.min_match_len..=longest).rev() {
            let suffix_start = seq_len - len;
            // The latest start that still has a continuation token.
            let mut start = seq_len - len - 1;
            // The window bounds how far back a start may sit.
            let window_floor = if self.config.search_window_tokens == 0 {
                0
            } else {
                (seq_len - len - 1).saturating_sub(self.config.search_window_tokens - 1)
            };
            loop {
                if (0..len).all(|k| at(start + k) == at(suffix_start + k)) {
                    let cont_start = start + len;
                    let cont_end = (cont_start + want).min(seq_len);
                    return (cont_start..cont_end).map(at).collect();
                }
                if start == window_floor {
                    break;
                }
                start -= 1;
            }
        }
        Vec::new()
    }

    /// One-hot rows at `tokens`, refusing a token outside the
    /// vocabulary by name rather than indexing past the row.
    fn one_hot_rows(&self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        tokens
            .iter()
            .map(|&t| {
                if (t as usize) >= self.vocab_size {
                    return Err(SpecError::SessionState {
                        detail: format!(
                            "n-gram proposer: token {t} is outside the vocabulary of {} \
                             the distributions are built at",
                            self.vocab_size
                        ),
                    });
                }
                let mut row = vec![0.0f32; self.vocab_size];
                row[t as usize] = 1.0;
                Ok(row)
            })
            .collect()
    }

    /// Propose up to `n` tokens after `pending` by prompt lookup,
    /// the inherent entry the session drives (the DFlash
    /// speculator's `propose_block` shape). The draw source is
    /// accepted for signature parity and never consumed: drawing
    /// from a one-hot distribution yields its token with
    /// probability 1, so the proposal is deterministic on both
    /// arms.
    pub fn propose_block(
        &mut self,
        pending: u32,
        n: usize,
        arm: DraftArm,
        _rng: &mut dyn FnMut() -> f64,
    ) -> Result<Proposal> {
        let tokens = self.lookup(pending, n);
        let dists = match arm {
            DraftArm::Greedy => Vec::new(),
            DraftArm::Sampled { .. } => self.one_hot_rows(&tokens)?,
        };
        self.last_probs = match arm {
            // The greedy arm keeps no distributions on the hot
            // path (the DFlash rule); the trait shim rebuilds the
            // one-hot view from the tokens on demand.
            DraftArm::Greedy => Vec::new(),
            DraftArm::Sampled { .. } => dists.clone(),
        };
        self.last_tokens.clone_from(&tokens);
        Ok(Proposal { tokens, dists })
    }
}

impl Speculator for NgramProposer {
    fn mode(&self) -> SpecMode {
        SpecMode::Ngram
    }

    fn max_depth(&self) -> usize {
        self.config.max_proposals
    }

    fn propose(&mut self, pending: u32, n: usize) -> Result<BlockProposal<'_>> {
        let mut no_draws = || 0.0f64;
        let proposal = self.propose_block(pending, n, DraftArm::Greedy, &mut no_draws)?;
        let _ = proposal;
        if self.last_probs.is_empty() {
            // Greedy proposals carry no stored distributions (see
            // propose_block); the trait's contract is a one-hot
            // view of the proposed tokens, built here on the
            // shim's cold path, exactly the DFlash shim's shape.
            self.last_probs = self.one_hot_rows(&self.last_tokens)?;
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
        self.history.extend_from_slice(tokens);
        if self.config.max_history_tokens > 0 && self.history.len() > self.config.max_history_tokens
        {
            let excess = self.history.len() - self.config.max_history_tokens;
            self.history.drain(..excess);
        }
        Ok(())
    }

    // note_captures keeps the trait default: the proposer conditions
    // on tokens only, so captured layer rows are ignored by design.
}

#[cfg(test)]
mod tests {
    use super::*;

    const VOCAB: usize = 128;

    fn proposer(config: NgramConfig) -> NgramProposer {
        NgramProposer::new(config, VOCAB).expect("valid config")
    }

    fn greedy(prop: &mut NgramProposer, pending: u32, n: usize) -> Vec<u32> {
        let mut no_draws = || 0.0f64;
        prop.propose_block(pending, n, DraftArm::Greedy, &mut no_draws)
            .expect("greedy proposal")
            .tokens
    }

    #[test]
    fn config_validation_refuses_by_name() {
        let bad = NgramConfig {
            max_proposals: 0,
            ..NgramConfig::default()
        };
        assert!(matches!(
            NgramProposer::new(bad, VOCAB),
            Err(SpecError::SessionState { .. })
        ));
        let bad = NgramConfig {
            min_match_len: 0,
            ..NgramConfig::default()
        };
        assert!(NgramProposer::new(bad, VOCAB).is_err());
        let bad = NgramConfig {
            max_match_len: 1,
            min_match_len: 3,
            ..NgramConfig::default()
        };
        assert!(NgramProposer::new(bad, VOCAB).is_err());
        assert!(NgramProposer::new(NgramConfig::default(), 0).is_err());
    }

    #[test]
    fn no_match_returns_empty() {
        // Every token distinct: no suffix of length >= 2 can occur
        // twice, so the proposer stays silent.
        let mut prop = proposer(NgramConfig::default());
        prop.commit(&[10, 20, 30, 40, 50]).expect("commit");
        assert!(greedy(&mut prop, 60, 4).is_empty());
        // A sequence too short to hold a match plus a continuation
        // is silent too.
        let mut prop = proposer(NgramConfig::default());
        prop.commit(&[7]).expect("commit");
        assert!(greedy(&mut prop, 7, 4).is_empty());
        // And n = 0 proposes nothing by construction.
        let mut prop = proposer(NgramConfig::default());
        prop.commit(&[1, 2, 1]).expect("commit");
        assert!(greedy(&mut prop, 2, 0).is_empty());
    }

    #[test]
    fn longest_match_wins_over_shorter() {
        // Sequence (history + pending): [1, 2, 3, 4, 9, 8, 3, 4, 7,
        // 1, 2, 3, 4]. The 4-gram suffix [1, 2, 3, 4] occurs at
        // index 0, followed by 9. The shorter 2-gram suffix [3, 4]
        // occurs more recently (index 6), followed by 7. The
        // longest match wins: the proposal follows the 4-gram's
        // continuation, not the fresher 2-gram's.
        let mut prop = proposer(NgramConfig::default());
        prop.commit(&[1, 2, 3, 4, 9, 8, 3, 4, 7, 1, 2, 3])
            .expect("commit");
        assert_eq!(greedy(&mut prop, 4, 4), vec![9, 8, 3, 4]);
    }

    #[test]
    fn most_recent_occurrence_wins_among_equal_matches() {
        // The 2-gram [1, 2] occurs followed by 5 (older, index 0)
        // and by 6 (most recent, index 4); no longer suffix
        // matches, so recency decides and 6 leads the proposal.
        let mut prop = proposer(NgramConfig::default());
        prop.commit(&[1, 2, 5, 8, 1, 2, 6, 7, 1]).expect("commit");
        assert_eq!(greedy(&mut prop, 2, 4), vec![6, 7, 1, 2]);
    }

    #[test]
    fn continuation_is_capped_by_depth_and_by_what_follows() {
        let mut prop = proposer(NgramConfig::default());
        prop.commit(&[1, 2, 3, 4, 5, 6, 7, 8, 1]).expect("commit");
        // Depth caps the proposal.
        assert_eq!(greedy(&mut prop, 2, 3), vec![3, 4, 5]);
        // A deeper ask returns the whole recorded continuation,
        // which runs to the end of the sequence (the pending
        // token's own position included: it is a recorded token).
        assert_eq!(greedy(&mut prop, 2, 16), vec![3, 4, 5, 6, 7, 8, 1, 2]);
        // max_proposals caps below the ask.
        let mut capped = proposer(NgramConfig {
            max_proposals: 2,
            ..NgramConfig::default()
        });
        capped.commit(&[1, 2, 3, 4, 5, 6, 1]).expect("commit");
        assert_eq!(greedy(&mut capped, 2, 16), vec![3, 4]);
        assert_eq!(capped.max_depth(), 2);
    }

    #[test]
    fn min_match_len_is_a_floor_not_a_ranking() {
        // Only a 1-token repeat exists ([2] occurs, followed by
        // different tokens); the default floor of 2 refuses it.
        let mut prop = proposer(NgramConfig::default());
        prop.commit(&[1, 2, 3, 2, 4, 2]).expect("commit");
        assert!(greedy(&mut prop, 5, 4).is_empty());
        // Lowering the floor to 1 lets the most recent [2] speak:
        // followed by 4.
        let cfg = NgramConfig {
            min_match_len: 1,
            ..NgramConfig::default()
        };
        let mut prop = proposer(cfg);
        prop.commit(&[1, 2, 3, 2]).expect("commit");
        assert_eq!(greedy(&mut prop, 4, 1), Vec::<u32>::new());
        prop.commit(&[4]).expect("commit");
        // Sequence is now [1, 2, 3, 2, 4] + pending 2: the most
        // recent prior [2] was followed by 4.
        assert_eq!(greedy(&mut prop, 2, 1), vec![4]);
    }

    #[test]
    fn search_window_bounds_how_far_back_a_match_may_sit() {
        // The only occurrence of [1, 2] sits at the very start; a
        // window of 4 tokens back from the end cannot see it.
        let cfg = NgramConfig {
            search_window_tokens: 4,
            ..NgramConfig::default()
        };
        let mut prop = proposer(cfg);
        prop.commit(&[1, 2, 3, 10, 11, 12, 13, 14, 15, 1])
            .expect("commit");
        assert!(greedy(&mut prop, 2, 4).is_empty());
        // The same history with the window open finds it.
        let mut prop = proposer(NgramConfig::default());
        prop.commit(&[1, 2, 3, 10, 11, 12, 13, 14, 15, 1])
            .expect("commit");
        assert_eq!(greedy(&mut prop, 2, 4), vec![3, 10, 11, 12]);
    }

    #[test]
    fn commit_appends_and_history_bound_trims_oldest() {
        let cfg = NgramConfig {
            max_history_tokens: 5,
            ..NgramConfig::default()
        };
        let mut prop = proposer(cfg);
        prop.commit(&[1, 2, 3]).expect("commit");
        assert_eq!(prop.committed_len(), 3);
        prop.commit(&[4, 5, 6, 7]).expect("commit");
        assert_eq!(prop.committed_len(), 5);
        // The retained history is the newest five: [3, 4, 5, 6, 7].
        // A match against the trimmed-away [1, 2] prefix is gone.
        assert!(greedy(&mut prop, 2, 4).is_empty());
    }

    #[test]
    fn note_captures_is_a_no_op() {
        let mut prop = proposer(NgramConfig::default());
        prop.commit(&[1, 2, 3]).expect("commit");
        prop.note_captures(0, &[]).expect("captures ignored");
        assert_eq!(prop.committed_len(), 3);
    }

    #[test]
    fn trait_view_agrees_with_propose_block_and_dists_are_one_hot() {
        let mut prop = proposer(NgramConfig::default());
        prop.commit(&[4, 5, 6, 4, 5]).expect("commit");
        let owned = greedy(&mut prop, 6, 3);
        let view = prop.propose(6, 3).expect("trait propose");
        assert_eq!(view.tokens, owned.as_slice());
        assert_eq!(view.dists.len(), owned.len());
        for (dist, &token) in view.dists.iter().zip(owned.iter()) {
            assert_eq!(dist.probs.len(), VOCAB);
            assert_eq!(dist.probs[token as usize], 1.0);
            assert_eq!(dist.probs.iter().sum::<f32>(), 1.0);
        }
        assert_eq!(prop.mode(), SpecMode::Ngram);
        assert_eq!(prop.max_depth(), NgramConfig::default().max_proposals);
    }

    #[test]
    fn sampled_arm_carries_one_hot_dists_and_the_same_tokens() {
        let mut prop = proposer(NgramConfig::default());
        prop.commit(&[4, 5, 6, 4, 5]).expect("commit");
        let mut no_draws = || 0.0f64;
        let sampled = prop
            .propose_block(6, 3, DraftArm::Sampled { temperature: 0.7 }, &mut no_draws)
            .expect("sampled proposal");
        assert_eq!(sampled.tokens, vec![4, 5, 6]);
        assert_eq!(sampled.dists.len(), 3);
        for (row, &token) in sampled.dists.iter().zip(sampled.tokens.iter()) {
            assert_eq!(row[token as usize], 1.0);
            assert_eq!(row.iter().sum::<f32>(), 1.0);
        }
        // Greedy carries no distributions (the walk never reads
        // them on that arm).
        let greedy_prop = prop
            .propose_block(6, 3, DraftArm::Greedy, &mut no_draws)
            .expect("greedy proposal");
        assert_eq!(greedy_prop.tokens, sampled.tokens);
        assert!(greedy_prop.dists.is_empty());
    }

    #[test]
    fn out_of_vocab_proposal_refuses_on_the_sampled_arm_by_name() {
        // A token outside the distribution width can only arrive
        // from a caller lying about the vocabulary; the proposer
        // refuses by name instead of indexing past the row.
        let mut prop = proposer(NgramConfig::default());
        prop.commit(&[1, 2, 900, 1]).expect("commit");
        let mut no_draws = || 0.0f64;
        let err = prop
            .propose_block(2, 2, DraftArm::Sampled { temperature: 1.0 }, &mut no_draws)
            .expect_err("out-of-vocab token must refuse");
        assert!(matches!(err, SpecError::SessionState { .. }), "{err}");
    }

    /// Lockstep acceptance simulation against a recorded stream:
    /// the session's greedy rule replayed at the proposer level.
    /// `history` starts as the stream's prompt; each round proposes
    /// after `pending`, the oracle stream accepts the matching
    /// prefix, and the round commits `[pending, accepted...]` and
    /// emits one further oracle token (the correction), exactly the
    /// session's accounting. Returns (proposed, accepted).
    fn simulate(stream: &[u32], prompt_len: usize, depth: usize) -> (usize, usize) {
        let mut prop = proposer(NgramConfig::default());
        prop.commit(&stream[..prompt_len]).expect("commit prompt");
        let mut pending = stream[prompt_len];
        let mut pos = prompt_len + 1;
        let (mut proposed, mut accepted_total) = (0usize, 0usize);
        while pos < stream.len() {
            let tokens = greedy(&mut prop, pending, depth);
            proposed += tokens.len();
            let mut accepted = 0;
            while accepted < tokens.len()
                && pos + accepted < stream.len()
                && tokens[accepted] == stream[pos + accepted]
            {
                accepted += 1;
            }
            accepted_total += accepted;
            // Commit [pending, accepted drafts...]; the correction
            // token at pos + accepted becomes the next pending.
            let mut committed = vec![pending];
            committed.extend_from_slice(&stream[pos..pos + accepted]);
            prop.commit(&committed).expect("commit round");
            if pos + accepted >= stream.len() {
                break;
            }
            pending = stream[pos + accepted];
            pos += accepted + 1;
        }
        (proposed, accepted_total)
    }

    // ----- Session-level tests: the proposer inside the loop -----

    use std::rc::Rc;
    use std::sync::Arc;

    use cpu_compute::CpuCompute;

    use crate::diag::NullDiag;
    use crate::driver::CpuTarget;
    use crate::session::SpecSession;
    use crate::speculator::SharedCompute;
    use crate::testkit;
    use crate::SpecConfig;

    fn fixture_target() -> CpuTarget {
        let compute = Rc::new(CpuCompute::new());
        let fixture = testkit::build_target(Box::new(SharedCompute(compute.clone())));
        CpuTarget::new(fixture.model, compute)
    }

    fn repetitive_prompt() -> Vec<u32> {
        let mut prompt = Vec::new();
        for _ in 0..6 {
            prompt.extend_from_slice(&[10, 20, 30, 40]);
        }
        prompt
    }

    fn prose_prompt() -> Vec<u32> {
        vec![
            3, 91, 7, 55, 120, 18, 44, 2, 88, 61, 30, 77, 12, 99, 25, 40, 66, 8, 110, 51, 21, 84,
            36, 70,
        ]
    }

    fn ngram_config(depth: usize) -> SpecConfig {
        SpecConfig {
            mode: SpecMode::Ngram,
            depth,
            ..SpecConfig::default()
        }
    }

    #[test]
    fn session_creation_refusals_are_named() {
        // No proposer attached under n-gram mode.
        let err =
            SpecSession::new_ngram(fixture_target(), None, ngram_config(4), Arc::new(NullDiag))
                .err()
                .expect("missing proposer must refuse");
        assert!(matches!(err, SpecError::DraftNotLoaded), "{err}");

        // A proposer attached under the DFlash mode: the pairing
        // is refused by name, never silently reinterpreted.
        let err = SpecSession::new_ngram(
            fixture_target(),
            Some(proposer(NgramConfig::default())),
            SpecConfig::default(),
            Arc::new(NullDiag),
        )
        .err()
        .expect("mode/proposer mismatch must refuse");
        assert!(matches!(err, SpecError::SessionState { .. }), "{err}");

        // Vocabulary disagreement with the target.
        let wrong_vocab = NgramProposer::new(NgramConfig::default(), 64).expect("proposer");
        let err = SpecSession::new_ngram(
            fixture_target(),
            Some(wrong_vocab),
            ngram_config(4),
            Arc::new(NullDiag),
        )
        .err()
        .expect("vocab mismatch must refuse");
        assert!(
            matches!(
                err,
                SpecError::DraftTargetMismatch {
                    field: "vocab_size",
                    expected: 128,
                    actual: 64
                }
            ),
            "{err}"
        );

        // Depth beyond the proposer's maximum proposals.
        let too_deep = SpecConfig {
            depth: NgramConfig::default().max_proposals + 1,
            ..ngram_config(4)
        };
        let err = SpecSession::new_ngram(
            fixture_target(),
            Some(proposer(NgramConfig::default())),
            too_deep,
            Arc::new(NullDiag),
        )
        .err()
        .expect("depth beyond the proposer must refuse");
        assert!(matches!(err, SpecError::DepthOutOfRange { .. }), "{err}");

        // Depth that breaks the round bound.
        let bound = SpecConfig {
            max_round_tokens: 4,
            ..ngram_config(4)
        };
        let err = SpecSession::new_ngram(
            fixture_target(),
            Some(proposer(NgramConfig::default())),
            bound,
            Arc::new(NullDiag),
        )
        .err()
        .expect("round bound must refuse");
        assert!(
            matches!(
                err,
                SpecError::RoundBoundTooSmall {
                    depth: 4,
                    max_round_tokens: 4
                }
            ),
            "{err}"
        );
    }

    /// The verifier property, n-gram arm: at temperature 0 the
    /// speculative stream equals plain greedy token for token,
    /// across prompt shapes, including the repetitive shape where
    /// the proposer speaks often and the prose shape where it
    /// mostly stays silent.
    #[test]
    fn session_greedy_matches_plain_across_prompt_shapes() {
        let prompts: Vec<Vec<u32>> = vec![repetitive_prompt(), prose_prompt(), vec![7]];
        for prompt in &prompts {
            for depth in [1usize, 4] {
                let mut session =
                    testkit::build_ngram_session(ngram_config(depth), NgramConfig::default());
                session.begin(prompt).expect("begin");
                let stream = session.generate(48, &[]).expect("generate");
                assert_eq!(
                    stream,
                    testkit::plain_greedy(prompt, 48),
                    "n-gram identity, prompt {prompt:?} depth {depth}"
                );
                let stats = session.stats();
                assert!(stats.accepted <= stats.proposed);
            }
        }
    }

    #[test]
    fn session_sampled_is_reproducible_per_seed() {
        let run = || {
            let config = SpecConfig {
                mode: SpecMode::Ngram,
                depth: 3,
                temperature: 0.7,
                seed: 42,
                ..SpecConfig::default()
            };
            let mut session = testkit::build_ngram_session(config, NgramConfig::default());
            session.begin(&[10, 20, 30, 40, 10, 20]).expect("begin");
            session.generate(32, &[]).expect("generate")
        };
        let stream = run();
        assert_eq!(stream.len(), 32);
        assert!(stream.iter().all(|&t| (t as usize) < VOCAB));
        assert_eq!(stream, run(), "same seed, same sampled stream");
    }

    /// The fixture acceptance measurement (the harness whose
    /// numbers `bench/ngram-spec-2026-11-07.md` reports): greedy
    /// n-gram sessions over the synthetic fixture target, on a
    /// repetitive prompt and a prose prompt, at three depths.
    /// Identity is asserted hard; the rates are printed for the
    /// record (run with `-- --nocapture` to read them).
    #[test]
    fn fixture_acceptance_measurement() {
        for (name, prompt) in [
            ("repetitive", repetitive_prompt()),
            ("prose", prose_prompt()),
        ] {
            for depth in [1usize, 4, 6] {
                let mut session =
                    testkit::build_ngram_session(ngram_config(depth), NgramConfig::default());
                session.begin(&prompt).expect("begin");
                let stream = session.generate(96, &[]).expect("generate");
                assert_eq!(stream, testkit::plain_greedy(&prompt, 96));
                let stats = session.finish();
                eprintln!(
                    "FIXTURE_ACCEPTANCE fixture={name} depth={depth} rounds={} proposed={} \
                     accepted={} acceptance_rate={:.4} tokens_emitted={}",
                    stats.rounds,
                    stats.proposed,
                    stats.accepted,
                    stats.acceptance_rate(),
                    stats.tokens_emitted
                );
            }
        }
    }

    #[test]
    fn lockstep_simulation_accepts_repetition_and_stays_quiet_on_novelty() {
        // A stream built from a repeating sentence accepts almost
        // everything after the first cycle.
        let sentence: Vec<u32> = (0..12).map(|i| 10 + i).collect();
        let mut repetitive = sentence.clone();
        for _ in 0..7 {
            repetitive.extend_from_slice(&sentence);
        }
        let (proposed, accepted) = simulate(&repetitive, 12, 6);
        assert!(proposed > 0, "repetitive stream must propose");
        assert!(
            accepted as f64 / proposed as f64 > 0.9,
            "repetitive acceptance {accepted}/{proposed}"
        );
        // A stream of distinct tokens never repeats a 2-gram, so
        // nothing is proposed and nothing can be accepted wrongly.
        let novel: Vec<u32> = (0..96).collect();
        let (proposed, accepted) = simulate(&novel, 16, 6);
        assert_eq!((proposed, accepted), (0, 0));
    }
}
