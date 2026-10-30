//! Sampling: turning a logit slice into one token id (TODO T-2.5,
//! PLAN S5).
//!
//! Every knob is a field of [`SamplingConfig`] with a default, read at
//! call time; no parameter is compiled into the math. The pipeline, in
//! the order llama.cpp applies its sampler chain:
//!
//! 1. Penalties against the token history: repetition (multiplicative,
//!    the CTRL rule: divide a positive logit by the penalty, multiply a
//!    negative one), then frequency and presence (additive, OpenAI
//!    style: frequency scales with the token's count in the history,
//!    presence applies once per distinct token).
//! 2. Greedy shortcut: at temperature 0 the draw is the argmax of the
//!    penalized logits and the RNG is never consulted, which is what
//!    makes temperature 0 a reproducible reference path (the golden
//!    harness, T-2.9, diffs against it).
//! 3. Otherwise: divide by temperature, keep the top-k logits, softmax
//!    in `f64`, cut the nucleus at top-p (the smallest highest-first
//!    prefix whose probability mass reaches top-p), renormalize, and
//!    draw one uniform from the seeded RNG against the cumulative
//!    distribution.
//!
//! Ties and ordering are canonical (logit descending, token id
//! ascending) so the same logits, config, and seed always produce the
//! same token, on any platform. The RNG is SplitMix64: small, std-only
//! (this workspace carries no external dependencies), and fully
//! determined by the seed in the config.
//!
//! Failure is loud and named, never a substituted token: an empty
//! logit slice is [`GlimmerError::EmptyLogits`], a slice with no
//! finite-mass token is [`GlimmerError::NoViableToken`], and a config
//! outside its meaningful range is [`GlimmerError::InvalidSampling`]
//! naming the field. NaN logits are excluded from the distribution
//! (a NaN is a kernel defect upstream; sampling treats it as zero
//! probability rather than letting it poison the softmax).

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

use crate::diag::{Diag, NullDiag};
use crate::error::{GlimmerError, Result};

/// Every sampling parameter, as config. Build one with
/// [`SamplingConfig::default`] and override the fields a caller cares
/// about, or fill the struct literally; [`SamplingConfig::validate`]
/// (run by [`Sampler::sample`] on every call) refuses values outside
/// their meaningful ranges.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingConfig {
    /// Temperature: logits are divided by this before the softmax.
    /// Exactly 0 selects greedy argmax and the RNG is not consulted.
    /// Must be finite and non-negative.
    pub temperature: f32,
    /// Top-k: keep only the `top_k` highest logits before the softmax.
    /// 0 disables the cut (the whole vocabulary competes).
    pub top_k: u32,
    /// Top-p (nucleus): keep the smallest highest-probability prefix
    /// whose mass reaches `top_p`. 1.0 disables the cut. Must lie in
    /// (0, 1].
    pub top_p: f32,
    /// Repetition penalty (CTRL rule): a token already in the history
    /// has its positive logit divided by this, its negative logit
    /// multiplied by it. 1.0 disables. Must be finite and positive.
    pub repetition_penalty: f32,
    /// Frequency penalty: subtract `frequency_penalty * count` from
    /// the logit of each token appearing `count` times in the history.
    /// 0.0 disables. Must be finite.
    pub frequency_penalty: f32,
    /// Presence penalty: subtract `presence_penalty` once from the
    /// logit of each distinct token in the history. 0.0 disables.
    /// Must be finite.
    pub presence_penalty: f32,
    /// Seed for the draw. The same seed, config, logits, and history
    /// reproduce the same token sequence exactly.
    pub seed: u64,
}

impl Default for SamplingConfig {
    /// Neutral defaults: temperature 1.0 (the distribution as the
    /// model produced it), no top-k or top-p cut, no penalties, seed 0.
    fn default() -> Self {
        SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 0,
        }
    }
}

impl SamplingConfig {
    /// Refuse parameters outside their meaningful ranges, naming the
    /// field. Returns `Ok(())` for a config [`Sampler::sample`] will
    /// run exactly as written.
    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(invalid(
                "temperature",
                "must be finite and non-negative (0 is greedy)",
            ));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(invalid("top_p", "must lie in (0, 1]"));
        }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            return Err(invalid(
                "repetition_penalty",
                "must be finite and positive (1.0 disables it)",
            ));
        }
        if !self.frequency_penalty.is_finite() {
            return Err(invalid("frequency_penalty", "must be finite"));
        }
        if !self.presence_penalty.is_finite() {
            return Err(invalid("presence_penalty", "must be finite"));
        }
        Ok(())
    }

    /// Whether this config takes the greedy path (temperature 0).
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }

    /// Whether any penalty differs from its disabled value.
    fn penalties_active(&self) -> bool {
        self.repetition_penalty != 1.0
            || self.frequency_penalty != 0.0
            || self.presence_penalty != 0.0
    }
}

fn invalid(field: &'static str, detail: &str) -> GlimmerError {
    GlimmerError::InvalidSampling {
        field,
        detail: detail.to_string(),
    }
}

/// A seeded, deterministic random source: SplitMix64. The whole
/// stream is a function of the seed, which is what makes a sampling
/// run replayable from its config alone.
#[derive(Clone, Debug)]
pub struct SeededRng {
    state: u64,
}

impl SeededRng {
    /// Start a stream from `seed`.
    pub fn new(seed: u64) -> Self {
        SeededRng { state: seed }
    }

    /// The next 64-bit value in the stream.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// The next draw as a double in [0, 1), from the top 53 bits.
    pub fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / (1u64 << 53) as f64;
        (self.next_u64() >> 11) as f64 * SCALE
    }
}

/// One token still in contention after the cuts, with its final
/// probability. [`distribution`] returns these in canonical order
/// (probability descending, token id ascending on ties) with
/// probabilities summing to 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    /// The token id (an index into the logit slice).
    pub token: u32,
    /// The logit after penalties and temperature scaling.
    pub logit: f32,
    /// The token's probability after top-k, top-p, and renormalization.
    pub prob: f64,
}

/// Apply the config's penalties to `logits` in place, against the
/// token `history` (everything generated so far, prompt included if
/// the caller wants it counted; the choice is the caller's and is the
/// same choice llama.cpp leaves to its penalty window). Tokens in the
/// history beyond the logit slice are ignored: they cannot be drawn
/// anyway. With all penalties at their disabled values this is a
/// no-op.
pub fn apply_penalties(logits: &mut [f32], history: &[u32], config: &SamplingConfig) {
    if !config.penalties_active() || history.is_empty() {
        return;
    }
    let mut counts: HashMap<u32, u32> = HashMap::new();
    for &token in history {
        *counts.entry(token).or_insert(0) += 1;
    }
    for (token, count) in counts {
        let Some(slot) = logits.get_mut(token as usize) else {
            continue;
        };
        let mut logit = *slot;
        if config.repetition_penalty != 1.0 {
            // CTRL rule: shrink positive logits toward zero, push
            // negative ones further down, both by the same factor.
            if logit > 0.0 {
                logit /= config.repetition_penalty;
            } else {
                logit *= config.repetition_penalty;
            }
        }
        logit -= config.frequency_penalty * count as f32;
        logit -= config.presence_penalty;
        *slot = logit;
    }
}

/// The greedy draw: the token id of the highest logit. NaN logits
/// never win. An empty slice is [`GlimmerError::EmptyLogits`]; a
/// slice whose every logit is NaN or negative infinity is
/// [`GlimmerError::NoViableToken`].
pub fn greedy_token(logits: &[f32]) -> Result<u32> {
    if logits.is_empty() {
        return Err(GlimmerError::EmptyLogits);
    }
    let mut best: Option<(usize, f32)> = None;
    for (index, &logit) in logits.iter().enumerate() {
        if logit.is_nan() {
            continue;
        }
        match best {
            Some((_, best_logit)) if logit <= best_logit => {}
            _ => best = Some((index, logit)),
        }
    }
    match best {
        Some((index, logit)) if logit > f32::NEG_INFINITY => Ok(index as u32),
        _ => Err(GlimmerError::NoViableToken),
    }
}

/// The distribution the draw samples from: temperature scaling,
/// top-k, softmax, top-p nucleus, renormalization. Penalties are NOT
/// applied here (they need the history; [`Sampler::sample`] applies
/// them first). At temperature 0 the result is a single candidate
/// with probability 1: the greedy token, so callers can inspect the
/// greedy path through the same shape.
///
/// Logit slices in this engine are vocabulary-sized and the vocab is
/// `u32`-bounded by construction, so token ids are indices cast to
/// `u32`.
pub fn distribution(logits: &[f32], config: &SamplingConfig) -> Result<Vec<Candidate>> {
    config.validate()?;
    if logits.is_empty() {
        return Err(GlimmerError::EmptyLogits);
    }
    if config.is_greedy() {
        let token = greedy_token(logits)?;
        return Ok(vec![Candidate {
            token,
            logit: logits[token as usize],
            prob: 1.0,
        }]);
    }

    // Temperature scaling; NaN logits are excluded outright, negative
    // infinity stays a hard zero through the softmax.
    let temperature = f64::from(config.temperature);
    let mut scaled: Vec<(u32, f64)> = logits
        .iter()
        .enumerate()
        .filter(|(_, logit)| !logit.is_nan())
        .map(|(index, logit)| (index as u32, f64::from(*logit) / temperature))
        .collect();
    if scaled.iter().all(|(_, z)| *z == f64::NEG_INFINITY) {
        return Err(GlimmerError::NoViableToken);
    }

    // Canonical order: logit descending, token id ascending on ties.
    scaled.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    // Top-k cut.
    if config.top_k > 0 && (config.top_k as usize) < scaled.len() {
        scaled.truncate(config.top_k as usize);
    }

    // Probabilities. Positive infinity is the limiting case of the
    // softmax: the infinite logits share the mass equally and every
    // finite logit gets zero.
    let probs: Vec<f64> = if scaled[0].1 == f64::INFINITY {
        let infinite = scaled
            .iter()
            .take_while(|(_, z)| *z == f64::INFINITY)
            .count();
        scaled
            .iter()
            .enumerate()
            .map(|(i, _)| {
                if i < infinite {
                    1.0 / infinite as f64
                } else {
                    0.0
                }
            })
            .collect()
    } else {
        let max = scaled[0].1;
        let weights: Vec<f64> = scaled.iter().map(|(_, z)| (z - max).exp()).collect();
        let sum: f64 = weights.iter().sum();
        weights.iter().map(|w| w / sum).collect()
    };

    // Zero-probability tail entries (finite logits next to an
    // infinite one) can never be drawn; drop them before the nucleus
    // cut so they cannot consume a top-k slot in the result.
    let mut kept: Vec<(u32, f64, f64)> = Vec::with_capacity(scaled.len());
    for (entry, prob) in scaled.iter().zip(probs.iter()) {
        if *prob > 0.0 {
            kept.push((entry.0, entry.1, *prob));
        }
    }
    // Top-p nucleus: the smallest prefix whose mass reaches top_p.
    // Always keeps the single most probable token.
    let mut cumulative = 0.0f64;
    let mut nucleus_len = kept.len();
    for (i, (_, _, prob)) in kept.iter().enumerate() {
        cumulative += prob;
        if cumulative >= f64::from(config.top_p) {
            nucleus_len = i + 1;
            break;
        }
    }
    kept.truncate(nucleus_len);

    // Renormalize the nucleus to sum to 1.
    let kept_mass: f64 = kept.iter().map(|(_, _, prob)| prob).sum();
    Ok(kept
        .iter()
        .map(|(token, z, prob)| Candidate {
            token: *token,
            logit: *z as f32,
            prob: prob / kept_mass,
        })
        .collect())
}

/// A sampler: a validated-at-draw-time [`SamplingConfig`], the seeded
/// RNG its draws come from, and the crate's diagnostic channel. One
/// sampler per generation stream; its RNG state is the stream's
/// position.
pub struct Sampler {
    config: SamplingConfig,
    rng: SeededRng,
    diag: Arc<dyn Diag>,
}

impl Sampler {
    /// A sampler with a silent diagnostic sink.
    pub fn new(config: SamplingConfig) -> Self {
        Sampler {
            rng: SeededRng::new(config.seed),
            config,
            diag: Arc::new(NullDiag),
        }
    }

    /// A sampler with a diagnostic sink attached. Draws and refusals
    /// emit through it.
    pub fn with_diag(config: SamplingConfig, diag: Arc<dyn Diag>) -> Self {
        Sampler {
            rng: SeededRng::new(config.seed),
            config,
            diag,
        }
    }

    /// The config this sampler draws under.
    pub fn config(&self) -> &SamplingConfig {
        &self.config
    }

    /// Draw one token from `logits`, with penalties applied against
    /// `history`. At temperature 0 this is the argmax of the penalized
    /// logits and the RNG state does not move. Failures are the named
    /// errors of [`distribution`] and [`greedy_token`], and every one
    /// emits a `sampling.refused` diagnostic before returning.
    pub fn sample(&mut self, logits: &[f32], history: &[u32]) -> Result<u32> {
        match self.draw(logits, history) {
            Ok(token) => Ok(token),
            Err(err) => {
                self.diag.event("sampling.refused", &err.to_string());
                Err(err)
            }
        }
    }

    fn draw(&mut self, logits: &[f32], history: &[u32]) -> Result<u32> {
        self.config.validate()?;
        if logits.is_empty() {
            return Err(GlimmerError::EmptyLogits);
        }
        self.diag.event(
            "sampling.begin",
            &format!(
                "logits={} history={} temperature={} top_k={} top_p={}",
                logits.len(),
                history.len(),
                self.config.temperature,
                self.config.top_k,
                self.config.top_p
            ),
        );

        let mut work = logits.to_vec();
        apply_penalties(&mut work, history, &self.config);

        if self.config.is_greedy() {
            let token = greedy_token(&work)?;
            self.diag
                .event("sampling.greedy", &format!("token={token}"));
            return Ok(token);
        }

        let candidates = distribution(&work, &self.config)?;
        let draw = self.rng.next_f64();
        let mut cumulative = 0.0f64;
        let mut chosen = candidates[candidates.len() - 1].token;
        for candidate in &candidates {
            cumulative += candidate.prob;
            if draw < cumulative {
                chosen = candidate.token;
                break;
            }
        }
        self.diag.event(
            "sampling.draw",
            &format!("token={chosen} candidates={}", candidates.len()),
        );
        Ok(chosen)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Probabilities derive from f32 logits, so agreement is at f32
    /// input precision: 1e-6, far tighter than the gaps between the
    /// candidate answers these tests distinguish.
    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn rng_stream_is_a_function_of_the_seed() {
        let mut a = SeededRng::new(42);
        let mut b = SeededRng::new(42);
        let mut c = SeededRng::new(43);
        let stream_a: Vec<u64> = (0..16).map(|_| a.next_u64()).collect();
        let stream_b: Vec<u64> = (0..16).map(|_| b.next_u64()).collect();
        let stream_c: Vec<u64> = (0..16).map(|_| c.next_u64()).collect();
        assert_eq!(stream_a, stream_b);
        assert_ne!(stream_a, stream_c);
        let mut d = SeededRng::new(7);
        for _ in 0..1024 {
            let v = d.next_f64();
            assert!((0.0..1.0).contains(&v), "draw {v} outside [0, 1)");
        }
    }

    /// TODO T-2.5 proof, first half: at temperature 0 the draw is the
    /// greedy argmax and the RNG state is exactly where it started,
    /// i.e. the RNG was never consulted.
    #[test]
    fn temp_zero_is_greedy_and_never_touches_the_rng() {
        let config = SamplingConfig {
            temperature: 0.0,
            seed: 1234,
            ..SamplingConfig::default()
        };
        let mut sampler = Sampler::new(config);
        let logits = [1.0f32, 4.5, -2.0, 4.25, 0.0];
        for _ in 0..8 {
            let token = sampler.sample(&logits, &[]).expect("greedy draw");
            assert_eq!(token, 1, "argmax of the logits is token 1");
        }
        assert_eq!(
            sampler.rng.state, 1234,
            "RNG state must not advance on the greedy path"
        );

        // Cross-check through behavior alone: a sampler that first
        // draws greedily and one that does not must produce the same
        // first stochastic token once both run the same warm config.
        let warm = SamplingConfig {
            temperature: 1.0,
            seed: 99,
            ..SamplingConfig::default()
        };
        let mut direct = Sampler::new(warm);
        let expected = direct.sample(&logits, &[]).expect("stochastic draw");
        let mut via_greedy = Sampler::new(SamplingConfig {
            temperature: 0.0,
            seed: 99,
            ..SamplingConfig::default()
        });
        let _ = via_greedy.sample(&logits, &[]).expect("greedy draw");
        via_greedy.config.temperature = 1.0;
        let actual = via_greedy.sample(&logits, &[]).expect("stochastic draw");
        assert_eq!(actual, expected);
    }

    /// The same seed and settings replay the same sequence, token for
    /// token, including the history feedback real generation has.
    #[test]
    fn same_seed_replays_the_same_sequence() {
        let config = SamplingConfig {
            temperature: 0.9,
            top_k: 8,
            top_p: 0.95,
            seed: 7,
            ..SamplingConfig::default()
        };
        let frames: Vec<Vec<f32>> = (0..12)
            .map(|step| {
                (0..16)
                    .map(|i| ((step * 31 + i * 17) % 23) as f32 * 0.35 - 3.0)
                    .collect()
            })
            .collect();
        let run = |config: SamplingConfig| -> Vec<u32> {
            let mut sampler = Sampler::new(config);
            let mut history: Vec<u32> = Vec::new();
            let mut out = Vec::new();
            for frame in &frames {
                let token = sampler.sample(frame, &history).expect("draw");
                history.push(token);
                out.push(token);
            }
            out
        };
        assert_eq!(run(config), run(config));
    }

    /// TODO T-2.5 proof, second half: overriding the config changes
    /// behavior, proven three ways on fixed logits.
    #[test]
    fn config_overrides_change_behavior() {
        let logits = [2.0f32, 1.0, 0.5];

        // Repetition penalty flips the greedy winner: token 0 leads
        // 2.0 to 1.0 until its history count divides it to 2/3.
        let base = SamplingConfig {
            temperature: 0.0,
            ..SamplingConfig::default()
        };
        let mut sampler = Sampler::new(base);
        assert_eq!(sampler.sample(&logits, &[0]).expect("draw"), 0);
        let penalized = SamplingConfig {
            temperature: 0.0,
            repetition_penalty: 3.0,
            ..SamplingConfig::default()
        };
        let mut sampler = Sampler::new(penalized);
        assert_eq!(sampler.sample(&logits, &[0]).expect("draw"), 1);

        // top_k = 1 collapses any temperature to the argmax.
        let top_one = SamplingConfig {
            temperature: 2.5,
            top_k: 1,
            seed: 5,
            ..SamplingConfig::default()
        };
        let mut sampler = Sampler::new(top_one);
        for _ in 0..16 {
            assert_eq!(sampler.sample(&logits, &[]).expect("draw"), 0);
        }

        // A tiny nucleus leaves only the argmax in contention.
        let nucleus = SamplingConfig {
            temperature: 1.0,
            top_p: 0.01,
            seed: 5,
            ..SamplingConfig::default()
        };
        let mut sampler = Sampler::new(nucleus);
        for _ in 0..16 {
            assert_eq!(sampler.sample(&logits, &[]).expect("draw"), 0);
        }
    }

    /// Hand-computed nucleus: logits ln(4), ln(3), ln(2), ln(1) give
    /// probabilities 0.4, 0.3, 0.2, 0.1 exactly. top_p 0.65 keeps the
    /// first two (0.4 alone is short, 0.4 + 0.3 = 0.7 reaches it),
    /// renormalized to 4/7 and 3/7.
    #[test]
    fn top_p_nucleus_matches_hand_computed_distribution() {
        let logits = [
            std::f32::consts::LN_2 * 2.0, // ln 4
            3.0f32.ln(),
            std::f32::consts::LN_2, // ln 2
            0.0,                    // ln 1
        ];
        let config = SamplingConfig {
            temperature: 1.0,
            top_p: 0.65,
            ..SamplingConfig::default()
        };
        let candidates = distribution(&logits, &config).expect("distribution");
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].token, 0);
        assert_eq!(candidates[1].token, 1);
        assert_close(candidates[0].prob, 4.0 / 7.0);
        assert_close(candidates[1].prob, 3.0 / 7.0);

        // A top_p under the leader's mass keeps only the leader. (The
        // mass here is 0.4 up to f32 input rounding, so the test uses
        // 0.39 rather than sitting exactly on the knife edge.)
        let config = SamplingConfig {
            top_p: 0.39,
            ..config
        };
        let candidates = distribution(&logits, &config).expect("distribution");
        assert_eq!(candidates.len(), 1);
        assert_close(candidates[0].prob, 1.0);

        // top_p 1.0 keeps the whole distribution, unrenormalized.
        let config = SamplingConfig {
            top_p: 1.0,
            ..config
        };
        let candidates = distribution(&logits, &config).expect("distribution");
        assert_eq!(candidates.len(), 4);
        assert_close(candidates[0].prob, 0.4);
        assert_close(candidates[1].prob, 0.3);
        assert_close(candidates[2].prob, 0.2);
        assert_close(candidates[3].prob, 0.1);
        let total: f64 = candidates.iter().map(|c| c.prob).sum();
        assert_close(total, 1.0);
    }

    #[test]
    fn top_k_keeps_the_highest_logits() {
        let logits = [0.0f32, 3.0, 1.0, 2.0, -1.0];
        let config = SamplingConfig {
            top_k: 2,
            ..SamplingConfig::default()
        };
        let candidates = distribution(&logits, &config).expect("distribution");
        let tokens: Vec<u32> = candidates.iter().map(|c| c.token).collect();
        assert_eq!(tokens, vec![1, 3]);
    }

    /// Penalty math on known logits, each rule isolated, then all
    /// three composed in pipeline order.
    #[test]
    fn penalty_math_on_known_logits() {
        let base = [4.0f32, -4.0, 2.0, 1.0];

        // Repetition: positive divides, negative multiplies, absent
        // tokens untouched.
        let config = SamplingConfig {
            repetition_penalty: 2.0,
            ..SamplingConfig::default()
        };
        let mut logits = base;
        apply_penalties(&mut logits, &[0, 1], &config);
        assert_eq!(logits, [2.0, -8.0, 2.0, 1.0]);

        // Frequency: scales with the count.
        let config = SamplingConfig {
            frequency_penalty: 0.5,
            ..SamplingConfig::default()
        };
        let mut logits = base;
        apply_penalties(&mut logits, &[2, 2, 2], &config);
        assert!((logits[2] - 0.5).abs() < 1e-6, "2.0 - 3*0.5 = 0.5");
        assert_eq!(logits[0], 4.0);

        // Presence: once per distinct token, however often it occurs.
        let config = SamplingConfig {
            presence_penalty: 0.25,
            ..SamplingConfig::default()
        };
        let mut logits = base;
        apply_penalties(&mut logits, &[3, 3, 3], &config);
        assert!((logits[3] - 0.75).abs() < 1e-6, "1.0 - 0.25 = 0.75");

        // Composed: token 0 appears twice. Repetition first
        // (4.0 / 2.0 = 2.0), then frequency (2 * 0.5 = 1.0) and
        // presence (0.25) subtract: 2.0 - 1.0 - 0.25 = 0.75.
        let config = SamplingConfig {
            repetition_penalty: 2.0,
            frequency_penalty: 0.5,
            presence_penalty: 0.25,
            ..SamplingConfig::default()
        };
        let mut logits = base;
        apply_penalties(&mut logits, &[0, 0], &config);
        assert!((logits[0] - 0.75).abs() < 1e-6);
        assert_eq!(logits[1], -4.0, "tokens outside the history are untouched");

        // History tokens beyond the vocabulary are ignored, not fatal.
        let mut logits = base;
        apply_penalties(&mut logits, &[999], &config);
        assert_eq!(logits, base);
    }

    #[test]
    fn empty_logits_is_a_named_error() {
        let mut sampler = Sampler::new(SamplingConfig::default());
        let err = sampler.sample(&[], &[]).expect_err("empty must fail");
        assert!(matches!(err, GlimmerError::EmptyLogits), "got {err:?}");
        let err = distribution(&[], &SamplingConfig::default()).expect_err("empty must fail");
        assert!(matches!(err, GlimmerError::EmptyLogits), "got {err:?}");
    }

    #[test]
    fn all_negative_infinity_is_a_named_error_not_a_panic() {
        let logits = [f32::NEG_INFINITY; 4];
        let mut greedy = Sampler::new(SamplingConfig {
            temperature: 0.0,
            ..SamplingConfig::default()
        });
        let err = greedy.sample(&logits, &[]).expect_err("no viable token");
        assert!(matches!(err, GlimmerError::NoViableToken), "got {err:?}");

        let mut warm = Sampler::new(SamplingConfig::default());
        let err = warm.sample(&logits, &[]).expect_err("no viable token");
        assert!(matches!(err, GlimmerError::NoViableToken), "got {err:?}");

        let nan_logits = [f32::NAN; 3];
        let err = warm.sample(&nan_logits, &[]).expect_err("NaN has no mass");
        assert!(matches!(err, GlimmerError::NoViableToken), "got {err:?}");
    }

    #[test]
    fn invalid_config_names_the_field() {
        let cases: Vec<(SamplingConfig, &'static str)> = vec![
            (
                SamplingConfig {
                    temperature: -0.5,
                    ..SamplingConfig::default()
                },
                "temperature",
            ),
            (
                SamplingConfig {
                    top_p: 0.0,
                    ..SamplingConfig::default()
                },
                "top_p",
            ),
            (
                SamplingConfig {
                    top_p: 1.5,
                    ..SamplingConfig::default()
                },
                "top_p",
            ),
            (
                SamplingConfig {
                    repetition_penalty: 0.0,
                    ..SamplingConfig::default()
                },
                "repetition_penalty",
            ),
            (
                SamplingConfig {
                    frequency_penalty: f32::INFINITY,
                    ..SamplingConfig::default()
                },
                "frequency_penalty",
            ),
        ];
        for (config, field) in cases {
            let mut sampler = Sampler::new(config);
            let err = sampler.sample(&[1.0, 2.0], &[]).expect_err("bad config");
            match err {
                GlimmerError::InvalidSampling { field: named, .. } => {
                    assert_eq!(named, field)
                }
                other => panic!("expected InvalidSampling, got {other:?}"),
            }
        }
    }

    #[test]
    fn nan_logits_are_excluded_not_drawn() {
        let logits = [f32::NAN, 1.0, 0.0];
        let candidates = distribution(&logits, &SamplingConfig::default()).expect("distribution");
        assert!(candidates.iter().all(|c| c.token != 0));
        assert_eq!(greedy_token(&logits).expect("greedy"), 1);
    }
}
