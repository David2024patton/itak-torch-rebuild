//! Sampling over the forward's finished logits.
//!
//! The protocol carries temperature, top-p, top-k, a seed, and the
//! presence/frequency penalties from the platform's request; this
//! module is where they are honored. Temperature at or below zero
//! is greedy (argmax, lowest id on ties), which is the temperature-0
//! behavior the golden-output proofs rely on. Penalties are applied
//! first, in the OpenAI form: a token's logit drops by
//! `frequency_penalty * count + presence_penalty * (count > 0)`,
//! where the count covers the prompt tokens (fed via
//! [`Sampler::observe`]) and every token this sampler has already
//! picked. Above zero temperature, the penalized logits are
//! truncated to the top-k set when top-k is set, softened with a
//! numerically stable softmax in f64, truncated again to the
//! smallest nucleus whose mass reaches top-p (top-p outside (0, 1)
//! disables truncation), and one token is drawn from the
//! renormalized nucleus with a seeded xorshift64* generator. The
//! seed is an explicit input so a sampled run is reproducible end
//! to end (TODO T-9.13); the server seeds from the clock only when
//! the request carries no seed.

use std::collections::HashMap;

/// One generation's sampling parameters, all from the request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerConfig {
    /// Sampling temperature; at or below 0 the sampler is greedy.
    pub temperature: f64,
    /// Nucleus mass cap; outside (0, 1) disables truncation.
    pub top_p: f64,
    /// Candidate set cap; 0 disables top-k truncation.
    pub top_k: usize,
    /// Flat logit penalty for any prior occurrence (OpenAI form).
    pub presence_penalty: f64,
    /// Per-occurrence logit penalty (OpenAI form).
    pub frequency_penalty: f64,
    /// Generator seed; any value is accepted (zero is remapped,
    /// xorshift has no zero state). Ignored when greedy.
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig {
            temperature: 0.0,
            top_p: 0.0,
            top_k: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: 0,
        }
    }
}

/// Greedy and stochastic token selection over one logits row, with
/// repetition state for the penalties.
pub struct Sampler {
    cfg: SamplerConfig,
    state: u64,
    counts: HashMap<u32, u32>,
}

impl Sampler {
    /// A sampler for one generation.
    pub fn new(cfg: SamplerConfig) -> Self {
        Sampler {
            cfg,
            state: if cfg.seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                cfg.seed
            },
            counts: HashMap::new(),
        }
    }

    /// Whether this sampler ignores its generator and takes argmax.
    pub fn is_greedy(&self) -> bool {
        self.cfg.temperature <= 0.0
    }

    /// Feed tokens into the penalty counts without sampling them:
    /// the prompt, so a penalty request counts prompt repetitions
    /// the way OpenAI's form specifies.
    pub fn observe(&mut self, tokens: &[u32]) {
        for &t in tokens {
            *self.counts.entry(t).or_insert(0) += 1;
        }
    }

    /// The penalized logit for one token id.
    fn adjusted(&self, id: u32, logit: f32) -> f64 {
        let count = self.counts.get(&id).copied().unwrap_or(0);
        if count == 0 || (self.cfg.presence_penalty == 0.0 && self.cfg.frequency_penalty == 0.0) {
            return f64::from(logit);
        }
        let mut z = f64::from(logit);
        z -= self.cfg.frequency_penalty * f64::from(count);
        if count > 0 {
            z -= self.cfg.presence_penalty;
        }
        z
    }

    /// Pick the next token id from a finished logits row. The pick
    /// joins the penalty counts, so a repeated pick costs more
    /// under the penalties than the first one did.
    pub fn pick(&mut self, logits: &[f32]) -> u32 {
        let token = self.select(logits);
        *self.counts.entry(token).or_insert(0) += 1;
        token
    }

    fn select(&mut self, logits: &[f32]) -> u32 {
        if logits.is_empty() {
            return 0;
        }
        // Penalized logits, in id order.
        let adjusted: Vec<f64> = logits
            .iter()
            .enumerate()
            .map(|(i, &z)| self.adjusted(i as u32, z))
            .collect();
        if self.is_greedy() || logits.len() <= 1 {
            return argmax(&adjusted);
        }
        // Candidate set: top-k by penalized logit when top-k is
        // set and smaller than the row, else the whole row.
        let mut candidates: Vec<u32> = (0..logits.len() as u32).collect();
        if self.cfg.top_k > 0 && self.cfg.top_k < candidates.len() {
            candidates.sort_by(|&a, &b| {
                adjusted[b as usize]
                    .partial_cmp(&adjusted[a as usize])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            candidates.truncate(self.cfg.top_k);
            candidates.sort_unstable();
        }
        // Stable softmax in f64 over the candidates.
        let temp = self.cfg.temperature;
        let max = candidates
            .iter()
            .map(|&id| adjusted[id as usize])
            .fold(f64::NEG_INFINITY, f64::max);
        let mut probs: Vec<(u32, f64)> = candidates
            .iter()
            .map(|&id| (id, ((adjusted[id as usize] - max) / temp).exp()))
            .collect();
        let total: f64 = probs.iter().map(|(_, p)| p).sum();
        for (_, p) in probs.iter_mut() {
            *p /= total;
        }
        // Nucleus truncation on the probability-sorted order.
        let mut order: Vec<usize> = (0..probs.len()).collect();
        order.sort_by(|&a, &b| {
            probs[b]
                .1
                .partial_cmp(&probs[a].1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(probs[a].0.cmp(&probs[b].0))
        });
        let mut keep = probs.len();
        if self.cfg.top_p > 0.0 && self.cfg.top_p < 1.0 {
            let mut cumulative = 0.0;
            for (rank, &idx) in order.iter().enumerate() {
                cumulative += probs[idx].1;
                if cumulative >= self.cfg.top_p {
                    keep = rank + 1;
                    break;
                }
            }
        }
        let kept_mass: f64 = order[..keep].iter().map(|&idx| probs[idx].1).sum();
        let draw = self.next_f64() * kept_mass;
        let mut cumulative = 0.0;
        for &idx in &order[..keep] {
            cumulative += probs[idx].1;
            if draw < cumulative {
                return probs[idx].0;
            }
        }
        // Floating-point edge: the draw rounded up to the kept mass.
        probs[order[keep - 1]].0
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64* (Marsaglia): full period over nonzero states.
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn next_f64(&mut self) -> f64 {
        // 53 random bits, uniform on [0, 1).
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The lowest id holding the maximum logit (llama.cpp's greedy tie
/// behavior, and what the reference harness's top-1 records show).
fn argmax(logits: &[f64]) -> u32 {
    let mut best = 0u32;
    let mut best_val = f64::NEG_INFINITY;
    for (i, &z) in logits.iter().enumerate() {
        if z > best_val {
            best_val = z;
            best = i as u32;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(temperature: f64, top_p: f64, seed: u64) -> SamplerConfig {
        SamplerConfig {
            temperature,
            top_p,
            seed,
            ..SamplerConfig::default()
        }
    }

    #[test]
    fn greedy_picks_argmax_with_lowest_id_ties() {
        let mut s = Sampler::new(cfg(0.0, 0.0, 42));
        assert_eq!(s.pick(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(s.pick(&[-5.0, -9.0]), 0);
    }

    #[test]
    fn tiny_top_p_collapses_to_argmax() {
        // The first token's mass alone exceeds a tiny top-p for any
        // seed, so the nucleus is a single token.
        for seed in 0..16 {
            let mut s = Sampler::new(cfg(1.0, 1e-9, seed));
            assert_eq!(s.pick(&[0.0, 10.0, 0.0, 0.0]), 1, "seed {seed}");
        }
    }

    #[test]
    fn seeded_sampling_is_deterministic() {
        let logits = [0.3, 1.2, -0.5, 2.0, 0.0, 0.7];
        let mut a = Sampler::new(cfg(0.8, 0.9, 1234));
        let mut b = Sampler::new(cfg(0.8, 0.9, 1234));
        for _ in 0..32 {
            assert_eq!(a.pick(&logits), b.pick(&logits));
        }
    }

    #[test]
    fn seeded_runs_match_across_a_whole_sequence() {
        // The T-9.13 determinism shape: two samplers with the same
        // full option set (seed, top-k, top-p, both penalties) fed
        // the same sequence of logits rows produce identical token
        // streams, penalty state included.
        let rows: Vec<Vec<f32>> = (0..24)
            .map(|i| {
                vec![
                    (i % 7) as f32 * 0.3,
                    1.1 - (i % 3) as f32 * 0.2,
                    0.4,
                    2.0 - (i % 5) as f32 * 0.35,
                    -0.6,
                ]
            })
            .collect();
        let full = SamplerConfig {
            temperature: 0.9,
            top_p: 0.95,
            top_k: 4,
            presence_penalty: 0.3,
            frequency_penalty: 0.2,
            seed: 777,
        };
        let run = |c: SamplerConfig| -> Vec<u32> {
            let mut s = Sampler::new(c);
            s.observe(&[3, 3, 1]);
            rows.iter().map(|row| s.pick(row)).collect()
        };
        assert_eq!(run(full), run(full));
        // A different seed diverges somewhere in the sequence with
        // overwhelming probability; if it ever does not, the seed
        // is not actually reaching the generator.
        let other = SamplerConfig { seed: 778, ..full };
        assert_ne!(run(full), run(other));
    }

    #[test]
    fn hot_sampling_spreads_across_tokens() {
        // At a huge temperature the distribution is near uniform, so
        // a run of seeded draws cannot sit on one token.
        let logits = [0.0, 0.1, -0.1, 0.05];
        let mut seen = std::collections::BTreeSet::new();
        for seed in 0..200u64 {
            let mut s = Sampler::new(cfg(1.0e6, 1.0, seed.wrapping_mul(7919) + 1));
            seen.insert(s.pick(&logits));
        }
        assert!(seen.len() >= 3, "only saw {seen:?}");
    }

    #[test]
    fn peaked_distribution_dominates_at_moderate_temperature() {
        let logits = [0.0, 20.0, 0.0];
        for seed in 0..16 {
            let mut s = Sampler::new(cfg(1.0, 1.0, seed * 31 + 7));
            assert_eq!(s.pick(&logits), 1, "seed {seed}");
        }
    }

    #[test]
    fn top_k_one_is_greedy_even_sampled() {
        let logits = [0.3, 1.2, -0.5, 2.0, 0.0];
        for seed in 0..16 {
            let mut c = cfg(1.3, 1.0, seed);
            c.top_k = 1;
            let mut s = Sampler::new(c);
            assert_eq!(s.pick(&logits), 3, "seed {seed}");
        }
    }

    #[test]
    fn frequency_penalty_pushes_the_pick_off_a_repeated_token() {
        // Token 0 leads by 0.1; after 5 observations under a 0.5
        // frequency penalty its adjusted logit drops by 2.5, so the
        // greedy pick moves to token 1. Without the penalty the
        // same history changes nothing.
        let logits = [2.0, 1.9, 0.0];
        let mut plain = Sampler::new(cfg(0.0, 0.0, 1));
        plain.observe(&[0, 0, 0, 0, 0]);
        assert_eq!(plain.pick(&logits), 0);

        let mut c = cfg(0.0, 0.0, 1);
        c.frequency_penalty = 0.5;
        let mut penalized = Sampler::new(c);
        penalized.observe(&[0, 0, 0, 0, 0]);
        assert_eq!(penalized.pick(&logits), 1);
    }

    #[test]
    fn presence_penalty_costs_a_token_once_regardless_of_count() {
        // Token 0 leads by 0.4. A presence penalty of 0.5 moves the
        // pick after a single observation, and further observations
        // do not move it more (contrast with frequency above).
        let logits = [2.0, 1.6, 0.0];
        let mut c = cfg(0.0, 0.0, 1);
        c.presence_penalty = 0.5;
        let mut once = Sampler::new(c);
        once.observe(&[0]);
        assert_eq!(once.pick(&logits), 1);
        let mut many = Sampler::new(c);
        many.observe(&[0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(many.pick(&logits), 1);
        // And with the weaker penalty the single hit is not enough.
        let mut weak = Sampler::new(SamplerConfig {
            presence_penalty: 0.3,
            ..c
        });
        weak.observe(&[0, 0, 0]);
        assert_eq!(weak.pick(&logits), 0);
    }

    #[test]
    fn picked_tokens_join_the_penalty_counts() {
        // Greedy with a frequency penalty over a constant row must
        // rotate: each pick makes its token costlier until another
        // token wins. Without the penalty the same row repeats.
        let logits = [1.0, 0.95, 0.9];
        let mut c = cfg(0.0, 0.0, 1);
        c.frequency_penalty = 0.2;
        let mut s = Sampler::new(c);
        assert_eq!(s.pick(&logits), 0);
        assert_eq!(s.pick(&logits), 1); // 0 now costs 0.2, sits at 0.8
        assert_eq!(s.pick(&logits), 2); // then 1 also sits at 0.75
    }
}
