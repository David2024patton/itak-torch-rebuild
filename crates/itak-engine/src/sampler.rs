//! Sampling over the forward's finished logits.
//!
//! The protocol carries temperature and top-p from the platform's
//! request; this module is where they are honored. Temperature at or
//! below zero is greedy (argmax, lowest id on ties), which is the
//! temperature-0 behavior the golden-output proofs rely on. Above
//! zero, logits are scaled, softened with a numerically stable
//! softmax in f64, truncated to the smallest nucleus whose mass
//! reaches top-p (top-p outside (0, 1) disables truncation), and one
//! token is drawn from the renormalized nucleus with a seeded
//! xorshift64* generator. The seed is an explicit input so tests are
//! deterministic; the server seeds it from the clock and a job
//! counter.

/// Greedy and stochastic token selection over one logits row.
pub struct Sampler {
    temperature: f64,
    top_p: f64,
    state: u64,
}

impl Sampler {
    /// A sampler for one generation. `seed` only matters when
    /// `temperature > 0`; any value is accepted (zero is remapped,
    /// xorshift has no zero state).
    pub fn new(temperature: f64, top_p: f64, seed: u64) -> Self {
        Sampler {
            temperature,
            top_p,
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    /// Whether this sampler ignores its generator and takes argmax.
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    /// Pick the next token id from a finished logits row.
    pub fn pick(&mut self, logits: &[f32]) -> u32 {
        if self.is_greedy() || logits.len() <= 1 {
            return argmax(logits);
        }
        let temp = self.temperature;
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        // (id, probability) in id order, stable softmax in f64.
        let mut probs: Vec<(u32, f64)> = logits
            .iter()
            .enumerate()
            .map(|(i, &z)| (i as u32, (f64::from(z - max) / temp).exp()))
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
                .then(a.cmp(&b))
        });
        let mut keep = probs.len();
        if self.top_p > 0.0 && self.top_p < 1.0 {
            let mut cumulative = 0.0;
            for (rank, &idx) in order.iter().enumerate() {
                cumulative += probs[idx].1;
                if cumulative >= self.top_p {
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
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut best_val = f32::NEG_INFINITY;
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

    #[test]
    fn greedy_picks_argmax_with_lowest_id_ties() {
        let mut s = Sampler::new(0.0, 0.0, 42);
        assert_eq!(s.pick(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(s.pick(&[-5.0, -9.0]), 0);
    }

    #[test]
    fn tiny_top_p_collapses_to_argmax() {
        // The first token's mass alone exceeds a tiny top-p for any
        // seed, so the nucleus is a single token.
        for seed in 0..16 {
            let mut s = Sampler::new(1.0, 1e-9, seed);
            assert_eq!(s.pick(&[0.0, 10.0, 0.0, 0.0]), 1, "seed {seed}");
        }
    }

    #[test]
    fn seeded_sampling_is_deterministic() {
        let logits = [0.3, 1.2, -0.5, 2.0, 0.0, 0.7];
        let mut a = Sampler::new(0.8, 0.9, 1234);
        let mut b = Sampler::new(0.8, 0.9, 1234);
        for _ in 0..32 {
            assert_eq!(a.pick(&logits), b.pick(&logits));
        }
    }

    #[test]
    fn hot_sampling_spreads_across_tokens() {
        // At a huge temperature the distribution is near uniform, so
        // a run of seeded draws cannot sit on one token.
        let logits = [0.0, 0.1, -0.1, 0.05];
        let mut seen = std::collections::BTreeSet::new();
        for seed in 0..200u64 {
            let mut s = Sampler::new(1.0e6, 1.0, seed.wrapping_mul(7919) + 1);
            seen.insert(s.pick(&logits));
        }
        assert!(seen.len() >= 3, "only saw {seen:?}");
    }

    #[test]
    fn peaked_distribution_dominates_at_moderate_temperature() {
        let logits = [0.0, 20.0, 0.0];
        for seed in 0..16 {
            let mut s = Sampler::new(1.0, 1.0, seed * 31 + 7);
            assert_eq!(s.pick(&logits), 1, "seed {seed}");
        }
    }
}
