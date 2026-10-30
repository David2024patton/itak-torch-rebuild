//! The loop's random source (SPEC-itak-spec open question Q6, settled by
//! this crate: the session owns one seeded generator, injected at
//! construction through [`crate::SpecConfig::seed`], never global state).
//!
//! The generator is xorshift64*, the same one the engine's sampler uses
//! (`crates/itak-engine/src/sampler.rs`), so a seed means the same stream
//! of draws in both places. Greedy rounds draw nothing; the generator only
//! advances when temperature sampling actually consumes a draw, which
//! keeps greedy runs perfectly reproducible regardless of the seed.

/// A seeded xorshift64* generator.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Seed the generator. A zero seed is remapped to a fixed nonzero
    /// constant, exactly as the engine sampler does, so seed 0 is still a
    /// well-defined reproducible stream.
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        };
        Rng { state }
    }

    /// The next 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// The next draw in `[0, 1)` with 53 bits of precision.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..64 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_different_streams() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let sa: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        let sb: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
        assert_ne!(sa, sb);
    }

    #[test]
    fn zero_seed_is_remapped_and_deterministic() {
        let mut a = Rng::new(0);
        let mut b = Rng::new(0);
        for _ in 0..16 {
            let x = a.next_f64();
            assert_eq!(x, b.next_f64());
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn draws_stay_in_unit_interval() {
        let mut r = Rng::new(7);
        for _ in 0..10_000 {
            let x = r.next_f64();
            assert!((0.0..1.0).contains(&x), "draw {x} escaped [0, 1)");
        }
    }
}
