//! Distribution construction and the correction walk (SPEC-itak-spec
//! Section 4): pure functions, no model, no cache, so the math is testable
//! in isolation (TODO T-3.7).
//!
//! Indexing contract, settled against the E3 real-file run
//! (`bench/dflash-e3-2026-10-30.md`): the target's block forward returns one
//! finished distribution per input position; row `p[i]` is the distribution
//! the target assigns to the token AFTER position `i`, so `p[i]` judges
//! draft token `t[i + 1]`. The draft's noise block obeys the same shape:
//! draft row `k` (0-based) is the draft distribution for draft token
//! `t[k + 1]` (1-based). The walk below takes proposal tokens `t[0..n]`
//! (0-based slice of the 1-based `t[1..=n]`), proposal distributions
//! `q[0..n]` aligned with them, and target rows `p[0..=n]`.
//!
//! The walk, exactly as the spec states it:
//!
//! - Greedy (temperature 0): accept `t[i]` iff it equals the target's
//!   argmax at `p[i - 1]`. At the first mismatch, emit the target's argmax
//!   at that row as the replacement. On full acceptance, emit the target's
//!   argmax at `p[n]` as the bonus. The emitted stream is then
//!   token-identical to non-speculative greedy generation.
//! - Sampled: draw `r` uniform in `[0, 1)`; accept `t[i]` iff
//!   `r < min(1, p(t[i]) / q(t[i]))`. At the first rejection `j`, emit a
//!   replacement drawn from `norm(max(0, p[j - 1] - q[j]))`. On full
//!   acceptance, emit a bonus drawn from `p[n]`. This is the standard
//!   speculative-sampling correction: the emitted distribution equals the
//!   target's, token for token in distribution.
//!
//! Distribution construction mirrors the engine sampler
//! (`crates/itak-engine/src/sampler.rs`) so both paths mean the same thing
//! by a temperature: a numerically stable softmax over `(z - max) / T` in
//! f64, nucleus truncation that sorts by probability descending with ties
//! broken by the lower token id, and argmax ties broken by the lowest id.

/// The greedy choice over a finished logit row: the largest logit, lowest
/// token id on ties (the engine sampler's tie rule).
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best as u32
}

/// Build a probability distribution from finished logits at a temperature.
///
/// Temperature at or below zero returns the one-hot distribution at the
/// argmax, which is the honest greedy distribution. Positive temperatures
/// apply the stable softmax; `top_p` in `(0, 1)` then nucleus-truncates
/// and renormalizes; any other `top_p` disables truncation.
pub fn distribution(logits: &[f32], temperature: f32, top_p: f32) -> Vec<f32> {
    if temperature <= 0.0 {
        let mut out = vec![0.0f32; logits.len()];
        out[argmax(logits) as usize] = 1.0;
        return out;
    }
    let t = f64::from(temperature);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f64> = logits
        .iter()
        .map(|&z| ((f64::from(z) - f64::from(max)) / t).exp())
        .collect();
    let sum: f64 = probs.iter().sum();
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }
    if top_p > 0.0 && top_p < 1.0 {
        nucleus_truncate(&mut probs, f64::from(top_p));
    }
    probs.iter().map(|&p| p as f32).collect()
}

/// Nucleus truncation in place: keep the smallest set of highest-probability
/// tokens whose cumulative mass reaches `top_p`, zero the rest, renormalize.
fn nucleus_truncate(probs: &mut [f64], top_p: f64) {
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut cumulative = 0.0f64;
    let mut keep = vec![false; probs.len()];
    for &idx in &order {
        keep[idx] = true;
        cumulative += probs[idx];
        if cumulative >= top_p {
            break;
        }
    }
    let mut kept_sum = 0.0f64;
    for (i, p) in probs.iter_mut().enumerate() {
        if keep[i] {
            kept_sum += *p;
        } else {
            *p = 0.0;
        }
    }
    if kept_sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= kept_sum;
        }
    }
}

/// Draw one token from a probability row using a uniform draw in `[0, 1)`.
/// The row need not be perfectly normalized; the draw is scaled by the
/// row's actual mass.
pub fn sample_from(probs: &[f32], r: f64) -> u32 {
    let total: f64 = probs.iter().map(|&p| f64::from(p)).sum();
    if total <= 0.0 {
        return argmax(probs);
    }
    let target = r * total;
    let mut cumulative = 0.0f64;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += f64::from(p);
        if target < cumulative {
            return i as u32;
        }
    }
    // Float edge: the draw landed past the accumulated mass. Fall back to
    // the last token carrying any mass.
    for (i, &p) in probs.iter().enumerate().rev() {
        if p > 0.0 {
            return i as u32;
        }
    }
    argmax(probs)
}

/// Draw from the normalized residual `norm(max(0, p - q))`, the correction
/// distribution at a rejection. If the residual has no mass (reachable
/// only through float rounding on a genuine rejection, where
/// `p(t) < q(t)` already guarantees mass somewhere else in exact math),
/// the draw falls back to `p` itself; that guard is numerical, not a
/// change to the correction rule.
pub fn sample_residual(p: &[f32], q: &[f32], r: f64) -> u32 {
    let residual: Vec<f64> = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| (f64::from(pi) - f64::from(qi)).max(0.0))
        .collect();
    let total: f64 = residual.iter().sum();
    if total <= 0.0 {
        return sample_from(p, r);
    }
    let target = r * total;
    let mut cumulative = 0.0f64;
    for (i, &m) in residual.iter().enumerate() {
        cumulative += m;
        if target < cumulative {
            return i as u32;
        }
    }
    sample_from(p, r)
}

/// The outcome of one correction walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalkOutcome {
    /// How many leading draft tokens were accepted.
    pub accepted: usize,
    /// The token the walk emits after the accepted prefix: the replacement
    /// at the first rejection, or the bonus on full acceptance.
    pub token: u32,
    /// The 1-based index of the first rejected draft token (`j` in the
    /// spec's walk), or `None` on full acceptance.
    pub first_rejection: Option<usize>,
}

/// The greedy walk. `p_argmax` holds the target's argmax at rows
/// `p[0..=n]` (one more row than the proposal: the last row supplies the
/// bonus). `proposal` holds the draft tokens `t[1..=n]`.
///
/// Pure prefix matching against the target's own greedy choices, which is
/// what makes the temperature-0 identity hold by construction.
pub fn correct_greedy(p_argmax: &[u32], proposal: &[u32]) -> WalkOutcome {
    let n = proposal.len();
    let mut accepted = 0;
    while accepted < n && proposal[accepted] == p_argmax[accepted] {
        accepted += 1;
    }
    if accepted == n {
        WalkOutcome {
            accepted,
            token: p_argmax[n],
            first_rejection: None,
        }
    } else {
        WalkOutcome {
            accepted,
            token: p_argmax[accepted],
            first_rejection: Some(accepted + 1),
        }
    }
}

/// The sampled walk (rejection sampling, SPEC-itak-spec Sections 4 and
/// T-3.2). `p_rows` holds the target distributions at rows `p[0..=n]`,
/// `q_rows` the draft distributions aligned with `proposal`. `rng` yields
/// uniform draws in `[0, 1)`; a full-acceptance walk consumes `n + 1`
/// draws (one per proposal token plus the bonus), a walk rejected at `j`
/// consumes `j + 1` (the rejections draws plus the residual draw).
pub fn correct_sampled(
    p_rows: &[&[f32]],
    q_rows: &[&[f32]],
    proposal: &[u32],
    rng: &mut dyn FnMut() -> f64,
) -> WalkOutcome {
    let n = proposal.len();
    for k in 0..n {
        let token = proposal[k] as usize;
        let p_t = f64::from(p_rows[k][token]);
        let q_t = f64::from(q_rows[k][token]);
        let ratio = if q_t > 0.0 {
            p_t / q_t
        } else if p_t > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        let r = rng();
        if r < ratio.min(1.0) {
            continue;
        }
        let replacement = sample_residual(p_rows[k], q_rows[k], rng());
        return WalkOutcome {
            accepted: k,
            token: replacement,
            first_rejection: Some(k + 1),
        };
    }
    let bonus = sample_from(p_rows[n], rng());
    WalkOutcome {
        accepted: n,
        token: bonus,
        first_rejection: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scripted draw source: the walk must consume draws in order.
    fn scripted(draws: &[f64]) -> (Vec<f64>, usize) {
        (draws.to_vec(), 0)
    }

    #[test]
    fn argmax_breaks_ties_by_lowest_id() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[-5.0, -1.0]), 1);
    }

    #[test]
    fn distribution_at_zero_temperature_is_one_hot() {
        let d = distribution(&[0.0, 2.0, 1.0], 0.0, 1.0);
        assert_eq!(d, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn distribution_softmax_matches_hand_math() {
        // logits [0, ln 2] at T=1: exp gives [1, 2], normalized [1/3, 2/3].
        let d = distribution(&[0.0, std::f32::consts::LN_2], 1.0, 1.0);
        assert!((d[0] - 1.0 / 3.0).abs() < 1e-6, "got {d:?}");
        assert!((d[1] - 2.0 / 3.0).abs() < 1e-6, "got {d:?}");
        // Temperature 2 flattens: exp([0, ln2 / 2]) = [1, sqrt(2)].
        let d2 = distribution(&[0.0, std::f32::consts::LN_2], 2.0, 1.0);
        let sqrt2 = std::f64::consts::SQRT_2;
        let expect0 = (1.0 / (1.0 + sqrt2)) as f32;
        assert!((d2[0] - expect0).abs() < 1e-6, "got {d2:?}");
    }

    #[test]
    fn distribution_nucleus_truncates_and_renormalizes() {
        // Probs before truncation: [0.5, 0.3, 0.2] via logits ln of those.
        let logits = [0.5f32.ln(), 0.3f32.ln(), 0.2f32.ln()];
        let d = distribution(&logits, 1.0, 0.6);
        // Top token alone is 0.5 < 0.6, so the nucleus is {0, 1} with mass
        // 0.8, renormalized to [0.625, 0.375, 0].
        assert!((d[0] - 0.625).abs() < 1e-5, "got {d:?}");
        assert!((d[1] - 0.375).abs() < 1e-5, "got {d:?}");
        assert_eq!(d[2], 0.0);
    }

    #[test]
    fn greedy_walk_accepts_the_matching_prefix() {
        // Target argmaxes per row p[0..=3]; proposal matches twice, then
        // diverges: replacement is the target argmax at the mismatch row.
        let p_argmax = [10, 20, 30, 40];
        let out = correct_greedy(&p_argmax, &[10, 20, 99]);
        assert_eq!(out.accepted, 2);
        assert_eq!(out.token, 30);
        assert_eq!(out.first_rejection, Some(3));
    }

    #[test]
    fn greedy_walk_full_acceptance_takes_the_bonus() {
        let p_argmax = [10, 20, 30, 40];
        let out = correct_greedy(&p_argmax, &[10, 20, 30]);
        assert_eq!(out.accepted, 3);
        assert_eq!(out.token, 40);
        assert_eq!(out.first_rejection, None);
    }

    #[test]
    fn greedy_walk_first_token_mismatch() {
        let p_argmax = [10, 20];
        let out = correct_greedy(&p_argmax, &[77]);
        assert_eq!(out.accepted, 0);
        assert_eq!(out.token, 10);
        assert_eq!(out.first_rejection, Some(1));
    }

    #[test]
    fn greedy_walk_empty_proposal_is_all_bonus() {
        let out = correct_greedy(&[55], &[]);
        assert_eq!(out.accepted, 0);
        assert_eq!(out.token, 55);
        assert_eq!(out.first_rejection, None);
    }

    #[test]
    fn sampled_walk_accepts_when_the_draw_beats_the_ratio() {
        // p = [0.5, 0.5], q = [0.25, 0.75]; proposal token 1 has
        // p/q = 2/3. Draw 0.5 accepts; the bonus then comes from p[1].
        let p0: &[f32] = &[0.5, 0.5];
        let p1: &[f32] = &[0.1, 0.9];
        let q0: &[f32] = &[0.25, 0.75];
        let (draws, mut used) = scripted(&[0.5, 0.5]);
        let mut rng = || {
            let d = draws[used];
            used += 1;
            d
        };
        let out = correct_sampled(&[p0, p1], &[q0], &[1], &mut rng);
        assert_eq!(out.accepted, 1);
        assert_eq!(out.first_rejection, None);
        // Bonus draw 0.5 against p1 = [0.1, 0.9] lands on token 1.
        assert_eq!(out.token, 1);
        assert_eq!(used, 2);
    }

    #[test]
    fn sampled_walk_rejection_draws_from_the_residual() {
        // Same setup; draw 0.7 > 2/3 rejects. Residual max(0, p - q) =
        // [0.25, 0], normalized [1, 0]: the replacement is token 0 for
        // any residual draw.
        let p0: &[f32] = &[0.5, 0.5];
        let p1: &[f32] = &[0.1, 0.9];
        let q0: &[f32] = &[0.25, 0.75];
        let (draws, mut used) = scripted(&[0.7, 0.99]);
        let mut rng = || {
            let d = draws[used];
            used += 1;
            d
        };
        let out = correct_sampled(&[p0, p1], &[q0], &[1], &mut rng);
        assert_eq!(out.accepted, 0);
        assert_eq!(out.token, 0);
        assert_eq!(out.first_rejection, Some(1));
        assert_eq!(used, 2);
    }

    #[test]
    fn sampled_walk_certain_token_always_accepts() {
        // q puts all mass on the proposed token and p agrees: ratio is 1,
        // and even a draw of 0.999 accepts (r < min(1, ratio) = 1).
        let p0: &[f32] = &[0.0, 1.0];
        let p1: &[f32] = &[1.0, 0.0];
        let q0: &[f32] = &[0.0, 1.0];
        let (draws, mut used) = scripted(&[0.999, 0.0]);
        let mut rng = || {
            let d = draws[used];
            used += 1;
            d
        };
        let out = correct_sampled(&[p0, p1], &[q0], &[1], &mut rng);
        assert_eq!(out.accepted, 1);
        assert_eq!(out.token, 0);
    }

    #[test]
    fn residual_distribution_is_the_normalized_positive_part() {
        // p = [0.2, 0.3, 0.5], q = [0.5, 0.25, 0.25]: residual is
        // [0, 0.05, 0.25], normalized [0, 1/6, 5/6].
        let p: &[f32] = &[0.2, 0.3, 0.5];
        let q: &[f32] = &[0.5, 0.25, 0.25];
        assert_eq!(sample_residual(p, q, 0.1), 1);
        assert_eq!(sample_residual(p, q, 0.5), 2);
        assert_eq!(sample_residual(p, q, 0.0), 1);
    }

    #[test]
    fn sample_from_uses_cumulative_mass_in_id_order() {
        let probs: &[f32] = &[0.25, 0.25, 0.5];
        assert_eq!(sample_from(probs, 0.0), 0);
        assert_eq!(sample_from(probs, 0.3), 1);
        assert_eq!(sample_from(probs, 0.9), 2);
    }
}
