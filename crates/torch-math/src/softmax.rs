//! Softmax over one row of logits: the distribution the sampler and
//! the attention weights both come from. The two properties this
//! module exists to guarantee are the ones a naive `exp(x) / sum`
//! gets wrong:
//!
//! - **Numerical stability.** The row maximum is subtracted before
//!   exponentiating, so logits in the thousands (routine for a
//!   confident LM head) produce the same distribution as their
//!   shifted selves instead of `inf / inf = NaN`.
//! - **Masked entries.** An entry of negative infinity is how an
//!   attention mask says "this position does not exist" (the
//!   convention in llama.cpp's softmax and in the HF mask builders).
//!   Such entries come out as exactly `0.0`, never as NaN: with the
//!   maximum taken over the finite entries, `exp(-inf - max)` is
//!   `exp(-inf) = 0`.
//!
//! A row where *every* entry is negative infinity has no defined
//! distribution; like everything else misuse-shaped in this crate it
//! is a named error, not a silent row of zeros. Callers that mask
//! (the attention crate) must refuse an all-masked row themselves,
//! as `attention` does with its `NoVisibleKeys` error.
//!
//! The exponentials and the denominator are accumulated in `f64` so
//! the reference is the most accurate version of the formula; the
//! probabilities are stored as `f32`.

use crate::{dbg_log, MathError, Result};

/// Softmax of one row of logits:
///
/// `out[i] = exp(x[i] - max(x)) / sum_j exp(x[j] - max(x))`
///
/// where `max(x)` is taken over the finite entries (subtracting the
/// maximum does not change the mathematical result; it keeps the
/// exponentials in range).
///
/// # Errors
///
/// - [`MathError::EmptyInput`] if `input` is empty.
/// - [`MathError::InvalidConfig`] if any entry is NaN or positive
///   infinity (neither has a probability meaning), or if every entry
///   is negative infinity (an all-masked row has no distribution).
pub fn softmax(input: &[f32]) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Err(MathError::EmptyInput { op: "softmax" });
    }
    let mut max = f64::NEG_INFINITY;
    for &v in input {
        if v.is_nan() || v == f32::INFINITY {
            return Err(MathError::InvalidConfig {
                op: "softmax",
                param: "input",
                reason: "logits must not be NaN or positive infinity",
            });
        }
        let v = f64::from(v);
        if v > max {
            max = v;
        }
    }
    if max == f64::NEG_INFINITY {
        return Err(MathError::InvalidConfig {
            op: "softmax",
            param: "input",
            reason: "every logit is negative infinity; the distribution is undefined",
        });
    }
    dbg_log!("softmax len={} max={max}", input.len());
    let mut out = Vec::with_capacity(input.len());
    let mut sum = 0.0f64;
    for &v in input {
        // For v = -inf this is exp(-inf) = 0 exactly: the masked
        // entry contributes nothing to the denominator and receives
        // exactly zero probability.
        let e = (f64::from(v) - max).exp();
        sum += e;
        out.push(e);
    }
    // sum >= exp(0) = 1 because the maximum entry itself contributes
    // exp(0); the division can never be by zero.
    Ok(out.iter().map(|e| (e / sum) as f32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::assert_vec_close;

    #[test]
    fn uniform_logits_give_uniform_probabilities() {
        let out = softmax(&[1.0, 1.0, 1.0]).unwrap();
        assert_vec_close(&out, &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 1e-6);
    }

    #[test]
    fn known_values() {
        // exp([0, 1, 2]) / (1 + e + e^2), constants from the
        // definition evaluated independently.
        let out = softmax(&[0.0, 1.0, 2.0]).unwrap();
        assert_vec_close(
            &out,
            &[
                0.090_030_573_170_380_46,
                0.244_728_471_054_797_64,
                0.665_240_955_774_821_9,
            ],
            1e-6,
        );
    }

    #[test]
    fn single_element_is_one() {
        assert_eq!(softmax(&[-42.5]).unwrap(), vec![1.0]);
    }

    #[test]
    fn large_logits_do_not_overflow() {
        // exp(1001) overflows f64, let alone f32; with the maximum
        // subtracted the result equals softmax([-1, 0, -2]).
        let out = softmax(&[1000.0, 1001.0, 999.0]).unwrap();
        assert_vec_close(
            &out,
            &[
                0.244_728_471_054_797_64,
                0.665_240_955_774_821_9,
                0.090_030_573_170_380_46,
            ],
            1e-6,
        );
    }

    #[test]
    fn large_negative_logits_do_not_underflow_to_zero_sum() {
        // Without the shift every exp underflows and the denominator
        // is 0; the distribution must still be the true one.
        let out = softmax(&[-1000.0, -1001.0]).unwrap();
        assert_vec_close(
            &out,
            &[0.731_058_578_630_004_9, 0.268_941_421_369_995_1],
            1e-6,
        );
    }

    #[test]
    fn negative_infinity_entries_get_exactly_zero() {
        let out = softmax(&[1.0, f32::NEG_INFINITY, 2.0]).unwrap();
        assert_eq!(out[1], 0.0);
        // The finite entries form softmax([1, 2]) between themselves.
        assert_vec_close(
            &[out[0], out[2]],
            &[0.268_941_421_369_995_1, 0.731_058_578_630_004_9],
            1e-6,
        );
    }

    #[test]
    fn all_negative_infinity_is_an_error() {
        assert!(matches!(
            softmax(&[f32::NEG_INFINITY, f32::NEG_INFINITY]),
            Err(MathError::InvalidConfig { param: "input", .. })
        ));
    }

    #[test]
    fn nan_and_positive_infinity_are_errors() {
        for bad in [f32::NAN, f32::INFINITY] {
            assert!(
                matches!(
                    softmax(&[0.0, bad]),
                    Err(MathError::InvalidConfig { param: "input", .. })
                ),
                "logit {bad} must be rejected"
            );
        }
    }

    #[test]
    fn empty_input_is_an_error() {
        assert_eq!(softmax(&[]), Err(MathError::EmptyInput { op: "softmax" }));
    }

    #[test]
    fn probabilities_sum_to_one() {
        let out = softmax(&[3.25, -1.5, 0.0, 8.75, 2.0]).unwrap();
        let sum: f64 = out.iter().map(|p| f64::from(*p)).sum();
        assert!((sum - 1.0).abs() <= 1e-6, "probabilities sum to {sum}");
    }
}
