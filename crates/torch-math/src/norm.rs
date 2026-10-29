//! RMSNorm: the normalization every dissected family uses in place of
//! LayerNorm. The one parameter the old Go engine got wrong by
//! hardcoding is the point of this module: `eps` is an argument.
//! Real models differ (Llama uses `1e-5`, Qwen2 and Gemma use `1e-6`),
//! so it is never a constant here.

use crate::{dbg_log, MathError, Result};

/// RMS-normalize `input` and scale elementwise by `weight`:
///
/// `out[i] = input[i] / sqrt(mean(input^2) + eps) * weight[i]`
///
/// The mean of squares is accumulated in `f64` so the reference stays
/// accurate for wide hidden dimensions (Glimmer: 6656).
///
/// # Errors
///
/// - [`MathError::EmptyInput`] if `input` is empty (the mean is undefined).
/// - [`MathError::LengthMismatch`] if `weight` is not the same length as `input`.
/// - [`MathError::InvalidConfig`] if `eps` is not finite and greater than zero.
pub fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Err(MathError::EmptyInput { op: "rms_norm" });
    }
    if weight.len() != input.len() {
        return Err(MathError::LengthMismatch {
            op: "rms_norm",
            expected: input.len(),
            actual: weight.len(),
        });
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(MathError::InvalidConfig {
            op: "rms_norm",
            param: "eps",
            reason: "eps must be finite and > 0",
        });
    }
    dbg_log!("rms_norm len={} eps={eps}", input.len());
    let len = input.len() as f64;
    let mean_sq = input
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        / len;
    let inv_rms = 1.0 / (mean_sq + f64::from(eps)).sqrt();
    Ok(input
        .iter()
        .zip(weight.iter())
        .map(|(v, w)| (f64::from(*v) * inv_rms * f64::from(*w)) as f32)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{assert_close, assert_vec_close};

    #[test]
    fn constant_vector_normalizes_to_weight() {
        // mean(x^2) = 4, so each element maps to 2 / sqrt(4 + eps) * 1,
        // which is 1 up to the eps correction (~0.99999988 for eps 1e-6).
        let out = rms_norm(&[2.0, 2.0, 2.0, 2.0], &[1.0, 1.0, 1.0, 1.0], 1e-6).unwrap();
        assert_vec_close(&out, &[1.0, 1.0, 1.0, 1.0], 1e-5);
    }

    #[test]
    fn known_vector() {
        // mean of squares of [1,2,3,4] is 7.5; 1/sqrt(7.5) = 0.36514837...
        // (eps 1e-5 shifts the fifth decimal only).
        let out = rms_norm(&[1.0, 2.0, 3.0, 4.0], &[1.0, 1.0, 1.0, 1.0], 1e-5).unwrap();
        assert_vec_close(&out, &[0.36515, 0.73030, 1.09545, 1.46061], 1e-4);
    }

    #[test]
    fn weight_scales_output() {
        let out = rms_norm(&[2.0, 2.0, 2.0, 2.0], &[2.0, 2.0, 2.0, 2.0], 1e-6).unwrap();
        assert_vec_close(&out, &[2.0, 2.0, 2.0, 2.0], 1e-5);
    }

    #[test]
    fn eps_is_a_parameter_not_a_constant() {
        // A large eps must visibly change the result: with eps = 4 the
        // divisor is sqrt(4 + 4) instead of sqrt(4), so the constant
        // vector [2,2,2,2] maps to 2/sqrt(8) = 0.70710678...
        let out = rms_norm(&[2.0, 2.0, 2.0, 2.0], &[1.0, 1.0, 1.0, 1.0], 4.0).unwrap();
        assert_close(out[0], std::f64::consts::FRAC_1_SQRT_2, 1e-6);
    }

    #[test]
    fn empty_input_is_an_error() {
        assert_eq!(
            rms_norm(&[], &[], 1e-6),
            Err(MathError::EmptyInput { op: "rms_norm" })
        );
    }

    #[test]
    fn weight_length_mismatch_is_an_error() {
        assert_eq!(
            rms_norm(&[1.0, 2.0], &[1.0], 1e-6),
            Err(MathError::LengthMismatch {
                op: "rms_norm",
                expected: 2,
                actual: 1
            })
        );
    }

    #[test]
    fn bad_eps_is_an_error() {
        for eps in [0.0, -1e-6, f32::NAN, f32::INFINITY] {
            assert!(
                matches!(
                    rms_norm(&[1.0], &[1.0], eps),
                    Err(MathError::InvalidConfig { param: "eps", .. })
                ),
                "eps {eps} must be rejected"
            );
        }
    }
}
