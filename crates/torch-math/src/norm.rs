//! RMSNorm: the normalization every dissected family uses in place of
//! LayerNorm. The one parameter the old Go engine got wrong by
//! hardcoding is the point of this module: `eps` is an argument.
//! Real models differ (Llama uses `1e-5`, Qwen2 and Gemma use `1e-6`),
//! so it is never a constant here.
//!
//! Two forms live here, and the weight convention is deliberate: the
//! weighted form multiplies by the stored weight as-is (a GGUF file
//! stores effective gains; checkpoint-side shifts like Gemma's
//! `(1 + w)` are baked into the stored tensor at conversion time),
//! and the scaleless form has no weight at all, for the places an
//! architecture normalizes without a learned gain (Muse Glimmer's
//! token embeddings). Tests below pin both readings against
//! independently computed reference vectors.

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
    let inv_rms = inv_rms(input, eps, "rms_norm")?;
    Ok(input
        .iter()
        .zip(weight.iter())
        .map(|(v, w)| (f64::from(*v) * inv_rms * f64::from(*w)) as f32)
        .collect())
}

/// The shared reciprocal-RMS factor of both norm forms:
/// `1 / sqrt(mean(input^2) + eps)`, accumulated in `f64`. `op` names
/// the calling form in refusals. Callers that already validated may
/// validate twice; the checks are cheap next to the norm itself.
fn inv_rms(input: &[f32], eps: f32, op: &'static str) -> Result<f64> {
    if input.is_empty() {
        return Err(MathError::EmptyInput { op });
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(MathError::InvalidConfig {
            op,
            param: "eps",
            reason: "eps must be finite and > 0",
        });
    }
    let len = input.len() as f64;
    let mean_sq = input
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        / len;
    Ok(1.0 / (mean_sq + f64::from(eps)).sqrt())
}

/// Parameter-free RMSNorm: `out[i] = input[i] / sqrt(mean(input^2) + eps)`,
/// with no weight at all.
///
/// Some architectures normalize with no learned gain in specific
/// places: Muse Glimmer RMS-normalizes its token embeddings exactly
/// this way, and its Q/K norms are scaleless in the checkpoint
/// definition (the GGUF converter materializes the Q-side scale
/// factor as a constant weight tensor instead; see the QK test
/// below). This function exists so a caller that needs the scaleless
/// form never fakes it with a synthesized ones tensor.
///
/// # Errors
///
/// - [`MathError::EmptyInput`] if `input` is empty.
/// - [`MathError::InvalidConfig`] if `eps` is not finite and greater than zero.
pub fn rms_norm_scaleless(input: &[f32], eps: f32) -> Result<Vec<f32>> {
    dbg_log!("rms_norm_scaleless len={} eps={eps}", input.len());
    let inv_rms = inv_rms(input, eps, "rms_norm_scaleless")?;
    Ok(input
        .iter()
        .map(|v| (f64::from(*v) * inv_rms) as f32)
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

    #[test]
    fn stored_weight_is_the_effective_gain_not_an_offset() {
        // CONVENTION PINNED (norm audit, 2026-10-29): the weight a
        // GGUF file stores is the effective gain, applied as-is. The
        // Gemma-family checkpoint form stores an offset and computes
        // with (1 + w); for GGUF files the converter bakes that shift
        // into the stored tensor, and llama.cpp's graphs multiply the
        // stored weight plainly. Reference vectors below were computed
        // independently (f64, from the formula) for a mixed-sign
        // weight, under which the two readings disagree in sign as
        // well as size, so this test cannot pass under the shifted
        // reading:
        //   plain:   [0.182574, -0.730296, 2.190889, 0.365148]
        //   shifted: [0.547722,  0.0,      3.286333, 1.825741]
        let out = rms_norm(&[1.0, 2.0, 3.0, 4.0], &[0.5, -1.0, 2.0, 0.25], 1e-5).unwrap();
        assert_vec_close(&out, &[0.182574, -0.730296, 2.190889, 0.365148], 1e-5);
    }

    #[test]
    fn scaleless_known_vector() {
        // Same input and eps as `known_vector`, no weight: the
        // outputs are the normalized values themselves
        // (1/sqrt(7.5 + 1e-5) = 0.365148...).
        let out = rms_norm_scaleless(&[1.0, 2.0, 3.0, 4.0], 1e-5).unwrap();
        assert_vec_close(&out, &[0.365148, 0.730296, 1.095444, 1.460593], 1e-5);
    }

    #[test]
    fn scaleless_constant_vector_normalizes_to_ones() {
        let out = rms_norm_scaleless(&[2.0, 2.0, 2.0, 2.0], 1e-6).unwrap();
        assert_vec_close(&out, &[1.0, 1.0, 1.0, 1.0], 1e-5);
    }

    #[test]
    fn scaleless_eps_dominates_tiny_inputs() {
        // mean(x^2) = 2.5e-6, below eps 1e-5, so the divisor is
        // sqrt(1.25e-5) and the outputs stay small instead of being
        // blown up to unit RMS: [0.282843, -0.565685] (independent
        // f64 reference). This is the regime where the eps VALUE is
        // load-bearing, which is why eps is never hardcoded.
        let out = rms_norm_scaleless(&[0.001, -0.002], 1e-5).unwrap();
        assert_vec_close(&out, &[0.282843, -0.565685], 1e-5);
    }

    #[test]
    fn scaleless_errors_name_the_scaleless_op() {
        assert_eq!(
            rms_norm_scaleless(&[], 1e-6),
            Err(MathError::EmptyInput {
                op: "rms_norm_scaleless"
            })
        );
        assert!(matches!(
            rms_norm_scaleless(&[1.0], 0.0),
            Err(MathError::InvalidConfig {
                op: "rms_norm_scaleless",
                param: "eps",
                ..
            })
        ));
    }

    #[test]
    fn constant_weight_reproduces_scaleless_times_the_constant() {
        // QK PIN (norm audit, 2026-10-29): Muse Glimmer's checkpoint
        // definition norms Q and K scalelessly, then multiplies Q by
        // the config's qk_scale_factor (3.87 on the released
        // checkpoint). The GGUF file materializes that definition as
        // constant weight tensors: blk.*.attn_q_norm is exactly 3.87
        // in every entry and attn_k_norm exactly 1.0 (measured in
        // integration run B). Under this module's plain form, a
        // constant weight c gives exactly c times the scaleless
        // norm, so the stored tensors reproduce the checkpoint
        // definition with no special case anywhere. The (1 + w)
        // reading would give 4.87 times instead and is excluded by
        // the second assertion. Independent f64 reference for
        // x = [0.5, -1.5, 2.0, 0.25], eps 1e-5:
        //   scaleless:     [0.390359, -1.171077, 1.561435, 0.195179]
        //   times 3.87:    [1.510689, -4.532066, 6.042755, 0.755344]
        let x = [0.5, -1.5, 2.0, 0.25];
        let scaleless = rms_norm_scaleless(&x, 1e-5).unwrap();
        assert_vec_close(&scaleless, &[0.390359, -1.171077, 1.561435, 0.195179], 1e-5);
        let weighted = rms_norm(&x, &[3.87; 4], 1e-5).unwrap();
        assert_vec_close(&weighted, &[1.510689, -4.532066, 6.042755, 0.755344], 1e-5);
        for (w, s) in weighted.iter().zip(scaleless.iter()) {
            assert_close(*w, f64::from(*s) * 3.87, 1e-5);
        }
    }
}
