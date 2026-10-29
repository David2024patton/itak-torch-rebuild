//! Output-head logit shaping: softcapping and the final scale.
//!
//! Glimmer's head applies both: logits are softcapped at `20.0`
//! (`cap * tanh(x / cap)`, the Gemma-2 technique that bounds logits
//! smoothly instead of clipping them) and multiplied by a final scale
//! of `0.1961`. Neither value belongs in this code: both are
//! parameters, read from the model's own configuration by the caller.
//! The old engine had no final-norm-to-head path at all, so neither
//! operation existed in it.

use crate::{dbg_log, MathError, Result};

/// Softcap logits elementwise: `cap * tanh(x / cap)`.
///
/// The result is bounded in magnitude by `cap`, passes through 0, and
/// is an odd function (`softcap(-x) == -softcap(x)`), so it compresses
/// large logits smoothly without the flat gradient of a hard clip.
/// (In `f32`, very large inputs saturate to exactly `+-cap`, because
/// `tanh` rounds to exactly 1.)
///
/// An empty input yields an empty output.
///
/// # Errors
///
/// - [`MathError::InvalidConfig`] if `cap` is not finite and greater than zero.
pub fn logit_softcap(logits: &[f32], cap: f32) -> Result<Vec<f32>> {
    if !cap.is_finite() || cap <= 0.0 {
        return Err(MathError::InvalidConfig {
            op: "logit_softcap",
            param: "cap",
            reason: "cap must be finite and > 0",
        });
    }
    dbg_log!("logit_softcap len={} cap={cap}", logits.len());
    Ok(logits.iter().map(|v| cap * (v / cap).tanh()).collect())
}

/// Multiply every logit by `scale`, the final head scale applied after
/// softcapping in Glimmer-style heads.
///
/// An empty input yields an empty output.
///
/// # Errors
///
/// - [`MathError::InvalidConfig`] if `scale` is not finite.
pub fn logit_scale(logits: &[f32], scale: f32) -> Result<Vec<f32>> {
    if !scale.is_finite() {
        return Err(MathError::InvalidConfig {
            op: "logit_scale",
            param: "scale",
            reason: "scale must be finite",
        });
    }
    dbg_log!("logit_scale len={} scale={scale}", logits.len());
    Ok(logits.iter().map(|v| v * scale).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{assert_close, assert_vec_close};

    #[test]
    fn softcap_known_value() {
        // At x == cap the result is cap * tanh(1) = cap * 0.7615941559557649.
        let out = logit_softcap(&[20.0], 20.0).unwrap();
        assert_close(out[0], 15.231_883_119_915_298, 1e-4);
    }

    #[test]
    fn softcap_is_odd_and_passes_through_zero() {
        let out = logit_softcap(&[0.0, 5.0, -5.0], 20.0).unwrap();
        assert_close(out[0], 0.0, 1e-7);
        assert_close(out[1] + out[2], 0.0, 1e-6);
    }

    #[test]
    fn softcap_is_bounded() {
        // No input, however large, exceeds the cap. In f32 the tanh
        // saturates, so a huge input lands exactly on the cap rather
        // than approaching it asymptotically; that is the correct
        // floating-point behavior and this test pins it down.
        let out = logit_softcap(&[1.0e6, -1.0e6], 20.0).unwrap();
        assert!(out[0] <= 20.0 && out[0] > 19.999, "got {}", out[0]);
        assert!(out[1] >= -20.0 && out[1] < -19.999, "got {}", out[1]);
    }

    #[test]
    fn softcap_is_a_parameter() {
        // Same input, different cap, different result.
        let wide = logit_softcap(&[10.0], 20.0).unwrap();
        let tight = logit_softcap(&[10.0], 5.0).unwrap();
        assert_close(wide[0], 9.242_343_145_200_195, 1e-4); // 20 * tanh(0.5)
        assert_close(tight[0], 4.820_137_788_460_383, 1e-4); // 5 * tanh(2)
    }

    #[test]
    fn softcap_bad_cap_is_an_error() {
        for cap in [0.0, -20.0, f32::NAN, f32::INFINITY] {
            assert!(
                matches!(
                    logit_softcap(&[1.0], cap),
                    Err(MathError::InvalidConfig { param: "cap", .. })
                ),
                "cap {cap} must be rejected"
            );
        }
    }

    #[test]
    fn scale_multiplies() {
        // Glimmer's own scale, applied to a known vector.
        let out = logit_scale(&[1.0, -2.0, 0.5], 0.1961).unwrap();
        assert_vec_close(&out, &[0.1961, -0.3922, 0.098_05], 1e-6);
    }

    #[test]
    fn scale_bad_value_is_an_error() {
        for scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                matches!(
                    logit_scale(&[1.0], scale),
                    Err(MathError::InvalidConfig { param: "scale", .. })
                ),
                "scale {scale} must be rejected"
            );
        }
    }

    #[test]
    fn empty_logits_are_allowed() {
        assert_eq!(logit_softcap(&[], 20.0).unwrap(), Vec::<f32>::new());
        assert_eq!(logit_scale(&[], 0.1961).unwrap(), Vec::<f32>::new());
    }
}
