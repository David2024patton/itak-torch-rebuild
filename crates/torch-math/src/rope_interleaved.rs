//! RoPE with interleaved (adjacent) pairing, kept in its own module
//! because it is a DIFFERENT convention from [`crate::rope`]'s
//! half-split pairing, not a variant of it:
//!
//! - Pair `i` joins channels `(2i, 2i + 1)` and rotates by
//!   `position * base^(-2i / head_dim)`.
//! - This is ggml's `GGML_ROPE_TYPE_NORM`, the mode llama.cpp
//!   selects for muse-glimmer (`llama_model_rope_type` in
//!   src/llama-model.cpp) and GPT-J-style models. Glimmer's
//!   converter unpermutes the Q/K projections into exactly this
//!   layout at conversion time (conversion/muse_glimmer.py:
//!   `permute` with `qk_order = NORM`), so the stored weights
//!   assume it.
//! - The pairings are not interchangeable: they pair different
//!   channels at the same rates, and a model converted for one is
//!   garbage under the other. The Run B follow-up (2026-10-29)
//!   found the Glimmer backend one edit away from the half-split
//!   form; giving each pairing its own named function is the
//!   structural fix, so a backend states its architecture's
//!   pairing at the call site.
//!
//! Partial rotary (only the first `rotary_dim` channels rotate, as
//! in llama-architecture files whose `rope.dimension_count` is
//! below the head dimension) is supported by
//! [`rope_interleaved_partial`]; the remaining channels pass
//! through unchanged. The frequency denominator is `rotary_dim`,
//! the same partial-rotary convention [`crate::rope_partial`]
//! documents for the half-split form; for the full form
//! [`rope_interleaved`] the two are equal anyway.

use crate::{dbg_log, MathError, Result};

/// Apply RoPE with interleaved (adjacent) pairing to one head vector
/// of length `head_dim` at `position`: pair `i` joins channels
/// `(2i, 2i + 1)` and rotates by `position * base^(-2i / head_dim)`.
/// Equivalent to [`rope_interleaved_partial`] with
/// `rotary_dim == head_dim`.
///
/// # Errors
///
/// - [`MathError::LengthMismatch`] if `input.len() != head_dim`.
/// - [`MathError::InvalidConfig`] if `head_dim` is odd or less than 2,
///   or if `base` is not finite and greater than 1.
pub fn rope_interleaved(
    input: &[f32],
    position: usize,
    base: f32,
    head_dim: usize,
) -> Result<Vec<f32>> {
    rope_interleaved_partial(input, position, base, head_dim, head_dim)
}

/// Apply RoPE with interleaved (adjacent) pairing to the first
/// `rotary_dim` channels of one head vector of length `head_dim` at
/// `position`; channels at or beyond `rotary_dim` are copied through
/// unchanged. Pair `i` joins channels `(2i, 2i + 1)` and rotates by
/// `position * base^(-2i / rotary_dim)`.
///
/// # Errors
///
/// - [`MathError::LengthMismatch`] if `input.len() != head_dim`.
/// - [`MathError::InvalidConfig`] if `head_dim` or `rotary_dim` is
///   odd or less than 2, if `rotary_dim > head_dim`, or if `base`
///   is not finite and greater than 1.
pub fn rope_interleaved_partial(
    input: &[f32],
    position: usize,
    base: f32,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<Vec<f32>> {
    if head_dim < 2 || head_dim % 2 != 0 {
        return Err(MathError::InvalidConfig {
            op: "rope_interleaved",
            param: "head_dim",
            reason: "head_dim must be even and >= 2",
        });
    }
    if input.len() != head_dim {
        return Err(MathError::LengthMismatch {
            op: "rope_interleaved",
            expected: head_dim,
            actual: input.len(),
        });
    }
    if rotary_dim < 2 || rotary_dim % 2 != 0 || rotary_dim > head_dim {
        return Err(MathError::InvalidConfig {
            op: "rope_interleaved",
            param: "rotary_dim",
            reason: "rotary_dim must be even, >= 2, and <= head_dim",
        });
    }
    if !base.is_finite() || base <= 1.0 {
        return Err(MathError::InvalidConfig {
            op: "rope_interleaved",
            param: "base",
            reason: "base must be finite and > 1",
        });
    }
    dbg_log!(
        "rope_interleaved_partial position={position} base={base} head_dim={head_dim} rotary_dim={rotary_dim}"
    );
    let mut out = input.to_vec();
    // Angles in f64, as in the half-split form: at long contexts the
    // product position * base^(-2i/d) loses real precision in f32.
    let pos = position as f64;
    let base64 = f64::from(base);
    for i in 0..rotary_dim / 2 {
        let exponent = -((2 * i) as f64) / (rotary_dim as f64);
        let angle = pos * base64.powf(exponent);
        let (sin, cos) = angle.sin_cos();
        let (x, y) = (f64::from(input[2 * i]), f64::from(input[2 * i + 1]));
        out[2 * i] = (x * cos - y * sin) as f32;
        out[2 * i + 1] = (x * sin + y * cos) as f32;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{assert_close, assert_vec_close};

    #[test]
    fn known_rotations() {
        // head_dim 4, base 10000, adjacent pairing: pair 0 joins
        // channels (0, 1) and turns at rate 1, pair 1 joins (2, 3)
        // at rate 10000^(-1/2) = 0.01. At position 1 the input
        // [1, 0, 1, 0] puts (1, 0) in both pairs, so the output is
        // (cos 1, sin 1, cos 0.01, sin 0.01). The half-split form on
        // the same input gives (-0.301, 0, 1.382, 0) instead, so
        // these values tell the pairings apart.
        let input = [1.0, 0.0, 1.0, 0.0];
        let out = rope_interleaved(&input, 1, 10_000.0, 4).unwrap();
        assert_vec_close(
            &out,
            &[
                0.540_302_305_868_139_8,   // cos(1), channel 0
                0.841_470_984_807_896_5,   // sin(1), channel 1
                0.999_950_000_416_665_3,   // cos(0.01), channel 2
                0.009_999_833_334_166_664, // sin(0.01), channel 3
            ],
            1e-5,
        );
    }

    #[test]
    fn rotation_preserves_norm_and_zero_position() {
        let input = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            rope_interleaved(&input, 0, 500_000.0, 4).unwrap(),
            input.to_vec()
        );
        let out = rope_interleaved(&input, 7, 500_000.0, 4).unwrap();
        let before: f64 = input.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
        let after: f64 = out.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
        assert_close(after as f32, before, 0.01);
    }

    #[test]
    fn partial_rotary_rotates_adjacent_pairs_and_passes_the_tail() {
        // head_dim 8, rotary_dim 4, base 10000, position 1: adjacent
        // pair 0 joins channels (0, 1) at rate 1, pair 1 joins
        // (2, 3) at rate 10000^(-2/4) = 0.01, and channels 4..8 are
        // beyond rotary_dim and pass through. The half-split
        // partial on the same input pairs (0, 2) and (1, 3)
        // instead, turning channel 0 with (cos 1 - sin 1), so
        // these values tell the pairings apart in the partial case
        // as well.
        let input = [1.0, 0.0, 1.0, 0.0, 9.0, -4.0, 7.0, 3.0];
        let out = rope_interleaved_partial(&input, 1, 10_000.0, 8, 4).unwrap();
        assert_vec_close(
            &out,
            &[
                0.540_302_305_868_139_8,   // cos(1), channel 0
                0.841_470_984_807_896_5,   // sin(1), channel 1
                0.999_950_000_416_665_3,   // cos(0.01), channel 2
                0.009_999_833_334_166_664, // sin(0.01), channel 3
                9.0,                       // tail, untouched
                -4.0,
                7.0,
                3.0,
            ],
            1e-5,
        );
        // The half-split partial form on the same input must NOT
        // produce this: pairings differ, not just names.
        let neox = crate::rope_partial(&input, 1, 10_000.0, 8, 4).unwrap();
        assert!(
            out.iter().zip(&neox).any(|(a, b)| (a - b).abs() > 0.1),
            "interleaved and half-split partials must differ on this input"
        );
    }

    #[test]
    fn partial_with_full_rotary_dim_equals_the_full_form() {
        let input = [0.3, -1.2, 2.5, 0.7, 1.1, -0.4, 0.9, 2.2];
        let partial = rope_interleaved_partial(&input, 5, 500_000.0, 8, 8).unwrap();
        let full = rope_interleaved(&input, 5, 500_000.0, 8).unwrap();
        assert_eq!(partial, full);
    }

    #[test]
    fn bad_config_is_an_error() {
        assert_eq!(
            rope_interleaved(&[1.0, 0.0], 0, 10_000.0, 4),
            Err(MathError::LengthMismatch {
                op: "rope_interleaved",
                expected: 4,
                actual: 2
            })
        );
        // Partial form refuses an impossible rotary_dim by name.
        assert!(matches!(
            rope_interleaved_partial(&[1.0, 0.0, 1.0, 0.0], 0, 10_000.0, 4, 8),
            Err(MathError::InvalidConfig {
                param: "rotary_dim",
                ..
            })
        ));
        assert!(matches!(
            rope_interleaved_partial(&[1.0, 0.0, 1.0, 0.0], 0, 10_000.0, 4, 3),
            Err(MathError::InvalidConfig {
                param: "rotary_dim",
                ..
            })
        ));
        assert!(matches!(
            rope_interleaved(&[1.0, 0.0, 1.0], 0, 10_000.0, 3),
            Err(MathError::InvalidConfig {
                param: "head_dim",
                ..
            })
        ));
        assert!(matches!(
            rope_interleaved(&[1.0, 0.0], 0, 1.0, 2),
            Err(MathError::InvalidConfig { param: "base", .. })
        ));
    }
}
