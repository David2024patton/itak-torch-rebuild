//! Rotary position embedding (RoPE). The old Go native engine had no
//! RoPE anywhere in its forward path, so position information never
//! entered the model at all. This module is the correction, written
//! against the conventions llama.cpp uses (the project's reference):
//!
//! - The head vector is treated as `head_dim / 2` adjacent pairs
//!   `(x[2i], x[2i + 1])` (the GPT-NeoX pairing llama.cpp applies).
//! - Pair `i` rotates by `position * base^(-2i / rotary_dim)`.
//! - `base` is a parameter: the classic value is `10000`, long-context
//!   models use more (Glimmer and Qwen2.5 use `500000`, Llama 3 uses
//!   `500000`, Mistral uses `1000000` in some releases). It is read
//!   from each model's config, never assumed.
//!
//! Partial rotary (only the first `rotary_dim` channels rotate, as in
//! some Phi and Gemma configurations) is supported by [`rope_partial`];
//! the remaining channels pass through unchanged. The frequency
//! denominator is `rotary_dim`, matching llama.cpp's partial-rotary
//! convention; for the full form [`rope`] the two are equal anyway.

use crate::{dbg_log, MathError, Result};

/// Apply RoPE to one head vector of length `head_dim` at `position`,
/// rotating every channel. Equivalent to [`rope_partial`] with
/// `rotary_dim == head_dim`.
///
/// # Errors
///
/// - [`MathError::LengthMismatch`] if `input.len() != head_dim`.
/// - [`MathError::InvalidConfig`] if `head_dim` is odd or less than 2,
///   or if `base` is not finite and greater than 1.
pub fn rope(input: &[f32], position: usize, base: f32, head_dim: usize) -> Result<Vec<f32>> {
    rope_partial(input, position, base, head_dim, head_dim)
}

/// Apply RoPE to the first `rotary_dim` channels of one head vector of
/// length `head_dim` at `position`; channels at or beyond `rotary_dim`
/// are copied through unchanged.
///
/// # Errors
///
/// - [`MathError::LengthMismatch`] if `input.len() != head_dim`.
/// - [`MathError::InvalidConfig`] if `head_dim` or `rotary_dim` is odd
///   or less than 2, if `rotary_dim > head_dim`, or if `base` is not
///   finite and greater than 1.
pub fn rope_partial(
    input: &[f32],
    position: usize,
    base: f32,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<Vec<f32>> {
    if head_dim < 2 || head_dim % 2 != 0 {
        return Err(MathError::InvalidConfig {
            op: "rope",
            param: "head_dim",
            reason: "head_dim must be even and >= 2",
        });
    }
    if input.len() != head_dim {
        return Err(MathError::LengthMismatch {
            op: "rope",
            expected: head_dim,
            actual: input.len(),
        });
    }
    if rotary_dim < 2 || rotary_dim % 2 != 0 || rotary_dim > head_dim {
        return Err(MathError::InvalidConfig {
            op: "rope",
            param: "rotary_dim",
            reason: "rotary_dim must be even, >= 2, and <= head_dim",
        });
    }
    if !base.is_finite() || base <= 1.0 {
        return Err(MathError::InvalidConfig {
            op: "rope",
            param: "base",
            reason: "base must be finite and > 1",
        });
    }
    dbg_log!("rope position={position} base={base} head_dim={head_dim} rotary_dim={rotary_dim}");
    let mut out = input.to_vec();
    // Angles are computed in f64: at long contexts the product
    // position * base^(-2i/d) loses real precision in f32, and the
    // reference implementation is the one that must not drift.
    let pos = position as f64;
    let base64 = f64::from(base);
    for (pair_idx, pair) in out[..rotary_dim].chunks_exact_mut(2).enumerate() {
        let exponent = -((2 * pair_idx) as f64) / (rotary_dim as f64);
        let angle = pos * base64.powf(exponent);
        let (sin, cos) = angle.sin_cos();
        if let [first, second] = pair {
            let (x, y) = (f64::from(*first), f64::from(*second));
            *first = (x * cos - y * sin) as f32;
            *second = (x * sin + y * cos) as f32;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{assert_close, assert_vec_close};

    #[test]
    fn position_zero_is_identity() {
        // Every angle is 0 at position 0, and rotation by 0 is exact.
        let input = [0.3, -1.2, 2.5, 0.7];
        let out = rope(&input, 0, 500_000.0, 4).unwrap();
        assert_eq!(out, input.to_vec());
    }

    #[test]
    fn known_rotations() {
        // head_dim 4, base 10000: pair 0 turns at rate 1, pair 1 at
        // rate 10000^(-1/2) = 0.01. At position 1 the unit vectors
        // [1,0] and [1,0] become (cos a, sin a) for their own a.
        let out = rope(&[1.0, 0.0, 1.0, 0.0], 1, 10_000.0, 4).unwrap();
        assert_vec_close(
            &out,
            &[
                0.540_302_305_868_139_8,   // cos(1)
                0.841_470_984_807_896_5,   // sin(1)
                0.999_950_000_416_665_3,   // cos(0.01)
                0.009_999_833_334_166_664, // sin(0.01)
            ],
            1e-5,
        );
    }

    #[test]
    fn rotation_preserves_norm() {
        // RoPE is an orthogonal transform per pair, so the vector's
        // sum of squares is invariant at any position and base.
        let input = [1.0, 2.0, 3.0, 4.0];
        let out = rope(&input, 7, 500_000.0, 4).unwrap();
        let before: f64 = input.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
        let after: f64 = out.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
        assert_close(after as f32, before, 0.01);
    }

    #[test]
    fn partial_rotary_leaves_the_tail_untouched() {
        let input = [1.0, 0.0, 9.0, -4.0];
        let out = rope_partial(&input, 3, 10_000.0, 4, 2).unwrap();
        // Channels 2 and 3 are beyond rotary_dim and pass through.
        assert_eq!(out[2], 9.0);
        assert_eq!(out[3], -4.0);
        // Channel pair 0 did rotate (angle 3 radians on [1, 0]).
        assert_close(out[0], -0.989_992_496_600_445_4, 1e-5); // cos(3)
        assert_close(out[1], 0.141_120_008_059_867_2, 1e-5); // sin(3)
    }

    #[test]
    fn base_is_a_parameter() {
        // With base 500000 and head_dim 4, pair 1 turns at rate
        // 500000^(-1/2) = 0.00141421..., far slower than base 10000.
        let out = rope(&[1.0, 0.0, 1.0, 0.0], 1, 500_000.0, 4).unwrap();
        assert_close(out[3], 0.001_414_213_5, 1e-6); // sin(500000^-0.5)
    }

    #[test]
    fn length_mismatch_is_an_error() {
        assert_eq!(
            rope(&[1.0, 0.0], 0, 10_000.0, 4),
            Err(MathError::LengthMismatch {
                op: "rope",
                expected: 4,
                actual: 2
            })
        );
    }

    #[test]
    fn bad_config_is_an_error() {
        // Odd head_dim.
        assert!(matches!(
            rope(&[1.0, 0.0, 1.0], 0, 10_000.0, 3),
            Err(MathError::InvalidConfig {
                param: "head_dim",
                ..
            })
        ));
        // rotary_dim beyond head_dim.
        assert!(matches!(
            rope_partial(&[1.0, 0.0, 1.0, 0.0], 0, 10_000.0, 4, 8),
            Err(MathError::InvalidConfig {
                param: "rotary_dim",
                ..
            })
        ));
        // Base of 1 would make every frequency 1.
        assert!(matches!(
            rope(&[1.0, 0.0], 0, 1.0, 2),
            Err(MathError::InvalidConfig { param: "base", .. })
        ));
    }
}
