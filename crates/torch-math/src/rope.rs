//! Rotary position embedding (RoPE). The old Go native engine had no
//! RoPE anywhere in its forward path, so position information never
//! entered the model at all. This module is the correction, written
//! against the conventions llama.cpp uses (the project's reference).
//!
//! ggml has two pairing conventions, and they are NOT interchangeable:
//! a model file's Q/K weights are stored for exactly one of them.
//!
//! - [`rope`] / [`rope_partial`] are ggml's `GGML_ROPE_TYPE_NORMAL`
//!   (value 0): the head vector is treated as `head_dim / 2` adjacent
//!   pairs `(x[2i], x[2i + 1])` (interleaved). Files served this way
//!   had their Q/K weights permuted at conversion time.
//! - [`rope_neox`] / [`rope_neox_partial`] are ggml's
//!   `GGML_ROPE_TYPE_NEOX` (value 2): pair `i` is the split-half pair
//!   `(x[i], x[i + rotary_dim / 2])`. This is the pairing llama.cpp
//!   assigns to the HF-derived architecture families (llama, qwen2,
//!   gemma, and the rest served from unpermuted HF weights), Muse
//!   Glimmer included: its file is an HF conversion and llama.cpp's
//!   graph for it rotates split-half. The Glimmer forward path uses
//!   this variant.
//!
//! Both share the frequency ladder: pair `i` rotates by
//! `position * base^(-2i / rotary_dim)`, and both rotate in the same
//! direction (`x' = x cos - y sin`, `y' = x sin + y cos`).
//!
//! `base` is a parameter: the classic value is `10000`, long-context
//! models use more (Glimmer and Qwen2.5 use `500000`, Llama 3 uses
//! `500000`, Mistral uses `1000000` in some releases). It is read
//! from each model's config, never assumed.
//!
//! Partial rotary (only the first `rotary_dim` channels rotate, as in
//! some Phi and Gemma configurations) is supported by the `_partial`
//! functions; the remaining channels pass through unchanged. The
//! frequency denominator is `rotary_dim`, matching ggml's convention
//! (its `n_dims` argument is the rotary count); for the full forms
//! the two are equal anyway.

use crate::{dbg_log, MathError, Result};

fn validate(input_len: usize, head_dim: usize, rotary_dim: usize, base: f32) -> Result<()> {
    if head_dim < 2 || head_dim % 2 != 0 {
        return Err(MathError::InvalidConfig {
            op: "rope",
            param: "head_dim",
            reason: "head_dim must be even and >= 2",
        });
    }
    if input_len != head_dim {
        return Err(MathError::LengthMismatch {
            op: "rope",
            expected: head_dim,
            actual: input_len,
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
    Ok(())
}

/// Rotation angle for pair `pair_idx` at `position`, in f64: at long
/// contexts the product position * base^(-2i/d) loses real precision
/// in f32, and the reference implementation is the one that must not
/// drift.
fn angle(position: usize, base: f32, rotary_dim: usize, pair_idx: usize) -> f64 {
    let exponent = -((2 * pair_idx) as f64) / (rotary_dim as f64);
    (position as f64) * f64::from(base).powf(exponent)
}

fn rotate_pair(x: f32, y: f32, angle: f64) -> (f32, f32) {
    let (sin, cos) = angle.sin_cos();
    let (x, y) = (f64::from(x), f64::from(y));
    (
        (x * cos - y * sin) as f32,
        (x * sin + y * cos) as f32,
    )
}

/// Apply NORMAL (interleaved) RoPE to one head vector of length
/// `head_dim` at `position`, rotating every channel: pair `i` is
/// `(x[2i], x[2i + 1])` (ggml `GGML_ROPE_TYPE_NORMAL`). Equivalent to
/// [`rope_partial`] with `rotary_dim == head_dim`.
///
/// # Errors
///
/// - [`MathError::LengthMismatch`] if `input.len() != head_dim`.
/// - [`MathError::InvalidConfig`] if `head_dim` is odd or less than 2,
///   or if `base` is not finite and greater than 1.
pub fn rope(input: &[f32], position: usize, base: f32, head_dim: usize) -> Result<Vec<f32>> {
    rope_partial(input, position, base, head_dim, head_dim)
}

/// Apply NORMAL (interleaved) RoPE to the first `rotary_dim` channels
/// of one head vector of length `head_dim` at `position`; channels at
/// or beyond `rotary_dim` are copied through unchanged.
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
    validate(input.len(), head_dim, rotary_dim, base)?;
    dbg_log!("rope position={position} base={base} head_dim={head_dim} rotary_dim={rotary_dim}");
    let mut out = input.to_vec();
    for pair_idx in 0..rotary_dim / 2 {
        let a = angle(position, base, rotary_dim, pair_idx);
        let (x, y) = rotate_pair(out[2 * pair_idx], out[2 * pair_idx + 1], a);
        out[2 * pair_idx] = x;
        out[2 * pair_idx + 1] = y;
    }
    Ok(out)
}

/// Apply NEOX (split-half) RoPE to one head vector of length
/// `head_dim` at `position`, rotating every channel: pair `i` is
/// `(x[i], x[i + head_dim / 2])` (ggml `GGML_ROPE_TYPE_NEOX`, the
/// pairing llama.cpp applies to Muse Glimmer). Equivalent to
/// [`rope_neox_partial`] with `rotary_dim == head_dim`.
///
/// # Errors
///
/// Same contract as [`rope`].
pub fn rope_neox(input: &[f32], position: usize, base: f32, head_dim: usize) -> Result<Vec<f32>> {
    rope_neox_partial(input, position, base, head_dim, head_dim)
}

/// Apply NEOX (split-half) RoPE to the first `rotary_dim` channels of
/// one head vector of length `head_dim` at `position`: pair `i` is
/// `(x[i], x[i + rotary_dim / 2])`; channels at or beyond `rotary_dim`
/// are copied through unchanged.
///
/// # Errors
///
/// Same contract as [`rope_partial`].
pub fn rope_neox_partial(
    input: &[f32],
    position: usize,
    base: f32,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<Vec<f32>> {
    validate(input.len(), head_dim, rotary_dim, base)?;
    dbg_log!(
        "rope_neox position={position} base={base} head_dim={head_dim} rotary_dim={rotary_dim}"
    );
    let mut out = input.to_vec();
    let half = rotary_dim / 2;
    for pair_idx in 0..half {
        let a = angle(position, base, rotary_dim, pair_idx);
        let (x, y) = rotate_pair(out[pair_idx], out[pair_idx + half], a);
        out[pair_idx] = x;
        out[pair_idx + half] = y;
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

    // ---- NEOX (split-half) convention: ggml GGML_ROPE_TYPE_NEOX,
    // the pairing llama.cpp applies to Muse Glimmer. Every expected
    // value below is computed from the split-half definition by hand
    // (literals), not by calling the code under test.

    #[test]
    fn neox_position_zero_is_identity() {
        let input = [0.3, -1.2, 2.5, 0.7];
        let out = rope_neox(&input, 0, 500_000.0, 4).unwrap();
        assert_eq!(out, input.to_vec());
    }

    #[test]
    fn neox_pairs_channels_across_the_halves() {
        // head_dim 4, base 10000, position 1, x = [1, 0, 1, 0].
        // Split-half pairs: pair 0 = (x[0], x[2]) = (1, 1) at rate 1,
        // pair 1 = (x[1], x[3]) = (0, 0) at rate 0.01.
        // Pair 0 rotates to (cos1 - sin1, sin1 + cos1) written back
        // into channels 0 and 2.
        let out = rope_neox(&[1.0, 0.0, 1.0, 0.0], 1, 10_000.0, 4).unwrap();
        assert_vec_close(
            &out,
            &[
                -0.301_168_678_939_756_7, // cos(1) - sin(1)
                0.0,
                1.381_773_290_676_036_3, // sin(1) + cos(1)
                0.0,
            ],
            1e-5,
        );
        // The interleaved convention on the same input gives
        // [cos1, sin1, cos0.01, sin0.01]: the two conventions are
        // observably different functions, so this test cannot pass
        // against the wrong pairing.
        let interleaved = rope(&[1.0, 0.0, 1.0, 0.0], 1, 10_000.0, 4).unwrap();
        assert!((f64::from(out[0]) - f64::from(interleaved[0])).abs() > 0.5);
        assert!((f64::from(out[2]) - f64::from(interleaved[2])).abs() > 0.3);
    }

    #[test]
    fn neox_second_half_channel_rotates_with_pair_zero() {
        // x = e_2 (channel 2) is the SECOND element of split-half
        // pair 0, so at position 1 with base 10000 it lands at
        // (-sin1, 0, cos1, 0). Under the interleaved convention
        // channel 2 is pair 1's first element and would land near
        // (0, 0, cos0.01, sin0.01) instead.
        let out = rope_neox(&[0.0, 0.0, 1.0, 0.0], 1, 10_000.0, 4).unwrap();
        assert_vec_close(
            &out,
            &[
                -0.841_470_984_807_896_5, // -sin(1)
                0.0,
                0.540_302_305_868_139_8, // cos(1)
                0.0,
            ],
            1e-5,
        );
    }

    #[test]
    fn neox_partial_rotary_pairs_inside_the_rotary_span_only() {
        // head_dim 8, rotary_dim 4, base 10000, position 2.
        // Pairs inside the span: (x[0], x[2]) at rate 1 (angle 2) and
        // (x[1], x[3]) at rate 10000^(-1/2) = 0.01 (angle 0.02).
        // Channels 4..8 pass through.
        let input = [1.0, 1.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0];
        let out = rope_neox_partial(&input, 2, 10_000.0, 8, 4).unwrap();
        assert_vec_close(
            &out,
            &[
                -0.416_146_836_547_142_4, // cos(2) on x[0]=1, y=0
                0.999_800_006_666_577_8,  // cos(0.02) on x[1]=1, y=0
                0.909_297_426_825_681_7,  // sin(2)
                0.019_998_666_693_333_08, // sin(0.02)
                5.0,
                6.0,
                7.0,
                8.0,
            ],
            1e-5,
        );
    }

    #[test]
    fn neox_glimmer_scale_frequency_ladder() {
        // Glimmer's real shape: head_dim 128, base 500000 (from the
        // file's muse-glimmer.rope.freq_base, pinned in muse-glimmer's
        // config tests). Two anchors on the ladder:
        //
        // Pair 0 = (channel 0, channel 64) at rate 1: e_64 at
        // position 3 gives (-sin3, cos3) in channels 0 and 64.
        let mut input = vec![0.0f32; 128];
        input[64] = 1.0;
        let out = rope_neox(&input, 3, 500_000.0, 128).unwrap();
        assert_close(out[0], -0.141_120_008_059_867_2, 1e-5); // -sin(3)
        assert_close(out[64], -0.989_992_496_600_445_4, 1e-5); // cos(3)
        //
        // Pair 63 = (channel 63, channel 127) at rate
        // 500000^(-126/128) = 2.4551408e-6: e_63 at position 1000
        // turns by 0.0024551408 rad.
        let mut input = vec![0.0f32; 128];
        input[63] = 1.0;
        let out = rope_neox(&input, 1000, 500_000.0, 128).unwrap();
        assert_close(out[63], 0.999_996_986_143_361_7, 1e-5); // cos(0.0024551408)
        assert_close(out[127], 0.002_455_138_324_650_322_6, 1e-6); // sin(0.0024551408)
        // Every other channel is exactly zero: no pair leaks.
        let leaked: f32 = out
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != 63 && *i != 127)
            .map(|(_, v)| v.abs())
            .sum();
        assert_eq!(leaked, 0.0);
    }

    #[test]
    fn neox_dot_product_depends_only_on_relative_position() {
        // The decode-time property the forward relies on: for any
        // offset d, <rope(q, n + d), rope(k, n)> == <rope(q, d),
        // rope(k, 0)>. If position handling dropped the offset (the
        // run-B freeze class), this identity would still hold
        // trivially at d = 0 only; here it is checked at n = 5,
        // d = 4 against positions 4 and 0, in f64 from the outputs.
        let q: Vec<f32> = (0..8).map(|i| (i as f32) * 0.37 - 1.1).collect();
        let k: Vec<f32> = (0..8).map(|i| 0.9 - (i as f32) * 0.21).collect();
        let dot = |a: &[f32], b: &[f32]| -> f64 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| f64::from(*x) * f64::from(*y))
                .sum()
        };
        let lhs = dot(
            &rope_neox(&q, 9, 500_000.0, 8).unwrap(),
            &rope_neox(&k, 5, 500_000.0, 8).unwrap(),
        );
        let rhs = dot(
            &rope_neox(&q, 4, 500_000.0, 8).unwrap(),
            &rope_neox(&k, 0, 500_000.0, 8).unwrap(),
        );
        assert!(
            (lhs - rhs).abs() < 1e-4,
            "relative-position identity: {lhs} vs {rhs}"
        );
        // And rotation must actually move with absolute position: the
        // same pair at the same RELATIVE offset but computed from a
        // zero base position is the rhs above; a position-blind
        // implementation would also make dot(q, k) itself match.
        let unrotated = dot(&q, &k);
        assert!((lhs - unrotated).abs() > 1e-3);
    }

    #[test]
    fn neox_validation_matches_the_interleaved_contract() {
        assert_eq!(
            rope_neox(&[1.0, 0.0], 0, 10_000.0, 4),
            Err(MathError::LengthMismatch {
                op: "rope",
                expected: 4,
                actual: 2
            })
        );
        assert!(matches!(
            rope_neox_partial(&[1.0, 0.0, 1.0, 0.0], 0, 10_000.0, 4, 8),
            Err(MathError::InvalidConfig {
                param: "rotary_dim",
                ..
            })
        ));
    }
}
