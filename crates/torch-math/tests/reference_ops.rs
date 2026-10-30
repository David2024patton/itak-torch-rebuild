//! Independent reference tests for the T-2.2 kernel ops (TODO T-2.2,
//! second half): RMSNorm, RoPE, softmax.
//!
//! Every expected value in this file comes from one of three sources,
//! none of them the code under test:
//!
//! 1. A naive implementation written here, directly from the formula
//!    in each op's definition, in a deliberately different style from
//!    the crate (explicit loops, different accumulation order, and
//!    for RoPE a different route to the frequencies).
//! 2. Closed-form values computed inline from published constants.
//! 3. Mathematical invariants the op must satisfy regardless of
//!    implementation (norm preservation, shift invariance, sums to
//!    one).

use torch_math::{rms_norm, rope, rope_partial, softmax, MathError};

/// Deterministic pseudo-random values in [-2, 2) from a small LCG,
/// so the wide-vector tests need no external crates and never flake.
fn lcg_values(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 40) as f32) / ((1u64 << 24) as f32);
            unit * 4.0 - 2.0
        })
        .collect()
}

fn assert_vec_close(actual: &[f32], expected: &[f64], tol: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (f64::from(*a) - e).abs();
        assert!(
            diff <= tol,
            "{what}[{i}]: expected {e}, got {a} (diff {diff} exceeds tol {tol})"
        );
    }
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

/// Naive RMSNorm, written from the definition with an explicit loop
/// that accumulates in reverse order (the crate sums forward through
/// an iterator). Weight applied per element at the end.
fn naive_rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f64> {
    let mut sum_sq = 0.0f64;
    for i in (0..input.len()).rev() {
        let v = f64::from(input[i]);
        sum_sq += v * v;
    }
    let mean_sq = sum_sq / input.len() as f64;
    let denom = (mean_sq + f64::from(eps)).sqrt();
    input
        .iter()
        .zip(weight.iter())
        .map(|(v, w)| f64::from(*v) / denom * f64::from(*w))
        .collect()
}

#[test]
fn rms_norm_matches_naive_on_a_mixed_vector() {
    let input = [0.5, -1.25, 3.0, 0.0, -0.75, 2.25, 1.5, -2.0];
    let weight = [1.0, 0.5, 2.0, 1.5, -1.0, 0.25, 1.0, 3.0];
    let out = rms_norm(&input, &weight, 1e-5).unwrap();
    assert_vec_close(
        &out,
        &naive_rms_norm(&input, &weight, 1e-5),
        1e-6,
        "rms_norm",
    );
}

#[test]
fn rms_norm_hand_computed_case() {
    // [3, 4]: mean of squares is exactly 12.5, so with eps negligible
    // at this tolerance the outputs are 3/sqrt(12.5) and 4/sqrt(12.5)
    // = 0.6*sqrt(2) and 0.8*sqrt(2) (the 3-4-5 triangle, normalized).
    let out = rms_norm(&[3.0, 4.0], &[1.0, 1.0], 1e-6).unwrap();
    let sqrt2 = std::f64::consts::SQRT_2;
    assert_vec_close(&out, &[0.6 * sqrt2, 0.8 * sqrt2], 1e-5, "rms_norm 3-4-5");
}

#[test]
fn rms_norm_eps_really_comes_from_the_argument() {
    // The historical bug this crate exists to fix was a hardcoded
    // eps. Two calls with different eps must each match the naive
    // result for *that* eps, and must differ from each other by far
    // more than float noise when eps is large.
    let input = [1.0, 2.0, 3.0, 4.0];
    let weight = [1.0; 4];
    let small = rms_norm(&input, &weight, 1e-6).unwrap();
    let large = rms_norm(&input, &weight, 0.5).unwrap();
    assert_vec_close(
        &small,
        &naive_rms_norm(&input, &weight, 1e-6),
        1e-6,
        "eps 1e-6",
    );
    assert_vec_close(
        &large,
        &naive_rms_norm(&input, &weight, 0.5),
        1e-6,
        "eps 0.5",
    );
    assert!(
        (small[0] - large[0]).abs() > 0.01,
        "eps must change the result: {small:?} vs {large:?}"
    );
    // The two eps values the real families use, checked individually.
    for eps in [1e-5_f32, 1e-6] {
        let out = rms_norm(&input, &weight, eps).unwrap();
        assert_vec_close(
            &out,
            &naive_rms_norm(&input, &weight, eps),
            1e-6,
            "family eps",
        );
    }
}

#[test]
fn rms_norm_matches_naive_at_glimmer_width() {
    // 6656 is Glimmer's hidden size: wide enough that a sloppy
    // accumulation would drift away from the naive sum.
    let input = lcg_values(6656, 0x5eed_0001);
    let weight = lcg_values(6656, 0x5eed_0002)
        .iter()
        .map(|w| w * 0.25)
        .collect::<Vec<_>>();
    let out = rms_norm(&input, &weight, 1e-6).unwrap();
    assert_vec_close(
        &out,
        &naive_rms_norm(&input, &weight, 1e-6),
        1e-5,
        "rms_norm wide",
    );
}

#[test]
fn rms_norm_zero_vector_stays_zero() {
    let out = rms_norm(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0], 1e-6).unwrap();
    assert_eq!(out, vec![0.0, 0.0, 0.0]);
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------

/// Naive RoPE, written from the GGUF convention as specified in the
/// crate docs: adjacent pairs `(x[2i], x[2i+1])`, pair `i` rotated by
/// `position * theta^(-2i / rotary_dim)`. The frequency is computed
/// through `exp(-(2i/d) * ln(theta))` instead of `powf`, so the two
/// implementations reach the angle by different arithmetic and still
/// have to agree. Channels past `rotary_dim` pass through.
fn naive_rope(input: &[f32], position: usize, theta: f32, rotary_dim: usize) -> Vec<f64> {
    let mut out: Vec<f64> = input.iter().map(|v| f64::from(*v)).collect();
    let ln_theta = f64::from(theta).ln();
    let pos = position as f64;
    let mut i = 0usize;
    while 2 * i + 1 < rotary_dim {
        let frac = (2 * i) as f64 / rotary_dim as f64;
        let angle = pos * (-frac * ln_theta).exp();
        let (sin, cos) = angle.sin_cos();
        let (x, y) = (out[2 * i], out[2 * i + 1]);
        out[2 * i] = x * cos - y * sin;
        out[2 * i + 1] = x * sin + y * cos;
        i += 1;
    }
    out
}

#[test]
fn rope_position_zero_is_exact_identity_at_width() {
    let input = lcg_values(128, 0x5eed_0003);
    let out = rope(&input, 0, 500_000.0, 128).unwrap();
    assert_eq!(out, input, "position 0 must not move any channel");
}

#[test]
fn rope_position_one_matches_naive_gguf_theta() {
    // Glimmer's GGUF theta (llama.rope.freq_base) is 500000.
    let input = [0.25, -1.5, 2.0, 0.75, -0.5, 1.25, 0.0, 3.0];
    let out = rope(&input, 1, 500_000.0, 8).unwrap();
    assert_vec_close(
        &out,
        &naive_rope(&input, 1, 500_000.0, 8),
        1e-6,
        "rope pos 1",
    );
}

#[test]
fn rope_large_position_matches_naive_and_preserves_norm() {
    // Position 131072 is the far end of a 128k context: the pair-0
    // angle is 131072 radians, where a sloppy (f32) angle computation
    // visibly drifts. The naive route is all-f64, so agreement here
    // pins the crate's angle precision, not just its formula.
    let input = lcg_values(128, 0x5eed_0004);
    let out = rope(&input, 131_072, 500_000.0, 128).unwrap();
    assert_vec_close(
        &out,
        &naive_rope(&input, 131_072, 500_000.0, 128),
        1e-4,
        "rope pos 131072",
    );
    let before: f64 = input.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    let after: f64 = out.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    assert!(
        (after - before).abs() <= 1e-3,
        "rotation must preserve the norm: {before} -> {after}"
    );
}

#[test]
fn rope_classic_theta_at_mid_context() {
    let input = lcg_values(64, 0x5eed_0005);
    let out = rope(&input, 4096, 10_000.0, 64).unwrap();
    assert_vec_close(
        &out,
        &naive_rope(&input, 4096, 10_000.0, 64),
        1e-5,
        "rope 4096",
    );
}

#[test]
fn rope_partial_matches_naive_and_leaves_tail_exact() {
    let input = lcg_values(16, 0x5eed_0006);
    let out = rope_partial(&input, 777, 500_000.0, 16, 4).unwrap();
    assert_vec_close(
        &out,
        &naive_rope(&input, 777, 500_000.0, 4),
        1e-5,
        "rope_partial",
    );
    assert_eq!(&out[4..], &input[4..], "tail past rotary_dim is a copy");
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

/// Naive softmax straight from the definition, with NO maximum
/// subtraction: only safe on the small logits it is fed here, which
/// is exactly why the large-logit tests compare the crate against
/// this naive form evaluated on pre-shifted inputs.
fn naive_softmax(input: &[f32]) -> Vec<f64> {
    let mut exps = Vec::with_capacity(input.len());
    let mut sum = 0.0f64;
    for &v in input {
        let e = f64::from(v).exp();
        sum += e;
        exps.push(e);
    }
    exps.iter().map(|e| e / sum).collect()
}

/// Naive masked softmax: `-inf` entries are excluded from the
/// denominator by hand and assigned exactly zero.
fn naive_masked_softmax(input: &[f32]) -> Vec<f64> {
    let mut exps = Vec::with_capacity(input.len());
    let mut sum = 0.0f64;
    for &v in input {
        if v == f32::NEG_INFINITY {
            exps.push(0.0);
        } else {
            let e = f64::from(v).exp();
            sum += e;
            exps.push(e);
        }
    }
    exps.iter().map(|e| e / sum).collect()
}

#[test]
fn softmax_matches_naive_definition() {
    let input = [0.3, -1.2, 2.5, 0.7, -0.4];
    let out = softmax(&input).unwrap();
    assert_vec_close(&out, &naive_softmax(&input), 1e-6, "softmax");
}

#[test]
fn softmax_large_logits_match_naive_on_shifted_inputs() {
    // Shift invariance is the stability claim: softmax(x) must equal
    // softmax(x - c) even when exp(x) itself would overflow. The
    // expected side is the naive definition on the shifted row.
    let big = [1000.0, 1001.0, 999.0, 1000.5];
    let small = [-1.0, 0.0, -2.0, -0.5];
    let out = softmax(&big).unwrap();
    assert_vec_close(&out, &naive_softmax(&small), 1e-6, "softmax large logits");

    let big_neg = [-2000.0, -2001.0, -1999.0];
    let small_neg = [-1.0, -2.0, 0.0];
    let out = softmax(&big_neg).unwrap();
    assert_vec_close(
        &out,
        &naive_softmax(&small_neg),
        1e-6,
        "softmax large negative",
    );
}

#[test]
fn softmax_masked_rows_match_naive() {
    // The causal-mask shape: a visible prefix, then -inf.
    let row = [0.5, 1.5, f32::NEG_INFINITY, f32::NEG_INFINITY];
    let out = softmax(&row).unwrap();
    assert_vec_close(
        &out,
        &naive_masked_softmax(&row),
        1e-6,
        "softmax causal row",
    );
    assert_eq!(out[2], 0.0);
    assert_eq!(out[3], 0.0);

    // Masked everywhere but one position: all mass lands there.
    let row = [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY, 2.0];
    let out = softmax(&row).unwrap();
    assert_eq!(out, vec![0.0, 0.0, 0.0, 1.0]);

    // Interleaved mask entries among large logits: stability and
    // masking at the same time.
    let row = [1001.0, f32::NEG_INFINITY, 1000.0];
    let out = softmax(&row).unwrap();
    let expected = naive_masked_softmax(&[1.0, f32::NEG_INFINITY, 0.0]);
    assert_vec_close(&out, &expected, 1e-6, "softmax large + masked");
    assert_eq!(out[1], 0.0);
}

#[test]
fn softmax_all_masked_row_is_a_named_error() {
    let row = [f32::NEG_INFINITY; 4];
    assert!(
        matches!(
            softmax(&row),
            Err(MathError::InvalidConfig { param: "input", .. })
        ),
        "an all-masked row has no distribution and must not return zeros or NaNs"
    );
}

#[test]
fn softmax_wide_row_sums_to_one_and_matches_naive() {
    // 257 arbitrary logits (a vocab-row shape in miniature): the
    // naive f64 definition is still safe at this scale, and the
    // probabilities must tile to exactly one distribution.
    let input = lcg_values(257, 0x5eed_0007)
        .iter()
        .map(|v| v * 3.0)
        .collect::<Vec<_>>();
    let out = softmax(&input).unwrap();
    assert_vec_close(&out, &naive_softmax(&input), 1e-6, "softmax wide");
    let sum: f64 = out.iter().map(|p| f64::from(*p)).sum();
    assert!((sum - 1.0).abs() <= 1e-6, "probabilities sum to {sum}");
    assert!(out.iter().all(|p| *p >= 0.0), "no negative probabilities");
}
