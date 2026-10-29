//! FFN activation functions. The gated families split the FFN input
//! into a gate half and an up half, run the gate through a nonlinearity,
//! and multiply. Which nonlinearity is a per-family fact, not a global
//! choice: most families (Llama, Mistral, Qwen2, Phi-3, OLMo) use
//! SwiGLU, Gemma uses GEGLU. The old Go engine hardcoded SwiGLU for
//! every family, which made Gemma wrong by construction. Here they are
//! two separate, separately tested functions, and the caller picks.

use std::f32::consts::PI;

use crate::{dbg_log, MathError, Result};

/// SiLU (also called swish): `x * sigmoid(x)`, the gate activation of
/// SwiGLU. Numerically safe at both tails: for very negative `x` the
/// exponential overflows to infinity and the quotient tends to 0,
/// which is the correct limit.
#[must_use]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// GELU with the tanh approximation, the form Gemma and GPT-2 use:
/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
#[must_use]
pub fn gelu_tanh(x: f32) -> f32 {
    let inner = (2.0 / PI).sqrt() * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// SwiGLU: `silu(gate) * up`, elementwise over the two FFN halves.
///
/// # Errors
///
/// - [`MathError::LengthMismatch`] if `gate` and `up` differ in length.
pub fn swiglu(gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
    gated(gate, up, "swiglu", silu)
}

/// GEGLU: `gelu_tanh(gate) * up`, elementwise over the two FFN halves.
///
/// # Errors
///
/// - [`MathError::LengthMismatch`] if `gate` and `up` differ in length.
pub fn geglu(gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
    gated(gate, up, "geglu", gelu_tanh)
}

/// Shared elementwise gate-times-up driver for the gated activations.
fn gated(gate: &[f32], up: &[f32], op: &'static str, act: fn(f32) -> f32) -> Result<Vec<f32>> {
    if gate.len() != up.len() {
        return Err(MathError::LengthMismatch {
            op,
            expected: gate.len(),
            actual: up.len(),
        });
    }
    dbg_log!("{op} len={}", gate.len());
    Ok(gate
        .iter()
        .zip(up.iter())
        .map(|(g, u)| act(*g) * *u)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{assert_close, assert_vec_close};

    #[test]
    fn silu_spot_values() {
        assert_close(silu(0.0), 0.0, 1e-7);
        // silu(1) = sigmoid(1) = 0.7310585786300049...
        assert_close(silu(1.0), 0.731_058_578_630_004_9, 1e-6);
        // silu(-1) = -sigmoid(-1) = -0.2689414213699951...
        assert_close(silu(-1.0), -0.268_941_421_369_995_1, 1e-6);
    }

    #[test]
    fn silu_tails_behave() {
        // No NaN or infinity at the extremes; limits are 0 and x.
        assert_close(silu(-100.0), 0.0, 1e-6);
        assert_close(silu(100.0), 100.0, 1e-4);
    }

    #[test]
    fn gelu_tanh_spot_values() {
        assert_close(gelu_tanh(0.0), 0.0, 1e-7);
        // Published value of the tanh-approximation GELU at 1.
        assert_close(gelu_tanh(1.0), 0.841_191_990_608_276_8, 1e-4);
        assert_close(gelu_tanh(-1.0), -0.158_808_009_391_723_2, 1e-4);
    }

    #[test]
    fn swiglu_spot_values() {
        // silu(1) * 2 = 1.4621171572600098...
        let out = swiglu(&[1.0], &[2.0]).unwrap();
        assert_vec_close(&out, &[1.462_117_157_260_009_8], 1e-6);
        // A closed gate (very negative) passes nothing through.
        let out = swiglu(&[-100.0], &[5.0]).unwrap();
        assert_vec_close(&out, &[0.0], 1e-6);
    }

    #[test]
    fn geglu_spot_values() {
        // gelu_tanh(1) * 3 = 2.5235759718248304...
        let out = geglu(&[1.0], &[3.0]).unwrap();
        assert_vec_close(&out, &[2.523_575_971_824_830_4], 3e-4);
    }

    #[test]
    fn swiglu_and_geglu_actually_differ() {
        // The regression guard for the original bug: if one of these
        // ever silently becomes the other, Gemma breaks again.
        let gate = [0.5, 1.5, -0.25];
        let up = [2.0, -1.0, 4.0];
        let s = swiglu(&gate, &up).unwrap();
        let g = geglu(&gate, &up).unwrap();
        assert!(
            s.iter().zip(g.iter()).any(|(a, b)| (a - b).abs() > 1e-3),
            "SwiGLU and GEGLU must not be the same function"
        );
    }

    #[test]
    fn length_mismatch_is_an_error() {
        assert_eq!(
            swiglu(&[1.0, 2.0], &[1.0]),
            Err(MathError::LengthMismatch {
                op: "swiglu",
                expected: 2,
                actual: 1
            })
        );
        assert!(matches!(
            geglu(&[1.0], &[1.0, 2.0]),
            Err(MathError::LengthMismatch { op: "geglu", .. })
        ));
    }
}
