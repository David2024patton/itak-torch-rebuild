//! `torch-math`: reference `f32` math primitives for the iTaK Torch engine.
//!
//! This crate is the math the audit of the old Go native engine demanded
//! (see `~/workspace/model-teardown/audit-native-engine.md`): the forward
//! there had no RoPE at all, no final norm before the LM head, an RMSNorm
//! epsilon hardcoded to `1e-6`, and the FFN activation hardcoded to
//! SwiGLU for every family. Here, every one of those choices is a
//! parameter: `eps`, the RoPE base, the softcap, and the final logit
//! scale are arguments, never constants, and SwiGLU and GEGLU are
//! separate functions because different families use different ones.
//!
//! Design rules:
//!
//! - `f32` reference implementations. Clarity over speed; SIMD and GPU
//!   kernels come later behind these same signatures and are verified
//!   against this crate.
//! - Accumulation (norm sums, dot products) is done in `f64` so the
//!   reference is the most accurate version of each formula; values
//!   are stored as `f32`.
//! - No panics on shape mismatch or bad configuration. Every fallible
//!   function returns [`Result`] with a named [`MathError`]. (The old
//!   engine swallowed load errors and silently kept random weights;
//!   this crate's contract is the opposite: misuse is loud and named.)
//! - Dependency-free on purpose: the reference crate stays auditable.
//!
//! Debug logging: every public function logs its shapes and config to
//! stderr when the flag set by [`set_debug`] is on. Off by default.

#![forbid(unsafe_code)]
// The module carries its own gates (David's rule: linting built in
// from birth), independent of the workspace manifest's lint settings.
#![deny(warnings)]
#![deny(clippy::all)]

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

pub mod activation;
pub mod linalg;
pub mod logits;
pub mod norm;
pub mod rope;
pub mod rope_interleaved;

pub use activation::{geglu, gelu_tanh, silu, swiglu};
pub use linalg::{matmul, matvec};
pub use logits::{logit_scale, logit_softcap};
pub use norm::rms_norm;
pub use rope::{rope, rope_partial};
pub use rope_interleaved::{rope_interleaved, rope_interleaved_partial};

/// Result type for `torch-math` operations.
pub type Result<T> = std::result::Result<T, MathError>;

/// Errors returned by `torch-math`. Shape and configuration problems
/// are reported here, never by panicking and never by silently
/// substituting anything.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MathError {
    /// An input that must be non-empty was empty.
    EmptyInput {
        /// Name of the operation that rejected the input.
        op: &'static str,
    },
    /// Two slices that must have equal length did not.
    LengthMismatch {
        /// Name of the operation that rejected the input.
        op: &'static str,
        /// The length the operation required.
        expected: usize,
        /// The length it was given.
        actual: usize,
    },
    /// A flat buffer did not match its declared `rows * cols` shape.
    ShapeMismatch {
        /// Name of the operation that rejected the input.
        op: &'static str,
        /// Element count implied by the declared shape.
        expected: usize,
        /// Element count of the buffer actually passed.
        actual: usize,
    },
    /// The inner dimensions of a matmul did not agree (`a_cols != b_rows`).
    DimensionMismatch {
        /// Name of the operation that rejected the input.
        op: &'static str,
        /// Columns of the left operand.
        a_cols: usize,
        /// Rows of the right operand.
        b_rows: usize,
    },
    /// A configuration parameter was out of range.
    InvalidConfig {
        /// Name of the operation that rejected the configuration.
        op: &'static str,
        /// Name of the offending parameter.
        param: &'static str,
        /// Why the value is invalid.
        reason: &'static str,
    },
}

impl fmt::Display for MathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput { op } => write!(f, "{op}: input must not be empty"),
            Self::LengthMismatch {
                op,
                expected,
                actual,
            } => write!(
                f,
                "{op}: length mismatch, expected {expected}, got {actual}"
            ),
            Self::ShapeMismatch {
                op,
                expected,
                actual,
            } => write!(
                f,
                "{op}: buffer holds {actual} elements, shape requires {expected}"
            ),
            Self::DimensionMismatch { op, a_cols, b_rows } => write!(
                f,
                "{op}: inner dimensions disagree, a_cols={a_cols} but b_rows={b_rows}"
            ),
            Self::InvalidConfig { op, param, reason } => {
                write!(f, "{op}: invalid {param}: {reason}")
            }
        }
    }
}

impl std::error::Error for MathError {}

static DEBUG_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable or disable structured debug logging to stderr.
///
/// When on, every public function logs its operation name, shapes, and
/// configuration values under a `[torch-math]` prefix. Off by default;
/// this is the crate's diagnostic seam, per the project rule that every
/// module ships with a way to pull its bug information out. The
/// `TORCH_MATH_DEBUG` environment variable is the flag's env twin
/// (checked on every [`debug_enabled`] call, the gguf shape), so a
/// shipped binary can turn the seam on with no code change.
pub fn set_debug(enabled: bool) {
    DEBUG_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Whether debug logging is currently enabled: the explicit flag,
/// or `TORCH_MATH_DEBUG` present in the environment (any value
/// other than `0` or empty). See [`set_debug`].
#[must_use]
pub fn debug_enabled() -> bool {
    if DEBUG_ENABLED.load(Ordering::Relaxed) {
        return true;
    }
    match std::env::var("TORCH_MATH_DEBUG") {
        Ok(v) => v != "0" && !v.is_empty(),
        Err(_) => false,
    }
}

/// Emit one already-formatted diagnostic line: the `dbg_log`
/// macro's single output path, so test builds capture exactly what
/// stderr receives and can prove the seam fired.
pub(crate) fn emit_debug(formatted: &str) {
    #[cfg(test)]
    capture(&format!("[torch-math] {formatted}"));
    eprintln!("[torch-math] {formatted}");
}

/// Test-only capture of emitted debug lines, so unit tests can prove
/// the seam actually fired instead of trusting that the macro call is
/// present in the source (the gguf channel's shape). Lines are the
/// exact formatted output; tests assert on their own probe strings,
/// which are unique per test, so parallel tests sharing this buffer
/// cannot satisfy each other's assertions.
#[cfg(test)]
static CAPTURED: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

#[cfg(test)]
fn capture(line: &str) {
    CAPTURED
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .push(line.to_string());
}

/// The debug lines captured since the process started (test builds
/// only).
#[cfg(test)]
pub(crate) fn captured_lines() -> Vec<String> {
    CAPTURED.lock().unwrap_or_else(|p| p.into_inner()).clone()
}

/// Internal logging macro: silent unless the debug flag is on.
macro_rules! dbg_log {
    ($($arg:tt)*) => {
        if $crate::debug_enabled() {
            $crate::emit_debug(&::std::format!($($arg)*));
        }
    };
}
pub(crate) use dbg_log;

#[cfg(test)]
pub(crate) mod test_util {
    /// Assert two floats agree within an absolute tolerance. Expected
    /// values in this crate's tests are `f64` constants computed by
    /// hand or taken from published values, so the comparison happens
    /// in `f64` against the `f32` result.
    pub fn assert_close(actual: f32, expected: f64, tol: f64) {
        let diff = (f64::from(actual) - expected).abs();
        assert!(
            diff <= tol,
            "expected {expected}, got {actual} (diff {diff} exceeds tol {tol})"
        );
    }

    /// [`assert_close`] elementwise over a slice.
    pub fn assert_vec_close(actual: &[f32], expected: &[f64], tol: f64) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: got {}, expected {}",
            actual.len(),
            expected.len()
        );
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_close(*a, *e, tol);
        }
    }
}

#[cfg(test)]
mod diag_tests {
    use super::*;

    /// Serializes the diag tests: the flag and the env var are
    /// process-global, so two tests flipping them in parallel would
    /// satisfy (or break) each other's assertions.
    static DIAG_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn env_twin_enables_debug() {
        let _guard = DIAG_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        set_debug(false);
        std::env::remove_var("TORCH_MATH_DEBUG");
        assert!(!debug_enabled());
        std::env::set_var("TORCH_MATH_DEBUG", "1");
        assert!(
            debug_enabled(),
            "TORCH_MATH_DEBUG=1 must enable debug (the gguf env-twin contract)"
        );
        std::env::set_var("TORCH_MATH_DEBUG", "0");
        assert!(!debug_enabled(), "TORCH_MATH_DEBUG=0 must not enable debug");
        std::env::set_var("TORCH_MATH_DEBUG", "");
        assert!(
            !debug_enabled(),
            "an empty TORCH_MATH_DEBUG must not enable debug"
        );
        std::env::remove_var("TORCH_MATH_DEBUG");
        assert!(!debug_enabled());
    }

    #[test]
    fn seam_fires_when_on_and_is_silent_when_off() {
        let _guard = DIAG_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        std::env::remove_var("TORCH_MATH_DEBUG");
        set_debug(false);
        crate::dbg_log!("silent probe torch_math_off_probe");
        assert!(
            !captured_lines()
                .iter()
                .any(|l| l.contains("torch_math_off_probe")),
            "the seam must stay silent while debug is off"
        );
        set_debug(true);
        crate::dbg_log!("firing probe torch_math_on_probe");
        set_debug(false);
        assert!(
            captured_lines()
                .iter()
                .any(|l| l.contains("[torch-math]") && l.contains("torch_math_on_probe")),
            "the seam must emit its prefixed line while debug is on"
        );
    }
}
