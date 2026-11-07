//! Diagnostic seam (David's debugging law): every refusal and load
//! event in this crate emits a structured line when debug is on.
//!
//! Runtime flag, the same shape as the sibling crates: enable with
//! [`set_debug`] or the `TOKENIZER_DEBUG` environment variable
//! (checked on every [`debug_enabled`] call, so a shipped binary
//! can turn the seam on with no code change). This seam was a
//! `debug-log` cargo feature until 2026-11-07; no dependent enabled
//! it, so every shipped build compiled the macro to nothing. The
//! feature is gone and the runtime flag is the only switch.

use std::sync::atomic::{AtomicBool, Ordering};

static DEBUG_ENABLED: AtomicBool = AtomicBool::new(false);

/// Turn diagnostic logging on or off at runtime.
pub fn set_debug(enabled: bool) {
    DEBUG_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Whether diagnostic logging is currently on: the explicit flag,
/// or `TOKENIZER_DEBUG` present in the environment (any value
/// other than `0` or empty).
pub fn debug_enabled() -> bool {
    if DEBUG_ENABLED.load(Ordering::Relaxed) {
        return true;
    }
    match std::env::var("TOKENIZER_DEBUG") {
        Ok(v) => v != "0" && !v.is_empty(),
        Err(_) => false,
    }
}

/// Emit one already-formatted diagnostic line: the `dbglog`
/// macro's single output path, so test builds capture exactly what
/// stderr receives and can prove the seam fired.
pub fn emit_debug(formatted: &str) {
    #[cfg(test)]
    capture(&format!("[tokenizer] {formatted}"));
    eprintln!("[tokenizer] {formatted}");
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

/// Internal logging macro: silent unless debug is on, and the
/// format arguments are never evaluated while it is off.
macro_rules! dbglog {
    ($($arg:tt)*) => {
        if $crate::diag::debug_enabled() {
            $crate::diag::emit_debug(&::std::format!($($arg)*));
        }
    };
}

pub(crate) use dbglog;

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes this crate's diag tests: the flag and the env var
    /// are process-global, so two tests flipping them in parallel
    /// would satisfy (or break) each other's assertions.
    static DIAG_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn env_twin_enables_debug() {
        let _guard = DIAG_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        set_debug(false);
        std::env::remove_var("TOKENIZER_DEBUG");
        assert!(!debug_enabled());
        std::env::set_var("TOKENIZER_DEBUG", "1");
        assert!(
            debug_enabled(),
            "TOKENIZER_DEBUG=1 must enable debug (the gguf env-twin contract)"
        );
        std::env::set_var("TOKENIZER_DEBUG", "0");
        assert!(!debug_enabled(), "TOKENIZER_DEBUG=0 must not enable debug");
        std::env::set_var("TOKENIZER_DEBUG", "");
        assert!(
            !debug_enabled(),
            "an empty TOKENIZER_DEBUG must not enable debug"
        );
        std::env::remove_var("TOKENIZER_DEBUG");
        assert!(!debug_enabled());
    }
    #[test]
    fn seam_fires_when_on_and_is_silent_when_off() {
        let _guard = DIAG_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        std::env::remove_var("TOKENIZER_DEBUG");
        set_debug(false);
        crate::diag::dbglog!("silent probe tokenizer_off_probe");
        assert!(
            !captured_lines()
                .iter()
                .any(|l| l.contains("tokenizer_off_probe")),
            "the seam must stay silent while debug is off"
        );
        set_debug(true);
        crate::diag::dbglog!("firing probe tokenizer_on_probe");
        set_debug(false);
        assert!(
            captured_lines()
                .iter()
                .any(|l| l.contains("[tokenizer]") && l.contains("tokenizer_on_probe")),
            "the seam must emit its prefixed line while debug is on"
        );
    }

    #[test]
    fn a_real_refusal_emits_through_the_seam() {
        // The seam on a production path, not just the macro: building
        // a tokenizer for an unsupported family must refuse AND say
        // so on the channel while debug is on.
        let _guard = DIAG_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        std::env::remove_var("TOKENIZER_DEBUG");
        set_debug(true);
        let data = crate::TokenizerData {
            model: Some("no-such-family".to_string()),
            ..Default::default()
        };
        let refused = crate::Tokenizer::new(data);
        set_debug(false);
        assert!(refused.is_err());
        assert!(
            captured_lines()
                .iter()
                .any(|l| l.contains("refusing unsupported tokenizer model")),
            "the refusal path must emit its diag line while debug is on"
        );
    }
}
