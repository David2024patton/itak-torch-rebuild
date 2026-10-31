//! Structured diagnostics for the GGUF loader.
//!
//! David's rule: debugging is built into every module from birth, not bolted
//! on after a failure. This module is the crate's diagnostic channel:
//!
//! - Enable the debug level with [`set_debug_enabled`] or the
//!   `GGUF_DEBUG` environment variable (checked on every
//!   [`debug_enabled`] call, so a test or a long-running process can
//!   flip it at runtime). The error level ([`error`]) is not gated:
//!   a refusal emits whether or not debug is on.
//! - Output is structured `key=value` lines on stderr, tagged
//!   `[gguf][debug]`, so logs can be grepped and parsed mechanically.
//! - Every parse and load event worth investigating (header values, tensor
//!   directory entries, dequantization runs, and every hard error) passes
//!   through here. Errors are logged at the point they are created, before
//!   they propagate, so the log shows the refusal and its reason even when
//!   the caller only prints the top-level error.

use std::sync::atomic::{AtomicBool, Ordering};

static DEBUG_ENABLED: AtomicBool = AtomicBool::new(false);

/// Turn structured debug logging on or off for the whole process.
pub fn set_debug_enabled(enabled: bool) {
    DEBUG_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Whether debug logging is active: the explicit flag, or `GGUF_DEBUG`
/// present in the environment (any value other than `0`).
pub fn debug_enabled() -> bool {
    if DEBUG_ENABLED.load(Ordering::Relaxed) {
        return true;
    }
    match std::env::var("GGUF_DEBUG") {
        Ok(v) => v != "0" && !v.is_empty(),
        Err(_) => false,
    }
}

/// Emit one structured debug line if diagnostics are enabled.
///
/// `event` is a short snake_case name; `fields` are `key=value` fragments the
/// caller has already formatted. Keeping formatting at the call site keeps
/// this channel allocation-light and dependency-free.
pub fn debug(event: &str, fields: &str) {
    if debug_enabled() {
        eprintln!("[gguf][debug] event={event} {fields}");
    }
}

/// Emit a structured line for a hard error, unconditionally.
/// Errors are the refusals the audit law says must be observable,
/// so unlike [`debug`] this is never gated by the debug flag: a
/// refusal that only shows when debug is on is a silent refusal.
/// Every `GgufError` construction inside the crate emits through
/// here exactly once, at the point the error is created (see
/// `GgufError::logged` in error.rs), before it propagates.
pub fn error(event: &str, fields: &str) {
    #[cfg(test)]
    capture(&format!("[gguf][error] event={event} {fields}"));
    eprintln!("[gguf][error] event={event} {fields}");
}

/// Test-only capture of emitted error lines, so unit tests can
/// prove a refusal actually fired its diag event instead of
/// trusting that the call is present in the source. Lines are the
/// exact formatted output; tests assert on their own event names
/// and field values, which are unique per test, so parallel tests
/// sharing this buffer cannot satisfy each other's assertions.
#[cfg(test)]
static CAPTURED: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

#[cfg(test)]
fn capture(line: &str) {
    CAPTURED
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .push(line.to_string());
}

/// The error lines captured since the process started (test builds
/// only). Callers compare a snapshot taken before the action under
/// test, or search for field values only their own test produces.
#[cfg(test)]
pub(crate) fn captured_errors() -> Vec<String> {
    CAPTURED.lock().unwrap_or_else(|p| p.into_inner()).clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_round_trips() {
        set_debug_enabled(true);
        assert!(debug_enabled());
        set_debug_enabled(false);
        // Env may still enable it in a developer shell; only assert the flag
        // store did not panic and the API is callable in both states.
        let _ = debug_enabled();
    }

    #[test]
    fn error_emits_with_debug_off() {
        // The documented contract: errors are unconditional. With
        // the flag off and regardless of env, the event must land
        // in the capture buffer (the same line stderr receives).
        set_debug_enabled(false);
        error("unconditional_probe", "probe=flag_off");
        assert!(captured_errors()
            .iter()
            .any(|l| l.contains("event=unconditional_probe") && l.contains("probe=flag_off")));
    }
}
