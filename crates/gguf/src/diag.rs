//! Structured diagnostics for the GGUF loader.
//!
//! David's rule: debugging is built into every module from birth, not bolted
//! on after a failure. This module is the crate's diagnostic channel:
//!
//! - Enable with [`set_debug_enabled`] or the `GGUF_DEBUG` environment
//!   variable (checked on every [`debug_enabled`] call, so a test or a
//!   long-running process can flip it at runtime).
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

/// Emit a structured line for a hard error, unconditionally at debug level.
/// Errors are the refusals the audit law says must be observable, so this is
/// called wherever a `GgufError` is constructed inside the crate.
pub fn error(event: &str, fields: &str) {
    if debug_enabled() {
        eprintln!("[gguf][error] event={event} {fields}");
    }
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
}
