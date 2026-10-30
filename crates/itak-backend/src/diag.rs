//! Structured diagnostics for the backend crate.
//!
//! David's rule: debugging is built into every module from birth, not
//! bolted on after a failure. This module is the crate's diagnostic
//! channel, shaped like the `itak-diag` seam of ARCHITECTURE 3.1 and
//! mirroring the `gguf` crate's channel until that crate lands:
//!
//! - Enable with [`set_debug_enabled`] or the `ITAK_BACKEND_DEBUG`
//!   environment variable (checked on every [`debug_enabled`] call,
//!   so a test or a long-running process can flip it at runtime).
//! - Output is structured `key=value` lines on stderr, tagged
//!   `[itak-backend][debug]`, so logs can be grepped and parsed
//!   mechanically.
//! - Selection and fallback events (BACKEND.md Section 6) are emitted
//!   here with their stable codes (`backend.selected`,
//!   `backend.fallback_cpu`, `backend.op_fallback_cpu`,
//!   `backend.vram_disagreement`, `backend.unavailable`) AND returned
//!   in the [`crate::HardwareReport`], so a fallback is a logged,
//!   queryable state, never a silent one.

use std::sync::atomic::{AtomicBool, Ordering};

static DEBUG_ENABLED: AtomicBool = AtomicBool::new(false);

/// Turn structured debug logging on or off for the whole process.
pub fn set_debug_enabled(enabled: bool) {
    DEBUG_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Whether debug logging is active: the explicit flag, or
/// `ITAK_BACKEND_DEBUG` present in the environment (any value other
/// than `0`).
pub fn debug_enabled() -> bool {
    if DEBUG_ENABLED.load(Ordering::Relaxed) {
        return true;
    }
    match std::env::var("ITAK_BACKEND_DEBUG") {
        Ok(v) => v != "0" && !v.is_empty(),
        Err(_) => false,
    }
}

/// Emit one structured debug line if diagnostics are enabled.
///
/// `event` is a short snake_case or dotted event name; `fields` are
/// `key=value` fragments the caller has already formatted.
pub fn debug(event: &str, fields: &str) {
    if debug_enabled() {
        eprintln!("[itak-backend][debug] event={event} {fields}");
    }
}

/// Emit a structured line for a refusal or fallback, if diagnostics
/// are enabled. Fallbacks and refusals are the events the audit law
/// says must be observable, so selection code calls this at every
/// ladder transition before returning.
pub fn error(event: &str, fields: &str) {
    if debug_enabled() {
        eprintln!("[itak-backend][error] event={event} {fields}");
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
        // Env may still enable it in a developer shell; only assert
        // the flag store did not panic and the API is callable.
        let _ = debug_enabled();
    }
}
