//! Structured diagnostics for the loader bridge.
//!
//! David's rule: debugging is built into every module from birth, not
//! bolted on after a failure. This module is the crate's diagnostic
//! channel, mirroring the `gguf` crate's channel so engine logs read
//! the same across crates:
//!
//! - Enable with [`set_debug_enabled`] or the `ITAK_LOADER_DEBUG`
//!   environment variable (checked on every [`debug_enabled`] call, so
//!   a test or a long-running process can flip it at runtime).
//! - Output is structured `key=value` lines on stderr, tagged
//!   `[itak-loader][debug]`, so logs can be grepped and parsed
//!   mechanically.
//! - The events worth investigating pass through here: config loads,
//!   coverage verdicts, tensors served, and every coercion the metadata
//!   adapter had to refuse (a key that is present but holds the wrong
//!   kind of value is exactly the fact a debugging session needs, and
//!   the `Metadata` trait's `Option` return has no room to carry it).

use std::sync::atomic::{AtomicBool, Ordering};

static DEBUG_ENABLED: AtomicBool = AtomicBool::new(false);

/// Turn structured debug logging on or off for the whole process.
pub fn set_debug_enabled(enabled: bool) {
    DEBUG_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Whether debug logging is active: the explicit flag, or
/// `ITAK_LOADER_DEBUG` present in the environment (any value other
/// than `0`).
pub fn debug_enabled() -> bool {
    if DEBUG_ENABLED.load(Ordering::Relaxed) {
        return true;
    }
    match std::env::var("ITAK_LOADER_DEBUG") {
        Ok(v) => v != "0" && !v.is_empty(),
        Err(_) => false,
    }
}

/// Emit one structured debug line if diagnostics are enabled.
///
/// `event` is a short snake_case name; `fields` are `key=value`
/// fragments the caller has already formatted. Keeping formatting at
/// the call site keeps this channel allocation-light and
/// dependency-free.
pub fn debug(event: &str, fields: &str) {
    if debug_enabled() {
        eprintln!("[itak-loader][debug] event={event} {fields}");
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
        // Env may still enable it in a developer shell; only assert the
        // flag store did not panic and the API is callable in both
        // states.
        let _ = debug_enabled();
    }
}
