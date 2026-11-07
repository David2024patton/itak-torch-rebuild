//! Diagnostics for the itak-kv crate.
//!
//! Debug logging behind a runtime flag (off by default), per the
//! standing rule that every module ships with a way to see what it
//! did and why it refused. Enable with [`set_debug_enabled`] or the
//! `ITAK_KV_DEBUG` environment variable (checked on every
//! [`debug_enabled`] call, the gguf shape, so a shipped binary can
//! turn the seam on with no code change). Lines go to stderr with a
//! fixed `[itak-kv::diag]` prefix so they can be grepped out of a
//! server log. For programmatic diagnosis without any logging,
//! [`crate::PrefixStore::stats`] returns the store's state directly.

use std::sync::atomic::{AtomicBool, Ordering};

static DEBUG_ENABLED: AtomicBool = AtomicBool::new(false);

/// Turn debug logging on or off at runtime.
pub fn set_debug_enabled(enabled: bool) {
    DEBUG_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Whether debug logging is currently on: the explicit flag, or
/// `ITAK_KV_DEBUG` present in the environment (any value other
/// than `0` or empty).
pub fn debug_enabled() -> bool {
    if DEBUG_ENABLED.load(Ordering::Relaxed) {
        return true;
    }
    match std::env::var("ITAK_KV_DEBUG") {
        Ok(v) => v != "0" && !v.is_empty(),
        Err(_) => false,
    }
}

/// Emit one already-formatted diagnostic line: the [`crate::diag`]
/// macro's single output path, so test builds capture exactly what
/// stderr receives and can prove the seam fired.
pub fn emit_debug(formatted: &str) {
    #[cfg(test)]
    capture(&format!("[itak-kv::diag] {formatted}"));
    eprintln!("[itak-kv::diag] {formatted}");
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

/// Emit one structured diagnostic line to stderr when debug is on.
///
/// Usage: `crate::diag!("put tokens={} bytes={}", n, bytes)`.
#[macro_export]
macro_rules! diag {
    ($($arg:tt)*) => {
        if $crate::diag::debug_enabled() {
            $crate::diag::emit_debug(&::std::format!($($arg)*));
        }
    };
}

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
        set_debug_enabled(false);
        std::env::remove_var("ITAK_KV_DEBUG");
        assert!(!debug_enabled());
        std::env::set_var("ITAK_KV_DEBUG", "1");
        assert!(
            debug_enabled(),
            "ITAK_KV_DEBUG=1 must enable debug (the gguf env-twin contract)"
        );
        std::env::set_var("ITAK_KV_DEBUG", "0");
        assert!(!debug_enabled(), "ITAK_KV_DEBUG=0 must not enable debug");
        std::env::set_var("ITAK_KV_DEBUG", "");
        assert!(
            !debug_enabled(),
            "an empty ITAK_KV_DEBUG must not enable debug"
        );
        std::env::remove_var("ITAK_KV_DEBUG");
        assert!(!debug_enabled());
    }
    #[test]
    fn seam_fires_when_on_and_is_silent_when_off() {
        let _guard = DIAG_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        std::env::remove_var("ITAK_KV_DEBUG");
        set_debug_enabled(false);
        crate::diag!("silent probe itak_kv_off_probe");
        assert!(
            !captured_lines()
                .iter()
                .any(|l| l.contains("itak_kv_off_probe")),
            "the seam must stay silent while debug is off"
        );
        set_debug_enabled(true);
        crate::diag!("firing probe itak_kv_on_probe");
        set_debug_enabled(false);
        assert!(
            captured_lines()
                .iter()
                .any(|l| l.contains("[itak-kv::diag]") && l.contains("itak_kv_on_probe")),
            "the seam must emit its prefixed line while debug is on"
        );
    }
}
