//! Debug logging seam for this crate, mirroring `torch-math`'s:
//! structured lines to stderr behind a flag, off by default. This is
//! the crate's built-in way to pull its bug information out (David's
//! rule: debugging built into every module from birth).
//!
//! Enable with [`set_debug`] or the `CPU_COMPUTE_DEBUG` environment
//! variable (checked on every [`debug_enabled`] call, the gguf
//! shape, so a shipped binary can turn the seam on with no code
//! change).

use std::sync::atomic::{AtomicBool, Ordering};

static DEBUG_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable or disable debug logging to stderr. When on, the backend
/// logs KV cache construction, per-call attention positions, and
/// resets under a `[cpu-compute]` prefix.
pub fn set_debug(enabled: bool) {
    DEBUG_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Whether debug logging is currently enabled: the explicit flag,
/// or `CPU_COMPUTE_DEBUG` present in the environment (any value other than
/// `0` or empty). See [`set_debug`].
#[must_use]
pub fn debug_enabled() -> bool {
    if DEBUG_ENABLED.load(Ordering::Relaxed) {
        return true;
    }
    match std::env::var("CPU_COMPUTE_DEBUG") {
        Ok(v) => v != "0" && !v.is_empty(),
        Err(_) => false,
    }
}

/// Emit one already-formatted diagnostic line: the `dbg_log`
/// macro's single output path, so test builds capture exactly what
/// stderr receives and can prove the seam fired.
pub fn emit_debug(formatted: &str) {
    #[cfg(test)]
    capture(&format!("[cpu-compute] {formatted}"));
    eprintln!("[cpu-compute] {formatted}");
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
        if $crate::diag::debug_enabled() {
            $crate::diag::emit_debug(&::std::format!($($arg)*));
        }
    };
}
pub(crate) use dbg_log;

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
        std::env::remove_var("CPU_COMPUTE_DEBUG");
        assert!(!debug_enabled());
        std::env::set_var("CPU_COMPUTE_DEBUG", "1");
        assert!(
            debug_enabled(),
            "CPU_COMPUTE_DEBUG=1 must enable debug (the gguf env-twin contract)"
        );
        std::env::set_var("CPU_COMPUTE_DEBUG", "0");
        assert!(
            !debug_enabled(),
            "CPU_COMPUTE_DEBUG=0 must not enable debug"
        );
        std::env::set_var("CPU_COMPUTE_DEBUG", "");
        assert!(
            !debug_enabled(),
            "an empty CPU_COMPUTE_DEBUG must not enable debug"
        );
        std::env::remove_var("CPU_COMPUTE_DEBUG");
        assert!(!debug_enabled());
    }
    #[test]
    fn seam_fires_when_on_and_is_silent_when_off() {
        let _guard = DIAG_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        std::env::remove_var("CPU_COMPUTE_DEBUG");
        set_debug(false);
        crate::diag::dbg_log!("silent probe cpu_compute_debug_off_probe");
        assert!(
            !captured_lines()
                .iter()
                .any(|l| l.contains("cpu_compute_debug_off_probe")),
            "the seam must stay silent while debug is off"
        );
        set_debug(true);
        crate::diag::dbg_log!("firing probe cpu_compute_debug_on_probe");
        set_debug(false);
        assert!(
            captured_lines()
                .iter()
                .any(|l| l.contains("[cpu-compute]") && l.contains("cpu_compute_debug_on_probe")),
            "the seam must emit its prefixed line while debug is on"
        );
    }
}
