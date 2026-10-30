//! Debug logging seam for this crate, mirroring `cpu-compute`'s:
//! structured lines to stderr behind a flag, off by default. This is
//! the crate's built-in way to pull its bug information out (David's
//! rule: debugging built into every module from birth).

use std::sync::atomic::{AtomicBool, Ordering};

static DEBUG_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable or disable debug logging to stderr. When on, the backend
/// logs adapter enumeration and selection, tensor uploads, and
/// dispatches under a `[wgpu-compute]` prefix.
pub fn set_debug(enabled: bool) {
    DEBUG_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Whether debug logging is currently enabled. See [`set_debug`].
#[must_use]
pub fn debug_enabled() -> bool {
    DEBUG_ENABLED.load(Ordering::Relaxed)
}

/// Internal logging macro: silent unless the debug flag is on.
macro_rules! dbg_log {
    ($($arg:tt)*) => {
        if $crate::diag::debug_enabled() {
            eprintln!("[wgpu-compute] {}", format!($($arg)*));
        }
    };
}
pub(crate) use dbg_log;
