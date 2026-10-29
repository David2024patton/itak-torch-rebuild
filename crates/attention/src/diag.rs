//! Diagnostics for the attention crate.
//!
//! Debug logging behind a runtime flag (off by default), per the
//! standing rule that every module ships with a way to see what it
//! did and why it refused. Lines go to stderr with a fixed
//! `[attention::diag]` prefix so they can be grepped out of a server
//! log. For programmatic diagnosis without any logging,
//! [`crate::KvCache::stats`] returns the per-layer state directly.

use std::sync::atomic::{AtomicBool, Ordering};

static DEBUG_ENABLED: AtomicBool = AtomicBool::new(false);

/// Turn debug logging on or off at runtime.
pub fn set_debug_enabled(enabled: bool) {
    DEBUG_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Whether debug logging is currently on.
pub fn debug_enabled() -> bool {
    DEBUG_ENABLED.load(Ordering::Relaxed)
}

/// Emit one structured diagnostic line to stderr when debug is on.
///
/// Usage: `crate::diag!("append layer={} position={}", layer, pos)`.
#[macro_export]
macro_rules! diag {
    ($($arg:tt)*) => {
        if $crate::diag::debug_enabled() {
            ::std::eprintln!("[attention::diag] {}", ::std::format!($($arg)*));
        }
    };
}
