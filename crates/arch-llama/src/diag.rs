//! Diagnostic channel for the arch-llama module.
//!
//! Project law: every module ships with diagnostics from birth, and
//! every refusal emits through the diagnostic channel with its reason.
//! This crate keeps the channel a trait so the engine's diag crate can
//! own the real sink later; the two sinks here are the silent default
//! and a stderr sink for local work. Same shape as the muse-glimmer
//! module's channel, so arch modules are interchangeable at the seam.

/// A sink for structured diagnostic events: an event name plus a short
/// detail string. Implementations must be cheap to call on hot paths or
/// document that they are not.
pub trait Diag: Send + Sync {
    /// Record one event.
    fn event(&self, event: &str, detail: &str);
}

/// The default sink: drops every event. Use it when the engine process
/// has not wired a real sink yet; errors still propagate as values, so
/// nothing fails silently even with this sink installed.
#[derive(Debug, Default)]
pub struct NullDiag;

impl Diag for NullDiag {
    fn event(&self, _event: &str, _detail: &str) {}
}

/// A sink that writes one line per event to stderr, for local runs and
/// test debugging.
#[derive(Debug, Default)]
pub struct StderrDiag;

impl Diag for StderrDiag {
    fn event(&self, event: &str, detail: &str) {
        eprintln!("[arch-llama] {event}: {detail}");
    }
}
