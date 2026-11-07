//! Diagnostic channel for the dflash module.
//!
//! Project law: every module ships with diagnostics from birth, and
//! every refusal emits through the diagnostic channel with its reason.
//! This crate keeps the channel a trait so the engine's diag crate can
//! own the real sink later; the two sinks here are the silent default
//! and a stderr sink for local work. Same shape as the other arch
//! modules' channels, so arch modules are interchangeable at the seam.

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
        eprintln!("[dflash] {event}: {detail}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// A capturing sink in the itak-engine test shape: records every
    /// event so the test can prove the seam delivers what it is
    /// given, instead of trusting that sink wiring is present in the
    /// source.
    #[derive(Default)]
    struct Capture {
        events: Mutex<Vec<(String, String)>>,
    }

    impl Diag for Capture {
        fn event(&self, event: &str, detail: &str) {
            self.events
                .lock()
                .expect("capture lock")
                .push((event.to_string(), detail.to_string()));
        }
    }

    #[test]
    fn seam_delivers_events_to_a_sink() {
        let capture = Capture::default();
        capture.event("seam_probe", "detail=fired");
        let events = capture.events.lock().expect("capture lock");
        assert!(
            events
                .iter()
                .any(|(e, d)| e == "seam_probe" && d == "detail=fired"),
            "a sink must receive the events emitted through the seam"
        );
        drop(events);
        // The shipped sinks behave: NullDiag drops silently and
        // StderrDiag writes without panicking.
        NullDiag.event("seam_probe", "detail=dropped");
        StderrDiag.event("seam_probe", "detail=stderr");
    }
}
