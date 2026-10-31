//! The engine's diagnostic channel.
//!
//! David's rule: debugging is built into every module from birth,
//! not bolted on after a failure. Until this module landed, the
//! engine's channel was the free-standing `log_line` helper in
//! main.rs; this module is that channel with the shape the sibling
//! crates use, so engine logs read the same across the workspace:
//!
//! - An emit function ([`emit`] for lifecycle and job events,
//!   [`error`] for refusals and failures) taking a named event and
//!   a detail string.
//! - A sink seam ([`Diag`], mirroring the arch crates' trait of the
//!   same name): the process sink is stderr by default
//!   ([`StderrDiag`]), and tests can install a capturing sink
//!   through [`set_sink`] to prove a refusal emitted its event.
//! - Named event kinds, catalogued here. Lifecycle and job events:
//!   `model_load_begin`, `model_loaded`, `stale_socket_removed`,
//!   `serving`, `concurrency`, `hello`, `generate_start`,
//!   `generate_prompt_ids`, `generate_done`, `generate_failed`,
//!   `generate_aborted`. Refusals and failures emit through
//!   [`error`] under their protocol code (`peer_refused`,
//!   `protocol_mismatch`, `bad_frame`, `bad_request`,
//!   `model_not_found`, `engine_busy`, `engine_died`,
//!   `context_overflow`, `no_chat_template`, `unknown_op`) or, for
//!   failures with no wire code, `model_load_failed`,
//!   `socket_bind_failed`, `accept_failed`, `connection_failed`,
//!   `connection_read_failed`.
//!
//! Both levels are unconditional. The engine's stderr is its
//! operator log (the platform supervisor forwards it), and a
//! refusal hidden behind a debug flag is exactly the failure mode
//! the audit law exists to prevent; this process has no
//! debug-level chatter to gate. The one content gate the engine
//! keeps, the operator's `--log-content` for full prompt ids and
//! completion text, stays at its call sites and is orthogonal to
//! this channel.
//!
//! The stderr sink prints the detail with the engine's historical
//! `[itak-engine {secs}]` prefix, byte for byte the lines the
//! engine has always logged, so existing log readers see one
//! shape. The event name rides the channel for sinks that record
//! structure: the capturing test sink today, a structured sink
//! whenever the platform grows one.

use std::sync::{Arc, OnceLock, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// A sink for the engine's diagnostic events: a named event plus a
/// detail string. Implementations must be cheap to call, because
/// any thread in the process may emit at any time.
pub trait Diag: Send + Sync {
    /// Record one event.
    fn event(&self, event: &str, detail: &str);
}

/// The default sink: one timestamped line per event on stderr, in
/// the format the engine has always logged.
#[derive(Debug, Default)]
pub struct StderrDiag;

impl Diag for StderrDiag {
    fn event(&self, _event: &str, detail: &str) {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        eprintln!("[itak-engine {secs}] {detail}");
    }
}

static SINK: OnceLock<RwLock<Arc<dyn Diag>>> = OnceLock::new();

fn sink_slot() -> &'static RwLock<Arc<dyn Diag>> {
    SINK.get_or_init(|| RwLock::new(Arc::new(StderrDiag)))
}

fn current_sink() -> Arc<dyn Diag> {
    let slot = sink_slot().read().unwrap_or_else(|p| p.into_inner());
    Arc::clone(&slot)
}

/// Install a different process sink. The seam exists for tests,
/// which install a capturing sink to prove a refusal emitted its
/// named event; production runs on the stderr default.
#[cfg(test)]
pub fn set_sink(sink: Arc<dyn Diag>) {
    *sink_slot().write().unwrap_or_else(|p| p.into_inner()) = sink;
}

/// Serializes tests that swap the process sink. The sink is
/// process-global, so two swapping tests running in parallel would
/// capture (and restore over) each other's events; a test that
/// installs a sink holds this lock for its whole install, drive,
/// read, restore sequence.
#[cfg(test)]
pub static SINK_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Emit a lifecycle or job event through the diagnostic channel.
pub fn emit(event: &str, detail: &str) {
    current_sink().event(event, detail);
}

/// Emit a refusal or failure through the diagnostic channel, under
/// its protocol code where the wire has one. Unconditional, like
/// every error path in this workspace: a refusal that only shows
/// under a debug flag is a silent refusal.
pub fn error(event: &str, detail: &str) {
    current_sink().event(event, detail);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn stderr_sink_event_does_not_panic() {
        StderrDiag.event("test_event", "detail=present");
    }

    #[test]
    fn emit_routes_to_the_installed_sink() {
        // A capturing sink defined here, the same way the engine's
        // connection tests capture: install, emit, read back.
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
        let _guard = SINK_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let capture = Arc::new(Capture::default());
        set_sink(capture.clone());
        emit("lifecycle_probe", "detail=one");
        error("refusal_probe", "detail=two");
        set_sink(Arc::new(StderrDiag));
        let events = capture.events.lock().expect("capture lock");
        assert!(events
            .iter()
            .any(|(e, d)| e == "lifecycle_probe" && d == "detail=one"));
        assert!(events
            .iter()
            .any(|(e, d)| e == "refusal_probe" && d == "detail=two"));
    }
}
