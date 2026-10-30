//! The diagnostic seam for the speculation loop (SPEC-itak-spec Section
//! 6.4). Every refusal, round, rewind, commit, and fallback emits through
//! this trait, so the engine can route the loop's evidence wherever its own
//! diagnostics go. The catalogue:
//!
//! - `spec.session.begin` / `spec.session.end`: session lifecycle, with the
//!   mode, depth, and (at the end) the cumulative acceptance figures.
//! - `spec.disabled`: speculation was configured off; the session generates
//!   plainly and says so once, never silently.
//! - `spec.round.begin`: round number, cache base, pending token.
//! - `spec.propose.end`: drafts proposed, draft wall time.
//! - `spec.verify.end`: verify wall time.
//! - `spec.round.accept`: proposed, accepted, first rejection, emitted.
//! - `spec.cache.rewind` / `spec.cache.commit`: cache movements, from/to.
//! - `spec.fallback.latched`: a verify refusal latched speculation off;
//!   carries the reason and the round.
//! - `spec.refused`: a creation or round refusal, with the error text.

/// One sink for the loop's structured diagnostic events.
pub trait Diag: Send + Sync {
    /// Record one event: a dotted name plus `key=value` detail fields.
    fn event(&self, name: &str, detail: &str);
}

/// A sink that drops every event (tests and quiet embeddings).
#[derive(Debug, Default)]
pub struct NullDiag;

impl Diag for NullDiag {
    fn event(&self, _name: &str, _detail: &str) {}
}

/// A sink that writes events to stderr, one line each.
#[derive(Debug, Default)]
pub struct StderrDiag;

impl Diag for StderrDiag {
    fn event(&self, name: &str, detail: &str) {
        eprintln!("[diag] {name} {detail}");
    }
}
