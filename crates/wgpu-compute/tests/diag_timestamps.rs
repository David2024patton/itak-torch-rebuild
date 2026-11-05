//! The diag instrumentation's off-path contract: with
//! `WGPU_COMPUTE_DIAG` unset (or not `timing`), a context
//! carries NO recorder, which is what guarantees no query
//! resources are created and verdict runs stay clean. The
//! record math itself is unit-tested in `timestamps.rs`; the
//! timing-mode smoke report lives in `resident_session.rs`.
//!
//! Needs a discrete Vulkan GPU under the backend's default
//! policy; skips with a printed reason where none qualifies.

use wgpu_compute::{DiagMode, WgpuCompute, WgpuError};

#[test]
fn off_mode_context_carries_no_recorder() {
    if DiagMode::from_env() == DiagMode::Timing {
        eprintln!("skipping off-mode check: this process runs with diag timing on");
        return;
    }
    let backend = match WgpuCompute::new() {
        Ok(b) => b,
        Err(WgpuError::NoSuitableAdapter { seen, .. }) => {
            eprintln!(
                "skipping off-mode check: no suitable adapter (enumerated: [{}])",
                seen.join("; ")
            );
            return;
        }
        Err(e) => panic!("backend construction failed unexpectedly: {e}"),
    };
    assert!(
        backend.context().diag_recorder().is_none(),
        "off mode must not create a timestamp recorder (no query resources)"
    );
    assert!(backend.context().diag.is_none());
}
