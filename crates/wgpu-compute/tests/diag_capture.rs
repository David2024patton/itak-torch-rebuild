//! The capture level's end-to-end contract through the
//! public API: two capture runs over the same values leave
//! sidecars that diff clean, and a run whose data changes at
//! one step diffs at exactly that step. No GPU: the capture
//! machinery is host-side by design (it fingerprints vectors
//! after their readback), so its whole contract is testable
//! here.

use std::path::{Path, PathBuf};

use wgpu_compute::{
    diff_capture_records, diff_manifest_fields, manifest_fields, manifest_sidecar_path,
    parse_capture_csv, register_model_file, registered_model_file, CaptureRecord, DiagAccumulator,
    DiagMode, ModelFileInfo, RunManifest, MODEL_FINGERPRINT_METHOD,
};

fn sample_manifest() -> RunManifest {
    RunManifest {
        diag_mode: "capture".to_string(),
        build_version: "0.1.0".to_string(),
        git: "unknown".to_string(),
        adapter_name: "test adapter".to_string(),
        adapter_backend: "Vulkan".to_string(),
        adapter_device_type: "DiscreteGpu".to_string(),
        adapter_driver: "radv".to_string(),
        adapter_driver_info: "Mesa test".to_string(),
        resolved_config: vec![("mmv_max_rows".to_string(), "16".to_string())],
        // Stated explicitly so the process model registry
        // (another test in this binary registers one) can
        // never leak into these manifests at begin time.
        model: Some(ModelFileInfo {
            path: "model.gguf".to_string(),
            size_bytes: 123,
            fingerprint: "0123456789abcdef".to_string(),
            method: MODEL_FINGERPRINT_METHOD.to_string(),
        }),
        session: vec![("kind".to_string(), "resident".to_string())],
    }
}

/// Drive one capture run over fixed values and finish it,
/// returning the run's report prefix. `second` tweaks the
/// second forward's logits when set.
fn run_capture(dir: &Path, name: &str, second: Option<f32>) -> PathBuf {
    let prefix = dir.join(name);
    let mut acc = DiagAccumulator::with_mode(Some(prefix.clone()), DiagMode::Capture);
    acc.begin_manifest(sample_manifest());
    acc.record_forward(CaptureRecord::KIND_FORWARD, &[0.0, 1.0, 0.5])
        .expect("first forward");
    let second_logits = [2.0, second.unwrap_or(1.0), 0.25];
    acc.record_forward(CaptureRecord::KIND_FORWARD, &second_logits)
        .expect("second forward");
    acc.record_argmax(7, 1.5).expect("argmax");
    acc.record_spec_round(&[10, 11, 12, 13], 3).expect("round");
    assert!(acc.write_report(0));
    prefix
}

fn read_records(prefix: &Path) -> Vec<CaptureRecord> {
    let text = std::fs::read_to_string(wgpu_compute::capture_sidecar_path(prefix))
        .expect("capture sidecar written");
    parse_capture_csv(&text).expect("capture sidecar parses")
}

fn test_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "wgpu-compute-diag-capture-it-{name}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

#[test]
fn identical_capture_runs_diff_clean() {
    let dir = test_dir("identical");
    let a = run_capture(&dir, "a.csv", None);
    let b = run_capture(&dir, "b.csv", None);
    let diff = diff_capture_records(&read_records(&a), &read_records(&b));
    assert!(diff.identical(), "divergence: {:?}", diff.divergence);
    assert_eq!(diff.compared, 4);
    let a_manifest = std::fs::read_to_string(manifest_sidecar_path(&a)).expect("manifest a");
    let b_manifest = std::fs::read_to_string(manifest_sidecar_path(&b)).expect("manifest b");
    assert_eq!(
        a_manifest, b_manifest,
        "same inputs render the same manifest"
    );
    assert!(
        diff_manifest_fields(&manifest_fields(&a_manifest), &manifest_fields(&b_manifest))
            .is_empty()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_changed_step_is_the_named_divergence() {
    let dir = test_dir("divergence");
    let a = run_capture(&dir, "a.csv", None);
    // The second forward's middle logit moves: record 0 must
    // still agree, record 1 must be the divergence, and its
    // field must be the fingerprint.
    let b = run_capture(&dir, "b.csv", Some(1.75));
    let diff = diff_capture_records(&read_records(&a), &read_records(&b));
    let div = diff.divergence.expect("the runs diverge");
    assert_eq!(div.seq, 1);
    assert_eq!(div.field, "hash");
    assert_eq!(diff.compared, 2, "record 0 agreed, record 1 decided it");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn model_registration_reaches_a_manifest_that_notes_nothing() {
    let dir = test_dir("registry");
    let model = dir.join("weights.gguf");
    std::fs::write(&model, b"GGUF registered for the manifest").expect("write model");
    let info = register_model_file(&model).expect("register");
    assert_eq!(info.size_bytes, 32);
    assert_eq!(registered_model_file(), Some(info.clone()));
    // A manifest begun with no model of its own picks the
    // registration up.
    let mut manifest = sample_manifest();
    manifest.model = None;
    let mut acc = DiagAccumulator::with_mode(None, DiagMode::Capture);
    acc.begin_manifest(manifest);
    let fields = manifest_fields(&acc.manifest_json().expect("manifest begun"));
    let path = fields
        .iter()
        .find(|(k, _)| k == "model.path")
        .map(|(_, v)| v.as_str());
    assert_eq!(path, Some(model.display().to_string().as_str()));
    let _ = std::fs::remove_dir_all(&dir);
}
