//! Diff two diag capture runs: the mechanical answer to
//! "where did the wrong run go wrong".
//!
//! Each argument is a `WGPU_COMPUTE_DIAG_OUT` prefix from a
//! run made with `WGPU_COMPUTE_DIAG=capture`. The tool reads
//! the two sidecars capture writes beside that prefix
//! (`<prefix>.capture.csv` and `<prefix>.manifest.json`) and
//! compares them:
//!
//! - Manifests first: every field that differs, is missing,
//!   or is extra is printed, so "were these even the same
//!   run" (same build, same adapter and driver, same
//!   resolved configuration, same model file) is answered
//!   before any data is compared.
//! - Then the capture records, position by position: the
//!   first record whose kind, value count, fingerprint,
//!   argmax, or accepted count differs is the divergence,
//!   printed with both runs' records. Everything before it
//!   provably computed the same values.
//!
//! Exit codes: 0 when the runs are identical, 1 when any
//! manifest field or capture record differs, 2 on a usage
//! or read error.
//!
//! Usage: diag_diff <diag_out A> <diag_out B>

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use wgpu_compute::{
    capture_sidecar_path, diff_capture_records, diff_manifest_fields, manifest_fields,
    manifest_sidecar_path, parse_capture_csv,
};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("usage: diag_diff <diag_out A> <diag_out B>");
        return ExitCode::from(2);
    }
    match run(&PathBuf::from(&args[0]), &PathBuf::from(&args[1])) {
        Ok(differs) => {
            if differs {
                ExitCode::from(1)
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(message) => {
            eprintln!("diag_diff: {message}");
            ExitCode::from(2)
        }
    }
}

fn run(a_prefix: &Path, b_prefix: &Path) -> Result<bool, String> {
    let mut differs = false;

    // Manifests: the same-run question.
    let a_manifest = read_optional(&manifest_sidecar_path(a_prefix))?;
    let b_manifest = read_optional(&manifest_sidecar_path(b_prefix))?;
    match (&a_manifest, &b_manifest) {
        (Some(a), Some(b)) => {
            let diffs = diff_manifest_fields(&manifest_fields(a), &manifest_fields(b));
            if diffs.is_empty() {
                println!("manifests: identical");
            } else {
                differs = true;
                println!("manifests: {} field(s) differ:", diffs.len());
                for line in diffs {
                    println!("  {line}");
                }
            }
        }
        (None, None) => println!("manifests: neither run wrote one"),
        (Some(_), None) => {
            differs = true;
            println!("manifests: only the first run wrote one");
        }
        (None, Some(_)) => {
            differs = true;
            println!("manifests: only the second run wrote one");
        }
    }

    // Capture records: the first-divergence question.
    let a_records = read_records(&capture_sidecar_path(a_prefix))?;
    let b_records = read_records(&capture_sidecar_path(b_prefix))?;
    let diff = diff_capture_records(&a_records, &b_records);
    match &diff.divergence {
        None => println!("capture records: identical ({} compared)", diff.compared),
        Some(div) => {
            differs = true;
            println!(
                "capture records: first divergence at seq {} (field {}), after {} agreeing record(s)",
                div.seq,
                div.field,
                diff.compared - 1
            );
            match &div.a {
                Some(r) => println!("  first : {}", r.to_csv_row()),
                None => println!("  first : (no record)"),
            }
            match &div.b {
                Some(r) => println!("  second: {}", r.to_csv_row()),
                None => println!("  second: (no record)"),
            }
        }
    }
    Ok(differs)
}

/// Read a sidecar that may legitimately be absent; absence
/// is `Ok(None)`, any other failure is an error.
fn read_optional(path: &Path) -> Result<Option<String>, String> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(Some(text)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("reading {} failed: {e}", path.display())),
    }
}

/// Read and parse a capture CSV sidecar; absence here is an
/// error (a capture run always writes one at finish).
fn read_records(path: &Path) -> Result<Vec<wgpu_compute::CaptureRecord>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("reading {} failed: {e}", path.display()))?;
    parse_capture_csv(&text)
}
