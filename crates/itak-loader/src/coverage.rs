//! The tensor inventory check: a file's tensor names against the arch
//! module's expected names.
//!
//! This is the mechanical form of the audit's central lesson. The
//! original engine loaded models through a Llama-shaped template and
//! never noticed the tensors it had not looked up; here the arch
//! module publishes its full expectation
//! ([`muse_glimmer::expected_tensor_names`]) and this module checks a
//! real file against it, name by name, before a single byte of weight
//! data moves. A miss fails by name, all of them at once.
//!
//! Names the file carries beyond the expectation are reported as
//! `unexpected`, not treated as failures: files legitimately carry
//! tensors a given arch module does not consume (a vision tower beside
//! the language model, for instance), and hiding them would be its own
//! kind of silence. They are printed so a human can judge them.

use std::collections::HashSet;

use gguf::GgufFile;
use muse_glimmer::{expected_tensor_names, GlimmerConfig};

use crate::diag;
use crate::error::{LoaderError, Result};

/// The outcome of one inventory check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoverageReport {
    expected: Vec<String>,
    present: Vec<String>,
    missing: Vec<String>,
    unexpected: Vec<String>,
}

impl CoverageReport {
    /// Every name the arch module expects, in its load order.
    pub fn expected(&self) -> &[String] {
        &self.expected
    }

    /// Expected names found in the file, in expected order.
    pub fn present(&self) -> &[String] {
        &self.present
    }

    /// Expected names absent from the file, in expected order.
    pub fn missing(&self) -> &[String] {
        &self.missing
    }

    /// Names the file carries that the arch module does not expect,
    /// in file order.
    pub fn unexpected(&self) -> &[String] {
        &self.unexpected
    }

    /// How many names the arch module expects.
    pub fn expected_count(&self) -> usize {
        self.expected.len()
    }

    /// How many expected names the file has.
    pub fn found_count(&self) -> usize {
        self.present.len()
    }

    /// True when every expected name is present.
    pub fn is_complete(&self) -> bool {
        self.missing.is_empty()
    }

    /// The gate form of the check: `Ok` when complete, otherwise a
    /// [`LoaderError::MissingTensors`] naming every miss.
    pub fn require_complete(&self) -> Result<()> {
        if self.missing.is_empty() {
            Ok(())
        } else {
            diag::debug(
                "coverage_incomplete",
                &format!("missing={}", self.missing.join(",")),
            );
            Err(LoaderError::MissingTensors(self.missing.clone()))
        }
    }
}

/// Run the inventory check of `file` against the expectation derived
/// from `cfg`.
pub fn coverage(file: &GgufFile, cfg: &GlimmerConfig) -> CoverageReport {
    let expected = expected_tensor_names(cfg);
    let mut present = Vec::new();
    let mut missing = Vec::new();
    for name in &expected {
        if file.has_tensor(name) {
            present.push(name.clone());
        } else {
            missing.push(name.clone());
        }
    }
    let expected_set: HashSet<&str> = expected.iter().map(String::as_str).collect();
    let unexpected: Vec<String> = file
        .tensor_names()
        .filter(|name| !expected_set.contains(name))
        .map(str::to_string)
        .collect();
    diag::debug(
        "coverage",
        &format!(
            "expected={} found={} missing={} unexpected={}",
            expected.len(),
            present.len(),
            missing.len(),
            unexpected.len()
        ),
    );
    CoverageReport {
        expected,
        present,
        missing,
        unexpected,
    }
}
