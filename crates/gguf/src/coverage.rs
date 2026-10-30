//! Per-file tensor coverage: the loader's truthfulness report.
//!
//! A coverage report answers, for one parsed file and one requirement
//! list (usually an architecture module's tensor map): is every required
//! tensor present, readable by this loader, and shaped as expected?
//! [`GgufFile::coverage_report`] only reports; [`GgufFile::verify_coverage`]
//! is the load gate and fails loudly, naming every tensor that is
//! missing, unreadable, or misshapen.
//!
//! There is no path from a failed check to substitute data: a tensor
//! that fails coverage fails the load. That is the audit rule this crate
//! exists to enforce (the old Go loader invented pseudo-random weights
//! for whatever it could not read).

use crate::diag;
use crate::error::{GgufError, Result};
use crate::file::GgufFile;

/// One tensor an architecture map requires from a file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequiredTensor {
    name: String,
    dims: Option<Vec<u64>>,
}

impl RequiredTensor {
    /// Require presence and readability, with no shape expectation.
    pub fn named(name: &str) -> Self {
        Self {
            name: name.to_string(),
            dims: None,
        }
    }

    /// Require presence, readability, and exactly these dimensions
    /// (GGUF order: contiguous dimension first).
    pub fn shaped(name: &str, dims: &[u64]) -> Self {
        Self {
            name: name.to_string(),
            dims: Some(dims.to_vec()),
        }
    }

    /// Required tensor name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Expected dimensions, when the requirement carries a shape.
    pub fn expected_dims(&self) -> Option<&[u64]> {
        self.dims.as_deref()
    }
}

/// The verdict for one required tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorStatus {
    /// Present, readable, and shaped as required.
    Ok,
    /// Not present in the file's tensor directory.
    Missing,
    /// Present, but this loader cannot dequantize its ggml type, so its
    /// bytes cannot be honestly turned into weights.
    Unreadable {
        /// Mainline ggml type id stored in the file.
        type_id: u32,
        /// ggml's name for that type id.
        type_name: &'static str,
    },
    /// Present and readable, but the stored dimensions differ from the
    /// required shape.
    Misshapen {
        /// Dimensions the requirement asked for.
        expected: Vec<u64>,
        /// Dimensions stored in the file.
        found: Vec<u64>,
    },
}

impl TensorStatus {
    /// Whether this verdict passes the load gate.
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Ok)
    }

    /// Plain-words reason, for error messages and diagnostics.
    pub fn describe(&self) -> String {
        match self {
            Self::Ok => "ok".to_string(),
            Self::Missing => "missing from the file".to_string(),
            Self::Unreadable { type_id, type_name } => {
                format!("unreadable: type {type_name} (ggml type id {type_id}) has no reader")
            }
            Self::Misshapen { expected, found } => {
                format!("misshapen: expected dims {expected:?}, file has {found:?}")
            }
        }
    }
}

/// Coverage verdict for one required tensor, with the file's own facts
/// about it when it exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorCoverage {
    /// Required tensor name.
    pub name: String,
    /// The verdict.
    pub status: TensorStatus,
    /// ggml type name of the stored tensor, when present.
    pub type_name: Option<&'static str>,
    /// Stored dimensions, when present.
    pub dims: Option<Vec<u64>>,
    /// Stored element count, when present.
    pub elements: Option<u64>,
}

/// The full coverage report for one file against one requirement list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoverageReport {
    /// One entry per required tensor, in requirement order.
    pub entries: Vec<TensorCoverage>,
    /// Names of tensors the file carries that were not required
    /// (informational; extra tensors never fail a load).
    pub unchecked: Vec<String>,
}

impl CoverageReport {
    /// True when every required tensor passed.
    pub fn is_complete(&self) -> bool {
        self.entries.iter().all(|e| e.status.is_ok())
    }

    /// Number of required tensors checked.
    pub fn required_count(&self) -> usize {
        self.entries.len()
    }

    /// Number of required tensors that passed.
    pub fn ok_count(&self) -> usize {
        self.entries.iter().filter(|e| e.status.is_ok()).count()
    }

    /// The failing entries, in requirement order.
    pub fn failures(&self) -> impl Iterator<Item = &TensorCoverage> {
        self.entries.iter().filter(|e| !e.status.is_ok())
    }

    /// One-line `key=value` summary for the diagnostic channel and logs.
    pub fn summary(&self) -> String {
        format!(
            "required={} ok={} failed={} unchecked={}",
            self.required_count(),
            self.ok_count(),
            self.required_count() - self.ok_count(),
            self.unchecked.len()
        )
    }
}

impl GgufFile {
    /// Build the coverage report for `required` without failing: every
    /// verdict is data the caller can inspect.
    pub fn coverage_report(&self, required: &[RequiredTensor]) -> CoverageReport {
        let mut entries = Vec::with_capacity(required.len());
        for req in required {
            let entry = match self.tensor_info(req.name()) {
                Err(_) => TensorCoverage {
                    name: req.name().to_string(),
                    status: TensorStatus::Missing,
                    type_name: None,
                    dims: None,
                    elements: None,
                },
                Ok(info) => {
                    let status = if info.ggml_type().is_none() {
                        TensorStatus::Unreadable {
                            type_id: info.type_id(),
                            type_name: info.type_name(),
                        }
                    } else if let Some(expected) = req.expected_dims() {
                        if expected == info.dims() {
                            TensorStatus::Ok
                        } else {
                            TensorStatus::Misshapen {
                                expected: expected.to_vec(),
                                found: info.dims().to_vec(),
                            }
                        }
                    } else {
                        TensorStatus::Ok
                    };
                    TensorCoverage {
                        name: req.name().to_string(),
                        status,
                        type_name: Some(info.type_name()),
                        dims: Some(info.dims().to_vec()),
                        elements: Some(info.element_count()),
                    }
                }
            };
            entries.push(entry);
        }
        let required_names: std::collections::HashSet<&str> =
            required.iter().map(RequiredTensor::name).collect();
        let unchecked = self
            .tensor_names()
            .filter(|n| !required_names.contains(n))
            .map(str::to_string)
            .collect();
        let report = CoverageReport { entries, unchecked };
        diag::debug("coverage_report", &report.summary());
        report
    }

    /// The load gate: build the coverage report and fail loudly when any
    /// required tensor is missing, unreadable, or misshapen. The error
    /// names every failing tensor; no weights are produced for a file
    /// that fails here.
    pub fn verify_coverage(&self, required: &[RequiredTensor]) -> Result<CoverageReport> {
        let report = self.coverage_report(required);
        if report.is_complete() {
            return Ok(report);
        }
        let failures: Vec<String> = report
            .failures()
            .map(|e| {
                diag::error(
                    "coverage_failure",
                    &format!("tensor={} reason={}", e.name, e.status.describe()),
                );
                format!("tensor {}: {}", e.name, e.status.describe())
            })
            .collect();
        Err(GgufError::CoverageIncomplete { failures })
    }

    /// Coverage of the file against itself: every stored tensor checked
    /// for readability, no shape expectations. This is the per-file
    /// report the conformance corpus records for each dissected GGUF.
    pub fn full_coverage(&self) -> CoverageReport {
        let required: Vec<RequiredTensor> =
            self.tensor_names().map(RequiredTensor::named).collect();
        self.coverage_report(&required)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_descriptions_name_the_problem() {
        assert_eq!(TensorStatus::Missing.describe(), "missing from the file");
        let u = TensorStatus::Unreadable {
            type_id: 15,
            type_name: "Q8_K",
        };
        assert!(u.describe().contains("Q8_K"));
        assert!(u.describe().contains("15"));
        let m = TensorStatus::Misshapen {
            expected: vec![1024, 256],
            found: vec![512, 256],
        };
        assert!(m.describe().contains("1024"));
        assert!(m.describe().contains("512"));
    }

    #[test]
    fn required_tensor_accessors() {
        let plain = RequiredTensor::named("a.weight");
        assert_eq!(plain.name(), "a.weight");
        assert_eq!(plain.expected_dims(), None);
        let shaped = RequiredTensor::shaped("b.weight", &[4, 2]);
        assert_eq!(shaped.expected_dims(), Some([4, 2].as_slice()));
    }
}
