//! `itak-backend`: the hardware grid slot of the iTaK Torch
//! engine (ARCHITECTURE Section 3.1, PLAN S7).
//!
//! This crate owns the hardware backends behind one seam. This
//! module lands the detection and auto-config part of the slot
//! (TODO T-4.3):
//!
//! - [`detect`] reads AMD VRAM from the amdgpu sysfs files, falls
//!   back to `nvidia-smi` for NVIDIA cards, and reports CPU-only
//!   when no GPU exists. The original engine read VRAM only from
//!   `nvidia-smi`, reported the RX 7900 XTX as 0 MiB, and forced
//!   CPU-only; the regression test named for that bug lives in
//!   this crate's test suite.
//! - [`auto_select`] turns a [`HardwareReport`] into a
//!   [`BackendSelection`] under an [`AutoConfig`] whose limits are
//!   config fields with defaults, never compiled-in constants.
//!   Every fallback is a named reason and a diag line.
//!
//! The `Backend` trait and its CPU SIMD and wgpu Vulkan backends
//! are the neighbouring TODO items (T-4.1, T-4.2) and land in this
//! crate beside this module.
//!
//! ```
//! let report = itak_backend::detect();
//! assert_eq!(report.version, itak_backend::HardwareReport::VERSION);
//! let selection = itak_backend::auto_select(&report, &itak_backend::AutoConfig::default());
//! // Whatever the machine, the verdict names its reason.
//! let _ = selection.reason;
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod config;
pub mod detect;
pub mod diag;

pub use config::{auto_select, AutoConfig, BackendKind, BackendSelection, SelectionReason};
pub use detect::{
    detect, CommandRunner, DetectSource, Detector, GpuInfo, GpuVendor, HardwareReport,
    SystemCommandRunner,
};
