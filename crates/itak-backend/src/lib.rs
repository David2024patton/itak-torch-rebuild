//! `itak-backend`: hardware backends behind one trait, the module
//! of ARCHITECTURE 3.1 that owns hardware detection, backend
//! selection, and the CPU reference backend.
//!
//! What lives here:
//!
//! - The [`Backend`] trait: the kernel surface of BACKEND.md
//!   Section 3 as pure functions over flat tensor buffers.
//! - [`detect`]: hardware detection and selection, implementing
//!   the BACKEND.md Section 2 adapter ranking and the Section 6
//!   fallback ladder (L0 discrete Vulkan, L1 non-discrete when
//!   configuration explicitly allows it, L4 CPU), with sysfs VRAM
//!   as the cross-check truth. Selection is by configuration and
//!   capability only, never by heuristics like file size
//!   (ARCHITECTURE 5.3).
//! - [`CpuBackend`]: the CPU backend. Always available, always
//!   compiled in, and the correctness reference every other backend
//!   is verified against. It serves on CPU alone with the GPU
//!   backend disabled (the TODO T-4.1 proof), delegating to
//!   `torch-math` and the `gguf` dequantizers where the workspace
//!   already owns reference math, and carrying scalar reference
//!   implementations for the ops that exist nowhere else yet.
//! - [`hwdetect`] and [`autoconfig`]: the sysfs detection half
//!   (TODO T-4.3). [`hwdetect::Detector`] reads AMD VRAM from the
//!   amdgpu sysfs files, falls back to `nvidia-smi` for NVIDIA
//!   cards, and reports CPU-only when no GPU exists; the original
//!   engine read VRAM only from `nvidia-smi`, reported the RX 7900
//!   XTX as 0 MiB, and forced CPU-only, and the regression test
//!   named for that bug lives in this crate's test suite.
//!   [`autoconfig::auto_select`] turns a detection report into a
//!   backend choice under limits that are config fields with
//!   defaults, never compiled-in constants. The `Detector` also
//!   implements the policy's [`AdapterProbe`] seam, so the ranking
//!   in [`detect`] can run against real sysfs enumeration.
//!
//! Two names exist in two shapes in this crate, kept side by side
//! on purpose (integration 2026-10-29): the selection policy's
//! decision record [`report::HardwareReport`] and the detection
//! pass's [`hwdetect::HardwareReport`]; and the policy's kind
//! vocabulary [`report::BackendKind`] and auto-config's verdict
//! [`autoconfig::BackendKind`]. The crate root exports the policy
//! types; the detection and auto-config types of the same name
//! live at their module paths. Likewise [`detect::detect`] is the
//! policy entry point and [`hwdetect::detect`] the detection one.
//!
//! The GPU backend (wgpu on Vulkan, the RX 7900 XTX path) lands in
//! TODO T-4.2 behind this same trait; the selection policy already
//! speaks its name so the policy is testable before it exists. The
//! `itak-diag` channel of ARCHITECTURE 3.1 is represented by the
//! [`diag`] module until that crate lands.
//!
//! Diagnostics and lint gates are built in from birth (David's
//! rule): every selection, fallback, and op emits through [`diag`],
//! warnings are denied, and the crate's tests pin the capability
//! report's scalar honesty.
//!
//! ```
//! let report = itak_backend::hwdetect::detect();
//! assert_eq!(report.version, itak_backend::hwdetect::HardwareReport::VERSION);
//! let selection = itak_backend::auto_select(&report, &itak_backend::AutoConfig::default());
//! // Whatever the machine, the verdict names its reason.
//! let _ = selection.reason;
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod autoconfig;
pub mod backend;
pub mod config;
pub mod cpu;
pub mod detect;
pub mod diag;
pub mod error;
pub mod hwdetect;
pub mod report;

pub use autoconfig::{auto_select, AutoConfig, BackendSelection, SelectionReason};
pub use backend::Backend;
pub use config::{BackendConfig, GpuPolicy};
pub use cpu::CpuBackend;
pub use detect::{detect, AdapterBackend, AdapterInfo, AdapterProbe, DeviceType, SystemProbe};
pub use error::BackendError;
pub use hwdetect::{CommandRunner, DetectSource, Detector, GpuInfo, GpuVendor, SystemCommandRunner};
pub use report::{
    BackendKind, CapabilityReport, CpuInfo, DiagEvent, FallbackLevel, HardwareReport, ImplKind, Op,
    OpStatus, OpSupport, SelectionOutcome, SimdFeatures, ALL_GGML_TYPES,
};
