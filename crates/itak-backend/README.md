# itak-backend

The backend trait of the iTaK Torch rebuild (ARCHITECTURE.md
Section 5, BACKEND.md) and its first implementation: the scalar
CPU reference backend (TODO T-4.1), plus the sysfs hardware
detection and auto-config of TODO T-4.3. The two halves were built
as separate lanes and unified into this one crate at integration
(2026-10-29); where both halves named a type, both types survive
at distinct paths (see the crate docs in `src/lib.rs`).

The engine talks to compute only through the `Backend` trait, so
any backend can be disabled by configuration and the rest of the
engine never learns which one ran. Selection lives in `detect`:
configuration and adapter capability in, a `HardwareReport` out,
walking the BACKEND.md ladder (L0 discrete Vulkan, L1
non-discrete only when configured, L4 CPU for everything).
Per-op fallback (L2) is planned from two backends' capability
reports with `report::plan_op_fallbacks`; device recreation (L3)
belongs to the GPU backend when it lands (T-4.2).

## The CPU backend

`CpuBackend` is the correctness reference and a supported serving
mode in its own right: `BackendConfig::cpu_only()` selects it even
on a host with a discrete GPU installed, and the selection tests
prove the CPU backend then serves a full op chain.

Where the workspace already has a reference op, the CPU backend
delegates to it: `torch-math` for matmul, matvec, RMSNorm, RoPE,
SwiGLU, GeGLU, and logit shaping, and the `gguf` crate's dequant
tables for every quantized type the loader accepts. Softmax and
single-query attention had no workspace op; both are scalar
implementations here, written to the same masking semantics as the
standalone `attention` crate.

## Scalar honesty

Every CPU path in this crate is scalar. There are no SIMD
intrinsics anywhere; `SimdFeatures` in the hardware report is
detection only (what the host CPU could do later), and the
capability report says `ImplKind::Scalar` for every op, pinned by
a test. Buffers are flat, contiguous, and row-major throughout, so
a future SIMD pass changes loop bodies, not signatures, and the
quantized products are deliberately dequantize-then-multiply per
call until the fused kernels (BACKEND.md K2/K3) land.

## Detection (T-4.3, the `hwdetect` module)

`hwdetect::detect() -> hwdetect::HardwareReport`, in fallback
order:

1. **AMD from amdgpu sysfs.** `/sys/class/drm/card<N>/device/vendor`
   names the vendor; `mem_info_vram_total` gives VRAM in bytes.
   This path exists because the original engine's `detectLinux`
   read VRAM only from `nvidia-smi`, reported the RX 7900 XTX as
   0 MiB, and forced CPU-only. The regression test
   `regression_7900_xtx_reported_zero_mib_forced_cpu_only` in
   `tests/detect_tests.rs` is named for that bug and pins the fix:
   the live card's real sysfs figure (24560 MiB) with no
   `nvidia-smi` present must detect in full and select the GPU.
2. **NVIDIA via `nvidia-smi`**, when sysfs could not size a card.
3. **CPU-only**: an empty report, a legitimate answer.

The sysfs root and the command runner are injectable
(`Detector::with_sysfs_root`, `Detector::with_command_runner`),
so the whole suite runs against a fake sysfs tree. The `Detector`
also implements the selection policy's `AdapterProbe` seam, so
`detect`'s ranking can run against real sysfs enumeration.

## Auto-config (the `autoconfig` module)

`auto_select(report, config) -> BackendSelection`. All limits are
config fields with defaults (minimum GPU VRAM 4096 MiB, VRAM
budget fraction 0.90, `prefer_gpu` on), never compiled-in
constants; `BackendConfig` carries the same limits with the same
defaults for operators who tune one struct. Every fallback is a
named `SelectionReason` and a line on the crate's diag channel.

## Debugging and lint

Diagnostics: the `diag` module, structured stderr lines, off by
default (`diag::set_debug_enabled(true)`, or
`diag::set_debug(true)` in the detection half's spelling, or the
`ITAK_BACKEND_DEBUG` environment variable). Lint: workspace gates
(warnings denied, clippy clean), `#![forbid(unsafe_code)]`. The
detection half adds no external dependencies: sysfs reads and
`nvidia-smi` go through std only.
