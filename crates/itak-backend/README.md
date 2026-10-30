# itak-backend

The hardware grid slot of the iTaK Torch engine (ARCHITECTURE
Section 3.1, PLAN S7): hardware detection and auto-config, with
the `Backend` trait and its CPU SIMD and wgpu Vulkan backends
landing beside them under TODO T-4.1 and T-4.2.

## Detection (T-4.3)

`detect() -> HardwareReport`, in fallback order:

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
so the whole suite runs against a fake sysfs tree.

## Auto-config

`auto_select(report, config) -> BackendSelection`. All limits are
`AutoConfig` fields with defaults (minimum GPU VRAM 4096 MiB,
VRAM budget fraction 0.90, `prefer_gpu` on), never compiled-in
constants. Every fallback is a named `SelectionReason` and a line
on the crate's diag channel (`diag::set_debug(true)`).

## Debugging and lint

Diagnostics: `diag` module, stderr lines under an `[itak-backend]`
prefix, off by default. Lint: workspace gates (warnings denied,
clippy clean), `#![forbid(unsafe_code)]`, no external deps.
