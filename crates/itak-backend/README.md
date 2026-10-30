# itak-backend

The backend trait of the iTaK Torch rebuild (ARCHITECTURE.md
Section 5, BACKEND.md) and its first implementation: the scalar
CPU reference backend. TODO T-4.1.

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
