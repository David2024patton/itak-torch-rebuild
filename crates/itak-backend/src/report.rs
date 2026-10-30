//! Reports: the queryable state of the backend layer.
//!
//! ARCHITECTURE Section 4 and BACKEND.md Section 6 require every
//! fallback to be an explicit, logged, queryable state. This module
//! is the queryable half: a [`CapabilityReport`] says what a backend
//! can execute and how (scalar, SIMD, or GPU kernel, and for which
//! quantization types), and a [`HardwareReport`] records what
//! [`crate::detect`] found, what it selected, at which fallback
//! level, why, and the diagnostic events it emitted on the way.

use std::fmt;

use gguf::GgmlType;

/// Every ggml element type the workspace's loader can dequantize
/// (the `gguf` crate's supported set). CPU capability entries list
/// these for the quantized ops; the list shrinks only if the loader's
/// set does.
pub const ALL_GGML_TYPES: [GgmlType; 7] = [
    GgmlType::F32,
    GgmlType::F16,
    GgmlType::Bf16,
    GgmlType::Q8_0,
    GgmlType::Q4K,
    GgmlType::Q5K,
    GgmlType::Q6K,
];

/// Which backend implementation a [`crate::Backend`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// The CPU backend: always available, always compiled in, the
    /// correctness reference for every kernel (BACKEND.md Section 6).
    Cpu,
    /// The wgpu-on-Vulkan GPU backend (the RX 7900 XTX path). Lands
    /// in TODO T-4.2; detection and selection already speak its name
    /// so the policy is testable before it exists.
    WgpuVulkan,
}

impl BackendKind {
    /// Stable machine-readable name, for reports and the diag channel.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::WgpuVulkan => "wgpu-vulkan",
        }
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// One operation of the [`crate::Backend`] surface. The capability
/// report and the load-time fallback plan (BACKEND.md Section 6,
/// level L2) are expressed in these terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// Dense matrix-matrix product (prefill shape).
    Matmul,
    /// Dense matrix-vector product (the decode shape).
    Matvec,
    /// Quantized weights times a dense matrix (prefill GEMM).
    MatmulQuantized,
    /// Quantized weights times a dense vector (the decode GEMV,
    /// BACKEND.md kernels K1 to K3).
    MatvecQuantized,
    /// Materialize a quantized tensor as f32.
    Dequantize,
    /// RMSNorm (BACKEND.md kernel K4, without the residual add).
    RmsNorm,
    /// RMSNorm fused with the residual add (K4 as it actually runs).
    RmsNormAdd,
    /// Rotary position embedding (K5's rotation half).
    Rope,
    /// Softmax over one vector (the score half of K6).
    Softmax,
    /// Grouped-query decode attention over flat key/value history
    /// (K6).
    Attention,
    /// SwiGLU gated activation (K7).
    SwiGlu,
    /// GEGLU gated activation (the Gemma-family variant of K7).
    GeGlu,
    /// Embedding row gather (K8's gather half).
    Embedding,
    /// Elementwise add (residual stream).
    Add,
    /// Elementwise multiply (attention gating).
    Mul,
    /// Logit softcap (K8's shaping half).
    LogitSoftcap,
    /// Final logit scale (K8's shaping half).
    LogitScale,
}

impl Op {
    /// Every op on the backend surface, in declaration order.
    pub const ALL: [Op; 17] = [
        Op::Matmul,
        Op::Matvec,
        Op::MatmulQuantized,
        Op::MatvecQuantized,
        Op::Dequantize,
        Op::RmsNorm,
        Op::RmsNormAdd,
        Op::Rope,
        Op::Softmax,
        Op::Attention,
        Op::SwiGlu,
        Op::GeGlu,
        Op::Embedding,
        Op::Add,
        Op::Mul,
        Op::LogitSoftcap,
        Op::LogitScale,
    ];

    /// Stable machine-readable name, for reports and the diag channel.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Matmul => "matmul",
            Self::Matvec => "matvec",
            Self::MatmulQuantized => "matmul_quantized",
            Self::MatvecQuantized => "matvec_quantized",
            Self::Dequantize => "dequantize",
            Self::RmsNorm => "rms_norm",
            Self::RmsNormAdd => "rms_norm_add",
            Self::Rope => "rope",
            Self::Softmax => "softmax",
            Self::Attention => "attention",
            Self::SwiGlu => "swiglu",
            Self::GeGlu => "geglu",
            Self::Embedding => "embedding",
            Self::Add => "add",
            Self::Mul => "mul",
            Self::LogitSoftcap => "logit_softcap",
            Self::LogitScale => "logit_scale",
        }
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// How an op is executed on the reporting backend. This is the
/// scalar/SIMD honesty record: a report that says `Scalar` means the
/// plain reference loops ran, whatever the host CPU could do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImplKind {
    /// Plain scalar reference loops (accuracy-first, f64
    /// accumulation where the reference crates do it).
    Scalar,
    /// SIMD-vectorized CPU kernels.
    Simd,
    /// A GPU compute kernel.
    GpuKernel,
}

impl ImplKind {
    /// Stable machine-readable name, for reports and the diag channel.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Simd => "simd",
            Self::GpuKernel => "gpu-kernel",
        }
    }
}

impl fmt::Display for ImplKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Whether one op is supported by a backend, and if so, how and for
/// which quantization types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpStatus {
    /// The op runs on this backend.
    Supported {
        /// How it runs (scalar, SIMD, GPU kernel).
        impl_kind: ImplKind,
        /// Quantization types accepted, for the ops that take a
        /// quantized operand (`MatvecQuantized`, `MatmulQuantized`,
        /// `Dequantize`). Empty for dense-only ops.
        quant_types: Vec<GgmlType>,
    },
    /// The op does not run on this backend.
    Unsupported {
        /// Why not, in operator language.
        reason: String,
    },
}

/// One row of a [`CapabilityReport`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpSupport {
    /// The operation.
    pub op: Op,
    /// Its support state on this backend.
    pub status: OpStatus,
}

/// What one backend can execute. Produced by
/// [`crate::Backend::capabilities`]; consumed at load time to decide
/// the per-op CPU fallback list (BACKEND.md Section 6, level L2)
/// before any tensor is served.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityReport {
    /// The backend this report describes.
    pub backend: BackendKind,
    /// One row per op the report covers. An op absent from the list
    /// counts as unsupported.
    pub ops: Vec<OpSupport>,
}

impl CapabilityReport {
    /// The row for one op, if the report carries one.
    #[must_use]
    pub fn row(&self, op: Op) -> Option<&OpSupport> {
        self.ops.iter().find(|r| r.op == op)
    }

    /// Whether this backend runs `op`, restricted to quantization
    /// type `ty` when the requirement names one (`None` asks about
    /// the dense form of the op).
    #[must_use]
    pub fn supports(&self, op: Op, ty: Option<GgmlType>) -> bool {
        match self.row(op) {
            Some(OpSupport {
                status: OpStatus::Supported { quant_types, .. },
                ..
            }) => match ty {
                None => true,
                Some(t) => quant_types.contains(&t),
            },
            _ => false,
        }
    }

    /// The requirements from `required` this backend cannot meet, in
    /// the order given. An empty result means the backend covers the
    /// whole requirement set.
    pub fn missing<I>(&self, required: I) -> Vec<(Op, Option<GgmlType>)>
    where
        I: IntoIterator<Item = (Op, Option<GgmlType>)>,
    {
        required
            .into_iter()
            .filter(|(op, ty)| !self.supports(*op, *ty))
            .collect()
    }
}

/// The per-op fallback list of BACKEND.md Section 6, level L2: the
/// (op, type) pairs the `primary` backend cannot run but the
/// `fallback` backend can. Decided at load time from the two
/// capability reports and disclosed in the load result; each pair
/// corresponds to one `backend.op_fallback_cpu` diag event at load.
///
/// Pairs neither backend can run are NOT in the result: those are a
/// load refusal for the caller to name, not a fallback.
pub fn plan_op_fallbacks(
    primary: &CapabilityReport,
    fallback: &CapabilityReport,
    required: &[(Op, Option<GgmlType>)],
) -> Vec<(Op, Option<GgmlType>)> {
    required
        .iter()
        .copied()
        .filter(|(op, ty)| !primary.supports(*op, *ty) && fallback.supports(*op, *ty))
        .collect()
}

/// One diagnostic event emitted during detection or selection. The
/// same events go to the diag channel as they happen; they are
/// collected here so the state is queryable afterwards, per
/// ARCHITECTURE Section 4.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagEvent {
    /// Stable event code (BACKEND.md Section 6 registry).
    pub code: &'static str,
    /// `key=value` detail fragments, operator language.
    pub detail: String,
}

/// The BACKEND.md Section 6 fallback ladder level a selection landed
/// on. Levels L2 (per-op CPU fallback) and L3 (device recreation) are
/// load-time and runtime states, not selection outcomes, so they do
/// not appear here: L2 is computed by [`plan_op_fallbacks`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackLevel {
    /// L0: wgpu Vulkan on the discrete adapter. Normal operation.
    L0,
    /// L1: wgpu Vulkan on a non-discrete adapter, entered only when
    /// configuration explicitly allows it.
    L1,
    /// L4: the CPU backend for everything (no suitable adapter, or
    /// configuration disables the GPU).
    L4,
}

/// What detection decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionOutcome {
    /// A backend was selected, at a ladder level.
    Selected {
        /// The selected backend.
        kind: BackendKind,
        /// The ladder level the selection landed on.
        level: FallbackLevel,
    },
    /// No backend satisfies the configured policy (a `Required` GPU
    /// policy with no suitable adapter). The engine refuses rather
    /// than falling back silently.
    Refused {
        /// Why no backend satisfied the policy.
        reason: String,
    },
}

/// CPU SIMD features relevant to the CPU backend's later SIMD pass.
/// Detected at runtime; reported for the record. The current CPU
/// backend executes scalar loops regardless (see [`ImplKind`]): this
/// struct describes the host, not the code path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SimdFeatures {
    /// AVX2 (x86_64).
    pub avx2: bool,
    /// AVX-512F (x86_64).
    pub avx512f: bool,
    /// FMA (x86_64).
    pub fma: bool,
    /// NEON (aarch64).
    pub neon: bool,
}

/// What detection learned about the host CPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuInfo {
    /// Model name (from `/proc/cpuinfo` on Linux; empty elsewhere).
    pub model: String,
    /// Logical cores available to the process.
    pub logical_cores: usize,
    /// SIMD features the host CPU advertises.
    pub simd: SimdFeatures,
}

/// The result of [`crate::detect`]: the hardware facts, the
/// selection, and the events emitted while deciding. This is the
/// summary the engine's Hello exchange and health surface carry
/// (ARCHITECTURE Section 4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareReport {
    /// The host CPU.
    pub cpu: CpuInfo,
    /// Every GPU adapter the probe enumerated, selectable or not.
    pub adapters: Vec<crate::detect::AdapterInfo>,
    /// The adapter the selection chose, when it chose one.
    pub chosen_adapter: Option<crate::detect::AdapterInfo>,
    /// Total VRAM from AMD sysfs (`mem_info_vram_total`), when
    /// readable. The cross-check source of BACKEND.md Section 2:
    /// sysfs is the fix for the old nvidia-smi-only detection bug.
    pub sysfs_vram_total_bytes: Option<u64>,
    /// VRAM the backend may plan against: sysfs total minus the
    /// configured reserve, when both are known.
    pub vram_budget_bytes: Option<u64>,
    /// What detection decided.
    pub outcome: SelectionOutcome,
    /// Operator-language account of the decision.
    pub reason: String,
    /// Diagnostic events emitted while deciding, in order.
    pub events: Vec<DiagEvent>,
}

impl HardwareReport {
    /// The selected backend kind, or `None` when selection refused.
    #[must_use]
    pub fn selected_kind(&self) -> Option<BackendKind> {
        match self.outcome {
            SelectionOutcome::Selected { kind, .. } => Some(kind),
            SelectionOutcome::Refused { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(ops: Vec<OpSupport>) -> CapabilityReport {
        CapabilityReport {
            backend: BackendKind::WgpuVulkan,
            ops,
        }
    }

    fn supported(op: Op, quant_types: &[GgmlType]) -> OpSupport {
        OpSupport {
            op,
            status: OpStatus::Supported {
                impl_kind: ImplKind::GpuKernel,
                quant_types: quant_types.to_vec(),
            },
        }
    }

    #[test]
    fn supports_checks_quant_types() {
        let r = report(vec![supported(Op::MatvecQuantized, &[GgmlType::Q4K])]);
        assert!(r.supports(Op::MatvecQuantized, Some(GgmlType::Q4K)));
        assert!(!r.supports(Op::MatvecQuantized, Some(GgmlType::Q5K)));
        // Dense form of a supported op counts as supported.
        assert!(r.supports(Op::MatvecQuantized, None));
        // Absent op counts as unsupported.
        assert!(!r.supports(Op::Softmax, None));
    }

    #[test]
    fn fallback_plan_lists_only_pairs_the_fallback_covers() {
        let primary = report(vec![supported(Op::MatvecQuantized, &[GgmlType::Q4K])]);
        let fallback = CapabilityReport {
            backend: BackendKind::Cpu,
            ops: vec![OpSupport {
                op: Op::MatvecQuantized,
                status: OpStatus::Supported {
                    impl_kind: ImplKind::Scalar,
                    quant_types: ALL_GGML_TYPES.to_vec(),
                },
            }],
        };
        let required = [
            (Op::MatvecQuantized, Some(GgmlType::Q4K)),
            (Op::MatvecQuantized, Some(GgmlType::Q5K)),
            (Op::MatvecQuantized, Some(GgmlType::Q6K)),
            (Op::Softmax, None),
        ];
        let plan = plan_op_fallbacks(&primary, &fallback, &required);
        // Q4K runs on the primary; Q5K and Q6K fall back; Softmax is
        // covered by neither, so it is a refusal, not a fallback.
        assert_eq!(
            plan,
            vec![
                (Op::MatvecQuantized, Some(GgmlType::Q5K)),
                (Op::MatvecQuantized, Some(GgmlType::Q6K)),
            ]
        );
    }
}
