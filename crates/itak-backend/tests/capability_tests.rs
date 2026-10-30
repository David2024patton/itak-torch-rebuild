//! Capability report tests: the CPU backend's report must cover
//! the whole backend surface, and it must be honest about how each
//! op executes (scalar today, whatever the host CPU advertises).

use gguf::GgmlType;
use itak_backend::{Backend, BackendKind, CpuBackend, ImplKind, Op, OpStatus, ALL_GGML_TYPES};

#[test]
fn cpu_report_covers_every_op() {
    let report = CpuBackend.capabilities();
    assert_eq!(report.backend, BackendKind::Cpu);
    for op in Op::ALL {
        let row = report.row(op).unwrap_or_else(|| panic!("no row for {op}"));
        assert!(
            matches!(row.status, OpStatus::Supported { .. }),
            "op {op} must be supported on the CPU backend"
        );
    }
}

#[test]
fn cpu_report_is_scalar_everywhere() {
    // The honesty pin: no op in this backend runs SIMD or GPU code
    // today. If a SIMD kernel lands, this test forces the report
    // (and the docs) to change consciously with it.
    let report = CpuBackend.capabilities();
    for row in &report.ops {
        match &row.status {
            OpStatus::Supported { impl_kind, .. } => {
                assert_eq!(
                    *impl_kind,
                    ImplKind::Scalar,
                    "op {} must report scalar execution",
                    row.op
                );
            }
            OpStatus::Unsupported { reason } => {
                panic!("op {} unexpectedly unsupported: {reason}", row.op)
            }
        }
    }
}

#[test]
fn cpu_report_quantized_ops_cover_the_loader_type_set() {
    let report = CpuBackend.capabilities();
    for op in [Op::MatvecQuantized, Op::MatmulQuantized, Op::Dequantize] {
        let row = report.row(op).expect("quantized op row exists");
        match &row.status {
            OpStatus::Supported { quant_types, .. } => {
                assert_eq!(
                    quant_types,
                    &vec![
                        GgmlType::F32,
                        GgmlType::F16,
                        GgmlType::Bf16,
                        GgmlType::Q8_0,
                        GgmlType::Q4K,
                        GgmlType::Q5K,
                        GgmlType::Q6K,
                    ],
                    "op {op} must accept every type the loader dequantizes"
                );
            }
            OpStatus::Unsupported { reason } => panic!("op {op} unsupported: {reason}"),
        }
    }
}

#[test]
fn cpu_backend_meets_the_full_requirement_set() {
    // The load-time question of BACKEND.md Section 6: given the
    // full requirement set (every op dense, every quantized op in
    // every type), the CPU backend misses nothing, which is what
    // makes it the universal L2/L4 fallback.
    let report = CpuBackend.capabilities();
    let mut required: Vec<(Op, Option<GgmlType>)> = Op::ALL.iter().map(|op| (*op, None)).collect();
    for op in [Op::MatvecQuantized, Op::MatmulQuantized, Op::Dequantize] {
        for ty in ALL_GGML_TYPES {
            required.push((op, Some(ty)));
        }
    }
    assert!(report.missing(required).is_empty());
}

#[test]
fn cpu_backend_identity() {
    let backend = CpuBackend;
    assert_eq!(backend.kind(), BackendKind::Cpu);
    assert_eq!(backend.name(), "cpu-scalar-reference");
    assert_eq!(backend.capabilities().backend, backend.kind());
}
