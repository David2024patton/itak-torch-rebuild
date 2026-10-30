//! Selection and configuration tests (TODO T-4.1's proof shape):
//! selection follows configuration and adapter capability only,
//! lands on the BACKEND.md ladder levels, records its events, and
//! the CPU-only configuration serves with a discrete GPU present
//! but disabled.

use itak_backend::{
    detect, AdapterBackend, AdapterInfo, AdapterProbe, Backend, BackendConfig, BackendKind,
    CpuBackend, CpuInfo, DeviceType, FallbackLevel, GpuPolicy, SelectionOutcome, SimdFeatures,
    SystemProbe,
};

struct FakeProbe {
    adapters: Vec<AdapterInfo>,
    sysfs_vram: Option<u64>,
}

impl AdapterProbe for FakeProbe {
    fn adapters(&self) -> Vec<AdapterInfo> {
        self.adapters.clone()
    }

    fn sysfs_vram_total_bytes(&self) -> Option<u64> {
        self.sysfs_vram
    }

    fn cpu_info(&self) -> CpuInfo {
        CpuInfo {
            model: "Fake CPU".to_string(),
            logical_cores: 8,
            simd: SimdFeatures::default(),
        }
    }
}

fn adapter(name: &str, backend: AdapterBackend, device_type: DeviceType) -> AdapterInfo {
    AdapterInfo {
        name: name.to_string(),
        backend,
        device_type,
        vram_bytes: None,
    }
}

fn discrete_xtx() -> AdapterInfo {
    AdapterInfo {
        name: "AMD Radeon RX 7900 XTX".to_string(),
        backend: AdapterBackend::Vulkan,
        device_type: DeviceType::DiscreteGpu,
        vram_bytes: Some(25_753_026_560),
    }
}

fn event_codes(report: &itak_backend::HardwareReport) -> Vec<&'static str> {
    report.events.iter().map(|e| e.code).collect()
}

#[test]
fn gpu_disabled_by_config_wins_over_a_present_discrete_gpu() {
    // The T-4.1 configuration: a 7900 XTX is right there, and the
    // config says no GPU. Selection must land on CPU at L4 without
    // consulting the adapter's capability, and say why.
    let probe = FakeProbe {
        adapters: vec![discrete_xtx()],
        sysfs_vram: Some(25_753_026_560),
    };
    let report = detect(&BackendConfig::cpu_only(), &probe);
    assert_eq!(
        report.outcome,
        SelectionOutcome::Selected {
            kind: BackendKind::Cpu,
            level: FallbackLevel::L4,
        }
    );
    assert!(report.chosen_adapter.is_none());
    assert!(report.reason.contains("disabled by configuration"));
    assert!(event_codes(&report).contains(&"backend.fallback_cpu"));
}

#[test]
fn cpu_only_config_is_the_disabled_policy() {
    assert_eq!(BackendConfig::cpu_only().gpu_policy, GpuPolicy::Disabled);
    assert_eq!(BackendConfig::default().gpu_policy, GpuPolicy::Auto);
}

#[test]
fn auto_with_no_adapters_lands_on_cpu_l4() {
    let probe = FakeProbe {
        adapters: Vec::new(),
        sysfs_vram: None,
    };
    let report = detect(&BackendConfig::default(), &probe);
    assert_eq!(report.selected_kind(), Some(BackendKind::Cpu));
    assert!(matches!(
        report.outcome,
        SelectionOutcome::Selected {
            level: FallbackLevel::L4,
            ..
        }
    ));
    assert!(event_codes(&report).contains(&"backend.fallback_cpu"));
}

#[test]
fn auto_with_only_a_software_adapter_lands_on_cpu_l4() {
    // llvmpipe counts as no adapter (BACKEND.md Sections 2 and 6):
    // it must not produce a pretend-GPU selection.
    let probe = FakeProbe {
        adapters: vec![adapter(
            "llvmpipe (LLVM 21.1.8)",
            AdapterBackend::Vulkan,
            DeviceType::Cpu,
        )],
        sysfs_vram: None,
    };
    let report = detect(&BackendConfig::default(), &probe);
    assert_eq!(report.selected_kind(), Some(BackendKind::Cpu));
    assert!(report.reason.contains("software"));
    assert!(report.chosen_adapter.is_none());
}

#[test]
fn auto_selects_the_discrete_adapter_from_the_bench_trio() {
    // The bench host's real adapter set (BACKEND.md Section 2).
    let probe = FakeProbe {
        adapters: vec![
            discrete_xtx(),
            adapter(
                "Intel(R) UHD Graphics 770",
                AdapterBackend::Vulkan,
                DeviceType::IntegratedGpu,
            ),
            adapter(
                "llvmpipe (LLVM 21.1.8)",
                AdapterBackend::Vulkan,
                DeviceType::Cpu,
            ),
        ],
        sysfs_vram: Some(25_753_026_560),
    };
    let report = detect(&BackendConfig::default(), &probe);
    assert_eq!(
        report.outcome,
        SelectionOutcome::Selected {
            kind: BackendKind::WgpuVulkan,
            level: FallbackLevel::L0,
        }
    );
    assert_eq!(
        report.chosen_adapter.as_ref().map(|a| a.name.as_str()),
        Some("AMD Radeon RX 7900 XTX")
    );
    assert!(event_codes(&report).contains(&"backend.selected"));
    // Adapter VRAM and sysfs agree here, so no disagreement event.
    assert!(!event_codes(&report).contains(&"backend.vram_disagreement"));
    // Budget is the sysfs total minus the configured reserve.
    assert_eq!(
        report.vram_budget_bytes,
        Some(25_753_026_560 - BackendConfig::default().vram_reserve_bytes)
    );
}

#[test]
fn vram_disagreement_is_a_diag_event() {
    // The adapter claims 8 GiB while sysfs says the card holds
    // 24 GiB: selection still proceeds, but the disagreement is on
    // the record (BACKEND.md Section 2 cross-check).
    let mut xtx = discrete_xtx();
    xtx.vram_bytes = Some(8 * 1024 * 1024 * 1024);
    let probe = FakeProbe {
        adapters: vec![xtx],
        sysfs_vram: Some(25_753_026_560),
    };
    let report = detect(&BackendConfig::default(), &probe);
    assert_eq!(report.selected_kind(), Some(BackendKind::WgpuVulkan));
    assert!(event_codes(&report).contains(&"backend.vram_disagreement"));
}

#[test]
fn non_discrete_adapter_needs_explicit_config_allowance() {
    let probe = FakeProbe {
        adapters: vec![adapter(
            "Intel(R) UHD Graphics 770",
            AdapterBackend::Vulkan,
            DeviceType::IntegratedGpu,
        )],
        sysfs_vram: None,
    };
    // Default config: L1 is gated off, so CPU at L4.
    let report = detect(&BackendConfig::default(), &probe);
    assert_eq!(report.selected_kind(), Some(BackendKind::Cpu));
    // With the allowance: the integrated adapter at L1.
    let config = BackendConfig {
        allow_non_discrete_gpu: true,
        ..BackendConfig::default()
    };
    let report = detect(&config, &probe);
    assert_eq!(
        report.outcome,
        SelectionOutcome::Selected {
            kind: BackendKind::WgpuVulkan,
            level: FallbackLevel::L1,
        }
    );
    assert!(event_codes(&report).contains(&"backend.selected"));
}

#[test]
fn required_gpu_policy_refuses_instead_of_falling_back() {
    let config = BackendConfig {
        gpu_policy: GpuPolicy::Required,
        ..BackendConfig::default()
    };
    let probe = FakeProbe {
        adapters: vec![adapter("llvmpipe", AdapterBackend::Vulkan, DeviceType::Cpu)],
        sysfs_vram: None,
    };
    let report = detect(&config, &probe);
    assert_eq!(report.selected_kind(), None);
    assert!(matches!(report.outcome, SelectionOutcome::Refused { .. }));
    assert!(event_codes(&report).contains(&"backend.unavailable"));
}

#[test]
fn system_probe_reports_the_live_host_and_lands_on_cpu() {
    // This build links no wgpu stack, so the production probe
    // enumerates no adapters and Auto honestly lands on CPU. The
    // CPU facts must still be real.
    let report = detect(&BackendConfig::default(), &SystemProbe);
    assert!(report.cpu.logical_cores >= 1);
    assert_eq!(report.selected_kind(), Some(BackendKind::Cpu));
}

#[test]
fn cpu_only_selection_serves_a_full_op_chain() {
    // The T-4.1 proof in miniature: with the GPU disabled by
    // configuration (and a discrete adapter present), the selected
    // CPU backend runs a transformer-shaped chain end to end:
    // embed, project, norm, attend, gate, head, softmax.
    let probe = FakeProbe {
        adapters: vec![discrete_xtx()],
        sysfs_vram: Some(25_753_026_560),
    };
    let report = detect(&BackendConfig::cpu_only(), &probe);
    assert_eq!(report.selected_kind(), Some(BackendKind::Cpu));
    let backend = CpuBackend;

    let table = [1.0f32, 0.0, 0.0, 1.0];
    let hidden = backend.embedding(&table, 2, 2, 1).unwrap();
    assert_eq!(hidden, vec![0.0, 1.0]);

    let projected = backend
        .matvec(&[1.0, 2.0, 3.0, 4.0], 2, 2, &hidden)
        .unwrap();
    assert_eq!(projected, vec![2.0, 4.0]);

    let (normed, _residual) = backend
        .rms_norm_add(&projected, &[0.0, 0.0], &[1.0, 1.0], 1e-6)
        .unwrap();
    assert!(normed.iter().all(|v| v.is_finite()));

    let attended = backend
        .attention(&normed, &normed, &normed, 1, 1, 2, 0, None, None)
        .unwrap();
    assert_eq!(attended.len(), 2);

    let gated = backend.swiglu(&attended, &[1.0, 1.0]).unwrap();
    let logits = backend
        .matvec(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], 3, 2, &gated)
        .unwrap();
    let probs = backend.softmax(&logits).unwrap();
    let total: f32 = probs.iter().sum();
    assert!((total - 1.0).abs() < 1e-5, "softmax sums to {total}");
}
