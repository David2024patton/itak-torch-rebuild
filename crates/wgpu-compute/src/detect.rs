//! Adapter selection: the policy from BACKEND.md Section 2 and the
//! `detect()` that applies it.
//!
//! Selection is a policy, not a driver default. This host class
//! exposes three Vulkan adapters at once (the discrete 7900 XTX, an
//! Intel iGPU, and llvmpipe), and `PowerPreference::HighPerformance`
//! picking the right one is driver behavior, not a contract. So the
//! ranking is written down here, as a pure function over adapter
//! facts that the unit tests can pin:
//!
//! - Only Vulkan adapters are candidates (DECISIONS 2026-10-26:
//!   Vulkan is the GPU path; ROCm and HIP are parked).
//! - A discrete GPU always qualifies and always wins.
//! - An integrated or "other" adapter qualifies only when the
//!   policy explicitly allows non-discrete GPUs (BACKEND.md
//!   Section 6, level L1: entered by configuration, never by
//!   preference drift).
//! - A CPU-type adapter (llvmpipe) never qualifies, under any
//!   policy: BACKEND.md Section 6 counts a software adapter as no
//!   adapter, because it would report success while running slower
//!   than this workspace's own CPU backend.
//!
//! When nothing qualifies, detection fails with
//! [`WgpuError::NoSuitableAdapter`] listing every adapter seen. The
//! CPU backend is the caller's fallback (level L4), chosen by the
//! caller, not smuggled in here.

use crate::device::GpuContext;
use crate::error::{Result, WgpuError};

use crate::diag::dbg_log;

/// The facts about one adapter that the ranking reads. Kept as a
/// plain struct (not a live `wgpu::Adapter`) so the policy is
/// testable without hardware.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterFacts {
    /// Adapter name as the driver reports it.
    pub name: String,
    /// wgpu backend the adapter came from.
    pub backend: wgpu::Backend,
    /// wgpu device type classification.
    pub device_type: wgpu::DeviceType,
}

impl AdapterFacts {
    /// The facts of a live adapter.
    fn of(adapter: &wgpu::Adapter) -> Self {
        let info = adapter.get_info();
        AdapterFacts {
            name: info.name,
            backend: info.backend,
            device_type: info.device_type,
        }
    }

    /// One-line description for refusals and the diag channel.
    fn describe(&self) -> String {
        format!(
            "{} (backend {:?}, type {:?})",
            self.name, self.backend, self.device_type
        )
    }
}

/// Adapter selection policy. The default is the production stance:
/// discrete Vulkan or nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DetectPolicy {
    /// Allow integrated and other non-discrete Vulkan adapters when
    /// no discrete GPU qualifies (BACKEND.md Section 6, level L1).
    pub allow_non_discrete: bool,
}

/// Rank class of one adapter under `policy`; lower wins, `None`
/// disqualifies the adapter outright.
fn rank_class(facts: &AdapterFacts, policy: &DetectPolicy) -> Option<u8> {
    if facts.backend != wgpu::Backend::Vulkan {
        return None;
    }
    match facts.device_type {
        wgpu::DeviceType::DiscreteGpu => Some(0),
        wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::Other => {
            if policy.allow_non_discrete {
                Some(1)
            } else {
                None
            }
        }
        // A software rasterizer counts as no adapter (BACKEND.md
        // Section 6), and a virtual GPU is a host decision this
        // policy does not make silently either.
        wgpu::DeviceType::Cpu | wgpu::DeviceType::VirtualGpu => None,
    }
}

/// The index of the adapter the policy selects, or `None` when no
/// adapter qualifies. Pure: the unit tests pin the ranking here.
#[must_use]
pub fn rank_adapters(facts: &[AdapterFacts], policy: &DetectPolicy) -> Option<usize> {
    let mut best: Option<(u8, usize)> = None;
    for (index, adapter) in facts.iter().enumerate() {
        if let Some(class) = rank_class(adapter, policy) {
            if best.is_none_or(|(best_class, _)| class < best_class) {
                best = Some((class, index));
            }
        }
    }
    best.map(|(_, index)| index)
}

/// Enumerate this machine's Vulkan adapters, apply the policy, and
/// open a device on the winner with the adapter's own limits (not
/// wgpu's defaults: a default-limits device caps buffers far below
/// what RADV grants, BACKEND.md Section 2).
///
/// # Errors
///
/// [`WgpuError::NoSuitableAdapter`] when the policy qualifies
/// nothing; [`WgpuError::DeviceRequest`] when the winning adapter
/// cannot create a device; [`WgpuError::LimitShortfall`] when the
/// granted limits fall below this crate's kernel floor.
pub fn detect(policy: &DetectPolicy) -> Result<GpuContext> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    let adapters = instance.enumerate_adapters(wgpu::Backends::VULKAN);
    let facts: Vec<AdapterFacts> = adapters.iter().map(AdapterFacts::of).collect();
    for adapter in &facts {
        dbg_log!("adapter seen: {}", adapter.describe());
    }
    let Some(index) = rank_adapters(&facts, policy) else {
        return Err(WgpuError::NoSuitableAdapter {
            seen: facts.iter().map(AdapterFacts::describe).collect(),
            allow_non_discrete: policy.allow_non_discrete,
        });
    };
    let chosen = &facts[index];
    dbg_log!("adapter selected: {}", chosen.describe());
    GpuContext::open(&adapters[index], chosen.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(name: &str, backend: wgpu::Backend, device_type: wgpu::DeviceType) -> AdapterFacts {
        AdapterFacts {
            name: name.to_string(),
            backend,
            device_type,
        }
    }

    fn server_trio() -> Vec<AdapterFacts> {
        // The three adapters the BACKEND.md probe measured on the
        // real host, in probe order.
        vec![
            facts(
                "AMD Radeon RX 7900 XTX (RADV NAVI31)",
                wgpu::Backend::Vulkan,
                wgpu::DeviceType::DiscreteGpu,
            ),
            facts(
                "Intel(R) UHD Graphics 770 (ADL-S GT1)",
                wgpu::Backend::Vulkan,
                wgpu::DeviceType::IntegratedGpu,
            ),
            facts(
                "llvmpipe (LLVM 21.1.8, 256 bits)",
                wgpu::Backend::Vulkan,
                wgpu::DeviceType::Cpu,
            ),
        ]
    }

    #[test]
    fn discrete_vulkan_wins_on_the_server_trio() {
        let policy = DetectPolicy::default();
        assert_eq!(rank_adapters(&server_trio(), &policy), Some(0));
    }

    #[test]
    fn llvmpipe_alone_is_no_adapter() {
        let trio = server_trio();
        let only_pipe = &trio[2..];
        assert_eq!(rank_adapters(only_pipe, &DetectPolicy::default()), None);
        // Even the permissive policy never selects a CPU adapter.
        let permissive = DetectPolicy {
            allow_non_discrete: true,
        };
        assert_eq!(rank_adapters(only_pipe, &permissive), None);
    }

    #[test]
    fn integrated_needs_the_explicit_policy() {
        let trio = server_trio();
        let no_discrete = &trio[1..];
        assert_eq!(rank_adapters(no_discrete, &DetectPolicy::default()), None);
        let permissive = DetectPolicy {
            allow_non_discrete: true,
        };
        assert_eq!(rank_adapters(no_discrete, &permissive), Some(0));
    }

    #[test]
    fn non_vulkan_discrete_is_not_a_candidate() {
        let metal_card = vec![facts(
            "Some GPU",
            wgpu::Backend::Metal,
            wgpu::DeviceType::DiscreteGpu,
        )];
        assert_eq!(rank_adapters(&metal_card, &DetectPolicy::default()), None);
    }

    #[test]
    fn first_discrete_wins_ties() {
        let two = vec![
            facts(
                "Card A",
                wgpu::Backend::Vulkan,
                wgpu::DeviceType::DiscreteGpu,
            ),
            facts(
                "Card B",
                wgpu::Backend::Vulkan,
                wgpu::DeviceType::DiscreteGpu,
            ),
        ];
        assert_eq!(rank_adapters(&two, &DetectPolicy::default()), Some(0));
    }
}
