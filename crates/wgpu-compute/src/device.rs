//! An opened GPU device: the selected adapter's identity, the
//! `wgpu` device and queue, and the granted limits the kernels are
//! checked against.
//!
//! The device is created with the ADAPTER's limits, not wgpu's
//! defaults (BACKEND.md Section 2: a default-limits device caps
//! buffers far below what RADV grants). The granted limits are then
//! verified against this crate's floor before any kernel runs: the
//! largest K1 tensor (an FFN gate at 74.8 MB) must bind whole, and
//! the dispatch scheme needs real workgroup range. A shortfall is a
//! named error here, BACKEND.md Section 6's fallback event, never a
//! surprise mid-load.

use crate::detect::AdapterFacts;
use crate::error::{Result, WgpuError};

use crate::diag::dbg_log;

/// Floor for `max_storage_buffer_binding_size`: the largest tensor
/// the K1 kernel binds whole is a Glimmer FFN gate or up projection
/// at 74,760,192 bytes; the floor rounds that up to 128 MiB so a
/// device that cannot bind one whole FFN tensor refuses up front.
/// (K3's output head, about 0.86 GiB, will raise this floor when
/// that kernel lands; the RADV adapter grants 2 GiB.)
pub const MIN_BINDING_BYTES: u64 = 128 * 1024 * 1024;

/// Floor for `max_compute_workgroups_per_dimension`: the K1 grid
/// scheme addresses rows through two dimensions and needs only a
/// modest first dimension, but a token floor of 1 workgroup would
/// make the scheme meaningless; 4,096 is far below every real
/// adapter (RADV grants 65,535) and far above a broken one.
pub const MIN_WORKGROUPS_PER_DIMENSION: u32 = 4_096;

/// One opened GPU: adapter identity plus device and queue.
#[derive(Debug)]
pub struct GpuContext {
    /// The adapter the policy selected, as facts.
    pub adapter: AdapterFacts,
    /// The opened device, created with the adapter's limits.
    pub device: wgpu::Device,
    /// The device's submission queue.
    pub queue: wgpu::Queue,
    /// Granted `max_buffer_size`, recorded for upload checks and
    /// the residency accounting in the bench reports.
    pub max_buffer_size: u64,
    /// Granted `max_storage_buffer_binding_size`.
    pub max_binding_size: u64,
    /// Features the device was actually granted (the subset of
    /// the adapter's features this crate asked for: SUBGROUP and
    /// SHADER_F16, the two the tuned K1 shapes select on).
    /// Kernels read this, through `device.features()`, at
    /// pipeline construction; a shape whose feature is absent
    /// refuses by name there.
    pub features: wgpu::Features,
}

impl GpuContext {
    /// Open a device on `adapter` with the adapter's own limits and
    /// verify the granted limits against this crate's floor.
    pub(crate) fn open(adapter: &wgpu::Adapter, facts: AdapterFacts) -> Result<Self> {
        // Ask for the features the tuned kernels select on, when
        // the adapter offers them (BACKEND.md Section 2 measured
        // both present on the server stack). Asking only for what
        // the adapter has keeps device creation infallible on
        // lesser stacks; the kernels then fall back by name at
        // pipeline construction (see shipped_shape).
        let wanted = wgpu::Features::SUBGROUP | wgpu::Features::SHADER_F16;
        let required_features = adapter.features() & wanted;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu-compute device"),
            required_features,
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .map_err(|e| WgpuError::DeviceRequest {
            detail: e.to_string(),
        })?;
        let features = device.features();
        dbg_log!(
            "device features granted: subgroup={} shader_f16={}",
            features.contains(wgpu::Features::SUBGROUP),
            features.contains(wgpu::Features::SHADER_F16)
        );
        let limits = device.limits();
        // wgpu reports the binding cap as u32; this crate's
        // accounting is u64 throughout, so widen once, here.
        let binding_size = u64::from(limits.max_storage_buffer_binding_size);
        dbg_log!(
            "device opened on {}: max_buffer_size={} max_binding={} max_workgroups_per_dim={}",
            facts.name,
            limits.max_buffer_size,
            binding_size,
            limits.max_compute_workgroups_per_dimension
        );
        if binding_size < MIN_BINDING_BYTES {
            return Err(WgpuError::LimitShortfall {
                what: "max_storage_buffer_binding_size",
                required: MIN_BINDING_BYTES,
                granted: binding_size,
            });
        }
        if limits.max_compute_workgroups_per_dimension < MIN_WORKGROUPS_PER_DIMENSION {
            return Err(WgpuError::LimitShortfall {
                what: "max_compute_workgroups_per_dimension",
                required: u64::from(MIN_WORKGROUPS_PER_DIMENSION),
                granted: u64::from(limits.max_compute_workgroups_per_dimension),
            });
        }
        if limits.max_buffer_size < binding_size {
            return Err(WgpuError::LimitShortfall {
                what: "max_buffer_size (below the granted binding size)",
                required: binding_size,
                granted: limits.max_buffer_size,
            });
        }
        Ok(GpuContext {
            adapter: facts,
            max_buffer_size: limits.max_buffer_size,
            max_binding_size: binding_size,
            features,
            device,
            queue,
        })
    }

    /// Verify a whole-tensor upload of `bytes` fits this device,
    /// naming the tensor and both numbers when it does not. This is
    /// the residency check the per-tensor buffer strategy
    /// (BACKEND.md Section 5) needs: one buffer per tensor, bound
    /// whole, or a refusal by name.
    pub fn check_tensor_fits(&self, name: &str, bytes: u64) -> Result<()> {
        if bytes > self.max_buffer_size {
            return Err(WgpuError::InvalidTensor {
                name: name.to_string(),
                detail: format!(
                    "tensor holds {bytes} bytes but the device caps a buffer at {}",
                    self.max_buffer_size
                ),
            });
        }
        if bytes > self.max_binding_size {
            return Err(WgpuError::InvalidTensor {
                name: name.to_string(),
                detail: format!(
                    "tensor holds {bytes} bytes but a single binding caps at {}",
                    self.max_binding_size
                ),
            });
        }
        Ok(())
    }
}
