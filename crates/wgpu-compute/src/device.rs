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
/// any kernel binds whole is K3's output head at 924,571,648 bytes
/// (about 0.86 GiB); the floor rounds that up to 1 GiB so a device
/// that cannot bind the head whole refuses up front, at device
/// opening, instead of mid-load. (The K1-only floor was 128 MiB,
/// sized to an FFN gate at 74,760,192 bytes; K3 landing is what
/// raised it, as this constant's earlier note anticipated. The
/// RADV adapter grants 2 GiB.)
pub const MIN_BINDING_BYTES: u64 = 1024 * 1024 * 1024;

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

    /// Create a storage buffer of `size` bytes with copy access in
    /// both directions: the scratch shape every one-shot kernel in
    /// this crate allocates for its inputs and outputs. (The
    /// resident weight buffers of BACKEND.md Section 5 are created
    /// by the kernels that own them; this helper is for the
    /// per-call activations the `Compute` trait hands over as host
    /// slices.)
    pub fn scratch_buffer(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Upload raw bytes into a fresh storage buffer (see
    /// [`GpuContext::scratch_buffer`]).
    pub fn upload_bytes(&self, label: &str, bytes: &[u8]) -> wgpu::Buffer {
        let buffer = self.scratch_buffer(label, bytes.len() as u64);
        self.queue.write_buffer(&buffer, 0, bytes);
        buffer
    }

    /// Upload f32 values into a fresh storage buffer.
    pub fn upload_f32(&self, label: &str, data: &[f32]) -> wgpu::Buffer {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.upload_bytes(label, &bytes)
    }

    /// Upload u32 values into a fresh storage buffer.
    pub fn upload_u32(&self, label: &str, data: &[u32]) -> wgpu::Buffer {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.upload_bytes(label, &bytes)
    }

    /// Create a uniform buffer from raw parameter bytes (the
    /// kernels' `Params` structs, laid out by hand in each kernel
    /// module).
    pub fn uniform_buffer(&self, label: &str, bytes: &[u8]) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (bytes.len() as u64).max(4),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, bytes);
        buffer
    }

    /// Read `count` f32 values back from a storage buffer: copy to
    /// a staging buffer, map, collect. This is the readback half of
    /// the trait's host-slice currency, shared by every one-shot
    /// kernel (K1's `run` keeps its own copy because the bench
    /// holds its buffers resident across iterations).
    ///
    /// # Errors
    ///
    /// [`WgpuError::Readback`] when the map fails or never fires.
    pub fn read_f32(&self, buffer: &wgpu::Buffer, count: usize) -> Result<Vec<f32>> {
        let bytes = (count * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback staging"),
            size: bytes.max(4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback copy"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
        self.queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| WgpuError::Readback {
                detail: e.to_string(),
            })?;
        rx.recv()
            .map_err(|_| WgpuError::Readback {
                detail: "map callback never fired".to_string(),
            })?
            .map_err(|e| WgpuError::Readback {
                detail: e.to_string(),
            })?;
        let mapped = slice.get_mapped_range();
        let out: Vec<f32> = mapped
            .chunks_exact(4)
            .take(count)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        drop(mapped);
        staging.unmap();
        Ok(out)
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
