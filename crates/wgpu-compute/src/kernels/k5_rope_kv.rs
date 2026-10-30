//! K5: RoPE rotation plus KV append (BACKEND.md Section 3).
//!
//! Two entry points, used back to back by the attention step:
//!
//! - **Rotate** ([`K5Pipeline::rotate`]): in-place interleaved
//!   (adjacent-pair, ggml NORM) RoPE over the rows of a projection
//!   buffer, one row per sequence position, each head of
//!   `head_dim` channels rotated at its absolute position. The
//!   pairing and the angle law are `torch-math`'s
//!   `rope_interleaved` exactly: pair `i` joins channels
//!   `(2i, 2i + 1)` and turns by `position * base^(-2i / head_dim)`.
//!   The base is a parameter (the SWA base the config resolved,
//!   `rope_freq_base_swa` with its train-base fallback), never a
//!   constant. Rotation runs on the sliding-window layers only;
//!   the global layers are NoPE and the caller simply does not
//!   call this kernel for them.
//! - **Append** ([`K5Pipeline::append`]): copy the rotated K rows
//!   and the V rows from their per-call staging buffers into the
//!   layer's resident KV buffers (K6's storage, see
//!   [`crate::kernels::k6_attention`]), at the slot the cache
//!   semantics assign: the position itself for global layers,
//!   `position % window` (the ring slot) for sliding layers. The
//!   slot law is the `attention` crate's `slot_of`, restated in
//!   the shader.
//!
//! Numerical note: `torch-math` computes its angles in f64; the
//! shader computes in f32 (`pow`, `sin`, `cos`). At Glimmer's
//! positions and base the two agree to float noise far below the
//! cross-backend tolerance the crate's tests assert; the
//! difference is stated here rather than hidden, the same
//! disclosure K4 makes for its f32 sum of squares.
//!
//! The rotation is deliberately NOT fused into the projection
//! epilogue in this landing (BACKEND.md allows either): the
//! `Compute` trait hands the attention step whole Q/K/V buffers,
//! so a standalone rotate over those buffers is the honest shape
//! of the seam as it exists today.

use crate::device::GpuContext;
use crate::error::{Result, WgpuError};

use crate::diag::dbg_log;

/// Threads per workgroup for both K5 kernels.
const WORKGROUP: u32 = 256;

/// The rotate shader: one thread per (row, head, pair).
const ROTATE_SHADER: &str = r#"
struct RotateParams {
    width: u32,
    head_dim: u32,
    total_pairs: u32,
    start_position: u32,
    base: f32,
    grid_x: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: RotateParams;

@compute @workgroup_size(256)
fn rotate_main(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>) {
    let flat = (wid.y * params.grid_x + wid.x) * 256u + lid.x;
    if (flat >= params.total_pairs) {
        return;
    }
    let pairs_per_row = params.width / 2u;
    let row = flat / pairs_per_row;
    let within = flat % pairs_per_row;
    let pairs_per_head = params.head_dim / 2u;
    let head = within / pairs_per_head;
    let pair = within % pairs_per_head;

    let position = f32(params.start_position + row);
    let angle = position * pow(params.base, -2.0 * f32(pair) / f32(params.head_dim));
    let c = cos(angle);
    let s = sin(angle);
    let i0 = row * params.width + head * params.head_dim + 2u * pair;
    let a = data[i0];
    let b = data[i0 + 1u];
    data[i0] = a * c - b * s;
    data[i0 + 1u] = a * s + b * c;
}
"#;

/// The append shader: one thread per element of the appended
/// rows. `window` is 0 for global layers (slot = position) and
/// the ring size for sliding layers (slot = position % window).
const APPEND_SHADER: &str = r#"
struct AppendParams {
    kv_width: u32,
    total: u32,
    start_position: u32,
    window: u32,
    grid_x: u32,
}

@group(0) @binding(0) var<storage, read> k_src: array<f32>;
@group(0) @binding(1) var<storage, read> v_src: array<f32>;
@group(0) @binding(2) var<storage, read_write> k_dst: array<f32>;
@group(0) @binding(3) var<storage, read_write> v_dst: array<f32>;
@group(0) @binding(4) var<uniform> params: AppendParams;

@compute @workgroup_size(256)
fn append_main(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>) {
    let flat = (wid.y * params.grid_x + wid.x) * 256u + lid.x;
    if (flat >= params.total) {
        return;
    }
    let row = flat / params.kv_width;
    let channel = flat % params.kv_width;
    let position = params.start_position + row;
    var slot = position;
    if (params.window != 0u) {
        slot = position % params.window;
    }
    let dst = slot * params.kv_width + channel;
    k_dst[dst] = k_src[flat];
    v_dst[dst] = v_src[flat];
}
"#;

/// The compiled K5 pipelines (rotate and append).
#[derive(Debug)]
pub struct K5Pipeline {
    rotate: wgpu::ComputePipeline,
    append: wgpu::ComputePipeline,
}

impl K5Pipeline {
    /// Compile both K5 pipelines on `device`.
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] when wgpu rejects a shader.
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        let build = |source: &str, entry: &'static str, label: &'static str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let rotate = build(ROTATE_SHADER, "rotate_main", "k5 rope rotate");
        let append = build(APPEND_SHADER, "append_main", "k5 kv append");
        dbg_log!("k5 pipelines built");
        Ok(K5Pipeline { rotate, append })
    }

    /// Rotate every row of `buffer` in place: `rows` rows of
    /// `heads * head_dim` channels, row `r` at absolute position
    /// `start_position + r`, interleaved pairing, the given base.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when the geometry is unusable
    /// (odd or zero head dim, zero rows).
    #[allow(clippy::too_many_arguments)]
    pub fn rotate(
        &self,
        ctx: &GpuContext,
        buffer: &wgpu::Buffer,
        rows: usize,
        heads: usize,
        head_dim: usize,
        start_position: usize,
        base: f32,
    ) -> Result<()> {
        if rows == 0 || heads == 0 || head_dim < 2 || head_dim % 2 != 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k5 rotate".to_string(),
                detail: format!(
                    "rows {rows}, heads {heads}, head_dim {head_dim}: rotation needs \
                     at least one row and head and an even head_dim of at least 2"
                ),
            });
        }
        if !base.is_finite() || base <= 1.0 {
            return Err(WgpuError::InvalidTensor {
                name: "k5 rotate".to_string(),
                detail: format!("rope base {base} is not finite and greater than 1"),
            });
        }
        let width = heads * head_dim;
        let total_pairs = (rows * width / 2) as u32;
        let total_wg = u64::from(total_pairs).div_ceil(u64::from(WORKGROUP));
        let grid = crate::kernels::k1_q4k_gemv::grid_dims(total_wg);
        let params: [u32; 6] = [
            width as u32,
            head_dim as u32,
            total_pairs,
            start_position as u32,
            base.to_bits(),
            grid.0,
        ];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buf = ctx.uniform_buffer("k5 rotate params", &param_bytes);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k5 rotate bind group"),
            layout: &self.rotate.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k5 rotate run"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("k5 rope rotate"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.rotate);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(grid.0, grid.1, 1);
        }
        ctx.queue.submit([encoder.finish()]);
        dbg_log!(
            "k5 rotate: rows={rows} heads={heads} head_dim={head_dim} start={start_position} base={base}"
        );
        Ok(())
    }

    /// Append `rows` rows of K and V from the staging buffers
    /// into the layer's resident KV buffers, starting at absolute
    /// position `start_position`. `window` is the layer's ring
    /// size, or `None` for a global layer (slot = position).
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when `rows` or `kv_width` is
    /// zero.
    #[allow(clippy::too_many_arguments)]
    pub fn append(
        &self,
        ctx: &GpuContext,
        k_src: &wgpu::Buffer,
        v_src: &wgpu::Buffer,
        k_dst: &wgpu::Buffer,
        v_dst: &wgpu::Buffer,
        rows: usize,
        kv_width: usize,
        start_position: usize,
        window: Option<usize>,
    ) -> Result<()> {
        if rows == 0 || kv_width == 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k5 append".to_string(),
                detail: format!("rows {rows}, kv_width {kv_width}: nothing to append"),
            });
        }
        let total = (rows * kv_width) as u32;
        let total_wg = u64::from(total).div_ceil(u64::from(WORKGROUP));
        let grid = crate::kernels::k1_q4k_gemv::grid_dims(total_wg);
        let params: [u32; 5] = [
            kv_width as u32,
            total,
            start_position as u32,
            window.map_or(0, |w| w as u32),
            grid.0,
        ];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buf = ctx.uniform_buffer("k5 append params", &param_bytes);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k5 append bind group"),
            layout: &self.append.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: k_src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: v_src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: k_dst.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: v_dst.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k5 append run"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("k5 kv append"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.append);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(grid.0, grid.1, 1);
        }
        ctx.queue.submit([encoder.finish()]);
        dbg_log!(
            "k5 append: rows={rows} kv_width={kv_width} start={start_position} window={window:?}"
        );
        Ok(())
    }
}

/// The ring slot of an absolute position, restating the
/// `attention` crate's `slot_of` for the host side of this
/// module's tests and callers: the position itself for global
/// layers, `position % window` for windowed ones.
#[must_use]
pub fn slot_of(position: usize, window: Option<usize>) -> usize {
    match window {
        None => position,
        Some(w) => position % w,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_of_matches_the_cache_law() {
        assert_eq!(slot_of(7, None), 7);
        assert_eq!(slot_of(7, Some(2048)), 7);
        assert_eq!(slot_of(2049, Some(2048)), 1);
        assert_eq!(slot_of(4096, Some(2048)), 0);
    }

    #[test]
    fn shaders_carry_their_entry_points() {
        assert!(ROTATE_SHADER.contains("fn rotate_main"));
        assert!(APPEND_SHADER.contains("fn append_main"));
        // The NORM pairing is adjacent channels, channels 2i and
        // 2i+1 of one head: the shader must index exactly that.
        assert!(ROTATE_SHADER.contains("2u * pair"));
    }
}
