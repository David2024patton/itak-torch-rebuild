//! K1: Q4_K dequantize-and-multiply GEMV (BACKEND.md Section 3).
//!
//! The bulk of the decode stream: 77.3 percent of Glimmer's bytes
//! per token are Q4_K (FFN gate and up, attention Q and K
//! projections, the embedding table's rows through `proj`-shaped
//! calls). This kernel's GB/s is effectively the backend's GB/s, so
//! it is the only kernel this landing implements; the rest sit
//! behind named refusals in their own modules.
//!
//! The shader reads the ggml block layout directly from the
//! tensor's raw bytes (144 bytes per 256-element super-block: f16
//! `d`, f16 `dmin`, 12 packed scale/min bytes, 128 nibble bytes),
//! expanding values in registers and never materializing a
//! dequantized weight anywhere, per BACKEND.md Section 4. The
//! dequantization arithmetic mirrors `gguf::dequant::dequant_q4_k`
//! term for term, including `get_scale_min_k4`'s 6-bit unpacking, so
//! GPU results can be checked against the CPU reference within
//! float reassociation noise.
//!
//! Parallel shape: one workgroup per (batch row, output row),
//! one wave of 32 lanes, one lane per super-block (Glimmer's rows
//! are 26 blocks; a wider row strides). Everything moves as
//! `vec4`: a super-block is exactly nine `vec4<u32>` (one header
//! vec4 with `d`, `dmin`, and all 12 scale bytes, then eight
//! vec4 of nibbles), and the activation span of a block is 64
//! `vec4<f32>`. A workgroup reduction over LDS produces the
//! row's dot product. The grid is two-dimensional because one
//! dispatch dimension caps at 65,535 workgroups (BACKEND.md
//! Section 2): the flat workgroup index wraps over `grid_x`,
//! which matters for the 202,048-row head class of shapes even
//! though K1's own widest tensor (FFN at 19,968 rows) fits in
//! one dimension.
//!
//! The tuning history is recorded in BACKEND.md's results
//! section; the summary is that the scalar first version
//! streamed at 148.6 GB/s, this vec4 form at 216.5 GB/s, and
//! BACKEND.md Section 4's tiled variant (four rows per
//! workgroup, activations staged in workgroup memory) was built
//! and measured at 147.6 GB/s, slower than this form, so it was
//! not kept: the staging barrier and workgroup-memory reads cost
//! more than the shared activation loads they saved, at these
//! shapes. A stream probe with the grid and access pattern but
//! no dequant math reached 746.9 GB/s, so the remaining gap to
//! llama.cpp's 708 GB/s lives in the dequant instruction mix,
//! not in the wgpu stack or the grid.

use crate::device::GpuContext;
use crate::error::{Result, WgpuError};

use crate::diag::dbg_log;

/// Elements per Q4_K super-block.
const BLOCK_ELEMS: usize = 256;
/// Bytes per Q4_K super-block.
const BLOCK_BYTES: usize = 144;
/// wgpu's per-dimension dispatch cap on this stack.
const MAX_WORKGROUPS_PER_DIM: u32 = 65_535;

/// The K1 shader, in WGSL. See the module docs for the layout it
/// reads and the parallel shape it runs.
const SHADER: &str = r#"
struct Params {
    blocks_per_row: u32,
    out_rows: u32,
    grid_x: u32,
    batch: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partials: array<f32, 32>;

// IEEE 754 half to f32, mirroring gguf::dequant::f16_to_f32 for
// every value a weight file can hold (normals, subnormals, zero;
// inf and NaN payloads keep their class).
fn f16_to_f32(h: u32) -> f32 {
    let sign = (h >> 15u) & 1u;
    let exp = (h >> 10u) & 31u;
    let frac = h & 1023u;
    if (exp == 31u) {
        return bitcast<f32>((sign << 31u) | 0x7f800000u | (frac << 13u));
    }
    if (exp == 0u) {
        // Subnormal half: value is frac * 2^-24, exact in f32.
        let v = f32(frac) * 5.960464477539063e-08;
        if (sign == 1u) { return -v; }
        return v;
    }
    return bitcast<f32>((sign << 31u) | ((exp + 112u) << 23u) | (frac << 13u));
}

// One of the 12 packed scale bytes of a super-block.
fn scale_byte(sw: array<u32, 3>, i: u32) -> u32 {
    return (sw[i / 4u] >> (8u * (i % 4u))) & 0xffu;
}

// ggml get_scale_min_k4: sub-block j's (scale, min) 6-bit pair.
fn scale_min_k4(j: u32, sw: array<u32, 3>) -> vec2<u32> {
    if (j < 4u) {
        return vec2<u32>(scale_byte(sw, j) & 63u, scale_byte(sw, j + 4u) & 63u);
    }
    return vec2<u32>(
        (scale_byte(sw, j + 4u) & 0xfu) | ((scale_byte(sw, j - 4u) >> 6u) << 4u),
        (scale_byte(sw, j + 4u) >> 4u) | ((scale_byte(sw, j) >> 6u) << 4u),
    );
}

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let flat = wid.y * params.grid_x + wid.x;
    // Whole-workgroup early out: every lane shares `flat`, so the
    // exit is uniform and the barriers below stay legal.
    if (flat >= params.out_rows * params.batch) {
        return;
    }
    let out_row = flat % params.out_rows;
    let batch_row = flat / params.out_rows;
    let lane = lid.x;

    // One lane per super-block (striding when a row has more
    // blocks than lanes). A block is nine vec4 words: header,
    // then eight of nibbles. The activation span of a block is
    // 64 vec4s, and every output row of a batch row reads the
    // same input vector.
    let row_block0 = out_row * params.blocks_per_row;
    let x_row_base = batch_row * params.blocks_per_row * 64u;

    var acc = 0.0;
    for (var blk = lane; blk < params.blocks_per_row; blk += 32u) {
        let bw = (row_block0 + blk) * 9u;
        let hdr = weights[bw];
        let d = f16_to_f32(hdr.x & 0xffffu);
        let dmin = f16_to_f32((hdr.x >> 16u) & 0xffffu);
        let sw = array<u32, 3>(hdr.y, hdr.z, hdr.w);
        let x_base = x_row_base + blk * 64u;
        // Four groups of 64 elements. Group g's nibble bytes are
        // vec4s 2g and 2g+1 after the header; within a byte, the
        // low nibble is element l and the high nibble element
        // l + 32 of the group, exactly ggml's grouping.
        for (var grp = 0u; grp < 4u; grp++) {
            let sm1 = scale_min_k4(2u * grp, sw);
            let sm2 = scale_min_k4(2u * grp + 1u, sw);
            let d1 = d * f32(sm1.x);
            let dm1 = dmin * f32(sm1.y);
            let d2 = d * f32(sm2.x);
            let dm2 = dmin * f32(sm2.y);
            var sum_qx_lo = 0.0;
            var sum_x_lo = 0.0;
            var sum_qx_hi = 0.0;
            var sum_x_hi = 0.0;
            // A group is 32 nibble bytes in two vec4s. Byte l of
            // the group pairs with activation l (low nibble) and
            // l + 32 (high nibble). The low span of a half is 16
            // activations = 4 vec4s; the high span sits 8 vec4s
            // later. All loop bounds are literal so the compiler
            // unrolls and the vec4 loads stay single instructions.
            for (var half = 0u; half < 2u; half++) {
                let qv = weights[bw + 1u + grp * 2u + half];
                for (var c = 0u; c < 4u; c++) {
                    let word = qv[c];
                    let xv_lo = input[x_base + grp * 16u + half * 4u + c];
                    let xv_hi = input[x_base + grp * 16u + 8u + half * 4u + c];
                    for (var j = 0u; j < 4u; j++) {
                        let qbyte = (word >> (8u * j)) & 0xffu;
                        let x_lo = xv_lo[j];
                        let x_hi = xv_hi[j];
                        sum_qx_lo += f32(qbyte & 0xfu) * x_lo;
                        sum_x_lo += x_lo;
                        sum_qx_hi += f32(qbyte >> 4u) * x_hi;
                        sum_x_hi += x_hi;
                    }
                }
            }
            acc += d1 * sum_qx_lo - dm1 * sum_x_lo + d2 * sum_qx_hi - dm2 * sum_x_hi;
        }
    }

    partials[lane] = acc;
    workgroupBarrier();
    var stride = 16u;
    while (stride > 0u) {
        if (lane < stride) {
            partials[lane] += partials[lane + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    if (lane == 0u) {
        output[batch_row * params.out_rows + out_row] = partials[0];
    }
}
"#;

/// Grid dimensions `(x, y)` for `total` workgroups under the
/// per-dimension cap: x takes as many as it may, y takes the rest,
/// and the shader recovers the flat index as `y * grid_x + x`.
#[must_use]
pub fn grid_dims(total: u64) -> (u32, u32) {
    if total == 0 {
        return (0, 0);
    }
    let x = total.min(u64::from(MAX_WORKGROUPS_PER_DIM)) as u32;
    let y = total.div_ceil(u64::from(x));
    (x, y.min(u64::from(u32::MAX)) as u32)
}

/// A Q4_K weight tensor resident on the device: one storage buffer
/// holding the tensor's raw GGUF bytes, bound whole per dispatch
/// (BACKEND.md Section 5's per-tensor buffer strategy).
#[derive(Debug)]
pub struct K1Weights {
    /// Whole-tensor byte length, for the bench's stream accounting.
    pub byte_len: u64,
    /// Input width (elements per row); a whole number of blocks.
    pub in_features: usize,
    /// Output rows.
    pub out_rows: usize,
    buffer: wgpu::Buffer,
}

/// One prepared K1 dispatch set: a bind group plus grid dimensions,
/// built once for a (weights, input, output, batch) combination and
/// encodable any number of times. The bench encodes the same call
/// repeatedly; `run` encodes it once.
#[derive(Debug)]
pub struct K1Call {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    grid: (u32, u32),
}

impl K1Call {
    /// Record this call's dispatch into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k1 q4_k gemv"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.grid.0, self.grid.1, 1);
    }
}

/// The compiled K1 pipeline. Compilation happens once, at backend
/// construction, so Naga translation cost stays out of the token
/// loop (BACKEND.md Section 4).
#[derive(Debug)]
pub struct K1Pipeline {
    pipeline: wgpu::ComputePipeline,
}

impl K1Pipeline {
    /// Compile the shader and build the pipeline on `device`.
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] if wgpu rejects the module or the
    /// pipeline. (wgpu reports shader errors through its error
    /// handling rather than a return value on some paths; the
    /// crate's conformance tests compile this pipeline on whatever
    /// adapter a machine offers, which is where a WGSL defect
    /// would surface first.)
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("k1 q4_k gemv shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k1 q4_k gemv pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Ok(K1Pipeline { pipeline })
    }

    /// Upload a Q4_K tensor's raw bytes into a fresh device buffer.
    /// The geometry is validated against the bytes before anything
    /// is uploaded: a mismatch is a refusal naming the tensor,
    /// never a truncated or padded upload.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] on any geometry or byte-count
    /// disagreement; [`WgpuError::LimitShortfall`] through
    /// [`GpuContext::check_tensor_fits`] when the device cannot
    /// hold or bind the tensor whole.
    pub fn upload(
        &self,
        ctx: &GpuContext,
        name: &str,
        bytes: &[u8],
        in_features: usize,
        out_rows: usize,
    ) -> Result<K1Weights> {
        if in_features == 0 || in_features % BLOCK_ELEMS != 0 {
            return Err(WgpuError::InvalidTensor {
                name: name.to_string(),
                detail: format!(
                    "input width {in_features} is not a positive whole number of {BLOCK_ELEMS}-element Q4_K blocks"
                ),
            });
        }
        if out_rows == 0 {
            return Err(WgpuError::InvalidTensor {
                name: name.to_string(),
                detail: "output row count is zero".to_string(),
            });
        }
        let expected = (in_features / BLOCK_ELEMS) * BLOCK_BYTES * out_rows;
        if bytes.len() != expected {
            return Err(WgpuError::InvalidTensor {
                name: name.to_string(),
                detail: format!(
                    "tensor holds {} bytes but [{in_features}, {out_rows}] of Q4_K needs {expected}",
                    bytes.len()
                ),
            });
        }
        ctx.check_tensor_fits(name, bytes.len() as u64)?;
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&buffer, 0, bytes);
        dbg_log!(
            "k1 upload {name}: {} bytes [{in_features} x {out_rows}]",
            bytes.len()
        );
        Ok(K1Weights {
            byte_len: bytes.len() as u64,
            in_features,
            out_rows,
            buffer,
        })
    }

    /// Prepare a dispatch of `weights` over a batch of input rows.
    /// `input` must hold `batch * weights.in_features` f32 and
    /// `output` must hold `batch * weights.out_rows` f32; both are
    /// caller-owned device buffers so the bench can hold them
    /// resident across iterations.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when the buffer sizes disagree
    /// with the batch the caller declares.
    pub fn prepare(
        &self,
        ctx: &GpuContext,
        weights: &K1Weights,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        batch: u32,
    ) -> Result<K1Call> {
        if batch == 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k1 call".to_string(),
                detail: "batch is zero".to_string(),
            });
        }
        let params: [u32; 4] = [
            (weights.in_features / BLOCK_ELEMS) as u32,
            weights.out_rows as u32,
            0, // grid_x, filled below once the grid is known
            batch,
        ];
        let total = u64::from(batch) * weights.out_rows as u64;
        let grid = grid_dims(total);
        let mut params = params;
        params[2] = grid.0;
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k1 params"),
            size: param_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&params_buffer, 0, &param_bytes);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k1 bind group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weights.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        Ok(K1Call {
            pipeline: self.pipeline.clone(),
            bind_group,
            grid,
        })
    }

    /// Convenience one-shot: upload `input` ([`batch`, in] rows),
    /// run the GEMV, read the [`batch`, out] result back to host.
    /// This is the `Compute::proj` data path; it pays an upload and
    /// a readback per call, which the crate docs state plainly.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when `input` is not a whole
    /// number of input rows; [`WgpuError::Readback`] when the
    /// result cannot be mapped back.
    pub fn run(&self, ctx: &GpuContext, weights: &K1Weights, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() || input.len() % weights.in_features != 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k1 input".to_string(),
                detail: format!(
                    "input holds {} elements, not a whole number of {}-wide rows",
                    input.len(),
                    weights.in_features
                ),
            });
        }
        let batch = (input.len() / weights.in_features) as u32;
        let in_bytes: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k1 input"),
            size: in_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&input_buffer, 0, &in_bytes);
        let out_elems = batch as usize * weights.out_rows;
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k1 output"),
            size: (out_elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k1 readback"),
            size: (out_elems * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let call = self.prepare(ctx, weights, &input_buffer, &output_buffer, batch)?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k1 run"),
            });
        call.encode(&mut encoder);
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, (out_elems * 4) as u64);
        ctx.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        ctx.device
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
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        drop(mapped);
        staging.unmap();
        dbg_log!(
            "k1 run: batch={batch} out_rows={} bytes_streamed={}",
            weights.out_rows,
            weights.byte_len
        );
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_dims_single_dimension_when_rows_fit() {
        // The widest K1 tensor: FFN gate/up at 19,968 rows.
        assert_eq!(grid_dims(19_968), (19_968, 1));
        assert_eq!(grid_dims(65_535), (65_535, 1));
    }

    #[test]
    fn grid_dims_wraps_to_two_dimensions_past_the_cap() {
        // The output head's 202,048 rows, the case the second
        // dimension exists for.
        let (x, y) = grid_dims(202_048);
        assert_eq!(x, 65_535);
        assert_eq!(y, 4);
        assert!(u64::from(x) * u64::from(y) >= 202_048);
        assert_eq!(grid_dims(65_536), (65_535, 2));
    }

    #[test]
    fn grid_dims_zero_is_zero() {
        assert_eq!(grid_dims(0), (0, 0));
    }
}
