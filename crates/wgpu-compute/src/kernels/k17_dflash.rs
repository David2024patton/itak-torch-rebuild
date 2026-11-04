//! K17: the two device primitives the DFlash draft forward
//! needs that no existing kernel provides.
//!
//! - **NEOX rotate.** K5's rotate is the interleaved
//!   (GPT-J) pairing Glimmer's arch uses; the draft's arch
//!   rotates with the half-split (GPT-NeoX) pairing of
//!   `torch_math::rope`: pair `i` joins channel `i` with
//!   channel `i + head_dim/2`, at angle
//!   `position * base^(-2i/head_dim)`. The inverse
//!   frequencies are computed on the host in f64 and stored
//!   f32, exactly the K5 convention, and the row positions
//!   come from a device array, because the draft's two
//!   rotate sites see different position sources (an inject
//!   chunk's caller-supplied positions, a block's
//!   `start..start+rows` span).
//! - **Draft attention.** K6 is causal, ring-addressed, and
//!   Glimmer-shaped; the draft's attention is (on the real
//!   file) non-causal, runs over a position-addressed
//!   committed set that can be sparse (the CPU cache's
//!   BTreeMap semantics), and mixes those committed rows
//!   with the current block's own K/V in one visible set,
//!   under a per-layer sliding-window mask stated over
//!   absolute positions. This kernel is that exact visible
//!   set: one workgroup per (query row, query head), the
//!   GQA group folded into the workgroup's KV head, an
//!   online softmax with f32 accumulation throughout (the
//!   same math as the CPU implementation's two-pass
//!   softmax, reassociated), and the mask evaluated over
//!   absolute positions with the CPU implementation's own
//!   comparisons.
//!
//! Everything else in the draft forward composes existing
//! kernels (K1/K2/K3 projections, K4 norms and residual
//! adds, K7 SwiGLU, K8 embedding gather); this module is
//! deliberately only the two missing primitives.

use crate::device::GpuContext;
use crate::diag::dbg_log;
use crate::error::{Result, WgpuError};
use crate::kernels::k1_q4k_gemv::grid_dims;
use std::sync::Arc;

/// Threads per rotate workgroup.
const ROTATE_WORKGROUP: u32 = 64;
/// Threads per attention workgroup (one per key-dimension
/// lane of the cooperative dot product).
const ATTEND_WORKGROUP: u32 = 128;
/// Value dimensions one lane accumulates at most: the
/// attention kernel refuses a `value_length` above
/// `ATTEND_WORKGROUP * MAX_VALUE_STRIDES` by name.
const MAX_VALUE_STRIDES: u32 = 8;

/// One WGSL source carrying both entry points, the K7
/// module's arrangement.
const SHADER: &str = r#"
struct RotateParams {
    rows: u32,
    heads: u32,
    head_dim: u32,
    _pad: u32,
}

struct AttendParams {
    rows: u32,
    q_heads: u32,
    kv_heads: u32,
    group: u32,
    key_len: u32,
    value_len: u32,
    block_rows: u32,
    block_start: u32,
    committed: u32,
    causal: u32,
    sliding: u32,
    window: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> rot_x: array<f32>;
@group(0) @binding(1) var<storage, read> rot_positions: array<u32>;
@group(0) @binding(2) var<storage, read> rot_inv_freq: array<f32>;
@group(0) @binding(3) var<uniform> rot_params: RotateParams;

@compute @workgroup_size(64)
fn rotate_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half = rot_params.head_dim / 2u;
    let per_row = rot_params.heads * half;
    let idx = gid.x;
    if (idx >= rot_params.rows * per_row) {
        return;
    }
    let row = idx / per_row;
    let rem = idx % per_row;
    let head = rem / half;
    let i = rem % half;
    let theta = f32(rot_positions[row]) * rot_inv_freq[i];
    let c = cos(theta);
    let s = sin(theta);
    let off = (row * rot_params.heads + head) * rot_params.head_dim + i;
    let x0 = rot_x[off];
    let x1 = rot_x[off + half];
    rot_x[off] = x0 * c - x1 * s;
    rot_x[off + half] = x0 * s + x1 * c;
}

@group(0) @binding(0) var<storage, read> att_q: array<f32>;
@group(0) @binding(1) var<storage, read> att_committed_k: array<f32>;
@group(0) @binding(2) var<storage, read> att_committed_v: array<f32>;
@group(0) @binding(3) var<storage, read> att_block_k: array<f32>;
@group(0) @binding(4) var<storage, read> att_block_v: array<f32>;
@group(0) @binding(5) var<storage, read> att_positions: array<u32>;
@group(0) @binding(6) var<storage, read_write> att_out: array<f32>;
@group(0) @binding(7) var<uniform> att_params: AttendParams;

var<workgroup> red: array<f32, 128>;
var<workgroup> sh_corr: f32;
var<workgroup> sh_w: f32;
var<workgroup> sh_l: f32;

fn att_visible(kp: u32, qp: u32) -> bool {
    if (att_params.causal != 0u && kp > qp) {
        return false;
    }
    if (att_params.sliding != 0u) {
        // The CPU implementation's comparison, in signed
        // arithmetic: key kp is visible to query qp iff
        // kp > qp - window.
        if (i32(kp) <= i32(qp) - i32(att_params.window)) {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(128)
fn attend_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.x / att_params.q_heads;
    let qh = wid.x % att_params.q_heads;
    let kvh = qh / att_params.group;
    let lane = lid.x;
    let qp = att_params.block_start + row;
    let key_len = att_params.key_len;
    let value_len = att_params.value_len;
    let k_width = att_params.kv_heads * key_len;
    let v_width = att_params.kv_heads * value_len;
    let q_off = (row * att_params.q_heads + qh) * key_len;
    var m = -1.0e30;
    var l = 0.0;
    var acc: array<f32, 8>;
    for (var j = 0u; j < 8u; j++) {
        acc[j] = 0.0;
    }
    let total = att_params.committed + att_params.block_rows;
    for (var idx = 0u; idx < total; idx++) {
        var kp: u32;
        var k_off: u32;
        var v_off: u32;
        var from_block = false;
        if (idx < att_params.committed) {
            kp = att_positions[idx];
            k_off = kp * k_width + kvh * key_len;
            v_off = kp * v_width + kvh * value_len;
        } else {
            let brow = idx - att_params.committed;
            kp = att_params.block_start + brow;
            k_off = brow * k_width + kvh * key_len;
            v_off = brow * v_width + kvh * value_len;
            from_block = true;
        }
        if (!att_visible(kp, qp)) {
            continue;
        }
        // Cooperative dot over the key dimension.
        var partial = 0.0;
        for (var d = lane; d < key_len; d += 128u) {
            var kv: f32;
            if (from_block) {
                kv = att_block_k[k_off + d];
            } else {
                kv = att_committed_k[k_off + d];
            }
            partial += att_q[q_off + d] * kv;
        }
        red[lane] = partial;
        workgroupBarrier();
        if (lane < 64u) { red[lane] += red[lane + 64u]; }
        workgroupBarrier();
        if (lane < 32u) { red[lane] += red[lane + 32u]; }
        workgroupBarrier();
        if (lane < 16u) { red[lane] += red[lane + 16u]; }
        workgroupBarrier();
        if (lane < 8u) { red[lane] += red[lane + 8u]; }
        workgroupBarrier();
        if (lane < 4u) { red[lane] += red[lane + 4u]; }
        workgroupBarrier();
        if (lane < 2u) { red[lane] += red[lane + 2u]; }
        workgroupBarrier();
        if (lane < 1u) { red[lane] += red[lane + 1u]; }
        workgroupBarrier();
        if (lane == 0u) {
            let score = red[0] * att_params.scale;
            let new_m = max(m, score);
            sh_corr = exp(m - new_m);
            sh_w = exp(score - new_m);
            m = new_m;
            l = l * sh_corr + sh_w;
            sh_l = l;
        }
        workgroupBarrier();
        let corr = sh_corr;
        let w = sh_w;
        for (var j = 0u; j < 8u; j++) {
            let d = lane + j * 128u;
            if (d >= value_len) {
                break;
            }
            var vv: f32;
            if (from_block) {
                vv = att_block_v[v_off + d];
            } else {
                vv = att_committed_v[v_off + d];
            }
            acc[j] = acc[j] * corr + w * vv;
        }
        workgroupBarrier();
    }
    // At least the query's own block row is always visible
    // (the block rows are part of the set and a position
    // sees itself under both masks), so l > 0 here; the
    // guard keeps a hypothetical empty set at zero instead
    // of NaN.
    var inv = 0.0;
    if (sh_l > 0.0) {
        inv = 1.0 / sh_l;
    }
    let o_off = (row * att_params.q_heads + qh) * value_len;
    for (var j = 0u; j < 8u; j++) {
        let d = lane + j * 128u;
        if (d >= value_len) {
            break;
        }
        att_out[o_off + d] = acc[j] * inv;
    }
}
"#;

/// Geometry validation for the rotate dispatch, kept pure
/// so the refusal shapes are unit-testable without a
/// device.
fn check_rotate(rows: usize, heads: usize, head_dim: usize) -> Result<()> {
    if rows == 0 || heads == 0 {
        return Err(WgpuError::InvalidTensor {
            name: "k17 rotate".to_string(),
            detail: format!(
                "rotate needs at least one row and head, got rows {rows}, heads {heads}"
            ),
        });
    }
    if head_dim < 2 || head_dim % 2 != 0 {
        return Err(WgpuError::InvalidTensor {
            name: "k17 rotate".to_string(),
            detail: format!("head_dim {head_dim} must be even and at least 2 for the NEOX pairing"),
        });
    }
    Ok(())
}

/// Geometry validation for the attention dispatch, pure
/// for the same reason as [`check_rotate`].
#[allow(clippy::too_many_arguments)]
fn check_attend(
    rows: usize,
    q_heads: usize,
    kv_heads: usize,
    key_len: usize,
    value_len: usize,
    block_rows: usize,
) -> Result<()> {
    if rows == 0 || block_rows == 0 {
        return Err(WgpuError::InvalidTensor {
            name: "k17 attend".to_string(),
            detail: format!(
                "attention needs at least one query row and one block row, got rows {rows}, block_rows {block_rows}"
            ),
        });
    }
    if q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0 {
        return Err(WgpuError::InvalidTensor {
            name: "k17 attend".to_string(),
            detail: format!(
                "q_heads {q_heads} must be a whole multiple of kv_heads {kv_heads}, both non-zero"
            ),
        });
    }
    if key_len == 0 {
        return Err(WgpuError::InvalidTensor {
            name: "k17 attend".to_string(),
            detail: "key_len is zero".to_string(),
        });
    }
    if value_len == 0 || value_len > (ATTEND_WORKGROUP * MAX_VALUE_STRIDES) as usize {
        return Err(WgpuError::InvalidTensor {
            name: "k17 attend".to_string(),
            detail: format!(
                "value_len {value_len} outside the kernel's supported range 1..={}",
                ATTEND_WORKGROUP * MAX_VALUE_STRIDES
            ),
        });
    }
    Ok(())
}

/// A prepared NEOX rotate dispatch: in place over `x`
/// (`rows x heads x head_dim`), one position per row read
/// from the positions array bound at prepare time.
#[derive(Debug)]
pub struct K17RotateCall {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    grid: (u32, u32),
}

impl K17RotateCall {
    /// Record this call's dispatch into `encoder` as its
    /// own compute pass.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k15 rotate dispatch"),
            timestamp_writes: None,
        });
        self.encode_into(&mut pass);
    }

    /// Record this dispatch into an already-open compute
    /// pass (the session composition form).
    pub fn encode_into(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.grid.0, self.grid.1, 1);
    }
}

/// A prepared draft-attention dispatch. The committed
/// count and the block start position are runtime values
/// (they change per block while the geometry does not), so
/// they live in the params buffer and are rewritten
/// through [`K17AttendCall::set_runtime`] between prepares
/// and submission, the K5 `set_start_position` pattern.
#[derive(Debug)]
pub struct K17AttendCall {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    grid: (u32, u32),
    params: Arc<wgpu::Buffer>,
}

/// Byte offset of the `block_start` field in the attend
/// params block (eight u32 fields precede it).
const ATTEND_BLOCK_START_OFFSET: u64 = 7 * 4;
/// Byte offset of the `committed` field in the attend
/// params block.
const ATTEND_COMMITTED_OFFSET: u64 = 8 * 4;

impl K17AttendCall {
    /// Record this call's dispatch into `encoder` as its
    /// own compute pass.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k15 attend dispatch"),
            timestamp_writes: None,
        });
        self.encode_into(&mut pass);
    }

    /// Record this dispatch into an already-open compute
    /// pass (the session composition form).
    pub fn encode_into(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.grid.0, self.grid.1, 1);
    }

    /// Rewrite the runtime params (block start position,
    /// committed position count) in place. Queue writes
    /// are ordered before the next submission, so a plan
    /// prepared once serves every block.
    pub fn set_runtime(&self, ctx: &GpuContext, block_start: u32, committed: u32) {
        ctx.queue.write_buffer(
            &self.params,
            ATTEND_BLOCK_START_OFFSET,
            &block_start.to_le_bytes(),
        );
        ctx.queue.write_buffer(
            &self.params,
            ATTEND_COMMITTED_OFFSET,
            &committed.to_le_bytes(),
        );
    }
}

/// The compiled K17 pipelines (NEOX rotate, draft
/// attention).
#[derive(Debug)]
pub struct K17Pipeline {
    rotate: wgpu::ComputePipeline,
    attend: wgpu::ComputePipeline,
}

impl K17Pipeline {
    /// Compile both K17 pipelines on `device`.
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] when wgpu rejects the
    /// shader (surfaced by the caller's first use; pipeline
    /// creation itself is lazy in wgpu).
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("k15 dflash shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let build = |entry: &'static str, label: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let rotate = build("rotate_main", "k15 rotate pipeline");
        let attend = build("attend_main", "k15 attend pipeline");
        dbg_log!("k15 pipelines built");
        Ok(K17Pipeline { rotate, attend })
    }

    /// Prepare an in-place NEOX rotate over `x`
    /// (`rows x heads x head_dim` f32), reading row
    /// positions from `positions` (at least `rows` u32
    /// entries) and the inverse frequencies for `base`
    /// from `inv_freq` (uploaded once per head geometry
    /// through [`K17Pipeline::upload_inv_freq`]).
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] on a zero row or head
    /// count, or an odd or tiny head dimension.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_rotate(
        &self,
        ctx: &GpuContext,
        x: &wgpu::Buffer,
        positions: &wgpu::Buffer,
        inv_freq: &wgpu::Buffer,
        rows: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<K17RotateCall> {
        check_rotate(rows, heads, head_dim)?;
        let total = (rows * heads * (head_dim / 2)) as u64;
        let grid = grid_dims(total.div_ceil(u64::from(ROTATE_WORKGROUP)));
        let params: [u32; 4] = [rows as u32, heads as u32, head_dim as u32, 0];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buf = ctx.uniform_buffer("k15 rotate params", &param_bytes);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k15 rotate bind group"),
            layout: &self.rotate.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: inv_freq.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        dbg_log!("k15 rotate prepared rows={rows} heads={heads} head_dim={head_dim}");
        Ok(K17RotateCall {
            pipeline: self.rotate.clone(),
            bind_group,
            grid,
        })
    }

    /// Upload the NEOX inverse-frequency table for
    /// `head_dim` at `base`: entry `i` is
    /// `base^(-2i/head_dim)`, computed in f64 and stored
    /// f32 (the K5 convention the cross-backend bars were
    /// proven under).
    pub fn upload_inv_freq(&self, ctx: &GpuContext, head_dim: usize, base: f32) -> wgpu::Buffer {
        let half = head_dim / 2;
        let mut freq = Vec::with_capacity(half);
        for i in 0..half {
            let exponent = -((2 * i) as f64) / (head_dim as f64);
            freq.push((f64::from(base).powf(exponent)) as f32);
        }
        ctx.upload_f32("k15 rope inv freq", &freq)
    }

    /// Prepare a draft-attention dispatch: `rows` query
    /// rows of `q_heads` heads (GQA group
    /// `q_heads / kv_heads`) attend over the committed
    /// positions in `positions` (whose K/V rows live in
    /// `committed_k` / `committed_v`, slot-addressed by
    /// absolute position) followed by the block's own
    /// `block_rows` K/V rows starting at absolute position
    /// `block_start`, under the causal flag and the
    /// per-layer sliding window. `out` receives
    /// `rows x q_heads x value_len` f32.
    ///
    /// The committed count and block start bound into the
    /// params here are the prepare-time values; rewrite
    /// them per block through
    /// [`K17AttendCall::set_runtime`].
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when the head counts
    /// are not a whole GQA multiple, a length is zero, or
    /// `value_len` exceeds the kernel's per-lane stride
    /// capacity.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_attend(
        &self,
        ctx: &GpuContext,
        q: &wgpu::Buffer,
        committed_k: &wgpu::Buffer,
        committed_v: &wgpu::Buffer,
        block_k: &wgpu::Buffer,
        block_v: &wgpu::Buffer,
        positions: &wgpu::Buffer,
        out: &wgpu::Buffer,
        rows: usize,
        q_heads: usize,
        kv_heads: usize,
        key_len: usize,
        value_len: usize,
        block_rows: usize,
        block_start: usize,
        committed: usize,
        causal: bool,
        sliding: Option<u32>,
        scale: f32,
    ) -> Result<K17AttendCall> {
        check_attend(rows, q_heads, kv_heads, key_len, value_len, block_rows)?;
        let group = q_heads / kv_heads;
        let (sliding_flag, window) = match sliding {
            Some(w) => (1u32, w),
            None => (0u32, 0u32),
        };
        let params: [u32; 16] = [
            rows as u32,
            q_heads as u32,
            kv_heads as u32,
            group as u32,
            key_len as u32,
            value_len as u32,
            block_rows as u32,
            block_start as u32,
            committed as u32,
            u32::from(causal),
            sliding_flag,
            window,
            scale.to_bits(),
            0,
            0,
            0,
        ];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buf = Arc::new(ctx.uniform_buffer("k15 attend params", &param_bytes));
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k15 attend bind group"),
            layout: &self.attend.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: committed_k.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: committed_v.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: block_k.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: block_v.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        dbg_log!(
            "k15 attend prepared rows={rows} q_heads={q_heads} kv_heads={kv_heads} \
             block_rows={block_rows} committed={committed} causal={causal} sliding={sliding:?}"
        );
        Ok(K17AttendCall {
            pipeline: self.attend.clone(),
            bind_group,
            grid: ((rows * q_heads) as u32, 1),
            params: params_buf,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_names_its_entry_points_and_masks() {
        assert!(SHADER.contains("fn rotate_main"));
        assert!(SHADER.contains("fn attend_main"));
        // The NEOX half-split pairing and the CPU mask's
        // own comparisons are the semantics under test;
        // pin their textual shape so an edit cannot drift
        // them quietly.
        assert!(SHADER.contains("rot_x[off + half]"));
        assert!(SHADER.contains("i32(kp) <= i32(qp) - i32(att_params.window)"));
        assert!(SHADER.contains("att_params.causal != 0u && kp > qp"));
    }

    #[test]
    fn check_rotate_rejects_bad_geometry() {
        assert!(check_rotate(2, 4, 8).is_ok());
        assert!(check_rotate(0, 4, 8).is_err());
        assert!(check_rotate(2, 0, 8).is_err());
        assert!(check_rotate(2, 4, 7).is_err());
        assert!(check_rotate(2, 4, 1).is_err());
    }

    #[test]
    fn check_attend_rejects_bad_geometry() {
        assert!(check_attend(4, 8, 2, 128, 128, 4).is_ok());
        assert!(check_attend(0, 8, 2, 128, 128, 4).is_err());
        assert!(check_attend(4, 8, 3, 128, 128, 4).is_err());
        assert!(check_attend(4, 8, 2, 0, 128, 4).is_err());
        assert!(check_attend(4, 8, 2, 128, 0, 4).is_err());
        assert!(check_attend(4, 8, 2, 128, 2048, 4).is_err());
        assert!(check_attend(4, 8, 2, 128, 128, 0).is_err());
    }

    #[test]
    fn attend_param_offsets_match_the_wgsl_layout() {
        // block_start is the eighth field, committed the
        // ninth; the runtime rewrite depends on it.
        assert_eq!(ATTEND_BLOCK_START_OFFSET, 28);
        assert_eq!(ATTEND_COMMITTED_OFFSET, 32);
    }
}
