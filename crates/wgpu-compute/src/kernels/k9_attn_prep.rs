//! K9: the fused attention preparation (a fusion product, not
//! a BACKEND.md numbered kernel): the Q and K per-head RMSNorms
//! (K4), the RoPE rotations (K5, sliding layers only), and the
//! KV append (K5) as ONE dispatch per layer.
//!
//! The unfused chain spends five dispatches here per sliding
//! layer (two norms, two rotates, one append) and three per
//! global layer, each over a few hundred to a few thousand
//! elements: at decode sizes these kernels are launch and
//! barrier cost, not math (bench/gpu-resident-2026-10-30.md
//! prices the resident chain's in-pass dispatch overhead at
//! roughly 12 microseconds a dispatch). The three steps also
//! chain with nothing between them but each other's outputs,
//! so they fuse without touching the step order or the math:
//!
//! - The norms are K4's weighted form at the file eps, with
//!   K4's exact per-lane distribution for a row of `head_dim`
//!   (at most 128 here, the K6 bound the session enforces):
//!   lane `d` holds channel `d`, lanes past the width
//!   contribute zero, and the sum of squares reduces through
//!   the same 256-lane partials tree K4 uses, so each norm's
//!   result is the value K4's own dispatch produces.
//! - The rotation is K5's interleaved (adjacent-pair, ggml
//!   NORM) form with K5's angle law, `position *
//!   base^(-2i / head_dim)` computed in f32, applied to the
//!   normed rows before they are written out. The normed row
//!   passes through workgroup memory so each adjacent pair is
//!   rotated by one lane; the pair values are the norm's own
//!   outputs, unrounded by any round trip.
//! - The append is K5's slot law: the position itself for
//!   global layers, `position % ring` for windowed ones (the
//!   ring capacity, not the mask window; see
//!   [`crate::kernels::k6_attention`]'s notes on the split).
//!   V rows are copied untouched, as the append always did.
//!
//! One workgroup of 256 lanes serves one head row. The grid
//! covers, in order, every Q head row, every K head row, and
//! every V head row of the chunk; the segment a workgroup
//! belongs to selects its source, its norm weight, and its
//! destination. Segment sources are addressed by a per-buffer
//! row stride plus a segment offset, so the same kernel reads
//! either three separate projection buffers (stride = the
//! buffer's own width, offset 0) or one concatenated QKV
//! buffer (stride = the concatenated width, offsets at the
//! segment starts); the session picks per layer.
//!
//! The trait path does not use this kernel: it composes the
//! K4 and K5 steps through the `Compute` trait's own seams,
//! and stays the executable reference this fusion is proven
//! against (the resident-session cross-path test and the
//! crate's fused-kernel test both compare the two).

use crate::device::GpuContext;
use crate::error::{Result, WgpuError};

use crate::diag::dbg_log;

/// Largest head dim this kernel serves: the workgroup-memory
/// rotation staging is one f32 per channel, sized for the K6
/// reduce bound the session already enforces.
const MAX_HEAD_DIM: u32 = 128;

/// The fused preparation shader. See the module docs for the
/// segment layout and the numerics contract.
const PREP_SHADER: &str = r#"
struct PrepParams {
    rows: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    start_pos: u32,
    ring: u32,
    rotate: u32,
    base: f32,
    eps: f32,
    grid_x: u32,
    q_stride: u32,
    q_off: u32,
    kv_stride: u32,
    k_off: u32,
    v_off: u32,
    v_stride: u32,
}

@group(0) @binding(0) var<storage, read> q_in: array<f32>;
@group(0) @binding(1) var<storage, read> k_in: array<f32>;
@group(0) @binding(2) var<storage, read> v_in: array<f32>;
@group(0) @binding(3) var<storage, read> q_w: array<f32>;
@group(0) @binding(4) var<storage, read> k_w: array<f32>;
@group(0) @binding(5) var<storage, read_write> q_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> kv_k: array<f32>;
@group(0) @binding(7) var<storage, read_write> kv_v: array<f32>;
@group(0) @binding(8) var<uniform> params: PrepParams;

var<workgroup> partials: array<f32, 256>;
var<workgroup> inv_shared: f32;
var<workgroup> vals: array<f32, 128>;

@compute @workgroup_size(256)
fn prep_main(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {
    let seg = wid.y * params.grid_x + wid.x;
    let lane = lid.x;
    let total_q = params.rows * params.q_heads;
    let total_kv = params.rows * params.kv_heads;
    // Whole-workgroup early out: every lane shares `seg`, so
    // the exit is uniform and the barriers below stay legal.
    if (seg >= total_q + 2u * total_kv) {
        return;
    }
    let d = lane;
    let kv_width = params.kv_heads * params.head_dim;

    if (seg >= total_q + total_kv) {
        // V segment: the append's plain copy, no norm, no
        // rotation.
        let s2 = seg - total_q - total_kv;
        let row = s2 / params.kv_heads;
        let head = s2 % params.kv_heads;
        if (d < params.head_dim) {
            let position = params.start_pos + row;
            var slot = position;
            if (params.ring != 0u) {
                slot = position % params.ring;
            }
            kv_v[slot * kv_width + head * params.head_dim + d] =
                v_in[row * params.v_stride + params.v_off + head * params.head_dim + d];
        }
        return;
    }

    let is_q = seg < total_q;
    var row = 0u;
    var head = 0u;
    var src_base = 0u;
    if (is_q) {
        row = seg / params.q_heads;
        head = seg % params.q_heads;
        src_base = row * params.q_stride + params.q_off + head * params.head_dim;
    } else {
        let s2 = seg - total_q;
        row = s2 / params.kv_heads;
        head = s2 % params.kv_heads;
        src_base = row * params.kv_stride + params.k_off + head * params.head_dim;
    }

    // The norm, K4's distribution and tree for a row of
    // head_dim: lane d holds channel d (lanes past the width
    // hold zero), partials reduce 128, 64, ..., 1.
    var x = 0.0;
    if (d < params.head_dim) {
        x = q_in[src_base + d];
        if (!is_q) {
            x = k_in[src_base + d];
        }
    }
    partials[lane] = x * x;
    workgroupBarrier();
    var stride = 128u;
    while (stride > 0u) {
        if (lane < stride) {
            partials[lane] += partials[lane + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    if (lane == 0u) {
        inv_shared = inverseSqrt(partials[0] / f32(params.head_dim) + params.eps);
    }
    workgroupBarrier();
    let inv = inv_shared;
    var y = 0.0;
    if (d < params.head_dim) {
        var w = q_w[d];
        if (!is_q) {
            w = k_w[d];
        }
        y = x * inv * w;
    }

    let position = params.start_pos + row;
    var slot = position;
    if (params.ring != 0u) {
        slot = position % params.ring;
    }
    let q_base = (row * params.q_heads + head) * params.head_dim;
    let k_base = slot * kv_width + head * params.head_dim;

    if (params.rotate == 1u) {
        if (d < params.head_dim) {
            vals[d] = y;
        }
        workgroupBarrier();
        // One lane per adjacent pair: K5's rotation, its angle
        // law and pairing, over the norm's own outputs.
        let pair = lane;
        if (pair < params.head_dim / 2u) {
            let angle = f32(position)
                * pow(params.base, -2.0 * f32(pair) / f32(params.head_dim));
            let c = cos(angle);
            let s = sin(angle);
            let a = vals[2u * pair];
            let b = vals[2u * pair + 1u];
            let r0 = a * c - b * s;
            let r1 = a * s + b * c;
            if (is_q) {
                q_out[q_base + 2u * pair] = r0;
                q_out[q_base + 2u * pair + 1u] = r1;
            } else {
                kv_k[k_base + 2u * pair] = r0;
                kv_k[k_base + 2u * pair + 1u] = r1;
            }
        }
        return;
    }
    if (d < params.head_dim) {
        if (is_q) {
            q_out[q_base + d] = y;
        } else {
            kv_k[k_base + d] = y;
        }
    }
}
"#;

/// The buffers and geometry one fused preparation dispatch
/// binds. Sources carry a row stride and a segment offset so
/// one kernel serves separate projection buffers and a
/// concatenated QKV buffer alike (see the module docs).
#[derive(Debug)]
pub struct PrepArgs<'a> {
    /// Query rows as projected (read).
    pub q_in: &'a wgpu::Buffer,
    /// Row stride of `q_in`, in elements.
    pub q_stride: usize,
    /// Element offset of the Q segment within each `q_in` row.
    pub q_off: usize,
    /// Key rows as projected (read).
    pub k_in: &'a wgpu::Buffer,
    /// Row stride of the K source, in elements.
    pub kv_stride: usize,
    /// Element offset of the K segment within each source row.
    pub k_off: usize,
    /// Value rows as projected (read).
    pub v_in: &'a wgpu::Buffer,
    /// Row stride of the V source, in elements (separate from
    /// the K stride: a partial batch can carry K in the
    /// concatenated rows while V keeps its own buffer).
    pub v_stride: usize,
    /// Element offset of the V segment within each source row.
    pub v_off: usize,
    /// Query norm weight, `[head_dim]`.
    pub q_w: &'a wgpu::Buffer,
    /// Key norm weight, `[head_dim]`.
    pub k_w: &'a wgpu::Buffer,
    /// Normed (and, on sliding layers, rotated) query rows
    /// (written), `[rows][q_heads * head_dim]`.
    pub q_out: &'a wgpu::Buffer,
    /// The layer's resident KV key buffer (written at slots).
    pub kv_k: &'a wgpu::Buffer,
    /// The layer's resident KV value buffer (written at slots).
    pub kv_v: &'a wgpu::Buffer,
    /// Chunk rows.
    pub rows: usize,
    /// Query head count.
    pub q_heads: usize,
    /// KV head count.
    pub kv_heads: usize,
    /// Per-head dimension (even, at most [`MAX_HEAD_DIM`]).
    pub head_dim: usize,
    /// Absolute position of the chunk's first row (rewritable
    /// through the returned call).
    pub start_pos: usize,
    /// The windowed ring capacity, or `None` for a global
    /// layer (slot = position).
    pub ring: Option<usize>,
    /// Rotate Q and K after their norms (sliding layers);
    /// global layers pass `false` (NoPE).
    pub rotate: bool,
    /// The RoPE base (the config's resolved SWA base).
    pub base: f32,
    /// The norm epsilon (the file's rms eps).
    pub eps: f32,
}

/// One prepared fused-preparation dispatch, encodable any
/// number of times into any command encoder. The start
/// position is the one parameter a sequence changes per
/// forward, so the call keeps its uniform buffer and rewrites
/// that one word through
/// [`K9PrepCall::set_start_position`]; everything else is
/// fixed at prepare time. This is the device-resident
/// forward's unit of composition (see [`crate::session`]);
/// the trait path never prepares one.
#[derive(Debug)]
pub struct K9PrepCall {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    grid: (u32, u32),
}

impl K9PrepCall {
    /// Record this call's dispatch into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k9 attention prep"),
            timestamp_writes: None,
        });
        self.encode_into(&mut pass);
    }

    /// Record this dispatch into an already-open compute
    /// `pass`: the composition form the device-resident
    /// session uses to chain a whole forward into one pass
    /// (wgpu inserts the hazard barriers between dispatches
    /// inside the pass).
    pub fn encode_into(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.grid.0, self.grid.1, 1);
    }

    /// Point the preparation at a new absolute start position
    /// (the prepared rows sit at `start_position + row` for
    /// rotation angles and append slots). Same queue-ordering
    /// contract as
    /// [`crate::kernels::k5_rope_kv::K5RotateCall::set_start_position`]:
    /// call it while no submission using this call is in
    /// flight.
    pub fn set_start_position(&self, ctx: &GpuContext, start_position: usize) {
        // PrepParams word 4, after rows, q_heads, kv_heads,
        // head_dim.
        ctx.queue
            .write_buffer(&self.params, 16, &(start_position as u32).to_le_bytes());
    }
}

/// The compiled K9 pipeline (one shader, one entry point).
#[derive(Debug)]
pub struct K9Pipeline {
    prep: wgpu::ComputePipeline,
}

impl K9Pipeline {
    /// Compile the K9 pipeline on `device`.
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] when wgpu rejects the shader.
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("k9 attention prep shader"),
            source: wgpu::ShaderSource::Wgsl(PREP_SHADER.into()),
        });
        let prep = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k9 attention prep pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("prep_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        dbg_log!("k9 pipeline built");
        Ok(K9Pipeline { prep })
    }

    /// Prepare a fused preparation dispatch over device
    /// buffers (see [`PrepArgs`]).
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when the geometry is
    /// unusable (zero rows or heads, an odd or oversized head
    /// dim, or a rotation base that is not finite and greater
    /// than 1 while rotating).
    pub fn prepare(&self, ctx: &GpuContext, args: &PrepArgs<'_>) -> Result<K9PrepCall> {
        if args.rows == 0 || args.q_heads == 0 || args.kv_heads == 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k9 prep".to_string(),
                detail: format!(
                    "rows {}, q_heads {}, kv_heads {}: nothing to prepare",
                    args.rows, args.q_heads, args.kv_heads
                ),
            });
        }
        if args.head_dim < 2 || args.head_dim % 2 != 0 || args.head_dim > MAX_HEAD_DIM as usize {
            return Err(WgpuError::InvalidTensor {
                name: "k9 prep".to_string(),
                detail: format!(
                    "head_dim {} is outside the even range 2..={MAX_HEAD_DIM} this \
                     kernel's workgroup staging serves",
                    args.head_dim
                ),
            });
        }
        if args.rotate && (!args.base.is_finite() || args.base <= 1.0) {
            return Err(WgpuError::InvalidTensor {
                name: "k9 prep".to_string(),
                detail: format!("rope base {} is not finite and greater than 1", args.base),
            });
        }
        // One workgroup per segment (not per 256 elements):
        // the grid covers segments, each exactly one workgroup.
        let total = (args.rows * (args.q_heads + 2 * args.kv_heads)) as u64;
        let grid = crate::kernels::k1_q4k_gemv::grid_dims(total.max(1));
        let params: [u32; 16] = [
            args.rows as u32,
            args.q_heads as u32,
            args.kv_heads as u32,
            args.head_dim as u32,
            args.start_pos as u32,
            args.ring.map_or(0, |r| r as u32),
            u32::from(args.rotate),
            args.base.to_bits(),
            args.eps.to_bits(),
            grid.0,
            args.q_stride as u32,
            args.q_off as u32,
            args.kv_stride as u32,
            args.k_off as u32,
            args.v_off as u32,
            args.v_stride as u32,
        ];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buf = ctx.uniform_buffer("k9 prep params", &param_bytes);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k9 prep bind group"),
            layout: &self.prep.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: args.q_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: args.k_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: args.v_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: args.q_w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: args.k_w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: args.q_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: args.kv_k.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: args.kv_v.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        Ok(K9PrepCall {
            pipeline: self.prep.clone(),
            bind_group,
            params: params_buf,
            grid,
        })
    }
}

/// The ring slot of an absolute position under this kernel's
/// law, restating the shader's for host-side callers and
/// tests: the position itself for global layers, `position %
/// ring` for windowed ones. (Same law as
/// [`crate::kernels::k5_rope_kv::slot_of`]; this kernel's
/// `ring` parameter is the capacity, never the mask window.)
#[must_use]
pub fn slot_of(position: usize, ring: Option<usize>) -> usize {
    match ring {
        None => position,
        Some(r) => position % r,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_carries_its_entry_point_and_laws() {
        assert!(PREP_SHADER.contains("fn prep_main"));
        // The NORM pairing is adjacent channels, as in K5.
        assert!(PREP_SHADER.contains("2u * pair"));
        // The slot law splits ring (storage) from window
        // (mask): the shader takes the ring only.
        assert!(PREP_SHADER.contains("slot = position % params.ring"));
        // The norm's inverse-RMS form is K4's.
        assert!(PREP_SHADER.contains("inverseSqrt(partials[0] / f32(params.head_dim)"));
    }

    #[test]
    fn slot_of_matches_the_append_law() {
        assert_eq!(slot_of(7, None), 7);
        assert_eq!(slot_of(7, Some(2048)), 7);
        assert_eq!(slot_of(2049, Some(2048)), 1);
        assert_eq!(slot_of(4096, Some(2111)), 4096 % 2111);
    }
}
