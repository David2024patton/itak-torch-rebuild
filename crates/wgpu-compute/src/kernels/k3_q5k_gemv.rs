//! K3: Q5_K dequantize-and-multiply GEMV (BACKEND.md Section 3).
//!
//! The untied output head: 6,656 in, 202,048 out, about 0.86
//! GiB, 5.5 percent of the stream, running every token to
//! produce the logits. Its 202,048 rows exceed the
//! 65,535-workgroup dispatch cap, which is why the grid scheme
//! here is two-dimensional from birth (the flat workgroup
//! index wraps over `grid_x`). The logit softcap and scale
//! fuse into this kernel's epilogue with K8; today the raw
//! projection is what lands, and `muse-glimmer`'s head math
//! applies the finishing map after `lm_head` returns, exactly
//! as it does over the CPU backend.
//!
//! The shader reads the ggml block layout directly from the
//! tensor's raw bytes (176 bytes per 256-element super-block:
//! f16 `d`, f16 `dmin`, the same 12 packed scale/min bytes as
//! Q4_K, a 32-byte high-bits plane `qh`, and 128 nibble bytes
//! `ql`), expanding values in registers and never materializing
//! a dequantized weight anywhere. The arithmetic mirrors
//! `gguf::dequant::dequant_q5_k` term for term: element value
//! is `d * scale * q - dmin * min` with the 5-bit `q` assembled
//! from a `ql` nibble plus one bit of `qh` (bit `s` of
//! `qh[l]` belongs to sub-block `s`), and the scale/min pairs
//! unpacked by `get_scale_min_k4`'s 6-bit packing, so GPU
//! results check against the CPU reference within float
//! reassociation noise. A Q5_K block is 176 bytes, a whole
//! number of words, so unlike K2 this kernel uploads the raw
//! bytes with no repacking.
//!
//! The parallel shape is K1's tuned shape, including its
//! thread assignment (see `k1_q4k_gemv`'s module docs and
//! BACKEND.md's tuning record): sixteen threads cooperate on
//! one super-block, each owning sixteen elements across four
//! sub-blocks; the eight scale/min pairs are extracted with
//! the same packed construction as K1's; a workgroup covers
//! several output rows so the activation quads serve every row
//! in the tile; and the reduction is the workgroup-memory tree
//! or the segmented subgroup butterfly with a shared-memory
//! finish. The one addition over K1's shader is the fifth-bit
//! assembly: one `qh` word per thread (its four bytes are the
//! high bits of its four element quads' shared positions), and
//! per sub-block the bit plane is `(qh >> s) & 0x01010101`,
//! one unpack away from the vec4 of bits the dot product
//! needs. No f16 activation variant is offered, for the same
//! reason as K2's: K1's f16 experiment failed this crate's
//! check tolerance and measured slower than its f32 twin.

use crate::device::GpuContext;
use crate::error::{Result, WgpuError};
use crate::kernels::k1_q4k_gemv::grid_dims;

use crate::diag::dbg_log;

/// Elements per Q5_K super-block.
const BLOCK_ELEMS: usize = 256;
/// Bytes per Q5_K super-block.
const BLOCK_BYTES: usize = 176;
/// Words per Q5_K super-block.
const BLOCK_WORDS: u32 = 44;

/// The parallel shape a K3 pipeline runs, parameterized the
/// same way as [`crate::kernels::k1_q4k_gemv::K1Shape`] (rows
/// per workgroup, workgroup size, reduction kind), minus the
/// legacy and f16 variants K1 carries for its own history.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct K3Shape {
    /// Output rows one workgroup covers (llama.cpp's NUM_ROWS).
    pub rows: u32,
    /// Threads per workgroup; a multiple of sixteen (one
    /// sixteen-thread group per super-block in flight).
    pub workgroup: u32,
    /// Reduce with the segmented subgroup butterfly (needs the
    /// device's SUBGROUP feature) instead of the pure
    /// workgroup-memory tree.
    pub subgroup: bool,
}

impl K3Shape {
    /// A shape with the given tile geometry and the
    /// workgroup-memory reduction.
    #[must_use]
    pub const fn llama(rows: u32, workgroup: u32) -> K3Shape {
        K3Shape {
            rows,
            workgroup,
            subgroup: false,
        }
    }

    /// This shape with the subgroup reduction.
    #[must_use]
    pub const fn with_subgroup(self) -> K3Shape {
        K3Shape {
            subgroup: true,
            ..self
        }
    }

    /// The bench/check name of this shape: `r<rows>` with an
    /// `sg` suffix for the subgroup reduction (workgroup size
    /// is a separate bench flag, defaulting to 128).
    #[must_use]
    pub fn name(&self) -> String {
        let mut name = format!("r{}", self.rows);
        if self.subgroup {
            name.push_str("sg");
        }
        name
    }

    /// Parse a bench/check shape name (see [`K3Shape::name`]),
    /// with workgroup size 128, the size K1's tuning pass
    /// measured best on this card (a bench flag can override
    /// it).
    #[must_use]
    pub fn from_name(name: &str) -> Option<K3Shape> {
        let rest = name.strip_prefix('r')?;
        let (digits, suffix) = rest.split_at(
            rest.find(|c: char| !c.is_ascii_digit())
                .unwrap_or(rest.len()),
        );
        let rows: u32 = digits.parse().ok()?;
        let mut shape = K3Shape::llama(rows, 128);
        match suffix {
            "" => {}
            "sg" => shape.subgroup = true,
            _ => return None,
        }
        Some(shape)
    }

    /// Check the shape against a device's granted features and
    /// the geometry rules, naming what is missing.
    fn validate(&self, features: wgpu::Features) -> Result<()> {
        if !(1..=8).contains(&self.rows) {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "k3 shape {}: rows per workgroup {} is outside 1..=8",
                    self.name(),
                    self.rows
                ),
            });
        }
        if self.workgroup == 0 || self.workgroup % 16 != 0 || self.workgroup > 256 {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "k3 shape {}: workgroup size {} is not a multiple of 16 in 16..=256",
                    self.name(),
                    self.workgroup
                ),
            });
        }
        if self.subgroup && !features.contains(wgpu::Features::SUBGROUP) {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "k3 shape {} needs the SUBGROUP device feature, which this device was not granted",
                    self.name()
                ),
            });
        }
        Ok(())
    }
}

/// The shape [`K3Pipeline::new`] builds for a device with the
/// given granted features: the configuration K1's tuning pass
/// measured best on this card and ggml-vulkan itself picks for
/// K-quants on RDNA3 (two rows per workgroup of 128, hybrid
/// reduction), falling back to the same geometry with the
/// workgroup-memory reduction when SUBGROUP was not granted.
/// K3's own bench record (BACKEND.md's K2/K3 results section)
/// tested this default against seven other shapes on the whole
/// head and confirmed it: at 26 blocks per row, rows per
/// workgroup pay and width does not, the inverse of K2's
/// geometry. If a future matrix disagrees, the default moves
/// with the evidence.
#[must_use]
pub fn shipped_shape(features: wgpu::Features) -> K3Shape {
    if features.contains(wgpu::Features::SUBGROUP) {
        K3Shape::llama(2, 128).with_subgroup()
    } else {
        K3Shape::llama(2, 128)
    }
}

/// The K3 shader for `shape`, generated with the tile geometry
/// as literals so the compiler unrolls the row loop and sizes
/// the reduction arrays exactly. Thread assignment follows
/// K1's (and llama.cpp's `mul_mat_vec_q5_k.comp` covers the
/// same elements with a strided variant of it): thread `t` of
/// a sixteen-thread group owns elements `l0..l0+3` of
/// sub-blocks `2h`, `2h+1`, `2h+4`, `2h+5` of each super-block
/// it strides, where `h = t / 8` and `l0 = 4 * (t % 8)` (the
/// ql plane starts at word 12 of the block, the qh plane at
/// word 4).
fn shader(shape: &K3Shape) -> String {
    let rows = shape.rows;
    let wg = shape.workgroup;
    let groups = wg / 16;
    // Reduction scratch: the tree variant keeps one partial per
    // lane per row; the subgroup variant only the per-group sums.
    let partials_len = if shape.subgroup {
        rows * groups
    } else {
        rows * wg
    };
    let reduction = if shape.subgroup {
        format!(
            r#"
    // Segmented butterfly inside each sixteen-thread group
    // (the xor masks never cross the aligned sixteen, so each
    // group reduces its own partials inside whatever hardware
    // subgroup width the device runs), then one shared-memory
    // finish across groups: ggml's HYBRID reduction.
    for (var r = 0u; r < {rows}u; r++) {{
        var v = acc[r];
        v += subgroupShuffleXor(v, 1u);
        v += subgroupShuffleXor(v, 2u);
        v += subgroupShuffleXor(v, 4u);
        v += subgroupShuffleXor(v, 8u);
        acc[r] = v;
    }}
    if (t == 0u) {{
        for (var r = 0u; r < {rows}u; r++) {{
            partials[r * {groups}u + group] = acc[r];
        }}
    }}
    workgroupBarrier();
    if (lane == 0u) {{
        for (var r = 0u; r < {rows}u; r++) {{
            if (row0 + r < params.out_rows) {{
                var s = 0.0;
                for (var g = 0u; g < {groups}u; g++) {{
                    s += partials[r * {groups}u + g];
                }}
                output[batch_row * params.out_rows + row0 + r] = s;
            }}
        }}
    }}"#
        )
    } else {
        format!(
            r#"
    for (var r = 0u; r < {rows}u; r++) {{
        partials[r * {wg}u + lane] = acc[r];
    }}
    workgroupBarrier();
    var stride = {half_wg}u;
    while (stride > 0u) {{
        if (lane < stride) {{
            for (var r = 0u; r < {rows}u; r++) {{
                partials[r * {wg}u + lane] += partials[r * {wg}u + lane + stride];
            }}
        }}
        workgroupBarrier();
        stride >>= 1u;
    }}
    if (lane == 0u) {{
        for (var r = 0u; r < {rows}u; r++) {{
            if (row0 + r < params.out_rows) {{
                output[batch_row * params.out_rows + row0 + r] = partials[r * {wg}u];
            }}
        }}
    }}"#,
            half_wg = wg / 2
        )
    };

    format!(
        r#"struct Params {{
    blocks_per_row: u32,
    out_rows: u32,
    grid_x: u32,
    batch: u32,
}}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partials: array<f32, {partials_len}>;

// IEEE 754 half to f32, mirroring gguf::dequant::f16_to_f32 for
// every value a weight file can hold (normals, subnormals, zero;
// inf and NaN payloads keep their class).
fn f16_to_f32(h: u32) -> f32 {{
    let sign = (h >> 15u) & 1u;
    let exp = (h >> 10u) & 31u;
    let frac = h & 1023u;
    if (exp == 31u) {{
        return bitcast<f32>((sign << 31u) | 0x7f800000u | (frac << 13u));
    }}
    if (exp == 0u) {{
        let v = f32(frac) * 5.960464477539063e-08;
        if (sign == 1u) {{ return -v; }}
        return v;
    }}
    return bitcast<f32>((sign << 31u) | ((exp + 112u) << 23u) | (frac << 13u));
}}

@compute @workgroup_size({wg})
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
    let row_groups = (params.out_rows + {rows}u - 1u) / {rows}u;
    let flat = wid.y * params.grid_x + wid.x;
    // Whole-workgroup early out: every lane shares `flat`, so
    // the exit is uniform and the barriers below stay legal.
    if (flat >= params.batch * row_groups) {{
        return;
    }}
    let batch_row = flat / row_groups;
    let row0 = (flat % row_groups) * {rows}u;
    let lane = lid.x;
    let group = lane / 16u;
    let t = lane % 16u;
    let half = t / 8u;
    let l0 = 4u * (t % 8u);
    // This thread's ql words within a super-block (the ql plane
    // starts at word 12; the second word sits 64 bytes later,
    // in group h + 2) and its first activation quad within the
    // block's 64-quad span. Its qh word is word 4 + l0 / 4: the
    // four bytes are the high bits of its four positions in
    // every sub-block.
    let w0_off = 12u + 8u * half + l0 / 4u;
    let qh_off = 4u + l0 / 4u;
    let yv = 16u * half + l0 / 4u;

    var acc = array<f32, {rows}>();
    for (var blk = group; blk < params.blocks_per_row; blk += {groups}u) {{
        // Activation quads and their sums are per-block, shared
        // by every row of the tile: loaded once, used {rows}
        // times. A pairs with sub-block 2h, B with 2h+1 (the
        // first word's high nibbles, 32 elements later), C and
        // D with 2h+4 and 2h+5 (the second word, 128 later).
        let xb0 = (batch_row * params.blocks_per_row + blk) * 64u;
        let xa = input[xb0 + yv];
        let xbq = input[xb0 + yv + 8u];
        let xc = input[xb0 + yv + 32u];
        let xd = input[xb0 + yv + 40u];
        let sa = dot(xa, vec4<f32>(1.0));
        let sb = dot(xbq, vec4<f32>(1.0));
        let sc_ = dot(xc, vec4<f32>(1.0));
        let sd = dot(xd, vec4<f32>(1.0));
        // The high bits are per-block too (one qh word covers
        // this thread's four positions in all eight sub-blocks;
        // bit s of each byte belongs to sub-block s).
        for (var r = 0u; r < {rows}u; r++) {{
            let row = row0 + r;
            if (row >= params.out_rows) {{
                continue;
            }}
            let bw = (row * params.blocks_per_row + blk) * {BLOCK_WORDS}u;
            let hw = weights[bw];
            let d = f16_to_f32(hw & 0xffffu);
            let dmin = f16_to_f32(hw >> 16u);
            // The packed scale extraction, identical to K1's
            // (the 12 scale bytes are packed the same way in
            // Q4_K and Q5_K): (scaleA, scaleB, minA, minB) and
            // (scaleC, scaleD, minC, minD) for sub-blocks 2h,
            // 2h+1, 2h+4, 2h+5, equal to get_scale_min_k4 on
            // those indices.
            let s16 = (weights[bw + 1u] >> (16u * half)) & 0xffffu;
            let m16 = (weights[bw + 2u] >> (16u * half)) & 0xffffu;
            let h16 = (weights[bw + 3u] >> (16u * half)) & 0xffffu;
            let lo = (m16 << 16u) | s16;
            let sc_lo = vec4<f32>(unpack4xU8(lo & 0x3f3f3f3fu));
            let hi = (((h16 << 12u) | h16) & 0x0f0f0f0fu) | ((lo & 0xc0c0c0c0u) >> 2u);
            let sc_hi = vec4<f32>(unpack4xU8(hi));
            let qhw = weights[bw + qh_off];
            let bits_a = vec4<f32>(unpack4xU8((qhw >> (2u * half)) & 0x01010101u));
            let bits_b = vec4<f32>(unpack4xU8((qhw >> (2u * half + 1u)) & 0x01010101u));
            let bits_c = vec4<f32>(unpack4xU8((qhw >> (2u * half + 4u)) & 0x01010101u));
            let bits_d = vec4<f32>(unpack4xU8((qhw >> (2u * half + 5u)) & 0x01010101u));
            let w0 = weights[bw + w0_off];
            let w1 = weights[bw + w0_off + 16u];
            let qa = vec4<f32>(unpack4xU8(w0 & 0x0f0f0f0fu)) + 16.0 * bits_a;
            let qb = vec4<f32>(unpack4xU8((w0 >> 4u) & 0x0f0f0f0fu)) + 16.0 * bits_b;
            let qc = vec4<f32>(unpack4xU8(w1 & 0x0f0f0f0fu)) + 16.0 * bits_c;
            let qd = vec4<f32>(unpack4xU8((w1 >> 4u) & 0x0f0f0f0fu)) + 16.0 * bits_d;
            acc[r] += d * (sc_lo.x * dot(qa, xa) + sc_lo.y * dot(qb, xbq)
                         + sc_hi.x * dot(qc, xc) + sc_hi.y * dot(qd, xd))
                    - dmin * (sc_lo.z * sa + sc_lo.w * sb
                            + sc_hi.z * sc_ + sc_hi.w * sd);
        }}
    }}
{reduction}
}}
"#
    )
}

/// A Q5_K weight tensor resident on the device: one storage
/// buffer holding the tensor's raw GGUF bytes, bound whole per
/// dispatch (BACKEND.md Section 5's per-tensor buffer
/// strategy).
#[derive(Debug)]
pub struct K3Weights {
    /// Whole-tensor byte length, for the bench's stream
    /// accounting.
    pub byte_len: u64,
    /// Input width (elements per row); a whole number of blocks.
    pub in_features: usize,
    /// Output rows.
    pub out_rows: usize,
    buffer: wgpu::Buffer,
}

/// One prepared K3 dispatch set: a bind group plus grid
/// dimensions, built once for a (weights, input, output, batch)
/// combination and encodable any number of times. The bench
/// encodes the same call repeatedly; `run` encodes it once.
#[derive(Debug)]
pub struct K3Call {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    grid: (u32, u32),
}

impl K3Call {
    /// Record this call's dispatch into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k3 q5_k gemv"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.grid.0, self.grid.1, 1);
    }
}

/// The compiled K3 pipeline, in one [`K3Shape`]. Compilation
/// happens once, at backend construction, so Naga translation
/// cost stays out of the token loop (BACKEND.md Section 4).
#[derive(Debug)]
pub struct K3Pipeline {
    pipeline: wgpu::ComputePipeline,
    shape: K3Shape,
}

impl K3Pipeline {
    /// Compile the shipped shape for `device` (see
    /// [`shipped_shape`]).
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] if wgpu rejects the module or
    /// the pipeline.
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        Self::with_shape(device, shipped_shape(device.features()))
    }

    /// Compile an explicit [`K3Shape`] on `device`. The shape
    /// is validated against the device's granted features
    /// first: a shape the device cannot run is a named refusal
    /// here, not a shader validation failure later.
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] on a shape the device cannot run
    /// or a shader wgpu rejects.
    pub fn with_shape(device: &wgpu::Device, shape: K3Shape) -> Result<Self> {
        shape.validate(device.features())?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("k3 q5_k gemv shader"),
            source: wgpu::ShaderSource::Wgsl(shader(&shape).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k3 q5_k gemv pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        dbg_log!("k3 pipeline built: shape {}", shape.name());
        Ok(K3Pipeline { pipeline, shape })
    }

    /// The shape this pipeline runs.
    #[must_use]
    pub fn shape(&self) -> K3Shape {
        self.shape
    }

    /// Upload a Q5_K tensor's raw bytes into a fresh device
    /// buffer. The geometry is validated against the bytes
    /// before anything is uploaded: a mismatch is a refusal
    /// naming the tensor, never a truncated or padded upload.
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
    ) -> Result<K3Weights> {
        if in_features == 0 || in_features % BLOCK_ELEMS != 0 {
            return Err(WgpuError::InvalidTensor {
                name: name.to_string(),
                detail: format!(
                    "input width {in_features} is not a positive whole number of {BLOCK_ELEMS}-element Q5_K blocks"
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
                    "tensor holds {} bytes but [{in_features}, {out_rows}] of Q5_K needs {expected}",
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
            "k3 upload {name}: {} bytes [{in_features} x {out_rows}]",
            bytes.len()
        );
        Ok(K3Weights {
            byte_len: bytes.len() as u64,
            in_features,
            out_rows,
            buffer,
        })
    }

    /// Prepare a dispatch of `weights` over a batch of input
    /// rows. `input` must hold `batch * weights.in_features`
    /// f32 elements and `output` must hold
    /// `batch * weights.out_rows` f32; both are caller-owned
    /// device buffers so the bench can hold them resident
    /// across iterations.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when the batch the caller
    /// declares is zero.
    pub fn prepare(
        &self,
        ctx: &GpuContext,
        weights: &K3Weights,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        batch: u32,
    ) -> Result<K3Call> {
        if batch == 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k3 call".to_string(),
                detail: "batch is zero".to_string(),
            });
        }
        let total =
            u64::from(batch) * (weights.out_rows as u64).div_ceil(u64::from(self.shape.rows));
        let grid = grid_dims(total);
        let params: [u32; 4] = [
            (weights.in_features / BLOCK_ELEMS) as u32,
            weights.out_rows as u32,
            grid.0,
            batch,
        ];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k3 params"),
            size: param_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&params_buffer, 0, &param_bytes);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k3 bind group"),
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
        Ok(K3Call {
            pipeline: self.pipeline.clone(),
            bind_group,
            grid,
        })
    }

    /// Convenience one-shot: upload `input` ([`batch`, in] f32
    /// rows), run the GEMV, read the [`batch`, out] result back
    /// to host. This is the `Compute::proj` and `Compute::lm_head`
    /// data path; it pays an upload and a readback per call,
    /// which the crate docs state plainly.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when `input` is not a whole
    /// number of input rows; [`WgpuError::Readback`] when the
    /// result cannot be mapped back.
    pub fn run(&self, ctx: &GpuContext, weights: &K3Weights, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() || input.len() % weights.in_features != 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k3 input".to_string(),
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
            label: Some("k3 input"),
            size: in_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&input_buffer, 0, &in_bytes);
        let out_elems = batch as usize * weights.out_rows;
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k3 output"),
            size: (out_elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k3 readback"),
            size: (out_elems * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let call = self.prepare(ctx, weights, &input_buffer, &output_buffer, batch)?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k3 run"),
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
            "k3 run: batch={batch} out_rows={} bytes_streamed={}",
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
    fn shape_names_round_trip() {
        for shape in [
            K3Shape::llama(1, 128),
            K3Shape::llama(2, 128),
            K3Shape::llama(4, 128),
            K3Shape::llama(2, 128).with_subgroup(),
            K3Shape::llama(4, 128).with_subgroup(),
        ] {
            assert_eq!(K3Shape::from_name(&shape.name()), Some(shape));
        }
        assert_eq!(K3Shape::from_name("bogus"), None);
        assert_eq!(K3Shape::from_name("r"), None);
        assert_eq!(K3Shape::from_name("r2zz"), None);
        assert_eq!(K3Shape::from_name("v2"), None);
    }

    #[test]
    fn shader_carries_the_shape() {
        let text = shader(&K3Shape::llama(4, 128).with_subgroup());
        assert!(text.contains("@workgroup_size(128)"));
        assert!(text.contains("subgroupShuffleXor"));
        assert!(text.contains("array<f32, 32>")); // 4 rows x 8 groups of partials
        let tree = shader(&K3Shape::llama(2, 64));
        assert!(tree.contains("@workgroup_size(64)"));
        assert!(tree.contains("array<f32, 128>")); // 2 rows x 64 lanes of partials
        assert!(!tree.contains("subgroupShuffleXor"));
    }
}
