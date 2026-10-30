//! K2: Q6_K dequantize-and-multiply GEMV (BACKEND.md Section 3).
//!
//! The second-largest stream in Glimmer: 17.1 percent of bytes
//! per token are Q6_K (the FFN down projection, 19,968 to 6,656,
//! and the attention V projection). Its input in the FFN is the
//! 19,968-wide intermediate; BACKEND.md Section 4 left the
//! staging choice (f16 in workgroup memory versus global memory
//! through the Infinity Cache) to this kernel's bench, and the
//! kernel reads its input from global memory like K1 does, so
//! the bench measures exactly that choice's global side. The
//! LDS-staged variant can be added behind a [`K2Shape`] flag if
//! the numbers ever ask for it.
//!
//! The shader reads the ggml block layout directly (210 bytes
//! per 256-element super-block: 128 bytes of low nibbles `ql`,
//! 64 bytes of high bit-pairs `qh`, 16 signed sub-block scales,
//! one f16 super-scale `d`), expanding values in registers and
//! never materializing a dequantized weight anywhere. The
//! arithmetic mirrors `gguf::dequant::dequant_q6_k` term for
//! term: element value is `d * scale * (q - 32)` with the 6-bit
//! `q` assembled from a `ql` nibble and two `qh` bits, the scale
//! chosen by the element's stream and 16-element half, so GPU
//! results check against the CPU reference within float
//! reassociation noise.
//!
//! One layout concession, stated plainly: a Q6_K super-block is
//! 210 bytes, which is not a multiple of four, so the raw tensor
//! bytes cannot be indexed as `array<u32>` with every block
//! aligned (odd blocks would start two bytes into a word, and
//! WGSL has no unaligned loads). The upload therefore repacks
//! each block into a 212-byte device stride (the 210 raw bytes
//! plus two zero pad bytes) on the host, once, at load time.
//! Every plane offset inside the block is a multiple of four, so
//! the padded stride makes every shader load aligned, and the
//! shader's structure stays the tuned K1 structure instead of
//! growing a per-load phase reconstruction. The stream
//! accounting ([`K2Weights::byte_len`]) stays on the raw file
//! bytes, the useful stream llama.cpp's 708 GB/s is also
//! measured on; the device traffic is 212/210 of that, recorded
//! in BACKEND.md's results.
//!
//! The parallel shape is K1's tuned shape (see
//! `k1_q4k_gemv`'s module docs and BACKEND.md's tuning record):
//! sixteen threads cooperate on one super-block, each thread
//! owning sixteen elements (the four streams' quads at its byte
//! offset within the half, the same assignment llama.cpp's
//! `mul_mat_vec_q6_k.comp` uses); a workgroup covers several
//! output rows so the activation quads each thread loads serve
//! every row in the tile; and the reduction is either a
//! workgroup-memory tree or the segmented subgroup butterfly
//! plus a short shared-memory finish. No f16 activation variant
//! is offered: K1's f16 experiment failed this crate's check
//! tolerance and measured slower than its f32 twin, and this
//! kernel's tolerance bar is the same, so the variant would
//! need its own proof before it could ever be selected.
//!
//! The grid is two-dimensional for the same reason as K1's: one
//! dispatch dimension caps at 65,535 workgroups (BACKEND.md
//! Section 2), and the flat workgroup index wraps over `grid_x`.

use crate::device::GpuContext;
use crate::error::{Result, WgpuError};
use crate::kernels::k1_q4k_gemv::grid_dims;

use crate::diag::dbg_log;

/// Elements per Q6_K super-block.
const BLOCK_ELEMS: usize = 256;
/// Bytes per Q6_K super-block in the file.
const BLOCK_BYTES_RAW: usize = 210;
/// Bytes per Q6_K super-block in the device buffer (raw bytes
/// plus two pad bytes; see the module docs).
const BLOCK_BYTES_DEV: usize = 212;
/// Words per device-stride Q6_K super-block.
const BLOCK_WORDS_DEV: u32 = 53;

/// The parallel shape a K2 pipeline runs, parameterized the
/// same way as [`crate::kernels::k1_q4k_gemv::K1Shape`] (rows
/// per workgroup, workgroup size, reduction kind), minus the
/// legacy and f16 variants K1 carries for its own history.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct K2Shape {
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

impl K2Shape {
    /// A shape with the given tile geometry and the
    /// workgroup-memory reduction.
    #[must_use]
    pub const fn llama(rows: u32, workgroup: u32) -> K2Shape {
        K2Shape {
            rows,
            workgroup,
            subgroup: false,
        }
    }

    /// This shape with the subgroup reduction.
    #[must_use]
    pub const fn with_subgroup(self) -> K2Shape {
        K2Shape {
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

    /// Parse a bench/check shape name (see [`K2Shape::name`]).
    /// The name carries no workgroup size; parsing defaults it
    /// to 128 and callers override it (the bench's `--wg` flag;
    /// 256 measured best for this kernel, see [`shipped_shape`]).
    #[must_use]
    pub fn from_name(name: &str) -> Option<K2Shape> {
        let rest = name.strip_prefix('r')?;
        let (digits, suffix) = rest.split_at(
            rest.find(|c: char| !c.is_ascii_digit())
                .unwrap_or(rest.len()),
        );
        let rows: u32 = digits.parse().ok()?;
        let mut shape = K2Shape::llama(rows, 128);
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
                    "k2 shape {}: rows per workgroup {} is outside 1..=8",
                    self.name(),
                    self.rows
                ),
            });
        }
        if self.workgroup == 0 || self.workgroup % 16 != 0 || self.workgroup > 256 {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "k2 shape {}: workgroup size {} is not a multiple of 16 in 16..=256",
                    self.name(),
                    self.workgroup
                ),
            });
        }
        if self.subgroup && !features.contains(wgpu::Features::SUBGROUP) {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "k2 shape {} needs the SUBGROUP device feature, which this device was not granted",
                    self.name()
                ),
            });
        }
        Ok(())
    }
}

/// The shape [`K2Pipeline::new`] builds for a device with the
/// given granted features. K2's own bench matrix (BACKEND.md's
/// K2/K3 results section) moved this default off K1's pick:
/// on Q6_K's geometry (the big tensor is 6,656 rows of 78
/// blocks) one row per workgroup with a wide 256-thread
/// workgroup measured best, median 831.9 GB/s against 818.6
/// for K1's two-rows-of-128 shape, every variant's runs
/// tightly clustered. The subgroup reduction is used when
/// SUBGROUP was granted, the workgroup-memory tree at the
/// same geometry otherwise.
#[must_use]
pub fn shipped_shape(features: wgpu::Features) -> K2Shape {
    if features.contains(wgpu::Features::SUBGROUP) {
        K2Shape::llama(1, 256).with_subgroup()
    } else {
        K2Shape::llama(1, 256)
    }
}

/// Repack a Q6_K tensor's raw bytes into the device layout:
/// each 210-byte super-block followed by two zero pad bytes,
/// so every block starts on a word boundary (module docs).
/// The caller validates the raw length first; this function
/// assumes a whole number of blocks.
#[must_use]
pub fn repack_q6k(raw: &[u8]) -> Vec<u8> {
    debug_assert_eq!(raw.len() % BLOCK_BYTES_RAW, 0);
    let blocks = raw.len() / BLOCK_BYTES_RAW;
    let mut out = vec![0u8; blocks * BLOCK_BYTES_DEV];
    for (i, block) in raw.chunks_exact(BLOCK_BYTES_RAW).enumerate() {
        out[i * BLOCK_BYTES_DEV..i * BLOCK_BYTES_DEV + BLOCK_BYTES_RAW].copy_from_slice(block);
    }
    out
}

/// The K2 shader for `shape`, generated with the tile geometry
/// as literals so the compiler unrolls the row loop and sizes
/// the reduction arrays exactly. The body follows llama.cpp's
/// `mul_mat_vec_q6_k.comp` (cited in the module docs): thread
/// `t` of a sixteen-thread group owns, within its half `h` of
/// the super-block, the quads of all four streams at byte
/// offset `l0 = 4 * (t % 8)`, with scale index base
/// `8h + (t % 8) / 4`.
fn shader(shape: &K2Shape) -> String {
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

// One signed sub-block scale: byte `j` of the block's 16-byte
// scale plane, whose four words the caller loaded. The shift
// through the top byte sign-extends the int8.
fn scale_at(scw: array<u32, 4>, j: u32) -> f32 {{
    let b = (scw[j >> 2u] >> (8u * (j & 3u))) & 0xffu;
    return f32(bitcast<i32>(b << 24u) >> 24);
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
    let v_in = t % 8u;
    // This thread's word offsets within a super-block (device
    // stride {BLOCK_WORDS_DEV} words): the ql word of stream 0
    // and 1 (byte 64h + 4*v_in into the ql plane, and 32 bytes
    // later), the qh word (byte 128 + 32h + 4*v_in), and the
    // activation quad of stream 0 within the block's 64-quad
    // span (streams sit 8 quads apart).
    let ql0_off = 16u * half + v_in;
    let qh_off = 32u + 8u * half + v_in;
    let yv = 32u * half + v_in;
    // Scale index base: stream s of this half reads scale
    // 8h + is + 2s, where is splits the 32-element streams at
    // element 16.
    let s_off = 8u * half + v_in / 4u;

    var acc = array<f32, {rows}>();
    for (var blk = group; blk < params.blocks_per_row; blk += {groups}u) {{
        // Activation quads are per-block, shared by every row
        // of the tile: loaded once, used {rows} times. xa pairs
        // with stream 0, xb with stream 1 (32 elements later),
        // xc with stream 2 (64 later), xd with stream 3 (96
        // later), exactly ggml's y offsets.
        let xb0 = (batch_row * params.blocks_per_row + blk) * 64u;
        let xa = input[xb0 + yv];
        let xb = input[xb0 + yv + 8u];
        let xc = input[xb0 + yv + 16u];
        let xd = input[xb0 + yv + 24u];
        for (var r = 0u; r < {rows}u; r++) {{
            let row = row0 + r;
            if (row >= params.out_rows) {{
                continue;
            }}
            let bw = (row * params.blocks_per_row + blk) * {BLOCK_WORDS_DEV}u;
            let ql0 = weights[bw + ql0_off];
            let ql32 = weights[bw + ql0_off + 8u];
            let qhw = weights[bw + qh_off];
            // The 6-bit values: ql nibble in the low four bits,
            // the qh bit-pair moved into bits 4 and 5 (the mask
            // positions are ggml's, one per stream). The -32
            // centers the unsigned 6-bit range, as in ggml and
            // in the CPU dequantizer.
            let q0 = vec4<f32>(unpack4xU8((ql0 & 0x0f0f0f0fu) | ((qhw & 0x03030303u) << 4u))) - vec4<f32>(32.0);
            let q1 = vec4<f32>(unpack4xU8((ql32 & 0x0f0f0f0fu) | ((qhw & 0x0c0c0c0cu) << 2u))) - vec4<f32>(32.0);
            let q2 = vec4<f32>(unpack4xU8(((ql0 >> 4u) & 0x0f0f0f0fu) | (qhw & 0x30303030u))) - vec4<f32>(32.0);
            let q3 = vec4<f32>(unpack4xU8(((ql32 >> 4u) & 0x0f0f0f0fu) | ((qhw & 0xc0c0c0c0u) >> 2u))) - vec4<f32>(32.0);
            let scw = array<u32, 4>(
                weights[bw + 48u], weights[bw + 49u],
                weights[bw + 50u], weights[bw + 51u],
            );
            let d = f16_to_f32(weights[bw + 52u] & 0xffffu);
            acc[r] += d * (scale_at(scw, s_off) * dot(q0, xa)
                         + scale_at(scw, s_off + 2u) * dot(q1, xb)
                         + scale_at(scw, s_off + 4u) * dot(q2, xc)
                         + scale_at(scw, s_off + 6u) * dot(q3, xd));
        }}
    }}
{reduction}
}}
"#
    )
}

/// A Q6_K weight tensor resident on the device: one storage
/// buffer holding the tensor's blocks in the padded device
/// layout (module docs), bound whole per dispatch (BACKEND.md
/// Section 5's per-tensor buffer strategy).
#[derive(Debug)]
pub struct K2Weights {
    /// Whole-tensor byte length in the FILE (raw 210-byte
    /// blocks), for the bench's stream accounting; the device
    /// buffer is 212/210 of this.
    pub byte_len: u64,
    /// Input width (elements per row); a whole number of blocks.
    pub in_features: usize,
    /// Output rows.
    pub out_rows: usize,
    buffer: wgpu::Buffer,
}

/// One prepared K2 dispatch set: a bind group plus grid
/// dimensions, built once for a (weights, input, output, batch)
/// combination and encodable any number of times. The bench
/// encodes the same call repeatedly; `run` encodes it once.
#[derive(Debug)]
pub struct K2Call {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    grid: (u32, u32),
}

impl K2Call {
    /// Record this call's dispatch into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k2 q6_k gemv"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.grid.0, self.grid.1, 1);
    }
}

/// The compiled K2 pipeline, in one [`K2Shape`]. Compilation
/// happens once, at backend construction, so Naga translation
/// cost stays out of the token loop (BACKEND.md Section 4).
#[derive(Debug)]
pub struct K2Pipeline {
    pipeline: wgpu::ComputePipeline,
    shape: K2Shape,
}

impl K2Pipeline {
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

    /// Compile an explicit [`K2Shape`] on `device`. The shape
    /// is validated against the device's granted features
    /// first: a shape the device cannot run is a named refusal
    /// here, not a shader validation failure later.
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] on a shape the device cannot run
    /// or a shader wgpu rejects.
    pub fn with_shape(device: &wgpu::Device, shape: K2Shape) -> Result<Self> {
        shape.validate(device.features())?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("k2 q6_k gemv shader"),
            source: wgpu::ShaderSource::Wgsl(shader(&shape).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k2 q6_k gemv pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        dbg_log!("k2 pipeline built: shape {}", shape.name());
        Ok(K2Pipeline { pipeline, shape })
    }

    /// The shape this pipeline runs.
    #[must_use]
    pub fn shape(&self) -> K2Shape {
        self.shape
    }

    /// Upload a Q6_K tensor's raw bytes into a fresh device
    /// buffer, repacking into the padded device layout (module
    /// docs). The geometry is validated against the bytes
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
    ) -> Result<K2Weights> {
        if in_features == 0 || in_features % BLOCK_ELEMS != 0 {
            return Err(WgpuError::InvalidTensor {
                name: name.to_string(),
                detail: format!(
                    "input width {in_features} is not a positive whole number of {BLOCK_ELEMS}-element Q6_K blocks"
                ),
            });
        }
        if out_rows == 0 {
            return Err(WgpuError::InvalidTensor {
                name: name.to_string(),
                detail: "output row count is zero".to_string(),
            });
        }
        let expected = (in_features / BLOCK_ELEMS) * BLOCK_BYTES_RAW * out_rows;
        if bytes.len() != expected {
            return Err(WgpuError::InvalidTensor {
                name: name.to_string(),
                detail: format!(
                    "tensor holds {} bytes but [{in_features}, {out_rows}] of Q6_K needs {expected}",
                    bytes.len()
                ),
            });
        }
        let device_bytes = repack_q6k(bytes);
        ctx.check_tensor_fits(name, device_bytes.len() as u64)?;
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size: device_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&buffer, 0, &device_bytes);
        dbg_log!(
            "k2 upload {name}: {} raw bytes ({} on device) [{in_features} x {out_rows}]",
            bytes.len(),
            device_bytes.len()
        );
        Ok(K2Weights {
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
        weights: &K2Weights,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        batch: u32,
    ) -> Result<K2Call> {
        if batch == 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k2 call".to_string(),
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
            label: Some("k2 params"),
            size: param_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&params_buffer, 0, &param_bytes);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k2 bind group"),
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
        Ok(K2Call {
            pipeline: self.pipeline.clone(),
            bind_group,
            grid,
        })
    }

    /// Convenience one-shot: upload `input` ([`batch`, in] f32
    /// rows), run the GEMV, read the [`batch`, out] result back
    /// to host. This is the `Compute::proj` data path; it pays
    /// an upload and a readback per call, which the crate docs
    /// state plainly.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when `input` is not a whole
    /// number of input rows; [`WgpuError::Readback`] when the
    /// result cannot be mapped back.
    pub fn run(&self, ctx: &GpuContext, weights: &K2Weights, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() || input.len() % weights.in_features != 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k2 input".to_string(),
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
            label: Some("k2 input"),
            size: in_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&input_buffer, 0, &in_bytes);
        let out_elems = batch as usize * weights.out_rows;
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k2 output"),
            size: (out_elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k2 readback"),
            size: (out_elems * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let call = self.prepare(ctx, weights, &input_buffer, &output_buffer, batch)?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k2 run"),
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
            "k2 run: batch={batch} out_rows={} bytes_streamed={}",
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
            K2Shape::llama(1, 128),
            K2Shape::llama(2, 128),
            K2Shape::llama(4, 128),
            K2Shape::llama(2, 128).with_subgroup(),
            K2Shape::llama(4, 128).with_subgroup(),
        ] {
            assert_eq!(K2Shape::from_name(&shape.name()), Some(shape));
        }
        assert_eq!(K2Shape::from_name("bogus"), None);
        assert_eq!(K2Shape::from_name("r"), None);
        assert_eq!(K2Shape::from_name("r2zz"), None);
        assert_eq!(K2Shape::from_name("v2"), None);
    }

    #[test]
    fn repack_pads_each_block_to_the_device_stride() {
        // Three blocks of distinct byte ramps; the device layout
        // must carry each block's 210 bytes verbatim and zero
        // the two pad bytes, including for odd block indices
        // (the blocks whose raw start is two bytes into a word).
        let mut raw = Vec::new();
        for b in 0..3u8 {
            raw.extend((0..210u16).map(|i| (i as u8).wrapping_add(b * 40)));
        }
        let dev = repack_q6k(&raw);
        assert_eq!(dev.len(), 3 * BLOCK_BYTES_DEV);
        for b in 0..3 {
            let src = &raw[b * BLOCK_BYTES_RAW..(b + 1) * BLOCK_BYTES_RAW];
            let dst = &dev[b * BLOCK_BYTES_DEV..b * BLOCK_BYTES_DEV + BLOCK_BYTES_RAW];
            assert_eq!(src, dst, "block {b} payload moved");
            assert_eq!(dev[b * BLOCK_BYTES_DEV + 210], 0);
            assert_eq!(dev[b * BLOCK_BYTES_DEV + 211], 0);
        }
    }

    #[test]
    fn shader_carries_the_shape() {
        let text = shader(&K2Shape::llama(4, 128).with_subgroup());
        assert!(text.contains("@workgroup_size(128)"));
        assert!(text.contains("subgroupShuffleXor"));
        assert!(text.contains("array<f32, 32>")); // 4 rows x 8 groups of partials
        let tree = shader(&K2Shape::llama(2, 64));
        assert!(tree.contains("@workgroup_size(64)"));
        assert!(tree.contains("array<f32, 128>")); // 2 rows x 64 lanes of partials
        assert!(!tree.contains("subgroupShuffleXor"));
    }
}
