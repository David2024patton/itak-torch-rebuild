//! K8: the bookends of the forward (BACKEND.md Section 3): the
//! embedding gather at the front and the head finishing math at
//! the back.
//!
//! **Embedding gather.** The trait feeds the embedding table as
//! a [`TensorView`], and this module accepts exactly the dtypes
//! `cpu-compute` (the spec) accepts there, refusing the rest by
//! name with the tensor and its type: F32, F16, BF16, Q8_0,
//! Q4_K, Q5_K, Q6_K. One kernel per storage family:
//!
//! - F32 rows: a plain indexed copy.
//! - F16 / BF16 rows: a two-byte element conversion (f16 via the
//!   WGSL unpack intrinsic, bf16 by the shift-into-place the
//!   format is).
//! - Quantized rows: the element is dequantized in-kernel from
//!   its block bytes. Every decoder restates `gguf`'s reference
//!   dequantizer instruction for instruction (the K4_K sub-block
//!   scale unpacking shared by Q5_K, Q5_K's shared fifth-bit
//!   array, Q6_K's four interleaved streams, Q8_0's signed
//!   bytes), the same discipline K1's shader keeps for Q4_K
//!   matrices. Elements are addressed globally (token row start
//!   plus channel), so a row that is not a whole number of
//!   blocks gathers exactly the way cpu-compute's whole-tensor
//!   fallback dequantizes it.
//!
//! A gather is one token row per dispatch; a chunk of tokens is
//! one call per row, matching how `cpu-compute`'s `embed` fills
//! its output row by row through `dequantize_rows`.
//!
//! **Head finishing.** The forward's last arithmetic after the
//! head projection: scale every logit by `cfg.logit_scale`, then
//! the tanh softcap `cap * tanh(x / cap)`, in that order (the
//! `muse_glimmer::softcap_scale` law). The kernel here is the GPU
//! form of that math for the device-resident pipeline it will
//! serve: one thread per logit over a two-dimensional grid,
//! because the 202,048 wide vocab exceeds a single dispatch
//! dimension's comfortable reach once rows multiply. The trait's
//! `lm_head` deliberately does NOT apply this finishing (the
//! model applies `softcap_scale` itself after `lm_head`
//! returns, so applying it twice would be a bug); the kernel is
//! proven against `softcap_scale` directly in this crate's
//! tests, which is the seam it will be wired into later.

use muse_glimmer::{Dtype, GlimmerConfig, GlimmerError, Result as GlimmerResult, TensorView};

use crate::device::GpuContext;
use crate::error::{Result, WgpuError};
use crate::kernels::k1_q4k_gemv::grid_dims;

use crate::diag::dbg_log;

/// Build a [`GlimmerError::StepFailed`] for `step` with `detail`.
fn step_failed(step: &'static str, detail: impl Into<String>) -> GlimmerError {
    GlimmerError::StepFailed {
        step,
        detail: detail.into(),
    }
}

/// The mainline ggml type id of a [`Dtype`], for naming types in
/// refusals (mirrors the table in the sibling compute modules).
fn dtype_id(dtype: Dtype) -> u32 {
    match dtype {
        Dtype::F32 => 0,
        Dtype::F16 => 1,
        Dtype::Q4_0 => 2,
        Dtype::Q8_0 => 8,
        Dtype::Q4K => 12,
        Dtype::Q5K => 13,
        Dtype::Q6K => 14,
        Dtype::Bf16 => 30,
        Dtype::Unknown(id) => id,
    }
}

/// Threads per workgroup for the gather and finish kernels.
const WORKGROUP: u32 = 256;

/// The F32 gather shader: one thread per output channel.
const GATHER_F32_SHADER: &str = r#"
struct GatherParams {
    width: u32,
    row_start: u32,
}

@group(0) @binding(0) var<storage, read> table: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: GatherParams;

@compute @workgroup_size(256)
fn gather_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c < params.width) {
        out[c] = table[params.row_start + c];
    }
}
"#;

/// The two-byte gather shader (F16 and BF16 element conversion).
const GATHER_HALF_SHADER: &str = r#"
struct GatherParams {
    width: u32,
    row_start: u32,
    dtype: u32,
}

@group(0) @binding(0) var<storage, read> table: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: GatherParams;

@compute @workgroup_size(256)
fn gather_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= params.width) {
        return;
    }
    let g = params.row_start + c;
    let bits = (table[g / 2u] >> (16u * (g % 2u))) & 0xFFFFu;
    if (params.dtype == 0u) {
        // F16: widen through the packed unpack intrinsic (the
        // second lane is padding).
        out[c] = unpack2x16float(bits).x;
    } else {
        // BF16: the value is the top half of an f32.
        out[c] = bitcast<f32>(bits << 16u);
    }
}
"#;

/// The quantized gather shader: one thread per output channel,
/// dequantizing its own element from the block bytes. Block
/// layouts and decoders restate `gguf`'s reference
/// implementations; dtype codes: 0 = Q4_K, 1 = Q6_K, 2 = Q5_K,
/// 3 = Q8_0.
const GATHER_QUANT_SHADER: &str = r#"
struct GatherParams {
    width: u32,
    row_start: u32,
    dtype: u32,
}

@group(0) @binding(0) var<storage, read> table: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: GatherParams;

// One byte of the table. The u32 view's word and lane are
// host-endian: the host uploads little-endian bytes and reads
// them back on the same machine class (see the K1 shader's note
// on the same pattern).
fn byte_at(i: u32) -> u32 {
    return (table[i / 4u] >> (8u * (i % 4u))) & 0xFFu;
}

fn f16_at(i: u32) -> f32 {
    let bits = byte_at(i) | (byte_at(i + 1u) << 8u);
    return unpack2x16float(bits).x;
}

// K4_K scale/min unpacking, gguf's get_scale_min_k4 restated.
// `b` is the block's byte base; scales start at b + 4.
fn k4_scale(b: u32, j: u32) -> vec2<u32> {
    if (j < 4u) {
        return vec2<u32>(byte_at(b + 4u + j) & 63u, byte_at(b + 4u + j + 4u) & 63u);
    }
    let sc = (byte_at(b + 4u + j + 4u) & 0xFu) | ((byte_at(b + 4u + j - 4u) >> 6u) << 4u);
    let mn = (byte_at(b + 4u + j + 4u) >> 4u) | ((byte_at(b + 4u + j) >> 6u) << 4u);
    return vec2<u32>(sc, mn);
}

fn dequant_q4k(g: u32) -> f32 {
    let b = (g / 256u) * 144u;
    let within = g % 256u;
    let d = f16_at(b);
    let dmin = f16_at(b + 2u);
    let group = within / 64u;
    let lane = within % 64u;
    let sm = k4_scale(b, 2u * group + lane / 32u);
    let packed = byte_at(b + 16u + group * 32u + lane % 32u);
    var q = packed & 0xFu;
    if (lane >= 32u) {
        q = packed >> 4u;
    }
    return d * f32(sm.x) * f32(q) - dmin * f32(sm.y);
}

fn dequant_q5k(g: u32) -> f32 {
    let b = (g / 256u) * 176u;
    let within = g % 256u;
    let d = f16_at(b);
    let dmin = f16_at(b + 2u);
    let group = within / 64u;
    let lane = within % 64u;
    let sub = lane / 32u;
    let sm = k4_scale(b, 2u * group + sub);
    let packed = byte_at(b + 48u + group * 32u + lane % 32u);
    var q = packed & 0xFu;
    if (sub == 1u) {
        q = packed >> 4u;
    }
    // The fifth bits share one 32-byte array: element lane l of
    // group g reads bit 2g (first sub-block) or 2g+1 (second) of
    // qh[l].
    let qh = byte_at(b + 16u + lane % 32u);
    if ((qh >> (2u * group + sub)) & 1u) == 1u {
        q += 16u;
    }
    return d * f32(sm.x) * f32(q) - dmin * f32(sm.y);
}

fn dequant_q6k(g: u32) -> f32 {
    let b = (g / 256u) * 210u;
    let within = g % 256u;
    let half = within / 128u;
    let l = within % 128u;
    // The reference's four 32-wide streams within a half: stream
    // n reads byte (n % 2) * 32 + e of the half's 64 ql bytes,
    // the low nibble for n < 2 and the high nibble above, and
    // bits 2n, 2n+1 of the qh byte at index e.
    let n = l / 32u;
    let e = l % 32u;
    let packed = byte_at(b + half * 64u + (n % 2u) * 32u + e);
    var q = packed & 0xFu;
    if (n >= 2u) {
        q = packed >> 4u;
    }
    let qbits = byte_at(b + 128u + half * 32u + e);
    q = q | (((qbits >> (2u * n)) & 1u) << 4u) | (((qbits >> (2u * n + 1u)) & 1u) << 5u);
    var sc = i32(byte_at(b + 192u + half * 8u + l / 16u));
    if (sc >= 128) {
        sc -= 256;
    }
    let d = f16_at(b + 208u);
    return d * f32(sc) * (f32(q) - 32.0);
}

fn dequant_q8_0(g: u32) -> f32 {
    let b = (g / 32u) * 34u;
    let within = g % 32u;
    let d = f16_at(b);
    var q = i32(byte_at(b + 2u + within));
    if (q >= 128) {
        q -= 256;
    }
    return d * f32(q);
}

@compute @workgroup_size(256)
fn gather_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= params.width) {
        return;
    }
    let g = params.row_start + c;
    if (params.dtype == 0u) {
        out[c] = dequant_q4k(g);
    } else if (params.dtype == 1u) {
        out[c] = dequant_q6k(g);
    } else if (params.dtype == 2u) {
        out[c] = dequant_q5k(g);
    } else {
        out[c] = dequant_q8_0(g);
    }
}
"#;

/// The head finishing shader: scale, then tanh softcap.
const FINISH_SHADER: &str = r#"
struct FinishParams {
    n: u32,
    scale: f32,
    cap: f32,
    grid_x: u32,
}

@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<uniform> params: FinishParams;

@compute @workgroup_size(256)
fn finish_main(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = (wid.y * params.grid_x + wid.x) * 256u + lid.x;
    if (i < params.n) {
        let x = logits[i] * params.scale;
        logits[i] = params.cap * tanh(x / params.cap);
    }
}
"#;

/// Dtype codes for the quantized gather's uniform.
const CODE_Q4_K: u32 = 0;
const CODE_Q6_K: u32 = 1;
const CODE_Q5_K: u32 = 2;
const CODE_Q8_0: u32 = 3;

/// Dtype codes for the two-byte gather's uniform.
const CODE_F16: u32 = 0;
const CODE_BF16: u32 = 1;

/// The compiled K8 pipelines (F32 gather, two-byte gather,
/// quantized gather, head finish).
#[derive(Debug)]
pub struct K8Pipeline {
    gather_f32: wgpu::ComputePipeline,
    gather_half: wgpu::ComputePipeline,
    gather_quant: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
}

impl K8Pipeline {
    /// Compile all four K8 pipelines on `device`.
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
        let gather_f32 = build(GATHER_F32_SHADER, "gather_main", "k8 gather f32");
        let gather_half = build(GATHER_HALF_SHADER, "gather_main", "k8 gather half");
        let gather_quant = build(GATHER_QUANT_SHADER, "gather_main", "k8 gather quant");
        let finish = build(FINISH_SHADER, "finish_main", "k8 head finish");
        dbg_log!("k8 pipelines built");
        Ok(K8Pipeline {
            gather_f32,
            gather_half,
            gather_quant,
            finish,
        })
    }

    /// Dispatch one gather pipeline for one token row and read
    /// the row back. `table` is the already-uploaded table
    /// buffer, `code` the family's dtype code (unused by the F32
    /// pipeline, whose uniform has no code field; callers pass
    /// the right pipeline for the family).
    fn gather_row(
        &self,
        ctx: &GpuContext,
        pipeline: &wgpu::ComputePipeline,
        table: &wgpu::Buffer,
        row_start: usize,
        width: usize,
        code: Option<u32>,
    ) -> Result<Vec<f32>> {
        let out_buf = ctx.scratch_buffer("k8 embedding row", (width * 4) as u64);
        let param_bytes: Vec<u8> = match code {
            Some(code) => [width as u32, row_start as u32, code]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            None => [width as u32, row_start as u32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        };
        let params_buf = ctx.uniform_buffer("k8 gather params", &param_bytes);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k8 gather bind group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: table.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let total_wg = (width as u64).div_ceil(u64::from(WORKGROUP));
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k8 gather run"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("k8 embedding gather"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(total_wg as u32, 1, 1);
        }
        ctx.queue.submit([encoder.finish()]);
        ctx.read_f32(&out_buf, width)
    }

    /// Gather the embedding rows for `tokens`, row-major
    /// `[tokens][hidden]`, validating exactly what `cpu-compute`
    /// validates (table shape against the config, byte length
    /// against the dtype's block geometry, token range) and
    /// accepting exactly its dtype set.
    ///
    /// # Errors
    ///
    /// [`GlimmerError::StepFailed`] under `"embed"` on any
    /// refusal; device failures are wrapped under the same step.
    pub fn embed(
        &self,
        ctx: &GpuContext,
        table: &TensorView,
        tokens: &[u32],
        cfg: &GlimmerConfig,
    ) -> GlimmerResult<Vec<f32>> {
        let step = "embed";
        let hidden = cfg.hidden as usize;
        let vocab = cfg.vocab_size as usize;
        if table.shape.len() != 2
            || table.shape[0] as usize != hidden
            || table.shape[1] as usize != vocab
        {
            return Err(step_failed(
                step,
                format!(
                    "embedding table {} has shape {:?}, expected [{}, {}] in GGUF stored order",
                    table.name, table.shape, hidden, cfg.vocab_size
                ),
            ));
        }
        let ty = match table.dtype {
            Dtype::F32 => gguf::GgmlType::F32,
            Dtype::F16 => gguf::GgmlType::F16,
            Dtype::Q8_0 => gguf::GgmlType::Q8_0,
            Dtype::Q4K => gguf::GgmlType::Q4K,
            Dtype::Q5K => gguf::GgmlType::Q5K,
            Dtype::Q6K => gguf::GgmlType::Q6K,
            Dtype::Bf16 => gguf::GgmlType::Bf16,
            other => {
                return Err(step_failed(
                    step,
                    format!(
                        "tensor {} has dtype {} (ggml type id {}), which the wgpu backend cannot dequantize for the embedding gather",
                        table.name,
                        gguf::ggml::type_name(dtype_id(other)),
                        dtype_id(other)
                    ),
                ))
            }
        };
        let elements = hidden * vocab;
        let expected = ty.byte_len(elements as u64).ok_or_else(|| {
            step_failed(
                step,
                format!(
                    "tensor {} has {elements} elements, not a whole number of {} blocks",
                    table.name,
                    gguf::ggml::type_name(ty.id())
                ),
            )
        })?;
        if expected != table.data.len() as u64 {
            return Err(step_failed(
                step,
                format!(
                    "tensor {} holds {} bytes but {elements} elements of {} need {expected}",
                    table.name,
                    table.data.len(),
                    gguf::ggml::type_name(ty.id())
                ),
            ));
        }
        for &token in tokens {
            if token as usize >= vocab {
                return Err(step_failed(
                    step,
                    format!(
                        "tensor {} has {vocab} rows, too few for the requested token {token}",
                        table.name
                    ),
                ));
            }
        }

        // One upload of the whole table, one dispatch per token
        // row. (The real table is 1.3 billion elements in Q4_K;
        // uploading it per embed call is the trait-seam cost this
        // landing accepts, the same one-shot shape as K1's run
        // before the weight cache. The gather kernels themselves
        // are what this module proves.)
        let as_err = |e: WgpuError| step_failed(step, e.to_string());
        let mut out = Vec::with_capacity(tokens.len() * hidden);
        match table.dtype {
            Dtype::F32 => {
                let values: Vec<f32> = table
                    .data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let buf = ctx.upload_f32("k8 embedding table", &values);
                for &token in tokens {
                    let row = self
                        .gather_row(
                            ctx,
                            &self.gather_f32,
                            &buf,
                            token as usize * hidden,
                            hidden,
                            None,
                        )
                        .map_err(as_err)?;
                    out.extend_from_slice(&row);
                }
            }
            Dtype::F16 | Dtype::Bf16 => {
                let code = if table.dtype == Dtype::F16 {
                    CODE_F16
                } else {
                    CODE_BF16
                };
                let buf = ctx.upload_bytes("k8 embedding table", &table.data);
                for &token in tokens {
                    let row = self
                        .gather_row(
                            ctx,
                            &self.gather_half,
                            &buf,
                            token as usize * hidden,
                            hidden,
                            Some(code),
                        )
                        .map_err(as_err)?;
                    out.extend_from_slice(&row);
                }
            }
            Dtype::Q4K | Dtype::Q5K | Dtype::Q6K | Dtype::Q8_0 => {
                let code = match table.dtype {
                    Dtype::Q4K => CODE_Q4_K,
                    Dtype::Q6K => CODE_Q6_K,
                    Dtype::Q5K => CODE_Q5_K,
                    _ => CODE_Q8_0,
                };
                let buf = ctx.upload_bytes("k8 embedding table", &table.data);
                for &token in tokens {
                    let row = self
                        .gather_row(
                            ctx,
                            &self.gather_quant,
                            &buf,
                            token as usize * hidden,
                            hidden,
                            Some(code),
                        )
                        .map_err(as_err)?;
                    out.extend_from_slice(&row);
                }
            }
            _ => unreachable!("the dtype was matched above"),
        }
        dbg_log!("k8 embed: {} tokens, hidden {hidden}", tokens.len());
        Ok(out)
    }

    /// The head finishing math over a logits buffer: upload,
    /// scale-then-softcap, read back. The GPU form of
    /// `muse_glimmer::softcap_scale`; see the module docs for
    /// why the trait does not call this today.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when `cap` is not positive
    /// and finite; [`WgpuError::Readback`] on map failure.
    pub fn finish(
        &self,
        ctx: &GpuContext,
        logits: &[f32],
        scale: f32,
        cap: f32,
    ) -> Result<Vec<f32>> {
        if !cap.is_finite() || cap <= 0.0 {
            return Err(WgpuError::InvalidTensor {
                name: "k8 finish".to_string(),
                detail: format!("softcap {cap} is not positive and finite"),
            });
        }
        if logits.is_empty() {
            return Ok(Vec::new());
        }
        let buf = ctx.upload_f32("k8 logits", logits);
        let total_wg = (logits.len() as u64).div_ceil(u64::from(WORKGROUP));
        let grid = grid_dims(total_wg);
        let params: [u32; 4] = [logits.len() as u32, scale.to_bits(), cap.to_bits(), grid.0];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buf = ctx.uniform_buffer("k8 finish params", &param_bytes);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k8 finish bind group"),
            layout: &self.finish.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
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
                label: Some("k8 finish run"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("k8 head finish"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.finish);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(grid.0, grid.1, 1);
        }
        ctx.queue.submit([encoder.finish()]);
        dbg_log!("k8 finish: n={} scale={scale} cap={cap}", logits.len());
        ctx.read_f32(&buf, logits.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaders_carry_their_entry_points() {
        assert!(GATHER_F32_SHADER.contains("fn gather_main"));
        assert!(GATHER_HALF_SHADER.contains("fn gather_main"));
        assert!(GATHER_QUANT_SHADER.contains("fn dequant_q4k"));
        assert!(GATHER_QUANT_SHADER.contains("fn dequant_q5k"));
        assert!(GATHER_QUANT_SHADER.contains("fn dequant_q6k"));
        assert!(GATHER_QUANT_SHADER.contains("fn dequant_q8_0"));
        assert!(FINISH_SHADER.contains("fn finish_main"));
        // The finish order is pinned: scale first, then the cap.
        assert!(FINISH_SHADER.contains("logits[i] * params.scale"));
        assert!(FINISH_SHADER.contains("params.cap * tanh"));
    }
}
