//! K21F: the finish-carrying form of K21, the Q5_K
//! register-streaming small-batch product with the
//! head's fused finishing map (logit scale, tanh
//! softcap) applied at the reduction's final store.
//! It exists to give K21 a production caller: the
//! resident session's Q5_K head sites.
//!
//! Why this module exists is the 2026-11-06 server
//! verdict in bench/mmv-q5k-2026-11-05.md, stated
//! plainly: K21 is correct (card-executed against the
//! CPU reference and against K3 at batches 2, 3, 5,
//! 7, 9, 16, both reduction shapes), its compiled
//! form is clean (zero spills, zero scratch, LDS
//! 1,024 B, VGPR 96 plateau at batch 16), and it beat
//! the K3 GEMV at nearly every in-band row of the
//! sweep harness's Q5_K cells. It also fired ZERO
//! times in the flagged real resident runs (46,739
//! per-dispatch diag lines, none of them
//! `k21_q5k_mmv`), because on the Muse-Glimmer-30B
//! Q4_K_M file the Q5_K tensors in the resident
//! forward are the head, and the head sites prepare
//! K3 with a fused finish (`prepare_with_finish`,
//! scale and softcap) that plain K21 deliberately
//! does not carry. A fast, correct kernel with no
//! production caller: this variant is the caller
//! shaped for it.
//!
//! The finish, and why fusing it here is the same
//! math the shipped head computes:
//!
//! - The finishing map is a pure elementwise function
//!   of the fully reduced dot product: mode
//!   [`FINISH_SCALE`] is `v * scale`; mode
//!   [`FINISH_SOFTCAP`] is `cap * tanh(scale * v /
//!   cap)` (K8's finish expressions, in K8's order).
//!   K3 applies it at the reduction's final store,
//!   the only store of each output element. This
//!   kernel applies the identical WGSL function
//!   ([`FINISH_APPLY_WGSL`], K3's text whole) at the
//!   identical position: each output element's only
//!   store, in both reduction shapes. Nothing about
//!   the map's inputs changes: it sees the same
//!   reduced sum the K3 head's map sees.
//! - The product under the map is K21's, whose
//!   dequant and dot text is K3's verbatim, so the
//!   raw sums agree with K3's inside the
//!   reassociation bound the family is already gated
//!   on (card-measured 2.0e-3 to 2.9e-3 absolute on
//!   fixture scales of 1.8e4 to 3.3e4). The map then
//!   SHRINKS any disagreement: it is Lipschitz with
//!   constant `scale` in both modes (scale mode
//!   exactly; softcap because the derivative of
//!   `cap * tanh(scale * v / cap)` is
//!   `scale * sech^2(..)`, at most `scale`). At
//!   Glimmer's logit scale (0.19611613) the finished
//!   outputs therefore agree to at most about a fifth
//!   of the raw bound. The one evaluation that could
//!   differ in principle, `tanh`, is the driver's
//!   WGSL `tanh` in both kernels on the same device,
//!   so the production comparison (this kernel
//!   against the shipped K3 finish) shares even that;
//!   the tests additionally bound it against a host
//!   f32 composition with slack for ulp-level `tanh`
//!   differences.
//! - The mode, scale, and cap travel exactly as
//!   K3's do: uniform words in the params tail (mode
//!   as a u32, scale and cap as f32 bit patterns),
//!   the mode constants are K3's own
//!   ([`FINISH_SCALE`], [`FINISH_SOFTCAP`], imported,
//!   never redefined here), and the prepare-time
//!   validation is K3's: `FINISH_NONE` and unknown
//!   modes refuse by name ("is not a finishing
//!   mode"), a softcap with a non-positive cap
//!   refuses by name ("is not positive"). A caller
//!   that asks this pipeline for a finish and does
//!   not get one fails at prepare, never silently at
//!   the store.
//!
//! What this variant covers, and what it does not:
//!
//! - The block head: the session prepares it at the
//!   chunk's full batch (`rows`), which lands inside
//!   this kernel's 2..=16 band on the speculative
//!   paths (the DFlash block is 16 rows; the borrowed
//!   Q5_K head is the single largest stream in the
//!   DFlash pass, bench/dflash-gpu-2026-11-03.md, so
//!   this is where the bytes are). Batches past the
//!   band refuse by name at prepare and the ladder
//!   keeps K3, exactly the family's composition
//!   contract.
//! - NOT the standard head: it runs at batch 1,
//!   below the band floor, and stays K3 by
//!   construction. Decode is the GEMV's, here as
//!   everywhere in the family.
//! - NOT the plain projections: they carry no finish
//!   and keep routing through `prepare_proj` to K21
//!   and K3 as today.
//! - Its dispatch count in a plan is 1, the same as
//!   K3's fused head dispatch and K21's own: routing
//!   the block head here changes the product engine,
//!   not the plan's dispatch accounting, and no K8
//!   finish dispatch appears or disappears (the Q5_K
//!   head never paid one).
//!
//! Routing is deliberately NOT in this module: the
//! backend registers the pipeline at construction
//! (`WgpuCompute::k21_finish`), and the session's
//! block-head site prepares it through
//! `try_mmv_q5k_finish` (beside `try_mmv_q5k`),
//! falling back to K3 on any refusal. The route is
//! governed by TWO gates, not the family threshold
//! alone: the backend's dedicated head switch
//! (`WGPU_COMPUTE_K21F_HEAD`, resolved in compute.rs;
//! on when unset or `1`, off on `0` or any
//! unrecognized value; default off under the guard
//! written after the 2026-11-07 defaults-confirmation
//! identity failure, see
//! bench/k21f-default-guard-2026-11-07.md, and back
//! to on by default since the wave18 retest met the
//! guard's reopen criterion, see
//! bench/k21f-retest-2026-11-07.md and
//! bench/k21f-default-flip-2026-11-07.md) AND the
//! family's shared `WGPU_COMPUTE_MMV_MAX_ROWS` band
//! admitting the block batch. Consistent with the
//! family, this module adds no env var of its own;
//! the switch is the backend's, not the kernel's.
//! The proof sequence that governs the route is in
//! bench/k21-finish-2026-11-06.md.
//!
//! The generator below mirrors K21's `mmv_shader`
//! structure line for line (the crate's per-kernel
//! generator idiom: K19, K20, and K21 each carry
//! their own), with exactly three differences: the
//! params tail carries the finish words, the
//! `finish_apply` function rides along, and the two
//! reduction store sites wrap their value in it. The
//! body, the dequant, the dot math, and the
//! reduction order are K21's text unchanged, so the
//! one structural variable against plain K21 is the
//! finish at the store, and against K3 it is the
//! workgroup organization, never the arithmetic.
//! The shared half-conversion text is K21's own
//! constant, bound by reference so the two
//! generators cannot drift on it.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::device::GpuContext;
use crate::error::{Result, WgpuError};
use crate::kernels::k1_q4k_gemv::grid_dims;
use crate::kernels::k21_q5k_mmv::{
    shipped_shape as k21_shipped_shape, K21Shape, F16_TO_F32_WGSL, MMV_MAX_BATCH, MMV_MIN_BATCH,
};
use crate::kernels::k3_q5k_gemv::{K3Weights, FINISH_SCALE, FINISH_SOFTCAP};

use crate::diag::dbg_log;

/// Elements per Q5_K super-block.
const BLOCK_ELEMS: usize = 256;

/// Check a finish request the way K3's
/// `prepare_with_finish` checks it (the shipped
/// head's contract, kept identical so the two
/// pipelines refuse the same requests with the same
/// words): the mode must be a finishing mode, and a
/// softcap's cap must be positive. Factored out of
/// the prepare so the refusals are testable without
/// a device.
///
/// # Errors
///
/// [`WgpuError::InvalidTensor`] on an unknown or
/// non-finishing mode, or a softcap mode with a
/// non-positive cap.
fn validate_finish(mode: u32, cap: f32) -> Result<()> {
    if mode != FINISH_SCALE && mode != FINISH_SOFTCAP {
        return Err(WgpuError::InvalidTensor {
            name: "k21 finish call".to_string(),
            detail: format!("finish mode {mode} is not a finishing mode"),
        });
    }
    if mode == FINISH_SOFTCAP && cap <= 0.0 {
        return Err(WgpuError::InvalidTensor {
            name: "k21 finish call".to_string(),
            detail: format!("softcap {cap} is not positive"),
        });
    }
    Ok(())
}

/// The params words a finish dispatch binds, in the
/// shader's `Params` field order: the product
/// geometry K21's params carry, then K3's finish
/// tail (mode, then the scale and cap as f32 bit
/// patterns). Factored out so the word order is
/// testable without a device: a transposed tail
/// would apply the wrong map and only a test that
/// pins the order catches it on a GPU-less machine.
fn finish_params(
    blocks_per_row: u32,
    out_rows: u32,
    grid_x: u32,
    mode: u32,
    scale: f32,
    cap: f32,
) -> [u32; 6] {
    [
        blocks_per_row,
        out_rows,
        grid_x,
        mode,
        scale.to_bits(),
        cap.to_bits(),
    ]
}

/// The head finishing map, K3's WGSL function whole
/// (see `k3_q5k_gemv`'s shader): applied once per
/// output element at the reduction's final store.
/// Mode 1 applies the logit scale; mode 2 applies
/// the scale and the tanh softcap, in K8's finish
/// shader's exact expression order. Mode 0 returns
/// the sum untouched and exists only because the
/// text is K3's whole function, kept whole on
/// purpose: [`K21FinishPipeline::prepare_with_finish`]
/// refuses it by name before any dispatch is built,
/// exactly as K3's does.
const FINISH_APPLY_WGSL: &str = r#"
// The head finishing map, applied once per output element at
// the reduction's final store. Mode 0 returns the sum untouched
// (bitwise the plain kernel); mode 1 applies the logit
// scale; mode 2 applies the scale and the tanh softcap, in
// K8's finish shader's exact expression order.
fn finish_apply(v: f32) -> f32 {
    if (params.finish_mode == 1u) {
        return v * params.finish_scale;
    }
    if (params.finish_mode == 2u) {
        let x = v * params.finish_scale;
        return params.finish_cap * tanh(x / params.finish_cap);
    }
    return v;
}
"#;

/// The finish-carrying K21 shader for `shape` at
/// `batch`: K21's generated shader (see
/// `k21_q5k_mmv`'s module docs for the shape and the
/// generation-literal rationale) with the finish
/// params in the uniform tail, K3's `finish_apply`
/// in the prelude, and the finish applied at the
/// reduction's two store sites.
fn finish_shader(shape: &K21Shape, batch: u32) -> String {
    let rows = shape.rows;
    let wg = shape.workgroup;
    let groups = wg / 16;
    let slices = rows * batch;
    let partials_len = if shape.subgroup { slices * groups } else { wg };

    // The main loop's body, unrolled per output row
    // of the tile and per batch row: K21's text
    // unchanged (the header, the packed scale
    // extraction, the qh word, the dequantized
    // register quads per row; the activation quads
    // and K3's dot math per (row, batch) pair).
    let mut body = String::new();
    for r in 0..rows {
        body.push_str(&format!(
            "        {{\n            let row = row0 + {r}u;\n            if (row < params.out_rows) {{\n"
        ));
        body.push_str(
            "                let bw = (row * params.blocks_per_row + blk) * 44u;\n                let hw = weights[bw];\n                let d = f16_to_f32(hw & 0xffffu);\n                let dmin = f16_to_f32(hw >> 16u);\n                let s16 = (weights[bw + 1u] >> (16u * half)) & 0xffffu;\n                let m16 = (weights[bw + 2u] >> (16u * half)) & 0xffffu;\n                let h16 = (weights[bw + 3u] >> (16u * half)) & 0xffffu;\n                let lo = (m16 << 16u) | s16;\n                let sc_lo = vec4<f32>(unpack4xU8(lo & 0x3f3f3f3fu));\n                let hi = (((h16 << 12u) | h16) & 0x0f0f0f0fu) | ((lo & 0xc0c0c0c0u) >> 2u);\n                let sc_hi = vec4<f32>(unpack4xU8(hi));\n                let qhw = weights[bw + qh_off];\n                let bits_a = vec4<f32>(unpack4xU8((qhw >> (2u * half)) & 0x01010101u));\n                let bits_b = vec4<f32>(unpack4xU8((qhw >> (2u * half + 1u)) & 0x01010101u));\n                let bits_c = vec4<f32>(unpack4xU8((qhw >> (2u * half + 4u)) & 0x01010101u));\n                let bits_d = vec4<f32>(unpack4xU8((qhw >> (2u * half + 5u)) & 0x01010101u));\n                let w0 = weights[bw + w0_off];\n                let w1 = weights[bw + w0_off + 16u];\n                let qa = vec4<f32>(unpack4xU8(w0 & 0x0f0f0f0fu)) + 16.0 * bits_a;\n                let qb = vec4<f32>(unpack4xU8((w0 >> 4u) & 0x0f0f0f0fu)) + 16.0 * bits_b;\n                let qc = vec4<f32>(unpack4xU8(w1 & 0x0f0f0f0fu)) + 16.0 * bits_c;\n                let qd = vec4<f32>(unpack4xU8((w1 >> 4u) & 0x0f0f0f0fu)) + 16.0 * bits_d;\n",
        );
        for b in 0..batch {
            let acc_idx = r * batch + b;
            body.push_str(&format!(
                "                {{\n                    let xb0 = ({b}u * params.blocks_per_row + blk) * 64u;\n                    let xa = input[xb0 + yv];\n                    let xbq = input[xb0 + yv + 8u];\n                    let xc = input[xb0 + yv + 32u];\n                    let xd = input[xb0 + yv + 40u];\n                    let sa = dot(xa, vec4<f32>(1.0));\n                    let sb = dot(xbq, vec4<f32>(1.0));\n                    let sc_ = dot(xc, vec4<f32>(1.0));\n                    let sd = dot(xd, vec4<f32>(1.0));\n                    acc[{acc_idx}] += d * (sc_lo.x * dot(qa, xa) + sc_lo.y * dot(qb, xbq)\n                            + sc_hi.x * dot(qc, xc) + sc_hi.y * dot(qd, xd))\n                        - dmin * (sc_lo.z * sa + sc_lo.w * sb + sc_hi.z * sc_ + sc_hi.w * sd);\n                }}\n"
            ));
        }
        body.push_str("            }\n        }\n");
    }

    // The reduction, generated per accumulator slice
    // as in K21's generator; the ONLY textual
    // difference from K21's is at the store: the
    // reduced sum passes through the finish before
    // it lands, in both shapes.
    let reduction = if shape.subgroup {
        let mut red = String::new();
        for s in 0..slices {
            red.push_str(&format!(
                "    {{ var v = acc[{s}]; v += subgroupShuffleXor(v, 1u); v += subgroupShuffleXor(v, 2u); v += subgroupShuffleXor(v, 4u); v += subgroupShuffleXor(v, 8u); acc[{s}] = v; }}\n"
            ));
        }
        red.push_str("    if (t == 0u) {\n");
        for s in 0..slices {
            red.push_str(&format!(
                "        partials[{idx} * {groups}u + group] = acc[{s}];\n",
                idx = s
            ));
        }
        red.push_str("    }\n    workgroupBarrier();\n    if (lane == 0u) {\n");
        for s in 0..slices {
            let r = s / batch;
            let b = s % batch;
            red.push_str(&format!(
                "        {{ var sum = 0.0; for (var g = 0u; g < {groups}u; g++) {{ sum += partials[{s} * {groups}u + g]; }} if (row0 + {r}u < params.out_rows) {{ output[{b}u * params.out_rows + row0 + {r}u] = finish_apply(sum); }} }}\n"
            ));
        }
        red.push_str("    }\n");
        red
    } else {
        let half_wg = wg / 2;
        let mut red = String::new();
        for s in 0..slices {
            let r = s / batch;
            let b = s % batch;
            red.push_str(&format!(
                "    partials[lane] = acc[{s}];\n    workgroupBarrier();\n    {{ var stride = {half_wg}u; while (stride > 0u) {{ if (lane < stride) {{ partials[lane] += partials[lane + stride]; }} workgroupBarrier(); stride >>= 1u; }} }}\n    if (lane == 0u && row0 + {r}u < params.out_rows) {{ output[{b}u * params.out_rows + row0 + {r}u] = finish_apply(partials[0]); }}\n"
            ));
        }
        red
    };

    format!(
        r#"struct Params {{
    blocks_per_row: u32,
    out_rows: u32,
    grid_x: u32,
    finish_mode: u32,
    finish_scale: f32,
    finish_cap: f32,
}}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

// Reduction partials only: the one workgroup array in
// the shader, carrying per-group (or per-lane) sums to
// the finish. No weight or activation tile is staged
// anywhere; weights dequantize in registers and the
// activations are read from the input buffer per row.
var<workgroup> partials: array<f32, {partials_len}>;
{F16_TO_F32_WGSL}{FINISH_APPLY_WGSL}
@compute @workgroup_size({wg})
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
    let row_groups = (params.out_rows + {rows}u - 1u) / {rows}u;
    let flat = wid.y * params.grid_x + wid.x;
    // Whole-workgroup early out: every lane shares
    // `flat`, so the exit is uniform and the barriers
    // in the reduction stay legal.
    if (flat >= row_groups) {{
        return;
    }}
    let row0 = flat * {rows}u;
    let lane = lid.x;
    let group = lane / 16u;
    let t = lane % 16u;
    let half = t / 8u;
    let l0 = 4u * (t % 8u);
    // This thread's word offsets within a super-block:
    // the qh plane starts at word 4 (after the header
    // words), the ql plane at word 12; the thread's
    // second ql word sits 64 bytes later. Its first
    // activation quad sits within the block's 64-quad
    // span: K21's assignment, unchanged.
    let w0_off = 12u + 8u * half + l0 / 4u;
    let qh_off = 4u + l0 / 4u;
    let yv = 16u * half + l0 / 4u;

    var acc = array<f32, {slices}>();
    for (var blk = group; blk < params.blocks_per_row; blk += {groups}u) {{
{body}    }}
{reduction}}}
"#
    )
}

/// One prepared finish-carrying K21 dispatch: a bind
/// group plus grid dimensions, built once for a
/// (weights, input, output, batch, finish)
/// combination and encodable any number of times.
#[derive(Debug)]
pub struct K21FinishCall {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    grid: (u32, u32),
    /// Diag timestamp stamp (see `crate::timestamps`);
    /// `None` unless the context carries a recorder.
    diag: Option<crate::timestamps::DiagStamp>,
}

impl K21FinishCall {
    /// Record this call's dispatch into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k21 q5_k mmv finish"),
            timestamp_writes: None,
        });
        self.encode_into(&mut pass);
    }

    /// Record this dispatch into an already-open
    /// compute `pass`: the composition form the
    /// device-resident session uses to chain a whole
    /// forward into one pass.
    pub fn encode_into(&self, pass: &mut wgpu::ComputePass<'_>) {
        let slot = self.diag.as_ref().and_then(|d| d.begin(pass));
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.grid.0, self.grid.1, 1);
        if let (Some(d), Some(s)) = (&self.diag, slot) {
            d.end(pass, s);
        }
    }
}

/// The compiled finish-carrying K21 pipeline set: one
/// shader variant per batch count in the band,
/// compiled lazily and cached, exactly K21's
/// arrangement (the batch is a generation literal;
/// the finish is NOT: mode, scale, and cap are
/// uniform words, so one variant serves every finish
/// the head can ask for, as K3's single shader does).
/// Compilation of the band-floor variant happens at
/// construction, so a WGSL defect in the family fails
/// at backend construction like every other kernel's.
#[derive(Debug)]
pub struct K21FinishPipeline {
    shape: K21Shape,
    variants: RefCell<HashMap<u32, wgpu::ComputePipeline>>,
}

impl K21FinishPipeline {
    /// Compile the shipped shape for `device` (K21's
    /// [`k21_shipped_shape`]: the finish changes what
    /// happens at the store, not the tile geometry).
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] on a shape the device
    /// cannot run or a shader wgpu rejects.
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        Self::with_shape(device, k21_shipped_shape(device.features()))
    }

    /// Compile an explicit [`K21Shape`] on `device`.
    /// The shape is validated by K21's own rules
    /// against the device's granted features first: a
    /// shape the device cannot run is a named refusal
    /// here, not a shader validation failure later.
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] on a shape the device
    /// cannot run or a shader wgpu rejects.
    pub fn with_shape(device: &wgpu::Device, shape: K21Shape) -> Result<Self> {
        // K21Shape's own validate is private to its
        // module, so the same three rules are checked
        // here with this kernel's name in the refusal
        // texts: rows in 1..=4, workgroup a multiple
        // of 16 in 16..=256, subgroup only where the
        // device grants SUBGROUP.
        if !(1..=4).contains(&shape.rows) {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "k21 finish shape {}: rows per workgroup {} is outside 1..=4",
                    shape.name(),
                    shape.rows
                ),
            });
        }
        if shape.workgroup == 0 || shape.workgroup % 16 != 0 || shape.workgroup > 256 {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "k21 finish shape {}: workgroup size {} is not a multiple of 16 in 16..=256",
                    shape.name(),
                    shape.workgroup
                ),
            });
        }
        if shape.subgroup && !device.features().contains(wgpu::Features::SUBGROUP) {
            return Err(WgpuError::Pipeline {
                detail: format!(
                    "k21 finish shape {} needs the SUBGROUP device feature, which this device was not granted",
                    shape.name()
                ),
            });
        }
        let this = K21FinishPipeline {
            shape,
            variants: RefCell::new(HashMap::new()),
        };
        // The construction smoke check, as K21's: the
        // band-floor variant compiles now, so the
        // family's WGSL is proven on this device
        // before any plan routes here.
        this.precompile(device, MMV_MIN_BATCH)?;
        dbg_log!("k21 finish pipeline built: shape {}", shape.name());
        Ok(this)
    }

    /// The shape this pipeline runs.
    #[must_use]
    pub fn shape(&self) -> K21Shape {
        self.shape
    }

    /// Compile (or fetch from the cache) the variant
    /// for `batch`: the session's plan builder calls
    /// this ahead of the first prepare at a batch
    /// count so Naga's translation cost stays out of
    /// the token loop, and `prepare_with_finish` calls
    /// it on demand regardless.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when `batch` is
    /// outside the band; [`WgpuError::Pipeline`] when
    /// wgpu rejects the variant's shader.
    pub fn precompile(&self, device: &wgpu::Device, batch: u32) -> Result<()> {
        self.variant(device, batch).map(|_| ())
    }

    /// The cached variant for `batch`, compiling it on
    /// first use.
    fn variant(&self, device: &wgpu::Device, batch: u32) -> Result<wgpu::ComputePipeline> {
        if !(MMV_MIN_BATCH..=MMV_MAX_BATCH).contains(&batch) {
            return Err(WgpuError::InvalidTensor {
                name: "k21 finish call".to_string(),
                detail: format!(
                    "batch {batch} is outside {MMV_MIN_BATCH}..={MMV_MAX_BATCH} \
                     (the register-streaming small-batch band)"
                ),
            });
        }
        if let Some(p) = self.variants.borrow().get(&batch) {
            return Ok(p.clone());
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("k21 q5_k mmv finish shader"),
            source: wgpu::ShaderSource::Wgsl(finish_shader(&self.shape, batch).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k21 q5_k mmv finish pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        dbg_log!(
            "k21 finish variant compiled: shape {} batch {batch}",
            self.shape.name()
        );
        self.variants.borrow_mut().insert(batch, pipeline.clone());
        Ok(pipeline)
    }

    /// Prepare a dispatch of `weights` over a batch of
    /// f32 input rows with the head finishing map
    /// fused into the reduction's final store: `mode`
    /// is [`FINISH_SCALE`] (multiply by `scale`) or
    /// [`FINISH_SOFTCAP`] (scale, then
    /// `cap * tanh(x / cap)`, K8's finish expressions),
    /// the same contract and the same signature as
    /// K3's `prepare_with_finish`, so the head site
    /// can swap product engines without changing what
    /// it asks for. `input` holds `batch *
    /// weights.in_features` f32 and `output` holds
    /// `batch * weights.out_rows` f32, both in the
    /// GEMV's `[batch][out_rows]` layout.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] on a non-finishing
    /// or unknown mode, a softcap mode with a
    /// non-positive cap, a batch outside the band, or
    /// a buffer smaller than the geometry addresses.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_with_finish(
        &self,
        ctx: &GpuContext,
        weights: &K3Weights,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        batch: u32,
        mode: u32,
        scale: f32,
        cap: f32,
    ) -> Result<K21FinishCall> {
        validate_finish(mode, cap)?;
        let pipeline = self.variant(&ctx.device, batch)?;
        let need_in = u64::from(batch) * weights.in_features as u64 * 4;
        let need_out = u64::from(batch) * weights.out_rows as u64 * 4;
        if input.size() < need_in || output.size() < need_out {
            return Err(WgpuError::InvalidTensor {
                name: "k21 finish call".to_string(),
                detail: format!(
                    "buffer sizes (in {}, out {}) are below the geometry's needs \
                     ({need_in}, {need_out})",
                    input.size(),
                    output.size()
                ),
            });
        }
        let total = (weights.out_rows as u64).div_ceil(u64::from(self.shape.rows));
        let grid = grid_dims(total);
        let params = finish_params(
            (weights.in_features / BLOCK_ELEMS) as u32,
            weights.out_rows as u32,
            grid.0,
            mode,
            scale,
            cap,
        );
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k21 finish params"),
            size: param_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&params_buffer, 0, &param_bytes);
        let entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: weights.buffer().as_entire_binding(),
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
        ];
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k21 finish bind group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        Ok(K21FinishCall {
            pipeline,
            bind_group,
            grid,
            diag: ctx.diag_stamp(crate::timestamps::DispatchMeta {
                kernel: "k21_q5k_mmv_finish",
                rows: weights.out_rows as u64,
                cols: weights.in_features as u64,
                quant: "Q5_K",
                weight_bytes: weights.byte_len,
                shape: format!("batch={batch} finish={mode} grid={}x{}", grid.0, grid.1),
            }),
        })
    }

    /// Convenience one-shot: run the finished product
    /// over host f32 rows and read the result back.
    /// The batch is the input's row count. The
    /// check/bench data path; the resident session
    /// prepares against its own buffers instead.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when the input is
    /// not a whole number of rows, its row count is
    /// outside the band, or the finish request is one
    /// [`validate_finish`] refuses;
    /// [`WgpuError::Readback`] when the result cannot
    /// be mapped back.
    #[allow(clippy::too_many_arguments)]
    pub fn run_with_finish(
        &self,
        ctx: &GpuContext,
        weights: &K3Weights,
        input: &[f32],
        mode: u32,
        scale: f32,
        cap: f32,
    ) -> Result<Vec<f32>> {
        if input.is_empty() || input.len() % weights.in_features != 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k21 finish input".to_string(),
                detail: format!(
                    "input holds {} elements, not a whole number of {}-wide rows",
                    input.len(),
                    weights.in_features
                ),
            });
        }
        let batch = (input.len() / weights.in_features) as u32;
        let input_buffer = ctx.upload_f32("k21 finish input", input);
        let out_elems = batch as usize * weights.out_rows;
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k21 finish output"),
            size: (out_elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let call = self.prepare_with_finish(
            ctx,
            weights,
            &input_buffer,
            &output_buffer,
            batch,
            mode,
            scale,
            cap,
        )?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k21 finish run"),
            });
        call.encode(&mut encoder);
        ctx.queue.submit([encoder.finish()]);
        let out = ctx.read_f32(&output_buffer, out_elems)?;
        dbg_log!(
            "k21 finish run: batch={batch} finish={mode} out_rows={} bytes_streamed={}",
            weights.out_rows,
            weights.byte_len
        );
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::k3_q5k_gemv::FINISH_NONE;

    #[test]
    fn finish_modes_are_k3s_constants() {
        // The modes are imported from K3, never
        // redefined; pin the values the head sites and
        // the diag stamps speak, so a renumbering
        // anywhere is a loud failure here.
        assert_eq!(FINISH_NONE, 0);
        assert_eq!(FINISH_SCALE, 1);
        assert_eq!(FINISH_SOFTCAP, 2);
    }

    #[test]
    fn finish_validation_matches_k3s_refusals() {
        // FINISH_NONE is a real mode value (the plain
        // product) but not a finishing mode: asking
        // this pipeline for it must refuse, as K3's
        // prepare_with_finish refuses, or a routing
        // bug could silently drop the head's finish.
        let err = validate_finish(FINISH_NONE, 20.0).unwrap_err();
        assert!(
            err.to_string().contains("is not a finishing mode"),
            "none-mode refusal names the rule: {err}"
        );
        let err = validate_finish(7, 20.0).unwrap_err();
        assert!(
            err.to_string().contains("is not a finishing mode"),
            "unknown-mode refusal names the rule: {err}"
        );
        // Softcap needs a positive cap (K8's finish
        // kernel refuses the same); the scale mode
        // never reads the cap, so any value passes
        // there, as it does for K3.
        for cap in [0.0, -20.0] {
            let err = validate_finish(FINISH_SOFTCAP, cap).unwrap_err();
            assert!(
                err.to_string().contains("is not positive"),
                "cap {cap} refusal names the rule: {err}"
            );
        }
        assert!(validate_finish(FINISH_SOFTCAP, 20.0).is_ok());
        assert!(validate_finish(FINISH_SCALE, 0.0).is_ok());
        assert!(validate_finish(FINISH_SCALE, 20.0).is_ok());
    }

    #[test]
    fn finish_params_tail_order_is_pinned() {
        // The uniform words must land in the shader's
        // Params field order: geometry first, then
        // mode, then the scale and cap as bit
        // patterns. A transposed tail would apply the
        // wrong map with no compile error anywhere;
        // this is the GPU-less machine's only gate on
        // it, so it pins every word.
        let words = finish_params(26, 202_048, 3157, FINISH_SOFTCAP, 0.5, 20.0);
        assert_eq!(
            words,
            [
                26,
                202_048,
                3157,
                FINISH_SOFTCAP,
                0.5f32.to_bits(),
                20.0f32.to_bits()
            ]
        );
        let words = finish_params(2, 100, 3, FINISH_SCALE, 0.19611613, 0.0);
        assert_eq!(words[3], FINISH_SCALE);
        assert_eq!(words[4], 0.19611613f32.to_bits());
        assert_eq!(words[5], 0.0f32.to_bits());
    }

    #[test]
    fn finish_shader_carries_the_finish_and_stages_no_weights() {
        for batch in [2, 7, 16] {
            for shape in [K21Shape::new(2, 128), K21Shape::new(2, 128).with_subgroup()] {
                let shader = finish_shader(&shape, batch);
                // The finish is present, whole: K3's
                // function text, both modes, K8's
                // expression order.
                assert!(shader.contains("fn finish_apply(v: f32) -> f32 {"));
                assert!(shader.contains("return v * params.finish_scale;"));
                assert!(shader.contains("let x = v * params.finish_scale;"));
                assert!(shader.contains("return params.finish_cap * tanh(x / params.finish_cap);"));
                // The params tail the words bind
                // against, in the pinned order.
                assert!(shader.contains(
                    "    finish_mode: u32,\n    finish_scale: f32,\n    finish_cap: f32,"
                ));
                // Both reduction shapes apply the
                // finish at the store, and no bare
                // store of the reduced sum survives
                // anywhere in the generated text.
                if shape.subgroup {
                    assert!(shader.contains("= finish_apply(sum);"));
                } else {
                    assert!(shader.contains("= finish_apply(partials[0]);"));
                }
                assert!(!shader.contains("= sum;"));
                assert!(!shader.contains("= partials[0];"));
                // K21's invariants ride along
                // unchanged: the accumulator array is
                // specialized by the literal slice
                // count, exactly one workgroup array
                // (the reduction partials), and the
                // dequant text is the family's.
                let slices = 2 * batch;
                assert!(shader.contains(&format!("array<f32, {slices}>()")));
                assert!(shader.contains(&format!("acc[{}]", slices - 1)));
                assert_eq!(shader.matches("var<workgroup>").count(), 1);
                assert!(shader.contains("var<workgroup> partials: array<f32,"));
                assert!(!shader.contains("w_qs"));
                assert!(!shader.contains("x_qs"));
                assert!(shader.contains("0x3f3f3f3fu"));
                assert!(shader.contains("* 44u"));
                assert!(shader.contains("let qh_off = 4u + l0 / 4u;"));
                assert!(shader.contains("let w0_off = 12u + 8u * half + l0 / 4u;"));
                // All four dequant lines pinned
                // verbatim. Pinning only qa's line let
                // a truncated 24-bit mask (0x0f0f0fu)
                // through on qc: it zeroed the fourth
                // byte's low nibble in every
                // super-block (bench/
                // k21f-mask-fix-2026-11-07.md). The
                // negative pin names the typo class
                // itself: the six-digit repeating-byte
                // mask must never appear in this text.
                assert!(
                    shader.contains(
                        "let qa = vec4<f32>(unpack4xU8(w0 & 0x0f0f0f0fu)) + 16.0 * bits_a;",
                    )
                );
                assert!(shader.contains(
                    "let qb = vec4<f32>(unpack4xU8((w0 >> 4u) & 0x0f0f0f0fu)) + 16.0 * bits_b;",
                ));
                assert!(
                    shader.contains(
                        "let qc = vec4<f32>(unpack4xU8(w1 & 0x0f0f0f0fu)) + 16.0 * bits_c;",
                    )
                );
                assert!(shader.contains(
                    "let qd = vec4<f32>(unpack4xU8((w1 >> 4u) & 0x0f0f0f0fu)) + 16.0 * bits_d;",
                ));
                assert!(!shader.contains("0x0f0f0fu"));
                // The half conversion is the shared
                // text (bound from K21's constant, not
                // a copy): its signature line is
                // present exactly once.
                assert_eq!(shader.matches("fn f16_to_f32(h: u32) -> f32 {").count(), 1);
            }
        }
        // The subgroup variant shuffles inside the
        // sixteen; the tree variant never names a
        // subgroup builtin (K21's gate, kept).
        let sg = finish_shader(&K21Shape::new(2, 128).with_subgroup(), 4);
        assert!(sg.contains("subgroupShuffleXor"));
        let tree = finish_shader(&K21Shape::new(2, 128), 4);
        assert!(!tree.contains("subgroupShuffleXor"));
        assert!(tree.contains("partials[lane] = acc[0];"));
    }

    #[test]
    fn every_finish_variant_validates_under_naga() {
        // The strongest shader gate a machine with no
        // GPU adapter at all can run: every generated
        // variant (the full band, both reductions)
        // must parse and validate under Naga, the same
        // front end wgpu feeds the driver with. This
        // proves the WGSL is well-formed and
        // well-typed; it does not prove a driver's
        // compiler accepts it or keeps the
        // accumulators in registers, which is the
        // server protocol's step 0 and step 1
        // (bench/k21-finish-2026-11-06.md).
        for batch in MMV_MIN_BATCH..=MMV_MAX_BATCH {
            for shape in [K21Shape::new(2, 128), K21Shape::new(2, 128).with_subgroup()] {
                let src = finish_shader(&shape, batch);
                let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| {
                    panic!(
                        "shape {} batch {batch}: WGSL parse failed: {}",
                        shape.name(),
                        e.emit_to_string(&src)
                    )
                });
                naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap_or_else(|e| {
                    panic!(
                        "shape {} batch {batch}: Naga validation failed: {e:?}",
                        shape.name()
                    )
                });
            }
        }
    }
}
