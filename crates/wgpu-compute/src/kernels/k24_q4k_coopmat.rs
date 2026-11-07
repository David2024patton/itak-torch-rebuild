//! K24: Q4_K matrix-matrix MMQ through the Vulkan
//! cooperative matrix extension, v2: the
//! register-resident form of K23 (the v1 coopmat
//! kernel, [`crate::kernels::k23_q4k_coopmat`]).
//!
//! K23 proved the concept on the card (correct
//! everywhere, full-model greedy identity exact) and
//! lost on craft: 4.0x slower than K13 at the kernel
//! line. The autopsy in bench/coopmat-v2-2026-11-07.md
//! convicts two structural causes, and this kernel is
//! the fix for exactly those two, with the staging,
//! tile geometry, buffer contract, and accumulate
//! identity of K23 unchanged:
//!
//! - Fragments are preloaded once per 32-element
//!   sub-block into register arrays (llama.cpp's
//!   `mul_mmq_cm1` Q4_K branch structure, its cache_a
//!   / cache_b), where K23 reloaded the A fragment per
//!   column tile and the B fragment per row tile: 12
//!   cooperative loads per sub-block instead of 32.
//! - The accumulator fold is in-register: the int32
//!   accumulator's elements are read directly
//!   (`acc[e]`) and folded into the per-thread f32
//!   sums with llama's cm1_accumulate identity in its
//!   plain shape, where K23 stored every 16x16 tile
//!   into a shared fold tile and folded it back with
//!   scalar reads (32 coopMatStores and 64 subgroup
//!   barriers per staged chunk, half of each subgroup
//!   idle in every fold). The fold tile is gone, so
//!   the staging set is 4,096 bytes smaller than
//!   K23's (see [`STAGING_BYTES`]).
//!
//! The element mapping the register fold stands on is
//! the load-bearing assumption, stated plainly: at
//! subgroup size 64 on GFX11 (RDNA3), Mesa's
//! cooperative matrix lowering
//! (`radv_nir_lower_cooperative_matrix` in
//! src/amd/vulkan/nir/radv_nir_lower_cooperative_matrix.c)
//! places accumulator element `e` of invocation `inv`
//! at row `(inv / 16) + 4 * e`, column `inv % 16` of
//! the 16x16 tile: its load, store, and
//! element-coordinate lowerings all compute
//! `base_row = inv / 16` (radv_get_base_row),
//! `row_iter = e * wave_size / 16`, `inner_idx =
//! inv & 15`, and the accumulator length
//! `16 * 16 / 64 = 4`. This is NOT llama's mapping:
//! llama runs its cm1 on AMD at wave32
//! (ggml-vulkan.cpp's `cm1_use_wave32` forces a
//! 32-wide subgroup through pipeline creation, a
//! control the wgpu passthrough path does not expose),
//! where the same Mesa formulas give row
//! `(inv / 16) + 2 * e` and 8 elements per thread.
//! GFX11.5/12 take a different branch of the Mesa
//! lowering and are out of scope, as they are for
//! K23; the shader header states the same. The
//! on-card numeric gate (tests/k24_coopmat.rs) is
//! the proof on the target card, and the routing
//! stays behind the coopmat flag with the v2
//! selector on top (see `WGPU_COMPUTE_COOPMAT_V2` in
//! `crate::compute`). Both switches shipped default
//! off until the card protocol in the bench note
//! discharged its decision rule (wave20: the mapping
//! proven numerically, the kernel line about 2.3x
//! faster than K13's, identity intact); the flip
//! (bench/coopmat-default-flip-2026-11-07.md) made
//! K24 the shipped default inside the rung, with K23
//! remaining the A/B reference and fallback behind
//! the selector's `0`.
//!
//! The shader is GLSL, not WGSL, for the same reason
//! as K23's: wgpu 27 has no cooperative matrix
//! support at any layer, so the module is compiled
//! ahead of time with glslangValidator and enters
//! through the passthrough path on a device the
//! coopmat bridge opened ([`crate::coopmat_bridge`]).
//! The source, the artifact, its disassembly, and
//! the rebuild script live in `kernels/coopmat/`
//! (rebuild-k24.sh; the script's op check asserts
//! the load and muladd ops are present and the store
//! op is ABSENT, the binary-level form of this
//! kernel's structural claim). Naga can never
//! validate this module; correctness rests on the
//! on-card numeric gate in the lane protocol.
//!
//! The shape contract is K23's, unchanged: the batch
//! stays capped at [`MMQ_MAX_BATCH`], and the input
//! width must be a whole number of super-blocks per
//! staged chunk (the 32-block count must divide by
//! the chunk's BK_STEP of 4).

use crate::coopmat_bridge::CoopmatInfo;
use crate::device::GpuContext;
use crate::error::{Result, WgpuError};
use crate::kernels::k12_q8_quant::{ds_vec4s, qs_words};
use crate::kernels::k13_q4k_mmq::MMQ_MAX_BATCH;
use crate::kernels::k1_q4k_gemv::K1Weights;

use crate::diag::dbg_log;

/// Elements per Q4_K super-block.
const BLOCK_ELEMS: usize = 256;
/// Output rows one workgroup computes (the shader's
/// W_ROWS).
const TILE_OUT: u32 = 128;
/// Batch rows one workgroup computes (the shader's
/// X_ROWS).
const TILE_BATCH: u32 = 64;
/// The subgroup size the checked-in shader's workgroup
/// geometry and accumulator element mapping are
/// compiled for (256 threads as four subgroups of 64;
/// the mapping's row stride is wave_size / 16 = 4).
/// The bridge probes the device's size; a device
/// reporting any other value gets no pipeline, by
/// name.
const REQUIRED_SUBGROUP_SIZE: u32 = 64;
/// Workgroup storage the shader stages, in bytes: the
/// two code tiles (128 and 64 rows at a 36-word
/// pitch) and the two scale arrays. K23's staging set
/// minus its 4,096-byte fold tile, which the
/// in-register fold deletes. Mirrors the shared
/// declarations in the GLSL source; the device must
/// grant at least this.
const STAGING_BYTES: u32 = 33_792;

/// The checked-in SPIR-V artifact, compiled from
/// `coopmat/k24_q4k_coopmat.comp` by
/// `coopmat/rebuild-k24.sh`.
const SPIRV: &[u8] = include_bytes!("coopmat/k24_q4k_coopmat.spv");

/// The artifact as SPIR-V words (the file is a whole
/// number of words by construction of the compiler;
/// the unit test walks its instruction stream).
fn spirv_words() -> Vec<u32> {
    SPIRV
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The shape refusals of [`K24Pipeline::prepare`], as a
/// pure check the unit tests can pin without a device:
/// the batch band, and the chunk condition on the input
/// width (the staged chunk is 4 32-blocks, so the
/// 32-block count must divide by 4; with the crate's
/// super-block-multiple widths that is automatic, and
/// the check names it rather than assuming it).
fn check_shape(in_features: usize, batch: u32) -> std::result::Result<(), String> {
    if batch == 0 || batch > MMQ_MAX_BATCH {
        return Err(format!(
            "batch {batch} is outside 1..={MMQ_MAX_BATCH} (the batch tile's row count)"
        ));
    }
    if in_features == 0 || in_features % BLOCK_ELEMS != 0 {
        return Err(format!(
            "input width {in_features} is not a whole number of {BLOCK_ELEMS}-element super-blocks"
        ));
    }
    if (in_features / 32) % 4 != 0 {
        return Err(format!(
            "input width {in_features} gives {} 32-blocks, not a whole number of 4-block staged chunks",
            in_features / 32
        ));
    }
    Ok(())
}

/// The dispatch grid for a shape: workgroups of
/// [`TILE_OUT`] output rows by [`TILE_BATCH`] batch
/// rows. Pure, so the unit tests pin it.
fn grid_for(out_rows: usize, batch: u32) -> (u32, u32) {
    (
        (out_rows as u32).div_ceil(TILE_OUT),
        batch.div_ceil(TILE_BATCH),
    )
}

/// One prepared K24 dispatch: a bind group plus grid
/// dimensions, built once for a (weights, quantized
/// input, output, batch) combination and encodable any
/// number of times.
#[derive(Debug)]
pub struct K24Call {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    grid: (u32, u32),
    /// Diag timestamp stamp (see `crate::timestamps`);
    /// `None` unless the context carries a recorder.
    diag: Option<crate::timestamps::DiagStamp>,
}

impl K24Call {
    /// Record this call's dispatch into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k24 q4_k coopmat mmq v2"),
            timestamp_writes: None,
        });
        self.encode_into(&mut pass);
    }

    /// Record this dispatch into an already-open compute
    /// `pass`: the composition form the device-resident
    /// session uses to chain a whole forward into one
    /// pass.
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

/// The compiled K24 pipeline. Compilation happens at
/// backend construction, and only on a bridged device:
/// the constructor takes the bridge's probe record as
/// its gate, so no stock device ever sees the
/// passthrough module.
#[derive(Debug)]
pub struct K24Pipeline {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

impl K24Pipeline {
    /// Build the pipeline on a bridged device, or `None`
    /// (with the refusal on the diag channel) when the
    /// device does not meet the shader's compiled
    /// assumptions: a subgroup size other than
    /// [`REQUIRED_SUBGROUP_SIZE`] (the accumulator
    /// element mapping is compiled for 64), or less
    /// workgroup storage than the staging set needs.
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] when a device that passes
    /// the gates rejects the pipeline.
    pub fn for_device(device: &wgpu::Device, coopmat: &CoopmatInfo) -> Result<Option<Self>> {
        if coopmat.subgroup_size != REQUIRED_SUBGROUP_SIZE {
            dbg_log!(
                "k24 coopmat v2 unavailable: device subgroup size {} is not the {} \
                 the shader geometry and element mapping are compiled for",
                coopmat.subgroup_size,
                REQUIRED_SUBGROUP_SIZE
            );
            return Ok(None);
        }
        let granted = device.limits().max_compute_workgroup_storage_size;
        if granted < STAGING_BYTES {
            dbg_log!(
                "k24 coopmat v2 unavailable: device grants {granted} bytes of workgroup \
                 storage, below the {STAGING_BYTES} the shader stages"
            );
            return Ok(None);
        }
        let words = spirv_words();
        let module = crate::coopmat_bridge::passthrough_shader_module(
            device,
            "k24 q4_k coopmat v2 shader",
            &words,
        );
        // The bind group layout is declared explicitly:
        // a passthrough module's bindings are the GLSL
        // source's contract (weights, qs, ds, output,
        // params, in that order), stated here rather than
        // reflected.
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("k24 bind group layout"),
            entries: &[
                storage(0, true),
                storage(1, true),
                storage(2, true),
                storage(3, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("k24 pipeline layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k24 q4_k coopmat mmq v2 pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        dbg_log!(
            "k24 coopmat v2 pipeline built: {} property rows, subgroup {}, tile {TILE_OUT}x{TILE_BATCH}",
            coopmat.property_rows,
            coopmat.subgroup_size
        );
        Ok(Some(K24Pipeline { pipeline, layout }))
    }

    /// Prepare a coopmat MMQ dispatch of `weights` over
    /// a batch of K12-quantized input rows: `qs` / `ds`
    /// hold the quantized form of `batch` rows of
    /// `weights.in_features` f32 (K12's layout), and
    /// `output` holds `batch * weights.out_rows` f32, in
    /// the same `[batch][width]` layout the GEMV
    /// prepares against, so a plan can swap one form for
    /// the other without moving a byte.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when the shape is
    /// outside the kernel's contract (see
    /// [`check_shape`]), the output needs more
    /// workgroups than one dispatch dimension holds, or
    /// a buffer is smaller than the geometry addresses.
    pub fn prepare(
        &self,
        ctx: &GpuContext,
        weights: &K1Weights,
        qs: &wgpu::Buffer,
        ds: &wgpu::Buffer,
        output: &wgpu::Buffer,
        batch: u32,
    ) -> Result<K24Call> {
        if let Err(detail) = check_shape(weights.in_features, batch) {
            return Err(WgpuError::InvalidTensor {
                name: "k24 call".to_string(),
                detail,
            });
        }
        let grid = grid_for(weights.out_rows, batch);
        if grid.0 > 65_535 || grid.1 > 65_535 {
            return Err(WgpuError::InvalidTensor {
                name: "k24 call".to_string(),
                detail: format!(
                    "{} output rows in batches of {batch} need a {grid:?} grid, \
                     past one dispatch dimension",
                    weights.out_rows
                ),
            });
        }
        let need_qs = (qs_words(batch as usize, weights.in_features) * 4) as u64;
        let need_ds = (ds_vec4s(batch as usize, weights.in_features) * 16) as u64;
        let need_out = batch as u64 * weights.out_rows as u64 * 4;
        if qs.size() < need_qs || ds.size() < need_ds || output.size() < need_out {
            return Err(WgpuError::InvalidTensor {
                name: "k24 call".to_string(),
                detail: format!(
                    "buffer sizes (qs {}, ds {}, out {}) are below the geometry's needs \
                     ({need_qs}, {need_ds}, {need_out})",
                    qs.size(),
                    ds.size(),
                    output.size()
                ),
            });
        }
        let params: [u32; 4] = [
            (weights.in_features / BLOCK_ELEMS) as u32,
            weights.out_rows as u32,
            batch,
            (weights.in_features / 32) as u32,
        ];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k24 params"),
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
                resource: qs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: ds.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buffer.as_entire_binding(),
            },
        ];
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k24 bind group"),
            layout: &self.layout,
            entries: &entries,
        });
        Ok(K24Call {
            pipeline: self.pipeline.clone(),
            bind_group,
            grid,
            diag: ctx.diag_stamp(crate::timestamps::DispatchMeta {
                kernel: "k24_q4k_coopmat",
                rows: weights.out_rows as u64,
                cols: weights.in_features as u64,
                quant: "Q4_K",
                weight_bytes: weights.byte_len,
                shape: format!("batch={batch} grid={}x{}", grid.0, grid.1),
            }),
        })
    }

    /// Convenience one-shot: run the coopmat MMQ over
    /// host-side quantized rows (`qs` packed words and
    /// flat `ds` quads, K12's layout for `batch` rows of
    /// the tensor's input width) and read the
    /// [`batch`, out] result back. The check/bench data
    /// path; the resident session prepares against its
    /// own buffers instead.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when the quantized
    /// inputs are not sized for the batch and width, or
    /// the batch is out of range; [`WgpuError::Readback`]
    /// when the result cannot be mapped back.
    pub fn run(
        &self,
        ctx: &GpuContext,
        weights: &K1Weights,
        qs: &[u32],
        ds: &[f32],
        batch: u32,
    ) -> Result<Vec<f32>> {
        if qs.len() != qs_words(batch as usize, weights.in_features)
            || ds.len() != ds_vec4s(batch as usize, weights.in_features) * 4
        {
            return Err(WgpuError::InvalidTensor {
                name: "k24 input".to_string(),
                detail: format!(
                    "quantized inputs hold {} words and {} scale values, not the \
                     {} words and {} values {batch} rows of width {} need",
                    qs.len(),
                    ds.len(),
                    qs_words(batch as usize, weights.in_features),
                    ds_vec4s(batch as usize, weights.in_features) * 4,
                    weights.in_features
                ),
            });
        }
        let qs_bytes: Vec<u8> = qs.iter().flat_map(|v| v.to_le_bytes()).collect();
        let qs_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k24 qs"),
            size: qs_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&qs_buffer, 0, &qs_bytes);
        let ds_bytes: Vec<u8> = ds.iter().flat_map(|v| v.to_le_bytes()).collect();
        let ds_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k24 ds"),
            size: ds_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&ds_buffer, 0, &ds_bytes);
        let out_elems = batch as usize * weights.out_rows;
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k24 output"),
            size: (out_elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let call = self.prepare(ctx, weights, &qs_buffer, &ds_buffer, &output_buffer, batch)?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k24 run"),
            });
        call.encode(&mut encoder);
        ctx.queue.submit([encoder.finish()]);
        let out = ctx.read_f32(&output_buffer, out_elems)?;
        dbg_log!(
            "k24 run: batch={batch} out_rows={} bytes_streamed={}",
            weights.out_rows,
            weights.byte_len
        );
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The GLSL source the artifact was compiled from,
    /// read back for the shader-text pins below: the
    /// structural choices this kernel exists for are
    /// pinned as text so a later edit cannot silently
    /// revert the craft (the K21F pin pattern).
    const SHADER_SRC: &str = include_str!("coopmat/k24_q4k_coopmat.comp");

    #[test]
    fn shape_contract() {
        // The Glimmer shapes: hidden and FFN widths and
        // the KV width all satisfy the contract.
        for width in [6656usize, 19968, 4096, 256] {
            assert!(check_shape(width, 64).is_ok(), "width {width}");
            assert!(check_shape(width, 17).is_ok(), "width {width}");
        }
        assert!(check_shape(6656, 0).is_err());
        assert!(check_shape(6656, 65).is_err());
        assert!(check_shape(6688, 64).is_err()); // not a super-block multiple
        assert!(check_shape(0, 64).is_err());
    }

    #[test]
    fn grid_math() {
        assert_eq!(grid_for(6656, 64), (52, 1));
        assert_eq!(grid_for(19968, 64), (156, 1));
        assert_eq!(grid_for(256, 62), (2, 1));
        assert_eq!(grid_for(128, 64), (1, 1));
    }

    #[test]
    fn staging_bytes_match_the_shader_arrays() {
        // The constructor's staging constant is the sum
        // of the shader's shared arrays, with NO fold
        // tile: two code tiles at the 36-word pitch
        // (4 B/word), the vec2 scale pairs (8 B each),
        // and the two split activation scale arrays
        // (4 B each). If the shader grows an array, this
        // arithmetic is where the constant is re-derived.
        let w_qs = 128 * 36 * 4;
        let w_dm = 128 * 4 * 8;
        let x_qs = 64 * 36 * 4;
        let x_scales = 64 * 4 * 4 * 2;
        assert_eq!(STAGING_BYTES as usize, w_qs + w_dm + x_qs + x_scales);
        assert_eq!(STAGING_BYTES, 33_792);
    }

    #[test]
    fn spirv_artifact_carries_coopmat_ops_and_no_store() {
        // Walk the artifact's instruction stream (word 0
        // of each instruction is word_count << 16 |
        // opcode) and require the cooperative matrix
        // instructions the shader exists for, and the
        // ABSENCE of the store: v2's fold is in-register,
        // so an OpCooperativeMatrixStoreKHR in the
        // artifact means the fold regressed to K23's
        // LDS round-trip. This is a smoke check on the
        // checked-in artifact; the authoritative
        // inspection is the disassembly committed beside
        // it, produced by rebuild-k24.sh (which asserts
        // the same inventory).
        const OP_TYPE_COOPERATIVE_MATRIX: u32 = 4456;
        const OP_COOPERATIVE_MATRIX_LOAD: u32 = 4457;
        const OP_COOPERATIVE_MATRIX_STORE: u32 = 4458;
        const OP_COOPERATIVE_MATRIX_MUL_ADD: u32 = 4459;
        let words = spirv_words();
        assert!(words.len() > 5, "artifact is implausibly small");
        assert_eq!(words[0], 0x0723_0203, "not a SPIR-V module");
        let mut found = [false; 3];
        let mut stores = 0usize;
        let mut i = 5; // past the 5-word header
        while i < words.len() {
            let head = words[i];
            let count = (head >> 16) as usize;
            assert!(count > 0, "zero-length instruction at word {i}");
            match head & 0xFFFF {
                OP_TYPE_COOPERATIVE_MATRIX => found[0] = true,
                OP_COOPERATIVE_MATRIX_LOAD => found[1] = true,
                OP_COOPERATIVE_MATRIX_STORE => stores += 1,
                OP_COOPERATIVE_MATRIX_MUL_ADD => found[2] = true,
                _ => {}
            }
            i += count;
        }
        assert_eq!(
            i,
            words.len(),
            "instruction walk must end on the stream end"
        );
        assert!(
            found.iter().all(|f| *f),
            "coopmat ops missing from the artifact: {found:?}"
        );
        assert_eq!(
            stores, 0,
            "the v2 artifact must contain no cooperative matrix store"
        );
    }

    #[test]
    fn shader_text_pins_the_v2_structure() {
        // The structural choices this kernel exists for,
        // pinned as source text (the K21F pin pattern):
        // a later edit that reverts the craft fails here
        // by name instead of silently re-shipping K23's
        // product loop under the K24 stamp.
        let src = SHADER_SRC;

        // The fragment preload arrays, at llama's sizes:
        // cms_per_row * K_SUB A fragments and
        // cms_per_col * K_SUB B fragments per sub-block.
        assert!(
            src.contains("cache_a[CMS_PER_ROW * K_SUB]"),
            "the A fragment preload array is pinned"
        );
        assert!(
            src.contains("cache_b[CMS_PER_COL * K_SUB]"),
            "the B fragment preload array is pinned"
        );

        // The tile and staging geometry, unchanged from
        // K23: the 36-word pitch with its bank skew, the
        // 128 x 64 workgroup tile, the 4-sub-block
        // staged chunk, and the shared arrays the Rust
        // staging constant is derived from.
        assert!(src.contains("const uint QPITCH = BK_STEP * (BK / 4) + 4;"));
        assert!(src.contains("const uint W_ROWS = 128;"));
        assert!(src.contains("const uint X_ROWS = 64;"));
        assert!(src.contains("const uint BK_STEP = 4;"));
        assert!(src.contains("shared uint buf_w_qs[W_ROWS * QPITCH];"));
        assert!(src.contains("shared vec2 buf_w_dm[W_ROWS * BK_STEP];"));
        assert!(src.contains("shared uint buf_x_qs[X_ROWS * QPITCH];"));
        assert!(src.contains("shared float buf_x_d[X_ROWS * BK_STEP];"));
        assert!(src.contains("shared float buf_x_s[X_ROWS * BK_STEP];"));

        // The accumulator element mapping at subgroup
        // size 64 on GFX11 (Mesa's lowering): 4 elements
        // per thread, element e at row (inv / TN) + 4e,
        // column inv % TN.
        assert!(
            src.contains("const uint CM_ELEMS = (TM * TN) / 64u;"),
            "the wave64 element count is pinned"
        );
        assert!(
            src.contains("return (gl_SubgroupInvocationID / TN) + 4u * e;"),
            "the wave64 element row formula is pinned"
        );
        assert!(
            src.contains("const uint elem_col0 = inv % TN;"),
            "the element column formula is pinned"
        );

        // The sums layout: [tile (r, c)][element e],
        // llama's indexing.
        assert!(src.contains("float sums[CMS_PER_ROW * CMS_PER_COL * CM_ELEMS];"));
        assert!(src.contains("sums[tile_idx * CM_ELEMS + e]"));

        // The fold is in-register: the source names no
        // fold tile, issues no cooperative store, and
        // synchronizes no subgroup anywhere.
        assert!(!src.contains("buf_tile"), "no shared fold tile may return");
        assert!(
            !src.contains("coopMatStore("),
            "no cooperative matrix store call may return"
        );
        assert!(
            !src.contains("subgroupBarrier("),
            "no subgroup barrier call may return"
        );
    }
}
