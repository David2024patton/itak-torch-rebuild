//! K23: Q4_K matrix-matrix MMQ through the Vulkan
//! cooperative matrix extension, the integer tensor-unit
//! product llama.cpp's Vulkan backend runs for prefill
//! on this card family (`mul_mmq_cm1.comp`; the parity
//! spec is bench/llama-parity-specs-2026-11-06.md, the
//! build record bench/coopmat-build-2026-11-06.md).
//!
//! K13 runs the same MMQ math (`vec_dot_q4_K_q8_1`, term
//! for term) as scalar DP4A dots out of LDS, and the
//! prefill autopsy convicted that whole MMQ class of an
//! occupancy wall: the compiled forms sit at the VGPR
//! 256 cap and stream the staged weights at ~45 GB/s
//! while the GEMV in the same run streams at ~650 GB/s.
//! K23 keeps K13's exact staging split (raw nibble codes
//! in shared rows, scale pairs beside them, K12's Q8_1
//! words and scaled sums as the activation side) but
//! issues the 16x16x16 products to the matrix units:
//! per 32-element sub-block, an s8 x s8 to s32
//! cooperative matrix product over the staged codes,
//! folded into f32 with the sub-block's scale pair,
//! which is llama's cm1 accumulate identity.
//!
//! The shader is GLSL, not WGSL: wgpu 27 has no
//! cooperative matrix support at any layer, so the
//! module is compiled ahead of time with
//! glslangValidator and enters through the passthrough
//! path on a device the coopmat bridge opened
//! ([`crate::coopmat_bridge`]). The source, the
//! artifact, its disassembly, and the rebuild script
//! live in `kernels/coopmat/`. Two consequences are
//! disclosed, not hidden: Naga can never validate this
//! module (correctness rests on the on-card numeric
//! gate in the lane protocol), and the pipeline can
//! only be built on a bridged device, so unlike the
//! other optional kernels it is not constructed on
//! stock devices at all, not even to fail early.
//!
//! Routing is config (see `WGPU_COMPUTE_COOPMAT` in
//! `crate::compute`): the shipped default keeps K23 out
//! of the Q4_K ladder and K13 serving, on the K22
//! precedent. The shape contract is narrower than
//! K13's: the batch stays capped at
//! [`MMQ_MAX_BATCH`], and the input width must be a
//! whole number of super-blocks per staged chunk
//! (in_features % 256 == 0 follows from the crate's
//! tensors, but the prepare states the chunk condition
//! it actually needs: the 32-block count must divide
//! by the chunk's BK_STEP of 4).

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
/// geometry is compiled for (256 threads as four
/// subgroups of 64, tiles assigned per subgroup). The
/// bridge probes the device's size; a device reporting
/// any other value gets no pipeline, by name.
const REQUIRED_SUBGROUP_SIZE: u32 = 64;
/// Workgroup storage the shader stages, in bytes: the
/// two code tiles (128 and 64 rows at a 36-word pitch),
/// the two scale arrays, and the per-subgroup int32
/// fold tile. Mirrors the shared declarations in the
/// GLSL source; the device must grant at least this.
const STAGING_BYTES: u32 = 37_888;

/// The checked-in SPIR-V artifact, compiled from
/// `coopmat/k23_q4k_coopmat.comp` by
/// `coopmat/rebuild.sh`.
const SPIRV: &[u8] = include_bytes!("coopmat/k23_q4k_coopmat.spv");

/// The artifact as SPIR-V words (the file is a whole
/// number of words by construction of the compiler;
/// the unit test walks its instruction stream).
fn spirv_words() -> Vec<u32> {
    SPIRV
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The shape refusals of [`K23Pipeline::prepare`], as a
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

/// One prepared K23 dispatch: a bind group plus grid
/// dimensions, built once for a (weights, quantized
/// input, output, batch) combination and encodable any
/// number of times.
#[derive(Debug)]
pub struct K23Call {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    grid: (u32, u32),
    /// Diag timestamp stamp (see `crate::timestamps`);
    /// `None` unless the context carries a recorder.
    diag: Option<crate::timestamps::DiagStamp>,
}

impl K23Call {
    /// Record this call's dispatch into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k23 q4_k coopmat mmq"),
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

/// The compiled K23 pipeline. Compilation happens at
/// backend construction, and only on a bridged device:
/// the constructor takes the bridge's probe record as
/// its gate, so no stock device ever sees the
/// passthrough module.
#[derive(Debug)]
pub struct K23Pipeline {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

impl K23Pipeline {
    /// Build the pipeline on a bridged device, or `None`
    /// (with the refusal on the diag channel) when the
    /// device does not meet the shader's compiled
    /// assumptions: a subgroup size other than
    /// [`REQUIRED_SUBGROUP_SIZE`], or less workgroup
    /// storage than the staging set needs.
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] when a device that passes
    /// the gates rejects the pipeline.
    pub fn for_device(device: &wgpu::Device, coopmat: &CoopmatInfo) -> Result<Option<Self>> {
        if coopmat.subgroup_size != REQUIRED_SUBGROUP_SIZE {
            dbg_log!(
                "k23 coopmat unavailable: device subgroup size {} is not the {} \
                 the shader geometry is compiled for",
                coopmat.subgroup_size,
                REQUIRED_SUBGROUP_SIZE
            );
            return Ok(None);
        }
        let granted = device.limits().max_compute_workgroup_storage_size;
        if granted < STAGING_BYTES {
            dbg_log!(
                "k23 coopmat unavailable: device grants {granted} bytes of workgroup \
                 storage, below the {STAGING_BYTES} the shader stages"
            );
            return Ok(None);
        }
        let words = spirv_words();
        let module = crate::coopmat_bridge::passthrough_shader_module(
            device,
            "k23 q4_k coopmat shader",
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
            label: Some("k23 bind group layout"),
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
            label: Some("k23 pipeline layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k23 q4_k coopmat mmq pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        dbg_log!(
            "k23 coopmat pipeline built: {} property rows, subgroup {}, tile {TILE_OUT}x{TILE_BATCH}",
            coopmat.property_rows,
            coopmat.subgroup_size
        );
        Ok(Some(K23Pipeline { pipeline, layout }))
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
    ) -> Result<K23Call> {
        if let Err(detail) = check_shape(weights.in_features, batch) {
            return Err(WgpuError::InvalidTensor {
                name: "k23 call".to_string(),
                detail,
            });
        }
        let grid = grid_for(weights.out_rows, batch);
        if grid.0 > 65_535 || grid.1 > 65_535 {
            return Err(WgpuError::InvalidTensor {
                name: "k23 call".to_string(),
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
                name: "k23 call".to_string(),
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
            label: Some("k23 params"),
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
            label: Some("k23 bind group"),
            layout: &self.layout,
            entries: &entries,
        });
        Ok(K23Call {
            pipeline: self.pipeline.clone(),
            bind_group,
            grid,
            diag: ctx.diag_stamp(crate::timestamps::DispatchMeta {
                kernel: "k23_q4k_coopmat",
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
                name: "k23 input".to_string(),
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
            label: Some("k23 qs"),
            size: qs_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&qs_buffer, 0, &qs_bytes);
        let ds_bytes: Vec<u8> = ds.iter().flat_map(|v| v.to_le_bytes()).collect();
        let ds_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k23 ds"),
            size: ds_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&ds_buffer, 0, &ds_bytes);
        let out_elems = batch as usize * weights.out_rows;
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k23 output"),
            size: (out_elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let call = self.prepare(ctx, weights, &qs_buffer, &ds_buffer, &output_buffer, batch)?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k23 run"),
            });
        call.encode(&mut encoder);
        ctx.queue.submit([encoder.finish()]);
        let out = ctx.read_f32(&output_buffer, out_elems)?;
        dbg_log!(
            "k23 run: batch={batch} out_rows={} bytes_streamed={}",
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
    fn spirv_artifact_carries_coopmat_ops() {
        // Walk the artifact's instruction stream (word 0
        // of each instruction is word_count << 16 |
        // opcode) and require the cooperative matrix
        // instructions the shader exists for. This is a
        // smoke check on the checked-in artifact; the
        // authoritative inspection is the disassembly
        // committed beside it, produced by rebuild.sh.
        const OP_TYPE_COOPERATIVE_MATRIX: u32 = 4456;
        const OP_COOPERATIVE_MATRIX_LOAD: u32 = 4457;
        const OP_COOPERATIVE_MATRIX_STORE: u32 = 4458;
        const OP_COOPERATIVE_MATRIX_MUL_ADD: u32 = 4459;
        let words = spirv_words();
        assert!(words.len() > 5, "artifact is implausibly small");
        assert_eq!(words[0], 0x0723_0203, "not a SPIR-V module");
        let mut found = [false; 4];
        let mut i = 5; // past the 5-word header
        while i < words.len() {
            let head = words[i];
            let count = (head >> 16) as usize;
            assert!(count > 0, "zero-length instruction at word {i}");
            match head & 0xFFFF {
                OP_TYPE_COOPERATIVE_MATRIX => found[0] = true,
                OP_COOPERATIVE_MATRIX_LOAD => found[1] = true,
                OP_COOPERATIVE_MATRIX_STORE => found[2] = true,
                OP_COOPERATIVE_MATRIX_MUL_ADD => found[3] = true,
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
    }
}
