//! Split-K reduction: the second half of the MMQ split-K
//! products (the split forms of K13 and K14; see
//! `bench/splitk-mmq-2026-11-07.md` for the design record).
//!
//! A split product partitions the reduction (K) dimension
//! across `S` workgroups per output tile: slice `s` walks a
//! contiguous range of super-block columns and writes its
//! f32 partial sums to slice `s` of a scratch buffer laid
//! out `[splits][batch][out_rows]`. This kernel folds the
//! slices back into the output, one thread per output
//! element, summing in ascending slice order:
//! `out[i] = p[0][i] + p[1][i] + ... + p[S-1][i]`, each add
//! an ordinary f32 add in that fixed order.
//!
//! Drift contract, stated plainly: the integer dots inside
//! each slice are exact and each slice folds its
//! per-sub-block scale terms in the same order the unsplit
//! kernel folds them, but the unsplit kernel accumulates
//! one chain over the whole K range while the split form
//! accumulates one chain per slice and then adds the slice
//! sums. f32 addition is not associative, so the split
//! result is NOT bit-identical to the unsplit product in
//! general; it differs by float reassociation noise, the
//! same class the MMQ correctness gates already bound
//! against the host reference running the same quantized
//! math. The reduction order here is fixed by construction
//! (ascending slices, one thread per element, no atomics),
//! so a split run is deterministic run to run.
//!
//! The kernel is family-agnostic (it folds f32 partials);
//! the caller passes its quantization family name purely so
//! the diag stamp attributes the dispatch.

use crate::device::GpuContext;
use crate::error::{Result, WgpuError};

use crate::diag::dbg_log;

/// The largest split factor a split product accepts: the
/// resolver's ceiling (`WGPU_COMPUTE_MMQ_SPLITK` admits
/// 1..=8) and the prepare-time refusal bound share it, so
/// the two can never disagree.
pub const SPLITK_MAX: u32 = 8;

/// Threads per reduce workgroup.
const REDUCE_WORKGROUP: u32 = 256;

/// The reduction shader. One thread per output element;
/// the slice loop is a runtime loop over at most
/// [`SPLITK_MAX`] terms, deliberately not unrolled: the
/// kernel is pure bandwidth over `splits * total` f32
/// reads and the loop bound is uniform across the grid.
const REDUCE_WGSL: &str = r#"struct Params {
    total: u32,
    splits: u32,
}

@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total) {
        return;
    }
    var acc = partials[idx];
    for (var s = 1u; s < params.splits; s++) {
        acc += partials[s * params.total + idx];
    }
    output[idx] = acc;
}
"#;

/// One prepared reduction dispatch: folds `splits` partial
/// slices of `total` f32 each into the output.
#[derive(Debug)]
pub struct SplitkReduceCall {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    grid: u32,
    /// Diag timestamp stamp (see `crate::timestamps`); `None`
    /// unless the context carries a recorder.
    diag: Option<crate::timestamps::DiagStamp>,
}

impl SplitkReduceCall {
    /// Record this dispatch into an already-open compute
    /// `pass`, after the split product that wrote the
    /// partials (the composition form the split calls use).
    pub fn encode_into(&self, pass: &mut wgpu::ComputePass<'_>) {
        let slot = self.diag.as_ref().and_then(|d| d.begin(pass));
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.grid, 1, 1);
        if let (Some(d), Some(s)) = (&self.diag, slot) {
            d.end(pass, s);
        }
    }
}

/// The compiled split-K reducer. One instance serves every
/// split product on the device; each split pipeline owns
/// one (compilation is one small shader, and ownership
/// keeps the split call types self-contained).
#[derive(Debug)]
pub struct SplitkReducer {
    pipeline: wgpu::ComputePipeline,
}

impl SplitkReducer {
    /// Compile the reducer on `device`.
    ///
    /// # Errors
    ///
    /// [`WgpuError::Pipeline`] when the device rejects the
    /// shader.
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("splitk reduce shader"),
            source: wgpu::ShaderSource::Wgsl(REDUCE_WGSL.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("splitk reduce pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        dbg_log!("splitk reducer built (workgroup {REDUCE_WORKGROUP})");
        Ok(SplitkReducer { pipeline })
    }

    /// Prepare a reduction of `partials` (laid out
    /// `[splits][total]`) into `output` (`total` f32).
    /// `quant` names the caller's quantization family for
    /// the diag stamp only.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] when `splits` is
    /// outside 2..=[`SPLITK_MAX`], `total` is zero, or a
    /// buffer is smaller than the geometry addresses.
    pub fn prepare(
        &self,
        ctx: &GpuContext,
        partials: &wgpu::Buffer,
        output: &wgpu::Buffer,
        total: u64,
        splits: u32,
        quant: &'static str,
    ) -> Result<SplitkReduceCall> {
        if !(2..=SPLITK_MAX).contains(&splits) {
            return Err(WgpuError::InvalidTensor {
                name: "splitk reduce call".to_string(),
                detail: format!(
                    "splits {splits} is outside 2..={SPLITK_MAX} (a split of 1 is the unsplit product)"
                ),
            });
        }
        if total == 0 {
            return Err(WgpuError::InvalidTensor {
                name: "splitk reduce call".to_string(),
                detail: "total is zero: nothing to reduce".to_string(),
            });
        }
        let need_partials = u64::from(splits) * total * 4;
        let need_out = total * 4;
        if partials.size() < need_partials || output.size() < need_out {
            return Err(WgpuError::InvalidTensor {
                name: "splitk reduce call".to_string(),
                detail: format!(
                    "buffer sizes (partials {}, out {}) are below the geometry's needs \
                     ({need_partials}, {need_out})",
                    partials.size(),
                    output.size()
                ),
            });
        }
        let total32 = u32::try_from(total).map_err(|_| WgpuError::InvalidTensor {
            name: "splitk reduce call".to_string(),
            detail: format!("total {total} exceeds one dispatch's index range"),
        })?;
        let grid = total32.div_ceil(REDUCE_WORKGROUP);
        let params: [u32; 2] = [total32, splits];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splitk reduce params"),
            size: param_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&params_buffer, 0, &param_bytes);
        let entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: partials.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ];
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("splitk reduce bind group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        Ok(SplitkReduceCall {
            pipeline: self.pipeline.clone(),
            bind_group,
            grid,
            diag: ctx.diag_stamp(crate::timestamps::DispatchMeta {
                kernel: "splitk_reduce",
                rows: total,
                cols: u64::from(splits),
                quant,
                weight_bytes: 0,
                shape: format!("splits={splits} total={total}"),
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_shader_sums_slices_in_ascending_order() {
        // The fixed reduction order is the determinism
        // contract: slice 0 seeds the accumulator and the
        // loop walks the remaining slices ascending.
        assert!(REDUCE_WGSL.contains("var acc = partials[idx];"));
        assert!(REDUCE_WGSL.contains("for (var s = 1u; s < params.splits; s++)"));
        assert!(REDUCE_WGSL.contains("acc += partials[s * params.total + idx];"));
        assert!(REDUCE_WGSL.contains("@workgroup_size(256)"));
    }

    #[test]
    fn reduce_shader_validates_under_naga() {
        let module = naga::front::wgsl::parse_str(REDUCE_WGSL)
            .unwrap_or_else(|e| panic!("reduce shader must parse: {e}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("reduce shader must validate: {e:?}"));
    }
}
