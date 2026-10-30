//! K6: grouped-query attention over resident GPU KV buffers
//! (BACKEND.md Section 3), plus the KV state the attention step
//! owns.
//!
//! The semantics are the `attention` crate's, restated for the
//! GPU and pinned by the crate's cross-backend tests: the score
//! scale is `1 / sqrt(head_dim)` (the crate's `AttentionParams`
//! default, which `cpu-compute` uses); a key at absolute position
//! `kp` is visible to a query at `p` when `kp <= p` (causal) and,
//! on windowed layers, when `kp` lies in `(p - window, p]`; query
//! heads group contiguously over KV heads (`kv_head = q_head /
//! (q_heads / kv_heads)`); softmax is max-subtracted.
//!
//! Two kernels:
//!
//! - **Scores**: one thread per (query row, Q head, stored key):
//!   the full dot product over the head dim, masked to negative
//!   infinity where the mask hides the key, written to a scratch
//!   buffer. This is the straightforward form the task allows for
//!   the multi-row (prefill) path, and it is also the decode
//!   path: at one row it is `q_heads * stored` threads, each
//!   reading one key.
//! - **Reduce**: one workgroup per (query row, Q head), one lane
//!   per head-dim channel, running an online (streaming) softmax
//!   over the stored keys in oldest-first order while
//!   accumulating the value mix per channel. Online softmax is
//!   mathematically the reference's two-pass max-subtracted form;
//!   the float summation order differs, which is reassociation
//!   noise, disclosed here the same way K4 and K5 disclose
//!   theirs. Decode leanness: a single query row launches
//!   `q_heads` workgroups and reads each value row exactly once.
//!
//! KV state: like `cpu-compute`, this backend owns its KV
//! storage, because the `Cache` trait the forward hands over is
//! a position contract only. [`GpuKvState`] holds the
//! `attention` crate's `KvCache` as the host mirror (positions,
//! cross-checks, and the exact rows the GPU rotated, read back
//! once per call) plus one device buffer pair per layer. Global
//! layers address slots by absolute position and grow
//! geometrically; windowed layers are a fixed ring of `window`
//! slots, slot = `position % window`, exactly the host cache's
//! law. The device buffers are the ones K5's append writes and
//! this module's scores kernel reads.
//!
//! Head-dim bound: the reduce workgroup is 128 lanes, one per
//! channel, so `head_dim` above 128 refuses by name. Glimmer's
//! head dim is exactly 128; a future wider head needs a strided
//! reduce, which is a named extension, not a silent truncation.

use attention::{KvCache, LayerKind as AttnLayerKind};
use muse_glimmer::{GlimmerConfig, GlimmerError, LayerKind, Result as GlimmerResult};

use crate::device::GpuContext;
use crate::error::{Result, WgpuError};

use crate::diag::dbg_log;

/// Build a [`GlimmerError::StepFailed`] for `step` with `detail`
/// (the state half of this module speaks the trait's error, like
/// `cpu-compute` does).
fn step_failed(step: &'static str, detail: impl Into<String>) -> GlimmerError {
    GlimmerError::StepFailed {
        step,
        detail: detail.into(),
    }
}

/// Workgroup width of the reduce kernel: one lane per head-dim
/// channel, so also the largest head dim it serves.
const REDUCE_LANES: u32 = 128;

/// Threads per workgroup of the scores kernel.
const SCORES_WORKGROUP: u32 = 256;

/// The scores shader: one thread per (row, Q head, stored key).
const SCORES_SHADER: &str = r#"
struct ScoresParams {
    rows: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    stored: u32,
    first_pos: u32,
    start_pos: u32,
    window: u32,
    scale: f32,
    total: u32,
    grid_x: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@group(0) @binding(3) var<uniform> params: ScoresParams;

@compute @workgroup_size(256)
fn scores_main(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>) {
    let flat = (wid.y * params.grid_x + wid.x) * 256u + lid.x;
    if (flat >= params.total) {
        return;
    }
    let idx = flat % params.stored;
    let head = (flat / params.stored) % params.q_heads;
    let row = flat / (params.stored * params.q_heads);
    let kp = params.first_pos + idx;
    let p = params.start_pos + row;

    // The mask, the attention crate's `attends` restated: causal
    // everywhere, plus the window on windowed layers (written as
    // kp + window > p so no subtraction can underflow).
    var visible = kp <= p;
    if (params.window != 0u && kp + params.window <= p) {
        visible = false;
    }
    if (!visible) {
        scores[flat] = -1.0e30;
        return;
    }
    var slot = kp;
    if (params.window != 0u) {
        slot = kp % params.window;
    }
    let kv_head = head / (params.q_heads / params.kv_heads);
    let q_base = (row * params.q_heads + head) * params.head_dim;
    let k_base = (slot * params.kv_heads + kv_head) * params.head_dim;
    var s = 0.0;
    for (var d = 0u; d < params.head_dim; d++) {
        s += q[q_base + d] * keys[k_base + d];
    }
    scores[flat] = s * params.scale;
}
"#;

/// The reduce shader: one workgroup per (row, Q head), one lane
/// per channel, online softmax over the stored keys.
const REDUCE_SHADER: &str = r#"
struct ReduceParams {
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    stored: u32,
    first_pos: u32,
    window: u32,
    grid_x: u32,
}

@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read> values: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: ReduceParams;

@compute @workgroup_size(128)
fn reduce_main(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>) {
    let flat_wg = wid.y * params.grid_x + wid.x;
    let head = flat_wg % params.q_heads;
    let row = flat_wg / params.q_heads;
    let d = lid.x;
    if (d >= params.head_dim) {
        return;
    }
    let kv_head = head / (params.q_heads / params.kv_heads);
    let scores_base = (row * params.q_heads + head) * params.stored;

    var m = -1.0e30;
    var sum = 0.0;
    var acc = 0.0;
    for (var idx = 0u; idx < params.stored; idx++) {
        let sc = scores[scores_base + idx];
        if (sc <= -1.0e29) {
            continue;
        }
        if (sc > m) {
            let corr = exp(m - sc);
            acc *= corr;
            sum *= corr;
            m = sc;
        }
        let w = exp(sc - m);
        let kp = params.first_pos + idx;
        var slot = kp;
        if (params.window != 0u) {
            slot = kp % params.window;
        }
        let v = values[(slot * params.kv_heads + kv_head) * params.head_dim + d];
        acc += w * v;
        sum += w;
    }
    out[(row * params.q_heads + head) * params.head_dim + d] = acc / sum;
}
"#;

/// The shape a [`GpuKvState`] was built from; the same discipline
/// as `cpu-compute`'s (a backend instance serves one model shape;
/// a different one arriving later is a refusal, not a rebuild).
#[derive(Clone, Debug, PartialEq, Eq)]
struct CacheShape {
    layers: Vec<LayerKind>,
    kv_heads: u32,
    key_length: u32,
    value_length: u32,
    sliding_window: u32,
}

impl CacheShape {
    fn of(cfg: &GlimmerConfig) -> Self {
        CacheShape {
            layers: cfg.layers.clone(),
            kv_heads: cfg.kv_heads,
            key_length: cfg.key_length,
            value_length: cfg.value_length,
            sliding_window: cfg.sliding_window,
        }
    }
}

/// One layer's resident device KV: a K buffer and a V buffer of
/// `capacity` rows each, one row being `[kv_heads][head_dim]`.
#[derive(Debug)]
pub struct LayerKv {
    /// Resident keys, slot-addressed (see the module docs).
    pub k: wgpu::Buffer,
    /// Resident values, slot-addressed like the keys.
    pub v: wgpu::Buffer,
    /// Rows the buffers currently hold.
    pub capacity: usize,
}

/// The backend-owned KV state: the host mirror cache (the
/// position contract's other half) plus per-layer device buffers.
#[derive(Debug)]
pub struct GpuKvState {
    host: KvCache,
    shape: CacheShape,
    layers: Vec<Option<LayerKv>>,
}

impl GpuKvState {
    /// Build the state for `cfg`'s shape: the host mirror from
    /// the config's layer pattern, no device buffers yet (they
    /// allocate on first append, per layer).
    ///
    /// # Errors
    ///
    /// [`GlimmerError::StepFailed`] under `step` when the
    /// `attention` crate refuses the pattern.
    pub fn build(cfg: &GlimmerConfig, step: &'static str) -> GlimmerResult<Self> {
        let pattern: Vec<AttnLayerKind> = cfg
            .layers
            .iter()
            .map(|kind| match kind {
                LayerKind::Global => AttnLayerKind::Global,
                LayerKind::SlidingWindow => AttnLayerKind::Windowed {
                    window: cfg.sliding_window as usize,
                },
            })
            .collect();
        let host = KvCache::new(&pattern, cfg.kv_heads as usize, cfg.key_length as usize)
            .map_err(|e| step_failed(step, e.to_string()))?;
        dbg_log!(
            "gpu kv state built layers={} kv_heads={} head_dim={} window={}",
            pattern.len(),
            cfg.kv_heads,
            cfg.key_length,
            cfg.sliding_window
        );
        Ok(GpuKvState {
            host,
            shape: CacheShape::of(cfg),
            layers: (0..pattern.len()).map(|_| None).collect(),
        })
    }

    /// Refuse when `cfg`'s shape differs from the built one.
    ///
    /// # Errors
    ///
    /// [`GlimmerError::StepFailed`] under `step` on a mismatch,
    /// with `cpu-compute`'s wording for the same refusal.
    pub fn check_shape(&self, cfg: &GlimmerConfig, step: &'static str) -> GlimmerResult<()> {
        if self.shape != CacheShape::of(cfg) {
            return Err(step_failed(
                step,
                "this WgpuCompute already serves a model of a different shape; \
                 construct a fresh WgpuCompute per model (or reset this one) \
                 instead of retargeting it mid-sequence",
            ));
        }
        Ok(())
    }

    /// The host mirror cache (positions and stored rows).
    #[must_use]
    pub fn host(&self) -> &KvCache {
        &self.host
    }

    /// The host mirror cache, mutably (appends land here after
    /// K5 rotates, with the read-back rows, so the mirror holds
    /// exactly what the device computed).
    pub fn host_mut(&mut self) -> &mut KvCache {
        &mut self.host
    }

    /// The sliding window this state was built with.
    fn window(&self) -> usize {
        self.shape.sliding_window as usize
    }

    /// Make sure layer `layer`'s device buffers hold at least
    /// the rows its kind needs after `appended_total` appends:
    /// exactly `window` rows for a windowed layer (allocated
    /// once), geometric growth for a global one (old contents
    /// copied across). Returns the layer's buffers.
    ///
    /// # Errors
    ///
    /// [`GlimmerError::StepFailed`] under `step` for an
    /// out-of-range layer.
    pub fn ensure_layer(
        &mut self,
        ctx: &GpuContext,
        layer: usize,
        kv_width: usize,
        appended_total: usize,
        step: &'static str,
    ) -> GlimmerResult<&LayerKv> {
        let kind = self
            .host
            .layer_kind(layer)
            .map_err(|e| step_failed(step, e.to_string()))?;
        let needed = match kind {
            AttnLayerKind::Global => appended_total,
            AttnLayerKind::Windowed { .. } => self.window(),
        };
        let slot = self.layers.get_mut(layer).ok_or_else(|| {
            step_failed(
                step,
                format!("layer {layer} is outside the KV state's layer pattern"),
            )
        })?;
        let current = slot.as_ref().map_or(0, |l| l.capacity);
        if current < needed {
            let new_capacity = if matches!(kind, AttnLayerKind::Windowed { .. }) {
                needed
            } else {
                needed.max(current.saturating_mul(2)).max(256)
            };
            let bytes = (new_capacity * kv_width * 4) as u64;
            let k = ctx.scratch_buffer(&format!("kv keys layer {layer}"), bytes);
            let v = ctx.scratch_buffer(&format!("kv values layer {layer}"), bytes);
            if let Some(old) = slot.as_ref() {
                let old_bytes = (old.capacity * kv_width * 4) as u64;
                let mut encoder =
                    ctx.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("kv grow copy"),
                        });
                encoder.copy_buffer_to_buffer(&old.k, 0, &k, 0, old_bytes);
                encoder.copy_buffer_to_buffer(&old.v, 0, &v, 0, old_bytes);
                ctx.queue.submit([encoder.finish()]);
            }
            dbg_log!(
                "kv layer {layer}: capacity {current} -> {new_capacity} rows ({bytes} bytes per buffer)"
            );
            *slot = Some(LayerKv {
                k,
                v,
                capacity: new_capacity,
            });
        }
        Ok(slot.as_ref().expect("the layer was just ensured"))
    }

    /// Layer `layer`'s device buffers, once ensured.
    ///
    /// # Errors
    ///
    /// [`GlimmerError::StepFailed`] under `step` when the layer
    /// was never ensured.
    pub fn layer(&self, layer: usize, step: &'static str) -> GlimmerResult<&LayerKv> {
        self.layers
            .get(layer)
            .and_then(|l| l.as_ref())
            .ok_or_else(|| {
                step_failed(
                    step,
                    format!("layer {layer} has no device KV buffers; ensure_layer runs first"),
                )
            })
    }
}

/// The geometry both K6 kernels share, bundled so the two calls
/// cannot disagree about the shape of the problem.
#[derive(Clone, Copy, Debug)]
pub struct AttendGeom {
    /// Query rows in this call (prompt chunk or 1 for decode).
    pub rows: usize,
    /// Query head count.
    pub q_heads: usize,
    /// KV head count.
    pub kv_heads: usize,
    /// Per-head dimension (keys and values share it here).
    pub head_dim: usize,
    /// Stored positions visible in the layer after the append.
    pub stored: usize,
    /// Absolute position of the oldest stored entry.
    pub first_pos: usize,
    /// Absolute position of the first query row.
    pub start_pos: usize,
    /// The layer's window, or `None` for a global layer.
    pub window: Option<usize>,
}

/// The compiled K6 pipelines (scores and reduce).
#[derive(Debug)]
pub struct K6Pipeline {
    scores: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
}

impl K6Pipeline {
    /// Compile both K6 pipelines on `device`.
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
        let scores = build(SCORES_SHADER, "scores_main", "k6 attention scores");
        let reduce = build(REDUCE_SHADER, "reduce_main", "k6 attention reduce");
        dbg_log!("k6 pipelines built");
        Ok(K6Pipeline { scores, reduce })
    }

    /// Validate the geometry both kernels are about to run.
    fn check_geom(geom: &AttendGeom) -> Result<()> {
        if geom.rows == 0 || geom.stored == 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k6 attention".to_string(),
                detail: format!(
                    "rows {} and stored {} must both be at least 1",
                    geom.rows, geom.stored
                ),
            });
        }
        if geom.q_heads == 0 || geom.kv_heads == 0 || geom.q_heads % geom.kv_heads != 0 {
            return Err(WgpuError::InvalidTensor {
                name: "k6 attention".to_string(),
                detail: format!(
                    "q_heads {} over kv_heads {} cannot group",
                    geom.q_heads, geom.kv_heads
                ),
            });
        }
        if geom.head_dim == 0 || geom.head_dim > REDUCE_LANES as usize {
            return Err(WgpuError::InvalidTensor {
                name: "k6 attention".to_string(),
                detail: format!(
                    "head_dim {} is outside 1..={REDUCE_LANES}: the reduce workgroup \
                     is one lane per channel, so a wider head refuses by name here \
                     instead of truncating",
                    geom.head_dim
                ),
            });
        }
        Ok(())
    }

    /// Run the scores kernel: `q` (device buffer, `[rows,
    /// q_width]`) against the layer's resident keys, producing
    /// the `[rows, q_heads, stored]` score buffer (device-side;
    /// the reduce kernel consumes it).
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] on an unusable geometry.
    pub fn scores(
        &self,
        ctx: &GpuContext,
        q: &wgpu::Buffer,
        kv: &LayerKv,
        geom: &AttendGeom,
    ) -> Result<wgpu::Buffer> {
        Self::check_geom(geom)?;
        let total = geom.rows * geom.q_heads * geom.stored;
        let scores_buf = ctx.scratch_buffer("k6 scores", (total * 4) as u64);
        let total_wg = (total as u64).div_ceil(u64::from(SCORES_WORKGROUP));
        let grid = crate::kernels::k1_q4k_gemv::grid_dims(total_wg);
        let scale = 1.0f32 / (geom.head_dim as f32).sqrt();
        let params: [u32; 11] = [
            geom.rows as u32,
            geom.q_heads as u32,
            geom.kv_heads as u32,
            geom.head_dim as u32,
            geom.stored as u32,
            geom.first_pos as u32,
            geom.start_pos as u32,
            geom.window.map_or(0, |w| w as u32),
            scale.to_bits(),
            total as u32,
            grid.0,
        ];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buf = ctx.uniform_buffer("k6 scores params", &param_bytes);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k6 scores bind group"),
            layout: &self.scores.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: kv.k.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scores_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k6 scores run"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("k6 attention scores"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scores);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(grid.0, grid.1, 1);
        }
        ctx.queue.submit([encoder.finish()]);
        dbg_log!(
            "k6 scores: rows={} q_heads={} stored={} first_pos={}",
            geom.rows,
            geom.q_heads,
            geom.stored,
            geom.first_pos
        );
        Ok(scores_buf)
    }

    /// Run the reduce kernel over a scores buffer from
    /// [`K6Pipeline::scores`] and the layer's resident values,
    /// returning the attention output `[rows, q_width]` on the
    /// host.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] on an unusable geometry;
    /// [`WgpuError::Readback`] on map failure.
    pub fn reduce(
        &self,
        ctx: &GpuContext,
        scores: &wgpu::Buffer,
        kv: &LayerKv,
        geom: &AttendGeom,
    ) -> Result<Vec<f32>> {
        Self::check_geom(geom)?;
        let out_elems = geom.rows * geom.q_heads * geom.head_dim;
        let out_buf = ctx.scratch_buffer("k6 attention output", (out_elems * 4) as u64);
        let total_wg = (geom.rows * geom.q_heads) as u64;
        let grid = crate::kernels::k1_q4k_gemv::grid_dims(total_wg);
        let params: [u32; 7] = [
            geom.q_heads as u32,
            geom.kv_heads as u32,
            geom.head_dim as u32,
            geom.stored as u32,
            geom.first_pos as u32,
            geom.window.map_or(0, |w| w as u32),
            grid.0,
        ];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params_buf = ctx.uniform_buffer("k6 reduce params", &param_bytes);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("k6 reduce bind group"),
            layout: &self.reduce.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scores.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: kv.v.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k6 reduce run"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("k6 attention reduce"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.reduce);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(grid.0, grid.1, 1);
        }
        ctx.queue.submit([encoder.finish()]);
        dbg_log!(
            "k6 reduce: rows={} q_heads={} stored={}",
            geom.rows,
            geom.q_heads,
            geom.stored
        );
        ctx.read_f32(&out_buf, out_elems)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaders_carry_their_entry_points() {
        assert!(SCORES_SHADER.contains("fn scores_main"));
        assert!(REDUCE_SHADER.contains("fn reduce_main"));
        // The mask sentinel and its guard must agree: scores
        // writes -1e30, the reduce skips anything at that depth.
        assert!(SCORES_SHADER.contains("-1.0e30"));
        assert!(REDUCE_SHADER.contains("-1.0e29"));
    }

    #[test]
    fn geom_checks_refuse_bad_shapes() {
        let mut geom = AttendGeom {
            rows: 1,
            q_heads: 32,
            kv_heads: 2,
            head_dim: 128,
            stored: 4,
            first_pos: 0,
            start_pos: 0,
            window: Some(2048),
        };
        assert!(K6Pipeline::check_geom(&geom).is_ok());
        geom.head_dim = 256;
        assert!(K6Pipeline::check_geom(&geom).is_err());
        geom.head_dim = 128;
        geom.q_heads = 5;
        assert!(K6Pipeline::check_geom(&geom).is_err());
        geom.q_heads = 32;
        geom.stored = 0;
        assert!(K6Pipeline::check_geom(&geom).is_err());
    }
}
