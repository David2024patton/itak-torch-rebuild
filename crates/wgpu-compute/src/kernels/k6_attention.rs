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
//! Three kernels, two forms of the same math:
//!
//! - **Scores**: one thread per (query row, Q head, stored key):
//!   the full dot product over the head dim, masked to negative
//!   infinity where the mask hides the key, written to a scratch
//!   buffer.
//! - **Reduce**: one workgroup per (query row, Q head), one lane
//!   per head-dim channel, running an online (streaming) softmax
//!   over the stored keys in oldest-first order while
//!   accumulating the value mix per channel. Online softmax is
//!   mathematically the reference's two-pass max-subtracted form;
//!   the float summation order differs, which is reassociation
//!   noise, disclosed here the same way K4 and K5 disclose
//!   theirs. Decode leanness: a single query row launches
//!   `q_heads` workgroups and reads each value row exactly once.
//!   Scores + reduce are the two-pass form; the trait path
//!   (`compute.rs`, one row per step) runs them.
//! - **Flash**: the fused form (see the flash shader below):
//!   scores, online softmax, and the value-weighted reduce in
//!   one dispatch, with no scores buffer at all. The resident
//!   session runs it at every shape, decode rows and prefill
//!   chunks alike (the chunk-plan switch is recorded in
//!   bench/flash-chunk-2026-11-04.md).
//! - **Flash, staged**: the same fused math with its memory
//!   traffic reorganized into llama.cpp's blocked form (see
//!   [`FLASH_STAGED_SHADER`]): K and V walked in blocks of
//!   32 keys staged in workgroup memory, 8 query rows per
//!   workgroup. Env-gated at the backend
//!   (`WGPU_COMPUTE_FLASH_STAGED` in `crate::compute`) and
//!   routed only for multi-row calls; the shipped flash
//!   remains the default and the single-row form.
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
    ring: u32,
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
    // The slot modulus is the ring capacity, NOT the mask
    // window: the two are the same for a decode-step ring, but
    // a chunked prefill ring is larger than the window (it must
    // hold every position any query in the chunk can see), and
    // slotting by the window there would alias positions a
    // chunk apart onto one slot.
    var slot = kp;
    if (params.ring != 0u) {
        slot = kp % params.ring;
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
    ring: u32,
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
        // Slot modulus is the ring capacity, as in the scores
        // kernel (see its note): never the mask window.
        var slot = kp;
        if (params.ring != 0u) {
            slot = kp % params.ring;
        }
        let v = values[(slot * params.kv_heads + kv_head) * params.head_dim + d];
        acc += w * v;
        sum += w;
    }
    out[(row * params.q_heads + head) * params.head_dim + d] = acc / sum;
}
"#;

/// The flash shader: scores, online softmax, and the
/// value-weighted reduce fused into one kernel, one workgroup
/// of 256 lanes per (query row, Q head), for the session
/// calls where the two-pass form's scores buffer and second
/// dispatch are pure overhead. The math is the two-pass math,
/// operation for operation: the score phase computes each
/// tile's 256 scores with the scores kernel's thread-to-key
/// mapping and dot order (masked keys get its -1e30 sentinel),
/// stages them in workgroup memory, and the lanes under the
/// head dim then run the reduce kernel's online-softmax loop
/// over the tile in the same ascending order, carrying their
/// (max, sum, accumulator) state across tiles in registers.
/// The sequence of float operations per output element is the
/// two-pass sequence, so the outputs agree with it to the bit
/// in exact arithmetic and to reassociation noise in floats
/// (the scores are computed identically; only their storage
/// differs). The kernel is shape-generic: one workgroup per
/// (query row, Q head) at any row count, the causal position
/// taken per row (`start_pos + row`), keys slotted modulo the
/// ring exactly as in the scores kernel, so the sliding mask
/// and the chunked ring carry over unchanged. The resident
/// session runs this form for decode rows and prefill chunks
/// alike; the two-pass form remains for the trait path.
const FLASH_SHADER: &str = r#"
struct FlashParams {
    rows: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    stored: u32,
    first_pos: u32,
    start_pos: u32,
    window: u32,
    ring: u32,
    scale: f32,
    grid_x: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> keys: array<f32>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: FlashParams;

var<workgroup> tile: array<f32, 256>;

@compute @workgroup_size(256)
fn flash_main(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>) {
    let flat_wg = wid.y * params.grid_x + wid.x;
    let head = flat_wg % params.q_heads;
    let row = flat_wg / params.q_heads;
    let lane = lid.x;
    let kv_head = head / (params.q_heads / params.kv_heads);
    let q_base = (row * params.q_heads + head) * params.head_dim;
    let p = params.start_pos + row;

    var m = -1.0e30;
    var sum = 0.0;
    var acc = 0.0;
    for (var t0 = 0u; t0 < params.stored; t0 += 256u) {
        // Score phase: lane l takes stored key t0 + l.
        let idx = t0 + lane;
        var sc = -1.0e30;
        if (idx < params.stored) {
            let kp = params.first_pos + idx;
            var visible = kp <= p;
            if (params.window != 0u && kp + params.window <= p) {
                visible = false;
            }
            if (visible) {
                var slot = kp;
                if (params.ring != 0u) {
                    slot = kp % params.ring;
                }
                let k_base = (slot * params.kv_heads + kv_head) * params.head_dim;
                var s = 0.0;
                for (var dd = 0u; dd < params.head_dim; dd++) {
                    s += q[q_base + dd] * keys[k_base + dd];
                }
                sc = s * params.scale;
            }
        }
        tile[lane] = sc;
        workgroupBarrier();
        // Reduce phase: the reduce kernel's loop over the
        // tile, ascending, state carried across tiles.
        if (lane < params.head_dim) {
            let tile_len = min(256u, params.stored - t0);
            for (var j = 0u; j < tile_len; j++) {
                let s2 = tile[j];
                if (s2 <= -1.0e29) {
                    continue;
                }
                if (s2 > m) {
                    let corr = exp(m - s2);
                    acc *= corr;
                    sum *= corr;
                    m = s2;
                }
                let w = exp(s2 - m);
                let kp = params.first_pos + t0 + j;
                var slot = kp;
                if (params.ring != 0u) {
                    slot = kp % params.ring;
                }
                let v = values[(slot * params.kv_heads + kv_head) * params.head_dim + lane];
                acc += w * v;
                sum += w;
            }
        }
        workgroupBarrier();
    }
    if (lane < params.head_dim) {
        out[q_base + lane] = acc / sum;
    }
}
"#;

/// Workgroup-memory bytes the staged flash form below
/// declares, pinned against its shader text by a unit test:
/// the K/V block tile (32 keys at a padded 129-float stride,
/// 4,128 f32), the query block tile (8 rows at the 128-float
/// head-dim bound, 1,024 f32), and the score tile (8 rows x
/// 32 keys, 256 f32): (4,128 + 1,024 + 256) * 4 = 21,632
/// bytes. A device granting less than this does not build
/// the staged pipeline at all (see
/// [`K6Pipeline::flash_staged_available`]); the backend
/// requests the adapter's own limits, and the server card
/// grants 64 KiB, so the staged set fits there with room.
pub const FLASH_STAGED_LDS_BYTES: u32 = 21_632;

/// The staged flash shader: the shipped flash math with the
/// memory traffic reorganized into llama.cpp's blocked form
/// (parity spec Family 5, bench/llama-parity-specs-2026-11-06.md
/// gap 5). One workgroup of 256 threads serves a block of
/// Br = 8 query rows for one Q head and walks the stored
/// keys in blocks of Bc = 32 (flash_attn_base.glsl:4-21's
/// Bc; the AMD scalar tuning at head dim 128 and row_split 1
/// picks block_rows 8 and block_cols 32, ggml-vulkan.cpp:
/// 1200-1245). Per KV block the workgroup stages the block's
/// K rows into workgroup memory with coalesced loads, scores
/// all 8 rows against the staged block, stages the block's
/// V rows into the same tile, and runs the online-softmax
/// accumulation reading V from workgroup memory. The query
/// block is staged once up front. The shipped form, by
/// contrast, stages only a 256-wide score tile: its score
/// phase reads every key row from global memory once per
/// query row and its reduce re-reads every value row from
/// global memory, which is the traffic that makes its cost
/// grow quadratically across a prompt (1,013.1 ms at 1,022
/// tokens, bench/prefill-profile-2026-11-05.md).
///
/// Thread map: `tid / 32` is the query row inside the block
/// and `tid % 32` is the lane. In the score phase lane `j`
/// owns key `j` of the block and computes the full dot over
/// the head dim from the two staged tiles. In the reduce
/// phase lane `l` owns channels `l`, `l + 32`, `l + 64`,
/// `l + 96` (those under the head dim) of its row and loops
/// the block's keys in ascending order. The K/V tile's row
/// stride is padded by one float (head_dim + 1) so the
/// score phase's simultaneous reads across keys and the
/// reduce phase's reads across channels both land on
/// distinct workgroup-memory banks at the head dims this
/// engine serves (128 and 64, both 0 mod 32).
///
/// Correctness contract: the per-element float operation
/// sequence is the shipped flash kernel's, exactly. Scores
/// are the same ascending-channel dot products of the same
/// values, multiplied by the same scale; masked keys take
/// the same -1.0e30 sentinel; the reduce carries (max, sum,
/// accumulator) across blocks in ascending key order with
/// the same per-key rescale and the same skip test. Only
/// the storage the operands pass through differs (workgroup
/// tiles instead of global buffers), so the outputs agree
/// with the shipped form to the bit in exact arithmetic and
/// to driver float-contraction noise in practice; the
/// device tests pin them against both the shipped flash and
/// the two-pass form. Staging reads the same KV slots the
/// shipped form reads: every key in the stored range is
/// visible to at least the block row at its own position,
/// so no slot outside the state's capacity is touched.
///
/// Bounds, stated: the head dim is capped at 128 by the
/// same `check_geom` the shipped form runs (the query tile
/// is sized at that bound), and the block walk is uniform
/// across the workgroup (every thread reaches every
/// barrier; rows past the call's row count stage as zeros,
/// compute, and are never written out).
const FLASH_STAGED_SHADER: &str = r#"
struct FlashParams {
    rows: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    stored: u32,
    first_pos: u32,
    start_pos: u32,
    window: u32,
    ring: u32,
    scale: f32,
    grid_x: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> keys: array<f32>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: FlashParams;

// The K/V block tile: 32 keys at a padded (head_dim + 1)
// stride, sized at the head-dim bound of 128.
var<workgroup> kv_tile: array<f32, 4128>;
// The query block tile: 8 rows at the head-dim bound.
var<workgroup> q_tile: array<f32, 1024>;
// The score tile: 8 rows x 32 keys.
var<workgroup> s_tile: array<f32, 256>;

@compute @workgroup_size(256)
fn flash_staged_main(@builtin(workgroup_id) wid: vec3<u32>,
                     @builtin(local_invocation_id) lid: vec3<u32>) {
    let flat_wg = wid.y * params.grid_x + wid.x;
    let head = flat_wg % params.q_heads;
    let row_block = flat_wg / params.q_heads;
    let kv_head = head / (params.q_heads / params.kv_heads);
    let tid = lid.x;
    let row_in_block = tid / 32u;
    let lane = tid % 32u;
    let row = row_block * 8u + row_in_block;
    let p = params.start_pos + row;
    let hd = params.head_dim;
    let stride = hd + 1u;

    // Stage the query block once: rows past the call's row
    // count stage as zeros (they compute and never write
    // out, keeping every barrier uniform).
    for (var e = tid; e < 8u * hd; e += 256u) {
        let r = e / hd;
        let d = e % hd;
        let grow = row_block * 8u + r;
        var v = 0.0;
        if (grow < params.rows) {
            v = q[(grow * params.q_heads + head) * hd + d];
        }
        q_tile[r * hd + d] = v;
    }
    workgroupBarrier();

    var m = -1.0e30;
    var sum = 0.0;
    var acc0 = 0.0;
    var acc1 = 0.0;
    var acc2 = 0.0;
    var acc3 = 0.0;
    for (var t0 = 0u; t0 < params.stored; t0 += 32u) {
        let block_len = min(32u, params.stored - t0);
        // Stage this block's K rows, slot-mapped exactly as
        // the shipped kernels slot them (the ring modulus is
        // the ring capacity, never the mask window).
        for (var e = tid; e < block_len * hd; e += 256u) {
            let j = e / hd;
            let d = e % hd;
            let kp = params.first_pos + t0 + j;
            var slot = kp;
            if (params.ring != 0u) {
                slot = kp % params.ring;
            }
            kv_tile[j * stride + d] =
                keys[(slot * params.kv_heads + kv_head) * hd + d];
        }
        workgroupBarrier();
        // Score phase: lane j scores key t0 + j for this
        // thread's row, the shipped mask and dot order.
        var sc = -1.0e30;
        if (lane < block_len) {
            let kp = params.first_pos + t0 + lane;
            var visible = kp <= p;
            if (params.window != 0u && kp + params.window <= p) {
                visible = false;
            }
            if (visible) {
                var s = 0.0;
                for (var d = 0u; d < hd; d++) {
                    s += q_tile[row_in_block * hd + d] * kv_tile[lane * stride + d];
                }
                sc = s * params.scale;
            }
        }
        s_tile[row_in_block * 32u + lane] = sc;
        workgroupBarrier();
        // Stage this block's V rows into the same tile (the
        // score phase's reads of the K staging are done).
        for (var e = tid; e < block_len * hd; e += 256u) {
            let j = e / hd;
            let d = e % hd;
            let kp = params.first_pos + t0 + j;
            var slot = kp;
            if (params.ring != 0u) {
                slot = kp % params.ring;
            }
            kv_tile[j * stride + d] =
                values[(slot * params.kv_heads + kv_head) * hd + d];
        }
        workgroupBarrier();
        // Reduce phase: the shipped online-softmax loop over
        // the block, ascending, state carried across blocks;
        // lane l accumulates channels l, l + 32, l + 64,
        // l + 96 of its row, reading V from the tile.
        for (var j = 0u; j < block_len; j++) {
            let s2 = s_tile[row_in_block * 32u + j];
            if (s2 <= -1.0e29) {
                continue;
            }
            if (s2 > m) {
                let corr = exp(m - s2);
                acc0 *= corr;
                acc1 *= corr;
                acc2 *= corr;
                acc3 *= corr;
                sum *= corr;
                m = s2;
            }
            let w = exp(s2 - m);
            let vbase = j * stride + lane;
            acc0 += w * kv_tile[vbase];
            if (lane + 32u < hd) {
                acc1 += w * kv_tile[vbase + 32u];
            }
            if (lane + 64u < hd) {
                acc2 += w * kv_tile[vbase + 64u];
            }
            if (lane + 96u < hd) {
                acc3 += w * kv_tile[vbase + 96u];
            }
            sum += w;
        }
        workgroupBarrier();
    }
    if (row < params.rows) {
        let out_base = (row * params.q_heads + head) * hd + lane;
        out[out_base] = acc0 / sum;
        if (lane + 32u < hd) {
            out[out_base + 32u] = acc1 / sum;
        }
        if (lane + 64u < hd) {
            out[out_base + 64u] = acc2 / sum;
        }
        if (lane + 96u < hd) {
            out[out_base + 96u] = acc3 / sum;
        }
    }
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

/// The device half of the KV design, factored out of
/// [`GpuKvState`] so the device-resident session
/// ([`crate::session`]) can own the same per-layer buffers
/// without the host mirror: one buffer pair per layer, global
/// layers addressed by absolute position and grown
/// geometrically, windowed layers a fixed ring of `window`
/// slots, exactly the growth and slotting law the trait path
/// runs. Positions themselves are the owner's business here;
/// this struct manages storage only.
#[derive(Debug)]
pub struct DeviceKv {
    kinds: Vec<AttnLayerKind>,
    ring: usize,
    layers: Vec<Option<LayerKv>>,
}

impl DeviceKv {
    /// Build the storage for a layer pattern: no buffers yet
    /// (they allocate on first append, per layer). `ring` is
    /// the capacity every windowed layer's ring gets: the
    /// sliding window itself for a step-at-a-time driver (the
    /// trait state), or the window plus chunk headroom for a
    /// chunked driver (the resident session; see
    /// [`AttendGeom::ring`] for why the two differ).
    #[must_use]
    pub fn new(pattern: &[AttnLayerKind], ring: usize) -> Self {
        DeviceKv {
            kinds: pattern.to_vec(),
            ring,
            layers: (0..pattern.len()).map(|_| None).collect(),
        }
    }

    /// The windowed layers' ring capacity.
    #[must_use]
    pub fn ring(&self) -> usize {
        self.ring
    }

    /// Make sure layer `layer`'s device buffers hold at least
    /// the rows its kind needs after `appended_total` appends:
    /// exactly the ring capacity for a windowed layer
    /// (allocated once), geometric growth for a global one
    /// (old contents copied across). Returns the layer's
    /// buffers.
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
        let kind = *self.kinds.get(layer).ok_or_else(|| {
            step_failed(
                step,
                format!("layer {layer} is outside the KV state's layer pattern"),
            )
        })?;
        let needed = match kind {
            AttnLayerKind::Global => appended_total,
            AttnLayerKind::Windowed { .. } => self.ring,
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

    /// How many layers the pattern holds.
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
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

/// The backend-owned KV state: the host mirror cache (the
/// position contract's other half) plus per-layer device buffers
/// (in [`DeviceKv`], the storage half).
#[derive(Debug)]
pub struct GpuKvState {
    host: KvCache,
    shape: CacheShape,
    kv: DeviceKv,
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
        let kv = DeviceKv::new(&pattern, cfg.sliding_window as usize);
        Ok(GpuKvState {
            host,
            shape: CacheShape::of(cfg),
            kv,
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

    /// Make sure layer `layer`'s device buffers hold at least
    /// the rows its kind needs after `appended_total` appends
    /// (the storage law lives in [`DeviceKv`]; this is the
    /// mirror-owning state's delegation to it). Returns the
    /// layer's buffers.
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
        self.kv
            .ensure_layer(ctx, layer, kv_width, appended_total, step)
    }

    /// Layer `layer`'s device buffers, once ensured.
    ///
    /// # Errors
    ///
    /// [`GlimmerError::StepFailed`] under `step` when the layer
    /// was never ensured.
    pub fn layer(&self, layer: usize, step: &'static str) -> GlimmerResult<&LayerKv> {
        self.kv.layer(layer, step)
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
    /// The KV ring's capacity for a windowed layer: positions
    /// are slotted modulo this in the device buffers. Equal to
    /// the window when the ring holds exactly one window (the
    /// trait state's layout); larger for the resident session,
    /// whose chunked prefill needs every position any query in
    /// the chunk can see to survive the chunk's appends. The
    /// mask distance is always `window`, never this. `None`
    /// for a global layer (absolute addressing).
    pub ring: Option<usize>,
}

/// One prepared scores dispatch. The geometry's position part
/// (stored count, first stored position, query start) changes
/// every forward, so the call keeps its uniform buffer and
/// rewrites it wholesale through
/// [`K6ScoresCall::set_geometry`]; the buffers and the head
/// geometry are fixed at prepare time. The scores buffer it
/// writes is the caller's, sized for the largest geometry the
/// caller will set. This is the device-resident forward's unit
/// of composition (see [`crate::session`]); the one-shot entry
/// points prepare, encode, and submit.
#[derive(Debug)]
pub struct K6ScoresCall {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    grid: (u32, u32),
    rows: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    window: Option<usize>,
    ring: Option<usize>,
    /// Diag timestamp stamp (see `crate::timestamps`); `None`
    /// unless the context carries a recorder.
    diag: Option<crate::timestamps::DiagStamp>,
}

impl K6ScoresCall {
    /// Record this call's dispatch into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k6 attention scores"),
            timestamp_writes: None,
        });
        self.encode_into(&mut pass);
    }

    /// Record this dispatch into an already-open compute
    /// `pass`: the composition form the device-resident
    /// session uses to chain a whole forward into one pass
    /// per segment (wgpu inserts the hazard barriers
    /// between dispatches inside the pass).
    pub fn encode_into(&self, pass: &mut wgpu::ComputePass<'_>) {
        let slot = self.diag.as_ref().and_then(|d| d.begin(pass));
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.grid.0, self.grid.1, 1);
        if let (Some(d), Some(s)) = (&self.diag, slot) {
            d.end(pass, s);
        }
    }

    /// Point the scores at a new position window: `stored`
    /// entries are visible, the oldest at absolute position
    /// `first_pos`, the query rows starting at `start_pos`.
    /// Rewrites the uniform and the dispatch grid in one go.
    /// Same queue-ordering contract as
    /// [`crate::kernels::k5_rope_kv::K5RotateCall::set_start_position`]:
    /// call it while no submission using this call is in flight.
    pub fn set_geometry(
        &mut self,
        ctx: &GpuContext,
        stored: usize,
        first_pos: usize,
        start_pos: usize,
    ) {
        let total = self.rows * self.q_heads * stored;
        let total_wg = (total as u64).div_ceil(u64::from(SCORES_WORKGROUP));
        self.grid = crate::kernels::k1_q4k_gemv::grid_dims(total_wg);
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let params: [u32; 12] = [
            self.rows as u32,
            self.q_heads as u32,
            self.kv_heads as u32,
            self.head_dim as u32,
            stored as u32,
            first_pos as u32,
            start_pos as u32,
            self.window.map_or(0, |w| w as u32),
            scale.to_bits(),
            total as u32,
            self.grid.0,
            self.ring.map_or(0, |r| r as u32),
        ];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        ctx.queue.write_buffer(&self.params, 0, &param_bytes);
    }
}

/// One prepared reduce dispatch, the scores call's sibling: the
/// position part of its geometry rewrites through
/// [`K6ReduceCall::set_geometry`]; its grid (one workgroup per
/// query row and head) is position-independent.
#[derive(Debug)]
pub struct K6ReduceCall {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    grid: (u32, u32),
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    window: Option<usize>,
    ring: Option<usize>,
    /// Diag timestamp stamp (see `crate::timestamps`); `None`
    /// unless the context carries a recorder.
    diag: Option<crate::timestamps::DiagStamp>,
}

impl K6ReduceCall {
    /// Record this call's dispatch into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k6 attention reduce"),
            timestamp_writes: None,
        });
        self.encode_into(&mut pass);
    }

    /// Record this dispatch into an already-open compute
    /// `pass`: the composition form the device-resident
    /// session uses to chain a whole forward into one pass
    /// per segment (wgpu inserts the hazard barriers
    /// between dispatches inside the pass).
    pub fn encode_into(&self, pass: &mut wgpu::ComputePass<'_>) {
        let slot = self.diag.as_ref().and_then(|d| d.begin(pass));
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.grid.0, self.grid.1, 1);
        if let (Some(d), Some(s)) = (&self.diag, slot) {
            d.end(pass, s);
        }
    }

    /// Point the reduce at a new position window: `stored`
    /// entries, the oldest at absolute position `first_pos`.
    /// Same queue-ordering contract as the scores call's setter.
    pub fn set_geometry(&mut self, ctx: &GpuContext, stored: usize, first_pos: usize) {
        let params: [u32; 8] = [
            self.q_heads as u32,
            self.kv_heads as u32,
            self.head_dim as u32,
            stored as u32,
            first_pos as u32,
            self.window.map_or(0, |w| w as u32),
            self.grid.0,
            self.ring.map_or(0, |r| r as u32),
        ];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        ctx.queue.write_buffer(&self.params, 0, &param_bytes);
    }
}

/// One prepared flash dispatch: the fused scores + online
/// softmax + reduce for session calls of any shape. The position
/// part of its geometry rewrites through
/// [`K6FlashCall::set_geometry`]; its grid (one workgroup per
/// query row and head) is position-independent, like the
/// reduce call's.
#[derive(Debug)]
pub struct K6FlashCall {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    grid: (u32, u32),
    rows: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    window: Option<usize>,
    ring: Option<usize>,
    /// Diag timestamp stamp (see `crate::timestamps`); `None`
    /// unless the context carries a recorder.
    diag: Option<crate::timestamps::DiagStamp>,
}

impl K6FlashCall {
    /// Record this call's dispatch into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("k6 attention flash"),
            timestamp_writes: None,
        });
        self.encode_into(&mut pass);
    }

    /// Record this dispatch into an already-open compute
    /// `pass`: the composition form the device-resident
    /// session uses to chain a whole forward into one pass
    /// per segment (wgpu inserts the hazard barriers
    /// between dispatches inside the pass).
    pub fn encode_into(&self, pass: &mut wgpu::ComputePass<'_>) {
        let slot = self.diag.as_ref().and_then(|d| d.begin(pass));
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.grid.0, self.grid.1, 1);
        if let (Some(d), Some(s)) = (&self.diag, slot) {
            d.end(pass, s);
        }
    }

    /// Point the flash at a new position window: `stored`
    /// entries are visible, the oldest at absolute position
    /// `first_pos`, the query rows starting at `start_pos`.
    /// Same queue-ordering contract as the scores call's
    /// setter.
    pub fn set_geometry(
        &mut self,
        ctx: &GpuContext,
        stored: usize,
        first_pos: usize,
        start_pos: usize,
    ) {
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let params: [u32; 11] = [
            self.rows as u32,
            self.q_heads as u32,
            self.kv_heads as u32,
            self.head_dim as u32,
            stored as u32,
            first_pos as u32,
            start_pos as u32,
            self.window.map_or(0, |w| w as u32),
            self.ring.map_or(0, |r| r as u32),
            scale.to_bits(),
            self.grid.0,
        ];
        let param_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        ctx.queue.write_buffer(&self.params, 0, &param_bytes);
    }
}

/// The compiled K6 pipelines (scores, reduce, and the fused
/// flash form, plus the staged flash form when the device
/// grants its workgroup-memory set).
#[derive(Debug)]
pub struct K6Pipeline {
    scores: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    flash: wgpu::ComputePipeline,
    /// The staged flash form (see [`FLASH_STAGED_SHADER`]);
    /// `None` on a device granting less workgroup storage
    /// than [`FLASH_STAGED_LDS_BYTES`], with the absence
    /// logged at construction. Routing to it is the
    /// backend's config (see `WGPU_COMPUTE_FLASH_STAGED` in
    /// `crate::compute`), never this struct's decision.
    flash_staged: Option<wgpu::ComputePipeline>,
}

impl K6Pipeline {
    /// Compile the K6 pipelines on `device`.
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
        let flash = build(FLASH_SHADER, "flash_main", "k6 attention flash");
        // The staged flash form is optional on the K13
        // terms: a device granting less workgroup storage
        // than its staging set does not build it, and the
        // session keeps the shipped flash form there. A
        // device that does grant the set builds it
        // unconditionally, so a shader defect fails at
        // construction, not on first routed use.
        let granted = device.limits().max_compute_workgroup_storage_size;
        let flash_staged = if granted >= FLASH_STAGED_LDS_BYTES {
            Some(build(
                FLASH_STAGED_SHADER,
                "flash_staged_main",
                "k6 attention flash staged",
            ))
        } else {
            dbg_log!(
                "k6 flash staged unavailable: device grants {granted} bytes of \
                 workgroup storage, below the staged set's {FLASH_STAGED_LDS_BYTES} bytes"
            );
            None
        };
        dbg_log!("k6 pipelines built");
        Ok(K6Pipeline {
            scores,
            reduce,
            flash,
            flash_staged,
        })
    }

    /// Whether the staged flash form was built on this
    /// device (see [`FLASH_STAGED_SHADER`]). The session
    /// consults this alongside the backend's routing switch;
    /// an unavailable staged form is a silent-by-construction
    /// fallback to the shipped flash, with the absence
    /// already named in the construction log above and in
    /// the backend's config line.
    #[must_use]
    pub fn flash_staged_available(&self) -> bool {
        self.flash_staged.is_some()
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

    /// Prepare a scores dispatch: `q` (device buffer, `[rows,
    /// q_width]`) against the layer's resident keys, writing the
    /// `[rows, q_heads, stored]` scores into the caller's
    /// `scores_buf` (sized for the largest geometry the caller
    /// will set through the returned call).
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] on an unusable geometry.
    pub fn prepare_scores(
        &self,
        ctx: &GpuContext,
        q: &wgpu::Buffer,
        kv: &LayerKv,
        scores_buf: &wgpu::Buffer,
        geom: &AttendGeom,
    ) -> Result<K6ScoresCall> {
        Self::check_geom(geom)?;
        let params_buf = ctx.uniform_buffer("k6 scores params", &[0u8; 48]);
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
        let mut call = K6ScoresCall {
            pipeline: self.scores.clone(),
            bind_group,
            params: params_buf,
            grid: (0, 0),
            rows: geom.rows,
            q_heads: geom.q_heads,
            kv_heads: geom.kv_heads,
            head_dim: geom.head_dim,
            window: geom.window,
            ring: geom.ring,
            diag: ctx.diag_stamp(crate::timestamps::DispatchMeta {
                kernel: "k6_attn_scores",
                rows: geom.rows as u64,
                cols: (geom.q_heads * geom.head_dim) as u64,
                quant: "",
                weight_bytes: 0,
                shape: format!(
                    "rows={} q_heads={} kv_heads={} head_dim={} window={:?}",
                    geom.rows, geom.q_heads, geom.kv_heads, geom.head_dim, geom.window
                ),
            }),
        };
        call.set_geometry(ctx, geom.stored, geom.first_pos, geom.start_pos);
        Ok(call)
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
        let call = self.prepare_scores(ctx, q, kv, &scores_buf, geom)?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k6 scores run"),
            });
        call.encode(&mut encoder);
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

    /// Prepare a reduce dispatch over a scores buffer and the
    /// layer's resident values, writing the attention output
    /// `[rows, q_width]` into the caller's `out` buffer.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] on an unusable geometry.
    pub fn prepare_reduce(
        &self,
        ctx: &GpuContext,
        scores: &wgpu::Buffer,
        kv: &LayerKv,
        out: &wgpu::Buffer,
        geom: &AttendGeom,
    ) -> Result<K6ReduceCall> {
        Self::check_geom(geom)?;
        let total_wg = (geom.rows * geom.q_heads) as u64;
        let grid = crate::kernels::k1_q4k_gemv::grid_dims(total_wg);
        let params_buf = ctx.uniform_buffer("k6 reduce params", &[0u8; 32]);
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
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let mut call = K6ReduceCall {
            pipeline: self.reduce.clone(),
            bind_group,
            params: params_buf,
            grid,
            q_heads: geom.q_heads,
            kv_heads: geom.kv_heads,
            head_dim: geom.head_dim,
            window: geom.window,
            ring: geom.ring,
            diag: ctx.diag_stamp(crate::timestamps::DispatchMeta {
                kernel: "k6_attn_reduce",
                rows: geom.rows as u64,
                cols: (geom.q_heads * geom.head_dim) as u64,
                quant: "",
                weight_bytes: 0,
                shape: format!(
                    "q_heads={} kv_heads={} head_dim={} window={:?}",
                    geom.q_heads, geom.kv_heads, geom.head_dim, geom.window
                ),
            }),
        };
        call.set_geometry(ctx, geom.stored, geom.first_pos);
        Ok(call)
    }

    /// Prepare a flash dispatch: `q` (device buffer, `[rows,
    /// q_width]`) against the layer's resident keys and
    /// values, writing the attention output `[rows, q_width]`
    /// into the caller's `out` buffer in the one fused kernel
    /// (see [`FLASH_SHADER`]). The resident session picks this
    /// form at every shape; the two-pass form remains for the
    /// trait path's step-at-a-time calls.
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] on an unusable geometry.
    pub fn prepare_flash(
        &self,
        ctx: &GpuContext,
        q: &wgpu::Buffer,
        kv: &LayerKv,
        out: &wgpu::Buffer,
        geom: &AttendGeom,
    ) -> Result<K6FlashCall> {
        self.prepare_flash_form(
            ctx,
            q,
            kv,
            out,
            geom,
            &self.flash,
            "k6_attn_flash",
            "k6 flash bind group",
            (geom.rows * geom.q_heads) as u64,
        )
    }

    /// Prepare a staged flash dispatch: the same contract,
    /// bindings, params, and output as [`Self::prepare_flash`],
    /// run through the staged shader (see
    /// [`FLASH_STAGED_SHADER`]): one workgroup per block of
    /// 8 query rows and Q head, K/V walked in staged blocks
    /// of 32 keys. The session routes here only when the
    /// backend's flash-staged switch is on and the call has
    /// more than one row (a one-row call would stage a block
    /// for a single active row of eight; llama routes N == 1
    /// to its scalar path on the same reasoning,
    /// ggml-vulkan.cpp's `get_fa_tuning_params`).
    ///
    /// # Errors
    ///
    /// [`WgpuError::InvalidTensor`] on an unusable geometry,
    /// or when the staged pipeline was not built on this
    /// device (workgroup storage below
    /// [`FLASH_STAGED_LDS_BYTES`]): a named refusal, never a
    /// silent substitution. Callers that want the fallback
    /// check [`Self::flash_staged_available`] first, as the
    /// session does.
    pub fn prepare_flash_staged(
        &self,
        ctx: &GpuContext,
        q: &wgpu::Buffer,
        kv: &LayerKv,
        out: &wgpu::Buffer,
        geom: &AttendGeom,
    ) -> Result<K6FlashCall> {
        let Some(pipeline) = &self.flash_staged else {
            return Err(WgpuError::InvalidTensor {
                name: "k6 attention flash staged".to_string(),
                detail: format!(
                    "the staged flash pipeline was not built on this device: \
                     its staging set needs {FLASH_STAGED_LDS_BYTES} bytes of \
                     workgroup storage"
                ),
            });
        };
        self.prepare_flash_form(
            ctx,
            q,
            kv,
            out,
            geom,
            pipeline,
            "k6_attn_flash_staged",
            "k6 flash staged bind group",
            ((geom.rows.div_ceil(8)) * geom.q_heads) as u64,
        )
    }

    /// The shared body of the two flash prepares: identical
    /// bind group, params, and call shape; the forms differ
    /// only in pipeline, diag kernel name, and dispatch grid
    /// (one workgroup per query row in the shipped form, per
    /// block of 8 query rows in the staged form).
    #[allow(clippy::too_many_arguments)]
    fn prepare_flash_form(
        &self,
        ctx: &GpuContext,
        q: &wgpu::Buffer,
        kv: &LayerKv,
        out: &wgpu::Buffer,
        geom: &AttendGeom,
        pipeline: &wgpu::ComputePipeline,
        kernel: &'static str,
        bind_label: &'static str,
        total_wg: u64,
    ) -> Result<K6FlashCall> {
        Self::check_geom(geom)?;
        let grid = crate::kernels::k1_q4k_gemv::grid_dims(total_wg);
        let params_buf = ctx.uniform_buffer("k6 flash params", &[0u8; 44]);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(bind_label),
            layout: &pipeline.get_bind_group_layout(0),
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
                    resource: kv.v.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let mut call = K6FlashCall {
            pipeline: pipeline.clone(),
            bind_group,
            params: params_buf,
            grid,
            rows: geom.rows,
            q_heads: geom.q_heads,
            kv_heads: geom.kv_heads,
            head_dim: geom.head_dim,
            window: geom.window,
            ring: geom.ring,
            diag: ctx.diag_stamp(crate::timestamps::DispatchMeta {
                kernel,
                rows: geom.rows as u64,
                cols: (geom.q_heads * geom.head_dim) as u64,
                quant: "",
                weight_bytes: 0,
                shape: format!(
                    "rows={} q_heads={} kv_heads={} head_dim={} window={:?}",
                    geom.rows, geom.q_heads, geom.kv_heads, geom.head_dim, geom.window
                ),
            }),
        };
        call.set_geometry(ctx, geom.stored, geom.first_pos, geom.start_pos);
        Ok(call)
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
        let call = self.prepare_reduce(ctx, scores, kv, &out_buf, geom)?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("k6 reduce run"),
            });
        call.encode(&mut encoder);
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
        assert!(FLASH_SHADER.contains("fn flash_main"));
        assert!(FLASH_STAGED_SHADER.contains("fn flash_staged_main"));
        // The mask sentinel and its guard must agree: scores
        // writes -1e30, the reduce skips anything at that depth.
        assert!(SCORES_SHADER.contains("-1.0e30"));
        assert!(REDUCE_SHADER.contains("-1.0e29"));
        assert!(FLASH_SHADER.contains("-1.0e30"));
        assert!(FLASH_SHADER.contains("-1.0e29"));
        assert!(FLASH_STAGED_SHADER.contains("-1.0e30"));
        assert!(FLASH_STAGED_SHADER.contains("-1.0e29"));
    }

    #[test]
    fn staged_lds_arithmetic_matches_the_shader() {
        // The three workgroup arrays, by their declared
        // sizes: the K/V block tile (32 keys at the padded
        // 129-float stride), the query block tile (8 rows
        // at the 128-float head-dim bound), and the score
        // tile (8 rows x 32 keys). Their byte sum is the
        // constant the pipeline build gates on; a shader
        // edit that grows a tile without moving the
        // constant (or the reverse) fails here, before a
        // device with exactly-enough storage misbuilds.
        assert!(FLASH_STAGED_SHADER.contains("var<workgroup> kv_tile: array<f32, 4128>;"));
        assert!(FLASH_STAGED_SHADER.contains("var<workgroup> q_tile: array<f32, 1024>;"));
        assert!(FLASH_STAGED_SHADER.contains("var<workgroup> s_tile: array<f32, 256>;"));
        assert_eq!((4128 + 1024 + 256) * 4, FLASH_STAGED_LDS_BYTES);
        assert_eq!(FLASH_STAGED_LDS_BYTES, 21_632);
    }

    #[test]
    fn staged_shader_validates_under_naga() {
        // The strongest shader gate a machine with no GPU
        // adapter at all can run: the staged shader must
        // parse and validate under Naga, the same front end
        // wgpu feeds the driver with. This proves the WGSL
        // is well-formed and well-typed; it does not prove
        // a driver's compiler accepts it or that the
        // staging wins on hardware, which is the server
        // protocol's work (bench/flash-rework-2026-11-06.md).
        let module = naga::front::wgsl::parse_str(FLASH_STAGED_SHADER).unwrap_or_else(|e| {
            panic!(
                "staged flash WGSL parse failed: {}",
                e.emit_to_string(FLASH_STAGED_SHADER)
            )
        });
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("staged flash Naga validation failed: {e:?}"));
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
            ring: Some(2048),
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
