//! Per-dispatch GPU timestamp instrumentation: every kernel,
//! every block (David's order, 2026-11-04). When enabled, each
//! prepared call carries a [`DiagStamp`]; its `encode_into`
//! brackets the dispatch with two timestamp writes inside the
//! pass, and a later [`DiagRecorder::drain`] resolves the query
//! set in one extra submission and emits one structured record
//! per dispatch through this crate's diagnostic channel (the
//! `[wgpu-compute-diag]` lines on stderr), so a smoke run
//! reports everything that fired, in order, with GPU durations.
//!
//! The drain is driven by the sessions themselves: every
//! resident session drains after each submitted chunk and
//! folds the records into a [`DiagAccumulator`], which writes
//! the per-kernel summary table to the [`DIAG_OUT_ENV`] file
//! when the run finishes. No caller has to collect anything
//! by hand for a real run to produce the report.
//!
//! Logger name: `wgpu-compute::timestamps`, records under the
//! `[wgpu-compute-diag]` prefix. Raise it with
//! `WGPU_COMPUTE_DIAG=timing`; the config default is off, and off
//! means no query set, no resolve buffers, and no behavior
//! change anywhere (verdict runs stay clean by construction).
//! `WGPU_COMPUTE_DIAG=capture` implies `timing` and adds the
//! data level: a run manifest (build, adapter, resolved
//! config, model file fingerprint, session config) and a
//! fingerprint of every completed forward's output, so two
//! runs can be diffed mechanically when one of them computed
//! the wrong thing (see the capture section below).
//! Failure modes: a device without the timestamp features
//! refuses the recorder at device open (named in the diag log,
//! instrumentation simply absent); a full query window drops
//! stamps and counts them rather than corrupting timings.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::error::{Result, WgpuError};

/// Environment variable gating the instrumentation:
/// `off` (default), `timing`, or `capture`.
pub const DIAG_ENV: &str = "WGPU_COMPUTE_DIAG";

/// Environment variable naming the file a session writes its
/// per-kernel summary table to when its run finishes (see
/// [`DiagAccumulator`]).
pub const DIAG_OUT_ENV: &str = "WGPU_COMPUTE_DIAG_OUT";

/// Hard wgpu cap on one query set: a `QuerySetDescriptor`
/// count above this fails validation when the set is created
/// (measured on the real card under RADV: an 8192-query set
/// panicked before any dispatch was timed). Every query set
/// this module creates is sized at or under it.
pub const MAX_QUERIES_PER_SET: u32 = 4096;

/// Dispatch capacity of one query window: the query set holds
/// two timestamps per dispatch, so one capped set yields
/// `MAX_QUERIES_PER_SET / 2` dispatch windows. One resident
/// forward of the full model fires on the order of a thousand
/// dispatches, so a window covers about two forwards; the
/// sessions drain after every chunk (one submission, a few
/// hundred dispatches), which keeps them far inside it.
/// Overflow drops and counts, never wraps into another
/// dispatch's slots.
///
/// One capped set, not several: the drain path resolves one
/// contiguous range into one resolve/staging buffer pair, and
/// a second set would add a second resolve range, a second
/// copy, and per-set slot mapping for partial windows, all to
/// buy capacity the per-forward drain cadence never reaches.
pub const MAX_DISPATCHES: u32 = MAX_QUERIES_PER_SET / 2;

// The sizing rule, pinned at compile time: the recorder's one
// query set can never exceed the wgpu cap, and the window can
// never shrink below the 2048-dispatch floor, whatever the
// constants above are later edited to.
const _: () = assert!(MAX_DISPATCHES * 2 <= MAX_QUERIES_PER_SET);
const _: () = assert!(MAX_QUERIES_PER_SET <= 4096);
const _: () = assert!(MAX_DISPATCHES >= 2048);

/// Queries in the recorder's one query set: two per dispatch
/// in the window, at or under [`MAX_QUERIES_PER_SET`] by the
/// compile-time pin above. The resolve and staging buffers
/// are sized off this same count.
fn window_query_count() -> u32 {
    MAX_DISPATCHES * 2
}

/// The instrumentation mode, parsed from [`DIAG_ENV`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagMode {
    /// No instrumentation: no query resources are created.
    Off,
    /// Per-dispatch GPU timestamps on.
    Timing,
    /// Timestamps on, plus the capture level: the run
    /// manifest and per-forward data fingerprints the
    /// accumulator records (see [`DiagAccumulator::begin_manifest`]
    /// and [`DiagAccumulator::record_forward`]). Capture
    /// implies timing everywhere the two are distinguished:
    /// it needs the same recorder, the same features, and the
    /// same session accumulator.
    Capture,
}

impl DiagMode {
    /// Parse a raw env value: only the exact strings `timing`
    /// and `capture` enable instrumentation; anything else
    /// (including a missing variable) is off, so a typo fails
    /// closed to the clean path.
    #[must_use]
    pub fn parse(raw: Option<&str>) -> DiagMode {
        match raw {
            Some("timing") => DiagMode::Timing,
            Some("capture") => DiagMode::Capture,
            _ => DiagMode::Off,
        }
    }

    /// The mode the current process environment requests.
    #[must_use]
    pub fn from_env() -> DiagMode {
        DiagMode::parse(std::env::var(DIAG_ENV).ok().as_deref())
    }

    /// The mode's env spelling, for manifests and log lines.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            DiagMode::Off => "off",
            DiagMode::Timing => "timing",
            DiagMode::Capture => "capture",
        }
    }

    /// Whether this mode runs the timestamp machinery at all:
    /// timing, and capture which implies it. Device open asks
    /// for the timestamp features exactly when this holds, so
    /// the off path's device request is unchanged.
    #[must_use]
    pub fn requests_timestamps(self) -> bool {
        self != DiagMode::Off
    }

    /// Whether this is the capture level.
    #[must_use]
    pub fn is_capture(self) -> bool {
        self == DiagMode::Capture
    }

    /// Whether this mode, on a device granted `features`,
    /// creates a recorder. Both timestamp features are
    /// required: set-level queries for resolution and
    /// inside-pass writes for per-dispatch brackets.
    #[must_use]
    pub fn wants_recorder(self, features: wgpu::Features) -> bool {
        self.requests_timestamps()
            && features.contains(
                wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
            )
    }
}

/// Static description of one dispatch, captured at prepare
/// time: everything the record needs except its duration.
#[derive(Clone, Debug)]
pub struct DispatchMeta {
    /// Kernel name, e.g. `k1_q4k_gemv`.
    pub kernel: &'static str,
    /// Output rows of the dispatch.
    pub rows: u64,
    /// Input columns of the dispatch.
    pub cols: u64,
    /// Quantization scheme of the weight stream, e.g. `Q4_K`;
    /// empty for kernels with no quantized weight stream.
    pub quant: &'static str,
    /// Weight bytes this dispatch streams from device memory;
    /// zero for kernels without a weight stream (norms,
    /// attention, elementwise), where the GB/s column of the
    /// summary is not meaningful.
    pub weight_bytes: u64,
    /// Free-form shape descriptor: batch, grid, stored KV
    /// length, whatever distinguishes this dispatch's shape.
    pub shape: String,
}

/// One resolved dispatch record: the meta plus its measured
/// GPU duration. This is the unit the smoke report aggregates.
#[derive(Clone, Debug)]
pub struct DispatchRecord {
    /// Global dispatch index within the recorder's lifetime,
    /// in encode order.
    pub index: u64,
    /// Kernel name (see [`DispatchMeta::kernel`]).
    pub kernel: &'static str,
    /// Output rows.
    pub rows: u64,
    /// Input columns.
    pub cols: u64,
    /// Quantization scheme, or empty.
    pub quant: &'static str,
    /// Weight bytes streamed.
    pub weight_bytes: u64,
    /// Shape descriptor.
    pub shape: String,
    /// Measured GPU duration in nanoseconds.
    pub duration_ns: u64,
}

impl DispatchRecord {
    /// The effective stream rate of this dispatch, in GB/s
    /// (decimal), or zero when it has no weight stream or no
    /// measured duration.
    #[must_use]
    pub fn gbps(&self) -> f64 {
        rate_gbps(self.weight_bytes, self.duration_ns)
    }
}

/// Stream rate in GB/s for `bytes` moved in `duration_ns`.
/// Zero duration or zero bytes yields zero, never NaN.
#[must_use]
pub fn rate_gbps(bytes: u64, duration_ns: u64) -> f64 {
    if bytes == 0 || duration_ns == 0 {
        return 0.0;
    }
    bytes as f64 / (duration_ns as f64 / 1_000_000_000.0) / 1_000_000_000.0
}

/// Convert a raw timestamp delta (device ticks, wrapping at
/// 2^64) to nanoseconds with the queue's period. The period is
/// read from the queue at device open and carried here; it is
/// never assumed (the often-quoted 10.0 ns is one device's
/// value, not a constant of the API).
#[must_use]
pub fn delta_ns(start: u64, end: u64, period_ns: f32) -> u64 {
    let ticks = end.wrapping_sub(start);
    (ticks as f64 * f64::from(period_ns)).round() as u64
}

/// Per-kernel aggregation of a record set: the smoke report's
/// rows.
#[derive(Clone, Debug)]
pub struct KernelSummary {
    /// Kernel name.
    pub kernel: String,
    /// Dispatches recorded.
    pub count: u64,
    /// Sum of GPU durations, nanoseconds.
    pub total_ns: u64,
    /// Sum of weight bytes streamed.
    pub total_bytes: u64,
}

impl KernelSummary {
    /// Total GPU time in milliseconds.
    #[must_use]
    pub fn total_ms(&self) -> f64 {
        self.total_ns as f64 / 1_000_000.0
    }

    /// Aggregate stream rate in GB/s across all its dispatches.
    #[must_use]
    pub fn gbps(&self) -> f64 {
        rate_gbps(self.total_bytes, self.total_ns)
    }
}

/// Aggregate records into per-kernel summaries, sorted by
/// total GPU time descending (the report's reading order), and
/// return them with the grand total duration in nanoseconds.
#[must_use]
pub fn summarize(records: &[DispatchRecord]) -> (Vec<KernelSummary>, u64) {
    let mut by_kernel: BTreeMap<&'static str, KernelSummary> = BTreeMap::new();
    let mut total_ns = 0u64;
    for r in records {
        total_ns += r.duration_ns;
        let entry = by_kernel.entry(r.kernel).or_insert_with(|| KernelSummary {
            kernel: r.kernel.to_string(),
            count: 0,
            total_ns: 0,
            total_bytes: 0,
        });
        entry.count += 1;
        entry.total_ns += r.duration_ns;
        entry.total_bytes += r.weight_bytes;
    }
    let mut rows: Vec<KernelSummary> = by_kernel.into_values().collect();
    rows.sort_by(|a, b| b.total_ns.cmp(&a.total_ns).then(a.kernel.cmp(&b.kernel)));
    (rows, total_ns)
}

/// Render the per-kernel summary table: one row per kernel
/// with dispatch count, total time, share of the run's GPU
/// time, streamed bytes, and the aggregate GB/s where a
/// weight stream exists.
#[must_use]
pub fn render_summary(records: &[DispatchRecord]) -> String {
    let (rows, total_ns) = summarize(records);
    render_summaries(&rows, total_ns, records.len() as u64)
}

/// Render a summary table from pre-aggregated rows (the
/// [`summarize`] output shape, or a [`DiagAccumulator`]'s
/// folded equivalent): byte-identical to [`render_summary`]
/// over the same record set, which the accumulator's tests
/// pin.
#[must_use]
pub fn render_summaries(rows: &[KernelSummary], total_ns: u64, total_records: u64) -> String {
    let mut out = String::new();
    out.push_str("kernel,count,total_ms,share_pct,weight_bytes,gbps\n");
    for row in rows {
        let share = if total_ns == 0 {
            0.0
        } else {
            row.total_ns as f64 * 100.0 / total_ns as f64
        };
        out.push_str(&format!(
            "{},{},{:.3},{:.1},{},{:.1}\n",
            row.kernel,
            row.count,
            row.total_ms(),
            share,
            row.total_bytes,
            row.gbps()
        ));
    }
    out.push_str(&format!(
        "TOTAL,{},{:.3},100.0,,\n",
        total_records,
        total_ns as f64 / 1_000_000.0
    ));
    out
}

/// Slot allocator for one query window: hands out dispatch
/// slots (two queries each) until the window is full, then
/// refuses and counts the drop. Pure and separately testable;
/// the recorder wraps it with the actual query set.
#[derive(Debug)]
pub struct SlotAllocator {
    next: AtomicU32,
    capacity: u32,
    dropped: AtomicU64,
}

impl SlotAllocator {
    /// A window holding `capacity` dispatches.
    #[must_use]
    pub fn new(capacity: u32) -> SlotAllocator {
        SlotAllocator {
            next: AtomicU32::new(0),
            capacity,
            dropped: AtomicU64::new(0),
        }
    }

    /// Take the next dispatch slot, or `None` (counted as a
    /// drop) when the window is full.
    pub fn alloc(&self) -> Option<u32> {
        let slot = self.next.fetch_add(1, Ordering::Relaxed);
        if slot < self.capacity {
            Some(slot)
        } else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Slots handed out so far in this window (including any
    /// that overflowed); the resolve range covers the first
    /// `min(used, capacity)` of them.
    #[must_use]
    pub fn used(&self) -> u32 {
        self.next.load(Ordering::Relaxed).min(self.capacity)
    }

    /// Dispatches dropped because the window was full.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Start a fresh window after a drain.
    pub fn reset(&self) {
        self.next.store(0, Ordering::Relaxed);
    }
}

/// A stamp resolved but not yet drained: its slot and meta.
#[derive(Debug)]
struct Pending {
    slot: u32,
    index: u64,
    meta: DispatchMeta,
}

/// The per-device timestamp recorder: one query set, its
/// resolve and staging buffers, and the pending/completed
/// record queues. Created by `GpuContext::open` only when the
/// diag mode is `timing` and the device granted the timestamp
/// features; `None` everywhere else, which is what keeps the
/// off path free of query resources by construction.
#[derive(Debug)]
pub struct DiagRecorder {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    staging: wgpu::Buffer,
    period_ns: f32,
    slots: SlotAllocator,
    index: AtomicU64,
    pending: Mutex<Vec<Pending>>,
    completed: Mutex<Vec<DispatchRecord>>,
}

impl DiagRecorder {
    /// Create the query resources on `device`. `period_ns` is
    /// the queue's own timestamp period, read by the caller
    /// from `queue.get_timestamp_period()`.
    #[must_use]
    pub fn new(device: &wgpu::Device, period_ns: f32) -> DiagRecorder {
        let queries = window_query_count();
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("wgpu-compute diag timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: queries,
        });
        let bytes = u64::from(queries) * 8;
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-compute diag resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-compute diag staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        DiagRecorder {
            query_set,
            resolve,
            staging,
            period_ns,
            slots: SlotAllocator::new(MAX_DISPATCHES),
            index: AtomicU64::new(0),
            pending: Mutex::new(Vec::new()),
            completed: Mutex::new(Vec::new()),
        }
    }

    /// The queue timestamp period this recorder converts with.
    #[must_use]
    pub fn period_ns(&self) -> f32 {
        self.period_ns
    }

    /// Dispatches dropped so far because a window filled
    /// before a drain. Cumulative for the recorder's lifetime:
    /// a drain resolves and resets the window but never this
    /// counter, so it is the per-backend total every session
    /// on the backend reports against.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.slots.dropped()
    }

    /// Dispatch slots handed out in the current window and
    /// not yet drained: how many stamps the next
    /// [`DiagRecorder::drain`] will resolve.
    #[must_use]
    pub fn pending_dispatches(&self) -> u32 {
        self.slots.used()
    }

    /// Open a stamp bracket inside `pass`: allocate a slot and
    /// write its start timestamp. `None` when the window is
    /// full (the dispatch proceeds unstamped and the drop is
    /// counted).
    fn begin(&self, pass: &mut wgpu::ComputePass<'_>) -> Option<u32> {
        let slot = self.slots.alloc()?;
        pass.write_timestamp(&self.query_set, slot * 2);
        Some(slot)
    }

    /// Close a stamp bracket: write the end timestamp and
    /// queue the record for the next drain.
    fn end(&self, pass: &mut wgpu::ComputePass<'_>, slot: u32, meta: DispatchMeta) {
        pass.write_timestamp(&self.query_set, slot * 2 + 1);
        let index = self.index.fetch_add(1, Ordering::Relaxed);
        self.pending
            .lock()
            .unwrap()
            .push(Pending { slot, index, meta });
    }

    /// Resolve every pending stamp in one extra submission,
    /// convert ticks to nanoseconds with the queue period,
    /// emit one structured record per dispatch through the
    /// diag channel, and queue the records for
    /// [`DiagRecorder::take_records`]. Returns how many records
    /// were resolved. The submission happens here, after the
    /// caller's own submissions, so the hot encode path never
    /// waits on the GPU for diagnostics.
    ///
    /// # Errors
    ///
    /// [`WgpuError::Readback`] when the staging map fails.
    pub fn drain(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<usize> {
        let used = self.slots.used();
        if used == 0 {
            return Ok(0);
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu-compute diag resolve"),
        });
        encoder.resolve_query_set(&self.query_set, 0..used * 2, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.staging, 0, u64::from(used) * 16);
        queue.submit([encoder.finish()]);

        let slice = self.staging.slice(..u64::from(used) * 16);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| WgpuError::Readback {
                detail: e.to_string(),
            })?;
        rx.recv()
            .map_err(|_| WgpuError::Readback {
                detail: "diag map callback never fired".to_string(),
            })?
            .map_err(|e| WgpuError::Readback {
                detail: e.to_string(),
            })?;
        let mapped = slice.get_mapped_range();
        let ticks: Vec<u64> = mapped
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect();
        drop(mapped);
        self.staging.unmap();

        let pending = std::mem::take(&mut *self.pending.lock().unwrap());
        let mut resolved = Vec::with_capacity(pending.len());
        for p in pending {
            let start = ticks[(p.slot * 2) as usize];
            let end = ticks[(p.slot * 2 + 1) as usize];
            let record = DispatchRecord {
                index: p.index,
                kernel: p.meta.kernel,
                rows: p.meta.rows,
                cols: p.meta.cols,
                quant: p.meta.quant,
                weight_bytes: p.meta.weight_bytes,
                shape: p.meta.shape,
                duration_ns: delta_ns(start, end, self.period_ns),
            };
            eprintln!(
                "[wgpu-compute-diag] dispatch #{} {} rows={} cols={} quant={} weight_bytes={} shape={} duration_ns={} gbps={:.1}",
                record.index,
                record.kernel,
                record.rows,
                record.cols,
                if record.quant.is_empty() { "-" } else { record.quant },
                record.weight_bytes,
                record.shape,
                record.duration_ns,
                record.gbps()
            );
            resolved.push(record);
        }
        let n = resolved.len();
        self.completed.lock().unwrap().extend(resolved);
        self.slots.reset();
        Ok(n)
    }

    /// Take all completed records, leaving the queue empty.
    /// A [`DiagAccumulator`] folds them out after each of its
    /// drains, so in a session-driven run the queue holds at
    /// most one window's worth at a time.
    #[must_use]
    pub fn take_records(&self) -> Vec<DispatchRecord> {
        std::mem::take(&mut *self.completed.lock().unwrap())
    }
}

/// The session-side half of the diag story: the running
/// summary one run accumulates as its session drains the
/// recorder. A session creates one only when its context
/// carries a recorder (timing mode on a capable device), so
/// its absence is the off path's guarantee that no report
/// work happens at all.
///
/// Cadence: the owning session calls
/// [`DiagAccumulator::drain_and_fold`] after every submitted
/// chunk. One chunk is one submission of a few hundred
/// dispatches (the full model's chunk measured 445; the
/// fixture's under a hundred), far inside the 2048-dispatch
/// window, so pending stamps never approach the cap and the
/// only way a stamp is lost is a single chunk bigger than
/// the whole window, which no plan shape produces; even then
/// the loss is the recorder's counted drop, surfaced in the
/// finish line, never a silent one. Folding after each drain
/// also bounds memory: the accumulator holds one aggregate
/// row per kernel, not one record per dispatch, however long
/// the run.
///
/// Finish: [`DiagAccumulator::write_report`] (the sessions
/// call it from their end-of-run finish, explicit or at drop)
/// prints the one-line tally through the diag channel and,
/// when [`DIAG_OUT_ENV`] named a file at construction,
/// writes the summary table there, in the exact
/// [`render_summary`] format. Diagnostics never fail the run
/// they observe: a failed drain or a failed write is counted
/// and reported on the channel, and generation proceeds.
///
/// Capture: an accumulator whose mode is
/// [`DiagMode::Capture`] additionally holds the run manifest
/// and the per-forward fingerprint records (the capture
/// section below); `write_report` then also writes the two
/// sidecar files beside the summary table. A timing-mode
/// accumulator holds neither, and its finish is unchanged.
#[derive(Debug)]
pub struct DiagAccumulator {
    out_path: Option<PathBuf>,
    mode: DiagMode,
    by_kernel: BTreeMap<&'static str, KernelSummary>,
    total_records: u64,
    total_ns: u64,
    drain_failures: u64,
    finished: bool,
    manifest: Option<RunManifest>,
    pending_model: Option<ModelFileInfo>,
    capture_records: Vec<CaptureRecord>,
}

impl DiagAccumulator {
    /// An accumulator reporting to `out_path` (or to the diag
    /// channel only, when `None`). Mode is
    /// [`DiagMode::Off`]: folds and reports timing records
    /// exactly as before, and records no capture data.
    #[must_use]
    pub fn new(out_path: Option<PathBuf>) -> DiagAccumulator {
        DiagAccumulator::with_mode(out_path, DiagMode::Off)
    }

    /// An accumulator reporting to `out_path` under an
    /// explicit diag `mode`. The mode only gates the capture
    /// level (manifest, forward fingerprints, sidecars); the
    /// timing fold is identical in every mode.
    #[must_use]
    pub fn with_mode(out_path: Option<PathBuf>, mode: DiagMode) -> DiagAccumulator {
        DiagAccumulator {
            out_path,
            mode,
            by_kernel: BTreeMap::new(),
            total_records: 0,
            total_ns: 0,
            drain_failures: 0,
            finished: false,
            manifest: None,
            pending_model: None,
            capture_records: Vec::new(),
        }
    }

    /// An accumulator whose report file is whatever
    /// [`DIAG_OUT_ENV`] names in the current environment
    /// (unset: channel only), under the mode [`DIAG_ENV`]
    /// requests. Both are read once, at construction, so a
    /// run's destination and level cannot move mid-flight.
    #[must_use]
    pub fn from_env() -> DiagAccumulator {
        DiagAccumulator::with_mode(
            std::env::var_os(DIAG_OUT_ENV).map(PathBuf::from),
            DiagMode::from_env(),
        )
    }

    /// The diag mode this accumulator runs at.
    #[must_use]
    pub fn mode(&self) -> DiagMode {
        self.mode
    }

    /// Whether this accumulator records capture data
    /// (manifest and forward fingerprints).
    #[must_use]
    pub fn is_capture(&self) -> bool {
        self.mode.is_capture()
    }

    /// The report file this accumulator writes at finish, if
    /// it has one.
    #[must_use]
    pub fn out_path(&self) -> Option<&std::path::Path> {
        self.out_path.as_deref()
    }

    /// Dispatch records folded so far.
    #[must_use]
    pub fn total_records(&self) -> u64 {
        self.total_records
    }

    /// Drains that failed (and were reported on the channel)
    /// so far.
    #[must_use]
    pub fn drain_failures(&self) -> u64 {
        self.drain_failures
    }

    /// Names of the kernels folded so far, in name order.
    #[must_use]
    pub fn kernel_names(&self) -> Vec<&'static str> {
        self.by_kernel.keys().copied().collect()
    }

    /// Fold resolved records into the running aggregates:
    /// the same per-kernel sums [`summarize`] computes, kept
    /// incrementally.
    fn fold_records(&mut self, records: &[DispatchRecord]) {
        for r in records {
            self.total_records += 1;
            self.total_ns += r.duration_ns;
            let entry = self
                .by_kernel
                .entry(r.kernel)
                .or_insert_with(|| KernelSummary {
                    kernel: r.kernel.to_string(),
                    count: 0,
                    total_ns: 0,
                    total_bytes: 0,
                });
            entry.count += 1;
            entry.total_ns += r.duration_ns;
            entry.total_bytes += r.weight_bytes;
        }
    }

    /// Drain the recorder once and fold whatever it resolved
    /// into the running aggregates. Returns how many records
    /// the drain resolved. A failed drain is counted and
    /// reported through the diag channel; it is never
    /// propagated, because diagnostics must not fail the
    /// generation they observe.
    pub fn drain_and_fold(
        &mut self,
        recorder: &DiagRecorder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> usize {
        match recorder.drain(device, queue) {
            Ok(_) => {
                let records = recorder.take_records();
                let n = records.len();
                self.fold_records(&records);
                n
            }
            Err(e) => {
                self.drain_failures += 1;
                eprintln!("[wgpu-compute-diag] drain failed: {e}");
                0
            }
        }
    }

    /// The summary table over everything folded so far, in
    /// the exact [`render_summary`] format.
    #[must_use]
    pub fn render_csv(&self) -> String {
        let mut rows: Vec<KernelSummary> = self.by_kernel.values().cloned().collect();
        rows.sort_by(|a, b| b.total_ns.cmp(&a.total_ns).then(a.kernel.cmp(&b.kernel)));
        render_summaries(&rows, self.total_ns, self.total_records)
    }

    /// Finish the run's report, once: print the tally line
    /// through the diag channel and write the summary table
    /// to the report file when one is configured and any
    /// records were folded. Under capture, the manifest and
    /// fingerprint sidecars are written first (their own
    /// emptiness rule: a begun manifest or any capture record
    /// is enough), so a capture run leaves its diffable record
    /// even when it folded no dispatch timings. `dropped` is
    /// the recorder's lifetime drop count, passed in because
    /// the accumulator does not own the recorder. Later calls
    /// are no-ops. Returns whether this call performed the
    /// finish.
    pub fn write_report(&mut self, dropped: u64) -> bool {
        if self.finished {
            return false;
        }
        self.finished = true;
        self.write_capture_sidecars();
        if self.total_records == 0 && dropped == 0 && self.drain_failures == 0 {
            return true;
        }
        let failures = if self.drain_failures == 0 {
            String::new()
        } else {
            format!(", {} drain failures", self.drain_failures)
        };
        match &self.out_path {
            Some(path) if self.total_records > 0 => match std::fs::write(path, self.render_csv()) {
                Ok(()) => eprintln!(
                    "[wgpu-compute-diag] session report: {} dispatches across {} kernels, \
                         {dropped} dropped{failures}, report written to {}",
                    self.total_records,
                    self.by_kernel.len(),
                    path.display()
                ),
                Err(e) => eprintln!(
                    "[wgpu-compute-diag] session report: {} dispatches across {} kernels, \
                         {dropped} dropped{failures}, report write to {} failed: {e}",
                    self.total_records,
                    self.by_kernel.len(),
                    path.display()
                ),
            },
            _ => eprintln!(
                "[wgpu-compute-diag] session report: {} dispatches across {} kernels, \
                 {dropped} dropped{failures}, no report file ({DIAG_OUT_ENV} unset)",
                self.total_records,
                self.by_kernel.len()
            ),
        }
        true
    }
}

// ----------------------------------------------------------------
// Capture level: data fingerprints and the run manifest.
//
// Timing answers "how long did each dispatch take". Capture
// answers the question timing cannot: "what did the run
// actually compute". A session running at the capture level
// emits a run manifest when it starts (what binary, what
// adapter and driver, what the resolved configuration was,
// which model file, what session shape) and records a
// fingerprint of every completed forward's output logits, so
// a wrong-data report stops being an argument about whose
// numbers look right: two capture runs diff mechanically, and
// the diff names the first forward where they diverge.
// ----------------------------------------------------------------

/// Word-wise FNV-1a 64: the fingerprint primitive for the
/// whole capture level. The byte stream is folded eight
/// bytes at a time: each little-endian `u64` word is XORed
/// into the state, which is then multiplied by the FNV
/// prime; a final partial word (the stream length mod 8)
/// folds byte by byte the classic way. Word folding is what
/// makes the hash affordable on a forward's logits: it costs
/// one multiply per eight bytes (measured in the bench
/// note), where the naive per-byte loop measured 35 times
/// slower and would have taxed every decode step. The
/// result is a function of the exact byte sequence only,
/// never of how the feeds were chunked: a staging buffer
/// holds any partial word until it completes or the stream
/// ends.
///
/// Chosen because it is stable across runs, processes, and
/// platforms (a fixed offset basis and prime over
/// little-endian words), needs no dependency, and runs over
/// a forward's logits on the CPU after the readback that
/// already happened. It is a change detector, not a
/// cryptographic hash: a collision needs the same 64-bit
/// state after the same length prefix, and any one-element
/// change avalanches through every later word.
#[derive(Clone, Debug)]
pub struct Fnv1a64 {
    state: u64,
    pending: [u8; 8],
    pending_len: usize,
}

impl Fnv1a64 {
    /// The FNV-1a 64 offset basis.
    pub const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    /// The FNV-1a 64 prime.
    pub const PRIME: u64 = 0x0000_0100_0000_01b3;

    /// A fresh hasher at the offset basis.
    #[must_use]
    pub fn new() -> Fnv1a64 {
        Fnv1a64 {
            state: Self::OFFSET,
            pending: [0; 8],
            pending_len: 0,
        }
    }

    fn fold_word(&mut self, word: u64) {
        self.state ^= word;
        self.state = self.state.wrapping_mul(Self::PRIME);
    }

    /// Fold raw bytes into the stream, in order. Bytes are
    /// staged until a full 8-byte word exists; the staging
    /// is what makes the fingerprint independent of feed
    /// chunk sizes.
    pub fn update_bytes(&mut self, mut bytes: &[u8]) {
        if self.pending_len > 0 {
            let take = (8 - self.pending_len).min(bytes.len());
            self.pending[self.pending_len..self.pending_len + take].copy_from_slice(&bytes[..take]);
            self.pending_len += take;
            bytes = &bytes[take..];
            if self.pending_len == 8 {
                let word = u64::from_le_bytes(self.pending);
                self.fold_word(word);
                self.pending_len = 0;
            }
        }
        let mut words = bytes.chunks_exact(8);
        for w in &mut words {
            self.fold_word(u64::from_le_bytes(w.try_into().expect("8-byte chunk")));
        }
        let rest = words.remainder();
        self.pending[..rest.len()].copy_from_slice(rest);
        self.pending_len = rest.len();
    }

    /// Fold one `u64` as little-endian bytes.
    pub fn update_u64(&mut self, value: u64) {
        self.update_bytes(&value.to_le_bytes());
    }

    /// Fold `f32` values by their bit patterns (the
    /// little-endian `to_bits` byte stream): two runs agree
    /// bitwise or the fingerprint differs, which is exactly
    /// the comparison a wrong-data hunt needs.
    pub fn update_f32_slice(&mut self, values: &[f32]) {
        if self.pending_len == 0 {
            // Fast path: pairs of f32 are exactly one word
            // of the byte stream; fold them directly.
            let mut pairs = values.chunks_exact(2);
            for p in &mut pairs {
                self.fold_word(u64::from(p[0].to_bits()) | u64::from(p[1].to_bits()) << 32);
            }
            if let [last] = pairs.remainder() {
                self.update_bytes(&last.to_bits().to_le_bytes());
            }
        } else {
            for v in values {
                self.update_bytes(&v.to_bits().to_le_bytes());
            }
        }
    }

    /// Fold `u32` values as little-endian bytes.
    pub fn update_u32_slice(&mut self, values: &[u32]) {
        if self.pending_len == 0 {
            let mut pairs = values.chunks_exact(2);
            for p in &mut pairs {
                self.fold_word(u64::from(p[0]) | u64::from(p[1]) << 32);
            }
            if let [last] = pairs.remainder() {
                self.update_bytes(&last.to_le_bytes());
            }
        } else {
            for v in values {
                self.update_bytes(&v.to_le_bytes());
            }
        }
    }

    /// The hash of the stream as it stands: the folded words
    /// plus any staged partial word folded byte by byte.
    #[must_use]
    pub fn finish(&self) -> u64 {
        let mut state = self.state;
        for &b in &self.pending[..self.pending_len] {
            state ^= u64::from(b);
            state = state.wrapping_mul(Self::PRIME);
        }
        state
    }
}

impl Default for Fnv1a64 {
    fn default() -> Fnv1a64 {
        Fnv1a64::new()
    }
}

/// Fingerprint a slice of `f32` values: FNV-1a over the
/// element count (as `u64` LE) followed by each value's bit
/// pattern. The length prefix means a prefix of a longer
/// vector can never collide with the shorter vector itself.
#[must_use]
pub fn fingerprint_f32(values: &[f32]) -> u64 {
    let mut h = Fnv1a64::new();
    h.update_u64(values.len() as u64);
    h.update_f32_slice(values);
    h.finish()
}

/// Fingerprint a slice of `u32` values (token ids): FNV-1a
/// over the element count followed by each id's LE bytes.
#[must_use]
pub fn fingerprint_u32(values: &[u32]) -> u64 {
    let mut h = Fnv1a64::new();
    h.update_u64(values.len() as u64);
    h.update_u32_slice(values);
    h.finish()
}

/// Render a fingerprint as 16 lowercase hex digits, the form
/// every capture record and manifest carries.
#[must_use]
pub fn fingerprint_hex(hash: u64) -> String {
    format!("{hash:016x}")
}

/// The index of the largest value in `values`, lowest index
/// on an exact tie: the host twin of the device argmax rule
/// `forward_argmax` documents, so a forward's recorded argmax
/// is the token the greedy path would have picked. `None` on
/// an empty slice. NaN never wins a comparison (every `>`
/// against it is false), so a NaN can only be reported when
/// it sits at index 0 and nothing later is strictly greater.
#[must_use]
pub fn argmax_of(values: &[f32]) -> Option<u32> {
    if values.is_empty() {
        return None;
    }
    let mut best = 0usize;
    for (i, &v) in values.iter().enumerate().skip(1) {
        if v > values[best] {
            best = i;
        }
    }
    Some(best as u32)
}

/// The model-file fingerprint method, stated wherever the
/// fingerprint is recorded: word-wise FNV-1a 64
/// ([`Fnv1a64`], hence the `w` in the method string) over
/// the file size as `u64` LE, then the whole file when it
/// is at most 2 MiB, else the first 1 MiB followed by the
/// last 1 MiB. Two samples at the file's two ends plus its
/// exact size catch a swapped, truncated, or re-quantized
/// file at open cost, not load cost. The file must not
/// change while it is being fingerprinted or while a run
/// uses it (the same immutability the mmap loader already
/// assumes).
pub const MODEL_FINGERPRINT_METHOD: &str =
    "fnv1a64w(size_u64le || whole_file_if_size<=2MiB_else_first_1MiB || last_1MiB)";

/// One model file as the manifest records it: where it is,
/// how big it is, and its content fingerprint under
/// [`MODEL_FINGERPRINT_METHOD`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelFileInfo {
    /// The path as given (what the run was pointed at).
    pub path: String,
    /// File size in bytes, from its metadata.
    pub size_bytes: u64,
    /// The content fingerprint, 16 lowercase hex digits.
    pub fingerprint: String,
    /// The fingerprint method (always
    /// [`MODEL_FINGERPRINT_METHOD`] today; carried in the
    /// record so a future method change is visible in old
    /// manifests instead of silently incomparable).
    pub method: String,
}

/// Fingerprint the model file at `path` under
/// [`MODEL_FINGERPRINT_METHOD`], reading at most 2 MiB of it
/// however large it is.
///
/// # Errors
///
/// The underlying `io` error when the file cannot be opened,
/// stated, or read.
pub fn fingerprint_model_file(path: &Path) -> std::io::Result<ModelFileInfo> {
    use std::io::{Read, Seek, SeekFrom};
    const MIB: u64 = 1024 * 1024;
    let mut file = std::fs::File::open(path)?;
    let size = file.metadata()?.len();
    let mut h = Fnv1a64::new();
    h.update_u64(size);
    let mut chunk = vec![0u8; 64 * 1024];
    let mut feed = |file: &mut std::fs::File, want: u64, h: &mut Fnv1a64| -> std::io::Result<()> {
        let mut left = want;
        while left > 0 {
            let take = left.min(chunk.len() as u64) as usize;
            let n = file.read(&mut chunk[..take])?;
            if n == 0 {
                break;
            }
            h.update_bytes(&chunk[..n]);
            left -= n as u64;
        }
        Ok(())
    };
    if size <= 2 * MIB {
        feed(&mut file, size, &mut h)?;
    } else {
        feed(&mut file, MIB, &mut h)?;
        file.seek(SeekFrom::Start(size - MIB))?;
        feed(&mut file, MIB, &mut h)?;
    }
    Ok(ModelFileInfo {
        path: path.display().to_string(),
        size_bytes: size,
        fingerprint: fingerprint_hex(h.finish()),
        method: MODEL_FINGERPRINT_METHOD.to_string(),
    })
}

/// The process-wide model-file registration: the one line a
/// driving binary calls after it opens its model file, so
/// sessions built later in the same process stamp their
/// manifests with the file without the session API having to
/// grow a path parameter (a session is built from weights,
/// and weights deliberately carry no path).
static MODEL_FILE_REGISTRY: std::sync::OnceLock<Mutex<Option<ModelFileInfo>>> =
    std::sync::OnceLock::new();

fn model_file_registry() -> &'static Mutex<Option<ModelFileInfo>> {
    MODEL_FILE_REGISTRY.get_or_init(|| Mutex::new(None))
}

/// Fingerprint the file at `path` and register it as this
/// process's model file, replacing any earlier registration.
/// Returns the recorded info.
///
/// # Errors
///
/// The `io` error from [`fingerprint_model_file`]; a failed
/// fingerprint registers nothing.
pub fn register_model_file(path: &Path) -> std::io::Result<ModelFileInfo> {
    let info = fingerprint_model_file(path)?;
    *model_file_registry().lock().unwrap() = Some(info.clone());
    Ok(info)
}

/// The registered model file, when a binary registered one.
#[must_use]
pub fn registered_model_file() -> Option<ModelFileInfo> {
    model_file_registry().lock().unwrap().clone()
}

/// One capture record: a single thing a capture run computed,
/// in run order. Forwards carry their output logits'
/// fingerprint; rounds carry the draft's proposed ids'
/// fingerprint plus the accepted count. Records are the unit
/// the diff utility compares.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureRecord {
    /// Record index within the run, from 0, in record order.
    pub seq: u64,
    /// Record kind: one of the `KIND_*` constants.
    pub kind: &'static str,
    /// Values fingerprinted: the logits length for forward
    /// kinds, 2 for an argmax record (the id/logit pair), the
    /// proposed count for a round.
    pub values: u64,
    /// The data fingerprint, 16 lowercase hex digits.
    pub hash: String,
    /// The forward's greedy token, when the record carries
    /// one (`forward` and `argmax` records).
    pub argmax: Option<u32>,
    /// Draft tokens accepted this round (`round` records).
    pub accepted: Option<u64>,
}

impl CaptureRecord {
    /// A plain forward's finished logits (`forward`).
    pub const KIND_FORWARD: &'static str = "forward";
    /// A device-argmax forward's winner (`forward_argmax`).
    pub const KIND_ARGMAX: &'static str = "argmax";
    /// A block forward's finished logits, every row
    /// (`forward_block_capturing`).
    pub const KIND_BLOCK: &'static str = "block";
    /// One speculative round: proposed ids plus acceptance.
    pub const KIND_ROUND: &'static str = "round";

    /// This record as one capture CSV row (no trailing
    /// newline); empty fields for data the kind does not
    /// carry.
    #[must_use]
    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{},{},{}",
            self.seq,
            self.kind,
            self.values,
            self.hash,
            self.argmax.map_or_else(String::new, |v| v.to_string()),
            self.accepted.map_or_else(String::new, |v| v.to_string()),
        )
    }
}

/// The capture CSV header line; a capture sidecar is this
/// line plus one [`CaptureRecord::to_csv_row`] per record.
pub const CAPTURE_CSV_HEADER: &str = "seq,kind,values,hash,argmax,accepted";

/// Render capture records as the capture CSV document.
#[must_use]
pub fn render_capture_csv(records: &[CaptureRecord]) -> String {
    let mut out = String::from(CAPTURE_CSV_HEADER);
    out.push('\n');
    for r in records {
        out.push_str(&r.to_csv_row());
        out.push('\n');
    }
    out
}

/// Parse a capture CSV document back into records, the
/// exact inverse of [`render_capture_csv`].
///
/// # Errors
///
/// A `String` naming the line when the header differs, a
/// row has the wrong field count, a kind is not one of the
/// four record kinds, or a number does not parse.
pub fn parse_capture_csv(text: &str) -> std::result::Result<Vec<CaptureRecord>, String> {
    let mut lines = text.lines();
    let header = lines.next().ok_or("empty capture csv")?;
    if header != CAPTURE_CSV_HEADER {
        return Err(format!("bad capture csv header: {header:?}"));
    }
    let mut records = Vec::new();
    for (i, line) in lines.enumerate() {
        if line.is_empty() {
            continue;
        }
        let lineno = i + 2;
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 6 {
            return Err(format!(
                "capture csv line {lineno}: {} fields, want 6",
                fields.len()
            ));
        }
        let bad = |what: &str, s: &str| format!("capture csv line {lineno}: bad {what} {s:?}");
        let seq = fields[0]
            .parse::<u64>()
            .map_err(|_| bad("seq", fields[0]))?;
        let kind = match fields[1] {
            "forward" => CaptureRecord::KIND_FORWARD,
            "argmax" => CaptureRecord::KIND_ARGMAX,
            "block" => CaptureRecord::KIND_BLOCK,
            "round" => CaptureRecord::KIND_ROUND,
            other => return Err(format!("capture csv line {lineno}: unknown kind {other:?}")),
        };
        let values = fields[2]
            .parse::<u64>()
            .map_err(|_| bad("values", fields[2]))?;
        let argmax = if fields[4].is_empty() {
            None
        } else {
            Some(
                fields[4]
                    .parse::<u32>()
                    .map_err(|_| bad("argmax", fields[4]))?,
            )
        };
        let accepted = if fields[5].is_empty() {
            None
        } else {
            Some(
                fields[5]
                    .parse::<u64>()
                    .map_err(|_| bad("accepted", fields[5]))?,
            )
        };
        records.push(CaptureRecord {
            seq,
            kind,
            values,
            hash: fields[3].to_string(),
            argmax,
            accepted,
        });
    }
    Ok(records)
}

/// The first point where two capture record sequences
/// disagree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureDivergence {
    /// Sequence number of the diverging record.
    pub seq: u64,
    /// The field that differs: `kind`, `values`, `hash`,
    /// `argmax`, `accepted`, or `presence` (one run has a
    /// record the other lacks).
    pub field: &'static str,
    /// The first run's record at the divergence, when it has
    /// one.
    pub a: Option<CaptureRecord>,
    /// The second run's record at the divergence, when it
    /// has one.
    pub b: Option<CaptureRecord>,
}

/// The outcome of diffing two capture record sequences.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureDiff {
    /// Records compared before the outcome was decided: all
    /// of them when identical, the diverging index + 1 when
    /// not.
    pub compared: u64,
    /// The first divergence, or `None` when the sequences
    /// agree record for record.
    pub divergence: Option<CaptureDivergence>,
}

impl CaptureDiff {
    /// Whether the two sequences are record-for-record equal.
    #[must_use]
    pub fn identical(&self) -> bool {
        self.divergence.is_none()
    }
}

/// Diff two capture record sequences position by position:
/// the first record whose kind, value count, fingerprint,
/// argmax, or accepted count differs is the divergence, and
/// a length mismatch diverges at the first record only one
/// run has. This is the mechanical answer to "where did the
/// wrong run go wrong": everything before the named step
/// provably computed the same values.
#[must_use]
pub fn diff_capture_records(a: &[CaptureRecord], b: &[CaptureRecord]) -> CaptureDiff {
    let shared = a.len().min(b.len());
    for i in 0..shared {
        let (x, y) = (&a[i], &b[i]);
        let field = if x.kind != y.kind {
            Some("kind")
        } else if x.values != y.values {
            Some("values")
        } else if x.hash != y.hash {
            Some("hash")
        } else if x.argmax != y.argmax {
            Some("argmax")
        } else if x.accepted != y.accepted {
            Some("accepted")
        } else {
            None
        };
        if let Some(field) = field {
            return CaptureDiff {
                compared: (i + 1) as u64,
                divergence: Some(CaptureDivergence {
                    seq: x.seq,
                    field,
                    a: Some(x.clone()),
                    b: Some(y.clone()),
                }),
            };
        }
    }
    if a.len() != b.len() {
        let (longer_a, extra) = if a.len() > b.len() {
            (true, a[shared].clone())
        } else {
            (false, b[shared].clone())
        };
        return CaptureDiff {
            compared: (shared + 1) as u64,
            divergence: Some(CaptureDivergence {
                seq: extra.seq,
                field: "presence",
                a: longer_a.then_some(extra.clone()),
                b: (!longer_a).then_some(extra),
            }),
        };
    }
    CaptureDiff {
        compared: shared as u64,
        divergence: None,
    }
}

/// Escape a string for the manifest JSON: quotes, backslash,
/// and control characters; everything else passes through
/// (the manifest is UTF-8 text, read by humans and by
/// [`manifest_fields`]).
fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn push_json_string(out: &mut String, value: &str) {
    out.push('"');
    out.push_str(&json_escape(value));
    out.push('"');
}

fn push_json_pairs(out: &mut String, pairs: &[(String, String)]) {
    out.push('{');
    for (i, (k, v)) in pairs.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        push_json_string(out, k);
        out.push(':');
        push_json_string(out, v);
    }
    out.push('}');
}

/// This crate's version: the build half of the manifest
/// every run can state without help.
#[must_use]
pub fn build_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// The source revision the binary was built from, when the
/// build exported `ITAK_GIT_HASH`; `"unknown"` otherwise.
/// Never guessed: a manifest that cannot know says so.
#[must_use]
pub fn build_git() -> &'static str {
    option_env!("ITAK_GIT_HASH").unwrap_or("unknown")
}

/// A run's manifest: everything a capture run states about
/// itself when its session starts, so a later diff can first
/// answer "were these even the same run" before it compares
/// a single fingerprint. Rendered as one JSON document
/// ([`RunManifest::render_json`]), printed on the diag
/// channel at session start and written to the
/// `.manifest.json` sidecar at finish.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RunManifest {
    /// The diag mode the run executed at (`capture`).
    pub diag_mode: String,
    /// Crate version ([`build_version`]).
    pub build_version: String,
    /// Source revision ([`build_git`]).
    pub git: String,
    /// Adapter name as the driver reported it.
    pub adapter_name: String,
    /// Adapter backend (`Vulkan` on the GPU path).
    pub adapter_backend: String,
    /// Adapter device type classification.
    pub adapter_device_type: String,
    /// Driver name string from the adapter.
    pub adapter_driver: String,
    /// Driver info string from the adapter.
    pub adapter_driver_info: String,
    /// The resolved configuration, as ordered (name, value)
    /// pairs: every `WGPU_COMPUTE_*` value the run actually
    /// resolved, in its resolved form, not the raw env text.
    pub resolved_config: Vec<(String, String)>,
    /// The model file, when one was noted or registered.
    pub model: Option<ModelFileInfo>,
    /// The session's own shape, as ordered (name, value)
    /// pairs (kind, layer count, capacities, as applicable).
    pub session: Vec<(String, String)>,
}

impl RunManifest {
    /// Render the manifest as one compact JSON document.
    /// Field order is fixed as written here, so two renders
    /// of the same manifest are byte-identical.
    #[must_use]
    pub fn render_json(&self) -> String {
        let mut out = String::new();
        out.push_str("{\"format\":\"wgpu-compute-diag-manifest\",\"format_version\":1");
        out.push_str(",\"diag_mode\":");
        push_json_string(&mut out, &self.diag_mode);
        out.push_str(",\"build\":{\"version\":");
        push_json_string(&mut out, &self.build_version);
        out.push_str(",\"git\":");
        push_json_string(&mut out, &self.git);
        out.push_str("},\"adapter\":{\"name\":");
        push_json_string(&mut out, &self.adapter_name);
        out.push_str(",\"backend\":");
        push_json_string(&mut out, &self.adapter_backend);
        out.push_str(",\"device_type\":");
        push_json_string(&mut out, &self.adapter_device_type);
        out.push_str(",\"driver\":");
        push_json_string(&mut out, &self.adapter_driver);
        out.push_str(",\"driver_info\":");
        push_json_string(&mut out, &self.adapter_driver_info);
        out.push_str("},\"resolved_config\":");
        push_json_pairs(&mut out, &self.resolved_config);
        out.push_str(",\"model\":");
        match &self.model {
            None => out.push_str("null"),
            Some(m) => {
                out.push_str("{\"path\":");
                push_json_string(&mut out, &m.path);
                out.push_str(",\"size_bytes\":");
                out.push_str(&m.size_bytes.to_string());
                out.push_str(",\"fingerprint\":");
                push_json_string(&mut out, &m.fingerprint);
                out.push_str(",\"fingerprint_method\":");
                push_json_string(&mut out, &m.method);
                out.push('}');
            }
        }
        out.push_str(",\"session\":");
        push_json_pairs(&mut out, &self.session);
        out.push('}');
        out
    }
}

/// Extract a manifest JSON document's scalar leaves as
/// (dotted path, value) pairs, sorted by path: strings
/// unescaped, numbers and literals as written, a `null`
/// object member as the value `"null"`. Deliberately a
/// small reader for the manifest shape [`RunManifest::render_json`]
/// emits (objects of scalars, one level of nesting), not a
/// general JSON parser: it stops at the first malformation
/// and returns what it has.
#[must_use]
pub fn manifest_fields(json: &str) -> Vec<(String, String)> {
    let bytes = json.as_bytes();
    let mut pos = 0usize;
    let mut out = Vec::new();
    manifest_skip_ws(bytes, &mut pos);
    if pos < bytes.len() && bytes[pos] == b'{' {
        manifest_object(bytes, &mut pos, "", &mut out);
    }
    out.sort();
    out
}

/// Compare two manifests' field sets ([`manifest_fields`]
/// output): one line per field that differs, is missing, or
/// is extra, sorted by field path. An empty result means the
/// manifests state the same run.
#[must_use]
pub fn diff_manifest_fields(a: &[(String, String)], b: &[(String, String)]) -> Vec<String> {
    let am: BTreeMap<&str, &str> = a.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    let bm: BTreeMap<&str, &str> = b.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    let mut diffs = Vec::new();
    for (k, av) in &am {
        match bm.get(k) {
            None => diffs.push(format!("{k}: only in first ({av})")),
            Some(bv) if bv != av => diffs.push(format!("{k}: {av} != {bv}")),
            _ => {}
        }
    }
    for (k, bv) in &bm {
        if !am.contains_key(k) {
            diffs.push(format!("{k}: only in second ({bv})"));
        }
    }
    diffs
}

fn manifest_skip_ws(bytes: &[u8], pos: &mut usize) {
    while *pos < bytes.len() && bytes[*pos].is_ascii_whitespace() {
        *pos += 1;
    }
}

/// Read one JSON string at `pos` (which must sit on its
/// opening quote), returning its unescaped contents and
/// advancing past the closing quote. `None` on malformation.
fn manifest_string(bytes: &[u8], pos: &mut usize) -> Option<String> {
    if *pos >= bytes.len() || bytes[*pos] != b'"' {
        return None;
    }
    *pos += 1;
    let mut out = String::new();
    while *pos < bytes.len() {
        let c = bytes[*pos];
        *pos += 1;
        match c {
            b'"' => return Some(out),
            b'\\' => {
                if *pos >= bytes.len() {
                    return None;
                }
                let e = bytes[*pos];
                *pos += 1;
                match e {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000C}'),
                    b'u' => {
                        if *pos + 4 > bytes.len() {
                            return None;
                        }
                        let hex = std::str::from_utf8(&bytes[*pos..*pos + 4]).ok()?;
                        let code = u32::from_str_radix(hex, 16).ok()?;
                        out.push(char::from_u32(code).unwrap_or('?'));
                        *pos += 4;
                    }
                    _ => return None,
                }
            }
            _ => {
                // Multi-byte UTF-8 passes through byte by
                // byte; the buffer is valid UTF-8 on both
                // sides of the escape handling, so collect
                // the raw byte and rebuild at the end is not
                // an option mid-string. Push the byte's char
                // via a one-byte re-decode: bytes >= 0x80 are
                // continuation/lead bytes of the original
                // &str, so decode the full sequence here.
                if c < 0x80 {
                    out.push(c as char);
                } else {
                    let width = if c >= 0xF0 {
                        4
                    } else if c >= 0xE0 {
                        3
                    } else {
                        2
                    };
                    let start = *pos - 1;
                    let end = (start + width).min(bytes.len());
                    match std::str::from_utf8(&bytes[start..end]) {
                        Ok(s) => {
                            out.push_str(s);
                            *pos = end;
                        }
                        Err(_) => return None,
                    }
                }
            }
        }
    }
    None
}

/// Parse one JSON value at `pos`, recording scalar leaves
/// under the dotted `path`. Objects recurse; anything else
/// scalar is recorded as written.
fn manifest_value(bytes: &[u8], pos: &mut usize, path: &str, out: &mut Vec<(String, String)>) {
    manifest_skip_ws(bytes, pos);
    if *pos >= bytes.len() {
        return;
    }
    match bytes[*pos] {
        b'{' => manifest_object(bytes, pos, path, out),
        b'"' => {
            if let Some(s) = manifest_string(bytes, pos) {
                out.push((path.to_string(), s));
            }
        }
        b'[' => {
            // The manifest shape carries no arrays; skip a
            // balanced one without recording it.
            let mut depth = 0usize;
            while *pos < bytes.len() {
                let c = bytes[*pos];
                *pos += 1;
                match c {
                    b'[' => depth += 1,
                    b']' => {
                        depth = depth.saturating_sub(1);
                        if depth == 0 {
                            break;
                        }
                    }
                    b'"' => {
                        *pos -= 1;
                        let _ = manifest_string(bytes, pos);
                    }
                    _ => {}
                }
            }
        }
        _ => {
            let start = *pos;
            while *pos < bytes.len()
                && !matches!(bytes[*pos], b',' | b'}' | b']')
                && !bytes[*pos].is_ascii_whitespace()
            {
                *pos += 1;
            }
            if *pos > start {
                if let Ok(s) = std::str::from_utf8(&bytes[start..*pos]) {
                    out.push((path.to_string(), s.to_string()));
                }
            }
        }
    }
}

fn manifest_object(bytes: &[u8], pos: &mut usize, prefix: &str, out: &mut Vec<(String, String)>) {
    // Caller has verified the current byte is '{'.
    *pos += 1;
    loop {
        manifest_skip_ws(bytes, pos);
        if *pos >= bytes.len() {
            return;
        }
        if bytes[*pos] == b'}' {
            *pos += 1;
            return;
        }
        let Some(key) = manifest_string(bytes, pos) else {
            return;
        };
        manifest_skip_ws(bytes, pos);
        if *pos >= bytes.len() || bytes[*pos] != b':' {
            return;
        }
        *pos += 1;
        let path = if prefix.is_empty() {
            key
        } else {
            format!("{prefix}.{key}")
        };
        manifest_value(bytes, pos, &path, out);
        manifest_skip_ws(bytes, pos);
        if *pos < bytes.len() && bytes[*pos] == b',' {
            *pos += 1;
            continue;
        }
        if *pos < bytes.len() && bytes[*pos] == b'}' {
            *pos += 1;
        }
        return;
    }
}

/// The capture half of [`DiagAccumulator`]: manifest,
/// forward fingerprints, and the sidecar finish. Every entry
/// point is a no-op (returning `None` where it returns a
/// record) unless the accumulator's mode is
/// [`DiagMode::Capture`], so timing and off runs pay one mode
/// check per forward and nothing else.
impl DiagAccumulator {
    /// Begin the run manifest at session start: restamp its
    /// mode field from this accumulator's own mode (the
    /// accumulator is the authority on the level it runs
    /// at), fill its model section from a model file noted
    /// earlier ([`DiagAccumulator::note_model_file`]) or,
    /// failing that, the process registration
    /// ([`registered_model_file`]), then print it on the diag
    /// channel. The stored manifest is what the finish writes
    /// to the `.manifest.json` sidecar. A no-op outside
    /// capture mode.
    pub fn begin_manifest(&mut self, mut manifest: RunManifest) {
        if !self.is_capture() {
            return;
        }
        manifest.diag_mode = self.mode.name().to_string();
        if manifest.model.is_none() {
            manifest.model = self.pending_model.take().or_else(registered_model_file);
        }
        eprintln!("[wgpu-compute-diag] manifest: {}", manifest.render_json());
        self.manifest = Some(manifest);
    }

    /// Fingerprint the model file at `path` into the
    /// manifest: into the begun manifest when there is one,
    /// held as pending for [`DiagAccumulator::begin_manifest`]
    /// when there is not yet.
    ///
    /// # Errors
    ///
    /// The `io` error from [`fingerprint_model_file`].
    pub fn note_model_file(&mut self, path: &Path) -> std::io::Result<()> {
        let info = fingerprint_model_file(path)?;
        match &mut self.manifest {
            Some(manifest) => manifest.model = Some(info),
            None => self.pending_model = Some(info),
        }
        Ok(())
    }

    /// The begun manifest rendered as JSON, when one was
    /// begun.
    #[must_use]
    pub fn manifest_json(&self) -> Option<String> {
        self.manifest.as_ref().map(RunManifest::render_json)
    }

    /// The capture records recorded so far, in order.
    #[must_use]
    pub fn capture_records(&self) -> &[CaptureRecord] {
        &self.capture_records
    }

    /// How many capture records were recorded so far.
    #[must_use]
    pub fn capture_record_count(&self) -> usize {
        self.capture_records.len()
    }

    fn push_capture(
        &mut self,
        kind: &'static str,
        values: u64,
        hash: String,
        argmax: Option<u32>,
        accepted: Option<u64>,
    ) -> CaptureRecord {
        let record = CaptureRecord {
            seq: self.capture_records.len() as u64,
            kind,
            values,
            hash,
            argmax,
            accepted,
        };
        eprintln!(
            "[wgpu-compute-diag] capture #{} {} values={} hash={} argmax={} accepted={}",
            record.seq,
            record.kind,
            record.values,
            record.hash,
            record
                .argmax
                .map_or_else(|| "-".to_string(), |v| v.to_string()),
            record
                .accepted
                .map_or_else(|| "-".to_string(), |v| v.to_string()),
        );
        self.capture_records.push(record.clone());
        record
    }

    /// Record one completed forward's output logits: their
    /// fingerprint, and for a plain `forward` the greedy
    /// token they imply. `kind` is
    /// [`CaptureRecord::KIND_FORWARD`] or
    /// [`CaptureRecord::KIND_BLOCK`] (a block's rows carry no
    /// single argmax). The hash runs over the vector the
    /// caller already read back; capture adds no GPU work and
    /// no readback. Returns the record, or `None` outside
    /// capture mode.
    pub fn record_forward(&mut self, kind: &'static str, logits: &[f32]) -> Option<CaptureRecord> {
        if !self.is_capture() {
            return None;
        }
        let hash = fingerprint_hex(fingerprint_f32(logits));
        let argmax = if kind == CaptureRecord::KIND_FORWARD {
            argmax_of(logits)
        } else {
            None
        };
        Some(self.push_capture(kind, logits.len() as u64, hash, argmax, None))
    }

    /// Record a device-argmax forward's winner: the
    /// fingerprint covers the (id, logit) pair, the only
    /// output that forward reads back. Returns the record,
    /// or `None` outside capture mode.
    pub fn record_argmax(&mut self, id: u32, logit: f32) -> Option<CaptureRecord> {
        if !self.is_capture() {
            return None;
        }
        let mut h = Fnv1a64::new();
        h.update_u64(2);
        h.update_u32_slice(&[id]);
        h.update_f32_slice(&[logit]);
        Some(self.push_capture(
            CaptureRecord::KIND_ARGMAX,
            2,
            fingerprint_hex(h.finish()),
            Some(id),
            None,
        ))
    }

    /// Record one speculative round: the draft's proposed
    /// ids (fingerprinted) and how many of them the target
    /// accepted. Returns the record, or `None` outside
    /// capture mode.
    pub fn record_spec_round(
        &mut self,
        proposed: &[u32],
        accepted: usize,
    ) -> Option<CaptureRecord> {
        if !self.is_capture() {
            return None;
        }
        let hash = fingerprint_hex(fingerprint_u32(proposed));
        Some(self.push_capture(
            CaptureRecord::KIND_ROUND,
            proposed.len() as u64,
            hash,
            None,
            Some(accepted as u64),
        ))
    }

    /// Write the capture sidecars at finish (capture mode
    /// only): the manifest JSON beside the report path, and
    /// the capture CSV beside it. Runs when a manifest was
    /// begun or any capture record exists; failures are
    /// reported on the diag channel, never propagated, like
    /// every other diag write.
    fn write_capture_sidecars(&mut self) {
        if !self.is_capture() {
            return;
        }
        if self.manifest.is_none() && self.capture_records.is_empty() {
            return;
        }
        let Some(out) = self.out_path.clone() else {
            eprintln!(
                "[wgpu-compute-diag] capture report: {} records, manifest {}begun, \
                 no sidecar files ({DIAG_OUT_ENV} unset)",
                self.capture_records.len(),
                if self.manifest.is_some() { "" } else { "not " },
            );
            return;
        };
        let mut notes: Vec<String> = Vec::new();
        if let Some(manifest) = &self.manifest {
            let path = manifest_sidecar_path(&out);
            match std::fs::write(&path, format!("{}\n", manifest.render_json())) {
                Ok(()) => notes.push(format!("manifest written to {}", path.display())),
                Err(e) => notes.push(format!("manifest write to {} failed: {e}", path.display())),
            }
        }
        let path = capture_sidecar_path(&out);
        match std::fs::write(&path, render_capture_csv(&self.capture_records)) {
            Ok(()) => notes.push(format!("capture records written to {}", path.display())),
            Err(e) => notes.push(format!(
                "capture records write to {} failed: {e}",
                path.display()
            )),
        }
        eprintln!(
            "[wgpu-compute-diag] capture report: {} records; {}",
            self.capture_records.len(),
            notes.join("; ")
        );
    }
}

/// The sidecar path for a run's manifest: the report path
/// with `.manifest.json` appended (`report.csv` becomes
/// `report.csv.manifest.json`).
#[must_use]
pub fn manifest_sidecar_path(out: &Path) -> PathBuf {
    let mut name = out.as_os_str().to_owned();
    name.push(".manifest.json");
    PathBuf::from(name)
}

/// The sidecar path for a run's capture records: the report
/// path with `.capture.csv` appended.
#[must_use]
pub fn capture_sidecar_path(out: &Path) -> PathBuf {
    let mut name = out.as_os_str().to_owned();
    name.push(".capture.csv");
    PathBuf::from(name)
}

/// One dispatch's stamp, cloned into a prepared call at
/// prepare time: the recorder plus the call's meta. A call
/// carries `Option<DiagStamp>`; `None` (the default-off world)
/// makes `encode_into` byte-identical to the uninstrumented
/// path.
#[derive(Clone, Debug)]
pub struct DiagStamp {
    recorder: Arc<DiagRecorder>,
    meta: DispatchMeta,
}

impl DiagStamp {
    /// Create a stamp binding `meta` to `recorder`.
    #[must_use]
    pub fn new(recorder: Arc<DiagRecorder>, meta: DispatchMeta) -> DiagStamp {
        DiagStamp { recorder, meta }
    }

    /// Write the opening timestamp for this dispatch inside
    /// `pass`; returns the slot to hand to
    /// [`DiagStamp::end`], or `None` when the window is full.
    pub fn begin(&self, pass: &mut wgpu::ComputePass<'_>) -> Option<u32> {
        self.recorder.begin(pass)
    }

    /// Write the closing timestamp and queue the record.
    pub fn end(&self, pass: &mut wgpu::ComputePass<'_>, slot: u32) {
        self.recorder.end(pass, slot, self.meta.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(kernel: &'static str, bytes: u64, ns: u64) -> DispatchRecord {
        DispatchRecord {
            index: 0,
            kernel,
            rows: 1,
            cols: 1,
            quant: "",
            weight_bytes: bytes,
            shape: String::new(),
            duration_ns: ns,
        }
    }

    #[test]
    fn mode_parses_timing_and_capture_only() {
        assert_eq!(DiagMode::parse(Some("timing")), DiagMode::Timing);
        assert_eq!(DiagMode::parse(Some("capture")), DiagMode::Capture);
        assert_eq!(DiagMode::parse(Some("off")), DiagMode::Off);
        assert_eq!(DiagMode::parse(Some("TIMING")), DiagMode::Off);
        assert_eq!(DiagMode::parse(Some("Capture")), DiagMode::Off);
        assert_eq!(DiagMode::parse(Some("capture ")), DiagMode::Off);
        assert_eq!(DiagMode::parse(Some("garbage")), DiagMode::Off);
        assert_eq!(DiagMode::parse(Some("")), DiagMode::Off);
        assert_eq!(DiagMode::parse(None), DiagMode::Off);
        // Capture implies timing everywhere the two are
        // distinguished; off requests nothing.
        assert!(DiagMode::Capture.requests_timestamps());
        assert!(DiagMode::Timing.requests_timestamps());
        assert!(!DiagMode::Off.requests_timestamps());
        assert!(DiagMode::Capture.is_capture());
        assert!(!DiagMode::Timing.is_capture());
        assert_eq!(DiagMode::Capture.name(), "capture");
        assert_eq!(DiagMode::Timing.name(), "timing");
        assert_eq!(DiagMode::Off.name(), "off");
    }

    #[test]
    fn off_mode_never_wants_a_recorder() {
        // The off path must create no query resources, whatever
        // the device offers; timing and capture (which implies
        // timing) still require both timestamp features.
        let all = wgpu::Features::all();
        assert!(!DiagMode::Off.wants_recorder(all));
        assert!(!DiagMode::Timing.wants_recorder(wgpu::Features::empty()));
        assert!(!DiagMode::Timing.wants_recorder(wgpu::Features::TIMESTAMP_QUERY));
        assert!(DiagMode::Timing.wants_recorder(
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
        ));
        assert!(!DiagMode::Capture.wants_recorder(wgpu::Features::empty()));
        assert!(!DiagMode::Capture.wants_recorder(wgpu::Features::TIMESTAMP_QUERY));
        assert!(DiagMode::Capture.wants_recorder(
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
        ));
    }

    #[test]
    fn delta_converts_ticks_with_the_given_period() {
        assert_eq!(delta_ns(100, 250, 1.0), 150);
        assert_eq!(delta_ns(100, 250, 2.5), 375);
        // Equal stamps are a zero duration, and a wrap of the
        // 64-bit counter still yields the small true delta.
        assert_eq!(delta_ns(7, 7, 10.0), 0);
        assert_eq!(delta_ns(u64::MAX - 4, 5, 1.0), 10);
    }

    #[test]
    fn rate_math_is_decimal_and_never_nan() {
        // 16 GB in 20 ms is 800 GB/s.
        assert!((rate_gbps(16_000_000_000, 20_000_000) - 800.0).abs() < 1e-9);
        assert_eq!(rate_gbps(0, 20_000_000), 0.0);
        assert_eq!(rate_gbps(16_000_000_000, 0), 0.0);
    }

    #[test]
    fn summarize_aggregates_and_sorts_by_time() {
        let records = vec![
            record("k1_q4k_gemv", 100, 300),
            record("k4_rmsnorm", 0, 900),
            record("k1_q4k_gemv", 300, 100),
        ];
        let (rows, total) = summarize(&records);
        assert_eq!(total, 1300);
        assert_eq!(rows.len(), 2);
        // k4 leads on time despite zero bytes.
        assert_eq!(rows[0].kernel, "k4_rmsnorm");
        assert_eq!(rows[0].count, 1);
        assert_eq!(rows[1].kernel, "k1_q4k_gemv");
        assert_eq!(rows[1].count, 2);
        assert_eq!(rows[1].total_ns, 400);
        assert_eq!(rows[1].total_bytes, 400);
        // 400 bytes in 400 ns is 1 byte per ns: exactly 1 GB/s.
        assert!((rows[1].gbps() - 1.0).abs() < 1e-9);
        let table = render_summary(&records);
        assert!(table.contains("k1_q4k_gemv,2,"));
        assert!(table.contains("TOTAL,3,"));
    }

    #[test]
    fn query_sizing_never_exceeds_the_wgpu_cap() {
        // The recorder creates exactly one query set, sized by
        // window_query_count(); wgpu validates the count at
        // creation and a set above the cap fails on the real
        // device, so the count the code path uses must sit at
        // or under it.
        assert_eq!(MAX_QUERIES_PER_SET, 4096);
        assert!(window_query_count() <= MAX_QUERIES_PER_SET);
        assert_eq!(window_query_count(), MAX_DISPATCHES * 2);
        // The 2048-dispatch window floor is pinned beside the
        // cap in the compile-time asserts above.
    }

    #[test]
    fn full_window_accounting_at_capacity() {
        // At the real capacity the allocator hands out exactly
        // MAX_DISPATCHES slots, then drops and counts; the
        // range the drain resolves (used slots, two queries
        // each) is exactly the query set, never beyond it.
        let slots = SlotAllocator::new(MAX_DISPATCHES);
        for expected in 0..MAX_DISPATCHES {
            assert_eq!(slots.alloc(), Some(expected));
        }
        assert_eq!(slots.alloc(), None);
        assert_eq!(slots.used(), MAX_DISPATCHES);
        assert_eq!(slots.used() * 2, window_query_count());
        assert_eq!(slots.dropped(), 1);
    }

    #[test]
    fn slot_allocator_refuses_and_counts_past_capacity() {
        let slots = SlotAllocator::new(2);
        assert_eq!(slots.alloc(), Some(0));
        assert_eq!(slots.alloc(), Some(1));
        assert_eq!(slots.alloc(), None);
        assert_eq!(slots.alloc(), None);
        assert_eq!(slots.used(), 2);
        assert_eq!(slots.dropped(), 2);
        slots.reset();
        assert_eq!(slots.alloc(), Some(0));
        assert_eq!(slots.used(), 1);
    }

    #[test]
    fn k20_q6k_mmv_records_group_under_their_kernel_name() {
        // Registration for the K20 register-streaming
        // kernel's diag stamp (`k20_q6k_mmv`, stamped at
        // K20 prepare in
        // crates/wgpu-compute/src/kernels/k20_q6k_mmv.rs):
        // records carrying the name aggregate under it
        // and render, like every other kernel's, beside
        // the Q6_K GEMV they share a family with.
        let records = vec![
            record("k20_q6k_mmv", 212, 100),
            record("k20_q6k_mmv", 212, 300),
            record("k2_q6k_gemv", 212, 50),
        ];
        let (rows, total) = summarize(&records);
        assert_eq!(total, 450);
        assert_eq!(rows.len(), 2);
        let k20 = rows
            .iter()
            .find(|r| r.kernel == "k20_q6k_mmv")
            .expect("k20 records group under their kernel name");
        assert_eq!(k20.count, 2);
        assert_eq!(k20.total_ns, 400);
        assert_eq!(k20.total_bytes, 424);
        let table = render_summary(&records);
        assert!(table.contains("k20_q6k_mmv,2,"));
    }

    #[test]
    fn accumulator_folds_and_renders_like_the_collector() {
        // The session-side fold must aggregate exactly what
        // summarize() computes over the raw records, across
        // any number of drains: the rendered table over the
        // folded state is byte-identical to render_summary
        // over the concatenated record set.
        let records = vec![
            record("k1_q4k_gemv", 100, 300),
            record("k4_rmsnorm", 0, 900),
            record("k1_q4k_gemv", 300, 100),
        ];
        let mut acc = DiagAccumulator::new(None);
        assert_eq!(acc.total_records(), 0);
        assert!(acc.kernel_names().is_empty());
        acc.fold_records(&records);
        assert_eq!(acc.total_records(), 3);
        assert_eq!(acc.kernel_names(), vec!["k1_q4k_gemv", "k4_rmsnorm"]);
        assert_eq!(acc.render_csv(), render_summary(&records));
        acc.fold_records(&records);
        assert_eq!(acc.total_records(), 6);
        let doubled: Vec<DispatchRecord> = records.iter().chain(&records).cloned().collect();
        assert_eq!(acc.render_csv(), render_summary(&doubled));
    }

    #[test]
    fn accumulator_writes_its_report_once() {
        let path =
            std::env::temp_dir().join(format!("wgpu-compute-diag-acc-{}.csv", std::process::id()));
        let _ = std::fs::remove_file(&path);

        // Nothing folded and nothing dropped: the finish is
        // performed (and spent) but no file appears.
        let mut empty = DiagAccumulator::new(Some(path.clone()));
        assert!(empty.write_report(0));
        assert!(!path.exists(), "an empty run writes no report file");
        assert!(!empty.write_report(0), "the finish happens once");

        // Records folded: the file lands with the summary
        // table, and a second finish neither rewrites it nor
        // reports again.
        let mut acc = DiagAccumulator::new(Some(path.clone()));
        acc.fold_records(&[record("k1_q4k_gemv", 100, 300)]);
        assert!(acc.write_report(2));
        let text = std::fs::read_to_string(&path).expect("report file written");
        assert!(text.contains("k1_q4k_gemv,1,"), "report rows: {text}");
        assert!(text.contains("TOTAL,1,"), "report total: {text}");
        std::fs::write(&path, "sentinel").expect("plant sentinel");
        assert!(!acc.write_report(2));
        assert_eq!(
            std::fs::read_to_string(&path).expect("sentinel readable"),
            "sentinel",
            "a second finish must not rewrite the report"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fingerprint_f32_is_stable_and_matches_its_golden_values() {
        // The format is pinned byte for byte: FNV-1a 64 over
        // the element count (u64 LE) then each f32's bits
        // (LE). These goldens were computed independently of
        // the implementation; if a future edit changes the
        // hash input shape, this test is where it shows, and
        // every stored capture record's comparability is the
        // reason it must not change silently.
        assert_eq!(fingerprint_hex(fingerprint_f32(&[])), "af63bd4c8601b7df");
        assert_eq!(fingerprint_hex(fingerprint_f32(&[1.0])), "d9c063aea94e0421");
        assert_eq!(
            fingerprint_hex(fingerprint_u32(&[201_818])),
            "c3bd5d17f5820309"
        );
        // Stability: the same values hash the same, every time.
        let values: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5 - 17.25).collect();
        assert_eq!(fingerprint_f32(&values), fingerprint_f32(&values.clone()));
        assert_eq!(fingerprint_u32(&[1, 2, 3]), fingerprint_u32(&[1, 2, 3]));
    }

    #[test]
    fn fingerprint_changes_with_any_one_element() {
        let base: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let want = fingerprint_f32(&base);
        // One element changed in value.
        let mut changed = base.clone();
        changed[200] += 1.0;
        assert_ne!(fingerprint_f32(&changed), want);
        // One element changed by a single bit (a denormal
        // step): bitwise comparison is the contract.
        let mut bit = base.clone();
        bit[7] = f32::from_bits(bit[7].to_bits() + 1);
        assert_ne!(fingerprint_f32(&bit), want);
        // A strict prefix never collides with the whole: the
        // length is part of the hash input.
        assert_ne!(fingerprint_f32(&base[..255]), want);
        // Same for token ids.
        let ids: Vec<u32> = (0..64).collect();
        let mut ids_changed = ids.clone();
        ids_changed[63] = 999;
        assert_ne!(fingerprint_u32(&ids_changed), fingerprint_u32(&ids));
    }

    #[test]
    fn fingerprint_does_not_depend_on_feed_chunking() {
        // The hash is a function of the byte sequence, not of
        // how the feeds were split: odd-sized byte feeds,
        // split slice feeds (including across the pair fast
        // path's boundary), and one whole feed all agree.
        let values: Vec<f32> = (0..1025).map(|i| i as f32 * 0.25 - 3.0).collect();
        let whole = fingerprint_f32(&values);
        let mut split = Fnv1a64::new();
        split.update_u64(values.len() as u64);
        split.update_f32_slice(&values[..333]);
        split.update_f32_slice(&values[333..]);
        assert_eq!(split.finish(), whole);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for v in &values {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        let mut bytewise = Fnv1a64::new();
        for chunk in bytes.chunks(7) {
            bytewise.update_bytes(chunk);
        }
        assert_eq!(bytewise.finish(), whole);
        // The same property for token ids.
        let ids: Vec<u32> = (0..777).map(|i| i * 31 + 5).collect();
        let mut split_ids = Fnv1a64::new();
        split_ids.update_u64(ids.len() as u64);
        split_ids.update_u32_slice(&ids[..100]);
        split_ids.update_u32_slice(&ids[100..]);
        assert_eq!(split_ids.finish(), fingerprint_u32(&ids));
    }

    #[test]
    fn argmax_picks_the_max_at_the_lowest_index() {
        assert_eq!(argmax_of(&[]), None);
        assert_eq!(argmax_of(&[3.0]), Some(0));
        assert_eq!(argmax_of(&[1.0, 9.0, 9.0, 2.0]), Some(1));
        assert_eq!(argmax_of(&[-5.0, -1.0, -9.0]), Some(1));
    }

    #[test]
    fn model_file_fingerprint_samples_head_and_tail() {
        let dir =
            std::env::temp_dir().join(format!("wgpu-compute-diag-model-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        // Small file: hashed whole.
        let small = dir.join("small.gguf");
        std::fs::write(&small, b"GGUF fake but deterministic contents").expect("write small");
        let a = fingerprint_model_file(&small).expect("fingerprint small");
        let b = fingerprint_model_file(&small).expect("fingerprint small again");
        assert_eq!(a, b, "the same file fingerprints the same");
        assert_eq!(a.size_bytes, 36);
        assert_eq!(a.method, MODEL_FINGERPRINT_METHOD);
        std::fs::write(&small, b"GGUF fake but deterministic contentX").expect("rewrite small");
        let c = fingerprint_model_file(&small).expect("fingerprint changed small");
        assert_ne!(a.fingerprint, c.fingerprint);
        // Large file (> 2 MiB): only the size and the first
        // and last 1 MiB are read. A byte in the unsampled
        // middle cannot move the fingerprint (that is the
        // stated method, pinned here so nobody mistakes it
        // for a whole-file hash); bytes in either sample can.
        let mib = 1024 * 1024usize;
        let big = dir.join("big.gguf");
        let mut bytes = vec![7u8; 2 * mib + 12_345];
        bytes[0] = 1;
        std::fs::write(&big, &bytes).expect("write big");
        let base = fingerprint_model_file(&big).expect("fingerprint big");
        assert_eq!(base.size_bytes, (2 * mib + 12_345) as u64);
        bytes[mib + 100] = 9; // the gap between head and tail samples
        std::fs::write(&big, &bytes).expect("rewrite big middle");
        let middle = fingerprint_model_file(&big).expect("fingerprint big middle");
        assert_eq!(base.fingerprint, middle.fingerprint);
        let tail_idx = bytes.len() - 10;
        bytes[tail_idx] = 9; // inside the tail sample
        std::fs::write(&big, &bytes).expect("rewrite big tail");
        let tail = fingerprint_model_file(&big).expect("fingerprint big tail");
        assert_ne!(base.fingerprint, tail.fingerprint);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn capture_record(
        seq: u64,
        kind: &'static str,
        hash: &str,
        argmax: Option<u32>,
        accepted: Option<u64>,
    ) -> CaptureRecord {
        CaptureRecord {
            seq,
            kind,
            values: 4,
            hash: hash.to_string(),
            argmax,
            accepted,
        }
    }

    #[test]
    fn capture_csv_round_trips_through_parse() {
        let records = vec![
            capture_record(
                0,
                CaptureRecord::KIND_FORWARD,
                "d9c063aea94e0421",
                Some(17),
                None,
            ),
            capture_record(
                1,
                CaptureRecord::KIND_ARGMAX,
                "aaaaaaaaaaaaaaaa",
                Some(3),
                None,
            ),
            capture_record(2, CaptureRecord::KIND_BLOCK, "bbbbbbbbbbbbbbbb", None, None),
            capture_record(
                3,
                CaptureRecord::KIND_ROUND,
                "c3bd5d17f5820309",
                None,
                Some(5),
            ),
        ];
        let text = render_capture_csv(&records);
        assert!(text.starts_with(CAPTURE_CSV_HEADER));
        let parsed = parse_capture_csv(&text).expect("our own csv parses");
        assert_eq!(parsed, records);
        // Malformed input refuses with a locating message.
        assert!(parse_capture_csv("").is_err());
        assert!(parse_capture_csv("seq,kind,values,hash,argmax\n").is_err());
        let bad = format!("{CAPTURE_CSV_HEADER}\n0,weird,4,aa,,\n");
        assert!(parse_capture_csv(&bad).unwrap_err().contains("line 2"));
    }

    #[test]
    fn capture_diff_names_the_first_divergence() {
        let a = vec![
            capture_record(0, CaptureRecord::KIND_FORWARD, "aa", Some(1), None),
            capture_record(1, CaptureRecord::KIND_FORWARD, "bb", Some(2), None),
            capture_record(2, CaptureRecord::KIND_ROUND, "cc", None, Some(4)),
        ];
        assert!(diff_capture_records(&a, &a).identical());
        assert_eq!(diff_capture_records(&a, &a).compared, 3);
        // A hash change at seq 1: everything before it is
        // proven equal, and the diff names seq 1 and the field.
        let mut b = a.clone();
        b[1].hash = "bd".to_string();
        let diff = diff_capture_records(&a, &b);
        let div = diff.divergence.expect("a divergence");
        assert_eq!(div.seq, 1);
        assert_eq!(div.field, "hash");
        assert_eq!(diff.compared, 2);
        // An acceptance change is its own field.
        let mut c = a.clone();
        c[2].accepted = Some(3);
        let div = diff_capture_records(&a, &c).divergence.expect("divergence");
        assert_eq!(div.seq, 2);
        assert_eq!(div.field, "accepted");
        // A run that stops early diverges by presence at the
        // first record it lacks.
        let div = diff_capture_records(&a, &a[..2])
            .divergence
            .expect("divergence");
        assert_eq!(div.seq, 2);
        assert_eq!(div.field, "presence");
        assert!(div.a.is_some());
        assert!(div.b.is_none());
    }

    fn sample_manifest() -> RunManifest {
        RunManifest {
            diag_mode: "capture".to_string(),
            build_version: build_version().to_string(),
            git: "unknown".to_string(),
            adapter_name: "AMD Radeon RX 7900 XTX".to_string(),
            adapter_backend: "Vulkan".to_string(),
            adapter_device_type: "DiscreteGpu".to_string(),
            adapter_driver: "radv".to_string(),
            adapter_driver_info: "Mesa 26.0.8".to_string(),
            resolved_config: vec![
                ("diag".to_string(), "capture".to_string()),
                ("mmv_max_rows".to_string(), "16".to_string()),
            ],
            model: None,
            session: vec![
                ("kind".to_string(), "resident".to_string()),
                ("layers".to_string(), "52".to_string()),
            ],
        }
    }

    #[test]
    fn manifest_renders_and_its_fields_diff() {
        let manifest = sample_manifest();
        let json = manifest.render_json();
        assert!(json.contains("\"format\":\"wgpu-compute-diag-manifest\""));
        assert!(json.contains("\"model\":null"));
        let fields = manifest_fields(&json);
        let lookup = |key: &str| {
            fields
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.as_str())
        };
        assert_eq!(lookup("diag_mode"), Some("capture"));
        assert_eq!(lookup("adapter.driver"), Some("radv"));
        assert_eq!(lookup("adapter.driver_info"), Some("Mesa 26.0.8"));
        assert_eq!(lookup("resolved_config.mmv_max_rows"), Some("16"));
        assert_eq!(lookup("session.layers"), Some("52"));
        assert_eq!(lookup("model"), Some("null"));
        // Identical manifests diff to nothing.
        assert!(diff_manifest_fields(&fields, &fields).is_empty());
        // A driver change diffs to exactly that field.
        let mut other = sample_manifest();
        other.adapter_driver_info = "Mesa 25.3".to_string();
        let diffs = diff_manifest_fields(&fields, &manifest_fields(&other.render_json()));
        assert_eq!(diffs.len(), 1, "diffs: {diffs:?}");
        assert!(
            diffs[0].starts_with("adapter.driver_info"),
            "diff: {}",
            diffs[0]
        );
        // A noted model file renders as an object and its
        // fingerprint becomes a diffable field.
        let mut with_model = sample_manifest();
        with_model.model = Some(ModelFileInfo {
            path: "C:\\models\\weird \"quoted\" name.gguf".to_string(),
            size_bytes: 16_756_683_904,
            fingerprint: "0123456789abcdef".to_string(),
            method: MODEL_FINGERPRINT_METHOD.to_string(),
        });
        let fields = manifest_fields(&with_model.render_json());
        let lookup = |key: &str| {
            fields
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.as_str())
        };
        assert_eq!(lookup("model.size_bytes"), Some("16756683904"));
        assert_eq!(lookup("model.fingerprint"), Some("0123456789abcdef"));
        assert_eq!(
            lookup("model.path"),
            Some("C:\\models\\weird \"quoted\" name.gguf"),
            "an escaped path round-trips through the field reader"
        );
    }

    #[test]
    fn accumulator_capture_flow_writes_both_sidecars() {
        let dir =
            std::env::temp_dir().join(format!("wgpu-compute-diag-capture-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let out = dir.join("run.csv");
        let mut acc = DiagAccumulator::with_mode(Some(out.clone()), DiagMode::Capture);
        acc.begin_manifest(sample_manifest());
        assert!(acc.manifest_json().is_some());
        let first = acc
            .record_forward(CaptureRecord::KIND_FORWARD, &[0.0, 1.0, 0.5])
            .expect("capture records");
        assert_eq!(first.seq, 0);
        assert_eq!(first.argmax, Some(1));
        acc.record_forward(CaptureRecord::KIND_BLOCK, &[2.0, 1.0])
            .expect("block records");
        acc.record_argmax(42, 3.25).expect("argmax records");
        acc.record_spec_round(&[10, 11, 12], 2)
            .expect("round records");
        assert_eq!(acc.capture_record_count(), 4);
        assert!(acc.write_report(0));
        let manifest_path = manifest_sidecar_path(&out);
        let capture_path = capture_sidecar_path(&out);
        let manifest_text = std::fs::read_to_string(&manifest_path).expect("manifest sidecar");
        assert!(manifest_text.contains("\"diag_mode\":\"capture\""));
        let capture_text = std::fs::read_to_string(&capture_path).expect("capture sidecar");
        let parsed = parse_capture_csv(&capture_text).expect("sidecar parses");
        assert_eq!(parsed.len(), 4);
        assert_eq!(parsed[3].kind, CaptureRecord::KIND_ROUND);
        assert_eq!(parsed[3].accepted, Some(2));
        assert!(diff_capture_records(&parsed, acc.capture_records()).identical());
        // A second finish writes nothing again.
        assert!(!acc.write_report(0));

        // Timing mode: capture entry points are all inert and
        // no sidecars appear beside the report.
        let tout = dir.join("timing.csv");
        let mut timing = DiagAccumulator::with_mode(Some(tout.clone()), DiagMode::Timing);
        assert!(timing
            .record_forward(CaptureRecord::KIND_FORWARD, &[1.0])
            .is_none());
        assert!(timing.record_spec_round(&[1], 1).is_none());
        timing.begin_manifest(sample_manifest());
        assert!(timing.manifest_json().is_none());
        timing.fold_records(&[record("k1_q4k_gemv", 100, 300)]);
        assert!(timing.write_report(0));
        assert!(!manifest_sidecar_path(&tout).exists());
        assert!(!capture_sidecar_path(&tout).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn accumulator_note_model_file_fills_the_manifest() {
        let dir = std::env::temp_dir().join(format!(
            "wgpu-compute-diag-notemodel-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let model = dir.join("model.gguf");
        std::fs::write(&model, b"GGUF note-model contents").expect("write model");
        let mut acc = DiagAccumulator::with_mode(None, DiagMode::Capture);
        // Noted BEFORE the manifest begins: held as pending
        // and merged at begin.
        acc.note_model_file(&model).expect("note before begin");
        acc.begin_manifest(sample_manifest());
        let json = acc.manifest_json().expect("manifest begun");
        let fields = manifest_fields(&json);
        let path = fields
            .iter()
            .find(|(k, _)| k == "model.path")
            .map(|(_, v)| v.as_str());
        assert_eq!(path, Some(model.display().to_string().as_str()));
        // Noted AFTER begin: replaces the manifest's model.
        let other = dir.join("other.gguf");
        std::fs::write(&other, b"GGUF other contents, different bytes").expect("write other");
        acc.note_model_file(&other).expect("note after begin");
        let fields = manifest_fields(&acc.manifest_json().expect("manifest still there"));
        let size = fields
            .iter()
            .find(|(k, _)| k == "model.size_bytes")
            .map(|(_, v)| v.as_str());
        assert_eq!(size, Some("36"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Cost probe for the bench note (not a gate): run with
    /// `--ignored --nocapture` to print what one forward's
    /// fingerprint costs at Glimmer's vocab (202,048 values)
    /// and at a full block (16 rows of it).
    #[test]
    #[ignore]
    fn capture_hash_cost_measurement() {
        let vocab: Vec<f32> = (0..202_048).map(|i| (i % 977) as f32 * 0.25).collect();
        let block: Vec<f32> = vocab.iter().copied().cycle().take(202_048 * 16).collect();
        let start = std::time::Instant::now();
        let mut acc = 0u64;
        for _ in 0..200 {
            acc ^= fingerprint_f32(&vocab);
        }
        let per_vocab = start.elapsed() / 200;
        let start = std::time::Instant::now();
        for _ in 0..20 {
            acc ^= fingerprint_f32(&block);
        }
        let per_block = start.elapsed() / 20;
        eprintln!(
            "fingerprint_f32: vocab 202048 values in {:?} ({:.1} MB/s of f32 data), \
             block 16x vocab in {:?} (checksum {acc:016x})",
            per_vocab,
            (202_048 * 4) as f64 / per_vocab.as_secs_f64() / 1.0e6,
            per_block,
        );
    }
}
