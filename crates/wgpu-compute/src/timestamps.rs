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
//! Failure modes: a device without the timestamp features
//! refuses the recorder at device open (named in the diag log,
//! instrumentation simply absent); a full query window drops
//! stamps and counts them rather than corrupting timings.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::error::{Result, WgpuError};

/// Environment variable gating the instrumentation:
/// `off` (default) or `timing`.
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
}

impl DiagMode {
    /// Parse a raw env value: only the exact string `timing`
    /// enables instrumentation; anything else (including a
    /// missing variable) is off, so a typo fails closed to the
    /// clean path.
    #[must_use]
    pub fn parse(raw: Option<&str>) -> DiagMode {
        match raw {
            Some("timing") => DiagMode::Timing,
            _ => DiagMode::Off,
        }
    }

    /// The mode the current process environment requests.
    #[must_use]
    pub fn from_env() -> DiagMode {
        DiagMode::parse(std::env::var(DIAG_ENV).ok().as_deref())
    }

    /// Whether this mode, on a device granted `features`,
    /// creates a recorder. Both timestamp features are
    /// required: set-level queries for resolution and
    /// inside-pass writes for per-dispatch brackets.
    #[must_use]
    pub fn wants_recorder(self, features: wgpu::Features) -> bool {
        self == DiagMode::Timing
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
#[derive(Debug)]
pub struct DiagAccumulator {
    out_path: Option<PathBuf>,
    by_kernel: BTreeMap<&'static str, KernelSummary>,
    total_records: u64,
    total_ns: u64,
    drain_failures: u64,
    finished: bool,
}

impl DiagAccumulator {
    /// An accumulator reporting to `out_path` (or to the diag
    /// channel only, when `None`).
    #[must_use]
    pub fn new(out_path: Option<PathBuf>) -> DiagAccumulator {
        DiagAccumulator {
            out_path,
            by_kernel: BTreeMap::new(),
            total_records: 0,
            total_ns: 0,
            drain_failures: 0,
            finished: false,
        }
    }

    /// An accumulator whose report file is whatever
    /// [`DIAG_OUT_ENV`] names in the current environment
    /// (unset: channel only). Read once, at construction, so
    /// a run's destination cannot move mid-flight.
    #[must_use]
    pub fn from_env() -> DiagAccumulator {
        DiagAccumulator::new(std::env::var_os(DIAG_OUT_ENV).map(PathBuf::from))
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
    /// records were folded. `dropped` is the recorder's
    /// lifetime drop count, passed in because the accumulator
    /// does not own the recorder. Later calls are no-ops.
    /// Returns whether this call performed the finish.
    pub fn write_report(&mut self, dropped: u64) -> bool {
        if self.finished {
            return false;
        }
        self.finished = true;
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
    fn mode_parses_timing_only() {
        assert_eq!(DiagMode::parse(Some("timing")), DiagMode::Timing);
        assert_eq!(DiagMode::parse(Some("off")), DiagMode::Off);
        assert_eq!(DiagMode::parse(Some("TIMING")), DiagMode::Off);
        assert_eq!(DiagMode::parse(Some("")), DiagMode::Off);
        assert_eq!(DiagMode::parse(None), DiagMode::Off);
    }

    #[test]
    fn off_mode_never_wants_a_recorder() {
        // The off path must create no query resources, whatever
        // the device offers; timing still requires both
        // timestamp features.
        let all = wgpu::Features::all();
        assert!(!DiagMode::Off.wants_recorder(all));
        assert!(!DiagMode::Timing.wants_recorder(wgpu::Features::empty()));
        assert!(!DiagMode::Timing.wants_recorder(wgpu::Features::TIMESTAMP_QUERY));
        assert!(DiagMode::Timing.wants_recorder(
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
}
