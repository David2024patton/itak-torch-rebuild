//! Activation dimensionality: how much of its space a captured
//! stage's signal actually occupies.
//!
//! The trace tooling ([`crate::trace`]) reduces a stage to means and
//! variances. Those numbers cannot see the failure lane 04 reproduced
//! synthetically and the norm audit suspects on the real file: a
//! forward whose per-token signal washes out across depth, every
//! position's vector converging toward the same point while the
//! end-to-end tokens still look plausible. A washed-out stage has a
//! measurable shape: its activation matrix, rows as positions and
//! columns as hidden units, collapses onto a line. This module
//! measures that shape with two numbers.
//!
//! - Participation ratio: the effective dimensionality of the
//!   covariance spectrum, `(sum of eigenvalues)^2 / (sum of squared
//!   eigenvalues)`. It counts how many directions carry the stage's
//!   variance: near 1 for a collapsed stage, near the hidden width
//!   for an isotropic one. Computed without an eigensolver: with `G`
//!   the Gram matrix of the centered rows, the covariance's nonzero
//!   eigenvalues are `G`'s scaled by a constant that cancels in the
//!   ratio, leaving `tr(G)^2 / ||G||_F^2`, accumulated in row blocks
//!   so memory stays at one block of rows however long the capture
//!   is. Needs at least 2 rows; with fewer there is no spectrum to
//!   speak of and the probe refuses by name instead.
//! - TwoNN intrinsic dimension (Facco et al., 2017): for each point,
//!   the ratio `mu` of the distances to its second and first nearest
//!   neighbours follows a Pareto law whose parameter is the data's
//!   intrinsic dimension; the estimate is `n / sum(ln mu)` over the
//!   usable points. Neighbour ratios are only meaningful with a
//!   crowd of points, so the estimator runs only at or above
//!   [`DimensionConfig::min_points_twonn`] rows; below the gate the
//!   probe refuses through the diagnostic channel rather than
//!   printing a junk number.
//!
//! Both estimators are diagnostics, not model math: they run on
//! captured tensors after the forward, cost `O(rows^2 * cols)`, and
//! never feed back into generation. Every refusal, degenerate input,
//! and completed measurement emits through the crate [`Diag`]
//! channel in the same key=value style as the rest of the crate.
//!
//! ## Line format (version 1)
//!
//! ```text
//! dim v=1 stage=<stage> rows=<r> cols=<c> pr=<e|na> twonn=<e|na>
//! ```
//!
//! Estimates print with `{:.6e}`; `na` marks an estimator that did
//! not run (below its gate, or a spectrum that does not exist). A
//! stage whose rows are all identical has a spectrum that exists and
//! is zero everywhere, so its participation ratio prints as exactly
//! 0: that is the loudest collapse reading the format can give.

use std::sync::Arc;

use crate::diag::{Diag, NullDiag};
use crate::error::{GlimmerError, Result};
use crate::trace::TraceSink;

/// Every dimension-measurement parameter, as config with defaults,
/// read at call time. Build with [`DimensionConfig::default`] and
/// override the fields a caller cares about;
/// [`DimensionConfig::validate`] (run by
/// [`DimensionProbe::analyze`] on every call) refuses values outside
/// their meaningful ranges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DimensionConfig {
    /// Minimum row count before the TwoNN estimator runs. TwoNN is a
    /// statistical estimate over neighbour ratios; on a handful of
    /// points it returns noise dressed as a dimension, so below this
    /// many rows the probe refuses it by name. Default 32.
    pub min_points_twonn: usize,
    /// Row block size for the Gram accumulation behind the
    /// participation ratio. Bounds the probe's working memory to one
    /// block of centered rows; the result is identical for any block
    /// size. Default 128.
    pub gram_block_rows: usize,
}

impl Default for DimensionConfig {
    fn default() -> Self {
        DimensionConfig {
            min_points_twonn: 32,
            gram_block_rows: 128,
        }
    }
}

impl DimensionConfig {
    /// Refuse a config whose fields cannot mean what they claim.
    pub fn validate(&self) -> Result<()> {
        if self.min_points_twonn < 2 {
            return Err(GlimmerError::InvalidDimension {
                detail: format!(
                    "min_points_twonn must be at least 2, got {}",
                    self.min_points_twonn
                ),
            });
        }
        if self.gram_block_rows == 0 {
            return Err(GlimmerError::InvalidDimension {
                detail: "gram_block_rows must be at least 1, got 0".to_string(),
            });
        }
        Ok(())
    }
}

/// One stage's dimensionality, as measured. An estimator that did
/// not run is `None`, never a placeholder number; the reason is on
/// the diagnostic channel (see the module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct DimensionReport {
    /// Rows (positions) in the measured matrix.
    pub rows: usize,
    /// Columns (hidden units) in the measured matrix.
    pub cols: usize,
    /// Participation ratio of the covariance spectrum. `None` when
    /// the capture has fewer than 2 rows. Exactly 0.0 when every row
    /// is identical: a spectrum that exists and is zero everywhere.
    pub participation_ratio: Option<f64>,
    /// TwoNN intrinsic-dimension estimate. `None` when the row count
    /// is below [`DimensionConfig::min_points_twonn`] or too few
    /// points have distinct neighbours to estimate from.
    pub twonn: Option<f64>,
    /// Points that contributed a neighbour ratio to the TwoNN
    /// estimate (rows whose nearest neighbour is at distance 0,
    /// exact duplicates, contribute nothing and are not counted).
    pub twonn_points: usize,
}

impl DimensionReport {
    /// This report in the stable version-1 line format documented
    /// at the top of this module.
    pub fn to_line(&self, stage: &str) -> String {
        format!(
            "dim v=1 stage={} rows={} cols={} pr={} twonn={}",
            stage,
            self.rows,
            self.cols,
            format_estimate(self.participation_ratio),
            format_estimate(self.twonn),
        )
    }
}

fn format_estimate(value: Option<f64>) -> String {
    match value {
        Some(v) => format!("{v:.6e}"),
        None => "na".to_string(),
    }
}

fn check_shape(tensor: &[f32], rows: usize, cols: usize) -> Result<()> {
    if cols == 0 {
        return Err(GlimmerError::InvalidDimension {
            detail: "cols is 0; an activation matrix needs at least one column".to_string(),
        });
    }
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| GlimmerError::InvalidDimension {
            detail: format!("rows {rows} x cols {cols} overflows a tensor length"),
        })?;
    if tensor.len() != expected {
        return Err(GlimmerError::InvalidDimension {
            detail: format!(
                "tensor holds {} values but rows {rows} x cols {cols} needs {expected}",
                tensor.len()
            ),
        });
    }
    Ok(())
}

/// Column means of the matrix, in f64.
fn column_means(tensor: &[f32], rows: usize, cols: usize) -> Vec<f64> {
    let mut means = vec![0.0f64; cols];
    for row in tensor.chunks_exact(cols) {
        for (mean, &value) in means.iter_mut().zip(row.iter()) {
            *mean += f64::from(value);
        }
    }
    let inv_rows = 1.0 / rows as f64;
    for mean in &mut means {
        *mean *= inv_rows;
    }
    means
}

/// `count` centered rows starting at row `start`, as f64.
fn centered_block(
    tensor: &[f32],
    means: &[f64],
    start: usize,
    count: usize,
    cols: usize,
) -> Vec<f64> {
    let mut out = vec![0.0f64; count * cols];
    for r in 0..count {
        let src = &tensor[(start + r) * cols..(start + r + 1) * cols];
        let dst = &mut out[r * cols..(r + 1) * cols];
        for (d, (&s, &m)) in dst.iter_mut().zip(src.iter().zip(means.iter())) {
            *d = f64::from(s) - m;
        }
    }
    out
}

/// `(tr(G), ||G||_F^2)` for `G` the Gram matrix of the centered
/// rows: the two spectrum sums the participation ratio is built
/// from. Accumulated block by block; no full Gram matrix is ever
/// materialized.
fn spectrum_sums(tensor: &[f32], rows: usize, cols: usize, block_rows: usize) -> (f64, f64) {
    let means = column_means(tensor, rows, cols);
    let mut trace = 0.0f64;
    let mut frobenius_sq = 0.0f64;
    let mut i0 = 0;
    while i0 < rows {
        let i_count = block_rows.min(rows - i0);
        let bi = centered_block(tensor, &means, i0, i_count, cols);
        let mut j0 = 0;
        while j0 < rows {
            let j_count = block_rows.min(rows - j0);
            let bj = centered_block(tensor, &means, j0, j_count, cols);
            for ii in 0..i_count {
                let a = &bi[ii * cols..(ii + 1) * cols];
                for jj in 0..j_count {
                    let b = &bj[jj * cols..(jj + 1) * cols];
                    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                    frobenius_sq += dot * dot;
                    if i0 + ii == j0 + jj {
                        trace += dot;
                    }
                }
            }
            j0 += block_rows;
        }
        i0 += block_rows;
    }
    (trace, frobenius_sq)
}

/// The participation ratio of the matrix's covariance spectrum:
/// `(sum of eigenvalues)^2 / (sum of squared eigenvalues)`, in
/// `(1, cols]` for any real spectrum. `Ok(None)` when there are
/// fewer than 2 rows (no spectrum exists); `Ok(Some(0.0))` when
/// every row is identical (the spectrum exists and is zero).
/// Pure: no config gate, no diagnostics; [`DimensionProbe`] owns
/// those.
pub fn participation_ratio(tensor: &[f32], rows: usize, cols: usize) -> Result<Option<f64>> {
    check_shape(tensor, rows, cols)?;
    if rows < 2 {
        return Ok(None);
    }
    let (trace, frobenius_sq) = spectrum_sums(
        tensor,
        rows,
        cols,
        DimensionConfig::default().gram_block_rows,
    );
    if trace <= 0.0 {
        return Ok(Some(0.0));
    }
    Ok(Some(trace * trace / frobenius_sq))
}

/// The TwoNN estimate over the matrix's rows, plus the number of
/// points that contributed. `None` when fewer than 2 points have a
/// distinct nearest neighbour, or every usable ratio is exactly 1
/// (the estimator's denominator vanishes). No minimum-points gate
/// here: the gate is policy and lives in [`DimensionProbe`]; this
/// is the bare estimator.
fn twonn_estimate(tensor: &[f32], rows: usize, cols: usize) -> (Option<f64>, usize) {
    let points: Vec<&[f32]> = tensor.chunks_exact(cols).collect();
    debug_assert_eq!(points.len(), rows);
    let mut sum_log_mu = 0.0f64;
    let mut usable = 0usize;
    for (i, point) in points.iter().enumerate() {
        let mut nearest = f64::INFINITY;
        let mut second = f64::INFINITY;
        for (j, other) in points.iter().enumerate() {
            if j == i {
                continue;
            }
            let dist_sq: f64 = point
                .iter()
                .zip(other.iter())
                .map(|(&a, &b)| {
                    let d = f64::from(a) - f64::from(b);
                    d * d
                })
                .sum();
            if dist_sq < nearest {
                second = nearest;
                nearest = dist_sq;
            } else if dist_sq < second {
                second = dist_sq;
            }
        }
        // Ratios of distances are ratios of squared distances
        // square-rooted; a zero nearest distance is an exact
        // duplicate and carries no ratio information.
        if nearest > 0.0 && second.is_finite() {
            let mu = (second / nearest).sqrt();
            sum_log_mu += mu.ln();
            usable += 1;
        }
    }
    if usable < 2 || sum_log_mu <= 0.0 {
        return (None, usable);
    }
    (Some(usable as f64 / sum_log_mu), usable)
}

/// The TwoNN intrinsic-dimension estimate (Facco et al.) over the
/// matrix's rows. `Ok(None)` under the same conditions as the
/// internal estimator; see [`twonn_estimate`]. Callers who want the
/// minimum-points gate and diagnostics use [`DimensionProbe`].
pub fn twonn_dimension(tensor: &[f32], rows: usize, cols: usize) -> Result<Option<f64>> {
    check_shape(tensor, rows, cols)?;
    Ok(twonn_estimate(tensor, rows, cols).0)
}

/// The probe: a [`DimensionConfig`] plus the diagnostic channel,
/// measuring captured activations into [`DimensionReport`]s. Every
/// refusal is emitted by name through the channel before the report
/// records the estimator as not run, so a dump's `na` always has its
/// reason in the diag stream beside it.
#[derive(Clone)]
pub struct DimensionProbe {
    config: DimensionConfig,
    diag: Arc<dyn Diag>,
}

impl std::fmt::Debug for DimensionProbe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The diag channel is a trait object with no Debug of its
        // own; the config is the probe's whole inspectable state.
        f.debug_struct("DimensionProbe")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl DimensionProbe {
    /// A probe with a silent diagnostic channel. Reports are still
    /// complete values; only the refusal reasons go unrecorded.
    pub fn new(config: DimensionConfig) -> Self {
        DimensionProbe {
            config,
            diag: Arc::new(NullDiag),
        }
    }

    /// A probe emitting through the given diagnostic channel.
    pub fn with_diag(config: DimensionConfig, diag: Arc<dyn Diag>) -> Self {
        DimensionProbe { config, diag }
    }

    /// The config this probe measures under.
    pub fn config(&self) -> DimensionConfig {
        self.config
    }

    /// Measure one captured stage: `tensor` is the row-major
    /// `rows x cols` activation matrix. The only error is a request
    /// that cannot describe a measurement (bad shape, bad config);
    /// gates and degenerate inputs are report values plus diagnostic
    /// events, not errors, because a trace run must survive the
    /// stages it exists to incriminate.
    pub fn analyze(&self, tensor: &[f32], rows: usize, cols: usize) -> Result<DimensionReport> {
        self.config.validate()?;
        check_shape(tensor, rows, cols)?;
        let mut report = DimensionReport {
            rows,
            cols,
            participation_ratio: None,
            twonn: None,
            twonn_points: 0,
        };

        if rows < 2 {
            self.diag.event(
                "dimension.pr_refused",
                &format!("rows={rows} min_points=2 estimator=participation_ratio"),
            );
        } else {
            let (trace, frobenius_sq) =
                spectrum_sums(tensor, rows, cols, self.config.gram_block_rows);
            if trace <= 0.0 {
                self.diag.event(
                    "dimension.degenerate",
                    &format!("rows={rows} cols={cols} total_variance=0"),
                );
                report.participation_ratio = Some(0.0);
            } else {
                report.participation_ratio = Some(trace * trace / frobenius_sq);
            }
        }

        if rows < self.config.min_points_twonn {
            self.diag.event(
                "dimension.twonn_refused",
                &format!(
                    "rows={rows} min_points={} estimator=twonn reason=below_min_points",
                    self.config.min_points_twonn
                ),
            );
        } else {
            let (estimate, usable) = twonn_estimate(tensor, rows, cols);
            report.twonn_points = usable;
            match estimate {
                Some(dimension) => report.twonn = Some(dimension),
                None => self.diag.event(
                    "dimension.twonn_refused",
                    &format!(
                        "rows={rows} usable_points={usable} estimator=twonn \
                         reason=no_usable_neighbour_pairs"
                    ),
                ),
            }
        }

        self.diag.event(
            "dimension.measured",
            &format!(
                "rows={rows} cols={cols} pr={} twonn={} twonn_points={}",
                format_estimate(report.participation_ratio),
                format_estimate(report.twonn),
                report.twonn_points
            ),
        );
        Ok(report)
    }
}

/// One captured stage and its dimension report, as collected by
/// [`DimensionLog`].
#[derive(Debug, Clone, PartialEq)]
pub struct DimensionEntry {
    /// Stage name as captured.
    pub stage: String,
    /// The stage's dimensionality.
    pub report: DimensionReport,
}

/// A collecting [`TraceSink`]: runs a [`DimensionProbe`] over every
/// capture and keeps the reports in capture order. The dimension
/// counterpart of [`crate::trace::TraceLog`]; the two compose
/// through any sink that forwards to both.
#[derive(Debug)]
pub struct DimensionLog {
    probe: DimensionProbe,
    entries: Vec<DimensionEntry>,
}

impl DimensionLog {
    /// A log measuring with the given probe.
    pub fn new(probe: DimensionProbe) -> Self {
        DimensionLog {
            probe,
            entries: Vec::new(),
        }
    }

    /// The entries so far, in capture order.
    pub fn entries(&self) -> &[DimensionEntry] {
        &self.entries
    }

    /// All entries in the stable line format, one per line.
    pub fn lines(&self) -> Vec<String> {
        self.entries
            .iter()
            .map(|entry| entry.report.to_line(&entry.stage))
            .collect()
    }
}

impl TraceSink for DimensionLog {
    fn capture(&mut self, stage: &str, tensor: &[f32], rows: usize, cols: usize) {
        match self.probe.analyze(tensor, rows, cols) {
            Ok(report) => self.entries.push(DimensionEntry {
                stage: stage.to_string(),
                report,
            }),
            Err(err) => self
                .probe
                .diag
                .event("dimension.failed", &format!("stage={stage} error={err}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::SeededRng;
    use std::f64::consts::PI;
    use std::sync::Mutex;

    /// Box-Muller over the seeded stream: deterministic Gaussians
    /// for fixture clouds, no external dependencies.
    fn gaussian(rng: &mut SeededRng) -> f64 {
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    fn gaussian_vec(rng: &mut SeededRng, cols: usize) -> Vec<f64> {
        (0..cols).map(|_| gaussian(rng)).collect()
    }

    fn normalized(v: &[f64]) -> Vec<f64> {
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v.iter().map(|x| x / norm).collect()
    }

    /// Points along one line in a high-dimensional space: offset +
    /// t * direction, t spread over [0, 10].
    fn line_cloud(seed: u64, rows: usize, cols: usize) -> Vec<f32> {
        let mut rng = SeededRng::new(seed);
        let direction = normalized(&gaussian_vec(&mut rng, cols));
        let offset = gaussian_vec(&mut rng, cols);
        let mut out = Vec::with_capacity(rows * cols);
        for _ in 0..rows {
            let t = rng.next_f64() * 10.0;
            for c in 0..cols {
                out.push((offset[c] + t * direction[c]) as f32);
            }
        }
        out
    }

    /// Points in a 2D plane embedded in a high-dimensional space:
    /// offset + u * a + v * b with a, b orthonormal.
    fn plane_cloud(seed: u64, rows: usize, cols: usize) -> Vec<f32> {
        let mut rng = SeededRng::new(seed);
        let a = normalized(&gaussian_vec(&mut rng, cols));
        let raw_b = gaussian_vec(&mut rng, cols);
        let proj: f64 = raw_b.iter().zip(a.iter()).map(|(x, y)| x * y).sum();
        let b = normalized(
            &raw_b
                .iter()
                .zip(a.iter())
                .map(|(x, y)| x - proj * y)
                .collect::<Vec<_>>(),
        );
        let offset = gaussian_vec(&mut rng, cols);
        let mut out = Vec::with_capacity(rows * cols);
        for _ in 0..rows {
            let u = rng.next_f64() * 10.0;
            let v = rng.next_f64() * 10.0;
            for c in 0..cols {
                out.push((offset[c] + u * a[c] + v * b[c]) as f32);
            }
        }
        out
    }

    #[derive(Default)]
    struct RecordingDiag {
        events: Mutex<Vec<(String, String)>>,
    }

    impl RecordingDiag {
        fn has_event(&self, name: &str) -> bool {
            self.events
                .lock()
                .unwrap()
                .iter()
                .any(|(event, _)| event == name)
        }

        fn detail_of(&self, name: &str) -> String {
            self.events
                .lock()
                .unwrap()
                .iter()
                .find(|(event, _)| event == name)
                .map(|(_, detail)| detail.clone())
                .unwrap_or_default()
        }
    }

    impl Diag for RecordingDiag {
        fn event(&self, event: &str, detail: &str) {
            self.events
                .lock()
                .unwrap()
                .push((event.to_string(), detail.to_string()));
        }
    }

    fn probe_with_recording() -> (DimensionProbe, Arc<RecordingDiag>) {
        let recording = Arc::new(RecordingDiag::default());
        let probe = DimensionProbe::with_diag(DimensionConfig::default(), recording.clone());
        (probe, recording)
    }

    #[test]
    fn line_cloud_reads_dimension_near_one() {
        let (probe, _) = probe_with_recording();
        let cloud = line_cloud(42, 200, 64);
        let report = probe.analyze(&cloud, 200, 64).unwrap();
        let pr = report.participation_ratio.unwrap();
        assert!((0.95..1.05).contains(&pr), "line pr={pr}");
        let twonn = report.twonn.unwrap();
        assert!((0.7..1.4).contains(&twonn), "line twonn={twonn}");
        assert_eq!(report.twonn_points, 200);
    }

    #[test]
    fn plane_cloud_reads_twonn_near_two() {
        let (probe, _) = probe_with_recording();
        let cloud = plane_cloud(7, 400, 48);
        let report = probe.analyze(&cloud, 400, 48).unwrap();
        let twonn = report.twonn.unwrap();
        assert!((1.6..2.5).contains(&twonn), "plane twonn={twonn}");
        let pr = report.participation_ratio.unwrap();
        assert!((1.7..2.35).contains(&pr), "plane pr={pr}");
    }

    #[test]
    fn isotropic_gaussian_reads_high_participation_ratio() {
        let (probe, _) = probe_with_recording();
        let mut rng = SeededRng::new(99);
        let (rows, cols) = (1024, 16);
        let mut cloud = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            cloud.push(gaussian(&mut rng) as f32);
        }
        let report = probe.analyze(&cloud, rows, cols).unwrap();
        let pr = report.participation_ratio.unwrap();
        // Sixteen independent directions share the variance almost
        // evenly; the ratio must land near the column count, nowhere
        // near the collapsed end of the scale.
        assert!(pr > 11.0 && pr <= 16.01, "gaussian pr={pr}");
    }

    #[test]
    fn twonn_refused_below_min_points_by_name() {
        let (probe, recording) = probe_with_recording();
        let cloud = line_cloud(3, 8, 16);
        let report = probe.analyze(&cloud, 8, 16).unwrap();
        assert!(report.twonn.is_none());
        assert_eq!(report.twonn_points, 0);
        // The participation ratio has no such gate and still runs.
        assert!(report.participation_ratio.is_some());
        assert!(recording.has_event("dimension.twonn_refused"));
        let detail = recording.detail_of("dimension.twonn_refused");
        assert!(detail.contains("rows=8"), "detail={detail}");
        assert!(detail.contains("min_points=32"), "detail={detail}");
        assert!(
            detail.contains("reason=below_min_points"),
            "detail={detail}"
        );
    }

    #[test]
    fn collapsed_cloud_reads_participation_ratio_near_one() {
        let (probe, _) = probe_with_recording();
        // The washout signature: every row is the same vector, with
        // only a whisper of variation left, all of it along the
        // shared direction itself (per-row scale jitter).
        let mut rng = SeededRng::new(5);
        let (rows, cols) = (64, 128);
        let base = gaussian_vec(&mut rng, cols);
        let mut cloud = Vec::with_capacity(rows * cols);
        for _ in 0..rows {
            let scale = 1.0 + (rng.next_f64() - 0.5) * 2e-3;
            for &component in &base {
                cloud.push((component * scale) as f32);
            }
        }
        let report = probe.analyze(&cloud, rows, cols).unwrap();
        let pr = report.participation_ratio.unwrap();
        assert!(pr > 0.99 && pr < 1.05, "collapsed pr={pr}");
    }

    #[test]
    fn identical_rows_read_zero_and_emit_degenerate() {
        let (probe, recording) = probe_with_recording();
        let (rows, cols) = (40, 8);
        let row: Vec<f32> = (0..cols).map(|c| c as f32).collect();
        let cloud: Vec<f32> = row.repeat(rows);
        let report = probe.analyze(&cloud, rows, cols).unwrap();
        assert_eq!(report.participation_ratio, Some(0.0));
        assert!(report.twonn.is_none());
        assert!(recording.has_event("dimension.degenerate"));
        assert!(recording.has_event("dimension.twonn_refused"));
    }

    #[test]
    fn single_row_refuses_participation_ratio_by_name() {
        let (probe, recording) = probe_with_recording();
        let report = probe.analyze(&[1.0, 2.0, 3.0], 1, 3).unwrap();
        assert!(report.participation_ratio.is_none());
        assert!(report.twonn.is_none());
        assert!(recording.has_event("dimension.pr_refused"));
    }

    #[test]
    fn shape_mismatch_is_a_named_error() {
        let probe = DimensionProbe::new(DimensionConfig::default());
        let err = probe.analyze(&[1.0; 6], 2, 4).unwrap_err();
        match err {
            GlimmerError::InvalidDimension { detail } => {
                assert!(detail.contains("needs 8"), "detail={detail}");
            }
            other => panic!("expected InvalidDimension, got {other}"),
        }
        // The bare estimators agree with the probe on shapes.
        assert!(participation_ratio(&[1.0; 6], 2, 4).is_err());
        assert!(twonn_dimension(&[1.0; 6], 2, 4).is_err());
    }

    #[test]
    fn dimension_log_collects_lines_in_capture_order() {
        let mut log = DimensionLog::new(DimensionProbe::new(DimensionConfig::default()));
        let cloud = line_cloud(11, 64, 32);
        log.capture("layer.0.out", &cloud, 64, 32);
        log.capture("logits", &[0.5, -0.5], 1, 2);
        let lines = log.lines();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("dim v=1 stage=layer.0.out rows=64 cols=32 pr="));
        assert!(!lines[0].contains("twonn=na"), "line={}", lines[0]);
        assert_eq!(
            lines[1],
            "dim v=1 stage=logits rows=1 cols=2 pr=na twonn=na"
        );
    }
}
