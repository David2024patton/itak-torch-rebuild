//! Per-layer activation tracing for the forward pass: the tooling
//! that localizes a forward numerics error to the first divergent
//! layer instead of guessing from end-to-end logits.
//!
//! [`GlimmerModel::forward_traced`](crate::GlimmerModel::forward_traced)
//! runs the ordinary forward and reports the residual stream at fixed
//! stage boundaries to a [`TraceSink`]:
//!
//! - `embed`: the embedding output, before any layer.
//! - `layer.<i>.mid`: layer `i`'s residual stream after the attention
//!   half (post-attention norm applied), entering the FFN half.
//! - `layer.<i>.out`: layer `i`'s residual stream after the full
//!   layer (post-FFN norm applied).
//! - `final_norm`: the final norm's output for the last position.
//! - `logits`: the finished logits (softcapped and scaled).
//!
//! A sink receives the raw tensor plus its row/column shape. The
//! residual-stream stages are `rows x hidden`; `final_norm` and
//! `logits` are single rows. [`TraceLog`] is the collecting sink: it
//! reduces every capture to a [`TraceRecord`] of statistics (mean,
//! variance, max abs, first values) over the whole tensor and over
//! the last row separately, and formats records in the stable text
//! format below so two runs, or our run and a reference dump reduced
//! to the same statistics, can be diffed line by line.
//!
//! ## Line format (version 1)
//!
//! ```text
//! trace v=1 stage=<stage> rows=<r> cols=<c> n=<count> mean=<e> var=<e> max_abs=<e> last_mean=<e> last_var=<e> last_max_abs=<e> first=[<f>,...] last_first=[<f>,...]
//! ```
//!
//! Statistics are printed with `{:.6e}`, first values with `{:.6}`,
//! comma-separated with no spaces. Variance is the population
//! variance. The last-row fields describe the final position's
//! vector, the one the head reads; the whole-tensor fields catch
//! position-dependent errors (for example a RoPE geometry mistake)
//! that a last-row-only view would hide. A record whose stage is
//! missing on one side of a diff, or whose statistics first disagree
//! at some layer, is the localization signal this module exists to
//! produce.

/// Statistics over a slice of activations, computed in f64 so the
/// reduction itself is never the noise source.
#[derive(Debug, Clone, PartialEq)]
pub struct VecStats {
    /// Number of values reduced.
    pub n: usize,
    /// Arithmetic mean.
    pub mean: f64,
    /// Population variance.
    pub var: f64,
    /// Largest absolute value.
    pub max_abs: f64,
    /// The first `first_n` values, in order (fewer if the slice is
    /// shorter).
    pub first: Vec<f32>,
}

/// Reduce `data` to its statistics, keeping the first `first_n`
/// values verbatim. An empty slice reduces to zeros with `n = 0`.
pub fn stats_of(data: &[f32], first_n: usize) -> VecStats {
    if data.is_empty() {
        return VecStats {
            n: 0,
            mean: 0.0,
            var: 0.0,
            max_abs: 0.0,
            first: Vec::new(),
        };
    }
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut max_abs = 0.0f64;
    for &v in data {
        let v = f64::from(v);
        sum += v;
        sumsq += v * v;
        if v.abs() > max_abs {
            max_abs = v.abs();
        }
    }
    let n = data.len() as f64;
    let mean = sum / n;
    // E[x^2] - mean^2 can go negligibly negative on constant input;
    // clamp at zero rather than emit a negative variance.
    let var = (sumsq / n - mean * mean).max(0.0);
    VecStats {
        n: data.len(),
        mean,
        var,
        max_abs,
        first: data.iter().take(first_n).copied().collect(),
    }
}

/// A receiver for forward-pass stage captures. Implementations must
/// be cheap or document their cost: the forward calls
/// [`TraceSink::capture`] once per stage boundary.
pub trait TraceSink {
    /// One stage boundary: `stage` names the point in the pass,
    /// `tensor` is the row-major `rows x cols` activation there.
    fn capture(&mut self, stage: &str, tensor: &[f32], rows: usize, cols: usize);
}

/// The default sink: drops every capture. The untraced
/// [`GlimmerModel::forward`](crate::GlimmerModel::forward) runs
/// through this sink, so the traced and untraced paths are the same
/// code.
#[derive(Debug, Default)]
pub struct NullTrace;

impl TraceSink for NullTrace {
    fn capture(&mut self, _stage: &str, _tensor: &[f32], _rows: usize, _cols: usize) {}
}

/// One reduced capture: stage name, shape, and the statistics of
/// the whole tensor and of its last row.
#[derive(Debug, Clone, PartialEq)]
pub struct TraceRecord {
    /// Stage name as captured (for example `layer.17.out`).
    pub stage: String,
    /// Row count of the captured tensor.
    pub rows: usize,
    /// Column count of the captured tensor.
    pub cols: usize,
    /// Statistics over the whole tensor.
    pub full: VecStats,
    /// Statistics over the last row (the final position).
    pub last_row: VecStats,
}

impl TraceRecord {
    /// This record in the stable version-1 line format documented
    /// at the top of this module.
    pub fn to_line(&self) -> String {
        format!(
            "trace v=1 stage={} rows={} cols={} n={} mean={:.6e} var={:.6e} max_abs={:.6e} \
             last_mean={:.6e} last_var={:.6e} last_max_abs={:.6e} first=[{}] last_first=[{}]",
            self.stage,
            self.rows,
            self.cols,
            self.full.n,
            self.full.mean,
            self.full.var,
            self.full.max_abs,
            self.last_row.mean,
            self.last_row.var,
            self.last_row.max_abs,
            format_first(&self.full.first),
            format_first(&self.last_row.first),
        )
    }
}

fn format_first(values: &[f32]) -> String {
    values
        .iter()
        .map(|v| format!("{v:.6}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// A collecting [`TraceSink`]: reduces every capture on arrival and
/// keeps the records in capture order. This is the sink the
/// layer-trace tooling dumps.
#[derive(Debug, Default)]
pub struct TraceLog {
    first_n: usize,
    records: Vec<TraceRecord>,
}

impl TraceLog {
    /// A log keeping the first `first_n` values of every tensor and
    /// of every last row.
    pub fn new(first_n: usize) -> Self {
        TraceLog {
            first_n,
            records: Vec::new(),
        }
    }

    /// The records so far, in capture order.
    pub fn records(&self) -> &[TraceRecord] {
        &self.records
    }

    /// All records in the stable line format, one per line.
    pub fn lines(&self) -> Vec<String> {
        self.records.iter().map(TraceRecord::to_line).collect()
    }
}

impl TraceSink for TraceLog {
    fn capture(&mut self, stage: &str, tensor: &[f32], rows: usize, cols: usize) {
        let last_row: &[f32] = if cols > 0 && tensor.len() >= cols {
            &tensor[tensor.len() - cols..]
        } else {
            &[]
        };
        self.records.push(TraceRecord {
            stage: stage.to_string(),
            rows,
            cols,
            full: stats_of(tensor, self.first_n),
            last_row: stats_of(last_row, self.first_n),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_of_a_known_vector() {
        // [1, 2, 3, 4]: mean 2.5, population variance 1.25, max 4.
        let s = stats_of(&[1.0, 2.0, 3.0, 4.0], 2);
        assert_eq!(s.n, 4);
        assert!((s.mean - 2.5).abs() < 1e-12);
        assert!((s.var - 1.25).abs() < 1e-12);
        assert!((s.max_abs - 4.0).abs() < 1e-12);
        assert_eq!(s.first, vec![1.0, 2.0]);
    }

    #[test]
    fn stats_of_negative_and_empty_inputs() {
        let s = stats_of(&[-3.0, 1.0], 8);
        assert_eq!(s.n, 2);
        assert!((s.max_abs - 3.0).abs() < 1e-12);
        // "first" saturates at the available length, never pads.
        assert_eq!(s.first, vec![-3.0, 1.0]);

        let empty = stats_of(&[], 8);
        assert_eq!(empty.n, 0);
        assert_eq!(empty.mean, 0.0);
        assert_eq!(empty.var, 0.0);
        assert!(empty.first.is_empty());

        // Constant input has exactly zero variance (no negative
        // clamp artifact leaking through).
        let flat = stats_of(&[2.0, 2.0, 2.0], 1);
        assert_eq!(flat.var, 0.0);
    }

    #[test]
    fn record_line_format_is_stable() {
        let record = TraceRecord {
            stage: "embed".to_string(),
            rows: 2,
            cols: 2,
            full: stats_of(&[1.0, 2.0, 3.0, 4.0], 2),
            last_row: stats_of(&[3.0, 4.0], 2),
        };
        assert_eq!(
            record.to_line(),
            "trace v=1 stage=embed rows=2 cols=2 n=4 mean=2.500000e0 var=1.250000e0 \
             max_abs=4.000000e0 last_mean=3.500000e0 last_var=2.500000e-1 \
             last_max_abs=4.000000e0 first=[1.000000,2.000000] last_first=[3.000000,4.000000]"
        );
    }

    #[test]
    fn trace_log_reduces_and_keeps_capture_order() {
        let mut log = TraceLog::new(1);
        log.capture("embed", &[1.0, 2.0, 3.0, 4.0], 2, 2);
        log.capture("logits", &[0.5, -0.5], 1, 2);
        let records = log.records();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].stage, "embed");
        // Last row of a 2x2 tensor is its second half.
        assert_eq!(records[0].last_row.first, vec![3.0]);
        assert!((records[0].last_row.mean - 3.5).abs() < 1e-12);
        assert_eq!(records[1].stage, "logits");
        assert_eq!(records[1].rows, 1);
        assert_eq!(log.lines().len(), 2);
    }

    #[test]
    fn trace_log_survives_a_degenerate_shape() {
        // A capture whose cols exceed the tensor must not panic;
        // the last-row view degrades to empty statistics.
        let mut log = TraceLog::new(4);
        log.capture("odd", &[1.0], 1, 8);
        let record = &log.records()[0];
        assert_eq!(record.full.n, 1);
        assert_eq!(record.last_row.n, 0);
    }
}
