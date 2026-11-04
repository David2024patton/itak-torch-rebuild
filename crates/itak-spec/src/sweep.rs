//! The draft-depth sweep (TODO T-9.7): run the speculative loop once
//! per candidate depth, log acceptance per depth, and name the winning
//! depth as the recorded default.
//!
//! Everything operational here is configuration with a default, never a
//! compiled-in constant (David's law): the candidate set and the
//! per-depth token budget live in [`SweepConfig`], and the depth each
//! session runs at is [`crate::SpecConfig::depth`], the same field
//! serving uses. The sweep only ever overrides that field on a clone of
//! the caller's base configuration.
//!
//! The runner is backend-agnostic by construction. It names no compute
//! backend and no model type: the caller hands it a factory that builds
//! a ready [`SpecSession`] for a given configuration, and the runner
//! drives `begin` / `generate` / `finish` and does the accounting. The
//! CPU example (`examples/spec_sweep.rs`) wires the factory to the CPU
//! reference backend today; when the GPU-resident loop lands (TODO
//! T-9.6's speed half), its lane wires the same runner to a factory
//! that builds the session over the GPU compute path, and the sweep,
//! the table, and the winner rule do not change.
//!
//! The measured wall is the `generate` call only: model loads and the
//! prompt prefill are excluded, so effective tok/s compares depths, not
//! disk speed. A depth whose session stops early (a stop token in the
//! base configuration) is measured on the tokens it actually emitted;
//! the row says how many, so a short row is never mistaken for a full
//! one.

use std::fmt;
use std::time::Instant;

use crate::config::SpecConfig;
use crate::diag::Diag;
use crate::driver::TargetDriver;
use crate::error::{Result, SpecError};
use crate::session::{SpecSession, SpecStats};

/// The default candidate depths, ascending: exactly the set the
/// 2026-10-30 llama.cpp sweep measured (`bench/gpu-proof-2026-10-30.
/// md`), from the floor to the real draft's block size (16) and
/// including 6, the standing default that sweep picked. Keeping the
/// set identical makes the two tables comparable cell for cell. A
/// default, not a constant: any caller can replace the set wholesale
/// through [`SweepConfig`].
pub const DEFAULT_CANDIDATE_DEPTHS: &[usize] = &[1, 2, 3, 4, 6, 8, 12, 16];

/// The default winner band: the recorded operating-point rule (SPEC
/// 9.4, applied in `bench/gpu-proof-2026-10-30.md`) picks the smallest
/// depth within 2% of the best measured speed, because the speed curve
/// is flat near its peak and the shallower round is the cheaper one to
/// be wrong about.
pub const DEFAULT_WINNER_BAND: f64 = 0.02;

/// The default generation budget per depth: enough rounds at every
/// candidate depth for the acceptance rate to settle, small enough for
/// a bench run to finish.
pub const DEFAULT_TOKENS_PER_DEPTH: usize = 128;

/// What a sweep measures: the candidate depths and the per-depth
/// generation budget. Validated before anything runs
/// ([`SweepConfig::validate`]); an unusable configuration refuses by
/// name instead of producing an empty table.
#[derive(Debug, Clone, PartialEq)]
pub struct SweepConfig {
    /// Candidate draft depths, in the order they are measured. Each
    /// must be at least 1 and appear at most once. A depth beyond the
    /// draft's block size is not refused here (the sweep does not know
    /// the draft); session creation refuses it by name when its turn
    /// comes, and the sweep stops there.
    pub depths: Vec<usize>,
    /// Tokens generated per depth (the `generate` budget handed to
    /// each session). Must be at least 1.
    pub tokens_per_depth: usize,
    /// The winner band, as a fraction in `[0, 1)`: the chosen default
    /// is the smallest depth whose effective tok/s is within this
    /// fraction of the best depth's. Default: 0.02 (SPEC 9.4's rule).
    pub winner_band: f64,
}

impl Default for SweepConfig {
    fn default() -> Self {
        SweepConfig {
            depths: DEFAULT_CANDIDATE_DEPTHS.to_vec(),
            tokens_per_depth: DEFAULT_TOKENS_PER_DEPTH,
            winner_band: DEFAULT_WINNER_BAND,
        }
    }
}

impl SweepConfig {
    /// Parse a comma-separated candidate list (`"1,2,4,6"`), whitespace
    /// tolerant. Refuses by name: an empty list or entry, a non-numeric
    /// entry, a zero depth, or a duplicate.
    pub fn parse_depths(text: &str) -> Result<Vec<usize>> {
        let invalid = |detail: String| SpecError::SweepConfigInvalid { detail };
        let mut depths = Vec::new();
        for part in text.split(',') {
            let part = part.trim();
            if part.is_empty() {
                return Err(invalid(format!("depth list {text:?} has an empty entry")));
            }
            let depth: usize = part
                .parse()
                .map_err(|_| invalid(format!("depth {part:?} in {text:?} is not a number")))?;
            depths.push(depth);
        }
        let config = SweepConfig {
            depths,
            ..SweepConfig::default()
        };
        config.validate()?;
        Ok(config.depths)
    }

    /// Check the configuration is measurable: a non-empty candidate
    /// set of positive, distinct depths and a positive token budget.
    pub fn validate(&self) -> Result<()> {
        let invalid = |detail: String| SpecError::SweepConfigInvalid { detail };
        if self.depths.is_empty() {
            return Err(invalid("candidate depth set is empty".to_string()));
        }
        if self.tokens_per_depth == 0 {
            return Err(invalid("tokens_per_depth is zero".to_string()));
        }
        if !(0.0..1.0).contains(&self.winner_band) {
            return Err(invalid(format!(
                "winner_band {} is outside [0, 1)",
                self.winner_band
            )));
        }
        for (i, &depth) in self.depths.iter().enumerate() {
            if depth == 0 {
                return Err(invalid("candidate depth 0 is not a depth".to_string()));
            }
            if self.depths[..i].contains(&depth) {
                return Err(invalid(format!("candidate depth {depth} appears twice")));
            }
        }
        Ok(())
    }
}

/// One measured depth: the session's own cumulative statistics plus the
/// generate wall they were produced under. All rates derive from the
/// counts, so a row can be rebuilt from a [`SpecStats`] and a wall time
/// without running anything (which is how the accounting is tested).
#[derive(Debug, Clone, PartialEq)]
pub struct DepthRow {
    /// The draft depth this row measured.
    pub depth: usize,
    /// Rounds the session ran.
    pub rounds: usize,
    /// Draft tokens proposed across all rounds.
    pub proposed: usize,
    /// Draft tokens accepted across all rounds.
    pub accepted: usize,
    /// `accepted / proposed`, in `[0, 1]`; zero when nothing proposed.
    pub acceptance_rate: f64,
    /// Round-emitted tokens per round.
    pub tokens_per_round: f64,
    /// Tokens emitted by rounds (the prefill's pending token excluded,
    /// exactly as [`SpecStats`] counts).
    pub tokens_emitted: usize,
    /// Wall seconds of the `generate` call (loads and prefill excluded).
    pub wall_secs: f64,
    /// `tokens_emitted / wall_secs`: the depth's effective speed.
    pub tok_per_sec: f64,
    /// Whether speculation latched off during this depth's session.
    pub fallback_latched: bool,
}

impl DepthRow {
    /// Build a row from a finished session's statistics and the wall
    /// time its generation took.
    pub fn from_stats(depth: usize, stats: &SpecStats, wall_secs: f64) -> DepthRow {
        let tok_per_sec = if wall_secs > 0.0 {
            stats.tokens_emitted as f64 / wall_secs
        } else {
            0.0
        };
        DepthRow {
            depth,
            rounds: stats.rounds,
            proposed: stats.proposed,
            accepted: stats.accepted,
            acceptance_rate: stats.acceptance_rate(),
            tokens_per_round: stats.tokens_per_round(),
            tokens_emitted: stats.tokens_emitted,
            wall_secs,
            tok_per_sec,
            fallback_latched: stats.fallback_latched,
        }
    }
}

/// The outcome of a sweep: one row per measured depth, in the order
/// they were measured, plus the winner band they were judged under
/// (carried from the [`SweepConfig`] that ran, so a filed report can
/// be re-judged by hand).
#[derive(Debug, Clone, PartialEq)]
pub struct SweepReport {
    /// The measured rows.
    pub rows: Vec<DepthRow>,
    /// The winner band fraction the chosen default is picked under.
    pub winner_band: f64,
}

impl Default for SweepReport {
    fn default() -> Self {
        SweepReport {
            rows: Vec::new(),
            winner_band: DEFAULT_WINNER_BAND,
        }
    }
}

impl SweepReport {
    /// The winning depth's row, by the recorded operating-point rule
    /// (SPEC 9.4): the smallest depth whose effective tok/s is within
    /// `winner_band` of the best depth's. The speed curve is flat near
    /// its peak, so the band keeps clock jitter from crowning a deep,
    /// expensive round over a shallow one that measured the same.
    /// `None` when no depth was measured.
    pub fn winner(&self) -> Option<&DepthRow> {
        let best = self
            .rows
            .iter()
            .map(|row| row.tok_per_sec)
            .fold(0.0_f64, f64::max);
        let floor = best * (1.0 - self.winner_band);
        self.rows
            .iter()
            .filter(|row| row.tok_per_sec >= floor)
            .min_by_key(|row| row.depth)
    }

    /// The winning depth, if any depth was measured.
    pub fn chosen_depth(&self) -> Option<usize> {
        self.winner().map(|row| row.depth)
    }

    /// The sweep table as markdown, in the shape the T-9.7 bench note
    /// files: one row per depth, then the chosen-default line naming
    /// the depth to record in [`crate::SpecConfig`] with its measured
    /// acceptance rate.
    pub fn render_table(&self) -> String {
        let mut out = String::new();
        out.push_str(
            "| Depth | Rounds | Proposed | Accepted | Acceptance rate | Tokens/round | Emitted | Wall (s) | Effective tok/s |\n",
        );
        out.push_str(
            "|------:|-------:|---------:|---------:|----------------:|-------------:|--------:|---------:|----------------:|\n",
        );
        for row in &self.rows {
            out.push_str(&format!(
                "| {} | {} | {} | {} | {:.3} | {:.2} | {} | {:.1} | {:.2} |\n",
                row.depth,
                row.rounds,
                row.proposed,
                row.accepted,
                row.acceptance_rate,
                row.tokens_per_round,
                row.tokens_emitted,
                row.wall_secs,
                row.tok_per_sec
            ));
        }
        match self.winner() {
            Some(w) => out.push_str(&format!(
                "\nChosen default depth: **{}** (acceptance rate {:.3}, effective {:.2} tok/s).\n",
                w.depth, w.acceptance_rate, w.tok_per_sec
            )),
            None => {
                out.push_str("\nChosen default depth: none (no depths measured).\n");
            }
        }
        out
    }
}

/// Why a sweep stopped, named. The loop's own refusals keep their
/// [`SpecError`] name; a factory that cannot build a session for a
/// depth (a model load, a backend init) reports in its own words, and
/// the sweep adds the depth it stopped at.
#[derive(Debug)]
pub enum SweepError {
    /// The loop refused: bad sweep configuration, a session creation
    /// refusal, or a round refusal.
    Spec(SpecError),
    /// The session factory failed to build for this depth.
    Build {
        /// The depth whose session never existed.
        depth: usize,
        /// The factory's own description of the failure.
        detail: String,
    },
}

impl fmt::Display for SweepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SweepError::Spec(err) => write!(f, "{err}"),
            SweepError::Build { depth, detail } => {
                write!(f, "session build for depth {depth} failed: {detail}")
            }
        }
    }
}

impl std::error::Error for SweepError {}

impl From<SpecError> for SweepError {
    fn from(err: SpecError) -> Self {
        SweepError::Spec(err)
    }
}

/// Run the sweep: for each candidate depth, build a session from
/// `factory` (handed the base configuration with only its depth
/// overridden), prefill `prompt`, generate `tokens_per_depth` tokens
/// under a wall clock, and fold the finished session's statistics into
/// a row.
///
/// The first refusal stops the sweep and is returned by name: a table
/// with a silent hole in it would be a lie about the depths it skipped.
/// Every step emits through `diag` (`spec.sweep.begin`, one
/// `spec.sweep.row` per depth, `spec.sweep.end`, and `spec.sweep.
/// refused` on the way out).
pub fn run_sweep<D, F>(
    sweep: &SweepConfig,
    base: &SpecConfig,
    prompt: &[u32],
    mut factory: F,
    diag: &dyn Diag,
) -> std::result::Result<SweepReport, SweepError>
where
    D: TargetDriver,
    F: FnMut(&SpecConfig) -> std::result::Result<SpecSession<D>, String>,
{
    if let Err(err) = sweep.validate() {
        diag.event("spec.sweep.refused", &format!("error={err}"));
        return Err(err.into());
    }
    diag.event(
        "spec.sweep.begin",
        &format!(
            "depths={:?} tokens_per_depth={} prompt_tokens={}",
            sweep.depths,
            sweep.tokens_per_depth,
            prompt.len()
        ),
    );
    let mut rows = Vec::with_capacity(sweep.depths.len());
    for &depth in &sweep.depths {
        let mut config = base.clone();
        config.depth = depth;
        let row = measure_depth(&mut factory, &config, prompt, sweep.tokens_per_depth);
        match row {
            Ok(row) => {
                diag.event(
                    "spec.sweep.row",
                    &format!(
                        "depth={} rounds={} proposed={} accepted={} acceptance_rate={:.4} tokens_per_round={:.2} wall_s={:.3} tok_per_sec={:.2} fallback_latched={}",
                        row.depth,
                        row.rounds,
                        row.proposed,
                        row.accepted,
                        row.acceptance_rate,
                        row.tokens_per_round,
                        row.wall_secs,
                        row.tok_per_sec,
                        row.fallback_latched
                    ),
                );
                rows.push(row);
            }
            Err(err) => {
                diag.event("spec.sweep.refused", &format!("depth={depth} error={err}"));
                return Err(err);
            }
        }
    }
    let report = SweepReport {
        rows,
        winner_band: sweep.winner_band,
    };
    match report.winner() {
        Some(w) => diag.event(
            "spec.sweep.end",
            &format!(
                "chosen_depth={} acceptance_rate={:.4} tok_per_sec={:.2}",
                w.depth, w.acceptance_rate, w.tok_per_sec
            ),
        ),
        None => diag.event("spec.sweep.end", "chosen_depth=none"),
    }
    Ok(report)
}

/// Build, run, and account one depth. Split from [`run_sweep`] so the
/// per-depth failure handling reads as one named refusal path.
fn measure_depth<D, F>(
    factory: &mut F,
    config: &SpecConfig,
    prompt: &[u32],
    tokens: usize,
) -> std::result::Result<DepthRow, SweepError>
where
    D: TargetDriver,
    F: FnMut(&SpecConfig) -> std::result::Result<SpecSession<D>, String>,
{
    let mut session = factory(config).map_err(|detail| SweepError::Build {
        depth: config.depth,
        detail,
    })?;
    session.begin(prompt)?;
    let started = Instant::now();
    session.generate(tokens, &[])?;
    let wall_secs = started.elapsed().as_secs_f64();
    let stats = session.finish();
    Ok(DepthRow::from_stats(config.depth, &stats, wall_secs))
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};

    use cpu_compute::CpuCompute;

    use super::*;
    use crate::diag::Diag;
    use crate::speculator::SharedCompute;
    use crate::testkit;
    use crate::NullDiag;

    const PROMPT: &[u32] = &[3, 17, 42];

    /// A diag sink that keeps every event, so tests can prove the
    /// sweep's evidence actually crossed the seam.
    #[derive(Debug, Default)]
    struct RecDiag(Mutex<Vec<String>>);

    impl Diag for RecDiag {
        fn event(&self, name: &str, detail: &str) {
            self.0
                .lock()
                .expect("recording diag lock")
                .push(format!("{name} {detail}"));
        }
    }

    impl RecDiag {
        fn saw(&self, needle: &str) -> bool {
            self.0
                .lock()
                .expect("recording diag lock")
                .iter()
                .any(|line| line.contains(needle))
        }
    }

    /// The CPU-fixture factory: the same wiring the example uses for
    /// the real files, over the synthetic geometry (draft block 16, so
    /// every candidate depth up to the default set's max is legal).
    fn fixture_factory(config: &SpecConfig) -> std::result::Result<SpecSession, String> {
        Ok(testkit::build_session(config.clone(), 16))
    }

    /// The fallible twin of [`fixture_factory`]: session creation
    /// refusals surface as the factory's error text instead of the
    /// testkit builder's expect, so a test can watch the sweep stop.
    fn try_fixture_factory(config: &SpecConfig) -> std::result::Result<SpecSession, String> {
        let compute = Rc::new(CpuCompute::new());
        let fixture = testkit::build_target(Box::new(SharedCompute(compute.clone())));
        let speculator = testkit::build_speculator(&fixture, 16);
        SpecSession::new(
            crate::driver::CpuTarget::new(fixture.model, compute),
            Some(speculator),
            config.clone(),
            Arc::new(NullDiag),
        )
        .map_err(|e| e.to_string())
    }

    fn fabricated_row(depth: usize, tok_per_sec: f64, acceptance_rate: f64) -> DepthRow {
        DepthRow {
            depth,
            rounds: 10,
            proposed: 100,
            accepted: (acceptance_rate * 100.0) as usize,
            acceptance_rate,
            tokens_per_round: 5.0,
            tokens_emitted: 50,
            wall_secs: 1.0,
            tok_per_sec,
            fallback_latched: false,
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn default_config_is_valid_and_covers_the_operating_point() {
        let config = SweepConfig::default();
        config
            .validate()
            .expect("the default sweep config validates");
        assert!(config.depths.contains(&6), "the standing default is swept");
        assert!(config.depths.contains(&16), "the draft block max is swept");
        assert_eq!(config.tokens_per_depth, DEFAULT_TOKENS_PER_DEPTH);
    }

    #[test]
    fn parse_depths_reads_lists() {
        assert_eq!(
            SweepConfig::parse_depths("1,2,4,6").expect("list parses"),
            vec![1, 2, 4, 6]
        );
        assert_eq!(
            SweepConfig::parse_depths(" 6 ").expect("single parses"),
            vec![6]
        );
        assert_eq!(
            SweepConfig::parse_depths("1, 2 , 12").expect("spaces parse"),
            vec![1, 2, 12]
        );
    }

    #[test]
    fn parse_depths_refuses_bad_input() {
        for text in ["", " ", "0", "3,0", "x", "1,,2", "4,4", "1,", "-3", "2.5"] {
            let err = SweepConfig::parse_depths(text).expect_err(&format!("{text:?} must refuse"));
            assert!(
                matches!(err, SpecError::SweepConfigInvalid { .. }),
                "{text:?} refused with {err:?}, want SweepConfigInvalid"
            );
        }
    }

    #[test]
    fn validate_refuses_empty_zero_duplicate_and_zero_budget() {
        let cases = [
            SweepConfig {
                depths: vec![],
                tokens_per_depth: 8,
                winner_band: DEFAULT_WINNER_BAND,
            },
            SweepConfig {
                depths: vec![0],
                tokens_per_depth: 8,
                winner_band: DEFAULT_WINNER_BAND,
            },
            SweepConfig {
                depths: vec![2, 2],
                tokens_per_depth: 8,
                winner_band: DEFAULT_WINNER_BAND,
            },
            SweepConfig {
                depths: vec![2],
                tokens_per_depth: 0,
                winner_band: DEFAULT_WINNER_BAND,
            },
            SweepConfig {
                depths: vec![2],
                tokens_per_depth: 8,
                winner_band: 1.0,
            },
        ];
        for config in cases {
            assert!(
                matches!(config.validate(), Err(SpecError::SweepConfigInvalid { .. })),
                "{config:?} must refuse validation"
            );
        }
    }

    #[test]
    fn row_math_comes_from_the_counts() {
        let stats = SpecStats {
            rounds: 4,
            proposed: 24,
            accepted: 9,
            rejections: 3,
            rollbacks: 3,
            tokens_emitted: 13,
            fallback_latched: false,
        };
        let row = DepthRow::from_stats(6, &stats, 2.0);
        assert_eq!(row.depth, 6);
        assert!(close(row.acceptance_rate, 0.375), "9/24 = 0.375");
        assert!(close(row.tokens_per_round, 3.25), "13/4 = 3.25");
        assert!(close(row.tok_per_sec, 6.5), "13 tokens over 2 s = 6.5");
        let idle = DepthRow::from_stats(1, &SpecStats::default(), 0.0);
        assert!(close(idle.acceptance_rate, 0.0), "no proposals, no rate");
        assert!(close(idle.tok_per_sec, 0.0), "no wall, no speed");
    }

    #[test]
    fn winner_follows_the_spec_9_4_band_rule() {
        // Outright speed: depth 2 sits outside the 2% band under the
        // best (60 * 0.98 = 58.8), so the fast deep row wins despite
        // its poor acceptance.
        let report = SweepReport {
            rows: vec![
                fabricated_row(2, 50.0, 0.90),
                fabricated_row(16, 60.0, 0.10),
            ],
            winner_band: DEFAULT_WINNER_BAND,
        };
        assert_eq!(
            report.chosen_depth(),
            Some(16),
            "outside the band, best speed wins"
        );

        // Inside the band: 49.5 is within 2% of 50 (floor 49.0), so
        // the smallest depth in the band is the pick.
        let report = SweepReport {
            rows: vec![fabricated_row(8, 50.0, 0.90), fabricated_row(4, 49.5, 0.80)],
            winner_band: DEFAULT_WINNER_BAND,
        };
        assert_eq!(
            report.chosen_depth(),
            Some(4),
            "inside the band, the smallest depth wins"
        );

        // Just outside the band: 48.0 < 49.0, so depth 4 does not
        // qualify and the best row stands.
        let report = SweepReport {
            rows: vec![fabricated_row(8, 50.0, 0.90), fabricated_row(4, 48.0, 0.95)],
            winner_band: DEFAULT_WINNER_BAND,
        };
        assert_eq!(report.chosen_depth(), Some(8), "outside the band, no steal");

        assert_eq!(SweepReport::default().chosen_depth(), None);
    }

    #[test]
    fn table_rendering_has_the_bench_note_shape() {
        let report = SweepReport {
            rows: vec![fabricated_row(4, 50.0, 0.9), fabricated_row(6, 60.0, 0.5)],
            winner_band: DEFAULT_WINNER_BAND,
        };
        let table = report.render_table();
        assert!(table.contains("| Depth | Rounds | Proposed | Accepted |"));
        assert_eq!(
            table.lines().filter(|l| l.starts_with('|')).count(),
            4,
            "header, separator, and one line per depth"
        );
        assert!(
            table.contains("Chosen default depth: **6**"),
            "the winner line names the chosen depth: {table}"
        );
        let empty = SweepReport::default().render_table();
        assert!(empty.contains("Chosen default depth: none"));
    }

    #[test]
    fn sweep_matches_direct_session_accounting() {
        let sweep = SweepConfig {
            depths: vec![1, 2],
            tokens_per_depth: 12,
            winner_band: DEFAULT_WINNER_BAND,
        };
        let base = SpecConfig::default();
        let diag = RecDiag::default();
        let report = run_sweep(&sweep, &base, PROMPT, fixture_factory, &diag)
            .expect("the fixture sweep runs");
        assert_eq!(report.rows.len(), 2);
        for row in &report.rows {
            // The same session, run directly at the same configuration,
            // must produce the identical counts: the sweep's accounting
            // is the session's accounting, not a second set of books.
            let mut config = base.clone();
            config.depth = row.depth;
            let mut session = testkit::build_session(config, 16);
            session.begin(PROMPT).expect("direct begin");
            session
                .generate(sweep.tokens_per_depth, &[])
                .expect("direct generate");
            let stats = session.finish();
            assert_eq!(row.rounds, stats.rounds, "depth {} rounds", row.depth);
            assert_eq!(row.proposed, stats.proposed, "depth {} proposed", row.depth);
            assert_eq!(row.accepted, stats.accepted, "depth {} accepted", row.depth);
            assert_eq!(
                row.tokens_emitted, stats.tokens_emitted,
                "depth {} emitted",
                row.depth
            );
            assert!(
                close(row.acceptance_rate, stats.acceptance_rate()),
                "depth {} acceptance rate is the stats rate",
                row.depth
            );
            assert!(row.wall_secs > 0.0, "a real wall was measured");
            assert!(row.tok_per_sec > 0.0, "tokens moved, so speed is real");
        }
        assert!(diag.saw("spec.sweep.begin"));
        assert!(diag.saw("spec.sweep.row depth=1"));
        assert!(diag.saw("spec.sweep.row depth=2"));
        assert!(diag.saw("spec.sweep.end chosen_depth="));
    }

    #[test]
    fn sweep_refuses_a_depth_beyond_the_draft_block_by_name() {
        let sweep = SweepConfig {
            depths: vec![99],
            tokens_per_depth: 4,
            winner_band: DEFAULT_WINNER_BAND,
        };
        let diag = RecDiag::default();
        let err = run_sweep(
            &sweep,
            &SpecConfig::default(),
            PROMPT,
            try_fixture_factory,
            &diag,
        )
        .expect_err("depth 99 cannot build against a block of 16");
        match err {
            SweepError::Build { depth, detail } => {
                assert_eq!(depth, 99);
                assert!(
                    detail.contains("draft depth 99 is out of range"),
                    "the session's named refusal survives the factory: {detail}"
                );
            }
            other => panic!("want a Build refusal for depth 99, got {other}"),
        }
        assert!(diag.saw("spec.sweep.refused depth=99"));
    }

    #[test]
    fn sweep_reports_a_factory_failure_with_its_depth() {
        let sweep = SweepConfig {
            depths: vec![3],
            tokens_per_depth: 4,
            winner_band: DEFAULT_WINNER_BAND,
        };
        let diag = RecDiag::default();
        let err = run_sweep(
            &sweep,
            &SpecConfig::default(),
            PROMPT,
            |_| -> std::result::Result<SpecSession, String> {
                Err("no backend in this test".to_string())
            },
            &diag,
        )
        .expect_err("a factory failure stops the sweep");
        match err {
            SweepError::Build { depth, detail } => {
                assert_eq!(depth, 3);
                assert!(detail.contains("no backend"));
            }
            other => panic!("want a Build refusal, got {other}"),
        }
        assert!(diag.saw("spec.sweep.refused depth=3"));
    }
}
