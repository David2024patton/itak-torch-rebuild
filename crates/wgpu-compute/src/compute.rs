//! [`WgpuCompute`]: the wgpu backend behind `muse_glimmer::Compute`.
//!
//! What runs on the device: `proj` (and through it `lm_head`)
//! for Q4_K, Q6_K, and Q5_K tensors, through kernels K1, K2,
//! and K3 respectively, selected by the tensor's dtype; the
//! norms and the residual add through K4; `swiglu_ffn` end to
//! end (K1 gate and up, K7 activation, K2 down) with
//! `attn_gate` through K7; and the embedding gather through
//! K8. The one step that still refuses, by name, is
//! `attention`: the K5 (RoPE plus KV append) and K6 (scores
//! plus reduce) pipelines are built, but their composition
//! over this backend (device KV residency mirroring
//! `cpu-compute`'s backend-owned storage, behind the `Cache`
//! trait's position contract) is not wired yet. Nothing is
//! silently substituted: BACKEND.md Section 6 puts per-op CPU
//! fallback at load time as the caller's disclosed composition
//! decision, so this backend never detours a step to the CPU
//! behind the caller's back.
//!
//! Data path, stated plainly: weight tensors upload once and are
//! cached by name (per-tensor device buffers, BACKEND.md
//! Section 5); each `proj` call uploads its input rows and reads
//! its output rows back, because the trait's currency is host
//! slices. When the forward itself becomes device-resident, the
//! per-call transfers are what go away; the kernels and the cache
//! stay.

use std::cell::RefCell;
use std::collections::HashMap;

use muse_glimmer::{
    AttentionRequest, Cache, Compute, Dtype, GlimmerConfig, GlimmerError, LayerKind, LayerWeights,
    Result, TensorView,
};

use crate::detect::{detect, DetectPolicy};
use crate::device::GpuContext;
use crate::kernels::k10_q4k_gemm::K10Pipeline;
use crate::kernels::k11_q6k_gemm::K11Pipeline;
use crate::kernels::k12_q8_quant::K12QuantPipeline;
use crate::kernels::k13_q4k_mmq::K13Pipeline;
use crate::kernels::k14_q6k_mmq::K14Pipeline;
use crate::kernels::k15_q4k_mmq_wide::K15Pipeline;
use crate::kernels::k16_q6k_mmq_wide::K16Pipeline;
use crate::kernels::k17_dflash::K17Pipeline;
use crate::kernels::k18_small_batch::{K18Q4KPipeline, K18Q5KPipeline, K18Q6KPipeline};
use crate::kernels::k19_q4k_mmv::{K19Pipeline, MMV_MAX_BATCH, MMV_MIN_BATCH};
use crate::kernels::k1_q4k_gemv::{K1Pipeline, K1Weights, ACT_GATED, ACT_NONE, ACT_SWIGLU};
use crate::kernels::k20_q6k_mmv::K20Pipeline;
use crate::kernels::k21_q5k_mmv::K21Pipeline;
use crate::kernels::k21_q5k_mmv_finish::K21FinishPipeline;
use crate::kernels::k22_q4k_mmq::K22Pipeline;
use crate::kernels::k23_q4k_coopmat::K23Pipeline as K23CoopmatPipeline;
use crate::kernels::k23_q4k_mmq::K23Pipeline;
use crate::kernels::k2_q6k_gemv::{K2Pipeline, K2Weights};
use crate::kernels::k3_q5k_gemv::{K3Pipeline, K3Weights};
use crate::kernels::k4_rmsnorm::K4Pipeline;
use crate::kernels::k5_rope_kv::K5Pipeline;
use crate::kernels::k6_attention::{AttendGeom, GpuKvState, K6Pipeline};
use crate::kernels::k7_swiglu::K7Pipeline;
use crate::kernels::k8_head::K8Pipeline;
use crate::kernels::k9_attn_prep::K9Pipeline;

use crate::diag::dbg_log;

/// Environment variable naming the wide-MMQ row threshold
/// override: the chunk row count at or past which the
/// session prefers the wide-tile double-buffered MMQ
/// products (K15/K16) over the standard ones (K13/K14).
/// Operational config, not a compiled constant: set it
/// past the batch cap (64) to take the wide form out of
/// the ladder entirely (the A/B control position the
/// prefill bench script uses), or lower it to widen the
/// wide form's reach. An unparsable value falls back to
/// [`DEFAULT_WIDE_MIN_ROWS`], with the fallback logged.
pub const WIDE_MIN_ROWS_ENV: &str = "WGPU_COMPUTE_MMQ_WIDE_MIN_ROWS";

/// The wide-MMQ row threshold when the environment does
/// not override it: past the chunk cap, so the wide form
/// is OUT of the ladder by default. The server proof of
/// 2026-11-04 (bench/mmq-wide-tile-2026-11-03.md) found
/// the wide kernels numerically correct but about 3.5x
/// slower per tensor than the standard MMQ, and the wide
/// ladder hung the real session outright (GPU context
/// lost). The standard MMQ is the shipping position; the
/// wide form stays in the tree for future work and can be
/// opted into with the env override.
pub const DEFAULT_WIDE_MIN_ROWS: u32 = 65;

/// Resolve the wide-MMQ row threshold from the raw
/// environment value: a plain u32, or the default when
/// unset or unparsable.
fn resolve_wide_min_rows(raw: Option<&str>) -> u32 {
    match raw {
        Some(s) => s.trim().parse::<u32>().unwrap_or(DEFAULT_WIDE_MIN_ROWS),
        None => DEFAULT_WIDE_MIN_ROWS,
    }
}

/// Environment variable naming the small-batch product
/// threshold override: the largest batch row count at
/// which the session prefers the small-batch products
/// (K18) over the GEMV forms, inside the band from 2
/// rows up. Operational config, not a compiled
/// constant: set it to 15 or 16 to enable K18 across
/// its measured band (the kernel itself refuses past
/// [`crate::kernels::k18_small_batch::SMALL_MAX_BATCH`]),
/// or leave it at the default to keep K18 out of the
/// ladder entirely. An unparsable value falls back to
/// [`DEFAULT_SMALL_BATCH_MAX_ROWS`], with the fallback
/// logged at construction.
pub const SMALL_BATCH_MAX_ROWS_ENV: &str = "WGPU_COMPUTE_SMALL_BATCH_MAX_ROWS";

/// The small-batch row threshold when the environment
/// does not override it: 1, which is OFF, since the
/// ladder only routes batches of 2 or more rows to
/// K18. The shipped default follows the wide-MMQ
/// precedent: an unproven kernel does not enter the
/// ladder by default. The server proof that can move
/// this value is sequenced in
/// bench/small-batch-2026-11-04.md.
pub const DEFAULT_SMALL_BATCH_MAX_ROWS: u32 = 1;

/// Resolve the small-batch row threshold from the raw
/// environment value: a plain u32, or the default when
/// unset or unparsable.
fn resolve_small_batch_max_rows(raw: Option<&str>) -> u32 {
    match raw {
        Some(s) => s
            .trim()
            .parse::<u32>()
            .unwrap_or(DEFAULT_SMALL_BATCH_MAX_ROWS),
        None => DEFAULT_SMALL_BATCH_MAX_ROWS,
    }
}

/// Environment variable naming the register-streaming
/// small-batch (MMV) threshold override: the largest
/// batch row count at which the session prefers the
/// register-streaming products (K19 for Q4_K, K20
/// for Q6_K, K21 for Q5_K) over the K18 and GEMV
/// forms, inside the band
/// from 2 rows up. Operational config, not a compiled
/// constant: the shipped default (16) enables the MMV
/// forms across their band (each kernel itself refuses
/// past its cap); set it to 1 to keep them out of the
/// ladder entirely, or to any value in between to clip
/// the band from above. The ceiling is only
/// the outer gate: the band routing table below
/// ([`MmvBands`]) narrows it per family and shape to
/// the cells the 2026-11-06 sweep measured as wins.
/// An unparsable value falls back to the closed
/// ceiling ([`MMV_MAX_ROWS_FALLBACK`]), never to the
/// shipped default, with the resolved value logged at
/// construction.
pub const MMV_MAX_ROWS_ENV: &str = "WGPU_COMPUTE_MMV_MAX_ROWS";

/// The MMV row threshold when the environment does not
/// override it: 16, the kernels' full band, so the MMV
/// forms are in the ladder wherever the band table
/// admits the batch. This default shipped at 1 (OFF:
/// the ladder only routes batches of 2 or more rows)
/// under the wide-MMQ and K18 precedent, that an
/// unproven kernel does not enter the ladder by
/// default. The server proofs sequenced in
/// bench/mmv-prototype-2026-11-05.md (Q4_K),
/// bench/mmv-q6k-2026-11-05.md (Q6_K) and
/// bench/mmv-q5k-2026-11-05.md (Q5_K), then the band
/// table of bench/mmv-band-routing-2026-11-06.md,
/// discharged that burden, and the baseline session's
/// card protocol (bench/baseline-session-2026-11-06.md
/// leg (d)) passed at this ceiling at both depths with
/// identity, golden, and byte-identical streams in
/// every arm; the default moved to 16 on 2026-11-06.
/// Rollback is one env value:
/// `WGPU_COMPUTE_MMV_MAX_ROWS=1` restores the pre-flip
/// ladder exactly.
pub const DEFAULT_MMV_MAX_ROWS: u32 = 16;

/// The MMV ceiling a present but unparsable env value
/// resolves to: 1, closed. The fallback is deliberately
/// NOT the shipped default: a typo in the environment
/// must never widen routing, so the malformed case
/// keeps the fail-closed value it resolved to before
/// the default moved.
const MMV_MAX_ROWS_FALLBACK: u32 = 1;

/// Resolve the MMV row threshold from the raw
/// environment value: a plain u32; the shipped default
/// when unset; the closed fallback when present but
/// unparsable. Production resolution goes through
/// [`resolve_mmv_max_rows_with_profile`]; this is the
/// tests' handle on the env layer alone.
#[cfg(test)]
fn resolve_mmv_max_rows(raw: Option<&str>) -> u32 {
    resolve_mmv_max_rows_with_profile(raw, None)
}

/// Resolve the MMV row threshold with the calibration
/// profile's layer in its precedence place: an env
/// value, when present, decides on its own (a present
/// but unparsable value still falls to the closed
/// fallback, never to the profile: a typo must not
/// widen routing through any layer); with the env
/// unset, the loaded profile's ceiling applies when
/// it states one; only then the shipped default.
fn resolve_mmv_max_rows_with_profile(raw: Option<&str>, profile_ceiling: Option<u32>) -> u32 {
    match raw {
        Some(s) => s.trim().parse::<u32>().unwrap_or(MMV_MAX_ROWS_FALLBACK),
        None => profile_ceiling.unwrap_or(DEFAULT_MMV_MAX_ROWS),
    }
}

/// The quant family an MMV routing decision belongs
/// to: the register-streaming products are one per
/// family (K19 for Q4_K, K20 for Q6_K, K21 for Q5_K),
/// and the 2026-11-06 sweep measured each family's
/// cells separately, so the routing table below is
/// keyed by family first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MmvFamily {
    Q4K,
    Q5K,
    Q6K,
}

impl MmvFamily {
    /// The family's short name, as it appears in the
    /// band env names and the construction log.
    pub(crate) fn name(self) -> &'static str {
        match self {
            MmvFamily::Q4K => "q4k",
            MmvFamily::Q5K => "q5k",
            MmvFamily::Q6K => "q6k",
        }
    }

    pub(crate) fn index(self) -> usize {
        match self {
            MmvFamily::Q4K => 0,
            MmvFamily::Q5K => 1,
            MmvFamily::Q6K => 2,
        }
    }
}

/// The shape class of a projection, as the sweep
/// measured it: the three output widths at the
/// model's hidden width (attn 6656, ffn 19968,
/// kv 256). Real projections are classified by
/// relative geometry in [`mmv_shape_of`], since a
/// session sees more widths than the sweep's three
/// (the Q projection at 4096 out, the down
/// projection at 19968 in).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MmvShape {
    /// Output near the input width (attention-sized).
    Attn,
    /// Output a wide multiple of the input (FFN up).
    Ffn,
    /// Output a small fraction of the input (K/V).
    Kv,
}

impl MmvShape {
    /// The shape class's short name, as it appears
    /// in the band env names and the logs.
    pub(crate) fn name(self) -> &'static str {
        match self {
            MmvShape::Attn => "attn",
            MmvShape::Ffn => "ffn",
            MmvShape::Kv => "kv",
        }
    }

    pub(crate) fn index(self) -> usize {
        match self {
            MmvShape::Attn => 0,
            MmvShape::Ffn => 1,
            MmvShape::Kv => 2,
        }
    }
}

/// Classify a projection's geometry into the
/// sweep's shape classes: a projection whose output
/// is at most an eighth of its input width is a KV
/// projection (Glimmer: 256 out of 6656 in); one
/// whose output is at least twice its input width
/// is an FFN projection (19968 out of 6656 in);
/// everything else is attention-class. The measured
/// cells sit far from both boundaries (256 is 1/26
/// of 6656, 19968 is 3x), so every real projection
/// classifies unambiguously; a geometry the sweep
/// did not measure (the Q projection at 4096 out,
/// the down projection at 19968 in) lands in the
/// attention class, the disclosure
/// bench/mmv-band-routing-2026-11-06.md records.
pub(crate) fn mmv_shape_of(in_features: usize, out_rows: usize) -> MmvShape {
    if out_rows * 8 <= in_features {
        MmvShape::Kv
    } else if out_rows >= 2 * in_features {
        MmvShape::Ffn
    } else {
        MmvShape::Attn
    }
}

/// The env name carrying the band cell for one
/// (family, shape) pair: the batch-row ranges in
/// which that family's register-streaming product
/// serves that shape class, in the syntax
/// [`parse_mmv_band`] reads. Operational config,
/// not a compiled constant, like every routing
/// threshold in this module. Set a cell to "1" to
/// close it entirely.
fn mmv_band_env(family: MmvFamily, shape: MmvShape) -> &'static str {
    match (family, shape) {
        (MmvFamily::Q4K, MmvShape::Attn) => "WGPU_COMPUTE_MMV_Q4K_ATTN_ROWS",
        (MmvFamily::Q4K, MmvShape::Ffn) => "WGPU_COMPUTE_MMV_Q4K_FFN_ROWS",
        (MmvFamily::Q4K, MmvShape::Kv) => "WGPU_COMPUTE_MMV_Q4K_KV_ROWS",
        (MmvFamily::Q5K, MmvShape::Attn) => "WGPU_COMPUTE_MMV_Q5K_ATTN_ROWS",
        (MmvFamily::Q5K, MmvShape::Ffn) => "WGPU_COMPUTE_MMV_Q5K_FFN_ROWS",
        (MmvFamily::Q5K, MmvShape::Kv) => "WGPU_COMPUTE_MMV_Q5K_KV_ROWS",
        (MmvFamily::Q6K, MmvShape::Attn) => "WGPU_COMPUTE_MMV_Q6K_ATTN_ROWS",
        (MmvFamily::Q6K, MmvShape::Ffn) => "WGPU_COMPUTE_MMV_Q6K_FFN_ROWS",
        (MmvFamily::Q6K, MmvShape::Kv) => "WGPU_COMPUTE_MMV_Q6K_KV_ROWS",
    }
}

/// The shipped band cell for one (family, shape)
/// pair, in the same syntax the env override uses,
/// so the default and the override share one parse
/// path. Every range here is derived cell by cell
/// from the 2026-11-06 sweep table in
/// bench/mmv-band-routing-2026-11-06.md: a batch is
/// in the cell only where the sweep measured the
/// MMV form beating the GEMV control there (a tie
/// or a loss falls through to the ladder), so the
/// ranges are the measured win map and nothing
/// else.
fn default_mmv_band(family: MmvFamily, shape: MmvShape) -> &'static str {
    match (family, shape) {
        (MmvFamily::Q4K, MmvShape::Attn) => "6-12",
        (MmvFamily::Q4K, MmvShape::Ffn) => "2,4-12",
        (MmvFamily::Q4K, MmvShape::Kv) => "2-3",
        (MmvFamily::Q5K, MmvShape::Attn) => "2-16",
        (MmvFamily::Q5K, MmvShape::Ffn) => "3-16",
        (MmvFamily::Q5K, MmvShape::Kv) => "16",
        (MmvFamily::Q6K, MmvShape::Attn) => "2-12",
        (MmvFamily::Q6K, MmvShape::Ffn) => "2-12",
        (MmvFamily::Q6K, MmvShape::Kv) => "12",
    }
}

/// Parse one band cell: comma-separated inclusive
/// ranges, each a single row ("16") or a span
/// ("6-12"), whitespace ignored. Parsed ranges are
/// intersected with the kernels' own band
/// ([`MMV_MIN_BATCH`]..=[`MMV_MAX_BATCH`]); a range
/// missing the band entirely drops out, so "1" is
/// a valid cell that routes nothing. Any malformed
/// item (empty, non-numeric, a reversed span) makes
/// the whole value unparsable: `None`, and the
/// caller falls back to the shipped cell, the
/// fail-closed direction every resolver in this
/// module takes.
fn parse_mmv_band(raw: &str) -> Option<Vec<(u32, u32)>> {
    let mut ranges = Vec::new();
    for item in raw.split(',') {
        let item = item.trim();
        if item.is_empty() {
            return None;
        }
        let (lo, hi) = match item.split_once('-') {
            Some((a, b)) => (a.trim().parse::<u32>().ok()?, b.trim().parse::<u32>().ok()?),
            None => {
                let v = item.parse::<u32>().ok()?;
                (v, v)
            }
        };
        if lo > hi {
            return None;
        }
        let lo = lo.max(MMV_MIN_BATCH);
        let hi = hi.min(MMV_MAX_BATCH);
        if lo <= hi {
            ranges.push((lo, hi));
        }
    }
    Some(ranges)
}

/// The resolved MMV routing table: one band cell
/// per (family, shape) pair. The table only ever
/// narrows the global [`MMV_MAX_ROWS_ENV`] ceiling:
/// a family's product serves a projection when the
/// batch is under the ceiling AND inside the cell,
/// so the shipped ceiling default (1) keeps every
/// product out of the ladder whatever the table
/// says, and opening the ceiling admits exactly the
/// measured-win cells instead of the whole band
/// whose middle the dose-response sweep convicted
/// (bench/mmv-verdict-2026-11-05.md, step 4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MmvBands {
    /// `cells[family.index()][shape.index()]`: the
    /// inclusive batch-row ranges the family's
    /// product serves for that shape class.
    cells: [[Vec<(u32, u32)>; 3]; 3],
}

impl MmvBands {
    /// Whether the family's register-streaming
    /// product serves this shape class at `batch`
    /// rows under the resolved global `ceiling`.
    pub(crate) fn routes(
        &self,
        ceiling: u32,
        family: MmvFamily,
        shape: MmvShape,
        batch: u32,
    ) -> bool {
        batch >= MMV_MIN_BATCH
            && batch <= ceiling
            && self.cells[family.index()][shape.index()]
                .iter()
                .any(|&(lo, hi)| lo <= batch && batch <= hi)
    }

    /// The table in the band syntax, for the
    /// construction log: the effective policy is
    /// always printable from the resolved config.
    fn describe(&self) -> String {
        let mut out = String::new();
        for family in [MmvFamily::Q4K, MmvFamily::Q5K, MmvFamily::Q6K] {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(family.name());
            out.push('[');
            for (i, shape) in [MmvShape::Attn, MmvShape::Ffn, MmvShape::Kv]
                .into_iter()
                .enumerate()
            {
                if i > 0 {
                    out.push(' ');
                }
                out.push_str(shape.name());
                out.push(' ');
                out.push_str(&fmt_mmv_band(&self.cells[family.index()][shape.index()]));
            }
            out.push(']');
        }
        out
    }
}

/// One band cell in the syntax [`parse_mmv_band`]
/// reads, for logs: "6-12", "2,4-12", "16", or
/// "none" when the cell routes nothing.
fn fmt_mmv_band(ranges: &[(u32, u32)]) -> String {
    if ranges.is_empty() {
        return "none".to_string();
    }
    ranges
        .iter()
        .map(|&(lo, hi)| {
            if lo == hi {
                lo.to_string()
            } else {
                format!("{lo}-{hi}")
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// The shipped routing table: every cell at its
/// [`default_mmv_band`]. Production resolution goes
/// through [`resolve_mmv_bands`] with the real env;
/// this is the tests' handle on the defaults alone.
#[cfg(test)]
pub(crate) fn default_mmv_bands() -> MmvBands {
    resolve_mmv_bands(|_| None)
}

/// Resolve the routing table: each cell from its
/// env override when set and parsable, else the
/// shipped cell. The defaults parse through the
/// same path as an override; a shipped cell that
/// stopped parsing is a bug, and the `expect` says
/// so at construction instead of routing silently.
/// `get` is the env lookup, passed in so tests
/// resolve against a table of their own.
#[cfg(test)]
fn resolve_mmv_bands(get: impl Fn(&str) -> Option<String>) -> MmvBands {
    resolve_mmv_bands_with_profile(get, None)
}

/// Resolve the routing table with the calibration
/// profile's layer in its precedence place: each cell
/// comes from its env override when set and parsable,
/// else from the loaded profile's cell for the pair
/// when it states one and it parses, else the shipped
/// cell. The layers share one parse path, so a cell
/// means the same thing whichever layer stated it;
/// an unparsable value at any layer falls through to
/// the next rather than disabling or widening the
/// cell, and the fallbacks are logged at construction
/// ([`log_mmv_band_fallbacks`],
/// [`log_mmv_profile_cell_fallbacks`]).
fn resolve_mmv_bands_with_profile(
    get: impl Fn(&str) -> Option<String>,
    profile: Option<&crate::profile::CalibrationProfile>,
) -> MmvBands {
    let mut cells: [[Vec<(u32, u32)>; 3]; 3] = Default::default();
    for family in [MmvFamily::Q4K, MmvFamily::Q5K, MmvFamily::Q6K] {
        for shape in [MmvShape::Attn, MmvShape::Ffn, MmvShape::Kv] {
            let parsed = get(mmv_band_env(family, shape))
                .and_then(|raw| parse_mmv_band(&raw))
                .or_else(|| {
                    profile
                        .and_then(|p| p.band_cell(family, shape))
                        .and_then(parse_mmv_band)
                })
                .or_else(|| parse_mmv_band(default_mmv_band(family, shape)));
            cells[family.index()][shape.index()] =
                parsed.expect("the shipped MMV band cells parse");
        }
    }
    MmvBands { cells }
}

/// Log, per cell, when an env override is present
/// but unparsable and a lower layer therefore
/// applies: a silent fallback in a routing table
/// is how a typo becomes a mystery cliff. The line
/// names the layer that actually governs the cell,
/// the profile's cell when a loaded profile states
/// one, else the shipped cell.
fn log_mmv_band_fallbacks(profile: Option<&crate::profile::CalibrationProfile>) {
    for family in [MmvFamily::Q4K, MmvFamily::Q5K, MmvFamily::Q6K] {
        for shape in [MmvShape::Attn, MmvShape::Ffn, MmvShape::Kv] {
            let env = mmv_band_env(family, shape);
            if let Ok(raw) = std::env::var(env) {
                if parse_mmv_band(&raw).is_none() {
                    match profile.and_then(|p| p.band_cell(family, shape)) {
                        Some(cell) => dbg_log!(
                            "mmv band env {env}={raw:?} is unparsable; \
                             the profile cell ({cell}) applies"
                        ),
                        None => dbg_log!(
                            "mmv band env {env}={raw:?} is unparsable; \
                             the shipped cell ({}) applies",
                            default_mmv_band(family, shape)
                        ),
                    }
                }
            }
        }
    }
}

/// Log, per cell, when a loaded profile states a
/// cell the band parser refuses and the shipped cell
/// therefore applies: a profile is hand-editable, so
/// its typos get the same named-fallback treatment
/// as the env's.
fn log_mmv_profile_cell_fallbacks(profile: Option<&crate::profile::CalibrationProfile>) {
    let Some(profile) = profile else {
        return;
    };
    for family in [MmvFamily::Q4K, MmvFamily::Q5K, MmvFamily::Q6K] {
        for shape in [MmvShape::Attn, MmvShape::Ffn, MmvShape::Kv] {
            if let Some(cell) = profile.band_cell(family, shape) {
                if parse_mmv_band(cell).is_none() {
                    dbg_log!(
                        "mmv profile cell {} {} ({cell:?}) is unparsable; \
                         the shipped cell ({}) applies",
                        family.name(),
                        shape.name(),
                        default_mmv_band(family, shape)
                    );
                }
            }
        }
    }
}

/// Load the calibration profile for this context's
/// adapter at construction, and state the outcome on
/// the diag channel either way: the loaded profile
/// (with its key, which the backend keeps for the
/// capture manifest), or `None` with the reason
/// logged (no path resolves, no file there, or a file
/// that failed verification). A rejected profile is
/// never half-applied: the shipped defaults govern,
/// exactly as on an uncalibrated machine.
fn load_construction_profile(
    ctx: &crate::device::GpuContext,
) -> (Option<crate::profile::CalibrationProfile>, Option<String>) {
    use crate::profile::{load_profile, profile_path, AdapterKey, ProfileLoad};
    let key = AdapterKey::from_facts(&ctx.adapter);
    let Some(path) = profile_path() else {
        dbg_log!(
            "mmv profile: no profile path resolves (no {}, \
             XDG_CONFIG_HOME, or HOME); shipped defaults apply",
            crate::profile::PROFILE_PATH_ENV
        );
        return (None, None);
    };
    match load_profile(&path, &key) {
        ProfileLoad::Loaded(profile) => {
            let profile = *profile;
            dbg_log!(
                "mmv profile: loaded [{}] from {}",
                profile.key.describe(),
                path.display()
            );
            let describe = profile.key.describe();
            (Some(profile), Some(describe))
        }
        ProfileLoad::Absent => {
            dbg_log!(
                "mmv profile: none at {}; shipped defaults apply",
                path.display()
            );
            (None, None)
        }
        ProfileLoad::Rejected(why) => {
            dbg_log!("mmv profile: {why}; shipped defaults apply");
            (None, None)
        }
    }
}

/// Environment variable naming the chunked-GEMV row
/// threshold override: the largest batch row count at
/// which the session serves a plain projection as
/// several windowed GEMV dispatches at power-of-two row
/// counts (the chunked form) instead of one GEMV dispatch
/// at the batch's own count, inside the band from 2 rows
/// up to below the matrix threshold. Operational config,
/// not a compiled constant: set it to 15 to enable the
/// chunked form across the whole speculative verify band,
/// or leave it at the default to keep the ladder's GEMV
/// rung exactly as shipped. An unparsable value falls
/// back to [`DEFAULT_CHUNKED_GEMV_MAX_ROWS`], with the
/// fallback logged at construction.
pub const CHUNKED_GEMV_MAX_ROWS_ENV: &str = "WGPU_COMPUTE_CHUNKED_GEMV_MAX_ROWS";

/// The chunked-GEMV row threshold when the environment
/// does not override it: 1, which is OFF, since the
/// ladder only routes batches of 2 or more rows to the
/// chunked form. The shipped default follows the K18
/// precedent: an unproven dispatch shape does not enter
/// the ladder by default. The head-to-head proof that
/// can move this value is sequenced in
/// bench/chunked-gemv-2026-11-05.md.
pub const DEFAULT_CHUNKED_GEMV_MAX_ROWS: u32 = 1;

/// Resolve the chunked-GEMV row threshold from the raw
/// environment value: a plain u32, or the default when
/// unset or unparsable.
fn resolve_chunked_gemv_max_rows(raw: Option<&str>) -> u32 {
    match raw {
        Some(s) => s
            .trim()
            .parse::<u32>()
            .unwrap_or(DEFAULT_CHUNKED_GEMV_MAX_ROWS),
        None => DEFAULT_CHUNKED_GEMV_MAX_ROWS,
    }
}

/// Environment variable naming the K22 (Q4_K MMQ,
/// register-organization prototype) routing switch: the
/// exact value `1` makes the resident session prefer the
/// K22 product over the standard K13 product for Q4_K MMQ
/// projections. Operational config, not a compiled
/// constant, and deliberately fail-closed: an unset
/// variable, or any other value, keeps K22 out of the
/// ladder entirely, so the shipped ladder is byte-for-byte
/// the ladder every existing proof ran on. The shipped
/// default follows the wide-MMQ and K18 precedent: an
/// unproven kernel does not enter the ladder by default.
/// The server proof that can move this switch's default
/// is sequenced in bench/k22-mmq-2026-11-05.md.
pub const K22_MMQ_ENV: &str = "WGPU_COMPUTE_K22_MMQ";

/// Resolve the K22 routing switch from the raw
/// environment value: only the exact value `1` (after
/// trimming) enables it; everything else is off.
fn resolve_k22_mmq(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), Some("1"))
}

/// Environment variable naming the staged-flash routing
/// switch: it decides whether the resident session runs
/// multi-row attention calls through K6's staged flash
/// form (K/V blocks of 32 keys staged in workgroup memory,
/// 8 query rows per workgroup; see
/// `crate::kernels::k6_attention::FLASH_STAGED_SHADER`)
/// instead of the shipped flash form. Operational config,
/// not a compiled constant. The staged form is the
/// DEFAULT: an unset variable resolves to on, and the
/// exact value `1` (after trimming) also resolves to on,
/// so a value that meant on before the default flip still
/// means on. The exact value `0` is the explicit off and
/// the one-variable rollback to the shipped form. Any
/// other value fails closed to off, with the refusal
/// logged by name at construction: a typo must never
/// silently select a form. Single-row calls keep the
/// shipped form in every switch position (a one-row
/// workgroup would stage a KV block for one active row
/// of eight); that exclusion is routing, not part of the
/// switch. The default flipped on the interleaved card
/// evidence in bench/flash-host-cost-2026-11-06.md,
/// which discharged the decision rule written in
/// bench/flash-rework-2026-11-06.md; the flip itself is
/// recorded in bench/flash-default-flip-2026-11-07.md.
pub const FLASH_STAGED_ENV: &str = "WGPU_COMPUTE_FLASH_STAGED";

/// Resolve the staged-flash routing switch from the raw
/// environment value: unset is ON (the shipped default),
/// the exact value `1` (after trimming) is ON, the exact
/// value `0` is the explicit OFF, and any other value
/// fails closed to OFF. The caller logs the refusal for
/// an unrecognized value at construction
/// ([`log_flash_staged_fallback`]), so the fail-closed
/// position is visible in the run log, never silent.
fn resolve_flash_staged(raw: Option<&str>) -> bool {
    match raw.map(str::trim) {
        None | Some("1") => true,
        Some("0") => false,
        Some(_) => false,
    }
}

/// Whether a raw staged-flash env value is one the
/// resolver recognizes (`1` or `0` after trimming, or
/// unset). An unrecognized value resolves off; the
/// construction log names it, following the MMV band
/// fallback pattern: a silent fallback in a routing
/// switch is how a typo becomes a mystery regression.
fn flash_staged_value_recognized(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), None | Some("1") | Some("0"))
}

/// Log, at construction, when the staged-flash env
/// value is present but unrecognized and the switch
/// therefore fails closed to off.
fn log_flash_staged_fallback(raw: Option<&str>) {
    if let Some(value) = raw {
        if !flash_staged_value_recognized(raw) {
            dbg_log!(
                "flash staged env {FLASH_STAGED_ENV}={value:?} is not a recognized \
                 value (\"1\" or unset enables, \"0\" disables); failing closed, \
                 the staged form is OFF"
            );
        }
    }
}

/// Environment variable naming the K23 (Q4_K MMQ,
/// staging-repack prototype) routing switch: the exact
/// value `1` makes the resident session prefer the K23
/// product over the standard K13 product for Q4_K MMQ
/// projections when the K22 switch is off (K22 keeps its
/// rung above K23; see `session::mmq_product_q4k`).
/// Operational config, not a compiled constant, and
/// deliberately fail-closed on the same terms as
/// [`K22_MMQ_ENV`]: an unset variable, or any other value,
/// keeps K23 out of the ladder entirely. The server proof
/// that can move this switch's default is sequenced in
/// bench/k23-mmq-repack-2026-11-06.md.
pub const K23_MMQ_ENV: &str = "WGPU_COMPUTE_K23_MMQ";

/// Resolve the K23 routing switch from the raw
/// environment value: only the exact value `1` (after
/// trimming) enables it; everything else is off.
fn resolve_k23_mmq(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), Some("1"))
}

/// Environment variable naming the K23 (Q4_K coopmat
/// MMQ) routing switch: the exact value `1` makes the
/// resident session prefer the cooperative-matrix
/// product over every other Q4_K MMQ form. Operational
/// config, not a compiled constant, and deliberately
/// fail-closed on the K22 pattern: an unset variable,
/// or any other value, keeps K23 out of the ladder
/// entirely. The pipeline itself exists only on a
/// device opened through the coopmat bridge
/// ([`crate::coopmat_bridge`], its own switch); with
/// the routing on but no pipeline present, the session
/// logs the refusal by name at plan build and the
/// standard ladder serves. The server proof that can
/// move this switch's default is sequenced in
/// bench/coopmat-build-2026-11-06.md.
pub const COOPMAT_ENV: &str = "WGPU_COMPUTE_COOPMAT";

/// Resolve the K23 coopmat routing switch from the raw
/// environment value: only the exact value `1` (after
/// trimming) enables it; everything else is off.
fn resolve_coopmat(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), Some("1"))
}

/// Environment variable naming the MMQ split-K routing
/// config for the standard products (K13 Q4_K, K14 Q6_K):
/// the split partitions a product's reduction dimension
/// across S workgroups per output tile writing f32
/// partial sums, folded by a reduction dispatch, so a
/// narrow output whose plain grid cannot fill the card
/// (the 256-wide Q6_K KV projections: 4 tiles walking 26
/// super-blocks each) still parallelizes. Values:
/// `off` (and unset) keeps the shipped unsplit products;
/// `1`..=`8` fixes the split factor; `auto` picks the
/// factor per projection from its tile count (see
/// [`mmq_splitk_factor`]). Operational config, not a
/// compiled constant, and deliberately fail-closed on
/// the K22 pattern: any other value resolves to
/// [`MmqSplitK::Off`] and is named in the construction
/// log. The default is OFF: the dispatch shape is
/// unproven on the card, and the verdict that can move
/// it is sequenced in bench/splitk-mmq-2026-11-07.md.
pub const MMQ_SPLITK_ENV: &str = "WGPU_COMPUTE_MMQ_SPLITK";

/// The resolved MMQ split-K mode (see [`MMQ_SPLITK_ENV`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MmqSplitK {
    /// Split-K off: the standard products serve as shipped.
    Off,
    /// A fixed split factor in 1..=[`crate::kernels::splitk_reduce::SPLITK_MAX`]
    /// (1 resolves to no split at the factor rule).
    Fixed(u32),
    /// The per-projection auto rule (see
    /// [`mmq_splitk_factor`]).
    Auto,
}

impl MmqSplitK {
    /// The diag manifest's spelling of this mode.
    pub(crate) fn describe(&self) -> String {
        match self {
            MmqSplitK::Off => "off".to_string(),
            MmqSplitK::Fixed(n) => format!("fixed:{n}"),
            MmqSplitK::Auto => "auto".to_string(),
        }
    }
}

/// Resolve the MMQ split-K config from the raw
/// environment value: unset or `off` is off, `auto` is
/// the auto rule, an integer in 1..=8 is a fixed factor,
/// and anything else fails closed to off.
fn resolve_mmq_splitk(raw: Option<&str>) -> MmqSplitK {
    match raw.map(str::trim) {
        None | Some("off") => MmqSplitK::Off,
        Some("auto") => MmqSplitK::Auto,
        Some(value) => match value.parse::<u32>() {
            Ok(n) if (1..=crate::kernels::splitk_reduce::SPLITK_MAX).contains(&n) => {
                MmqSplitK::Fixed(n)
            }
            _ => MmqSplitK::Off,
        },
    }
}

/// Whether a raw MMQ split-K env value is one the
/// resolver recognizes (unset, `off`, `auto`, or an
/// integer in 1..=8 after trimming). An unrecognized
/// value resolves off; the construction log names it
/// (the flash fallback pattern).
fn mmq_splitk_value_recognized(raw: Option<&str>) -> bool {
    match raw.map(str::trim) {
        None | Some("off") | Some("auto") => true,
        Some(value) => matches!(
            value.parse::<u32>(),
            Ok(n) if (1..=crate::kernels::splitk_reduce::SPLITK_MAX).contains(&n)
        ),
    }
}

/// Log, at construction, when the MMQ split-K env value
/// is present but unrecognized and the config therefore
/// fails closed to off.
fn log_mmq_splitk_fallback(raw: Option<&str>) {
    if let Some(value) = raw {
        if !mmq_splitk_value_recognized(raw) {
            dbg_log!(
                "mmq splitk env {MMQ_SPLITK_ENV}={value:?} is not a recognized value \
                 (\"off\", \"auto\", or 1..=8); failing closed, split-K is OFF"
            );
        }
    }
}

/// The split factor for one projection under `mode`,
/// from the shape math (bench/splitk-mmq-2026-11-07.md):
/// `tiles` is the unsplit grid's workgroup count
/// (output rows over the kernel's tile) and
/// `blocks_per_row` the super-block columns each
/// workgroup would walk. The factor never exceeds the
/// block count (a slice past the last block is empty
/// work) and never exceeds [`crate::kernels::splitk_reduce::SPLITK_MAX`].
/// The auto rule targets roughly 64 workgroups: the KV
/// shape's 4 tiles get the full factor of 8 (32
/// workgroups, about 4 blocks per slice), a 32-tile
/// projection doubles to 64, and any projection already
/// filling the card (64+ tiles) stays unsplit. The rule
/// is a heuristic pending the card A/B; the fixed mode
/// exists so the verdict can sweep the factor directly.
pub(crate) fn mmq_splitk_factor(mode: MmqSplitK, tiles: u32, blocks_per_row: u32) -> u32 {
    let blocks = blocks_per_row.max(1);
    let cap = blocks.min(crate::kernels::splitk_reduce::SPLITK_MAX);
    match mode {
        MmqSplitK::Off => 1,
        MmqSplitK::Fixed(n) => n.clamp(1, cap),
        MmqSplitK::Auto => 64u32.div_ceil(tiles.max(1)).clamp(1, cap),
    }
}

/// Environment variable naming the K21F finish-carrying
/// head routing switch: it allows the resident session's
/// block head to route its Q5_K product to the
/// finish-carrying MMV form (K21F, see
/// `crate::kernels::k21_q5k_mmv_finish`) when the shared
/// MMV band also admits the block batch. Operational
/// config, not a compiled constant. The route is the
/// DEFAULT: an unset variable resolves to on (see
/// [`DEFAULT_K21F_HEAD`]), and the exact value `1`
/// (after trimming) also resolves to on, so a value
/// that meant on under the guard still means on. The
/// exact value `0` is the explicit off and the
/// one-variable rollback to the K3 fused head. Any
/// other value fails closed to off
/// ([`K21F_HEAD_FALLBACK`]), with the refusal logged by
/// name at construction: a typo must never silently
/// select a route. The switch is the head route's own
/// gate, separate from the shared MMV ceiling
/// ([`MMV_MAX_ROWS_ENV`]): the route used to ride that
/// ceiling alone, so when the ceiling's default opened
/// to 16 the head route went live by default, and the
/// 2026-11-07 defaults-confirmation session
/// (bench/defaults-confirmation-2026-11-07.md) proved
/// speculative identity FAILS on the real file with
/// the route live (deterministic divergence at stream
/// index 23, depths 1 and 6). The guard that closed it
/// by default is recorded in
/// bench/k21f-default-guard-2026-11-07.md; the
/// root-cause fix (a truncated dequant mask in the
/// K21F shader, bench/k21f-mask-fix-2026-11-07.md) and
/// the wave18 card retest that met the guard's reopen
/// criterion (bench/k21f-retest-2026-11-07.md) are what
/// moved the default back to on; the flip itself is
/// recorded in bench/k21f-default-flip-2026-11-07.md.
pub const K21F_HEAD_ENV: &str = "WGPU_COMPUTE_K21F_HEAD";

/// Whether the K21F head route is live when the
/// environment does not override it: on. The route
/// originally shipped live by riding the shared MMV
/// ceiling; the 2026-11-07 defaults-confirmation
/// identity failure closed it behind this switch with
/// the default off (bench/k21f-default-guard-2026-11-07.md)
/// until the numerics were fixed and the route
/// re-earned the default under the guard's written
/// reopen criterion. The mask fix
/// (bench/k21f-mask-fix-2026-11-07.md) plus the wave18
/// card retest (bench/k21f-retest-2026-11-07.md:
/// identity and golden PASS at depths 1 and 6 with the
/// switch on, K21F's finished output bit-identical to
/// K3's at the real head geometry) discharged that
/// criterion, and the default moved back to on.
/// Rollback is one env value:
/// `WGPU_COMPUTE_K21F_HEAD=0` restores the guarded
/// (K3 fused head) composition exactly.
pub const DEFAULT_K21F_HEAD: bool = true;

/// The K21F head switch position a present but
/// unrecognized env value resolves to: off, closed.
/// The fallback is deliberately NOT the shipped
/// default: a typo in the environment must never
/// widen routing, so the malformed case keeps the
/// fail-closed value it resolved to while the guard's
/// default was off.
const K21F_HEAD_FALLBACK: bool = false;

/// Resolve the K21F head routing switch from the raw
/// environment value: unset is ON (the shipped
/// default, [`DEFAULT_K21F_HEAD`]), the exact value
/// `1` (after trimming) is ON, the exact value `0` is
/// the explicit OFF, and any other value fails closed
/// to OFF ([`K21F_HEAD_FALLBACK`]). The caller logs
/// the refusal for an unrecognized value at
/// construction ([`log_k21f_head_fallback`]), so the
/// fail-closed position is visible in the run log,
/// never silent.
fn resolve_k21f_head(raw: Option<&str>) -> bool {
    match raw.map(str::trim) {
        None => DEFAULT_K21F_HEAD,
        Some("1") => true,
        Some("0") => false,
        Some(_) => K21F_HEAD_FALLBACK,
    }
}

/// Whether a raw K21F head env value is one the
/// resolver recognizes (`1` or `0` after trimming, or
/// unset). An unrecognized value resolves off; the
/// construction log names it, following the MMV band
/// and flash-staged fallback pattern: a silent
/// fallback in a routing switch is how a typo becomes
/// a mystery regression.
fn k21f_head_value_recognized(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), None | Some("1") | Some("0"))
}

/// Log, at construction, when the K21F head env value
/// is present but unrecognized and the switch
/// therefore fails closed to off.
fn log_k21f_head_fallback(raw: Option<&str>) {
    if let Some(value) = raw {
        if !k21f_head_value_recognized(raw) {
            dbg_log!(
                "k21f head env {K21F_HEAD_ENV}={value:?} is not a recognized \
                 value (\"1\" or unset enables, \"0\" disables); failing closed, \
                 the K21F head route is OFF"
            );
        }
    }
}

/// Build a [`GlimmerError::StepFailed`] for `step` with `detail`.
fn step_failed(step: &'static str, detail: impl Into<String>) -> GlimmerError {
    GlimmerError::StepFailed {
        step,
        detail: detail.into(),
    }
}

/// The mainline ggml type id of a [`Dtype`], for naming types in
/// refusals (the table is pinned in `muse-glimmer`'s weights
/// module; `Bf16` is 30).
fn dtype_id(dtype: Dtype) -> u32 {
    match dtype {
        Dtype::F32 => 0,
        Dtype::F16 => 1,
        Dtype::Q4_0 => 2,
        Dtype::Q8_0 => 8,
        Dtype::Q4K => 12,
        Dtype::Q5K => 13,
        Dtype::Q6K => 14,
        Dtype::Bf16 => 30,
        Dtype::Unknown(id) => id,
    }
}

/// A weight tensor resident on the device, in whichever kernel's
/// layout its dtype dictates. The cache keys these by tensor
/// name, and the name is an identity: a later tensor arriving
/// under a cached name with a different dtype or geometry is a
/// refusal, not a re-upload.
#[derive(Debug)]
enum CachedWeights {
    /// Q4_K, through kernel K1.
    Q4K(K1Weights),
    /// Q6_K, through kernel K2.
    Q6K(K2Weights),
    /// Q5_K, through kernel K3.
    Q5K(K3Weights),
}

impl CachedWeights {
    /// The (input width, output rows, raw byte length) the
    /// cached upload was made from.
    fn geometry(&self) -> (usize, usize, u64) {
        match self {
            CachedWeights::Q4K(w) => (w.in_features, w.out_rows, w.byte_len),
            CachedWeights::Q6K(w) => (w.in_features, w.out_rows, w.byte_len),
            CachedWeights::Q5K(w) => (w.in_features, w.out_rows, w.byte_len),
        }
    }

    /// The kernel name this entry's dtype runs through, for
    /// refusal messages.
    fn kernel(&self) -> &'static str {
        match self {
            CachedWeights::Q4K(_) => "K1 (Q4_K)",
            CachedWeights::Q6K(_) => "K2 (Q6_K)",
            CachedWeights::Q5K(_) => "K3 (Q5_K)",
        }
    }
}

/// The wgpu [`Compute`] backend. Constructing one selects an
/// adapter and opens a device (see [`crate::detect`]), so
/// construction is the fallible step; a constructed backend has a
/// live device and every kernel pipeline compiled. Stateful in
/// exactly one way, like `cpu-compute`: it owns the KV cache of
/// the one sequence it is serving (device buffers plus a host
/// mirror, in [`GpuKvState`]). Construct one per sequence, and
/// call [`WgpuCompute::reset`] to abandon a sequence and start
/// over.
#[derive(Debug)]
pub struct WgpuCompute {
    ctx: GpuContext,
    k1: K1Pipeline,
    k1_gated: K1Pipeline,
    k1_swiglu: K1Pipeline,
    k2: K2Pipeline,
    k2_swiglu: K2Pipeline,
    k3: K3Pipeline,
    k4: K4Pipeline,
    k5: K5Pipeline,
    k6: K6Pipeline,
    k7: K7Pipeline,
    k8: K8Pipeline,
    k9: K9Pipeline,
    k10: Option<K10Pipeline>,
    k10_gated: Option<K10Pipeline>,
    k10_swiglu: Option<K10Pipeline>,
    k11: Option<K11Pipeline>,
    k11_swiglu: Option<K11Pipeline>,
    k12: K12QuantPipeline,
    k12_gated: K12QuantPipeline,
    k12_swiglu: K12QuantPipeline,
    k13: Option<K13Pipeline>,
    k14: Option<K14Pipeline>,
    k22: Option<K22Pipeline>,
    k22_mmq: bool,
    /// Whether the resident session routes multi-row
    /// attention calls through K6's staged flash form,
    /// resolved once at construction from
    /// [`FLASH_STAGED_ENV`], never a constant compiled into
    /// the routing. The pipeline itself lives in the K6 set
    /// and may be absent on a low-storage device; the
    /// session checks both.
    flash_staged: bool,
    k23: Option<K23Pipeline>,
    k23_mmq: bool,
    k23_coopmat: Option<K23CoopmatPipeline>,
    coopmat: bool,
    /// The MMQ split-K routing config for the standard
    /// products, resolved once at construction from
    /// [`MMQ_SPLITK_ENV`], never a constant compiled into
    /// the routing. The split pipelines live inside the
    /// K13/K14 pipelines and are built regardless of this
    /// config; the session reads the mode and the factor
    /// rule at plan build.
    mmq_splitk: MmqSplitK,
    k15: Option<K15Pipeline>,
    k16: Option<K16Pipeline>,
    wide_min_rows: u32,
    k17: K17Pipeline,
    k18_q4k: Option<K18Q4KPipeline>,
    k18_q6k: Option<K18Q6KPipeline>,
    k18_q5k: Option<K18Q5KPipeline>,
    small_batch_max_rows: u32,
    k19: K19Pipeline,
    k20: K20Pipeline,
    k21: K21Pipeline,
    // The finish-carrying K21 pipeline: read by the
    // resident session's block head routing
    // (bench/k21-finish-2026-11-06.md).
    k21_finish: K21FinishPipeline,
    /// Whether the resident session's block head may
    /// route to the finish-carrying K21 form at all:
    /// resolved once at construction from
    /// [`K21F_HEAD_ENV`], never a constant compiled
    /// into the routing. `true` (the shipped default,
    /// [`DEFAULT_K21F_HEAD`]) lets the route prepare
    /// wherever the shared band admits the block
    /// batch; `false` (the explicit `0`, or a
    /// fail-closed unrecognized value) keeps the
    /// block head on the K3 fused call no matter
    /// where the shared MMV ceiling sits. The route
    /// additionally requires the shared band to admit
    /// the block batch (see
    /// `session::try_mmv_q5k_finish`).
    k21f_head: bool,
    mmv_max_rows: u32,
    mmv_bands: MmvBands,
    /// The calibration profile that governed this
    /// backend's MMV resolution, as its adapter key's
    /// one-line description, or `None` when no profile
    /// loaded (uncalibrated machine, or a profile that
    /// failed verification): kept so the capture
    /// manifest can state which profile a run ran
    /// under, or that none did. The profile's values
    /// themselves are already folded into
    /// `mmv_max_rows` and `mmv_bands` at construction.
    mmv_profile_key: Option<String>,
    chunked_gemv_max_rows: u32,
    weights: RefCell<HashMap<String, CachedWeights>>,
    kv: RefCell<Option<GpuKvState>>,
}

impl WgpuCompute {
    /// Open a backend under the default selection policy (discrete
    /// Vulkan or refusal).
    ///
    /// # Errors
    ///
    /// [`crate::error::WgpuError::NoSuitableAdapter`] when no adapter qualifies,
    /// plus the device and pipeline failures of
    /// [`WgpuCompute::with_policy`].
    pub fn new() -> crate::error::Result<Self> {
        Self::with_policy(&DetectPolicy::default())
    }

    /// Open a backend under an explicit selection policy.
    ///
    /// # Errors
    ///
    /// [`crate::error::WgpuError::NoSuitableAdapter`], [`crate::error::WgpuError::DeviceRequest`],
    /// [`crate::error::WgpuError::LimitShortfall`], or [`crate::error::WgpuError::Pipeline`]
    /// when device opening or pipeline compilation fails.
    pub fn with_policy(policy: &DetectPolicy) -> crate::error::Result<Self> {
        Self::from_context(detect(policy)?)
    }

    /// Build a backend over an already-opened context.
    ///
    /// # Errors
    ///
    /// [`crate::error::WgpuError::Pipeline`] when any kernel
    /// shader fails to compile.
    pub fn from_context(ctx: GpuContext) -> crate::error::Result<Self> {
        let k1 = K1Pipeline::new(&ctx.device)?;
        let k1_gated = K1Pipeline::with_shape_act(
            &ctx.device,
            crate::kernels::k1_q4k_gemv::shipped_shape(ctx.device.features()),
            ACT_GATED,
        )?;
        let k1_swiglu = K1Pipeline::with_shape_act(
            &ctx.device,
            crate::kernels::k1_q4k_gemv::shipped_shape(ctx.device.features()),
            ACT_SWIGLU,
        )?;
        let k2 = K2Pipeline::new(&ctx.device)?;
        let k2_swiglu = K2Pipeline::with_shape_act(
            &ctx.device,
            crate::kernels::k2_q6k_gemv::shipped_shape(ctx.device.features()),
            ACT_SWIGLU,
        )?;
        let k3 = K3Pipeline::new(&ctx.device)?;
        let k4 = K4Pipeline::new(&ctx.device)?;
        let k5 = K5Pipeline::new(&ctx.device)?;
        let k6 = K6Pipeline::new(&ctx.device)?;
        let k7 = K7Pipeline::new(&ctx.device)?;
        let k8 = K8Pipeline::new(&ctx.device)?;
        let k9 = K9Pipeline::new(&ctx.device)?;
        // The prefill GEMMs are the one optional pipeline set:
        // a device granting too little workgroup storage for
        // the smallest staged tile gets `None` here, and the
        // resident session keeps its GEMV chunk plans on it
        // (the K10 module docs own the fallback contract).
        let k10 = K10Pipeline::for_device(&ctx.device, ACT_NONE)?;
        let k10_gated = K10Pipeline::for_device(&ctx.device, ACT_GATED)?;
        let k10_swiglu = K10Pipeline::for_device(&ctx.device, ACT_SWIGLU)?;
        let k11 = K11Pipeline::for_device(&ctx.device, ACT_NONE)?;
        let k11_swiglu = K11Pipeline::for_device(&ctx.device, ACT_SWIGLU)?;
        // The prefill MMQ set: the quantizer (K12) is
        // unconditional (its workgroup footprint is two
        // 256-entry arrays), while the product kernels
        // (K13/K14) are optional on the same terms as the
        // GEMMs: a device granting too little workgroup
        // storage for their staging sets gets `None`, and the
        // resident session keeps its GEMM chunk plans there
        // (the K13 module docs own the fallback contract).
        let k12 = K12QuantPipeline::new(&ctx.device, ACT_NONE)?;
        let k12_gated = K12QuantPipeline::new(&ctx.device, ACT_GATED)?;
        let k12_swiglu = K12QuantPipeline::new(&ctx.device, ACT_SWIGLU)?;
        let k13 = K13Pipeline::for_device(&ctx.device)?;
        let k14 = K14Pipeline::for_device(&ctx.device)?;
        // The K22 Q4_K MMQ prototype (the register-
        // organization rework of K13): optional on the
        // same terms as the standard MMQ set, and built
        // regardless of its routing switch so a shader
        // defect fails at construction, not on first
        // routed use. Only its routing is config, read
        // here (see K22_MMQ_ENV); the default keeps it
        // out of the ladder until the server proof
        // moves it.
        let k22 = K22Pipeline::for_device(&ctx.device)?;
        let k22_mmq = resolve_k22_mmq(std::env::var(K22_MMQ_ENV).ok().as_deref());
        dbg_log!(
            "mmq k22 config: enabled={k22_mmq} (env {K22_MMQ_ENV}, only \"1\" enables); \
             k22={}",
            k22.is_some()
        );
        // The staged flash form (K6's blocked rework of
        // the shipped flash): its pipeline was built (or
        // not) inside the K6 set above, on the device's
        // workgroup-storage grant; only its routing is
        // config, read here (see FLASH_STAGED_ENV). The
        // config line names both halves, so a run log shows
        // a switch that cannot engage (enabled with no
        // pipeline) instead of silently timing the shipped
        // form under the staged name.
        let flash_staged_raw = std::env::var(FLASH_STAGED_ENV).ok();
        let flash_staged = resolve_flash_staged(flash_staged_raw.as_deref());
        log_flash_staged_fallback(flash_staged_raw.as_deref());
        dbg_log!(
            "flash staged config: enabled={flash_staged} (env {FLASH_STAGED_ENV}, \
             \"1\" or unset enables, \"0\" disables); staged pipeline={}",
            k6.flash_staged_available()
        );
        // The K23 Q4_K MMQ prototype (K13's organization
        // with llama's staging repack): optional on the
        // same terms as K22, and built regardless of its
        // routing switch so a shader defect fails at
        // construction, not on first routed use. Only its
        // routing is config, read here (see K23_MMQ_ENV);
        // the default keeps it out of the ladder until
        // the server proof moves it.
        let k23 = K23Pipeline::for_device(&ctx.device)?;
        let k23_mmq = resolve_k23_mmq(std::env::var(K23_MMQ_ENV).ok().as_deref());
        dbg_log!(
            "mmq k23 config: enabled={k23_mmq} (env {K23_MMQ_ENV}, only \"1\" enables); \
             k23={}",
            k23.is_some()
        );
        // The K23 Q4_K coopmat MMQ: constructible only
        // on a bridged device (the passthrough module is
        // legal only where the bridge enabled the
        // extension), so it is built from the context's
        // coopmat record rather than unconditionally;
        // its routing switch is read here (see
        // COOPMAT_ENV), default off.
        let k23_coopmat = match &ctx.coopmat {
            Some(info) => K23CoopmatPipeline::for_device(&ctx.device, info)?,
            None => None,
        };
        let coopmat = resolve_coopmat(std::env::var(COOPMAT_ENV).ok().as_deref());
        dbg_log!(
            "mmq coopmat config: enabled={coopmat} (env {COOPMAT_ENV}, only \"1\" enables); \
             k23_coopmat={}",
            k23_coopmat.is_some()
        );
        // The MMQ split-K routing config (see
        // MMQ_SPLITK_ENV): resolved here like every
        // routing switch; the split pipelines themselves
        // were built inside K13/K14 above regardless of
        // the mode, so a defect in either form fails at
        // construction. Default off (unproven dispatch
        // shape; the card verdict is sequenced in
        // bench/splitk-mmq-2026-11-07.md).
        let mmq_splitk_raw = std::env::var(MMQ_SPLITK_ENV).ok();
        log_mmq_splitk_fallback(mmq_splitk_raw.as_deref());
        let mmq_splitk = resolve_mmq_splitk(mmq_splitk_raw.as_deref());
        dbg_log!(
            "mmq splitk config: mode={} (env {MMQ_SPLITK_ENV}: \"off\"/unset is off, \
             \"auto\" is the tile-count rule, 1..=8 a fixed factor)",
            mmq_splitk.describe()
        );
        let k17 = K17Pipeline::new(&ctx.device)?;
        // The wide-tile double-buffered MMQ set (K15/K16):
        // optional on the same terms as the standard MMQ
        // set (a device granting too little workgroup
        // storage for the double-buffered staging sets gets
        // `None`, and the session keeps its standard MMQ
        // form there). The row threshold at which the
        // session starts preferring them is config, read
        // here (see WIDE_MIN_ROWS_ENV).
        let k15 = K15Pipeline::for_device(&ctx.device)?;
        let k16 = K16Pipeline::for_device(&ctx.device)?;
        let wide_min_rows = resolve_wide_min_rows(std::env::var(WIDE_MIN_ROWS_ENV).ok().as_deref());
        dbg_log!(
            "mmq wide config: min_rows={wide_min_rows} (env {WIDE_MIN_ROWS_ENV}, \
             default {DEFAULT_WIDE_MIN_ROWS}); k15={} k16={}",
            k15.is_some(),
            k16.is_some()
        );
        // The small-batch product set (K18): optional on
        // the same terms as the MMQ sets (a device
        // granting too little workgroup storage for a
        // family's staging set gets `None` for it, and
        // the session keeps that family's GEMV form
        // there). The row threshold at which the session
        // starts preferring them is config, read here
        // (see SMALL_BATCH_MAX_ROWS_ENV); the default
        // keeps them out of the ladder until the server
        // proof moves it.
        let k18_q4k = K18Q4KPipeline::for_device(&ctx.device)?;
        let k18_q6k = K18Q6KPipeline::for_device(&ctx.device)?;
        let k18_q5k = K18Q5KPipeline::for_device(&ctx.device)?;
        let small_batch_max_rows =
            resolve_small_batch_max_rows(std::env::var(SMALL_BATCH_MAX_ROWS_ENV).ok().as_deref());
        dbg_log!(
            "small batch config: max_rows={small_batch_max_rows} (env \
             {SMALL_BATCH_MAX_ROWS_ENV}, default {DEFAULT_SMALL_BATCH_MAX_ROWS}); \
             k18 q4k={} q6k={} q5k={}",
            k18_q4k.is_some(),
            k18_q6k.is_some(),
            k18_q5k.is_some()
        );
        // The register-streaming products (K19 for
        // Q4_K, K20 for Q6_K, K21 for Q5_K): not optional
        // like the staging sets (their only workgroup
        // memory is their reduction partials, so no
        // device class refuses them), and built
        // unconditionally so a shader defect fails at
        // construction like K1's. Only their routing is
        // config (see MMV_MAX_ROWS_ENV); the shipped
        // default opens the gate to the kernels' full
        // band, the server proof having moved it on
        // 2026-11-06 (bench/baseline-session-2026-11-06.md
        // leg (d)), and the band table narrows the gate
        // per family and shape from there.
        let k19 = K19Pipeline::new(&ctx.device)?;
        let k20 = K20Pipeline::new(&ctx.device)?;
        let k21 = K21Pipeline::new(&ctx.device)?;
        // The finish-carrying form of K21, for the
        // Q5_K head sites: plain K21 cannot serve them
        // (they prepare K3 with the fused scale and
        // softcap finish), which is why the 2026-11-06
        // verdict shows K21 firing zero times in the
        // real forward. Built unconditionally like its
        // sibling. Only its block-head routing is
        // config, read here (see K21F_HEAD_ENV): the
        // route requires BOTH this switch on AND the
        // shared MMV band admitting the block batch.
        // The switch defaulted off after the 2026-11-07
        // defaults-confirmation identity failure
        // (bench/k21f-default-guard-2026-11-07.md) and
        // returned to on when the wave18 retest met
        // the guard's reopen criterion
        // (bench/k21f-retest-2026-11-07.md); an
        // unrecognized value fails closed and is
        // logged by name.
        let k21_finish = K21FinishPipeline::new(&ctx.device)?;
        let k21f_head_raw = std::env::var(K21F_HEAD_ENV).ok();
        let k21f_head = resolve_k21f_head(k21f_head_raw.as_deref());
        log_k21f_head_fallback(k21f_head_raw.as_deref());
        dbg_log!(
            "k21f head config: enabled={k21f_head} (env {K21F_HEAD_ENV}, \
             \"1\" or unset enables, \"0\" disables)"
        );
        // The calibration profile for this adapter
        // (see crate::profile): measured routing values
        // from a prior calibration run on this exact
        // card and driver, loaded before the MMV policy
        // resolves so they take their precedence place
        // under the env overrides and over the shipped
        // table. No profile (the uncalibrated machine)
        // resolves exactly as before.
        let (mmv_profile, mmv_profile_key) = load_construction_profile(&ctx);
        let mmv_max_rows = resolve_mmv_max_rows_with_profile(
            std::env::var(MMV_MAX_ROWS_ENV).ok().as_deref(),
            mmv_profile.as_ref().and_then(|p| p.mmv_max_rows),
        );
        log_mmv_band_fallbacks(mmv_profile.as_ref());
        log_mmv_profile_cell_fallbacks(mmv_profile.as_ref());
        let mmv_bands =
            resolve_mmv_bands_with_profile(|name| std::env::var(name).ok(), mmv_profile.as_ref());
        dbg_log!(
            "mmv config: max_rows={mmv_max_rows} (env {MMV_MAX_ROWS_ENV}, \
             default {DEFAULT_MMV_MAX_ROWS}); profile {}; k19 shape {}, \
             k20 shape {}, k21 shape {}; bands {}",
            mmv_profile_key.as_deref().unwrap_or("none"),
            k19.shape().name(),
            k20.shape().name(),
            k21.shape().name(),
            mmv_bands.describe()
        );
        // The chunked-GEMV dispatch shape: no pipelines
        // of its own (it re-dispatches the GEMV kernels
        // over row windows), only the row threshold at
        // which the session prefers it over the single
        // GEMV dispatch, read here (see
        // CHUNKED_GEMV_MAX_ROWS_ENV); the default keeps
        // it out of the ladder until the head-to-head
        // proof moves it.
        let chunked_gemv_max_rows =
            resolve_chunked_gemv_max_rows(std::env::var(CHUNKED_GEMV_MAX_ROWS_ENV).ok().as_deref());
        dbg_log!(
            "chunked gemv config: max_rows={chunked_gemv_max_rows} (env \
             {CHUNKED_GEMV_MAX_ROWS_ENV}, default {DEFAULT_CHUNKED_GEMV_MAX_ROWS})"
        );
        dbg_log!("backend ready on {}", ctx.adapter.name);
        Ok(WgpuCompute {
            ctx,
            k1,
            k1_gated,
            k1_swiglu,
            k2,
            k2_swiglu,
            k3,
            k4,
            k5,
            k6,
            k7,
            k8,
            k9,
            k10,
            k10_gated,
            k10_swiglu,
            k11,
            k11_swiglu,
            k12,
            k12_gated,
            k12_swiglu,
            k13,
            k14,
            k22,
            k22_mmq,
            flash_staged,
            k23,
            k23_mmq,
            k23_coopmat,
            coopmat,
            mmq_splitk,
            k15,
            k16,
            wide_min_rows,
            k17,
            k18_q4k,
            k18_q6k,
            k18_q5k,
            small_batch_max_rows,
            k19,
            k20,
            k21,
            k21_finish,
            k21f_head,
            mmv_max_rows,
            mmv_bands,
            mmv_profile_key,
            chunked_gemv_max_rows,
            weights: RefCell::new(HashMap::new()),
            kv: RefCell::new(None),
        })
    }

    /// The device context this backend runs on.
    #[must_use]
    pub fn context(&self) -> &GpuContext {
        &self.ctx
    }

    /// The K1 (Q4_K GEMV) pipeline. Crate-internal: the
    /// device-resident session ([`crate::session`]) composes the
    /// same pipelines this backend's trait implementation uses,
    /// so the two paths can never drift onto different kernels.
    pub(crate) fn k1(&self) -> &K1Pipeline {
        &self.k1
    }

    /// The K1 pipeline with the attention-gate activation
    /// fused in (the resident session's output projection).
    /// See [`WgpuCompute::k1`].
    pub(crate) fn k1_gated(&self) -> &K1Pipeline {
        &self.k1_gated
    }

    /// The K1 pipeline with the SwiGLU activation fused in
    /// (the resident session's down projection, when the down
    /// tensor is Q4_K). See [`WgpuCompute::k1`].
    pub(crate) fn k1_swiglu(&self) -> &K1Pipeline {
        &self.k1_swiglu
    }

    /// The K2 (Q6_K GEMV) pipeline. See [`WgpuCompute::k1`].
    pub(crate) fn k2(&self) -> &K2Pipeline {
        &self.k2
    }

    /// The K2 pipeline with the SwiGLU activation fused in
    /// (the resident session's down projection, when the down
    /// tensor is Q6_K). See [`WgpuCompute::k1`].
    pub(crate) fn k2_swiglu(&self) -> &K2Pipeline {
        &self.k2_swiglu
    }

    /// The K3 (Q5_K GEMV) pipeline. See [`WgpuCompute::k1`].
    pub(crate) fn k3(&self) -> &K3Pipeline {
        &self.k3
    }

    /// The K4 (RMSNorm and residual add) pipeline. See
    /// [`WgpuCompute::k1`].
    pub(crate) fn k4(&self) -> &K4Pipeline {
        &self.k4
    }

    /// The K6 (grouped-query attention) pipeline. See
    /// [`WgpuCompute::k1`].
    pub(crate) fn k6(&self) -> &K6Pipeline {
        &self.k6
    }

    /// The K7 (SwiGLU and attention gate) pipeline. See
    /// [`WgpuCompute::k1`].
    pub(crate) fn k7(&self) -> &K7Pipeline {
        &self.k7
    }

    /// The K8 (embedding gather and head finishing) pipeline.
    /// See [`WgpuCompute::k1`].
    pub(crate) fn k8(&self) -> &K8Pipeline {
        &self.k8
    }

    /// The K9 (fused attention preparation) pipeline. Only the
    /// device-resident session prepares its calls; the trait
    /// implementation composes K4 and K5 instead. See
    /// [`WgpuCompute::k1`].
    pub(crate) fn k9(&self) -> &K9Pipeline {
        &self.k9
    }

    /// The K10 (Q4_K GEMM) pipeline, when this device grants
    /// the workgroup storage its smallest tile needs. Only
    /// the device-resident session prepares its calls, for
    /// chunk plans at or past the GEMM row threshold; `None`
    /// is the session's signal to keep the GEMV form. See
    /// [`WgpuCompute::k1`].
    pub(crate) fn k10(&self) -> Option<&K10Pipeline> {
        self.k10.as_ref()
    }

    /// The K10 pipeline with the attention-gate activation
    /// fused in. See [`WgpuCompute::k10`].
    pub(crate) fn k10_gated(&self) -> Option<&K10Pipeline> {
        self.k10_gated.as_ref()
    }

    /// The K10 pipeline with the SwiGLU activation fused in.
    /// See [`WgpuCompute::k10`].
    pub(crate) fn k10_swiglu(&self) -> Option<&K10Pipeline> {
        self.k10_swiglu.as_ref()
    }

    /// The K11 (Q6_K GEMM) pipeline, when this device grants
    /// the workgroup storage. See [`WgpuCompute::k10`].
    pub(crate) fn k11(&self) -> Option<&K11Pipeline> {
        self.k11.as_ref()
    }

    /// The K11 pipeline with the SwiGLU activation fused in.
    /// See [`WgpuCompute::k10`].
    pub(crate) fn k11_swiglu(&self) -> Option<&K11Pipeline> {
        self.k11_swiglu.as_ref()
    }

    /// The K12 (Q8_1 activation quantizer) pipeline. Only the
    /// device-resident session prepares its calls, for chunk
    /// plans at or past the GEMM row threshold when an MMQ
    /// product pipeline exists to consume the quantized form.
    /// See [`WgpuCompute::k1`].
    pub(crate) fn k12(&self) -> &K12QuantPipeline {
        &self.k12
    }

    /// The K12 pipeline with the attention-gate activation
    /// applied before quantization. See [`WgpuCompute::k12`].
    pub(crate) fn k12_gated(&self) -> &K12QuantPipeline {
        &self.k12_gated
    }

    /// The K12 pipeline with the SwiGLU activation applied
    /// before quantization. See [`WgpuCompute::k12`].
    pub(crate) fn k12_swiglu(&self) -> &K12QuantPipeline {
        &self.k12_swiglu
    }

    /// The K13 (Q4_K MMQ) pipeline, when this device grants
    /// the workgroup storage its staging set needs. Only the
    /// device-resident session prepares its calls; `None` is
    /// the session's signal to keep the GEMM form. See
    /// [`WgpuCompute::k10`].
    pub(crate) fn k13(&self) -> Option<&K13Pipeline> {
        self.k13.as_ref()
    }

    /// The K14 (Q6_K MMQ) pipeline, when this device grants
    /// the workgroup storage. See [`WgpuCompute::k13`].
    pub(crate) fn k14(&self) -> Option<&K14Pipeline> {
        self.k14.as_ref()
    }

    /// The K22 (Q4_K MMQ, register-organization
    /// prototype) pipeline, when this device grants the
    /// workgroup storage its staging set needs. See
    /// [`WgpuCompute::k13`] for the fallback contract.
    pub(crate) fn k22(&self) -> Option<&K22Pipeline> {
        self.k22.as_ref()
    }

    /// Whether the resident session prefers the K22
    /// product over the standard K13 product for Q4_K MMQ
    /// projections: config resolved at construction from
    /// [`K22_MMQ_ENV`], never a constant compiled into the
    /// selection seam. `false` (the default) leaves the
    /// shipped ladder untouched.
    pub(crate) fn k22_mmq(&self) -> bool {
        self.k22_mmq
    }

    /// Whether the resident session routes multi-row
    /// attention calls through K6's staged flash form:
    /// config resolved at construction from
    /// [`FLASH_STAGED_ENV`], never a constant compiled into
    /// the routing. `true` is the default (the variable
    /// unset); the explicit off is the value `0` on the
    /// same variable. The session also requires
    /// the staged pipeline to exist on this device
    /// (`K6Pipeline::flash_staged_available`) and the call
    /// to carry more than one row.
    pub(crate) fn flash_staged(&self) -> bool {
        self.flash_staged
    }

    /// The K23 (Q4_K MMQ, staging-repack prototype)
    /// pipeline, when this device grants the workgroup
    /// storage its staging set needs. See
    /// [`WgpuCompute::k13`] for the fallback contract.
    pub(crate) fn k23(&self) -> Option<&K23Pipeline> {
        self.k23.as_ref()
    }

    /// Whether the resident session prefers the K23
    /// product over the standard K13 product for Q4_K MMQ
    /// projections (below K22's rung): config resolved at
    /// construction from [`K23_MMQ_ENV`], never a constant
    /// compiled into the selection seam. `false` (the
    /// default) leaves the shipped ladder untouched.
    pub(crate) fn k23_mmq(&self) -> bool {
        self.k23_mmq
    }

    /// The K23 (Q4_K coopmat MMQ) pipeline, present only
    /// on a device opened through the coopmat bridge
    /// whose probes passed. See [`WgpuCompute::k13`] for
    /// the fallback contract.
    pub(crate) fn k23_coopmat(&self) -> Option<&K23CoopmatPipeline> {
        self.k23_coopmat.as_ref()
    }

    /// Whether the resident session prefers the K23
    /// coopmat product over the other Q4_K MMQ forms:
    /// config resolved at construction from
    /// [`COOPMAT_ENV`], never a constant compiled into
    /// the selection seam. `false` (the default) leaves
    /// the shipped ladder untouched.
    pub(crate) fn coopmat(&self) -> bool {
        self.coopmat
    }

    /// The MMQ split-K routing config: config resolved
    /// at construction from [`MMQ_SPLITK_ENV`], never a
    /// constant compiled into the selection seam.
    /// [`MmqSplitK::Off`] (the default) leaves the
    /// shipped ladder untouched.
    pub(crate) fn mmq_splitk(&self) -> MmqSplitK {
        self.mmq_splitk
    }

    /// The split factor this backend's config selects
    /// for a projection with the given unsplit tile count
    /// and super-block columns (see [`mmq_splitk_factor`]
    /// for the rule). A factor of 1 means the unsplit
    /// product serves.
    pub(crate) fn mmq_splitk_factor_for(&self, tiles: u32, blocks_per_row: u32) -> u32 {
        mmq_splitk_factor(self.mmq_splitk, tiles, blocks_per_row)
    }

    /// The K17 (DFlash NEOX rotate + draft attention)
    /// pipeline. Crate-internal: the draft session
    /// ([`crate::dflash_session`]) composes it.
    pub(crate) fn k17(&self) -> &K17Pipeline {
        &self.k17
    }

    /// The K15 (Q4_K wide-tile double-buffered MMQ)
    /// pipeline, when this device grants the workgroup
    /// storage its double-buffered staging set needs. See
    /// [`WgpuCompute::k13`] for the fallback contract.
    pub(crate) fn k15(&self) -> Option<&K15Pipeline> {
        self.k15.as_ref()
    }

    /// The K16 (Q6_K wide-tile double-buffered MMQ)
    /// pipeline, when this device grants the workgroup
    /// storage. See [`WgpuCompute::k15`].
    pub(crate) fn k16(&self) -> Option<&K16Pipeline> {
        self.k16.as_ref()
    }

    /// The chunk row count at or past which the resident
    /// session prefers the wide MMQ products (K15/K16)
    /// over the standard ones (K13/K14): config resolved
    /// at construction from [`WIDE_MIN_ROWS_ENV`], never a
    /// constant compiled into the selection seam.
    pub(crate) fn wide_min_rows(&self) -> u32 {
        self.wide_min_rows
    }

    /// The K18 (Q4_K small-batch) pipeline, when this
    /// device grants the workgroup storage its staging
    /// set needs. See [`WgpuCompute::k13`] for the
    /// fallback contract.
    pub(crate) fn k18_q4k(&self) -> Option<&K18Q4KPipeline> {
        self.k18_q4k.as_ref()
    }

    /// The K18 (Q6_K small-batch) pipeline, when this
    /// device grants the workgroup storage. See
    /// [`WgpuCompute::k18_q4k`].
    pub(crate) fn k18_q6k(&self) -> Option<&K18Q6KPipeline> {
        self.k18_q6k.as_ref()
    }

    /// The K18 (Q5_K small-batch) pipeline, when this
    /// device grants the workgroup storage. See
    /// [`WgpuCompute::k18_q4k`].
    pub(crate) fn k18_q5k(&self) -> Option<&K18Q5KPipeline> {
        self.k18_q5k.as_ref()
    }

    /// The largest batch row count at which the resident
    /// session prefers the small-batch products (K18)
    /// over the GEMV forms, inside the band from 2 rows
    /// up: config resolved at construction from
    /// [`SMALL_BATCH_MAX_ROWS_ENV`], never a constant
    /// compiled into the selection seam. The shipped
    /// default (1) routes nothing.
    pub(crate) fn small_batch_max_rows(&self) -> u32 {
        self.small_batch_max_rows
    }

    /// The K19 (Q4_K register-streaming small-batch)
    /// pipeline. Unconditional: the kernel's only
    /// workgroup memory is its reduction partials, so no
    /// device class refuses it (contrast the K18 staging
    /// sets, which are `Option`s). It is still only
    /// routed to under [`WgpuCompute::mmv_max_rows`].
    pub(crate) fn k19(&self) -> &K19Pipeline {
        &self.k19
    }

    /// The K20 (Q6_K register-streaming small-batch)
    /// pipeline. Unconditional: the kernel's only
    /// workgroup memory is its reduction partials, so no
    /// device class refuses it (contrast the K18 staging
    /// sets, which are `Option`s). It is still only
    /// routed to under [`WgpuCompute::mmv_max_rows`].
    pub(crate) fn k20(&self) -> &K20Pipeline {
        &self.k20
    }

    /// The K21 (Q5_K register-streaming small-batch)
    /// pipeline. Unconditional: its only workgroup
    /// memory is the reduction partials, so no device
    /// class refuses it. See [`WgpuCompute::k18_q4k`]
    /// for the staged sets' fallback contract.
    pub(crate) fn k21(&self) -> &K21Pipeline {
        &self.k21
    }

    /// The K21 finish-carrying pipeline (the Q5_K
    /// register-streaming small-batch product with
    /// K3's fused head finish at its reduction's final
    /// store), for the resident session's Q5_K head
    /// sites. Registered at construction; the block
    /// head site routes to it through the session's
    /// finish ladder rung (see
    /// bench/k21-finish-2026-11-06.md), and only when
    /// [`WgpuCompute::k21f_head`] also allows it.
    pub(crate) fn k21_finish(&self) -> &K21FinishPipeline {
        &self.k21_finish
    }

    /// Whether the resident session's block head may
    /// route its Q5_K product to the finish-carrying
    /// K21 form: config resolved at construction from
    /// [`K21F_HEAD_ENV`], never a constant compiled
    /// into the selection seam. `true` (the shipped
    /// default, [`DEFAULT_K21F_HEAD`]) is necessary
    /// but not sufficient, the shared band must also
    /// admit the block batch (see
    /// `session::try_mmv_q5k_finish`); `false` (the
    /// explicit `0`, or a fail-closed unrecognized
    /// value) leaves the block head on the K3 fused
    /// call at every shared-MMV-ceiling position.
    pub(crate) fn k21f_head(&self) -> bool {
        self.k21f_head
    }

    /// The largest batch row count at which the resident
    /// session prefers the register-streaming products
    /// (the MMV family: K19 for Q4_K, K20 for Q6_K,
    /// K21 for Q5_K) over the K18 and GEMV forms, inside
    /// the band from 2 rows up: config resolved at
    /// construction from [`MMV_MAX_ROWS_ENV`], never a
    /// constant compiled into the selection seam. The
    /// shipped default (16) opens the gate to the
    /// kernels' full band. One threshold
    /// serves the whole register-streaming family, so an
    /// A/B enables all three arms together; the band
    /// table ([`WgpuCompute::mmv_routes`]) then narrows
    /// the ceiling per family and shape.
    pub(crate) fn mmv_max_rows(&self) -> u32 {
        self.mmv_max_rows
    }

    /// Whether the register-streaming product for
    /// `family` serves a projection of shape class
    /// `shape` at `batch` rows: the batch must sit
    /// under the [`WgpuCompute::mmv_max_rows`] ceiling
    /// AND inside the family's band cell for the shape
    /// (resolved at construction from the
    /// `WGPU_COMPUTE_MMV_*_ROWS` env cells, shipped
    /// defaults derived in
    /// bench/mmv-band-routing-2026-11-06.md). The
    /// ceiling alone used to be the whole policy; the
    /// dose-response cliff in the middle of its band
    /// is what the table removes.
    pub(crate) fn mmv_routes(&self, family: MmvFamily, shape: MmvShape, batch: u32) -> bool {
        self.mmv_bands
            .routes(self.mmv_max_rows, family, shape, batch)
    }

    /// The resolved band table itself, for routing
    /// seams that resolve their candidates from the
    /// policy values directly (the draft session's
    /// ladder; see `dflash_session::proj_candidates`).
    pub(crate) fn mmv_bands(&self) -> &MmvBands {
        &self.mmv_bands
    }

    /// The resolved MMV band table in its one-line described
    /// form, for the capture manifest: the bands are resolved
    /// configuration, so a manifest that omits them cannot
    /// state the run it claims to describe.
    pub(crate) fn mmv_bands_describe(&self) -> String {
        self.mmv_bands.describe()
    }

    /// The calibration profile that governed this
    /// backend's MMV resolution, as the one-line
    /// description of its adapter key, or `None` when
    /// no profile loaded: the capture manifest states
    /// it (as `"none"` when absent) beside the resolved
    /// bands, so a run's governing table is never a
    /// mystery. See [`crate::profile`].
    pub(crate) fn mmv_profile_key(&self) -> Option<&str> {
        self.mmv_profile_key.as_deref()
    }

    /// The largest batch row count at which the resident
    /// session prefers the chunked GEMV form (several
    /// windowed GEMV dispatches at power-of-two row
    /// counts) over the single GEMV dispatch, inside the
    /// band from 2 rows up: config resolved at
    /// construction from [`CHUNKED_GEMV_MAX_ROWS_ENV`],
    /// never a constant compiled into the selection
    /// seam. The shipped default (1) routes nothing.
    pub(crate) fn chunked_gemv_max_rows(&self) -> u32 {
        self.chunked_gemv_max_rows
    }

    /// Drop all stored KV state, device buffers and host mirror.
    /// The next attention call rebuilds the state from its
    /// config and expects the caller's sequence position to be 0
    /// again. Mirrors `CpuCompute::reset`.
    pub fn reset(&self) {
        dbg_log!("gpu kv state reset");
        *self.kv.borrow_mut() = None;
    }

    /// Ensure a device copy of `view`'s tensor exists in the
    /// cache, uploading it through its dtype's kernel on first
    /// use. A later tensor arriving under a cached name with a
    /// different dtype or geometry is a refusal, not a
    /// re-upload: the name is the identity the loader gave it.
    fn ensure_uploaded(&self, view: &TensorView, step: &'static str) -> Result<()> {
        let (inn, out) = (view.shape[0] as usize, view.shape[1] as usize);
        {
            let cache = self.weights.borrow();
            if let Some(existing) = cache.get(&view.name) {
                let (e_in, e_out, e_bytes) = existing.geometry();
                let kind_matches = matches!(
                    (view.dtype, existing),
                    (Dtype::Q4K, CachedWeights::Q4K(_))
                        | (Dtype::Q6K, CachedWeights::Q6K(_))
                        | (Dtype::Q5K, CachedWeights::Q5K(_))
                );
                if !kind_matches || e_in != inn || e_out != out || e_bytes != view.data.len() as u64
                {
                    return Err(step_failed(
                        step,
                        format!(
                            "tensor {} was already uploaded through {} as [{} x {}] ({} bytes) \
                             but now arrives as [{inn} x {out}] ({} bytes) of another kind; \
                             a tensor name is an identity in this backend, not a slot",
                            view.name,
                            existing.kernel(),
                            e_in,
                            e_out,
                            e_bytes,
                            view.data.len()
                        ),
                    ));
                }
                return Ok(());
            }
        }
        let uploaded = match view.dtype {
            Dtype::Q4K => CachedWeights::Q4K(
                self.k1
                    .upload(&self.ctx, &view.name, &view.data, inn, out)
                    .map_err(|e| step_failed(step, e.to_string()))?,
            ),
            Dtype::Q6K => CachedWeights::Q6K(
                self.k2
                    .upload(&self.ctx, &view.name, &view.data, inn, out)
                    .map_err(|e| step_failed(step, e.to_string()))?,
            ),
            Dtype::Q5K => CachedWeights::Q5K(
                self.k3
                    .upload(&self.ctx, &view.name, &view.data, inn, out)
                    .map_err(|e| step_failed(step, e.to_string()))?,
            ),
            other => {
                return Err(step_failed(
                    step,
                    format!(
                        "tensor {} has dtype {} (ggml type id {}), which has no GPU kernel in \
                         this backend yet (K1 covers Q4_K, K2 covers Q6_K, K3 covers Q5_K); \
                         BACKEND.md Section 6 level L2 runs such types on the CPU backend \
                         as a load-time composition decision, not inside this step",
                        view.name,
                        gguf::ggml::type_name(dtype_id(other)),
                        dtype_id(other)
                    ),
                ))
            }
        };
        self.weights
            .borrow_mut()
            .insert(view.name.clone(), uploaded);
        Ok(())
    }

    /// Shared projection body, mirroring `cpu-compute`'s: the FFN
    /// calls it under `swiglu_ffn`, the head under `lm_head`, so
    /// refusals name the step the forward thinks it is in.
    fn proj_impl(
        &self,
        x: &[f32],
        weight: &TensorView,
        rows: usize,
        step: &'static str,
    ) -> Result<Vec<f32>> {
        if weight.shape.len() != 2 {
            return Err(step_failed(
                step,
                format!(
                    "projection tensor {} has shape {:?}, expected [in, out] in GGUF stored order",
                    weight.name, weight.shape
                ),
            ));
        }
        let inn = weight.shape[0] as usize;
        if x.len() != rows * inn {
            return Err(step_failed(
                step,
                format!(
                    "input holds {} elements but {rows} rows of width {inn} need {}",
                    x.len(),
                    rows * inn
                ),
            ));
        }
        self.ensure_uploaded(weight, step)?;
        let cache = self.weights.borrow();
        let cached = cache.get(&weight.name).ok_or_else(|| {
            step_failed(
                step,
                format!(
                    "tensor {} was uploaded but is missing from the cache",
                    weight.name
                ),
            )
        })?;
        match (weight.dtype, cached) {
            (Dtype::Q4K, CachedWeights::Q4K(w)) => self
                .k1
                .run(&self.ctx, w, x)
                .map_err(|e| step_failed(step, e.to_string())),
            (Dtype::Q6K, CachedWeights::Q6K(w)) => self
                .k2
                .run(&self.ctx, w, x)
                .map_err(|e| step_failed(step, e.to_string())),
            (Dtype::Q5K, CachedWeights::Q5K(w)) => self
                .k3
                .run(&self.ctx, w, x)
                .map_err(|e| step_failed(step, e.to_string())),
            _ => Err(step_failed(
                step,
                format!(
                    "tensor {} changed kind between upload and run; the cache holds it \
                     through {}, which its dtype does not select",
                    weight.name,
                    cached.kernel()
                ),
            )),
        }
    }
}

impl Compute for WgpuCompute {
    fn embed(&self, table: &TensorView, tokens: &[u32], cfg: &GlimmerConfig) -> Result<Vec<f32>> {
        self.k8.embed(&self.ctx, table, tokens, cfg)
    }

    fn rms_norm(&self, x: &[f32], weight: &TensorView, rows: usize, eps: f32) -> Result<Vec<f32>> {
        let step = "rms_norm";
        // The norm weight is a small F32 vector in the real file;
        // dequantize it here (the same view-to-values step, with
        // the same byte-length validation, that the CPU backend
        // takes in its dequantize_view) and let K4 run the
        // normalization itself on the device.
        let ty = match weight.dtype {
            Dtype::F32 => gguf::GgmlType::F32,
            Dtype::F16 => gguf::GgmlType::F16,
            Dtype::Q8_0 => gguf::GgmlType::Q8_0,
            Dtype::Q4K => gguf::GgmlType::Q4K,
            Dtype::Q5K => gguf::GgmlType::Q5K,
            Dtype::Q6K => gguf::GgmlType::Q6K,
            Dtype::Bf16 => gguf::GgmlType::Bf16,
            other => {
                return Err(step_failed(
                    step,
                    format!(
                        "norm weight {} has dtype {} (ggml type id {}), which this backend \
                         cannot dequantize for K4",
                        weight.name,
                        gguf::ggml::type_name(dtype_id(other)),
                        dtype_id(other)
                    ),
                ))
            }
        };
        let elements = weight.elements() as usize;
        let expected = ty.byte_len(elements as u64).ok_or_else(|| {
            step_failed(
                step,
                format!(
                    "norm weight {} has {elements} elements, not a whole number of blocks",
                    weight.name
                ),
            )
        })?;
        if expected != weight.data.len() as u64 {
            return Err(step_failed(
                step,
                format!(
                    "norm weight {} holds {} bytes but {elements} elements need {expected}",
                    weight.name,
                    weight.data.len()
                ),
            ));
        }
        let w = gguf::dequant::dequantize(ty, &weight.data, elements);
        self.k4
            .rms_norm(&self.ctx, x, &w, rows, eps)
            .map_err(|e| step_failed(step, e.to_string()))
    }

    fn rms_norm_plain(&self, x: &[f32], rows: usize, eps: f32) -> Result<Vec<f32>> {
        self.k4
            .rms_norm_plain(&self.ctx, x, rows, eps)
            .map_err(|e| step_failed("rms_norm_plain", e.to_string()))
    }

    fn proj(&self, x: &[f32], weight: &TensorView, rows: usize) -> Result<Vec<f32>> {
        self.proj_impl(x, weight, rows, "proj")
    }

    fn attention(&self, req: &AttentionRequest<'_>, cache: &mut dyn Cache) -> Result<Vec<f32>> {
        let step = "attention";
        let cfg = req.cfg;
        // The validation mirrors cpu-compute's, the spec: the
        // attention crate stores keys and values at one head
        // dimension, the request's kind must be the config's, and
        // the projections must be whole rows.
        if cfg.key_length != cfg.value_length {
            return Err(step_failed(
                step,
                format!(
                    "key_length {} != value_length {}: the attention crate's KvCache stores keys and values at one head dimension",
                    cfg.key_length, cfg.value_length
                ),
            ));
        }
        if cfg.layer_kind(req.layer) != Some(req.kind) {
            return Err(step_failed(
                step,
                format!(
                    "request kind {:?} does not match the config's kind for layer {}",
                    req.kind, req.layer
                ),
            ));
        }
        let q_width = cfg.q_width() as usize;
        let k_width = cfg.k_width() as usize;
        let v_width = cfg.v_width() as usize;
        if q_width == 0 || req.q.len() % q_width != 0 {
            return Err(step_failed(
                step,
                format!(
                    "query projection holds {} elements, not a whole number of {q_width}-wide rows",
                    req.q.len()
                ),
            ));
        }
        let rows = req.q.len() / q_width;
        if req.k.len() != rows * k_width || req.v.len() != rows * v_width {
            return Err(step_failed(
                step,
                format!(
                    "key/value projections hold {}/{} elements for {rows} rows, expected {}/{}",
                    req.k.len(),
                    req.v.len(),
                    rows * k_width,
                    rows * v_width
                ),
            ));
        }

        let start = cache.position();
        let mut guard = self.kv.borrow_mut();
        if guard.is_none() {
            *guard = Some(GpuKvState::build(cfg, step)?);
        }
        let state = guard.as_mut().expect("kv state was just built");
        state.check_shape(cfg, step)?;
        let appended = state
            .host()
            .next_position(req.layer)
            .map_err(|e| step_failed(step, e.to_string()))?;
        if appended != start {
            return Err(step_failed(
                step,
                format!(
                    "KV state for layer {} is at position {appended} but the sequence cache is at position {start}: \
                     the two disagree, so attending would read the wrong history. \
                     Pair one Cache with one WgpuCompute per sequence, or call WgpuCompute::reset \
                     when a sequence is abandoned (a forward that failed midway leaves the KV state \
                     ahead of the cache and also needs a reset)",
                    req.layer
                ),
            ));
        }

        let head_dim = cfg.key_length as usize;
        let q_heads = cfg.q_heads as usize;
        let kv_heads = cfg.kv_heads as usize;
        let window = match req.kind {
            LayerKind::Global => None,
            LayerKind::SlidingWindow => Some(cfg.sliding_window as usize),
        };

        // Rotate Q and K in place on the device, sliding layers
        // only, at the SWA base; global layers are NoPE and pass
        // through unrotated (src/models/muse-glimmer.cpp).
        let q_buf = self.ctx.upload_f32("attention q", req.q);
        let k_buf = self.ctx.upload_f32("attention k", req.k);
        if req.kind == LayerKind::SlidingWindow {
            self.k5
                .rotate(
                    &self.ctx,
                    &q_buf,
                    rows,
                    q_heads,
                    head_dim,
                    start,
                    cfg.rope_freq_base_swa,
                )
                .map_err(|e| step_failed(step, e.to_string()))?;
            self.k5
                .rotate(
                    &self.ctx,
                    &k_buf,
                    rows,
                    kv_heads,
                    head_dim,
                    start,
                    cfg.rope_freq_base_swa,
                )
                .map_err(|e| step_failed(step, e.to_string()))?;
        }
        // The rotated keys come back exactly once: the host
        // mirror appends what the device computed, so the mirror
        // (positions, cross-checks, tests) and the device buffers
        // can never drift apart in what they hold.
        let k_rot = self
            .ctx
            .read_f32(&k_buf, req.k.len())
            .map_err(|e| step_failed(step, e.to_string()))?;
        for i in 0..rows {
            state
                .host_mut()
                .append(
                    req.layer,
                    &k_rot[i * k_width..(i + 1) * k_width],
                    &req.v[i * v_width..(i + 1) * v_width],
                    start + i,
                )
                .map_err(|e| step_failed(step, e.to_string()))?;
        }
        let (stored, first_pos) = {
            let view = state
                .host()
                .view(req.layer)
                .map_err(|e| step_failed(step, e.to_string()))?;
            let first = view.position_at(0).ok_or_else(|| {
                step_failed(
                    step,
                    "the layer's cache view is empty right after an append",
                )
            })?;
            (view.len(), first)
        };
        let appended_total = state
            .host()
            .next_position(req.layer)
            .map_err(|e| step_failed(step, e.to_string()))?;
        let layer_kv = state.ensure_layer(&self.ctx, req.layer, k_width, appended_total, step)?;
        // K5's second half: append the rotated K rows and the V
        // rows into the layer's resident buffers, at the slots
        // the cache semantics assign.
        let v_buf = self.ctx.upload_f32("attention v", req.v);
        self.k5
            .append(
                &self.ctx,
                &k_buf,
                &v_buf,
                &layer_kv.k,
                &layer_kv.v,
                rows,
                k_width,
                start,
                window,
            )
            .map_err(|e| step_failed(step, e.to_string()))?;
        // K6: scores plus the online-softmax reduce over the
        // resident buffers.
        let geom = AttendGeom {
            rows,
            q_heads,
            kv_heads,
            head_dim,
            stored,
            first_pos,
            start_pos: start,
            window,
            // The trait state's ring holds exactly one window,
            // so its slot modulus is the window itself.
            ring: window,
        };
        let scores = self
            .k6
            .scores(&self.ctx, &q_buf, layer_kv, &geom)
            .map_err(|e| step_failed(step, e.to_string()))?;
        let out = self
            .k6
            .reduce(&self.ctx, &scores, layer_kv, &geom)
            .map_err(|e| step_failed(step, e.to_string()))?;
        dbg_log!(
            "attention layer={} kind={:?} start={} rows={} stored={}",
            req.layer,
            req.kind,
            start,
            rows,
            stored
        );
        Ok(out)
    }

    fn attn_gate(&self, attn_out: &[f32], gate: &[f32]) -> Result<Vec<f32>> {
        self.k7
            .attn_gate(&self.ctx, attn_out, gate)
            .map_err(|e| step_failed("attn_gate", e.to_string()))
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.k4
            .add(&self.ctx, a, b)
            .map_err(|e| step_failed("add", e.to_string()))
    }

    fn swiglu_ffn(
        &self,
        x: &[f32],
        w: &LayerWeights,
        rows: usize,
        _cfg: &GlimmerConfig,
    ) -> Result<Vec<f32>> {
        let step = "swiglu_ffn";
        // The FFN end to end: gate and up through K1 (Q4_K in the
        // real file), the activation through K7, and the down
        // projection through K2 (Q6_K).
        let gate = self.proj_impl(x, &w.ffn_gate, rows, step)?;
        let up = self.proj_impl(x, &w.ffn_up, rows, step)?;
        let act = self
            .k7
            .swiglu(&self.ctx, &gate, &up)
            .map_err(|e| step_failed(step, e.to_string()))?;
        self.proj_impl(&act, &w.ffn_down, rows, step)
    }

    fn lm_head(&self, x: &[f32], head: &TensorView, cfg: &GlimmerConfig) -> Result<Vec<f32>> {
        let step = "lm_head";
        if head.shape.len() != 2 {
            return Err(step_failed(
                step,
                format!(
                    "output head {} has shape {:?}, expected [hidden, vocab] in GGUF stored order",
                    head.name, head.shape
                ),
            ));
        }
        let (inn, vocab) = (head.shape[0] as usize, head.shape[1] as usize);
        if inn != x.len() {
            return Err(step_failed(
                step,
                format!(
                    "output head {} reads {inn} inputs but the normed hidden state holds {}",
                    head.name,
                    x.len()
                ),
            ));
        }
        if vocab != cfg.vocab_size as usize {
            return Err(step_failed(
                step,
                format!(
                    "output head {} produces {vocab} logits but the config's vocabulary is {}",
                    head.name, cfg.vocab_size
                ),
            ));
        }
        self.proj_impl(x, head, 1, step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_min_rows_resolution() {
        // Unset: the default.
        assert_eq!(resolve_wide_min_rows(None), DEFAULT_WIDE_MIN_ROWS);
        // Plain values pass through, including the two
        // bench positions: 1 (wide everywhere the matrix
        // path runs) and 65 (past the batch cap: wide out
        // of the ladder, the A/B control).
        assert_eq!(resolve_wide_min_rows(Some("1")), 1);
        assert_eq!(resolve_wide_min_rows(Some("65")), 65);
        assert_eq!(resolve_wide_min_rows(Some(" 48 ")), 48);
        // Unparsable values fall back to the default rather
        // than failing backend construction.
        assert_eq!(resolve_wide_min_rows(Some("wide")), DEFAULT_WIDE_MIN_ROWS);
        assert_eq!(resolve_wide_min_rows(Some("")), DEFAULT_WIDE_MIN_ROWS);
        assert_eq!(resolve_wide_min_rows(Some("-3")), DEFAULT_WIDE_MIN_ROWS);
    }

    #[test]
    fn small_batch_max_rows_resolution() {
        // Unset: the default, which is off (1 routes
        // nothing, since the ladder's band starts at 2).
        assert_eq!(
            resolve_small_batch_max_rows(None),
            DEFAULT_SMALL_BATCH_MAX_ROWS
        );
        assert_eq!(DEFAULT_SMALL_BATCH_MAX_ROWS, 1);
        // Plain values pass through, including the
        // proof positions: 15 (the routed band under
        // GEMM_MIN_ROWS) and 16 (the kernel's full
        // band).
        assert_eq!(resolve_small_batch_max_rows(Some("15")), 15);
        assert_eq!(resolve_small_batch_max_rows(Some("16")), 16);
        assert_eq!(resolve_small_batch_max_rows(Some(" 0 ")), 0);
        // Unparsable values fall back to the default
        // rather than failing backend construction.
        assert_eq!(
            resolve_small_batch_max_rows(Some("small")),
            DEFAULT_SMALL_BATCH_MAX_ROWS
        );
        assert_eq!(
            resolve_small_batch_max_rows(Some("")),
            DEFAULT_SMALL_BATCH_MAX_ROWS
        );
        assert_eq!(
            resolve_small_batch_max_rows(Some("-2")),
            DEFAULT_SMALL_BATCH_MAX_ROWS
        );
    }

    #[test]
    fn mmv_max_rows_resolution() {
        // Unset: the shipped default, 16, the kernels'
        // full band. The default moved from 1 (off) to
        // 16 on the baseline session's card verdict
        // (bench/baseline-session-2026-11-06.md leg
        // (d): cap 16 beat the unset control at both
        // depths, 21.73 vs 17.39 and 16.33 vs 10.14,
        // with identity and golden green in every arm).
        assert_eq!(resolve_mmv_max_rows(None), DEFAULT_MMV_MAX_ROWS);
        assert_eq!(DEFAULT_MMV_MAX_ROWS, 16);
        // Plain values pass through, including the
        // proof positions: 15 (the routed band under
        // GEMM_MIN_ROWS) and 16 (the kernels' full
        // band). The explicit close still closes: a set
        // "1" wins over the shipped default.
        assert_eq!(resolve_mmv_max_rows(Some("1")), 1);
        assert_eq!(resolve_mmv_max_rows(Some("15")), 15);
        assert_eq!(resolve_mmv_max_rows(Some("16")), 16);
        assert_eq!(resolve_mmv_max_rows(Some(" 0 ")), 0);
        // Unparsable values fall back to the CLOSED
        // ceiling, not the shipped default, rather than
        // failing backend construction: a typo in the
        // env must never widen routing. This is the
        // fail-closed behavior the malformed case had
        // before the default moved, kept deliberately.
        assert_eq!(MMV_MAX_ROWS_FALLBACK, 1);
        assert_eq!(resolve_mmv_max_rows(Some("mmv")), MMV_MAX_ROWS_FALLBACK);
        assert_eq!(resolve_mmv_max_rows(Some("")), MMV_MAX_ROWS_FALLBACK);
        assert_eq!(resolve_mmv_max_rows(Some("-2")), MMV_MAX_ROWS_FALLBACK);
    }

    #[test]
    fn mmv_band_parse_forms() {
        // Single rows, spans, and lists, whitespace
        // tolerated.
        assert_eq!(parse_mmv_band("16"), Some(vec![(16, 16)]));
        assert_eq!(parse_mmv_band("6-12"), Some(vec![(6, 12)]));
        assert_eq!(parse_mmv_band("2,4-12"), Some(vec![(2, 2), (4, 12)]));
        assert_eq!(parse_mmv_band(" 2 , 4 - 12 "), Some(vec![(2, 2), (4, 12)]));
        // Ranges are intersected with the kernels'
        // own band (2..=16): a span reaching past it
        // clamps, and a span missing it entirely
        // drops out, so "1" is a valid cell that
        // routes nothing.
        assert_eq!(parse_mmv_band("0-99"), Some(vec![(2, 16)]));
        assert_eq!(parse_mmv_band("1"), Some(vec![]));
        assert_eq!(parse_mmv_band("17-20"), Some(vec![]));
    }

    #[test]
    fn mmv_band_parse_rejects_malformed() {
        // Any malformed item poisons the whole
        // value: resolution then falls back to the
        // shipped cell rather than guessing a band.
        for raw in [
            "",
            "x",
            "2-",
            "-3",
            "5-2",
            "2,,3",
            "2-3-4",
            "1.5",
            "99999999999999999999",
        ] {
            assert_eq!(parse_mmv_band(raw), None, "input {raw:?}");
        }
    }

    #[test]
    fn mmv_band_resolution_falls_back_to_the_shipped_cell() {
        // Unset: the shipped cell.
        let bands = resolve_mmv_bands(|_| None);
        assert!(bands.routes(16, MmvFamily::Q4K, MmvShape::Attn, 6));
        assert!(!bands.routes(16, MmvFamily::Q4K, MmvShape::Attn, 2));
        // A parsable override replaces exactly its
        // own cell; the rest of the table is the
        // shipped one.
        let bands = resolve_mmv_bands(|name| {
            (name == mmv_band_env(MmvFamily::Q4K, MmvShape::Attn)).then(|| "2-3".to_string())
        });
        assert!(bands.routes(16, MmvFamily::Q4K, MmvShape::Attn, 2));
        assert!(!bands.routes(16, MmvFamily::Q4K, MmvShape::Attn, 6));
        assert!(bands.routes(16, MmvFamily::Q6K, MmvShape::Attn, 6));
        // An unparsable override falls back to the
        // shipped cell: it neither disables nor
        // widens the cell.
        let bands = resolve_mmv_bands(|name| {
            (name == mmv_band_env(MmvFamily::Q6K, MmvShape::Kv)).then(|| "junk".to_string())
        });
        assert!(bands.routes(16, MmvFamily::Q6K, MmvShape::Kv, 12));
        assert!(!bands.routes(16, MmvFamily::Q6K, MmvShape::Kv, 6));
        // An override of "1" closes its cell.
        let bands = resolve_mmv_bands(|name| {
            (name == mmv_band_env(MmvFamily::Q5K, MmvShape::Attn)).then(|| "1".to_string())
        });
        assert!(!bands.routes(16, MmvFamily::Q5K, MmvShape::Attn, 7));
    }

    #[test]
    fn mmv_profile_layer_sits_between_env_and_shipped() {
        use crate::profile::{AdapterKey, CalibrationProfile};
        // A profile measured on some other sweep: it
        // opens Q4_K KV at batch 12 (a batch the
        // shipped cell "2-3" leaves closed), narrows
        // Q4_K attention to "2-3" (the shipped cell is
        // "6-12"), and states an unparsable Q5_K KV
        // cell, which must fall through to the shipped
        // "16" rather than poison the table.
        let mut profile = CalibrationProfile::new(AdapterKey {
            name: "test adapter".to_string(),
            backend: "Vulkan".to_string(),
            driver: "test driver".to_string(),
            driver_info: "test driver info".to_string(),
        });
        profile.set_band_cell(MmvFamily::Q4K, MmvShape::Kv, "12".to_string());
        profile.set_band_cell(MmvFamily::Q4K, MmvShape::Attn, "2-3".to_string());
        profile.set_band_cell(MmvFamily::Q5K, MmvShape::Kv, "junk".to_string());

        // No env: the profile governs the cells it
        // states, the shipped table the rest.
        let bands = resolve_mmv_bands_with_profile(|_| None, Some(&profile));
        assert!(bands.routes(16, MmvFamily::Q4K, MmvShape::Kv, 12));
        assert!(!bands.routes(16, MmvFamily::Q4K, MmvShape::Kv, 4));
        assert!(bands.routes(16, MmvFamily::Q4K, MmvShape::Attn, 2));
        assert!(!bands.routes(16, MmvFamily::Q4K, MmvShape::Attn, 6));
        assert!(bands.routes(16, MmvFamily::Q5K, MmvShape::Kv, 16));
        assert!(!bands.routes(16, MmvFamily::Q5K, MmvShape::Kv, 2));
        assert!(bands.routes(16, MmvFamily::Q6K, MmvShape::Attn, 6));

        // An env override beats the profile for the
        // same cell, and only that cell.
        let bands = resolve_mmv_bands_with_profile(
            |name| {
                (name == mmv_band_env(MmvFamily::Q4K, MmvShape::Attn)).then(|| "6-9".to_string())
            },
            Some(&profile),
        );
        assert!(bands.routes(16, MmvFamily::Q4K, MmvShape::Attn, 6));
        assert!(!bands.routes(16, MmvFamily::Q4K, MmvShape::Attn, 2));
        assert!(bands.routes(16, MmvFamily::Q4K, MmvShape::Kv, 12));

        // An unparsable env value falls through to the
        // profile layer, not past it: the profile is
        // the next statement of the cell's value.
        let bands = resolve_mmv_bands_with_profile(
            |name| {
                (name == mmv_band_env(MmvFamily::Q4K, MmvShape::Attn)).then(|| "junk".to_string())
            },
            Some(&profile),
        );
        assert!(bands.routes(16, MmvFamily::Q4K, MmvShape::Attn, 2));
        assert!(!bands.routes(16, MmvFamily::Q4K, MmvShape::Attn, 6));

        // The ceiling resolves in the same precedence:
        // env over profile over shipped, and a present
        // but unparsable env value still lands on the
        // closed fallback without ever reaching the
        // profile.
        assert_eq!(resolve_mmv_max_rows_with_profile(None, Some(4)), 4);
        assert_eq!(resolve_mmv_max_rows_with_profile(None, None), 16);
        assert_eq!(resolve_mmv_max_rows_with_profile(Some("9"), Some(4)), 9);
        assert_eq!(resolve_mmv_max_rows_with_profile(Some("junk"), Some(4)), 1);
    }

    #[test]
    fn mmv_shape_classification() {
        // The sweep's measured geometries.
        assert_eq!(mmv_shape_of(6656, 6656), MmvShape::Attn);
        assert_eq!(mmv_shape_of(6656, 19968), MmvShape::Ffn);
        assert_eq!(mmv_shape_of(6656, 256), MmvShape::Kv);
        // The real model's other projections land in
        // the attention class: Q (4096 out), O (4096
        // in), down (19968 in).
        assert_eq!(mmv_shape_of(6656, 4096), MmvShape::Attn);
        assert_eq!(mmv_shape_of(4096, 6656), MmvShape::Attn);
        assert_eq!(mmv_shape_of(19968, 6656), MmvShape::Attn);
        // The class boundaries themselves: an output
        // at exactly an eighth of the input is KV, at
        // exactly double is FFN.
        assert_eq!(mmv_shape_of(2048, 256), MmvShape::Kv);
        assert_eq!(mmv_shape_of(6656, 832), MmvShape::Kv);
        assert_eq!(mmv_shape_of(6656, 833), MmvShape::Attn);
        assert_eq!(mmv_shape_of(6656, 13312), MmvShape::Ffn);
        assert_eq!(mmv_shape_of(6656, 202048), MmvShape::Ffn);
    }

    #[test]
    fn mmv_default_bands_match_the_measured_win_map() {
        // The shipped table against the verdict of
        // the 2026-11-06 sweep, cell by cell
        // (bench/results/mmv-sweep-20261106-212335/,
        // derived in
        // bench/mmv-band-routing-2026-11-06.md). Each
        // row is batches 2..=16 in order, true where
        // the family's product routes under a fully
        // open ceiling. A cell the sweep measured as
        // a tie or a loss for the MMV arm is false;
        // rows the sweep did not measure (10, 11, 13,
        // 14, 15) are true only where two measured
        // wins bracket them (Q5_K's run to 16; Q4_K
        // and Q6_K end at 12, their last measured
        // win, since row 16 on those families is the
        // MMQ rung in both arms and carries no MMV
        // measurement).
        let bands = default_mmv_bands();
        let expected: [(MmvFamily, MmvShape, [bool; 15]); 9] = [
            (
                MmvFamily::Q4K,
                MmvShape::Attn,
                [
                    false, false, false, false, true, true, true, true, true, true, true, false,
                    false, false, false,
                ],
            ),
            (
                MmvFamily::Q4K,
                MmvShape::Ffn,
                [
                    true, false, true, true, true, true, true, true, true, true, true, false,
                    false, false, false,
                ],
            ),
            (
                MmvFamily::Q4K,
                MmvShape::Kv,
                [
                    true, true, false, false, false, false, false, false, false, false, false,
                    false, false, false, false,
                ],
            ),
            (MmvFamily::Q5K, MmvShape::Attn, [true; 15]),
            (
                MmvFamily::Q5K,
                MmvShape::Ffn,
                [
                    false, true, true, true, true, true, true, true, true, true, true, true, true,
                    true, true,
                ],
            ),
            (
                MmvFamily::Q5K,
                MmvShape::Kv,
                [
                    false, false, false, false, false, false, false, false, false, false, false,
                    false, false, false, true,
                ],
            ),
            (
                MmvFamily::Q6K,
                MmvShape::Attn,
                [
                    true, true, true, true, true, true, true, true, true, true, true, false, false,
                    false, false,
                ],
            ),
            (
                MmvFamily::Q6K,
                MmvShape::Ffn,
                [
                    true, true, true, true, true, true, true, true, true, true, true, false, false,
                    false, false,
                ],
            ),
            (
                MmvFamily::Q6K,
                MmvShape::Kv,
                [
                    false, false, false, false, false, false, false, false, false, false, true,
                    false, false, false, false,
                ],
            ),
        ];
        for (family, shape, table) in expected {
            for (i, &want) in table.iter().enumerate() {
                let batch = i as u32 + 2;
                assert_eq!(
                    bands.routes(16, family, shape, batch),
                    want,
                    "{} {} batch {batch}",
                    family.name(),
                    shape.name()
                );
            }
        }
        // The construction log prints this same
        // table in band syntax.
        assert!(bands
            .describe()
            .contains("q4k[attn 6-12 ffn 2,4-12 kv 2-3]"));
        assert!(bands.describe().contains("q5k[attn 2-16 ffn 3-16 kv 16]"));
        assert!(bands.describe().contains("q6k[attn 2-12 ffn 2-12 kv 12]"));
    }

    #[test]
    fn mmv_routes_ceiling_composition() {
        let bands = default_mmv_bands();
        // The explicit close (ceiling 1) routes nothing,
        // whatever the table says.
        for family in [MmvFamily::Q4K, MmvFamily::Q5K, MmvFamily::Q6K] {
            for shape in [MmvShape::Attn, MmvShape::Ffn, MmvShape::Kv] {
                for batch in 2..=16 {
                    assert!(
                        !bands.routes(1, family, shape, batch),
                        "{} {} batch {batch} routed under the closed ceiling",
                        family.name(),
                        shape.name()
                    );
                }
            }
        }
        // The shipped ceiling is the kernels' full band
        // (16 since the 2026-11-06 flip, on the baseline
        // session's card verdict,
        // bench/baseline-session-2026-11-06.md leg (d)),
        // so at the default the ceiling itself clips
        // nothing and routing is exactly the band
        // table's: measured wins route, measured losses
        // and holes do not.
        assert_eq!(DEFAULT_MMV_MAX_ROWS, 16);
        assert!(bands.routes(DEFAULT_MMV_MAX_ROWS, MmvFamily::Q4K, MmvShape::Attn, 6));
        assert!(!bands.routes(DEFAULT_MMV_MAX_ROWS, MmvFamily::Q4K, MmvShape::Attn, 2));
        assert!(!bands.routes(DEFAULT_MMV_MAX_ROWS, MmvFamily::Q4K, MmvShape::Ffn, 3));
        assert!(bands.routes(DEFAULT_MMV_MAX_ROWS, MmvFamily::Q4K, MmvShape::Kv, 2));
        assert!(!bands.routes(DEFAULT_MMV_MAX_ROWS, MmvFamily::Q4K, MmvShape::Kv, 12));
        assert!(bands.routes(DEFAULT_MMV_MAX_ROWS, MmvFamily::Q5K, MmvShape::Kv, 16));
        assert!(bands.routes(DEFAULT_MMV_MAX_ROWS, MmvFamily::Q6K, MmvShape::Kv, 12));
        assert!(!bands.routes(DEFAULT_MMV_MAX_ROWS, MmvFamily::Q6K, MmvShape::Attn, 16));
        // A ceiling clips cells from above: at cap 4
        // the Q4_K attention cell (6-12) admits
        // nothing, while Q6_K attention admits its
        // measured wins at rows 2..=4 and no further.
        assert!(!bands.routes(4, MmvFamily::Q4K, MmvShape::Attn, 6));
        assert!(bands.routes(4, MmvFamily::Q6K, MmvShape::Attn, 4));
        assert!(!bands.routes(4, MmvFamily::Q6K, MmvShape::Attn, 5));
        // The session dose positions: at cap 12 the
        // Q5_K KV cell (16 only) is still closed; at
        // cap 16 it opens.
        assert!(!bands.routes(12, MmvFamily::Q5K, MmvShape::Kv, 16));
        assert!(bands.routes(16, MmvFamily::Q5K, MmvShape::Kv, 16));
        // Batch 1 never routes (the kernels' band
        // starts at 2), even under a huge ceiling, and
        // neither does a batch past the kernels' cap.
        assert!(!bands.routes(u32::MAX, MmvFamily::Q5K, MmvShape::Attn, 1));
        assert!(!bands.routes(u32::MAX, MmvFamily::Q5K, MmvShape::Attn, 17));
    }

    #[test]
    fn k22_mmq_switch_resolution() {
        // Unset: off. The shipped ladder is the ladder
        // with K22 nowhere in it.
        assert!(!resolve_k22_mmq(None));
        // The exact value "1" (trimmed) is the only
        // enabling position: the switch is fail-closed by
        // design, so a typo routes the proven kernel, not
        // the prototype.
        assert!(resolve_k22_mmq(Some("1")));
        assert!(resolve_k22_mmq(Some(" 1 ")));
        assert!(!resolve_k22_mmq(Some("0")));
        assert!(!resolve_k22_mmq(Some("true")));
        assert!(!resolve_k22_mmq(Some("")));
        assert!(!resolve_k22_mmq(Some("11")));
    }

    #[test]
    fn flash_staged_switch_resolution() {
        // Unset: ON. The default flipped on the
        // interleaved card evidence in
        // bench/flash-host-cost-2026-11-06.md: cache-
        // controlled arms show the staged form winning
        // the prefill wall at 1,022 tokens (about 6,150
        // ms on against 6,490 ms off) and at every
        // longer shape measured, with the wall delta
        // matching the kernel delta and identical
        // session stats, discharging the decision rule
        // in bench/flash-rework-2026-11-06.md. The flip
        // is recorded in
        // bench/flash-default-flip-2026-11-07.md.
        assert!(resolve_flash_staged(None));
        // The exact value "1" (trimmed) still means on:
        // a value that enabled the staged form before
        // the flip enables it after.
        assert!(resolve_flash_staged(Some("1")));
        assert!(resolve_flash_staged(Some(" 1 ")));
        // The exact value "0" (trimmed) is the explicit
        // off, the one-variable rollback to the shipped
        // flash form.
        assert!(!resolve_flash_staged(Some("0")));
        assert!(!resolve_flash_staged(Some(" 0 ")));
        // Any other value fails closed to off: a typo
        // never silently selects a form, and the refusal
        // is logged by name at construction.
        assert!(!resolve_flash_staged(Some("true")));
        assert!(!resolve_flash_staged(Some("")));
        assert!(!resolve_flash_staged(Some("11")));
        assert!(!resolve_flash_staged(Some("on")));
        assert_eq!(FLASH_STAGED_ENV, "WGPU_COMPUTE_FLASH_STAGED");
    }

    #[test]
    fn flash_staged_value_recognition() {
        // The recognized set is exactly the positions
        // the resolver gives meaning to: unset, "1",
        // "0" (either after trimming). Everything else
        // is the unrecognized set the construction log
        // names when it fails closed.
        assert!(flash_staged_value_recognized(None));
        assert!(flash_staged_value_recognized(Some("1")));
        assert!(flash_staged_value_recognized(Some(" 1 ")));
        assert!(flash_staged_value_recognized(Some("0")));
        assert!(flash_staged_value_recognized(Some(" 0 ")));
        assert!(!flash_staged_value_recognized(Some("true")));
        assert!(!flash_staged_value_recognized(Some("")));
        assert!(!flash_staged_value_recognized(Some("11")));
        assert!(!flash_staged_value_recognized(Some("on")));
    }

    #[test]
    fn k23_mmq_switch_resolution() {
        // Unset: off. The shipped ladder is the ladder
        // with K23 nowhere in it.
        assert!(!resolve_k23_mmq(None));
        // The exact value "1" (trimmed) is the only
        // enabling position: the switch is fail-closed by
        // design, so a typo routes the proven kernel, not
        // the prototype.
        assert!(resolve_k23_mmq(Some("1")));
        assert!(resolve_k23_mmq(Some(" 1 ")));
        assert!(!resolve_k23_mmq(Some("0")));
        assert!(!resolve_k23_mmq(Some("true")));
        assert!(!resolve_k23_mmq(Some("")));
        assert!(!resolve_k23_mmq(Some("11")));
    }

    #[test]
    fn k21f_head_switch_resolution() {
        // Unset: ON (the shipped default). The route
        // earned the default back under the guard's
        // written reopen criterion: the wave18 card
        // retest (bench/k21f-retest-2026-11-07.md) ran
        // the mask-fixed kernel on the real file with
        // the switch on and recorded IDENTITY and
        // GOLDEN PASS at depths 1 and 6, with the
        // finish output bit-identical to K3's at the
        // real head geometry. Before that proof the
        // default was off (the guard, bench/k21f-
        // default-guard-2026-11-07.md, after the
        // defaults-confirmation identity failure).
        // (This assertion also pins DEFAULT_K21F_HEAD
        // itself: the unset arm returns the constant.)
        assert!(resolve_k21f_head(None));
        // The exact value "1" (trimmed) is on: the
        // value that meant on under the guard still
        // means on after the flip.
        assert!(resolve_k21f_head(Some("1")));
        assert!(resolve_k21f_head(Some(" 1 ")));
        // The exact value "0" (trimmed) is the
        // explicit off: the one-variable rollback to
        // the K3 fused head.
        assert!(!resolve_k21f_head(Some("0")));
        assert!(!resolve_k21f_head(Some(" 0 ")));
        // Anything else fails closed to the fallback,
        // never to the shipped default: a typo must
        // not silently widen routing (the MMV
        // ceiling's fallback discipline).
        assert!(!resolve_k21f_head(Some("true")));
        assert!(!resolve_k21f_head(Some("")));
        assert!(!resolve_k21f_head(Some("11")));
        // (These arms pin K21F_HEAD_FALLBACK the same
        // way: the unrecognized arm returns it.)
        assert!(!resolve_k21f_head(Some("on")));
        assert_eq!(K21F_HEAD_ENV, "WGPU_COMPUTE_K21F_HEAD");
    }

    #[test]
    fn k21f_head_value_recognition() {
        // The recognized positions are exactly the
        // ones the resolver gives a non-fallback
        // meaning: unset, "1", "0" (each trimmed).
        // The construction log names the rest.
        assert!(k21f_head_value_recognized(None));
        assert!(k21f_head_value_recognized(Some("1")));
        assert!(k21f_head_value_recognized(Some(" 0 ")));
        assert!(!k21f_head_value_recognized(Some("true")));
        assert!(!k21f_head_value_recognized(Some("")));
        assert!(!k21f_head_value_recognized(Some("11")));
        assert!(!k21f_head_value_recognized(Some("on")));
    }

    #[test]
    fn coopmat_switch_resolution() {
        // Unset: off. The shipped ladder is the ladder
        // with K23 nowhere in it.
        assert!(!resolve_coopmat(None));
        // The exact value "1" (trimmed) is the only
        // enabling position: fail-closed, like every
        // routing switch in this backend.
        assert!(resolve_coopmat(Some("1")));
        assert!(resolve_coopmat(Some(" 1 ")));
        assert!(!resolve_coopmat(Some("0")));
        assert!(!resolve_coopmat(Some("true")));
        assert!(!resolve_coopmat(Some("")));
        assert!(!resolve_coopmat(Some("11")));
    }

    #[test]
    fn mmq_splitk_resolution() {
        // Unset: off. The default keeps the shipped
        // unsplit products (unproven dispatch shape).
        assert_eq!(resolve_mmq_splitk(None), MmqSplitK::Off);
        assert_eq!(resolve_mmq_splitk(Some("off")), MmqSplitK::Off);
        assert_eq!(resolve_mmq_splitk(Some(" off ")), MmqSplitK::Off);
        // The auto rule.
        assert_eq!(resolve_mmq_splitk(Some("auto")), MmqSplitK::Auto);
        assert_eq!(resolve_mmq_splitk(Some(" auto ")), MmqSplitK::Auto);
        // Fixed factors across the admitted range.
        assert_eq!(resolve_mmq_splitk(Some("1")), MmqSplitK::Fixed(1));
        assert_eq!(resolve_mmq_splitk(Some("4")), MmqSplitK::Fixed(4));
        assert_eq!(resolve_mmq_splitk(Some(" 8 ")), MmqSplitK::Fixed(8));
        // Everything else fails closed to off: zero, past
        // the ceiling, booleans, words, empty.
        assert_eq!(resolve_mmq_splitk(Some("0")), MmqSplitK::Off);
        assert_eq!(resolve_mmq_splitk(Some("9")), MmqSplitK::Off);
        assert_eq!(resolve_mmq_splitk(Some("true")), MmqSplitK::Off);
        assert_eq!(resolve_mmq_splitk(Some("on")), MmqSplitK::Off);
        assert_eq!(resolve_mmq_splitk(Some("")), MmqSplitK::Off);
        assert_eq!(resolve_mmq_splitk(Some("-2")), MmqSplitK::Off);
        assert_eq!(MMQ_SPLITK_ENV, "WGPU_COMPUTE_MMQ_SPLITK");
    }

    #[test]
    fn mmq_splitk_value_recognition() {
        // The recognized positions are exactly the ones
        // the resolver gives a non-fallback meaning.
        assert!(mmq_splitk_value_recognized(None));
        assert!(mmq_splitk_value_recognized(Some("off")));
        assert!(mmq_splitk_value_recognized(Some("auto")));
        assert!(mmq_splitk_value_recognized(Some("1")));
        assert!(mmq_splitk_value_recognized(Some(" 8 ")));
        // The unrecognized rest resolve off and are
        // named in the construction log.
        assert!(!mmq_splitk_value_recognized(Some("0")));
        assert!(!mmq_splitk_value_recognized(Some("9")));
        assert!(!mmq_splitk_value_recognized(Some("true")));
        assert!(!mmq_splitk_value_recognized(Some("")));
    }

    #[test]
    fn mmq_splitk_factor_rule() {
        // Off never splits.
        assert_eq!(mmq_splitk_factor(MmqSplitK::Off, 4, 26), 1);
        assert_eq!(mmq_splitk_factor(MmqSplitK::Off, 512, 26), 1);
        // Fixed: the factor itself, capped at the block
        // count (past it the extra slices are empty) and
        // floored at 1. Fixed(1) is an explicit no-split.
        assert_eq!(mmq_splitk_factor(MmqSplitK::Fixed(4), 4, 26), 4);
        assert_eq!(mmq_splitk_factor(MmqSplitK::Fixed(8), 4, 3), 3);
        assert_eq!(mmq_splitk_factor(MmqSplitK::Fixed(1), 4, 26), 1);
        // Auto: target roughly 64 workgroups. The KV
        // shape (4 tiles, 26 blocks) takes the full 8;
        // a 32-tile projection doubles; anything at or
        // past 64 tiles stays unsplit.
        assert_eq!(mmq_splitk_factor(MmqSplitK::Auto, 4, 26), 8);
        assert_eq!(mmq_splitk_factor(MmqSplitK::Auto, 32, 26), 2);
        assert_eq!(mmq_splitk_factor(MmqSplitK::Auto, 64, 26), 1);
        assert_eq!(mmq_splitk_factor(MmqSplitK::Auto, 312, 26), 1);
        // Auto never exceeds the block count either: a
        // 4-tile projection over a single super-block
        // cannot split.
        assert_eq!(mmq_splitk_factor(MmqSplitK::Auto, 4, 1), 1);
        assert_eq!(mmq_splitk_factor(MmqSplitK::Auto, 8, 2), 2);
    }

    #[test]
    fn chunked_gemv_max_rows_resolution() {
        // Unset: the default, which is off (1 routes
        // nothing, since the ladder's band starts at 2).
        assert_eq!(
            resolve_chunked_gemv_max_rows(None),
            DEFAULT_CHUNKED_GEMV_MAX_ROWS
        );
        assert_eq!(DEFAULT_CHUNKED_GEMV_MAX_ROWS, 1);
        // Plain values pass through, including the
        // head-to-head proof position: 15 (the routed
        // band under GEMM_MIN_ROWS).
        assert_eq!(resolve_chunked_gemv_max_rows(Some("15")), 15);
        assert_eq!(resolve_chunked_gemv_max_rows(Some(" 7 ")), 7);
        assert_eq!(resolve_chunked_gemv_max_rows(Some(" 0 ")), 0);
        // Unparsable values fall back to the default
        // rather than failing backend construction.
        assert_eq!(
            resolve_chunked_gemv_max_rows(Some("chunked")),
            DEFAULT_CHUNKED_GEMV_MAX_ROWS
        );
        assert_eq!(
            resolve_chunked_gemv_max_rows(Some("")),
            DEFAULT_CHUNKED_GEMV_MAX_ROWS
        );
        assert_eq!(
            resolve_chunked_gemv_max_rows(Some("-2")),
            DEFAULT_CHUNKED_GEMV_MAX_ROWS
        );
    }

    #[test]
    fn k21_finish_pipeline_is_registered() {
        // Registration proof: a backend that
        // constructs carries the finish-carrying K21
        // pipeline, under the same shipped shape as
        // plain K21 (the finish changes the store, not
        // the tile). Skips where no adapter qualifies
        // (this VM); executes on the card, where the
        // construction-time compile of the band-floor
        // finish variant is itself part of the gate.
        let backend = match WgpuCompute::new() {
            Ok(b) => b,
            Err(crate::error::WgpuError::NoSuitableAdapter { seen, .. }) => {
                eprintln!(
                    "skipping k21 finish registration test: no suitable adapter \
                     (enumerated: [{}])",
                    seen.join("; ")
                );
                return;
            }
            Err(e) => panic!("backend construction failed unexpectedly: {e}"),
        };
        assert_eq!(backend.k21_finish().shape(), backend.k21().shape());
    }
}
