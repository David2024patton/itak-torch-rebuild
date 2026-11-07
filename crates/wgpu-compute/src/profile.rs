//! Calibration profiles: the per-hardware routing
//! policy the backend loads at construction, measured
//! once on the machine it will run on instead of
//! shipped as one table for every card.
//!
//! The MMV routing policy (the band table and ceiling
//! in [`crate::compute`]) shipped as constants derived
//! from one card's sweep (the 7900 XTX numbers in
//! bench/mmv-band-routing-2026-11-06.md). That is the
//! right table for that card and an unexamined guess
//! for any other. A calibration profile replaces the
//! guess: the `calibrate_profile` example runs the
//! sweep surface ([`crate::ProjBench`]) on the local
//! adapter, derives the winning cells under the same
//! win rule the shipped table used, and writes them
//! here, keyed to the adapter and driver that produced
//! them. Construction then resolves each value in the
//! order env override, profile, shipped default, so a
//! calibrated machine runs its own table, an
//! uncalibrated one runs the shipped table, and an env
//! override still wins over both, per cell.
//!
//! The format is a hand-rolled `key=value` text file,
//! one value per line, in the same parsing style as
//! the band cells themselves: this crate carries no
//! serde dependency, and the profile's whole content
//! is nine band cells in the env's own syntax, one
//! ceiling, and the adapter key, so a line format
//! loses nothing and keeps the file diffable and
//! hand-editable. The format carries a version field
//! ([`PROFILE_FORMAT_VERSION`]); a file from another
//! version is refused by name, never half-read.

use std::path::{Path, PathBuf};

use crate::compute::{MmvFamily, MmvShape};
use crate::detect::AdapterFacts;

/// Environment variable naming the calibration
/// profile file directly. When set (and non-empty) it
/// is the profile path, ahead of the per-user default
/// location; tests point it at a scratch file, and a
/// deployment can pin a profile beside its binary.
pub const PROFILE_PATH_ENV: &str = "WGPU_COMPUTE_PROFILE_PATH";

/// The profile format version this build writes and
/// reads. A profile whose `format_version` differs is
/// rejected at load: the format's meaning is versioned
/// so a future field change never silently reinterprets
/// an old file.
pub const PROFILE_FORMAT_VERSION: u32 = 1;

/// The profile's file name at the default location.
const PROFILE_FILE_NAME: &str = "wgpu-compute-profile.txt";

/// The per-user directory the profile lives in under
/// the config root: this project's own namespace, so
/// the file never collides with another tool's.
const PROFILE_DIR_NAME: &str = "itak";

/// The identity a profile is keyed to: the adapter
/// and driver stack that produced it, taken from the
/// same [`AdapterFacts`] the diag capture manifest
/// records, so a manifest and a profile name the same
/// machine the same way. A profile applies only when
/// every field matches the live adapter exactly: a
/// different card, a different backend, or a driver
/// change (the `driver_info` field carries the
/// version) all make it a different key, and the
/// profile is refused rather than applied to hardware
/// it was not measured on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterKey {
    /// Adapter name as the driver reports it
    /// (`AdapterInfo.name`).
    pub name: String,
    /// wgpu backend, in the manifest's `Debug` form
    /// (`Vulkan` on the GPU path).
    pub backend: String,
    /// Driver name string (`AdapterInfo.driver`).
    pub driver: String,
    /// Driver info string (`AdapterInfo.driver_info`),
    /// the version-bearing half of the driver identity:
    /// a driver upgrade changes this field and so
    /// retires profiles measured under the old driver.
    pub driver_info: String,
}

impl AdapterKey {
    /// The key of a live adapter, from the facts the
    /// backend already gathered at detection.
    #[must_use]
    pub fn from_facts(facts: &AdapterFacts) -> Self {
        AdapterKey {
            name: facts.name.clone(),
            backend: format!("{:?}", facts.backend),
            driver: facts.driver.clone(),
            driver_info: facts.driver_info.clone(),
        }
    }

    /// The key as one line, for construction logs and
    /// the diag manifest's `mmv_profile` pair: a run
    /// can always state which profile governed it, or
    /// that none did.
    #[must_use]
    pub fn describe(&self) -> String {
        format!(
            "{} | {} | {} | {}",
            self.name, self.backend, self.driver, self.driver_info
        )
    }
}

/// One calibration profile: the routing values a
/// calibration run measured for one [`AdapterKey`].
/// Every value is optional individually: a profile
/// states opinions only where it has them, and
/// resolution falls through to the shipped defaults
/// for the rest. Band cells are stored in the env
/// overrides' own text syntax and parsed by the same
/// parser at resolution time, so a profile cell and
/// an env cell can never drift onto different
/// meanings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CalibrationProfile {
    /// The format version the file was written under
    /// ([`PROFILE_FORMAT_VERSION`] for anything this
    /// build writes).
    pub format_version: u32,
    /// The adapter and driver the values were
    /// measured on.
    pub key: AdapterKey,
    /// The MMV ceiling the calibration ran under and
    /// recommends for this adapter, when it states
    /// one.
    pub mmv_max_rows: Option<u32>,
    /// `bands[family.index()][shape.index()]`: the
    /// cell text for that (family, shape) pair, or
    /// `None` when the profile has no opinion there.
    bands: [[Option<String>; 3]; 3],
}

impl CalibrationProfile {
    /// A profile for `key` with no routing opinions
    /// yet: the calibration example starts here and
    /// fills in what it measures.
    #[must_use]
    pub fn new(key: AdapterKey) -> Self {
        CalibrationProfile {
            format_version: PROFILE_FORMAT_VERSION,
            key,
            mmv_max_rows: None,
            bands: Default::default(),
        }
    }

    /// The profile's band cell text for one (family,
    /// shape) pair, when it states one.
    #[must_use]
    pub(crate) fn band_cell(&self, family: MmvFamily, shape: MmvShape) -> Option<&str> {
        self.bands[family.index()][shape.index()].as_deref()
    }

    /// Record the band cell text for one (family,
    /// shape) pair, in the env syntax
    /// ([`format_band_cell`] produces it from measured
    /// ranges).
    pub(crate) fn set_band_cell(&mut self, family: MmvFamily, shape: MmvShape, cell: String) {
        self.bands[family.index()][shape.index()] = Some(cell);
    }

    /// The profile's band cell text for the pair the
    /// short names identify ("q4k", "attn"), when it
    /// states one: the name-keyed view of the same
    /// cells, for callers outside the crate (the
    /// calibration example), where the routing enums
    /// themselves do not reach.
    #[must_use]
    pub fn band_cell_named(&self, family: &str, shape: &str) -> Option<&str> {
        let (family, shape) = band_key_family_shape(&format!("band_{family}_{shape}"))?;
        self.band_cell(family, shape)
    }

    /// Record the band cell text for the pair the
    /// short names identify; `false` (and nothing
    /// recorded) when either name is not a family or
    /// shape the table knows.
    pub fn set_band_cell_named(&mut self, family: &str, shape: &str, cell: String) -> bool {
        let Some((family, shape)) = band_key_family_shape(&format!("band_{family}_{shape}")) else {
            return false;
        };
        self.set_band_cell(family, shape, cell);
        true
    }

    /// Render the profile in its file format: a
    /// comment header, the version, the adapter key,
    /// then the routing values. Field order is fixed
    /// as written here, so two renders of the same
    /// profile are byte-identical.
    #[must_use]
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str("# wgpu-compute calibration profile\n");
        out.push_str(
            "# Measured on the adapter named below by the calibrate_profile\n\
             # example; construction applies it only to that exact adapter\n\
             # and driver. Hand edits are read back through the same\n\
             # parsers the WGPU_COMPUTE_MMV_* env overrides use.\n",
        );
        out.push_str(&format!("format_version={}\n", self.format_version));
        out.push_str(&format!(
            "adapter_name={}\n",
            sanitize_value(&self.key.name)
        ));
        out.push_str(&format!(
            "adapter_backend={}\n",
            sanitize_value(&self.key.backend)
        ));
        out.push_str(&format!(
            "adapter_driver={}\n",
            sanitize_value(&self.key.driver)
        ));
        out.push_str(&format!(
            "adapter_driver_info={}\n",
            sanitize_value(&self.key.driver_info)
        ));
        if let Some(ceiling) = self.mmv_max_rows {
            out.push_str(&format!("mmv_max_rows={ceiling}\n"));
        }
        for family in [MmvFamily::Q4K, MmvFamily::Q5K, MmvFamily::Q6K] {
            for shape in [MmvShape::Attn, MmvShape::Ffn, MmvShape::Kv] {
                if let Some(cell) = self.band_cell(family, shape) {
                    out.push_str(&format!(
                        "band_{}_{}={}\n",
                        family.name(),
                        shape.name(),
                        cell
                    ));
                }
            }
        }
        out
    }

    /// Parse a profile from its file text. Format
    /// level only: field presence, duplicates, and the
    /// ceiling's shape are checked here; the version
    /// and adapter-key checks are the loader's
    /// ([`load_profile`]), and the band cells are
    /// validated by the band parser at resolution time
    /// like any env value. Unknown keys are ignored,
    /// so a newer file with extra fields still loads
    /// under its own version; a line that is neither
    /// blank, a `#` comment, nor `key=value` is
    /// malformed and poisons the whole file.
    ///
    /// # Errors
    ///
    /// A description naming the offending line or the
    /// missing field when the text is not a profile.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut format_version: Option<u32> = None;
        let mut name: Option<String> = None;
        let mut backend: Option<String> = None;
        let mut driver: Option<String> = None;
        let mut driver_info: Option<String> = None;
        let mut mmv_max_rows: Option<u32> = None;
        let mut bands: [[Option<String>; 3]; 3] = Default::default();
        let mut seen: Vec<&str> = Vec::new();
        for (index, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((raw_key, value)) = line.split_once('=') else {
                return Err(format!("line {} is not key=value: {line:?}", index + 1));
            };
            let key = raw_key.trim();
            let value = value.trim();
            let known = key == "format_version"
                || key == "adapter_name"
                || key == "adapter_backend"
                || key == "adapter_driver"
                || key == "adapter_driver_info"
                || key == "mmv_max_rows"
                || band_key_family_shape(key).is_some();
            if !known {
                continue;
            }
            if seen.contains(&key) {
                return Err(format!("duplicate key {key:?}"));
            }
            seen.push(key);
            match key {
                "format_version" => {
                    format_version = Some(value.parse::<u32>().map_err(|_| {
                        format!("format_version {value:?} is not a version number")
                    })?);
                }
                "adapter_name" => name = Some(value.to_string()),
                "adapter_backend" => backend = Some(value.to_string()),
                "adapter_driver" => driver = Some(value.to_string()),
                "adapter_driver_info" => driver_info = Some(value.to_string()),
                "mmv_max_rows" => {
                    mmv_max_rows = Some(
                        value
                            .parse::<u32>()
                            .map_err(|_| format!("mmv_max_rows {value:?} is not a row count"))?,
                    );
                }
                _ => {
                    let (family, shape) =
                        band_key_family_shape(key).expect("known keys checked above");
                    bands[family.index()][shape.index()] = Some(value.to_string());
                }
            }
        }
        let missing = |field: &str| format!("the profile states no {field}");
        Ok(CalibrationProfile {
            format_version: format_version.ok_or_else(|| missing("format_version"))?,
            key: AdapterKey {
                name: name.ok_or_else(|| missing("adapter_name"))?,
                backend: backend.ok_or_else(|| missing("adapter_backend"))?,
                driver: driver.ok_or_else(|| missing("adapter_driver"))?,
                driver_info: driver_info.ok_or_else(|| missing("adapter_driver_info"))?,
            },
            mmv_max_rows,
            bands,
        })
    }
}

/// The (family, shape) a `band_<family>_<shape>` key
/// names, or `None` when the key is not a band key in
/// the families' and shapes' short names.
fn band_key_family_shape(key: &str) -> Option<(MmvFamily, MmvShape)> {
    let rest = key.strip_prefix("band_")?;
    let (family, shape) = rest.split_once('_')?;
    let family = match family {
        "q4k" => MmvFamily::Q4K,
        "q5k" => MmvFamily::Q5K,
        "q6k" => MmvFamily::Q6K,
        _ => return None,
    };
    let shape = match shape {
        "attn" => MmvShape::Attn,
        "ffn" => MmvShape::Ffn,
        "kv" => MmvShape::Kv,
        _ => return None,
    };
    Some((family, shape))
}

/// A value as the line format can carry it: the
/// format is one value per line, so a stray newline
/// in an adapter string becomes a space on write
/// instead of corrupting the file's shape.
fn sanitize_value(raw: &str) -> String {
    raw.chars()
        .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
        .collect()
}

/// The outcome of looking for a profile at a path,
/// in the three shapes construction cares about: a
/// file that loaded and matches this adapter, no file
/// at all (the normal uncalibrated case), or a file
/// that is there but cannot govern this adapter, with
/// the reason stated. Nothing in here panics and
/// nothing applies a profile it could not verify:
/// every rejection falls back to the shipped defaults.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProfileLoad {
    /// The file parsed, its version is this build's,
    /// and its key is the live adapter's key. Boxed: a
    /// profile is the enum's one large payload, and it
    /// is produced once per construction.
    Loaded(Box<CalibrationProfile>),
    /// No file exists at the path. Not an error: most
    /// machines are uncalibrated.
    Absent,
    /// A file exists (or its path does) but cannot be
    /// used: unreadable, malformed, another format
    /// version, or keyed to different hardware. The
    /// string names the path and the reason.
    Rejected(String),
}

/// Load and verify the profile at `path` for the
/// adapter `key` names. The checks are the whole
/// point of keying: a profile measured on other
/// hardware, or under another driver, or written in
/// another format version, must never silently route
/// this adapter.
#[must_use]
pub fn load_profile(path: &Path, key: &AdapterKey) -> ProfileLoad {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return ProfileLoad::Absent,
        Err(e) => {
            return ProfileLoad::Rejected(format!(
                "profile at {} is unreadable: {e}",
                path.display()
            ));
        }
    };
    let profile = match CalibrationProfile::parse(&text) {
        Ok(profile) => profile,
        Err(why) => {
            return ProfileLoad::Rejected(format!(
                "profile at {} is malformed: {why}",
                path.display()
            ));
        }
    };
    if profile.format_version != PROFILE_FORMAT_VERSION {
        return ProfileLoad::Rejected(format!(
            "profile at {} is format version {}, this build reads {}",
            path.display(),
            profile.format_version,
            PROFILE_FORMAT_VERSION
        ));
    }
    if profile.key != *key {
        return ProfileLoad::Rejected(format!(
            "profile at {} is keyed to [{}], this adapter is [{}]",
            path.display(),
            profile.key.describe(),
            key.describe()
        ));
    }
    ProfileLoad::Loaded(Box::new(profile))
}

/// Write a profile to `path`, creating its parent
/// directories. The write is a plain file write: a
/// calibration run is a deliberate act, and a torn
/// write is caught by the loader's parse on the next
/// construction, never applied half-way.
///
/// # Errors
///
/// The `io` error when the directories or the file
/// cannot be created or written.
pub fn write_profile(path: &Path, profile: &CalibrationProfile) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(path, profile.render())
}

/// The path the backend looks for a profile at, from
/// the process environment (see [`resolve_profile_path`]
/// for the rule itself). `None` when no location
/// resolves: no override, no config home, no profile.
#[must_use]
pub fn profile_path() -> Option<PathBuf> {
    resolve_profile_path(|name| std::env::var(name).ok())
}

/// The profile path rule, over an injected env lookup
/// so tests resolve against a table of their own:
///
/// 1. [`PROFILE_PATH_ENV`], when set and non-empty,
///    is the path, exactly as given.
/// 2. Else the per-user config location:
///    `$XDG_CONFIG_HOME/itak/wgpu-compute-profile.txt`
///    when `XDG_CONFIG_HOME` is set and non-empty,
///    else `$HOME/.config/itak/wgpu-compute-profile.txt`.
/// 3. Else (neither variable set) there is no default
///    location and no profile is looked for.
#[must_use]
pub fn resolve_profile_path(get: impl Fn(&str) -> Option<String>) -> Option<PathBuf> {
    if let Some(raw) = get(PROFILE_PATH_ENV) {
        if !raw.trim().is_empty() {
            return Some(PathBuf::from(raw));
        }
    }
    let config_root = match get("XDG_CONFIG_HOME") {
        Some(xdg) if !xdg.trim().is_empty() => PathBuf::from(xdg),
        _ => PathBuf::from(get("HOME")?).join(".config"),
    };
    Some(config_root.join(PROFILE_DIR_NAME).join(PROFILE_FILE_NAME))
}

/// The win ratio the shipped band table was derived
/// under (bench/mmv-band-routing-2026-11-06.md): the
/// MMV form wins a cell row only when its median is
/// under this fraction of the GEMV control's median.
/// A tie (within the band either side) is not a win:
/// ties and losses fall through to the ladder's later
/// rungs, which is what keeps the table a measured win
/// map and nothing else.
pub const MMV_WIN_RATIO: f64 = 0.98;

/// Whether the MMV form wins one swept row under the
/// shipped derivation's rule. A non-positive control
/// median is no measurement at all and wins nothing.
#[must_use]
pub fn mmv_wins(mmv_median_ms: f64, gemv_median_ms: f64) -> bool {
    gemv_median_ms > 0.0 && mmv_median_ms < MMV_WIN_RATIO * gemv_median_ms
}

/// Group a sweep's winning rows into the band ranges
/// a cell states. `sweep_rows` is the swept row list
/// in order and `wins` its per-row verdicts; a run of
/// wins adjacent IN THE SWEEP becomes one numeric
/// range from its first row to its last, the shipped
/// table's own convention (its Q4_K FFN cell "2,4-12"
/// is the row-2 win plus the 4-through-12 run, whose
/// unmeasured interior rows the span covers because
/// measured wins bracket them). Adjacency is in the
/// sweep list, not numeric: rows 9 and 12 sit next to
/// each other in the sweep and join one range.
#[must_use]
pub fn derive_band_ranges(sweep_rows: &[u32], wins: &[bool]) -> Vec<(u32, u32)> {
    let mut ranges = Vec::new();
    let mut run: Option<(u32, u32)> = None;
    for (i, &row) in sweep_rows.iter().enumerate() {
        if wins.get(i).copied().unwrap_or(false) {
            match &mut run {
                Some((_, hi)) => *hi = row,
                None => run = Some((row, row)),
            }
        } else if let Some(done) = run.take() {
            ranges.push(done);
        }
    }
    if let Some(done) = run.take() {
        ranges.push(done);
    }
    ranges
}

/// One band cell in the syntax the resolution parser
/// reads, from derived ranges: "6-12", "2,4-12",
/// "16". An empty range list is the closed cell,
/// spelled "1" (a single row below the kernels' band,
/// which parses to no ranges and routes nothing): a
/// calibrated cell with no wins must close, never
/// fall through to the shipped cell.
#[must_use]
pub fn format_band_cell(ranges: &[(u32, u32)]) -> String {
    if ranges.is_empty() {
        return "1".to_string();
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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key() -> AdapterKey {
        AdapterKey {
            name: "AMD Radeon RX 7900 XTX (RADV NAVI31)".to_string(),
            backend: "Vulkan".to_string(),
            driver: "RADV".to_string(),
            driver_info: "Mesa 26.0.8".to_string(),
        }
    }

    fn full_profile() -> CalibrationProfile {
        let mut profile = CalibrationProfile::new(test_key());
        profile.mmv_max_rows = Some(16);
        profile.set_band_cell(MmvFamily::Q4K, MmvShape::Attn, "6-12".to_string());
        profile.set_band_cell(MmvFamily::Q4K, MmvShape::Ffn, "2,4-12".to_string());
        profile.set_band_cell(MmvFamily::Q4K, MmvShape::Kv, "2-3".to_string());
        profile.set_band_cell(MmvFamily::Q5K, MmvShape::Attn, "2-16".to_string());
        profile.set_band_cell(MmvFamily::Q5K, MmvShape::Ffn, "3-16".to_string());
        profile.set_band_cell(MmvFamily::Q5K, MmvShape::Kv, "16".to_string());
        profile.set_band_cell(MmvFamily::Q6K, MmvShape::Attn, "2-12".to_string());
        profile.set_band_cell(MmvFamily::Q6K, MmvShape::Ffn, "2-12".to_string());
        profile.set_band_cell(MmvFamily::Q6K, MmvShape::Kv, "12".to_string());
        profile
    }

    /// A scratch directory under the system temp dir,
    /// unique to this test run, removed by the caller.
    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "wgpu-compute-profile-test-{tag}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    #[test]
    fn profile_round_trip_through_text() {
        // Render then parse is the identity: the file
        // format loses nothing the profile carries.
        let profile = full_profile();
        let parsed = CalibrationProfile::parse(&profile.render()).expect("rendered parses");
        assert_eq!(parsed, profile);
        // A profile with no opinions round-trips too:
        // absent values stay absent, never defaulted
        // into existence by the format.
        let bare = CalibrationProfile::new(test_key());
        let parsed = CalibrationProfile::parse(&bare.render()).expect("bare renders parses");
        assert_eq!(parsed, bare);
        assert_eq!(parsed.mmv_max_rows, None);
        assert_eq!(parsed.band_cell(MmvFamily::Q4K, MmvShape::Attn), None);
    }

    #[test]
    fn profile_round_trip_through_a_file() {
        // Write to a scratch file, load it back under
        // the same key: Loaded, and identical. The
        // loader the backend runs is the one tested
        // here, not a stand-in.
        let dir = scratch_dir("round-trip");
        let path = dir.join("nested").join("profile.txt");
        let profile = full_profile();
        write_profile(&path, &profile).expect("write");
        match load_profile(&path, &test_key()) {
            ProfileLoad::Loaded(loaded) => assert_eq!(*loaded, profile),
            other => panic!("expected Loaded, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    #[test]
    fn profile_load_missing_file_is_absent_not_rejected() {
        // The uncalibrated machine is the normal case:
        // Absent is its own outcome, distinct from a
        // file that failed verification.
        let dir = scratch_dir("absent");
        let path = dir.join("no-such-profile.txt");
        assert_eq!(load_profile(&path, &test_key()), ProfileLoad::Absent);
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    #[test]
    fn profile_load_adapter_key_mismatch_is_rejected() {
        // A profile measured on other hardware never
        // applies: the rejection names both keys so
        // the construction log says whose table was
        // refused and whose adapter refused it.
        let dir = scratch_dir("key-mismatch");
        let path = dir.join("profile.txt");
        write_profile(&path, &full_profile()).expect("write");
        let mut other = test_key();
        other.driver_info = "Mesa 25.0.0".to_string();
        match load_profile(&path, &other) {
            ProfileLoad::Rejected(why) => {
                assert!(why.contains("Mesa 26.0.8"), "reason: {why}");
                assert!(why.contains("Mesa 25.0.0"), "reason: {why}");
                assert!(why.contains(&path.display().to_string()), "reason: {why}");
            }
            other_outcome => panic!("expected Rejected, got {other_outcome:?}"),
        }
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    #[test]
    fn profile_load_malformed_file_is_rejected_with_its_path() {
        let dir = scratch_dir("malformed");
        let path = dir.join("profile.txt");
        std::fs::write(&path, "this is not a profile\n").expect("write garbage");
        match load_profile(&path, &test_key()) {
            ProfileLoad::Rejected(why) => {
                assert!(why.contains("malformed"), "reason: {why}");
                assert!(why.contains(&path.display().to_string()), "reason: {why}");
            }
            other => panic!("expected Rejected, got {other:?}"),
        }
        // A missing required field is malformed too,
        // not a profile with a hole.
        std::fs::write(&path, "format_version=1\n").expect("write partial");
        match load_profile(&path, &test_key()) {
            ProfileLoad::Rejected(why) => assert!(why.contains("malformed"), "reason: {why}"),
            other => panic!("expected Rejected, got {other:?}"),
        }
        // A directory at the profile path is unreadable
        // as a file: Rejected, never Absent, never a
        // panic.
        match load_profile(&dir, &test_key()) {
            ProfileLoad::Rejected(why) => assert!(why.contains("unreadable"), "reason: {why}"),
            other => panic!("expected Rejected, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    #[test]
    fn profile_load_version_mismatch_is_rejected() {
        // The version field is the format's contract:
        // a file from another version is refused by
        // name even when every other field is right.
        let dir = scratch_dir("version");
        let path = dir.join("profile.txt");
        let mut future = full_profile();
        future.format_version = PROFILE_FORMAT_VERSION + 1;
        write_profile(&path, &future).expect("write");
        match load_profile(&path, &test_key()) {
            ProfileLoad::Rejected(why) => {
                assert!(why.contains("format version"), "reason: {why}");
            }
            other => panic!("expected Rejected, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    #[test]
    fn profile_path_resolution_order() {
        // The explicit override wins, exactly as given.
        let got = resolve_profile_path(|name| match name {
            PROFILE_PATH_ENV => Some("/scratch/pinned.txt".to_string()),
            "XDG_CONFIG_HOME" => Some("/xdg".to_string()),
            "HOME" => Some("/home/tester".to_string()),
            _ => None,
        });
        assert_eq!(got, Some(PathBuf::from("/scratch/pinned.txt")));
        // XDG config home next.
        let got = resolve_profile_path(|name| match name {
            "XDG_CONFIG_HOME" => Some("/xdg".to_string()),
            "HOME" => Some("/home/tester".to_string()),
            _ => None,
        });
        assert_eq!(
            got,
            Some(PathBuf::from("/xdg/itak/wgpu-compute-profile.txt"))
        );
        // HOME composes the .config default.
        let got = resolve_profile_path(|name| match name {
            "HOME" => Some("/home/tester".to_string()),
            _ => None,
        });
        assert_eq!(
            got,
            Some(PathBuf::from(
                "/home/tester/.config/itak/wgpu-compute-profile.txt"
            ))
        );
        // An empty override is no override, and with
        // no config variables at all there is no path.
        let got = resolve_profile_path(|name| match name {
            PROFILE_PATH_ENV => Some("  ".to_string()),
            _ => None,
        });
        assert_eq!(got, None);
    }

    #[test]
    fn mmv_win_rule_matches_the_shipped_derivation() {
        // bench/mmv-band-routing-2026-11-06.md: WIN is
        // a median strictly under 0.98x the control;
        // the boundary itself is a tie, and a tie is
        // not a win.
        assert!(mmv_wins(0.97, 1.0));
        assert!(!mmv_wins(0.98, 1.0));
        assert!(!mmv_wins(1.0, 1.0));
        assert!(!mmv_wins(1.5, 1.0));
        // A zero control median is no measurement.
        assert!(!mmv_wins(0.0, 0.0));
        assert!(!mmv_wins(0.5, 0.0));
    }

    #[test]
    fn band_derivation_groups_sweep_adjacent_wins() {
        // The sweep's row list; wins at 2 and at
        // 4..=12 (adjacent in the sweep from 4 on)
        // derive the shipped table's own Q4_K FFN
        // shape, "2,4-12".
        let rows = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16];
        let wins = [
            false, true, false, true, true, true, true, true, true, true, false,
        ];
        let ranges = derive_band_ranges(&rows, &wins);
        assert_eq!(ranges, vec![(2, 2), (4, 12)]);
        assert_eq!(format_band_cell(&ranges), "2,4-12");
        // No wins derive the closed cell, spelled so
        // it parses to no ranges.
        assert_eq!(derive_band_ranges(&rows, &[false; 11]), Vec::new());
        assert_eq!(format_band_cell(&[]), "1");
        // A lone win is a single-row cell; a full run
        // is one span.
        let wins = [
            false, false, false, false, false, false, false, false, false, false, true,
        ];
        assert_eq!(format_band_cell(&derive_band_ranges(&rows, &wins)), "16");
        assert_eq!(
            format_band_cell(&derive_band_ranges(&rows, &[true; 11])),
            "1-16"
        );
    }
}
