//! Store configuration: where the prefix store lives and how big
//! it may grow, resolved from the environment in the codebase's
//! fail-closed style (named env vars, exact spellings, malformed
//! values logged and fallen back, never silently widened).
//!
//! - `ITAK_KV_PREFIX_STORE`: the master switch for the session
//!   seam. Only the exact value `1` enables; unset or anything else
//!   is off. Default off.
//! - `ITAK_KV_PREFIX_DIR`: the store root directory. Unset means
//!   `$HOME/.cache/itak/prefix-kv` (or a `prefix-kv` directory
//!   under the system temp dir when `HOME` is unset).
//! - `ITAK_KV_PREFIX_MAX_BYTES`: the store's byte cap. Unset or
//!   malformed means [`DEFAULT_MAX_BYTES`]; the cap is a config
//!   value with a default, never a compiled-in limit on callers,
//!   who may always pass their own [`StoreConfig`].

use std::path::PathBuf;

/// Env var naming the store root directory.
pub const DIR_ENV: &str = "ITAK_KV_PREFIX_DIR";
/// Env var naming the store byte cap.
pub const MAX_BYTES_ENV: &str = "ITAK_KV_PREFIX_MAX_BYTES";
/// Env var switching the prefix store seam on.
pub const ENABLE_ENV: &str = "ITAK_KV_PREFIX_STORE";

/// Default byte cap for the store: 8 GiB. At Glimmer geometry a
/// stored prefix costs about 104 KiB per token before the sliding
/// windows saturate (see the crate's bench note), so the default
/// holds roughly thirty 4k-token prefixes or two full-context
/// snapshots.
pub const DEFAULT_MAX_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Whether the prefix store seam is enabled, resolved from a raw
/// env value: only the exact string `"1"` (after trimming) turns
/// it on. This is the codebase's fail-closed switch shape (the
/// K21F head switch resolves the same way): unset, `"0"`, `"true"`,
/// and every other value are off.
#[must_use]
pub fn prefix_store_enabled_from(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), Some("1"))
}

/// Whether the prefix store seam is enabled in this process.
#[must_use]
pub fn prefix_store_enabled() -> bool {
    prefix_store_enabled_from(std::env::var(ENABLE_ENV).ok().as_deref())
}

/// Resolve the byte cap from a raw env value: a parsable positive
/// integer wins; unset means the default; a present but malformed
/// or zero value falls back to the default and is logged by name
/// (a cap is a bound, not a gate, so the fallback is the named
/// default rather than the closed position a switch would take).
#[must_use]
pub fn max_bytes_from(raw: Option<&str>) -> u64 {
    match raw.map(str::trim) {
        None => DEFAULT_MAX_BYTES,
        Some(text) => match text.parse::<u64>() {
            Ok(bytes) if bytes > 0 => bytes,
            _ => {
                crate::diag!(
                    "{MAX_BYTES_ENV} value {text:?} is not a positive byte count; \
                     falling back to the default cap {DEFAULT_MAX_BYTES}"
                );
                DEFAULT_MAX_BYTES
            }
        },
    }
}

/// The default store root when [`DIR_ENV`] is unset.
#[must_use]
pub fn default_root() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        if !home.is_empty() {
            return PathBuf::from(home)
                .join(".cache")
                .join("itak")
                .join("prefix-kv");
        }
    }
    std::env::temp_dir().join("itak-prefix-kv")
}

/// Fully resolved store configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreConfig {
    /// Root directory holding one file per entry.
    pub root: PathBuf,
    /// Byte cap across all entries; LRU eviction enforces it.
    pub max_bytes: u64,
}

impl StoreConfig {
    /// A config with an explicit root and the default cap.
    pub fn new(root: PathBuf) -> Self {
        StoreConfig {
            root,
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }

    /// Resolve the config from the process environment.
    #[must_use]
    pub fn from_env() -> Self {
        let root = std::env::var_os(DIR_ENV)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(default_root);
        let max_bytes = max_bytes_from(std::env::var(MAX_BYTES_ENV).ok().as_deref());
        StoreConfig { root, max_bytes }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enable_switch_is_exact_one_only() {
        assert!(prefix_store_enabled_from(Some("1")));
        assert!(prefix_store_enabled_from(Some(" 1 ")));
        assert!(!prefix_store_enabled_from(None));
        assert!(!prefix_store_enabled_from(Some("0")));
        assert!(!prefix_store_enabled_from(Some("true")));
        assert!(!prefix_store_enabled_from(Some("11")));
        assert!(!prefix_store_enabled_from(Some("")));
    }

    #[test]
    fn cap_resolution() {
        assert_eq!(max_bytes_from(None), DEFAULT_MAX_BYTES);
        assert_eq!(max_bytes_from(Some("1024")), 1024);
        assert_eq!(max_bytes_from(Some(" 2048 ")), 2048);
        assert_eq!(max_bytes_from(Some("0")), DEFAULT_MAX_BYTES);
        assert_eq!(max_bytes_from(Some("-5")), DEFAULT_MAX_BYTES);
        assert_eq!(max_bytes_from(Some("lots")), DEFAULT_MAX_BYTES);
    }
}
