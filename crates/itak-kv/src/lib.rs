//! itak-kv: KV storage contracts for the iTaK Torch rebuild
//! (TODO T-2.4).
//!
//! The `attention` crate owns the in-memory KV cache. This crate
//! owns everything that makes KV state outlive a process:
//!
//! - [`snapshot`]: the versioned [`KvSnapshot`] format. A snapshot
//!   is the KV state of exactly one sequence prefix, captured from
//!   an `attention` [`KvCache`](attention::KvCache), serializable
//!   to bytes with a trailing checksum, and restorable into a
//!   fresh cache with named refusals for anything corrupt,
//!   truncated, or geometrically foreign.
//! - [`store`]: [`PrefixStore`], the persistent tiered prefix-KV
//!   store. Entries are keyed by the hash of a token-id prefix,
//!   carry the full token-id list so a hit is verified token for
//!   token (never trusted from the hash alone), are written
//!   atomically, and are evicted least-recently-used under a byte
//!   cap that is configuration with a default, not a constant.
//!   Lookup returns the longest stored prefix of a query.
//! - [`config`]: the environment resolution for the store (root
//!   directory, byte cap, and the fail-closed master switch the
//!   session seam will consult).
//!
//! The session wiring (a `ResidentSession` restore path that
//! installs a snapshot into device KV and continues from the
//! first unmatched token) is specified in
//! `bench/prefix-kv-store-2026-11-07.md`; this crate provides the
//! store, the format, the switch, and the KV-state-level restore
//! ([`KvSnapshot::restore`]) that the wiring consumes.
//!
//! Everything is f32 reference data, no unsafe, named errors
//! instead of panics on anything read from disk, and debug logging
//! behind a flag (see [`diag`]).

#![forbid(unsafe_code)]

pub mod config;
pub mod diag;
pub mod error;
pub mod hash;
pub mod snapshot;
pub mod store;

pub use config::{
    default_root, max_bytes_from, prefix_store_enabled, prefix_store_enabled_from, StoreConfig,
    DEFAULT_MAX_BYTES, DIR_ENV, ENABLE_ENV, MAX_BYTES_ENV,
};
pub use error::KvError;
pub use hash::{checksum, prefix_discriminator, prefix_hash};
pub use snapshot::{KvGeometry, KvSnapshot, LayerSnapshot, SNAPSHOT_FORMAT_VERSION};
pub use store::{now_unix, EntryMeta, PrefixHit, PrefixStore, StoreStats, ENTRY_FORMAT_VERSION};
