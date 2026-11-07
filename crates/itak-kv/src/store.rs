//! The persistent prefix store: one file per entry under a root
//! directory, entries keyed by token prefix, the whole store
//! bounded by a byte cap with LRU eviction.
//!
//! Entry file layout (all integers little-endian):
//!
//! ```text
//! preamble, fixed 72 bytes:
//!   magic "ITKVPENT"        [8]u8
//!   format_version          u32   (= 1)
//!   reserved                u32   (zero)
//!   created_unix            u64
//!   last_used_unix          u64   (offset 24; rewritten in place
//!                                  on a verified hit)
//!   prefix_hash             u64
//!   token_count             u64
//!   tokens_bytes            u64   (= 4 * token_count)
//!   payload_bytes           u64   (the snapshot's byte length)
//!   reserved                u64   (zero)
//! token ids                 token_count * u32
//! snapshot bytes            payload_bytes (see snapshot.rs)
//! ```
//!
//! Trust model: the filename and the preamble are an index, never
//! a proof. A lookup candidate is only a hit after the stored
//! token ids hash back to the entry's key AND equal the query's
//! prefix token for token AND the snapshot payload decodes (its
//! own checksum and self-consistency checks) AND its token count
//! matches the preamble AND, when the caller requires them, its
//! geometry and model fingerprint match. Any failure refuses that
//! entry by name in the diag log and moves to the next-longest
//! candidate; a forged or corrupt entry is never applied and
//! never stops a shorter honest prefix from serving.
//!
//! Writes are atomic: the entry is written to a `.tmp-` file in
//! the same directory, synced, and renamed over its final name.
//! A torn temp file left by a crashed writer is ignored by the
//! scanner (it never carries the `.kv` suffix) and a torn final
//! file fails its decode checks and is refused by name.
//!
//! Last-used is the preamble stamp, updated by an 8-byte in-place
//! write on a verified hit (never a content rewrite: entries can
//! be gigabytes). Eviction runs after every `put`: while the
//! store is over its cap, the least recently used entry other
//! than the one just written is deleted. An entry that alone
//! exceeds the cap is refused at `put` time.
//!
//! The index is built by scanning the directory at `open` and is
//! maintained by this instance's own `put`/`lookup` calls; files
//! another process writes after `open` are not seen until the
//! store is reopened. That is the session model: a serving
//! process opens the store once at startup.

use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::config::StoreConfig;
use crate::error::KvError;
use crate::hash;
use crate::snapshot::{KvGeometry, KvSnapshot};

/// Entry file magic.
const ENTRY_MAGIC: &[u8; 8] = b"ITKVPENT";
/// The entry format version this build writes and reads.
pub const ENTRY_FORMAT_VERSION: u32 = 1;
/// Fixed preamble length in bytes.
const PREAMBLE_LEN: usize = 72;
/// Byte offset of the last-used stamp inside the preamble.
const LAST_USED_OFFSET: u64 = 24;
/// Entry filename suffix.
const ENTRY_SUFFIX: &str = ".kv";
/// Temp filename prefix; temp files are never indexed.
const TEMP_PREFIX: &str = ".tmp-";

/// One indexed entry: everything the store knows without opening
/// the payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryMeta {
    /// Filename inside the store root.
    pub file_name: String,
    /// The prefix key (hash of the stored token ids).
    pub prefix_hash: u64,
    /// Token count of the stored prefix.
    pub token_count: usize,
    /// Total file size in bytes (the eviction accounting unit).
    pub bytes: u64,
    /// Creation stamp (unix seconds, caller-supplied clock).
    pub created_unix: u64,
    /// Last verified-hit stamp (unix seconds).
    pub last_used_unix: u64,
}

/// A verified lookup result: the matched prefix and its snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefixHit {
    /// How many leading tokens of the query the entry covers.
    pub token_count: usize,
    /// The stored token ids (equal to the query's first
    /// `token_count` ids; returned so callers can log exactly what
    /// was trusted).
    pub tokens: Vec<u32>,
    /// The decoded, verified KV snapshot for those positions.
    pub snapshot: KvSnapshot,
}

/// Point-in-time store state, for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoreStats {
    /// Indexed entries.
    pub entries: usize,
    /// Sum of indexed entry file sizes.
    pub total_bytes: u64,
    /// The byte cap in force.
    pub max_bytes: u64,
}

/// The current unix time in seconds (the production clock for
/// entry stamps; tests pass their own).
#[must_use]
pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The persistent prefix store.
#[derive(Debug)]
pub struct PrefixStore {
    root: PathBuf,
    max_bytes: u64,
    entries: Vec<EntryMeta>,
}

impl PrefixStore {
    /// Open (creating if needed) the store described by `config`,
    /// scanning the root and indexing every well-formed entry.
    /// Malformed final files are refused by name in the diag log
    /// and left on disk, unindexed; temp files are ignored.
    ///
    /// # Errors
    ///
    /// [`KvError::InvalidConfig`] on a zero cap, or
    /// [`KvError::Io`] when the root cannot be created or read.
    pub fn open(config: &StoreConfig) -> Result<Self, KvError> {
        if config.max_bytes == 0 {
            return Err(KvError::InvalidConfig {
                detail: "the store byte cap is zero".to_string(),
            });
        }
        fs::create_dir_all(&config.root)
            .map_err(|e| KvError::io(format!("create store root {}", config.root.display()), e))?;
        let mut store = PrefixStore {
            root: config.root.clone(),
            max_bytes: config.max_bytes,
            entries: Vec::new(),
        };
        store.rescan()?;
        crate::diag!(
            "prefix store opened root={} entries={} bytes={} cap={}",
            store.root.display(),
            store.entries.len(),
            store.total_bytes(),
            store.max_bytes
        );
        Ok(store)
    }

    /// The store root directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Indexed entries, longest prefix first.
    #[must_use]
    pub fn entries(&self) -> Vec<EntryMeta> {
        let mut entries = self.entries.clone();
        entries.sort_by_key(|e| std::cmp::Reverse(e.token_count));
        entries
    }

    /// Point-in-time store state.
    #[must_use]
    pub fn stats(&self) -> StoreStats {
        StoreStats {
            entries: self.entries.len(),
            total_bytes: self.total_bytes(),
            max_bytes: self.max_bytes,
        }
    }

    fn total_bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.bytes).sum()
    }

    /// Rebuild the index from the directory. Public so a long-lived
    /// store can pick up entries another writer added.
    ///
    /// # Errors
    ///
    /// [`KvError::Io`] when the root cannot be listed.
    pub fn rescan(&mut self) -> Result<(), KvError> {
        let mut entries = Vec::new();
        let listing = fs::read_dir(&self.root)
            .map_err(|e| KvError::io(format!("list store root {}", self.root.display()), e))?;
        for item in listing {
            let item = item.map_err(|e| KvError::io("read store root entry".to_string(), e))?;
            let file_name = item.file_name().to_string_lossy().into_owned();
            if file_name.starts_with(TEMP_PREFIX) || !file_name.ends_with(ENTRY_SUFFIX) {
                continue;
            }
            match read_preamble(&item.path()) {
                Ok(meta) => entries.push(meta),
                Err(error) => {
                    crate::diag!("prefix store refuses entry {file_name}: {error}");
                }
            }
        }
        self.entries = entries;
        Ok(())
    }

    /// Store `snapshot` under the prefix `tokens`, stamped `now`.
    /// Replaces any existing entry carrying the same token list.
    /// Afterwards, evicts least-recently-used entries until the
    /// store is within its cap.
    ///
    /// # Errors
    ///
    /// [`KvError::EmptyPrefix`], [`KvError::TokenCountMismatch`],
    /// [`KvError::EntryTooLarge`] when the entry alone exceeds the
    /// cap, a snapshot encode refusal, or [`KvError::Io`].
    pub fn put(&mut self, tokens: &[u32], snapshot: &KvSnapshot, now: u64) -> Result<(), KvError> {
        if tokens.is_empty() {
            return Err(KvError::EmptyPrefix);
        }
        if snapshot.token_count != tokens.len() {
            return Err(KvError::TokenCountMismatch {
                snapshot: snapshot.token_count,
                tokens: tokens.len(),
            });
        }
        let payload = snapshot.to_bytes()?;
        let key = hash::prefix_hash(tokens);
        let entry_bytes = (PREAMBLE_LEN + tokens.len() * 4 + payload.len()) as u64;
        if entry_bytes > self.max_bytes {
            return Err(KvError::EntryTooLarge {
                bytes: entry_bytes,
                cap: self.max_bytes,
            });
        }
        let file_name = self.pick_file_name(tokens, key)?;
        let path = self.root.join(&file_name);
        // Atomic write: temp file in the same directory, sync,
        // rename over the final name.
        let temp_name = format!("{TEMP_PREFIX}{file_name}-{}", std::process::id());
        let temp_path = self.root.join(&temp_name);
        let preamble = encode_preamble(&Preamble {
            created_unix: now,
            last_used_unix: now,
            prefix_hash: key,
            token_count: tokens.len(),
            payload_bytes: payload.len() as u64,
        });
        {
            let mut file = fs::File::create(&temp_path)
                .map_err(|e| KvError::io(format!("create temp entry {temp_name}"), e))?;
            file.write_all(&preamble)
                .and_then(|()| {
                    for token in tokens {
                        file.write_all(&token.to_le_bytes())?;
                    }
                    file.write_all(&payload)
                })
                .and_then(|()| file.sync_all())
                .map_err(|e| KvError::io(format!("write temp entry {temp_name}"), e))?;
        }
        fs::rename(&temp_path, &path)
            .map_err(|e| KvError::io(format!("rename entry {file_name}"), e))?;
        crate::diag!(
            "prefix store put tokens={} bytes={entry_bytes} file={file_name}",
            tokens.len()
        );
        // Replace any prior index record for this file, then index
        // the new entry.
        self.entries.retain(|e| e.file_name != file_name);
        self.entries.push(EntryMeta {
            file_name,
            prefix_hash: key,
            token_count: tokens.len(),
            bytes: entry_bytes,
            created_unix: now,
            last_used_unix: now,
        });
        self.evict_except(&self.entries.last().expect("just pushed").file_name.clone())
    }

    /// Find the longest stored prefix of `query` and return its
    /// verified snapshot, or `None` when no stored prefix matches.
    /// When `expected_geometry` / `expected_fingerprint` are given,
    /// a candidate whose snapshot disagrees is refused by name and
    /// the search continues to shorter prefixes. A verified hit
    /// restamps the entry's last-used time to `now` (an 8-byte
    /// in-place write).
    ///
    /// # Errors
    ///
    /// [`KvError::Io`] when a candidate file cannot be read at
    /// all; verification failures of individual candidates are
    /// named refusals in the diag log, not errors.
    pub fn lookup(
        &mut self,
        query: &[u32],
        expected_geometry: Option<&KvGeometry>,
        expected_fingerprint: Option<&str>,
        now: u64,
    ) -> Result<Option<PrefixHit>, KvError> {
        let mut candidates: Vec<EntryMeta> = self
            .entries
            .iter()
            .filter(|e| e.token_count <= query.len())
            .cloned()
            .collect();
        candidates.sort_by_key(|e| std::cmp::Reverse(e.token_count));
        for candidate in candidates {
            match self.verify_candidate(&candidate, query, expected_geometry, expected_fingerprint)
            {
                Ok(Some(hit)) => {
                    self.touch(&candidate.file_name, now)?;
                    if let Some(meta) = self
                        .entries
                        .iter_mut()
                        .find(|e| e.file_name == candidate.file_name)
                    {
                        meta.last_used_unix = now;
                    }
                    crate::diag!(
                        "prefix store hit tokens={} file={}",
                        hit.token_count,
                        candidate.file_name
                    );
                    return Ok(Some(hit));
                }
                Ok(None) => {}
                Err(error) => {
                    crate::diag!(
                        "prefix store refuses entry {}: {error}",
                        candidate.file_name
                    );
                }
            }
        }
        Ok(None)
    }

    /// Read and fully verify one candidate against the query.
    /// `Ok(None)` means the entry is well-formed but is not a
    /// prefix of the query (a hash collision or a different branch
    /// of the prefix tree); `Err` means the entry failed a trust
    /// check and was refused by name.
    fn verify_candidate(
        &self,
        meta: &EntryMeta,
        query: &[u32],
        expected_geometry: Option<&KvGeometry>,
        expected_fingerprint: Option<&str>,
    ) -> Result<Option<PrefixHit>, KvError> {
        let path = self.root.join(&meta.file_name);
        let bytes = fs::read(&path)
            .map_err(|e| KvError::io(format!("read entry {}", meta.file_name), e))?;
        let preamble = parse_preamble(&bytes)?;
        if preamble.prefix_hash != meta.prefix_hash || preamble.token_count != meta.token_count {
            return Err(KvError::InconsistentSnapshot {
                detail: format!(
                    "entry {} changed between scan and read (index hash {:#x} / {} tokens, \
                     file hash {:#x} / {} tokens)",
                    meta.file_name,
                    meta.prefix_hash,
                    meta.token_count,
                    preamble.prefix_hash,
                    preamble.token_count
                ),
            });
        }
        let tokens_end = PREAMBLE_LEN + preamble.token_count * 4;
        if bytes.len() < tokens_end {
            return Err(KvError::Truncated {
                what: "entry",
                expected: tokens_end,
                actual: bytes.len(),
            });
        }
        let mut tokens = Vec::with_capacity(preamble.token_count);
        for chunk in bytes[PREAMBLE_LEN..tokens_end].chunks_exact(4) {
            tokens.push(u32::from_le_bytes(chunk.try_into().expect("chunks of 4")));
        }
        // Trust check 1: the stored ids must hash back to the key.
        let computed = hash::prefix_hash(&tokens);
        if computed != preamble.prefix_hash {
            return Err(KvError::PrefixHashMismatch {
                stored: preamble.prefix_hash,
                computed,
            });
        }
        // Prefix check: not an error, just not this query's branch.
        if tokens.as_slice() != &query[..tokens.len()] {
            return Ok(None);
        }
        // Trust check 2: the payload must decode whole and agree
        // with the preamble.
        let payload = &bytes[tokens_end..];
        if payload.len() as u64 != preamble.payload_bytes {
            return Err(KvError::Truncated {
                what: "entry payload",
                expected: preamble.payload_bytes as usize,
                actual: payload.len(),
            });
        }
        let snapshot = KvSnapshot::from_bytes(payload)?;
        if snapshot.token_count != preamble.token_count {
            return Err(KvError::SnapshotTokenMismatch {
                entry: preamble.token_count,
                snapshot: snapshot.token_count,
            });
        }
        // Trust check 3: caller-required geometry and fingerprint.
        if let Some(geometry) = expected_geometry {
            if snapshot.geometry != *geometry {
                return Err(KvError::GeometryMismatch {
                    detail: format!(
                        "entry {} snapshot geometry differs from the required geometry",
                        meta.file_name
                    ),
                });
            }
        }
        if let Some(fingerprint) = expected_fingerprint {
            if snapshot.model_fingerprint != fingerprint {
                return Err(KvError::FingerprintMismatch {
                    expected: fingerprint.to_string(),
                    actual: snapshot.model_fingerprint.clone(),
                });
            }
        }
        Ok(Some(PrefixHit {
            token_count: preamble.token_count,
            tokens,
            snapshot,
        }))
    }

    /// Choose the filename for a put: the hash/length/discriminator
    /// name, unless another entry with different tokens already
    /// owns it (a double hash collision), in which case a counter
    /// suffix disambiguates.
    fn pick_file_name(&self, tokens: &[u32], key: u64) -> Result<String, KvError> {
        let disc = hash::prefix_discriminator(tokens);
        let base = format!("{key:016x}-{}-{disc:016x}", tokens.len());
        for counter in 0..1024_u32 {
            let name = if counter == 0 {
                format!("{base}{ENTRY_SUFFIX}")
            } else {
                format!("{base}-c{counter}{ENTRY_SUFFIX}")
            };
            match self.entries.iter().find(|e| e.file_name == name) {
                None => {
                    // The name may still exist on disk from a writer
                    // this instance has not indexed; check the ids.
                    let path = self.root.join(&name);
                    if path.exists() {
                        if let Ok(existing) = read_entry_tokens(&path) {
                            if existing == tokens {
                                return Ok(name);
                            }
                            continue;
                        }
                    }
                    return Ok(name);
                }
                Some(meta) => {
                    let path = self.root.join(&name);
                    if let Ok(existing) = read_entry_tokens(&path) {
                        if existing == tokens {
                            return Ok(name);
                        }
                    }
                    let _ = meta;
                    continue;
                }
            }
        }
        Err(KvError::InvalidConfig {
            detail: format!(
                "no free filename for prefix hash {key:#x} after 1024 collisions; \
                 the store directory is not behaving like a prefix store"
            ),
        })
    }

    /// Rewrite one entry's last-used stamp in place.
    fn touch(&self, file_name: &str, now: u64) -> Result<(), KvError> {
        let path = self.root.join(file_name);
        let mut file = fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .map_err(|e| KvError::io(format!("open entry {file_name} to restamp"), e))?;
        file.seek(SeekFrom::Start(LAST_USED_OFFSET))
            .and_then(|_| file.write_all(&now.to_le_bytes()))
            .map_err(|e| KvError::io(format!("restamp entry {file_name}"), e))?;
        Ok(())
    }

    /// Evict least-recently-used entries (never `keep`) until the
    /// store is within its cap.
    fn evict_except(&mut self, keep: &str) -> Result<(), KvError> {
        while self.total_bytes() > self.max_bytes {
            let victim = self
                .entries
                .iter()
                .filter(|e| e.file_name != keep)
                .min_by(|a, b| {
                    a.last_used_unix
                        .cmp(&b.last_used_unix)
                        .then_with(|| a.file_name.cmp(&b.file_name))
                })
                .map(|e| e.file_name.clone());
            let Some(victim) = victim else {
                // Only the kept entry remains and it alone fits by
                // construction (put refuses oversize entries), so
                // this is unreachable unless the cap shrank after
                // the fact; stop rather than evict the new entry.
                break;
            };
            let path = self.root.join(&victim);
            fs::remove_file(&path).map_err(|e| KvError::io(format!("evict entry {victim}"), e))?;
            self.entries.retain(|e| e.file_name != victim);
            crate::diag!("prefix store evicted {victim} (LRU, over cap)");
        }
        Ok(())
    }
}

/// The decoded entry preamble.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Preamble {
    created_unix: u64,
    last_used_unix: u64,
    prefix_hash: u64,
    token_count: usize,
    payload_bytes: u64,
}

fn encode_preamble(preamble: &Preamble) -> [u8; PREAMBLE_LEN] {
    let mut out = [0_u8; PREAMBLE_LEN];
    out[0..8].copy_from_slice(ENTRY_MAGIC);
    out[8..12].copy_from_slice(&ENTRY_FORMAT_VERSION.to_le_bytes());
    out[16..24].copy_from_slice(&preamble.created_unix.to_le_bytes());
    out[24..32].copy_from_slice(&preamble.last_used_unix.to_le_bytes());
    out[32..40].copy_from_slice(&preamble.prefix_hash.to_le_bytes());
    out[40..48].copy_from_slice(&(preamble.token_count as u64).to_le_bytes());
    out[48..56].copy_from_slice(&((preamble.token_count * 4) as u64).to_le_bytes());
    out[56..64].copy_from_slice(&preamble.payload_bytes.to_le_bytes());
    out
}

fn parse_preamble(bytes: &[u8]) -> Result<Preamble, KvError> {
    if bytes.len() < PREAMBLE_LEN {
        return Err(KvError::Truncated {
            what: "entry",
            expected: PREAMBLE_LEN,
            actual: bytes.len(),
        });
    }
    if &bytes[0..8] != ENTRY_MAGIC {
        return Err(KvError::BadMagic { what: "entry" });
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes"));
    if version != ENTRY_FORMAT_VERSION {
        return Err(KvError::UnsupportedVersion {
            what: "entry",
            version,
        });
    }
    let field = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().expect("8 bytes"));
    let token_count = usize::try_from(field(40)).map_err(|_| KvError::InconsistentSnapshot {
        detail: "entry token count does not fit in usize".to_string(),
    })?;
    let tokens_bytes = field(48);
    let expected_tokens_bytes = (token_count as u64).saturating_mul(4);
    if tokens_bytes != expected_tokens_bytes {
        return Err(KvError::InconsistentSnapshot {
            detail: format!(
                "entry tokens_bytes is {tokens_bytes} but token count {token_count} \
                 implies {expected_tokens_bytes}"
            ),
        });
    }
    Ok(Preamble {
        created_unix: field(16),
        last_used_unix: field(24),
        prefix_hash: field(32),
        token_count,
        payload_bytes: field(56),
    })
}

/// Read just the preamble of an entry file into index metadata.
fn read_preamble(path: &Path) -> Result<EntryMeta, KvError> {
    let mut file =
        fs::File::open(path).map_err(|e| KvError::io(format!("open {}", path.display()), e))?;
    let mut head = [0_u8; PREAMBLE_LEN];
    file.read_exact(&mut head)
        .map_err(|e| KvError::io(format!("read preamble of {}", path.display()), e))?;
    let preamble = parse_preamble(&head)?;
    let bytes = file
        .metadata()
        .map_err(|e| KvError::io(format!("stat {}", path.display()), e))?
        .len();
    Ok(EntryMeta {
        file_name: path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        prefix_hash: preamble.prefix_hash,
        token_count: preamble.token_count,
        bytes,
        created_unix: preamble.created_unix,
        last_used_unix: preamble.last_used_unix,
    })
}

/// Read an entry's stored token ids (preamble + token region).
fn read_entry_tokens(path: &Path) -> Result<Vec<u32>, KvError> {
    let bytes = fs::read(path).map_err(|e| KvError::io(format!("read {}", path.display()), e))?;
    let preamble = parse_preamble(&bytes)?;
    let end = PREAMBLE_LEN + preamble.token_count * 4;
    if bytes.len() < end {
        return Err(KvError::Truncated {
            what: "entry",
            expected: end,
            actual: bytes.len(),
        });
    }
    Ok(bytes[PREAMBLE_LEN..end]
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("chunks of 4")))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::StoreConfig;
    use attention::{KvCache, LayerKind};

    fn temp_root(tag: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("itak-kv-store-test-{}-{}", std::process::id(), tag));
        let _ = fs::remove_dir_all(&root);
        root
    }

    fn geometry() -> KvGeometry {
        KvGeometry {
            layers: vec![LayerKind::Windowed { window: 4 }, LayerKind::Global],
            num_kv_heads: 2,
            head_dim: 3,
        }
    }

    fn snapshot_for(tokens: &[u32], fingerprint: &str) -> KvSnapshot {
        let geometry = geometry();
        let mut cache = KvCache::new(&geometry.layers, geometry.num_kv_heads, geometry.head_dim)
            .expect("builds");
        let row = geometry.row_floats();
        for (position, &token) in tokens.iter().enumerate() {
            for layer in 0..geometry.layers.len() {
                let k: Vec<f32> = (0..row)
                    .map(|i| (token as f32) + (position * 10 + layer + i) as f32 / 11.0)
                    .collect();
                let v: Vec<f32> = (0..row)
                    .map(|i| -((token as f32) + (position + layer * 3 + i) as f32 / 5.0))
                    .collect();
                cache.append(layer, &k, &v, position).expect("append");
            }
        }
        KvSnapshot::capture(&cache, fingerprint).expect("capture")
    }

    fn open_store(root: &Path, cap: u64) -> PrefixStore {
        PrefixStore::open(&StoreConfig {
            root: root.to_path_buf(),
            max_bytes: cap,
        })
        .expect("store opens")
    }

    #[test]
    fn open_refuses_a_planted_garbage_entry_by_name() {
        // Critic panel regression (failure seat): a planted
        // 72-byte entry whose token_count is u64::MAX and whose
        // tokens_bytes disagrees made parse_preamble's refusal
        // message evaluate token_count * 4 unchecked, which
        // PANICKED debug builds during open/rescan (release
        // wrapped and printed a wrong number). The message now
        // formats the saturating product the check already
        // computed, so both profiles refuse the entry by name and
        // the store opens with it unindexed.
        let root = temp_root("planted");
        fs::create_dir_all(&root).expect("root");
        let mut bytes = vec![0_u8; PREAMBLE_LEN];
        bytes[0..8].copy_from_slice(ENTRY_MAGIC);
        bytes[8..12].copy_from_slice(&ENTRY_FORMAT_VERSION.to_le_bytes());
        bytes[40..48].copy_from_slice(&u64::MAX.to_le_bytes()); // token count
        bytes[48..56].copy_from_slice(&0_u64.to_le_bytes()); // tokens_bytes: disagrees
        fs::write(root.join("planted.kv"), &bytes).expect("plant");
        let store = open_store(&root, 1 << 30);
        assert!(store.entries().is_empty());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn put_lookup_round_trip() {
        let root = temp_root("roundtrip");
        let mut store = open_store(&root, 1 << 30);
        let tokens = vec![10_u32, 20, 30, 40, 50];
        let snapshot = snapshot_for(&tokens, "model-a");
        store.put(&tokens, &snapshot, 100).expect("put");
        let hit = store
            .lookup(&[10, 20, 30, 40, 50, 60, 70], None, None, 200)
            .expect("lookup")
            .expect("hit");
        assert_eq!(hit.token_count, 5);
        assert_eq!(hit.tokens, tokens);
        assert_eq!(hit.snapshot, snapshot);
        // The hit restamped last-used in the index and on disk.
        assert_eq!(store.entries()[0].last_used_unix, 200);
        let reopened = open_store(&root, 1 << 30);
        assert_eq!(reopened.entries()[0].last_used_unix, 200);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn longest_prefix_wins_among_nested_entries() {
        let root = temp_root("nested");
        let mut store = open_store(&root, 1 << 30);
        let short = vec![1_u32, 2];
        let mid = vec![1_u32, 2, 3, 4];
        let long = vec![1_u32, 2, 3, 4, 5, 6];
        store
            .put(&short, &snapshot_for(&short, "model-a"), 1)
            .expect("put short");
        store
            .put(&mid, &snapshot_for(&mid, "model-a"), 2)
            .expect("put mid");
        store
            .put(&long, &snapshot_for(&long, "model-a"), 3)
            .expect("put long");
        // A query extending the longest entry hits the longest.
        let hit = store
            .lookup(&[1, 2, 3, 4, 5, 6, 7, 8], None, None, 10)
            .expect("lookup")
            .expect("hit");
        assert_eq!(hit.token_count, 6);
        // A query that branches after 4 tokens hits the mid entry.
        let hit = store
            .lookup(&[1, 2, 3, 4, 99], None, None, 11)
            .expect("lookup")
            .expect("hit");
        assert_eq!(hit.token_count, 4);
        // A query that branches after 2 tokens hits the short one.
        let hit = store
            .lookup(&[1, 2, 77], None, None, 12)
            .expect("lookup")
            .expect("hit");
        assert_eq!(hit.token_count, 2);
        // No shared prefix at all misses cleanly.
        assert!(store
            .lookup(&[9, 9], None, None, 13)
            .expect("lookup")
            .is_none());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn put_refuses_mismatched_and_oversize() {
        let root = temp_root("refusals");
        let mut store = open_store(&root, 1 << 30);
        let tokens = vec![1_u32, 2, 3];
        let other = snapshot_for(&[7, 8], "model-a");
        let error = store.put(&tokens, &other, 1).expect_err("must refuse");
        assert!(
            matches!(error, KvError::TokenCountMismatch { .. }),
            "{error}"
        );
        let error = store.put(&[], &other, 1).expect_err("must refuse");
        assert!(matches!(error, KvError::EmptyPrefix), "{error}");
        // A cap smaller than one entry refuses the put by name.
        let root2 = temp_root("oversize");
        let mut small = open_store(&root2, 128);
        let snap = snapshot_for(&tokens, "model-a");
        let error = small.put(&tokens, &snap, 1).expect_err("must refuse");
        assert!(matches!(error, KvError::EntryTooLarge { .. }), "{error}");
        let _ = fs::remove_dir_all(&root);
        let _ = fs::remove_dir_all(&root2);
    }
}
