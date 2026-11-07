//! The versioned KV snapshot format: the KV state of exactly one
//! sequence prefix, serializable to bytes and restorable into a
//! fresh [`KvCache`].
//!
//! A snapshot covers positions `0..token_count` of one sequence.
//! Per layer it stores only the rows still visible at that point:
//! every row for a global layer, and the last `window` rows for a
//! windowed layer (older rows were evicted by the ring and no
//! consumer can ever see them again, so storing them would be
//! dead weight). Each layer block names its `first_position`, so
//! the restore side replays appends in true position order.
//!
//! Byte layout (all integers little-endian):
//!
//! ```text
//! magic "ITKV"            [4]u8
//! format_version          u32   (= 1)
//! fingerprint_len         u32, then that many UTF-8 bytes
//! token_count             u64
//! num_kv_heads            u32
//! head_dim                u32
//! layer_count             u32
//! per layer: kind u8 (0 global, 1 windowed), 3 reserved zero
//!            bytes, window u64 (0 for global), first_position
//!            u64, row_count u64
//! per layer: keys   (row_count * kv_heads * head_dim) f32
//!            values (same count) f32
//! checksum                u64   FNV-1a 64 over all preceding bytes
//! ```
//!
//! Decoding is strict and total: bad magic, an unknown version, a
//! short buffer, a checksum mismatch, or per-layer rows that do
//! not add up against the header each refuse by name, and a
//! snapshot is only ever handed out whole. Restoring builds a
//! fresh cache and returns it only when every append succeeded,
//! so a refused snapshot leaves nothing partially applied.

use attention::{KvCache, LayerKind};

use crate::error::KvError;
use crate::hash;

/// Snapshot format magic.
const MAGIC: &[u8; 4] = b"ITKV";
/// The snapshot format version this build writes and reads.
pub const SNAPSHOT_FORMAT_VERSION: u32 = 1;

/// The layer geometry a snapshot is bound to: the layer pattern
/// plus the KV width every layer shares. A snapshot restores only
/// into this exact geometry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvGeometry {
    /// One entry per layer, in layer order.
    pub layers: Vec<LayerKind>,
    /// KV heads per layer.
    pub num_kv_heads: usize,
    /// Dimension of each KV head.
    pub head_dim: usize,
}

impl KvGeometry {
    /// Floats in one position's row for one of K or V:
    /// `[num_kv_heads][head_dim]` laid out flat.
    #[must_use]
    pub fn row_floats(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// Payload bytes one token of prefix adds before any window
    /// saturates: every layer stores one K row and one V row per
    /// token.
    #[must_use]
    pub fn bytes_per_token(&self) -> u64 {
        let per_layer = 2 * self.row_floats() as u64 * 4;
        per_layer * self.layers.len() as u64
    }

    /// Payload bytes a snapshot of a `tokens`-long prefix carries
    /// under this geometry: global layers store every row,
    /// windowed layers store at most `window` rows.
    #[must_use]
    pub fn stored_bytes_for(&self, tokens: usize) -> u64 {
        let per_row = 2 * self.row_floats() as u64 * 4;
        let rows: u64 = self
            .layers
            .iter()
            .map(|kind| match kind {
                LayerKind::Global => tokens as u64,
                LayerKind::Windowed { window } => tokens.min(*window) as u64,
            })
            .sum();
        rows * per_row
    }

    /// The geometry of a live cache.
    #[must_use]
    pub fn of_cache(cache: &KvCache) -> Self {
        let layers = (0..cache.layer_count())
            .map(|layer| {
                cache
                    .layer_kind(layer)
                    .expect("layer index from the cache's own count")
            })
            .collect();
        KvGeometry {
            layers,
            num_kv_heads: cache.num_kv_heads(),
            head_dim: cache.head_dim(),
        }
    }
}

/// One layer's stored rows inside a snapshot: the visible rows at
/// capture time, oldest first, each row `row_floats` long for keys
/// and the same for values.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerSnapshot {
    /// Absolute position of the first stored row.
    pub first_position: usize,
    /// Key rows, oldest first, flattened.
    pub keys: Vec<f32>,
    /// Value rows, oldest first, flattened.
    pub values: Vec<f32>,
}

impl LayerSnapshot {
    /// Stored row count, given the geometry's row width.
    #[must_use]
    pub fn rows(&self, row_floats: usize) -> usize {
        self.keys.len().checked_div(row_floats).unwrap_or(0)
    }
}

/// A whole-sequence-prefix KV snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct KvSnapshot {
    /// The geometry the rows belong to.
    pub geometry: KvGeometry,
    /// Opaque model identity (for example the GGUF file's SHA-256
    /// hex). Recorded at capture; a restore may require an exact
    /// match so KV from one model is never served to another.
    pub model_fingerprint: String,
    /// Positions the snapshot covers: `0..token_count`.
    pub token_count: usize,
    /// One block per layer, in layer order.
    pub layers: Vec<LayerSnapshot>,
}

impl KvSnapshot {
    /// Check the snapshot against its own header: layer count,
    /// per-layer first positions and row counts implied by the
    /// token count and windows, and exact row widths. Called by
    /// both the encoder and the decoder, so a snapshot that exists
    /// in memory has already passed the same checks a file must.
    ///
    /// # Errors
    ///
    /// [`KvError::InconsistentSnapshot`] naming the first
    /// disagreement.
    pub fn validate(&self) -> Result<(), KvError> {
        let geometry = &self.geometry;
        if self.layers.len() != geometry.layers.len() {
            return Err(inconsistent(format!(
                "snapshot carries {} layer blocks but its geometry has {} layers",
                self.layers.len(),
                geometry.layers.len()
            )));
        }
        if geometry.num_kv_heads == 0 || geometry.head_dim == 0 {
            return Err(inconsistent(
                "geometry has a zero KV head count or head dimension".to_string(),
            ));
        }
        let row = geometry.row_floats();
        for (index, (kind, layer)) in geometry.layers.iter().zip(&self.layers).enumerate() {
            let expected_rows = match kind {
                LayerKind::Global => self.token_count,
                LayerKind::Windowed { window } => self.token_count.min(*window),
            };
            let expected_first = self.token_count - expected_rows;
            if layer.first_position != expected_first {
                return Err(inconsistent(format!(
                    "layer {index}: first stored position is {} but {expected_first} follows \
                     from token count {} and kind {kind:?}",
                    layer.first_position, self.token_count
                )));
            }
            if layer.keys.len() != expected_rows * row || layer.values.len() != expected_rows * row
            {
                return Err(inconsistent(format!(
                    "layer {index}: holds {} key and {} value floats but {expected_rows} rows \
                     of {row} floats were expected",
                    layer.keys.len(),
                    layer.values.len()
                )));
            }
        }
        Ok(())
    }

    /// Capture a snapshot of a live cache. Every layer must have
    /// appended the same number of positions (the cache is one
    /// sequence); a cache mid-round with divergent layer positions
    /// is refused by name instead of being frozen half-way.
    ///
    /// # Errors
    ///
    /// [`KvError::IncoherentCache`] on divergent layer positions,
    /// or an attention refusal while reading a layer view.
    pub fn capture(cache: &KvCache, model_fingerprint: impl Into<String>) -> Result<Self, KvError> {
        let geometry = KvGeometry::of_cache(cache);
        let token_count = cache.next_position(0)?;
        let row = geometry.row_floats();
        let mut layers = Vec::with_capacity(geometry.layers.len());
        for index in 0..geometry.layers.len() {
            let appended = cache.next_position(index)?;
            if appended != token_count {
                return Err(KvError::IncoherentCache {
                    detail: format!(
                        "layer 0 has appended {token_count} positions but layer {index} has \
                         appended {appended}; a snapshot is one sequence"
                    ),
                });
            }
            let view = cache.view(index)?;
            let visible = view.len();
            let mut keys = Vec::with_capacity(visible * row);
            let mut values = Vec::with_capacity(visible * row);
            for slot in 0..visible {
                for head in 0..geometry.num_kv_heads {
                    let key = view
                        .key(slot, head)
                        .ok_or_else(|| KvError::IncoherentCache {
                            detail: format!("layer {index} view has no key row at slot {slot}"),
                        })?;
                    let value = view
                        .value(slot, head)
                        .ok_or_else(|| KvError::IncoherentCache {
                            detail: format!("layer {index} view has no value row at slot {slot}"),
                        })?;
                    keys.extend_from_slice(key);
                    values.extend_from_slice(value);
                }
            }
            layers.push(LayerSnapshot {
                first_position: token_count - visible,
                keys,
                values,
            });
        }
        let snapshot = KvSnapshot {
            geometry,
            model_fingerprint: model_fingerprint.into(),
            token_count,
            layers,
        };
        snapshot.validate()?;
        crate::diag!(
            "snapshot captured tokens={} layers={} payload_bytes={}",
            snapshot.token_count,
            snapshot.layers.len(),
            snapshot.geometry.stored_bytes_for(snapshot.token_count)
        );
        Ok(snapshot)
    }

    /// Restore the snapshot into a fresh cache. Windowed layers
    /// replay zero rows for the evicted positions ahead of the
    /// first stored row, so every ring slot and the layer's
    /// appended count end exactly where a cache that ran the
    /// prefix itself would have put them; the zeros land only in
    /// slots the stored rows are about to overwrite (or in slots
    /// whose positions are already outside the window and never
    /// read). The fresh cache is returned only on full success.
    ///
    /// # Errors
    ///
    /// [`KvError::InconsistentSnapshot`] on a self-disagreeing
    /// snapshot, or an attention refusal while building the cache.
    pub fn restore(&self) -> Result<KvCache, KvError> {
        self.validate()?;
        let geometry = &self.geometry;
        let mut cache = KvCache::new(&geometry.layers, geometry.num_kv_heads, geometry.head_dim)?;
        let row = geometry.row_floats();
        let zero_row = vec![0.0_f32; row];
        for (index, layer) in self.layers.iter().enumerate() {
            for position in 0..layer.first_position {
                cache.append(index, &zero_row, &zero_row, position)?;
            }
            let rows = layer.rows(row);
            for r in 0..rows {
                let start = r * row;
                cache.append(
                    index,
                    &layer.keys[start..start + row],
                    &layer.values[start..start + row],
                    layer.first_position + r,
                )?;
            }
        }
        crate::diag!(
            "snapshot restored tokens={} layers={}",
            self.token_count,
            self.layers.len()
        );
        Ok(cache)
    }

    /// Restore after requiring an exact geometry and model
    /// fingerprint match: the store's trust check, so KV captured
    /// for one model or layer geometry is never served to another.
    ///
    /// # Errors
    ///
    /// [`KvError::GeometryMismatch`] or
    /// [`KvError::FingerprintMismatch`] naming the difference,
    /// then whatever [`KvSnapshot::restore`] refuses.
    pub fn restore_checked(
        &self,
        expected_geometry: &KvGeometry,
        expected_fingerprint: &str,
    ) -> Result<KvCache, KvError> {
        if self.geometry != *expected_geometry {
            return Err(KvError::GeometryMismatch {
                detail: format!(
                    "snapshot geometry ({} layers, {} KV heads, head dim {}) differs from the \
                     required geometry ({} layers, {} KV heads, head dim {})",
                    self.geometry.layers.len(),
                    self.geometry.num_kv_heads,
                    self.geometry.head_dim,
                    expected_geometry.layers.len(),
                    expected_geometry.num_kv_heads,
                    expected_geometry.head_dim,
                ),
            });
        }
        if self.model_fingerprint != expected_fingerprint {
            return Err(KvError::FingerprintMismatch {
                expected: expected_fingerprint.to_string(),
                actual: self.model_fingerprint.clone(),
            });
        }
        self.restore()
    }

    /// Serialize to the versioned byte format (see the module
    /// docs). Validates first: bytes that cannot be produced by a
    /// coherent snapshot are never written.
    ///
    /// # Errors
    ///
    /// [`KvError::InconsistentSnapshot`] on a self-disagreeing
    /// snapshot.
    pub fn to_bytes(&self) -> Result<Vec<u8>, KvError> {
        self.validate()?;
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&SNAPSHOT_FORMAT_VERSION.to_le_bytes());
        let fingerprint = self.model_fingerprint.as_bytes();
        out.extend_from_slice(&(fingerprint.len() as u32).to_le_bytes());
        out.extend_from_slice(fingerprint);
        out.extend_from_slice(&(self.token_count as u64).to_le_bytes());
        out.extend_from_slice(&(self.geometry.num_kv_heads as u32).to_le_bytes());
        out.extend_from_slice(&(self.geometry.head_dim as u32).to_le_bytes());
        out.extend_from_slice(&(self.geometry.layers.len() as u32).to_le_bytes());
        for (kind, layer) in self.geometry.layers.iter().zip(&self.layers) {
            let (tag, window) = match kind {
                LayerKind::Global => (0_u8, 0_u64),
                LayerKind::Windowed { window } => (1_u8, *window as u64),
            };
            out.push(tag);
            out.extend_from_slice(&[0, 0, 0]);
            out.extend_from_slice(&window.to_le_bytes());
            out.extend_from_slice(&(layer.first_position as u64).to_le_bytes());
            out.extend_from_slice(&(layer.rows(self.geometry.row_floats()) as u64).to_le_bytes());
        }
        for layer in &self.layers {
            for value in &layer.keys {
                out.extend_from_slice(&value.to_le_bytes());
            }
            for value in &layer.values {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        let sum = hash::checksum(&out);
        out.extend_from_slice(&sum.to_le_bytes());
        Ok(out)
    }

    /// Decode the versioned byte format. Strict: any disagreement
    /// refuses the whole snapshot by name.
    ///
    /// # Errors
    ///
    /// [`KvError::BadMagic`], [`KvError::UnsupportedVersion`],
    /// [`KvError::Truncated`], [`KvError::ChecksumMismatch`], or
    /// [`KvError::InconsistentSnapshot`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, KvError> {
        let mut reader = Reader::new(bytes);
        if reader.take(4)? != MAGIC {
            return Err(KvError::BadMagic { what: "snapshot" });
        }
        let version = reader.u32()?;
        if version != SNAPSHOT_FORMAT_VERSION {
            return Err(KvError::UnsupportedVersion {
                what: "snapshot",
                version,
            });
        }
        let fingerprint_len = reader.u32()? as usize;
        let fingerprint_bytes = reader.take(fingerprint_len)?;
        let model_fingerprint = String::from_utf8(fingerprint_bytes.to_vec()).map_err(|_| {
            KvError::InconsistentSnapshot {
                detail: "model fingerprint is not valid UTF-8".to_string(),
            }
        })?;
        let token_count = reader.u64_usize("token count")?;
        let num_kv_heads = reader.u32()? as usize;
        let head_dim = reader.u32()? as usize;
        let layer_count = reader.u32()? as usize;
        let mut kinds = Vec::with_capacity(layer_count);
        let mut blocks = Vec::with_capacity(layer_count);
        for _ in 0..layer_count {
            let tag = reader.take(1)?[0];
            let reserved = reader.take(3)?;
            if reserved != [0, 0, 0] {
                return Err(inconsistent(
                    "layer header reserved bytes are not zero".to_string(),
                ));
            }
            let window = reader.u64()?;
            let first_position = reader.u64_usize("first position")?;
            let row_count = reader.u64_usize("row count")?;
            let kind = match tag {
                0 => LayerKind::Global,
                1 => LayerKind::windowed(usize::try_from(window).map_err(|_| {
                    KvError::InconsistentSnapshot {
                        detail: "window does not fit in usize".to_string(),
                    }
                })?)
                .map_err(|_| KvError::InconsistentSnapshot {
                    detail: "windowed layer carries a zero window".to_string(),
                })?,
                other => {
                    return Err(inconsistent(format!(
                        "layer kind tag {other} is not a known kind"
                    )))
                }
            };
            kinds.push(kind);
            blocks.push((first_position, row_count));
        }
        let geometry = KvGeometry {
            layers: kinds,
            num_kv_heads,
            head_dim,
        };
        let row = geometry.row_floats();
        let mut layers = Vec::with_capacity(layer_count);
        for (first_position, row_count) in blocks {
            let floats =
                row_count
                    .checked_mul(row)
                    .ok_or_else(|| KvError::InconsistentSnapshot {
                        detail: "layer row count overflows the row width".to_string(),
                    })?;
            let keys = reader.f32s(floats)?;
            let values = reader.f32s(floats)?;
            layers.push(LayerSnapshot {
                first_position,
                keys,
                values,
            });
        }
        let stored_checksum = reader.u64()?;
        if reader.remaining() != 0 {
            return Err(inconsistent(format!(
                "{} trailing bytes after the checksum",
                reader.remaining()
            )));
        }
        let computed = hash::checksum(&bytes[..bytes.len() - 8]);
        if stored_checksum != computed {
            return Err(KvError::ChecksumMismatch {
                stored: stored_checksum,
                computed,
            });
        }
        let snapshot = KvSnapshot {
            geometry,
            model_fingerprint,
            token_count,
            layers,
        };
        snapshot.validate()?;
        Ok(snapshot)
    }
}

fn inconsistent(detail: String) -> KvError {
    KvError::InconsistentSnapshot { detail }
}

/// A strict little-endian cursor over a snapshot byte stream.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Reader { bytes, at: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.at
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], KvError> {
        if self.remaining() < n {
            return Err(KvError::Truncated {
                what: "snapshot",
                expected: self.at + n,
                actual: self.bytes.len(),
            });
        }
        let slice = &self.bytes[self.at..self.at + n];
        self.at += n;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32, KvError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes(
            bytes.try_into().expect("take(4) yields 4 bytes"),
        ))
    }

    fn u64(&mut self) -> Result<u64, KvError> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes(
            bytes.try_into().expect("take(8) yields 8 bytes"),
        ))
    }

    fn u64_usize(&mut self, what: &str) -> Result<usize, KvError> {
        let value = self.u64()?;
        usize::try_from(value).map_err(|_| KvError::InconsistentSnapshot {
            detail: format!("{what} does not fit in usize"),
        })
    }

    fn f32s(&mut self, count: usize) -> Result<Vec<f32>, KvError> {
        let byte_count = count
            .checked_mul(4)
            .ok_or_else(|| KvError::InconsistentSnapshot {
                detail: "float count overflows the byte count".to_string(),
            })?;
        let bytes = self.take(byte_count)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunks of 4")))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> KvGeometry {
        KvGeometry {
            layers: vec![
                LayerKind::Windowed { window: 4 },
                LayerKind::Global,
                LayerKind::Windowed { window: 4 },
            ],
            num_kv_heads: 2,
            head_dim: 3,
        }
    }

    fn filled_cache(tokens: usize) -> KvCache {
        let geometry = geometry();
        let mut cache = KvCache::new(&geometry.layers, geometry.num_kv_heads, geometry.head_dim)
            .expect("fixture geometry builds");
        let row = geometry.row_floats();
        for position in 0..tokens {
            for layer in 0..geometry.layers.len() {
                let k: Vec<f32> = (0..row)
                    .map(|i| (position * 100 + layer * 10 + i) as f32 / 7.0)
                    .collect();
                let v: Vec<f32> = (0..row)
                    .map(|i| -((position * 50 + layer * 5 + i) as f32) / 3.0)
                    .collect();
                cache
                    .append(layer, &k, &v, position)
                    .expect("in-order appends");
            }
        }
        cache
    }

    #[test]
    fn bytes_round_trip() {
        let cache = filled_cache(10);
        let snapshot = KvSnapshot::capture(&cache, "model-a").expect("capture");
        let bytes = snapshot.to_bytes().expect("encode");
        let decoded = KvSnapshot::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded, snapshot);
    }

    #[test]
    fn restore_rebuilds_the_same_views() {
        let cache = filled_cache(10);
        let snapshot = KvSnapshot::capture(&cache, "model-a").expect("capture");
        let restored = snapshot.restore().expect("restore");
        for layer in 0..3 {
            assert_eq!(
                restored.next_position(layer).expect("position"),
                cache.next_position(layer).expect("position")
            );
            let want = cache.view(layer).expect("view");
            let got = restored.view(layer).expect("view");
            assert_eq!(got.len(), want.len());
            for slot in 0..want.len() {
                assert_eq!(got.position_at(slot), want.position_at(slot));
                for head in 0..2 {
                    assert_eq!(got.key(slot, head), want.key(slot, head));
                    assert_eq!(got.value(slot, head), want.value(slot, head));
                }
            }
        }
    }

    #[test]
    fn restore_can_continue_appending() {
        let cache = filled_cache(6);
        let snapshot = KvSnapshot::capture(&cache, "model-a").expect("capture");
        let mut restored = snapshot.restore().expect("restore");
        let row = geometry().row_floats();
        let k = vec![1.5; row];
        let v = vec![-2.5; row];
        for layer in 0..3 {
            restored.append(layer, &k, &v, 6).expect("append at 6");
        }
        assert_eq!(restored.next_position(0).expect("position"), 7);
    }

    #[test]
    fn capture_refuses_a_divergent_cache() {
        // A cache whose layers disagree on position cannot be built
        // through balanced appends; simulate by appending to one
        // layer only.
        let geometry = geometry();
        let mut cache = KvCache::new(&geometry.layers, geometry.num_kv_heads, geometry.head_dim)
            .expect("builds");
        let row = geometry.row_floats();
        cache
            .append(0, &vec![0.0; row], &vec![0.0; row], 0)
            .expect("append");
        let error = KvSnapshot::capture(&cache, "model-a").expect_err("must refuse");
        assert!(matches!(error, KvError::IncoherentCache { .. }), "{error}");
    }

    #[test]
    fn decode_refuses_corruption() {
        let cache = filled_cache(5);
        let snapshot = KvSnapshot::capture(&cache, "model-a").expect("capture");
        let mut bytes = snapshot.to_bytes().expect("encode");
        // Flip a payload byte: the checksum must catch it.
        let at = bytes.len() - 12;
        bytes[at] ^= 0xff;
        let error = KvSnapshot::from_bytes(&bytes).expect_err("must refuse");
        assert!(matches!(error, KvError::ChecksumMismatch { .. }), "{error}");
        // Truncation refuses by name too.
        let error = KvSnapshot::from_bytes(&bytes[..8]).expect_err("must refuse");
        assert!(matches!(error, KvError::Truncated { .. }), "{error}");
        // Bad magic refuses by name.
        let mut bad = snapshot.to_bytes().expect("encode");
        bad[0] = b'X';
        let error = KvSnapshot::from_bytes(&bad).expect_err("must refuse");
        assert!(matches!(error, KvError::BadMagic { .. }), "{error}");
    }

    #[test]
    fn restore_checked_refuses_geometry_and_fingerprint() {
        let cache = filled_cache(4);
        let snapshot = KvSnapshot::capture(&cache, "model-a").expect("capture");
        let mut other = geometry();
        other.head_dim = 9;
        let error = snapshot
            .restore_checked(&other, "model-a")
            .expect_err("must refuse");
        assert!(matches!(error, KvError::GeometryMismatch { .. }), "{error}");
        let error = snapshot
            .restore_checked(&geometry(), "model-b")
            .expect_err("must refuse");
        assert!(
            matches!(error, KvError::FingerprintMismatch { .. }),
            "{error}"
        );
        snapshot
            .restore_checked(&geometry(), "model-a")
            .expect("matching checks restore");
    }

    #[test]
    fn glimmer_sizing_math() {
        // The recorded Glimmer geometry: 52 layers (39 windowed at
        // 2048, 13 global), 2 KV heads, head dim 128.
        let layers = attention::glimmer_layer_pattern();
        let geometry = KvGeometry {
            layers,
            num_kv_heads: 2,
            head_dim: 128,
        };
        // 52 layers * 2 (K and V) * 256 floats * 4 bytes.
        assert_eq!(geometry.bytes_per_token(), 106_496);
        // A 4096-token prefix: global layers hold 4096 rows,
        // windowed layers hold 2048.
        assert_eq!(geometry.stored_bytes_for(4096), 272_629_760);
        // Full context: 13 * 131072 + 39 * 2048 stored rows.
        assert_eq!(geometry.stored_bytes_for(131_072), 3_653_238_784);
    }
}
