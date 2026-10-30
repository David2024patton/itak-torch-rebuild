//! [`DflashCache`]: the draft's own KV storage.
//!
//! The draft's cache is filled by the injection pass
//! ([`crate::DflashModel::inject`]) with K/V projected from the
//! fused target conditioning at TARGET positions, and read by the
//! drafting pass. It is not the target's cache, it is not
//! checkpointed by the target-side machinery, and the drafting pass
//! never writes to it: block K/V built while proposing are
//! speculative state, discarded when the block ends (SPEC-itak-spec
//! Section 4 step 7: after a round, the draft's state equals having
//! processed exactly the committed sequence, which arrives through
//! `inject` and nothing else).
//!
//! Storage is one position map per layer over a shared position
//! set (an injection writes every layer at the same positions).
//! Sliding-window eviction is a READ-side mask, applied in the
//! attention step exactly as the reference graph's mask states it
//! (a key at `kp` is visible to a query at `p` when
//! `kp > p - window`): keeping the rows and masking at read time is
//! visibility-identical to the reference's ring eviction, and it
//! keeps overwrite-after-rewind semantics trivially correct.
//! Re-injecting a held position overwrites its rows, matching
//! llama.cpp's cache, which sets the cell at a position when the
//! target reprocesses it.

use std::collections::BTreeMap;

use crate::config::DflashConfig;

/// One position's stored rows for one layer.
#[derive(Clone, Debug, PartialEq)]
pub struct KvRow {
    /// Key row, `[kv_heads * key_length]`, RoPE already applied.
    pub k: Vec<f32>,
    /// Value row, `[kv_heads * value_length]`.
    pub v: Vec<f32>,
}

/// The draft model's KV cache: per-layer position maps over one
/// shared set of committed positions.
#[derive(Clone, Debug)]
pub struct DflashCache {
    layers: Vec<BTreeMap<usize, KvRow>>,
}

impl DflashCache {
    /// An empty cache shaped for `cfg`'s layer count.
    pub fn new(cfg: &DflashConfig) -> Self {
        DflashCache {
            layers: (0..cfg.block_count as usize)
                .map(|_| BTreeMap::new())
                .collect(),
        }
    }

    /// Drop every stored position. The next injection starts a
    /// fresh sequence.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    /// Number of committed positions held (the same for every
    /// layer; layer 0's map is the count of record).
    pub fn len(&self) -> usize {
        self.layers.first().map_or(0, BTreeMap::len)
    }

    /// Whether no position has been injected yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether `position` has been injected.
    pub fn contains(&self, position: usize) -> bool {
        self.layers
            .first()
            .is_some_and(|layer| layer.contains_key(&position))
    }

    /// Every held position, ascending.
    pub fn positions(&self) -> Vec<usize> {
        self.layers
            .first()
            .map(|layer| layer.keys().copied().collect())
            .unwrap_or_default()
    }

    /// The stored row for (`layer`, `position`), when present.
    pub fn get(&self, layer: usize, position: usize) -> Option<&KvRow> {
        self.layers.get(layer)?.get(&position)
    }

    /// Iterate one layer's rows in ascending position order.
    pub fn iter_layer(&self, layer: usize) -> impl Iterator<Item = (usize, &KvRow)> {
        self.layers
            .get(layer)
            .into_iter()
            .flat_map(|map| map.iter().map(|(p, row)| (*p, row)))
    }

    /// Store (or overwrite) one position's row on one layer. Called
    /// only by the model's injection pass, which validates widths
    /// and position order before any store happens.
    pub(crate) fn insert(&mut self, layer: usize, position: usize, row: KvRow) {
        if let Some(map) = self.layers.get_mut(layer) {
            map.insert(position, row);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::tiny_config;

    fn row(fill: f32) -> KvRow {
        KvRow {
            k: vec![fill; 4],
            v: vec![fill; 4],
        }
    }

    #[test]
    fn insert_get_and_overwrite() {
        let cfg = tiny_config();
        let mut cache = DflashCache::new(&cfg);
        assert!(cache.is_empty());
        cache.insert(0, 3, row(1.0));
        cache.insert(0, 1, row(2.0));
        cache.insert(1, 3, row(3.0));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.positions(), vec![1, 3]);
        assert!(cache.contains(3));
        assert!(!cache.contains(2));
        assert_eq!(cache.get(0, 3).expect("row").k, vec![1.0; 4]);
        assert_eq!(cache.get(1, 3).expect("row").v, vec![3.0; 4]);
        // Overwrite replaces, never duplicates.
        cache.insert(0, 3, row(9.0));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(0, 3).expect("row").k, vec![9.0; 4]);
        // Ascending iteration regardless of insertion order.
        let seen: Vec<usize> = cache.iter_layer(0).map(|(p, _)| p).collect();
        assert_eq!(seen, vec![1, 3]);
        cache.reset();
        assert!(cache.is_empty());
    }
}
