//! The seam fixture: a small model shape where a "cold" run
//! (append the prefix, then continue) and a "restored" run
//! (append the prefix, snapshot it through the store, restore
//! into a fresh cache, then continue) must produce bitwise
//! identical attention outputs and identical greedy streams.
//!
//! The surface here is the KV-state level (attention crate math),
//! which is the level this lane wires; the ResidentSession wiring
//! that consumes the same restore is specified in
//! bench/prefix-kv-store-2026-11-07.md.

use std::fs;
use std::path::PathBuf;

use attention::{gqa_attention, AttentionParams, KvCache, LayerKind};
use itak_kv::{KvGeometry, KvSnapshot, PrefixStore, StoreConfig};

const Q_HEADS: usize = 4;
const LOGITS: usize = 16;

/// Small deterministic PRNG stream keyed by a seed word.
fn lcg(seed: u64) -> impl Iterator<Item = f32> {
    let mut state = seed.max(1);
    std::iter::from_fn(move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        Some(((state >> 40) as u32 as f32) / (1_u32 << 24) as f32 - 0.5)
    })
}

fn geometry() -> KvGeometry {
    KvGeometry {
        layers: vec![
            LayerKind::Windowed { window: 4 },
            LayerKind::Global,
            LayerKind::Global,
            LayerKind::Windowed { window: 4 },
        ],
        num_kv_heads: 2,
        head_dim: 8,
    }
}

/// One position's K and V rows for one layer, a pure function of
/// (layer, position, token): the "model" of this fixture.
fn rows(geometry: &KvGeometry, layer: usize, position: usize, token: u32) -> (Vec<f32>, Vec<f32>) {
    let row = geometry.row_floats();
    let seed = (layer as u64) << 40 | (position as u64) << 16 | u64::from(token);
    let mut stream = lcg(seed);
    let k: Vec<f32> = stream.by_ref().take(row).collect();
    let v: Vec<f32> = stream.take(row).collect();
    (k, v)
}

fn append_token(cache: &mut KvCache, geometry: &KvGeometry, position: usize, token: u32) {
    for layer in 0..geometry.layers.len() {
        let (k, v) = rows(geometry, layer, position, token);
        cache.append(layer, &k, &v, position).expect("append");
    }
}

/// Fixed pseudo-head: project the concatenated per-layer
/// attention outputs for the newest position into LOGITS values.
fn logits_for(cache: &KvCache, geometry: &KvGeometry, position: usize, token: u32) -> Vec<f32> {
    let head_dim = geometry.head_dim;
    let q_len = Q_HEADS * head_dim;
    let seed = 0x5eed_u64 ^ ((position as u64) << 20) ^ u64::from(token);
    let q: Vec<f32> = lcg(seed).take(q_len).collect();
    let mut logits = vec![0.0_f32; LOGITS];
    for layer in 0..geometry.layers.len() {
        let view = cache.view(layer).expect("view");
        let out = gqa_attention(&q, 1, Q_HEADS, &[position], &view, &AttentionParams::new())
            .expect("attention");
        for (o, slot) in logits.iter_mut().enumerate() {
            for (i, &value) in out.iter().enumerate() {
                let w = (((o * 31 + i * 7 + layer * 3) % 13) as f32) / 7.0 - 0.8;
                *slot += w * value;
            }
        }
    }
    logits
}

fn argmax(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("no NaNs in fixture"))
        .map(|(i, _)| i as u32)
        .expect("nonempty")
}

/// Run the continuation against a prepared cache and record the
/// per-step logits and greedy picks. The continuation tokens are
/// chosen by the fixture (not by the picks) so both runs walk the
/// same positions; the picks are the stream under test.
fn continue_run(
    cache: &mut KvCache,
    geometry: &KvGeometry,
    start: usize,
    continuation: &[u32],
) -> (Vec<Vec<f32>>, Vec<u32>) {
    let mut all_logits = Vec::new();
    let mut stream = Vec::new();
    for (step, &token) in continuation.iter().enumerate() {
        let position = start + step;
        append_token(cache, geometry, position, token);
        let logits = logits_for(cache, geometry, position, token);
        stream.push(argmax(&logits));
        all_logits.push(logits);
    }
    (all_logits, stream)
}

#[test]
fn restored_cache_continues_bitwise_identically() {
    let geometry = geometry();
    let prefix: Vec<u32> = vec![11, 22, 33, 44, 55, 66, 77, 88, 99];
    let continuation: Vec<u32> = vec![5, 4, 3, 2, 1];

    // Cold run: prefix and continuation in one cache.
    let mut cold =
        KvCache::new(&geometry.layers, geometry.num_kv_heads, geometry.head_dim).expect("builds");
    for (position, &token) in prefix.iter().enumerate() {
        append_token(&mut cold, &geometry, position, token);
    }
    let (cold_logits, cold_stream) =
        continue_run(&mut cold, &geometry, prefix.len(), &continuation);

    // Restored run: prefix only, snapshot through the store,
    // restore, compare state, then continue.
    let root: PathBuf =
        std::env::temp_dir().join(format!("itak-kv-fixture-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let mut store = PrefixStore::open(&StoreConfig {
        root: root.clone(),
        max_bytes: 1 << 30,
    })
    .expect("store opens");
    let mut warm =
        KvCache::new(&geometry.layers, geometry.num_kv_heads, geometry.head_dim).expect("builds");
    for (position, &token) in prefix.iter().enumerate() {
        append_token(&mut warm, &geometry, position, token);
    }
    let snapshot = KvSnapshot::capture(&warm, "fixture-model").expect("capture");
    store.put(&prefix, &snapshot, 100).expect("put");
    let query: Vec<u32> = prefix.iter().chain(&continuation).copied().collect();
    let hit = store
        .lookup(&query, Some(&geometry), Some("fixture-model"), 200)
        .expect("lookup")
        .expect("hit");
    assert_eq!(hit.token_count, prefix.len());
    let mut restored = hit
        .snapshot
        .restore_checked(&geometry, "fixture-model")
        .expect("restore checked");

    // State equality against the cold cache's prefix state: the
    // cold cache has already continued, so compare against a
    // fresh prefix-only cache instead.
    let mut prefix_only =
        KvCache::new(&geometry.layers, geometry.num_kv_heads, geometry.head_dim).expect("builds");
    for (position, &token) in prefix.iter().enumerate() {
        append_token(&mut prefix_only, &geometry, position, token);
    }
    for layer in 0..geometry.layers.len() {
        let want = prefix_only.view(layer).expect("view");
        let got = restored.view(layer).expect("view");
        assert_eq!(got.len(), want.len(), "layer {layer} visible rows");
        for slot in 0..want.len() {
            assert_eq!(got.position_at(slot), want.position_at(slot));
            for head in 0..geometry.num_kv_heads {
                assert_eq!(got.key(slot, head), want.key(slot, head));
                assert_eq!(got.value(slot, head), want.value(slot, head));
            }
        }
        assert_eq!(
            restored.next_position(layer).expect("position"),
            prefix.len()
        );
    }

    let (restored_logits, restored_stream) =
        continue_run(&mut restored, &geometry, prefix.len(), &continuation);
    assert_eq!(restored_stream, cold_stream);
    assert_eq!(restored_logits, cold_logits);
    let _ = fs::remove_dir_all(&root);
}
