//! Integration tests for the KV cache: ring eviction, global growth,
//! the Glimmer layer pattern, and fail-loud validation.

use attention::{glimmer_layer_pattern, AttentionError, KvCache, LayerKind};

/// One position's row filled with a marker value.
fn row(kv_heads: usize, head_dim: usize, fill: f32) -> Vec<f32> {
    vec![fill; kv_heads * head_dim]
}

#[test]
fn glimmer_pattern_is_39_windowed_13_global() {
    let pattern = glimmer_layer_pattern();
    assert_eq!(pattern.len(), 52);
    let globals: Vec<usize> = pattern
        .iter()
        .enumerate()
        .filter(|(_, k)| **k == LayerKind::Global)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        globals,
        vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51]
    );
    let windowed = pattern
        .iter()
        .filter(|k| matches!(k, LayerKind::Windowed { .. }))
        .count();
    assert_eq!(windowed, 39);
    for kind in &pattern {
        if let LayerKind::Windowed { window } = kind {
            assert_eq!(*window, 2048);
        }
    }
}

#[test]
fn ring_buffer_wraps_and_evicts_oldest() {
    // Window 4, append positions 0..=6: only 3,4,5,6 survive, and the
    // data read back through the view is theirs, not the evicted rows.
    let mut cache = KvCache::new(&[LayerKind::windowed(4).unwrap()], 1, 1).unwrap();
    for pos in 0..=6usize {
        let marker = pos as f32;
        cache
            .append(0, &row(1, 1, marker), &row(1, 1, marker), pos)
            .unwrap();
    }
    assert_eq!(cache.stored_len(0).unwrap(), 4);
    assert_eq!(cache.next_position(0).unwrap(), 7);
    let view = cache.view(0).unwrap();
    assert_eq!(view.len(), 4);
    assert!(!view.is_empty());
    for (vi, expected_pos) in [3usize, 4, 5, 6].iter().enumerate() {
        assert_eq!(view.position_at(vi), Some(*expected_pos));
        assert_eq!(view.value(vi, 0), Some(&[*expected_pos as f32][..]));
        assert_eq!(view.key(vi, 0), Some(&[*expected_pos as f32][..]));
    }
    assert_eq!(view.position_at(4), None);
    let stats = cache.stats();
    assert_eq!(stats[0].appended, 7);
    assert_eq!(stats[0].stored, 4);
}

#[test]
fn ring_buffer_holds_exactly_the_glimmer_window() {
    // Glimmer's real window: 2048. Appending 2050 positions must cap
    // the stored count at 2048 with the oldest visible position at 2.
    let mut cache = KvCache::new(&[LayerKind::windowed(2048).unwrap()], 2, 4).unwrap();
    for pos in 0..2050usize {
        let k = row(2, 4, pos as f32);
        cache.append(0, &k, &k, pos).unwrap();
    }
    assert_eq!(cache.stored_len(0).unwrap(), 2048);
    let view = cache.view(0).unwrap();
    assert_eq!(view.position_at(0), Some(2));
    assert_eq!(view.position_at(2047), Some(2049));
    assert_eq!(view.value(0, 0).unwrap()[0], 2.0);
    assert_eq!(view.value(2047, 1).unwrap()[0], 2049.0);
}

#[test]
fn global_layer_keeps_everything() {
    let mut cache = KvCache::new(&[LayerKind::Global], 2, 2).unwrap();
    for pos in 0..10usize {
        let r = row(2, 2, pos as f32);
        cache.append(0, &r, &r, pos).unwrap();
    }
    assert_eq!(cache.stored_len(0).unwrap(), 10);
    let view = cache.view(0).unwrap();
    assert_eq!(view.len(), 10);
    assert_eq!(view.window(), None);
    for pos in 0..10usize {
        assert_eq!(view.position_at(pos), Some(pos));
        assert_eq!(view.value(pos, 1).unwrap()[0], pos as f32);
    }
}

#[test]
fn append_rejects_bad_shapes_and_out_of_order_positions() {
    let mut cache = KvCache::new(&[LayerKind::Global], 2, 4).unwrap();
    // k too short: 2 heads x 4 dims = 8 expected.
    let err = cache
        .append(0, &row(1, 4, 0.0), &row(2, 4, 0.0), 0)
        .unwrap_err();
    assert_eq!(
        err,
        AttentionError::ShapeMismatch {
            what: "k",
            expected: 8,
            actual: 4
        }
    );
    // Skipping position 0 is refused, and nothing was stored.
    let err = cache
        .append(0, &row(2, 4, 0.0), &row(2, 4, 0.0), 1)
        .unwrap_err();
    assert_eq!(
        err,
        AttentionError::OutOfOrderPosition {
            layer: 0,
            expected: 0,
            actual: 1
        }
    );
    assert_eq!(cache.stored_len(0).unwrap(), 0);
    // The correct append then succeeds.
    cache
        .append(0, &row(2, 4, 0.0), &row(2, 4, 0.0), 0)
        .unwrap();
    assert_eq!(cache.stored_len(0).unwrap(), 1);
}

#[test]
fn construction_rejects_zero_dimensions_and_window() {
    assert_eq!(
        KvCache::new(&[], 2, 128).unwrap_err(),
        AttentionError::ZeroDimension {
            what: "layer pattern"
        }
    );
    assert_eq!(
        KvCache::new(&[LayerKind::Global], 0, 128).unwrap_err(),
        AttentionError::ZeroDimension {
            what: "num_kv_heads"
        }
    );
    assert_eq!(
        KvCache::new(&[LayerKind::Global], 2, 0).unwrap_err(),
        AttentionError::ZeroDimension { what: "head_dim" }
    );
    assert_eq!(
        LayerKind::windowed(0).unwrap_err(),
        AttentionError::InvalidWindow { window: 0 }
    );
    // A zero window smuggled in via the struct literal is also caught
    // at cache construction.
    assert_eq!(
        KvCache::new(&[LayerKind::Windowed { window: 0 }], 2, 128).unwrap_err(),
        AttentionError::InvalidWindow { window: 0 }
    );
}

#[test]
fn construction_refuses_sizes_that_overflow() {
    // A window whose storage size (window times row) overflows usize
    // is refused by name, not wrapped into a small allocation and
    // not attempted as an unrepresentable one.
    assert_eq!(
        KvCache::new(&[LayerKind::Windowed { window: usize::MAX }], 2, 128).unwrap_err(),
        AttentionError::SizeOverflow {
            what: "windowed layer storage"
        }
    );
    // The KV row product itself overflowing is refused the same way,
    // even on a global layer, which allocates no ring storage: the
    // row length is checked before any storage decision.
    assert_eq!(
        KvCache::new(&[LayerKind::Global], usize::MAX, 2).unwrap_err(),
        AttentionError::SizeOverflow {
            what: "kv row length"
        }
    );
}

#[test]
fn invalid_layer_is_an_error_not_a_panic() {
    let mut cache = KvCache::new(&[LayerKind::Global], 1, 1).unwrap();
    assert_eq!(
        cache.view(5).unwrap_err(),
        AttentionError::InvalidLayer {
            layer: 5,
            layer_count: 1
        }
    );
    assert_eq!(
        cache
            .append(5, &row(1, 1, 0.0), &row(1, 1, 0.0), 0)
            .unwrap_err(),
        AttentionError::InvalidLayer {
            layer: 5,
            layer_count: 1
        }
    );
    assert_eq!(cache.layer_kind(0).unwrap(), LayerKind::Global);
}

#[test]
fn empty_view_is_empty_and_indexes_to_none() {
    let cache = KvCache::new(&[LayerKind::windowed(8).unwrap()], 1, 2).unwrap();
    let view = cache.view(0).unwrap();
    assert!(view.is_empty());
    assert_eq!(view.len(), 0);
    assert_eq!(view.position_at(0), None);
    assert_eq!(view.key(0, 0), None);
    assert_eq!(view.window(), Some(8));
}

#[test]
fn bulk_restore_matches_append_replay_bitwise() {
    // The bulk constructor must land a windowed layer in exactly
    // the state the append replay (zero rows for evicted
    // positions, then the stored rows) produces: same views,
    // same appended count, continuations identical.
    let pattern = [
        LayerKind::windowed(4).unwrap(),
        LayerKind::Global,
        LayerKind::windowed(4).unwrap(),
    ];
    let (kv_heads, head_dim) = (2usize, 3usize);
    let row_len = kv_heads * head_dim;
    let tokens = 9usize;
    let rows_for = |layer: usize, position: usize| {
        let k: Vec<f32> = (0..row_len)
            .map(|i| (position * 100 + layer * 10 + i) as f32 / 7.0)
            .collect();
        let v: Vec<f32> = (0..row_len)
            .map(|i| -((position * 50 + layer * 5 + i) as f32) / 3.0)
            .collect();
        (k, v)
    };
    // Replay: append every position in order.
    let mut replay = KvCache::new(&pattern, kv_heads, head_dim).unwrap();
    for position in 0..tokens {
        for layer in 0..pattern.len() {
            let (k, v) = rows_for(layer, position);
            replay.append(layer, &k, &v, position).unwrap();
        }
    }
    // Bulk: per layer, the visible rows only (windowed layers
    // carry the last 4, tagged with their first position).
    let mut bulk = KvCache::new(&pattern, kv_heads, head_dim).unwrap();
    for (layer, kind) in pattern.iter().enumerate() {
        let (first, count) = match kind {
            LayerKind::Global => (0, tokens),
            LayerKind::Windowed { window } => (tokens - tokens.min(*window), tokens.min(*window)),
        };
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for position in first..first + count {
            let (k, v) = rows_for(layer, position);
            keys.extend_from_slice(&k);
            values.extend_from_slice(&v);
        }
        bulk.restore_bulk(layer, first, &keys, &values).unwrap();
    }
    for layer in 0..pattern.len() {
        assert_eq!(bulk.next_position(layer).unwrap(), tokens);
        let want = replay.view(layer).unwrap();
        let got = bulk.view(layer).unwrap();
        assert_eq!(got.len(), want.len(), "layer {layer} visible rows");
        for slot in 0..want.len() {
            assert_eq!(got.position_at(slot), want.position_at(slot));
            for head in 0..kv_heads {
                assert_eq!(got.key(slot, head), want.key(slot, head));
                assert_eq!(got.value(slot, head), want.value(slot, head));
            }
        }
        // Both continue identically.
        let (k, v) = rows_for(layer, tokens);
        replay.append(layer, &k, &v, tokens).unwrap();
        bulk.append(layer, &k, &v, tokens).unwrap();
    }
}

#[test]
fn bulk_restore_refusals_are_named() {
    let pattern = [LayerKind::windowed(4).unwrap(), LayerKind::Global];
    let mut cache = KvCache::new(&pattern, 1, 2).unwrap();
    // More rows than the window holds.
    let err = cache
        .restore_bulk(0, 0, &[0.0; 10], &[0.0; 10])
        .expect_err("too many windowed rows must refuse");
    assert!(matches!(err, AttentionError::ShapeMismatch { .. }), "{err}");
    // A global layer whose rows do not start at position zero.
    let err = cache
        .restore_bulk(1, 3, &[0.0; 4], &[0.0; 4])
        .expect_err("global bulk restore must start at zero");
    assert!(matches!(err, AttentionError::ShapeMismatch { .. }), "{err}");
    // A non-fresh layer refuses, and the refusal names it.
    cache
        .append(0, &row(1, 2, 1.0), &row(1, 2, 1.0), 0)
        .unwrap();
    let err = cache
        .restore_bulk(0, 0, &[0.0; 2], &[0.0; 2])
        .expect_err("bulk restore into a used layer must refuse");
    assert!(
        matches!(
            err,
            AttentionError::BulkRestoreNotFresh {
                layer: 0,
                appended: 1
            }
        ),
        "{err}"
    );
    // Mismatched key/value lengths refuse.
    let mut fresh = KvCache::new(&pattern, 1, 2).unwrap();
    let err = fresh
        .restore_bulk(0, 0, &[0.0; 4], &[0.0; 2])
        .expect_err("mismatched row sets must refuse");
    assert!(matches!(err, AttentionError::ShapeMismatch { .. }), "{err}");
}
