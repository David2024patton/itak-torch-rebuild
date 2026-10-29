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
