//! Integration tests for grouped-query attention: the window mask,
//! the causal mask, global visibility, GQA head sharing, and
//! fail-loud validation. All tensors are small and synthetic; the
//! expected outputs are computed by hand.

use attention::{
    gqa_attention, kv_head_for_query_head, AttentionError, AttentionParams, KvCache, LayerKind,
};

/// Build a single-layer cache and append the given per-position rows
/// in order. Each row is the full `[num_kv_heads][head_dim]` layout.
fn cache_with(
    kind: LayerKind,
    kv_heads: usize,
    head_dim: usize,
    rows: &[(Vec<f32>, Vec<f32>)],
) -> KvCache {
    let mut cache = KvCache::new(&[kind], kv_heads, head_dim).unwrap();
    for (pos, (k, v)) in rows.iter().enumerate() {
        cache.append(0, k, v, pos).unwrap();
    }
    cache
}

fn close(actual: f32, expected: f32, tol: f32) -> bool {
    (actual - expected).abs() <= tol
}

#[test]
fn windowed_layer_provably_cannot_see_beyond_the_window() {
    // All keys identical, so all scores are equal and attention is a
    // plain average of the visible values. Position 0 carries the
    // marker value 1000: if the window (2) leaked, the query at
    // position 2 would average (1000 + 3 + 5) / 3 = 336 instead of
    // (3 + 5) / 2 = 4.
    let rows = vec![
        (vec![1.0], vec![1000.0]),
        (vec![1.0], vec![3.0]),
        (vec![1.0], vec![5.0]),
    ];
    let cache = cache_with(LayerKind::windowed(2).unwrap(), 1, 1, &rows);
    let view = cache.view(0).unwrap();
    let out = gqa_attention(&[1.0], 1, 1, &[2], &view, &AttentionParams::new()).unwrap();
    assert!(close(out[0], 4.0, 1e-6), "got {}", out[0]);
}

#[test]
fn global_layer_sees_everything() {
    // The exact setup of the window test on a global layer: the
    // marker at position 0 IS visible, so the same query averages
    // all three values.
    let rows = vec![
        (vec![1.0], vec![1000.0]),
        (vec![1.0], vec![3.0]),
        (vec![1.0], vec![5.0]),
    ];
    let cache = cache_with(LayerKind::Global, 1, 1, &rows);
    let view = cache.view(0).unwrap();
    let out = gqa_attention(&[1.0], 1, 1, &[2], &view, &AttentionParams::new()).unwrap();
    assert!(close(out[0], 336.0, 1e-3), "got {}", out[0]);
}

#[test]
fn causal_mask_blocks_future_positions() {
    // Keys are crafted so the future keys would dominate if visible:
    // key 0 scores 0, keys 1 and 2 score 10 against query [1.0].
    let rows = vec![
        (vec![0.0], vec![1.0]),
        (vec![10.0], vec![100.0]),
        (vec![10.0], vec![1000.0]),
    ];
    let cache = cache_with(LayerKind::Global, 1, 1, &rows);
    let view = cache.view(0).unwrap();
    // Two queries in one call: position 0 sees only itself (output
    // exactly its own value, weight 1); position 1 sees keys 0 and 1
    // (softmax over scores 0 and 10 gives ~99.995), never key 2's
    // 1000.
    let out = gqa_attention(&[1.0, 1.0], 2, 1, &[0, 1], &view, &AttentionParams::new()).unwrap();
    assert!(close(out[0], 1.0, 1e-6), "got {}", out[0]);
    assert!(
        out[1] > 99.9 && out[1] < 100.0,
        "position 1 output {} must sit at ~99.995, below 100",
        out[1]
    );
}

#[test]
fn gqa_head_sharing_glimmer_shape_32_to_2() {
    // The mapping function itself, exhaustively, for Glimmer's 32/2.
    for h in 0..32 {
        let expected = if h < 16 { 0 } else { 1 };
        assert_eq!(kv_head_for_query_head(h, 32, 2).unwrap(), expected);
    }
    // End to end: one cached position, so every head's softmax is a
    // single weight of 1 and each query head returns exactly its KV
    // head's value (10 for KV 0, 20 for KV 1).
    let rows = vec![(vec![1.0, 0.0, 1.0, 0.0], vec![10.0, 0.0, 20.0, 0.0])];
    let cache = cache_with(LayerKind::Global, 2, 2, &rows);
    let view = cache.view(0).unwrap();
    let mut q = Vec::new();
    for _ in 0..32 {
        q.extend_from_slice(&[1.0, 0.0]);
    }
    let out = gqa_attention(&q, 1, 32, &[0], &view, &AttentionParams::new()).unwrap();
    for h in 0..32 {
        let expected = if h < 16 { 10.0 } else { 20.0 };
        assert!(
            close(out[h * 2], expected, 1e-6),
            "head {h} got {} expected {expected}",
            out[h * 2]
        );
        assert!(close(out[h * 2 + 1], 0.0, 1e-6));
    }
}

#[test]
fn gqa_head_sharing_smaller_ratio_4_to_2() {
    for (h, expected) in [(0, 0), (1, 0), (2, 1), (3, 1)] {
        assert_eq!(kv_head_for_query_head(h, 4, 2).unwrap(), expected);
    }
    let rows = vec![(vec![1.0, 1.0], vec![7.0, 9.0])];
    let cache = cache_with(LayerKind::Global, 2, 1, &rows);
    let view = cache.view(0).unwrap();
    let out = gqa_attention(
        &[1.0, 1.0, 1.0, 1.0],
        1,
        4,
        &[0],
        &view,
        &AttentionParams::new(),
    )
    .unwrap();
    for (h, expected) in [(0, 7.0), (1, 7.0), (2, 9.0), (3, 9.0)] {
        assert!(
            close(out[h], expected, 1e-6),
            "head {h} got {} expected {expected}",
            out[h]
        );
    }
}

#[test]
fn scale_defaults_to_one_over_sqrt_head_dim_and_is_a_parameter() {
    let params = AttentionParams::new();
    assert!(close(params.scale_for(128), 0.088_388_35, 1e-7));
    assert!(close(params.scale_for(1), 1.0, 1e-7));
    let pinned = AttentionParams::with_scale(0.25);
    assert!(close(pinned.scale_for(128), 0.25, 1e-7));
    assert_eq!(AttentionParams::default(), params);
}

#[test]
fn attention_validation_is_named_errors_not_panics() {
    // Empty cache: a query has no visible keys.
    let cache = KvCache::new(&[LayerKind::Global], 2, 2).unwrap();
    let view = cache.view(0).unwrap();
    let err = gqa_attention(
        &[1.0, 0.0, 1.0, 0.0],
        1,
        2,
        &[0],
        &view,
        &AttentionParams::new(),
    )
    .unwrap_err();
    assert_eq!(err, AttentionError::NoVisibleKeys { position: 0 });

    // 3 query heads across 2 KV heads does not divide.
    let rows = vec![(vec![1.0, 0.0, 1.0, 0.0], vec![1.0, 0.0, 1.0, 0.0])];
    let cache = cache_with(LayerKind::Global, 2, 2, &rows);
    let view = cache.view(0).unwrap();
    let q = vec![1.0; 3 * 2];
    let err = gqa_attention(&q, 1, 3, &[0], &view, &AttentionParams::new()).unwrap_err();
    assert_eq!(
        err,
        AttentionError::NonDivisibleHeads {
            num_q_heads: 3,
            num_kv_heads: 2
        }
    );

    // q slice of the wrong length.
    let err = gqa_attention(&[1.0], 1, 2, &[0], &view, &AttentionParams::new()).unwrap_err();
    assert_eq!(
        err,
        AttentionError::ShapeMismatch {
            what: "q",
            expected: 4,
            actual: 1
        }
    );

    // query_positions length disagrees with num_queries.
    let err = gqa_attention(&[1.0; 4], 2, 2, &[0], &view, &AttentionParams::new()).unwrap_err();
    assert_eq!(
        err,
        AttentionError::ShapeMismatch {
            what: "query_positions",
            expected: 2,
            actual: 1
        }
    );

    // The mapping helper refuses the same non-divisible shape.
    assert_eq!(
        kv_head_for_query_head(0, 3, 2).unwrap_err(),
        AttentionError::NonDivisibleHeads {
            num_q_heads: 3,
            num_kv_heads: 2
        }
    );
}
