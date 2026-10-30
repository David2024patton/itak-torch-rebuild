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

/// Independent f64 brute-force attention over an explicit visible
/// set: no crate code, just the definition (scores scaled by
/// 1/sqrt(head_dim), softmax, weighted value sum) for ONE query head
/// reading ONE KV head. Used to pin the crate's mask and grouping
/// against a second implementation.
fn brute_force_head(
    q: &[f64],
    keys: &[Vec<f64>],
    values: &[Vec<f64>],
    visible: &[usize],
) -> Vec<f64> {
    let head_dim = q.len();
    let scale = 1.0 / (head_dim as f64).sqrt();
    let scores: Vec<f64> = visible
        .iter()
        .map(|&j| q.iter().zip(&keys[j]).map(|(a, b)| a * b).sum::<f64>() * scale)
        .collect();
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let mut out = vec![0.0; head_dim];
    for (e, &j) in exps.iter().zip(visible.iter()) {
        for (d, o) in out.iter_mut().enumerate() {
            *o += (e / sum) * values[j][d];
        }
    }
    out
}

fn to_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| f64::from(*x)).collect()
}

#[test]
fn window_edge_membership_matches_brute_force() {
    // Window 3, query at position 5: the visible set is exactly
    // {3, 4, 5} (positions in (p - window, p]). Scores are all
    // distinct, so including position 2 or excluding position 3
    // changes the output well beyond tolerance; the marker value at
    // position 2 (500) makes an off-by-one leak unmissable.
    let head_dim = 2;
    let rows: Vec<(Vec<f32>, Vec<f32>)> = (0..6)
        .map(|j| {
            let k = vec![(j as f32) * 0.5 - 1.0, 0.25];
            let v = if j == 2 {
                vec![500.0, 500.0]
            } else {
                vec![(j as f32) * 3.0, 1.0 - (j as f32)]
            };
            (k, v)
        })
        .collect();
    let cache = cache_with(LayerKind::windowed(3).unwrap(), 1, head_dim, &rows);
    let view = cache.view(0).unwrap();
    let q = vec![0.7f32, -0.4];
    let out = gqa_attention(&q, 1, 1, &[5], &view, &AttentionParams::new()).unwrap();
    let keys: Vec<Vec<f64>> = rows.iter().map(|(k, _)| to_f64(k)).collect();
    let values: Vec<Vec<f64>> = rows.iter().map(|(_, v)| to_f64(v)).collect();
    let want = brute_force_head(&to_f64(&q), &keys, &values, &[3, 4, 5]);
    for d in 0..head_dim {
        assert!(
            close(out[d], want[d] as f32, 1e-5),
            "dim {d}: got {}, brute force {}",
            out[d],
            want[d]
        );
    }
    // The same query on a global layer sees all six positions and
    // must differ (the marker at position 2 alone guarantees it).
    let cache = cache_with(LayerKind::Global, 1, head_dim, &rows);
    let view = cache.view(0).unwrap();
    let out_global = gqa_attention(&q, 1, 1, &[5], &view, &AttentionParams::new()).unwrap();
    assert!((out_global[0] - out[0]).abs() > 1.0);
}

#[test]
fn gqa_grouping_through_softmax_matches_brute_force() {
    // 4 query heads over 2 KV heads, 3 cached positions, head_dim 2.
    // Every head's output is checked against the brute-force result
    // for its OWN KV head (heads 0,1 -> KV 0; heads 2,3 -> KV 1)
    // with real softmax weights, not the single-position weight-1
    // shortcut of the sharing tests above.
    let head_dim = 2;
    let kv_rows: Vec<(Vec<f32>, Vec<f32>)> = vec![
        (vec![1.0, 0.0, 0.0, 1.0], vec![2.0, 0.0, 0.0, 3.0]),
        (vec![0.0, 1.0, 1.0, 0.0], vec![0.0, 5.0, 7.0, 0.0]),
        (vec![1.0, 1.0, -1.0, 0.5], vec![1.0, 1.0, 1.0, 1.0]),
    ];
    let cache = cache_with(LayerKind::Global, 2, head_dim, &kv_rows);
    let view = cache.view(0).unwrap();
    // Distinct query per head so a head mix-up changes the numbers.
    let q: Vec<f32> = vec![
        1.0, 0.0, // head 0
        0.0, 1.0, // head 1
        1.0, 1.0, // head 2
        -0.5, 0.25, // head 3
    ];
    let out = gqa_attention(&q, 1, 4, &[2], &view, &AttentionParams::new()).unwrap();
    for h in 0..4 {
        let kv = h / 2;
        let keys: Vec<Vec<f64>> = kv_rows
            .iter()
            .map(|(k, _)| to_f64(&k[kv * head_dim..(kv + 1) * head_dim]))
            .collect();
        let values: Vec<Vec<f64>> = kv_rows
            .iter()
            .map(|(_, v)| to_f64(&v[kv * head_dim..(kv + 1) * head_dim]))
            .collect();
        let qh = to_f64(&q[h * head_dim..(h + 1) * head_dim]);
        let want = brute_force_head(&qh, &keys, &values, &[0, 1, 2]);
        for d in 0..head_dim {
            assert!(
                close(out[h * head_dim + d], want[d] as f32, 1e-5),
                "head {h} dim {d}: got {}, brute force {}",
                out[h * head_dim + d],
                want[d]
            );
        }
    }
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
