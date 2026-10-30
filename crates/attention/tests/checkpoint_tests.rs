//! Integration tests for the KvCache checkpoint and rewind API
//! (SPEC-itak-spec Section 5, dependency E2): the windowed ring's
//! evicted rows come back byte for byte, a mid-block rewind keeps
//! exactly the committed prefix, global layers truncate, and every
//! contract violation is a named refusal.
//!
//! Rows carry distinct, exactly representable values per position,
//! head, and element, so "restored" is checked with exact equality:
//! a restore that recomputed, shifted, or zeroed anything would fail.

use attention::{
    glimmer_layer_pattern, gqa_attention, AttentionError, AttentionParams, KvCache, LayerKind,
};

/// One position's key row: element values encode (position, head,
/// dim) so a misplaced row is visible in the numbers themselves.
fn k_row(kv_heads: usize, head_dim: usize, position: usize) -> Vec<f32> {
    (0..kv_heads * head_dim)
        .map(|e| (position * 1000 + (e / head_dim) * 100 + (e % head_dim)) as f32)
        .collect()
}

/// One position's value row, offset from the key row so keys and
/// values can never be confused for each other.
fn v_row(kv_heads: usize, head_dim: usize, position: usize) -> Vec<f32> {
    (0..kv_heads * head_dim)
        .map(|e| (position * 1000 + 500 + (e / head_dim) * 100 + (e % head_dim)) as f32)
        .collect()
}

fn append(cache: &mut KvCache, layer: usize, kv_heads: usize, head_dim: usize, position: usize) {
    cache
        .append(
            layer,
            &k_row(kv_heads, head_dim, position),
            &v_row(kv_heads, head_dim, position),
            position,
        )
        .unwrap();
}

fn append_all(cache: &mut KvCache, kv_heads: usize, head_dim: usize, positions: &[usize]) {
    for layer in 0..cache.layer_count() {
        for &p in positions {
            append(cache, layer, kv_heads, head_dim, p);
        }
    }
}

/// Everything a view exposes, flattened oldest first: for each
/// visible position, its absolute position and every head's key and
/// value rows. Two caches with equal snapshots have byte-identical
/// views.
fn view_snapshot(cache: &KvCache, layer: usize) -> Vec<f32> {
    let view = cache.view(layer).unwrap();
    let mut out = Vec::new();
    for vi in 0..view.len() {
        out.push(view.position_at(vi).unwrap() as f32);
        for head in 0..view.num_kv_heads() {
            out.extend_from_slice(view.key(vi, head).unwrap());
            out.extend_from_slice(view.value(vi, head).unwrap());
        }
    }
    out
}

/// A reference cache fed only `positions` of the standard rows.
fn reference_cache(
    pattern: &[LayerKind],
    kv_heads: usize,
    head_dim: usize,
    positions: &[usize],
) -> KvCache {
    let mut cache = KvCache::new(pattern, kv_heads, head_dim).unwrap();
    append_all(&mut cache, kv_heads, head_dim, positions);
    cache
}

#[test]
fn windowed_rewind_restores_rows_evicted_by_speculative_appends() {
    // Window 4. The committed prefix is positions 0..=5, so the ring
    // holds 2,3,4,5. The speculative round appends 6,7,8, evicting
    // 2,3,4, rows the committed prefix still attends to. A full
    // abort (rewind to base) must bring all three back byte for
    // byte: this is the corruption case of SPEC Section 5.1.
    let pattern = [LayerKind::windowed(4).unwrap()];
    let mut cache = reference_cache(&pattern, 1, 2, &[0, 1, 2, 3, 4, 5]);
    let committed_view = view_snapshot(&cache, 0);

    let cp = cache.checkpoint(3).unwrap();
    assert!(cache.checkpoint_active());
    assert_eq!(cp.base(0), Some(6));
    for p in [6, 7, 8] {
        append(&mut cache, 0, 1, 2, p);
    }
    // The round really did evict: the view no longer matches.
    assert_ne!(view_snapshot(&cache, 0), committed_view);

    cache.rewind_to(&cp, 6).unwrap();
    cache.release(&cp).unwrap();
    assert!(!cache.checkpoint_active());

    assert_eq!(view_snapshot(&cache, 0), committed_view);
    let stats = cache.stats();
    assert_eq!(stats[0].appended, 6);
    assert_eq!(stats[0].stored, 4);

    // Attention after the rewind must be bit-identical to attention
    // on a cache that never took the speculative appends: same data
    // through the same math in the same order.
    let fresh = reference_cache(&pattern, 1, 2, &[0, 1, 2, 3, 4, 5]);
    let q = [0.5f32, -0.25];
    let params = AttentionParams::new();
    let after = gqa_attention(&q, 1, 1, &[5], &cache.view(0).unwrap(), &params).unwrap();
    let never = gqa_attention(&q, 1, 1, &[5], &fresh.view(0).unwrap(), &params).unwrap();
    assert_eq!(after, never);
}

#[test]
fn windowed_rewind_to_mid_block_keeps_exactly_the_committed_prefix() {
    // The T-2.10 proof shape: rewind a windowed ring to a mid-block
    // position and the cache view is byte-identical to a cache fed
    // only the kept prefix. Base 6, block 6,7,8, keep only 6.
    let pattern = [LayerKind::windowed(4).unwrap()];
    let mut cache = reference_cache(&pattern, 2, 2, &[0, 1, 2, 3, 4, 5]);
    let cp = cache.checkpoint(3).unwrap();
    for p in [6, 7, 8] {
        append(&mut cache, 0, 2, 2, p);
    }
    cache.rewind_to(&cp, 7).unwrap();
    cache.release(&cp).unwrap();

    // The kept prefix includes the speculative row for position 6,
    // exactly as the verify forward computed it (rows are standard
    // by position here, so the reference just feeds 0..=6).
    let reference = reference_cache(&pattern, 2, 2, &[0, 1, 2, 3, 4, 5, 6]);
    assert_eq!(view_snapshot(&cache, 0), view_snapshot(&reference, 0));
    assert_eq!(cache.stats()[0].appended, 7);

    // The sequence then continues from 7 with no seam.
    append(&mut cache, 0, 2, 2, 7);
    let reference = reference_cache(&pattern, 2, 2, &[0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(view_snapshot(&cache, 0), view_snapshot(&reference, 0));
}

#[test]
fn windowed_checkpoint_saved_slots_wrap_the_ring() {
    // Base 7 with window 4: the saved slots for positions 7 and 8
    // are slots 3 and 0, wrapping the ring end.
    let pattern = [LayerKind::windowed(4).unwrap()];
    let mut cache = reference_cache(&pattern, 1, 2, &[0, 1, 2, 3, 4, 5, 6]);
    let cp = cache.checkpoint(2).unwrap();
    for p in [7, 8] {
        append(&mut cache, 0, 1, 2, p);
    }
    cache.rewind_to(&cp, 7).unwrap();
    cache.release(&cp).unwrap();
    let reference = reference_cache(&pattern, 1, 2, &[0, 1, 2, 3, 4, 5, 6]);
    assert_eq!(view_snapshot(&cache, 0), view_snapshot(&reference, 0));
    assert_eq!(cache.stats()[0].appended, 7);
}

#[test]
fn global_rewind_truncates_and_keeps_the_prefix() {
    let pattern = [LayerKind::Global];
    let mut cache = reference_cache(&pattern, 2, 2, &[0, 1, 2, 3, 4]);
    let cp = cache.checkpoint(3).unwrap();
    // A global layer saves no rows at all.
    assert_eq!(cp.saved_bytes(), 0);
    for p in [5, 6, 7] {
        append(&mut cache, 0, 2, 2, p);
    }
    assert_eq!(cache.stored_len(0).unwrap(), 8);
    cache.rewind_to(&cp, 6).unwrap();
    cache.release(&cp).unwrap();

    let reference = reference_cache(&pattern, 2, 2, &[0, 1, 2, 3, 4, 5]);
    assert_eq!(view_snapshot(&cache, 0), view_snapshot(&reference, 0));
    assert_eq!(cache.stats()[0].appended, 6);
    assert_eq!(cache.stored_len(0).unwrap(), 6);
}

#[test]
fn rewind_to_a_checkpoint_taken_before_any_append() {
    // Mixed pattern, both kinds at base 0: a full abort empties the
    // cache and the positions can be appended again from scratch.
    let pattern = [LayerKind::windowed(4).unwrap(), LayerKind::Global];
    let mut cache = KvCache::new(&pattern, 1, 2).unwrap();
    let cp = cache.checkpoint(2).unwrap();
    assert_eq!(cp.base(0), Some(0));
    assert_eq!(cp.base(1), Some(0));
    append_all(&mut cache, 1, 2, &[0, 1]);
    cache.rewind_to(&cp, 0).unwrap();
    cache.release(&cp).unwrap();
    for layer in 0..2 {
        assert_eq!(cache.stats()[layer].appended, 0);
        assert!(cache.view(layer).unwrap().is_empty());
    }
    append_all(&mut cache, 1, 2, &[0, 1]);
    let reference = reference_cache(&pattern, 1, 2, &[0, 1]);
    for layer in 0..2 {
        assert_eq!(
            view_snapshot(&cache, layer),
            view_snapshot(&reference, layer)
        );
    }
}

#[test]
fn second_rewind_restores_the_snapshot_again() {
    // Documented contract: while the handle is live, rewinding to
    // the same (or a smaller) target restores the snapshot again,
    // discarding whatever a post-rewind append wrote into the saved
    // slots.
    let pattern = [LayerKind::windowed(4).unwrap()];
    let mut cache = reference_cache(&pattern, 1, 2, &[0, 1, 2, 3, 4, 5]);
    let committed_view = view_snapshot(&cache, 0);
    let cp = cache.checkpoint(3).unwrap();
    for p in [6, 7, 8] {
        append(&mut cache, 0, 1, 2, p);
    }
    cache.rewind_to(&cp, 6).unwrap();
    // A divergent continuation writes slot 2 again (position 6).
    append(&mut cache, 0, 1, 2, 6);
    assert_ne!(view_snapshot(&cache, 0), committed_view);
    // Rewinding to base once more restores the snapshot exactly.
    cache.rewind_to(&cp, 6).unwrap();
    cache.release(&cp).unwrap();
    assert_eq!(view_snapshot(&cache, 0), committed_view);
}

#[test]
fn checkpoint_contract_violations_are_named_refusals() {
    let pattern = [LayerKind::windowed(4).unwrap(), LayerKind::Global];
    let mut cache = reference_cache(&pattern, 1, 2, &[0, 1]);

    // Zero round bound.
    assert_eq!(
        cache.checkpoint(0).unwrap_err(),
        AttentionError::ZeroDimension {
            what: "max_round_tokens"
        }
    );
    // Round bound beyond the window: a round that long would write
    // a ring slot twice.
    assert_eq!(
        cache.checkpoint(5).unwrap_err(),
        AttentionError::CheckpointTooDeep {
            max_round_tokens: 5,
            window: 4
        }
    );

    let cp = cache.checkpoint(2).unwrap();
    // One checkpoint at a time.
    assert_eq!(
        cache.checkpoint(1).unwrap_err(),
        AttentionError::CheckpointBusy
    );
    // Appends stop at the saved range: base 2 + bound 2 = limit 4.
    append_all(&mut cache, 1, 2, &[2, 3]);
    let err = cache
        .append(0, &k_row(1, 2, 4), &v_row(1, 2, 4), 4)
        .unwrap_err();
    assert_eq!(
        err,
        AttentionError::CheckpointRangeExceeded {
            layer: 0,
            position: 4,
            limit: 4
        }
    );
    // Rewind targets outside [base, appended] refuse whole.
    assert_eq!(
        cache.rewind_to(&cp, 1).unwrap_err(),
        AttentionError::RewindOutOfRange {
            layer: 0,
            base: 2,
            appended: 4,
            target: 1
        }
    );
    assert_eq!(
        cache.rewind_to(&cp, 5).unwrap_err(),
        AttentionError::RewindOutOfRange {
            layer: 0,
            base: 2,
            appended: 4,
            target: 5
        }
    );
    // The refused rewinds changed nothing.
    assert_eq!(cache.stats()[0].appended, 4);

    cache.release(&cp).unwrap();
    // The handle is dead now: rewind and release both refuse by name.
    assert!(matches!(
        cache.rewind_to(&cp, 2),
        Err(AttentionError::StaleCheckpoint { .. })
    ));
    assert!(matches!(
        cache.release(&cp),
        Err(AttentionError::StaleCheckpoint { .. })
    ));
    // A handle from another cache is stale there too, even with a
    // checkpoint of the same serial live on it.
    let mut other = reference_cache(&pattern, 1, 2, &[0, 1]);
    let foreign = other.checkpoint(2).unwrap();
    let mine = cache.checkpoint(2).unwrap();
    assert!(matches!(
        cache.rewind_to(&foreign, 2),
        Err(AttentionError::StaleCheckpoint { .. })
    ));
    cache.release(&mine).unwrap();
    other.release(&foreign).unwrap();
}

#[test]
fn checkpoint_memory_bound_at_glimmer_geometry() {
    // The spec's bound, measured: 52 layers (39 windowed), 2 KV
    // heads, head dim 128, round bound 17 (pending + depth 16, the
    // spec.max_round_tokens default). Each saved slot is one key row
    // plus one value row: 2 * (2 * 128) floats * 4 bytes = 2048
    // bytes. Total: 39 * 17 * 2048 = 1,357,824 bytes < ~1.4 MB,
    // independent of the 2048 window.
    let pattern = glimmer_layer_pattern();
    let mut cache = KvCache::new(&pattern, 2, 128).unwrap();
    append_all(&mut cache, 2, 128, &[0]);
    let cp = cache.checkpoint(17).unwrap();
    assert_eq!(cp.base(0), Some(1));
    assert_eq!(cp.base(51), Some(1));
    assert_eq!(cp.max_round_tokens(), 17);
    assert_eq!(cp.saved_bytes(), 39 * 17 * 2048);
    assert_eq!(cp.saved_bytes(), 1_357_824);
    assert!(cp.saved_bytes() < 1_400_000);
    cache.release(&cp).unwrap();
}
