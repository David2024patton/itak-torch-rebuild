//! Gate: K8's on-device argmax against the host greedy pick
//! (maximum value, lowest index on an exact tie, the
//! `itak-engine` sampler's convention), over vectors sized
//! and shaped like the real head's output, including an
//! engineered exact tie and a maximum in the last reduction
//! group's tail.
//!
//! The one-shot [`K8Pipeline::argmax`] uploads host logits
//! and reads back the eight result bytes. Devices without a
//! suitable adapter skip, per the crate's test convention.

use wgpu_compute::{K8Pipeline, WgpuCompute, WgpuError};

fn backend_or_skip(test: &str) -> Option<WgpuCompute> {
    match WgpuCompute::new() {
        Ok(b) => Some(b),
        Err(WgpuError::NoSuitableAdapter { seen, .. }) => {
            eprintln!(
                "skipping {test}: no suitable adapter (enumerated: [{}])",
                seen.join("; ")
            );
            None
        }
        Err(e) => panic!("backend construction failed unexpectedly: {e}"),
    }
}

/// The host greedy pick: a strictly-greater scan, so the
/// lowest index wins an exact tie.
fn host_greedy(logits: &[f32]) -> (u32, f32) {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    (best as u32, best_val)
}

/// Deterministic pseudo-random logits in a bounded band.
fn lcg_logits(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 40.0
        })
        .collect()
}

#[test]
fn k8_argmax_matches_host_greedy() {
    let Some(backend) = backend_or_skip("k8 argmax") else {
        return;
    };
    let ctx = backend.context();
    let k8 = K8Pipeline::new(&ctx.device).expect("k8 pipelines build");

    // The real vocabulary width, plus small widths around the
    // reduction's edges (one element, one workgroup, exactly
    // the stage stride, one past it).
    for n in [1usize, 255, 256, 257, 32_768, 32_769, 202_048] {
        let logits = lcg_logits(n, 0xA66E_0001 + n as u32);
        let (id, value) = k8.argmax(ctx, &logits).expect("device argmax");
        let (want_id, want_value) = host_greedy(&logits);
        assert_eq!(id, want_id, "winner disagrees at n={n}");
        assert_eq!(
            value, want_value,
            "winning logit must be bit-exact at n={n} (the reduction moves values, \
             never recomputes them)"
        );
    }

    // An exact tie between distant indices: the lowest wins.
    let mut logits = lcg_logits(202_048, 7);
    logits[123] = 99.5;
    logits[150_000] = 99.5;
    logits[150_001] = 99.5;
    let (id, value) = k8.argmax(ctx, &logits).expect("device argmax (tie)");
    assert_eq!(
        (id, value),
        (123, 99.5),
        "the lowest index wins an exact tie"
    );

    // The maximum in the final element, and an all-negative
    // vector (the winner below zero).
    let mut logits = lcg_logits(202_048, 99);
    let last = logits.len() - 1;
    logits[last] = 88.25;
    let (id, _) = k8.argmax(ctx, &logits).expect("device argmax (tail)");
    assert_eq!(id as usize, last);
    let logits: Vec<f32> = (0..4096).map(|i| -1.0 - (i % 97) as f32 * 0.01).collect();
    let (id, value) = k8.argmax(ctx, &logits).expect("device argmax (negative)");
    assert_eq!((id, value), host_greedy(&logits));

    // Empty input refuses by name rather than reducing nothing.
    let err = k8.argmax(ctx, &[]).expect_err("empty argmax refuses");
    assert!(err.to_string().contains("zero"), "unexpected error: {err}");
}
