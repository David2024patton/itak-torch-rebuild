//! K1 against a synthetic Q4_K tensor with known contents: the
//! GPU result must match the CPU reference (the `gguf` crate's
//! dequantizer plus `torch-math`'s matvec) within float
//! reassociation noise, and the `Compute::proj` wiring must refuse
//! a Q6_K tensor through K2 by name.
//!
//! These tests need a discrete Vulkan GPU under the default
//! selection policy. On a machine without one (a CI box, this
//! workstation) they print the refusal and pass: the policy's
//! whole point is that llvmpipe does not count, and the real-file
//! proof runs on the server through the `k1_check` example. A
//! skip here is a reported event, never a silent one.

use std::sync::Arc;

use gguf::GgmlType;
use muse_glimmer::{Compute, Dtype, TensorView};
use wgpu_compute::{WgpuCompute, WgpuError};

/// A backend, or `None` after printing why this machine cannot
/// provide one.
fn backend_or_skip() -> Option<WgpuCompute> {
    match WgpuCompute::new() {
        Ok(b) => Some(b),
        Err(WgpuError::NoSuitableAdapter { seen, .. }) => {
            eprintln!(
                "skipping GPU test: no suitable adapter (enumerated: [{}])",
                seen.join("; ")
            );
            None
        }
        Err(e) => panic!("backend construction failed unexpectedly: {e}"),
    }
}

/// One Q4_K super-block with chosen header values and
/// deterministic pseudo-random payload bytes.
fn synthetic_block(d: u16, dmin: u16, seed: u8) -> [u8; 144] {
    let mut block = [0u8; 144];
    block[0..2].copy_from_slice(&d.to_le_bytes());
    block[2..4].copy_from_slice(&dmin.to_le_bytes());
    let mut state = seed;
    for byte in &mut block[4..144] {
        state = state.wrapping_mul(31).wrapping_add(17);
        *byte = state;
    }
    block
}

/// A [256, 3] Q4_K tensor: one block per row, three different
/// (d, dmin) headers so the rows exercise different scales.
fn synthetic_tensor() -> TensorView {
    let mut data = Vec::new();
    data.extend_from_slice(&synthetic_block(0x3c00, 0x3400, 11)); // d 1.0, dmin 0.25
    data.extend_from_slice(&synthetic_block(0x3800, 0x0000, 47)); // d 0.5, dmin 0
    data.extend_from_slice(&synthetic_block(0x4000, 0x2e66, 91)); // d 2.0, dmin ~0.1
    TensorView {
        name: "synthetic.q4k".to_string(),
        dtype: Dtype::Q4K,
        shape: vec![256, 3],
        data: Arc::from(data.into_boxed_slice()),
    }
}

fn lcg_input(seed: u64, count: usize) -> Vec<f32> {
    let mut state = seed.max(1);
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as u32) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

#[test]
fn k1_proj_matches_cpu_reference_on_synthetic_rows() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let view = synthetic_tensor();
    let input = lcg_input(3, 256);
    let weights = gguf::dequant::dequantize(GgmlType::Q4K, &view.data, 256 * 3);
    let expected = torch_math::matvec(&weights, 3, 256, &input).expect("cpu reference");

    // One row (decode shape) and two rows (batch through one
    // dispatch) must both match.
    let one = backend.proj(&input, &view, 1).expect("gpu proj, one row");
    let mut two_input = input.clone();
    two_input.extend_from_slice(&input);
    let two = backend
        .proj(&two_input, &view, 2)
        .expect("gpu proj, two rows");
    let cases: [(&str, &[f32]); 3] = [
        ("one row", &one),
        ("batched row 0", &two[0..3]),
        ("batched row 1", &two[3..6]),
    ];
    for (label, got) in cases {
        let mut max_abs = 0.0f32;
        let mut scale = 0.0f32;
        for (g, e) in got.iter().zip(expected.iter()) {
            max_abs = max_abs.max((g - e).abs());
            scale = scale.max(e.abs());
        }
        let rel = max_abs / scale.max(1.0);
        assert!(
            rel < 1e-4,
            "{label}: max_abs_err {max_abs} at scale {scale} (rel {rel}) exceeds reassociation noise; got {got:?}, expected {expected:?}"
        );
    }
}

#[test]
fn k1_proj_refuses_q6k_through_k2_by_name() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    // Geometry is never reached: the dtype routes first. The data
    // length is the honest Q6_K size for [256, 1] anyway.
    let view = TensorView {
        name: "synthetic.q6k".to_string(),
        dtype: Dtype::Q6K,
        shape: vec![256, 1],
        data: Arc::from(vec![0u8; 210].into_boxed_slice()),
    };
    let err = backend
        .proj(&[0.0; 256], &view, 1)
        .expect_err("Q6_K must refuse until K2 lands");
    match err {
        muse_glimmer::GlimmerError::StepFailed { step, detail } => {
            assert_eq!(step, "proj");
            assert!(detail.contains("K2"), "refusal must name K2: {detail}");
        }
        other => panic!("expected StepFailed, got {other:?}"),
    }
}

#[test]
fn k1_reupload_under_a_cached_name_with_new_geometry_refuses() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let view = synthetic_tensor();
    let input = lcg_input(5, 256);
    backend.proj(&input, &view, 1).expect("first upload");
    // Same name, different shape and byte count: the name is an
    // identity, so this refuses rather than replacing the buffer.
    let impostor = TensorView {
        name: "synthetic.q4k".to_string(),
        dtype: Dtype::Q4K,
        shape: vec![256, 1],
        data: Arc::from(
            synthetic_block(0x3c00, 0x3400, 3)
                .to_vec()
                .into_boxed_slice(),
        ),
    };
    let err = backend
        .proj(&input, &impostor, 1)
        .expect_err("geometry change under a cached name must refuse");
    assert!(
        matches!(
            err,
            muse_glimmer::GlimmerError::StepFailed { step: "proj", .. }
        ),
        "got {err:?}"
    );
}
