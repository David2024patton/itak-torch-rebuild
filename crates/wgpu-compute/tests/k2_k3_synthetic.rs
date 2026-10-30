//! K2 and K3 against synthetic Q6_K and Q5_K tensors with known
//! contents: the GPU results must match the CPU reference (the
//! `gguf` crate's dequantizer plus `torch-math`'s matvec) within
//! float reassociation noise, end to end through
//! `Compute::proj`. K3 additionally proves the two-dimensional
//! dispatch on a row count whose workgroup total crosses the
//! 65,535 per-dimension cap, the head geometry in miniature.
//!
//! These tests need a discrete Vulkan GPU under the default
//! selection policy. On a machine without one (a CI box, this
//! workstation) they print the refusal and pass: the policy's
//! whole point is that llvmpipe does not count, and the real-file
//! proofs run on the server through the `k2_check` and
//! `k3_check` examples. A skip here is a reported event, never a
//! silent one.

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

fn fill_payload(bytes: &mut [u8], seed: u8) {
    let mut state = seed;
    for byte in bytes.iter_mut() {
        state = state.wrapping_mul(31).wrapping_add(17);
        *byte = state;
    }
}

/// One Q6_K super-block (210 bytes): payload in the ql and qh
/// planes, a spread of signed scales (including negatives, which
/// exercise the sign extension), and the chosen f16 `d`.
fn synthetic_q6k_block(d_bits: u16, seed: u8) -> [u8; 210] {
    let mut block = [0u8; 210];
    fill_payload(&mut block[0..192], seed);
    for (i, sc) in block[192..208].iter_mut().enumerate() {
        *sc = (i as i8).wrapping_mul(3).wrapping_sub(20) as u8;
    }
    block[208..210].copy_from_slice(&d_bits.to_le_bytes());
    block
}

/// A [256, 3] Q6_K tensor: one block per row, three different
/// `d` headers.
fn synthetic_q6k_tensor() -> TensorView {
    let mut data = Vec::new();
    data.extend_from_slice(&synthetic_q6k_block(0x3c00, 13)); // d 1.0
    data.extend_from_slice(&synthetic_q6k_block(0x3800, 59)); // d 0.5
    data.extend_from_slice(&synthetic_q6k_block(0x4000, 97)); // d 2.0
    TensorView {
        name: "synthetic.q6k".to_string(),
        dtype: Dtype::Q6K,
        shape: vec![256, 3],
        data: Arc::from(data.into_boxed_slice()),
    }
}

/// One Q5_K super-block (176 bytes): d and dmin headers, the
/// packed scale/min bytes, the qh bit plane, and the ql nibbles.
fn synthetic_q5k_block(d_bits: u16, dmin_bits: u16, seed: u8) -> [u8; 176] {
    let mut block = [0u8; 176];
    block[0..2].copy_from_slice(&d_bits.to_le_bytes());
    block[2..4].copy_from_slice(&dmin_bits.to_le_bytes());
    fill_payload(&mut block[4..176], seed);
    block
}

/// A [256, 3] Q5_K tensor: one block per row, three different
/// (d, dmin) headers.
fn synthetic_q5k_tensor() -> TensorView {
    let mut data = Vec::new();
    data.extend_from_slice(&synthetic_q5k_block(0x3c00, 0x3400, 21)); // d 1.0, dmin 0.25
    data.extend_from_slice(&synthetic_q5k_block(0x3800, 0x0000, 67)); // d 0.5, dmin 0
    data.extend_from_slice(&synthetic_q5k_block(0x4000, 0x2e66, 111)); // d 2.0, dmin ~0.1
    TensorView {
        name: "synthetic.q5k".to_string(),
        dtype: Dtype::Q5K,
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

/// Compare a GPU result against the CPU reference at the
/// projection tolerance (1e-4 relative to the largest expected
/// magnitude, the reassociation noise band).
fn assert_matches(label: &str, got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    let mut max_abs = 0.0f32;
    let mut scale = 0.0f32;
    for (g, e) in got.iter().zip(expected.iter()) {
        max_abs = max_abs.max((g - e).abs());
        scale = scale.max(e.abs());
    }
    let rel = max_abs / scale.max(1.0);
    assert!(
        rel < 1e-4,
        "{label}: max_abs_err {max_abs} at scale {scale} (rel {rel}) exceeds reassociation noise"
    );
}

#[test]
fn k2_proj_matches_cpu_reference_on_synthetic_rows() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let view = synthetic_q6k_tensor();
    let input = lcg_input(3, 256);
    let weights = gguf::dequant::dequantize(GgmlType::Q6K, &view.data, 256 * 3);
    let expected = torch_math::matvec(&weights, 3, 256, &input).expect("cpu reference");

    let one = backend.proj(&input, &view, 1).expect("gpu proj, one row");
    let mut two_input = input.clone();
    two_input.extend_from_slice(&input);
    let two = backend
        .proj(&two_input, &view, 2)
        .expect("gpu proj, two rows");
    assert_matches("k2 one row", &one, &expected);
    assert_matches("k2 batched row 0", &two[0..3], &expected);
    assert_matches("k2 batched row 1", &two[3..6], &expected);
}

#[test]
fn k3_proj_matches_cpu_reference_on_synthetic_rows() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let view = synthetic_q5k_tensor();
    let input = lcg_input(3, 256);
    let weights = gguf::dequant::dequantize(GgmlType::Q5K, &view.data, 256 * 3);
    let expected = torch_math::matvec(&weights, 3, 256, &input).expect("cpu reference");

    let one = backend.proj(&input, &view, 1).expect("gpu proj, one row");
    let mut two_input = input.clone();
    two_input.extend_from_slice(&input);
    let two = backend
        .proj(&two_input, &view, 2)
        .expect("gpu proj, two rows");
    assert_matches("k3 one row", &one, &expected);
    assert_matches("k3 batched row 0", &two[0..3], &expected);
    assert_matches("k3 batched row 1", &two[3..6], &expected);
}

#[test]
fn k3_proj_matches_cpu_reference_across_the_2d_grid() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    // 140,000 output rows at two rows per workgroup is 70,000
    // workgroups: past the 65,535 per-dimension cap, so the
    // dispatch wraps into the second grid dimension, the head
    // geometry in miniature (the real head is 202,048 rows).
    // Seven distinct blocks cycle through the rows; the cycle
    // length does not divide the grid span, so a row-indexing
    // error across the wrap lands on the wrong block and shows.
    const ROWS: usize = 140_000;
    let blocks: Vec<[u8; 176]> = (0..7u8)
        .map(|i| synthetic_q5k_block(0x3c00, 0x3400, 21 + i * 13))
        .collect();
    let mut data = Vec::with_capacity(ROWS * 176);
    for r in 0..ROWS {
        data.extend_from_slice(&blocks[r % blocks.len()]);
    }
    let view = TensorView {
        name: "synthetic.q5k.grid".to_string(),
        dtype: Dtype::Q5K,
        shape: vec![256, ROWS as u64],
        data: Arc::from(data.into_boxed_slice()),
    };
    let input = lcg_input(9, 256);
    let weights = gguf::dequant::dequantize(GgmlType::Q5K, &view.data, 256 * ROWS);
    let expected = torch_math::matvec(&weights, ROWS, 256, &input).expect("cpu reference");
    let got = backend.proj(&input, &view, 1).expect("gpu proj, 2d grid");
    assert_matches("k3 2d grid", &got, &expected);
}
