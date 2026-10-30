//! CPU backend op tests. Every expected value is computed in the
//! test from the op's definition (hand arithmetic or an explicit
//! f64 formula written here), never by calling the library the
//! backend delegates to: the point is an independent reference.

use gguf::GgmlType;
use itak_backend::{Backend, BackendError, CpuBackend};

fn assert_vec_close(actual: &[f32], expected: &[f64], tol: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: got {}, expected {}",
        actual.len(),
        expected.len()
    );
    for (a, e) in actual.iter().zip(expected.iter()) {
        let diff = (f64::from(*a) - e).abs();
        assert!(
            diff <= tol,
            "expected {e}, got {a} (diff {diff} > tol {tol})"
        );
    }
}

const B: CpuBackend = CpuBackend;

#[test]
fn matvec_matches_hand_product() {
    // [[1, 2], [3, 4]] * [5, 6] = [17, 39].
    let out = B.matvec(&[1.0, 2.0, 3.0, 4.0], 2, 2, &[5.0, 6.0]).unwrap();
    assert_vec_close(&out, &[17.0, 39.0], 1e-6);
}

#[test]
fn matmul_matches_hand_product() {
    // [[1,2,3],[4,5,6]] * [[7,8],[9,10],[11,12]] = [[58,64],[139,154]].
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let out = B.matmul(&a, 2, 3, &b, 3, 2).unwrap();
    assert_vec_close(&out, &[58.0, 64.0, 139.0, 154.0], 1e-5);
}

#[test]
fn matvec_shape_error_is_named() {
    let err = B.matvec(&[1.0, 2.0, 3.0], 2, 2, &[1.0, 1.0]).unwrap_err();
    assert!(matches!(err, BackendError::Math(_)), "got {err}");
}

/// One Q8_0 block: f16 scale bits followed by 32 signed bytes.
fn q8_0_block(scale_bits: u16, qs: [i8; 32]) -> Vec<u8> {
    let mut block = Vec::with_capacity(34);
    block.extend_from_slice(&scale_bits.to_le_bytes());
    block.extend(qs.iter().map(|q| *q as u8));
    block
}

#[test]
fn matvec_quantized_q8_0_matches_known_dequant() {
    // f16 0x3800 = 0.5, 0x3400 = 0.25 (exact powers of two).
    // Row 0: 0.5 * 2 = 1.0 per element; row 1: 0.25 * 4 = 1.0.
    let mut payload = q8_0_block(0x3800, [2; 32]);
    payload.extend(q8_0_block(0x3400, [4; 32]));
    let ones = [1.0f32; 32];
    let out = B
        .matvec_quantized(GgmlType::Q8_0, &payload, 2, 32, &ones)
        .unwrap();
    assert_vec_close(&out, &[32.0, 32.0], 1e-6);
    // Alternating 1/0 vector halves each dot product.
    let alt: Vec<f32> = (0..32)
        .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
        .collect();
    let out = B
        .matvec_quantized(GgmlType::Q8_0, &payload, 2, 32, &alt)
        .unwrap();
    assert_vec_close(&out, &[16.0, 16.0], 1e-6);
}

#[test]
fn matmul_quantized_q8_0_matches_known_dequant() {
    let mut payload = q8_0_block(0x3800, [2; 32]);
    payload.extend(q8_0_block(0x3400, [4; 32]));
    let input = [1.0f32; 32];
    let out = B
        .matmul_quantized(GgmlType::Q8_0, &payload, 2, 32, &input, 1)
        .unwrap();
    assert_vec_close(&out, &[32.0, 32.0], 1e-6);
}

#[test]
fn matvec_quantized_f32_payload_is_the_dense_product() {
    // The quantized route with an F32 payload must equal the dense
    // product of the same numbers.
    let mut payload = Vec::new();
    for v in [1.0f32, 2.0, 3.0, 4.0] {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    let out = B
        .matvec_quantized(GgmlType::F32, &payload, 2, 2, &[5.0, 6.0])
        .unwrap();
    assert_vec_close(&out, &[17.0, 39.0], 1e-6);
}

#[test]
fn matvec_quantized_zero_k_blocks_are_zero() {
    // An all-zero K-quant block has scale 0 and min 0, so every
    // element dequantizes to exactly 0 regardless of layout.
    let ones = vec![1.0f32; 256];
    for (ty, block_bytes) in [
        (GgmlType::Q4K, 144),
        (GgmlType::Q5K, 176),
        (GgmlType::Q6K, 210),
    ] {
        let payload = vec![0u8; block_bytes];
        let out = B.matvec_quantized(ty, &payload, 1, 256, &ones).unwrap();
        assert_eq!(out, vec![0.0], "type {ty:?}");
    }
}

#[test]
fn dequantize_q8_0_known_values() {
    // f16 0x4000 = 2.0; elements are 2.0 * q.
    let mut qs = [0i8; 32];
    qs[0] = 1;
    qs[1] = -1;
    qs[2] = 2;
    qs[3] = -2;
    let payload = q8_0_block(0x4000, qs);
    let out = B.dequantize(GgmlType::Q8_0, &payload, 32).unwrap();
    assert_vec_close(&out[..4], &[2.0, -2.0, 4.0, -4.0], 1e-6);
    assert!(out[4..].iter().all(|v| *v == 0.0));
}

#[test]
fn quantized_payload_length_is_checked_before_indexing() {
    // 33 bytes for 32 Q8_0 elements (34 required) is a named
    // refusal, not a short read.
    let err = B.dequantize(GgmlType::Q8_0, &[0u8; 33], 32).unwrap_err();
    assert_eq!(
        err,
        BackendError::ByteLengthMismatch {
            op: "dequantize",
            ty: GgmlType::Q8_0,
            expected: 34,
            actual: 33,
        }
    );
    // 100 elements is not a whole number of Q4_K blocks (256).
    let err = B.dequantize(GgmlType::Q4K, &[0u8; 144], 100).unwrap_err();
    assert_eq!(
        err,
        BackendError::BadElementCount {
            op: "dequantize",
            ty: GgmlType::Q4K,
            elements: 100,
        }
    );
}

#[test]
fn rms_norm_matches_independent_formula() {
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let weight = [1.0f32, 2.0, 0.5, 1.0];
    let eps = 1e-5f32;
    let out = B.rms_norm(&input, &weight, eps).unwrap();
    let mean_sq: f64 = input
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        / 4.0;
    let denom = (mean_sq + f64::from(eps)).sqrt();
    let expected: Vec<f64> = input
        .iter()
        .zip(weight.iter())
        .map(|(v, w)| f64::from(*v) / denom * f64::from(*w))
        .collect();
    assert_vec_close(&out, &expected, 1e-5);
}

#[test]
fn rms_norm_add_returns_normed_and_the_summed_residual() {
    // input + residual = [1, 1]; RMS of [1, 1] is 1, so the normed
    // output is [1, 1] up to the eps correction, and the residual
    // stream is the sum itself.
    let (normed, residual) = B
        .rms_norm_add(&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0], 1e-6)
        .unwrap();
    assert_vec_close(&residual, &[1.0, 1.0], 0.0);
    let expected = 1.0 / (1.0f64 + 1e-6).sqrt();
    assert_vec_close(&normed, &[expected, expected], 1e-5);
}

#[test]
fn rope_matches_independent_rotation() {
    // Position 0 is the identity, exactly.
    let input = [0.3f32, -1.2, 2.5, 0.7];
    assert_eq!(B.rope(&input, 0, 500_000.0, 4, 4).unwrap(), input.to_vec());
    // head_dim 4, base 10000, position 1: pair 0 rotates by 1
    // radian, pair 1 by 10000^(-1/2) = 0.01 radians.
    let out = B.rope(&[1.0, 0.0, 1.0, 0.0], 1, 10_000.0, 4, 4).unwrap();
    let expected = [1.0f64.cos(), 1.0f64.sin(), 0.01f64.cos(), 0.01f64.sin()];
    assert_vec_close(&out, &expected, 1e-5);
}

#[test]
fn softmax_matches_independent_formula() {
    let input = [1.0f32, 2.0, 3.0];
    let out = B.softmax(&input).unwrap();
    let exps: Vec<f64> = input.iter().map(|v| f64::from(*v).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let expected: Vec<f64> = exps.iter().map(|e| e / sum).collect();
    assert_vec_close(&out, &expected, 1e-6);
    // A probability distribution sums to 1.
    let total: f32 = out.iter().sum();
    assert!((total - 1.0).abs() < 1e-6);
}

#[test]
fn softmax_is_stable_at_large_logits() {
    // Without max-subtraction these exps overflow f32 and f64 alike
    // at larger offsets; the distribution must stay the logistic
    // pair for a unit gap.
    let out = B.softmax(&[1000.0, 1001.0]).unwrap();
    let e = std::f64::consts::E;
    assert_vec_close(&out, &[1.0 / (1.0 + e), e / (1.0 + e)], 1e-6);
}

#[test]
fn softmax_empty_input_is_an_error() {
    assert!(B.softmax(&[]).is_err());
}

#[test]
fn attention_matches_hand_computed_softmax_mix() {
    // One query head, one KV head, head_dim 2, two keys.
    // keys: pos0 [1, 0], pos1 [0, 1]; values: pos0 [10, 0], pos1 [0, 20].
    // Query [1, 0] at position 1 with scale 1: scores are 1 and 0,
    // softmax weights are e/(e+1) and 1/(e+1).
    let q = [1.0f32, 0.0];
    let keys = [1.0f32, 0.0, 0.0, 1.0];
    let values = [10.0f32, 0.0, 0.0, 20.0];
    let out = B
        .attention(&q, &keys, &values, 1, 1, 2, 1, None, Some(1.0))
        .unwrap();
    let e = std::f64::consts::E;
    let w0 = e / (e + 1.0);
    let w1 = 1.0 / (e + 1.0);
    assert_vec_close(&out, &[w0 * 10.0, w1 * 20.0], 1e-5);
}

#[test]
fn attention_window_limits_visibility() {
    // Same setup, window 1: only the newest key (position 1) is
    // visible to a query at position 1, so the output is exactly
    // that position's value vector.
    let q = [1.0f32, 0.0];
    let keys = [1.0f32, 0.0, 0.0, 1.0];
    let values = [10.0f32, 0.0, 0.0, 20.0];
    let out = B
        .attention(&q, &keys, &values, 1, 1, 2, 1, Some(1), Some(1.0))
        .unwrap();
    assert_vec_close(&out, &[0.0, 20.0], 1e-6);
}

#[test]
fn attention_is_causal_for_early_positions() {
    // A query at position 0 sees only position 0 even when later
    // keys exist.
    let q = [1.0f32, 0.0];
    let keys = [1.0f32, 0.0, 0.0, 1.0];
    let values = [10.0f32, 0.0, 0.0, 20.0];
    let out = B
        .attention(&q, &keys, &values, 1, 1, 2, 0, None, Some(1.0))
        .unwrap();
    assert_vec_close(&out, &[10.0, 0.0], 1e-6);
}

#[test]
fn attention_shares_kv_heads_across_query_groups() {
    // GQA: 2 query heads over 1 KV head, one key. Both heads must
    // read the single KV head and return its value.
    let q = [1.0f32, 0.0, 0.0, 1.0];
    let keys = [1.0f32, 0.0];
    let values = [3.0f32, 4.0];
    let out = B
        .attention(&q, &keys, &values, 2, 1, 2, 0, None, Some(1.0))
        .unwrap();
    assert_vec_close(&out, &[3.0, 4.0, 3.0, 4.0], 1e-6);
}

#[test]
fn attention_rejects_bad_head_geometry() {
    let q = [1.0f32, 0.0];
    let keys = [1.0f32, 0.0];
    let values = [3.0f32, 4.0];
    // 3 query heads over 2 KV heads does not divide evenly.
    let q3 = [1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0];
    assert!(B
        .attention(&q3, &keys, &values, 3, 2, 2, 0, None, None)
        .is_err());
    // Empty history is an error, not a zero vector.
    assert!(B.attention(&q, &[], &[], 1, 1, 2, 0, None, None).is_err());
}

#[test]
fn swiglu_matches_independent_silu() {
    // silu(x) = x / (1 + e^-x); expected from the formula here.
    let gate = [1.0f32, -2.0, 0.5];
    let up = [2.0f32, 3.0, -4.0];
    let out = B.swiglu(&gate, &up).unwrap();
    let expected: Vec<f64> = gate
        .iter()
        .zip(up.iter())
        .map(|(g, u)| {
            let g = f64::from(*g);
            g / (1.0 + (-g).exp()) * f64::from(*u)
        })
        .collect();
    assert_vec_close(&out, &expected, 1e-5);
}

#[test]
fn geglu_zero_gate_passes_nothing() {
    // gelu_tanh(0) = 0 exactly.
    let out = B.geglu(&[0.0], &[5.0]).unwrap();
    assert_vec_close(&out, &[0.0], 1e-7);
}

#[test]
fn embedding_gathers_the_named_row() {
    let table = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_eq!(B.embedding(&table, 3, 2, 2).unwrap(), vec![5.0, 6.0]);
    assert_eq!(B.embedding(&table, 3, 2, 0).unwrap(), vec![1.0, 2.0]);
    let err = B.embedding(&table, 3, 2, 3).unwrap_err();
    assert_eq!(
        err,
        BackendError::IndexOutOfRange {
            op: "embedding",
            index: 3,
            len: 3,
        }
    );
}

#[test]
fn elementwise_ops_and_their_length_check() {
    assert_eq!(B.add(&[1.0, 2.0], &[10.0, 20.0]).unwrap(), vec![11.0, 22.0]);
    assert_eq!(B.mul(&[2.0, 3.0], &[4.0, 5.0]).unwrap(), vec![8.0, 15.0]);
    assert!(B.add(&[1.0], &[1.0, 2.0]).is_err());
    assert!(B.mul(&[1.0, 2.0], &[1.0]).is_err());
}

#[test]
fn logit_shaping_matches_formulas() {
    // softcap at x == cap is cap * tanh(1).
    let out = B.logit_softcap(&[20.0], 20.0).unwrap();
    assert_vec_close(&out, &[20.0 * 1.0f64.tanh()], 1e-4);
    let out = B.logit_scale(&[1.0, -2.0], 0.1961).unwrap();
    assert_vec_close(&out, &[0.1961, -0.3922], 1e-6);
}
