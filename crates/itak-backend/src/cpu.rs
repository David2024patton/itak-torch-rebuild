//! The CPU backend: the correctness reference for every backend
//! op, and a supported serving mode in its own right (BACKEND.md
//! Section 6: it is the L4 landing and the T-4.1 proof target).
//!
//! Where the workspace already owns reference math, this backend is
//! a thin, honest delegation: dense products, RMSNorm, RoPE, the
//! gated activations, and the logit shaping go to `torch-math`, and
//! dequantization goes to the `gguf` crate's ggml-exact
//! dequantizers. The quantized products dequantize through that same
//! path and then run the dense product: unlike the GPU kernels of
//! BACKEND.md Section 3, which fuse dequantize-and-multiply and
//! never materialize f32 weights, the CPU reference materializes the
//! f32 matrix per call. That is a deliberate correctness-first
//! choice, stated here so nobody mistakes this for the speed path.
//!
//! Ops with no existing reference implementation anywhere in the
//! workspace (softmax as a standalone op, decode attention over
//! flat buffers, embedding gather, elementwise add and multiply)
//! are implemented in this module as scalar reference versions and
//! are named as such in the capability report.
//!
//! SCALAR HONESTY: every op in this backend is
//! [`ImplKind::Scalar`]. No SIMD intrinsics run anywhere in this
//! crate, whatever the host CPU advertises; the data layout (flat,
//! contiguous, row-major) is the layout the SIMD pass will consume,
//! so vectorization later changes loop bodies, not interfaces. The
//! capability report says `Scalar` for every op, and a test pins
//! that, so the claim cannot silently drift.

use gguf::GgmlType;
use torch_math::MathError;

use crate::backend::{Backend, Result};
use crate::diag;
use crate::error::BackendError;
use crate::report::{
    BackendKind, CapabilityReport, ImplKind, Op, OpStatus, OpSupport, ALL_GGML_TYPES,
};

/// The CPU reference backend. Stateless: every op is a pure
/// function, so one instance serves any number of callers.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend;

/// Validate a quantized payload before any byte is indexed: the
/// element count must be a whole number of blocks and the byte
/// length must be the exact length the type requires
/// (ARCHITECTURE 5.2).
fn check_payload(op: &'static str, ty: GgmlType, data: &[u8], elements: usize) -> Result<()> {
    match ty.byte_len(elements as u64) {
        None => Err(BackendError::BadElementCount { op, ty, elements }),
        Some(expected) if expected != data.len() as u64 => Err(BackendError::ByteLengthMismatch {
            op,
            ty,
            expected,
            actual: data.len() as u64,
        }),
        Some(_) => Ok(()),
    }
}

/// `rows * cols` with overflow reported as a math error, in the
/// workspace's no-panic convention.
fn checked_len(op: &'static str, rows: usize, cols: usize) -> Result<usize> {
    rows.checked_mul(cols)
        .ok_or(BackendError::Math(MathError::InvalidConfig {
            op,
            param: "shape",
            reason: "rows * cols overflows usize",
        }))
}

/// Scalar softmax, max-subtracted, with the exponential sum
/// accumulated in f64 so the reference is the accurate version of
/// the formula (the `torch-math` house style).
fn softmax_scalar(input: &[f32]) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Err(BackendError::Math(MathError::EmptyInput { op: "softmax" }));
    }
    let max = input
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |a, v| a.max(f64::from(v)));
    let mut out: Vec<f64> = input.iter().map(|v| (f64::from(*v) - max).exp()).collect();
    let sum: f64 = out.iter().sum();
    for v in &mut out {
        *v /= sum;
    }
    Ok(out.into_iter().map(|v| v as f32).collect())
}

/// Scalar grouped-query decode attention over flat buffers; the
/// masking rule mirrors the `attention` crate (causal everywhere,
/// plus the sliding-window interval `(p - window, p]`), and scores
/// and accumulations run in f64. The `attention` crate's own
/// implementation works over its KV cache view instead; this is the
/// stateless backend-op form of the same math.
#[allow(clippy::too_many_arguments)]
fn attention_scalar(
    q: &[f32],
    keys: &[f32],
    values: &[f32],
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    query_position: usize,
    window: Option<usize>,
    scale: Option<f32>,
) -> Result<Vec<f32>> {
    let op = "attention";
    if num_q_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
        return Err(BackendError::Math(MathError::InvalidConfig {
            op,
            param: "heads",
            reason: "num_q_heads, num_kv_heads, and head_dim must all be positive",
        }));
    }
    if num_q_heads % num_kv_heads != 0 {
        return Err(BackendError::Math(MathError::InvalidConfig {
            op,
            param: "num_kv_heads",
            reason: "query heads must divide evenly across KV heads",
        }));
    }
    if let Some(w) = window {
        if w == 0 {
            return Err(BackendError::Math(MathError::InvalidConfig {
                op,
                param: "window",
                reason: "window must be positive when set",
            }));
        }
    }
    if let Some(s) = scale {
        if !s.is_finite() || s <= 0.0 {
            return Err(BackendError::Math(MathError::InvalidConfig {
                op,
                param: "scale",
                reason: "scale must be finite and > 0",
            }));
        }
    }
    if values.len() != keys.len() {
        return Err(BackendError::Math(MathError::LengthMismatch {
            op,
            expected: keys.len(),
            actual: values.len(),
        }));
    }
    let kv_width = checked_len(op, num_kv_heads, head_dim)?;
    if keys.len() % kv_width != 0 {
        return Err(BackendError::Math(MathError::InvalidConfig {
            op,
            param: "keys",
            reason: "key buffer is not a whole number of per-position KV rows",
        }));
    }
    let seq = keys.len() / kv_width;
    if seq == 0 {
        return Err(BackendError::Math(MathError::EmptyInput { op }));
    }
    let expected_q = checked_len(op, num_q_heads, head_dim)?;
    if q.len() != expected_q {
        return Err(BackendError::Math(MathError::ShapeMismatch {
            op,
            expected: expected_q,
            actual: q.len(),
        }));
    }
    let group = num_q_heads / num_kv_heads;
    let scale64 = f64::from(scale.unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt()));
    let p = query_position as i64;
    let visible: Vec<usize> = (0..seq)
        .filter(|&kp| {
            let kp = kp as i64;
            kp <= p && window.is_none_or(|w| kp > p - (w as i64))
        })
        .collect();
    if visible.is_empty() {
        return Err(BackendError::Math(MathError::InvalidConfig {
            op,
            param: "query_position",
            reason: "no key is visible to this query position under the causal/window mask",
        }));
    }

    let mut out = vec![0.0f64; expected_q];
    let mut scores: Vec<f64> = Vec::with_capacity(visible.len());
    for h in 0..num_q_heads {
        let kv_head = h / group;
        let q_off = h * head_dim;
        let qh = &q[q_off..q_off + head_dim];
        scores.clear();
        let mut max = f64::NEG_INFINITY;
        for &kp in &visible {
            let k_off = (kp * num_kv_heads + kv_head) * head_dim;
            let k = &keys[k_off..k_off + head_dim];
            let s: f64 = qh
                .iter()
                .zip(k.iter())
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum::<f64>()
                * scale64;
            if s > max {
                max = s;
            }
            scores.push(s);
        }
        let mut sum = 0.0f64;
        for s in &mut scores {
            *s = (*s - max).exp();
            sum += *s;
        }
        for (i, &kp) in visible.iter().enumerate() {
            let w = scores[i] / sum;
            let v_off = (kp * num_kv_heads + kv_head) * head_dim;
            let v = &values[v_off..v_off + head_dim];
            for d in 0..head_dim {
                out[q_off + d] += w * f64::from(v[d]);
            }
        }
    }
    Ok(out.into_iter().map(|v| v as f32).collect())
}

/// Elementwise binary op over equal-length slices, scalar.
fn elementwise(op: &'static str, a: &[f32], b: &[f32], f: fn(f32, f32) -> f32) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(BackendError::Math(MathError::LengthMismatch {
            op,
            expected: a.len(),
            actual: b.len(),
        }));
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| f(*x, *y)).collect())
}

impl Backend for CpuBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn name(&self) -> &str {
        "cpu-scalar-reference"
    }

    fn capabilities(&self) -> CapabilityReport {
        let dense = |op: Op| OpSupport {
            op,
            status: OpStatus::Supported {
                impl_kind: ImplKind::Scalar,
                quant_types: Vec::new(),
            },
        };
        let quantized = |op: Op| OpSupport {
            op,
            status: OpStatus::Supported {
                impl_kind: ImplKind::Scalar,
                quant_types: ALL_GGML_TYPES.to_vec(),
            },
        };
        CapabilityReport {
            backend: BackendKind::Cpu,
            ops: vec![
                dense(Op::Matmul),
                dense(Op::Matvec),
                quantized(Op::MatmulQuantized),
                quantized(Op::MatvecQuantized),
                quantized(Op::Dequantize),
                dense(Op::RmsNorm),
                dense(Op::RmsNormAdd),
                dense(Op::Rope),
                dense(Op::Softmax),
                dense(Op::Attention),
                dense(Op::SwiGlu),
                dense(Op::GeGlu),
                dense(Op::Embedding),
                dense(Op::Add),
                dense(Op::Mul),
                dense(Op::LogitSoftcap),
                dense(Op::LogitScale),
            ],
        }
    }

    fn matmul(
        &self,
        a: &[f32],
        a_rows: usize,
        a_cols: usize,
        b: &[f32],
        b_rows: usize,
        b_cols: usize,
    ) -> Result<Vec<f32>> {
        diag::debug(
            "op",
            &format!("name=matmul a={a_rows}x{a_cols} b={b_rows}x{b_cols}"),
        );
        Ok(torch_math::matmul(a, a_rows, a_cols, b, b_rows, b_cols)?)
    }

    fn matvec(&self, matrix: &[f32], rows: usize, cols: usize, vector: &[f32]) -> Result<Vec<f32>> {
        diag::debug("op", &format!("name=matvec rows={rows} cols={cols}"));
        Ok(torch_math::matvec(matrix, rows, cols, vector)?)
    }

    fn matvec_quantized(
        &self,
        ty: GgmlType,
        data: &[u8],
        rows: usize,
        cols: usize,
        vector: &[f32],
    ) -> Result<Vec<f32>> {
        diag::debug(
            "op",
            &format!("name=matvec_quantized ty={ty:?} rows={rows} cols={cols}"),
        );
        let elements = checked_len("matvec_quantized", rows, cols)?;
        check_payload("matvec_quantized", ty, data, elements)?;
        let dense = gguf::dequant::dequantize(ty, data, elements);
        Ok(torch_math::matvec(&dense, rows, cols, vector)?)
    }

    fn matmul_quantized(
        &self,
        ty: GgmlType,
        data: &[u8],
        rows: usize,
        cols: usize,
        input: &[f32],
        input_cols: usize,
    ) -> Result<Vec<f32>> {
        diag::debug(
            "op",
            &format!(
                "name=matmul_quantized ty={ty:?} rows={rows} cols={cols} input_cols={input_cols}"
            ),
        );
        let elements = checked_len("matmul_quantized", rows, cols)?;
        check_payload("matmul_quantized", ty, data, elements)?;
        let dense = gguf::dequant::dequantize(ty, data, elements);
        Ok(torch_math::matmul(
            &dense, rows, cols, input, cols, input_cols,
        )?)
    }

    fn dequantize(&self, ty: GgmlType, data: &[u8], elements: usize) -> Result<Vec<f32>> {
        diag::debug(
            "op",
            &format!("name=dequantize ty={ty:?} elements={elements}"),
        );
        check_payload("dequantize", ty, data, elements)?;
        Ok(gguf::dequant::dequantize(ty, data, elements))
    }

    fn rms_norm(&self, input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
        diag::debug(
            "op",
            &format!("name=rms_norm len={} eps={eps}", input.len()),
        );
        Ok(torch_math::rms_norm(input, weight, eps)?)
    }

    fn rms_norm_add(
        &self,
        input: &[f32],
        residual: &[f32],
        weight: &[f32],
        eps: f32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        diag::debug(
            "op",
            &format!("name=rms_norm_add len={} eps={eps}", input.len()),
        );
        let summed = elementwise("rms_norm_add", input, residual, |a, b| a + b)?;
        let normed = torch_math::rms_norm(&summed, weight, eps)?;
        Ok((normed, summed))
    }

    fn rope(
        &self,
        input: &[f32],
        position: usize,
        base: f32,
        head_dim: usize,
        rotary_dim: usize,
    ) -> Result<Vec<f32>> {
        diag::debug(
            "op",
            &format!("name=rope position={position} base={base} head_dim={head_dim} rotary_dim={rotary_dim}"),
        );
        Ok(torch_math::rope_partial(
            input, position, base, head_dim, rotary_dim,
        )?)
    }

    fn softmax(&self, input: &[f32]) -> Result<Vec<f32>> {
        diag::debug("op", &format!("name=softmax len={}", input.len()));
        softmax_scalar(input)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        q: &[f32],
        keys: &[f32],
        values: &[f32],
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        query_position: usize,
        window: Option<usize>,
        scale: Option<f32>,
    ) -> Result<Vec<f32>> {
        diag::debug(
            "op",
            &format!(
                "name=attention q_heads={num_q_heads} kv_heads={num_kv_heads} head_dim={head_dim} position={query_position} window={window:?}"
            ),
        );
        attention_scalar(
            q,
            keys,
            values,
            num_q_heads,
            num_kv_heads,
            head_dim,
            query_position,
            window,
            scale,
        )
    }

    fn swiglu(&self, gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
        diag::debug("op", &format!("name=swiglu len={}", gate.len()));
        Ok(torch_math::swiglu(gate, up)?)
    }

    fn geglu(&self, gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
        diag::debug("op", &format!("name=geglu len={}", gate.len()));
        Ok(torch_math::geglu(gate, up)?)
    }

    fn embedding(&self, table: &[f32], rows: usize, cols: usize, token: u32) -> Result<Vec<f32>> {
        diag::debug(
            "op",
            &format!("name=embedding rows={rows} cols={cols} token={token}"),
        );
        let expected = checked_len("embedding", rows, cols)?;
        if table.len() != expected {
            return Err(BackendError::Math(MathError::ShapeMismatch {
                op: "embedding",
                expected,
                actual: table.len(),
            }));
        }
        let row = token as usize;
        if row >= rows {
            return Err(BackendError::IndexOutOfRange {
                op: "embedding",
                index: u64::from(token),
                len: rows as u64,
            });
        }
        Ok(table[row * cols..(row + 1) * cols].to_vec())
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        diag::debug("op", &format!("name=add len={}", a.len()));
        elementwise("add", a, b, |x, y| x + y)
    }

    fn mul(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        diag::debug("op", &format!("name=mul len={}", a.len()));
        elementwise("mul", a, b, |x, y| x * y)
    }

    fn logit_softcap(&self, logits: &[f32], cap: f32) -> Result<Vec<f32>> {
        diag::debug(
            "op",
            &format!("name=logit_softcap len={} cap={cap}", logits.len()),
        );
        Ok(torch_math::logit_softcap(logits, cap)?)
    }

    fn logit_scale(&self, logits: &[f32], scale: f32) -> Result<Vec<f32>> {
        diag::debug(
            "op",
            &format!("name=logit_scale len={} scale={scale}", logits.len()),
        );
        Ok(torch_math::logit_scale(logits, scale)?)
    }
}
