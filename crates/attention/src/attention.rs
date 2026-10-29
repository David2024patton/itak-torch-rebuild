//! Grouped-query attention over a [`CacheView`]: the f32 reference
//! implementation. Clarity first; the specialized kernels that come
//! later must match this one's outputs.
//!
//! RoPE is the caller's job: this module takes pre-rotated queries
//! and keys. What it owns is the masking (causal everywhere, plus the
//! sliding-window mask on windowed layers), the GQA head sharing, and
//! the softmax.

use crate::cache::CacheView;
use crate::error::AttentionError;

/// Attention knobs. The score scale defaults to `1 / sqrt(head_dim)`
/// (standard scaled dot-product attention) and can be pinned
/// explicitly; the original Go engine hardcoded its equivalent, here
/// it is a parameter like everything else measured to matter.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AttentionParams {
    /// Explicit score scale; `None` means `1 / sqrt(head_dim)`.
    pub scale: Option<f32>,
}

impl AttentionParams {
    /// Default parameters: scale is `1 / sqrt(head_dim)`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Parameters with an explicit score scale.
    pub fn with_scale(scale: f32) -> Self {
        Self { scale: Some(scale) }
    }

    /// The scale in effect for a given head dimension.
    pub fn scale_for(&self, head_dim: usize) -> f32 {
        self.scale.unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt())
    }
}

/// Which KV head a query head reads from. Query heads are grouped
/// contiguously: with 32 query heads and 2 KV heads (Glimmer), heads
/// 0..16 map to KV head 0 and heads 16..32 to KV head 1.
pub fn kv_head_for_query_head(
    q_head: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
) -> Result<usize, AttentionError> {
    if num_kv_heads == 0 {
        return Err(AttentionError::ZeroDimension {
            what: "num_kv_heads",
        });
    }
    if num_q_heads == 0 {
        return Err(AttentionError::ZeroDimension {
            what: "num_q_heads",
        });
    }
    if num_q_heads % num_kv_heads != 0 {
        return Err(AttentionError::NonDivisibleHeads {
            num_q_heads,
            num_kv_heads,
        });
    }
    if q_head >= num_q_heads {
        return Err(AttentionError::ShapeMismatch {
            what: "q_head",
            expected: num_q_heads,
            actual: q_head,
        });
    }
    Ok(q_head / (num_q_heads / num_kv_heads))
}

/// The mask: a key at position `kp` is visible to a query at position
/// `p` when it is not in the future (causal mask) and, on windowed
/// layers, when it lies in `(p - window, p]`.
fn attends(kp: usize, p: usize, window: Option<usize>) -> bool {
    if kp > p {
        return false;
    }
    match window {
        None => true,
        Some(w) => (kp as i64) > (p as i64) - (w as i64),
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Run grouped-query attention for one or more query positions
/// against one layer's cache view.
///
/// * `q`: flat `[num_queries][num_q_heads][head_dim]` queries,
///   already RoPE-rotated by the caller.
/// * `query_positions`: the absolute position of each query. A query
///   at position `p` attends to keys at positions `<= p` (causal
///   mask) and, on windowed layers, only to positions in
///   `(p - window, p]`.
/// * Each query head reads the KV head given by
///   [`kv_head_for_query_head`].
///
/// Returns flat `[num_queries][num_q_heads][head_dim]` outputs.
pub fn gqa_attention(
    q: &[f32],
    num_queries: usize,
    num_q_heads: usize,
    query_positions: &[usize],
    view: &CacheView<'_>,
    params: &AttentionParams,
) -> Result<Vec<f32>, AttentionError> {
    let head_dim = view.head_dim();
    let num_kv_heads = view.num_kv_heads();

    if query_positions.len() != num_queries {
        return Err(AttentionError::ShapeMismatch {
            what: "query_positions",
            expected: num_queries,
            actual: query_positions.len(),
        });
    }
    let expected_q = num_queries * num_q_heads * head_dim;
    if q.len() != expected_q {
        return Err(AttentionError::ShapeMismatch {
            what: "q",
            expected: expected_q,
            actual: q.len(),
        });
    }
    if num_queries == 0 {
        return Ok(Vec::new());
    }
    if num_q_heads == 0 {
        return Err(AttentionError::ZeroDimension {
            what: "num_q_heads",
        });
    }
    if num_q_heads % num_kv_heads != 0 {
        return Err(AttentionError::NonDivisibleHeads {
            num_q_heads,
            num_kv_heads,
        });
    }
    let group = num_q_heads / num_kv_heads;
    let scale = params.scale_for(head_dim);
    let window = view.window();
    crate::diag!(
        "gqa_attention queries={} q_heads={} kv_heads={} view_len={} window={:?} scale={}",
        num_queries,
        num_q_heads,
        num_kv_heads,
        view.len(),
        window,
        scale
    );

    let mut out = vec![0.0; q.len()];
    let mut scores: Vec<f32> = Vec::with_capacity(view.len());

    for (qi, &p) in query_positions.iter().enumerate() {
        // The mask depends only on the query position, so resolve the
        // visible entries once per query, not once per head.
        let mut visible: Vec<usize> = Vec::new();
        for vi in 0..view.len() {
            if let Some(kp) = view.position_at(vi) {
                if attends(kp, p, window) {
                    visible.push(vi);
                }
            }
        }
        if visible.is_empty() {
            return Err(AttentionError::NoVisibleKeys { position: p });
        }

        for h in 0..num_q_heads {
            let kv_head = h / group;
            let q_off = (qi * num_q_heads + h) * head_dim;
            let qh = &q[q_off..q_off + head_dim];

            scores.clear();
            let mut max = f32::NEG_INFINITY;
            for &vi in &visible {
                // Invariant: vi < view.len() and kv_head <
                // num_kv_heads, so the view always has this key.
                let k = view
                    .key(vi, kv_head)
                    .expect("visible index and kv head are in range by construction");
                let s = dot(qh, k) * scale;
                if s > max {
                    max = s;
                }
                scores.push(s);
            }

            // Softmax in place, max-subtracted for stability.
            let mut sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - max).exp();
                sum += *s;
            }

            let o_off = (qi * num_q_heads + h) * head_dim;
            for (i, &vi) in visible.iter().enumerate() {
                let w = scores[i] / sum;
                let v = view
                    .value(vi, kv_head)
                    .expect("visible index and kv head are in range by construction");
                for d in 0..head_dim {
                    out[o_off + d] += w * v[d];
                }
            }
        }
    }
    Ok(out)
}
