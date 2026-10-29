//! Dense `f32` matrix math. All buffers are flat, row-major: element
//! `(i, j)` of an `rows x cols` matrix lives at `buf[i * cols + j]`.
//! Shapes travel as explicit `rows`/`cols` parameters and are always
//! validated against the buffer lengths before any arithmetic happens.
//!
//! Decode is matvec-bound (one token against the whole weight matrix),
//! so [`matvec`] is a first-class operation here, not a special case of
//! [`matmul`]. Dot products accumulate in `f64`, as everywhere in this
//! crate, so the reference is the accurate version of the formula.
//! SIMD and GPU kernels will replace the inner loops behind these same
//! signatures and be verified against these implementations.

use crate::{dbg_log, MathError, Result};

/// `rows * cols` with overflow reported as an error instead of a panic
/// (debug builds panic on arithmetic overflow; this crate never panics).
fn checked_len(op: &'static str, rows: usize, cols: usize) -> Result<usize> {
    rows.checked_mul(cols).ok_or(MathError::InvalidConfig {
        op,
        param: "shape",
        reason: "rows * cols overflows usize",
    })
}

/// Matrix-vector product: `out[i] = sum_j matrix[i, j] * vector[j]`,
/// where `matrix` is `rows x cols` in row-major order and `vector`
/// has length `cols`. The output has length `rows`.
///
/// With `cols == 0` the product is the zero vector of length `rows`
/// (an empty sum), provided `matrix` is empty and `vector` is empty.
///
/// # Errors
///
/// - [`MathError::ShapeMismatch`] if `matrix.len() != rows * cols`.
/// - [`MathError::LengthMismatch`] if `vector.len() != cols`.
/// - [`MathError::InvalidConfig`] if `rows * cols` overflows.
pub fn matvec(matrix: &[f32], rows: usize, cols: usize, vector: &[f32]) -> Result<Vec<f32>> {
    let expected = checked_len("matvec", rows, cols)?;
    if matrix.len() != expected {
        return Err(MathError::ShapeMismatch {
            op: "matvec",
            expected,
            actual: matrix.len(),
        });
    }
    if vector.len() != cols {
        return Err(MathError::LengthMismatch {
            op: "matvec",
            expected: cols,
            actual: vector.len(),
        });
    }
    dbg_log!("matvec rows={rows} cols={cols}");
    if cols == 0 {
        return Ok(vec![0.0; rows]);
    }
    Ok(matrix
        .chunks_exact(cols)
        .map(|row| {
            row.iter()
                .zip(vector.iter())
                .map(|(m, v)| f64::from(*m) * f64::from(*v))
                .sum::<f64>() as f32
        })
        .collect())
}

/// Matrix-matrix product: `out = a * b`, where `a` is `a_rows x a_cols`
/// and `b` is `b_rows x b_cols`, all row-major. The output is
/// `a_rows x b_cols` row-major.
///
/// # Errors
///
/// - [`MathError::DimensionMismatch`] if `a_cols != b_rows`.
/// - [`MathError::ShapeMismatch`] if either buffer's length disagrees
///   with its declared shape.
/// - [`MathError::InvalidConfig`] if any shape product overflows.
pub fn matmul(
    a: &[f32],
    a_rows: usize,
    a_cols: usize,
    b: &[f32],
    b_rows: usize,
    b_cols: usize,
) -> Result<Vec<f32>> {
    if a_cols != b_rows {
        return Err(MathError::DimensionMismatch {
            op: "matmul",
            a_cols,
            b_rows,
        });
    }
    let a_expected = checked_len("matmul", a_rows, a_cols)?;
    if a.len() != a_expected {
        return Err(MathError::ShapeMismatch {
            op: "matmul",
            expected: a_expected,
            actual: a.len(),
        });
    }
    let b_expected = checked_len("matmul", b_rows, b_cols)?;
    if b.len() != b_expected {
        return Err(MathError::ShapeMismatch {
            op: "matmul",
            expected: b_expected,
            actual: b.len(),
        });
    }
    let out_len = checked_len("matmul", a_rows, b_cols)?;
    dbg_log!("matmul a={a_rows}x{a_cols} b={b_rows}x{b_cols}");
    let mut out = vec![0.0f32; out_len];
    if a_cols == 0 || b_cols == 0 {
        // Degenerate shapes: the product is all zeros by definition
        // (empty inner sum), and chunks_exact(0) would panic.
        return Ok(out);
    }
    // Definition order, one output element at a time, each dot
    // product accumulated in f64: this is the reference the optimized
    // kernels are checked against, so it favors obviousness over
    // memory-access cleverness.
    for (i, a_row) in a.chunks_exact(a_cols).enumerate() {
        for j in 0..b_cols {
            let mut acc = 0.0f64;
            for (k, &a_ik) in a_row.iter().enumerate() {
                acc += f64::from(a_ik) * f64::from(b[k * b_cols + j]);
            }
            out[i * b_cols + j] = acc as f32;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::assert_vec_close;

    #[test]
    fn matvec_known_product() {
        // [[1, 2], [3, 4]] * [5, 6] = [1*5 + 2*6, 3*5 + 4*6] = [17, 39].
        let out = matvec(&[1.0, 2.0, 3.0, 4.0], 2, 2, &[5.0, 6.0]).unwrap();
        assert_vec_close(&out, &[17.0, 39.0], 1e-6);
    }

    #[test]
    fn matvec_rectangular() {
        // [[1, 2, 3], [4, 5, 6]] * [1, 0, -1] = [-2, -2].
        let out = matvec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3, &[1.0, 0.0, -1.0]).unwrap();
        assert_vec_close(&out, &[-2.0, -2.0], 1e-6);
    }

    #[test]
    fn matvec_zero_cols_is_zero_vector() {
        let out = matvec(&[], 3, 0, &[]).unwrap();
        assert_eq!(out, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn matvec_shape_errors() {
        // Buffer shorter than the declared shape.
        assert_eq!(
            matvec(&[1.0, 2.0, 3.0], 2, 2, &[1.0, 1.0]),
            Err(MathError::ShapeMismatch {
                op: "matvec",
                expected: 4,
                actual: 3
            })
        );
        // Vector longer than cols.
        assert_eq!(
            matvec(&[1.0, 2.0, 3.0, 4.0], 2, 2, &[1.0, 1.0, 1.0]),
            Err(MathError::LengthMismatch {
                op: "matvec",
                expected: 2,
                actual: 3
            })
        );
    }

    #[test]
    fn matmul_known_product() {
        // [[1,2,3],[4,5,6]] * [[7,8],[9,10],[11,12]]
        // = [[58, 64], [139, 154]] (hand-computed).
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let out = matmul(&a, 2, 3, &b, 3, 2).unwrap();
        assert_vec_close(&out, &[58.0, 64.0, 139.0, 154.0], 1e-5);
    }

    #[test]
    fn matmul_identity() {
        // A * I = A, with A = [[3, -1], [0.5, 2]].
        let a = [3.0, -1.0, 0.5, 2.0];
        let identity = [1.0, 0.0, 0.0, 1.0];
        let out = matmul(&a, 2, 2, &identity, 2, 2).unwrap();
        assert_vec_close(&out, &[3.0, -1.0, 0.5, 2.0], 1e-6);
    }

    #[test]
    fn matmul_agrees_with_matvec_on_a_column() {
        // The (single) column of A * [x] must equal matvec(A, x):
        // the two code paths are independent implementations of the
        // same product, so agreement cross-checks both.
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 x 3
        let x = [7.0, -8.0, 9.0];
        let via_mul = matmul(&a, 2, 3, &x, 3, 1).unwrap();
        let via_vec = matvec(&a, 2, 3, &x).unwrap();
        assert_eq!(via_mul, via_vec);
    }

    #[test]
    fn matmul_inner_dimension_mismatch_is_an_error() {
        assert_eq!(
            matmul(&[1.0; 6], 2, 3, &[1.0; 8], 4, 2),
            Err(MathError::DimensionMismatch {
                op: "matmul",
                a_cols: 3,
                b_rows: 4
            })
        );
    }

    #[test]
    fn matmul_shape_mismatch_is_an_error() {
        assert!(matches!(
            matmul(&[1.0; 5], 2, 3, &[1.0; 6], 3, 2),
            Err(MathError::ShapeMismatch {
                op: "matmul",
                expected: 6,
                actual: 5
            })
        ));
    }

    #[test]
    fn matmul_degenerate_shapes() {
        // 2x0 times 0x3 is a 2x3 zero matrix, not a panic.
        let out = matmul(&[], 2, 0, &[], 0, 3).unwrap();
        assert_eq!(out, vec![0.0; 6]);
    }
}
