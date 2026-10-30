//! The `Backend` trait: the one interface every hardware backend
//! implements (ARCHITECTURE 3.1, module `itak-backend`).
//!
//! The surface is the kernel set of BACKEND.md Section 3 expressed
//! as pure functions over flat, row-major tensor buffers, in the
//! conventions the workspace's math crates already use: shapes are
//! explicit parameters validated before any arithmetic, misuse is a
//! named [`BackendError`] never a panic, and results are freshly
//! allocated vectors (the reference style of `torch-math`; the GPU
//! backend's residency model is BACKEND.md Section 5 and does not
//! change this surface, only where the buffers live).
//!
//! Every method has a corresponding [`crate::Op`] so the capability
//! report and the load-time fallback plan can name it.

use gguf::GgmlType;

use crate::error::BackendError;
use crate::report::{BackendKind, CapabilityReport};

/// Result type for backend operations.
pub type Result<T> = std::result::Result<T, BackendError>;

/// A hardware backend behind one trait: CPU (always available) or
/// wgpu on Vulkan (TODO T-4.2). Implementations must report their
/// capabilities truthfully through [`Backend::capabilities`]; the
/// load-time per-op fallback plan is computed from that report, so
/// an op claimed there and refused here is a contract violation.
pub trait Backend: Send + Sync {
    /// Which backend this is.
    fn kind(&self) -> BackendKind;

    /// Stable human-readable name, for reports and the diag channel.
    fn name(&self) -> &str;

    /// What this backend can execute, and how. See
    /// [`CapabilityReport`].
    fn capabilities(&self) -> CapabilityReport;

    /// Dense matrix-matrix product: `a` (`a_rows x a_cols`) times
    /// `b` (`b_rows x b_cols`), row-major; the prefill shape.
    ///
    /// # Errors
    /// Shape and dimension mismatches, as [`BackendError::Math`].
    fn matmul(
        &self,
        a: &[f32],
        a_rows: usize,
        a_cols: usize,
        b: &[f32],
        b_rows: usize,
        b_cols: usize,
    ) -> Result<Vec<f32>>;

    /// Dense matrix-vector product: `matrix` (`rows x cols`,
    /// row-major) times `vector` (length `cols`); the decode shape.
    ///
    /// # Errors
    /// Shape and length mismatches, as [`BackendError::Math`].
    fn matvec(&self, matrix: &[f32], rows: usize, cols: usize, vector: &[f32]) -> Result<Vec<f32>>;

    /// Quantized matrix-vector product: `data` holds a `rows x cols`
    /// matrix in element type `ty` (BACKEND.md kernels K1 to K3);
    /// `vector` has length `cols`.
    ///
    /// # Errors
    /// [`BackendError::ByteLengthMismatch`] or
    /// [`BackendError::BadElementCount`] if the payload disagrees
    /// with `ty` and `rows * cols`; shape errors as
    /// [`BackendError::Math`].
    fn matvec_quantized(
        &self,
        ty: GgmlType,
        data: &[u8],
        rows: usize,
        cols: usize,
        vector: &[f32],
    ) -> Result<Vec<f32>>;

    /// Quantized matrix-matrix product: `data` holds a
    /// `rows x cols` matrix in element type `ty`; `input` is a dense
    /// `cols x input_cols` matrix, row-major (the prefill GEMM).
    ///
    /// # Errors
    /// As [`Backend::matvec_quantized`], plus shape errors as
    /// [`BackendError::Math`].
    fn matmul_quantized(
        &self,
        ty: GgmlType,
        data: &[u8],
        rows: usize,
        cols: usize,
        input: &[f32],
        input_cols: usize,
    ) -> Result<Vec<f32>>;

    /// Materialize `elements` elements of type `ty` from their
    /// on-disk bytes as f32.
    ///
    /// # Errors
    /// [`BackendError::ByteLengthMismatch`] or
    /// [`BackendError::BadElementCount`] on a payload that disagrees
    /// with `ty` and `elements`.
    fn dequantize(&self, ty: GgmlType, data: &[u8], elements: usize) -> Result<Vec<f32>>;

    /// RMSNorm: `input / sqrt(mean(input^2) + eps) * weight`,
    /// elementwise. `eps` comes from the arch module, never from a
    /// constant in the backend.
    ///
    /// # Errors
    /// Empty input, length mismatch, or bad `eps`, as
    /// [`BackendError::Math`].
    fn rms_norm(&self, input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>>;

    /// RMSNorm fused with the residual add (BACKEND.md kernel K4):
    /// sums `input` and `residual`, then RMS-normalizes the sum.
    /// Returns `(normed, new_residual)`; the caller keeps
    /// `new_residual` as the residual stream, which is the point of
    /// the fusion.
    ///
    /// # Errors
    /// Length mismatches or bad `eps`, as [`BackendError::Math`].
    fn rms_norm_add(
        &self,
        input: &[f32],
        residual: &[f32],
        weight: &[f32],
        eps: f32,
    ) -> Result<(Vec<f32>, Vec<f32>)>;

    /// RoPE on one head vector: rotate the first `rotary_dim`
    /// channels of a `head_dim` vector at `position`, with the arch
    /// module's `base`.
    ///
    /// # Errors
    /// Length and configuration errors, as [`BackendError::Math`].
    fn rope(
        &self,
        input: &[f32],
        position: usize,
        base: f32,
        head_dim: usize,
        rotary_dim: usize,
    ) -> Result<Vec<f32>>;

    /// Softmax over one vector, numerically stabilized.
    ///
    /// # Errors
    /// [`BackendError::Math`] on empty input.
    fn softmax(&self, input: &[f32]) -> Result<Vec<f32>>;

    /// Grouped-query decode attention for one query position:
    /// `q` is `[num_q_heads][head_dim]`; `keys` and `values` are
    /// `[seq][num_kv_heads][head_dim]` holding positions
    /// `0..seq`. The query at `query_position` attends causally and,
    /// when `window` is set, only to positions in
    /// `(query_position - window, query_position]` (the
    /// sliding-window mask of BACKEND.md kernel K6). `scale` pins
    /// the score scale; `None` means `1 / sqrt(head_dim)`.
    ///
    /// # Errors
    /// Shape, divisibility, and visibility errors, as
    /// [`BackendError::Math`].
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
    ) -> Result<Vec<f32>>;

    /// SwiGLU: `silu(gate) * up`, elementwise (kernel K7).
    ///
    /// # Errors
    /// Length mismatch, as [`BackendError::Math`].
    fn swiglu(&self, gate: &[f32], up: &[f32]) -> Result<Vec<f32>>;

    /// GEGLU: `gelu_tanh(gate) * up`, elementwise (the Gemma-family
    /// variant of K7).
    ///
    /// # Errors
    /// Length mismatch, as [`BackendError::Math`].
    fn geglu(&self, gate: &[f32], up: &[f32]) -> Result<Vec<f32>>;

    /// Embedding gather: copy row `token` of a `rows x cols` table
    /// (kernel K8's gather half).
    ///
    /// # Errors
    /// [`BackendError::IndexOutOfRange`] if `token >= rows`; shape
    /// errors as [`BackendError::Math`].
    fn embedding(&self, table: &[f32], rows: usize, cols: usize, token: u32) -> Result<Vec<f32>>;

    /// Elementwise add.
    ///
    /// # Errors
    /// Length mismatch, as [`BackendError::Math`].
    fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>>;

    /// Elementwise multiply.
    ///
    /// # Errors
    /// Length mismatch, as [`BackendError::Math`].
    fn mul(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>>;

    /// Logit softcap: `cap * tanh(x / cap)` elementwise (kernel
    /// K8's shaping half; `cap` from the arch module).
    ///
    /// # Errors
    /// Bad `cap`, as [`BackendError::Math`].
    fn logit_softcap(&self, logits: &[f32], cap: f32) -> Result<Vec<f32>>;

    /// Final logit scale: multiply every logit by `scale` (kernel
    /// K8's shaping half; `scale` from the arch module).
    ///
    /// # Errors
    /// Non-finite `scale`, as [`BackendError::Math`].
    fn logit_scale(&self, logits: &[f32], scale: f32) -> Result<Vec<f32>>;
}
