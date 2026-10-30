# torch-math

Reference `f32` math primitives for the iTaK Torch engine, one module per
math family:

| Module | Owns |
| :--- | :--- |
| `norm` | `rms_norm` with `eps` as a parameter (never a constant) |
| `rope` | `rope` / `rope_partial`, configurable base (`500000` for Glimmer), llama.cpp pairing and partial-rotary conventions |
| `softmax` | `softmax` over one row: max-subtracted for stability, `-inf` (mask) entries get exactly `0.0`, an all-masked row is a named error |
| `activation` | `silu`, `gelu_tanh`, and the gated forms `swiglu` / `geglu` as separate functions |
| `logits` | `logit_softcap` and `logit_scale`, both parameterized (Glimmer: cap `20.0`, scale `0.1961`) |
| `linalg` | `matvec` and `matmul` over flat row-major buffers, shapes always validated |

Rules this crate lives by:

- Clarity over speed. SIMD and GPU kernels land later behind these same
  signatures and are verified against these implementations.
- Accumulation in `f64`, storage in `f32`, so the reference is the most
  accurate version of each formula.
- No panics on shape mismatch or bad config: everything fallible returns
  `Result` with a named `MathError`.
- No dependencies.
- Debug logging behind `set_debug(true)`: every public function logs its
  shapes and config to stderr under a `[torch-math]` prefix.
- Tests are hand-computed vectors (published constants, exact identities,
  and independent-route recomputation), never snapshots of the code's own
  output.
