# muse-glimmer

The Muse Glimmer architecture module for the iTaK Torch Rust engine
(TODO T-P.3). One crate, one architecture: the 52-layer hybrid measured
from the real Muse-Glimmer-30B GGUF on 2026-10-29 (39 sliding-window
layers, window 2048; 13 global layers at indices 3, 7, ..., 51; hidden
6656; FFN 19968; 32 Q heads over 2 KV heads; head dim 128; RoPE base
500000; logit softcap 20.0, scale 0.19611613; untied Q5_K head; vocab
202048). Dissection: `~/workspace/glimmer-blueprint/GLIMMER-DISSECTION.md`.

## Status: wired and serving

This crate is the architecture the serving engine runs. Its trait
seams are implemented by the sibling crates, and the full forward
generates on the real file end to end (integration record:
crates/itak-loader/INTEGRATION-RUN-B.md and the serving proofs in
bench/ at the repo root).

Real today:

- `GlimmerConfig` parsed from GGUF metadata through a trait seam, using
  the file's own `muse-glimmer.*` keys. Missing keys are named errors;
  nothing is defaulted silently. Metadata magnitudes are bounded at
  parse: the sliding window may not exceed the context length, the
  block count is capped (MAX_BLOCK_COUNT) and may not exceed the
  context length, and the head width products are checked, with
  refusals named as `GlimmerError::InvalidMetadata`. The layer pattern
  comes from metadata when a file carries it, else from the measured
  39/13 pattern, with the origin recorded (`PatternSource`).
- `Weights`: the full measured tensor inventory, 14 tensors per layer
  (including attn_q_norm, attn_k_norm, attn_gate, post_attention_norm,
  post_ffw_norm, which the old engine's Llama-shaped template never
  loaded), behind the `TensorSource` trait. Missing tensor: named error.
  The untied output head is never replaced by the embedding table.
- `GlimmerModel::forward`: input validation, the complete per-layer
  loop (norm, Q/K/V, QK norms, attention by layer kind, gate, output,
  residual, post norm, norm, SwiGLU, residual, post norm), final norm,
  head, and the real softcap-and-scale logit finishing (scale applied
  first, then the tanh softcap, matching the reference).
- Diagnostics trait with every refusal emitted; errors are typed and
  named (`GlimmerError`).

Trait implementations, landed in the sibling crates:

- Metadata and tensors: the `gguf` loader crate, bridged by
  `itak-loader`, implements `Metadata` and `TensorSource` over the
  mapped file, behind the tensor coverage gate.
- Arithmetic: `cpu-compute` implements `Compute` (the serving
  backend), and `wgpu-compute` implements it for the GPU bench path.
- KV storage: the `attention` crate implements `Cache` with ring
  buffers for sliding layers and full caches for global layers, with
  checked sizing at construction.

## Gates

`cargo build`, `cargo test`, `cargo clippy` clean, per repo rules.
Tests include the real 52-layer pattern (exactly 13 globals at the
measured indices), config parsing from the real file's measured
metadata values, missing-key and missing-tensor named errors, the
metadata bound refusals, and a full structural walk of the forward.
