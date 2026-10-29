# muse-glimmer

The Muse Glimmer architecture module for the iTaK Torch Rust engine
(TODO T-P.3). One crate, one architecture: the 52-layer hybrid measured
from the real Muse-Glimmer-30B GGUF on 2026-10-29 (39 sliding-window
layers, window 2048; 13 global layers at indices 3, 7, ..., 51; hidden
6656; FFN 19968; 32 Q heads over 2 KV heads; head dim 128; RoPE base
500000; logit softcap 20.0, scale 0.19611613; untied Q5_K head; vocab
202048). Dissection: `~/workspace/glimmer-blueprint/GLIMMER-DISSECTION.md`.

## Status: skeleton

Real today:

- `GlimmerConfig` parsed from GGUF metadata through a trait seam, using
  the file's own `muse-glimmer.*` keys. Missing keys are named errors;
  nothing is defaulted silently. The layer pattern comes from metadata
  when a file carries it, else from the measured 39/13 pattern, with the
  origin recorded (`PatternSource`).
- `Weights`: the full measured tensor inventory, 14 tensors per layer
  (including attn_q_norm, attn_k_norm, attn_gate, post_attention_norm,
  post_ffw_norm, which the old engine's Llama-shaped template never
  loaded), behind the `TensorSource` trait. Missing tensor: named error.
  The untied output head is never replaced by the embedding table.
- `GlimmerModel::forward`: input validation, the complete per-layer
  loop (norm, Q/K/V, QK norms, attention by layer kind, gate, output,
  residual, post norm, norm, SwiGLU, residual, post norm), final norm,
  head, and the real softcap-and-scale logit finishing.
- Diagnostics trait with every refusal emitted; errors are typed and
  named (`GlimmerError`).

Trait-bound, landing with sibling crates:

- Metadata and tensors: the `gguf` loader crate (T-2.1) implements
  `Metadata` and `TensorSource` over the mapped file.
- Arithmetic: the kernels crate (T-2.2) implements `Compute`. Until
  then `UnimplementedCompute` refuses every step by name, and the tests
  walk the forward's structure against a recording stub.
- KV storage: the KV crate (T-2.4) implements `Cache` with ring buffers
  for sliding layers and full caches for global layers. `PositionCache`
  here tracks position only.

## Gates

`cargo build`, `cargo test`, `cargo clippy` clean, per repo rules.
Tests include the real 52-layer pattern (exactly 13 globals at the
measured indices), config parsing from the real file's measured
metadata values, missing-key and missing-tensor named errors, and a
full structural walk of the forward.
