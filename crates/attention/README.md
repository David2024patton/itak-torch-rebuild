# attention

KV cache and grouped-query attention for the iTaK Torch rebuild's
Rust token loop: one module in the workspace grid, owning exactly
the pieces the original Go engine left unwired (sliding_window.go,
paged_attention.go, kv_cache.go were never called by anything).

Shaped by Muse Glimmer's hybrid design: 52 layers, 39
sliding-window (window 2048) and 13 global, 32 query heads sharing
2 KV heads at head dim 128.

## What is here

| Piece | File | Job |
| :--- | :--- | :--- |
| `KvCache`, `CacheView`, `LayerKind` | `src/cache.rs` | Per-layer K/V storage. Global layers grow; windowed layers are ring buffers of exactly `window` slots (slot = position % window, oldest evicted). Layer kind is chosen per layer at construction from a pattern; `glimmer_layer_pattern()` returns Glimmer's 39/13 split. |
| `gqa_attention`, `AttentionParams` | `src/attention.rs` | f32 reference scaled dot-product attention over a cache view: causal mask everywhere, window mask `(p - window, p]` on windowed layers, contiguous GQA head groups. RoPE is applied by the caller; the score scale defaults to 1/sqrt(head_dim) and is a parameter. |
| `AttentionError` | `src/error.rs` | Named errors for every refusal. No panics on shape mismatches. |
| `diag` | `src/diag.rs` | Debug logging behind a runtime flag (`diag::set_debug_enabled`), stderr lines prefixed `[attention::diag]`. `KvCache::stats()` returns per-layer state programmatically. |

## Rules baked in

- Appends arrive in position order per layer and are validated;
  out-of-order or mis-shaped appends are named errors, never silent
  placeholder weights.
- Every module gate runs from birth: workspace lints deny warnings,
  clippy is clean, and `cargo test` covers ring eviction, the window
  mask, the causal mask, global visibility, and GQA head sharing at
  both the 32/2 Glimmer ratio and a smaller 4/2 ratio.
- f32 reference math, clarity first. Specialized kernels must match
  this crate's outputs; the cache allocates windowed storage once at
  construction.
