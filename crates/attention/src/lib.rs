//! attention: the KV cache and grouped-query attention for the iTaK
//! Torch rebuild's Rust token loop.
//!
//! Shaped by Muse Glimmer's hybrid design (52 layers: 39
//! sliding-window with window 2048, 13 global; 32 query heads sharing
//! 2 KV heads; head dim 128). This crate is the wired-by-construction
//! replacement for the original Go engine's sliding_window.go,
//! paged_attention.go, and kv_cache.go, which no code path ever
//! called: here the cache is the only way attention can obtain keys
//! and values, so the wiring cannot be skipped.
//!
//! Everything is f32 reference math: clarity first, no unsafe, named
//! errors instead of panics on shape mismatches, and debug logging
//! behind a flag (see [`diag`]).

#![forbid(unsafe_code)]

pub mod cache;
pub mod diag;
pub mod error;

pub use cache::{glimmer_layer_pattern, CacheView, KvCache, LayerKind, LayerStats};
pub use error::AttentionError;
