//! Shared fixtures for this crate's tests: metadata maps carrying the
//! measured real-file values and a tiny synthetic model shape. Test code
//! only; nothing here ships in the library's API surface.

use crate::config::*;
use crate::metadata::MapMetadata;

/// Metadata exactly as measured in the real Muse-Glimmer-30B GGUF
/// (census 2026-10-29). Building the config from these values is the
/// closest a unit test gets to the real file without a 15 GiB fixture.
pub fn real_glimmer_metadata() -> MapMetadata {
    let mut m = MapMetadata::new();
    m.insert_string(KEY_ARCHITECTURE, "muse-glimmer");
    m.insert_string(KEY_NAME, "Muse Glimmer Hf");
    m.insert_u32(KEY_BLOCK_COUNT, 52);
    m.insert_u32(KEY_CONTEXT_LENGTH, 131_072);
    m.insert_u32(KEY_HIDDEN, 6656);
    m.insert_u32(KEY_FFN_HIDDEN, 19_968);
    m.insert_u32(KEY_Q_HEADS, 32);
    m.insert_u32(KEY_KV_HEADS, 2);
    m.insert_u32(KEY_KEY_LENGTH, 128);
    m.insert_u32(KEY_VALUE_LENGTH, 128);
    m.insert_f32(KEY_ROPE_FREQ_BASE, 500_000.0);
    m.insert_f32(KEY_RMS_EPS, 1e-5);
    m.insert_u32(KEY_SLIDING_WINDOW, 2048);
    m.insert_f32(KEY_LOGIT_SOFTCAP, 20.0);
    m.insert_f32(KEY_LOGIT_SCALE, 0.196_116_13);
    m.insert_array_len(KEY_VOCAB_TOKENS, 202_048);
    m
}

/// A two-layer synthetic model: hidden 32, FFN 64, 4 Q heads over 2 KV
/// heads, head dim 8, vocab 128. Small enough that tests can build its
/// full weight set from dummy tensors and walk the whole forward shape.
pub fn tiny_metadata() -> MapMetadata {
    let mut m = MapMetadata::new();
    m.insert_string(KEY_ARCHITECTURE, "muse-glimmer");
    m.insert_u32(KEY_BLOCK_COUNT, 2);
    m.insert_u32(KEY_CONTEXT_LENGTH, 128);
    m.insert_u32(KEY_HIDDEN, 32);
    m.insert_u32(KEY_FFN_HIDDEN, 64);
    m.insert_u32(KEY_Q_HEADS, 4);
    m.insert_u32(KEY_KV_HEADS, 2);
    m.insert_u32(KEY_KEY_LENGTH, 8);
    m.insert_u32(KEY_VALUE_LENGTH, 8);
    m.insert_f32(KEY_ROPE_FREQ_BASE, 10_000.0);
    m.insert_f32(KEY_RMS_EPS, 1e-6);
    m.insert_u32(KEY_SLIDING_WINDOW, 16);
    m.insert_f32(KEY_LOGIT_SOFTCAP, 20.0);
    m.insert_f32(KEY_LOGIT_SCALE, 1.0);
    m.insert_array_len(KEY_VOCAB_TOKENS, 128);
    m
}
