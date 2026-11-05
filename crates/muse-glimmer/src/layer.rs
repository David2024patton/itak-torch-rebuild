//! The per-layer attention pattern: which layers are sliding-window and
//! which are global.
//!
//! Measured from the real Muse Glimmer GGUF on 2026-10-29 (file census in
//! the model teardown volume; write-up in
//! `~/workspace/glimmer-blueprint/GLIMMER-DISSECTION.md`): 52 layers, 39
//! sliding-window and 13 global, the global layers at indices
//! 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, every fourth layer
//! starting at index 3.
//!
//! The file does carry the pattern, under
//! `muse-glimmer.attention.sliding_window_pattern`, as an array of 52
//! bools (header read 2026-11-05): `[true, true, true, false]` repeated
//! 13 times, `true` marking a sliding-window layer. That is the shape
//! llama.cpp reads first (`load_swa_pattern` in src/llama-model.cpp
//! reads the key as an array of `is_swa` flags before any fallback).
//! The 2026-10-29 census recorded the window size but not this key's
//! stored type, and the loader probed the key as a string instead, so
//! every load fell back to the measured pattern and the probe logged a
//! type mismatch. The file's array and the measured fallback agree
//! exactly (the fallback is llama.cpp's own period-4 pattern:
//! `set_swa_pattern(4, false)` in src/llama-hparams.cpp marks layer
//! `il` sliding iff `il % 4 < 3`), and the array now wins, as it should:
//! the file's own declaration over a measurement. [`PatternSource`]
//! records which one produced the pattern so the origin is never
//! silent.

use crate::config::{KEY_BLOCK_COUNT, MAX_BLOCK_COUNT};
use crate::error::{GlimmerError, Result};

/// How one layer attends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// Attention limited to the last `sliding_window` positions, served
    /// by a ring buffer in the KV layer.
    SlidingWindow,
    /// Attention over the full sequence.
    Global,
}

/// Where a layer pattern came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PatternSource {
    /// The file's own metadata carried an explicit per-layer pattern.
    Metadata,
    /// The file carried no pattern key; the pattern measured from the
    /// real Muse Glimmer file on 2026-10-29 was used. Recorded, never
    /// silent: a caller can see that this config rests on measurement
    /// rather than the file's own declaration.
    MeasuredFallback,
}

/// Metadata key for the file's explicit pattern: an array of bools,
/// one per layer in layer order, `true` for a sliding-window layer and
/// `false` for a global one. This is the shape the real file stores
/// (GGUF array of BOOL, 52 elements) and the shape llama.cpp reads
/// (`load_swa_pattern`'s array read into `is_swa_impl`).
pub const SWA_PATTERN_KEY: &str = "muse-glimmer.attention.sliding_window_pattern";

/// Derive the layer pattern.
///
/// A `block_count` above [`MAX_BLOCK_COUNT`] is refused before any
/// allocation: the pattern vector is sized from the count, so an
/// unbounded count from a corrupt or hostile file would abort the
/// process here instead of failing by name.
///
/// With `pattern` present (read from [`SWA_PATTERN_KEY`]), it must hold
/// exactly `block_count` elements, one per layer; a different length
/// is an [`GlimmerError::InvalidMetadata`] naming the key. Without it,
/// the measured fallback applies: every fourth layer starting at
/// index 3 is global, the rest sliding-window (the same pattern
/// llama.cpp's period-4 default produces for this architecture).
pub fn derive_layer_pattern(
    block_count: u32,
    pattern: Option<&[bool]>,
) -> Result<(Vec<LayerKind>, PatternSource)> {
    if block_count > MAX_BLOCK_COUNT {
        return Err(GlimmerError::InvalidMetadata {
            key: KEY_BLOCK_COUNT,
            detail: format!(
                "block_count {block_count} exceeds the plausible layer ceiling of {MAX_BLOCK_COUNT}"
            ),
        });
    }
    match pattern {
        Some(flags) => {
            if flags.len() != block_count as usize {
                return Err(GlimmerError::InvalidMetadata {
                    key: SWA_PATTERN_KEY,
                    detail: format!(
                        "pattern has {} layers but block_count is {block_count}",
                        flags.len()
                    ),
                });
            }
            let layers = flags
                .iter()
                .map(|&is_swa| {
                    if is_swa {
                        LayerKind::SlidingWindow
                    } else {
                        LayerKind::Global
                    }
                })
                .collect();
            Ok((layers, PatternSource::Metadata))
        }
        None => {
            let layers = (0..block_count)
                .map(|i| {
                    if i % 4 == 3 {
                        LayerKind::Global
                    } else {
                        LayerKind::SlidingWindow
                    }
                })
                .collect();
            Ok((layers, PatternSource::MeasuredFallback))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_52_layer_pattern_has_13_globals_at_measured_indices() {
        let (layers, source) = derive_layer_pattern(52, None).expect("fallback pattern");
        assert_eq!(source, PatternSource::MeasuredFallback);
        assert_eq!(layers.len(), 52);
        let globals: Vec<usize> = layers
            .iter()
            .enumerate()
            .filter(|(_, k)| **k == LayerKind::Global)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            globals,
            vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51]
        );
        let sliding = layers
            .iter()
            .filter(|k| **k == LayerKind::SlidingWindow)
            .count();
        assert_eq!(sliding, 39);
    }

    #[test]
    fn explicit_metadata_pattern_wins() {
        let (layers, source) =
            derive_layer_pattern(4, Some(&[true, false, true, false])).expect("pattern");
        assert_eq!(source, PatternSource::Metadata);
        assert_eq!(
            layers,
            vec![
                LayerKind::SlidingWindow,
                LayerKind::Global,
                LayerKind::SlidingWindow,
                LayerKind::Global
            ]
        );
    }

    #[test]
    fn pattern_length_must_match_block_count() {
        let err =
            derive_layer_pattern(52, Some(&[true, false])).expect_err("short pattern must fail");
        match err {
            GlimmerError::InvalidMetadata { key, .. } => assert_eq!(key, SWA_PATTERN_KEY),
            other => panic!("expected InvalidMetadata, got {other:?}"),
        }
    }

    /// The pattern array exactly as stored in the real file's GGUF
    /// header, read 2026-11-05 from
    /// `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`: key
    /// `muse-glimmer.attention.sliding_window_pattern`, ARRAY of BOOL,
    /// 52 elements, `[1, 1, 1, 0]` repeated 13 times (1 = the layer is
    /// a sliding-window layer, llama.cpp's `is_swa` semantics).
    const REAL_FILE_PATTERN: [bool; 52] = [
        true, true, true, false, //
        true, true, true, false, //
        true, true, true, false, //
        true, true, true, false, //
        true, true, true, false, //
        true, true, true, false, //
        true, true, true, false, //
        true, true, true, false, //
        true, true, true, false, //
        true, true, true, false, //
        true, true, true, false, //
        true, true, true, false, //
        true, true, true, false,
    ];

    #[test]
    fn real_file_array_derives_the_measured_fallback_pattern() {
        // Equivalence proof for the read fix: the file's own array,
        // quoted above from its header, must derive exactly the layer
        // pattern the measured fallback produces, so accepting the
        // array changes provenance (Metadata, not MeasuredFallback)
        // and nothing else.
        let (from_file, file_source) =
            derive_layer_pattern(52, Some(&REAL_FILE_PATTERN)).expect("file pattern");
        let (from_fallback, fallback_source) =
            derive_layer_pattern(52, None).expect("fallback pattern");
        assert_eq!(file_source, PatternSource::Metadata);
        assert_eq!(fallback_source, PatternSource::MeasuredFallback);
        assert_eq!(from_file, from_fallback);
        let globals: Vec<usize> = from_file
            .iter()
            .enumerate()
            .filter(|(_, k)| **k == LayerKind::Global)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            globals,
            vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51]
        );
    }

    #[test]
    fn pattern_refuses_block_count_above_the_ceiling() {
        let err = derive_layer_pattern(MAX_BLOCK_COUNT + 1, None)
            .expect_err("absurd count must fail before allocation");
        match err {
            GlimmerError::InvalidMetadata { key, .. } => assert_eq!(key, KEY_BLOCK_COUNT),
            other => panic!("expected InvalidMetadata, got {other:?}"),
        }
    }
}
