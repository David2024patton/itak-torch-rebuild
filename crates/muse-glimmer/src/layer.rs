//! The per-layer attention pattern: which layers are sliding-window and
//! which are global.
//!
//! Measured from the real Muse Glimmer GGUF on 2026-10-29 (file census in
//! the model teardown volume; write-up in
//! `~/workspace/glimmer-blueprint/GLIMMER-DISSECTION.md`): 52 layers, 39
//! sliding-window and 13 global, the global layers at indices
//! 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, every fourth layer
//! starting at index 3. The file's metadata carries the window size but
//! no per-layer pattern key, so the measured pattern is the recorded
//! fallback. If a future file ships an explicit pattern key, it wins, and
//! [`PatternSource`] records which one produced the pattern so the origin
//! is never silent.

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

/// Metadata key for an explicit pattern: a string of `S` (sliding) and
/// `G` (global), one character per layer, in layer order. No shipping
/// Glimmer file carries it yet; it is read first so a file that grows
/// the key is honored automatically.
pub const SWA_PATTERN_KEY: &str = "muse-glimmer.attention.sliding_window_pattern";

/// Derive the layer pattern.
///
/// With `pattern` present (read from [`SWA_PATTERN_KEY`]), it must hold
/// exactly `block_count` characters of `S` or `G`; anything else is an
/// [`GlimmerError::InvalidMetadata`] naming the key. Without it, the
/// measured fallback applies: every fourth layer starting at index 3 is
/// global, the rest sliding-window.
pub fn derive_layer_pattern(
    block_count: u32,
    pattern: Option<&str>,
) -> Result<(Vec<LayerKind>, PatternSource)> {
    match pattern {
        Some(text) => {
            let mut layers = Vec::with_capacity(block_count as usize);
            for ch in text.chars() {
                match ch {
                    'S' => layers.push(LayerKind::SlidingWindow),
                    'G' => layers.push(LayerKind::Global),
                    _ => {
                        return Err(GlimmerError::InvalidMetadata {
                            key: SWA_PATTERN_KEY,
                            detail: format!("pattern character {ch:?} is neither 'S' nor 'G'"),
                        })
                    }
                }
            }
            if layers.len() != block_count as usize {
                return Err(GlimmerError::InvalidMetadata {
                    key: SWA_PATTERN_KEY,
                    detail: format!(
                        "pattern has {} layers but block_count is {block_count}",
                        layers.len()
                    ),
                });
            }
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
        let (layers, source) = derive_layer_pattern(4, Some("SGSG")).expect("pattern");
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
        let err = derive_layer_pattern(52, Some("SG")).expect_err("short pattern must fail");
        match err {
            GlimmerError::InvalidMetadata { key, .. } => assert_eq!(key, SWA_PATTERN_KEY),
            other => panic!("expected InvalidMetadata, got {other:?}"),
        }
    }

    #[test]
    fn pattern_rejects_unknown_characters() {
        let err = derive_layer_pattern(3, Some("SXG")).expect_err("bad character must fail");
        assert!(matches!(err, GlimmerError::InvalidMetadata { .. }));
    }
}
