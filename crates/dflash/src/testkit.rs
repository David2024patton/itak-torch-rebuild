//! Shared fixtures for this crate's tests: metadata maps carrying
//! the measured real-file values and a tiny synthetic model shape,
//! plus an in-memory tensor source. Test code only; nothing here
//! ships in the library's API surface.

use std::collections::HashMap;

use crate::config::*;
use crate::error::{DflashError, Result};
use crate::metadata::MapMetadata;
use crate::weights::{TensorData, TensorSource};

/// Metadata exactly as measured in the real draft GGUF (census
/// 2026-10-30). Building the config from these values is the closest
/// a unit test gets to the real file without a 1.5 GiB fixture.
pub fn real_draft_metadata() -> MapMetadata {
    let mut m = MapMetadata::new();
    m.insert_string(KEY_ARCHITECTURE, "dflash");
    m.insert_string(KEY_NAME, "Hf_Museglimmer");
    m.insert_u32(KEY_BLOCK_COUNT, 5);
    m.insert_u32(KEY_CONTEXT_LENGTH, 131_072);
    m.insert_u32(KEY_HIDDEN, 6656);
    m.insert_u32(KEY_FFN_HIDDEN, 19_968);
    m.insert_u32(KEY_Q_HEADS, 32);
    m.insert_u32(KEY_KV_HEADS, 8);
    m.insert_u32(KEY_KEY_LENGTH, 128);
    m.insert_u32(KEY_VALUE_LENGTH, 128);
    m.insert_f32(KEY_ROPE_FREQ_BASE, 500_000.0);
    m.insert_f32(KEY_RMS_EPS, 1e-5);
    m.insert_u32(KEY_BLOCK_SIZE, 16);
    m.insert_u32_array(KEY_TARGET_LAYERS, &[2, 14, 26, 38, 50]);
    m.insert_u32(KEY_SLIDING_WINDOW, 2048);
    m.insert_bool_array(KEY_SLIDING_PATTERN, &[true, true, true, true, true]);
    m.insert_u32(KEY_MASK_TOKEN, 201_818);
    m.insert_array_len(KEY_VOCAB_TOKENS, 202_048);
    m
}

/// The real file's config, parsed from [`real_draft_metadata`].
pub fn real_config() -> crate::config::DflashConfig {
    crate::config::DflashConfig::from_metadata(&real_draft_metadata()).expect("real config")
}

/// A two-layer synthetic draft: hidden 8, FFN 16, 2 Q heads over
/// 1 KV head, head dims 4, vocab 16, block size 4, target layers
/// [0, 1] over a target hidden of 8 (so conditioning rows are 16
/// wide), layer 0 sliding with a wide window, layer 1 global.
pub fn tiny_metadata() -> MapMetadata {
    let mut m = MapMetadata::new();
    m.insert_string(KEY_ARCHITECTURE, "dflash");
    m.insert_string(KEY_NAME, "tiny-dflash");
    m.insert_u32(KEY_BLOCK_COUNT, 2);
    m.insert_u32(KEY_CONTEXT_LENGTH, 64);
    m.insert_u32(KEY_HIDDEN, 8);
    m.insert_u32(KEY_FFN_HIDDEN, 16);
    m.insert_u32(KEY_Q_HEADS, 2);
    m.insert_u32(KEY_KV_HEADS, 1);
    m.insert_u32(KEY_KEY_LENGTH, 4);
    m.insert_u32(KEY_VALUE_LENGTH, 4);
    m.insert_f32(KEY_ROPE_FREQ_BASE, 10_000.0);
    m.insert_f32(KEY_RMS_EPS, 1e-5);
    m.insert_u32(KEY_BLOCK_SIZE, 4);
    m.insert_u32_array(KEY_TARGET_LAYERS, &[0, 1]);
    m.insert_u32(KEY_SLIDING_WINDOW, 2048);
    m.insert_bool_array(KEY_SLIDING_PATTERN, &[true, false]);
    m.insert_u32(KEY_MASK_TOKEN, 15);
    m.insert_array_len(KEY_VOCAB_TOKENS, 16);
    m
}

/// The tiny synthetic config, parsed from [`tiny_metadata`].
pub fn tiny_config() -> crate::config::DflashConfig {
    crate::config::DflashConfig::from_metadata(&tiny_metadata()).expect("tiny config")
}

/// Build one zero-filled (or constant-filled) tensor.
pub fn tensor(name: &str, shape: &[u64], fill: f32) -> TensorData {
    let n: u64 = shape.iter().product();
    TensorData {
        name: name.to_string(),
        shape: shape.to_vec(),
        data: vec![fill; n as usize],
    }
}

/// An in-memory [`TensorSource`] over named tensors.
#[derive(Debug, Default)]
pub struct FakeSource {
    tensors: HashMap<String, TensorData>,
}

impl FakeSource {
    /// A source carrying every tensor [`crate::Weights::load`]
    /// requires for `cfg`, zero-filled, plus the draft's own
    /// embedding table and head when `with_own_tables` is set.
    pub fn full(cfg: &crate::config::DflashConfig, with_own_tables: bool) -> Self {
        let mut source = FakeSource::default();
        let hidden = u64::from(cfg.hidden);
        let cond_in = (cfg.target_layers.len() * 8) as u64; // target hidden 8 in fixtures
        source.insert(tensor("fc.weight", &[cond_in, hidden], 0.0));
        source.insert(tensor("enc.output_norm.weight", &[hidden], 0.0));
        source.insert(tensor("output_norm.weight", &[hidden], 0.0));
        if with_own_tables {
            source.insert(tensor(
                "token_embd.weight",
                &[hidden, u64::from(cfg.vocab_size)],
                0.0,
            ));
            source.insert(tensor(
                "output.weight",
                &[hidden, u64::from(cfg.vocab_size)],
                0.0,
            ));
        }
        for layer in 0..cfg.block_count {
            let p = format!("blk.{layer}");
            source.insert(tensor(&format!("{p}.attn_norm.weight"), &[hidden], 0.0));
            source.insert(tensor(
                &format!("{p}.attn_q.weight"),
                &[hidden, u64::from(cfg.q_width())],
                0.0,
            ));
            source.insert(tensor(
                &format!("{p}.attn_k.weight"),
                &[hidden, u64::from(cfg.k_width())],
                0.0,
            ));
            source.insert(tensor(
                &format!("{p}.attn_v.weight"),
                &[hidden, u64::from(cfg.v_width())],
                0.0,
            ));
            source.insert(tensor(
                &format!("{p}.attn_q_norm.weight"),
                &[u64::from(cfg.key_length)],
                0.0,
            ));
            source.insert(tensor(
                &format!("{p}.attn_k_norm.weight"),
                &[u64::from(cfg.key_length)],
                0.0,
            ));
            source.insert(tensor(
                &format!("{p}.attn_output.weight"),
                &[u64::from(cfg.attn_out_width()), hidden],
                0.0,
            ));
            source.insert(tensor(&format!("{p}.ffn_norm.weight"), &[hidden], 0.0));
            source.insert(tensor(
                &format!("{p}.ffn_gate.weight"),
                &[hidden, u64::from(cfg.ffn_hidden)],
                0.0,
            ));
            source.insert(tensor(
                &format!("{p}.ffn_up.weight"),
                &[hidden, u64::from(cfg.ffn_hidden)],
                0.0,
            ));
            source.insert(tensor(
                &format!("{p}.ffn_down.weight"),
                &[u64::from(cfg.ffn_hidden), hidden],
                0.0,
            ));
        }
        source
    }

    /// Insert or replace one tensor.
    pub fn insert(&mut self, t: TensorData) {
        self.tensors.insert(t.name.clone(), t);
    }

    /// Remove one tensor by name.
    pub fn remove(&mut self, name: &str) {
        self.tensors.remove(name);
    }
}

impl TensorSource for FakeSource {
    fn tensor(&self, name: &str) -> Result<TensorData> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| DflashError::MissingTensor {
                name: name.to_string(),
            })
    }

    fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}
