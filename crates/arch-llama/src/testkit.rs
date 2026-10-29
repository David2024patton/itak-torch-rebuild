//! Shared fixtures for this crate's tests: a minimal GGUF v3 byte
//! builder (so config and load tests run through the real gguf
//! parser, not a stub), metadata sets carrying the measured values
//! of the two conformance files, and a deterministic value generator
//! for synthetic weights. Test code only; nothing here ships in the
//! library's API surface.

use std::collections::HashMap;

use gguf::MetadataValue;

use crate::config::LlamaConfig;
use crate::error::{LlamaError, Result};
use crate::weights::{TensorData, TensorSource};

/// One tensor to write into a synthetic GGUF: name, dimensions in
/// GGUF stored order, and its F32 values (written as ggml type F32).
pub struct TestTensor {
    /// Tensor name.
    pub name: String,
    /// Dimensions in GGUF stored order (innermost first).
    pub dims: Vec<u64>,
    /// Element values; length must equal the product of `dims`.
    pub values: Vec<f32>,
}

/// Little-endian bytes for F32 values.
pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn write_scalar(out: &mut Vec<u8>, v: &MetadataValue) {
    match v {
        MetadataValue::UInt8(x) => out.push(*x),
        MetadataValue::Int8(x) => out.push(*x as u8),
        MetadataValue::UInt16(x) => out.extend_from_slice(&x.to_le_bytes()),
        MetadataValue::Int16(x) => out.extend_from_slice(&x.to_le_bytes()),
        MetadataValue::UInt32(x) => out.extend_from_slice(&x.to_le_bytes()),
        MetadataValue::Int32(x) => out.extend_from_slice(&x.to_le_bytes()),
        MetadataValue::Float32(x) => out.extend_from_slice(&x.to_le_bytes()),
        MetadataValue::Bool(x) => out.push(u8::from(*x)),
        MetadataValue::String(s) => write_string(out, s),
        MetadataValue::UInt64(x) => out.extend_from_slice(&x.to_le_bytes()),
        MetadataValue::Int64(x) => out.extend_from_slice(&x.to_le_bytes()),
        MetadataValue::Float64(x) => out.extend_from_slice(&x.to_le_bytes()),
        MetadataValue::Array(_) => panic!("arrays do not nest in GGUF"),
    }
}

fn write_value(out: &mut Vec<u8>, v: &MetadataValue) {
    out.extend_from_slice(&v.type_id().to_le_bytes());
    match v {
        MetadataValue::Array(items) => {
            let elem_type = items.first().map_or(8, MetadataValue::type_id);
            out.extend_from_slice(&elem_type.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                write_scalar(out, item);
            }
        }
        scalar => write_scalar(out, scalar),
    }
}

fn align_up(v: u64, a: u64) -> u64 {
    v.div_ceil(a) * a
}

/// Build a complete GGUF v3 file in memory: the given metadata and
/// F32 tensors, tensor data aligned to 32 bytes like a real file.
pub fn build_gguf(metadata: &[(&str, MetadataValue)], tensors: &[TestTensor]) -> Vec<u8> {
    const ALIGNMENT: u64 = 32;
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (key, value) in metadata {
        write_string(&mut out, key);
        write_value(&mut out, value);
    }
    // Tensor directory, with offsets assigned in layout order.
    let mut offset: u64 = 0;
    let mut offsets = Vec::with_capacity(tensors.len());
    for t in tensors {
        offset = align_up(offset, ALIGNMENT);
        offsets.push(offset);
        offset += (t.values.len() * 4) as u64;
    }
    for (t, &off) in tensors.iter().zip(&offsets) {
        assert_eq!(
            t.values.len() as u64,
            t.dims.iter().product::<u64>(),
            "test tensor {} value count disagrees with its dims",
            t.name
        );
        write_string(&mut out, &t.name);
        out.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&0u32.to_le_bytes()); // ggml type 0: F32
        out.extend_from_slice(&off.to_le_bytes());
    }
    // Pad to the data section, then the tensor bytes at their offsets.
    let data_start = align_up(out.len() as u64, ALIGNMENT);
    out.resize(data_start as usize, 0);
    for (t, &off) in tensors.iter().zip(&offsets) {
        let at = data_start as usize + off as usize;
        if out.len() < at + t.values.len() * 4 {
            out.resize(at + t.values.len() * 4, 0);
        }
        out[at..at + t.values.len() * 4].copy_from_slice(&f32_bytes(&t.values));
    }
    out
}

/// A token array of `n` distinct short strings, standing in for a
/// real vocabulary in metadata fixtures.
pub fn tokens(n: usize) -> MetadataValue {
    MetadataValue::Array(
        (0..n)
            .map(|i| MetadataValue::String(format!("t{i}")))
            .collect(),
    )
}

/// Metadata exactly as measured in the real TinyLlama-1.1B GGUF
/// (census 2026-10-29): llama arch, no attention biases, untied head.
pub fn tinyllama_metadata() -> Vec<(String, MetadataValue)> {
    vec![
        (
            "general.architecture".into(),
            MetadataValue::String("llama".into()),
        ),
        (
            "general.name".into(),
            MetadataValue::String("tinyllama_tinyllama-1.1b-chat-v1.0".into()),
        ),
        ("llama.block_count".into(), MetadataValue::UInt32(22)),
        ("llama.context_length".into(), MetadataValue::UInt32(2048)),
        ("llama.embedding_length".into(), MetadataValue::UInt32(2048)),
        (
            "llama.feed_forward_length".into(),
            MetadataValue::UInt32(5632),
        ),
        (
            "llama.attention.head_count".into(),
            MetadataValue::UInt32(32),
        ),
        (
            "llama.attention.head_count_kv".into(),
            MetadataValue::UInt32(4),
        ),
        (
            "llama.attention.layer_norm_rms_epsilon".into(),
            MetadataValue::Float32(1e-5),
        ),
        (
            "llama.rope.freq_base".into(),
            MetadataValue::Float32(10_000.0),
        ),
        (
            "llama.rope.dimension_count".into(),
            MetadataValue::UInt32(64),
        ),
        ("tokenizer.ggml.tokens".into(), tokens(32_000)),
    ]
}

/// Metadata exactly as measured in the real Qwen2.5-1.5B GGUF
/// (census 2026-10-29): qwen2 arch, attention biases in the tensor
/// map, and no `rope.dimension_count` key in the file.
pub fn qwen2_metadata() -> Vec<(String, MetadataValue)> {
    vec![
        (
            "general.architecture".into(),
            MetadataValue::String("qwen2".into()),
        ),
        (
            "general.name".into(),
            MetadataValue::String("qwen2.5-1.5b-instruct".into()),
        ),
        ("qwen2.block_count".into(), MetadataValue::UInt32(28)),
        ("qwen2.context_length".into(), MetadataValue::UInt32(32_768)),
        ("qwen2.embedding_length".into(), MetadataValue::UInt32(1536)),
        (
            "qwen2.feed_forward_length".into(),
            MetadataValue::UInt32(8960),
        ),
        (
            "qwen2.attention.head_count".into(),
            MetadataValue::UInt32(12),
        ),
        (
            "qwen2.attention.head_count_kv".into(),
            MetadataValue::UInt32(2),
        ),
        (
            "qwen2.attention.layer_norm_rms_epsilon".into(),
            MetadataValue::Float32(1e-6),
        ),
        (
            "qwen2.rope.freq_base".into(),
            MetadataValue::Float32(1_000_000.0),
        ),
        ("tokenizer.ggml.tokens".into(), tokens(151_936)),
    ]
}

/// Metadata for a two-layer synthetic model of the given family:
/// hidden 32, FFN 48, 4 Q heads over 2 KV heads (head dim 8),
/// vocab 64, context 128. Small enough for full forward tests.
pub fn tiny_metadata(arch: &str) -> Vec<(String, MetadataValue)> {
    let p = |suffix: &str| format!("{arch}.{suffix}");
    let mut entries = vec![
        (
            "general.architecture".to_string(),
            MetadataValue::String(arch.to_string()),
        ),
        (
            "general.name".to_string(),
            MetadataValue::String(format!("tiny-{arch}")),
        ),
        (p("block_count"), MetadataValue::UInt32(2)),
        (p("context_length"), MetadataValue::UInt32(128)),
        (p("embedding_length"), MetadataValue::UInt32(32)),
        (p("feed_forward_length"), MetadataValue::UInt32(48)),
        (p("attention.head_count"), MetadataValue::UInt32(4)),
        (p("attention.head_count_kv"), MetadataValue::UInt32(2)),
        (
            p("attention.layer_norm_rms_epsilon"),
            MetadataValue::Float32(1e-6),
        ),
        (p("rope.freq_base"), MetadataValue::Float32(10_000.0)),
        ("tokenizer.ggml.tokens".to_string(), tokens(64)),
    ];
    if arch == "llama" {
        // TinyLlama carries the key; Qwen2.5 does not. Mirror both.
        entries.push((p("rope.dimension_count"), MetadataValue::UInt32(8)));
    }
    entries
}

/// Deterministic pseudo-values in a stable range, so synthetic
/// weights are neither zeros (which hide wiring bugs: every path
/// would output zero) nor random per run (which hides regressions).
/// Values land in [-0.3, 0.3).
pub fn deterministic_values(count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E37_79B9);
    (0..count)
        .map(|_| {
            // LCG, Numerical Recipes constants.
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / (1u32 << 24) as f32;
            unit * 0.6 - 0.3
        })
        .collect()
}

/// Convert owned metadata fixtures to the borrowed shape
/// [`build_gguf`] takes.
pub fn refs(entries: &[(String, MetadataValue)]) -> Vec<(&str, MetadataValue)> {
    entries
        .iter()
        .map(|(k, v)| (k.as_str(), v.clone()))
        .collect()
}

/// A [`LlamaConfig`] for the tiny synthetic shape [`tiny_metadata`]
/// describes, built directly (config parsing itself is tested
/// through the GGUF byte path in the config module).
pub fn tiny_config(arch: &str) -> LlamaConfig {
    LlamaConfig {
        architecture: arch.to_string(),
        name: Some(format!("tiny-{arch}")),
        block_count: 2,
        context_length: 128,
        hidden: 32,
        ffn_hidden: 48,
        q_heads: 4,
        kv_heads: 2,
        head_dim: 8,
        rope_freq_base: 10_000.0,
        rope_dim: 8,
        rms_eps: 1e-6,
        vocab_size: 64,
    }
}

/// How a [`FakeSource`] fills the attention bias tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiasMode {
    /// No bias tensors in the source at all.
    None,
    /// Bias tensors present, all zeros (present but inert: proves
    /// the forward applies the values, not just their presence).
    Zero,
    /// Bias tensors present with deterministic nonzero values.
    NonZero,
}

/// An in-memory [`TensorSource`] holding a complete tiny weight set
/// for a config, unless a test removes or replaces tensors.
#[derive(Debug)]
pub struct FakeSource {
    tensors: HashMap<String, TensorData>,
}

impl FakeSource {
    /// A complete weight set for `cfg`: every tensor the load
    /// requires, plus biases per `biases`, plus the untied head
    /// unless `tied`. Values are deterministic per tensor.
    pub fn full(cfg: &LlamaConfig, tied: bool, biases: BiasMode) -> Self {
        let mut tensors: HashMap<String, TensorData> = HashMap::new();
        // Each tensor's values derive from its NAME, never from
        // insertion order, so two sources that differ only in which
        // optional tensors they carry (biases, the untied head)
        // still hold identical values for every shared tensor. The
        // bias and tied tests rely on exactly that.
        fn seed_of(name: &str) -> u32 {
            // FNV-1a over the name.
            let mut hash = 0x811C_9DC5u32;
            for byte in name.bytes() {
                hash ^= u32::from(byte);
                hash = hash.wrapping_mul(0x0100_0193);
            }
            hash
        }
        let put = |tensors: &mut HashMap<String, TensorData>, name: String, shape: &[u64]| {
            let count: u64 = shape.iter().product();
            let seed = seed_of(&name);
            tensors.insert(
                name.clone(),
                TensorData {
                    name,
                    shape: shape.to_vec(),
                    data: deterministic_values(count as usize, seed),
                },
            );
        };
        let hidden = u64::from(cfg.hidden);
        let vocab = u64::from(cfg.vocab_size);
        let ffn = u64::from(cfg.ffn_hidden);
        let q_w = u64::from(cfg.q_width());
        let k_w = u64::from(cfg.k_width());
        let v_w = u64::from(cfg.v_width());
        put(
            &mut tensors,
            "token_embd.weight".to_string(),
            &[hidden, vocab],
        );
        put(&mut tensors, "output_norm.weight".to_string(), &[hidden]);
        if !tied {
            put(&mut tensors, "output.weight".to_string(), &[hidden, vocab]);
        }
        for layer in 0..cfg.block_count {
            let p = |suffix: &str| format!("blk.{layer}.{suffix}");
            put(&mut tensors, p("attn_norm.weight"), &[hidden]);
            put(&mut tensors, p("attn_q.weight"), &[hidden, q_w]);
            put(&mut tensors, p("attn_k.weight"), &[hidden, k_w]);
            put(&mut tensors, p("attn_v.weight"), &[hidden, v_w]);
            if biases != BiasMode::None {
                for (suffix, width) in [
                    ("attn_q.bias", q_w),
                    ("attn_k.bias", k_w),
                    ("attn_v.bias", v_w),
                ] {
                    let name = p(suffix);
                    let data = match biases {
                        BiasMode::Zero => vec![0.0; width as usize],
                        _ => deterministic_values(width as usize, seed_of(&name)),
                    };
                    tensors.insert(
                        name.clone(),
                        TensorData {
                            name,
                            shape: vec![width],
                            data,
                        },
                    );
                }
            }
            put(&mut tensors, p("attn_output.weight"), &[q_w, hidden]);
            put(&mut tensors, p("ffn_norm.weight"), &[hidden]);
            put(&mut tensors, p("ffn_gate.weight"), &[hidden, ffn]);
            put(&mut tensors, p("ffn_up.weight"), &[hidden, ffn]);
            put(&mut tensors, p("ffn_down.weight"), &[ffn, hidden]);
        }
        FakeSource { tensors }
    }

    /// Remove a tensor, for missing-tensor tests.
    pub fn remove(&mut self, name: &str) {
        self.tensors.remove(name);
    }

    /// Replace or insert a tensor, for shape and tied-head tests.
    pub fn set(&mut self, tensor: TensorData) {
        self.tensors.insert(tensor.name.clone(), tensor);
    }

    /// Look up a tensor the source holds.
    pub fn get(&self, name: &str) -> Option<&TensorData> {
        self.tensors.get(name)
    }

    /// Convert the whole set into [`TestTensor`]s for the GGUF byte
    /// builder, for the end-to-end-through-the-real-loader test.
    pub fn into_test_tensors(self) -> Vec<TestTensor> {
        let mut out: Vec<TestTensor> = self
            .tensors
            .into_values()
            .map(|t| TestTensor {
                name: t.name,
                dims: t.shape,
                values: t.data,
            })
            .collect();
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }
}

impl TensorSource for FakeSource {
    fn tensor(&self, name: &str) -> Result<TensorData> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| LlamaError::MissingTensor {
                name: name.to_string(),
            })
    }

    fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}
