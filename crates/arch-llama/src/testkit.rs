//! Shared fixtures for this crate's tests: a minimal GGUF v3 byte
//! builder (so config and load tests run through the real gguf
//! parser, not a stub), metadata sets carrying the measured values
//! of the two conformance files, and a deterministic value generator
//! for synthetic weights. Test code only; nothing here ships in the
//! library's API surface.

use gguf::MetadataValue;

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

/// Convert owned metadata fixtures to the borrowed shape
/// [`build_gguf`] takes.
pub fn refs(entries: &[(String, MetadataValue)]) -> Vec<(&str, MetadataValue)> {
    entries
        .iter()
        .map(|(k, v)| (k.as_str(), v.clone()))
        .collect()
}
